"""
Arquant v1.0 - KIS Broker (확장판)
한국투자증권 OpenAPI 전체 카테고리 지원:
  국내주식 시세/주문/잔고, 해외주식, 장내채권, 해외선물옵션, 국내선물옵션
  일봉/분봉 실시간 데이터 CSV 누적 수집
"""
import asyncio, aiohttp, time, logging, csv, json, math
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger("KIS")
KST = timezone(timedelta(hours=9))
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
# KIS access token (24h 유효). 디스크에 캐싱 → 프로세스 재시작/멀티 클라이언트라도 만료 전엔 재발급 안 함.
TOKEN_CACHE_FILE = DATA_DIR / "kis_token.json"

import re as _re

try:  # 공용 시세(EOD) 레이어 — pykrx/yfinance 우선, 실패 시 기존 KIS 경로 폴백
    import arcmarket
except ImportError:
    arcmarket = None


def _clean_kis_msg(msg: str) -> str:
    """사장 피드백 2026-05-16: KIS 원문(msg1)에서 '해외투자영업부(3276-5300)문의'·전화번호 같은
    안내 보일러플레이트를 제거하고 **실제 거부/처리 사유만** 남긴다.
    (예: "가격 $0.01 미만시 온라인 주문불가조건입니다. 해외투자영업부(3276-5300)문의"
       → "가격 $0.01 미만시 온라인 주문불가조건입니다.")
    과거 해외 ETP 거래제한이 이 메시지로 뜬 적이 있으나, 지금은 거래 가능 — 표시는 실제 사유만."""
    if not msg:
        return ""
    s = str(msg)
    # 부서/센터 + (전화) + 문의 형태의 안내 꼬리 제거
    s = _re.sub(r"\s*[가-힣A-Za-z]{0,12}(?:영업부|고객\s*센터|상담\s*센터|콜센터|데스크)\s*"
                r"[\(（]?\s*\d[\d\-\s]{4,}[\)）]?\s*(?:로)?\s*문의\s*(?:바랍니다|하세요|요망)?\.?",
                "", s)
    # 잔여 전화번호 토큰 제거
    s = _re.sub(r"[\(（]?\b\d{2,4}-\d{3,4}-?\d{0,4}\b[\)）]?", "", s)
    # 꼬리에 남은 '문의' 단독 잔재 제거
    s = _re.sub(r"\s*문의\s*(?:바랍니다|하세요)?\.?\s*$", "", s)
    s = _re.sub(r"\s{2,}", " ", s).strip(" .·,")
    return s.strip()


def _sanitize_overseas(krw, stock, exrt, *, min_valid_exrt: float = 500.0):
    """모의서버 비정상 기준환율(exrt < min_valid_exrt) 시 해외 평가(krw)·주식분(stock)을
    함께 0 으로 — garbage 전파 차단. krw 는 총평가 합산용, stock 은 매크로 비중계산
    (_overseas_stock_krw)용. 기존엔 krw 만 0 처리하고 stock 은 오염값을 캐시에 흘려보내
    주식비중이 100% 로 부풀어 매수가 영구 차단되던 버그(uid2). exrt 0/None(미상)은
    건드리지 않는다(조회 실패와 비정상 환율을 구분 — 보수적). 정상 환율이면 입력 그대로."""
    if exrt and float(exrt) < min_valid_exrt:
        return 0.0, 0.0
    return krw, stock


def _real_usdkrw() -> float:
    """실환율(USD/KRW). 모르면 0.0 — 모르면 지어내지 않는다."""
    try:
        from tools.market_data import get_usdkrw
        return float(get_usdkrw(0.0) or 0.0)
    except Exception:
        return 0.0


def kr_net_valuation(scts_eval: float, cash_d2: float, cash_d1: float,
                     prev_settled: Optional[float] = None):
    """현재 평가액(국내 구성) = 국내 유가증권평가액 + D+2 예수금. (해외 외화평가총액은 호출부에서 더한다)

    사장 지시 2026-05-28: D+2(prvs_rcdl_excc_amt)가 정상 정산예수금이다. KIS가 결제 과도기에
    D+2/D+1 을 0 으로 깜빡이면 직전 정상 D+2 를 유지한다 — D0(dnca_tot_amt)로 폴백하면 미결제
    매수분이 아직 안 빠져 부풀려진 값이라 자산곡선이 스파이크 친다(hh09080 +4.9M 유령점프, 2026-05-28).
    시그니처에 D0 를 받지 않아 구조적으로 폴백이 불가능하다.
    반환: (kr_valuation, settled_cash) — settled_cash 는 carry-forward 적용된 D+2."""
    if cash_d2 and cash_d2 > 0:
        settled = float(cash_d2)
    elif prev_settled and prev_settled > 0:
        settled = float(prev_settled)
    elif cash_d1 and cash_d1 > 0:
        settled = float(cash_d1)
    else:
        settled = 0.0
    return float(scts_eval or 0.0) + settled, settled


def marketable_us_limit(side: str, limit_price: float, cur_price: float):
    """US 명시 지정가가 호가 반대쪽이면 체결가능 가격으로 클램프. 사장 지시 2026-05-28.

    매수 지정가 < 현재가(시세 아래 매수 = 미체결) → 현재가×1.003(센트 올림).
    매도 지정가 > 현재가(시세 위 매도 = 미체결) → 현재가×0.997(센트 내림).
    이미 체결가능(매수 limit≥현재가 / 매도 limit≤현재가)이면 지정가 그대로.
    현재가 미확보(cur_price<=0)면 지정가 유지(주문 미차단 — 데이터 결손으로 주문 막지 않음).
    반환: (price, clamped: bool)."""
    lp = round(float(limit_price or 0.0), 2)
    if not cur_price or cur_price <= 0:
        return lp, False
    if side == "buy" and lp < cur_price:
        return math.ceil(cur_price * 1.003 * 100) / 100.0, True
    if side == "sell" and lp > cur_price:
        return math.floor(cur_price * 0.997 * 100) / 100.0, True
    return lp, False


def excd_to_excg(excd: Optional[str]) -> str:
    """시세 프로브 거래소코드(NAS/NYS/AMS) → 주문 거래소코드(NASD/NYSE/AMEX).
    미상·기타는 NASD 로 안전 폴백. 이미 주문코드면 그대로 통과."""
    e = (excd or "").strip().upper()
    if e in ("NASD", "NYSE", "AMEX"):
        return e
    return {"NAS": "NASD", "NYS": "NYSE", "AMS": "AMEX"}.get(e, "NASD")


def kr_tick_size(price: float) -> int:
    """KRX/NXT 공통 호가단위 (2023~ 개정 기준)."""
    p = float(price or 0)
    if p < 2000:    return 1
    if p < 5000:    return 5
    if p < 20000:   return 10
    if p < 50000:   return 50
    if p < 200000:  return 100
    if p < 500000:  return 500
    return 1000


def round_to_tick(price: float) -> int:
    """가장 가까운 유효 호가로 반올림(nearest, Python round=banker's rounding). 밴드가 이미 공격성을 부여하므로 방향성 라운딩 불필요."""
    p = float(price or 0)
    if p <= 0:
        return 0
    t = kr_tick_size(p)
    return int(round(p / t) * t)


def ceil_to_tick(price: float) -> int:
    """국내주식 지정가를 해당 가격대의 유효 호가로 올림한다.

    LLM·전략이 27,196원처럼 호가단위에 맞지 않는 가격을 제시해도 주문 전송 직전에
    27,200원으로 보정한다. 0 이하는 시장가 표식이므로 그대로 0을 반환한다.
    """
    p = float(price or 0)
    if p <= 0:
        return 0
    tick = kr_tick_size(p)
    return int(math.ceil(p / tick) * tick)


def compute_nxt_limit_price(last_price: float, *, side: str, slippage_pct: float,
                            ref_price: float = None, max_premium_pct: float = None) -> int:
    """시간외 지정가 = NXT시세 ± 슬리피지 밴드, 호가단위 반올림. last_price<=0 이면 0(주문 보류 신호).

    ref_price(정규 전일종가)·max_premium_pct 가 주어지면 **정규가 대비 프리미엄을 캡**한다:
    매수는 ref×(1+캡)을 넘지 못하고, 매도는 ref×(1−캡) 아래로 내려가지 못한다. 얇은 NXT
    프리마켓이 큰 프리미엄/디스카운트를 호가해도 그걸 추종해 과지불/과소매도하지 않게 한다
    (2026-06-15: 003490 을 정규 26,600 대비 +4.9% 27,900 에 시장가 추종 체결한 버그 수정).
    """
    last = float(last_price or 0)
    if last <= 0:
        return 0
    band = (float(slippage_pct or 0) / 100.0)
    raw = last * (1 + band) if side == "buy" else last * (1 - band)
    ref = float(ref_price or 0)
    cap = float(max_premium_pct or 0) / 100.0
    if ref > 0 and cap > 0:
        if side == "buy":
            raw = min(raw, ref * (1 + cap))   # 정규가 대비 프리미엄 상한
        else:
            raw = max(raw, ref * (1 - cap))   # 정규가 대비 디스카운트 하한
    return round_to_tick(raw)


class OrderSide(str, Enum):
    BUY = "buy"; SELL = "sell"
class PriceType(str, Enum):
    MARKET = "market"; LIMIT = "limit"
class OrderDraft(BaseModel):
    ticker: str; side: OrderSide; qty: int = Field(gt=0)
    price_type: PriceType = PriceType.MARKET; limit_price: Optional[float] = None
    market: str = "KR"; exchange: str = "KRX"; reason: str = ""; approved: bool = False
    rejection_reason: Optional[str] = None

class KISBroker:
    # 같은 KIS 자격증명을 쓰는 여러 계정/브로커 인스턴스가 한 이벤트루프에서 호출 간격을 공유한다.
    # KIS 제한은 Python 객체별이 아니라 appkey·서버별인데, 종전 인스턴스 락은 동시 호출을 못 막았다.
    _SHARED_RATE_STATES: Dict[tuple, Dict[str, Any]] = {}

    def __init__(self, creds: dict, token_path=None):
        # Phase 2: credentials are injected per-uid. No more config globals.
        self.app_key = creds["kis_app_key"]; self.app_secret = creds["kis_app_secret"]
        self.base_url = creds.get("kis_base_url") or "https://openapi.koreainvestment.com:9443"
        self.account_no = creds["kis_account_no"]
        self._token_path = Path(token_path) if token_path else TOKEN_CACHE_FILE
        # 사장 피드백 2026-05-16: 실전/모의 정식 지원. base_url 이 KIS 모의투자 서버면
        # 주문/잔고 tr_id 를 모의용으로 변환 (실전 경로는 전혀 안 건드림 — 무위험).
        _bu = (self.base_url or "")
        self.is_mock = ("openapivts" in _bu) or (":29443" in _bu)
        self._token: Optional[str] = None; self._token_exp: float = 0
        self._session: Optional[aiohttp.ClientSession] = None
        # 사장 지시 2026-06-01: 전역 호출간격 락 — 해외 거래소순회·페이징·5분폴러·멀티테넌트 동시호출이
        # 겹쳐도 KIS 초당제한(EGW00201)에 안 걸리게 사전 직렬화한다(거부 후 백오프보다 안정적).
        # 모의서버는 1.0s, 실전은 0.10s(≈10TPS) 간격. 이론 한도에 딱 맞춘 0.5s/0.06s는
        # 재시작 직후 계정별 잔고·시세 호출이 겹칠 때 서버 윈도우 오차로 반복 거부됐다.
        self._rate_lock = asyncio.Lock()
        self._last_call: float = 0.0
        # 사장 지시 2026-06-17: 고정 간격이 KIS 실측 한도를 넘는 버스트 구간엔 거부 폭주가 났다.
        # base 에서 시작해 rate-limit 거부 시 상향(_note_rate_limited)·무거부 시 점감(_decay_interval).
        self._rate_base: float = 1.0 if self.is_mock else 0.10
        self._min_interval: float = self._rate_base
        self._nxt_supported = None   # None=미탐, True=지원확인, False=미지원(시간외 스킵)

    async def _s(self):
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self._session

    async def close(self):
        if self._session and not self._session.closed: await self._session.close()

    # token re-issue is allowed only when within this many seconds of expiry (keep a safety margin)
    _TOKEN_SAFETY_SEC = 600

    def _load_token_file(self) -> Optional[Dict[str, Any]]:
        try:
            if self._token_path.exists():
                d = json.loads(self._token_path.read_text(encoding="utf-8"))
                if isinstance(d, dict) and d.get("access_token") and d.get("appkey") == self.app_key:
                    return d
        except Exception as e:
            logger.warning(f"토큰 캐시 읽기 실패: {e}")
        return None

    def _save_token_file(self, token: str, exp: float):
        try:
            self._token_path.write_text(json.dumps(
                {"access_token": token, "expires_at": exp, "appkey": self.app_key,
                 "issued_at": time.time()}, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning(f"토큰 캐시 저장 실패: {e}")

    # KIS가 토큰을 '서버측에서 조기 무효화'했을 때 응답에 나타나는 마커 (rt_cd≠0 + 아래 문구/코드).
    # 로컬 expires_at 은 미래인데 KIS는 만료로 거부하는 글리치(잔고 0원의 근본원인)를 _authed_json 이 잡아 강제 재발급한다.
    _TOKEN_EXPIRED_MARKERS = ("기간이 만료된 token", "expired token", "egw00123", "egw00121")
    # KIS 초당 거래건수/호출 제한 거부 마커 — 거부(rt_cd≠0, 미체결 확정)라 재전송 안전. 사장 지시 2026-05-28.
    _RATE_LIMIT_MARKERS = ("초당 거래건수", "거래건수를 초과", "egw00201", "초당 허용", "초당 호출")
    _RATE_LIMIT_BACKOFF_SEC = 0.35
    _RATE_LIMIT_MAX_RETRY = 3
    # 적응적 간격(사장 지시 2026-06-17): 거부 폭주를 스스로 완화.
    _RATE_MAX_INTERVAL = 2.0    # 상향 상한(모의 base 0.5보다 커야 거부 후 실제로 느려진다)
    _RATE_BUMP = 1.6            # 거부 1회당 곱셈 상향
    _RATE_DECAY = 0.92         # 무거부 호출마다 base 로 점감

    def _resp_token_expired(self, d: Any) -> bool:
        """KIS 응답이 '토큰 만료/무효' 거부인가. 정상(rt_cd==0)·다른 거부 사유는 False."""
        if not isinstance(d, dict) or str(d.get("rt_cd", "")) == "0":
            return False
        blob = f"{d.get('msg_cd','')} {d.get('msg1','')}".lower()
        return any(m in blob for m in self._TOKEN_EXPIRED_MARKERS)

    def _resp_rate_limited(self, d: Any) -> bool:
        """KIS 응답이 '초당 거래건수/호출 초과' 거부인가. 정상(rt_cd==0)·다른 거부는 False."""
        if not isinstance(d, dict) or str(d.get("rt_cd", "")) == "0":
            return False
        blob = f"{d.get('msg_cd','')} {d.get('msg1','')}".lower()
        return any(m in blob for m in self._RATE_LIMIT_MARKERS)

    def _note_rate_limited(self) -> None:
        """rate-limit 거부 관측 → 호출 간격을 곱셈 상향(상한까지)해 버스트를 스스로 벌린다."""
        state = getattr(self, "_active_rate_state", None)
        current = max(float(self._min_interval),
                      float((state or {}).get("min_interval") or 0.0))
        self._min_interval = min(self._RATE_MAX_INTERVAL, current * self._RATE_BUMP)
        if state is not None:
            state["min_interval"] = self._min_interval
            self._min_interval = state["min_interval"]

    def _decay_interval(self) -> None:
        """거부 없이 호출이 흐르면 간격을 base 로 점감 복귀(base 아래로는 내리지 않음)."""
        if self._min_interval > self._rate_base:
            self._min_interval = max(self._rate_base, self._min_interval * self._RATE_DECAY)
        state = getattr(self, "_active_rate_state", None)
        if state is not None:
            cur = float(state.get("min_interval") or self._rate_base)
            state["min_interval"] = max(self._rate_base, cur * self._RATE_DECAY)
            self._min_interval = max(self._min_interval, state["min_interval"])

    async def _authed_json(self, make_request):
        """make_request: async (tok:str) -> dict(파싱된 KIS JSON). 토큰을 주입해 1회 호출하고,
        KIS가 '만료 토큰'으로 거부하면 token(force=True) 강제 재발급 후 **딱 1회** 재시도한다.
        추가로 '초당 거래건수 초과' rate-limit 거부면 간격을 두고 재전송한다(주문 드롭 금지 —
        거부는 미체결 확정이라 재전송이 안전, 사장 지시 2026-05-28).
        모든 잔고/주문 경로가 이 한 곳을 통해 죽은-토큰 고착·rate-limit 드롭을 자가치유한다."""
        tok = await self.token()
        await self._pace()
        d = await make_request(tok)
        if self._resp_token_expired(d):
            logger.warning("KIS '기간이 만료된 token' 응답 — 토큰 강제 재발급 후 1회 재시도")
            try:
                tok = await self.token(force=True)
            except Exception as e:
                logger.error(f"토큰 강제 재발급 실패: {e}")
                return d
            await self._pace()
            d = await make_request(tok)
        attempts = 0
        while self._resp_rate_limited(d) and attempts < self._RATE_LIMIT_MAX_RETRY:
            attempts += 1
            self._note_rate_limited()   # 적응적 간격 상향 — 이후 호출이 스스로 벌어져 거부 연쇄를 줄인다
            delay = self._RATE_LIMIT_BACKOFF_SEC * attempts
            logger.warning(f"KIS rate-limit(초당 거래건수 초과) — {delay:.2f}s 후 재전송 {attempts}/{self._RATE_LIMIT_MAX_RETRY}")
            if delay > 0:
                await asyncio.sleep(delay)
            await self._pace()
            d = await make_request(tok)
        return d

    async def _get_json(self, path: str, tr_id: str, params: Dict[str, Any]) -> Dict:
        """단순 GET 조회 보일러플레이트(세션·토큰·헤더 구성)를 한 곳으로 모으고,
        _authed_json 을 태워 **시세/조회 경로도 토큰 만료 자가치유**되게 한다
        (과거엔 주문/잔고만 self-heal 됐고 시세는 await self.token() 직호출이라 비대칭).
        path 는 base_url 뒤에 붙는 절대경로. 반환: 파싱된 KIS JSON 전체."""
        async def _do(tok):
            s = await self._s()
            async with s.get(f"{self.base_url}{path}", headers=self._h(tok, tr_id), params=params) as r:
                return await r.json()
        return await self._authed_json(_do)

    async def _pace(self) -> None:
        """KIS 호출 사전 간격 보장(초당제한 회피). _min_interval 만큼 직전 호출과 벌린다.
        appkey·서버가 같은 모든 브로커 인스턴스를 공유 lock 으로 직렬화한다."""
        loop = asyncio.get_running_loop()
        key = (loop, self.base_url, self.app_key)
        state = self._SHARED_RATE_STATES.get(key)
        if state is None:
            state = {"lock": asyncio.Lock(), "last_call": 0.0,
                     "min_interval": float(self._min_interval)}
            self._SHARED_RATE_STATES[key] = state
        self._active_rate_state = state
        async with state["lock"]:
            interval = max(float(self._min_interval), float(state.get("min_interval") or 0.0))
            wait = interval - (loop.time() - float(state.get("last_call") or 0.0))
            if wait > 0:
                await asyncio.sleep(wait)
            now = loop.time()
            state["last_call"] = now
            self._last_call = now
            self._decay_interval()   # 거부 없이 흐르면 base 로 점감 복귀

    async def _paged_get(self, path: str, tr_id: str, params: Dict[str, Any],
                         fk_key: str = "CTX_AREA_FK100", nk_key: str = "CTX_AREA_NK100",
                         out_keys=("output1",), max_depth: int = 10) -> Dict:
        """KIS 연속조회(tr_cont) 표준 루프 + 부분성공 보존.
        응답 헤더 tr_cont∈{F,M}이면 다음 요청에 헤더 tr_cont='N' + 직전 응답의 ctx_area_fk/nk 를 실어 재호출.
        out_keys 의 list 출력은 누적, dict 출력(요약 output2/3)은 최신값으로 유지.
        rt_cd≠0: 첫 페이지면 ok=False, 이후 페이지면 누적분을 ok=True·partial=True 로 보존(사장 지시 2026-06-01)."""
        acc: Dict[str, Any] = {k: [] for k in out_keys}
        last = {"rt_cd": None, "msg1": "", "msg_cd": ""}
        p = dict(params)
        tr_cont = ""
        ok, partial, page = True, False, 0
        while page < max_depth:
            hdr: Dict[str, str] = {}

            async def _do(tok, _p=dict(p), _trc=tr_cont, _hdr=hdr):
                s = await self._s()
                headers = self._h(tok, tr_id)
                if _trc:
                    headers["tr_cont"] = _trc
                async with s.get(f"{self.base_url}{path}", headers=headers, params=_p) as r:
                    _hdr["tr_cont"] = r.headers.get("tr_cont", "") or ""
                    return await r.json()

            body = await self._authed_json(_do)
            body = body if isinstance(body, dict) else {}
            rc = str(body.get("rt_cd", ""))
            last = {"rt_cd": rc, "msg1": body.get("msg1", ""), "msg_cd": body.get("msg_cd", "")}
            if rc != "0":
                if page == 0:
                    ok = False
                else:
                    partial = True
                break
            for k in out_keys:
                v = body.get(k)
                if isinstance(v, list):
                    acc[k].extend(v)
                elif isinstance(v, dict):
                    acc[k] = v   # 요약 객체 — 최신 페이지 값 유지
            page += 1
            if (hdr.get("tr_cont") or "") in ("F", "M"):
                tr_cont = "N"
                p[fk_key] = body.get(fk_key.lower(), "")
                p[nk_key] = body.get(nk_key.lower(), "")
            else:
                break
        return {**acc, "ok": ok, "partial": partial, **last}

    async def token(self, force: bool = False) -> str:
        now = time.time()
        if not force:
            # 1) in-memory
            if self._token and now < self._token_exp - self._TOKEN_SAFETY_SEC:
                return self._token
            # 2) disk cache (survives restarts / shared across clients using the same appkey)
            cached = self._load_token_file()
            if cached and now < float(cached.get("expires_at", 0)) - self._TOKEN_SAFETY_SEC:
                self._token = cached["access_token"]; self._token_exp = float(cached["expires_at"])
                return self._token
        # 3) expired (or none, or forced) → issue a fresh one, once, and persist it
        s = await self._s()
        async with s.post(f"{self.base_url}/oauth2/tokenP", json={
            "grant_type":"client_credentials","appkey":self.app_key,"appsecret":self.app_secret}) as r:
            d = await r.json()
        if "access_token" not in d:
            # rate-limited (EGW00133) or transient error → fall back to a still-valid disk token if any.
            # 단, force(=KIS가 방금 그 토큰을 만료로 거부)면 같은 죽은 토큰을 재사용해선 안 된다.
            if not force:
                cached = self._load_token_file()
                if cached and now < float(cached.get("expires_at", 0)) - 60:
                    logger.warning(f"토큰 재발급 실패({d}) — 캐시된 토큰 재사용")
                    self._token = cached["access_token"]; self._token_exp = float(cached["expires_at"])
                    return self._token
            raise Exception(f"토큰 실패: {d}")
        self._token = d["access_token"]; self._token_exp = now + d.get("expires_in", 86400)
        self._save_token_file(self._token, self._token_exp)
        logger.info(f"KIS {'강제 ' if force else ''}신규 토큰 발급 (만료 {datetime.fromtimestamp(self._token_exp, KST):%Y-%m-%d %H:%M})")
        return self._token

    # 모의투자 tr_id: 대부분 실전 첫 글자 'T'→'V'. 단 해외주식 매도만 예외
    # (실전 TTTT1006U → 모의 VTTT1001U, 단순 T→V 아님). 시세성(FH...)은 변환 불필요.
    _MOCK_TR_OVERRIDE = {"TTTT1006U": "VTTT1001U"}

    # (side, exchange) → tr_id.  KRX = 검증된 구 TR 그대로(EXCG 미포함). NXT = 신 통합주문 TR.
    _KR_ORD_TR = {
        ("buy",  "KRX"): "TTTC0802U", ("sell", "KRX"): "TTTC0801U",
        ("buy",  "NXT"): "TTTC0012U", ("sell", "NXT"): "TTTC0011U",
    }

    # NXT 미지원으로 판정할 메시지 시그니처(모의서버). 일반 거부(잔고부족 등)와 구분.
    # 2026-06-08 라이브검증 확정: 모의서버 실제 거부 = "모의투자에서 대체거래소 서비스를 제공하지 않습니다."
    # (msg_cd=41050000) → "대체거래소 서비스"·"제공하지" 로 매칭. 잔고/가격 거부엔 안 나오는 문구라 오탐 없음.
    _NXT_UNSUPPORTED_HINTS = ("미지원", "지원하지", "지원되지", "제공되지", "제공하지",
                              "사용할 수 없", "대체거래소 서비스")

    def _note_nxt_result(self, exchange: str, resp: dict) -> None:
        """NXT 주문 응답으로 거래소 지원 여부 학습. KRX 주문은 무관."""
        if exchange != "NXT":
            return
        if (resp or {}).get("rt_cd") == "0":
            if self._nxt_supported is not True:
                self._nxt_supported = True
            return
        msg = str((resp or {}).get("msg1", ""))   # 원본 msg1 사용 — 힌트 키워드는 정제 전 원문에 들어있음
        if any(h in msg for h in self._NXT_UNSUPPORTED_HINTS):
            if self._nxt_supported is not False:
                logger.warning(f"[NXT] 거래소 미지원 감지 — 시간외 매매 비활성화. msg={msg}")
                self._nxt_supported = False
        # 그 외 거부(잔고부족 등)는 지원 여부 미확정 → 플래그 불변

    def nxt_supported(self):
        return self._nxt_supported

    def _mock_tr(self, tr_id: str) -> str:
        if not self.is_mock or not tr_id:
            return tr_id
        if tr_id in self._MOCK_TR_OVERRIDE:
            return self._MOCK_TR_OVERRIDE[tr_id]
        # KIS 표준(kis_auth): 모의 변환 대상은 첫 글자 T/J/C (시세성 FH… 등은 불변).
        # 사장 지시 표준화 2026-06-01: 'T' 만 변환하던 것을 샘플과 일치시킴(CTRP→VTRP 등).
        return ("V" + tr_id[1:]) if tr_id[0] in ("T", "J", "C") else tr_id

    def _h(self, tok, tr_id):
        from config import PAPER_ONLY
        from urllib.parse import urlsplit
        if PAPER_ONLY and str(tr_id).endswith("U"):
            endpoint = urlsplit(self.base_url)
            if (endpoint.scheme, endpoint.hostname, endpoint.port) != (
                    "https", "openapivts.koreainvestment.com", 29443):
                raise PermissionError("모의매매 전용: 실전 주문·정정·취소·환전 요청을 차단했습니다")
        return {"content-type":"application/json;charset=utf-8","authorization":f"Bearer {tok}",
                "appkey":self.app_key,"appsecret":self.app_secret,"tr_id":self._mock_tr(tr_id)}

    def _acnt(self):
        # Tolerate formats like "12345678-01", "1234567801", with/without spaces.
        digits = "".join(ch for ch in (self.account_no or "") if ch.isdigit())
        cano = digits[:8]
        prdt = digits[8:10] if len(digits) >= 9 else "01"
        return (cano, prdt or "01")

    # ═══════════════════ 국내주식 시세 ═══════════════════
    async def kr_price(self, code: str, market: str = "J") -> Dict:
        # market: J=KRX(기본), NX=NXT, UN=통합 (FID_COND_MRKT_DIV_CODE)
        d = await self._get_json("/uapi/domestic-stock/v1/quotations/inquire-price", "FHKST01010100",
            {"FID_COND_MRKT_DIV_CODE": market, "FID_INPUT_ISCD": code})
        return d.get("output", {})

    async def kr_price_str(self, code: str) -> str:
        d = await self.kr_price(code)
        return (f"[국내시세] {code} | {d.get('stck_prpr','?')}원 | "
                f"전일비: {d.get('prdy_vrss','')} ({d.get('prdy_ctrt','')}%) | "
                f"거래량: {d.get('acml_vol','')}")

    async def kr_daily_chart(self, code: str, days: int = 60) -> List[Dict]:
        """일봉 조회 → CSV 누적"""
        tok = await self.token(); s = await self._s()
        end = datetime.now(KST).strftime("%Y%m%d")
        start = (datetime.now(KST) - timedelta(days=days*2)).strftime("%Y%m%d")
        await self._pace()
        async with s.get(f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
            headers=self._h(tok,"FHKST03010100"),
            params={"FID_COND_MRKT_DIV_CODE":"J","FID_INPUT_ISCD":code,
                    "FID_INPUT_DATE_1":start,"FID_INPUT_DATE_2":end,
                    "FID_PERIOD_DIV_CODE":"D","FID_ORG_ADJ_PRC":"0"}) as r:
            data = (await r.json()).get("output2",[])
        rows = []
        for x in data:
            rows.append({"date":x.get("stck_bsop_date",""),"open":x.get("stck_oprc",""),
                "high":x.get("stck_hgpr",""),"low":x.get("stck_lwpr",""),
                "close":x.get("stck_clpr",""),"volume":x.get("acml_vol","")})
        self._append_csv(f"daily_{code}.csv", rows, ["date","open","high","low","close","volume"])
        return rows

    async def _arcmarket_daily_rows(self, symbol: str, *, kr: bool,
                                    days: int, adjusted: bool = False) -> Optional[List[Dict]]:
        """arcmarket(pykrx/yfinance) EOD 일봉 → KIS 경로와 동일한 row 규격.
        사장 지시 2026-07-02: 시세는 pykrx/yfinance 일원화, 없는 정보만 KIS.
        블로킹 IO 라서 to_thread 로 실행. 실패/미가용이면 None → KIS 폴백."""
        if arcmarket is None:
            return None
        def _fetch():
            df = (arcmarket.kr_daily(symbol, days=days) if kr
                  else arcmarket.us_daily(symbol, days=days, adjusted=adjusted))
            if df is None or df.empty:
                return None
            df = df.fillna(0)
            rows = []
            for d, x in df.iterrows():
                if kr:
                    rows.append({"date": d.strftime("%Y-%m-%d"),
                                 "open": int(x["open"]), "high": int(x["high"]),
                                 "low": int(x["low"]), "close": int(x["close"]),
                                 "volume": int(x["volume"])})
                else:
                    rows.append({"date": d.strftime("%Y-%m-%d"),
                                 "open": float(x["open"]), "high": float(x["high"]),
                                 "low": float(x["low"]), "close": float(x["close"]),
                                 "volume": int(x["volume"])})
            return rows or None
        try:
            return await asyncio.to_thread(_fetch)
        except Exception as e:
            logger.warning(f"[arcmarket] {symbol} EOD 조회 실패 — KIS 폴백: {e}")
            return None

    async def kr_daily_chart_deep(self, code: str, years: int = 2, max_calls: int = 10) -> List[Dict]:
        """국내 일봉 깊은 수집(~years년) → CSV 누적.
        1순위 arcmarket(pykrx/yfinance — 사장 지시 2026-07-02 시세 일원화),
        실패 시 기존 KIS 날짜 윈도우 페이지네이션 폴백 (호출당 ~100행)."""
        via = await self._arcmarket_daily_rows(code, kr=True, days=int(max(1, years) * 365))
        if via:
            self._append_csv(f"daily_{code}.csv", via, ["date", "open", "high", "low", "close", "volume"])
            logger.info(f"[일봉deep] {code}: arcmarket {len(via)}건 (pykrx/yfinance)")
            return via
        tok = await self.token(); s = await self._s()
        target = (datetime.now(KST) - timedelta(days=int(max(1, years) * 365))).strftime("%Y%m%d")
        win_end = datetime.now(KST)
        all_rows: List[Dict] = []
        seen: set = set()
        for _ in range(max(1, max_calls)):
            e = win_end.strftime("%Y%m%d")
            b = (win_end - timedelta(days=150)).strftime("%Y%m%d")  # ≈100 거래일
            try:
                await self._pace()
                async with s.get(f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
                    headers=self._h(tok, "FHKST03010100"),
                    params={"FID_COND_MRKT_DIV_CODE":"J","FID_INPUT_ISCD":code,
                            "FID_INPUT_DATE_1":b,"FID_INPUT_DATE_2":e,
                            "FID_PERIOD_DIV_CODE":"D","FID_ORG_ADJ_PRC":"0"}) as r:
                    data = (await r.json()).get("output2", []) or []
            except Exception as ex:
                logger.warning(f"[일봉deep] {code} 조회 예외: {ex}")
                break
            win_rows = []
            for x in data:
                d = (x.get("stck_bsop_date") or "").strip()
                if len(d) != 8 or d in seen:
                    continue
                seen.add(d)
                win_rows.append({"date": f"{d[:4]}-{d[4:6]}-{d[6:8]}",
                    "open":x.get("stck_oprc",""),"high":x.get("stck_hgpr",""),
                    "low":x.get("stck_lwpr",""),"close":x.get("stck_clpr",""),
                    "volume":x.get("acml_vol","")})
            if not win_rows:
                break
            all_rows.extend(win_rows)
            oldest = min(r["date"].replace("-", "") for r in win_rows)
            if oldest <= target:
                break
            win_end = datetime.strptime(oldest, "%Y%m%d") - timedelta(days=1)
            await asyncio.sleep(0.25)  # KIS 유량 보호
        if all_rows:
            all_rows.sort(key=lambda r: r["date"])
            self._append_csv(f"daily_{code}.csv", all_rows, ["date","open","high","low","close","volume"])
        return all_rows

    async def kr_minute_chart(self, code: str, interval: str = "1") -> List[Dict]:
        """분봉 조회 → CSV 누적"""
        tok = await self.token(); s = await self._s()
        now = datetime.now(KST).strftime("%H%M%S")
        await self._pace()
        async with s.get(f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice",
            headers=self._h(tok,"FHKST03010200"),
            params={"FID_COND_MRKT_DIV_CODE":"J","FID_INPUT_ISCD":code,
                    "FID_INPUT_HOUR_1":now,"FID_PW_DATA_INCU_YN":"Y"}) as r:
            data = (await r.json()).get("output2",[])
        rows = []
        for x in data:
            rows.append({"datetime":f"{x.get('stck_bsop_date','')}{x.get('stck_cntg_hour','')}",
                "open":x.get("stck_oprc",""),"high":x.get("stck_hgpr",""),
                "low":x.get("stck_lwpr",""),"close":x.get("stck_prpr",""),
                "volume":x.get("cntg_vol","")})
        self._append_csv(f"minute_{code}.csv", rows, ["datetime","open","high","low","close","volume"])
        return rows

    # ═══════════════════ 국내주식 주문 ═══════════════════
    async def kr_buy(self, code: str, qty: int, price: int = 0, exchange: str = "KRX") -> str:
        raw_price = float(price or 0)
        price = ceil_to_tick(raw_price)
        if raw_price > 0 and price != raw_price:
            logger.info("[국내매수] 지정가 호가단위 올림: %s %s → %s원", code, raw_price, price)
        s = await self._s(); c, p = self._acnt()
        # 사장 지시 2026-05-19: 지정가(price>0)면 ORD_DVSN="00"(지정가), 없으면 "01"(시장가).
        # 기존 '"01" if price else "01"'은 지정가 주문도 시장가로 체결시키던 버그.
        ord_dvsn = "00" if (price or (exchange == "NXT")) else "01"   # NXT는 시장가 미지원→지정가 강제
        body = {"CANO":c,"ACNT_PRDT_CD":p,"PDNO":code,"ORD_DVSN":ord_dvsn,
                "ORD_QTY":str(qty),"ORD_UNPR":str(price) if price else "0","CTAC_TLNO":""}
        if exchange == "NXT":
            body["EXCG_ID_DVSN_CD"] = "NXT"; body["CNDT_PRIC"] = ""
        tr = self._KR_ORD_TR[("buy", exchange)]
        async def _do(tk):
            async with s.post(f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash",
                headers=self._h(tk, tr), json=body) as r:
                return await r.json()
        d = await self._authed_json(_do)
        self._note_nxt_result(exchange, d)
        return f"[국내매수] {code} {qty}주 → {_clean_kis_msg(d.get('msg1',''))}" if d.get("rt_cd")=="0" else f"[실패] {_clean_kis_msg(d.get('msg1',''))}"

    async def kr_pending_orders(self, code: Optional[str] = None) -> List[Dict]:
        """정정·취소가능 주문 조회(TTTC0084R). code 주면 해당 종목으로 필터.
        반환 각 행: {odno, ord_gno_brno, pdno, ord_qty, ord_unpr, ord_dvsn_cd, sll_buy_dvsn_cd, ord_tmd, ...}.
        sll_buy_dvsn_cd: '01'=매도, '02'=매수.
        """
        s = await self._s(); c, p = self._acnt()
        async def _do(tk):
            out: List[Dict] = []
            fk = ""; nk = ""
            for page in range(5):
                if page:
                    await self._pace()
                async with s.get(f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-psbl-rvsecncl",
                    headers=self._h(tk, "TTTC0084R"),
                    params={"CANO":c,"ACNT_PRDT_CD":p,
                            "CTX_AREA_FK100":fk,"CTX_AREA_NK100":nk,
                            "INQR_DVSN_1":"0","INQR_DVSN_2":"0"}) as r:
                    d = await r.json()
                if d.get("rt_cd") != "0":
                    return {"ok": False, "rows": []}
                out.extend(d.get("output") or [])
                fk = (d.get("ctx_area_fk100") or "").strip()
                nk = (d.get("ctx_area_nk100") or "").strip()
                if not nk:
                    break
            return {"ok": True, "rows": out}
        try:
            d = await self._authed_json(_do)
        except Exception as e:
            logger.warning(f"[펜딩조회] 실패: {e}")
            return []
        if not d.get("ok"):
            return []
        rows = d.get("rows") or []
        if code:
            t = (code or "").lstrip("0")
            rows = [r for r in rows if ((r.get("pdno") or "").lstrip("0") == t)]
        return rows

    async def kr_cancel(self, order: Dict) -> str:
        """KR 펜딩 주문 취소(TTTC0803U). order = kr_pending_orders() 의 한 행."""
        s = await self._s(); c, p = self._acnt()
        body = {
            "CANO": c, "ACNT_PRDT_CD": p,
            "KRX_FWDG_ORD_ORGNO": (order.get("ord_gno_brno") or "").strip(),
            "ORGN_ODNO": (order.get("odno") or "").strip(),
            "ORD_DVSN": (order.get("ord_dvsn_cd") or order.get("ord_dvsn") or "00").strip(),
            "RVSE_CNCL_DVSN_CD": "02",            # 02=취소
            "ORD_QTY": "0",
            "ORD_UNPR": "0",
            "QTY_ALL_ORD_YN": "Y",                # 전량 취소
        }
        async def _do(tk):
            async with s.post(f"{self.base_url}/uapi/domestic-stock/v1/trading/order-rvsecncl",
                headers=self._h(tk, "TTTC0803U"), json=body) as r:
                return await r.json()
        d = await self._authed_json(_do)
        pdno = order.get("pdno", ""); odno = order.get("odno", "")
        return (f"[국내취소] {pdno} {odno} → {_clean_kis_msg(d.get('msg1',''))}"
                if d.get("rt_cd") == "0"
                else f"[취소실패] {pdno} {odno} → {_clean_kis_msg(d.get('msg1',''))}")

    async def kr_sell(self, code: str, qty: int, price: int = 0, exchange: str = "KRX") -> str:
        raw_price = float(price or 0)
        price = ceil_to_tick(raw_price)
        if raw_price > 0 and price != raw_price:
            logger.info("[국내매도] 지정가 호가단위 올림: %s %s → %s원", code, raw_price, price)
        # 사장 지시 2026-05-28: 새 매도 판단이 들어오면 같은 종목의 살아있는 펜딩 매도는 폐기하고 신규로 대체.
        # 배경: KIS는 펜딩 주문이 ord_psbl_qty(매도가능수량)를 깎아, 보유 1주에 28,000원 펜딩 매도가 있으면
        # 후속 매도 시도가 모두 "주문 가능한 수량을 초과했습니다"로 거부된다(003490 사례 5/28 14:09·15:20).
        # 펜딩 취소가 실패해도 신규 매도는 어떻게든 전송(다중 폴백 — 사장 룰).
        try:
            pending = await self.kr_pending_orders(code)
            for r in pending:
                if (r.get("sll_buy_dvsn_cd") or "").strip() == "01":
                    cres = await self.kr_cancel(r)
                    logger.info(f"[국내매도] 사전 펜딩 취소: {cres}")
        except Exception as e:
            logger.warning(f"[국내매도] 펜딩 취소 시도 실패(무시하고 신규 전송): {e}")
        s = await self._s(); c, p = self._acnt()
        # 사장 지시 2026-05-19: 지정가(price>0)면 ORD_DVSN="00"(지정가), 없으면 "01"(시장가).
        # 기존 '"01" if price else "01"'은 지정가 매도도 시장가로 체결시키던 버그.
        ord_dvsn = "00" if (price or (exchange == "NXT")) else "01"   # NXT는 시장가 미지원→지정가 강제
        body = {"CANO":c,"ACNT_PRDT_CD":p,"PDNO":code,"ORD_DVSN":ord_dvsn,
                "ORD_QTY":str(qty),"ORD_UNPR":str(price) if price else "0","CTAC_TLNO":""}
        if exchange == "NXT":
            body["EXCG_ID_DVSN_CD"] = "NXT"; body["SLL_TYPE"] = "01"; body["CNDT_PRIC"] = ""
        tr = self._KR_ORD_TR[("sell", exchange)]
        async def _do(tk):
            async with s.post(f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash",
                headers=self._h(tk, tr), json=body) as r:
                return await r.json()
        d = await self._authed_json(_do)
        self._note_nxt_result(exchange, d)
        return f"[국내매도] {code} {qty}주 → {_clean_kis_msg(d.get('msg1',''))}" if d.get("rt_cd")=="0" else f"[실패] {_clean_kis_msg(d.get('msg1',''))}"

    # ── one canonical balance read (paginated) → cached 8s; everything else derives from it ──
    _SNAP_TTL = 8.0
    # 보유=0인데 (총평가 − 예수금)이 이 값(원)보다 크면 KIS 빈-보유 글리치로 보고 직전 보유를 유지.
    _HOLDINGS_GLITCH_MIN_GAP = 5000.0
    async def _raw_balance(self) -> Dict:
        """Single inquire-balance call with tr_cont pagination. Returns {output1:[...],output2:{...},ok,rt_cd,msg1}.
        토큰 만료 거부 시 _authed_json 이 강제 재발급+1회 재시도(잔고 0원 글리치 자가치유)."""
        c, p = self._acnt()
        async def _do(tok):
            s = await self._s()
            out1: List[Dict] = []; out2: Dict = {}; rt = ""; msg = ""
            fk = ""; nk = ""; tr_cont = ""
            for page in range(5):
                if page:
                    await self._pace()
                headers = self._h(tok, "TTTC8434R")
                if tr_cont in ("F", "M"):
                    headers["tr_cont"] = "N"
                async with s.get(f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance",
                    headers=headers,
                    params={"CANO":c,"ACNT_PRDT_CD":p,"AFHR_FLPR_YN":"N","OFL_YN":"","INQR_DVSN":"02",
                            "UNPR_DVSN":"01","FUND_STTL_ICLD_YN":"N","FNCG_AMT_AUTO_RDPT_YN":"N",
                            "PRCS_DVSN":"01","CTX_AREA_FK100":fk,"CTX_AREA_NK100":nk}) as r:
                    tr_cont = r.headers.get("tr_cont","")
                    d = await r.json()
                rt = d.get("rt_cd",""); msg = d.get("msg1","")
                if rt != "0":
                    break
                out1.extend(d.get("output1") or [])
                o2list = d.get("output2") or []
                if o2list: out2 = o2list[0]
                fk = (d.get("ctx_area_fk100") or "").strip(); nk = (d.get("ctx_area_nk100") or "").strip()
                if tr_cont not in ("F", "M") or not nk:
                    break
            return {"output1": out1, "output2": out2, "ok": (rt == "0"), "rt_cd": rt, "msg1": msg}
        d = await self._authed_json(_do)
        o2 = d.get("output2") or {}
        logger.info(f"[잔고] rt_cd={d.get('rt_cd')} msg='{d.get('msg1')}' output1_rows={len(d.get('output1') or [])} "
                    f"tot_evlu={o2.get('tot_evlu_amt','?')} dnca={o2.get('dnca_tot_amt','?')}")
        return d

    async def kr_account_snapshot(self, force: bool = False) -> Dict:
        """{"buying_power":{cash,total_eval,pnl_ratio,ok}, "holdings":[{code,name,qty,avg_price,cur_price,pnl_amt,pnl_pct}], "ok":bool, "ts":float}.
        Cached for _SNAP_TTL seconds so repeated callers (cycle + /api/balance) don't double-hit KIS."""
        now = time.time()
        cached = getattr(self, "_acct_snap", None)
        if cached and not force and (now - cached.get("ts", 0)) < self._SNAP_TTL:
            return cached
        _f, _i = self._num, self._int   # 숫자 파서는 _num/_int(staticmethod)로 일원화
        try:
            d = await self._raw_balance()
        except Exception as e:
            logger.warning(f"[잔고스냅샷] 실패: {e}")
            d = {"output1": [], "output2": {}, "ok": False}
        o2 = d.get("output2") or {}
        # ── Cash: D+2 정산예수금(prvs_rcdl_excc_amt, 매도대금 반영분)이 실제 가용·평가 기준이다.
        #    D0(dnca_tot_amt)는 미결제 매수분이 안 빠져 부풀려져 자산곡선을 튀게 하므로 평가·사이징에
        #    절대 쓰지 않는다(사장 지시 2026-05-28). D+2 가 0 으로 깜빡이면 직전 정상 D+2 를 유지. ──
        cash_d2 = _f(o2.get("prvs_rcdl_excc_amt")); cash_d1 = _f(o2.get("nxdy_excc_amt"))
        scts = _f(o2.get("scts_evlu_amt"))   # 국내 유가증권평가금액
        # 현재 평가액(자산곡선) = 국내 유가증권평가액 + D+2 예수금. 해외(외화평가총액)는 portfolio_holdings 가 더한다.
        # 사장 지시 2026-05-28: D+2(prvs_rcdl_excc_amt)가 정상 정산예수금이다. KIS가 결제 과도기에 D+2/D+1 을
        # 0 으로 깜빡이면 직전 정상 D+2 를 유지(disk 영속) — D0(dnca_tot_amt)로 폴백하면 미결제 매수분이 아직
        # 안 빠져 부풀려진 값이라 자산곡선이 스파이크 친다(hh09080 +4.9M 유령점프, 사장 보고 2026-05-28).
        _prev_settled = self._get_settled_cash_cache()
        total, settled = kr_net_valuation(scts, cash_d2, cash_d1, _prev_settled)
        cash = settled                       # 예수금=D+2(정산). 주문 사이징도 D0 부풀림 없이 실제 가용분만 쓴다.
        if cash_d2 > 0:
            self._set_settled_cash_cache(cash_d2)
        pnl = _f(o2.get("evlu_pfls_smtl_amt"))
        pnl_ratio = (pnl / total) if total > 0 else 0.0
        holdings = []
        for h in (d.get("output1") or []):
            q = _i(h.get("hldg_qty"))
            if q <= 0:
                continue
            holdings.append({"code": (h.get("pdno") or "").strip(), "name": (h.get("prdt_name") or "").strip(), "qty": q,
                             "avg_price": _f(h.get("pchs_avg_pric")), "cur_price": _f(h.get("prpr")),
                             "pnl_amt": _f(h.get("evlu_pfls_amt")), "pnl_pct": _f(h.get("evlu_pfls_rt")),
                             # 사장 지시 2026-06-11(라이브 진단): 매도가능수량 — 미체결 매도 주문이
                             # 물량을 잠그면 hldg_qty>0 인데 ord_psbl_qty=0 이 되어, 보유수량 기준
                             # 매도가 '잔고내역이 없습니다'로 반복 거부된다(uid2 041830 사례).
                             "sellable_qty": _i(h.get("ord_psbl_qty"))})
        snap = {"buying_power": {"cash": cash, "total_eval": total, "pnl_ratio": pnl_ratio, "ok": bool(d.get("ok"))},
                "holdings": holdings, "ok": bool(d.get("ok")), "ts": now}
        # Keep last-good holdings if this read succeeded structurally but came back empty while
        # total_eval clearly implies positions exist (transient KIS quirk) — avoids UI flicker.
        # 직전 정상 보유목록: in-memory 우선, 없으면(재시작 콜드스타트) 디스크 캐시로 폴백(사장 제보 2026-05-29).
        _prev_h = (cached.get("holdings") if cached else None)
        # 글리치 carry-forward 기준은 KR 기준 총평가(total_eval_kr)다 — 아래에서 total_eval 에 해외분을
        # 더하므로, _prev_total 이 해외 포함값을 집으면 글리치 복원 시 해외분이 이중 가산된다.
        _prev_bp_c = (cached.get("buying_power") or {}) if cached else {}
        _prev_total = float(_prev_bp_c.get("total_eval_kr") or _prev_bp_c.get("total_eval") or 0.0)
        if not _prev_h:
            _dh, _dt, _dts = self._get_holdings_cache()
            if _dh and (now - _dts) < self._HOLDINGS_CACHE_TTL:
                _prev_h = _dh
                if _prev_total <= 0:
                    _prev_total = _dt
        if snap["ok"] and not holdings and total > cash + self._HOLDINGS_GLITCH_MIN_GAP and _prev_h:
            snap["holdings"] = _prev_h
            snap["holdings_stale"] = True
            # 사장 지시 2026-05-21: KIS가 보유목록을 빈 채 주면서 nass_amt(총평가)를 부풀려
            # 반환하는 글리치 폴(보유=0인데 총평가−예수금이 큼) → 자산곡선·주문 사이징이 튄다.
            # 보유목록뿐 아니라 총평가도 직전 정상 값으로 유지해 안정화한다.
            if _prev_total > 0:
                snap["buying_power"]["total_eval"] = _prev_total
                snap["buying_power"]["total_stale"] = True
        elif not snap["ok"] and _prev_h:
            # 조회 실패(rt_cd≠0/예외, 모의서버 토큰 rate-limit 등): 빈 결과로 다운스트림
            # (사이클 사전 게이트·사후관리실장 매도 평가)을 오염시키지 말고 직전 정상 보유목록·총평가·
            # 예수금을 유지한다(stale). 매수는 guardrails 가 ok=False 를 보고 여전히 보수적으로 반려하므로
            # 이 carry-forward 는 사이클 진행·매도 평가만 살리고 오발주를 만들지 않는다. (사장 제보 2026-05-29)
            snap["holdings"] = _prev_h
            snap["holdings_stale"] = True
            if _prev_total > 0:
                snap["buying_power"]["total_eval"] = _prev_total
                snap["buying_power"]["total_stale"] = True
            _prev_cash = float((cached.get("buying_power") or {}).get("cash") or 0.0) if cached else 0.0
            if _prev_cash <= 0:
                _prev_cash = self._get_settled_cash_cache()
            if _prev_cash > 0:
                snap["buying_power"]["cash"] = _prev_cash
                snap["buying_power"]["cash_stale"] = True
        # 결제예수금 글리치 방어 (cash 가 0 으로 resolve = D+2/D+1 모두 0 + last-good 캐시도 없음):
        # 평소엔 kr_net_valuation 의 D+2 carry-forward(disk 영속)가 막지만, 콜드스타트/깊은 글리치로
        # settled 가 0 이 되면 cash·total 이 함께 무너진다(2026-05-14 cash=0 버그, 2026-05-22 결제 과도기).
        # 이 때만 직전 정상 스냅샷(cash·total)을 유지해 자산곡선 스파이크를 막는다. D0 로는 절대 폴백하지 않는다.
        if cached and snap["ok"] and cash <= 0:
            _prev_bp = cached.get("buying_power") or {}
            prev_cash = float(_prev_bp.get("cash") or 0.0)
            prev_total = float(_prev_bp.get("total_eval_kr") or _prev_bp.get("total_eval") or 0.0)
            if prev_cash > 0:
                snap["buying_power"]["cash"] = prev_cash
                snap["buying_power"]["cash_stale"] = True
            if prev_total > 0:
                snap["buying_power"]["total_eval"] = prev_total
                snap["buying_power"]["total_stale"] = True
            logger.warning(f"[잔고스냅샷] 결제예수금=0 글리치 — 직전 정상값 유지 "
                           f"(cash={prev_cash:,.0f}, total={prev_total:,.0f})")
        # 보유목록 디스크 영속(재시작 갭 방어, 사장 제보 2026-05-29): 정상(보유 있음) 읽기면 last-good 저장,
        # 진짜 평탄(빈 보유 + 총평가≈예수금)이면 무효화(유령 보유 방지). 글리치(빈 보유 + 총평가≫예수금)면 캐시 유지.
        if holdings:
            self._set_holdings_cache(holdings, total, now)
        elif snap["ok"] and total <= cash + self._HOLDINGS_GLITCH_MIN_GAP:
            self._clear_holdings_cache()
        # 사장 지시 2026-06-03: 사이클/리스크/표시용 총평가는 KR(국내 유가증권 + D+2 예수금)에
        # 해외 외화평가총액(frcr_evlu_tota 원화환산)을 합산한다. KR nass_amt만 쓰면 US 보유가 0으로
        # 사라져 자산곡선이 매수 때마다 계단식 하락하고 pnl 이 0 에 고정된다.
        #   - total_eval_kr: KR 기준(글리치 carry-forward·곡선 합산의 base). total_eval: KR+해외(헤드라인).
        #   - 해외분은 portfolio_holdings 가 매 폴링마다 갱신·디스크 영속하는 캐시값을 쓴다(추가 호출 0).
        #   - #2(0/동결 금지): 캐시는 실패 시 직전값을 보존하고, 실제 US 전량매도 때만 0으로 무효화되므로
        #     '>0' 이면 항상 더한다(TTL 무시) — US 보유가 일시 조회실패로 사라지지 않게.
        _kr_total = self._num(snap["buying_power"]["total_eval"])
        snap["buying_power"]["total_eval_kr"] = _kr_total
        _ov_krw, _ov_ts = self._get_overseas_cache()
        # 2026-07-29: `>0` 였던 가드는 **음수 해외분(모의 통합증거금 USD 부채)을 통째로 버렸다** —
        # 캐시가 -28.99M 인데 안 더해져 사이클/사이징이 보는 총평가가 71M 대신 100M 이 됐다.
        # 0(=해외 없음)은 더해도 무의미하므로 '0이 아니면' 부호 그대로 반영한다.
        if _ov_krw:
            snap["buying_power"]["total_eval"] = _kr_total + _ov_krw
            snap["buying_power"]["overseas_krw"] = _ov_krw
        self._acct_snap = snap
        return snap

    async def kr_balance(self) -> str:
        snap = await self.kr_account_snapshot()
        bp = snap["buying_power"]
        if not bp["ok"]:
            return f"[잔고실패] KIS inquire-balance rt_cd≠0"
        lines = [f"[국내 계좌잔고] 예수금: {bp['cash']:,.0f}원 | 총평가: {bp['total_eval']:,.0f}원 | 평가손익률: {bp['pnl_ratio']*100:.2f}%\n"]
        for h in snap["holdings"]:
            lines.append(f"  📊 {h['name']} ({h['code']}): {h['qty']}주 | 평단 {h['avg_price']:,.0f} | 현재 {h['cur_price']:,.0f} | 손익 {h['pnl_amt']:,.0f} ({h['pnl_pct']:+.2f}%)")
        return "\n".join(lines)

    async def kr_holdings(self) -> List[Dict]:
        return (await self.kr_account_snapshot())["holdings"]

    async def kr_buying_power(self) -> Dict[str, float]:
        return (await self.kr_account_snapshot())["buying_power"]

    async def kr_last_price(self, code: str, market: str = "J") -> float:
        """Current price as a float. KIS primary → 네이버 금융 폴백 (사장 지시 2026-05-14 — 028670 0원 이슈 해결).
        market: J=KRX(기본), NX=NXT, UN=통합. 네이버 폴백은 KRX 근사로 fallback 허용.
        Returns 0.0 only when both sources fail."""
        try:
            d = await self.kr_price(code, market=market)
            px = self._num(d.get("stck_prpr"))
            if px > 0:
                return px
        except Exception as e:
            logger.warning(f"[가격] KIS 조회 예외 ({code}): {e}")
        # Fallback: Naver Finance polling API (KIS가 0/실패 반환 시)
        try:
            loop = asyncio.get_event_loop()
            from tools.market_data import kr_price_naver
            px = await loop.run_in_executor(None, kr_price_naver, code)
            if px > 0:
                logger.info(f"[가격] {code} KIS 0원 → 네이버 폴백 성공: {px:,.0f}원")
            return float(px)
        except Exception as e:
            logger.warning(f"[가격] 네이버 폴백 실패 ({code}): {e}")
            return 0.0

    # ═══════════════════ 통합 종목 포트폴리오 (국내주식+해외주식+국내채권+펀드) ═══════════════════
    @staticmethod
    def _num(x):
        try: return float(str(x).replace(",",""))
        except (TypeError, ValueError): return 0.0
    @staticmethod
    def _int(x):
        try: return int(float(str(x).replace(",","")))
        except (TypeError, ValueError): return 0

    async def _overseas_holdings(self) -> List[Dict]:
        """해외주식 보유 종목 (NASD/NYSE/AMEX 순회). 실패/미보유 시 [].
        사장 피드백 2026-05-20(2차): KIS는 동일 보유분을 거래소별 inquire-balance 응답마다
        중복 노출한다(예: UUP가 NASD·AMEX 양쪽에 등장). 따라서 code 기준으로 **중복 제거**(첫 건만)
        해야 한다 — 이전엔 qty를 합산해 2주가 4주로 부풀던 버그. 실보유는 present-balance로 검증함."""
        seen: Dict[str, Dict] = {}
        # [] 하나만으로는 '실제 전량매도'와 '세 거래소 조회가 모두 실패'를 구분할 수 없다.
        # portfolio_holdings 의 모의 원장 정정은 세 거래소가 모두 정상 응답한 빈 잔고에서만
        # 허용한다. 매 호출 시작 때 False 로 되돌려 이전 성공 상태가 새 실패에 남지 않게 한다.
        self._overseas_holdings_authoritative = False
        successful_exchanges = 0
        try:
            c, p = self._acnt()
            for excd in ("NASD", "NYSE", "AMEX"):
                try:
                    # 사장 지시 2026-06-01: 거래소별 연속조회(tr_cont 페이징) — 1페이지(~100건) 초과
                    # 보유가 통째 누락되던 KR/US 비대칭 버그 수정(국내 _raw_balance 와 동형). 부분성공 보존.
                    d = await self._paged_get(
                        "/uapi/overseas-stock/v1/trading/inquire-balance", "TTTS3012R",
                        {"CANO": c, "ACNT_PRDT_CD": p, "OVRS_EXCG_CD": excd, "TR_CRCY_CD": "USD",
                         "CTX_AREA_FK200": "", "CTX_AREA_NK200": ""},
                        fk_key="CTX_AREA_FK200", nk_key="CTX_AREA_NK200", out_keys=("output1",))
                    if not d.get("ok") or d.get("partial"):
                        continue
                    successful_exchanges += 1
                    for h in (d.get("output1") or []):
                        q = self._int(h.get("ovrs_cblc_qty"))
                        if q <= 0:
                            continue
                        code = (h.get("ovrs_pdno") or "").strip()
                        if not code or code in seen:
                            continue   # 거래소 간 중복 노출 — 첫 건만 채택(합산 금지)
                        seen[code] = {"code": code,
                                      "name": (h.get("ovrs_item_name") or "").strip() or code,
                                      "qty": q, "avg_price": self._num(h.get("pchs_avg_pric")),
                                      "cur_price": self._num(h.get("now_pric2")),
                                      "pnl_amt": self._num(h.get("frcr_evlu_pfls_amt") or h.get("evlu_pfls_amt")),
                                      "pnl_pct": self._num(h.get("evlu_pfls_rt")),
                                      "category": "해외주식", "ccy": "USD"}
                except Exception:
                    continue
        except Exception:
            pass
        self._overseas_holdings_authoritative = (successful_exchanges == 3)
        return list(seen.values())

    async def _bond_holdings(self) -> List[Dict]:
        """장내채권 보유 잔고. API 미지원/실패 시 []."""
        out: List[Dict] = []
        try:
            tok = await self.token(); s = await self._s(); c, p = self._acnt()
            await self._pace()
            async with s.get(f"{self.base_url}/uapi/domestic-bond/v1/trading/inquire-balance",
                headers=self._h(tok,"CTSC8407R"),
                params={"CANO":c,"ACNT_PRDT_CD":p,"INQR_CNDT":"00","PDNO":"","BUY_DT":"",
                        "CTX_AREA_FK200":"","CTX_AREA_NK200":""}) as r:
                d = await r.json()
            for h in (d.get("output1") or d.get("output") or []):
                if not isinstance(h, dict):
                    continue
                q = self._int(h.get("cblc_qty") or h.get("hldg_qty") or h.get("nrcvb_buy_qty"))
                if q <= 0:
                    continue
                out.append({"code": (h.get("pdno") or "").strip(),
                            "name": (h.get("prdt_name") or "").strip() or (h.get("pdno") or "").strip(),
                            "qty": q, "avg_price": self._num(h.get("buy_unpr") or h.get("pchs_avg_pric")),
                            "cur_price": self._num(h.get("prpr") or h.get("evlu_pric") or h.get("bond_prpr")),
                            "pnl_amt": self._num(h.get("evlu_pfls_amt")), "pnl_pct": self._num(h.get("evlu_pfls_rt")),
                            "category": "국내채권", "ccy": "KRW"})
        except Exception:
            pass
        return out

    async def _fund_holdings(self) -> List[Dict]:
        """펀드 보유 잔고. KIS OpenAPI는 종목 단위 펀드 잔고 조회를 공개하지 않아 빈 목록 반환(안전).
        (국내주식 inquire-balance의 FUND_STTL_ICLD_YN='Y'는 펀드결제분 '금액'만 포함하며 보유내역이 아님.)"""
        return []

    async def _overseas_present_krw(self) -> Dict:
        """해외주식 원화환산 평가합계 + 기준환율 (CTRP6504R inquire-present-balance).
        사장 피드백 2026-05-20: 국내 inquire-balance 의 nass_amt(총평가)는 해외주식 평가를
        포함하지 않으므로, 통합 총평가 보정에 쓸 해외 원화평가합계를 별도 조회한다.
        실패/모의투자 미지원 시 {ok:False}. (실전 전용 — 모의 base_url 이면 시도하되 실패 허용.)
        사장 지시 2026-06-01: NATN_CD '000'(전체국가)로 호출(미국 외 보유 누락 방지)하고, 같은 응답의
        output3 통합총자산(tot_asst_amt)·총예수금·총평가손익, output2 외화예수금(frcr_dncl_amt_2)까지 파싱해
        반환한다(추가 호출 0) — 통합총자산 교차검증·예수금 직접산출에 쓴다."""
        try:
            c, p = self._acnt()
            d = await self._get_json("/uapi/overseas-stock/v1/trading/inquire-present-balance", "CTRP6504R",
                {"CANO": c, "ACNT_PRDT_CD": p, "WCRC_FRCR_DVSN_CD": "02", "NATN_CD": "000",
                 "TR_MKET_CD": "00", "INQR_DVSN_CD": "00"})
            if str(d.get("rt_cd", "")) != "0":
                return {"ok": False, "krw_value": 0.0, "stock_value": 0.0, "exrt": 0.0}
            o3 = d.get("output3") or {}
            if isinstance(o3, list):
                o3 = o3[0] if o3 else {}
            # 사장 지시 2026-05-28: 외화평가총액 = frcr_evlu_tota (USD 예수금 포함, 원화환산).
            # 이전엔 evlu_amt_smtl_amt(해외 '주식만' 평가합계)를 더해 USD 예수금 484불(~727K)이 빠져
            # 우리 화면이 KIS 앱 총자산보다 ~737K 낮게 표시됐다(사장 보고 2026-05-28).
            krw_value = self._num(o3.get("frcr_evlu_tota"))
            if krw_value <= 0:
                # 스키마 변동·모의계좌·USD 미보유 → 이전 필드로 폴백
                krw_value = self._num(o3.get("evlu_amt_smtl_amt"))
            # 사장 지시 2026-05-30: 주식분(예수금 제외) 평가액 — 권위 조회 실패 시 라이브 가격으로
            # 주식분만 재계산하고 예수금분(총액−주식분)은 캐시값을 보존하기 위함(자산곡선 동결 방지).
            stock_value = self._num(o3.get("evlu_amt_smtl_amt"))
            o2 = d.get("output2") or []
            if isinstance(o2, dict):
                o2 = [o2]
            exrt = self._num(o2[0].get("frst_bltn_exrt")) if o2 else 0.0
            # 사장 지시 2026-06-01: 외화예수금을 역산(총액−주식분) 대신 직접 필드로 — 시점차·환율차 누적오차 제거.
            deposit_frcr = self._num(o2[0].get("frcr_dncl_amt_2")) if o2 else 0.0
            deposit_krw = deposit_frcr * exrt if exrt > 0 else 0.0
            return {"ok": True, "krw_value": krw_value, "stock_value": stock_value, "exrt": exrt,
                    "tot_asst_amt": self._num(o3.get("tot_asst_amt")),
                    "tot_dncl_amt": self._num(o3.get("tot_dncl_amt")),
                    "tot_evlu_pfls_amt": self._num(o3.get("tot_evlu_pfls_amt")),
                    "deposit_frcr": deposit_frcr,   # 외화예수금 원통화(USD) — USD→KRW 역환전 판단용
                    "deposit_krw": deposit_krw}
        except Exception as e:
            logger.warning(f"[해외원화평가] CTRP6504R 실패: {e}")
            return {"ok": False, "krw_value": 0.0, "stock_value": 0.0, "exrt": 0.0}

    async def idle_usd_deposit(self) -> Dict:
        """유휴 USD 예수금(원통화·원화환산·기준환율) — read-only. USD→KRW 역환전 판단 전용.
        CTRP6504R(_overseas_present_krw)의 외화예수금(frcr_dncl_amt_2)을 그대로 노출한다.
        주문·표시·자산곡선에 무영향. 실패/모의/비정상환율 → {ok:False}.
        ※ KRW 한도와 섞지 말 것: 여기서 주는 usd 는 'USD 평가(예수금)'다."""
        try:
            pk = await self._overseas_present_krw()
            if not pk.get("ok"):
                return {"ok": False, "usd": 0.0, "krw_value": 0.0, "exrt": 0.0}
            exrt = float(pk.get("exrt") or 0.0)
            usd = float(pk.get("deposit_frcr") or 0.0)
            krw = float(pk.get("deposit_krw") or 0.0)
            # exrt 비정상(모의서버 garbage <500)이면 USD 환산을 신뢰 불가 — 0 처리(보수적, 환전 오발 방지).
            if exrt and exrt < 500:
                return {"ok": False, "usd": 0.0, "krw_value": 0.0, "exrt": exrt}
            return {"ok": True, "usd": usd, "krw_value": krw, "exrt": exrt}
        except Exception as e:
            logger.warning(f"[유휴USD조회] 실패: {e}")
            return {"ok": False, "usd": 0.0, "krw_value": 0.0, "exrt": 0.0}

    async def us_to_krw_exchange(self, usd_amount: float, *, dry_run: bool = True,
                                 reason: str = "") -> Dict:
        """USD→KRW 역환전 '실행' 단일 진입점. **실제 환전은 반드시 여기 한 곳에서만** 수행한다.

        현실(2026-06-26 확인): KIS OpenAPI 는 공개 '환전' TR/엔드포인트를 제공하지 않는다 — 공식
        open-trading-api 저장소·본 코드베이스 모두 환전 엔드포인트 0건. 정방향(KRW→USD)은
        통합증거금이 결제 시 자동 처리할 뿐 명시 API 호출이 없다. 따라서 '엔드포인트 발명 금지'
        원칙에 따라 여기서 임의 URL/TR 을 호출하지 않는다:
          - is_mock / dry_run → 절대 실주문 금지(no-op, 의도만 로깅).
          - config.KIS_FX_EXCHANGE_TR 미설정(기본) → {ok:False, manual_required:True}(수동 환전 신호).
        KIS 가 환전 TR 을 공개/계약 제공하면, 검증된 TR·URL·body 를 *오직 이 메서드 안에서만*
        배선한다(다른 곳에서 환전을 호출/조립하지 말 것)."""
        usd_amount = float(usd_amount or 0.0)
        if usd_amount <= 0:
            return {"ok": False, "reason": "환전액 0", "manual_required": False}
        if self.is_mock:
            logger.info(f"[USD→KRW] 모의계좌 — 환전 미실행(no-op) ${usd_amount:,.2f} ({reason})")
            return {"ok": False, "reason": "모의계좌 — 환전 미지원", "manual_required": False}
        if dry_run:
            logger.info(f"[USD→KRW][DRY-RUN] 환전 미실행 ${usd_amount:,.2f} ({reason})")
            return {"ok": False, "reason": "dry-run — 환전 미실행", "manual_required": False, "dry_run": True}
        try:
            from config import KIS_FX_EXCHANGE_TR as _FX_TR
        except Exception:
            _FX_TR = ""
        if not _FX_TR:
            # KIS 공개 환전 TR 없음 → 자동 실환전 불가. 조용히 누락 금지: 수동 환전 필요 신호로 반환.
            logger.warning(f"[USD→KRW] 환전 TR 미설정 — 자동 환전 불가, 수동 환전 필요 ${usd_amount:,.2f} ({reason})")
            return {"ok": False, "reason": "KIS 공개 환전 TR 없음 — 수동 환전 필요",
                    "manual_required": True, "usd": usd_amount}
        # ── 확장점(미배선): KIS 환전 TR 이 확보되면 *여기서만* POST 한다. 검증된 TR/URL/body 가
        #    없는 상태에서 임의 엔드포인트를 호출하면 실주문 사고이므로 절대 금지. ──
        logger.error(f"[USD→KRW] KIS_FX_EXCHANGE_TR={_FX_TR} 설정됐으나 실행 경로 미배선 — 수동 환전 필요 "
                     f"${usd_amount:,.2f} ({reason})")
        return {"ok": False, "reason": "환전 실행 경로 미배선(미검증 TR)",
                "manual_required": True, "usd": usd_amount}

    # ═══════════════ 신규 권위조회 (사장 지시 2026-06-01, KIS 공식샘플 정독 반영) ═══════════════
    # 잔고/주문 한도를 추정(D+2·환율 합성) 대신 KIS 권위 전용조회로. 실전 전용 TR(6548/6010/8494)은
    # 모의 미지원이라 is_mock 이면 호출 자체를 skip(매 사이클 무용 호출·거부로그·TPS 소모 방지).
    async def kr_psbl_order(self, code: str, unpr: float = 0.0) -> Dict:
        """국내 매수가능 (TTTC8908R). ★ORD_DVSN='01'(시장가)로 호출해야 종목 증거금율이 반영된
        nrcvb_buy_qty(미수없는 매수가능수량)를 준다(지정가 '00'은 미반영→과대). 실패 시 {ok:False,buy_qty:None}."""
        c, p = self._acnt()
        d = await self._get_json("/uapi/domestic-stock/v1/trading/inquire-psbl-order", "TTTC8908R",
            {"CANO": c, "ACNT_PRDT_CD": p, "PDNO": code, "ORD_UNPR": str(int(unpr or 0)),
             "ORD_DVSN": "01", "CMA_EVLU_AMT_ICLD_YN": "N", "OVRS_ICLD_YN": "N"})
        if str(d.get("rt_cd", "")) != "0":
            return {"ok": False, "buy_qty": None, "cash": 0.0, "msg1": d.get("msg1", "")}
        o = d.get("output") or {}
        return {"ok": True, "buy_qty": self._int(o.get("nrcvb_buy_qty")),
                "cash": self._num(o.get("ord_psbl_cash"))}

    async def kr_psbl_sell_qty(self, code: str) -> Optional[int]:
        """국내 매도가능수량 (TTTC8408R). ord_psbl_qty. 빈값/실패 시 None(폴백은 호출부에서 hldg_qty —
        주문 절대 드롭 금지)."""
        c, p = self._acnt()
        d = await self._get_json("/uapi/domestic-stock/v1/trading/inquire-psbl-sell", "TTTC8408R",
            {"CANO": c, "ACNT_PRDT_CD": p, "PDNO": code})
        if str(d.get("rt_cd", "")) != "0":
            return None
        o = d.get("output")
        if isinstance(o, list):
            o = o[0] if o else {}
        v = (o or {}).get("ord_psbl_qty")
        return self._int(v) if v not in (None, "") else None

    async def us_buying_power(self, ticker: str, unpr: float, excg: Optional[str] = None) -> Dict:
        """해외 매수가능 (TTTS3007R). USD 주문가능금액(ord_psbl_frcr_amt)·최대수량·환율을 직접 준다 —
        KR 원화예수금을 환율로 나눈 합성 대신 사용(통화혼용·과대사이징 방지). 실패 시 {ok:False}.

        거래소 결정(버그 2026-06-17, uid1 NYSE 매수 거부 반복): 호출부(클램프)가 excg 를
        주지 않아 'NASD' 로 고정되면 NYSE/AMEX 종목이 '상품이 없습니다'(rt_cd≠0)로 거부되어
        ok=False → 클램프가 스킵되고 못 살 주문이 KIS 까지 갔다. 실제 주문(_overseas_order_body)이
        쓰는 _us_excd_cache(시세 프로브 자동판별)와 동일하게 맞춘다 — excg 미지정 시 캐시(없으면
        1회 프로브)에서 확보, 명시값(NAS/NYS/AMS)도 excd_to_excg 로 정규화."""
        tk = (ticker or "").upper()
        if excg:
            excg = excd_to_excg(excg)
        else:
            cached = self._us_excd_cache.get(tk)
            if not cached:
                try:
                    await self.us_last_price(tk)   # 거래소 자동판별 → 캐시 채움
                except Exception:
                    pass
                cached = self._us_excd_cache.get(tk)
            excg = excd_to_excg(cached) if cached else "NASD"
        c, p = self._acnt()
        d = await self._get_json("/uapi/overseas-stock/v1/trading/inquire-psamount", "TTTS3007R",
            {"CANO": c, "ACNT_PRDT_CD": p, "OVRS_EXCG_CD": excg,
             "OVRS_ORD_UNPR": f"{float(unpr or 0):.4f}", "ITEM_CD": tk})
        if str(d.get("rt_cd", "")) != "0":
            return {"ok": False, "usd": 0.0, "qty": 0, "exrt": 0.0, "msg1": d.get("msg1", "")}
        o = d.get("output") or {}
        # 통합증거금(버그 2026-06-17, uid1 US 신규매수 전부 '예수금 $0 제외'): KIS 는 KRW 를 환율로
        # 환산한 해외 매수력을 ovrs_ord_psbl_amt(해외 주문가능금액)·max_ord_psbl_qty 로 정확히 준다.
        # 순수 USD 현금(ord_psbl_frcr_amt)만 읽으면 USD 0 계좌에서 통합증거금이 무력화돼 매수가 전부
        # 제외됐다(라이브 확인: ord_psbl_frcr_amt=0 · ovrs_ord_psbl_amt=1657.94 · max_ord_psbl_qty=13).
        # usd 는 ovrs_ord_psbl_amt → frcr_ord_psbl_amt1 → ord_psbl_frcr_amt 순 첫 양수(qty 클램프가 최종 방어).
        _usd = (self._num(o.get("ovrs_ord_psbl_amt"))
                or self._num(o.get("frcr_ord_psbl_amt1"))
                or self._num(o.get("ord_psbl_frcr_amt")))
        return {"ok": True, "usd": _usd,
                "qty": self._int(o.get("max_ord_psbl_qty")), "exrt": self._num(o.get("exrt"))}

    async def kr_account_asset(self) -> Dict:
        """투자계좌자산현황 (CTRP6548R) — KR+US 통합 총자산(tot_asst_amt). 대시보드 '현재 총자산' 표시 전용
        (예수금이 D0 기반이라 자산곡선엔 쓰지 않는다 — 5/28 D0 금지). 모의 미지원 → skip."""
        if self.is_mock:
            return {"ok": False}
        c, p = self._acnt()
        d = await self._get_json("/uapi/domestic-stock/v1/trading/inquire-account-balance", "CTRP6548R",
            {"CANO": c, "ACNT_PRDT_CD": p, "INQR_DVSN_1": "", "BSPR_BF_DT_APLY_YN": ""})
        if str(d.get("rt_cd", "")) != "0":
            return {"ok": False}
        o = d.get("output2") or {}
        if isinstance(o, list):
            o = o[0] if o else {}
        return {"ok": True, "tot_asst_amt": self._num(o.get("tot_asst_amt")),
                "tot_dncl_amt": self._num(o.get("tot_dncl_amt"))}

    async def _overseas_settled_krw(self) -> Dict:
        """해외 결제기준잔고 (CTRP6010R) — 외화잔고 원화평가합계(frcr_cblc_wcrc_evlu_amt_smtl, 결제기준).
        자산곡선 해외분을 국내 D+2와 같은 결제기준으로 맞춰 결제 과도기 출렁임을 줄인다. 모의 미지원 → skip."""
        if self.is_mock:
            return {"ok": False}
        c, p = self._acnt()
        bass = datetime.now(KST).strftime("%Y%m%d")
        d = await self._get_json("/uapi/overseas-stock/v1/trading/inquire-paymt-stdr-balance", "CTRP6010R",
            {"CANO": c, "ACNT_PRDT_CD": p, "BASS_DT": bass, "WCRC_FRCR_DVSN_CD": "01", "INQR_DVSN_CD": "00"})
        if str(d.get("rt_cd", "")) != "0":
            return {"ok": False}
        o = d.get("output3") or {}
        if isinstance(o, list):
            o = o[0] if o else {}
        return {"ok": True, "krw": self._num(o.get("frcr_cblc_wcrc_evlu_amt_smtl")),
                "tot_asst_amt2": self._num(o.get("tot_asst_amt2"))}

    async def overseas_fills(self, start_ymd: str, end_ymd: str) -> List[Dict]:
        """해외주식 기간 체결내역 (TTTS3035R inquire-ccnl, 실전 전용 — 모의 미지원 → 빈 목록).
        사장 지시 2026-06-11: 실거래 원장의 '미결제 USD 매도대금' 시드와 실현손익 backfill 의
        권위 소스 — 결제 과도기엔 통합총자산/외화예수금 TR 이 전부 0 을 줘(라이브 확인:
        6504·6010 모두 0, 6548 만 포함) 체결내역이 유일하게 신뢰 가능한 per-trade 기록이다.
        반환: [{date(YYYYMMDD), ticker, side, qty, price, amount, ccy}] — 체결 수량 있는 행만."""
        if self.is_mock:
            return []
        try:
            c, p = self._acnt()
            d = await self._paged_get("/uapi/overseas-stock/v1/trading/inquire-ccnl", "TTTS3035R",
                {"CANO": c, "ACNT_PRDT_CD": p, "PDNO": "%",
                 "ORD_STRT_DT": str(start_ymd), "ORD_END_DT": str(end_ymd),
                 "SLL_BUY_DVSN": "00", "CCLD_NCCS_DVSN": "01",
                 "OVRS_EXCG_CD": "%", "SORT_SQN": "DS", "ORD_DT": "",
                 "ORD_GNO_BRNO": "", "ODNO": "",
                 "CTX_AREA_NK200": "", "CTX_AREA_FK200": ""},
                fk_key="CTX_AREA_FK200", nk_key="CTX_AREA_NK200", out_keys=("output",))
            if not d.get("ok"):
                return []
            rows: List[Dict] = []
            for r in d.get("output") or []:
                qty = self._num(r.get("ft_ccld_qty"))
                if qty <= 0:
                    continue
                name = str(r.get("sll_buy_dvsn_cd_name") or "")
                side = "sell" if ("매도" in name or str(r.get("sll_buy_dvsn_cd")) == "01") else "buy"
                price = self._num(r.get("ft_ccld_unpr3"))
                rows.append({"date": str(r.get("ord_dt") or ""), "ticker": str(r.get("pdno") or "").strip(),
                             "side": side, "qty": int(qty), "price": price,
                             "amount": self._num(r.get("ft_ccld_amt3")) or (qty * price),
                             "ccy": str(r.get("tr_crcy_cd") or "USD")})
            return rows
        except Exception as e:
            logger.warning(f"[해외체결내역] TTTS3035R 실패: {e}")
            return []

    async def kr_realized_pnl_audit(self) -> Dict:
        """국내 실현손익(TTTC8494R, 비용반영 COST_ICLD_YN='Y') — 우리 체결기반 실현손익 KPI 교차검증용
        (주문 무영향). 모의 미지원 가능 → skip. output1 의 rlzt_pfls 합산."""
        if self.is_mock:
            return {"ok": False}
        c, p = self._acnt()
        d = await self._get_json("/uapi/domestic-stock/v1/trading/inquire-balance-rlz-pl", "TTTC8494R",
            {"CANO": c, "ACNT_PRDT_CD": p, "AFHR_FLPR_YN": "N", "OFL_YN": "",
             "INQR_DVSN": "00", "UNPR_DVSN": "01", "FUND_STTL_ICLD_YN": "N",
             "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "00", "COST_ICLD_YN": "Y",
             "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""})
        if str(d.get("rt_cd", "")) != "0":
            return {"ok": False}
        # 라이브 검증 2026-06-01: 실현손익은 output2(요약) rlzt_pfls/rlzt_erng_rt 에 있다(output1 아님).
        o2 = d.get("output2") or {}
        if isinstance(o2, list):
            o2 = o2[0] if o2 else {}
        return {"ok": True, "realized": self._num(o2.get("rlzt_pfls")),
                "realized_rt": self._num(o2.get("rlzt_erng_rt"))}

    # 사장 지시 2026-05-21: 해외 원화평가 캐시 — KIS 해외잔고 조회가 간헐 실패하면 US 평가가
    # 통째로 빠져 통합 총평가가 ~16% 급락(자산곡선 -16% 글리치)한다. 마지막 정상값을 디스크에
    # 영속해 일시적 조회 실패를 메우고(재시작 콜드스타트 포함), 실제 매도(조회 성공+평가 0)면
    # 즉시 캐시를 비운다.
    _OVERSEAS_CACHE_TTL = 7200  # 2시간 — 일시 실패 보강용(실제 매도는 즉시 반영되므로 무관)

    def _overseas_cache_path(self) -> Path:
        # Phase 2: 토큰 캐시와 같은 per-uid 디렉터리(data/<uid>/)에 둔다. 전역 단일 파일이면
        # 한 계정의 해외 원화평가가 다른 계정(특히 실전→모의)으로 누출된다.
        return self._token_path.parent / "overseas_krw_cache.json"

    def _get_overseas_cache(self):
        """(총평가_krw, ts). 부수적으로 주식분(_overseas_stock_krw)·기준환율(_overseas_exrt)도
        인스턴스에 적재한다 — present-balance 실패 시 예수금 보존 폴백(portfolio_holdings)에서 쓴다."""
        c = getattr(self, "_overseas_krw_cache", None)
        if c is not None:
            return c
        self._overseas_stock_krw = 0.0
        self._overseas_exrt = 0.0
        try:
            p = self._overseas_cache_path()
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                self._overseas_krw_cache = (float(d.get("krw") or 0.0), float(d.get("ts") or 0.0))
                self._overseas_stock_krw = float(d.get("stock") or 0.0)
                self._overseas_exrt = float(d.get("exrt") or 0.0)
                return self._overseas_krw_cache
        except Exception:
            pass
        self._overseas_krw_cache = (0.0, 0.0)
        return self._overseas_krw_cache

    def _overseas_selfcalc_krw(self, holdings: List[Dict]) -> Dict:
        """모의서버 기준환율이 garbage 일 때 해외 순평가를 자체 산출한다 (사장 지시 2026-07-22).

        실측(uid2 모의, 2026-07-22):
          • 보유수량(IEF 97)·현재가($93.31)는 실제와 **일치** → 신뢰 가능.
          • 오염된 건 기준환율(frst_bltn_exrt 218.31 vs 실제 1480)과 그걸로 환산된
            총평가(frcr_evlu_tota 357M)뿐. → 환율만 우리 실환율로 갈아끼우면 된다.
          • 국내 nass_amt = 국내유가증권 + D+2예수금 (차이 0) → 해외분 **미포함**, 더해야 한다.

        ⚠️ 주식분만 더하면 안 된다: 모의는 US 매수 때 KRW 를 전혀 차감하지 않았고(7/21 22:39
        IEF 97주 매수 전후 D+2 예수금 불변), 그 대가가 **USD 부채**로 남아 있다(원장 cash_usd
        -9,086). 주식분 13.4M 만 더하면 그만큼 가짜 이득이 된다. 그래서 USD 예수금(음수 포함)을
        함께 환산해 **순액**으로 더한다 — 매수 시점 총자산이 보존되고(주식 +13.4M, USD현금 -13.4M),
        이후 IEF 가격 변동만 손익으로 잡힌다.

        USD 예수금 출처: KIS 모의는 frcr_dncl_amt_2 를 0 으로 오보하므로(부채를 안 알려줌)
        우리 체결 원장의 cash_usd 를 쓴다. 원장이 없으면 산출을 포기한다(0 처리 — 종전 동작).

        ⚠️ 감시 필요(2026-07-22): 모의가 **뒤늦게(US 결제일 T+2~3) KRW 를 차감**하면 부채가
        이중 계상된다 — KIS 쪽 KRW 예수금이 줄고 우리 원장 cash_usd 도 여전히 음수라 총평가가
        13.4M 헛빠진다. 아래 INFO 로그가 매 폴마다 (주식 / USD예수금 / 순액)을 찍으니,
        총평가가 US 매수액만큼 계단식으로 떨어지면 이 경로를 먼저 의심할 것. 그때는 USD 예수금
        출처를 원장 대신 KIS 실측(또는 KRW 차감 감지 후 원장 cash_usd 상계)으로 바꿔야 한다.

        반환: {ok, krw(순액), stock_krw, usd_cash, fx}. 산출 불가면 ok=False.
        """
        fx = _real_usdkrw()
        usd_stock = sum(self._num(h.get("qty")) * self._num(h.get("cur_price"))
                        for h in (holdings or []) if h.get("ccy") == "USD")
        # 2026-07-29: `usd_stock<=0` 조기반환은 **부채를 지웠다**. 모의는 US 매수 때 KRW 를 안
        # 깎고 USD 부채로 남기므로, 해외 보유목록이 비어도(조회 실패·전량매도 직후) 원장
        # cash_usd 는 그대로 남는다. 주식분이 0 이어도 부채는 계속 순평가에 반영해야 한다.
        if fx <= 0:
            return {"ok": False, "krw": 0.0, "stock_krw": 0.0, "usd_cash": 0.0, "fx": fx}
        # 브로커는 uid 를 들고 있지 않다. 원장은 data/<uid>/ledger.json 이고 토큰 경로가
        # data/<uid>/kis_token.json 이므로, 같은 디렉터리에서 직접 읽는다(_settled_cash_path 와 동형).
        try:
            _p = self._token_path.parent / "ledger.json"
            if not _p.exists():
                return {"ok": False, "krw": 0.0, "stock_krw": 0.0, "usd_cash": 0.0, "fx": fx}
            led = json.loads(_p.read_text(encoding="utf-8"))
            if not isinstance(led, dict) or "cash_usd" not in led:
                return {"ok": False, "krw": 0.0, "stock_krw": 0.0, "usd_cash": 0.0, "fx": fx}
            usd_cash = float(led.get("cash_usd") or 0.0)
            # 해외 잔고 API가 한 틱 비면 라이브 주식분은 0이지만 USD 부채는 그대로여서
            # 순평가가 매수원금만큼 급락했다(uid2: 약 -62M). 라이브 USD 보유가 하나도 없을
            # 때는 원장의 직전 수량·가격으로 주식분도 함께 carry-forward 한다. 실제 전량매도는
            # 아래 _reconcile_mock_us_flat 이 3회 권위 확인 후 원장 포지션을 청산한다.
            if usd_stock <= 0:
                usd_stock = sum(
                    self._num(pos.get("qty"))
                    * (self._num(pos.get("last_price")) or self._num(pos.get("avg_cost")))
                    for pos in (led.get("positions") or {}).values()
                    if str(pos.get("ccy") or "").upper() == "USD"
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("[해외자체산출] 원장 USD 예수금 조회 실패(%s): %s — 자체산출 포기",
                           self._token_path.parent, e)
            return {"ok": False, "krw": 0.0, "stock_krw": 0.0, "usd_cash": 0.0, "fx": fx}
        stock_krw = usd_stock * fx
        return {"ok": True, "krw": stock_krw + usd_cash * fx, "stock_krw": stock_krw,
                "usd_cash": usd_cash, "fx": fx}

    _MOCK_US_FLAT_CONFIRMATIONS = 3

    def _reconcile_mock_us_flat(self, authoritative_empty: bool) -> bool:
        """모의 KIS가 US 빈 잔고를 연속 확인하면 원장 허수 포지션을 청산한다.

        모의 해외 잔고는 일시적으로 []를 반환하므로 단발성 빈 응답에는 손대지 않는다. NASD·NYSE·
        AMEX 세 조회가 모두 성공한 빈 잔고가 3회 연속일 때만, 원장에 남은 USD 포지션을 직전가로
        청산한 것으로 정정한다. 매도대금은 cash_usd에 더해 누적 US 손익은 보존한다.
        """
        if not self.is_mock:
            return False
        p = self._token_path.parent / "ledger.json"
        try:
            if not p.exists():
                return False
            led = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(led, dict) or not isinstance(led.get("positions"), dict):
                return False
            def _persist() -> None:
                tmp = p.with_name(p.name + ".mock-flat.tmp")
                tmp.write_text(json.dumps(led, ensure_ascii=False, indent=2), encoding="utf-8")
                tmp.replace(p)  # 같은 파일시스템 원자 교체 — 중간 JSON 노출 방지
            streak_key = "_mock_us_flat_streak"
            if not authoritative_empty:
                if streak_key in led:
                    led.pop(streak_key, None)
                    _persist()
                return False
            usd_positions = {
                code: pos for code, pos in led["positions"].items()
                if str((pos or {}).get("ccy") or "").upper() == "USD"
                and self._int((pos or {}).get("qty")) > 0
            }
            if not usd_positions:
                led.pop(streak_key, None)
                return False
            streak = self._int(led.get(streak_key)) + 1
            if streak < self._MOCK_US_FLAT_CONFIRMATIONS:
                led[streak_key] = streak
                _persist()
                return False

            proceeds = 0.0
            removed = []
            for code, pos in usd_positions.items():
                qty = self._int(pos.get("qty"))
                px = self._num(pos.get("last_price")) or self._num(pos.get("avg_cost"))
                # 원장과 동일한 US 매도비용 0.3%. 직전가 정정이라 exact fill이 아닌 점은 감사
                # 기록에 남긴다. 현금을 0으로 덮지 않아 기존 실현/미실현 손익은 보존한다.
                proceeds += qty * px * (1.0 - 0.003)
                removed.append({"ticker": code, "qty": qty, "price": px})
                led["positions"].pop(code, None)
            before_cash = self._num(led.get("cash_usd"))
            led["cash_usd"] = before_cash + proceeds
            led.pop(streak_key, None)
            audit = led.setdefault("reconciliations", [])
            audit.append({
                "ts": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
                "kind": "mock_us_authoritative_flat",
                "before_cash_usd": before_cash,
                "proceeds_usd_approx": proceeds,
                "positions": removed,
            })
            led["reconciliations"] = audit[-50:]
            _persist()
            logger.warning(
                "[모의US원장정정 uid=%s] 해외 빈 잔고 %s회 연속 권위확인 — 허수 %s종목 청산, "
                "USD현금 %.2f→%.2f",
                self._token_path.parent.name, self._MOCK_US_FLAT_CONFIRMATIONS,
                len(removed), before_cash, led["cash_usd"],
            )
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("[모의US원장정정 uid=%s] 실패: %s", self._token_path.parent.name, e)
            return False

    def _set_overseas_cache(self, krw: float, ts: float, stock: Optional[float] = None,
                            exrt: Optional[float] = None) -> None:
        self._overseas_krw_cache = (float(krw), float(ts))
        if stock is not None:
            self._overseas_stock_krw = float(stock)
        if exrt is not None and exrt > 0:
            self._overseas_exrt = float(exrt)
        try:
            self._overseas_cache_path().write_text(
                json.dumps({"krw": float(krw), "ts": float(ts),
                            "stock": float(getattr(self, "_overseas_stock_krw", 0.0) or 0.0),
                            "exrt": float(getattr(self, "_overseas_exrt", 0.0) or 0.0)},
                           ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    # 사장 지시 2026-05-28: D+2 예수금 last-good 영속 — KIS가 결제 과도기에 D+2/D+1 을 0 으로 깜빡일 때
    # D0(부풀려진 값)로 폴백하지 않고 직전 정상 D+2 를 유지하기 위함(재시작 콜드스타트 포함). per-uid.
    def _settled_cash_path(self) -> Path:
        return self._token_path.parent / "settled_cash_cache.json"

    def _get_settled_cash_cache(self) -> float:
        c = getattr(self, "_settled_cash", None)
        if c is not None:
            return c
        try:
            p = self._settled_cash_path()
            if p.exists():
                self._settled_cash = float(json.loads(p.read_text(encoding="utf-8")).get("d2") or 0.0)
                return self._settled_cash
        except Exception:
            pass
        self._settled_cash = 0.0
        return self._settled_cash

    def _set_settled_cash_cache(self, d2: float) -> None:
        self._settled_cash = float(d2)
        try:
            self._settled_cash_path().write_text(
                json.dumps({"d2": float(d2)}, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    # 사장 제보 2026-05-29: 보유목록 last-good 디스크 영속 — KIS가 보유목록을 빈 채 주는 글리치가
    # 서버 재시작 직후(in-memory 스냅샷 소실=콜드스타트) 나면 기존 가드(in-memory cached 의존)가
    # 못 막아 '보유 종목 없음'이 떴다(모의계좌 대한항공 사라짐 현상). settled_cash·overseas_krw 와
    # 동형으로 디스크에 영속해 재시작 갭을 메운다. 정상 매도로 평탄해지면 즉시 무효화(유령 보유 방지). per-uid.
    _HOLDINGS_CACHE_TTL = 86400.0  # 24h — 재시작·글리치 보강용. 진짜 매도는 즉시 무효화되므로 무관.

    def _holdings_cache_path(self) -> Path:
        return self._token_path.parent / "holdings_cache.json"

    def _get_holdings_cache(self):
        """(holdings:list, total:float, ts:float). 미존재/실패 시 ([],0,0)."""
        c = getattr(self, "_holdings_cache", None)
        if c is not None:
            return c
        try:
            p = self._holdings_cache_path()
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                self._holdings_cache = (list(d.get("holdings") or []),
                                        float(d.get("total") or 0.0), float(d.get("ts") or 0.0))
                return self._holdings_cache
        except Exception:
            pass
        self._holdings_cache = ([], 0.0, 0.0)
        return self._holdings_cache

    def _set_holdings_cache(self, holdings, total: float, ts: float) -> None:
        hl = list(holdings or [])
        self._holdings_cache = (hl, float(total or 0.0), float(ts or 0.0))
        try:
            self._holdings_cache_path().write_text(
                json.dumps({"holdings": hl, "total": float(total or 0.0), "ts": float(ts or 0.0)},
                           ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _clear_holdings_cache(self) -> None:
        self._holdings_cache = ([], 0.0, 0.0)
        try:
            p = self._holdings_cache_path()
            if p.exists():
                p.unlink()
        except Exception:
            pass

    async def portfolio_holdings(self) -> Dict:
        """대시보드 '종목 포트폴리오' — 국내주식 + 해외주식 + 국내채권 + 펀드 통합 잔고.
        kr_account_snapshot()의 국내주식 잔고를 기준으로 나머지를 best-effort 병합한다.
        사장 피드백 2026-05-20: (code, category) 키로 최종 dedupe — 외부 거래 종목도 KIS 잔고가
        신뢰원천이므로 자동 포함되지만, 중복 노출만 방지(같은 카테고리 내 같은 code 합치기)."""
        snap = await self.kr_account_snapshot()
        kr = [{**h, "category": "국내주식", "ccy": "KRW"} for h in (snap.get("holdings") or [])]
        extra: List[Dict] = []
        for fn in (self._overseas_holdings, self._bond_holdings, self._fund_holdings):
            try:
                extra += await fn()
            except Exception:
                pass
        merged: Dict = {}
        for h in kr + extra:
            code = (h.get("code") or "").strip()
            cat = h.get("category") or ""
            if not code:
                continue
            key = (code, cat)
            if key in merged:
                ex = merged[key]
                q1, q2 = self._int(ex.get("qty")), self._int(h.get("qty"))
                tot = q1 + q2
                if tot > 0:
                    ex["avg_price"] = ((self._num(ex.get("avg_price")) * q1) + (self._num(h.get("avg_price")) * q2)) / tot
                ex["qty"] = tot
                ex["pnl_amt"] = self._num(ex.get("pnl_amt")) + self._num(h.get("pnl_amt"))
                if self._num(h.get("cur_price")) > 0:
                    ex["cur_price"] = self._num(h.get("cur_price"))
                if h.get("pnl_pct"):
                    ex["pnl_pct"] = h.get("pnl_pct")
            else:
                merged[key] = dict(h)
        holdings = list(merged.values())
        # ── 통합 총평가 보정: 국내 nass_amt(buying_power.total_eval)는 해외주식 평가를 포함하지
        #    않는다. USD 보유가 있으면 present-balance(원화환산 평가합계)를 더해 총자산을 맞춘다.
        #    + 각 USD 종목에 원화환산 평가액(krw_value)을 부여(프론트 표시용). ──
        bp = dict(snap["buying_power"])
        # 사장 지시 2026-06-03: snap.total_eval 은 이제 캐시 해외분을 포함(kr_account_snapshot)하므로,
        # 곡선용 합산은 KR 기준(total_eval_kr)에서 다시 시작해 아래 fresh/결제기준 해외분만 더한다(이중계상 방지).
        if bp.get("total_eval_kr") is not None:
            bp["total_eval"] = self._num(bp["total_eval_kr"])
        _now = time.time()
        _us_in_holdings = any(h.get("ccy") == "USD" for h in holdings)
        # 사장 결정 2026-06-01: 모의계정도 해외평가를 표시한다(_mock_tr 표준화로 VTRP6504R 이 읽혀
        # frcr_evlu_tota≈379M 가 HTS 와 근접). 단 모의서버 기준환율(frst_bltn_exrt)이 비정상(221.9)일 수
        # 있어, 아래에서 환율을 전역 FX 캐시에 먹일 땐 sanity 가드(>500)로 garbage 전파를 막는다.
        # (총평가에 더하는 krw_value=frcr_evlu_tota 는 KIS 산출값이라 환율 오류와 무관.)
        # 항상 권위 조회(ok 플래그 보유)로 US 원화평가를 확인 — 실패/진짜없음/정상을 구분해
        # 곡선·총평가가 조회 실패로 ~16% 급락하지 않게 한다.
        pk = await self._overseas_present_krw()
        # 모의 원장에는 이미 매도된 US 포지션이 체결 누락으로 남을 수 있다. 세 거래소 조회가
        # 모두 성공한 빈 잔고만 연속 확인해 정정하며, 조회 실패/부분성공이면 스트릭을 리셋한다.
        if self.is_mock:
            self._reconcile_mock_us_flat(
                bool(not _us_in_holdings
                     and getattr(self, "_overseas_holdings_authoritative", False))
            )
        krw = None
        # 사장 보고 2026-07-29(수익률 ±40% 튐): 모의 자체산출 순평가는 **음수**(USD 부채)일 수
        # 있고, 그 부채는 US 보유목록이 비어도 남는다. 종전엔 자체산출 진입 조건이
        # `_us_in_holdings` 라, 해외 보유 조회가 빈 폴마다 -29M 부채가 사라져 총평가가
        # 71M↔100M 로 튀었다(uid2). 자체산출은 보유목록 유무와 무관하게 시도한다(모의 한정).
        def _selfcalc_into_bp():
            """원장 기반 해외 순평가 → (krw, stock_krw) | None. bp 에 투명성 필드도 채운다."""
            _sc = self._overseas_selfcalc_krw(holdings)
            if not _sc["ok"] or not (_sc["krw"] or _sc["stock_krw"]):
                return None
            bp["overseas_selfcalc"] = True
            # 표시 투명성: 보유목록엔 US 주식 평가만 보이고 그 대가인 USD 부채는 안 보이므로,
            # 합이 안 맞아 보인다. 부채분을 별도 필드로 노출한다.
            bp["overseas_stock_krw"] = _sc["stock_krw"]
            bp["overseas_usd_cash_krw"] = _sc["usd_cash"] * _sc["fx"]
            logger.info("[해외자체산출 uid=%s] 주식 %s + USD예수금 %s USD × %s = 순 %s원 "
                        "(모의 기준환율 %s 무시)", self._token_path.parent.name,
                        f"{_sc['stock_krw']:,.0f}", f"{_sc['usd_cash']:,.2f}",
                        f"{_sc['fx']:,.1f}", f"{_sc['krw']:,.0f}", pk.get("exrt"))
            return _sc["krw"], _sc["stock_krw"]
        if pk["ok"] and pk["krw_value"] > 0:
            krw = pk["krw_value"]                         # 조회 성공 + 평가 있음 = 권위값
            # 사장 지시 2026-06-10: 모의서버 해외 데이터는 평가·기준환율뿐 아니라 보유 가격·수량까지
            # garbage 다(라이브 확인: exrt 224, US종목 평가 145M인데 모의 계좌는 100M짜리 — 물리적 불가).
            # 가격×수량×환율 재계산도 입력이 전부 오염돼 무의미하므로, exrt 비정상(<500)이면 해외평가를
            # 신뢰 불가로 0 처리(제외)한다 → 모의 equity = 국내+현금만(안정·정직). 실거래(exrt~1500)는 영향 없음.
            krw, _ov_stock = _sanitize_overseas(krw, pk.get("stock_value"), pk.get("exrt"))
            # 사장 지시 2026-07-22: 종전엔 여기서 0 처리하고 끝이라 모의계정의 US 평가·손익이
            # 자산곡선에 영영 안 잡혔다. 수량·현재가는 멀쩡하므로 실환율로 순평가를 자체 산출한다.
            if krw <= 0:
                _r = _selfcalc_into_bp()
                if _r:
                    krw, _ov_stock = _r
            self._set_overseas_cache(krw, _now, stock=_ov_stock, exrt=pk.get("exrt"))
            # 사장 지시 2026-06-01: 기준환율 sanity 가드 — 모의서버가 비정상 환율(221.9 등)을 주면
            # 전역 FX 캐시(예산·리스크 환산)가 오염돼 실전 계정까지 망가진다. USD/KRW 는 역사적으로 500↑.
            if pk["exrt"] > 500:
                bp["fx_rate"] = pk["exrt"]
                # 사장 지시 2026-05-22: 5분 폴러가 매번 호출하는 이 경로의 KIS 기준환율을
                # 라이브 FX 캐시에 먹여, 예산·리스크 환산이 최신 환율을 쓰게 한다.
                try:
                    from tools.market_data import set_usdkrw
                    set_usdkrw(pk["exrt"])
                except Exception:
                    pass
                for h in holdings:
                    if h.get("ccy") == "USD":
                        h["krw_value"] = round(self._num(h.get("qty")) * self._num(h.get("cur_price")) * pk["exrt"])
            else:
                # 2026-07-22: 기준환율이 비정상(모의)이면 종목별 원화표시가 통째로 빠져
                # 대시보드 보유목록에서 US 종목만 원화가 비어 보였다 — 실환율로 채운다.
                _fx = _real_usdkrw()
                if _fx > 0:
                    for h in holdings:
                        if h.get("ccy") == "USD":
                            h["krw_value"] = round(self._num(h.get("qty")) * self._num(h.get("cur_price")) * _fx)
        elif pk["ok"] and pk["krw_value"] == 0 and not _us_in_holdings:
            # 조회 성공 + 평가 0 + 보유목록도 US 없음 = 실제 매도 → 캐시 무효화.
            # 단 모의는 US 매수분이 KRW 를 안 깎고 USD 부채로 남으므로, 주식이 0 이어도 원장에
            # 부채가 있으면 그 부채를 유지해야 한다(0 으로 지우면 총평가가 부채만큼 튄다).
            _r = _selfcalc_into_bp() if self.is_mock else None
            if _r:
                krw, _ov_stock = _r
                self._set_overseas_cache(krw, _now, stock=_ov_stock)
            else:
                self._set_overseas_cache(0.0, _now)
        else:
            # 조회 실패(ok=False) 또는 모순(보유목록엔 US인데 평가 0) → 최근 캐시로 보강(곡선 안정)
            _ck, _ct = self._get_overseas_cache()
            # 2026-07-29: `_ck > 0` 는 음수 캐시(모의 USD 부채)를 폴백에서 제외해, 조회 실패 폴마다
            # 부채가 사라지고 총평가가 튀게 만들었다 → 0이 아니면(부호 무관) 폴백한다.
            if _ck and (_now - _ct) < self._OVERSEAS_CACHE_TTL:
                # 사장 지시 2026-05-30: stale 캐시값으로 '동결'하면 US 세션 내내 자산곡선이 안 움직인다.
                # 보유종목 라이브 현재가로 '주식분'을 재계산하고 캐시에 보존된 'USD 예수금분'(총액−주식분)을
                # 더한다 — 예수금 정확도(2026-05-28 수정)는 지키면서 곡선이 라이브로 움직인다.
                # 라이브 주식분/환율/캐시 주식분 중 하나라도 없으면 기존처럼 캐시 총액을 그대로 쓴다(안전).
                _exrt = float(getattr(self, "_overseas_exrt", 0.0) or 0.0)
                if _exrt <= 0:
                    try:
                        from tools.market_data import get_usdkrw
                        _exrt = float(get_usdkrw(0.0) or 0.0)
                    except Exception:
                        _exrt = 0.0
                _live_stock = sum(self._num(h.get("qty")) * self._num(h.get("cur_price"))
                                  for h in holdings if h.get("ccy") == "USD") * _exrt
                _cached_stock = float(getattr(self, "_overseas_stock_krw", 0.0) or 0.0)
                if _live_stock > 0 and _exrt > 0 and _cached_stock > 0:
                    # 예수금분 = 캐시총액 − 캐시주식분. `max(0,…)` 로 바닥을 치면 모의 USD
                    # 부채(음수 예수금)가 지워져 총평가가 부채만큼 튄다 → 부호 보존(2026-07-29).
                    krw = _live_stock + (_ck - _cached_stock)
                else:
                    krw = _ck
                bp["overseas_krw_stale"] = True
        # 사장 결정 2026-06-01: 자산곡선의 해외분은 '결제기준'(CTRP6010R)으로 — 국내 D+2와 같은 결제기준에
        # 맞춰 결제 과도기 곡선 출렁임을 줄인다. 결제기준 조회 성공(실전)이면 그 값을, 실패(모의 등)면 위에서
        # 구한 실시간 krw 로 폴백한다. (per-종목 krw_value 표시는 실시간 환율 유지 — '표시=실시간')
        settled = await self._overseas_settled_krw()
        _use_settled = bool(settled.get("ok") and self._num(settled.get("krw")) > 0)
        curve_overseas = settled["krw"] if _use_settled else krw
        # 2026-07-22/29: `> 0` 가드는 '해외분은 항상 양수 자산'을 전제했는데, 자체산출 순액은
        # USD 부채(통합증거금 매수)가 주식평가를 넘으면 **음수일 수 있다**. 그 경우 가드에 걸려
        # 통째로 누락되면 US 손익이 안 보이고 총평가가 부채만큼 튄다 → 0이 아니면 부호 그대로
        # 반영한다(0/None 은 '조회 실패·해외 없음' 이라 더해도 의미 없음 = 종전 동작 유지).
        if curve_overseas:
            bp["total_eval"] = self._num(bp.get("total_eval")) + curve_overseas
            bp["overseas_krw"] = curve_overseas
            if _use_settled:
                bp["overseas_settled"] = True
        # 사장 결정 2026-06-01: 대시보드 '현재 총자산'은 KIS 통합총자산(CTRP6548R tot_asst_amt)으로 — HTS와
        # 일치(실시간). 단 이 값은 예수금이 D0 기반이라 자산곡선(total_eval)엔 절대 반영하지 않는다(5/28 D0 금지).
        # 곡선식 total_eval 과 괴리가 크면(>1%) 경고로 표면화 — 'US 평가 동결/날조' 조기탐지.
        asset = await self.kr_account_asset()
        if asset.get("ok") and self._num(asset.get("tot_asst_amt")) > 0:
            _ta = self._num(asset["tot_asst_amt"])
            bp["display_total_asset"] = _ta
            _cv = self._num(bp.get("total_eval"))
            if _cv > 0 and abs(_ta - _cv) / _cv > 0.01:
                logger.warning(f"[총자산 검증] KIS통합 {_ta:,.0f}원 vs 곡선식 {_cv:,.0f}원 "
                               f"({(_ta - _cv) / _cv * 100:+.1f}%) — 구성차(D0예수금/미결제 등) 확인")
                # 2026-06-10 진단(사장 지시: 괴리 해결): 곡선식 vs KIS통합의 구성요소를 1회성 분해해
                # 638K 괴리가 '국내(D+2 vs D0)'에서 오는지 '해외(결제기준 vs 실시간/USD예수금)'에서
                # 오는지 못박는다. 원인 확정 후 진단 로그는 제거하고 타깃 수정한다.
                try:
                    logger.warning(
                        "[총자산 진단] 곡선=국내(유가+D2) %s + 해외결제 %s | 해외실시간(frcr_evlu_tota) %s "
                        "= 주식 %s + USD예수금 %s | KIS통합 6548=%s · 6504=%s · 6010(2)=%s",
                        f"{self._num(bp.get('total_eval_kr')):,.0f}",
                        f"{self._num(curve_overseas):,.0f}",
                        f"{self._num(pk.get('krw_value')):,.0f}",
                        f"{self._num(pk.get('stock_value')):,.0f}",
                        f"{self._num(pk.get('deposit_krw')):,.0f}",
                        f"{self._num(asset.get('tot_asst_amt')):,.0f}",
                        f"{self._num(pk.get('tot_asst_amt')):,.0f}",
                        f"{self._num(settled.get('tot_asst_amt2')):,.0f}")
                except Exception as _de:
                    logger.warning("[총자산 진단] 분해 실패: %s", _de)
        return {"buying_power": bp, "holdings": holdings,
                "holdings_stale": snap.get("holdings_stale", False), "ok": snap.get("ok", False)}

    # ═══════════════════ 국내주식 순위/업종 ═══════════════════
    async def kr_volume_rank(self) -> str:
        d = await self._get_json("/uapi/domestic-stock/v1/quotations/volume-rank", "FHPST01710000",
            {"FID_COND_MRKT_DIV_CODE":"J","FID_COND_SCR_DIV_CODE":"20171",
             "FID_INPUT_ISCD":"0000","FID_DIV_CLS_CODE":"0","FID_BLNG_CLS_CODE":"0",
             "FID_TRGT_CLS_CODE":"111111111","FID_TRGT_EXLS_CLS_CODE":"000000",
             "FID_INPUT_PRICE_1":"","FID_INPUT_PRICE_2":"","FID_VOL_CNT":"",
             "FID_INPUT_DATE_1":""})
        data = d.get("output", [])
        self._last_volrank = data  # cache raw for kr_volume_rank_list()
        lines = ["[거래량 순위 TOP 10]\n"]
        for i, d in enumerate(data[:10], 1):
            lines.append(f"  {i}. {d.get('hts_kor_isnm','')} ({d.get('mksc_shrn_iscd','')}) | "
                         f"{d.get('stck_prpr','')}원 | {d.get('prdy_ctrt','')}% | 거래량: {d.get('acml_vol','')}")
        return "\n".join(lines)

    async def kr_volume_rank_list(self) -> List[Dict]:
        """Parsed volume-rank: [{code,name,price}]. [] on failure. (Reuses the last raw if fresh.)"""
        try:
            data = getattr(self, "_last_volrank", None)
            if data is None:
                await self.kr_volume_rank()
                data = getattr(self, "_last_volrank", []) or []
            out = []
            for d in data:
                px = self._num(d.get("stck_prpr"))
                code = (d.get("mksc_shrn_iscd") or "").strip()
                if code and px > 0:
                    out.append({"code": code, "name": (d.get("hts_kor_isnm") or "").strip(), "price": px})
            return out
        except Exception:
            return []

    async def kr_sector(self, sector_code: str = "0001") -> str:
        d = await self._get_json("/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice", "FHKUP03500100",
            {"FID_COND_MRKT_DIV_CODE":"U","FID_INPUT_ISCD":sector_code,
             "FID_INPUT_DATE_1":(datetime.now()-timedelta(days=30)).strftime("%Y%m%d"),
             "FID_INPUT_DATE_2":datetime.now().strftime("%Y%m%d"),
             "FID_PERIOD_DIV_CODE":"D"})
        data = d.get("output2", [])
        return f"[업종지수 {sector_code}] {len(data)}일 데이터 조회 완료"

    async def kr_index_daily(self, index_code: str = "0001", days: int = 40) -> List[Dict]:
        """지수 일별 종가 (KOSPI=0001, KOSDAQ=1001). 벤치마크 오버레이용.
        반환: [{"date":"YYYY-MM-DD","close":float}, ...] 오름차순. 실패 시 []."""
        end = datetime.now(KST).strftime("%Y%m%d")
        start = (datetime.now(KST) - timedelta(days=days * 2)).strftime("%Y%m%d")
        out: List[Dict] = []
        try:
            resp = await self._get_json("/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice", "FHKUP03500100",
                {"FID_COND_MRKT_DIV_CODE":"U","FID_INPUT_ISCD":index_code,
                 "FID_INPUT_DATE_1":start,"FID_INPUT_DATE_2":end,
                 "FID_PERIOD_DIV_CODE":"D"})
            data = resp.get("output2", []) or []
            for x in data:
                d = (x.get("stck_bsop_date") or "").strip()
                c = x.get("bstp_nmix_prpr") or x.get("stck_clpr") or ""
                if len(d) == 8 and c not in ("", None):
                    try:
                        out.append({"date": f"{d[:4]}-{d[4:6]}-{d[6:8]}", "close": float(c)})
                    except (TypeError, ValueError):
                        pass
            out.sort(key=lambda r: r["date"])
        except Exception as ex:
            logger.warning(f"[지수일봉] {index_code} 조회 예외: {ex}")
        return out

    async def kr_index_now(self, index_code: str = "0001") -> float:
        """KOSPI(0001)/KOSDAQ(1001) 현재 지수값 (float). 실패 시 0.0.
        사장 지시 2026-05-21: 벤치마크 일중 오버레이는 5분 폴링이 equity 포인트에 지수값을
        같이 찍어 만든다. 여기선 검증된 일봉(kr_index_daily)의 '당일' 마지막 종가를 현재값으로
        재사용한다 — 장중엔 당일 바가 실시간 갱신되므로 별도(미검증) 분봉 API가 불필요하다."""
        try:
            rows = await self.kr_index_daily(index_code=index_code, days=3)
            return float(rows[-1]["close"]) if rows else 0.0
        except Exception:
            return 0.0

    # ═══════════════════ 해외주식 ═══════════════════
    # KIS overseas-price (HHDFS00000300) EXCD codes: NAS(나스닥) / NYS(뉴욕) / AMS(아멕스).
    # JNJ·PG are NYSE names so a NASDAQ-only query returns an empty 'last'. Probe these, cache the hit.
    _US_EXCH_CODES = ("NAS", "NYS", "AMS")
    _us_excd_cache: Dict[str, str] = {}
    # 2026-05-19: 모든 거래소가 rt_cd=0(정상)인데 일봉 0건인 티커(예: 상장폐지 "X"=US Steel)를
    # 기억해 이후 사이클의 3회 중복 KIS 호출·경보 로그를 생략. 프로세스 재시작 시 비워져 재검증됨.
    _us_dataless: set = set()

    async def _us_price_raw(self, ticker: str) -> Dict:
        """First non-empty overseas-price 'output' across NAS/NYS/AMS ({} on total failure).
        Remembers which exchange answered for a ticker so later lookups are a single request."""
        tk = (ticker or "").strip().upper()
        if not tk:
            return {}
        tok = await self.token(); s = await self._s()
        codes = list(self._US_EXCH_CODES)
        if tk in self._us_excd_cache:  # try the known exchange first
            codes = [self._us_excd_cache[tk]] + [c for c in codes if c != self._us_excd_cache[tk]]
        for i, excd in enumerate(codes):
            if i:
                await asyncio.sleep(0.3)  # ease KIS TPS between probes
            # 버그수정 2026-06-05: rate-limit(초당 거래건수 초과)면 그 거래소를 '미스'로 건너뛰지 말고
            # 백오프 후 재시도한다 — 올바른 거래소를 rate-limit 때문에 놓쳐 시세 0 → 주문 무음 스킵되던 문제.
            for attempt in range(1, self._RATE_LIMIT_MAX_RETRY + 1):
                try:
                    await self._pace()
                    async with s.get(f"{self.base_url}/uapi/overseas-price/v1/quotations/price",
                        headers=self._h(tok,"HHDFS00000300"),
                        params={"AUTH":"","EXCD":excd,"SYMB":tk}) as r:
                        full = await r.json()
                    if self._resp_rate_limited(full):
                        self._note_rate_limited()
                        logger.warning(f"[US시세] {tk} {excd} rate-limit — "
                                       f"{self._RATE_LIMIT_BACKOFF_SEC*attempt:.2f}s 후 재시도 {attempt}/{self._RATE_LIMIT_MAX_RETRY}")
                        await asyncio.sleep(self._RATE_LIMIT_BACKOFF_SEC * attempt)
                        continue  # 같은 거래소 재시도
                    d = full.get("output", {}) or {}
                    if d.get("last") not in (None, "", "0", "0.0", "0.00", "0.0000"):
                        self._us_excd_cache[tk] = excd
                        return {**d, "_excd": excd}
                    break  # rate-limit 아닌 빈 응답 → 다음 거래소
                except Exception:
                    break
        return {}

    async def us_price(self, ticker: str, excd: Optional[str] = None) -> str:
        if excd:
            tok = await self.token(); s = await self._s()
            await self._pace()
            async with s.get(f"{self.base_url}/uapi/overseas-price/v1/quotations/price",
                headers=self._h(tok,"HHDFS00000300"),
                params={"AUTH":"","EXCD":excd,"SYMB":ticker}) as r:
                d = (await r.json()).get("output",{}) or {}
        else:
            d = await self._us_price_raw(ticker)
        if not d or d.get("last") in (None, ""):
            return f"[US시세] {ticker} | 시세 조회 실패 (거래소 미확인)"
        return f"[US시세] {ticker} | ${d.get('last','?')} | {d.get('rate','')}% | 거래량: {d.get('tvol','')}" + (f" | {d.get('_excd')}" if d.get('_excd') else "")

    async def us_last_price(self, ticker: str) -> float:
        """US 현재가(float, USD). 거래소를 자동 탐색. 실패 시 0.0."""
        d = await self._us_price_raw(ticker)
        try:
            return float(str(d.get("last", "0")).replace(",", ""))
        except (TypeError, ValueError):
            return 0.0

    async def us_daily_chart(self, ticker: str, days: int = 100) -> List[Dict]:
        """미국주식 일봉 조회 (HHDFS76240000) — 거래소(NAS/NYS/AMS) 직접 프로브 후 CSV 누적.
        Returns parsed rows; writes to data/daily_US_{ticker}.csv with the same OHLCV schema as KR.
        사장 지시(2026-05-14): 미국 종목도 historical 데이터를 가져야 퀀트 분석이 정상 작동.

        버그 수정 2026-05-19: 과거엔 실시간시세 프로브(_us_price_raw)로만 거래소를
        알아내, 그게 실패하면 excd가 'NAS'로 폴백돼 NYSE 종목(예: BAC)이 0행이 됐다.
        일봉은 실시간 호가가 필요 없으므로 dailyprice 엔드포인트 자체를 거래소별로
        직접 시도하고 첫 비어있지 않은 응답을 채택한다 (장시간·시세구독과 무관)."""
        tk = (ticker or "").strip().upper()
        if not tk:
            return []
        # 1순위 arcmarket(yfinance, 수정주가 — KIS MODP=1 과 정합). 실패 시 KIS 폴백.
        via = await self._arcmarket_daily_rows(tk, kr=False, days=max(30, int(days * 1.6)), adjusted=True)
        if via:
            via = via[-max(1, days):]
            self._append_csv(f"daily_US_{tk}.csv", via, ["date", "open", "high", "low", "close", "volume"])
            logger.info(f"[US일봉] {tk}: arcmarket {len(via)}건 (yfinance)")
            return via
        if tk in self._us_dataless:
            logger.info(f"[US일봉] {tk} 상장폐지/데이터없음 캐시 — KIS 조회 생략 (재시작 시 재검증)")
            return []
        tok = await self.token(); s = await self._s()
        # 알려진 거래소가 있으면 그것부터, 없으면 NAS→NYS→AMS 순으로 일봉 직접 프로브
        codes = list(self._US_EXCH_CODES)
        if tk in self._us_excd_cache:
            codes = [self._us_excd_cache[tk]] + [c for c in codes if c != self._us_excd_cache[tk]]
        data: List[Dict] = []
        exchange_results = []
        for i, excd in enumerate(codes):
            if i:
                await asyncio.sleep(0.3)  # KIS TPS 완화
            # 버그수정 2026-06-05: rate-limit(초당 거래건수 초과)면 그 거래소를 백오프 후 재시도한다 —
            # CVX(NYS) 처럼 올바른 거래소가 rate-limit으로 0건이 돼 일봉 결손→퀀트/주문 누락되던 문제.
            _out = []; status = "?"; resp_json = {}
            for attempt in range(1, self._RATE_LIMIT_MAX_RETRY + 1):
                try:
                    await self._pace()
                    async with s.get(f"{self.base_url}/uapi/overseas-price/v1/quotations/dailyprice",
                        headers=self._h(tok, "HHDFS76240000"),
                        params={"AUTH":"","EXCD":excd,"SYMB":tk,
                                "GUBN":"0",          # 0=일, 1=주, 2=월
                                "BYMD": datetime.now(KST).strftime("%Y%m%d"),
                                "MODP":"1"}) as r:   # 1=수정주가 반영
                        resp_json = await r.json()
                    if self._resp_rate_limited(resp_json):
                        self._note_rate_limited()
                        logger.warning(f"[US일봉] {tk} {excd} rate-limit — "
                                       f"{self._RATE_LIMIT_BACKOFF_SEC*attempt:.2f}s 후 재시도 {attempt}/{self._RATE_LIMIT_MAX_RETRY}")
                        await asyncio.sleep(self._RATE_LIMIT_BACKOFF_SEC * attempt)
                        continue  # 같은 거래소 재시도
                    status = resp_json.get("rt_cd", "?")
                    _out = resp_json.get("output2", []) or []
                    break
                except Exception as e:
                    logger.warning(f"[US일봉] {tk} {excd} 조회 예외: {e}")
                    _out = []; status = "예외"
                    break
            exchange_results.append({"excd": excd, "status": status, "output2_len": len(_out)})
            if status != "0" and status not in ("?",):
                logger.warning(f"[US일봉] {tk} {excd} rt_cd={status} msg1={resp_json.get('msg1','')} output2_len={len(_out)}")
            elif status == "0" and not _out:
                logger.info(f"[US일봉] {tk} {excd} rt_cd=0 but output2 empty (msg1={resp_json.get('msg1','')})")
            if _out:
                data = _out
                self._us_excd_cache[tk] = excd   # 다음 조회는 단일 요청으로
                break
        if not data:
            details = "; ".join(f"{r['excd']}: status={r['status']}, output2_len={r['output2_len']}" for r in exchange_results)
            _all_clean_empty = (len(exchange_results) == len(codes)
                                and all(r["status"] == "0" and r["output2_len"] == 0 for r in exchange_results))
            if _all_clean_empty:
                # 모든 거래소가 rt_cd=0(정상)인데 데이터 0 → 상장폐지/미지원 종목으로 확정.
                # 캐시 등록해 이후 사이클의 중복 호출·경보를 제거 (프로세스 재시작 시 재검증).
                self._us_dataless.add(tk)
                logger.info(f"[US일봉] {tk} 상장폐지/데이터없음 확정 (전 거래소 rt_cd=0+빈응답) — "
                            f"이후 사이클 조회 생략. 상세: {details}")
            else:
                # 일부 거래소 에러/비정상 → 일시적일 수 있어 캐시하지 않고 경고 유지(다음 사이클 재시도).
                logger.warning(f"[US일봉] {tk} 모든 거래소(NAS/NYS/AMS) output2 비어있음 — 상세: {details}")
            return []
        rows = []
        for x in data[:max(1, days)]:
            d = (x.get("xymd") or "").strip()       # yyyymmdd
            if not d or len(d) < 8: continue
            iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            try:
                rows.append({"date": iso,
                             "open":  float(x.get("open")  or 0),
                             "high":  float(x.get("high")  or 0),
                             "low":   float(x.get("low")   or 0),
                             "close": float(x.get("clos")  or 0),
                             "volume":int(float(x.get("tvol") or 0))})
            except Exception:
                continue
        # newest-first → sort ascending so CSV reads naturally
        rows.sort(key=lambda r: r["date"])
        self._append_csv(f"daily_US_{tk}.csv", rows, ["date","open","high","low","close","volume"])
        return rows

    async def _overseas_order_body(self, ticker: str, qty: int, price: float,
                                   *, side: str, excd: str):
        """KIS 해외주식 주문 바디 조립.

        버그(2026-05-19): KIS 해외주식 주문 TR(TTTT1002U/1006U)은 국내 '01'
        시장가가 없고 유효 지정가 단가를 요구한다. price 미지정(시장가 의도)에
        OVRS_ORD_UNPR="0" 을 보내면 KIS 가 "가격 $0.01 미만 주문불가" 로 거부.
        → 시장가 의도면 현재가 기반 **체결가능 지정가**로 환산해 전송한다.
        현재가 미확보면 0 전송 대신 실패 문자열 반환(원칙 #14: 데이터 결손을
        주문가능 조건으로 오인 금지). 거래소는 시세 프로브가 캐싱한 값을 매핑.
        반환: dict(주문 바디) 또는 str(실패 — 미전송)."""
        tk = (ticker or "").strip().upper()
        explicit = False
        try:
            lp = float(price or 0)
        except (TypeError, ValueError):
            lp = 0.0
        if lp > 0:
            explicit = True   # 호출자가 명시 지정가 → 그대로 사용(버퍼 X)
        else:
            # 시장가 의도 → 다중 폴백으로 끝까지 가격 확보(주문 스킵 금지).
            lp = await self.us_last_price(tk)
            if not lp or lp <= 0:                 # ① 실시간 실패 → 일봉 종가
                try:
                    rows = await self.us_daily_chart(tk, days=5)
                except Exception:
                    rows = []
                if rows:
                    try:
                        lp = float(rows[-1].get("close") or 0)
                    except (TypeError, ValueError):
                        lp = 0.0
            if not lp or lp <= 0:
                # 실시간·일봉 모두 비어 KIS 지정가에 넣을 단가가 물리적으로
                # 없음(상장폐지/미지원 추정). 0 전송은 원래 버그 재현이라 금지.
                return (f"[US{'매수' if side == 'buy' else '매도'} 실패] {tk} "
                        f"현재가·일봉 모두 미확보 — 단가 산출 불가, 주문 미전송")
        if explicit:
            # Preserve the committee's price ceiling/floor, even if unfilled.
            unpr = (math.floor(lp * 100) if side == "buy" else math.ceil(lp * 100)) / 100
        elif side == "buy":    # 매수: 현재가보다 살짝 위(체결 보장), 센트 올림
            unpr = math.ceil(lp * 1.003 * 100) / 100.0
        else:                   # 매도: 현재가보다 살짝 아래, 센트 내림
            unpr = math.floor(lp * 0.997 * 100) / 100.0
        excg = excd_to_excg(self._us_excd_cache.get(tk) or excd)
        c, p = self._acnt()
        return {"CANO": c, "ACNT_PRDT_CD": p, "OVRS_EXCG_CD": excg, "PDNO": tk,
                "ORD_QTY": str(int(qty)), "OVRS_ORD_UNPR": f"{unpr:.2f}",
                "ORD_SVR_DVSN_CD": "0", "ORD_DVSN": "00"}

    async def us_buy(self, ticker: str, qty: int, price: float = 0, excd: str = "NASD") -> str:
        body = await self._overseas_order_body(ticker, qty, price, side="buy", excd=excd)
        if isinstance(body, str):
            return body
        s = await self._s()
        async def _do(tk):
            async with s.post(f"{self.base_url}/uapi/overseas-stock/v1/trading/order",
                headers=self._h(tk,"TTTT1002U"), json=body) as r:
                return await r.json()
        d = await self._authed_json(_do)
        msg = _clean_kis_msg(d.get('msg1', ''))
        # KR(kr_buy)과 대칭: rt_cd≠0(거부)면 [실패] 프리픽스 — 호출부 accepted 휴리스틱이
        # "주문가능금액을 초과" 같은 거부를 잡아 잠정 체결로 오판하지 않게 한다.
        return (f"[US매수] {ticker} {qty}주 @ ${body['OVRS_ORD_UNPR']} → {msg}"
                if d.get("rt_cd") == "0" else
                f"[US매수 실패] {ticker} {qty}주 → {msg}")

    async def us_sell(self, ticker: str, qty: int, price: float = 0, excd: str = "NASD") -> str:
        body = await self._overseas_order_body(ticker, qty, price, side="sell", excd=excd)
        if isinstance(body, str):
            return body
        s = await self._s()
        async def _do(tk):
            async with s.post(f"{self.base_url}/uapi/overseas-stock/v1/trading/order",
                headers=self._h(tk,"TTTT1006U"), json=body) as r:
                return await r.json()
        d = await self._authed_json(_do)
        msg = _clean_kis_msg(d.get('msg1', ''))
        return (f"[US매도] {ticker} {qty}주 @ ${body['OVRS_ORD_UNPR']} → {msg}"
                if d.get("rt_cd") == "0" else
                f"[US매도 실패] {ticker} {qty}주 → {msg}")

    async def us_balance(self) -> str:
        # NASD/NYSE/AMEX 전 거래소를 순회·중복제거하는 _overseas_holdings 를 재사용한다.
        # (과거 버그: 여기서 OVRS_EXCG_CD=NASD 하드코딩이라 NYSE/AMEX 종목이 통째로 누락됐다.)
        holdings = await self._overseas_holdings()
        if not holdings:
            return "[해외 계좌잔고] 보유 종목 없음"
        total_eval = sum(self._num(h.get("qty")) * self._num(h.get("cur_price")) for h in holdings)
        total_pnl = sum(self._num(h.get("pnl_amt")) for h in holdings)
        lines = [f"[해외 계좌잔고] 총평가: ${total_eval:,.2f} | 평가손익: ${total_pnl:,.2f}\n"]
        for h in holdings:
            lines.append(f"  📊 {h['name']} ({h['code']}): {h['qty']}주 | "
                         f"${self._num(h.get('cur_price')):,.2f} | 손익: ${self._num(h.get('pnl_amt')):,.2f}")
        return "\n".join(lines)

    # ═══════════════════ 장내채권 ═══════════════════
    async def bond_price(self, code: str) -> str:
        resp = await self._get_json("/uapi/domestic-bond/v1/quotations/inquire-price", "FHKBJ773401C0",
            {"FID_COND_MRKT_DIV_CODE":"B","FID_INPUT_ISCD":code})
        d = resp.get("output", {})
        return f"[채권시세] {code} | {d.get('bond_prpr','')} | 수익률: {d.get('bond_ytm','')}"

    async def bond_buy(self, code: str, qty: int, price: float) -> str:
        tok = await self.token(); s = await self._s(); c, p = self._acnt()
        await self._pace()
        async with s.post(f"{self.base_url}/uapi/domestic-bond/v1/trading/order",
            headers=self._h(tok,"TTTC0951U"),
            json={"CANO":c,"ACNT_PRDT_CD":p,"PDNO":code,"ORD_DVSN":"00",
                  "ORD_QTY":str(qty),"BOND_ORD_UNPR":str(price)}) as r:
            d = await r.json()
        return f"[채권매수] {code} {qty}매 → {d.get('msg1','')}"

    # ═══════════════════ 해외선물옵션 ═══════════════════
    async def futures_price(self, code: str, excd: str = "CME") -> str:
        resp = await self._get_json("/uapi/overseas-futureoption/v1/quotations/inquire-price", "HHDFS76200200",
            {"EXCD":excd,"SYMB":code})
        d = resp.get("output", {})
        return f"[해외선물] {code} | {d.get('last','?')} | {d.get('rate','')}%"

    async def futures_buy(self, code: str, qty: int, price: float, excd: str = "CME") -> str:
        tok = await self.token(); s = await self._s(); c, p = self._acnt()
        await self._pace()
        async with s.post(f"{self.base_url}/uapi/overseas-futureoption/v1/trading/order",
            headers=self._h(tok,"TTTS6036U"),
            json={"CANO":c,"ACNT_PRDT_CD":p,"OVRS_FUOP_ECNG_MRKT_CD":excd,
                  "PDNO":code,"SLL_BUY_DVSN_CD":"02","ORD_QTY":str(qty),
                  "OVRS_FUOP_LMT_PRIC":str(price),"ORD_DVSN_CD":"00"}) as r:
            d = await r.json()
        return f"[해외선물매수] {code} → {d.get('msg1','')}"

    # ═══════════════════ 국내선물옵션 ═══════════════════
    async def kr_futures_price(self, code: str) -> str:
        resp = await self._get_json("/uapi/domestic-futureoption/v1/quotations/inquire-price", "FHMIF10000000",
            {"FID_COND_MRKT_DIV_CODE":"F","FID_INPUT_ISCD":code})
        d = resp.get("output", {})
        return f"[국내선물] {code} | {d.get('futs_prpr','?')} | {d.get('prdy_ctrt','')}%"

    # ═══════════════════ 통합 잔고 ═══════════════════
    async def get_account_balance(self) -> str:
        kr = await self.kr_balance(); us = await self.us_balance()
        return f"{kr}\n\n{us}"

    # ═══════════════════ 통합 주문 ═══════════════════
    async def place_order(self, order: OrderDraft) -> str:
        if not order.approved:
            return "[주문 거부] 리스크관리실 승인 필요"
        if order.market == "KR":
            if order.side == OrderSide.BUY:
                return await self.kr_buy(order.ticker, order.qty, int(order.limit_price or 0), exchange=order.exchange)
            else:
                return await self.kr_sell(order.ticker, order.qty, int(order.limit_price or 0), exchange=order.exchange)
        elif order.market == "US":
            if order.side == OrderSide.BUY:
                return await self.us_buy(order.ticker, order.qty, order.limit_price or 0)
            else:
                return await self.us_sell(order.ticker, order.qty, order.limit_price or 0)
        elif order.market in ("BOND", "FUTURES"):
            # 채권·선물은 시장가 개념이 없어 유효 지정가가 필수다. price=0 전송은 거부 유발이므로
            # 단가 없이는 주문을 보내지 않는다(원칙: 결손을 주문가능 조건으로 오인 금지).
            if not order.limit_price or order.limit_price <= 0:
                return f"[주문 거부] {order.market} 주문은 지정가(limit_price)가 필수입니다 — 단가 미지정, 미전송"
            if order.market == "BOND":
                return await self.bond_buy(order.ticker, order.qty, order.limit_price)
            return await self.futures_buy(order.ticker, order.qty, order.limit_price)
        return f"[에러] 미지원 시장: {order.market}"

    # ═══════════════════ CSV 누적 ═══════════════════
    def _append_csv(self, filename: str, rows: List[Dict], columns: List[str]):
        """CSV upsert — 같은 키(columns[0]=date/datetime)는 덮어쓴다(당일 봉 갱신·종가 확정).
        2026-06-15: 동일 날짜 스킵으로 당일 봉이 첫 조회값에 고정되던 버그 수정. 겹침 없으면 append."""
        if not rows: return
        key = columns[0]
        path = DATA_DIR / filename
        exists = path.exists()
        existing: List[Dict] = []
        seen = set()
        if exists:
            with open(path, "r", encoding="utf-8") as f:
                for line in csv.DictReader(f):
                    existing.append(line); seen.add(line.get(key, ""))
        new_keys = {str(r.get(key, "")) for r in rows if str(r.get(key, "")) != ""}
        overlap = new_keys & seen
        if not overlap:
            new_rows = [r for r in rows if str(r.get(key, "")) != "" and str(r.get(key, "")) not in seen]
            if not new_rows: return
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=columns)
                if not exists: w.writeheader()
                w.writerows(new_rows)
            logger.info(f"CSV 누적: {filename} +{len(new_rows)}행")
            return
        # 겹침 → 동일 키 덮어쓰기(read-modify-write)
        new_by_key = {str(r.get(key, "")): r for r in rows if str(r.get(key, "")) != ""}
        merged: Dict[str, Dict] = {row.get(key, ""): row for row in existing}
        merged.update(new_by_key)  # 같은 키는 새 값으로 교체
        out = sorted(merged.values(), key=lambda r: r.get(key, ""))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            w.writeheader(); w.writerows(out)
        logger.info(f"CSV 갱신: {filename} ({len(new_by_key)}행 upsert, 총 {len(out)})")

# Phase 2: the global broker singleton is retired. Brokers are owned by UserContext
# (one per uid, built with that uid's injected credentials). This shim raises so any
# stray caller is caught loudly instead of silently trading on the wrong account.
def get_broker():
    raise RuntimeError(
        "get_broker() is retired in Phase 2 — use UserContext.broker (per-uid). "
        "A caller still references the global broker singleton.")

"""
Arquant v1.0 - OpenDart Disclosure Crawler
공시 데이터 크롤링. DART 인증키는 환경변수 OPENDART_API_KEY 로만 주입한다
(config.OPENDART_API_KEY). 절대 소스/문서에 키 값을 박지 말 것.
"""
import aiohttp, logging, re as _re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional
logger = logging.getLogger("DART")
DART_BASE = "https://opendart.fss.or.kr/api"

# ─── 상태 구분 (사장 피드백 2026-05-19 ITEM2a) ─────────────────────────────────
# 리스크관리실장이 "조회 실패"와 "공시 없음(정상)"을 명확히 구분할 수 있도록.
DART_STATE_OK           = "OK"           # 데이터 정상 수신
DART_STATE_NO_DISCLOSURE = "NO_DISCLOSURE"  # 조회 성공했으나 해당 기간 공시/재무 없음 (정상)
DART_STATE_QUERY_FAILED = "QUERY_FAILED"  # API 키 없음·HTTP 오류·타임아웃·파싱 실패 → 시스템 리스크


@dataclass
class DartResult:
    """DART 조회 결과를 명시적 상태로 감싸는 경량 구조체.

    state  : DART_STATE_* 중 하나
    text   : 기존 호출자와 호환되는 사람-읽기 요약 문자열
    payload: 구조화 데이터(재무 dict 등). 없으면 None.
    """
    state: str
    text: str
    payload: Optional[Dict] = None

    # 하위 호환: str() 또는 f-string 에서 text 값만 나오도록
    def __str__(self) -> str:
        return self.text

    @property
    def ok(self) -> bool:
        return self.state == DART_STATE_OK

    @property
    def failed(self) -> bool:
        return self.state == DART_STATE_QUERY_FAILED

    @property
    def no_disclosure(self) -> bool:
        return self.state == DART_STATE_NO_DISCLOSURE

    # 리스크 시스템에서 빠르게 상태 접두사를 붙인 텍스트를 생성
    def risk_text(self) -> str:
        if self.state == DART_STATE_QUERY_FAILED:
            return f"⚠️ [DART 조회 실패 — 시스템 리스크 주의] {self.text}"
        if self.state == DART_STATE_NO_DISCLOSURE:
            return f"ℹ️ [DART 공시 없음 — 특이사항 없음] {self.text}"
        return self.text


# ─── 재무 수치 파싱 (ITEM2b) ───────────────────────────────────────────────────
_AMOUNT_RE = _re.compile(r"([-+]?[\d,]+(?:\.\d+)?)")


def _parse_amount(s: str) -> Optional[float]:
    """재무 계정 금액 문자열('1,234,567원', '-5,000,000' 등)을 float으로 변환.
    파싱 불가 또는 빈 값이면 None 반환."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip().rstrip("원").replace(",", "").replace(" ", "")
    try:
        return float(s)
    except (ValueError, TypeError):
        m = _AMOUNT_RE.search(s)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except (ValueError, TypeError):
                pass
        return None


def parse_balance_sheet_sanity(bs_items: Dict[str, str]) -> Dict:
    """재무상태표 핵심 계정(자산총계/부채총계/자본총계)의 수치를 파싱하고 결정론 검증.

    Returns {
        'assets': float|None,    # 자산총계
        'liabilities': float|None,  # 부채총계
        'equity': float|None,    # 자본총계
        'debt_ratio': float|None,  # 부채총계/자산총계 (0~1+ 범위)
        'state': 'OK'|'PARSE_FAILED'|'IMPOSSIBLE',
        'note': str,             # 사람 읽기 진단 메시지
    }

    state == 'IMPOSSIBLE' : 부채>자산 또는 자본≠자산-부채 불일치 (데이터 오류)
    state == 'PARSE_FAILED': 하나 이상 수치 파싱 실패
    state == 'OK'          : 모든 수치 정상 파싱 및 계리 내적 일관성 확인
    """
    assets      = _parse_amount(bs_items.get("자산총계") or "")
    liabilities = _parse_amount(bs_items.get("부채총계") or "")
    equity      = _parse_amount(bs_items.get("자본총계") or "")

    if assets is None or liabilities is None or equity is None:
        missing = [k for k, v in [("자산총계", assets), ("부채총계", liabilities), ("자본총계", equity)] if v is None]
        return {
            "assets": assets, "liabilities": liabilities, "equity": equity,
            "debt_ratio": None, "state": "PARSE_FAILED",
            "note": f"재무데이터 파싱 불가 — 누락/비파싱 계정: {', '.join(missing)}",
        }

    # 계리 일관성 검증: 자산 = 부채 + 자본 (5% 허용오차)
    reconstructed = liabilities + equity
    if assets != 0:
        discrepancy = abs(reconstructed - assets) / abs(assets)
    else:
        discrepancy = 0.0 if reconstructed == 0 else 1.0

    if liabilities > assets:
        return {
            "assets": assets, "liabilities": liabilities, "equity": equity,
            "debt_ratio": None, "state": "IMPOSSIBLE",
            "note": (f"재무데이터 파싱 불가 — 부채총계({liabilities:,.0f})>자산총계({assets:,.0f}): "
                     f"데이터 오류 또는 단위/부호 혼재. 자본잠식이 의심되면 자본총계({equity:,.0f})가 "
                     f"음수인지 확인하십시오. 이 값으로 부채>자산 판정은 금지."),
        }
    if discrepancy > 0.05:
        return {
            "assets": assets, "liabilities": liabilities, "equity": equity,
            "debt_ratio": liabilities / assets if assets else None,
            "state": "IMPOSSIBLE",
            "note": (f"재무데이터 내적 불일치 — 자산({assets:,.0f}) ≠ 부채({liabilities:,.0f})+자본({equity:,.0f}) "
                     f"(오차 {discrepancy*100:.1f}%). 단위 혼재 또는 연결/별도 구분 오류로 추정. "
                     f"이 데이터 기반 재무 판단은 신뢰 불가."),
        }

    debt_ratio = liabilities / assets if assets > 0 else 0.0
    note = (f"자산총계 {assets:,.0f}원 / 부채총계 {liabilities:,.0f}원 / 자본총계 {equity:,.0f}원 "
            f"→ 부채비율 {debt_ratio*100:.1f}% (자산 대비). 계리 내적 일관성 정상.")
    return {
        "assets": assets, "liabilities": liabilities, "equity": equity,
        "debt_ratio": debt_ratio, "state": "OK",
        "note": note,
    }

async def search_disclosures(corp_name: str = "", bgn_de: str = "", end_de: str = "",
                              pblntf_ty: str = "", page_count: int = 20) -> DartResult:
    """공시 목록 조회. 반환값은 DartResult — str()로 하위 호환 텍스트도 제공.

    state 해석:
      OK           : 공시 데이터 정상 수신
      NO_DISCLOSURE: 조회 성공했으나 해당 기간 공시 없음 (특이사항 없음)
      QUERY_FAILED : API 키 없음·HTTP 오류·타임아웃·파싱 실패 → 시스템 리스크 신호
    """
    from config import OPENDART_API_KEY
    if not end_de: end_de = datetime.now().strftime("%Y%m%d")
    if not bgn_de: bgn_de = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
    if not OPENDART_API_KEY:
        return DartResult(
            state=DART_STATE_QUERY_FAILED,
            text="[DART 공시조회] OPENDART_API_KEY 없음 — 조회 불가 (시스템 리스크)",
        )
    params = {"crtfc_key": OPENDART_API_KEY, "bgn_de": bgn_de, "end_de": end_de,
              "page_count": str(page_count), "sort": "date", "sort_mth": "desc"}
    if corp_name: params["corp_name"] = corp_name
    if pblntf_ty: params["pblntf_ty"] = pblntf_ty
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{DART_BASE}/list.json", params=params, timeout=aiohttp.ClientTimeout(total=15)) as r:
                data = await r.json()
        if data.get("status") != "000":
            return DartResult(
                state=DART_STATE_QUERY_FAILED,
                text=f"[DART 공시조회] API 오류: {data.get('message','')} (status={data.get('status')}) — 시스템 리스크",
            )
        items = data.get("list", [])
        if not items:
            return DartResult(
                state=DART_STATE_NO_DISCLOSURE,
                text=f"[DART 공시] {bgn_de}~{end_de} | 공시 없음 — 특이사항 없음",
            )
        lines = [f"[DART 공시] {bgn_de}~{end_de} | {len(items)}건\n"]
        for i, d in enumerate(items[:15], 1):
            lines.append(f"  {i}. 📋 [{d.get('corp_name','')}] {d.get('report_nm','')}")
            lines.append(f"     접수일: {d.get('rcept_dt','')} | 공시유형: {d.get('pblntf_ty_nm','')}")
        return DartResult(state=DART_STATE_OK, text="\n".join(lines))
    except Exception as e:
        return DartResult(
            state=DART_STATE_QUERY_FAILED,
            text=f"[DART 공시조회 에러] {e} — 시스템 리스크",
        )

# 사장 피드백 2026-05-15 (#7): 직전연도 요약재무상태표 + 손익계산서 + 최근 공시를 한 번에.
# 핵심 계정만 골라 LLM 토큰 절약 (회계 전 계정을 다 보낼 필요는 없음).
_BS_KEYS = ("자산총계", "부채총계", "자본총계", "유동자산", "유동부채", "비유동자산", "비유동부채")
_IS_KEYS = ("매출액", "수익(매출액)", "영업수익", "매출원가", "매출총이익", "판매비와관리비", "영업이익", "영업이익(손실)",
            "법인세비용차감전순이익", "당기순이익", "지배기업소유주지분 순이익")


def _parse_fin_items(items: list) -> tuple:
    """fnlttSinglAcntAll list → (bs_lines, is_lines) 핵심 계정만."""
    bs_lines: List[str] = []
    is_lines: List[str] = []
    for it in items:
        name = (it.get("account_nm") or "").strip()
        amount = (it.get("thstrm_amount") or "").strip()
        sj_div = (it.get("sj_div") or "").strip()  # BS=재무상태표, IS=손익, CIS=포괄손익
        if not name or not amount:
            continue
        if sj_div == "BS" and name in _BS_KEYS and not any(name in l for l in bs_lines):
            bs_lines.append(f"  - {name}: {amount}원")
        elif sj_div in ("IS", "CIS") and name in _IS_KEYS and not any(name in l for l in is_lines):
            is_lines.append(f"  - {name}: {amount}원")
    return bs_lines, is_lines


# 최신성 순서: 3분기 > 반기 > 1분기 > (해당연도 사업보고서). 연도는 올해→작년.
_REPRT_RECENCY = [("11014", "3분기"), ("11012", "반기"), ("11013", "1분기"), ("11011", "사업")]


async def get_financial_summary(corp_code: str, bsns_year: Optional[int] = None) -> DartResult:
    """**가장 최근** 분기/반기/연간 요약재무상태표 + 손익계산서.
    (연도 × 보고서코드)를 최신→과거로 순회하며 데이터가 잡히는 첫 보고서를 반환.
    corp_code는 8자리 DART 고유번호. 사장 피드백 2026-05-18 (최신 재무 강제).

    반환 DartResult.payload = {"bs_sanity": parse_balance_sheet_sanity 결과 dict} (OK/IMPOSSIBLE 시)
    DartResult.state == QUERY_FAILED → API 오류·키 없음·네트워크 장애 (시스템 리스크)
    DartResult.state == NO_DISCLOSURE → 최근 3년 모든 보고서에 재무 미발견 (특이사항 없음)
    DartResult.state == OK → 정상 데이터. 단 payload['bs_sanity']['state'] == 'IMPOSSIBLE' 이면
                             수치 내적 불일치 — 리스크관리실장은 이 데이터로 부채>자산 판정 금지.
    """
    from config import OPENDART_API_KEY
    if not OPENDART_API_KEY:
        return DartResult(
            state=DART_STATE_QUERY_FAILED,
            text="[DART 재무요약] OPENDART_API_KEY 없음 — 조회 불가 (시스템 리스크)",
        )
    this_year = datetime.now().year
    years = [bsns_year] if bsns_year else [this_year, this_year - 1, this_year - 2]
    last_msg = ""
    had_network_error = False
    try:
        async with aiohttp.ClientSession() as s:
            for yr in years:
                for reprt_code, reprt_ko in _REPRT_RECENCY:
                    for fs_div in ("CFS", "OFS"):  # 연결 우선, 없으면 별도
                        params = {"crtfc_key": OPENDART_API_KEY, "corp_code": corp_code,
                                  "bsns_year": str(yr), "reprt_code": reprt_code, "fs_div": fs_div}
                        try:
                            async with s.get(f"{DART_BASE}/fnlttSinglAcntAll.json", params=params,
                                              timeout=aiohttp.ClientTimeout(total=15)) as r:
                                data = await r.json()
                        except Exception as e:
                            last_msg = str(e); had_network_error = True; continue
                        if data.get("status") != "000":
                            last_msg = data.get("message", ""); continue
                        bs_lines, is_lines = _parse_fin_items(data.get("list", []))
                        if not bs_lines and not is_lines:
                            continue
                        kind = "연결" if fs_div == "CFS" else "별도"
                        parts = [f"[DART 요약재무 — {yr}년 {reprt_ko}보고서 ({kind}) · 최신 가용분]"]

                        # ── 재무상태표 수치 결정론 검증 (ITEM2b) ──────────────────
                        bs_dict: Dict[str, str] = {}
                        for it in data.get("list", []):
                            name = (it.get("account_nm") or "").strip()
                            amount = (it.get("thstrm_amount") or "").strip()
                            sj_div = (it.get("sj_div") or "").strip()
                            if sj_div == "BS" and name in _BS_KEYS:
                                bs_dict[name] = amount
                        sanity = parse_balance_sheet_sanity(bs_dict)

                        if sanity["state"] == "IMPOSSIBLE":
                            # 수치 내적 불일치 → 텍스트에 경고 삽입, QUERY_FAILED 로 격상
                            parts.append(f"⚠️ 재무상태표 내적 불일치 — {sanity['note']}")
                            parts.append("(위 불일치로 인해 부채>자산·부채비율 자동 계산 금지. 수치만 참고용.)")
                        elif sanity["state"] == "OK":
                            parts.append(f"✅ 재무상태표 검증: {sanity['note']}")

                        if bs_lines:
                            parts.append("재무상태표 핵심:"); parts.extend(bs_lines)
                        if is_lines:
                            parts.append("손익계산서 핵심:"); parts.extend(is_lines)

                        text = "\n".join(parts)
                        # IMPOSSIBLE이면 상태를 OK 유지하되 payload에 경고 플래그 전달
                        # (호출자가 QUERY_FAILED로 재분류 가능)
                        result_state = DART_STATE_OK
                        if sanity["state"] == "IMPOSSIBLE":
                            result_state = DART_STATE_QUERY_FAILED  # 불일치 데이터는 신뢰 불가
                        return DartResult(state=result_state, text=text,
                                          payload={"bs_sanity": sanity})

        if had_network_error:
            return DartResult(
                state=DART_STATE_QUERY_FAILED,
                text=f"[DART 재무요약] {corp_code}: 네트워크/타임아웃 오류 — 시스템 리스크 ({last_msg})",
            )
        return DartResult(
            state=DART_STATE_NO_DISCLOSURE,
            text=f"[DART 재무요약] {corp_code}: 최근 3년 분기/반기/연간 보고서에서 재무 미발견 ({last_msg}) — 특이사항 없음",
        )
    except Exception as e:
        return DartResult(
            state=DART_STATE_QUERY_FAILED,
            text=f"[DART 재무요약 에러] {e} — 시스템 리스크",
        )


# DART 고유번호 (corp_code 8자리) ↔ stock_code (6자리) 매핑.
# OpenDART corpCode.xml로 한 번 받아서 메모리에 캐싱. 첫 호출 시 다운로드 ~수 초.
_CORP_CODE_CACHE: Dict[str, str] = {}        # stock_code → corp_code
_CORP_NAME_TO_CODE: Dict[str, str] = {}       # corp_name → corp_code
_CORP_CACHE_LOADED = False


async def _load_corp_code_map() -> None:
    """DART corpCode.xml을 다운로드해 stock_code → corp_code 매핑을 캐시.
    한 번만 로드되며 메모리 사용량 ~1MB. 사장 피드백 2026-05-15 #7."""
    global _CORP_CACHE_LOADED
    if _CORP_CACHE_LOADED:
        return
    from config import OPENDART_API_KEY
    if not OPENDART_API_KEY:
        _CORP_CACHE_LOADED = True
        return
    try:
        import zipfile, io, xml.etree.ElementTree as ET
        params = {"crtfc_key": OPENDART_API_KEY}
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{DART_BASE}/corpCode.xml", params=params, timeout=aiohttp.ClientTimeout(total=30)) as r:
                blob = await r.read()
        # corpCode.xml은 zip으로 옴
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".xml"):
                    root = ET.fromstring(zf.read(name).decode("utf-8", errors="ignore"))
                    for el in root.findall("list"):
                        stock_code = (el.findtext("stock_code") or "").strip()
                        corp_code = (el.findtext("corp_code") or "").strip()
                        corp_name = (el.findtext("corp_name") or "").strip()
                        if corp_code:
                            if stock_code:
                                _CORP_CODE_CACHE[stock_code] = corp_code
                            if corp_name:
                                _CORP_NAME_TO_CODE[corp_name] = corp_code
                    break
        logger.info(f"[DART] corpCode 매핑 {len(_CORP_CODE_CACHE)}건 로드 (stock_code 기준)")
    except Exception as e:
        logger.warning(f"[DART] corpCode.xml 로드 실패: {e}")
    finally:
        _CORP_CACHE_LOADED = True


async def get_financial_summary_by_stock_code(stock_code: str) -> DartResult:
    """6자리 stock_code → corp_code → 직전연도 요약재무. DartResult 반환.
    사장 피드백 2026-05-15 #7."""
    if not stock_code or not stock_code.isdigit() or len(stock_code) != 6:
        return DartResult(state=DART_STATE_NO_DISCLOSURE, text="")
    await _load_corp_code_map()
    corp_code = _CORP_CODE_CACHE.get(stock_code)
    if not corp_code:
        # corp_code 매핑 없음 — corpCode.xml 미수록은 DART 데이터 없는 것(정상)
        return DartResult(
            state=DART_STATE_NO_DISCLOSURE,
            text=f"[DART 재무요약] {stock_code}: corp_code 매핑 없음 (corpCode.xml 미수록) — 특이사항 없음",
        )
    return await get_financial_summary(corp_code)


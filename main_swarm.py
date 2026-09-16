"""
Arquant v1.0 - Continuous Market Surveillance & Trading Orchestrator
Watchlist: 10 global indices (KOSPI, KOSDAQ, S&P500, NASDAQ, etc.)
When strategist returns target stocks → 3yr daily + supply crawl + minute chart
"""
import json, re, asyncio, logging, time, difflib, subprocess, types
from enum import Enum
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timezone, timedelta

from agents.base_agent import BaseAgent
from agents.specialists import (create_macro_analyst, create_quant_analyst, create_news_analyst,
                                create_trader, create_post_manager, create_ops_support,
                                create_bond_manager, create_commodity_manager, create_insight_analyst)
from agents.guardrails import create_risk_guard, validate_order_draft
from agents import compliance as compliance_mod   # 컴플라이언스실장 부활(사장 지시 2026-07-22)
from infra.kis_broker import OrderDraft, PriceType, compute_nxt_limit_price
from infra import cycle_store, notifier, metrics, admin_config
from infra import nxt_blacklist, trade_ledger
from infra.error_log import record_error
from infra.market_intel import get_intel_store
from tools.news_monitor import get_monitor
from tools.dart_disclosure import (search_disclosures, get_financial_summary_by_stock_code,
                                    DART_STATE_QUERY_FAILED, DART_STATE_NO_DISCLOSURE)
# 사장 피드백 2026-05-15 (8차): Tavily → alibaba/tongyi-deepresearch로 전환
from tools.global_search import deep_research
from tools.account_weight import compute_stock_weight
from tools.market_data import (
    crawl_index_snapshot, crawl_company_full, format_quant_data_for_agent, INDEX_WATCHLIST,
    get_index_data, format_indices_for_macro, get_stock_name, fetch_investor_data, _csv_row_count,
    resolve_kr_stock_code, get_usdkrw
)
from config import (HEADLINE_DEDUP_RATIO,
                    MACRO_CACHE_TTL_SEC, NEWS_CACHE_TTL_SEC, DART_CACHE_TTL_SEC,
                    LIVE_TRADING, PERIODIC_CYCLE_SEC,
                    NEWS_PREFILTER_TRIGGER, NEWS_PREFILTER_LIMIT)
# 자산슬리브 엔진(채권·원자재 공통) — 순수 함수는 infra.asset_sleeves 단일 진실원천.
from infra.asset_sleeves import (
    SLEEVES, BOND_SLEEVE, COMMODITY_SLEEVE, get_sleeve, all_sleeve_pool_codes,
    sleeve_codes, sleeve_for_code, parse_macro_sleeve_pct, sleeve_pool_for_session,
    split_sleeve_holdings, current_sleeve_weight, size_sleeve_action,
    cap_sleeve_buy_notional, parse_sleeve_decisions, parse_sleeve_weights,
    assemble_sleeve_orders, should_execute_sleeve_buy, normalize_sell_directive,
    build_exec_list, format_sleeve_holdings_block,
)
import runtime  # live strategy overrides — runtime.get("KEY") → override or config default

logger = logging.getLogger("ARQUANT")
KST = timezone(timedelta(hours=9))

SCHEDULE = {
    # 넥스트레이드(NXT) 프리마켓 — 지정가 한정, 실제 거래 가능 시장 (2026-06-03 폐지된 '분석전용 프리장'과 다름)
    # 사장 지시 2026-06-03: 개장 전 '분석전용' 프리장 폐지. 여기서의 KR_PRE_MARKET은 NXT 실거래 세션이다.
    "kr_pre_market":   {"start":(8,0),  "end":(8,50),  "desc":"NXT 프리마켓"},
    "kr_trading":      {"start":(9,0),  "end":(15,30), "desc":"KRX 장중"},
    "kr_close_review": {"start":(15,35),"end":(15,50), "desc":"장 마감 리뷰"},
    # NXT 애프터마켓 연속지정가 — 마감리뷰(15:35–15:50) 이후 15:50 시작 (15:30–15:40 시가단일가 제외)
    "kr_after_market": {"start":(15,50),"end":(20,0),  "desc":"NXT 애프터마켓"},
    "us_trading":      {"start":(22,30),"end":(5,0),   "desc":"US 장중 (야간)"},
}
NEWS_CHECK_INTERVAL = 900     # 뉴스 반영(파일 읽기) 주기 15분 — 실제 크롤은 news-collector.timer 5분 (2026-09-14)
# 사장 피드백 2026-05-16: 뉴스 크롤링·분류는 유저별이 아니라 **단일 스왐 프로세스에서
# 한 번만 수행된다(get_monitor()는 프로세스 전역 싱글턴).
# 결과는 data/news_history.json 에 영속되고 /api/news 가 전체 유저에게 동일하게 제공 →
# 모든 유저가 같은 뉴스를 공유 (개별 크롤링 없음).
# 환율 폴백 — 5분 크롤 라이브 환율(get_usdkrw)이 비어 있을 때만 쓰는 기본값 (사장 지시 2026-05-22).
USDKRW_FALLBACK = 1510.0
# 최저가 KR 종목 1주(현실적 최저가 ~5,000원)도 못 살 예수금이면 분석 비용만 들므로 사이클 스킵.
MIN_TRADABLE_CASH_KRW = 5000
# 전체 두뇌 사이클은 정시 1회만 돈다. 아래 안전감시는 LLM·종목선정·재량매매를
# 호출하지 않고 보유 포지션의 확정 손절 조건만 60초마다 확인한다.
SAFETY_WATCH_INTERVAL_SEC = 60
SAFETY_ORDER_COOLDOWN_SEC = 15 * 60
SAFETY_FAILURE_BACKOFF_SEC = 5 * 60
# 뉴스 크롤이 정각을 몇 초 넘길 수 있어 2분까지는 그 시간의 정기 사이클로 인정한다.
# 이 창 밖에서는 밀린 사이클을 개장 시각에 따라잡지 않고 다음 정각을 기다린다.
HOURLY_TRIGGER_WINDOW_SEC = 2 * 60


def _low_cash_skip_reason(snap) -> Optional[str]:
    """'예수금 부족' 사이클 스킵 사유 — *신뢰 가능한*(ok=True) 잔고 스냅샷에만 적용한다.
    조회 실패(ok=False)는 '돈이 없음'이 아니라 '데이터를 모름'이므로 스킵하지 않는다
    (사장 제보 2026-05-29 — 모의계좌 잔고 일시 실패로 사이클이 통째로 안 도는 버그). None=진행."""
    bp = (snap or {}).get("buying_power") or {}
    if not snap or not snap.get("ok") or not bp.get("ok"):
        return None
    cash = float(bp.get("cash", 0.0) or 0.0)
    if cash < MIN_TRADABLE_CASH_KRW:
        return f"가용 예수금 부족 ({cash:,.0f}원 < {MIN_TRADABLE_CASH_KRW:,}원) — 분석 비용만 듦, 스킵"
    return None


def _is_kr_code(code: Any) -> bool:
    """국내(KR) 종목코드 판정 — 6자리 숫자면 KR, 아니면 US(티커). 이 한 함수로 KR/US 분기를
    통일해 'kr_* 만 부르고 us_* 빠뜨리는' 비대칭 버그의 표면을 좁힌다."""
    s = str(code or "").strip()
    return s.isdigit() and len(s) == 6


def _safety_sell_reason(holding: Dict[str, Any], thesis: Optional[Dict[str, Any]],
                        stop_loss_pct: float) -> Optional[str]:
    """연속 안전감시가 허용하는 결정론적 전량매도 조건. 재량 신호·익절은 여기서 다루지 않는다."""
    try:
        cur = float(holding.get("cur_price") or 0.0)
        pnl = float(holding.get("pnl_pct") or 0.0)
        thesis_stop = float((thesis or {}).get("stop_price") or 0.0)
        strategy_stop = abs(float(stop_loss_pct or 0.0))
    except (TypeError, ValueError):
        return None
    if cur > 0 and thesis_stop > 0 and cur <= thesis_stop:
        return f"포트폴리오 계획 손절가 도달 ({cur:,.2f} ≤ {thesis_stop:,.2f})"
    if strategy_stop > 0 and pnl <= -strategy_stop:
        return f"전략 하드 손절 도달 ({pnl:+.2f}% ≤ -{strategy_stop:.2f}%)"
    return None


def _safety_watch_active(session: str) -> bool:
    """브로커가 실제 주문을 받을 수 있는 세션에서만 안전감시를 연다."""
    return session == "US_TRADING" or is_kr_tradable(session)


# 기업리서치팀장은 '개별 기업'에만 붙는다(사장 지시 2026-07-21) — ETF/ETN/레버리지·인버스·
# 채권/원자재 상품은 기업 분석 대상이 아니므로 이름 힌트로 제외한다.
_ETF_NAME_HINTS = ("ETF", "ETN", "레버리지", "인버스", "KODEX", "TIGER", "ARIRANG", "KBSTAR",
                   "KOSEF", "HANARO", "국고채", "종합채권", "채권액티브", "원유", "금현물", "선물")


def _is_company_code(code: Any, name: str = "") -> bool:
    """개별 기업이면 True, ETF·채권·원자재 등 상품이면 False (이름 힌트 기반)."""
    n = str(name or "").upper()
    return not any(h.upper() in n for h in _ETF_NAME_HINTS)


# ─── Cost-reduction helpers ────────────────────────────────────────────────
def _norm_title(title: str) -> str:
    return re.sub(r"[\s\W_]+", "", (title or "")).lower()

def _is_dup_title(title: str, existing_titles: List[str]) -> bool:
    n = _norm_title(title)
    if not n:
        return True
    for e in existing_titles:
        en = _norm_title(e)
        if not en:
            continue
        if n == en or difflib.SequenceMatcher(None, n, en).ratio() >= HEADLINE_DEDUP_RATIO:
            return True
    return False

# Simple TTL caches for slow-moving inputs (macro analysis text, DART disclosures, deep-research)
_macro_cache = {"ts": 0.0, "value": ""}
# 뉴스 감성 리포트 30분 공유 캐시(사장 지시 2026-07-21) — 매크로와 동형. 프롬프트가 세션에 따라
# 달라지므로('지금 매매 가능한 시장') 세션이 바뀌면 재사용하지 않는다.
_news_cache = {"ts": 0.0, "value": "", "session": ""}
_dart_cache = {"ts": 0.0, "value": ""}
# 사장 피드백 2026-05-15 (8차): 세션별 alibaba 매크로 리서치 결과 30분 캐시 (Tavily 캐시 대체).
_research_cache: Dict[str, Dict] = {}  # session → {ts, value}

def _now_kst(): return datetime.now(KST)

def _current_hour_key():
    """현재 KST 시각을 시(hour) 단위로 내림한 datetime — 사이클 정시(:00) 앵커."""
    return _now_kst().replace(minute=0, second=0, microsecond=0)

def _current_hour_key_str() -> str:
    """공유 스토어 키용 직렬화 (예: '2026-06-08 10')."""
    return _current_hour_key().strftime("%Y-%m-%d %H")

# 사장 지시 2026-06-03: 하드코딩 휴장일 목록 폐지. 주말만 자명 처리하고, 그 외 휴장은
# '오늘 실제 거래(당일 거래량/일봉)가 있었는가'를 KIS 실데이터로만 판정한다. 매년 음력·임시
# 공휴일을 손으로 갱신하다 빠뜨리던 구조적 누락(예: 2026-06-03 지방선거 임시공휴일)을 없앤다.
#
# 실데이터 검증 결과 캐시(시장·날짜 단위, 프로세스 전역 — 휴장은 유저 무관 시장 공통).
#   key: "KR:YYYY-MM-DD"(KST 날짜) · "US:YYYY-MM-DD"(미 동부 거래일)
#   _VERIFIED_TRADED: 당일 봉 확인 → 거래일 / _VERIFIED_CLOSED: 개장 후에도 당일 봉 없음 → 휴장.
# 동기 함수(is_market_session_now 등)는 이 캐시를 읽어 평가금액 기록/표시를 게이팅한다.
_VERIFIED_TRADED: set = set()
_VERIFIED_CLOSED: set = set()

def _mkt_day_key(market: str, d: Optional[datetime] = None) -> str:
    """시장의 '거래일' 키. KR=KST 날짜. US=미 동부 거래일(KST 22시 이후=같은 날, 자정~새벽=전날)."""
    d = d or _now_kst()
    if market == "US":
        ref = d if d.hour >= 22 else (d - timedelta(days=1))
        return "US:" + ref.strftime("%Y-%m-%d")
    return "KR:" + d.strftime("%Y-%m-%d")

def _market_day_verified_closed(market: str, d: Optional[datetime] = None) -> bool:
    """실데이터 검증으로 '오늘 휴장'이 확정된 거래일이면 True (개장으로 재확인되면 False)."""
    k = _mkt_day_key(market, d)
    return k in _VERIFIED_CLOSED and k not in _VERIFIED_TRADED

def is_kr_weekend(d: Optional[datetime] = None) -> bool:
    """KST 기준 주말(토/일)이면 True. 주말은 KIS 호출 없이 자명하게 휴장 처리."""
    d = d or _now_kst()
    return d.weekday() >= 5  # 토(5)·일(6)

def is_us_weekend(d: Optional[datetime] = None) -> bool:
    """미 동부 기준 주말(토/일)을 KST 시각으로 환산해 판정. 주말은 KIS 호출 없이 자명 휴장.
    US 정규장은 KST 22:30~05:00 (미 동부 09:30~16:00)."""
    d = d or _now_kst()
    if d.hour < 5 and d.weekday() in (6, 0):  # KST 일/월 새벽 = 미 동부 토/일
        return True
    if d.hour >= 22 and d.weekday() == 4:  # KST 금 밤 = 미 동부 금 → 거래일(주말 아님)
        return False
    if d.hour >= 22 and d.weekday() in (5, 6):  # KST 토/일 밤 = 미 동부 토/일 → 주말
        return True
    return False

def _in_schedule(name):
    h,m = _now_kst().hour, _now_kst().minute; t = h*60+m
    s = SCHEDULE[name]; st = s["start"][0]*60+s["start"][1]; en = s["end"][0]*60+s["end"][1]
    return (st<=t<en) if st<=en else (t>=st or t<en)
def is_trading_hours():
    # 시간대만 본다(요일·휴장 무관 — 그건 is_market_session_now). NXT 시간외(프리/애프터) 포함:
    # 사이클 실행 게이트가 이 함수를 쓰므로 시간외에도 사이클이 돌려면 여기 포함돼야 한다(사장 지시 2026-06-08).
    return (_in_schedule("kr_pre_market") or _in_schedule("kr_trading")
            or _in_schedule("kr_after_market") or _in_schedule("us_trading"))
def get_current_session():
    if _in_schedule("kr_pre_market"):   return "KR_PRE_MARKET"
    if _in_schedule("kr_trading"):      return "KR_TRADING"
    if _in_schedule("kr_close_review"): return "KR_CLOSE_REVIEW"
    if _in_schedule("kr_after_market"): return "KR_AFTER_MARKET"
    if _in_schedule("us_trading"):      return "US_TRADING"
    return "OFF_HOURS"

# ── 세션→거래소 결정 헬퍼군 (Task 2) ──────────────────────────────────────────
KR_SESSIONS          = ("KR_TRADING", "KR_PRE_MARKET", "KR_AFTER_MARKET", "KR_CLOSE_REVIEW")
KR_TRADABLE_SESSIONS = ("KR_TRADING", "KR_PRE_MARKET", "KR_AFTER_MARKET")  # 리뷰는 매매 X

def is_kr_session(s):        return s in KR_SESSIONS
def is_kr_tradable(s):       return s in KR_TRADABLE_SESSIONS


def _buy_time_window_skip(session, skip_open_min: int, skip_close_min: int) -> str:
    """개장 후 skip_open_min / 마감 전 skip_close_min 안이면 사유 문자열, 아니면 '' (사장 지시 2026-07-21).
    KR 정규장 09:00–15:30, US 정규장 22:30–05:00(KST). 주말·해당없는 세션은 ''(무영향)."""
    try:
        from datetime import datetime as _dt, time as _tm, timedelta as _td
        now = _dt.now(KST)
        if session in ("KR_TRADING", "KR_PRE_MARKET", "KR_AFTER_MARKET", "KR_CLOSE_REVIEW"):
            if now.weekday() >= 5:
                return ""
            open_t, close_t = _tm(9, 0), _tm(15, 30)
        elif session == "US_TRADING":
            open_t, close_t = _tm(22, 30), _tm(5, 0)
        else:
            return ""
        t = now.time()
        if skip_open_min > 0:
            om = (_dt.combine(now.date(), open_t) + _td(minutes=int(skip_open_min))).time()
            if open_t <= t <= om:
                return f"개장 {skip_open_min}분 이내"
        if skip_close_min > 0:
            cm = (_dt.combine(now.date(), close_t) - _td(minutes=int(skip_close_min))).time()
            if cm <= t <= close_t:
                return f"마감 {skip_close_min}분 이내"
        return ""
    except Exception:
        return ""
def is_kr_extended_hours(s): return s in ("KR_PRE_MARKET", "KR_AFTER_MARKET")
def kr_exchange_for_session(s):  # "KRX" | "NXT"
    return "NXT" if is_kr_extended_hours(s) else "KRX"
# ──────────────────────────────────────────────────────────────────────────────

def _post_manager_session_hint(session: str) -> str:
    """사후관리실장(매도 판단) 프롬프트에 넣을 '현재 어느 장이 열려 있는가' 안내문.
    버그 2026-05-23 04:47: 세션 정보가 없어 LLM이 'KR 장중이라 미국 시장 닫힘'으로 환각 →
    US_TRADING 인데도 매도 신호를 '매매 불가'로 무시하고 보유. 입력 보유 종목은 이미 현재
    세션에서 거래 가능한 것만 필터링돼 들어오므로, 여기서 '지금 매도 가능'을 사실로 못박는다."""
    if session == "US_TRADING":
        return ("⚠️ 현재 세션은 **미국 정규장(US_TRADING)** — 지금 열려 있는 시장은 미국이고 한국 장은 마감입니다. "
                "아래 보유 종목은 전부 미국 종목이며 **지금 즉시 매도 가능**합니다. "
                "'미국 시장이 닫혀 매매 불가' 같은 판단은 사실과 다르니 절대 쓰지 말고, 세션을 임의로 추측하지 마십시오.")
    if is_kr_session(session):
        return ("⚠️ 현재 세션은 **한국 장(KR)** — 지금 열려 있는 시장은 한국이고 미국 장은 마감입니다. "
                "아래 보유 종목은 전부 국내 종목이며 **지금 즉시 매도 가능**합니다. "
                "'한국 시장이 닫혀 매매 불가' 같은 판단은 사실과 다르니 절대 쓰지 말고, 세션을 임의로 추측하지 마십시오.")
    return ("현재 장외 — 아래 보유 종목은 다음 개장 사이클에서 매매됩니다. "
            "매도/보유는 오직 매크로·퀀트·뉴스·손익 신호로만 판단하십시오.")

class SwarmState(str, Enum):
    IDLE="IDLE";MONITORING="MONITORING";MACRO_ANALYSIS="MACRO_ANALYSIS"
    QUANT_ANALYSIS="QUANT_ANALYSIS";NEWS_ANALYSIS="NEWS_ANALYSIS"
    ORDER_DRAFTING="ORDER_DRAFTING";RISK_VALIDATION="RISK_VALIDATION"
    EXECUTION="EXECUTION";REPORT="REPORT";COOLDOWN="COOLDOWN"
    ERROR="ERROR";STOPPED="STOPPED";OFF_HOURS="OFF_HOURS";DATA_COLLECTION="DATA_COLLECTION"

class SwarmCycleLog:
    def __init__(self):
        self.started_at = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"); self.state_logs: List[Dict] = []; self.final_report = ""
    def log(self, state, agent, message):
        self.state_logs.append({"timestamp":datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),"state":state,"agent":agent,"message":(message or "")[:20000]})
    def to_dict(self): return {"started_at":self.started_at,"logs":self.state_logs,"final_report":self.final_report}

# ─── Real-time event/response log (data/<uid>/trade_log.json) ───────────────
# Accumulates *forever* (per user request) until the dashboard "초기화" button calls
# clear_event_log(uid). Soft-capped so the file can't grow without bound.
#
# Phase 2 멀티테넌트: 과거엔 전역 claude_response.json 하나에 모든 유저의 에이전트/이벤트가
# 뒤섞여, 동시 운용되는 두 계정의 대시보드가 서로의 에이전트를 봤다(주식운용실장 등 페르소나
# 중복 표시). equity 와 동일하게 유저별 파일(data/<uid>/trade_log.json — user_paths.trade_log_path)
# 로 분리한다. log_response_event/get_recent_events/get_trade_history/clear_* 가 uid 인자를 받는다.
# uid 가 None 이면 파일 기록을 건너뛴다(WS 로만 전달 — 부팅 IDLE 등 전역 시스템 이벤트).
from pathlib import Path as _Path
_RESPONSE_LOG_CAP = 4000  # keep at most this many entries on disk
_DISPLAY_EVENT_TYPES = {"status","news","trigger","agent_msg","cycle_complete",
                        "execution_ready","execution_skipped","trade_executed","trade_failed","error",
                        # 사장 지시 2026-05-21: 체결 신청·장 마감 이벤트 (모바일 알림 4종 중 2종)
                        "order_submitted","market_close"}

def _response_log_path(uid):
    """그 유저의 이벤트/응답 로그 파일 경로. uid 가 None 이면 None(파일 미사용)."""
    if uid is None:
        return None
    from infra import user_paths
    return user_paths.trade_log_path(int(uid))

def _read_response_log(uid=None) -> list:
    p = _response_log_path(uid)
    if p is None:
        return []
    try:
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            return d if isinstance(d, list) else []
    except Exception:
        pass
    return []

def _now_kst_iso() -> str:
    """KST 연월일시분초 timestamp (e.g. '2026-05-13 17:13:57'). Used by every log/trade entry
    so the dashboard can render KST without guessing the server's tz (the OCI box runs UTC).
    Note: legacy callers may pass these to datetime.fromisoformat — '%Y-%m-%d %H:%M:%S' is still
    parseable by fromisoformat (Python 3.11+ accepts space separator)."""
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def log_response_event(entry: dict, uid=None):
    """Append an event/response record to that uid's log (full text, no truncation).
    Phase 2 멀티테넌트: uid 가 주어지면 data/<uid>/trade_log.json 에 적는다. uid 가 None 이면
    파일 기록을 건너뛴다(이벤트는 _broadcast 의 WS 콜백으로 여전히 전달된다)."""
    p = _response_log_path(uid)
    if p is None:
        return  # 전역 시스템 이벤트(uid 없음) — 파일에 남기지 않는다(WS 로만).
    try:
        data = _read_response_log(uid)
        data.append({"ts": _now_kst_iso(), **entry})
        if len(data) > _RESPONSE_LOG_CAP:
            data = data[-_RESPONSE_LOG_CAP:]
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as _e:
        logger.warning(f"이벤트 로그 기록 실패(uid={uid}): {_e}")

def get_recent_events(limit: int = 500, uid=None) -> list:
    """UI replay: the display-relevant events (newest-last), so a page reload restores
    the trade/agent log without re-running anything. uid 별 로그만 반환한다."""
    evs = [e for e in _read_response_log(uid)
           if e.get("source") == "system_event" and e.get("type") in _DISPLAY_EVENT_TYPES]
    return evs[-max(1, int(limit)):]

# ─── Equity curve (for the 수익률 tab) ──────────────────────────────────────
# Phase 2 멀티테넌트: 전역 equity_curve.json 폐지. equity 는 유저별 data/<uid>/equity_curve.json
# 에 기록·조회되며, record_equity/get_equity_series/performance_kpis 가 equity_path 인자를 받는다.

# 사장 지시 2026-05-24: 거래 내역을 '비우기' 해도 승률·매도수·보유일 통계가 사라지지 않도록,
# 비우기 직전에 누적 실현손익 통계를 이 파일에 적립한다. performance_kpis 가 라이브 거래 통계에
# 이 베이스라인을 더해 표시하므로, 화면의 거래 리스트를 비워도 승률이 유지된다.
_PERF_BASELINE = _Path(__file__).parent / "data" / "perf_baseline.json"

def _read_perf_baseline() -> dict:
    try:
        if _PERF_BASELINE.exists():
            d = json.loads(_PERF_BASELINE.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
    except Exception:
        pass
    return {}

def _write_perf_baseline(d: dict):
    try:
        _PERF_BASELINE.parent.mkdir(parents=True, exist_ok=True)
        _PERF_BASELINE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as _e:
        logger.warning(f"perf_baseline 기록 실패: {_e}")

def _parse_ts_any(ts_str) -> Optional[datetime]:
    """Parse a stored ts that may be ISO ('2026-05-13T17:13:57.123+09:00') or
    new-format ('2026-05-13 17:13:57'). Returns naive local datetime (KST) or None."""
    s = (ts_str or "").strip()
    if not s:
        return None
    try:
        if "T" in s or "+" in s:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        try:
            return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

def _detect_external_flow(prev: Optional[Dict], curr: Dict) -> float:
    """입출금(외부 현금흐름) 자동 감지 — 사장 지시 2026-05-21 비활성화.

    KIS 총평가/예수금이 폴 간 일시적으로 크게 흔들릴 때(해외종목 시세 일시 0/누락 추정)
    이를 '입금'으로 오탐해 누적 보정값을 오염시켰다: 2026-05-20 15:44 +3,263,616원 유령
    입금이 누적되어 자산곡선이 실제 ~3.76M 대신 ~415K로 표시됨(사장 보고). 금액 임계값으로는
    진짜 입금과 이 변동을 구분할 수 없어, 신뢰 불가한 자동 감지를 끈다.
    이제 adj_total_eval == total_eval (실계좌 총평가) 라 그래프가 실제 자산을 그대로 표시한다."""
    return 0.0

def record_equity(equity_path, bp: dict, source: str = "poll", holdings: Optional[List[Dict]] = None,
                  kospi: Optional[float] = None, nasdaq: Optional[float] = None,
                  ledger_eval: Optional[float] = None, is_mock: bool = False,
                  reconcile_adj: float = 0.0):
    """Append a {ts,total_eval,cash,pnl_ratio,holdings,external_flow_cum,kospi,nasdaq} point — at most one per 60s.
    Caps at 2000 points. `holdings` (optional list of {code,qty}) is used to detect external cashflow
    (deposits/withdrawals) vs trade-driven changes — see _detect_external_flow().
    `kospi`/`nasdaq` (사장 지시 2026-05-21): 5분 폴링이 잔고와 함께 수집한 지수 현재값. equity 포인트와
    동일 타임스탬프에 저장돼, 벤치마크 오버레이가 일중 움직임을 보여준다(미검증 분봉 API 불필요).

    Phase 2 멀티테넌트: `equity_path`(유저별 data/<uid>/equity_curve.json)를 명시적으로 받아
    해당 유저 곡선에만 기록한다. 전역 _EQUITY_LOG 는 더 이상 쓰지 않는다."""
    equity_path = _Path(equity_path)
    if holdings is not None:
        try:
            bp = {**bp, "_holdings": holdings}
        except Exception: pass
    try:
        if not bp or not bp.get("ok"):
            return
        data = []
        if equity_path.exists():
            try: data = json.loads(equity_path.read_text(encoding="utf-8"))
            except Exception: data = []
        if not isinstance(data, list): data = []
        # 사장 보고 2026-05-21: ts 를 KST 로 저장해야 차트 가로축이 KST 로 표시된다.
        # (OCI 서버는 UTC 로 동작 → datetime.now() 는 UTC naive. _ts_to_kst 는 공백포맷을
        #  '이미 KST'로 간주하므로, 여기서 UTC 로 저장하면 9시간 어긋난 라벨이 나왔다.)
        now = datetime.now(KST)
        if data:
            try:
                last = _parse_ts_any(data[-1]["ts"])
                if last:
                    last_aware = last if last.tzinfo else last.replace(tzinfo=KST)
                    if (now - last_aware).total_seconds() < 60:
                        return
            except Exception: pass
        # holdings snapshot {code: qty} — used to distinguish external cashflow from trade-driven changes
        try:
            holdings_snap = {str(h.get("code","")).strip(): int(h.get("qty") or 0)
                             for h in (bp.get("_holdings") or []) if h.get("code")}
        except Exception:
            holdings_snap = {}
        entry = {"ts": now.strftime("%Y-%m-%d %H:%M:%S"),
                 "total_eval": float(bp.get("total_eval") or 0.0),
                 "cash": float(bp.get("cash") or 0.0),
                 "pnl_ratio": float(bp.get("pnl_ratio") or 0.0),
                 "src": source}
        # 사장 지시 2026-06-11: 실거래 원장 평가(KIS 집계 TR 비의존). 이 값이 있는 포인트가
        # 곡선·KPI 의 1순위 시리즈가 된다 (_equity_points 참조).
        # 사장 지시 2026-06-16: 단, 모의계정은 제외한다 — 원장이 KIS 실보유와 어긋나는 드리프트가
        # 있어(2026-07-22 실측: 114260 KIS 111주 vs 원장 129주) 모의 곡선의 1순위로 쓰기엔 위험하다.
        # 모의는 KIS total_eval 을 쓴다. 단 그 total_eval 은 이제 US 평가를 포함한다 — 2026-07-22
        # 부터 kis_broker._overseas_selfcalc_krw 가 '수량×현재가×실환율 + USD예수금' 으로 순평가를
        # 자체 산출한다(모의 기준환율 218.31 은 무시). 종전 주석의 'exrt 스케일로 체결' 근거는
        # 실측으로 반증됐다: 모의는 US 매수 때 KRW 를 아예 차감하지 않고 USD 부채로 남긴다.
        try:
            if ledger_eval is not None and float(ledger_eval) > 0 and not is_mock:
                entry["ledger_eval"] = float(ledger_eval)
        except (TypeError, ValueError):
            pass
        # 사장 지시 2026-06-17: 리시드/허수정정으로 ledger_eval 이 단차로 변하면 그 델타를
        # reconcile_adj 로 박는다 — 자산곡선 cum_pnl 이 이를 '매매손익'이 아닌 '장부정정'으로
        # carry-forward 하게 한다(가짜 -31만원 손실 방지). 표시 평가액은 실제값 그대로.
        try:
            if reconcile_adj and not is_mock:
                entry["reconcile_adj"] = float(reconcile_adj)
        except (TypeError, ValueError):
            pass
        if holdings_snap:
            entry["holdings"] = holdings_snap
        # 사장 지시 2026-05-21: 잔고와 함께 수집한 지수 현재값을 같이 저장 (벤치마크 일중 오버레이)
        try:
            if kospi and float(kospi) > 0:  entry["kospi"]  = float(kospi)
            if nasdaq and float(nasdaq) > 0: entry["nasdaq"] = float(nasdaq)
        except (TypeError, ValueError):
            pass
        # 사장 지시 2026-07-22: 지수를 안 넘기는 경로(source="cycle")와 폴 실패 포인트가
        # 벤치마크 선을 끊어놓았다(uid2 결측률 48.8%). 값이 없으면 직전 포인트의 지수를
        # 그대로 이어받아(carry-forward) 선이 끊기지 않게 한다 — 없는 값을 지어내는 게 아니라
        # '직전 관측 유지'이므로 5분 폴링 해상도에서 왜곡이 없다.
        _prev = data[-1] if data else None
        if _prev:
            for _k in ("kospi", "nasdaq"):
                if not entry.get(_k) and _prev.get(_k):
                    entry[_k] = _prev[_k]
        # Detect external cashflow (deposits/withdrawals) — total_eval delta unexplained by holdings change.
        # Carry forward cumulative external flow so charts can show a "trade-only" equity curve.
        prev_flow = 0.0
        if data:
            try: prev_flow = float(data[-1].get("external_flow_cum", 0.0) or 0.0)
            except Exception: prev_flow = 0.0
        entry["external_flow_cum"] = prev_flow
        flow_delta = _detect_external_flow(data[-1] if data else None, entry)
        if flow_delta:
            entry["external_flow_cum"] = prev_flow + flow_delta
            entry["external_flow"] = flow_delta
            logger.info(f"[equity] 외부 현금흐름 감지: {flow_delta:+,.0f}원 (입출금) — 수익률 베이스라인 보정")
        data.append(entry)
        equity_path.parent.mkdir(parents=True, exist_ok=True)
        equity_path.write_text(json.dumps(data[-2000:], ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as _eq_e:
        # 동작 보존(여전히 raise 안 함) + 조용한 실패 표면화: equity 영속 실패는
        # 대시보드 P&L 을 말없이 고착시킨다.
        metrics.incr("equity_record_error")
        notifier.alert("WARN", "equity 기록 실패", str(_eq_e),
                       dedup_key="equity_record_error")

def _ts_to_kst(ts_str: str) -> Optional[datetime]:
    """Parse an ISO ('2026-05-13T17:13:57+09:00') or new-format ('2026-05-13 17:13:57') timestamp.
    Legacy ISO entries without tz are treated as UTC (the OCI server runs UTC, old format was naive UTC).
    New-format entries (space separator) are already KST (we strftime in KST). Returns KST-aware datetime."""
    s = (ts_str or "").strip()
    if not s:
        return None
    try:
        if "T" in s or "+" in s or "Z" in s:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(KST)
        dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=KST)
    except Exception:
        return None

def is_market_session_now(dt: Optional[datetime] = None) -> bool:
    """지금(또는 dt)이 '실제 정규장 세션'인지 — 시간대 + 요일 + 휴장(실데이터 검증)까지 본다.
    is_trading_hours()/get_current_session()은 시간대만 보므로 주말 밤 US 시간대
    (토 22:30~일 05:00 KST = 실제 미국 토요일·휴장)를 장중으로 오인한다. 평가금액 기록·차트 표시는
    이 함수로 게이팅해 장외/주말/휴장 포인트가 쌓이거나 표시되지 않게 한다 (사장 지시 2026-05-24).
    휴장 판정(사장 지시 2026-06-03): 하드코딩 목록을 쓰지 않고, ① 주말은 자명 휴장,
    ② 그 외 평일은 '거래량 검증으로 휴장이 확정된 날'(_VERIFIED_CLOSED)만 휴장 처리한다.
    개장 후 _verify_market_open 이 당일 봉 유무로 이 캐시를 채운다. 검증 전(평일 기본)엔 개장으로 본다.
    KR 정규장 09:00-15:30(평일) 또는 US 정규장 22:30-05:00(미 거래일)이면 True."""
    dt = dt or _now_kst()
    t = dt.hour * 60 + dt.minute
    # NXT 프리마켓(08:00–08:50)·애프터마켓(15:50–20:00) — KRX와 동일 거래일(주말/휴장 게이트 공유)
    if (8*60 <= t < 8*60+50) or (15*60+50 <= t < 20*60):
        return not is_kr_weekend(dt) and not _market_day_verified_closed("KR", dt)
    if 9 * 60 <= t < 15 * 60 + 30:        # KR 정규장
        return not is_kr_weekend(dt) and not _market_day_verified_closed("KR", dt)
    if t >= 22 * 60 + 30 or t < 5 * 60:   # US 정규장 (야간 wrap)
        return not is_us_weekend(dt) and not _market_day_verified_closed("US", dt)
    return False

def _hint_conflicts_authority(hint_pct, auth_pct) -> bool:
    """매도 '인용 평가손익'(hint_pct)이 권위 실현손익(auth_pct)과 부호가 반대면 True.
    재매수가 미결제(D+2)인 동안 KIS 평단이 옛값으로 잡혀 인용이 실제와 반대 부호로 기록되던
    오해(uid1 375500: 인용 +12.9% vs 권위 실현 -2.7%/-12,500원) 차단 — 충돌 시 호출부가 인용을
    숨긴다. 둘 중 하나라도 None/0 이면 False(판단 보류)."""
    try:
        if hint_pct is None or auth_pct is None:
            return False
        h = float(hint_pct); a = float(auth_pct)
    except (TypeError, ValueError):
        return False
    if h == 0 or a == 0:
        return False
    return (h > 0) != (a > 0)


def _equity_points(raw_equity, *, glitch_pct: float = 0.10):
    """정렬된 [(dt, adj_total, point)] — 입출금 보정(adj) 적용 + 결제 글리치 carry-forward.

    사장 지시 2026-06-11(수익률 환각 수정): 실거래 원장 평가(ledger_eval)가 기록된 포인트가
    하나라도 있으면 '그 시리즈만' 사용한다. KIS 집계 TR(총평가)은 3종 TR 자기불일치·USD 결제
    과도기·해외평가 증발(외화예수금 미포함)로 가짜 -43%류 수익률을 만들었다 — 원장 평가는
    우리 체결만으로 굴러가 결정론적이다. (원장 포인트가 없으면 기존 KIS 곡선 로직 유지.)

    사장 지시 2026-05-22: 보유 종목 변동이 없는데 총평가가 비정상 급변(>glitch_pct)하면 KIS
    결제 과도기 글리치로 보고 직전 값을 유지(carry-forward)해, 누적수익·MDD·그래프에 가짜
    스파이크가 끼지 않게 한다. (보유가 실제로 바뀐 시점은 정상 변동이라 그대로 둔다.)"""
    ledger_pts = []
    for p in (raw_equity or []):
        if not isinstance(p, dict):
            continue
        try:
            lv = float(p.get("ledger_eval") or 0.0)
        except (TypeError, ValueError):
            lv = 0.0
        if lv <= 0:
            continue
        dt = _ts_to_kst(p.get("ts", ""))
        if dt:
            ledger_pts.append((dt, lv, {**p, "adj_total_eval": lv}))
    if ledger_pts:
        ledger_pts.sort(key=lambda x: x[0])
        return ledger_pts
    enriched = []
    for p in (raw_equity or []):
        if not isinstance(p, dict) or not p.get("total_eval"):
            continue
        dt = _ts_to_kst(p.get("ts", ""))
        if not dt:
            continue
        try:
            ext = float(p.get("external_flow_cum", 0.0) or 0.0)
        except Exception:
            ext = 0.0
        try:
            adj = float(p["total_eval"]) - ext
        except Exception:
            continue
        enriched.append((dt, adj, p))
    enriched.sort(key=lambda x: x[0])
    out = []
    prev_total = None
    prev_hold = None
    n = len(enriched)
    for i, (dt, adj, p) in enumerate(enriched):
        hold = p.get("holdings")
        use = adj
        # 사장 지시 2026-05-24 (강화): 결제 글리치(전산오류) 배제 강화.
        # 실제 거래로 보유 수량이 '명시적으로' 바뀐 경우는 정상 변동이라 그대로 둔다.
        holdings_changed = (prev_hold is not None and hold is not None and hold != prev_hold)
        if (prev_total is not None and prev_total > 0
                and not holdings_changed
                and abs(adj - prev_total) / prev_total > glitch_pct):
            # (a) 보유 수량이 직전과 '동일함이 확인'되면 결제 글리치로 보고 즉시 직전값 유지(기존 동작).
            holds_same_known = (prev_hold is not None and hold is not None and hold == prev_hold)
            # (b) 보유 스냅샷이 미상(holdings 필드 없음)이면, 다음 포인트가 직전값으로 되돌아오는
            #     '일시 스파이크'일 때만 글리치로 본다 — 며칠에 걸친 실제 손익(되돌림 없음)은 보존.
            reverts = (i + 1 < n and abs(enriched[i + 1][1] - prev_total) / prev_total <= glitch_pct)
            if holds_same_known or reverts:
                use = prev_total  # 글리치 의심 — 직전 값 유지
        out.append((dt, use, {**p, "adj_total_eval": use}))
        prev_total = use
        if hold is not None:
            prev_hold = hold
    return out


def get_equity_series(equity_path, limit: int = 500, view: str = "realtime") -> list:
    """Return equity-curve points for the dashboard chart.
    - view='realtime' : KR/US 거래시간 포인트만, 5분 버킷의 마지막 값으로 다운샘플 (라벨은 'MM-DD HH:MM' KST)
    - view='daily'    : KST 일자별 마지막 포인트 (라벨 'YYYY-MM-DD')
    - view='monthly'  : KST 월별 마지막 포인트 (라벨 'YYYY-MM')
    `limit`은 최종 반환 포인트 수의 상한.
    Each point's `total_eval` is the RAW account value; `adj_total_eval` is the
    deposit/withdrawal-adjusted value (subtracts cumulative external cashflow so the
    chart shows trade-driven P&L only — per 사장 지시 2026-05-14).
    Phase 2 멀티테넌트: `equity_path` 로 요청 유저의 곡선을 읽는다."""
    equity_path = _Path(equity_path)
    try:
        if not equity_path.exists():
            return []
        raw = json.loads(equity_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return []
    except Exception:
        return []
    # 사장 지시 2026-05-22: 결제 글리치 carry-forward + 누적수익(cum_pnl, 0원 시작) 부여.
    cleaned = _equity_points(raw)
    if not cleaned:
        return []
    _baseline = cleaned[0][1]
    # 사장 지시 2026-06-17: 리시드/허수정정 단차(reconcile_adj)는 매매손익이 아니므로 cum_pnl
    # 에서 carry-forward 한다(상쇄). 표시 평가액(adj_total_eval)은 실제값 그대로 두고, 수익선만
    # 정정 단차를 제거 — 허수 제거가 '가짜 손실'로 보이지 않게 한다.
    try:
        _recon0 = float(cleaned[0][2].get("reconcile_adj") or 0.0)
    except Exception:
        _recon0 = 0.0
    _recon_cum = 0.0
    enriched = []
    for dt, adj, p in cleaned:
        try:
            _recon_cum += float(p.get("reconcile_adj") or 0.0)
            p = {**p, "cum_pnl": (adj - _baseline) - (_recon_cum - _recon0)}
        except Exception:
            pass
        enriched.append((dt, p))
    if not enriched:
        return []
    out = []
    if view == "daily":
        bucket: Dict[str, tuple] = {}
        for dt, p in enriched:
            bucket[dt.strftime("%Y-%m-%d")] = (dt, p)
        for k in sorted(bucket):
            dt, p = bucket[k]
            out.append({**p, "label": k, "ts_kst": dt.strftime("%Y-%m-%d %H:%M")})
    elif view == "monthly":
        bucket: Dict[str, tuple] = {}
        for dt, p in enriched:
            bucket[dt.strftime("%Y-%m")] = (dt, p)
        for k in sorted(bucket):
            dt, p = bucket[k]
            out.append({**p, "label": k, "ts_kst": dt.strftime("%Y-%m-%d %H:%M")})
    else:  # realtime
        bucket: Dict[str, tuple] = {}
        for dt, p in enriched:
            if not is_market_session_now(dt):
                continue
            # 사장 지시 2026-05-21: 5분 체결 확인 폴링 주기에 맞춰 5분 버킷으로 다운샘플
            key = dt.strftime("%Y-%m-%d %H:") + f"{(dt.minute // 5) * 5:02d}"
            bucket[key] = (dt, p)
        for k in sorted(bucket):
            dt, p = bucket[k]
            label = dt.strftime("%m-%d %H:") + f"{(dt.minute // 5) * 5:02d}"
            out.append({**p, "label": label, "ts_kst": dt.strftime("%Y-%m-%d %H:%M")})
    return out[-max(1, int(limit)):]


def _sell_pnl(e: dict):
    """매도 거래 하나에서 (detail, 실현손익) 추출. 매도가 아니거나 손익 미상이면 None.
    realized_pnl 우선, 없으면 total_pnl 폴백 — _trade_realized_stats/_realized_perf_buckets 공유."""
    if str((e or {}).get("side") or "").lower() != "sell":
        return None
    det = e.get("detail") or {}
    pnl = det.get("realized_pnl")
    if pnl is None:
        pnl = det.get("total_pnl")
    return None if pnl is None else (det, pnl)


def _trade_realized_stats(trades: Optional[list]) -> dict:
    """매도 거래의 FIFO/KIS 실현손익으로 승률·매도수·보유일 합계를 집계한다.
    performance_kpis(표시) 와 clear_trade_log(비우기 시 베이스라인 적립) 가 공유한다."""
    wins = total = 0
    hold_sum = 0.0; hold_n = 0
    for e in (trades or []):
        sp = _sell_pnl(e)
        if sp is None:
            continue
        det, pnl = sp
        total += 1
        if pnl > 0:
            wins += 1
        sell_dt = _ts_to_kst(e.get("ts", ""))
        for m in (det.get("matched") or []):
            bdt = _ts_to_kst(m.get("buy_ts", ""))
            if sell_dt and bdt:
                hold_sum += max(0.0, (sell_dt - bdt).total_seconds() / 86400.0); hold_n += 1
    return {"sell_count": total, "win_count": wins, "hold_days_sum": hold_sum, "hold_days_n": hold_n}


# 사장 지시 2026-05-27: 해외(US) 주식 거래비용 — 매수·매도 각 leg 0.3%. 국내(KR)는 미적용.
US_TRADE_COST_RATE = 0.003


def _net_realized_pnl(buy_price, sell_price, qty, is_us: bool):
    """체결 실현손익(거래비용 반영). 국내 0%, 해외 매수·매도 각 0.3%.
    net = (매도가-매수가)×수량 − (해외면) 0.003×(매수가+매도가)×수량.
    값이 비어있으면(None/0) None 반환(미상)."""
    if not (buy_price and sell_price and qty):
        return None
    gross = (sell_price - buy_price) * qty
    if not is_us:
        return gross
    cost = US_TRADE_COST_RATE * (buy_price + sell_price) * qty
    return gross - cost


def _realized_perf_buckets(trades: Optional[list], now: datetime, fx: Optional[float] = None) -> dict:
    """체결된 매도의 (비용반영) 실현손익을 누적/오늘/주/월로 집계 — 결정론적, 잔고스냅샷 비의존.
    각 매도 detail.realized_pnl(_enrich_trade_history 가 비용반영해 기록) 과 매입원금(cost_basis×qty)을
    버킷별로 합산해, 잔고 글리치에 흔들리지 않는 수익률(원/%)을 만든다 (사장 지시 2026-05-27).
    %는 pnl / 매입원금합 × 100. realized_pnl 이 없으면 total_pnl 폴백, 둘 다 없으면 건너뛴다.

    사장 지시 2026-05-30: 해외(USD) 실현손익·매입원금은 원/달러를 그대로 더하면 무의미하므로
    환율(fx)로 원화 환산해 합산한다. fx 미지정 시 5분 크롤 라이브 환율(get_usdkrw)로 채운다.
    detail.currency 가 'USD' 인 항목만 환산하고, 없으면(레거시/국내) 원화로 간주한다."""
    if fx is None:
        try:
            fx = get_usdkrw(USDKRW_FALLBACK)
        except Exception:
            fx = USDKRW_FALLBACK
    today0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week0 = today0 - timedelta(days=today0.weekday())
    month0 = today0.replace(day=1)
    # key → [pnl_sum, basis_sum]
    acc = {"cum": [0.0, 0.0], "today": [0.0, 0.0], "week": [0.0, 0.0], "month": [0.0, 0.0]}
    has = False
    for e in (trades or []):
        sp = _sell_pnl(e)
        if sp is None:
            continue
        det, pnl = sp
        try:
            qty = float(det.get("qty") or 0)
        except (TypeError, ValueError):
            qty = 0.0
        try:
            basis_per = float(det.get("cost_basis") or 0)
        except (TypeError, ValueError):
            basis_per = 0.0
        basis = basis_per * qty
        if basis <= 0:  # cost_basis 없으면 FIFO 매칭 매수금액으로 매입원금 추정
            basis = sum((m.get("buy_price") or 0) * (m.get("buy_qty") or 0)
                        for m in (det.get("matched") or []))
        # 사장 지시 2026-05-30: 해외(USD) 손익·원금을 원화로 환산 — KRW/USD 혼합 합산 방지.
        if str(det.get("currency") or "").upper() == "USD":
            rate = fx or USDKRW_FALLBACK
            pnl = pnl * rate
            basis = basis * rate
        sdt = _ts_to_kst(e.get("ts", ""))
        has = True
        acc["cum"][0] += pnl; acc["cum"][1] += basis
        for key, boundary in (("today", today0), ("week", week0), ("month", month0)):
            if sdt and sdt >= boundary:
                acc[key][0] += pnl; acc[key][1] += basis

    def _pct(pnl, basis):
        return (pnl / basis * 100.0) if basis else 0.0
    return {
        "cumulative_pnl": acc["cum"][0], "cumulative_pct": _pct(*acc["cum"]),
        "today_pnl": acc["today"][0], "today_pct": _pct(*acc["today"]),
        "week_pnl": acc["week"][0], "week_pct": _pct(*acc["week"]),
        "month_pnl": acc["month"][0], "month_pct": _pct(*acc["month"]),
        "_realized_has": has,
    }


def _equity_change_buckets(pts, now: datetime) -> dict:
    """자산곡선(adj total) 기준 평가금액 변동액(원)·변동률(%) — 전체/오늘/이번주/이번달.
    각 기간 시작 '직전' 마지막 평가금액을 기준값으로 현재값과의 차이를 낸다(전일 종가식).
    곡선이 기간 안에서 시작했으면(직전 포인트 없음) 그 기간 첫 포인트를 기준으로 폴백.
    실현손익 버킷(_realized_perf_buckets)과 달리 미실현 평가까지 반영한 mark-to-market 변동이다.
    pts: [(dt, adj_total)] 정렬 리스트(_equity_points 결과). 사장 지시 2026-06-09."""
    if not pts:
        return {}
    cur = pts[-1][1]
    first = pts[0][1]
    today0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week0 = today0 - timedelta(days=today0.weekday())
    month0 = today0.replace(day=1)

    def _baseline(boundary):
        before = [v for dt, v in pts if dt < boundary]
        if before:
            return before[-1]               # 기간 시작 직전 마지막 평가금액
        after = [v for dt, v in pts if dt >= boundary]
        return after[0] if after else None  # 곡선이 기간 안에서 시작 → 그 첫 포인트 기준

    def _chg(base):
        if base is None:
            return None, None
        d = cur - base
        return d, ((d / base * 100.0) if base else None)

    out = {"eq_all_chg": cur - first, "eq_all_pct": ((cur - first) / first * 100.0) if first else None}
    for key, boundary in (("today", today0), ("week", week0), ("month", month0)):
        c, p = _chg(_baseline(boundary))
        out[f"eq_{key}_chg"] = c
        out[f"eq_{key}_pct"] = p
    return out


def performance_kpis(equity_path=None, raw_equity: Optional[list] = None,
                     trades: Optional[list] = None, now: Optional[datetime] = None,
                     uid=None) -> dict:
    """수익률 탭 KPI 카드용 요약 (사장 지시 2026-05-21).

    equity_curve 의 입출금 보정값(adj total) 기준으로 누적·오늘·이번주·이번달 수익(원/%)
    과 최대낙폭(MDD)을, 거래 FIFO 실현손익 기준으로 승률·평균 보유일을 계산한다.
    `raw_equity`/`trades`/`now` 를 주입하면 디스크 없이 순수 계산 — 단위 테스트용.
    Phase 2 멀티테넌트: `equity_path`(유저별)를 주면 그 곡선을 읽는다."""
    now = now or datetime.now(KST)
    if raw_equity is None:
        try:
            ep = _Path(equity_path) if equity_path is not None else None
            raw_equity = (json.loads(ep.read_text(encoding="utf-8"))
                          if (ep is not None and ep.exists()) else [])
            if not isinstance(raw_equity, list):
                raw_equity = []
        except Exception:
            raw_equity = []

    # 평가금액 곡선: current(현재 자산가치)·MDD(자산 낙폭) 표시용으로만 쓴다.
    # 사장 지시 2026-05-22: 결제 글리치(보유 불변·총액 급변)를 carry-forward 해 가짜 낙폭/급등 배제.
    _pts3 = _equity_points(raw_equity)
    pts = [(dt, v) for dt, v, _p in _pts3]
    # 사장 지시 2026-06-18(버그 A): '평가금액 변동' KPI 도 get_equity_series.cum_pnl 처럼
    # reconcile_adj(리시드/허수정정 장부단차)를 매매손실로 안 친다. current/MDD 는 실제값 그대로
    # 두고, 변동 버킷 계산용 시리즈에서만 누적 정정분을 상쇄한다(첫 포인트 기준 carry-forward).
    _recon0 = 0.0
    if _pts3:
        try:
            _recon0 = float(_pts3[0][2].get("reconcile_adj") or 0.0)
        except (TypeError, ValueError, AttributeError):
            _recon0 = 0.0
    _adj_pts = []
    _rc = 0.0
    for _dt, _v, _p in _pts3:
        try:
            _rc += float(_p.get("reconcile_adj") or 0.0)
        except (TypeError, ValueError, AttributeError):
            pass
        _adj_pts.append((_dt, _v - (_rc - _recon0)))

    kpi: Dict[str, Any] = {"has_equity": False, "has_trades": False}
    if pts:
        cur = pts[-1][1]
        first = pts[0][1]
        peak = pts[0][1]
        mdd = 0.0
        for _, v in pts:
            if v > peak:
                peak = v
            if peak > 0:
                mdd = min(mdd, v / peak - 1.0)
        kpi.update({
            "has_equity": True, "current": cur, "start": first, "points": len(pts),
            "mdd_pct": mdd * 100.0,
        })
        # 사장 지시 2026-06-09: 평가금액(자산곡선) 변동 — 전체/오늘/주/월 (KPI 상단 4칸).
        # 변동액은 정정분 상쇄된 _adj_pts 로 계산(버그 A). pct 분모는 첫 포인트 실제값 유지.
        kpi.update(_equity_change_buckets(_adj_pts, now))

    if trades is None:
        try:
            trades = get_trade_history(limit=2000, uid=uid)
        except Exception:
            trades = []

    # 사장 지시 2026-05-27: 수익률(누적/오늘/주/월)은 잔고 스냅샷이 아닌 '체결 실현손익(비용반영)'
    # 합산 기반 — KIS 결제 글리치로 자산곡선이 튀어도 수익률이 흔들리지 않는다. (미실현 평가손익은
    # 보유 종목 목록에 별도 표시.) 거래비용: 국내 0%, 해외 매수·매도 각 0.3%.
    kpi.update({k: v for k, v in _realized_perf_buckets(trades, now).items() if k != "_realized_has"})
    live = _trade_realized_stats(trades)
    # 사장 지시 2026-05-24: '거래 내역 비우기' 후에도 승률·통계가 유지되도록, 비우기 때 적립한
    # 누적 베이스라인을 라이브 거래 통계에 더한다(평가금액 추이는 equity_curve 가 별도 보존).
    base = _read_perf_baseline()
    total = live["sell_count"] + int(base.get("sell_count", 0) or 0)
    wins = live["win_count"] + int(base.get("win_count", 0) or 0)
    hold_sum = live["hold_days_sum"] + float(base.get("hold_days_sum", 0.0) or 0.0)
    hold_n = live["hold_days_n"] + int(base.get("hold_days_n", 0) or 0)
    if total > 0:
        kpi.update({
            "has_trades": True, "sell_count": total, "win_count": wins,
            "win_rate_pct": wins / total * 100.0,
            "avg_hold_days": (hold_sum / hold_n) if hold_n else None,
        })
    return kpi

def _enrich_trade_history(events: list) -> list:
    """사장 피드백 2026-05-15 (5차): 거래 내역 상세보기용 enrich.
    각 trade event에 다음 추가:
    - est_price (추정 체결가 — cycle_store orders_planned reason의 ≈X원/$X 패턴 파싱)
    - est_currency ('KRW'/'USD')
    - cycle_id (소속 사이클 id)
    - pnl_pct_hint (매도 reason에 박힌 평가손익 %)
    - detail.matched (sell 시 FIFO 매칭된 매수 이력 + 실현 손익)
    체결가는 시장가라 reason text의 ≈X원이 1차 근사치. KIS 체결내역 API로 정확 가능."""
    if not events:
        return []
    try:
        from infra import cycle_store
        cycles = cycle_store.list_cycles(limit=300)
    except Exception:
        cycles = []
    # (ticker, side) → 사이클 후보 리스트
    order_lookup: Dict[tuple, list] = {}
    for c in cycles:
        op = c.get("orders_planned") or []
        if isinstance(op, str):
            try: op = json.loads(op)
            except Exception: op = []
        for o in (op or []):
            tk = str(o.get("ticker") or "").strip()
            sd = str(o.get("side") or "buy").strip().lower()
            reason = str(o.get("reason") or "")
            if not tk: continue
            est_kr = None; est_usd = None
            mr = re.search(r"≈\s*([\d,]+)\s*원", reason)
            if mr:
                try: est_kr = float(mr.group(1).replace(",", ""))
                except ValueError: pass
            mu = re.search(r"≈\s*\$\s*([\d.]+)", reason)
            if mu:
                try: est_usd = float(mu.group(1))
                except ValueError: pass
            pnl_pct_hint = None
            mp = re.search(r"평가손익\s*([+\-]?[\d.]+)\s*%", reason)
            if mp:
                try: pnl_pct_hint = float(mp.group(1))
                except ValueError: pass
            order_lookup.setdefault((tk, sd), []).append({
                "cycle_ts": c.get("started_at") or "", "ended_at": c.get("ended_at") or "",
                "cycle_id": c.get("id"), "reason": reason[:300],
                "est_kr": est_kr, "est_usd": est_usd, "pnl_pct_hint": pnl_pct_hint,
            })
    # ts 오름차순으로 FIFO 처리
    sorted_evs = sorted(events, key=lambda e: e.get("ts", ""))
    positions: Dict[str, list] = {}  # ticker → [{qty, price, ts, source}, ...]
    for e in sorted_evs:
        tk = (e.get("ticker") or "").strip()
        side = str(e.get("side") or "buy").strip().lower()
        try: qty = int(e.get("qty") or 0)
        except (TypeError, ValueError): qty = 0
        filled = str(e.get("filled", "")).lower() in ("true", "1", "yes")
        ts = e.get("ts", "") or ""
        is_us = not (_is_kr_code(tk))
        # 사장 피드백 2026-05-15 (6차): 실제 체결가 우선, 없으면 reason 텍스트 추정으로 폴백.
        actual_fill = e.get("fill_price")
        try: actual_fill = float(actual_fill) if actual_fill is not None else None
        except (TypeError, ValueError): actual_fill = None
        # 가장 가까운 (ts ≤ event ts) 사이클의 가격 후보 선택 (추정값)
        est_price = None; cycle_id = None; pnl_hint = None
        candidates = order_lookup.get((tk, side), [])
        best = None
        for cand in candidates:
            cts = cand.get("cycle_ts") or ""
            if cts <= ts and (best is None or cts > (best.get("cycle_ts") or "")):
                best = cand
        if best:
            est_price = best.get("est_usd") if is_us else best.get("est_kr")
            cycle_id = best.get("cycle_id"); pnl_hint = best.get("pnl_pct_hint")
        # effective_price: 실제 체결가가 있으면 그것, 없으면 추정
        effective_price = actual_fill if actual_fill is not None else est_price
        price_source = "actual" if actual_fill is not None else ("estimated" if est_price else "unknown")
        e["fill_price"] = actual_fill  # 실제 체결가 (있을 때)
        e["est_price"] = est_price     # 추정 체결가 (reason 텍스트)
        e["est_currency"] = "USD" if is_us else "KRW"
        e["cycle_id"] = cycle_id
        e["pnl_pct_hint"] = pnl_hint
        e["price_source"] = price_source
        if not filled:
            e["detail"] = {"unfilled": True, "note": "체결 미확인 — 호가 미체결 가능 (KIS 자동 취소)"}
            continue
        if side == "buy":
            if effective_price and qty > 0:
                positions.setdefault(tk, []).append({"qty": qty, "price": effective_price,
                                                       "ts": ts, "source": price_source})
            e["detail"] = {"buy_price": effective_price, "qty": qty, "currency": e["est_currency"],
                          "price_source": price_source, "avg_cost": e.get("avg_cost")}
        elif side == "sell":
            queue = positions.get(tk, [])
            remaining = qty
            matches = []
            # 사장 피드백 6차: 매도가 우선순위 — 실제 fill_price → est_price → buy_price × (1 + hint%) → buy_price
            while remaining > 0 and queue:
                head = queue[0]
                take = min(head["qty"], remaining)
                effective_sell = actual_fill if actual_fill is not None else est_price
                sell_inferred = (actual_fill is None) and (est_price is None)
                if effective_sell is None and pnl_hint is not None and head["price"]:
                    effective_sell = head["price"] * (1 + pnl_hint / 100.0)
                if effective_sell is None:
                    effective_sell = head["price"]
                pnl = 0.0; pnl_pct = 0.0
                if effective_sell and head["price"]:
                    # 사장 지시 2026-05-27: 거래비용 반영 실현손익 — 국내 0%, 해외 매수·매도 각 0.3%.
                    pnl = _net_realized_pnl(head["price"], effective_sell, take, is_us) or 0.0
                    pnl_pct = (pnl / (head["price"] * take) * 100) if head["price"] and take else 0.0
                matches.append({"buy_qty": take, "buy_price": head["price"], "buy_ts": head["ts"],
                                "buy_source": head.get("source", "unknown"),
                                "sell_price": effective_sell, "pnl": pnl, "pnl_pct": pnl_pct,
                                "sell_price_inferred": sell_inferred})
                head["qty"] -= take
                if head["qty"] <= 0: queue.pop(0)
                remaining -= take
            total_pnl = sum(m["pnl"] for m in matches)
            shown_sell = (actual_fill if actual_fill is not None
                          else (est_price if est_price else (matches[0]["sell_price"] if matches else None)))
            # 사장 지시 2026-05-19: 매도 거래에 박힌 KIS 실매입평균가(avg_cost)가 있으면
            # 그것을 실현손익의 권위 기준으로 삼는다 — 세전(총손익).
            # FIFO 재구성은 로그가 잘리면 어긋나 명목/실제 불일치의 근본 원인이었다.
            # avg_cost가 없는 과거 이벤트만 FIFO 폴백(cost_source로 구분).
            rec_avg = e.get("avg_cost")
            try:
                rec_avg = float(rec_avg) if rec_avg not in (None, "", 0, "0", 0.0) else None
            except (TypeError, ValueError):
                rec_avg = None
            eff_sell_auth = (actual_fill if actual_fill is not None
                             else (est_price if est_price else shown_sell))
            realized_pnl = realized_pct = None
            if rec_avg and rec_avg > 0 and eff_sell_auth:
                # 사장 지시 2026-05-27: KIS 실평단(매수가) 기준 실현손익에 거래비용 반영 — 국내 0%, 해외 매수·매도 각 0.3%.
                realized_pnl = _net_realized_pnl(rec_avg, eff_sell_auth, qty, is_us)
                realized_pct = (realized_pnl / (rec_avg * qty) * 100.0) if (realized_pnl is not None and rec_avg and qty) else None
                if realized_pnl is not None:
                    total_pnl = realized_pnl  # 표시 P&L을 권위값(KIS 평단 기준, 비용반영)으로 교체
            # 사장 지시 2026-05-22(R5): KIS 실평단(avg_cost)도 없고 매수 이력도 없으면(거래내역 초기화)
            # 손익을 0/추정으로 '부정확하게' 찍지 말고 '미상'으로 표기한다.
            _no_basis = (not (rec_avg and rec_avg > 0)) and (not matches)
            if _no_basis:
                total_pnl = None
            e["detail"] = {"matched": matches, "sell_price": shown_sell, "qty": qty,
                          "total_pnl": total_pnl, "unmatched_qty": remaining,
                          "currency": e["est_currency"], "price_source": price_source,
                          "cost_basis": rec_avg, "realized_pnl": realized_pnl,
                          "realized_pnl_pct": realized_pct,
                          "cost_source": ("kis_avg" if (rec_avg and rec_avg > 0)
                                          else ("none" if _no_basis else "fifo_reconstructed")),
                          "no_cost_basis": _no_basis,
                          "sell_price_inferred": (actual_fill is None and est_price is None)}
            # 사장 지시 2026-06-16: 인용 평가손익(pnl_pct_hint)이 권위 실현손익과 부호 충돌하면
            # 숨김(None) — 재매수 미결제 중 옛 평단 인용으로 손실을 이익으로 오해하던 것 차단.
            _auth_pct = (realized_pct if realized_pct is not None
                         else ((total_pnl / (matches[0]["buy_price"] * qty) * 100.0)
                               if (total_pnl and matches and matches[0].get("buy_price") and qty) else None))
            if _hint_conflicts_authority(e.get("pnl_pct_hint"), _auth_pct):
                e["pnl_pct_hint"] = None
    return sorted_evs


def get_trade_history(limit: int = 500, uid=None) -> list:
    """All trade events (executed/failed), newest first, for the 거래 내역 list.
    사장 피드백 2026-05-15 (5차): 추정 가격 + FIFO 매칭 P&L detail 첨부.
    Phase 2 멀티테넌트: uid 별 로그만 반환한다."""
    evs = [e for e in _read_response_log(uid) if e.get("type") in ("trade_executed", "trade_failed")]
    enriched = _enrich_trade_history(evs[-max(1, int(limit)):])
    return list(reversed(enriched))

def clear_event_log(uid=None):
    """Wipe that uid's accumulated event/response log (the dashboard '초기화' button)."""
    p = _response_log_path(uid)
    if p is None:
        return
    try:
        p.write_text("[]", encoding="utf-8")
        log_response_event({"source": "system_event", "type": "status", "state": "IDLE",
                            "message": "로그 초기화됨 (사용자 요청)"}, uid=uid)
    except Exception as _e:
        logger.warning(f"이벤트 로그 초기화 실패(uid={uid}): {_e}")

def clear_trade_log(uid=None) -> int:
    """Wipe just the trade events from claude_response.json (keeps everything else) + clear in-memory trade log.
    Used by the 수익률 탭 '거래 내역 비우기' button. Returns the number of trade entries removed.

    사장 지시 2026-05-24: 비우기 해도 ① 승률·매도수·보유일 통계는 영속 베이스라인에 적립해 유지하고,
    ② 제거되는 거래 이벤트는 타임스탬프 백업으로 남긴다('기억 못해서 평가금액이 틀어지는' 문제 방지).
    누적 평가금액(누적수익)은 equity_curve.json 가 별도 보존하므로 비우기와 무관하게 정확하다."""
    removed = 0
    p = _response_log_path(uid)
    if p is None:
        return 0
    # ① 비우기 직전 실현손익 통계를 베이스라인에 적립 (승률이 0으로 초기화되지 않게)
    try:
        st = _trade_realized_stats(get_trade_history(limit=4000, uid=uid))
        if st["sell_count"] > 0:
            base = _read_perf_baseline()
            base["sell_count"] = int(base.get("sell_count", 0) or 0) + st["sell_count"]
            base["win_count"] = int(base.get("win_count", 0) or 0) + st["win_count"]
            base["hold_days_sum"] = float(base.get("hold_days_sum", 0.0) or 0.0) + st["hold_days_sum"]
            base["hold_days_n"] = int(base.get("hold_days_n", 0) or 0) + st["hold_days_n"]
            _write_perf_baseline(base)
    except Exception as _e:
        logger.warning(f"승률 베이스라인 적립 실패(무시): {_e}")
    try:
        data = _read_response_log(uid)
        trade_evs = [e for e in data if e.get("type") in ("trade_executed", "trade_failed")]
        kept = [e for e in data if e.get("type") not in ("trade_executed", "trade_failed")]
        removed = len(data) - len(kept)
        # ② 제거 전 백업 (안전망) — uid 별로 구분 저장.
        if trade_evs:
            try:
                _bdir = _Path(__file__).parent / "data" / "_reset_backup"
                _bdir.mkdir(parents=True, exist_ok=True)
                (_bdir / f"trades.uid{uid}.{datetime.now(KST):%Y%m%d-%H%M%S}.json").write_text(
                    json.dumps(trade_evs, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as _be:
                logger.warning(f"거래 내역 백업 실패(무시): {_be}")
        p.write_text(json.dumps(kept[-_RESPONSE_LOG_CAP:], ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as _e:
        logger.warning(f"거래 내역 초기화 실패(uid={uid}): {_e}")
    # Phase 2 멀티테넌트: 이 uid 의 스왐 인메모리 trade_log 만 비운다(이미 생성된 스왐만 —
    # lazy 프로퍼티 강제생성 안 함). 다른 유저의 인메모리 거래 상태는 건드리지 않는다.
    try:
        from infra.user_context import REGISTRY
        for _ctx in REGISTRY.all_contexts().values():
            if int(getattr(_ctx, "uid", -1)) != int(uid):
                continue
            sw = getattr(_ctx, "_swarm", None)
            if sw is not None:
                sw._trade_log.clear()
                sw._trades_executed = 0
    except Exception:
        pass
    log_response_event({"source": "system_event", "type": "status", "state": "IDLE",
                        "message": f"거래 내역 초기화됨 ({removed}건 제거 · 승률·통계는 유지)"}, uid=uid)
    return removed

_broadcast_callback = None
def set_broadcast_callback(cb):
    global _broadcast_callback; _broadcast_callback = cb
    # 운영자 알림을 대시보드로도 흘린다 (파일 싱크가 진실의 원천 — UI 는 best-effort).
    def _alert_bridge(entry: dict):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_broadcast(entry))
        except RuntimeError:
            pass  # 실행 중인 이벤트 루프 없음 → 파일/로그 싱크로 충분
    try:
        notifier.set_broadcast_callback(_alert_bridge)
    except Exception:
        pass
async def _broadcast(msg, uid=None):
    """이벤트를 영속 로그에 적고 WS 콜백으로 흘린다.
    Phase 2 멀티테넌트: uid 가 주어지면(오케스트레이터 사이클 이벤트) 해당 유저 연결에만,
    uid 가 None 이면(시스템 알림 등) 전체 연결에 송신한다. 라우팅은 app.py 가 등록한
    콜백(_route)이 결정한다."""
    # 사장 지시(재발 2026-07-29): 대시보드는 마크다운을 렌더하지 않아 `**` 가 그대로 보인다.
    # LLM 응답은 base_agent/response_text 에서 이미 지우지만, 코드가 직접 만든 문구에도
    # `**` 가 남아 있었다(예: '굵직한 **40건** 선별'). 브로드캐스트 단일 관문에서 최종 제거.
    try:
        if isinstance(msg, dict):
            _fix = {k: msg[k] for k in ("message", "report")
                    if isinstance(msg.get(k), str) and "**" in msg[k]}
            if _fix:
                from infra.local_llm_client import strip_markdown_emphasis
                msg = {**msg, **{k: strip_markdown_emphasis(v) for k, v in _fix.items()}}
    except Exception:
        pass
    try:
        if isinstance(msg, dict): log_response_event({"source":"system_event", **msg}, uid=uid)
    except Exception: pass
    if _broadcast_callback:
        try: await _broadcast_callback(msg, uid)
        except: pass

_TICKER_SKIP = {"APPROVED","JSON","KOSPI","KOSDAQ","KPI","NYSE","NASD","NASDAQ","AMEX","ETF","RSI","MACD",
                "EPS","PER","PBR","PSR","ROE","ROA","ROIC","ESG","GDP","CPI","PPI","PCE","USD","KRW","JPY",
                "EUR","CEO","CFO","COO","AI","ML","HBM","DRAM","NAND","HQ","IPO","M&A","FOMC","ETN","REIT",
                "DJI","SPX","IXIC","SHS","NKY","WTI","SMA","EMA","BB","ADX","OBV"}

def _clean_codes(kr_raw, us_raw) -> List[str]:
    kr, seen = [], set()
    for c in kr_raw:
        c = c.strip()
        if re.fullmatch(r"\d{6}", c) and c not in seen:
            seen.add(c); kr.append(c)
    us = []
    for t in us_raw:
        t = t.strip().upper()
        if re.fullmatch(r"[A-Z]{1,5}", t) and t not in _TICKER_SKIP and t not in seen:
            seen.add(t); us.append(t)
    return kr[:10] + us[:8]

def _extract_stock_codes(text: str) -> List[str]:
    """Prefer the explicit '대상종목: ...' / '종목코드: ...' line the orchestrator is required to emit;
    fall back to scanning the whole text for 6-digit codes / US tickers (filtered)."""
    text = text or ""
    code6 = r"(?<!\d)\d{6}(?!\d)"
    m = re.search(r"(?:대상\s*종목|종목\s*코드|target\s*stocks?)\s*[:：]\s*(.+)", text, re.IGNORECASE)
    if m:
        seg = m.group(1).splitlines()[0]
        codes = _clean_codes(re.findall(code6, seg), re.findall(r"[A-Za-z]{1,5}", seg))
        if codes:
            return codes
    # fallback: whole-text scan (handles Korean particles glued to the code, e.g. "005930을")
    return _clean_codes(re.findall(code6, text), re.findall(r"\b[A-Z]{1,5}\b", text))

def _extract_codes_after(text: str, *labels: str) -> List[str]:
    """Codes on the line that starts with one of `labels` (e.g. '후보종목:' / '최종종목:' / '최종승인:').
    Returns [] when the label line isn't present (caller can distinguish 'no line' from 'empty list')."""
    text = text or ""
    code6 = r"(?<!\d)\d{6}(?!\d)"
    pat = r"(?:" + "|".join(re.escape(l) for l in labels) + r")\s*[:：]\s*(.+)"
    m = re.search(pat, text, re.IGNORECASE)
    if not m:
        return []
    seg = m.group(1).splitlines()[0]
    return _clean_codes(re.findall(code6, seg), re.findall(r"[A-Za-z]{1,5}", seg))

def _has_label(text: str, *labels: str) -> bool:
    return bool(re.search(r"(?:" + "|".join(re.escape(l) for l in labels) + r")\s*[:：]", text or "", re.IGNORECASE))


def _parse_macro_stock_pct(text: Optional[str]) -> Optional[float]:
    """글로벌리서치팀장 매크로 보고에서 권고 '주식 X%' 비중을 분수(0.01)로 추출한다.
    asset_sleeves.parse_macro_sleeve_pct 에 위임(DRY) — 못 찾으면 None(호출부 fail-open)."""
    return parse_macro_sleeve_pct(text, "주식")


def _settle_fill_qty(side, before_qty, after_qty, order_qty):
    """체결 정산 — (실제 체결 증분, 잔여 수량). 원장/로그엔 주문수량이 아닌 '잔고 증분'을 기록한다
    (2026-06-15: 부분체결이 전량체결로 둔갑하던 버그 수정). 증분이 주문량을 넘으면 주문량으로 캡한다."""
    b = int(before_qty or 0); a = int(after_qty or 0); o = int(order_qty or 0)
    filled = (a - b) if side == "buy" else (b - a)
    filled = max(0, min(filled, o))
    return filled, max(0, o - filled)


def _poll_increment(side, base_qty, after_qty, target, recorded):
    """부분체결 잔여 폴링의 1회 증분 정산. 이동 base 대비 새로 늘어난 잔고만 기록하고(중복 방지)
    목표(target) 초과분은 캡한다. 반환 (record_qty, new_base, new_recorded, done)."""
    b = int(base_qty or 0); a = int(after_qty or 0)
    t = int(target or 0); rec = int(recorded or 0)
    delta = (a - b) if side == "buy" else (b - a)
    delta = max(0, min(delta, t - rec))
    new_rec = rec + delta
    new_base = a if delta > 0 else b
    return delta, new_base, new_rec, (new_rec >= t)


def _thesis_prune_msg(scope: str, removed) -> str:
    """보유 0이 된 코드의 thesis 정리 메시지. 보유 0은 '이번 사이클 전량 매도'일 수도, 애초
    '미보유(미체결·과거 청산)'일 수도 있어 매도를 단정하지 않는다(2026-06-15: 보유 0인 DBC 를
    '전량 매도'로 오기하던 confab 수정)."""
    return f"[{scope}] 보유 없음 — thesis 정리: {removed}"


MACRO_DEPLOY_FLOOR_GAP = 0.05   # 매크로 목표가 현재 주식비중을 이만큼(5%p) 넘어야 플로어 발동


def apply_macro_deploy_floor(per_order_ratio: float, cycle_ratio: float, *,
                             macro_target_pct, current_stock_ratio, params: dict):
    """매크로 주식 목표 > 현재 주식비중(여력 있음)이면 per-order·per-cycle 예산 비율에 최소
    플로어를 적용해 ops 예산 컷이 목표 수렴을 과도하게 늦추지 못하게 한다. 여력이 없거나(목표≤현재)
    토글 OFF 면 원값 유지(방어 의도 보존). (2026-06-15 사장 선택: '목표 향한 예산 플로어'.)"""
    if not params.get("MACRO_DEPLOY_FLOOR_ENABLED", True):
        return per_order_ratio, cycle_ratio
    if macro_target_pct is None or current_stock_ratio is None:
        return per_order_ratio, cycle_ratio
    if (float(macro_target_pct) - float(current_stock_ratio)) < MACRO_DEPLOY_FLOOR_GAP:
        return per_order_ratio, cycle_ratio
    po_floor = float(params.get("PER_ORDER_BUDGET_FLOOR_RATIO", 0.10) or 0.0)
    cyc_floor = float(params.get("MAX_CYCLE_BUDGET_FLOOR_RATIO", 0.30) or 0.0)
    return max(per_order_ratio, po_floor), max(cycle_ratio, cyc_floor)


def seed_pending_news(articles, now_iso, window_min: int = 90):
    """재시작 직후 빈 대기(pending) 뉴스풀을 최근 history 로 시드한다(사장 지시 2026-06-04).

    재시작 시 인메모리 대기풀은 비고, crawl_once 는 영속 seen_links 와 대조해 '이미 본' 최근 기사를
    다시 안 담는다 → 첫 사이클이 '뉴스 0' sell-only 로 눈먼다(history엔 최신 뉴스가 있는데도).
    crawled_at 이 now 기준 window_min 분 이내인 기사만 돌려준다. crawled_at 누락/파싱불가는 제외.
    2026-06-04 뉴스 풀 단일화: KR/US 시장 분기 폐지 — 단일 리스트 반환(market 필드 미사용)."""
    out = []
    try:
        now = datetime.strptime(str(now_iso)[:19], "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return out
    for a in (articles or []):
        ts = a.get("crawled_at")
        if not ts:
            continue
        try:
            t = datetime.strptime(str(ts)[:19], "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            continue
        mins = (now - t).total_seconds() / 60.0
        if mins < 0 or mins > window_min:
            continue
        out.append(a)
    return out


def pick_cycle_news(pending, recent_fallback, fallback_n: int = 20):
    """사이클에 쓸 뉴스 선택(사장 지시 2026-06-04 뉴스 풀 단일화).
    풀에 뉴스가 있으면 그대로 쓰고, 비어 있으면 최신 fallback_n개(history)로 폴백 — 뉴스 없이
    헛도는 사이클 방지. Returns (news_list, used_fallback: bool)."""
    if pending:
        return list(pending), False
    return list(recent_fallback or [])[:fallback_n], True


def _filter_news_by_freshness(articles, decay_hours):
    """뉴스 신선도 감쇠(사장 지시 2026-07-21) — NEWS_DECAY_HOURS 초과된 오래된 뉴스 제외.
    decay_hours<=0 → 무필터. article['date'] 'YYYY-MM-DD HH:MM'. 파싱 실패 기사는 보수적으로 유지."""
    try:
        dh = float(decay_hours or 0.0)
    except (TypeError, ValueError):
        dh = 0.0
    if dh <= 0 or not articles:
        return articles
    from datetime import datetime as _dt, timedelta as _td
    now = _dt.now(KST).replace(tzinfo=None)
    cutoff = now - _td(hours=dh)
    kept = []
    for a in articles:
        ds = str((a or {}).get("date") or "").strip()
        if not ds:
            kept.append(a); continue
        parsed = None
        for _fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                parsed = _dt.strptime(ds[:19], _fmt); break
            except ValueError:
                continue
        if parsed is None or parsed >= cutoff:
            kept.append(a)
    return kept


def parse_news_sentiment(news_report: Optional[str], code: str, name: Optional[str] = None) -> Optional[float]:
    """뉴스 리포트에서 해당 후보(코드 또는 종목명)의 감성 점수(-1..1)를 파싱(사장 지시 2026-06-04).
    형식 예: '📰 에치에프알(230240) ...: 감성 +0.80'. 못 찾으면 None(뉴스 차원 제외 폴백)."""
    if not news_report:
        return None
    text = str(news_report)
    for anchor in (code, name):
        if not anchor:
            continue
        idx = text.find(str(anchor))
        if idx >= 0:
            m = re.search(r"감성\s*([+-]?\d+(?:\.\d+)?)", text[idx:idx + 200])
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    return None
    return None


def assemble_quant_score(ind: Dict[str, Any], sentiment: Optional[float], macro_pct: Optional[float],
                          qiw: Dict[str, float], dw: Dict[str, float]):
    """결정론 최종 퀀트점수 조립(사장 지시 2026-06-04). 지표 부족(빈 ind) → 0(매수 제외, 레거시 패리티).
    Returns (int 0~10, breakdown)."""
    from tools.quant_score import compute_indicator_score, news_score, macro_score, combine_dimensions
    if not ind:
        return 0, {"S_quant": None, "S_news": None, "S_macro": None, "reason": "지표 데이터 부족"}
    s_quant, ind_bd = compute_indicator_score(ind, qiw)
    s_news = news_score(sentiment)
    s_macro = macro_score(macro_pct)
    final = combine_dimensions({"QUANT": s_quant, "NEWS": s_news, "MACRO": s_macro}, dw)
    bd = {"S_quant": round(s_quant, 1),
          "S_news": (round(s_news, 1) if s_news is not None else None),
          "S_macro": (round(s_macro, 1) if s_macro is not None else None),
          "indicators": {k: round(v, 2) for k, v in ind_bd.items()}}
    return int(round(final)), bd


def filter_targets_by_score(target_codes, quant_scores: Dict[str, int], min_score: int, max_names: int = 0):
    """Reject unscored names, then rank scored candidates and enforce the name cap."""
    ms = int(min_score or 0)
    survivors, dropped = [], []
    for c in (target_codes or []):
        c = str(c).strip()
        if not c:
            continue
        if c not in (quant_scores or {}):
            dropped.append(c)
        elif ms <= 0:
            survivors.append(c)
        elif int(quant_scores.get(c) or 0) >= ms:
            survivors.append(c)
        else:
            dropped.append(c)                         # 점수 미달 제거
    def _key(c):
        return quant_scores[c]
    survivors.sort(key=lambda c: -float(_key(c)))
    mn = int(max_names or 0)
    if mn > 0 and len(survivors) > mn:
        dropped.extend(survivors[mn:])
        survivors = survivors[:mn]
    return survivors, dropped


def format_scoring_rubric_block(qiw: Dict[str, float], dw: Dict[str, float], min_score: int) -> str:
    """주식운용실장 PASS1 선정 프롬프트에 주입할 채점 루브릭 요약(사장 지시 2026-06-04 ①).
    어떤 지표가 가점/감점되는지(상위 가중 3개)와 최소 퀀트점수 게이트를 알려, LLM이 루브릭 정렬된
    후보를 제안하게 한다. 점수는 시스템(파이썬)이 확정하므로 LLM은 '선정 기준'으로만 참고."""
    _names = {"rsi": "RSI(과매도 가점)", "macd": "MACD 모멘텀", "adx": "ADX 추세강도",
              "vwap": "VWAP 이격(추격 감점)", "vol": "저변동", "mom": "모멘텀(1·3M)",
              "cmf": "CMF 매집", "flow": "외인·기관 수급", "high52": "52주 신고가 근접",
              "leadlag": "선행-후행(그랜저)", "vol_surge": "거래량 급증"}
    pos = sorted(((k, float(v)) for k, v in (qiw or {}).items() if float(v) > 0), key=lambda kv: -kv[1])[:3]
    drivers = ", ".join(_names.get(k, k) for k, _ in pos) or "(가중치 미설정)"
    lines = ["[채점 루브릭 — 주식운용실장 선정 참고]",
             f"시스템 퀀트점수(0~10)는 다음을 가장 크게 반영: {drivers}.",
             f"최종 매수는 퀀트점수 {int(min_score or 0)}점 이상만 통과하니, 이 기준에 부합할 종목을 우선 고르십시오.",
             "점수는 시스템이 확정합니다(이 블록은 선정 기준 참고용)."]
    return "\n".join(lines)


def format_strategy_param_block(params: Dict[str, Any]) -> str:
    """계량분석팀장 호출 프롬프트에 주입할 '매수 필터' advisory 블록(사장 지시 2026-06-04).
    2026-06-04 결정론 점수 엔진 도입으로 채점 가중치는 파이썬이 처리(QIW_*)하므로 제거 — 여기선 '활성
    (비-0/True) 필터'만 advisory로 표기(주식운용실장 선정·해설 참고용). 비활성(0/off)은 생략."""
    lines = ["[전략 파라미터 — 운용지원실장 지정, 선정·해설 참고]"]
    filt = []
    _mv = float(params.get("MAX_BUY_VOLATILITY_PCT") or 0)
    if _mv > 0:
        filt.append(f"연환산 변동성 {_mv:.0f}% 초과 종목은 매수부적합(점수 대폭 하향).")
    _rsi = int(params.get("RSI_OVERBOUGHT_SKIP") or 0)
    if _rsi > 0:
        filt.append(f"RSI(14) {_rsi} 초과(과매수) 종목은 신규매수 회피(점수 하향).")
    _adx = int(params.get("MIN_ADX_FOR_BUY") or 0)
    if _adx > 0:
        filt.append(f"ADX(14) {_adx} 미만(추세 약함) 종목은 매수부적합(추세추종 모드).")
    if params.get("REQUIRE_FOREIGN_NET_BUY"):
        filt.append("외국인 순매수(+)가 아닌 종목은 매수부적합(수급 우선).")
    _ext = float(params.get("MAX_PRICE_EXTENSION_PCT") or 0)
    if _ext > 0:
        filt.append(f"현재가가 VWAP/이평 대비 +{_ext:.0f}% 초과 이격이면 추격매수 회피(점수 하향).")
    _mq = int(params.get("MIN_QUANT_SCORE") or 0)
    if _mq > 0:
        filt.append(f"최종 매수는 퀀트점수 {_mq}점 이상만 — 미만은 시스템이 제거함.")
    if filt:
        lines.append("매수 필터: " + " ".join(filt))
    return "\n".join(lines)


def _macro_blocks_new_buys(macro_report: Optional[str], equity_weight: float):
    """사장 지시 2026-06-04: 매크로 권고 주식비중 ≤ 현재 주식 평가비중이면 '추가 매수 여력 없음'
    으로 보고 신규 매수 평가를 건너뛴다. Returns (blocked: bool, stock_pct or None).
    주식% 파싱 실패 시 (False, None) — fail-open(잘못된 차단으로 거래가 전면중단되는 것 방지)."""
    pct = _parse_macro_stock_pct(macro_report)
    if pct is None:
        return False, None
    return (pct <= float(equity_weight or 0.0) + 1e-9), pct


def buy_limit_below_market(side, entry_mode, entry_limit, market_price) -> bool:
    """매수 진입 지정가가 현재 시장가보다 낮은가(=미체결 위험)? (사장 지시 2026-06-12)

    LLM(주식운용실장/계량분석팀장)이 시장가보다 낮은 매수 진입가를 지정하면 KIS limit 매수가
    접수만 되고 체결되지 않아(매수호가<매도호가) 매수가 조용히 증발한다('주문 절대 스킵 금지'
    위반, uid1 cyc312 RKLB 진입가 105 vs 시장가 112.83 → accepted/filled=False). True 면
    호출부가 시장가 매수로 전환한다. 매도·시장가·시장가 미상(0/None)은 손대지 않는다(보수)."""
    if side != "buy" or entry_mode != "limit":
        return False
    try:
        el = float(entry_limit or 0.0)
        mp = float(market_price or 0.0)
    except (TypeError, ValueError):
        return False
    return el > 0 and mp > 0 and el < mp


# 긴급청산 판별용 키워드 — reason 문자열에 하나라도 있으면 '반드시 빠져나가야 하는' 매도.
# '익절'(자동 익절·목표 도달)·'절반'·'N주'는 의도된 지정가 목표라 제외(시장가 위 지정가 유지).
_URGENT_LIQ_WORDS = ("전량", "청산", "손절", "트레일링", "편중")


def _is_urgent_liquidation(reason) -> bool:
    """매도 사유가 긴급청산(전량/손절/트레일링 스톱/편중축소)인가? (사장 정책 2026-06-22)

    긴급청산은 '반드시 지금 빠져나가야' 하는 매도라, 지정가가 시장가 위라 미체결되면 위험이
    방치된다 → sell_limit_above_market 과 결합해 시장가로 전환한다. 의도된 익절 목표가
    (자동 익절·절반·N주)는 제외 — 시장가 위에서 더 받으려 대기하는 정상 지정가다."""
    r = str(reason or "")
    return any(w in r for w in _URGENT_LIQ_WORDS)


def sell_limit_above_market(side, entry_mode, entry_limit, market_price) -> bool:
    """매도 진입 지정가가 현재 시장가보다 위인가(=미체결·물량잠김 위험)? (사장 정책 2026-06-22)

    buy_limit_below_market 의 매도 대칭. 사후관리실장이 '전량 손절'을 시장가 위 지정가로 제시하면
    KIS limit 매도가 접수만 되고 체결 안 돼(매도호가>매수호가) 보유물량이 잠긴다(uid2 316140
    32,500 지정가 vs 30,000 시장가 → sellable 0 누적). True 면 호출부가 '긴급청산 한정' 시장가로
    전환한다. 매수·시장가·시세 미상(0/None)은 손대지 않는다(보수). US 경로는 marketable_us_limit
    이 가격 자체를 체결가능으로 클램프해 이미 보호되지만, KR 도메스틱엔 그 대칭이 없었다."""
    if side != "sell" or entry_mode != "limit":
        return False
    try:
        el = float(entry_limit or 0.0)
        mp = float(market_price or 0.0)
    except (TypeError, ValueError):
        return False
    return el > 0 and mp > 0 and el > mp


def _names_match(a: str, b: str) -> bool:
    """종목명 a(주어진 이름)와 b(코드의 실제 이름)가 사실상 같은지(공백 제거 후 포함 관계).
    한쪽이라도 비면 검증 불가로 보고 통과(보수). '삼성전자' vs '삼성전자보통주'류 허용."""
    import re as _re
    a2 = _re.sub(r"\s+", "", str(a or "")); b2 = _re.sub(r"\s+", "", str(b or ""))
    if not a2 or not b2:
        return True
    return a2 in b2 or b2 in a2


def _correct_display_codes(text: str, *, resolver=None, name_check=None, limit: int = 12) -> str:
    """뉴스·후보 표시 텍스트의 '종목명(6자리코드)' 토큰에서 코드가 종목명과 불일치하면
    이름→코드 재조회로 교정한다(사장 지시 2026-07-20 — 화면에 환각 코드 노출 방지).
    resolver(name)->code, name_check(code)->name 미주입 시 원문 그대로(best-effort).
    limit: 교정 시도(불일치 시 resolver 호출)를 이 횟수로 제한해 네트워크 폭주 방지."""
    if not text or resolver is None or name_check is None:
        return text
    _tries = [0]
    def _fix(m):
        nm, code, mkt = m.group(1).strip(), m.group(2), (m.group(3) or "")
        if _tries[0] >= limit:
            return m.group(0)
        try:
            actual = name_check(code)
            if actual and _names_match(nm, actual):
                return m.group(0)   # 코드 정확 — 그대로
            _tries[0] += 1
            corrected = resolver(nm) or ""
            if re.fullmatch(r"\d{6}", corrected or "") and corrected != code:
                logger.info(f"[표시코드 교정] {nm}({code}→{corrected})")
                return f"{nm}({corrected}{mkt})"   # 시장 태그(, KR/US)는 보존
        except Exception:
            pass
        return m.group(0)
    try:
        # '이름(코드)' 및 '이름(코드, KR/US)' 두 표기 모두 매칭(시장 태그는 group3 로 보존).
        return re.sub(
            r"([가-힣A-Za-z][가-힣A-Za-z0-9]{1,15})\s*[\(（]\s*(\d{6})\s*(,\s*[A-Za-z]{2})?\s*[\)）]",
            _fix, text)
    except Exception:
        return text


def _resolve_candidate_codes(allocation: str, *, session: Optional[str] = None,
                             resolver=None, name_check=None, limit: int = 5) -> List[str]:
    """'후보종목:' 라인을 파싱해 **실제 종목코드**로 해석한다 (사장 지시 2026-05-22).

    주식운용실장(LLM)은 종목명은 알아도 6자리 코드를 몰라 환각(123456 등)한다. 그래서
    `종목명(코드)`/`종목명`/`코드`/US티커가 섞여 들어온다. 규칙:
      - 6자리 코드가 유효(name_check 로 이름 조회 성공)하면 그대로 사용.
      - 코드가 무효/누락이면 **종목명을 resolver(이름→코드)로 검색**해 보정. 이름도 없으면 버린다.
      - 영문 토큰은 US 세션(또는 세션 불명)에서만 미국 티커로 본다. KR 세션의 'HPSP' 같은
        영문 종목명은 코드로 해석한다.
    resolver/name_check 미주입(테스트·오프라인) 시 보정 없이 코드만 신뢰한다."""
    m = re.search(r"(?:후보종목|대상종목|종목코드|target stocks)\s*[:：]\s*(.+)", allocation or "", re.IGNORECASE)
    if not m:
        return []
    seg = m.group(1).splitlines()[0]
    is_kr = is_kr_session(session)
    out: List[str] = []
    for tok in re.split(r"[,，;]", seg):
        tok = tok.strip().strip(".·").strip()
        if not tok:
            continue
        mm = re.match(r"(?P<name>.*?)\s*[\(（]\s*(?P<code>[A-Za-z0-9.]{1,6})\s*[\)）]\s*$", tok)
        name = mm.group("name").strip() if mm else ""
        inner = (mm.group("code").strip() if mm else tok)
        code = inner if re.fullmatch(r"\d{6}", inner) else ""
        # 영문 토큰 → US 티커 (단, KR 세션이면 영문 종목명일 수 있으니 코드 해석으로 넘김)
        if not code and re.fullmatch(r"[A-Za-z]{1,5}", inner) and not is_kr:
            t = inner.upper()
            # KR 과 대칭으로 이름↔티커 검증 (2026-08-04). 'Rocket Lab(RVLV)' 처럼 **실존하는**
            # 티커로 오배정되면 하류 일봉 게이트에 안 걸려 엉뚱한 회사를 매수한다.
            # 한글 이름은 영문 회사명과 대조 불가라 건너뛰고, 조회 실패('')도 통과 — fail-open.
            if name and name_check and not re.search(r"[가-힣]", name):
                actual = name_check(t)
                if actual and not _names_match(name, actual):
                    logger.warning(f"[후보검증] '{name}' ↔ 티커 {t}(실제 {actual}) 불일치 — 후보 제외")
                    continue
            if t not in out:
                out.append(t)
            if len(out) >= limit:
                break
            continue
        nm = name or ("" if code else tok)
        resolved = ""
        if code:
            if name_check is None:
                resolved = code  # 검증 불가 환경(테스트/오프라인) → 코드 신뢰
            else:
                actual = name_check(code)
                if actual:
                    # 코드 유효 — 단, 이름이 함께 주어졌으면 코드의 실제 종목명과 일치하는지 검증.
                    # 불일치면 LLM 코드 환각으로 보고 이름으로 재해석(못 찾으면 후보 제외) — 2026-06-15
                    # '벡트(290650→엘앤씨바이오)' 오매수 방지.
                    if nm and not _names_match(nm, actual):
                        resolved = (resolver(nm) if resolver else "") or ""
                        if not resolved:
                            logger.warning(f"[후보검증] '{nm}' ↔ 코드 {code}(실제 {actual}) 불일치 — 후보 제외")
                    else:
                        resolved = code
        if not resolved and nm and resolver:
            resolved = resolver(nm) or ""
        if resolved and resolved not in out:
            out.append(resolved)
        if len(out) >= limit:
            break
    return out


# C (사장 지시 2026-05-28): 후보 해석이 0건인데 뉴스에 종목 신호가 있을 때, 뉴스분석 리포트의
# 괄호표기 종목(`종목명(MU)`/`종목명(005930)`)으로 후보를 보강한다 — '뉴스만 있고 후보 0'으로
# 사이클이 낭비되는 것 방지(로그 리뷰: uid2 cycle24 MU/NVDA 신호 무시). 보강 후보도 downstream
# 퀀트≥6·DART·리스크 게이트를 그대로 통과해야 매수되므로 과매수 위험은 낮다.
_NEWS_TICKER_STOPWORDS = {
    "AI", "ETF", "ETN", "CEO", "CFO", "USD", "KRW", "GDP", "FED", "CPI", "PPI",
    "IPO", "EPS", "PER", "PBR", "ROE", "AND", "OR", "US", "UK", "EU", "OK",
    "ESG", "API", "OPEC", "WTI", "FOMC", "ECB", "BOJ", "GPU", "CPU", "HBM",
}

def seed_candidates_from_news(news_report: str, session: Optional[str] = None,
                              limit: int = 5) -> List[str]:
    """뉴스분석 리포트의 괄호표기 종목에서 후보를 추출한다(세션 경계·중복·상한 적용).
    US 세션=영문 티커, KR 세션=6자리 코드. 흔한 비종목 약어(AI/ETF/FED 등)는 제외한다."""
    text = news_report or ""
    is_kr = is_kr_session(session)
    out: List[str] = []
    for m in re.finditer(r"[\(（]\s*([A-Za-z]{1,5}|\d{6})\s*[\)）]", text):
        tok = m.group(1)
        if re.fullmatch(r"\d{6}", tok):
            if is_kr and tok not in out:
                out.append(tok)
        else:
            t = tok.upper()
            if (not is_kr) and t not in _NEWS_TICKER_STOPWORDS and t not in out:
                out.append(t)
        if len(out) >= limit:
            break
    return out


def cycle_is_idle(sell_only: bool, holdings) -> bool:
    """D (사장 지시 2026-05-28): 뉴스0·미개장(sell_only)이고 보유 종목도 없으면 매수(뉴스게이트)·
    매도(보유없음) 둘 다 불가 → 매크로 리서치/LLM 호출을 생략해 비용을 아낀다. 보유가 있으면
    매도 평가가 필요하므로 절대 idle 로 보지 않는다(손절/익절 누락 방지)."""
    return bool(sell_only) and not (holdings or [])


def _affordable_one_share(price: float, cash: float, total: float,
                          *, cycle_remaining: Optional[float] = None) -> bool:
    """사장 결정 2026-05-16: '1주 예산'(총평가의 10%) 비율과 무관하게,
    1주 가격이 **가용 예수금** 이내면 최소 1주 매수를 허용한다.
    (소액 계좌에서 $85짜리 정상 종목이 비율 한도 때문에 무조건 제외되던 문제 해결)

    price/cash/total/cycle_remaining 은 **모두 같은 통화**로 전달된다 (KR=원, US=USD 환산).

    Returns True ⇒ '1주는 살 수 있다'고 보고 주문 초안에 포함.

    예수금 게이트(보수형 B): 슬리피지·수수료 여유분(runtime 'MIN_CASH_BUFFER', 보통 1.10)을
    곱한 `price * buffer <= cash` 일 때만 통과 — 리스크관리실장 예수금 게이트와 동일 기준이라
    주문 초안이 곧바로 리스크 단에서 반려되는 모순을 막는다.

    사장 결정 2026-06-16: 1주가 예수금 이내라도 **사이클 매수예산(잔여)을 초과하면 매수하지
    않는다**. `cycle_remaining`(= cash×MAX_CYCLE_BUDGET_RATIO − 이번 사이클 기집행액)을 주면
    그 안일 때만 1주를 허용한다. 이전엔 사이징이 예수금만 보고 1주를 통과시켜, 고가주(AMD $551
    등)가 리스크 가드레일의 MAX_CYCLE_BUDGET_RATIO 초과로 반려되어 — 매 사이클 골랐다 반려되는
    데드존 + final_report '예수금 이내 → 매수 가능' 모순 메시지를 만들었다. 사이징과 가드레일을
    같은 기준으로 일치시켜 데드존·모순을 제거한다. (cycle_remaining 미전달 시 예산 체크 생략 —
    하위호환; 호출부가 사이클 잔여예산을 알 때만 적용.)
    """
    import runtime as _rt
    if price <= 0 or cash <= 0:
        return False
    _buffer = float(_rt.get("MIN_CASH_BUFFER") or 1.10)
    if price * _buffer > cash:
        return False
    if cycle_remaining is not None and price > cycle_remaining:
        return False
    return True

_SELL_HOLD_WORDS = {"보유", "유지", "hold", "keep", "유보", "관망"}
_SELL_ALL_WORDS  = {"전량", "전부", "모두", "all", "full", "100%", "청산"}
_SELL_HALF_WORDS = {"절반", "반", "1/2", "half", "50%"}


def _load_trailing_peaks(uid) -> dict:
    """트레일링 익절용 종목별 고점 로드(없으면 빈 dict)."""
    from infra import user_paths
    try:
        p = user_paths.trailing_peaks_path(uid)
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
    except Exception:
        pass
    return {}


def _save_trailing_peaks(uid, peaks: dict, holdings) -> None:
    """현재 보유 종목의 고점만 남겨 저장 — 매도/청산된 종목 고점은 정리(무한증식 방지)."""
    from infra import user_paths
    try:
        held = set()
        for h in (holdings or []):
            c = str(h.get("code", "")).strip()
            if c:
                held.add(c if _is_kr_code(c) else c.upper())
        pruned = {k: v for k, v in (peaks or {}).items() if k in held}
        user_paths.trailing_peaks_path(uid).write_text(
            json.dumps(pruned, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _load_locked_streak(uid) -> dict:
    """매도 잠김 연속 카운트 로드(버그 C)."""
    from infra import user_paths
    try:
        p = user_paths.locked_sell_streak_path(uid)
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
    except Exception:
        pass
    return {}


def _save_locked_streak(uid, streak: dict) -> None:
    from infra import user_paths
    try:
        user_paths.locked_sell_streak_path(uid).write_text(
            json.dumps(streak or {}, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _load_drift_streak(uid) -> dict:
    """원장↔KIS 괴리 연속 사이클 카운트 로드(전이 오탐 알림 억제, 2026-06-22)."""
    from infra import user_paths
    try:
        p = user_paths.ledger_drift_streak_path(uid)
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
    except Exception:
        pass
    return {}


def _save_drift_streak(uid, streak: dict) -> None:
    from infra import user_paths
    try:
        user_paths.ledger_drift_streak_path(uid).write_text(
            json.dumps(streak or {}, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _assemble_sell_orders(holdings, sell_directives, *, enable_rebalance, take_profit_pct,
                          stop_loss_pct, trim_over_ratio, conservative_ratio, per_stock_cap, total,
                          sell_prices=None, trailing_pct=0.0, peaks=None):
    """보유종목 → 매도 주문 리스트 + price_map. KR(6자리)·US(티커) 모두 처리.
    자동 익절·손절·트레일링은 계정 공통 하드 안전망이며 사후관리실장의 '보유/절반'보다 우선한다.
    안전망 미발동 종목만 sell_directives 판단을 따른다. 자동매도를 원치 않는 프로필은
    ENABLE_SELL_REBALANCE=False 로 명시적으로 끌 수 있다.
    편중축소(TRIM)는 KRW per_stock_cap 기준이라 KR에만 적용 — US(USD 평가액)와 통화를
    섞으면 안 됨(버그 2026-05-22). 반환 order dict 의 market 으로 실행부가 us_sell/kr_sell 라우팅."""
    sell_directives = sell_directives or {}
    orders = []
    price_map: dict = {}
    for h in holdings:
        code = str(h.get("code", "")).strip()
        qty = int(h.get("qty") or 0)
        if not code or qty < 1:
            continue
        is_kr = _is_kr_code(code)
        pkey = code if is_kr else code.upper()
        pnl = float(h.get("pnl_pct") or 0.0)
        cur = float(h.get("cur_price") or 0.0)
        price_map[pkey] = cur
        # 트레일링 익절(2026-06-18): 종목별 고점(peak_price·peak_pnl)을 매 사이클 갱신·영속.
        _tt = float(trailing_pct or 0.0)
        if peaks is not None and _tt > 0:
            _pk = peaks.get(pkey) or {}
            peaks[pkey] = {"peak_price": max(float(_pk.get("peak_price") or 0.0), cur),
                           "peak_pnl": max(float(_pk.get("peak_pnl") or 0.0), pnl)}
        reason = None
        sell_qty = 0
        automatic_safety = False
        directive = sell_directives.get(code) or sell_directives.get(code.upper())
        # 계정 공통 결정론 안전망. LLM이 '보유'를 반환해도 설정된 손익 규율은 무력화되지 않는다.
        if enable_rebalance and float(take_profit_pct or 0.0) > 0 and pnl >= float(take_profit_pct):
            reason = f"자동 익절 — 평가손익 {pnl:+.1f}% ≥ +{take_profit_pct:.0f}%"; sell_qty = qty
            automatic_safety = True
        elif enable_rebalance and float(stop_loss_pct or 0.0) > 0 and pnl <= -float(stop_loss_pct):
            reason = f"자동 손절 — 평가손익 {pnl:+.1f}% ≤ -{stop_loss_pct:.0f}%"; sell_qty = qty
            automatic_safety = True
        elif enable_rebalance and _tt > 0 and peaks is not None and cur > 0:
            _pp = float((peaks.get(pkey) or {}).get("peak_price") or 0.0)
            _ppnl = float((peaks.get(pkey) or {}).get("peak_pnl") or 0.0)
            _retr = ((_pp - cur) / _pp * 100.0) if _pp > 0 else 0.0
            if _ppnl >= _tt and _retr >= _tt:
                reason = (f"트레일링 익절 — 고점 +{_ppnl:.1f}% → 현재 {pnl:+.1f}% "
                          f"(고점가 대비 -{_retr:.1f}% 되밀림 ≥ {_tt:.1f}%)"); sell_qty = qty
                automatic_safety = True

        if reason is None and directive is not None:
            dl = str(directive).strip().lower()
            if dl in _SELL_HOLD_WORDS:
                continue
            elif dl in _SELL_ALL_WORDS:
                sell_qty = qty; reason = "사후관리실장 매도 판단 — 전량"
            elif dl in _SELL_HALF_WORDS:
                sell_qty = max(1, qty // 2); reason = "사후관리실장 매도 판단 — 절반"
            else:
                mnum = re.match(r"(\d+)", dl)
                if mnum:
                    sell_qty = max(1, min(int(mnum.group(1)), qty)); reason = f"사후관리실장 매도 판단 — {sell_qty}주"
                else:
                    continue  # 알 수 없는 지시 → 보유로 간주
        if reason is None and directive is None and enable_rebalance:
            # 하드 안전망·LLM 지시가 모두 없는 종목의 편중만 축소한다.
            if is_kr and trim_over_ratio and per_stock_cap > 0 and cur > 0 and (cur * qty) > per_stock_cap:
                over = int(((cur * qty) - per_stock_cap) // cur) + 1
                sell_qty = max(1, min(over, qty))
                reason = f"편중 축소 — 비중 {cur*qty/total*100:.1f}% > {conservative_ratio*100:.0f}% 한도"
        if reason and sell_qty > 0:
            # 사장 지시 2026-05-22: 계량분석팀장이 '매도가'를 숫자로 제시하면 그 지정가로 매도.
            # (시장가/미지정·안전망 자동 익절손절은 시장가 유지 — 즉시 청산.)
            # 실제 손익으로 발동한 하드 안전망은 LLM의 매도가를 사용하지 않는다. 오래되거나
            # 잘못 생성된 지정가 때문에 익절·손절 주문이 미체결로 남는 것을 막는다.
            _sp = None if automatic_safety else (
                (sell_prices or {}).get(code) or (sell_prices or {}).get(pkey)
                or (sell_prices or {}).get(code.upper()))
            _lim = _sp.get("limit_price") if (_sp and _sp.get("mode") == "limit") else None
            _od = {"ticker": pkey, "side": "sell", "qty": sell_qty,
                   "price_type": "limit" if _lim else "market",
                   "market": "KR" if is_kr else "US",
                   "reason": (f"{h.get('name', pkey)} {reason} (보유 {qty}주, 평가손익 {pnl:+.1f}%"
                              + (f", 매도지정가 {_lim:,.0f}" if _lim else "") + ")")}
            if _lim:
                _od["entry_mode"] = "limit"; _od["entry_limit"] = _lim
            orders.append(_od)
    return orders, price_map


def _codes_for_session(holdings, session) -> List[str]:
    """현재 세션 시장에 해당하는 보유 종목코드만 반환. US세션→US티커, KR세션→6자리, 장외→전체.
    계량분석·매도평가 대상이 세션 시장과 일치하도록 한다 — 과거 6자리(KR)만 골라 US 보유분이
    분석에서 통째 빠지던 버그(2026-05-22) 방지. [[arquant-kr-us-asymmetry-bugs]]"""
    out: List[str] = []
    for h in (holdings or []):
        c = str(h.get("code", "")).strip()
        if not c:
            continue
        is_kr = _is_kr_code(c)
        if session == "US_TRADING":
            if not is_kr:
                out.append(c)
        elif is_kr_session(session):
            if is_kr:
                out.append(c)
        else:  # OFF_HOURS 등 — 양쪽 모두
            out.append(c)
    return out


def _looks_like_tool_call(text: str) -> bool:
    """계량분석팀장 등이 분석 대신 도구호출 JSON/코드를 뱉었는지 감지 (사장 피드백 2026-05-18).
    BaseAgent는 tool-exec 루프가 없으므로 이런 출력은 그대로 화면에 찍히는 버그였다."""
    if not text:
        return False
    t = text.strip()
    head = t[:400]
    # 1) 응답이 사실상 JSON/코드펜스로 시작 (분석 줄글이 아님)
    if t.startswith("{") or t.startswith("```"):
        return True
    # 2) tool-call 시그니처 — "api"/"params"/함수호출 표기 + 점수 줄 부재
    sig = ('"api"' in head or '"params"' in head or '"tool"' in head
           or "analyze_stock_technical(" in head or '"function"' in head)
    if sig and "퀀트점수" not in t:
        return True
    return False


def _strip_leading_section_marker(text: str, *markers: str) -> str:
    """LLM이 프롬프트의 `[후보 종목 선정]`/`[최종 매수 종목 결정]` 머리말을 그대로 따라 적어
    브로드캐스트 래퍼와 겹쳐 두 번 보이는 문제 제거 (사장 피드백 2026-05-18).
    선두에서 해당 마커로 시작하는 줄만 걷어낸다 (본문 중간 인용은 보존)."""
    if not text:
        return text
    s = text.lstrip()
    changed = True
    while changed:
        changed = False
        for mk in markers:
            if s.startswith(mk):
                nl = s.find("\n")
                s = (s[nl + 1:] if nl >= 0 else "").lstrip()
                changed = True
    return s


def _parse_sell_decisions(text: str) -> Dict[str, str]:
    """Parse the post-manager's '매도결정: 005930=전량, 000660=절반, AAPL=보유' line → {code: directive}."""
    out: Dict[str, str] = {}
    m = re.search(r"매도\s*결정\s*[:：]\s*(.+)", text or "", re.IGNORECASE)
    if not m:
        return out
    for part in re.split(r"[,，;]", m.group(1).splitlines()[0]):
        mm = re.match(r"\s*([0-9]{6}|[A-Za-z]{1,5})\s*=\s*([^\s,，;]+)", part)
        if mm:
            # 사장 지시 2026-07-20: '1매'/'5주'/'3매도' 등 단위 접미사를 주수로 정규화 —
            # 종전엔 isdigit() 판정에 걸려 매도가 조용히 누락됐다(#481 261220=1매).
            out[mm.group(1).upper()] = normalize_sell_directive(mm.group(2).strip())
    return out


_CLAIMED_CURRENT_PRICE_RE = re.compile(
    r"현재가\s*(?:\(\s*|[:：]?\s*(?:약\s*)?)([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:원|달러|USD|\$)?",
    re.IGNORECASE,
)


def sanitize_sell_directives_by_authoritative_prices(
        text: str, directives: Dict[str, str], holdings: List[Dict],
        *, max_deviation_pct: float = 3.0) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """LLM이 보유 현재가를 크게 틀린 채 만든 매도지시를 보유로 격리한다.

    자동 익절·손절은 이후 _assemble_sell_orders 가 권위 pnl_pct로 별도 집행하므로 여기서는
    자유서술 기반 지시만 보수적으로 차단한다. 가격 언급이 없거나 오차 허용범위 안이면 그대로 둔다.
    """
    cleaned = dict(directives or {})
    conflicts: List[Dict[str, Any]] = []
    if not text or not cleaned:
        return cleaned, conflicts
    facts = {}
    for h in holdings or []:
        code = str(h.get("code") or "").strip().upper()
        cur = float(h.get("cur_price") or 0.0)
        if code and cur > 0:
            facts[code] = {"cur": cur, "name": str(h.get("name") or "")}
    code_starts = {}
    upper_text = text.upper()
    for code, fact in facts.items():
        starts = [p for p in (upper_text.find(code), upper_text.find(fact["name"].upper())) if p >= 0]
        if starts:
            code_starts[code] = min(starts)
    ordered = sorted((p, c) for c, p in code_starts.items())
    for code, directive in list(cleaned.items()):
        key = str(code).strip().upper()
        if str(directive or "").strip().lower() in _SELL_HOLD_WORDS or key not in code_starts:
            continue
        start = code_starts[key]
        later = [p for p, other in ordered if p > start and other != key]
        section = text[start:(min(later) if later else len(text))]
        m = _CLAIMED_CURRENT_PRICE_RE.search(section)
        if not m:
            continue
        try:
            claimed = float(m.group(1).replace(",", ""))
        except (TypeError, ValueError):
            continue
        actual = facts[key]["cur"]
        deviation = abs(claimed - actual) / actual * 100.0 if actual > 0 else 0.0
        if claimed > 0 and deviation > float(max_deviation_pct):
            cleaned[code] = "보유"
            conflicts.append({"code": key, "claimed": claimed, "actual": actual,
                              "deviation_pct": deviation, "directive": directive})
    return cleaned, conflicts


def _build_sleeve_prompt(spec, macro, news, pool_txt, weight_ctx, thesis_reminder="", holdings_txt=""):
    """슬리브 매니저(채권/원자재) 프롬프트 조립(순수 함수). 사장 결정 2026-06-09:
    주식 계량분석은 부적합 → **매크로 + 뉴스만** 주입. 비중 밴드 안이어도 신호로 매도 판단하도록 지시.
    버그 2026-06-12: 현재 보유·가격을 정형 주입하지 않아 매니저(LLM)가 보유분을 '미보유'로
    단정하고 가격을 달러로 날조하던 환각(hh0908 137610·$가격) 차단 — holdings_txt 주입."""
    return (
        f"[매수 가능 {spec.manager_name} ETF 풀] — 이 안에서만 선택(코드 그대로):\n{pool_txt}\n\n"
        f"[현재 이 슬리브 보유 — 아래가 사실이다. 보유/가격을 추측하지 말 것]\n{holdings_txt or '현재 이 슬리브 보유 없음.'}\n\n"
        f"글로벌리서치팀장 매크로 보고:\n{(macro or '')[:1500]}\n\n"
        f"마켓센티먼트팀장 뉴스 분석:\n{(news or '')[:1000]}\n\n"
        f"{weight_ctx}\n"
        + (f"\n{thesis_reminder}\n" if thesis_reminder else "")
        + "\n⚠️ 비중이 목표 밴드 안이어도, 매크로·뉴스 신호가 악화됐으면 매도를 판단하십시오 — "
          "'비중%가 맞으니 보류'는 금지입니다(신호로 판단). 풀의 ETF 코드에 대해 매수/절반/보유/매도주수를 정하십시오.\n"
        # 사장 지시 2026-07-20: 매수 ETF의 배분 비중을 결정 줄에 `코드=매수:비중`(정수 %, 합 100)으로
        # 명시하게 한다 — 종전엔 매니저가 본문에 '50/30/20'을 써도 조립부가 균등 분할해 무시했다.
        + f"마지막 줄은 반드시 (다른 텍스트 없이) `{spec.decision_keyword}: 코드=값, ...` 형식.\n"
        + "  · 매수 ETF는 배분 비중을 함께: `코드=매수:비중` (정수 %, 매수분 합이 100이 되게). "
          f"예) `{spec.decision_keyword}: 153130=매수:50, 357870=매수:30, 114260=매수:20`\n"
        + "  · 매도는 `코드=전량`/`코드=절반`/`코드=주수`, 유지는 `코드=보유`.")


def _build_sleeve_sell_orders(decisions, sleeve_holdings, price_lookup, pool=None):
    """사후관리실장 통합 매도결정 중 **슬리브 풀 코드만** 골라 매도 주문으로 조립(순수 함수).
    슬리브별 spec 으로 그룹핑해 reason 을 정확히 붙인다. 사장 결정 2026-06-09: 슬리브 매도의
    최종 권한은 사후관리실장(주식+슬리브 종합). 코드리뷰 #5: pool 을 활성 슬리브 코드로 한정하면
    OFF 슬리브 보유는 주식 매도 트랙이 처리(여기선 제외)해 이중 조립을 피한다(기본 None=전체 풀)."""
    if pool is None:
        pool = all_sleeve_pool_codes()
    by_key: Dict[str, list] = {}
    for code, d in (decisions or {}).items():
        cu = str(code).strip().upper()
        if cu not in pool:
            continue
        spec = sleeve_for_code(cu)
        if spec is None:
            continue
        by_key.setdefault(spec.key, [spec, {}])[1][cu] = d
    orders: list = []
    for _key, (spec, dirs) in by_key.items():
        orders.extend(assemble_sleeve_orders(spec, "sell", 0, dirs, sleeve_holdings, price_lookup))
    return orders


def dedupe_sell_orders(orders):
    """같은 종목 매도 주문 중복 제거 — (ticker) 당 1건만 남긴다(순수 함수).

    사장 보고 2026-07-29: 주식 매도 트랙(_assemble_sell_orders)과 슬리브 매도 트랙
    (_build_sleeve_sell_orders)이 같은 사후관리실장 지시로 **같은 종목에 각각 매도 주문**을
    만들어 보유의 2배를 팔았다(261220: 보유 45 → 22+22 → 1주 잔여, 이후 매 사이클
    '잔고내역이 없습니다' 실패 + 사이클 화면에 결정 중복 표시). 두 생산자가 합류하는
    지점에서 '한 사이클 · 한 종목 · 매도 1건' 불변식을 강제한다.
    수량이 다르면 **큰 쪽**을 남긴다(전량 지시가 부분 지시에 잘리지 않게). 매수는 무관.
    반환: (정리된 orders, 제거된 [(ticker, dropped_qty)])"""
    seen: Dict[str, int] = {}      # ticker → orders 내 인덱스
    dropped: List[tuple] = []
    out: List[Dict] = []
    for o in (orders or []):
        if str(o.get("side") or "buy").lower() != "sell":
            out.append(o)
            continue
        tk = str(o.get("ticker") or "").strip().upper()
        if not tk:
            out.append(o)
            continue
        idx = seen.get(tk)
        if idx is None:
            seen[tk] = len(out)
            out.append(o)
            continue
        keep, drop = out[idx], o
        if int(float(drop.get("qty") or 0)) > int(float(keep.get("qty") or 0)):
            out[idx], drop = drop, keep
        dropped.append((tk, int(float(drop.get("qty") or 0))))
    return out, dropped


# 사장 피드백 2026-05-15 (#4): 계량분석팀장의 `진입가: code=값` 한 줄을 파싱.
# 값 패턴: '시장가' | 숫자(지정가) | '관망 ±X%' | '관망 X원' | 빈문자열 (시장가)
def _parse_entry_directive(text: str, code: str) -> Dict[str, Any]:
    """Returns {"mode": "market"/"limit"/"watch", "limit_price": float|None, "watch_pct": float|None, "raw": str}."""
    if not text or not code:
        return {"mode": "market", "limit_price": None, "watch_pct": None, "raw": ""}
    # 종목 라인 찾기 (단일 종목 응답이므로 한 줄만 봐도 됨)
    m = re.search(rf"진입가\s*[:：][^\n]*\b{re.escape(code)}\s*=\s*([^\n,，;]+)", text, re.IGNORECASE)
    if not m:
        return {"mode": "market", "limit_price": None, "watch_pct": None, "raw": ""}
    raw = m.group(1).strip()
    s = raw.lower().replace(",", "")
    if not s or "시장가" in s or "market" in s or "now" in s or "즉시" in s:
        return {"mode": "market", "limit_price": None, "watch_pct": None, "raw": raw}
    # 관망 ±X% 패턴
    mm = re.match(r"(?:관망|wait|watch|대기)\s*([+\-]?[\d.]+)\s*%", s)
    if mm:
        try:
            pct = float(mm.group(1))
            # 음수 표시가 빠진 경우 '하락' 단어 확인
            if pct > 0 and any(k in s for k in ("하락", "drop", "down", "내려")): 
                pct = -pct
            # 관망 비율 상한: ±10% 이내로 제한 (극단적 값 방지)
            MAX_WATCH = 10.0
            if pct < -MAX_WATCH:
                pct = -MAX_WATCH
            elif pct > MAX_WATCH:
                pct = MAX_WATCH
            return {"mode": "watch", "limit_price": None, "watch_pct": pct, "raw": raw}
        except ValueError: pass
    # 단순 숫자 (지정가)
    mm = re.match(r"([+\-]?[\d.]+)", s)
    if mm:
        try:
            px = float(mm.group(1))
            if px > 0:
                return {"mode": "limit", "limit_price": px, "watch_pct": None, "raw": raw}
        except ValueError: pass
    # 인식 실패 → 시장가 폴백
    return {"mode": "market", "limit_price": None, "watch_pct": None, "raw": raw}


def _parse_sell_price(text: str, code: str) -> Dict[str, Any]:
    """계량분석팀장의 `매도가: code=값` 한 줄 파싱 (사장 지시 2026-05-22).

    값: '시장가' | 숫자(지정가) | 빈문자열(시장가). 매도는 관망 모드 없음 — 시장가 또는 지정가만.
    Returns {"mode": "market"/"limit", "limit_price": float|None, "raw": str}."""
    if not text or not code:
        return {"mode": "market", "limit_price": None, "raw": ""}
    m = re.search(rf"매도가\s*[:：][^\n]*\b{re.escape(code)}\s*=\s*([^\n,，;]+)", text, re.IGNORECASE)
    if not m:
        return {"mode": "market", "limit_price": None, "raw": ""}
    raw = m.group(1).strip()
    s = raw.lower().replace(",", "")
    if not s or "시장가" in s or "market" in s or "즉시" in s:
        return {"mode": "market", "limit_price": None, "raw": raw}
    mm = re.match(r"([+\-]?[\d.]+)", s)
    if mm:
        try:
            px = float(mm.group(1))
            if px > 0:
                return {"mode": "limit", "limit_price": px, "raw": raw}
        except ValueError:
            pass
    return {"mode": "market", "limit_price": None, "raw": raw}


def _quant_ctx_for(report: str, code: str, width: int = 200) -> str:
    """주문 사유에 첨부할 '그 종목'의 퀀트 분석 앞부분을 돌려준다.

    quant_report 는 종목별 섹션을 `\\n\\n---\\n\\n` 으로 이은 것이고, 각 섹션엔
    `퀀트점수: {code}=` / `진입가: {code}=` 가 들어 있다. 그 앵커로 섹션을 찾아
    인접 종목 텍스트 오염을 막는다.
    버그 2026-05-22: 옛 구현은 report.find(code) 후 [idx-100:idx+200] 윈도우를 ctx[:90]
    으로 잘라, idx-100 이 직전 종목 섹션의 꼬리를 끌어와 SK텔레콤(017670) 주문에
    지아이이노베이션(358570) 퀀트가 붙었다."""
    if not report or not code:
        return ""
    anchor = re.compile(rf"(?:퀀트점수|진입가)\s*[:：][^\n]*\b{re.escape(code)}\s*=", re.IGNORECASE)
    for sec in report.split("\n\n---\n\n"):
        if anchor.search(sec):
            body = sec.split("\n", 1)[1] if "\n" in sec else sec
            return body.replace("\n", " ").strip()[:width]
    # 섹션 매칭 실패 — 코드 첫 등장 위치에서 '앞으로'만(뒤로 끌어오지 않음)
    m = re.search(re.escape(code), report, re.IGNORECASE)
    if m:
        return report[m.start(): m.start() + width].replace("\n", " ").strip()
    return ""


def _affordable_buy_qty(price: float, *, per_order_budget: float, per_stock_cap: float,
                        cycle_remaining: float) -> int:
    """세 한도(주문당 예산·단일종목 한도·사이클 잔여예산)의 최소값 안에서 매수 수량을 산정.

    리스크관리실장 결정론 검증(guardrails._check_single_order)과 **같은 한도**를 사이징
    단계에서 선반영해, 종목을 골라놓고 비중·예산 초과로 반려되는 낭비를 막는다.
    버그 2026-05-22: 개장 사이클이 per_stock_cap 을 60%로 완화했으나 검증은 25%로 반려 →
    삼성전자 6주(58.6%)·대한항공 69주(59.9%)가 매번 반려, 매수 0건."""
    if price <= 0:
        return 0
    budget = min(per_order_budget, per_stock_cap, cycle_remaining)
    return int(budget // price) if budget > 0 else 0


def _is_us_buy_fail(o: dict) -> bool:
    """exec_result 가 'USD 부족류' US 매수 거부인가 (주문가능금액 초과/부족).
    KR 거부·접수·체결은 제외. 회로차단기 카운터 입력."""
    if not isinstance(o, dict):
        return False
    if o.get("market") != "US" or o.get("side") != "buy":
        return False
    if o.get("accepted") or o.get("filled"):
        return False
    txt = f"{o.get('result') or ''} {o.get('fill_note') or ''}"
    return ("주문가능금액" in txt) or ("초과" in txt)


def _us_activity_freed_cash(exec_results) -> bool:
    """이번 사이클에 USD 를 확보한 US 활동이 있었나 — US 매도 접수 또는 US 매수 성공.
    있으면 회로차단기 스트릭을 리셋한다(자가치유: 살 여력이 생겼다는 신호)."""
    for o in (exec_results or []):
        if not isinstance(o, dict) or o.get("market") != "US":
            continue
        if o.get("side") == "sell" and o.get("accepted"):
            return True
        if o.get("side") == "buy" and (o.get("accepted") or o.get("filled")):
            return True
    return False


def _update_us_buy_fail_streak(prev: int, exec_results) -> int:
    """연속 US 매수실패(순수 낭비 사이클) 카운터 갱신.
    - US 성공(매도 접수/매수 체결) → 0 리셋
    - 그 외 US 매수실패 건수만큼 += (US 활동 없으면 유지)."""
    if _us_activity_freed_cash(exec_results):
        return 0
    fails = sum(1 for o in (exec_results or []) if _is_us_buy_fail(o))
    return prev + fails if fails else prev


def _us_buy_circuit_open(streak: int, threshold: int) -> bool:
    """연속 US 매수실패가 임계치 이상이면 회로 개방(US 매수 차단). threshold≤0 이면 비활성."""
    return threshold > 0 and streak >= threshold


def _us_buy_usd_basis(bp_ok: bool, real_usd: float, *, krw_cash: float, krw_per_usd: float) -> float:
    """US 매수 사이징의 USD 현금 기준 (사장 지시 2026-06-17, uid1 phantom 매수 버그).

    KIS 실제 USD 주문가능액(us_buying_power.usd, ok=True)을 우선 쓴다 — 음수/0은 0으로 바닥.
    조회 실패(ok=False: 모의서버 해외 TR 미지원·에러)시에만 KRW예수금÷환율 합성으로 폴백한다
    (주문 드롭 금지 원칙 — 폴백 시엔 집행 직전 클램프가 최종 방어). 합성을 기본값으로 쓰면
    통합증거금을 과대평가해 USD 0 계좌에 phantom 1주 초안→KIS '주문가능금액 초과'가 반복된다."""
    if bp_ok:
        return max(0.0, float(real_usd or 0.0))
    return max(0.0, float(krw_cash or 0.0) / float(krw_per_usd or 1.0))


def _one_share_exclude_reason(price: float, cash: float, cyc_rem: float) -> str:
    """1주조차 못 사 제외할 때의 '진짜 binding 한도'를 사람이 읽을 문구로 (사장 지시 2026-06-17).
    cyc430 final_report 가 UAL($119.46 < 잔여예산 $175.29)을 '예산 한도 초과'로 잘못 표기했다 —
    실제 binding 은 예수금 $0.00. 예수금을 우선 사유로(자금이 근본), 다음 사이클 예산 순."""
    if float(cash or 0.0) < float(price or 0.0):
        return "예수금 부족"
    if float(cyc_rem or 0.0) < float(price or 0.0):
        return "사이클 예산 초과"
    return "한도 초과"


def _clamp_qty_to_buyable(qty, buyable_qty) -> int:
    """주문수량을 KIS 권위 매수가능수량(nrcvb_buy_qty / max_ord_psbl_qty)으로 클램프.
    리스크 승인(bp.cash 기반)과 KIS 실제 주문가능금액(증거금·D+2 미결제 반영)의 불일치로
    마지막 매수가 '주문가능금액 초과' 거부되던 문제(uid1 cycle380 241710) 방어. buyable_qty
    None(조회 실패)이면 원래 수량 유지(폴백 — 조회 실패로 매수를 막지 않음), 0/음수면 0
    (호출부가 보류 + 사유 발화 — 조용한 드롭 금지)."""
    if buyable_qty is None:
        return int(qty)
    return max(0, min(int(qty), int(buyable_qty)))


def _locked_sell_action(sellable, before_qty, qty, has_pending_sell,
                        locked_streak: int = 0, escalate_after: int = 3):
    """KR 매도 집행 결정 — 매도가능수량(sellable, ord_psbl_qty) 기준. (사장 지시 2026-06-16)

    지정가 매도가 시세 위라 미체결되면 그 주문이 보유물량을 잠가 ord_psbl_qty=0 이 된다. 기존엔
    '매도가능 0'이면 무조건 보류라 kr_sell(펜딩 취소 후 신규 전송)을 영영 못 불러 영원히 잠겼다
    (375500 익절이 종일 미체결). 이제 미체결 펜딩 매도가 '실제로 있으면' 펜딩 취소 후 시장가로
    재청산(재가격+재시도)하고, 없으면(결제/글리치 잠금) 보류한다 — 무리한 시장가 청산은 금물.

    사장 지시 2026-06-18(버그 C): 펜딩 없는 잠김(결제/제도 잠금 추정)이 escalate_after 사이클
    연속되면 무한 보류를 끊고 'escalate_market'(강제 시장가 재청산 시도 + 큰 경고)으로 표면화한다 —
    손절이 종일 집행 불가로 방치되던 위험(007340 -5.5% 등) 제거. kr_sell 은 펜딩취소+다중폴백이라
    잠금이 거짓(모의/글리치 false-negative)이면 풀리고, 진짜 결제잠금이면 KIS 가 거부를 명시한다.

    Returns (action, value):
      ('proceed', qty)            — 정상 진행
      ('clamp', sellable)         — 매도가능 일부 → 그만큼으로 축소
      ('reprice_market', qty)     — 잠김+펜딩존재 → 펜딩 취소+보유 전량 시장가 재청산
      ('hold', None)              — 잠김+펜딩없음+스트릭<임계 → 보류(결제 해소 기회)
      ('escalate_market', qty)    — 잠김+펜딩없음+스트릭≥임계 → 강제 시장가 재청산 시도+에스컬레이션
    """
    if sellable is None or int(before_qty or 0) <= 0:
        return ("proceed", int(qty))
    s = int(sellable)
    if s > 0:
        return ("clamp", s) if int(qty) > s else ("proceed", int(qty))
    # s <= 0 — 물량이 잠겼다
    if has_pending_sell:
        return ("reprice_market", int(before_qty))
    if int(locked_streak or 0) >= max(1, int(escalate_after)):
        return ("escalate_market", int(before_qty))   # N사이클 연속 잠김 → 에스컬레이션
    return ("hold", None)


def _locked_sell_escalation_alert(tk, before_qty, escalate_after, is_mock=False):
    """매도 잠김 에스컬레이션 알림 문구 → (severity, title, message). (사장 지시 2026-06-22)

    모의서버는 '오늘 매도 누적(thdt_sll_qty)'이 잔여 보유분의 ord_psbl_qty 를 0 으로 깎는 일일
    카운터 quirk 가 있어(uid2 316140: hldg 100·ord_psbl 0·thdt_sll 160·펜딩 0, 익일 리셋 시
    자동 해소) 실계정용 '수동 확인 필요' WARN 이 오해를 샀다. 모의는 INFO + '일일 매도한도 제약
    (익일 해소)'로 순화하고, 실계정은 진짜 결제/제도 잠금일 수 있어 기존 WARN(수동 확인 유도)을 보존한다."""
    if is_mock:
        return ("INFO", "모의 매도 일시 제약",
                f"{tk} 모의서버 일일 매도한도 제약 — 보유 {before_qty}주 매도가능 0주(오늘 매도 누적·펜딩없음). "
                f"강제 시장가 재청산을 시도하나 모의서버가 당일 거부할 수 있다(익일 카운터 리셋 시 자동 해소).")
    return ("WARN", "손절 매도 잠김 에스컬레이션",
            f"{tk} 매도 {escalate_after}사이클 연속 잠김(매도가능 0·펜딩없음) — "
            f"보유 {before_qty}주 강제 시장가 재청산 시도. 결제/제도 잠금이면 KIS가 거부를 명시한다(수동 확인 필요)")


# 체결 미확인 주문을 다시 확인하기까지의 대기 시간(초). 모듈 상수로 둬서 테스트가
# 5분 실대기 없이 재확인 로직을 검증할 수 있게 한다(seam).
_REVERIFY_DELAY_SEC = 300
# 사장 지시 2026-05-21: 미체결 주문을 5분마다 반복 폴링하는 최대 횟수 (5분 × 36 = 3시간).
# 보통은 해당 시장 마감(KIS 자동취소) 시 조기 종료되고, 이 상한은 안전 장치(루프 누수 방지)다.
_POLL_MAX_ATTEMPTS = 36


def _trade_event_type(ok: bool) -> str:
    """체결 이벤트 타입을 누적 카운트 기준(ok)과 일치시킨다.
    ok=True 면(체결 확인 또는 US 잠정 접수) trade_executed, 아니면 trade_failed.
    이전엔 filled 기준이라 US 접수만(ok=True, filled=False) 주문이 카운트엔 포함되면서도
    이벤트는 trade_failed 로 나가 거래내역이 불일치했다(2026-05-21 회귀)."""
    return "trade_executed" if ok else "trade_failed"


def _fill_badge(filled: bool, is_mock: bool = False) -> str:
    """체결 알림 배지. 모의계정이면 '실매매' 가 아니라 '모의' 로 표기한다.
    버그 2026-06-12: 본문엔 '모의투자' 가 오는데 제목만 '실매매 체결확인' 으로 나가
    운영자가 모의 체결을 실거래로 오인하던 표시 결함(hh0908)."""
    if not filled:
        return "⚠ 주문 실패"
    return "✅ 모의 체결확인" if is_mock else "✅ 실매매 체결확인"


def _poll_close_message(ticker, side, remaining_qty, reason: str) -> str:
    """미체결 폴링 종료 시 사용자에게 상태를 닫는 메시지 — 사장 지시 2026-06-16.
    기존 '조용히 폐기'로 '접수 후 확인 중'이 영영 안 닫히던 문제(uid2 375500) 해결.
    reason='market_close'(장 마감 자동취소) | 'max_attempts'(체결 확인 한도 도달). 미체결이므로
    체결 카운트와 무관 — 표시만 닫는다."""
    act = "매수" if (side or "buy") == "buy" else "매도"
    why = "장 마감으로 자동취소(KIS)" if reason == "market_close" else "체결 확인 한도 도달(미체결 종료)"
    return f"⛔ {ticker} {act} 미체결 {max(0, int(remaining_qty or 0))}주 — {why} · 주문 종료"


def _extract_code_news(news_report: str, code: str, name: str = "") -> str:
    """센티먼트 리포트에서 **해당 종목의 단락만** 추출(없으면 '').
    버그 2026-06-12: thesis 작성 시 종목 무관한 news_report 앞부분(센티 리포트는 항상
    최고 화제주=삼성으로 시작)을 통째 주입 → 대한항공 thesis 에 삼성 사유가 박히는 오염
    (hh0908 003490). '- 📰 종목(코드)' 불릿 단위로 쪼개 코드·이름 일치 블록만 돌려준다."""
    if not news_report:
        return ""
    blocks: List[str] = []
    cur: List[str] = []
    for ln in str(news_report).splitlines():
        s = ln.lstrip()
        if s.startswith("- 📰") or s.startswith("📰"):
            if cur:
                blocks.append("\n".join(cur))
            cur = [ln]
        elif cur:
            cur.append(ln)
    if cur:
        blocks.append("\n".join(cur))
    code = str(code or "").strip()
    name = str(name or "").strip()
    hits = [b for b in blocks
            if (code and code in b) or (name and name in b)]
    return "\n".join(hits).strip()


def _format_exec_for_report(exec_results: List[Dict]) -> str:
    """전략실장 최종요약(LLM) 입력용 실행요약 — 체결/접수대기/실패를 정직히 구분한다.
    접수만 되고 미체결인 주문(US 즉시확인 불가·KR 지정가 미도달)을 '실패'로 넣으면
    LLM이 사유를 지어내므로(유동성 부족·VWAP 괴리 등 환각), '접수·체결대기'로 표기한다.
    거부(accepted=False)만 '실패'."""
    if not exec_results:
        return "없음"
    parts = []
    for e in exec_results:
        tk = str(e.get("ticker") or "").strip()
        side = e.get("side") or "buy"
        qty = e.get("qty", 0)
        if e.get("filled"):
            status = "체결"
        elif e.get("accepted"):
            status = "접수·체결대기(미확인)"
        else:
            status = "실패(거부)"
        parts.append(f"{tk} {side} x{qty} → {status}")
    return "; ".join(parts)


def _build_cycle_final_report(exec_results: List[Dict], risk_result=None,
                              sizing_notes=None, *, dart_vetoed=None) -> str:
    """사이클 최종 보고를 실행 원장만으로 조립한다.

    최종 보고를 LLM에 다시 맡기면 체결되지 않은 주문을 체결로 바꾸거나, 실패 사유를
    추측하는 문제가 생긴다. 사용자에게 보이는 상태는 주문 원장의 accepted/filled 값과
    리스크 엔진 결과만 사용하고, 시장 전망은 별도 에이전트 로그에 남긴다.
    """
    rows = list(exec_results or [])
    filled = [e for e in rows if e.get("filled")]
    pending = [e for e in rows if e.get("accepted") and not e.get("filled")]
    failed = [e for e in rows if not e.get("accepted") and not e.get("watch")]
    watching = [e for e in rows if e.get("watch")]

    def _label(e):
        side = "매도" if (e.get("side") or "buy") == "sell" else "매수"
        q = e.get('qty') or 0
        oq = int(e.get('order_qty') or 0)
        base = f"{e.get('ticker') or '?'} {side} {q}주"
        # 사장 지시 2026-06-16: 부분체결은 잔여 수량을 최종 보고에 명시(uid2 012510 1/84 사례).
        if oq and int(q or 0) < oq:
            base += f"(부분 {q}/{oq}, 잔여 {oq - int(q or 0)})"
        return base

    lines = []
    if filled:
        lines.append("체결 확인: " + ", ".join(_label(e) for e in filled) + ".")
    if pending:
        lines.append("주문 접수 후 체결 확인 중: " + ", ".join(_label(e) for e in pending) + ".")
    if watching:
        lines.append("진입 조건 감시 중: " + ", ".join(_label(e) for e in watching) + ".")
    if failed:
        details = []
        for e in failed:
            reason = re.sub(r"\s+", " ", str(e.get("result") or "거래소 거부")).strip()
            details.append(f"{_label(e)} ({reason[:120]})")
        lines.append("주문 실패: " + "; ".join(details) + ".")

    rejected = []
    for result in (risk_result or {}).get("results") or []:
        if result.get("status") != "APPROVED":
            rejected.append(str(result.get("ticker") or "?"))
    if rejected:
        lines.append("리스크 반려: " + ", ".join(rejected) + ".")
    # 사장 지시 2026-06-16(투명성): DART 2차 공시 재심 반려는 1차 risk_result 엔 APPROVED 로
    # 남아 위 rejected 에 안 잡힌다 — 미집행 매수를 별도로 명시해 '승인했는데 실행 안 됨' 유령
    # 주문(감사 추적 불가)을 없앤다. OPS 가 실패 사유를 못 봐 '유동성 부족'으로 환각하던 근본 원인.
    # (cycle 379 035250 강원랜드 사례.)
    _vetoed = sorted({str(t).upper() for t in (dart_vetoed or set()) if str(t).strip()})
    if _vetoed:
        lines.append("DART 공시 반려(미집행 매수): " + ", ".join(_vetoed) + ".")
    if not lines:
        lines.append("이번 사이클은 실제 주문과 체결 없이 종료되었습니다.")

    notes = [re.sub(r"\s+", " ", str(n)).strip() for n in (sizing_notes or []) if str(n).strip()]
    if notes:
        lines.append("다음 사이클 유의: " + " | ".join(notes)[:500] + ".")
    elif failed or rejected:
        lines.append("다음 사이클 유의: 실패·반려 원인을 재확인한 뒤 주문을 다시 판단합니다.")
    else:
        lines.append("다음 사이클 유의: 새 데이터로 후보와 보유 포지션을 다시 검증합니다.")
    return "\n".join(lines)


def _format_sizing_notes_for_report(notes) -> str:
    """주문 조립 단계의 스킵/사이징 사유(order_obj['sizing_notes'])를 보고용 한 줄 블록으로.
    비면 ''. 주식운용실장이 '왜 안 샀는지'(예: 시세조회 실패)를 사실대로 보고하게 한다(관측성)."""
    if not notes:
        return ""
    return "주문 조립 메모(미체결·제외 사유): " + " | ".join(str(n) for n in notes if n)


def _format_order_disposition(risk_result) -> str:
    """리스크 검증 결과 → '주문 처리 결과' 결정론 요약 (주식운용실장 리포트 환각 방지).
    사장 검토 2026-05-29: 리포트가 리스크 반려된 매수를 '최종 매수 선정'으로 단정하는
    환각(선정≠체결 혼동)을 막기 위해, 승인/반려를 종목·방향·사유와 함께 명시 주입한다."""
    results = (risk_result or {}).get("results") or []
    if not results:
        return "주문 처리 결과: 검증된 주문 없음."
    appr, rej = [], []
    for r in results:
        tk = str(r.get("ticker") or "?").strip()
        side = "매수" if (r.get("side") or "buy") != "sell" else "매도"
        qty = r.get("qty", "?")
        if r.get("status") == "APPROVED":
            appr.append(f"{tk} {side} x{qty}")
        else:
            why = "; ".join(r.get("issues") or []) or "사유 미상"
            rej.append(f"{tk} {side} x{qty} — 반려: {why}")
    lines = ["승인: " + (", ".join(appr) if appr else "없음"),
             "반려: " + ("; ".join(rej) if rej else "없음")]
    return "주문 처리 결과 (리스크관리실장 결정론 검증):\n" + "\n".join(lines)


def cycle_health_warnings(exec_results: List[Dict]) -> List[str]:
    """사이클 실행 결과의 '진짜 이상'을 코드로 점검 — 거부·전건 미접수는 경고로 띄운다.
    접수-후-미체결(US 폴링 대기·KR 지정가 미도달)은 정상 대기라 경고하지 않는다(잡음 방지).
    체결률이 낮아도 ok:true 로 묻히던 문제(2026-05-27 진단)를 가시화하기 위함."""
    real = [e for e in (exec_results or []) if not e.get("watch")]
    if not real:
        return []
    warnings: List[str] = []
    # 사장 보고 2026-06-26: 시간외(NXT) '사전 보류'는 '거부'가 아니라 정규장에서 재시도 예정인
    # 정상 보류다(NXT 미상장/시세없음/블랙리스트 → _finalize_kr_order_for_session 의도된 deferral).
    # 이를 거부로 묶어 '전건 미접수 → 자격증명·잔고 점검' 경보를 띄우면 거짓경보가 된다(밤사이 3회 발생).
    # 따라서 시간외 사전 보류는 별도 정보성 경고로만 가시화하고, 자격증명·잔고 경보에선 제외한다.
    def _is_ext_hours_defer(e):
        return str(e.get("fill_note") or "").strip() == "시간외 주문 사전 보류"
    unaccepted = [e for e in real if not e.get("accepted")]
    deferred = [e for e in unaccepted if _is_ext_hours_defer(e)]
    rejected = [e for e in unaccepted if not _is_ext_hours_defer(e)]
    if deferred:
        # 정상 시간외 보류는 경보가 아니다(자격증명·잔고 무관) — 실행부에서 이미 종목별 trade_failed
        # 로 발화됐다. 여기선 디버그 로깅만 남기고 운영자 경보(자기검증 WARN)는 띄우지 않는다.
        _dp = "; ".join(str(e.get("ticker") or "?") for e in deferred)
        logger.debug(f"[자기검증] 시간외(NXT) 주문 {len(deferred)}건 정규장 재시도 보류 — {_dp}")
    if rejected:
        # 사장 피드백 2026-06-16: '사유 점검 필요'로 떠넘기지 말고 exec_result 의 실제 사유를
        # 경고에 직접 담는다(fill_note > result 순). 종목별로 무엇이 왜 막혔는지 한 줄에 보인다.
        def _why(e):
            w = (e.get("fill_note") or e.get("result") or "사유 미상")
            return str(w).strip().splitlines()[0][:80]
        _parts = "; ".join(f"{str(e.get('ticker') or '?')}({_why(e)})" for e in rejected)
        warnings.append(f"주문 거부·보류 {len(rejected)}건 — {_parts}")
    # '전건 미접수' 경보는 *실거부*가 있을 때만. 정상 시간외 보류만으로는 자격증명·잔고 점검을 띄우지 않는다
    # (사장 보고 2026-06-26: NXT 보류만으로 '자격증명·잔고 점검' 거짓경보가 밤사이 3회 발생).
    if rejected and not any(e.get("accepted") for e in real):
        warnings.append("이번 사이클 주문 전건 미접수(거부) — 실행 경로·자격증명·잔고 점검 필요")
    return warnings


async def _llm_classify_directive(message: str, response: str) -> str:
    """사장 지시 2026-05-21(2026-07-03 확장): 사장 채팅 메시지를 경량 LLM으로 3분류.
    STRATEGY — 본인 매매 전략 서술(종목 선정 기준·매수/매도 규칙·지표 조건 등)을 이후 운용에 적용.
    STANDING — 전략 서술까지는 아니나 매 운용에 지속 적용할 상시 원칙·정책.
    ONESHOT  — 일회성 질문·현황 조회·단발 명령. 실패 시 보수적으로 ONESHOT(미저장)."""
    try:
        from config import MODEL_ASSIGNMENTS
        from infra.local_llm_client import chat_completion, response_text
        model = MODEL_ASSIGNMENTS.get("news_curator") or ""
        sys_p = ("당신은 분류기입니다. 사장이 운용역에게 보낸 메시지를 셋 중 하나로 분류하세요. "
                 "STRATEGY: 사장이 '본인의 매매 전략'(종목 선정 기준, 매수/매도 규칙, 지표 조건, "
                 "포트폴리오 구성 방식 등)을 서술하며 앞으로 그 전략대로 운용하라는 내용. "
                 "STANDING: 전략 전체 서술은 아니지만 앞으로 매 운용에 지속 적용할 상시 원칙·정책 "
                 "(포트폴리오 비중·자산배분·익절/손절·금지/우선 종목·리스크 한도 등 단편 규칙). "
                 "ONESHOT: 단순 질문·현재 현황 조회·1회성 실행 요청. "
                 "오직 한 단어로만 답하세요: STRATEGY 또는 STANDING 또는 ONESHOT.")
        usr_p = f"[사장 메시지]\n{message}\n\n[운용역 응답 요지]\n{(response or '')[:600]}"
        d = await chat_completion(
            api_key="", model=model,
            messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
            max_tokens=8, temperature=0.0, timeout_sec=30, thinking=False,
        )
        reply = (response_text(d) or "").upper()
        if "STRATEGY" in reply:
            return "STRATEGY"
        if "STANDING" in reply:
            return "STANDING"
        return "ONESHOT"
    except Exception as _e:
        logger.warning(f"지시 분류 실패(보수적 ONESHOT): {_e}")
        return "ONESHOT"


class _OpsRouterMixin:
    async def ceo_directive(self, message: str) -> str:
        # 사장 지시 2026-05-20: 운용지원실장은 ADMIN·일반 유저 모두 사용 가능
        # (프로필 한정 파라미터 조정만 — 코드 자가수정·산하 팀장 폐지).
        _auid, _admin = self._active_actor()
        # 선두 @<태그> 만 라우팅 태그로 인정한다(문장 중간 @멘션은 본문으로 취급).
        mention = re.match(r"\s*@(\S+)", message or "")
        if mention:
            name = mention.group(1)
            # 사장 지시 2026-05-20: 운용지원실장은 ADMIN·일반 유저 모두 사용 가능(프로필 한정
            # 파라미터 조정만). 산하 팀장 멘션 라우팅은 폐지(빈 매핑) — 일반 경로로 흐른다.
            agent = self._agents_map.get(name)
            if agent:
                # 운용지원실장: 자동 분류 후 적절한 팀장 role로 spawn
                if name == "운용지원실장":
                    # 운용지원실장은 ADMIN·일반 유저 모두 '프로필 한정 파라미터 튜닝'을 수행한다.
                    # (운용지원실장은 애초에 소스/서버를 건드릴 능력이 없으므로 별도 거부 게이트가 불필요.)
                    return await self._ops_support_execute(message)
                # 일반 에이전트는 대화 페르소나 — 설정/코드는 직접 못 바꿈.
                # 전략/예산 조정이 필요하면 운용지원실장 라인이 '이 프로필 전용' 파라미터로 반영.
                _guide = ("(참고: 당신은 시스템 설정·소스코드를 직접 변경할 수 없습니다. 이 지시가 전략/예산/매매규칙 "
                          "조정을 요구하면 의견·이유는 자유롭게 답하시되, '운용지원실장 라인을 통해 이 프로필 전용 "
                          "파라미터로 반영하겠다'라는 식으로 진행 의도를 밝히십시오. 단순 질의·분석이면 평소 역할대로 "
                          "답하고, 무성의하게 거절하지 마십시오.)")
                # 사장 지시 2026-05-19: 포지션 인지 에이전트(프롭트레이딩팀장·사후관리실장)가
                # 실시간 잔고 없이 '보유 0주·이미 정리됨' 같은 환각을 답한 사례(한화시스템
                # 272210: 09:14 매수 1주 보유 중인데 09:39 "정리됨"이라 거짓 보고)를 차단 —
                # 멘션 시 실제 KIS 잔고를 컨텍스트로 주입한다.
                _live_ctx = None
                if name in ("프롭트레이딩팀장", "사후관리실장"):
                    try:
                        _live_ctx = await self.broker.kr_balance()
                    except Exception as _be:
                        record_error(name, _be, context="ceo_directive 멘션 잔고 주입 실패", uid=self.uid)
                        _live_ctx = ("[국내 계좌잔고] 조회 실패 — 보유 수량을 단정하지 말고 "
                                     "'확인 불가'로 답하십시오.")
                resp = await agent.think(f"[🔴 사장 직접 지시] {message}\n\n{_guide}", context=_live_ctx)
                await self._emit({"type":"agent_msg","agent":name,"message":resp})
                # 자동 체이닝 — 운용지원실장(프로필 한정 파라미터 조정)으로 (ADMIN·일반 공통).
                if self._needs_ops_chain(resp):
                    await self._auto_chain_to_ops(message, source_agent=name, source_response=resp)
                asyncio.create_task(self._auto_persist_directive(message, resp))   # 체크박스 대체: 자동 판단 저장
                return resp
            # Task 8: 잘못된/알 수 없는 태그 → 하드 에러로 막지 말고 가장 적절한 에이전트(주식운용실장)에게
            # 핸드오프한다. 선두의 잘못된 @태그 토큰만 제거하고 본문을 주식운용실장에게 넘긴다.
            logger.info("ceo_directive: 알 수 없는 태그 '@%s' — 주식운용실장으로 핸드오프", name)
            _body = re.sub(r"^\s*@\S+\s*", "", message or "", count=1)
            message = _body if _body.strip() else (message or "")
        resp = await self.orchestrator.think(f"[🔴 사장 직접 지시] {message}")
        await self._emit({"type":"agent_msg","agent":"주식운용실장","message":resp})
        # 주식운용실장 응답도 자동 체이닝 — 운용지원실장(프로필 한정 조정)으로 (ADMIN·일반 공통).
        if self._needs_ops_chain(resp):
            await self._auto_chain_to_ops(message, source_agent="주식운용실장", source_response=resp)
        asyncio.create_task(self._auto_persist_directive(message, resp))   # 체크박스 대체: 자동 판단 저장
        return resp

    async def _auto_persist_directive(self, message: str, response: str) -> None:
        """사장 지시 2026-05-21: 체크박스 제거 — 지시가 '지속 적용 상시 원칙'인지 LLM이 판단해
        활성 계정의 standing_directive 로 자동 저장한다. 질문·일회성·에러응답은 저장하지 않는다.
        확장 2026-07-03: '본인 매매 전략' 서술(STRATEGY)이면 user_strategy 로 저장해 다음 사이클부터
        계량분석팀장 평가에 적용하고, 운용지원실장 자동 튜닝을 끈다(전략과 자동 튜닝 충돌 방지)."""
        try:
            _auid, _ = self._active_actor()
            if _auid is None:
                return
            if not (message or "").strip():
                return
            if (response or "").startswith("["):   # 에이전트 에러 응답이면 판단 보류
                return
            verdict = await _llm_classify_directive(message, response)
            if verdict == "ONESHOT":
                return
            text = message.strip()
            if text.startswith("@"):               # 선두 @멘션 토큰 제거
                parts = text.split(None, 1)
                text = parts[1] if len(parts) > 1 else ""
            if not text:
                return
            if verdict == "STRATEGY":
                from infra import user_strategy
                user_strategy.set_strategy(_auid, text, source="chat")
                runtime.set_ops_feedback(False, uid=_auid, by="user_strategy")
                await self._emit({"type": "agent_msg", "agent": "시스템",
                                  "message": "🧭 위 메시지를 '사용자 전략'으로 인식해 저장했습니다 — 다음 사이클부터 "
                                             "계량분석팀장 종목 평가에 이 전략을 적용하고, 운용지원실장 자동 튜닝을 "
                                             "껐습니다. (프로필 › 지시사항 관리에서 전략 확인·삭제, 전략 삭제 시 "
                                             "운용지원실장 재활성화)"})
                return
            from infra.standing_directives import append_directive
            if append_directive(_auid, text):
                await self._emit({"type": "agent_msg", "agent": "시스템",
                                  "message": "📌 위 지시를 '지속 운영 원칙'으로 판단해 상시 지시로 저장했습니다 "
                                             "— 프로필 › 지시사항 관리에서 확인·삭제할 수 있습니다."})
        except Exception as _e:
            logger.warning(f"지시 자동 저장 판단 실패: {_e}")

    @staticmethod
    def _needs_ops_chain(response_text: str) -> bool:
        """사장 지시 2026-05-14: 일반 에이전트 응답에서 '운용지원실장에게 위임' 신호 감지.
        과거에는 사장이 직접 다시 @운용지원실장 멘션해야 했지만 이제 시스템이 자동 체이닝한다.
        무한 루프 방지: 응답 자체가 워커 spawn 메시지(🛠 ... 별도 프로세스 워커)면 매칭 안 함."""
        if not response_text: return False
        # 워커 spawn 알림 자체엔 절대 매칭 금지
        if "별도 프로세스 워커" in response_text or "워커 spawn" in response_text:
            return False
        signals = (
            "@운용지원실장",
            "@투자관리팀장", "@경영관리팀장", "@재무관리팀장",
            "운용지원실장에게 지시", "운용지원실장에게 말",
            "운용지원실장에 지시", "운용지원실장에 말",
            "직접 반영할 수 없", "시스템 설정 변경", "시스템에 반영", "코드 수정", "코드 변경",
            "직접 변경할 수 없", "설정 변경이 필요",
        )
        return any(s in response_text for s in signals)

    async def _auto_chain_to_ops(self, original_message: str, source_agent: str, source_response: str):
        """사장 지시 2026-05-14: 자동 지명 호출 — 사장이 다시 멘션할 필요 없이 시스템이 직접 워커 spawn.
        사장 지시 2026-05-20: 산하 팀장 폐지 → 모든 지시는 운용지원실장(ops_support) 단일 처리."""
        # 사용자에게 체이닝 사실을 명확히 표시 (UI 로그에 노출)
        await self._emit({"type": "agent_msg", "agent": "시스템",
                          "message": f"🔁 자동 지명 호출: {source_agent} → 운용지원실장. "
                                     f"사장님 원지시를 그대로 전달합니다."})
        # 워커는 원래 사장 지시 메시지를 받아 동일하게 처리
        await self._ops_support_execute(original_message)

    def _active_actor(self) -> tuple:
        """This orchestrator's (uid, is_admin). Phase 2: identity comes from the owning
        UserContext, not a global active account."""
        return self.uid, self.is_admin

    def _spawn_ops_support_worker(self, cycle_id: Optional[int] = None, manual_directive: Optional[str] = None,
                                  role: str = "ops_support"):
        """Spawn the standalone 운용지원실장 worker — fire-and-forget.

        사장 지시 2026-05-20: 코드 자가수정·서버 재시작·산하 팀장 위임 폐지. 워커는 별도
        프로세스에서 진단 + **프로필 한정 전략 파라미터(param_overrides)** 조정·제안만
        수행한다. ADMIN·일반 유저 모두 spawn 되며 적용은 각자 프로필로 분리되고,
        안전성은 워커가 소스 코드·서버를 절대 건드리지 않음으로써 보장된다."""
        # 회귀 가드 2026-06-09: pytest 실행 중에는 절대 실제 워커 subprocess 를 띄우지 않는다.
        # (테스트가 실제 uid 오케스트레이터로 ceo_directive 를 호출하면 라이브 LLM·param_overrides·
        #  ops_history·대시보드 broadcast 까지 오염시킨 사고가 있었다 — defense in depth.)
        import os as _os
        if _os.environ.get("PYTEST_CURRENT_TEST"):
            logger.info("ops 워커 spawn 스킵 — pytest 환경(운영 부작용 차단)")
            return
        worker = _Path(__file__).parent / "infra" / "ops_support_worker.py"
        if not worker.exists():
            logger.warning(f"ops_support_worker.py 없음 — 스킵 ({worker})")
            return
        _auid, _admin = self._active_actor()
        # 운용지원실장 피드백 on/off 토글(프로필별) — 활성 계정 토글이 OFF면 spawn 안 함.
        try:
            import runtime as _rt
            if not _rt.ops_feedback_enabled(_auid):
                logger.info(f"운용지원 워커 spawn 스킵 — 피드백 토글 OFF "
                            f"(uid={_auid}, cycle_id={cycle_id})")
                return
        except Exception as _e:
            logger.warning(f"ops_feedback 토글 조회 실패 — 그대로 진행: {_e}")
        cmd = ["python3.11", str(worker), "--role", role]
        if cycle_id is not None:
            cmd += ["--cycle-id", str(int(cycle_id))]
        if manual_directive:
            cmd += ["--manual", manual_directive]
        if _auid is not None:
            cmd += ["--actor-user", str(int(_auid))]
        # 워커는 더 이상 actor-admin 으로 동작을 분기하지 않지만(코드 자가수정 제거),
        # 로깅·호환을 위해 실제 값 전달.
        cmd += ["--actor-admin", "1" if _admin else "0"]
        log_path = _Path(__file__).parent / "data" / "ops_support.spawn.log"
        try:
            f = open(log_path, "a", encoding="utf-8", buffering=1)
            f.write(f"\n=== {datetime.now(KST):%Y-%m-%d %H:%M:%S} spawn (role={role}, cycle_id={cycle_id}, manual={'Y' if manual_directive else 'N'}, actor_uid={_auid}, admin={_admin}) ===\n")
            subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True, cwd=str(_Path(__file__).parent))
            logger.info(f"운용지원 워커 spawn: role={role}, cycle_id={cycle_id}, actor_uid={_auid}, admin={_admin}")
        except Exception as e:
            logger.warning(f"운용지원 워커 spawn 예외: {e}")

    async def _ops_support_execute(self, message: str, role: Optional[str] = None) -> str:
        """@운용지원실장 또는 @투자관리팀장/@경영관리팀장/@재무관리팀장 멘션 처리.
        사장 지시 2026-05-14:
         - 질의형 키워드(이력/히스토리/수정한/변경한/어떤 코드/list)가 보이면 → 즉시 이력 텍스트 응답
         - 그 외 일반 지시 → 키워드 분류 후 해당 role로 워커 spawn (자동 지명 호출).
         - role을 명시적으로 받으면 분류를 건너뛰고 그 role로 spawn (사장이 @팀장 직접 호출한 경우).
        운용지원실장 산하 팀장은 모두 동일 워커를 다른 페르소나(--role)로 실행한다 — 보안 가드는 공유."""
        query_keywords = ("이력", "히스토리", "수정한", "변경한", "어떤 코드", "고친", "고친게",
                          "수정 내역", "변경 내역", "한 게 있", "한게 있", "리스트", "list", "history")
        # 사장 지시 2026-05-20: 운용지원실장은 ADMIN·일반 유저 모두 사용 가능
        # (프로필 한정 파라미터 조정·진단만 — 코드 자가수정 없음). 산하 팀장 라인은 폐지됨.
        _auid, _admin = self._active_actor()

        if any(kw in message for kw in query_keywords):
            from infra import ops_history
            text = ops_history.summary_text(limit=20, only_with_changes=True)
            st = ops_history.stats()
            stats_line = (f"\n\n📊 전체 통계: 워커 실행 {st['runs']}회 / "
                          f"실제 조정 발생 {st['runs_with_changes']}회 / "
                          f"반영된 파라미터 {st['total_applied_changes']}건")
            full = text + stats_line
            await self._emit({"type": "agent_msg", "agent": "운용지원실장", "message": full})
            return full

        # 사장 피드백 2026-05-18: 운용지원실장 피드백 토글이 OFF면 새 지시는 받지 않는다
        # (과거 이력 조회는 위에서 이미 처리되므로 영향 없음 — 끄면 '안내'만).
        try:
            import runtime as _rt
            if not _rt.ops_feedback_enabled(_auid):
                msg = ("⏸ 이 계정의 운용지원실장 피드백이 현재 **꺼짐(OFF)** 상태입니다. "
                       "대시보드의 '운용지원 피드백' 토글을 켜면 지시·자동 사이클 분석이 재개됩니다.")
                await self._emit({"type": "agent_msg", "agent": "운용지원실장", "message": msg})
                return msg
        except Exception:
            pass

        # 팀장 폐지(2026-05-20) — 모든 지시는 운용지원실장 단일 역할로 처리.
        display = "운용지원실장"

        # 사장이 내린 지시 원문도 대시보드 로그에 남긴다.
        await self._emit({"type":"agent_msg","agent":"사장",
                          "message": f"🗣 [사장 → {display}] {message}"})
        self._spawn_ops_support_worker(cycle_id=None, manual_directive=message)
        msg = (f"🛠 {display}: 지시 받았습니다. 다만 소스 코드 직접 수정·배포는 제 권한 밖이라 그 자체는 못 합니다 — "
               f"대신 같은 목적을 **이 프로필 전용 전략 파라미터**(전략·예산·익절/손절 등 적용 가능 항목)로 옮겨 조정안을 잡아 보겠습니다. "
               f"반영되면 다음 로그인 시 활성화되고, '@운용지원실장 이력 보여줘'로 반영 내역을 확인하실 수 있습니다.")
        await self._emit({"type":"agent_msg","agent":display,"message":msg})
        return msg



class _MarketCalendarMixin:
    async def _verify_market_open(self, session: str) -> Optional[bool]:
        """개장 5분 경과 후, KIS 실시세로 '오늘 실제 거래가 있었는지'를 1회 확인한다 (사장 지시 2026-05-24).
        휴일을 하드코딩 목록으로만 추정하지 말고, 실데이터로 한번 검증하라는 취지.
          KR: KOSPI(0001) 지수 일봉의 최신 봉 날짜가 '오늘'이면 개장.
          US: 유동성 큰 티커(AAPL) 일봉의 최신 봉 날짜가 'US 거래일'이면 개장.
        반환: True=거래중(개장 확정) / False=당일 데이터 없음(휴장 추정) / None=시기상조·확인불가(보류).
        주의(거래 누락 방지): True(개장)만 self 캐시에 사이클간 보존한다 — False/None 은 보존하지 않아
        다음 사이클에 재확인 → KIS 당일 봉이 늦게 채워진 일시적 지연이 자기 교정된다.
        (단 결과는 프로세스 전역 _VERIFIED_TRADED/_VERIFIED_CLOSED 에 기록해 동기 게이트가 참조한다.)"""
        now = _now_kst()
        is_kr = is_kr_session(session)
        is_us = session == "US_TRADING"
        if not (is_kr or is_us):
            return None
        market = "KR" if is_kr else "US"
        # ── 개장 +5분 경과 판정 ──
        if is_kr:
            sh, sm = SCHEDULE["kr_trading"]["start"]
            past_open5 = (now.hour * 60 + now.minute) >= (sh * 60 + sm + 5)
            expect = now.strftime("%Y-%m-%d")
        else:
            # US 정규장 22:30 개장(야간) — 22:35 이후(당일) 또는 자정~05:00(이미 경과).
            past_open5 = (now.hour == 22 and now.minute >= 35) or (now.hour == 23) or (now.hour < 5)
            # US 거래일: KST 저녁(22~24시)=같은 날짜, 자정~새벽(00~05시)=전날.
            expect = now.strftime("%Y-%m-%d") if now.hour >= 22 else (now - timedelta(days=1)).strftime("%Y-%m-%d")
        if not past_open5:
            return None
        ck = f"{market}:{now.strftime('%Y-%m-%d')}"
        if self._mkt_open_verified.get(ck):
            return True
        try:
            if is_kr:
                rows = await self.broker.kr_index_daily("0001", days=5)
            else:
                rows = await self.broker.us_daily_chart("AAPL", days=5)
        except Exception as _e:
            logger.warning(f"[개장확인] {market} 실시세 확인 실패(폴백) — {_e}")
            return None
        if not rows:
            logger.info(f"[개장확인] {market} 실시세 빈 응답 — 확인 보류(폴백)")
            return None
        modkey = f"{market}:{expect}"
        latest = str(rows[-1].get("date", ""))
        if latest == expect:
            self._mkt_open_verified[ck] = True
            _VERIFIED_TRADED.add(modkey); _VERIFIED_CLOSED.discard(modkey)
            logger.info(f"[개장확인] {market} {expect}: 개장 확정 (KIS 최신 봉={latest})")
            return True
        if latest and latest < expect:
            _VERIFIED_CLOSED.add(modkey)
            logger.info(f"[개장확인] {market} {expect}: 당일 봉 없음 → 휴장 확정 (KIS 최신 봉={latest})")
            return False
        logger.info(f"[개장확인] {market} {expect}: 판정 보류 (KIS 최신 봉={latest})")
        return None

    async def _market_closed_today(self, session: str) -> tuple[bool, str]:
        """이 세션의 시장이 오늘 휴장인지 판정 (사장 지시 2026-06-03 — 하드코딩 휴장일 목록 폐지):
          1) 주말 → 무조건 휴장 (KIS 호출 안 함, 자명).
          2) 그 외엔 KIS 실데이터로 '오늘 실제 거래(당일 봉/거래량)가 있는가'만 본다:
             • 당일 봉 확인  → 개장 → 진행.
             • 개장 후에도 당일 봉 없음 → 휴장 확정 → 스킵.
             • 시기상조/일시적 확인불가(개장 직후 5분 등) → 보류 → 진행하되 다음 사이클에 재확인
               (개장 직후 봉이 늦게 채워지는 정상 거래일에 거짓 '휴장'으로 거래를 누락하지 않기 위함).
        반환: (휴장이면 True, 사유 문자열)."""
        now = _now_kst()
        is_kr = is_kr_session(session)
        is_us = session == "US_TRADING"
        if not (is_kr or is_us):
            return False, ""
        # 1) 주말 — 자명한 휴장 (유일한 하드코딩)
        if (is_kr and is_kr_weekend(now)) or (is_us and is_us_weekend(now)):
            return True, f"주말 — 사이클 스킵 ({now.strftime('%Y-%m-%d %a')})"
        # 2) KIS 실데이터(당일 거래량/봉)로만 휴장 판정
        verified = await self._verify_market_open(session)
        if verified is False:
            mk = "KR" if is_kr else "US"
            return True, f"{mk} 당일 거래량 없음 — 휴장 확정(KIS 당일 봉 없음), 사이클 스킵"
        # True(개장) 또는 None(보류) → 진행
        return False, ""

    def _market_closed_for(self, is_kr: bool) -> bool:
        """해당 종목 시장이 (지정가 미체결분이 더는 체결될 수 없는) 마감 상태인가.
        KR: NXT 프리/애프터마켓·정규장·마감리뷰 포함 KR 세션이 아니면 마감. US: US_TRADING 이 아니면 마감."""
        sess = get_current_session()
        if is_kr:
            return not is_kr_session(sess)
        return sess != "US_TRADING"

    async def _maybe_market_close_alert(self, prev_session: str, cur_session: str):
        """장 마감 전환 시 1회 — 당일·누적 수익률을 모바일 알림(type=market_close)으로 보고
        (사장 지시 2026-05-21, 모바일 알림 4종 중 ④). KR_TRADING→그외=한국장 마감,
        US_TRADING→그외=미국장 마감. 전환 순간에만 호출되므로 중복 발송이 없다."""
        kr_closed = prev_session == "KR_TRADING" and cur_session != "KR_TRADING"
        us_closed = prev_session == "US_TRADING" and cur_session != "US_TRADING"
        if not (kr_closed or us_closed):
            return
        # 사장 지시 2026-05-30(비운영일 알림 버그): get_current_session()은 시간대만 보므로 주말/휴장일에도
        # KR_TRADING/US_TRADING 으로 잡혀 가짜 '장 마감' 전환이 생긴다. 그 시장이 실제 거래일이었을
        # 때만 알림을 보낸다. 마감 직전 시각(−30분)을 기준으로 보면 US 자정 경계도 정확히 판정된다.
        ref = _now_kst() - timedelta(minutes=30)
        if kr_closed and (is_kr_weekend(ref) or _market_day_verified_closed("KR", ref)):
            return
        if us_closed and (is_us_weekend(ref) or _market_day_verified_closed("US", ref)):
            return
        mkt = "한국" if kr_closed else "미국"
        try:
            # uid 누락 시 거래내역이 빈 배열로 로드돼 당일·누적이 항상 0% 였다(+0% 버그). uid 전달.
            k = performance_kpis(self.equity_path, uid=self.uid)
        except Exception:
            k = {}
        def _p(v): return f"{v:+.2f}%" if isinstance(v, (int, float)) else "-"
        def _w(v): return f"{v:+,.0f}원" if isinstance(v, (int, float)) else "-"
        await self._emit({"type": "market_close", "agent": "프롭트레이딩팀장", "market": mkt,
            "message": (f"🔔 {mkt} 장 마감 — 당일 {_p(k.get('today_pct'))} ({_w(k.get('today_pnl'))}) · "
                        f"누적 {_p(k.get('cumulative_pct'))} ({_w(k.get('cumulative_pnl'))})"),
            "today_pct": k.get("today_pct"), "today_pnl": k.get("today_pnl"),
            "cumulative_pct": k.get("cumulative_pct"), "cumulative_pnl": k.get("cumulative_pnl"),
            "trades_total": self._trades_executed})



class _ExecutionMixin:
    async def _dilution_skip(self, code: str):
        """ENABLE_DILUTION_GATE(기본 OFF) 시 매수 직전 DART 희석 공시(전환사채/유증/신주인수권)를
        점검해 high 심각도면 보류 사유 반환, 아니면 None. OFF면 즉시 None(DART 호출 없음·핫패스 비용 0).
        2026-06-15 ROI#4 — 자율 에이전트 없이 결정론 게이트(거버넌스 방침)."""
        if not runtime.get("ENABLE_DILUTION_GATE", uid=self.uid):
            return None
        try:
            from tools.dart_disclosure import search_disclosures
            from tools.dilution import detect_dilution
            from tools.market_data import canonical_name
            name = canonical_name(code) or code
            end = datetime.now(KST).strftime("%Y%m%d")
            bgn = (datetime.now(KST) - timedelta(days=45)).strftime("%Y%m%d")
            res = await search_disclosures(corp_name=name, bgn_de=bgn, end_de=end)
            if not getattr(res, "ok", False):
                return None
            d = detect_dilution(getattr(res, "text", ""))
            if d["dilutive"] and d["severity"] == "high":
                return f"{name}({code}) 희석 공시 감지({', '.join(d['kinds'][:3])}) — 매수 보류(희석 게이트)"
        except Exception as _de:
            logger.warning(f"[희석게이트] {code} 점검 실패(통과): {_de}")
        return None

    async def _build_orders(self, target_codes: List[str], candidate_codes: List[str], quant_report: str, news_report: str,
                            holdings: List[Dict], sell_directives: Optional[Dict[str, str]] = None,
                            market_open: bool = False,
                            entry_dirs: Optional[Dict[str, Dict[str, Any]]] = None,
                            sell_prices: Optional[Dict[str, Dict[str, Any]]] = None,
                            quant_scores: Optional[Dict[str, int]] = None,
                            quant_sigmas: Optional[Dict[str, float]] = None,
                            macro_stock_pct: Optional[float] = None,
                            systematic_policy: Optional[Dict] = None):
        """Assemble OrderDraft JSON in Python (no LLM):
          1) SELL from current holdings — if 사후관리실장 gave a `sell_directives` map ({code: '전량'|'절반'|'보유'|'N주'})
             it is authoritative for the holdings it addresses; holdings it didn't mention fall back to the auto rules
             (take profit ≥TAKE_PROFIT_PCT / stop loss ≤-STOP_LOSS_PCT / trim over CONSERVATIVE_STOCK_RATIO).
          2) BUY targets (2패스 최종종목), sized to available cash: qty = floor( min(cash·PER_ORDER_BUDGET_RATIO,
             total·CONSERVATIVE_STOCK_RATIO) / price ), capped by MAX_ORDER_QTY (if >0). Skip names already held.
          3) If no approved target is affordable, retain cash.
        US targets get a conservative qty=1 (needs USD cash). Returns (order_obj, price_map, buying_power)."""
        sell_directives = sell_directives or {}
        # live strategy params
        PER_ORDER_BUDGET_RATIO = runtime.get("PER_ORDER_BUDGET_RATIO", uid=self.uid); CONSERVATIVE_STOCK_RATIO = runtime.get("CONSERVATIVE_STOCK_RATIO", uid=self.uid)
        PER_ORDER_BUDGET_OVERSHOOT = float(runtime.get("PER_ORDER_BUDGET_OVERSHOOT", uid=self.uid) or 1.20)
        MAX_ORDER_QTY = runtime.get("MAX_ORDER_QTY", uid=self.uid); MAX_TRADES_PER_CYCLE = runtime.get("MAX_TRADES_PER_CYCLE", uid=self.uid)
        ENABLE_SELL_REBALANCE = runtime.get("ENABLE_SELL_REBALANCE", uid=self.uid); TAKE_PROFIT_PCT = runtime.get("TAKE_PROFIT_PCT", uid=self.uid)
        if systematic_policy is not None:
            # Let the weekly rotation/aftercare manage winners; hard stops stay.
            TAKE_PROFIT_PCT = 0.0
        STOP_LOSS_PCT = runtime.get("STOP_LOSS_PCT", uid=self.uid); TRIM_OVER_RATIO = runtime.get("TRIM_OVER_RATIO", uid=self.uid)
        ENABLE_CHEAP_FALLBACK = runtime.get("ENABLE_CHEAP_FALLBACK", uid=self.uid); ALLOW_US_STOCKS = runtime.get("ALLOW_US_STOCKS", uid=self.uid)
        ALLOW_DERIVATIVES = runtime.get("ALLOW_DERIVATIVES", uid=self.uid)
        MAX_CYCLE_BUDGET_RATIO = float(runtime.get("MAX_CYCLE_BUDGET_RATIO", uid=self.uid) or 0.4) * float(getattr(self, "_regime_budget_mult", 1.0))  # 리스크검증과 동일한 사이클 예산 한도 + 레짐 틸트(사장 지시 2026-07-21)
        report = (quant_report or "") + "\n" + (news_report or "")
        snap = await self.broker.kr_account_snapshot()
        bp = snap["buying_power"]
        # 사장 보고 2026-07-29(261220 중복매도): `holdings or snap[...]` 는 **빈 리스트를 falsy 로**
        # 보고 KIS 전체 보유로 갈아끼웠다. 보유가 전부 슬리브 ETF(채권·원자재)인 계정(uid2)은
        # 주식 매도 트랙 입력(stock_holdings)이 정확히 `[]` 라, 이 폴백이 슬리브 ETF를 되살려
        # 슬리브 트랙과 **같은 종목에 매도 주문을 2건** 만들었다(45주 → 22+22 → 1주 잔여 → 이후
        # 매 사이클 '모의투자 잔고내역이 없습니다' 실패). None(미주입)일 때만 폴백한다.
        holdings = (snap.get("holdings") or []) if holdings is None else holdings
        # 사장 지시 2026-05-21: 자산곡선은 KR+US 통합 총평가로 기록한다(주문 사이징은 KR 기준 bp 유지).
        # 사장 지시 2026-05-24: 실제 정규장 세션(요일·휴장 반영)에만 평가금액 추이를 기록한다(장외/주말/휴장 제외).
        if is_market_session_now():
            try:
                _pf = await self.broker.portfolio_holdings()
                record_equity(self.equity_path, _pf.get("buying_power") or bp, "cycle", holdings=_pf.get("holdings") or holdings)
            except Exception:
                record_equity(self.equity_path, bp, "cycle", holdings=holdings)
        cash = float(bp.get("cash", 0.0) or 0.0)
        total = float(bp.get("total_eval", 0.0) or 0.0) or cash
        # 사장 지시 2026-06-15: 매크로 주식목표 > 현재 주식비중(여력)이면 ops 예산 컷이 목표
        # 수렴을 과도하게 늦추지 않도록 per-order·per-cycle 예산 비율에 최소 플로어 적용.
        _stock_eval = sum(float(h.get("qty") or 0) * float(h.get("cur_price") or 0)
                          for h in (holdings or []) if not sleeve_for_code(str(h.get("code", "")).strip()))
        _cur_stock_ratio = (_stock_eval / total) if total > 0 else 0.0
        PER_ORDER_BUDGET_RATIO, MAX_CYCLE_BUDGET_RATIO = apply_macro_deploy_floor(
            PER_ORDER_BUDGET_RATIO, MAX_CYCLE_BUDGET_RATIO,
            macro_target_pct=macro_stock_pct, current_stock_ratio=_cur_stock_ratio,
            params={k: runtime.get(k, uid=self.uid) for k in (
                "MACRO_DEPLOY_FLOOR_ENABLED", "PER_ORDER_BUDGET_FLOOR_RATIO", "MAX_CYCLE_BUDGET_FLOOR_RATIO")})
        # 사장 지시 2026-05-14: 1주 예산은 **총평가액** 기준 (실제 spend는 cash로 캡).
        target_per_order = total * PER_ORDER_BUDGET_RATIO if total > 0 else 0.0
        # 사장 지시 2026-05-14: 개장 사이클은 한도 해제 — 100건 뉴스 정보를 적극 활용해 큰 포지션 가능
        if market_open:
            target_per_order = total * min(1.0, PER_ORDER_BUDGET_RATIO * 5.0)  # 5배 확대 (균형형이면 50%까지)
        per_order_budget = min(target_per_order, cash) if cash > 0 else 0.0
        # 단일 종목 비중 한도는 리스크관리실장 검증(total*CONSERVATIVE_STOCK_RATIO)과 동일하게 유지한다.
        # 버그 2026-05-22: 개장 사이클에서만 이 한도를 60%로 풀어 큰 주문을 만들었으나, 검증부는
        # 완화를 몰라 25%로 반려 → 고가주가 매번 반려됐다. 사이징을 검증과 일치시켜 통과시킨다.
        per_stock_cap = total * CONSERVATIVE_STOCK_RATIO if total > 0 else 0.0
        cycle_budget = cash * MAX_CYCLE_BUDGET_RATIO if cash > 0 else 0.0
        spent_krw = 0.0  # 이번 사이클 매수 누적(원화 환산) — 검증부 사이클 예산 한도를 사이징에 선반영
        held_codes = {str(h.get("code","")).strip() for h in holdings}
        # 사장 지시 2026-05-22: 사이클 예산을 '매수 대상 종목 수'로 균등 배분해 분산 매수한다.
        # (한 종목이 예산을 독식해 다음 종목이 '사이클 누적 매수예산 초과'로 반려되던 케이스 방지 —
        #  QUBT 49주 승인 후 IBM 반려처럼.)
        _n_buy = max(1, len({str(c).strip() for c in (target_codes or [])
                             if str(c).strip() and str(c).strip() not in held_codes}))
        per_name_budget = (cycle_budget / _n_buy) if cycle_budget > 0 else float("inf")
        # 사장 지시 2026-06-04 ②: 리스크기반 사이징 — 점수↑·변동성↓ 종목에 per-name 예산을 기울인다(KR/US 공통).
        # equal 모드/strength=0 이면 전원 1.0(기존 균등분배). per_stock_cap·cycle_remaining 하드 한도는 그대로 위에 적용.
        from tools.position_sizing import compute_sizing_weights
        _buy_names = [str(c).strip() for c in (target_codes or [])
                      if str(c).strip() and str(c).strip() not in held_codes]
        _sizing_w = compute_sizing_weights(
            _buy_names, quant_scores or {}, quant_sigmas or {},
            mode=str(runtime.get("POSITION_SIZING_MODE", uid=self.uid) or "equal"),
            strength=float(runtime.get("SIZING_TILT_STRENGTH", uid=self.uid) or 0.0),
            max_tilt=float(runtime.get("SIZING_MAX_TILT", uid=self.uid) or 2.0))
        orders: List[Dict] = []
        price_map: Dict[str, float] = {}
        notes: List[str] = []

        # ── 1) SELL — 사후관리실장 매도결정 우선, 미언급 종목은 자동 규칙 (KR·US 모두) ──
        # 선행-후행 매도(사장 지시 2026-07-21) — 선행주가 최근 30분+ 시장대비 급락한 보유 후행주(KR)를
        # 절반 부분매도. 사후관리실장이 이미 결정한(sell_directives에 있는) 종목은 그 판단 우선(미언급만).
        if bool(runtime.get("ENABLE_LEADLAG_SELL", uid=self.uid)):
            try:
                from tools import leadlag as _ll_sell
                _ll_thr = float(runtime.get("LEADLAG_MIN_BUY_SIGNAL", uid=self.uid) or 0.5)
                for _h in (holdings or []):
                    _hc = str(_h.get("code") or "").strip().zfill(6)
                    if not _is_kr_code(_hc) or _hc in sell_directives:
                        continue
                    _s = _ll_sell.leadlag_signal(_hc)
                    if _s <= -_ll_thr:
                        sell_directives[_hc] = "절반"
                        logger.info(f"[선행-후행 매도] {_hc} 신호 {_s:.2f} ≤ -{_ll_thr:.2f} → 절반 부분매도 지시")
            except Exception as _lse:
                logger.warning(f"선행-후행 매도 판정 실패(무시): {_lse}")
        # 트레일링 익절(2026-06-18): 종목별 고점 영속 로드 → _assemble_sell_orders 가 갱신 → 저장.
        _TRAILING_PCT = float(runtime.get("TRAILING_TAKE_PROFIT_PCT", uid=self.uid) or 0.0)
        _peaks = _load_trailing_peaks(self.uid) if _TRAILING_PCT > 0 else None
        if ENABLE_SELL_REBALANCE or sell_directives:
            _sell_orders, _sell_px = _assemble_sell_orders(
                holdings, sell_directives, enable_rebalance=ENABLE_SELL_REBALANCE,
                take_profit_pct=TAKE_PROFIT_PCT, stop_loss_pct=STOP_LOSS_PCT,
                trim_over_ratio=TRIM_OVER_RATIO, conservative_ratio=CONSERVATIVE_STOCK_RATIO,
                per_stock_cap=per_stock_cap, total=total, sell_prices=sell_prices,
                trailing_pct=_TRAILING_PCT, peaks=_peaks)
            orders.extend(_sell_orders)
            price_map.update(_sell_px)
        if _peaks is not None:
            _save_trailing_peaks(self.uid, _peaks, holdings)

        # ── 2) BUY targets ──────────────────────────────────────────────
        affordable_buy_found = False
        # 사장 지시 2026-06-17: US 매수가능액은 KRW예수금÷환율 합성이 아니라 KIS 실제 USD
        # 주문가능액(us_buying_power) 기준. 사이클당 1회 조회 후 USD 소진을 추적해 phantom
        # 매수(USD 0 계좌에 1주 초안→KIS '주문가능금액 초과' 거부)를 사이징 단계에서 차단.
        _us_bp_usd = None; _us_bp_ok = False; _us_spent_usd = 0.0
        # 회로차단기: 세션 바뀌면 스트릭 리셋(T+2 결제로 USD 생길 수 있음 → 새 세션 재시도).
        from config import US_BUY_FAIL_STREAK_LIMIT as _US_FAIL_LIMIT
        _cur_sess = get_current_session()
        if getattr(self, "_us_circuit_session", None) != _cur_sess:
            self._us_circuit_session = _cur_sess
            self._us_buy_fail_streak = 0
        _us_circuit_open = _us_buy_circuit_open(getattr(self, "_us_buy_fail_streak", 0), _US_FAIL_LIMIT)
        for code in (target_codes or [])[:8]:
            code = str(code).strip()
            if not code:
                continue
            is_kr = _is_kr_code(code)
            if not is_kr and not ALLOW_US_STOCKS:
                notes.append(f"{code}: 해외주식 비활성(ALLOW_US_STOCKS=False) → 제외"); continue
            ctx = _quant_ctx_for(report, code)
            if is_kr:
                if code in held_codes:
                    notes.append(f"{code}: 이미 보유 → 신규 매수 생략(분산)"); continue
                _dil = await self._dilution_skip(code)   # ROI#4 희석 게이트(기본 OFF)
                if _dil:
                    notes.append(_dil)
                    await self._emit({"type": "agent_msg", "agent": "리스크관리실장", "message": f"🚫 {_dil}"})
                    continue
                price = await self.broker.kr_last_price(code)
                await asyncio.sleep(0.25)  # ease KIS TPS
                price_map[code] = price
                if price <= 0 or per_order_budget <= 0:
                    notes.append(f"{code}: 사이즈 산정 불가(가격={price:,.0f}원, 총평가={total:,.0f}원·예수금={cash:,.0f}원) → 제외"); continue
                _w = _sizing_w.get(code, 1.0)
                qty = _affordable_buy_qty(
                    price, per_order_budget=min(per_order_budget, per_name_budget * _w),
                    per_stock_cap=(per_stock_cap if per_stock_cap > 0 else float("inf")),
                    cycle_remaining=max(0.0, cycle_budget - spent_krw))
                if MAX_ORDER_QTY and MAX_ORDER_QTY > 0:
                    qty = min(qty, MAX_ORDER_QTY)
                if qty < 1:
                    # 사장 결정 2026-05-16/2026-06-16: per-order 비율 예산과 무관하게 1주가 예수금 이내면
                    # 1주 허용하되, '사이클 매수예산(잔여)'을 초과하면 제외 — 사이징과 리스크 가드레일을
                    # 같은 기준으로 일치시켜 고가주 데드존(사이징 통과→가드레일 반려)·모순 메시지를 막는다.
                    _cyc_rem = max(0.0, cycle_budget - spent_krw)
                    if _affordable_one_share(price, cash, total, cycle_remaining=_cyc_rem) and not (MAX_ORDER_QTY and 0 < MAX_ORDER_QTY < 1):
                        qty = 1
                        notes.append(f"{code}: 1주 {price:,.0f}원 — 예수금 {cash:,.0f}원·사이클 잔여예산 {_cyc_rem:,.0f}원 이내 → 1주 매수")
                    else:
                        notes.append(f"{code}: 1주 {price:,.0f}원 — 예수금 {cash:,.0f}원/사이클 잔여예산 {_cyc_rem:,.0f}원 {_one_share_exclude_reason(price, cash, _cyc_rem)} → 제외"); continue
                affordable_buy_found = True
                spent_krw += qty * price
                # 사장 피드백 2026-05-15 (#4): 계량분석팀장이 지정한 진입가 directive 첨부 (시장가 default)
                _ed = (entry_dirs or {}).get(code, {"mode": "market", "limit_price": None, "watch_pct": None, "raw": ""})
                orders.append({"ticker": code, "side": "buy", "qty": qty, "price_type": "market", "market": "KR",
                               "entry_mode": _ed.get("mode"), "entry_limit": _ed.get("limit_price"),
                               "entry_watch_pct": _ed.get("watch_pct"), "entry_raw": _ed.get("raw"),
                               "reason": f"주식운용실장 지정 · {qty}주(≈{qty*price:,.0f}원, 총평가 {PER_ORDER_BUDGET_RATIO*100:.0f}%·종목 {CONSERVATIVE_STOCK_RATIO*100:.0f}% 한도 내) · 진입가:{_ed.get('raw') or '시장가'}{' [관망: 다음 사이클 재평가]' if _ed.get('mode') == 'watch' else ''}"})
            else:
                # US stock: 사장 피드백 2026-05-15 (#14): SOUN $8인데 1주만 산 버그 — 예산 내에서 가능한 만큼 매수.
                # 환율 1,500원/$ 가정으로 KRW 예산을 USD로 환산 후 정수 주수 산정.
                tk = code.upper()
                if _us_circuit_open:   # 회로차단기 개방 — 이번 세션 US 매수 보류(매도로 USD 확보 시 자동 해제)
                    notes.append(f"{tk}: US 매수 회로차단 — 연속 USD부족 실패 {self._us_buy_fail_streak}회(임계 {_US_FAIL_LIMIT}) → 이번 세션 US 매수 보류(매도로 USD 확보 시 자동 해제)")
                    continue
                us_px = await self.broker.us_last_price(tk)
                price_map[tk] = us_px
                await asyncio.sleep(0.25)
                if us_px <= 0:
                    notes.append(f"{tk}: 해외 시세 조회 실패(거래소 미확인) → 제외"); continue
                if us_px < 0.01:
                    notes.append(f"{tk}: 가격 ${us_px:.4f} — KIS 온라인 주문 최소단위($0.01) 미만 → 주문불가, 제외"); continue
                _krw_per_usd = get_usdkrw(USDKRW_FALLBACK)  # 사장 지시 2026-05-22: 5분 크롤 라이브 환율(폴백)
                _budget_usd = per_order_budget / _krw_per_usd  # 표시용(주문당 예산)
                _w = _sizing_w.get(tk, 1.0)
                # 실제 USD 주문가능액 — 사이클당 1회 조회(조회 실패 시 KRW÷환율 폴백, 클램프가 최종 방어)
                if _us_bp_usd is None:
                    try:
                        _pb = await self.broker.us_buying_power(tk, us_px, None)
                        _us_bp_ok = bool(_pb.get("ok"))
                        _us_bp_usd = float(_pb.get("usd") or 0.0) if _us_bp_ok else 0.0
                    except Exception:
                        _us_bp_ok = False; _us_bp_usd = 0.0
                _avail_usd = max(0.0, _us_buy_usd_basis(_us_bp_ok, _us_bp_usd,
                                                        krw_cash=cash, krw_per_usd=_krw_per_usd) - _us_spent_usd)
                qty_us = _affordable_buy_qty(
                    us_px, per_order_budget=min(min(per_order_budget, per_name_budget * _w) / _krw_per_usd, _avail_usd),
                    per_stock_cap=((per_stock_cap / _krw_per_usd) if per_stock_cap > 0 else float("inf")),
                    cycle_remaining=max(0.0, cycle_budget - spent_krw) / _krw_per_usd)
                if MAX_ORDER_QTY and MAX_ORDER_QTY > 0:
                    qty_us = min(qty_us, MAX_ORDER_QTY)
                if qty_us < 1:
                    # 사장 결정 2026-05-16/2026-06-16: 예수금(USD 환산) 기준 1주 허용하되 사이클 잔여예산
                    # 초과면 제외 — KR 과 동일 기준(고가주 데드존 방지). 사이클예산은 KRW 라 USD 로 환산.
                    # 2026-06-17: 예수금 기준은 KIS 실제 USD 주문가능액(_avail_usd) — KRW 합성 phantom 차단.
                    _cash_usd = _avail_usd
                    _total_usd = total / _krw_per_usd
                    _cyc_rem_usd = max(0.0, cycle_budget - spent_krw) / _krw_per_usd
                    if _affordable_one_share(us_px, _cash_usd, _total_usd, cycle_remaining=_cyc_rem_usd):
                        qty_us = 1
                        notes.append(f"{tk}: 1주 ${us_px:.2f} — 예수금 ${_cash_usd:,.2f}·사이클 잔여예산 ${_cyc_rem_usd:,.2f} 이내 → 1주 매수")
                    else:
                        notes.append(f"{tk}: 1주 ${us_px:.2f} — 예수금 ${_cash_usd:,.2f}/사이클 잔여예산 ${_cyc_rem_usd:,.2f} {_one_share_exclude_reason(us_px, _cash_usd, _cyc_rem_usd)} → 제외"); continue
                est_krw = us_px * qty_us * _krw_per_usd
                affordable_buy_found = True
                spent_krw += est_krw
                _us_spent_usd += us_px * qty_us   # 사이클 내 후속 US 종목은 잔여 USD 로 제한
                _ed = (entry_dirs or {}).get(tk, {"mode": "market", "limit_price": None, "watch_pct": None, "raw": ""})
                orders.append({"ticker": tk, "side": "buy", "qty": qty_us, "price_type": "market", "market": "US",
                               "entry_mode": _ed.get("mode"), "entry_limit": _ed.get("limit_price"),
                               "entry_watch_pct": _ed.get("watch_pct"), "entry_raw": _ed.get("raw"),
                               "reason": f"주식운용실장 지정 해외종목 · {qty_us}주(≈${us_px*qty_us:,.2f} / ≈{est_krw:,.0f}원, 예산 ${_budget_usd:.2f}) · 진입가:{_ed.get('raw') or '시장가'}{' [관망: 다음 사이클 재평가]' if _ed.get('mode') == 'watch' else ''}"})

        # Keep cash when approved names are unaffordable. No unreviewed substitute.
        if target_codes and not affordable_buy_found:
            notes.append("최종 심의 종목을 예산 내에서 살 수 없어 현금을 유지합니다")

        # ── 4) Session-aware market validation (item 11) ──────────────────────
        # KR orders should only go through during KR trading hours; US during US hours.
        # This prevents errors like "장운영일자가 주문일과 상이합니다".
        session = get_current_session()
        filtered_orders = []
        for o in orders:
            mkt = o.get("market", "KR")
            if mkt == "KR" and not is_kr_tradable(session):
                notes.append(f"{o['ticker']}: 현재 KR 장외시간({session}) → KR 주문 제외 (장운영일자 불일치 방지)")
                continue
            if mkt == "US" and session != "US_TRADING":
                notes.append(f"{o['ticker']}: 현재 US 장외시간({session}) → US 주문 제외")
                continue
            filtered_orders.append(o)
        orders = filtered_orders

        # sells first (risk-reducing), then buys; clamp to MAX_TRADES_PER_CYCLE
        orders.sort(key=lambda o: 0 if o["side"] == "sell" else 1)
        if not ALLOW_DERIVATIVES:
            orders = [o for o in orders if o.get("market") not in ("BOND", "FUTURES")]
        # 사장 지시 2026-06-04: 1회 주문 한도 초과 매수는 '반려'가 아니라 한도까지 clamp (주문 드롭 금지).
        # 저가주에 예산이 몰려 산출된 큰 수량(052900 10,227주)이 리스크 게이트에서 전량 반려되던 버그 방지.
        orders = self._clamp_orders_to_max_qty(orders)
        # 사장 지시 2026-06-01: 매수/매도 직전 KIS 권위 '주문가능' 조회로 수량 clamp
        # (국내 증거금·해외 USD한도·매도가능 반영). 조회 실패/글리치 시 현행 유지(주문 절대 드롭 금지).
        orders = await self._clamp_orders_to_psbl(orders, price_map)
        return {"orders": orders, "sizing_notes": notes}, price_map, bp

    def _clamp_orders_to_max_qty(self, orders: List[Dict]) -> List[Dict]:
        """1회 주문 수량 상한으로 clamp — 매수 한정(반려가 아니라 깎아서 통과시킨다).

        상한 = MAX_ORDER_QTY(런타임·>0) 우선, 없으면 config.HARD_MAX_ORDER_QTY(=1000).
        저가주에 사이클 예산이 몰리면 budget/단가 = 큰 수량이 산출돼 리스크 게이트의 '1회 한도 초과'에
        걸려 매수가 통째로 반려(매수 증발)되던 버그(052900) 방지. 매도는 위험회피(전량청산)이고
        보유수량으로 이미 한정되므로 적용하지 않는다(전량 매도가 한도에 막히면 안 됨)."""
        from config import HARD_MAX_ORDER_QTY
        _mq = runtime.get("MAX_ORDER_QTY", uid=self.uid)
        ceiling = int(_mq) if (_mq and _mq > 0) else int(HARD_MAX_ORDER_QTY)
        if ceiling <= 0:
            return orders
        for o in orders:
            if (o.get("side") or "buy") != "buy":
                continue
            try:
                qty = int(float(o.get("qty") or 0))
            except (TypeError, ValueError):
                continue
            if qty > ceiling:
                o["qty"] = ceiling
                o["maxqty_clamped"] = True
                _r = str(o.get("reason") or "")
                o["reason"] = f"{_r} [⚠ 1회 한도 {ceiling}주로 clamp — 원수량 {qty}주, 예산 일부만 집행]"
        return orders

    async def _clamp_orders_to_psbl(self, orders: List[Dict], price_map: Dict[str, float]) -> List[Dict]:
        """매수/매도 직전 KIS 권위 주문가능 조회로 수량을 clamp.
          - 국내 매수: kr_psbl_order(TTTC8908R, ORD_DVSN='01') nrcvb_buy_qty 상한
          - 해외 매수: us_buying_power(TTTS3007R) max_ord_psbl_qty 상한(USD 단위)
          - 국내 매도: kr_psbl_sell_qty(TTTC8408R) ord_psbl_qty 상한
        조회 실패(ok=False/None)는 현행 수량 유지 — 주문 절대 드롭 금지(사장 원칙). 매도가능 0 글리치는
        1회 재조회 후에도 0이면 전량 유지. 매수가 0 으로 clamp되면(살 수 없음) 0주 주문 금지로 제거."""
        def _iq(v):
            try:
                return int(float(v or 0))
            except (TypeError, ValueError):
                return 0
        for o in orders:
            try:
                code = str(o.get("ticker") or "").strip()
                side = o.get("side")
                mkt = o.get("market", "KR")
                qty = _iq(o.get("qty"))
                if qty <= 0 or not code:
                    continue
                if side == "buy" and mkt == "US":
                    pb = await self.broker.us_buying_power(code, price_map.get(code) or price_map.get(code.upper()) or 0.0, None)
                    if pb.get("ok") and pb.get("qty") is not None and qty > _iq(pb.get("qty")):
                        o["qty"] = _iq(pb.get("qty")); o["psbl_clamped"] = True
                elif side == "buy":
                    pb = await self.broker.kr_psbl_order(code, price_map.get(code) or 0.0)
                    if pb.get("ok") and pb.get("buy_qty") is not None and qty > _iq(pb.get("buy_qty")):
                        o["qty"] = _iq(pb.get("buy_qty")); o["psbl_clamped"] = True
                elif side == "sell" and mkt == "KR":
                    sq = await self.broker.kr_psbl_sell_qty(code)
                    if sq is not None and _iq(sq) < qty:
                        if _iq(sq) <= 0:
                            sq2 = await self.broker.kr_psbl_sell_qty(code)   # 글리치 의심 → 1회 재조회
                            if sq2 is not None and _iq(sq2) > 0:
                                o["qty"] = min(qty, _iq(sq2)); o["psbl_clamped"] = True
                            # else: 현행 유지(매도 드롭 금지)
                        else:
                            o["qty"] = _iq(sq); o["psbl_clamped"] = True
            except Exception:
                continue   # 조회 실패는 현행 유지(주문 드롭 금지)
        # 매수가 0주로 clamp된 건만 제거(0주 주문 금지). 매도는 유지.
        return [o for o in orders if not (o.get("side") == "buy" and _iq(o.get("qty")) < 1)]

    async def _dart_risk_review(self, buy_orders: List[Dict], per_dart: Dict[str, str]) -> tuple:
        """리스크관리실장 2차 재심 — 매수 주문 종목의 DART 공시를 읽고 리스크성 공시가 있으면 반려.

        사장 지시 2026-05-14: 더 이상 자체적으로 broadcast/log 하지 않음 — 호출처(`_run_analysis_cycle`)에서
        1차 결과와 묶어 단일 메시지로 출력. Returns (vetoed_set, llm_response_text). Fail-open."""
        relevant = {}
        for r in (buy_orders or []):
            tk = str(r.get("ticker", "")).strip()
            d = per_dart.get(tk) or per_dart.get(tk.upper())
            if d:
                relevant[tk] = d
        if not relevant:
            return set(), "DART 재심: 관련 공시 없음 — 1차 결과 유지"
        tickers = [str(r.get("ticker", "")).strip() for r in buy_orders]
        digest = "\n\n".join(f"[{k}]\n{(v or '')[:2500]}" for k, v in relevant.items())
        # ITEM2b: QUERY_FAILED 종목 목록 추출 — 해당 종목은 특별 주의 지시
        _qf_tickers = []
        for r in (buy_orders or []):
            tk_qf = str(r.get("ticker", "")).strip()
            d_qf = per_dart.get(tk_qf) or per_dart.get(tk_qf.upper()) or ""
            if "시스템 리스크" in d_qf or "QUERY_FAILED" in d_qf:
                _qf_tickers.append(tk_qf)
        _qf_warning = ""
        if _qf_tickers:
            _qf_warning = (
                f"\n\n⚠️ [DART 조회 실패 종목 — 시스템 리스크 주의]: {', '.join(_qf_tickers)}\n"
                f"위 종목은 API 오류·키 없음·네트워크 장애로 DART 데이터를 가져오지 못했습니다. "
                f"'재무 미확보'로 처리하되, 조회 실패 자체를 **시스템 리스크 경고**로 간주하십시오. "
                f"다른 명백한 승인 근거가 없으면 보수적으로 반려를 검토하십시오.\n"
            )
        try:
            resp = await self.risk_guard.think(
                f"1차 결정론 검증을 통과한 매수 주문 종목: {', '.join(tickers)}\n\n"
                f"각 종목 — 최근 ~90일 DART 공시 + **가장 최근 가용 분기/반기/연간 요약재무**:\n{digest}"
                f"{_qf_warning}\n\n"
                f"① 공시 리스크 신호(관리종목/거래정지/상장폐지심사/횡령·배임/회계감리/불성실공시/감자/대규모 유상증자/주요 소송·제재 등) "
                f"② 재무 적신호(자본잠식·부채비율 과다·연속 적자·매출 급감) 를 종합해, 해당하는 종목은 반려하십시오. "
                f"⚠️ '부채>자산' 판정은 **재무상태표 검증 통과(✅ 재무상태표 검증: ...)** 표기가 있을 때만 하십시오. "
                f"'⚠️ 재무상태표 내적 불일치' 또는 '시스템 리스크' 경고가 있는 종목의 재무 수치를 근거로 부채>자산·자본잠식을 단정하지 마십시오. "
                f"명백한 악재가 없으면 승인 유지(공시·재무 정보 없음만으로는 반려 금지). "
                f"마지막 줄에 반드시 `최종승인: 코드, 코드` (유지할 종목만; 전부 반려면 `최종승인: 없음`).")
            if not _has_label(resp, "최종승인"):
                return set(), f"[DART 재심 응답]\n{resp}\n\n⚠ '최종승인' 라인 없음 — 1차 결과 유지(fail-open)"
            kept_upper = {c.upper() for c in _extract_codes_after(resp, "최종승인")}
            vetoed = {tk.upper() for tk in relevant if tk.upper() not in kept_upper}
            return vetoed, resp
        except Exception as e:
            logger.warning(f"DART 재심 실패: {e}")
            return set(), f"DART 재심 예외 — 1차 결과 유지: {e}"

    async def _entry_watch_task(self, ticker: str, qty: int, market: str,
                                 watch_pct: float, baseline_holdings: List[Dict],
                                 reason: str = ""):
        """사장 피드백 2026-05-15 (#4): 분봉 진입 타이밍 모니터.
        - 1분 주기로 현재가 폴링, baseline 대비 watch_pct 도달 시 시장가 매수.
        - 최대 3시간(180분) 대기 후 타임아웃 → 시장가 매수 (계량분석팀장 의도대로 일단 진입).
        - 도중에 해당 시장이 마감되면 매수 취소(주문 안 냄)."""
        start = time.time()
        max_wait = 3 * 3600  # 3시간
        poll_interval = 60   # 1분
        is_kr = (market == "KR")
        try:
            initial_px = await (self.broker.kr_last_price(ticker) if is_kr else self.broker.us_last_price(ticker))
            if not initial_px or initial_px <= 0:
                await self._emit({"type":"trade_failed",
                    "message": f"⏰ {ticker} 대기 매수 취소 — 초기가 조회 실패"})
                return
            target_px = initial_px * (1 + watch_pct / 100.0)
            triggered = False
            elapsed_min = 0
            while time.time() - start < max_wait:
                if self._stop_event.is_set():
                    await self._emit({"type":"trade_failed",
                        "message": f"⏰ {ticker} 대기 매수 취소 — 사용자 중지"})
                    return
                # 시장 마감 체크
                sess = get_current_session()
                if (is_kr and not is_kr_tradable(sess)) or (not is_kr and sess != "US_TRADING"):
                    await self._emit({"type":"trade_failed",
                        "message": f"⏰ {ticker} 대기 매수 취소 — 장 마감 ({sess})"})
                    return
                cur = await (self.broker.kr_last_price(ticker) if is_kr else self.broker.us_last_price(ticker))
                if cur and cur > 0:
                    move_pct = (cur / initial_px - 1) * 100
                    # 조건 충족 검사 (음수 pct = 하락 트리거, 양수 pct = 돌파 트리거)
                    if (watch_pct < 0 and cur <= target_px) or (watch_pct > 0 and cur >= target_px):
                        triggered = True
                        break
                    if elapsed_min % 5 == 0:  # 5분마다 진행상황 broadcast
                        await self._emit({"type":"agent_msg","agent":"프롭트레이딩팀장",
                            "message": f"⏱ {ticker} 분봉 모니터 ({elapsed_min}분 경과): 현재가 {cur:,.2f} / 목표 {target_px:,.2f} ({move_pct:+.2f}%)"})
                await asyncio.sleep(poll_interval)
                elapsed_min += 1
            # 시장가 매수 발동 (트리거 or 타임아웃)
            od = OrderDraft(ticker=ticker, side="buy", qty=qty, price_type="market",
                            market=market, reason=f"분봉 대기 매수 ({'트리거' if triggered else '3시간 타임아웃'}) — {reason[:80]}",
                            approved=True)
            # 시장 마감 직전 다시 한 번 체크 (KIS가 거부할 수 있으므로)
            sess = get_current_session()
            if (is_kr and not is_kr_tradable(sess)) or (not is_kr and sess != "US_TRADING"):
                await self._emit({"type":"trade_failed",
                    "message": f"⏰ {ticker} 대기 매수 취소 — 매수 시점에 장 마감"})
                return
            # 주문 직전 보유 스냅샷 — 즉시 체결 확인 + (미확인 시) 폴링 baseline (KR+US)
            try: _pre_kr = await self.broker.kr_holdings()
            except Exception: _pre_kr = list(baseline_holdings or [])
            _pre_us = []
            if not is_kr:
                try: _pre_us = await self.broker._overseas_holdings()
                except Exception: _pre_us = []
            od, _nxt_skip = await self._finalize_kr_order_for_session(od, get_current_session())
            if _nxt_skip:
                await self._emit({"type":"trade_failed", "message": f"⚠️ {_nxt_skip}"})
                return
            res = await self.broker.place_order(od)
            badge = "⏱ 분봉 진입 매수 발동" if triggered else "⌛ 3시간 타임아웃 매수"
            accepted = all(bad not in res for bad in ("실패", "에러", "거부", "예외", "REJECT"))
            # 사장 지시 2026-06-11: NXT 미상장 거부 학습 (전 계정 공유) — 메인 실행 경로와 동일.
            if getattr(od, "exchange", "") == "NXT" and nxt_blacklist.looks_nxt_unsupported(res):
                try:
                    nxt_blacklist.record(ticker, note=res[:120])
                except Exception as _ne:
                    logger.warning(f"NXT 블랙리스트 기록 실패({ticker}): {_ne}")
            # 즉시 체결 확인 (KR 2초 후 보유 재조회 / US 는 즉시 확인 불가 → 폴링)
            filled = False; fill_note = ""; fill_price = None; avg_cost = None
            if accepted and is_kr:
                try:
                    await asyncio.sleep(2.0)
                    self.broker._acct_snap = None
                    after = await self.broker.kr_holdings()
                    before_qty = next((h["qty"] for h in _pre_kr if h.get("code") == ticker), 0) or 0
                    after_qty = next((h["qty"] for h in after if h.get("code") == ticker), 0) or 0
                    if after_qty > before_qty:
                        filled = True; fill_note = f"보유 {before_qty}→{after_qty}주"
                        # 사장 지시 2026-05-27: 매수가·수량 항상 기록 — 평단 변화로 체결가 역산.
                        before_avg = next((h.get("avg_price") for h in _pre_kr if h.get("code") == ticker), 0) or 0
                        after_avg = next((h.get("avg_price") for h in after if h.get("code") == ticker), 0) or 0
                        bq = after_qty - before_qty
                        if bq > 0 and after_avg and after_qty:
                            fill_price = (after_avg * after_qty - before_avg * before_qty) / bq
                        avg_cost = after_avg or before_avg or None
                except Exception as _e:
                    fill_note = f"체결확인 실패({_e})"
            # 사장 지시 2026-05-21: 즉시 체결 확인된 경우에만 +1. 접수만(미확인)이면 폴링이 확정 시 +1.
            if filled:
                self._trades_executed += 1
                self._trade_log.append({"ts": _now_kst_iso(), "ticker": ticker, "side": "buy",
                                        "qty": qty, "result": res, "filled": True, "ok": True, "watch": True,
                                        "fill_price": fill_price, "avg_cost": avg_cost,
                                        "fill_currency": "KRW"})
                # 사장 지시 2026-06-11: 실거래 원장에 체결 반영.
                try:
                    trade_ledger.apply_fill(self.uid, ticker=ticker, side="buy", qty=qty,
                                            price=fill_price, ccy="KRW", avg_cost=avg_cost,
                                            note="entry_watch")
                except Exception as _le:
                    logger.warning(f"[원장 uid={self.uid}] 진입감시 체결 반영 실패 {ticker}: {_le}")
            # 모바일 알림 ①: 체결 신청(접수)
            if accepted:
                await self._emit({"type": "order_submitted", "agent": "프롭트레이딩팀장",
                    "message": f"📨 {ticker} 매수 {qty}주 주문 접수 — 체결 확인 중",
                    "ticker": ticker, "side": "buy", "qty": qty})
            # 모바일 알림 ②: 체결 완료(즉시) / 또는 실패
            if filled or not accepted:
                await self._emit({"type": _trade_event_type(filled),
                    "message": f"{badge} — {ticker} {qty}주: {res}" + (f" | {fill_note}" if fill_note else ""),
                    "ticker": ticker, "side": "buy", "qty": qty, "filled": filled,
                    "trades_total": self._trades_executed})
            elif accepted:  # 접수만(미확인) → 5분마다 반복 폴링으로 확정 시 카운트
                asyncio.create_task(self._poll_fills_until_confirmed(
                    [{"ticker": ticker, "side": "buy", "qty": qty}],
                    list(_pre_kr) + list(_pre_us)))
            metrics.incr("orders_filled" if filled else "orders_unfilled",
                         market="KR" if is_kr else "US")
        except Exception as e:
            logger.error(f"_entry_watch_task({ticker}) 예외: {e}")
            await self._emit({"type":"trade_failed",
                "message": f"⚠ {ticker} 분봉 모니터 예외 — {e}"})
            metrics.incr("entry_watch_error", ticker=ticker)
            notifier.alert("CRITICAL", "분봉 진입 모니터 예외",
                           f"{ticker} {qty}주 진입 감시 중단 — {e}",
                           dedup_key=f"entry_watch_error:{ticker}")

    async def _finalize_kr_order_for_session(self, od, session):
        """KR 주문을 세션 거래소로 데코레이트. 시간외면 NXT 지정가 산정.
        반환 (order, skip_reason). skip_reason 이 있으면 호출부가 주문 보류 + 사유 발화(조용히 누락 금지)."""
        if getattr(od, "market", "KR") != "KR":
            return od, None
        od.exchange = kr_exchange_for_session(session)
        if not is_kr_extended_hours(session):
            return od, None
        if bool(getattr(self.broker, "is_mock", False)):
            return od, "KIS 모의투자는 NXT 주문을 지원하지 않습니다. KRX 정규장에서 재평가합니다"
        # 사장 지시 2026-06-11: 과거 NXT 거부('종목정보가 없습니다')로 학습된 종목은 주문 시도
        # 자체를 생략한다 (전 계정 공유 블랙리스트). 153130 매도가 4사이클 연속 거부된 사례 방지.
        if nxt_blacklist.is_blocked(od.ticker):
            return od, (f"{od.ticker} NXT 거래불가 학습 종목 — 시간외 주문 생략 "
                        f"(정규장에서 재시도)")
        # NXT 전용 시세가 있어야 NXT 주문을 낸다. KRX(J) 시세로 폴백하면 NXT 미상장
        # ETF도 가격 산정에 성공해 거래소에서 반복 거부된다.
        try:
            last = await self.broker.kr_last_price(od.ticker, market="NX")
        except Exception:
            last = 0.0
        if not last or last <= 0:
            return od, f"NXT 시세 없음 — {od.ticker} NXT 거래 가능 여부를 확인할 수 없어 시간외 주문 보류"
        slip = float(runtime.get("EXT_HOURS_LIMIT_SLIPPAGE_PCT", 0.5, uid=self.uid) or 0.5)
        max_prem = float(runtime.get("EXT_HOURS_MAX_PREMIUM_PCT", 1.5, uid=self.uid) or 0.0)
        # 정규(KRX) 기준가 — 프리마켓엔 전일종가. 이 대비 프리미엄을 캡해 얇은 NXT 추종 방지
        # (2026-06-15: NXT 프리미엄 추종으로 정규가 대비 +4.9% 과지불 + 즉시 평가손→조기손절 수정).
        try:
            ref = await self.broker.kr_last_price(od.ticker, market="J")
        except Exception:
            ref = 0.0
        side = od.side.value if hasattr(od.side, "value") else str(od.side)
        od.limit_price = compute_nxt_limit_price(last, side=side, slippage_pct=slip,
                                                 ref_price=ref, max_premium_pct=max_prem)
        od.price_type = PriceType.LIMIT
        return od, None

    def _extended_hours_blocked(self, session):
        """시간외 세션을 건너뛸 사유(문자열). None 이면 진행. (마스터/구간 토글 OFF·NXT미지원)"""
        if not is_kr_extended_hours(session):
            return None
        # 사장 지시 2026-06-08: 모의투자는 NXT(대체거래소) 미지원 확정(msg_cd=41050000) → 아예 비활성.
        # 능력감지(nxt_supported)보다 앞서 하드 차단해 거부 주문 시도 자체를 막는다.
        if getattr(self.broker, "is_mock", False):
            return "모의투자 계정 — NXT 시간외 매매 비활성(모의서버 미지원)"
        if not runtime.get("ENABLE_NXT_EXTENDED_HOURS", True, uid=self.uid):
            return "NXT 시간외 매매 비활성(마스터 OFF)"
        if session == "KR_PRE_MARKET" and not runtime.get("ENABLE_NXT_PRE_MARKET", True, uid=self.uid):
            return "프리마켓 비활성"
        if session == "KR_AFTER_MARKET" and not runtime.get("ENABLE_NXT_AFTER_MARKET", True, uid=self.uid):
            return "애프터마켓 비활성"
        if self.broker.nxt_supported() is False:
            return "이 계정은 NXT 미지원(모의 등) — 시간외 스킵"
        return None

    async def _poll_fills_until_confirmed(self, pending: List[Dict], baseline_holdings: List[Dict], cyc=None):
        # Recovery must not book partial fills while this live poll owns them.
        with trade_ledger.pending_fill_poll(self.uid, pending or []):
            return await self._poll_fills_active(pending, baseline_holdings, cyc)

    async def _poll_fills_active(self, pending: List[Dict], baseline_holdings: List[Dict], cyc=None):
        """접수됐지만 체결 미확인인 주문을 5분마다 '반복' 폴링한다 (사장 지시 2026-05-21).

        - pending: [{ticker, side, qty, ...}] — 실행부에서 즉시 체결이 확인되지 않은 주문들.
          (즉시 체결된 주문은 실행부에서 이미 +1 했으므로 여기 오지 않는다.)
        - baseline_holdings: 주문 직전 holdings 스냅샷 (qty 비교 기준; US false-positive 방지).

        매 주기마다 보유 변동을 재확인한다:
          • 체결 확인  → 그때 누적 카운트 +1, trade_log 기록, 통신로그 '체결 확인됨'
                         (type=trade_executed → 모바일 '체결 완료' 알림). 그 주문은 목록에서 제거.
          • 아직 미체결 → 채팅 메시지도, 카운트도 올리지 않고 '조용히' 다음 주기 재시도.
          • 해당 시장 마감 → 그 주문의 폴링을 조용히 종료 (KIS가 미체결 지정가를 자동 취소).
        루프 누수 방지를 위해 _POLL_MAX_ATTEMPTS 회 후엔 무조건 종료한다."""
        remaining = [dict(e) for e in (pending or [])]
        attempts = 0
        try:
            while remaining and attempts < _POLL_MAX_ATTEMPTS and not self._stop_event.is_set():
                await asyncio.sleep(_REVERIFY_DELAY_SEC)  # 5분 (테스트는 0으로 단축)
                attempts += 1
                try:
                    self.broker._acct_snap = None  # force fresh read
                    after_kr = await self.broker.kr_holdings()
                    try:
                        after_us = await self.broker._overseas_holdings()
                    except Exception:
                        after_us = []
                except Exception as e:
                    logger.warning(f"_poll_fills 보유조회 실패(다음 주기 재시도): {e}")
                    continue
                still: List[Dict] = []
                for e in remaining:
                    tk = str(e.get("ticker", "")).strip()
                    side = e.get("side", "buy")
                    qty = e.get("qty", 0)
                    is_kr_tk = _is_kr_code(tk)
                    after = after_kr if is_kr_tk else after_us
                    # 이동 base + 누적 recorded 로 '증분만' 기록 — 부분체결 잔여분을 목표 도달까지
                    # 다단계로 추적(2026-06-15 후속). _base_qty/_base_avg 가 없으면 baseline 으로 초기화.
                    if "_base_qty" not in e:
                        e["_base_qty"] = next((h["qty"] for h in (baseline_holdings or []) if h.get("code") == tk), 0) or 0
                        e["_base_avg"] = next((h.get("avg_price") for h in (baseline_holdings or []) if h.get("code") == tk), 0) or 0
                    e.setdefault("_recorded", 0)
                    target = int(e.get("qty", 0) or 0)
                    after_h = next((h for h in after if h.get("code") == tk), None)
                    after_qty = (after_h or {}).get("qty", 0) or 0
                    inc, new_base, new_rec, done = _poll_increment(side, e["_base_qty"], after_qty, target, e["_recorded"])
                    first_fill = (int(e["_recorded"]) == 0 and inc > 0)
                    if inc > 0:
                        # 사장 지시 2026-05-27: 폴링 확정 체결도 체결가·매수원가(avg_cost)를 '항상' 기록(실현손익 근거).
                        after_avg = (after_h or {}).get("avg_price") or 0
                        base_qty = int(e["_base_qty"]); base_avg = float(e.get("_base_avg") or 0)
                        fill_price = None; avg_cost = None
                        if side == "buy":
                            # 증분 체결가 = (after 평가 − base 평가)/증분. 평단(avg_cost)은 블렌딩 후 평단.
                            if after_avg and after_qty:
                                fill_price = (after_avg * after_qty - base_avg * base_qty) / inc
                            else:
                                fill_price = after_avg or None
                            avg_cost = after_avg or base_avg or None
                        else:  # sell — 체결가 ≈ 확정시점 현재가, 매수원가 = 매도 직전 평단
                            base_cur = next((h.get("cur_price") for h in (baseline_holdings or []) if h.get("code") == tk), 0) or 0
                            fill_price = ((after_h or {}).get("cur_price") or base_cur) or None
                            avg_cost = base_avg or None
                        # 사장 지시 2026-05-30(KR/US 비대칭): US 장중 평단/현재가 결손 → 라이브 호가로 보정.
                        if not is_kr_tk and not fill_price:
                            try:
                                _lp = await self.broker.us_last_price(tk)
                                fill_price = float(_lp) if _lp and _lp > 0 else fill_price
                            except Exception:
                                pass
                        if not is_kr_tk and not avg_cost and side == "buy":
                            avg_cost = fill_price
                        if first_fill:
                            self._trades_executed += 1   # 다단계라도 주문당 1건만 카운트
                        _ppart = f" · ⚠ 부분체결 누적 {new_rec}/{target}주" if not done else (
                            f" (부분체결 합계 {new_rec}/{target}주)" if int(e["_recorded"]) > 0 else "")
                        self._trade_log.append({"ts": _now_kst_iso(), "ticker": tk, "side": side,
                                                "qty": inc, "order_qty": target, "filled": True, "ok": True,
                                                "fill_note": "5분 폴링 후 체결 확인" + _ppart,
                                                "fill_price": fill_price, "avg_cost": avg_cost,
                                                "fill_currency": ("KRW" if is_kr_tk else "USD")})
                        try:
                            trade_ledger.apply_fill(self.uid, ticker=tk, side=side, qty=inc,
                                                    price=fill_price, ccy=("KRW" if is_kr_tk else "USD"),
                                                    avg_cost=avg_cost, note="poll_confirm")
                        except Exception as _le:
                            logger.warning(f"[원장 uid={self.uid}] 폴링 체결 반영 실패 {tk}: {_le}")
                        await self._emit({"type": "trade_executed", "agent": "프롭트레이딩팀장",
                            "message": (f"✅ {tk} {('매수' if side == 'buy' else '매도')} {inc}주 체결 확인됨 — "
                                        f"누적 {new_rec}/{target}주 (총 체결 {self._trades_executed}건)"),
                            "ticker": tk, "side": side, "qty": inc, "filled": True,
                            "fill_price": fill_price, "avg_cost": avg_cost,
                            "fill_currency": ("KRW" if is_kr_tk else "USD"),
                            "trades_total": self._trades_executed})
                        # thesis 는 주문당 1회(첫 체결 시) 기록 — US 비동기 매수 누락 방지(2026-05-29).
                        if side == "buy" and cyc is not None and first_fill:
                            _buy_rec = {"ticker": tk, "side": "buy", "ok": True,
                                        "fill_price": fill_price, "avg_cost": avg_cost,
                                        "fill_currency": ("KRW" if is_kr_tk else "USD")}
                            try:
                                await self._record_buy_thesis(_buy_rec, cyc)
                            except Exception as _the:
                                logger.warning(f"[펀드기획] 폴링 체결 thesis 기록 실패 {tk}: {_the}")
                        e["_base_qty"] = new_base; e["_base_avg"] = after_avg or base_avg; e["_recorded"] = new_rec
                    if done:
                        pass  # 전량 체결 완료 → 재큐잉 안 함
                    elif not self._market_closed_for(is_kr_tk):
                        still.append(e)  # 장중 잔여 미체결 → 다음 주기 재확인
                    else:
                        # 사장 지시 2026-06-16: 시장 마감 → 잔여 미체결 확정. 기존 '조용히 폐기'는
                        # '접수 후 확인 중'이 영영 안 닫혔다(uid2 375500) → 종결 메시지 발화(상태 닫기).
                        # KIS 가 미체결 지정가를 자동취소. 미체결이므로 체결 카운트는 올리지 않는다.
                        _rem_q = int(e.get("qty", 0) or 0) - int(e.get("_recorded", 0) or 0)
                        await self._emit({"type": "trade_failed",
                            "message": _poll_close_message(tk, side, _rem_q, "market_close"),
                            "ticker": tk, "side": side, "qty": max(0, _rem_q), "filled": False})
                remaining = still
        except Exception as e:
            logger.warning(f"_poll_fills_until_confirmed 예외: {e}")
            # 폴링 실패 = 체결 카운트가 누락될 수 있음 → 운영자에게 표면화.
            metrics.incr("poll_fills_error")
            notifier.alert("WARN", "체결 폴링 예외",
                           f"미체결 주문 반복 확인 중단 — {e}",
                           dedup_key="poll_fills_error")



class ArquantOrchestrator(_OpsRouterMixin, _MarketCalendarMixin, _ExecutionMixin):
    def __init__(self, ctx):
        self.ctx = ctx
        self.uid = ctx.uid
        self.is_admin = ctx.is_admin
        # 회로차단기 상태(사장 지시 2026-06-17): 연속 US 매수실패 누적. 세션 바뀌면 리셋.
        self._us_buy_fail_streak = 0
        self._us_circuit_session = None
        from infra import user_paths
        self.equity_path = user_paths.equity_path(ctx.uid)
        self.trade_log_path = user_paths.trade_log_path(ctx.uid)
        _inj = {"uid": ctx.uid}
        self.orchestrator = BaseAgent(name="주식운용실장", role="chief_orchestrator", model_key="chief_orchestrator", injection=_inj,
            system_prompt="""당신은 ArQuant v1.0의 주식운용실장입니다. 의사결정은 **2단계(2패스)**로 진행됩니다.

## 데이터 소스
1. **글로벌 지수**: KOSPI, KOSDAQ, S&P500, NASDAQ, 다우, 상해, 니케이, 환율, WTI
2. **뉴스**: 네이버 금융 실시간 뉴스
3. **DART 공시**: 최근 공시 자동 수집
4. **퀀트 데이터**: 3년치 일봉 + 수급 + KIS 분봉

## [후보 종목 선정]
- 글로벌리서치팀장의 매크로 보고와 검증된 지수만 근거로, 분석할 **후보 종목 정확히 5개**를 고릅니다.
- ⚠️ **대형주·메가캡에 치우치지 마십시오.** 시가총액 상·중·소형, 서로 다른 업종/테마를 골고루 섞으십시오.
  (예: 한 종목은 대형 반도체, 한 종목은 중형 소재, 한 종목은 소형 성장주, 한 종목은 금융/유틸 ...)
  같은 업종 3개 이상, '삼성전자·SK하이닉스만' 같은 구성은 금지.
- 이미 보유 중인 종목은 가급적 제외하고 새 후보를 우선합니다.
- 응답 **마지막 줄**에 반드시 이 형식으로만(다른 텍스트 없이):
  `후보종목: 삼성전자(005930), SK하이닉스(000660), 에코프로비엠(247540), 클래시스(214150), AAPL`
  ← **종목명(코드)** 형식, 미국은 티커. 정확히 5개. 코드가 불확실하면 종목명만 정확히 적으면 시스템이 코드를 채웁니다 — 코드를 지어내지 마십시오.

## [최종 매수 종목 결정]
- 계량분석팀장의 정량 평가와 마켓센티먼트팀장의 감성 분석을 받아, 후보 5개 중에서 **실제 매수할 종목을 좁힙니다.**
- 매수 개수는 프롬프트에 주어진 '최대 매수 개수 N'(전략 설정값)을 넘기지 마십시오.
- 퀀트 점수가 6점 미만이거나 뉴스 감성이 부정적인 종목은 제외합니다. 마땅한 게 없으면 N개보다 적게 골라도 됩니다.
- 응답 **마지막 줄**에 반드시 이 형식으로만:
  `최종종목: 005930, AAPL`  ← 후보 목록에 있던 코드만, N개 이하. 없으면 `최종종목: 없음`

## 공통
- 표에 없는 수치는 추정·생성하지 않습니다. 사장님(@사장)의 직접 지시는 최우선입니다.""")
        self.macro_analyst = create_macro_analyst(injection=_inj)
        self.quant_analyst = create_quant_analyst(injection=_inj)
        self.news_analyst = create_news_analyst(injection=_inj)
        self.insight_analyst = create_insight_analyst(injection=_inj)  # 기업리서치팀장(사장 지시 2026-07-21)
        self.trader = create_trader(injection=_inj)
        self.risk_guard = create_risk_guard(injection=_inj)
        # 사장 피드백 2026-05-18: 수탁자책임실장(policy_filter) 폐지 → 역할은 risk_guard 통합
        self.post_manager = create_post_manager(injection=_inj)
        # 사장 지시 2026-05-28(우선순위 3 단독): 사후관리실장 산하 펀드기획팀장 — 매수 시점 thesis 박고 매도 직전 상기.
        from agents.specialists import create_fund_planner
        self.fund_planner = create_fund_planner(injection=_inj)
        self.bond_manager = create_bond_manager(injection=_inj)
        self.commodity_manager = create_commodity_manager(injection=_inj)  # 사장 지시 2026-06-09 — 원자재 슬리브
        # 슬리브 role → 매니저 에이전트 (asset_sleeves SLEEVES 순회 시 사용)
        self._sleeve_managers = {"bond_manager": self.bond_manager,
                                 "commodity_manager": self.commodity_manager}
        self.ops_support = create_ops_support(injection=_inj)
        # 컴플라이언스실장(사장 지시 2026-07-22) — 사이클 반려는 결정론 compliance_mod.screen()
        # 이 수행하고, 이 페르소나는 @멘션 정책 설명 전용이다.
        self.compliance_officer = compliance_mod.create_compliance_officer(injection=_inj)
        # 뉴스 헤드라인 사전 선별기 — 40건 초과 시 굵직한 40건만 추리는 경량 페르소나 (대시보드 @멘션은 안 받음)
        # 사장 지시 2026-07-22: 뉴스 큐레이션은 **마켓센티먼트팀장의 겸직**이다.
        # (2026-05-14 엔 macro_analyst 와 같은 '글로벌리서치팀장' 명찰을 써서 로그·조직도에서 구분이
        #  안 됐고 소관도 어긋났다 — 큐레이션 산출물의 소비자가 곧 마켓센티먼트팀장이므로 그쪽으로 귀속.
        #  별도 '뉴스큐레이터' 정체를 새로 만들지 않는다.)
        self.news_curator = BaseAgent(
            name="마켓센티먼트팀장", role="news_curator", model_key="news_curator", injection=_inj,
            system_prompt=("당신은 ArQuant '마켓센티먼트팀장'의 뉴스 큐레이션 페르소나입니다. 다수의 증권 속보 헤드라인 중 시장·종목 분석에 가장 가치 있는 것만 골라내는 게 이 단계의 역할입니다.\n"
                "선정 기준: ① 실적/M&A/규제/소송/증자·감자/관리종목·거래정지/실적 가이던스/대규모 계약 등 실질 이벤트 우선, "
                "② 단순 시황 요약·일반 사설·반복 속보·재배포는 후순위, ③ 같은 사건 중복 보도는 1건만.\n"
                "응답은 오직 한 줄 — `선정: 1, 4, 7, 12, ...` (1-base 인덱스 콤마 구분). 다른 설명/주석 절대 금지."))
        self.broker = ctx.broker
        self.news_monitor = get_monitor()
        self.current_state = SwarmState.IDLE
        self.cycle_log: Optional[SwarmCycleLog] = None
        self.validation_attempts = 0
        self._cycle_history: List[Dict] = []
        self._stop_event = asyncio.Event()
        # 사장 지시 2026-06-04: 뉴스 풀 단일화 — KR/US 분기 폐지. 모든 뉴스를 한 풀에 쌓고 사이클마다
        # 전체를 마켓센티먼트팀장에게 전달(분석 후 비움). 시장 구분은 마켓센티먼트팀장·세션 게이트가 담당.
        self._pending_news: List[Dict] = []
        self._last_cycle_at: float = 0.0   # epoch of last analysis cycle (for the hourly periodic trigger)
        # 사전 게이트로 스킵한 직전 사유 — 같은 사유 반복 알림을 억제한다(사유가 바뀌거나
        # 사이클이 실제로 돌면 리셋되어, 상태 변화는 놓치지 않는다).
        self._last_gate_reason: Optional[str] = None
        # 안전감시 주문 접수 시각. 정시 사이클과 안전감시가 같은 종목을 중복 매도하지 않게 한다.
        self._safety_last_order_at: Dict[str, float] = {}
        # 실패한 주문은 즉시 예약을 풀되 짧은 재시도 유예를 둔다. UI 장애가 생겨도 60초마다
        # 같은 주문으로 브라우저를 두드려 정시 사이클까지 방해하지 않게 한다.
        self._safety_retry_after: Dict[str, float] = {}
        self._last_session: Optional[str] = None   # previous loop iteration's session (for market-open detection)
        self._last_cycle_hour_key = None   # 사장 지시 2026-06-08: 벽시계 시(hour) 앵커 (정시 정렬)
        self._last_status_state: Optional[str] = None  # last broadcast status state (suppress 1-min OFF_HOURS spam)
        # 사장 지시 2026-05-24: 개장 5분 후 KIS 실시세로 '오늘 개장 확정'을 1회 확인한 결과 캐시.
        # 키 "KR:YYYY-MM-DD"/"US:YYYY-MM-DD" → True(개장 확정)만 적재(이후 호출 생략).
        # 휴장/확인불가는 적재하지 않아 다음 사이클에 재확인(데이터 지연 자기 교정).
        self._mkt_open_verified: Dict[str, bool] = {}
        self._trades_executed = 0
        # 이번 사이클 '생산자 부재' 확정 플래그 — 부재일 때만 래치해 남은 공유 단계를 즉시 자체계산.
        self._producer_absent_this_cycle = False
        self._trade_log: List[Dict] = []
        self._agents_map = {
            "주식운용실장": self.orchestrator, "글로벌리서치팀장": self.macro_analyst,
            "계량분석팀장": self.quant_analyst, "마켓센티먼트팀장": self.news_analyst,
            "프롭트레이딩팀장": self.trader, "리스크관리실장": self.risk_guard,
            "사후관리실장": self.post_manager, "운용지원실장": self.ops_support,
            "포트폴리오기획팀장": self.fund_planner,
            "채권운용실장": self.bond_manager,
            "원자재운용실장": self.commodity_manager,
            "컴플라이언스실장": self.compliance_officer,
        }
        # 사장 지시 2026-05-20: 산하 팀장(investment/operations/finance) 및 코드 자가수정 폐지.
        # 운용지원실장 단일 역할만 남으며, 팀장 멘션 라우팅은 빈 매핑으로 비활성화한다.
        self._ops_team_leaders: Dict[str, str] = {}

    async def _emit(self, msg):
        """이 오케스트레이터(=이 유저)의 사이클 이벤트를 그 유저 WS 연결에만 송신한다.
        Phase 2 멀티테넌트: 다른 유저 대시보드로 이벤트가 새지 않게 uid 로 라우팅한다."""
        await _broadcast(msg, uid=self.uid)

    async def _emit_news_activity(self, msg):
        """뉴스 크롤·분석 활동 메시지 — 사장 지시 2026-06-08: ADMIN(hh09080)에게만 노출.
        (비관리자 대시보드는 /api/news 헤드라인만 보고, 크롤/분석 '활동'은 가린다.)"""
        if self.is_admin:
            await self._emit(msg)

    def _should_run_periodic(self) -> bool:
        """정각 2분 창에서만 시간당 1회 발화한다.

        장외 동안 마지막 사이클 시각이 오래돼도 22:30 같은 개장 시각에 밀린 사이클을
        따라잡지 않는다. 정각 창을 놓친 시간은 앵커만 현재 시로 넘겨 다음 :00을 기다린다.
        ``None``은 .force_first_cycle 마커가 만든 명시적 즉시 실행 상태다.
        """
        if self._last_cycle_hour_key is None:
            return True
        now = _now_kst()
        hour_key = _current_hour_key()
        if hour_key == self._last_cycle_hour_key:
            return False
        elapsed = now.minute * 60 + now.second
        if elapsed >= HOURLY_TRIGGER_WINDOW_SEC:
            self._last_cycle_hour_key = hour_key
            return False
        return True

    def _recent_cycle_dedup_active(self) -> bool:
        """직전 '실제 사이클' 후 5분 이내면 트리거 중복 스킵. 부팅 직후(_last_cycle_at==0.0)는
        비활성 — :00 5분 이내 재시작이 그 정시 사이클을 잃지 않도록(2026-06-19)."""
        return self._last_cycle_at > 0 and (time.time() - self._last_cycle_at) < 300

    def _init_cycle_anchors(self, force: bool) -> None:
        """루프 시작 시 트리거 앵커 초기화.
        • _last_cycle_hour_key = 현재 시(hour) 앵커 → 같은 시(hour) 내 재발화 방지.
        • _last_cycle_at = 0.0 → 5분 중복가드 비활성. 부팅 직후엔 '직전 사이클'이 없으므로,
          :00 직전 5분 이내에 재시작해도 그 정시 사이클이 중복가드에 걸려 누락되지 않게 한다(2026-06-19).
        • force(=.force_first_cycle 마커) 면 hour 앵커를 None 으로 비워 시작 즉시 1사이클(periodic_due=True)."""
        self._last_cycle_hour_key = None if force else _current_hour_key()
        self._last_cycle_at = 0.0

    def _intelligence_producer_uid(self):
        """실행 중인 프로필 중 하나가 공통 분석을 맡는다. 정지된 계정은 기다리지 않는다."""
        from infra.user_context import REGISTRY
        running = [uid for uid, ctx in REGISTRY.all_contexts().items()
                   if ctx.task is not None and not ctx.task.done()]
        return min(running) if running else self.uid

    async def _shared_or_compute(self, kind, fingerprint, compute):
        """실행 중인 대표 프로필이 계산·게시하고 다른 프로필은 결과를 공유한다.
        compute: zero-arg async 콜러블(기존 LLM 호출). 사장 지시 2026-06-08.
        플래그는 runtime.get (override-or-config-default) — main_swarm 은 from config import 만 하고
        import config 는 안 하므로 config.X 직접 참조 금지(프로젝트 관용)."""
        if not runtime.get("SHARE_MARKET_INTELLIGENCE", uid=self.uid):
            return await compute()
        store = get_intel_store()
        hk = _current_hour_key_str()
        if self.uid == self._intelligence_producer_uid():
            r = await compute()
            if r:                                               # 성공/비어있지 않음만 게시
                await store.publish(kind, hk, r, fingerprint, uid=self.uid, now=time.time())
            return r
        # 사장 지시 2026-07-22: 소비자(uid2 모의)가 매 사이클 21분 걸리던 원인 —
        # ① 대기 120초가 생산자(uid1)의 매크로 게시 시점보다 짧아 매번 타임아웃했고,
        # ② 첫 타임아웃이 _producer_absent_this_cycle 을 래치해 뒤따르는 공유 단계
        #    (macro_research·macro_report)까지 전부 자체계산으로 떨어뜨렸으며,
        # ③ 그 자체계산이 생산자와 같은 로컬 LLM 을 놓고 경합해 서로를 더 느리게 만들었다.
        # → 직전 게시분(TTL 내) 재사용을 먼저 시도하고, 대기 한도를 넉넉히 준다.
        #    '느린 생산자'를 기다리는 건 낭비가 아니라 이득이다 — 자체계산은 같은 GPU 를
        #    두고 생산자와 싸워 양쪽을 다 늦춘다. 반대로 생산자가 **아예 없으면** 단계마다
        #    헛기다리므로(420초×3), 그 경우만 래치해 이후 단계는 즉시 자체계산한다.
        #    둘의 구분: 생산자가 살아있다면 이 kind 를 한 번은 게시했을 것 → 게시 이력 유무.
        _fresh_sec = float(runtime.get("SHARE_STALE_OK_SEC", uid=self.uid) or MACRO_CACHE_TTL_SEC)
        hit = store.peek(kind, hk, fingerprint)
        if hit is None:
            hit = store.peek_recent(kind, fingerprint, max_age_sec=_fresh_sec, now=time.time())
        if hit is None and not self._producer_absent_this_cycle:
            _wait = float(runtime.get("SHARE_PRODUCER_WAIT_SEC", uid=self.uid) or 420)
            hit = await store.wait_for(kind, hk, fingerprint, timeout=_wait)
            if hit is None:
                hit = store.peek_recent(kind, fingerprint, max_age_sec=_fresh_sec, now=time.time())
            if hit is None and not store.ever_published(kind):
                self._producer_absent_this_cycle = True   # 게시 이력 자체가 없음 = 생산자 부재
        if hit is not None:
            return hit
        return await compute()

    async def _research_macro_themes(self, session: str, force: bool = False,
                                     news_digest: str = "", index_digest: str = "") -> str:
        """alibaba/tongyi-deepresearch가 검색·합성을 통합 수행 (search 단계).
        사장 피드백 2026-05-16: **마켓센티먼트팀장이 짚은 포인트 + 실제 검증 지수**를 검색 쿼리에
        주입해, 정적 질문이 아니라 '오늘 실제로 움직인 것'을 출발점으로 심층 검색하게 한다.
        세션 캐시(30분)지만 뉴스/지수 컨텍스트가 바뀌면 캐시를 무시하고 새로 검색.
        실패 시 빈 문자열 — fail-open.
        모델 분담(사장 확인 2026-06-11): 이 search 단계 = macro_researcher(Qwen3.6-35B-A3B-Uncensored-Claude-Genesis-Q8_0.gguf,
        Hermes 도구 tool-calling 필수라 pro 불가) → 최종 매크로 리포트·자산배분 '결정' =
        macro_analyst(글로벌리서치팀장, Qwen3.6-35B-A3B-Uncensored-Claude-Genesis-Q8_0.gguf+thinking)가 작성한다."""
        cache_key = "KR" if is_kr_session(session) else ("US" if session == "US_TRADING" else "OFF")
        # 컨텍스트 시그니처 — 뉴스/지수가 실질적으로 바뀌면 캐시 무효화
        _ctx_sig = str(hash((cache_key, (news_digest or "")[:1500], (index_digest or "")[:600])))
        cached = _research_cache.get(cache_key)
        if (not force and cached and (time.time() - cached.get("ts", 0)) < MACRO_CACHE_TTL_SEC
                and cached.get("value") and cached.get("sig") == _ctx_sig):
            return cached["value"]
        # 세션별 종합 리서치 관점
        if cache_key == "KR":
            _focus = ("1. 외국인 수급 동향 (최근 매도/매수 흐름, 시점, 펀더멘털 변화 vs 리밸런싱 해석)\n"
                      "2. 한국은행 금리 정책 전망 (다음 금통위, 시장 컨센서스, 분기 인하/인상 기대)\n"
                      "3. 코스피·코스닥 투자심리 (밸류에이션 부담·기대, 섹터 로테이션)\n"
                      "4. 원/달러 환율 영향 (수출·수입 비중, 자금 흐름에 미치는 효과)")
        elif cache_key == "US":
            _focus = ("1. 연준(Fed) 통화정책 방향 (FOMC dovish/hawkish 스탠스, 시장 기대치)\n"
                      "2. 미국 노동·물가 지표가 시사하는 매크로 (CPI·고용·임금 trend)\n"
                      "3. S&P/Nasdaq 투자심리·포지셔닝 (섹터 로테이션, 헤지펀드·소매 흐름)\n"
                      "4. 美 국채 금리·달러 강세가 주식 시장에 주는 영향")
        else:  # OFF_HOURS — 글로벌 시각
            _focus = ("1. 글로벌 주식시장 투자심리 (위험선호/회피 사이클)\n"
                      "2. 미중 무역·관세 관계 동향 (협상 진전, 갈등 격화 가능성)\n"
                      "3. 지정학적 리스크 (중동·우크라이나·대만 등 공급망 영향)\n"
                      "4. 원자재·유가 수급 (OPEC 회의, 미국 셰일, 수요 전망)")
        _ctx_block = ""
        if (index_digest or "").strip():
            _ctx_block += f"\n[금일 검증된 지수 움직임 — 실제 수치]\n{index_digest.strip()[:800]}\n"
        if (news_digest or "").strip():
            _ctx_block += f"\n[마켓센티먼트팀장이 이번 사이클에 짚은 포인트]\n{news_digest.strip()[:1800]}\n"
        query = (
            f"현재 {'한국' if cache_key=='KR' else '미국' if cache_key=='US' else '글로벌'} 증시 시황을 "
            f"종합·심층 분석해 주세요. 다음 4가지 관점:\n{_focus}\n"
            f"{_ctx_block}\n"
            "⚠️ 위 '실제 지수 움직임'과 '마켓센티먼트팀장이 짚은 포인트'를 **출발점**으로 삼아, "
            "관련된 최신 정책·수급·심리·지정학 동향을 구체적으로 검색해 분석하십시오. "
            "각 관점은 (1) 무엇이 (2) 왜 (3) 어떤 경로로 시장에 영향을 주는지 3~5문장으로 상세히, "
            "가능하면 날짜·기관·발언 주체를 명시. 가격 수치는 부차적 — 정책·심리·구조 해설 위주.")
        result = await deep_research(
            query, max_tokens=8000, timeout_sec=180,
            api_key="",
        )
        if not result:
            return ""  # fail-open
        _research_cache[cache_key] = {"ts": time.time(), "value": result, "sig": _ctx_sig}
        return result

    async def _collect_index_data(self):
        """글로벌 지수 워치리스트 크롤링 (blocking → run_in_executor). Returns the structured dict."""
        self.current_state = SwarmState.DATA_COLLECTION
        await self._emit({"type":"status","state":"DATA_COLLECTION","message":"글로벌 지수 수집 중"})
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, get_index_data)

    async def _collect_company_data(self, codes: List[str]) -> str:
        """전략가가 반환한 종목의 일봉 + 수급 + KIS 분봉 수집.

        Fallback chain (사장 피드백 2026-05-18 — KIS 우선·'데이터 부족' 해소):
          KR 일봉: ① KIS API 페이지네이션(`kr_daily_chart_deep`, ~2년) 먼저 →
                   ② KIS 0행이면 네이버 금융 크롤(`crawl_company_full`) 폴백
          KR 수급: 네이버 크롤(KIS 미제공) — 항상 보강 시도
          US: KIS 시세 + 일봉 (primary) → 실패 시 메시지에 명시"""
        if not codes:
            return "[종목 데이터] 분석 대상 종목 없음"
        await self._emit({"type":"status","state":"DATA_COLLECTION","message":f"종목 {len(codes)}개 데이터 수집 중"})
        loop = asyncio.get_event_loop()
        summaries = []
        empty_us_codes = set()
        for code in codes[:8]:  # cap to avoid rate limits (was 5 — bumped for 6 후보 + 보유 종목)
            if _is_kr_code(code):
                # ── KR 일봉: KIS API 먼저 (페이지네이션으로 ~2년) ──
                kis_rows = 0
                try:
                    kis_rows = len(await self.broker.kr_daily_chart_deep(code, years=2))
                except Exception as e:
                    summaries.append(f"  ⚠ [{code}] KIS 일봉 조회 예외: {e}")
                if kis_rows > 0:
                    csv_daily = _csv_row_count(_Path("data") / f"daily_{code}.csv")
                    summaries.append(f"[{code}] 일봉: KIS +{kis_rows}행 / 누적 {csv_daily}행")
                else:
                    # 사장님 지시 2026-05-19: KIS 신규 0행이면 누적 CSV가 있어도(=stale 가능성)
                    # 네이버 금융(KRX Sim 로직 fetch_stock_daily)으로 최신 봉 top-up — stale 데이터로
                    # 분석하던 문제 차단. (기존엔 csv_daily>0이면 폴백을 아예 안 했음.)
                    _n = 0
                    try:
                        from tools.market_data import fetch_stock_daily as _fsd
                        _ndf = await loop.run_in_executor(None, _fsd, code, 2)
                        _n = len(_ndf) if _ndf is not None else 0
                    except Exception as _e:
                        summaries.append(f"  ⚠ [{code}] 네이버 일봉 폴백 예외: {_e}")
                    csv_daily = _csv_row_count(_Path("data") / f"daily_{code}.csv")
                    summaries.append(f"  🔁 [{code}] KIS 0행 → 네이버 폴백 +{_n}행 / 누적 {csv_daily}행")
                # ── KR 수급(기관·외인): KIS 미제공 → 네이버 크롤로 항상 보강 ──
                try:
                    inv_df = await loop.run_in_executor(None, fetch_investor_data, code)
                    inv_total = _csv_row_count(_Path("data") / f"investor_{code}.csv")
                    summaries.append(f"  [{code}] 수급: 신규 +{len(inv_df) if inv_df is not None else 0} / 누적 {inv_total}행")
                except Exception as e:
                    summaries.append(f"  ⚠ [{code}] 수급 크롤 실패: {e}")
                # KR: KIS 분봉 (장중이면)
                if is_trading_hours():
                    try:
                        minute = await self.broker.kr_minute_chart(code)
                        summaries.append(f"  [{code}] KIS 분봉 {len(minute)}건 수집")
                    except Exception as e:
                        summaries.append(f"  ⚠ [{code}] KIS 분봉 실패: {e}")
            else:
                # US: KIS API로 시세 + 일봉 (사장 지시 2026-05-14 — 미국도 historical 데이터 필요)
                try:
                    price = await self.broker.us_price(code)
                    summaries.append(price)
                except Exception as e:
                    summaries.append(f"[US시세] {code} | ⚠ 조회 예외: {e}")
                try:
                    us_rows = await self.broker.us_daily_chart(code, days=100)
                    if us_rows:
                        summaries.append(f"  📈 [{code}] KIS 미국 일봉 +{len(us_rows)}행 누적")
                    else:
                        summaries.append(f"  ⚠️ [{code}] KIS US 일봉 수집 실패 (0행)")
                        empty_us_codes.add(code)
                except Exception as e:
                    summaries.append(f"  ⚠ [{code}] 미국 일봉 수집 실패: {e}")
            await asyncio.sleep(0.5)

        # Build formatted quant data — KR과 US 종목 모두 (load_daily_csv가 자동 분기)
        quant_parts = []
        for code in codes[:8]:
            quant_parts.append(format_quant_data_for_agent(code))
            if code in empty_us_codes:
                quant_parts.append(f"⚠️ [{code}] KIS US 일봉 수집 실패 (0행)")
        return "\n".join(summaries + quant_parts)

    async def _prefilter_news(self, articles: List[Dict], limit: Optional[int] = None) -> List[Dict]:
        """헤드라인이 너무 많으면 굵직한 `limit` 건만 추려낸다.
        사장 피드백 2026-05-15 (#22): LLM 응답 파싱이 자주 실패하던 이유로 결정론적 키워드 스코어링으로 전환.
        뉴스는 이미 crawl 시점에 KR/US/BOTH로 분류되어 있고 MATERIAL_NEWS_KEYWORDS도 있으니 LLM 호출 없이 정렬 가능."""
        from config import MATERIAL_NEWS_KEYWORDS
        effective_limit = int(limit) if limit and int(limit) > 0 else NEWS_PREFILTER_LIMIT
        if not articles or len(articles) <= max(NEWS_PREFILTER_TRIGGER, effective_limit):
            return articles or []

        def _score(a: Dict) -> int:
            text = (a.get("title", "") or "") + " " + (a.get("summary", "") or "")
            s = 0
            for kw in MATERIAL_NEWS_KEYWORDS:
                if kw in text:
                    s += 2
            # 종목 코드/티커 직접 언급 가중치
            if re.search(r"(?<!\d)\d{6}(?!\d)", text):
                s += 3
            if re.search(r"\b[A-Z]{2,5}\b", text):
                s += 1
            # 너무 짧거나 광고성 헤드라인 페널티
            if len(a.get("title", "") or "") < 12:
                s -= 1
            return s

        # 원래 순서를 보존하면서 점수 내림차순 정렬 → 같은 점수면 최신순 (articles는 최신 last가 일반적이므로 reversed)
        scored = sorted(enumerate(articles), key=lambda kv: (-_score(kv[1]), -kv[0]))
        picked = [a for _, a in scored[:effective_limit]]
        # 원래 시간 순서로 다시 정렬해 뉴스 분석가가 흐름을 잡기 쉽게
        picked.sort(key=lambda a: articles.index(a))
        await self._emit({"type": "agent_msg", "agent": "마켓센티먼트팀장",
            "message": f"🗂️ 누적 헤드라인 {len(articles)}건 → 결정론적 점수로 굵직한 **{len(picked)}건** 선별 (LLM 미호출, 키워드+종목코드 가중치)"})
        return picked

    async def _collect_per_stock_dart(self, codes: List[str], days: int = 90, limit: int = 8):
        """Per-stock 최근 공시(~90일) + **가장 최근** 분기/반기/연간 요약재무.
        사장 피드백 2026-05-18: 공시 윈도우를 14→90일로 넓혀 관리종목·증자·소송 등을 놓치지 않음.
        ITEM2a: DartResult.state 를 사람-읽기 텍스트로 접두사 표기 — 리스크관리실장이
                QUERY_FAILED(시스템 리스크 주의)와 NO_DISCLOSURE(특이사항 없음)를 명확히 구분.
        Returns (dart_dict, name_map). Best-effort — skips a code on any error."""
        out: Dict[str, str] = {}; names: Dict[str, str] = {}
        loop = asyncio.get_event_loop()
        for code in [c for c in (codes or []) if _is_kr_code(c)][:limit]:
            try:
                nm = await loop.run_in_executor(None, get_stock_name, code)
                if nm:
                    names[code] = nm
                    # 최근 공시 (기존 동작) — DartResult 반환
                    dd_result = await search_disclosures(corp_name=nm, bgn_de=(datetime.now() - timedelta(days=days)).strftime("%Y%m%d"))
                    # 사장 피드백 #7: 직전연도 요약재무상태표 + 손익계산서 추가 — DartResult 반환
                    fin_result = await get_financial_summary_by_stock_code(code)

                    # QUERY_FAILED 이면 리스크관리실장이 알 수 있도록 명시적 경고 접두사 삽입
                    dd_text = dd_result.risk_text() if dd_result.failed else str(dd_result)
                    fin_text = fin_result.risk_text() if fin_result.failed else str(fin_result)

                    sections = [f"{nm}({code})", dd_text]
                    if fin_text:
                        sections.append("")
                        sections.append(fin_text)
                    out[code] = "\n".join(sections)
            except Exception as _e:
                logger.warning(f"per-stock DART {code} 실패: {_e}")
        return out, names

    # markets whose *open* should force an immediate cycle off the accumulated news.
    # 2026-06-03 폐지된 것은 '분석전용 프리장'(거래 불가). NXT 시간외(KR_PRE_MARKET 08:00·KR_AFTER_MARKET 15:50)는
    # 실제 거래 가능 시장이라 개장 트리거에 포함한다(사장 지시 2026-06-08). 두 집합은 의도상 동일 멤버십
    # (개장-트리거 세션 = 라이브 세션) — market_open = '라이브 밖→라이브 진입' 전환 판정용. 둘을 항상 함께 갱신할 것.
    _MARKET_OPEN_SESSIONS = ("KR_PRE_MARKET", "KR_TRADING", "KR_AFTER_MARKET", "US_TRADING")
    _LIVE_SESSIONS        = ("KR_PRE_MARKET", "KR_TRADING", "KR_AFTER_MARKET", "US_TRADING")

    async def _set_status(self, state: str, message: str, *, force: bool = False):
        """Broadcast a status update only when the state actually changes (or force=True).
        Prevents the every-60s OFF_HOURS spam in the dashboard log."""
        if not force and state == self._last_status_state:
            return
        self._last_status_state = state
        try:
            self.current_state = SwarmState(state)
        except ValueError:
            pass
        await self._emit({"type": "status", "state": state, "message": message})

    async def _equity_poller(self):
        """Poll account balance every 5 min during market hours so equity_curve has dense points
        between analysis cycles. 사장 지시 2026-05-24: 장 운영시간(KR/US 정규장) 외에는 기록하지 않는다
        — 장외 평가금액 추이는 의미가 없어 포인트를 남기지 않는다."""
        while not self._stop_event.is_set():
            try:
                # 사장 지시 2026-05-24: 장 운영시간(KR/US 정규장) 외에는 평가금액 추이를 기록하지 않는다
                # — 장외엔 잔고 폴링·기록 모두 생략(불필요한 KIS 호출도 절약).
                # 사장 지시 2026-05-21: 자산곡선을 KR+US 통합 총평가로 — portfolio_holdings 가
                # 국내 nass_amt 에 해외주식 원화환산 평가를 더한 총평가를 돌려준다.
                snap = await self.broker.portfolio_holdings() if is_market_session_now() else None
                bp = (snap or {}).get("buying_power") or {}
                if snap is not None and bp.get("ok"):
                    # 사장 지시 2026-05-21: 잔고 확인과 동시에 KOSPI·NASDAQ(QQQ) 현재값을 수집 →
                    # equity 포인트와 같은 타임스탬프에 저장(벤치마크 일중 오버레이, 검증된 현재가 API만 사용).
                    kospi = nasdaq = None
                    try: kospi = await self.broker.kr_index_now("0001") or None
                    except Exception: kospi = None
                    try:
                        _q = await self.broker.us_last_price("QQQ")
                        nasdaq = _q if (_q and _q > 0) else None
                    except Exception: nasdaq = None
                    # 원장-KIS 보유수량 대조 (30분에 1회) — record_equity 전에 실행해, 허수 자동정정으로
                    # 평가액이 떨어지면 같은 포인트에 reconcile_adj 로 박히게 한다(가짜 손실단차 방지).
                    recon_adj = 0.0
                    try:
                        self._ledger_recon_tick = getattr(self, "_ledger_recon_tick", 0) + 1
                        if self._ledger_recon_tick % 6 == 1:
                            _hold = snap.get("holdings") or []
                            # ① 지속 허수(KR·연속확인) 자동 하향 정정 — ledger_eval 부풀림·가짜 곡선단차 차단
                            #    (2026-06-17: 047810 2주 6일 허수 재발 방지)
                            try:
                                from config import LEDGER_PHANTOM_PRUNE_CONFIRMATIONS as _PRUNE_N
                                _pr = trade_ledger.prune_phantoms(self.uid, _hold, min_confirmations=_PRUNE_N)
                                if _pr.get("value_krw_removed"):
                                    recon_adj = -float(_pr["value_krw_removed"])   # 평가액 하락 = 장부정정(매매손실 아님)
                                    logger.warning(f"[원장정정 uid={self.uid}] 허수 자동제거: {'; '.join(_pr['pruned'])}")
                                    notifier.alert("INFO", "원장 허수 자동정정",
                                                   f"uid={self.uid} {'; '.join(_pr['pruned'][:5])} — 자산곡선엔 매매손실 아닌 장부정정으로 반영",
                                                   dedup_key=f"ledger_prune:{self.uid}")
                            except Exception:
                                pass
                            # ①b 지속 누락(KR·KIS>원장·연속확인) 자동 상향 채택 — 매도 이중계상 등으로
                            #     원장이 KIS 아래로 떨어져 고착되는 것 방지(2026-06-19, prune 의 대칭).
                            try:
                                from config import LEDGER_ORPHAN_ADOPT_CONFIRMATIONS as _ADOPT_N
                                _ad = trade_ledger.adopt_orphans(self.uid, _hold, min_confirmations=_ADOPT_N)
                                if _ad.get("value_krw_added"):
                                    recon_adj += float(_ad["value_krw_added"])   # 평가액 상승 = 장부정정(매매이익 아님)
                                    logger.warning(f"[원장채택 uid={self.uid}] 누락분 자동채택: {'; '.join(_ad['adopted'])}")
                                    notifier.alert("INFO", "원장 누락분 자동채택",
                                                   f"uid={self.uid} {'; '.join(_ad['adopted'][:5])} — 자산곡선엔 매매이익 아닌 장부정정으로 반영",
                                                   dedup_key=f"ledger_adopt:{self.uid}")
                            except Exception:
                                pass
                            # ② 남은 괴리(US 등)는 기존처럼 경고만
                            _diffs = trade_ledger.reconcile(self.uid, _hold)
                            if _diffs:
                                logger.warning(f"[원장대조 uid={self.uid}] 보유수량 괴리: {'; '.join(_diffs)}")
                                notifier.alert("WARN", "원장-KIS 보유 괴리",
                                               f"uid={self.uid} {'; '.join(_diffs[:5])} — 수동거래/입출금 시 /api/ledger/reseed 필요",
                                               dedup_key=f"ledger_recon:{self.uid}")
                    except Exception:
                        pass
                    # 사장 지시 2026-06-11: 실거래 원장 평가 — KIS 집계 TR 대신 자체 체결 원장 M2M.
                    # (최초 1회 KIS 보유/예수금으로 시드 후, 이후엔 우리 체결만으로 진화.) prune 후 평가.
                    led_val = None
                    try:
                        led_val = await trade_ledger.ensure_value(
                            self.uid, self.broker, snap, fx=get_usdkrw(USDKRW_FALLBACK))
                    except Exception as _le:
                        logger.warning(f"[원장 uid={self.uid}] 평가 실패(이번 포인트는 KIS 곡선만): {_le}")
                    record_equity(self.equity_path, bp, "poll", holdings=snap.get("holdings") or [],
                                  kospi=kospi, nasdaq=nasdaq, ledger_eval=led_val,
                                  is_mock=bool(getattr(self.broker, "is_mock", False)),
                                  reconcile_adj=recon_adj)
                    # 사장 지시 2026-06-01: KIS 실현손익(TTTC8494R) 감사 대조 — 실전·약 30분마다 1회
                    # 로깅(주문·표시 무영향). 우리 체결기반 수익률 KPI 와 교차검증해 드리프트 조기탐지.
                    try:
                        self._audit_tick = getattr(self, "_audit_tick", 0) + 1
                        if not self.broker.is_mock and self._audit_tick % 6 == 1:
                            _aud = await self.broker.kr_realized_pnl_audit()
                            if _aud.get("ok"):
                                logger.info(f"[실현손익 감사] KIS 누적 실현손익 {_aud.get('realized', 0):,.0f}원 "
                                            f"({_aud.get('realized_rt', 0):.2f}%) — 체결기반 KPI 교차검증")
                    except Exception:
                        pass
                    # 사장 보고 2026-06-26: 유휴 USD → KRW 자동 역환전 점검(주기 스윕, 약 30분마다).
                    # KRW→USD 정방향은 통합증거금이 자동 처리하지만 역방향은 KIS 공개 환전 TR 부재로
                    # 자동 경로가 없었다(KR/US 비대칭) → 유휴 USD 감지 시 (기본) 수동 환전 알림으로 환기.
                    # 실환전은 LIVE_TRADING + AUTO_USD_TO_KRW_RECONVERT + KIS_FX_EXCHANGE_TR 가 모두
                    # 있을 때만(dry-run/모의/미설정이면 안전 no-op + 알림). KRW 한도/USD 평가 분리 유지.
                    try:
                        if not self.broker.is_mock and self._audit_tick % 6 == 1:
                            from infra import fx_reconvert
                            await fx_reconvert.maybe_reconvert_idle_usd(
                                self.broker, dry_run=(not LIVE_TRADING), uid=self.uid,
                                notifier=notifier)
                    except Exception as _fxe:
                        logger.warning(f"[USD역환전] 점검 실패(무시): {_fxe}")
            except Exception as e:
                logger.warning(f"[equity_poller] {e}")
            # 5분 슬립 — 60초 게이트와 결합해 잔고가 자주 안 흔들리는 한 가벼움
            for _ in range(300):
                if self._stop_event.is_set(): break
                await asyncio.sleep(1)

    async def _weekly_review_scheduler(self):
        """매 시간마다 토요일 06시 이후인지 체크 → 트리거 (사장 지시 2026-05-14).
        marker 파일로 중복 실행 방지 — 한 주에 한 번만 운용지원실장 호출."""
        from infra import weekly_review
        while not self._stop_event.is_set():
            try:
                _wk_msg = weekly_review.trigger_if_due(uid=self.uid, is_admin=self.is_admin)
                if _wk_msg:
                    await self._emit({"type": "agent_msg", "agent": "시스템",
                        "message": _wk_msg if isinstance(_wk_msg, str) else
                        "📅 [주간 피드백 루프] 지난 7일 운영을 점검했습니다."})
            except Exception as e:
                logger.warning(f"[weekly_review_scheduler] {e}")
            # 1시간마다 체크
            for _ in range(3600):
                if self._stop_event.is_set(): break
                await asyncio.sleep(1)

    async def _submit_safety_sell(self, holding: Dict[str, Any], reason: str,
                                  baseline_holdings: List[Dict]) -> bool:
        """안전감시 전용 전량매도. 전체 분석·위원회·신규매수 경로와 분리된 좁은 주문 경로다."""
        code = str(holding.get("code") or "").strip()
        try:
            qty = int(float(holding.get("qty") or 0))
        except (TypeError, ValueError):
            qty = 0
        if not code or qty <= 0:
            return False
        is_kr = _is_kr_code(code)
        if not LIVE_TRADING:
            self._safety_last_order_at[code] = time.time()
            await self._emit({"type": "agent_msg", "agent": "리스크관리실장",
                              "message": f"🛡️ 안전감시 감지(모의 실행) — {code} {reason}"})
            return True
        if is_kr:
            try:
                sellable = await self.broker.kr_psbl_sell_qty(code)
                if sellable is not None:
                    sellable = int(float(sellable or 0))
                    if sellable <= 0:
                        self._safety_last_order_at[code] = time.time()
                        await self._emit({"type": "agent_msg", "agent": "리스크관리실장",
                                          "message": f"🛡️ {code} 손절 감지, 매도가능수량 0주 — 기존 주문 여부 확인 중"})
                        return False
                    qty = min(qty, sellable)
            except Exception as exc:
                logger.warning(f"[safety-watch] {code} 매도가능수량 조회 실패, 보유수량으로 진행: {exc}")
        order = OrderDraft(ticker=code, side="sell", qty=qty, price_type="market",
                           market=("KR" if is_kr else "US"),
                           reason=f"연속 안전감시 — {reason}", approved=True)
        order, nxt_skip = await self._finalize_kr_order_for_session(order, get_current_session())
        if nxt_skip:
            await self._emit({"type": "trade_failed", "ticker": code, "side": "sell", "qty": qty,
                              "filled": False, "message": f"🛡️ 안전매도 보류 — {nxt_skip}"})
            return False
        result = ""
        for attempt in range(3):
            if attempt:
                await asyncio.sleep(2.5)
            try:
                result = await self.broker.place_order(order)
            except Exception as exc:
                result = f"[주문 예외] {exc}"
            if not any(x in result for x in ("초당", "거래건수", "EGW", "유량", "TPS")):
                break
        accepted = all(x not in result for x in
                       ("실패", "에러", "거부", "예외", "REJECT", "초당", "거래건수"))
        if accepted:
            self._safety_last_order_at[code] = time.time()
            self._safety_retry_after.pop(code, None)
            await self._emit({"type": "order_submitted", "ticker": code, "side": "sell", "qty": qty,
                              "filled": False,
                              "message": f"🛡️ 안전매도 접수 — {code} {qty}주 | {reason} | {result}"})
            asyncio.create_task(self._poll_fills_until_confirmed(
                [{"ticker": code, "side": "sell", "qty": qty}], list(baseline_holdings or [])))
            metrics.incr("safety_orders_submitted", market=("KR" if is_kr else "US"))
            return True
        self._safety_retry_after[code] = time.time() + SAFETY_FAILURE_BACKOFF_SEC
        await self._emit({"type": "trade_failed", "ticker": code, "side": "sell", "qty": qty,
                          "filled": False, "message": f"⚠️ 안전매도 실패 — {code}: {result}"})
        metrics.incr("safety_order_failed", market=("KR" if is_kr else "US"))
        return False

    async def _safety_watchdog(self):
        """60초마다 확정 손절만 감시한다. 종목선정·LLM·익절·비중조절은 정시 사이클에 남긴다."""
        while not self._stop_event.is_set():
            try:
                session = get_current_session()
                active = _safety_watch_active(session)
                if bool(getattr(self.broker, "is_mock", False)) and is_kr_extended_hours(session):
                    active = False
                if active and self.current_state != SwarmState.EXECUTION:
                    holdings = await self._session_holdings(session)
                    active_holdings = [h for h in holdings if
                                       (_is_kr_code(h.get("code")) if session != "US_TRADING"
                                        else not _is_kr_code(h.get("code")))]
                    from infra import position_thesis
                    stop_pct = float(runtime.get("STOP_LOSS_PCT", uid=self.uid) or 0.0)
                    now = time.time()
                    self._safety_last_order_at = {
                        c: ts for c, ts in self._safety_last_order_at.items()
                        if now - ts < SAFETY_ORDER_COOLDOWN_SEC
                    }
                    self._safety_retry_after = {
                        c: retry_at for c, retry_at in self._safety_retry_after.items()
                        if retry_at > now
                    }
                    for holding in active_holdings:
                        code = str(holding.get("code") or "").strip()
                        if (not code or code in self._safety_last_order_at
                                or self._safety_retry_after.get(code, 0.0) > now):
                            continue
                        reason = _safety_sell_reason(holding, position_thesis.get(self.uid, code), stop_pct)
                        if reason:
                            # await 없는 예약을 먼저 남겨, 정시 실행부가 동시에 같은 매도를 조립해도
                            # 중복 주문을 걸러낸다. 실행부가 먼저 시작했다면 아래 재검사에서 멈춘다.
                            if self.current_state == SwarmState.EXECUTION:
                                break
                            reserved_at = time.time()
                            self._safety_last_order_at[code] = reserved_at
                            accepted = await self._submit_safety_sell(holding, reason, holdings)
                            if not accepted and self._safety_last_order_at.get(code) == reserved_at:
                                self._safety_last_order_at.pop(code, None)
            except Exception as exc:
                logger.warning(f"[safety-watch] 감시 실패(다음 주기 재시도): {exc}")
                metrics.incr("safety_watch_error")
            for _ in range(SAFETY_WATCH_INTERVAL_SEC):
                if self._stop_event.is_set():
                    break
                await asyncio.sleep(1)

    async def start_continuous(self, user_directive: Optional[str] = None):
        self._stop_event.clear(); self.news_monitor.is_running = True
        self._last_session = get_current_session()
        # 사장 지시 2026-06-09(디버그/검증): data/<uid>/.force_first_cycle 마커가 있으면 정시(:00) 대기·
        # 5분 중복가드를 우회해 시작 즉시 1사이클을 발화한다. 1회성 — 발동 즉시 마커를 소비(삭제)한다.
        # 마커가 없으면 평소와 100% 동일(무해). 채권ETF 등 신기능 라이브 검증용.
        _force = False
        try:
            from infra import user_paths as _up_force
            _force_marker = _up_force.user_dir(self.uid) / ".force_first_cycle"
            if _force_marker.exists():
                _force = True
                _force_marker.unlink(missing_ok=True)
                logger.info(f"[force-cycle] uid={self.uid} 시작 즉시 1사이클 강제(마커 소비)")
        except Exception as _fe:
            logger.warning(f"[force-cycle] uid={self.uid} 훅 실패(무시하고 정상 진행): {_fe}")
        # 트리거 앵커 초기화 — 현재 시(hour) 앵커 + _last_cycle_at=0.0(5분 중복가드 비활성).
        # 이로써 :00 직전 5분 이내 재시작이 그 정시 사이클을 잃지 않는다(2026-06-19 버그 수정).
        self._init_cycle_anchors(force=_force)
        self._last_status_state = None
        # 사장 지시 2026-06-04: 재시작 직후 첫 사이클이 '뉴스 0' sell-only 로 눈머는 문제 방지 —
        # 인메모리 대기풀이 비어 있으면 최근 history(기본 90분 이내 크롤)를 시장별로 시드한다.
        # (crawl_once 는 영속 seen_links 와 대조해 '이미 본' 최근 기사를 다시 안 담으므로 빈 채로 시작됨.)
        try:
            if not self._pending_news:
                _recent = self.news_monitor.get_recent_articles(300)
                _seeded = seed_pending_news(_recent, _now_kst_iso(), window_min=90)
                if _seeded:
                    self._pending_news = list(_seeded)
                    logger.info(f"📰 재시작 뉴스 시드: {len(_seeded)}건 (최근 90분 history, 첫 사이클 눈멈 방지)")
        except Exception as _se:
            logger.warning(f"재시작 뉴스 시드 실패(무시하고 진행): {_se}")
        # Equity poller — keeps the dashboard 수익률 chart populated even outside cycles
        try: asyncio.create_task(self._equity_poller())
        except Exception: pass
        # 두뇌는 정시 1회만, 확정 손절 안전장치는 60초마다 독립 감시한다.
        try: asyncio.create_task(self._safety_watchdog())
        except Exception: pass
        # 사장 지시 2026-05-14: 주간 피드백 스케줄러 — 토요일 KST에 운용지원실장 자동 호출
        try: asyncio.create_task(self._weekly_review_scheduler())
        except Exception: pass
        if self._last_session == "OFF_HOURS":
            await self._set_status("OFF_HOURS", f"장외 ({_now_kst().strftime('%H:%M')} KST) — 뉴스 수집만, 다음 개장 시 사이클", force=True)
        else:
            await self._set_status("MONITORING",
                "1시간 두뇌 사이클 + 60초 안전감시 시작", force=True)
        while not self._stop_event.is_set():
            try:
                session = get_current_session()
                # 사장 지시 2026-05-21: 장 마감 전환 시 당일·누적 수익률 모바일 알림 (전환 순간 1회)
                try:
                    await self._maybe_market_close_alert(self._last_session, session)
                except Exception as _mce:
                    logger.warning(f"market_close 알림 실패: {_mce}")
                # status badge: only emit on transition (no 1-minute 장외 spam)
                if session == "OFF_HOURS":
                    await self._set_status("OFF_HOURS", f"장외 ({_now_kst().strftime('%H:%M')} KST)")
                else:
                    await self._set_status("MONITORING", "감시 중")

                # crawl + accumulate news into the single pool (dedup near-identical headlines).
                # 사장 지시 2026-06-04: KR/US 시장 분기·미러링·LLM 시장분류 폐지 — 모든 뉴스를 한 풀에 쌓고
                # 마켓센티먼트팀장이 사이클에서 직접 시장을 구분한다(market 필드 미사용).
                # 2026-09-14: 크롤은 독립 수집기(news-collector.timer)가 한다 — 여기선 결과만 읽는다.
                new_articles = self.news_monitor.pull_from_disk()
                if new_articles:
                    existing = [a.get("title", "") for a in self._pending_news]
                    added = []
                    for a in new_articles:
                        title = a.get("title", "")
                        if not _is_dup_title(title, existing):
                            self._pending_news.append(a); existing.append(title); added.append(a)
                    if added:
                        await self._emit_news_activity({"type": "news", "count": len(added), "ts": self.news_monitor.last_crawl_time,
                            "message": (f"📰 +{len(added)}건 (누적 {len(self._pending_news)}건) | "
                                        f"크롤 {self.news_monitor.last_crawl_time or ''}"),
                            "articles": [{"title": a.get("title", ""), "link": a.get("link", "")} for a in added[:5]]})

                # ── 전체 두뇌 사이클: 모든 프로필 공통으로 벽시계 정시에만 1회. ──
                # market_open 은 정시가 개장과 겹쳤을 때 분석/사이징 문맥에만 쓰며 별도 트리거가 아니다.
                market_open = (session in self._MARKET_OPEN_SESSIONS) and (self._last_session not in self._LIVE_SESSIONS)
                periodic_due = self._should_run_periodic()    # 사장 지시 2026-06-08: 벽시계 :00 앵커

                # ── 사장 지시 2026-05-14/2026-05-24: 사이클 사전 게이트 ──
                # (1) 휴장 — 주말은 자명 스킵, 그 외엔 개장 5분 후 KIS 실시세로 '오늘 개장'을 1회 확인
                #     (하드코딩 휴장일 목록은 폴백). → _market_closed_today
                # (2) cash 부족 — 가용 예수금이 최소 매매 단위(현실적 최저가 1주 ~5000원)도 안 되면 스킵
                # (3) 너무 잦은 사이클 방지 — 직전 사이클이 5분 이내라면 스킵 (트리거 중복 가드)
                skip_reason = None
                if periodic_due:
                    _xh_block = self._extended_hours_blocked(session)
                    if _xh_block:
                        skip_reason = _xh_block
                    elif self._recent_cycle_dedup_active():
                        skip_reason = f"직전 사이클이 {int((time.time()-self._last_cycle_at)/60)}분 전 — 트리거 중복 스킵"
                    else:
                        _closed, _closed_reason = await self._market_closed_today(session)
                        if _closed:
                            skip_reason = _closed_reason
                        else:
                            # cash 가용성 체크 — 한 번의 KIS 호출. 조회 실패(ok=False)는
                            # '돈 없음'이 아니라 '데이터 미상'이므로 스킵하지 않고 진행한다.
                            try:
                                _snap = await self.broker.kr_account_snapshot()
                                skip_reason = _low_cash_skip_reason(_snap)
                            except Exception:
                                pass  # 잔고 조회 실패는 진행 (broker가 사이클 안에서 재시도)
                if skip_reason and periodic_due:
                    # 같은 사유가 이어지면 첫 회만 알린다(사장 지시 2026-07-21). 모의 계정은
                    # NXT 미지원이라 시간외 내내 같은 문구가 나오는데, 연속 사이클이 3분마다
                    # 재발화하면서 대시보드를 도배했다. force=True 가 _set_status 의 중복
                    # 억제를 우회하므로 여기서 사유 기준으로 한 번 더 거른다.
                    if skip_reason != self._last_gate_reason:
                        await self._set_status("MONITORING", f"⏭ 사이클 사전 게이트: {skip_reason}",
                                               force=True)
                        self._last_gate_reason = skip_reason
                    self._last_cycle_at = time.time()
                    self._last_cycle_hour_key = _current_hour_key()  # 이 시각엔 재발화 방지(다음 :00 재시도)
                    self._last_session = session
                    _ns = _now_kst()
                    _sleep_steps = max(1, min(int(admin_config.news_crawl_interval(NEWS_CHECK_INTERVAL)),
                                              3600 - (_ns.minute * 60 + _ns.second) + 2))
                    for _ in range(_sleep_steps):
                        if self._stop_event.is_set(): break
                        await asyncio.sleep(1)
                    continue

                if periodic_due and is_trading_hours():
                    self._last_gate_reason = None              # 정상 사이클 → 게이트 알림 억제 해제
                    # 사장 지시 2026-06-04: 단일 풀 전체를 사이클에 쓰고 비운다. 풀이 비면 최신 20개(history)로
                    # 폴백 — 뉴스 없이 헛도는 사이클 방지. 시장 구분은 마켓센티먼트팀장·세션 게이트가 담당.
                    cycle_news, _used_fb = pick_cycle_news(
                        list(self._pending_news), self.news_monitor.get_recent_articles(20), fallback_n=20)
                    self._pending_news.clear()
                    _fb_note = " (신규 뉴스 없음 → 최신 20건 폴백)" if _used_fb else ""
                    if market_open:
                        await self._set_status("MONITORING",
                            f"📈 {session} 개장 — 뉴스 {len(cycle_news)}건 기반 사이클{_fb_note}", force=True)
                    else:
                        await self._set_status("MONITORING", f"⏱️ 1시간 정기 사이클 — 뉴스 {len(cycle_news)}건{_fb_note}", force=True)
                    with metrics.timer("analysis_cycle", session=str(session),
                                       market_open=bool(market_open)):
                        await self._run_analysis_cycle(cycle_news, user_directive,
                                                       session, market_open=bool(market_open),
                                                       fresh_news=not _used_fb)
                    self._last_cycle_at = time.time()
                    self._last_cycle_hour_key = _current_hour_key()   # 이 시각 발화 완료 — 다음 :00까지 대기
                    # 쿨다운 없음 — 사이클 끝나면 곧장 감시 상태로 복귀 (배지 고착 방지를 위해 명시적 브로드캐스트)
                    if not self._stop_event.is_set():
                        session = get_current_session()
                        if session == "OFF_HOURS":
                            await self._set_status("OFF_HOURS", f"장외 ({_now_kst().strftime('%H:%M')} KST) — 감시 재개", force=True)
                        else:
                            # 사장 지시 2026-06-11: '다음 정시'를 실제 시각으로 표시하되, 그 정시가 거래
                            # 세션이 아니면(장 마감 등) 시각 없이 '감시 재개'만.
                            _next_hr = _current_hour_key() + timedelta(hours=1)
                            _resume_msg = (f"사이클 완료 — 감시 재개 (다음 {_next_hr.strftime('%H:%M')} 사이클)"
                                           if is_market_session_now(_next_hr) else "사이클 완료 — 감시 재개")
                            await self._set_status("MONITORING", _resume_msg, force=True)

                self._last_session = session
                # 정기 사이클 :00 앵커(드리프트 방지): 뉴스 크롤 간격과 다음 정시 중 빠른 쪽에 깬다.
                _ns = _now_kst()
                _sleep_steps = max(1, min(int(admin_config.news_crawl_interval(NEWS_CHECK_INTERVAL)),
                                          3600 - (_ns.minute * 60 + _ns.second) + 2))
                for _ in range(_sleep_steps):
                    if self._stop_event.is_set(): break
                    await asyncio.sleep(1)
            except Exception as e:
                self.current_state = SwarmState.ERROR; logger.error(f"루프 오류: {e}")
                await self._emit({"type": "error", "message": str(e)})
                # 감시 루프 크래시는 '거래 중'인 줄 알고 멈추는 최악의 조용한 실패.
                metrics.incr("engine_loop_error")
                notifier.alert("CRITICAL", "감시 루프 오류 — 30초 후 자동 재시도",
                               str(e), dedup_key="engine_loop_error")
                await asyncio.sleep(30)
        self.current_state = SwarmState.STOPPED; self.news_monitor.is_running = False
        await self._emit({"type": "status", "state": "STOPPED", "message": "감시 중지됨"})

    async def _session_holdings(self, session: str) -> List[Dict]:
        """매도 평가용 보유종목. US 세션엔 해외 보유분을 합쳐야 미국 포지션이 사후관리실장
        매도 평가에 들어온다. (버그 2026-05-22: kr_holdings()만 써서 US_TRADING 사이클에
        미국 보유분이 통째로 누락 → 항상 "보유 종목 없음 — 분석 생략"으로 빠졌다.)
        _overseas_holdings()는 내부에서 예외를 삼키고 []를 반환하므로 KR 보유분은 보존된다."""
        holdings = await self.broker.kr_holdings()
        if session == "US_TRADING":
            holdings = holdings + await self.broker._overseas_holdings()
        return holdings

    async def _run_analysis_cycle(self, news_articles, user_directive, session, market_open: bool = False,
                                  fresh_news: bool = True):
        # 리팩터링 2026-05-27: 거대 단일 메서드를 단계별 헬퍼(_cyc_stage_*)로 분리.
        # 본문은 최상위 early-return 없는 선형 시퀀스이므로, 단계 간 공유 지역변수를
        # `cyc`(SimpleNamespace)에 담아 순서대로 호출만 한다 — 로직·순서·동작 1바이트 불변.
        cyc = types.SimpleNamespace()
        cyc.news_articles = news_articles
        cyc.user_directive = user_directive
        cyc.session = session
        cyc.market_open = market_open
        cyc.fresh_news = fresh_news   # 2026-07-22: 스테이지 분리 때 누락돼 news_macro 가 매 사이클 NameError 로 죽던 버그
        cyc.total_eval = 0.0  # news_macro 스테이지가 buying-power 조회로 채움(실패 시 0 → 슬리브 스테이지 스킵)
        cyc.cash = 0.0         # news_macro 스테이지가 예수금으로 채움(슬리브 사이징 예수금 cap 용)
        # 자산슬리브 스테이지(_cyc_stage_sleeves, finalize_sell '앞'에서 실행)가 채우는 필드:
        cyc.sleeve_buy_orders = []        # 슬리브 매수 주문(자산배분); build_orders 합류
        cyc.sleeve_sell_proposals = {}    # {sleeve_key: {code: directive}} — 사후관리실장 종합 입력
        cyc.sleeve_price_map = {}         # 슬리브 가격조회 dict; build_orders 가 price_map 에 합류
        cyc.sleeve_holdings_by_key = {}   # {sleeve_key: [보유 슬리브 ETF]} — 매도 조립용
        cyc.sleeve_buy_meta = {}          # {code: {name,sleeve_label,chief_label,rationale,weight_ctx,sleeve_key}} — 위원회 슬리브 심의용(2026-07-20)
        cyc.thesis_reminders = {}         # {manager_name: reminder} — 포트폴리오기획팀장 보유계획 일괄 상기
        cyc.stock_holdings = None  # C2: 슬리브 제외 주식 보유(슬리브 ON일 때만 채움)
        self.cycle_log = SwarmCycleLog(); self.validation_attempts = 0
        self._producer_absent_this_cycle = False   # 사이클 시작 — 공유 부재 플래그 리셋
        # 누적 헤드라인이 NEWS_PREFILTER_TRIGGER(기본 40)을 넘으면 큐레이터로 굵직한 N건만 선별.
        # 개장(market_open) 사이클은 N을 100으로 상향 — 누적된 종일치 뉴스를 폭넓게 흡수.
        prefilter_limit = 100 if market_open else None  # None → config 기본(NEWS_PREFILTER_LIMIT=40)
        cyc.news_articles = await self._prefilter_news(cyc.news_articles, limit=prefilter_limit)
        # 뉴스 신선도 감쇠(사장 지시 2026-07-21) — NEWS_DECAY_HOURS 초과한 오래된 뉴스는 감성 반영에서 제외.
        cyc.news_articles = _filter_news_by_freshness(
            cyc.news_articles, float(runtime.get("NEWS_DECAY_HOURS", uid=self.uid) or 0.0))
        cyc.formatted_news = self.news_monitor.format_articles_for_agent(cyc.news_articles)
        # ITEM6: 활성 계정의 상시 지시사항 로드 — 주식운용실장 프롬프트에 주입 (계정 격리)
        _active_uid, _ = self._active_actor()
        try:
            from infra.standing_directives import build_orchestrator_directive_block
            cyc.standing_directive_block = build_orchestrator_directive_block(_active_uid)
        except Exception as _sde:
            logger.warning("상시지시 로드 실패(uid=%s): %s — 사이클 계속", _active_uid, _sde)
            cyc.standing_directive_block = ""
        try:
            await self._cyc_stage_collect(cyc)
            await self._cyc_stage_news_macro(cyc)
            await self._cyc_stage_select(cyc)
            await self._cyc_stage_data_quant(cyc)
            # 사장 지시 2026-06-09(#1): 슬리브(채권·원자재) 매니저가 매크로+뉴스로 매도 판단을 *먼저* 내리고,
            # 그 제안을 사후관리실장이 주식 매도와 *종합*(finalize_sell)한다. 순서 = 슬리브 → finalize_sell.
            await self._cyc_stage_sleeves(cyc)
            await self._cyc_stage_finalize_sell(cyc)
            # QuantInSight 이식(사장 지시 2026-07-18): 매수 대상별 위원회 심의(찬반토론+정직성
            # 가드+결정론 게이트). 차단·회피 종목은 여기서 매수 대상에서 빠진다(매도 트랙 불간섭).
            await self._cyc_stage_committee(cyc)
            await self._cyc_stage_build_orders(cyc)
            await self._cyc_stage_risk(cyc)
            await self._cyc_stage_execute(cyc)
            await self._cyc_stage_report(cyc)
            await self._emit_advisory_signals(cyc)   # ROI 섀도우 신호(레짐·IC·데이터품질) — 관찰 전용
        except Exception as e:
            self.current_state = SwarmState.ERROR; logger.error(f"사이클 오류: {e}")
            if self.cycle_log: self.cycle_log.log("ERROR","시스템",str(e)); self._cycle_history.append(self.cycle_log.to_dict())
            try:
                cycle_store.record_cycle({
                    "uid": self.uid,
                    "started_at": self.cycle_log.started_at if self.cycle_log else _now_kst().strftime("%Y-%m-%d %H:%M:%S"),
                    "ended_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
                    "session": session, "market_open": market_open,
                    "news_count": len(cyc.news_articles) if cyc.news_articles else 0,
                    "error": str(e)[:2000],
                })
            except Exception: pass

    async def _cyc_stage_collect(self, cyc):
            session = cyc.session
            # [1] GLOBAL INDEX DATA (validated numbers only — no garbage)
            index_data = await self._collect_index_data()
            index_report = crawl_index_snapshot(index_data)
            index_facts = format_indices_for_macro(index_data)
            # 사장 피드백 2026-05-15 (4차): 글로벌 지수 수집 결과는 글로벌리서치팀장 이름으로 표기 (이름만, 결과는 그대로 수집된 데이터).
            self.cycle_log.log("DATA", "글로벌리서치팀장", index_report)
            await self._emit({"type":"agent_msg","agent":"글로벌리서치팀장","message":f"📈 지수 수집 완료\n{index_report}"})

            # current holdings — used both to diversify the orchestrator's picks and for fill confirmation
            holdings = []
            try:
                holdings = await self._session_holdings(session)
            except Exception as _he:
                # 클린코드 2026-05-19: 침묵 삼킴 제거 — 빈 holdings로 진행하면 보유 종목
                # 회피·체결확인이 불가하므로 최소한 원인을 남긴다.
                record_error("_collect_company_data", _he, context="세션 보유종목 조회 실패 → 빈 목록 진행", uid=self.uid)
            holdings_str = ("; ".join(
                f"{h['name']}({h['code']}) {h['qty']}주 "
                f"평단 {float(h.get('avg_price') or 0):,.2f} 현재가 {float(h.get('cur_price') or 0):,.2f} "
                f"손익 {float(h.get('pnl_pct') or 0):+.1f}%"
                for h in holdings) or "없음")

            # [2] DART — 사장 피드백 2026-05-15 (#20): DART는 국내 종목만 있으니 KR 장 시간에만 30분 간격으로 가동.
            # US_TRADING/OFF_HOURS 사이클은 DART 호출 자체를 생략.
            _is_kr_session = is_kr_session(session)
            _dart_ttl = 30 * 60 if _is_kr_session else DART_CACHE_TTL_SEC  # KR 세션은 30분, 그 외는 캐시 유지
            if not _is_kr_session:
                dart_report = "[DART] US/장외 세션 — 국내 공시 조회 생략"
                # 사장 피드백 2026-05-16: 장외 DART 생략은 정상 동작이므로 대시보드에 알리지 않음 (내부 처리만).
            elif (time.time() - _dart_cache["ts"]) < _dart_ttl and _dart_cache["value"]:
                dart_report = _dart_cache["value"]
                # 캐시 재사용은 내부 동작이라 대화창에 노출하지 않는다(흐름을 끊는 시스템 노트 제거, 사장 지시 2026-06-03). cycle_log엔 남김.
            else:
                _dr = await search_disclosures(bgn_de=(datetime.now()-timedelta(days=3)).strftime("%Y%m%d"))
                # ITEM2a: QUERY_FAILED는 시스템 리스크 경고, NO_DISCLOSURE는 정상으로 구분
                dart_report = _dr.risk_text() if _dr.failed else str(_dr)
                # QUERY_FAILED 응답은 캐시하지 않음 (다음 사이클에서 재시도)
                if not _dr.failed:
                    _dart_cache.update(ts=time.time(), value=dart_report)
            self.cycle_log.log("DATA", "시스템", dart_report)
            cyc.index_report = index_report
            cyc.index_facts = index_facts
            cyc.holdings = holdings
            cyc.holdings_str = holdings_str
            cyc.dart_report = dart_report

    async def _cyc_stage_news_macro(self, cyc):
            news_articles = cyc.news_articles
            market_open = cyc.market_open
            session = cyc.session
            formatted_news = cyc.formatted_news
            index_facts = cyc.index_facts
            dart_report = cyc.dart_report
            # [3] NEWS ANALYSIS first — macro now reflects the analyzed news (사장 지시 2026-05-14).
            # 개장 사이클은 뉴스 100건이 들어오므로 macro/orchestrator가 모두 그것을 흡수해야 함.
            # 사장 지시 2026-05-22: 신규 뉴스 0건(정기 사이클)이면 마켓센티먼트팀장 호출 자체를 생략하고
            # 보유 종목 매도 평가(계량분석)만 진행한다 — 뉴스가 없는데 뉴스분석 LLM을 부르는 건 무의미.
            _sell_only = (not news_articles) and (not market_open)
            self.current_state = SwarmState.NEWS_ANALYSIS
            await self._emit({"type":"status","state":"NEWS_ANALYSIS","message":"뉴스 분석 (증권 속보)"})
            # 사장 지시 2026-06-04: 뉴스 풀 단일화 — 전체 누적 뉴스(KR+US 혼재)를 전달하고, 현재 세션의
            # '지금 매매 가능한 시장'을 알려 마켓센티먼트팀장이 직접 KR/US 를 구분 표기하게 한다.
            _tradeable_now = "미국(US) 종목" if session == "US_TRADING" else "국내(KR) 종목"
            if _sell_only:
                news_report = "[마켓센티먼트팀장] 신규 뉴스 없음 — 뉴스 분석 생략. 이번 사이클은 보유 종목 매도 평가(계량분석)만 진행합니다."
                self.cycle_log.log("NEWS", "마켓센티먼트팀장", news_report)
                await self._emit_news_activity({"type":"agent_msg","agent":"마켓센티먼트팀장",
                    "message":"🔕 신규 뉴스 없음 — 뉴스 분석 생략, 보유 종목 매도 평가(계량분석)만 진행합니다."})
            else:
                _news_prompt = (
                    f"네이버 금융 '증권 속보' 크롤링 결과입니다 (전체 누적 {len(news_articles)}건 — 국내·미국 뉴스 혼재).\n"
                    f"현재 세션은 **{session}** — 지금 실제로 매매 가능한 시장은 **{_tradeable_now}**입니다.\n{formatted_news}\n\n"
                    f"이 뉴스들을 분석해 다음을 정리하십시오:\n"
                    f"① 직접 언급되거나 직접 영향을 받는 **종목/업종**과 각각의 호재·악재·이벤트 — 가능하면 종목명(또는 6자리 코드/미국 티커)을 함께 적되, "
                    f"**각 종목이 국내(KR)인지 미국(US)인지 시장을 표기**하고, 뉴스에 실제로 나온 것만 쓰고 모르는 코드는 지어내지 마십시오. "
                    f"지금 매매 가능한 시장({_tradeable_now})의 종목을 우선 부각하되, 반대 시장 뉴스도 맥락·테마로 정리하십시오.\n"
                    f"② 시장 전반 분위기·주목 테마 (1~3줄).\n"
                    f"③ 매크로(금리/환율/원자재/지정학) 시사점 — 글로벌리서치팀장 매크로 분석에 영감을 줄 수 있는 포인트 1~3개.\n"
                    f"이 분석은 글로벌리서치팀장 매크로 분석 및 주식운용실장 종목 선정에 최우선으로 반영됩니다.")
                # 30분 공유 캐시(사장 지시 2026-07-21) — 단, **신규 뉴스가 없을 때만** 쓴다.
                # market_open 은 세션이 바뀌는 첫 사이클 1회만 True 라(4216행) 그것만으론 방어가
                # 얇다: 사이클 진입부에서 _pending_news 를 이미 비웠으므로(4280행), 신규 뉴스를
                # 받아놓고 캐시를 쓰면 그 뉴스는 분석되지 않고 그대로 버려진다(다음 사이클에도
                # 돌아오지 않음). fresh_news=False(=최신 20건 폴백, 즉 신규 0건)일 때만 재사용해
                # 절감 효과는 얻고 신선도는 잃지 않는다.
                _use_news_cache = ((not market_open) and (not cyc.fresh_news) and _news_cache["value"]
                                   and _news_cache["session"] == session
                                   and (time.time() - _news_cache["ts"]) < NEWS_CACHE_TTL_SEC)
                _news_cache_min = None
                if _use_news_cache:
                    news_report = _news_cache["value"]
                    _news_cache_min = int((time.time() - _news_cache["ts"]) / 60)
                else:
                    news_report = await self._shared_or_compute(
                        "news_report", None, lambda: self.news_analyst.think(_news_prompt))
                    # 사장 지시 2026-07-20: 마켓센티먼트팀장(LLM)이 종목코드를 환각(가비아 103200 등)해도
                    # 화면·저장 텍스트의 '이름(코드)' 를 이름→코드 재조회로 교정(KR 세션 한정, best-effort).
                    if is_kr_session(session):
                        try:
                            news_report = await asyncio.to_thread(
                                _correct_display_codes, news_report,
                                resolver=resolve_kr_stock_code, name_check=get_stock_name)
                        except Exception as _cce:
                            logger.warning(f"뉴스 표시코드 교정 실패(무시): {_cce}")
                    # 매크로 캐시와 같은 방어(4578행 주석): 빈/실패 응답을 캐시하면 TTL 내내 그 빈 값이
                    # 재사용된다 → 유효 응답(40자 이상)만 캐시한다. 교정까지 끝난 값을 넣는다.
                    if len((news_report or "").strip()) >= 40:
                        _news_cache.update(ts=time.time(), value=news_report, session=session)
                self.cycle_log.log("NEWS", "마켓센티먼트팀장", news_report)
                _news_msg = (f"♻️ 직전({_news_cache_min}분 전) 뉴스 분석을 재사용합니다(LLM 비용 절감) — "
                             f"아래 종목·감성은 그 분석 시점 기준입니다.\n\n{news_report}"
                             if _news_cache_min is not None else news_report)
                await self._emit_news_activity({"type":"agent_msg","agent":"마켓센티먼트팀장","message":_news_msg})

            # [4] MACRO ANALYSIS — cached up to MACRO_CACHE_TTL_SEC. 개장 사이클(market_open)에선 캐시 무시
            # (방금 분석한 뉴스/지수를 반드시 흡수해야 하므로 새로 호출).
            self.current_state = SwarmState.MACRO_ANALYSIS
            await self._emit({"type":"status","state":"MACRO_ANALYSIS","message":"매크로 분석"})
            _idle = cycle_is_idle(_sell_only, getattr(cyc, "holdings", None))
            _use_macro_cache = (not market_open) and (time.time() - _macro_cache["ts"]) < MACRO_CACHE_TTL_SEC and _macro_cache["value"]
            _macro_cache_min = None  # 캐시 재사용 시 '직전 N분 전 판단 유지'를 글로벌리서치팀장이 직접 말하게(시스템 노트 대신)
            if _idle:
                # D (사장 지시 2026-05-28): 보유0 + 신규뉴스0 = 매수·매도 둘 다 불가 → 매크로 리서치/LLM 생략(비용 절감).
                # (아래 macro_report 후처리 else 분기가 cycle_log+emit 를 담당 — 캐시 재사용 분기와 동형.)
                macro_report = "[글로벌리서치팀장] 보유·신규뉴스 없음 — 매수·매도 대상이 없어 분석을 생략합니다(LLM 비용 절감)."
                metrics.incr("cycle_idle_skip")
            elif _use_macro_cache:
                macro_report = _macro_cache["value"]
                _macro_cache_min = int((time.time() - _macro_cache["ts"]) / 60)
            else:
                _cache_hint = ("⚡ 개장 사이클 — 캐시 무시, 누적 뉴스 100건 분석 결과를 반영합니다.\n" if market_open else "")
                # 사장 피드백 2026-05-15 (8차): alibaba/tongyi-deepresearch로 매크로 종합 리서치 (Tavily 대체).
                # 30분 캐시 + 세션별 종합 쿼리. 실패해도 매크로 분석은 진행 (fail-open).
                _macro_research = ""
                try:
                    # 사장 피드백 2026-05-16: 리서치 진행/완료 알림은 내부 단계라 대시보드 미표시.
                    # (최종 매크로 환경 요약만 글로벌리서치팀장 메시지로 노출됨)
                    _macro_research = await self._shared_or_compute(
                        "macro_research", None,
                        lambda: self._research_macro_themes(
                            session, force=market_open,
                            news_digest=news_report, index_digest=index_facts))
                except Exception as _te:
                    logger.warning(f"매크로 리서치 실패: {_te}")
                # 사장 피드백 2026-05-15 (#18): 직전 사이클의 자산 배분 권고를 명시적으로 첨부.
                # cycle_store에서 가장 최근 매크로 리포트를 가져와 핵심(자산 배분 권고 라인)만 추출.
                _prev_macro_hint = ""
                try:
                    _recent_cycles = cycle_store.list_cycles(limit=3)
                    for _rc in _recent_cycles:
                        _prev = (_rc.get("macro_report") or "").strip()
                        if _prev and len(_prev) > 40:
                            # 자산 배분 권고 줄만 추출 (핵심 비중 %)
                            _alloc_lines = []
                            for _ln in _prev.splitlines():
                                if ("주식" in _ln or "채권" in _ln or "현금" in _ln) and "%" in _ln:
                                    _alloc_lines.append(_ln.strip())
                                if len(_alloc_lines) >= 5:
                                    break
                            if _alloc_lines:
                                _prev_macro_hint = ("\n[직전 사이클 자산 배분 권고 — 사장 피드백 #18: 이 권고와 다르게 가려면 'X% → Y%'로 변경 폭 명시]\n"
                                                    + "\n".join(_alloc_lines[:5]) + "\n")
                                break
                except Exception as _pme:
                    logger.warning(f"직전 매크로 권고 로드 실패: {_pme}")
                _research_section = (
                    f"\n[매크로 종합 리서치 — alibaba/tongyi-deepresearch, 세션 {session}]\n"
                    f"⚠️ 아래는 **시황 해설·정책 전망·투자심리·수급 분석**용 합성 정보입니다. "
                    f"**가격·지수 수치는 절대 인용하지 마십시오** — 가격은 위 '검증된 글로벌 지수'(네이버 크롤)만 신뢰합니다.\n\n"
                    f"{_macro_research}\n"
                    if _macro_research else "")
                _macro_prompt = (
                    f"{_cache_hint}{index_facts}\n{_prev_macro_hint}"
                    f"{_research_section}\n"
                    f"마켓센티먼트팀장 뉴스 분석 (감성·이벤트 정리, 본 사이클 {len(news_articles)}건 기반):\n{news_report}\n\n"
                    f"최신 공시:\n{dart_report}\n\n세션: {session}.\n"
                    f"위 정보를 종합하여 매크로 분석과 자산배분 가이드라인을 제시하십시오. **가격 데이터 우선순위**:\n"
                    f"  1순위 (수치): '검증된 글로벌 지수' (네이버 크롤) — 모든 가격·% 인용은 여기서만\n"
                    f"  2순위 (해설): 매크로 리서치 (alibaba)의 시황·정책·수급·심리 분석 — 가격은 인용 금지, 해설만 활용\n"
                    f"  3순위 (이벤트): 마켓센티먼트팀장 분석 — 감성·이벤트 흐름\n"
                    f"⚠️ 매크로 리서치 결과에 가격·지수 수치가 있어도 **출처를 알 수 없으므로 인용 금지**. "
                    f"가격이 필요하면 위 검증된 지수 표에서만 가져오십시오.\n"
                    f"⚠️ 리서치 답변에서 *해설·정책 방향·투자심리·수급 흐름·지정학적 영향*은 적극 활용·인용 OK.\n"
                    f"⚠️ 뉴스에서 짚은 매크로 시사점(금리/환율/원자재/지정학)을 반드시 매크로 결론에 통합하십시오.\n"
                    f"⚠️ 직전 권고와 다른 비중을 권고할 때는 반드시 정확한 변경 폭을 표기하십시오.\n"
                    f"⚠️ 사장 피드백 2026-05-16: **상세히** 작성하십시오 — 매크로 환경 요약은 핵심 동인별로 "
                    f"(무엇이/왜/시장 영향 경로) 풀어쓰고, 자산 배분 권고는 주식·채권·현금 각각에 대해 "
                    f"근거 1~2줄씩, 핵심 리스크는 3개 이상 + 각 리스크의 트리거/모니터링 포인트를 함께 적으십시오. "
                    f"단, 가격·지수 수치 인용 규칙(검증 지수만)은 그대로 지키고 표/마크다운 강조는 쓰지 마십시오.\n"
                    f"표에 없는 수치는 추정·생성 금지.")
                macro_report = await self._shared_or_compute(
                    "macro_report", None, lambda: self.macro_analyst.think(_macro_prompt))
                # 빈/실패 응답이 캐시되면
                # MACRO_CACHE_TTL(30분) 동안 모든 사이클이 그 빈 값을 '캐시 재사용'해
                # 글로벌리서치팀장이 계속 '매크로 분석 실패'만 반복했다.
                # → 유효 응답(비어있지 않고 40자 이상)일 때만 캐시한다.
                if (macro_report or "").strip() and len((macro_report or "").strip()) >= 40:
                    _macro_cache.update(ts=time.time(), value=macro_report)
            # 사장 피드백 2026-05-16 (OPS#21): 매크로 LLM이 빈/너무 짧은 응답을 줄 때
            # 글로벌리서치팀장 칸이 '아무 말 없이' 비어 주식운용실장이 참조할 게 없던 문제.
            # 빈 응답은 명확히 표시하고, 주식운용실장에는 '매크로 분석 불가' 신호를 전달.
            if not (macro_report or "").strip() or len((macro_report or "").strip()) < 40:
                macro_report = "[글로벌리서치팀장] ⚠️ 매크로 분석 실패(LLM 빈 응답) — 이번 사이클은 검증 지수·뉴스만으로 판단합니다."
                self.cycle_log.log("MACRO", "글로벌리서치팀장", macro_report)
                await self._emit({"type":"agent_msg","agent":"글로벌리서치팀장","message":macro_report})
            else:
                self.cycle_log.log("MACRO", "글로벌리서치팀장", macro_report)
                # 캐시 재사용 배너(사장 지시 2026-07-20 착시 수정): 종전 "그새 바뀐 게 없습니다"는
                # 아래 본문이 '주식 5→4%' 식 재배분을 보여주는데도 "안 바뀜"이라 말해 모순됐다(배너의
                # '직전'=캐시 시점, 본문의 '직전'=그 전 사이클로 기준점이 달랐음). 또 본문 수치는 캐시
                # 시점 것이라 방금 수집한 지수와 불일치. → '재사용·수치는 그 시점 기준'임을 명시한다.
                _macro_msg = (f"♻️ 직전({_macro_cache_min}분 전) 매크로 분석을 재사용합니다(LLM 비용 절감) — "
                              f"아래 자산배분·수치는 그 분석 시점 기준이며, 현재 지수는 위 '지수 수집'을 참조하십시오.\n\n{macro_report}"
                              if _macro_cache_min is not None else macro_report)
                await self._emit({"type":"agent_msg","agent":"글로벌리서치팀장","message":_macro_msg})

            # 1주 매수 예산 힌트 — 사장 지시 2026-05-14: **총평가액** 기준 (실제 spend는 예수금으로 캡됨)
            _budget_hint = ""
            _macro_buy_blocked = False
            _macro_stock_pct = None
            _equity_weight = 0.0
            try:
                _bp0 = (await self.broker.kr_account_snapshot()).get("buying_power", {}) or {}
                _cash0 = float(_bp0.get("cash", 0.0) or 0.0)
                _total0 = float(_bp0.get("total_eval", 0.0) or 0.0) or _cash0
                cyc.total_eval = _total0   # 슬리브 스테이지(_cyc_stage_sleeves)가 비중 산정에 재사용
                cyc.cash = _cash0          # 슬리브 스테이지가 예수금 cap(*_PER_CYCLE_RATIO 와 함께)에 재사용
                _por = float(runtime.get("PER_ORDER_BUDGET_RATIO", uid=self.uid) or 0.10)
                _pob = min(_total0 * _por, _cash0) if _cash0 > 0 else 0.0
                if _total0 > 0:
                    _budget_hint = (f"\n참고 — 현재 총평가 {_total0:,.0f}원 / 예수금 {_cash0:,.0f}원, "
                        f"1종목당 1주 매수 예산은 약 {_pob:,.0f}원(총평가의 {_por*100:.0f}%, 예수금으로 캡)입니다. "
                        f"1주 가격이 이 예산을 크게 초과하는 초고가 종목은 사실상 매수가 어려우니 후보에서 빼거나 신중히 고르십시오.")
                    # 사장 지시 2026-06-04: 매크로 권고 주식비중 ≤ 현재 주식 평가비중(=추가 매수 여력 없음)
                    # 이면 신규 매수 후보 선정·평가 자체를 건너뛴다(매수엔진↔매크로 배분 불일치 churn 차단).
                    # MACRO_STOCK_GATE_ENABLED 로 on/off (운용지원실장 튜너블).
                    # 버그수정 2026-06-05: 해외 USD 예수금을 주식으로 오분류하던 (total−KR현금)/total 폐지.
                    # 실제 주식가치(KR주식+해외주식분)만으로 비중 산정 → 모의계정 영구 동결 해소.
                    _total_kr0 = float(_bp0.get("total_eval_kr") or _total0)
                    _os_stock0 = 0.0
                    try:
                        self.broker._get_overseas_cache()   # _overseas_stock_krw 채움
                        _os_stock0 = float(getattr(self.broker, "_overseas_stock_krw", 0.0) or 0.0)
                    except Exception:
                        _os_stock0 = 0.0
                    _equity_weight = compute_stock_weight(_total0, _cash0, _total_kr0, _os_stock0)
                    if bool(runtime.get("MACRO_STOCK_GATE_ENABLED", uid=self.uid)):
                        _macro_buy_blocked, _macro_stock_pct = _macro_blocks_new_buys(macro_report, _equity_weight)
            except Exception:
                pass
            cyc._sell_only = _sell_only
            cyc._macro_buy_blocked = _macro_buy_blocked
            cyc._macro_stock_pct = _macro_stock_pct
            cyc._equity_weight = _equity_weight
            cyc.news_report = news_report
            cyc.macro_report = macro_report
            cyc._budget_hint = _budget_hint

    async def _cyc_stage_select(self, cyc):
            session = cyc.session
            market_open = cyc.market_open
            news_articles = cyc.news_articles
            user_directive = cyc.user_directive
            _standing_directive_block = cyc.standing_directive_block
            _sell_only = cyc._sell_only
            # 사장 지시 2026-06-04: 매크로가 '주식 추가 매수 불가'로 판단되면(권고 주식비중 ≤ 현재 주식비중)
            # 신규 매수 평가를 건너뛴다. _sell_only 와 합쳐 _skip_buys 로 매수 파이프라인만 게이트한다.
            _macro_buy_blocked = getattr(cyc, "_macro_buy_blocked", False)
            # The research desk owns candidates and target weights. News
            # availability and LLM macro percentages are no longer alpha signals.
            from config import SYSTEMATIC_POLICY_ENABLED
            _systematic_plan = None
            if SYSTEMATIC_POLICY_ENABLED:
                from tools import stock_policy
                _market = "USA" if session == "US_TRADING" else "KRX"
                try:
                    _systematic_plan = await asyncio.to_thread(stock_policy.plan, self.uid, _market)
                except Exception as _policy_error:
                    logger.warning("정량 운용안 실패: %s", type(_policy_error).__name__)
                    _systematic_plan = {"status": "error", "weights": {}, "rows": [], "mode": "systematic"}
                cyc.systematic_policy = _systematic_plan
                cyc._macro_stock_pct = _systematic_plan.get("exposure", 0.)
                cyc._macro_buy_blocked = False
                _sell_only = _systematic_plan.get("status") != "ready"
                cyc._sell_only = _sell_only
                _macro_buy_blocked = False
                await self._emit({"type": "agent_msg", "agent": "계량분석팀장",
                    "message": f"정량 운용안 { _systematic_plan.get('as_of', '자료 확인 중') } · "
                               f"상태 {_systematic_plan.get('status')} · 목표 주식 비중 "
                               f"{_systematic_plan.get('exposure', 0)*100:.0f}%. "
                               "60·20일 상대강도와 변동성으로 후보를 정하고 5거래일 유지합니다. "
                               "위원회가 공시·뉴스 위험을 심의합니다. 실전·모의 공통 전략이며 수익성은 검증 중입니다."})
            _skip_buys = _sell_only or _macro_buy_blocked
            # 연속 손절 회로차단(사장 지시 2026-07-21) — 최근 실현매도가 N회 연속 손실이면 신규매수 중단.
            _loss_halt = int(runtime.get("LOSS_STREAK_HALT", uid=self.uid) or 0)
            if _loss_halt > 0 and not _skip_buys:
                try:
                    from infra import trade_ledger as _tl
                    _streak = _tl.recent_loss_streak(self.uid)
                    if _streak >= _loss_halt:
                        _skip_buys = True
                        await self._emit({"type": "agent_msg", "agent": "리스크관리실장",
                            "message": f"🛑 연속 손절 {_streak}회(≥{_loss_halt}) — 이번 사이클 신규매수 중단(회로차단). 매도·관리만 진행합니다."})
                except Exception as _lse:
                    logger.warning(f"연속손절 회로차단 확인 실패: {_lse}")
            # 개장·마감 매수회피(사장 지시 2026-07-21) — 변동성 큰 개장·마감 임박 구간엔 신규매수 스킵.
            _skip_open_m = int(runtime.get("SKIP_OPEN_MIN", uid=self.uid) or 0)
            _skip_close_m = int(runtime.get("SKIP_CLOSE_MIN", uid=self.uid) or 0)
            if (not _skip_buys) and (_skip_open_m > 0 or _skip_close_m > 0):
                _tw_reason = _buy_time_window_skip(session, _skip_open_m, _skip_close_m)
                if _tw_reason:
                    _skip_buys = True
                    await self._emit({"type": "agent_msg", "agent": "리스크관리실장",
                        "message": f"⏱ {_tw_reason} — 신규매수 회피(변동성 구간). 매도·관리만 진행합니다."})
            # 레짐 자동전환(사장 지시 2026-07-21) — KOSPI 20일선 대비 추세로 방어/공격 틸트.
            # 방어(약세): MIN_QUANT_SCORE +bump(엄선) · 사이클예산 ×0.7. 공격(강세): 예산 ×1.15.
            # 효과값은 self._regime_* 에 담아 게이트/예산 계산부에서 읽는다(스토어 파라미터 불변, 투명).
            self._regime_qs_bump = 0
            self._regime_budget_mult = 1.0
            if bool(runtime.get("REGIME_ADAPTIVE", uid=self.uid)) and is_kr_tradable(session):
                try:
                    _lb = int(runtime.get("REGIME_TREND_LOOKBACK", uid=self.uid) or 20)
                    _idx = await self.broker.kr_index_daily("0001", days=_lb + 6)
                    _cl = [float(r.get("close") or 0) for r in (_idx or []) if r.get("close")]
                    if len(_cl) >= _lb:
                        _ma = sum(_cl[-_lb:]) / _lb
                        _last = _cl[-1]
                        if _last < _ma:
                            self._regime_qs_bump = int(runtime.get("REGIME_DEFENSIVE_QS_BUMP", uid=self.uid) or 1)
                            self._regime_budget_mult = float(runtime.get("REGIME_DEFENSIVE_BUDGET_MULT", uid=self.uid) or 0.7)
                            _rlabel = f"방어(약세) — KOSPI {_last:,.0f} < {_lb}일선 {_ma:,.0f}"
                        else:
                            self._regime_budget_mult = float(runtime.get("REGIME_AGGRESSIVE_BUDGET_MULT", uid=self.uid) or 1.15)
                            _rlabel = f"공격(강세) — KOSPI {_last:,.0f} ≥ {_lb}일선 {_ma:,.0f}"
                        cyc._regime_adaptive = _rlabel
                        await self._emit({"type": "agent_msg", "agent": "글로벌리서치팀장",
                            "message": f"🧭 레짐 자동전환 — {_rlabel}: 퀀트하한 +{self._regime_qs_bump}, 사이클예산 ×{self._regime_budget_mult:.2f}"})
                except Exception as _rge:
                    logger.warning(f"레짐 자동전환 판정 실패(중립): {_rge}")
            macro_report = cyc.macro_report
            news_report = cyc.news_report
            index_facts = cyc.index_facts
            holdings = cyc.holdings
            holdings_str = cyc.holdings_str
            _budget_hint = cyc._budget_hint
            # ── PASS 1: 주식운용실장 → 후보 5종목 (뉴스 우선 + 시총·업종 분산) ────────
            # Session-aware: 사장 지시(2026-05-14) — US 정규장엔 **반드시 미국 티커만**, KR 세션엔 **반드시 국내 코드만**.
            # (반대편 시장 뉴스는 다른 풀에 누적되어 다음 세션에 사용됨.)
            if session == "US_TRADING":
                session_hint = ("⚠️ 현재 세션은 **미국 정규장(US_TRADING)** — 지금 체결 가능한 시장은 미국뿐입니다. "
                    "후보 5개는 **반드시 미국 상장 티커만**, 대형/중형/소형·서로 다른 섹터를 골고루 (메가캡만 금지). "
                    "**국내 6자리 코드는 절대 금지** — 국내 종목은 한국 장 시간에 분석합니다.")
            elif is_kr_tradable(session):
                session_hint = ("현재 세션은 **한국 장(또는 장 시작 전)** — 후보 5개는 **반드시 국내 6자리 종목코드만**, "
                    "대형/중형/소형주·서로 다른 업종을 골고루 (삼성전자·SK하이닉스 같은 메가캡 편중 금지). "
                    "**해외 티커는 절대 금지** — 미국 종목은 미국 장 시간에 분석합니다.")
            else:
                session_hint = ("현재 장외시간 — 다음 개장(한국/미국) 기준으로 후보 5개 (국내 6자리 / 미국 티커, 시총·업종 골고루).")
            if _skip_buys:
                # 사장 피드백 2026-05-16: 신규 뉴스 0건 → 신규 매수 후보 선정 자체를 건너뜀.
                # 사장 지시 2026-06-04: 매크로 추가 매수 여력 없음도 동일하게 후보 선정 생략.
                allocation = "후보종목: 없음"
                if _macro_buy_blocked and not _sell_only:
                    _pct = getattr(cyc, "_macro_stock_pct", None)
                    _ew = getattr(cyc, "_equity_weight", 0.0)
                    _skip_msg = (f"[후보 종목 선정] 글로벌리서치팀장 권고 주식비중 {(_pct or 0)*100:.0f}% ≤ 현재 주식 평가비중 "
                                 f"{_ew*100:.0f}% — 추가 매수 여력이 없어 신규 매수 후보 선정·평가를 생략합니다. "
                                 f"보유 종목 매도·관리만 진행합니다.")
                else:
                    _skip_msg = "[후보 종목 선정] 신규 뉴스 없음 — 신규 매수 후보 선정 생략, 보유 종목 매도 평가만 진행합니다."
                await self._emit({"type":"agent_msg","agent":"주식운용실장","message":_skip_msg})
            elif _systematic_plan is not None:
                _policy_candidates = stock_policy.candidates(
                    _systematic_plan, holdings or [], float(getattr(cyc, "total_eval", 0) or 0),
                    usdkrw=get_usdkrw(USDKRW_FALLBACK))
                allocation = "후보종목: " + (", ".join(_policy_candidates) or "없음")
                self.cycle_log.log("MACRO", "주식운용실장", allocation)
                await self._emit({"type": "agent_msg", "agent": "주식운용실장",
                                 "message": "[정량 운용안 후보]\n"+allocation})
            else:
                # 사장 지시 2026-06-04 ①: 채점 루브릭(가중 상위 지표·최소 퀀트점수)을 선정 프롬프트에 주입 —
                # LLM이 시스템 점수 기준에 부합할 후보를 고르게 한다(점수는 시스템 확정, 선정 가이드용).
                _rubric_block = format_scoring_rubric_block(
                    runtime.quant_weights(self.uid), runtime.dim_weights(self.uid),
                    int(runtime.get("MIN_QUANT_SCORE", uid=self.uid) or 0))
                allocation = await self.orchestrator.think(
                    f"[후보 종목 선정]\n전략리서치팀 매크로 보고:\n{macro_report}\n\n"
                    f"마켓센티먼트팀장 뉴스 분석 (증권 속보 기반):\n{news_report}\n\n"
                    f"검증된 지수:\n{index_facts}\n\n현재 보유 종목: {holdings_str}\n현재 세션: {session}\n"
                    f"{('사장 지시: '+user_directive if user_directive else '')}\n"
                    f"{_budget_hint}\n\n"
                    f"{session_hint}\n"
                    + (f"\n{_standing_directive_block}\n" if _standing_directive_block else "")
                    + f"\n{_rubric_block}\n"
                    + f"분석할 후보 종목 **정확히 5개**를 고르십시오.\n"
                    f"⚠️ **최우선**: 위 마켓센티먼트팀장이 짚은 종목/업종을 먼저 후보에 넣으십시오. (뉴스 원문은 제공되지 않으니 마켓센티먼트팀장 분석만 참고) "
                    f"뉴스 기반 적격 종목이 5개에 못 미칠 때만 매크로 판단으로 나머지를 채우십시오.\n"
                    f"⚠️ 각 종목은 반드시 **`종목명(종목코드)`** 형식으로 적으십시오 (예: `삼성전자(005930)`). "
                    f"코드가 확실치 않으면 **종목명만** 정확히 적으면 됩니다 — 시스템이 정확한 코드를 채웁니다. "
                    f"코드를 임의로 지어내지 마십시오(가짜 코드는 종목 자체가 누락됩니다). 미국 종목은 티커(예: AAPL).\n"
                    f"그 외에는 대형주에 치우치지 말고 시가총액·업종을 분산하고, 위 1주 예산 안에서 매수 가능한(또는 근접한) 종목을 우선하며, 이미 보유한 종목은 가급적 제외하십시오.\n"
                    f"[대화 흐름] 이건 이어지는 팀 회의입니다 — 첫 문장은 글로벌리서치팀장·마켓센티먼트팀장 발언을 받아 잇는 인계 코멘트로 시작하고, 지수·환율 등 이미 나온 수치는 다시 나열하지 말고 짧게 참조만 하십시오.\n"
                    f"⚠️ 응답 마지막 줄은 반드시 `후보종목: 종목명(코드), ...` (5개, 다른 텍스트 없이).",
                    # 후보 5개를 뽑는 정형 단계에 pro 기본 40k thinking 예산을 쓰면 모델이
                    # 30k+ 토큰을 생성하며 사이클이 장 마감까지 정체될 수 있다. 심층 심의는
                    # 뒤 위원회에서 수행하므로 이 단계만 짧고 결정론적인 출력으로 제한한다.
                    max_tokens=2500, timeout_sec=120, thinking=False)
                allocation = _strip_leading_section_marker(allocation, "[후보 종목 선정]", "[최종 매수 종목 결정]")
                self.cycle_log.log("MACRO", "주식운용실장", allocation)
                await self._emit({"type":"agent_msg","agent":"주식운용실장","message":f"[후보 종목 선정]\n{allocation}"})
            # 사장 지시 2026-05-22: 주식운용실장의 후보 코드 환각을 이름→코드 검색으로 보정.
            candidate_codes = _resolve_candidate_codes(
                allocation, session=session, resolver=resolve_kr_stock_code, name_check=get_stock_name
            ) or _extract_stock_codes(allocation)
            # Hard filter — enforce session/market boundary on candidates (LLM이 가끔 어김).
            # US 세션엔 KR 6자리 코드 제외, KR 세션엔 US 티커 제외. 장외엔 양쪽 모두 허용.
            if session == "US_TRADING":
                candidate_codes = [c for c in candidate_codes if not (_is_kr_code(c))]
            elif is_kr_tradable(session):
                candidate_codes = [c for c in candidate_codes if _is_kr_code(c)]
            # 사장 지시 2026-06-11: 시간외(NXT) 세션에선 NXT 거래불가 학습 종목을 매수 후보에서
            # 선제 제외한다 (전 계정 공유) — 주문 단계 보류 메시지가 매 사이클 반복되는 것 방지.
            if is_kr_extended_hours(session) and candidate_codes:
                _nxt_drop = [c for c in candidate_codes if nxt_blacklist.is_blocked(c)]
                if _nxt_drop:
                    candidate_codes = [c for c in candidate_codes if c not in _nxt_drop]
                    await self._emit({"type": "agent_msg", "agent": "시스템",
                        "message": f"🚫 NXT 거래불가 학습 종목 후보 제외: {', '.join(_nxt_drop)} (정규장에서만 거래)"})
            candidate_codes = candidate_codes[:5]
            # 그랜저 선행-후행 신호 매수(사장 지시 2026-07-21) — 선행주가 최근 30분+ 시장대비 오른
            # 후행주(KOSPI200/KOSDAQ150)를 매수 후보에 보강한다. ENABLE_LEADLAG_SIGNAL 파라미터로 on/off.
            # 보강 후보도 계량≥MIN·DART·위원회·리스크 게이트를 그대로 통과해야만 매수된다(안전).
            if _systematic_plan is None and is_kr_tradable(session) and not _skip_buys and bool(runtime.get("ENABLE_LEADLAG_SIGNAL", uid=self.uid)):
                try:
                    from tools import leadlag as _leadlag
                    _ll_thr = float(runtime.get("LEADLAG_MIN_BUY_SIGNAL", uid=self.uid) or 0.5)
                    _ll = _leadlag.leadlag_candidates(top_n=3, min_signal=_ll_thr)
                    _ll_new = [x for x in _ll if x["code"] not in candidate_codes]
                    if _ll_new:
                        candidate_codes = (candidate_codes + [x["code"] for x in _ll_new])[:6]
                        await self._emit({"type": "agent_msg", "agent": "계량분석팀장",
                            "message": "🔗 그랜저 선행-후행 신호 — 선행주 상승에 따른 후행주 매수후보 보강: " +
                                       ", ".join(f"{x['code']}(신호{x['signal']:+.2f} ← 선행 {','.join(x['leaders'][:2])})" for x in _ll_new)})
                except Exception as _le:
                    logger.warning(f"선행-후행 후보 보강 실패(무시): {_le}")
            # C (사장 지시 2026-05-28): 후보 해석이 0건인데 뉴스 신호가 있으면 뉴스 괄호표기 종목으로 보강 —
            # '뉴스만 있고 후보 0'으로 사이클이 낭비되는 것 방지(uid2 cycle24 MU/NVDA 무시 사례).
            # 보강 후보도 downstream 퀀트≥6·DART·리스크 게이트를 그대로 통과해야 매수된다.
            if _systematic_plan is None and not candidate_codes and not _skip_buys:
                _seeded = seed_candidates_from_news(getattr(cyc, "news_report", ""), session)
                if _seeded:
                    candidate_codes = _seeded[:5]
                    metrics.incr("cycle_candidate_seeded_from_news")
                    await self._emit({"type": "agent_msg", "agent": "주식운용실장",
                        "message": f"⚠️ 후보 해석 0건 — 뉴스 신호 종목으로 자동 보강: {', '.join(candidate_codes)} "
                                   f"(퀀트·리스크 게이트 통과 시에만 매수)"})
                else:
                    metrics.incr("cycle_no_candidate")

            # ── 사장 지시 2026-06-04 ③: 유니버스 결정론 스크리닝 — 후보 풀에서 레버리지/저가/거래대금미달 배제 ──
            #    (후보 아이디어 풀만 거른다 — LLM 명시 최종 주문은 불간섭. 배제 내역은 로그·메시지로 투명 공개.)
            _excl_lev = bool(runtime.get("UNIVERSE_EXCLUDE_LEVERAGED", uid=self.uid))
            _min_price = float(runtime.get("UNIVERSE_MIN_PRICE", uid=self.uid) or 0)
            _min_turn = float(runtime.get("UNIVERSE_MIN_TURNOVER", uid=self.uid) or 0)
            if candidate_codes and (_excl_lev or _min_price > 0 or _min_turn > 0):
                try:
                    from tools.universe_screen import screen_universe
                    _loop = asyncio.get_event_loop()
                    _uitems = []
                    for _c in candidate_codes:
                        _c = str(_c).strip()
                        _is_kr = _is_kr_code(_c)
                        _nm_c = _c
                        try:
                            _nm_c = (await _loop.run_in_executor(None, get_stock_name, _c)) or _c if _is_kr else _c
                        except Exception:
                            _nm_c = _c
                        _it = {"code": _c, "name": _nm_c}
                        if _is_kr and _min_price > 0:   # 가격 임계 활성 시에만 시세 조회(비용 절약)
                            try:
                                _px = await self.broker.kr_last_price(_c); await asyncio.sleep(0.12)
                                if _px and _px > 0:
                                    _it["price"] = _px
                            except Exception:
                                pass
                        _uitems.append(_it)
                    _ukept, _udropped = screen_universe(
                        _uitems, min_price=_min_price, min_turnover=_min_turn, exclude_leveraged=_excl_lev)
                    if _udropped:
                        logger.info(f"[유니버스 스크린 uid={self.uid}] 배제 {len(_udropped)}건: " +
                                    "; ".join(f"{c}({r})" for c, r in _udropped))
                        await self._emit({"type": "agent_msg", "agent": "리스크관리실장",
                            "message": "🧹 유니버스 스크리닝 — 후보 배제: " +
                                       ", ".join(f"{c}: {r}" for c, r in _udropped)})
                        candidate_codes = _ukept
                except Exception as _ue:
                    logger.warning(f"유니버스 스크리닝 실패(후보 유지): {_ue}")

            # ── 사장 지시 2026-05-14: 후보 사전 필터 — 데이터 수집/퀀트 평가 전에 예산 초과 종목 제거 ──
            # 1주 가격이 예산의 1.5배를 넘으면 어차피 못 사니까 LLM 비용 낭비 차단.
            try:
                _bp_snap = (await self.broker.kr_account_snapshot()).get("buying_power", {}) or {}
                _cash_pre = float(_bp_snap.get("cash", 0.0) or 0.0)
                _total_pre = float(_bp_snap.get("total_eval", 0.0) or 0.0) or _cash_pre
                _krw_usd_pre = get_usdkrw(USDKRW_FALLBACK)  # 사장 지시 2026-05-22: 5분 크롤 라이브 환율(폴백)
                # 버그수정 2026-06-05: 사전필터를 리스크부와 **동일한 사이클 매수예산 기준**으로 강화.
                # 기존 '예수금 1주' 기준은 조립부의 1주 예외와만 맞고 리스크부(cash×MAX_CYCLE_BUDGET_RATIO)와
                # 어긋나, 595K~5.95M 가격대 종목이 '선정→무조건 반려'되는 데드존을 만들었다(LG이노텍 1.13M 사례).
                from tools.affordable_prefilter import affordable_within_cycle_budget
                _cyc_ratio = float(runtime.get("MAX_CYCLE_BUDGET_RATIO", uid=self.uid) or 0.25)
                _overshoot = float(runtime.get("PER_ORDER_BUDGET_OVERSHOOT", uid=self.uid) or 1.2)
                if _cash_pre > 0:
                    kept, dropped = [], []
                    for c in candidate_codes:
                        try:
                            if _is_kr_code(c):
                                px = await self.broker.kr_last_price(c)
                                await asyncio.sleep(0.1)
                                if affordable_within_cycle_budget(px, _cash_pre, _cyc_ratio, _overshoot):
                                    kept.append(c)  # 가격 조회 실패(px≤0)는 보수적으로 통과
                                else:
                                    dropped.append(f"{c}({px:,.0f}원)")
                            else:
                                px = await self.broker.us_last_price(c.upper())
                                await asyncio.sleep(0.1)
                                _px_krw = px * _krw_usd_pre
                                if affordable_within_cycle_budget(_px_krw, _cash_pre, _cyc_ratio, _overshoot):
                                    kept.append(c)
                                else:
                                    dropped.append(f"{c.upper()}(${px:,.2f})")
                        except Exception:
                            kept.append(c)
                    if dropped:
                        # 사장 피드백 2026-05-15 (4차): 사전 필터 결과를 주식운용실장 후보 선정 메시지의 연장으로 통합.
                        await self._emit({"type":"agent_msg","agent":"주식운용실장",
                            "message": (f"후보 사전 필터 — 1주 가격이 사이클 매수예산(현금×{_cyc_ratio:.0%}×{_overshoot:.1f}) 초과라 제외: {', '.join(dropped)}\n"
                                        f"최종 후보 종목: {', '.join(kept) or '없음'}")})
                        candidate_codes = kept
            except Exception as _e:
                logger.warning(f"후보 사전 필터 실패: {_e}")

            # 포트폴리오 상관 분산(사장 지시 2026-07-21) — 보유종목과 30분 수익률 상관이 높은 신규후보 제외.
            _max_corr = float(runtime.get("MAX_PORTFOLIO_CORRELATION", uid=self.uid) or 0.0)
            if candidate_codes and _max_corr > 0 and is_kr_tradable(session):
                try:
                    from tools import leadlag as _ll_corr
                    _hold_codes = [str(h.get("code") or "").zfill(6) for h in (holdings or [])
                                   if _is_kr_code(str(h.get("code") or ""))]
                    if _hold_codes:
                        _ck, _cd = [], []
                        for _c in candidate_codes:
                            if not _is_kr_code(_c):
                                _ck.append(_c); continue
                            _mc = _ll_corr.max_correlation_with(_c, _hold_codes)
                            (_cd.append((_c, _mc)) if _mc > _max_corr else _ck.append(_c))
                        if _cd:
                            candidate_codes = _ck
                            await self._emit({"type": "agent_msg", "agent": "리스크관리실장",
                                "message": "🔀 포트폴리오 상관 분산 — 보유와 상관 높은 후보 제외: " +
                                           ", ".join(f"{c}(상관 {m:.2f}>{_max_corr:.2f})" for c, m in _cd)})
                except Exception as _ce:
                    logger.warning(f"상관 분산 필터 실패(후보 유지): {_ce}")

            if _systematic_plan is not None:
                _allowed = _systematic_plan.get("entry_weights", _systematic_plan.get("weights", {}))
                candidate_codes = [c for c in candidate_codes if c in _allowed]
            # 분석 대상 = 후보 ∪ 보유(KR) — 보유분은 사후관리실장의 매도 판단을 위해 함께 분석
            held_session = _codes_for_session(holdings, session)
            analysis_codes, _seen = [], set()
            for c in (candidate_codes + held_session):
                if c and c not in _seen:
                    _seen.add(c); analysis_codes.append(c)
            analysis_codes = analysis_codes[:8]
            cyc.candidate_codes = candidate_codes
            cyc.held_session = held_session
            cyc.analysis_codes = analysis_codes

    async def _cyc_stage_data_quant(self, cyc):
            session = cyc.session
            market_open = cyc.market_open
            news_report = cyc.news_report
            # 버그 2026-06-04(라이브): 결정론 점수 엔진이 이 스테이지에서 _parse_macro_stock_pct(macro_report)를
            # 호출하는데 macro_report 를 cyc 에서 읽지 않아 NameError 로 매 라이브 사이클이 붕괴했다. cyc 에서 바인딩.
            macro_report = getattr(cyc, "macro_report", "") or ""
            candidate_codes = cyc.candidate_codes
            held_session = cyc.held_session
            analysis_codes = cyc.analysis_codes
            # [4] DATA — 3yr daily/supply + 분봉 for analysis_codes, + per-stock DART (+ 종목명 맵)
            company_data = ""; per_dart: Dict[str, str] = {}; name_map: Dict[str, str] = {}
            if analysis_codes:
                await self._emit({"type":"status","state":"DATA_COLLECTION","message":f"종목 {len(analysis_codes)}개 데이터 수집"})
                company_data = await self._collect_company_data(analysis_codes)
                per_dart, name_map = await self._collect_per_stock_dart(analysis_codes, days=90, limit=8)
                if per_dart:
                    company_data += "\n\n[종목별 최근 DART 공시]\n" + "\n".join(f"[{k}]\n{v}" for k, v in per_dart.items())
                # 사장 피드백 2026-05-15 (4차): 데이터 수집 완료는 계량분석팀장 이름으로 (실제 분석 주체).
                self.cycle_log.log("DATA", "계량분석팀장", f"분석 종목: {analysis_codes}")
                await self._emit({"type":"agent_msg","agent":"계량분석팀장","message":f"📊 데이터 수집 완료 — 후보 {len(candidate_codes)}개 / 보유 {len(held_session)}개 (공시 {len(per_dart)}건)"})

                # ── 데이터 없는 종목 자동 제외 가드 (사장 지시 2026-05-14) ──
                # 데이터 수집 후 일봉 누적 0행인 종목은 분석/매수 대상에서 제외 — LLM이 빈 데이터로
                # 0점 평가하는 패턴 차단. 보유 종목은 매도 판단 위해 유지.
                from tools.market_data import load_daily_csv as _load_daily
                held_set = set(held_session)
                no_data_codes = []
                for c in list(candidate_codes):
                    if c in held_set:
                        continue
                    df = _load_daily(c)
                    if df is None or len(df) == 0:
                        no_data_codes.append(c)
                if no_data_codes:
                    _kept = [c for c in candidate_codes if c not in no_data_codes]
                    # 사장 피드백 2026-05-15 (#23, 4차): 상장폐지 의심 종목·일봉 누락 안내를 계량분석팀장 이름으로 통합 1줄로.
                    _delisted_suspects = []
                    for _c in no_data_codes:
                        if _is_kr_code(_c):
                            try:
                                _px = await self.broker.kr_last_price(_c)
                                if _px <= 0:
                                    _delisted_suspects.append(_c)
                            except Exception:
                                _delisted_suspects.append(_c)
                            await asyncio.sleep(0.1)
                    _msg_lines = [f"🚫 일봉 데이터 없는 종목 평가 제외 (LLM 계량평가 생략): {', '.join(no_data_codes)} → 후보 {len(candidate_codes)} → {len(_kept)}개"]
                    if _delisted_suspects:
                        _msg_lines.append(f"⚠️ 상장폐지/거래정지 의심: {', '.join(_delisted_suspects)} (다음 사이클 후보 선정 시 사전 확인 필요)")
                    await self._emit({"type":"agent_msg","agent":"계량분석팀장",
                        "message": "\n".join(_msg_lines)})
                    candidate_codes = _kept
                    # 버그+정책 수정 2026-05-19 (사장님 지시: "데이터 없으면 평가 자체를 안 함"):
                    # 기존엔 candidate_codes만 필터링하고 퀀트 루프는 analysis_codes를
                    # 순회해 데이터 없는 종목(예: BAC)이 그대로 LLM 평가·자의적 채점됐다.
                    # analysis_codes에서도 제거해 LLM 호출 자체를 생략한다.
                    # (보유 종목은 위 가드에서 no_data_codes에 넣지 않으므로 매도 판단은 유지)
                    analysis_codes = [c for c in analysis_codes if c not in no_data_codes]

            def _nm(c: str) -> str:
                c = str(c).strip()
                return f"{c}({name_map[c]})" if c in name_map else c

            # [5] Volume rank
            vol_rank = ""
            try:
                vol_rank = await self.broker.kr_volume_rank()
            except Exception: pass

            # [6] QUANT — 사장 피드백 2026-05-15 (#23): 종목당 1회 호출 → 별개 메시지로 출력.
            # 직전엔 후보+보유 전체를 한 번에 평가해 응답이 토큰 한계에 걸려 글자가 깨지는 일이 잦았음.
            # 시장 분리(#4): US 세션엔 KR 보유종목 분석 생략, KR 세션엔 US 보유종목 분석 생략.
            self.current_state = SwarmState.QUANT_ANALYSIS
            await self._emit({"type":"status","state":"QUANT_ANALYSIS","message":"퀀트 분석 (종목별 평가)"})
            # 후보·보유 종목 라인 — Pass 2 orchestrator 프롬프트에서 사용 (사장 피드백 2026-05-15 — 누락 복구)
            _cand_line = ", ".join(_nm(c) for c in candidate_codes) or "(없음)"
            _held_line = ", ".join(_nm(c) for c in held_session) or "(없음)"
            _quant_codes = list(analysis_codes or [])
            # 세션별 시장 필터 — 반대편 시장 종목은 평가에서 제외
            # 사장 피드백 2026-05-16: 미국장 세션엔 KR 종목을 보유분 포함 **완전 제외**.
            # (KR 주문은 어차피 KR 세션에만 체결되므로 매도 판단도 KR 세션에서 수행)
            if session == "US_TRADING":
                _quant_codes = [c for c in _quant_codes if not (_is_kr_code(c))]
            elif is_kr_tradable(session):
                _quant_codes = [c for c in _quant_codes if _is_kr_code(c)]
            _quant_scores: Dict[str, int] = {}
            _quant_tags: Dict[str, str] = {}       # P2 2026-08-02: 코드→알파 계열(복기 귀인용)
            _quant_sigmas: Dict[str, float] = {}   # 사장 지시 2026-06-04 ②: 리스크기반 사이징용 변동성(σ20)
            cyc.fundamental_research = getattr(cyc, "fundamental_research", {}) or {}
            _quant_sections: List[str] = []
            _entry_dirs: Dict[str, Dict[str, Any]] = {}  # 사장 피드백 #4: 진입가 directive 저장
            _sell_prices: Dict[str, Dict[str, Any]] = {}  # 사장 지시 2026-05-22: 보유종목 매도가 directive
            # 사장 지시 2026-06-04: 운용지원실장이 조정한 전략 파라미터(채점 가중치·매수 필터)를 계량분석팀장
            # 호출 프롬프트에 주입 — 전략이 점수·선정에 실제로 반영되게 한다(매수 후보 한정).
            _strat_params = {k: runtime.get(k, uid=self.uid) for k in (
                "MAX_BUY_VOLATILITY_PCT", "RSI_OVERBOUGHT_SKIP", "MIN_ADX_FOR_BUY",
                "REQUIRE_FOREIGN_NET_BUY", "MAX_PRICE_EXTENSION_PCT", "MIN_QUANT_SCORE")}
            _strategy_block = format_strategy_param_block(_strat_params)
            # 사장 지시 2026-07-03: 채팅으로 등록된 '사용자 전략'을 계량분석팀장 평가에 주입.
            # 전략이 활성이면 결정론 채점을 우회한다 — 결정론 모드에선 점수가 파이썬 확정이라
            # 전략이 점수에 반영될 길이 없기 때문(전략 삭제 시 원래 모드로 복귀).
            try:
                from infra.user_strategy import build_quant_block as _build_ustrat_block
                _user_strategy_block = _build_ustrat_block(self.uid)
            except Exception as _use:
                logger.warning(f"[사용자전략] 로드 실패(미주입 진행): {_use}")
                _user_strategy_block = ""
            # 사장 지시 2026-06-04: 결정론 점수 엔진 — DETERMINISTIC_SCORING이면 퀀트점수를 파이썬이 확정,
            # 계량분석팀장은 해설만(점수 파싱 안 함). 지표별 QIW·차원 DW·매크로%·뉴스감성으로 산정.
            _det_scoring = bool(runtime.get("DETERMINISTIC_SCORING", uid=self.uid)) and not _user_strategy_block
            _qiw = runtime.quant_weights(self.uid)   # runtime.QIW_KEYS 단일 정본(선행-후행·거래량급증 포함)
            _dw = runtime.dim_weights(self.uid)
            _leadlag_on = bool(runtime.get("ENABLE_LEADLAG_SIGNAL", uid=self.uid))
            _max_gap_up = float(runtime.get("MAX_GAP_UP_PCT", uid=self.uid) or 0.0)   # 갭업 추격 회피(사장 지시 2026-07-21)
            _macro_pct = _parse_macro_stock_pct(macro_report)
            # 사장 지시 2026-07-22: 종목별 계량분석은 서로 독립인데 순차 호출이라 사이클의 최대
            # 병목이었다(실측 13종목 × 40초 ≈ 14분 / 28분 사이클). 동시 실행한다.
            #
            # ⚠️ 단, self.quant_analyst 를 공유한 채 동시 실행하면 안 된다 — think() 가
            # conversation_history 를 읽고 append 하므로(base_agent.py:96,164), 아래 루프가
            # 매 종목 reset_history() 로 막고 있던 '종목 간 컨텍스트 누수'가 그대로 되살아난다.
            # 퀀트점수는 매수·매도로 직결되니 누수는 곧 판단 오염이다. → 태스크마다 **전용
            # 인스턴스**를 쓴다(self.quant_analyst 는 대시보드 @멘션 라우팅용으로 그대로 둔다).
            _sections_by_code: Dict[str, str] = {}    # 순서 재조립용 — append 는 동시 실행에서 순서가 흔들린다
            _qsem = asyncio.Semaphore(max(1, int(runtime.get("QUANT_CONCURRENCY", uid=self.uid) or 2)))

            async def _quant_one(_qcode):
                _qname = _nm(_qcode)
                _is_held = _qcode in held_session
                _qrole = "보유" if _is_held else "후보"
                # 버그 수정 2026-05-19 (치명적): 마커가 '\n[{code}]'였는데 실제 퀀트 블록은
                # format_quant_data_for_agent가 '[{code} 퀀트 데이터]'로 출력 → 마커가 요약줄
                # '[{code}] 일봉: KIS +N행'에 매칭돼 행수만 전달, 실데이터(OHLCV/지표/수급)는
                # 누락 → 계량분석팀장이 매번 '데이터 부족'으로 블라인드 평가(실매매까지 발생).
                # 실제 출력 헤더 '[{code} 퀀트 데이터]'를 직접 타깃팅한다.
                _stock_section_marker = f"[{_qcode} 퀀트 데이터]"
                _stock_data = ""
                _idx = company_data.find(_stock_section_marker)
                if _idx >= 0:
                    _end = company_data.find("\n[", _idx + len(_stock_section_marker))
                    _stock_data = company_data[_idx:_end] if _end > _idx else company_data[_idx:_idx + 4000]
                _per_dart_str = per_dart.get(_qcode, "")
                _fundamental_research = None
                _fundamental_prompt = ""
                try:
                    from tools.fundamental_rigor import assess_fundamental_research, format_research_for_prompt
                    from infra import fundamental_research_store
                    _fr_ts = _now_kst().strftime("%Y-%m-%d %H:%M:%S")
                    _fundamental_research = assess_fundamental_research(
                        _qcode, _qname, dart_text=_per_dart_str, financial_text=_per_dart_str,
                        source="ai_berkshire_advisory")
                    _fundamental_research.update({
                        "uid": self.uid,
                        "cycle_started_at": getattr(self.cycle_log, "started_at", None) or _fr_ts,
                        "ts": _fr_ts,
                    })
                    fundamental_research_store.record_snapshot(_fundamental_research)
                    cyc.fundamental_research[_qcode] = _fundamental_research
                    _fundamental_prompt = format_research_for_prompt(_fundamental_research)
                except Exception as _fre:
                    logger.debug(f"[펀더멘털리서치] {_qcode} 적재 생략: {_fre}")
                # 결정론 점수 산정(사장 지시 2026-06-04) — DETERMINISTIC_SCORING이면 파이썬이 점수 확정.
                _det_score = None
                _det_bd = None
                if _det_scoring:
                    try:
                        from tools.market_data import compute_quant_indicators
                        _ind = compute_quant_indicators(_qcode)
                        # 그랜저 선행-후행(30분 지평) 신호를 계량 팩터로 편입(사장 지시 2026-07-21) — KR 종목만.
                        if _leadlag_on and _ind is not None and _is_kr_code(_qcode):
                            try:
                                from tools import leadlag
                                _ll_s = leadlag.leadlag_signal(_qcode)
                                if _ll_s:
                                    _ind["leadlag"] = _ll_s
                            except Exception:
                                pass
                        _sent = parse_news_sentiment(news_report, _qcode, _qname)
                        _det_score, _det_bd = assemble_quant_score(_ind, _sent, _macro_pct, _qiw, _dw)
                        # 갭업 추격 회피(사장 지시 2026-07-21) — 매수 후보(보유 아님)가 시가 갭업 과대면 점수 0(게이트 배제).
                        if (not _is_held) and _max_gap_up > 0 and _ind is not None \
                                and _ind.get("gap_up_pct") is not None and float(_ind["gap_up_pct"]) > _max_gap_up:
                            logger.info(f"[갭업배제] {_qcode} 갭업 {_ind['gap_up_pct']:.1f}% > {_max_gap_up:.1f}% → 매수 후보 배제")
                            _det_bd["gap_up_excluded"] = round(float(_ind["gap_up_pct"]), 1)
                            _det_score = 0
                        if _ind and _ind.get("sigma20") is not None:
                            _quant_sigmas[_qcode] = float(_ind["sigma20"])
                    except Exception as _de:
                        logger.warning(f"[결정론점수] {_qcode} 계산 실패 → LLM 폴백: {_de}")
                        _det_score = None
                _policy = getattr(cyc, "systematic_policy", None)
                if _policy is not None and _qcode in candidate_codes:
                    _row = next((r for r in _policy.get("rows", []) if r["code"] == _qcode), None)
                    if _row and not (_det_bd or {}).get("gap_up_excluded"):
                        _det_score = max(6, min(10, int(round(5+5*_row["score"]))))
                        _det_bd = {"S_quant": _det_score, "S_news": None, "S_macro": None,
                                   "indicators": {"relative_strength": _row["score"]}, "policy": _policy.get("version")}
                        _quant_sigmas[_qcode] = _row["sigma_pct"] * (252**.5)
                    elif not _row:
                        _det_score, _det_bd = 0, {"indicators": {}, "policy": "missing_evidence"}
                # 자동 재시도: 실패/도구호출JSON 응답 시 최대 2회 retry (사장 피드백 2026-05-18)
                _quant_resp = ""
                if _det_score is not None:
                    # 해설자 모드: 점수는 시스템 확정, LLM은 한국어 해설만(점수 못 바꿈).
                    _score_directive = (
                        f"⚙️ 시스템 결정론 점수(확정, 바꾸지 말 것) = **{_det_score}/10**. "
                        f"구성: 퀀트 {_det_bd.get('S_quant')} · 뉴스 {_det_bd.get('S_news')} · 매크로 {_det_bd.get('S_macro')} "
                        f"(지표 기여: {_det_bd.get('indicators')}).\n"
                        f"이 점수가 왜 합당한지와 핵심 리스크를 한국어 줄글로 해설하십시오. **점수는 시스템이 확정**했으니 "
                        f"임의 변경 금지. 마지막 줄은 반드시 `퀀트점수: {_qcode}={_det_score}` 그대로 적으십시오"
                        + (f". 보유 종목이므로 매도 관점 코멘트와 `매도가: {_qcode}=시장가|숫자` 줄도 제시하십시오."
                           if _is_held else f". 후보면 `진입가: {_qcode}=시장가|숫자` 줄도 제시 가능."))
                else:
                    _score_directive = (
                        f"위 데이터를 **여러 계량 기법으로 평가**하고, 매수 적합도 0~10점 + 한 줄 코멘트로 응답하십시오. "
                        f"마지막 줄은 반드시 `퀀트점수: {_qcode}=점수` (1종목 1점수)"
                        + (f". 보유 종목이므로 매수 분석이 아니라 **매도 분석**을 하고 `매도가: {_qcode}=시장가|숫자` 줄도 제시하십시오."
                           if _is_held else f". 후보 종목이면 `진입가: {_qcode}=시장가|숫자` 줄도 제시 가능."))
                _base_prompt = (
                    f"[종목 단독 평가 — {_qrole}] {_qname}\n"
                    f"현재 세션: {session} / 최대 매수 종목 수 (참고): {runtime.get('MAX_TRADES_PER_CYCLE', uid=self.uid)}\n\n"
                    + (f"{_user_strategy_block}\n\n" if _user_strategy_block else "")
                    + ("" if (_is_held or _det_score is not None) else f"{_strategy_block}\n\n")
                    + f"마켓센티먼트팀장 분석 발췌:\n{news_report[:1200]}\n\n"
                    f"종목 데이터:\n{_stock_data or '(데이터 부족)'}\n\n"
                    f"최근 DART 공시:\n{_per_dart_str or '(공시 없음)'}\n\n"
                    f"장기 펀더멘털 리서치 참고(ai-berkshire 방식, advisory):\n{_fundamental_prompt or '(특이 리서치 없음)'}\n\n"
                    + _score_directive)
                _agent = create_quant_analyst(injection={"uid": self.uid})   # 태스크 전용 — 히스토리 공유 금지
                for _attempt in range(3):
                    try:
                        _agent.reset_history()  # 종목 간 컨텍스트 누수 방지
                        _prompt = _base_prompt
                        if _attempt > 0 and _looks_like_tool_call(_quant_resp):
                            _prompt = ("⚠️ 직전 응답이 도구호출 JSON/코드였습니다. 도구를 호출하지 마십시오 — "
                                       "데이터는 아래에 이미 전부 주어져 있습니다. JSON·코드펜스·함수호출 표기 없이 "
                                       "한국어 분석 줄글로만 답하십시오.\n\n" + _base_prompt)
                        _quant_resp = await _agent.think(_prompt)
                        if (_quant_resp and "에러" not in _quant_resp[:80]
                                and "퀀트점수" in _quant_resp and not _looks_like_tool_call(_quant_resp)):
                            break
                    except Exception as _e:
                        _quant_resp = f"[계량분석팀장 에러] {_e}"
                    if _attempt < 2:
                        await asyncio.sleep(1.0)
                # 최종 방어: 그래도 도구호출 JSON이면 원문을 화면에 노출하지 않고 데이터부족으로 강등.
                if _looks_like_tool_call(_quant_resp):
                    logger.warning(f"[계량분석팀장] {_qcode} 도구호출JSON 3회 — 데이터부족 처리")
                    _quant_resp = (f"{_qname}\n- 분석 실패: 모델이 분석 대신 도구호출을 반복 출력해 "
                                   f"해당 종목은 데이터 부족으로 처리합니다 (매수 후보에서 제외 권고).\n"
                                   f"퀀트점수: {_qcode}=0")
                await self._emit({"type":"agent_msg","agent":"계량분석팀장",
                    "message": f"📊 [{_qrole}] {_qname}\n\n{_quant_resp}"})
                # 점수 확정 — 결정론 모드면 파이썬 값 사용(LLM 파싱 무시), 아니면 LLM 응답 파싱.
                if _det_score is not None:
                    _quant_scores[_qcode] = _det_score
                    # P2: 이 점수를 끌어올린 알파 계열을 태깅 — 청산 후 계열별 성과 귀인의 유일한 근거.
                    from tools.quant_score import alpha_tag as _alpha_tag
                    _quant_tags[_qcode] = _alpha_tag(_det_bd)
                else:
                    _ms = re.search(rf"퀀트점수\s*[:：][^\n]*\b{re.escape(_qcode)}\s*=\s*(\d+)", _quant_resp or "")
                    if _ms:
                        try:
                            _quant_scores[_qcode] = int(_ms.group(1))
                        except ValueError: pass
                # 사장 지시 2026-06-04 ④: 에이전트 예측 구조화 적재(성과귀인). 매매동작 불변(부수 적재, 실패 무해).
                try:
                    from infra import scorecard_store
                    _sc_ts = _now_kst().strftime("%Y-%m-%d %H:%M:%S")
                    scorecard_store.record_signal({
                        "uid": self.uid,
                        "cycle_started_at": getattr(self.cycle_log, "started_at", None) or _sc_ts,
                        "ts": _sc_ts, "code": _qcode, "name": _qname,
                        "news_sentiment": parse_news_sentiment(news_report, _qcode, _qname),
                        "quant_score": _quant_scores.get(_qcode),
                        "det_breakdown": _det_bd,
                        "fundamental_verdict": (_fundamental_research or {}).get("verdict"),
                        "business_quality_score": (_fundamental_research or {}).get("business_quality_score"),
                        "moat_score": (_fundamental_research or {}).get("moat_score"),
                        "management_score": (_fundamental_research or {}).get("management_score"),
                        "valuation_margin_score": (_fundamental_research or {}).get("valuation_margin_score"),
                        "thesis_invalidators": (_fundamental_research or {}).get("thesis_invalidators"),
                    })
                except Exception as _sce:
                    logger.debug(f"[스코어카드] 신호 적재 생략 {_qcode}: {_sce}")
                # 사장 피드백 2026-05-15 (#4): 진입가 directive 파싱 (선택)
                _entry = _parse_entry_directive(_quant_resp, _qcode)
                if _entry["mode"] != "market":
                    _entry_dirs[_qcode] = _entry
                # 사장 지시 2026-05-22: 보유 종목은 매도가(지정가) directive 파싱 — 프롭트레이딩팀장이 그 가격으로 매도.
                if _is_held:
                    _sp = _parse_sell_price(_quant_resp, _qcode)
                    if _sp["mode"] == "limit":
                        _sell_prices[_qcode] = _sp
                _sections_by_code[_qcode] = f"[{_qrole}] {_qname}\n{_quant_resp}"

            async def _quant_one_guarded(_qcode):
                async with _qsem:
                    try:
                        await _quant_one(_qcode)
                    except Exception as _qe:   # 1종목 실패가 나머지 종목을 죽이지 않게 격리
                        logger.warning(f"[계량분석팀장] {_qcode} 평가 실패(해당 종목 제외): {_qe}")

            if _quant_codes:
                await asyncio.gather(*(_quant_one_guarded(_c) for _c in _quant_codes))
            # 순서 재조립 — 동시 실행이라 완료 순서가 뒤섞이므로 원래 후보 순서로 되돌린다
            # (quant_report 텍스트가 주식운용실장 프롬프트에 그대로 들어가므로 순서가 재현돼야 한다).
            _quant_sections = [_sections_by_code[_c] for _c in _quant_codes if _c in _sections_by_code]
            # 통합 리포트 (이후 주식운용실장·사후관리실장 입력용)
            quant_report = "\n\n---\n\n".join(_quant_sections) if _quant_sections else "[퀀트 평가 데이터 없음]"
            if _quant_scores:
                quant_report += "\n\n퀀트점수: " + ", ".join(f"{k}={v}" for k, v in _quant_scores.items())
            if _entry_dirs:
                quant_report += "\n\n진입가 지시: " + "; ".join(
                    f"{k}={v.get('raw') or v.get('mode')}" for k, v in _entry_dirs.items())
            if _sell_prices:
                quant_report += "\n\n매도가 지시: " + "; ".join(
                    f"{k}={v.get('raw') or v.get('mode')}" for k, v in _sell_prices.items())
            self.cycle_log.log("QUANT", "계량분석팀장", quant_report)
            # 사장 지시 2026-06-03: 종목별 장문 6연속 뒤에 한 줄 종합(요약 헤드라인)을 붙여 대화 리듬 회복.
            if _quant_scores:
                _gate = int(runtime.get("MIN_QUANT_SCORE", uid=self.uid) or 6)
                _passed = sorted((k for k, v in _quant_scores.items() if v >= _gate), key=lambda k: -_quant_scores[k])
                _top = max(_quant_scores.items(), key=lambda kv: kv[1])
                _digest = (f"📊 계량분석팀장 — 평가 종합: {len(_quant_scores)}종목 중 {_gate}점 이상 {len(_passed)}개"
                           + (f" ({', '.join(_passed)})" if _passed else "")
                           + f". 최고 점수 {_top[0]}={_top[1]}. 상세는 위 종목별 리포트 참조 — 주식운용실장이 이 점수로 최종 선정합니다.")
                await self._emit({"type": "agent_msg", "agent": "계량분석팀장", "message": _digest})
            cyc.candidate_codes = candidate_codes
            cyc.quant_report = quant_report
            cyc.per_dart = per_dart
            cyc.name_map = name_map
            cyc._entry_dirs = _entry_dirs
            cyc._sell_prices = _sell_prices
            cyc._cand_line = _cand_line
            cyc._quant_scores = _quant_scores   # 사장 지시 2026-06-04: MIN_QUANT_SCORE 결정론 게이트용
            cyc._quant_tags = _quant_tags       # P2: 진입 복기에 알파 계열을 실어보내기 위한 스냅샷
            cyc._quant_sigmas = _quant_sigmas   # 사장 지시 2026-06-04 ②: 리스크기반 사이징용 변동성

    async def _cyc_stage_finalize_sell(self, cyc):
            session = cyc.session
            market_open = cyc.market_open
            _sell_only = cyc._sell_only
            _standing_directive_block = cyc.standing_directive_block
            candidate_codes = cyc.candidate_codes
            _cand_line = cyc._cand_line
            macro_report = cyc.macro_report
            quant_report = cyc.quant_report
            news_report = cyc.news_report
            holdings = cyc.holdings
            holdings_str = cyc.holdings_str
            _budget_hint = cyc._budget_hint
            index_facts = cyc.index_facts  # 검증 지수 스냅샷 — 팀 간 수치 일관성(사장 지시 2026-06-03)
            # 사장 지시 2026-06-09 #1: 슬리브 ON 이면 슬리브 ETF 를 사후관리실장의 *주식* 매도 평가
            # 입력(holdings_str)·자동 익절손절 트랙에서 제외한다(슬리브 매도는 매니저 제안+사후관리 종합으로
            # 별도 처리 — _cyc_stage_sleeves 가 finalize_sell '앞'에서 제안을 만든다). 자동 익절/손절용으론
            # cyc.stock_holdings(슬리브 제외)를 따로 둔다.
            # 슬리브 보유는 cyc.holdings(원본)에 보존 — _cyc_stage_sleeves 가 비중계산·매도에 쓴다.
            # 전 슬리브 OFF 면 stock_holdings 미설정 → build_orders 는 cyc.holdings 전체 사용(기존 동작 불변).
            _full_holdings = holdings
            _bond_etf_on = any(bool(runtime.get(_s.enable_key, uid=self.uid)) for _s in SLEEVES)
            if _bond_etf_on:
                _stocks_only, _ = split_sleeve_holdings(holdings, self._enabled_sleeve_codes())
                holdings = _stocks_only  # 사후관리실장 *주식* 매도 평가 입력 = 주식만(활성 슬리브 제외; OFF 슬리브 보유는 폴백)
                # 프롬프트 보유 문자열도 슬리브 제외 — 슬리브 매도는 매니저 제안+사후관리 종합으로 처리.
                holdings_str = ("; ".join(
                    f"{h.get('name', h.get('code'))}({h.get('code')}) {h.get('qty')}주 "
                    f"평단 {float(h.get('avg_price') or 0):,.2f} 현재가 {float(h.get('cur_price') or 0):,.2f} "
                    f"손익 {float(h.get('pnl_pct') or 0.0):+.1f}%" for h in holdings) or "없음")
            # Experimental signal remains shadow-only until forward validation.
            # Do not expose its ranking to the committee: that would affect trades.
            from tools.empirical_edge import evaluate_candidates
            from tools.market_data import load_daily_csv
            _edge_evidence = {}
            if candidate_codes and not _sell_only:
                _edge_evidence = await asyncio.to_thread(
                    evaluate_candidates, candidate_codes, load_daily_csv,
                    min_net_edge_pct=float(runtime.get("MIN_NET_EDGE_PCT", uid=self.uid) or 0.0))
                await self._emit({"type": "agent_msg", "agent": "계량분석팀장",
                    "message": "실험 진입 규칙을 비교 기록했습니다. 과거 구간 검증에서 비교 기준을 "
                               "넘지 못해 매수 선정과 비중에는 반영하지 않습니다."})
            cyc.edge_evidence = {"mode": "shadow", "candidates": _edge_evidence}
            # ── PASS 2: 주식운용실장 → 최종 매수 종목 (전략 설정에 따라 1~N개) ──────
            _N = int(runtime.get("MAX_TRADES_PER_CYCLE", uid=self.uid) or 2)
            # 사장 지시 2026-05-14: 개장 사이클은 뉴스 100건 정보를 최대 활용 — 매매 건수 한도 확대
            if market_open:
                _N = min(8, max(_N, _N * 3))  # 3배 또는 8개 중 작은 값
            # _max_trades_runtime은 실행 단계에서도 동일한 한도를 쓰도록 별도 저장
            _exec_N = _N
            # 사장 지시 2026-06-03(흐름): 최종 매수 결정 '발화'는 사후관리실장 매도 검토 뒤에 노출한다
            # (사람 회의처럼 보유 점검 먼저 → 신규 매수 나중). 계산 순서는 그대로 두고 발화만 지연한다.
            _pass2_emit_msg = None
            if _sell_only or not candidate_codes:
                # 사장 피드백 2026-05-16: 신규 뉴스 없음(또는 후보 0) → 신규 매수 결정 자체를 생략.
                final_view = "최종종목: 없음"
                if _sell_only:
                    _pass2_emit_msg = "[최종 매수 종목 결정] 신규 매수 보류 (신규 뉴스 없는 사이클) — 보유 종목 매도 평가로 진행."
                picked = []
            else:
                # 과거 매매 복기 교훈(사장 지시 2026-07-31) — 최근 청산 결과를 최종 선정 판단에 재주입.
                _lessons = ""
                try:
                    from infra import trade_reflections
                    _lessons = trade_reflections.past_context(self.uid, None, n_same=0, n_cross=3)
                except Exception:
                    pass
                final_view = await self.orchestrator.think(
                    f"[최종 매수 종목 결정]\n1차 후보: {_cand_line}\n최대 매수 개수 N = {_N} (전략 설정)\n현재 세션: {session}\n"
                    f"{_budget_hint}\n\n"
                    + (f"{_standing_directive_block}\n\n" if _standing_directive_block else "")
                    + (f"[과거 매매 복기 교훈 — 최근 청산 결과, 같은 실수 반복 금지]\n{_lessons}\n\n" if _lessons else "")
                    + f"검증된 글로벌 지수(수치는 이 스냅샷에서만 그대로 인용):\n{index_facts}\n\n"
                    + f"글로벌리서치팀장 자산 배분 권고(글로벌리서치팀장의 매크로 판단 — 참고하여 반영):\n{macro_report[:600]}\n\n"
                    f"계량분석팀장 평가:\n{quant_report}\n\n마켓센티먼트팀장 평가:\n{news_report}\n\n"
                    f"[대화 흐름] 이건 이어지는 팀 회의입니다 — 첫 문장은 계량분석팀장 점수·글로벌리서치팀장 권고를 받아 잇는 인계 코멘트로 시작하고, 앞서 나온 지수·환율 수치는 다시 적지 말고 짧게 참조만 하십시오.\n"
                    f"후보 {len(candidate_codes)}개 중에서 실제로 매수할 종목을 **N개 이하**로 좁히십시오. "
                    f"퀀트 점수가 낮거나 뉴스 감성이 부정적이거나 1주 예산을 크게 넘는 종목은 빼십시오. 마땅한 게 없으면 더 적게 골라도, 0개여도 됩니다.\n"
                    f"⚠️ 분산 매수(사장 지시 2026-06-03): 비슷하게 유망한 종목이 2개 이상이면 **억지로 1개로 몰지 말고 함께 고르십시오**. "
                    f"시스템이 이번 사이클 매수 예산을 '선정 종목 수'로 나눠 각 종목에 비율 분배하므로, N개를 고르면 한 종목당 예산이 1/N로 줄어 자연히 분산됩니다 "
                    f"(한 종목 집중보다 분산이 안전). 단 확신이 한 종목에 뚜렷하게 쏠리면 1개로 좁히는 것도 정당합니다 — 유망도가 비슷할 때 굳이 1개만 버리지 말라는 뜻입니다.\n"
                    f"⚠️ **글로벌리서치팀장**의 주식 비중 권고가 확대면 N개 풀로, 축소면 N개보다 적게(또는 0개), 유지면 N의 절반~80% 수준 매수를 고려하십시오. "
                    f"이는 글로벌리서치팀장의 매크로 판단이며 사장님 지시가 아닙니다 — 매수 사유를 쓸 때 '사장님 지시/피드백에 따라 비중 축소' 같은 표현을 쓰지 말고, "
                    f"'글로벌리서치팀장 매크로 권고에 따라'로 정확히 출처를 밝히십시오. (사장님이 직접 비중 지시를 내린 게 아니면 사장님을 근거로 인용 금지)\n"
                    f"⚠️ 응답 마지막 줄은 반드시 `최종종목: ...` (후보 목록에 있던 코드만, N개 이하; 없으면 `최종종목: 없음`).")
                final_view = _strip_leading_section_marker(final_view, "[최종 매수 종목 결정]", "[후보 종목 선정]")
                self.cycle_log.log("DRAFT", "주식운용실장", final_view)
                _pass2_emit_msg = f"[최종 매수 종목 결정]\n{final_view}"
                picked = _extract_codes_after(final_view, "최종종목", "대상종목", "종목코드", "target stocks")
            cand_set = set(candidate_codes)
            target_codes = [c for c in picked if c in cand_set][:_N]
            # Enforce session boundary on final picks too (defence-in-depth)
            if session == "US_TRADING":
                target_codes = [c for c in target_codes if not (_is_kr_code(c))]
            elif is_kr_tradable(session):
                target_codes = [c for c in target_codes if _is_kr_code(c)]
            # 사장 지시 2026-06-04: MIN_QUANT_SCORE 결정론 게이트 + ① 랭크-인지 선정(MAX_BUY_NAMES) —
            # 점수 미달 제거 + 퀀트점수 상위부터 자금배정, 최대 종목수 캡(투명 로깅).
            _min_qs = int(runtime.get("MIN_QUANT_SCORE", uid=self.uid) or 0) + int(getattr(self, "_regime_qs_bump", 0))  # 레짐 방어 시 엄선(사장 지시 2026-07-21)
            _max_names = int(runtime.get("MAX_BUY_NAMES", uid=self.uid) or 0)
            if (_min_qs > 0 or _max_names > 0) and target_codes:
                target_codes, _qs_dropped = filter_targets_by_score(
                    target_codes, getattr(cyc, "_quant_scores", {}) or {}, _min_qs, max_names=_max_names)
                if _qs_dropped:
                    _qs = getattr(cyc, "_quant_scores", {}) or {}
                    _nmap = getattr(cyc, "name_map", {}) or {}
                    _dl = ", ".join(f"{_nmap.get(c, c)}({c})={_qs.get(c, '?')}" for c in _qs_dropped)
                    await self._emit({"type": "agent_msg", "agent": "주식운용실장",
                        "message": f"⛔ 선정 게이트(MIN_QUANT_SCORE={_min_qs}점·MAX_BUY_NAMES={_max_names}) — 제외/캡: {_dl}. "
                                   f"매수대상(점수순): {', '.join(target_codes) if target_codes else '없음'}."})


            # ── 사후관리실장: 보유 종목 매도 판단 ─────────────────────────────────
            # 사장 피드백 2026-05-15 (#5): 현재 세션 시장의 보유 종목이 없으면 분석 자체를 시작하지 않음.
            # KR 보유 종목이 있는데 US 세션이면 → 매도 판단 보류, 그 반대도 동일.
            _holdings_this_market = []
            for _h in (holdings or []):
                _hc = str(_h.get("code", "")).strip()
                _is_kr_holding = _is_kr_code(_hc)
                if session == "US_TRADING" and not _is_kr_holding:
                    _holdings_this_market.append(_h)
                elif is_kr_session(session) and _is_kr_holding:
                    _holdings_this_market.append(_h)
                elif session == "OFF_HOURS":
                    _holdings_this_market.append(_h)  # 장외엔 양쪽 모두 분석 가능
            sell_directives: Dict[str, str] = {}
            if _holdings_this_market:
                await self._emit({"type":"status","state":"QUANT_ANALYSIS","message":"사후관리 — 보유 종목 매도 판단"})
                _orig_holdings = holdings  # 매도 결정 후 모든 종목에 다시 사용
                holdings = _holdings_this_market  # 사후관리실장 입력은 이 시장 종목만
                # 보유기간 정보(holdings_history 기반) — 사후관리실장이 단기/중장기 판단할 때 사용
                period_lines = []
                for h in holdings:
                    hp = cycle_store.get_holding_period(h.get("code",""), uid=self.uid)
                    if hp and hp.get("days_held") is not None:
                        period_lines.append(f"  - {h.get('name',h.get('code'))}({h.get('code')}): 보유 {hp['days_held']:.1f}일 (관찰 시작 {hp['first_seen']})")
                holding_period_str = "\n".join(period_lines) if period_lines else "  (보유기간 데이터 없음 — 신규 관찰)"
                # 사장 피드백 2026-05-15 (#24): 데이트레이딩 회피 룰은 전략 설정에서 토글.
                # ALLOW_DAY_TRADING=True(기본)면 보유기간 가이드 자체를 안 보냄 — 사후관리실장이 자유롭게 판단.
                _allow_day = bool(runtime.get("ALLOW_DAY_TRADING", uid=self.uid))
                _min_hold = float(runtime.get("MIN_HOLDING_DAYS_FOR_SELL", uid=self.uid) or 0.0)
                if _allow_day:
                    _hold_guide = "보유기간은 참고만 — **데이트레이딩 허용(전략 설정)**, 신호가 명확하면 단기 매도 OK."
                else:
                    _hold_guide = (f"**보유기간 가중** — {_min_hold}일 미만은 데이트레이딩 회피, "
                                   f"7일 이상은 트렌드 점검.")
                # 사장 지시 2026-05-24: 사후관리실장 프롬프트에 현재 세션을 명시한다.
                # (버그 2026-05-23 04:47: 세션 정보가 없어 LLM이 'KR 장중이라 미국 시장 닫힘'으로
                #  환각 → US_TRADING 인데도 QUBT/UUP/DAL 매도 신호를 '매매 불가'로 무시하고 보유.)
                # 아래 보유 종목은 이미 현재 세션에서 거래 가능한 것만 필터링돼 들어온다.
                _pm_session_hint = _post_manager_session_hint(session)
                # 사장 지시 2026-05-28(우선순위 3): 펀드기획팀장 thesis 상기 — 사후관리실장이 매도 판단 직전
                # 진입 시점의 목표가/손절가/계획 보유기간을 보고 무계획 단타를 막는다.
                # 사장 지시 2026-06-09 #2: thesis 상기는 _cyc_stage_sleeves 에서 포트폴리오기획팀장이
                # 3매니저(사후관리·채권·원자재)에게 일괄 발화·보관(cyc.thesis_reminders). 여기선 주식분만 주입.
                _thesis_reminder = (getattr(cyc, "thesis_reminders", {}) or {}).get("사후관리실장", "")
                _fundamental_block = ""
                try:
                    from tools.fundamental_rigor import format_research_for_prompt
                    from infra import fundamental_research_store
                    _fr_lines = []
                    _fr_map = getattr(cyc, "fundamental_research", {}) or {}
                    for _h in (holdings or []):
                        _hc = str(_h.get("code", "")).strip()
                        if not _hc:
                            continue
                        _fr = _fr_map.get(_hc) or fundamental_research_store.latest(self.uid, _hc)
                        _fp = format_research_for_prompt(_fr)
                        if _fp:
                            _fr_lines.append(f"- {_hc}: {_fp}")
                    if _fr_lines:
                        _fundamental_block = ("\n[장기 펀더멘털 리서치 참고 — ai-berkshire 방식, advisory]\n"
                                              + "\n".join(_fr_lines) + "\n")
                except Exception as _frbe:
                    logger.debug(f"[펀더멘털리서치] 사후관리 주입 생략: {_frbe}")
                # 사장 지시 2026-06-09 #1: 슬리브(채권·원자재) 매니저의 매도 제안을 사후관리실장이 종합한다.
                _sleeve_prop_block = ""
                _sp = getattr(cyc, "sleeve_sell_proposals", {}) or {}
                if _sp:
                    _lines = []
                    for _skey, _props in _sp.items():
                        try:
                            _smgr = get_sleeve(_skey).manager_name
                        except Exception:
                            _smgr = _skey
                        for _c, _d in _props.items():
                            _lines.append(f"  - [{_smgr}] {_c} → 제안: {_d}")
                    if _lines:
                        _sleeve_prop_block = ("\n[채권·원자재 매니저 매도 제안 — 매크로·뉴스 기반]\n"
                            + "\n".join(_lines) + "\n")
                _pm_prompt = (
                    f"{_pm_session_hint}\n\n"
                    + (f"{_thesis_reminder}\n\n" if _thesis_reminder else "")
                    + f"검증된 글로벌 지수(수치는 이 스냅샷에서만 그대로 인용):\n{index_facts}\n\n"
                    + f"글로벌리서치팀장 매크로 보고:\n{macro_report}\n\n현재 보유 종목: {holdings_str}\n\n"
                    + (_fundamental_block + "\n" if _fundamental_block else "")
                    + f"보유기간 (Arquant 자체 관찰):\n{holding_period_str}\n\n"
                    f"계량분석팀장 평가:\n{quant_report}\n\n마켓센티먼트팀장 평가:\n{news_report}\n\n"
                    + _sleeve_prop_block
                    + "⚠️ 현재 보유 종목 줄의 평단·현재가·손익이 이 사이클의 권위 사실입니다. 다른 가격을 추정하거나 "
                      "과거 가격을 현재가로 쓰지 마십시오. 권위 현재가와 모순되는 가격을 근거로 만든 매도결정은 시스템이 폐기합니다.\n"
                    + f"위 매크로 → 퀀트 → 뉴스 → 평가손익 순으로 가중해 보유 종목별로 매도/보유를 결정하십시오. "
                    f"{_hold_guide} 종목별 사유에 '시장이 닫혀 매매 불가' 같은 세션 추측을 쓰지 마십시오 (위 세션 안내가 사실). "
                    + ("⚠️ 위 [장기 펀더멘털 리서치 참고]에서 QUALITY_VETO 또는 thesis invalidator가 있으면, "
                       "단기 수급이 양호하더라도 thesis 훼손 여부를 매도/보유 사유에 명시하십시오. 이 블록은 advisory지만 "
                       "장기 보유 명분을 약화시키는 증거입니다.\n"
                       if _fundamental_block else "")
                    + ("⚠️ 포트폴리오기획팀장 thesis 가 위에 명시된 종목은 진입 사유·목표가·손절가·계획 보유기간을 "
                       "**강력 권고로 우선 반영**하십시오. 목표 미달·손절 미터치·계획 기간 미경과인 종목을 미세 손익만으로 "
                       "청산하는 것은 무계획 단타입니다 — 계획 유지가 원칙이며, 그럼에도 매도하려면 명확한 신호 변화 이유를 "
                       "사유에 반드시 적으십시오. (단 최종 매도 권한은 사후관리실장 본인에게 있습니다.)\n"
                       if _thesis_reminder else "")
                    + ("⚠️ 위 [채권·원자재 매니저 매도 제안]을 **주식 매도와 함께 종합**하십시오 — 비중%가 적정하다는 "
                       "이유만으로 무조건 보류하지 말고, 신호로 판단해 매도결정 라인에 해당 슬리브 코드도 포함(매도/절반/보유)하십시오.\n"
                       if _sleeve_prop_block else "")
                    + f"마지막 줄은 반드시 `매도결정: 코드=전량/절반/보유, ...` (보유 종목 + 슬리브 제안 코드 전체).")
                # 사장 지시 2026-05-22: 빈 응답/에러 시 재시도 (계량분석팀장과 동일 패턴) —
                # 직전 사이클에서 사후관리실장이 'API 응답 비어있음'으로 매도 평가가 통째 빠진 버그.
                pm_view = ""
                for _pm_attempt in range(3):
                    try:
                        pm_view = await self.post_manager.think(_pm_prompt)
                    except Exception as _pe:
                        pm_view = f"[사후관리실장 에러] {_pe}"
                    if pm_view and pm_view.strip() and "에러" not in pm_view[:80] and "응답 비어" not in pm_view:
                        break
                    if _pm_attempt < 2:
                        await asyncio.sleep(1.0)
                self.cycle_log.log("RISK", "사후관리실장", pm_view)
                await self._emit({"type":"agent_msg","agent":"사후관리실장","message":pm_view})
                sell_directives = _parse_sell_decisions(pm_view)
                sell_directives, _price_conflicts = sanitize_sell_directives_by_authoritative_prices(
                    pm_view, sell_directives, holdings)
                if _price_conflicts:
                    _facts = "; ".join(
                        f"{x['code']} LLM {x['claimed']:,.2f} vs 실제 {x['actual']:,.2f}"
                        for x in _price_conflicts)
                    logger.warning("[매도사실충돌 uid=%s] %s — LLM 매도지시 보유로 격리", self.uid, _facts)
                    await self._emit({"type": "agent_msg", "agent": "시스템",
                                      "message": f"🛡️ 매도 근거 가격 불일치 차단 — {_facts}. "
                                                 "자동 익절·손절은 실제 손익 기준으로 별도 적용합니다."})
                cyc._pm_view = pm_view  # QuantInSight 이식(2026-07-18): 위원회 '보유 심의' 기록용 원문
                # 사장 지시 2026-06-09 #1: 사후관리실장이 슬리브(채권·원자재) 코드를 매도결정에 포함하면
                # 그대로 *종합*한다(구 strip 폐지). 슬리브 코드는 주식 매도 트랙(stock_holdings 기반)엔 안 걸리고,
                # build_orders 가 슬리브 매도로 따로 조립한다 — 사후관리실장이 주식+슬리브 최종 종합권.
                holdings = _orig_holdings  # 후속 처리(주문 조립·이력 기록)는 주식 보유(슬리브 제외 시 stocks-only)
                # 사장 지시 2026-06-08: 포트폴리오기획팀장 '거부권'(매도결정을 '보유'로 강제 오버라이드)은
                # 권한이 과도하여 폐지했다. thesis 는 위 사후관리실장 프롬프트에 '강력 권고'로 주입만 하고
                # (format_thesis_reminder), 사후관리실장의 매도결정을 그대로 존중한다.
                # 사장 피드백 2026-05-15 (#5, #13): 반대편 시장 보유 종목은 자동 '보유' 처리.
                # 사후관리실장이 분석을 안 했으므로 _build_orders의 자동 익절/손절 룰이 잘못 매도하는 것 방지.
                _existing_lower = {k.lower() for k in sell_directives.keys()}
                for _h in (_orig_holdings or []):
                    _hc = str(_h.get("code", "")).strip()
                    if not _hc or _hc.lower() in _existing_lower:
                        continue
                    _is_kr_h = _is_kr_code(_hc)
                    _opposite = ((session == "US_TRADING" and _is_kr_h) or
                                 (is_kr_session(session) and not _is_kr_h))
                    if _opposite:
                        sell_directives[_hc] = "보유"
            elif holdings:
                # 반대편 시장 보유만 있는 상황 — 사장 피드백 2026-05-15 (#5, #13): 분석 생략 + 전체 자동 '보유'.
                _other_mkt = "US" if is_kr_session(session) else "KR"
                await self._emit({"type":"agent_msg","agent":"사후관리실장",
                    "message": f"📭 현재 세션({session})의 보유 종목 없음 — 분석 생략 ({_other_mkt} 보유분은 해당 시장 사이클에서 처리)."})
                for _h in (holdings or []):
                    _hc = str(_h.get("code", "")).strip()
                    if _hc:
                        sell_directives[_hc] = "보유"
            else:
                # 사장 지시 2026-06-11: 사후관리실장은 *주식*만 다룬다 — 채권·원자재 슬리브 ETF를
                # 보유 중이어도 여기 holdings(주식)는 비어 "보유 종목 없음"으로 오인됐다. 범위를 명시.
                await self._emit({"type":"agent_msg","agent":"사후관리실장",
                    "message":"보유 주식 없음 — 매도 판단 불필요 (채권·원자재 ETF는 각 운용실장이 별도 관리)"})
            # 사장 지시 2026-06-03(흐름): 보유 점검(사후관리·펀드기획)이 끝난 뒤에야 최종 매수 결정을 노출한다.
            if _pass2_emit_msg:
                await self._emit({"type":"agent_msg","agent":"주식운용실장","message":_pass2_emit_msg})
            cyc.target_codes = target_codes
            # 사장 지시 2026-06-09 #1: 슬리브 매도 제안을 최종 종합. post_manager 가 다룬 슬리브 코드는
            # 그 결정이 우선(override), 사후관리실장이 안 다룬 슬리브 코드는 매니저 제안을 그대로 반영(누락 방지).
            # post_manager 미실행(주식 보유 없음) 사이클에서도 슬리브 매도가 살아남는다.
            _sleeve_flat = {c: d for _props in (getattr(cyc, "sleeve_sell_proposals", {}) or {}).values()
                            for c, d in _props.items()}
            sell_directives = {**_sleeve_flat, **sell_directives}
            if getattr(cyc, "systematic_policy", None) is not None:
                from tools.stock_policy import exit_proposals
                _policy_exits = exit_proposals(cyc.systematic_policy, _full_holdings or [],
                    total_eval=float(getattr(cyc, "total_eval", 0) or 0), usdkrw=get_usdkrw(USDKRW_FALLBACK),
                    is_sleeve=lambda c: sleeve_for_code(c) is not None)
                sell_directives.update(_policy_exits)
                if _policy_exits:
                    await self._emit({"type": "agent_msg", "agent": "사후관리실장",
                        "message": "정량 운용안 교체·비중 축소 제안: " + ", ".join(f"{c}={v}" for c,v in _policy_exits.items())})
            cyc.sell_directives = sell_directives
            # cyc.holdings 는 항상 *전체* 보유(_full_holdings)를 유지(_cyc_stage_sleeves 가 이미 실행됐고,
            # build_orders 가 슬리브 매도 조립에 cyc.holdings 를 쓴다). 주식 매도 트랙용으론 cyc.stock_holdings
            # (슬리브 제외)를 따로 둔다(슬리브 ON 일 때만; OFF 면 미설정 → build_orders 가 cyc.holdings 전체 사용).
            cyc.holdings = _full_holdings
            if _bond_etf_on:
                cyc.stock_holdings = holdings  # 슬리브 제외(주식만) — _build_orders 매도 트랙 입력
            cyc._exec_N = _exec_N

    def _enabled_sleeve_codes(self):
        """현재 활성(enable=ON) 슬리브들의 풀 코드 합집합(대문자). 코드리뷰 #5(2026-06-09):
        주식 매도 트랙 제외/슬리브 매도 조립은 *활성* 슬리브만 대상으로 해야 한다 — OFF 슬리브의
        보유 ETF 는 슬리브 트랙이 스킵하므로, 활성 코드만 제외하면 OFF 슬리브 보유분이 주식
        매도 트랙(자동 익절/손절·사후관리 판단)으로 폴백돼 orphan(청산경로 전무)을 면한다."""
        out = set()
        for spec in SLEEVES:
            try:
                _on = bool(runtime.get(spec.enable_key, uid=self.uid))
            except Exception:
                _on = True  # uid 비정상 등 runtime 오류 → 보수적으로 활성 취급(주식 트랙서 제외, 기존 동작)
            if _on:
                out |= sleeve_codes(spec)
        return out

    def _sleeve_rt(self, key):
        """슬리브 수치 파라미터를 None-safe 로 읽는다(코드리뷰 #2). runtime override 의 정당한 0.0
        (비중 0%·데드존 0 등)을 `or DEFAULT`가 둔갑시키던 falsy-zero 버그 방지 — None 일 때만
        config 모듈 기본값(슬리브별 정확값: 채권 0.40/0.15, 원자재 0.20/0.10)으로 폴백."""
        import config as _cfg
        v = runtime.get(key, uid=self.uid)
        if v is None:
            v = getattr(_cfg, key, 0.0)
        return float(v)

    def _collect_thesis_reminders(self, cyc):
        """포트폴리오기획팀장 보유계획 일괄 상기(사장 지시 2026-06-09 #2) — 사후관리실장(주식)·
        채권운용실장(채권)·원자재운용실장(원자재) 각각에게 진입 thesis 를 강력 권고로 상기.
        반환 {manager_name: reminder_text}. 호출부가 단일 발화로 emit + 각 프롬프트에 주입."""
        from infra import sleeve_thesis, position_thesis
        from agents.specialists import format_thesis_reminder, format_sleeve_thesis_reminder
        now = _now_kst_iso()
        holdings = getattr(cyc, "holdings", None) or []
        out: Dict[str, str] = {}
        # 주식 (사후관리실장) — 목표가/손절가 포함 thesis
        try:
            stock_h, _ = split_sleeve_holdings(holdings, all_sleeve_pool_codes())
            st = position_thesis.get_all(self.uid)
            if st and stock_h:
                r = format_thesis_reminder(st, stock_h, now)
                if r:
                    out["사후관리실장"] = r
        except Exception as _e:
            logger.warning(f"[보유계획상기] 주식 thesis 실패: {_e}")
        # 슬리브 (채권·원자재) — 보유기간 thesis
        for spec in SLEEVES:
            try:
                _, sh = split_sleeve_holdings(holdings, sleeve_codes(spec))
                th = sleeve_thesis.get_all(self.uid, spec.key)
                if th and sh:
                    r = format_sleeve_thesis_reminder(th, sh, now, manager_name=spec.manager_name)
                    if r:
                        out[spec.manager_name] = r
            except Exception as _e:
                logger.warning(f"[보유계획상기] {spec.key} thesis 실패: {_e}")
        return out

    async def _cyc_stage_sleeves(self, cyc):
        """자산슬리브 트랙(채권·원자재) — 매크로 권고 비중을 ETF 매수/매도로 실현(사장 지시 2026-06-09).
        finalize_sell '앞'에서 실행. 매니저는 **매크로+뉴스**로 판단(주식 퀀트 제외):
          - 매수(저비중)는 자산배분으로 즉시 집행 → cyc.sleeve_buy_orders.
          - 매도(신호 악화)는 *제안*만 → cyc.sleeve_sell_proposals(사후관리실장이 주식과 종합).
        ENABLE_*_ETF=OFF 슬리브는 스킵. 포트폴리오기획팀장 보유계획 일괄 상기도 여기서 발화."""
        # 포트폴리오기획팀장 보유계획 일괄 상기(한 번에) — 각 매니저 프롬프트 주입용으로 보관
        cyc.thesis_reminders = self._collect_thesis_reminders(cyc)
        if cyc.thesis_reminders:
            _who = " · ".join(cyc.thesis_reminders.keys())
            _body = "\n\n".join(cyc.thesis_reminders.values())
            await self._emit({"type": "agent_msg", "agent": "포트폴리오기획팀장",
                "message": f"📌 [보유계획 일괄 상기] {_who}께 매수 때 세운 계획을 다시 한 번 상기시킵니다.\n\n{_body}"})
        session = cyc.session
        usdkrw = get_usdkrw(USDKRW_FALLBACK)
        total_eval = float(getattr(cyc, "total_eval", 0.0) or 0.0)
        # 매매(매수/매도)용 보유 = 현재 세션 시장(KR 세션→KR, US→해외) — 그 세션에서만 거래 가능.
        if session == "US_TRADING":
            try:
                holdings = await self.broker._overseas_holdings()
            except Exception:
                holdings = getattr(cyc, "holdings", None) or []
            if not holdings:
                holdings = getattr(cyc, "holdings", None) or []
        else:
            holdings = getattr(cyc, "holdings", None) or []
        # 코드리뷰 수정 #1(2026-06-09): 비중 측정은 *양 시장 전체* 슬리브 보유로 해야 한다.
        # (편측 세션 보유만 numerator로 쓰면 반대편 시장 슬리브가 안 보여 cur_w 과소→*_TARGET_MAX 초과 매수.)
        # total_eval 은 KR+US 전체이므로 numerator 도 전체여야 일관. US세션 cyc.holdings 는 이미 KR+US,
        # KR세션 cyc.holdings 는 KR-only 라 해외분을 best-effort 합산(실패 시 그대로).
        _full_h = list(getattr(cyc, "holdings", None) or [])
        if session != "US_TRADING":
            try:
                _full_h = _full_h + list(await self.broker._overseas_holdings() or [])
            except Exception:
                pass
        for spec in SLEEVES:
            mgr_name = spec.manager_name
            if not bool(runtime.get(spec.enable_key, uid=self.uid)):
                continue
            try:
                pool = sleeve_pool_for_session(
                    spec, session, us_allowed=bool(runtime.get("ALLOW_US_STOCKS", uid=self.uid)))
                if not pool:
                    continue
                pool_codes = [c for c, *_ in pool]
                _, sleeve_h = split_sleeve_holdings(holdings, pool_codes)
                cyc.sleeve_holdings_by_key[spec.key] = sleeve_h
                rec = parse_macro_sleeve_pct(getattr(cyc, "macro_report", "") or "", spec.macro_keyword)
                # 비중은 양 시장 전체 슬리브 풀(sleeve_codes)·전체 보유(_full_h)로 측정(코드리뷰 #1).
                cur_w = current_sleeve_weight(_full_h, total_eval, sleeve_codes(spec), usdkrw)
                # 코드리뷰 #2: runtime override 의 정당한 0.0(예: 비중 0%·데드존 0)을 `or DEFAULT`가
                # 둔갑시키지 않도록 None 일 때만 config 기본값(슬리브별 정확값)으로 폴백.
                action, notional = size_sleeve_action(
                    rec, cur_w, total_eval,
                    self._sleeve_rt(spec.target_max_key), self._sleeve_rt(spec.band_key))
                _rec_s = f"{rec*100:.1f}%" if rec is not None else "권고없음"
                # 실효 목표(상한 클램프 반영) — 데드존 비교는 min(권고, 상한) 기준이므로 메시지도 이 값을
                # 보여야 한다. 종전엔 원권고(예 60%)만 표시해 '42%가 60% 데드존 이내'라는 모순 인상을 줬다(사장 지시 2026-07-21).
                _tmax_v = float(self._sleeve_rt(spec.target_max_key) or 0.0)
                _eff_tgt = min(float(rec), _tmax_v) if rec is not None else None
                _tgt_s = ("권고없음" if _eff_tgt is None else
                          (f"목표 {_eff_tgt*100:.1f}%" +
                           (f"(권고 {_rec_s}→상한 {_tmax_v*100:.0f}% 적용)" if _eff_tgt < float(rec) - 1e-9 else "")))
                # 매도 평가는 보유가 있으면 밴드 안이어도 항상 수행(사장 지시 #1). 매수는 action=='buy'만.
                if action != "buy" and not sleeve_h:
                    _why = (f"매크로 {spec.macro_keyword} 권고 없음·보유 없음 → 트랙 스킵" if action == "skip"
                            else f"{_tgt_s}·현재 비중 {cur_w*100:.1f}% 데드존 이내 + 보유 없음 → 매매 보류")
                    await self._emit({"type": "agent_msg", "agent": mgr_name, "message": f"🟦 {_why}."})
                    self.cycle_log.log("DRAFT", mgr_name, _why)
                    continue
                # 매수 예산 cap(슬리브 전용 사이클 예산·예수금) — 리스크검증 게이트 면제분 사이징 단계서 제한.
                if action == "buy":
                    _pcr = self._sleeve_rt(spec.per_cycle_key)  # 코드리뷰 #2: 0.0 마스킹 방지
                    _cash = float(getattr(cyc, "cash", 0.0) or 0.0)
                    _mcb = float(runtime.get("MIN_CASH_BUFFER", uid=self.uid) or 1.0)
                    _uncapped = notional
                    notional = cap_sleeve_buy_notional(notional, total_eval, _cash, _pcr, _mcb)
                    if notional < _uncapped:
                        self.cycle_log.log("DRAFT", mgr_name,
                            f"{spec.macro_keyword} 매수예산 cap: {_uncapped:,.0f}→{notional:,.0f}원 (전용비율·예수금)")
                _pool_lines = "\n".join(f"  - {c} {n} ({d}·{k})" for c, n, d, k, *_ in pool)
                _act_kr = ("부족분 매수 + 보유 매도평가" if action == "buy"
                           else ("초과분 매도평가" if action == "sell" else "보유 매도평가"))
                _weight_ctx = (f"권고 {spec.macro_keyword}비중 {_rec_s} / 현재 평가비중 {cur_w*100:.1f}% "
                               f"→ 조치: **{_act_kr}**" + (f" (매수예산 약 {notional:,.0f}원)" if action == "buy" else ""))
                _prompt = _build_sleeve_prompt(
                    spec, getattr(cyc, "macro_report", "") or "", getattr(cyc, "news_report", "") or "",
                    _pool_lines, _weight_ctx, cyc.thesis_reminders.get(mgr_name, ""),
                    holdings_txt=format_sleeve_holdings_block(sleeve_h))
                manager = self._sleeve_managers.get(spec.role)
                view = ""
                for _attempt in range(3):
                    try:
                        view = await manager.think(_prompt)
                    except Exception as _be:
                        view = f"[{mgr_name} 에러] {_be}"
                    if view and view.strip() and "에러" not in view[:80] and "응답 비어" not in view:
                        break
                    if _attempt < 2:
                        await asyncio.sleep(1.0)
                self.cycle_log.log("DRAFT", mgr_name, view)
                await self._emit({"type": "agent_msg", "agent": mgr_name, "message": view})
                decisions = parse_sleeve_decisions(view, spec.decision_keyword, pool_codes)
                # 사장 지시 2026-07-20: 매니저가 명시한 매수 배분 비중(코드=매수:비중) 파싱 → 조립부에 전달.
                sleeve_weights = parse_sleeve_weights(view, spec.decision_keyword, pool_codes)
                _pool_name = {str(c).strip().upper(): n for c, n, *_ in pool}
                # 가격 사전조회(결정 코드 + 보유) → build_orders 가 price_map 합류(미주입 시 매수/매도 반려).
                _need = set(decisions.keys()) | {str(h.get("code", "")).strip().upper() for h in (sleeve_h or [])}
                for code in _need:
                    try:
                        px = (await self.broker.kr_last_price(code) if _is_kr_code(code)
                              else await self.broker.us_last_price(code))
                        cyc.sleeve_price_map[code] = float(px or 0.0)
                    except Exception as _pe:
                        logger.warning(f"[{spec.key}] 가격 조회 실패 {code}: {_pe}")
                        cyc.sleeve_price_map[code] = 0.0
                # 매수: 자산배분으로 즉시 집행. 매도: 사후관리실장 종합용 *제안*만 보관.
                _has_buy_dir = any(str(d).strip() == "매수" for d in decisions.values())
                _has_sell_dir = any(str(d).strip() not in ("매수", "보유") for d in decisions.values())
                _exec_buy, _is_rotation = should_execute_sleeve_buy(action, _has_buy_dir, _has_sell_dir)
                if _exec_buy:
                    if _is_rotation:
                        # 사장 지시 2026-06-12(Q4): 데드존(hold/sell)이어도 매도+매수 동시 제안은
                        # '비중 유지 회전'(순비중은 매도로 상쇄) → 매수를 사이클 전용 예산까지 허용(Option B).
                        _pcr = self._sleeve_rt(spec.per_cycle_key)   # 0.0 마스킹 방지(_sleeve_rt)
                        _cash = float(getattr(cyc, "cash", 0.0) or 0.0)
                        _mcb = float(runtime.get("MIN_CASH_BUFFER", uid=self.uid) or 1.0)
                        notional = cap_sleeve_buy_notional(total_eval, total_eval, _cash, _pcr, _mcb)
                        await self._emit({"type": "agent_msg", "agent": mgr_name,
                            "message": (f"🔁 {spec.macro_keyword} 비중 유지 회전 — 현재 평가비중 {cur_w*100:.1f}%가 "
                                        f"데드존(±{self._sleeve_rt(spec.band_key)*100:.0f}%p) 이내이나 매도+매수 동시 제안이라 "
                                        f"매수 집행(예산 약 {notional:,.0f}원, 매도로 순비중 상쇄).")})
                        self.cycle_log.log("DRAFT", mgr_name,
                            f"{spec.macro_keyword} 비중유지 회전 매수 집행(데드존, 매도동반): 예산 {notional:,.0f}원")
                    buys = assemble_sleeve_orders(
                        spec, "buy", notional, decisions, sleeve_h, cyc.sleeve_price_map.get,
                        usdkrw, weights=sleeve_weights)
                    cyc.sleeve_buy_orders.extend(buys)
                    # 위원회 슬리브 심의(_cyc_stage_committee)용 메타 — 매니저 논거·라벨·비중맥락 보관.
                    for _bo in buys:
                        _bc = str(_bo.get("ticker", "")).strip().upper()
                        cyc.sleeve_buy_meta[_bc] = {
                            "name": _pool_name.get(_bc, _bc),
                            "sleeve_label": spec.macro_keyword,
                            "chief_label": mgr_name,
                            "rationale": view,
                            "weight_ctx": _weight_ctx,
                            "sleeve_key": spec.key,
                        }
                    if not buys and _has_buy_dir:
                        await self._emit({"type": "agent_msg", "agent": mgr_name,
                            "message": f"⚠️ {spec.macro_keyword} 매수 0건 — 예산<1주 단가 또는 가격조회 실패(다음 사이클 이월)."})
                elif _has_buy_dir:
                    # 사장 지시 2026-06-11(투명성): 매니저가 매수 의견을 냈으나 사이징이 hold/sell
                    # (현재 평가비중이 권고 대비 리밸런싱 데드존 이내 등)이고 매도 동반도 없어 매수 예산 0 →
                    # 집행 안 됨. 종전엔 아무 안내 없이 조용히 누락돼 "매수한다더니 왜 안 사?"로 보였다.
                    _bp = [c for c, d in decisions.items() if str(d).strip() == "매수"]
                    await self._emit({"type": "agent_msg", "agent": mgr_name,
                        "message": (f"ℹ️ {spec.macro_keyword} 매수 의견({', '.join(_bp)})이나 현재 평가비중 "
                                    f"{cur_w*100:.1f}%가 {_tgt_s} 데드존(±{self._sleeve_rt(spec.band_key)*100:.0f}%p) "
                                    f"이내 → 리밸런싱 매수 보류(이미 목표 비중 근접·매도 동반 없음, 추가 매수 불필요).")})
                    self.cycle_log.log("DRAFT", mgr_name,
                        f"{spec.macro_keyword} 매수의견 보류(데드존): 현재 {cur_w*100:.1f}% vs {_tgt_s}")
                _sell_dirs = {c: d for c, d in decisions.items()
                              if str(d).strip() not in ("매수", "보유")}
                if _sell_dirs:
                    cyc.sleeve_sell_proposals[spec.key] = _sell_dirs
            except Exception as e:
                logger.warning(f"[{spec.key}] 슬리브 스테이지 오류: {e}")
                await self._emit({"type": "agent_msg", "agent": mgr_name,
                                  "message": f"⚠️ {spec.macro_keyword} 트랙 오류 — 이번 사이클 생략: {e}"})

    async def _cyc_stage_committee(self, cyc):
            """QuantInSight 위원회 심의 이식 (사장 지시 2026-07-18) — fail-open.

            PASS 2 매수 대상별로: 출처검증 리포트 → 찬반토론(매수 심사역↔보유 심사역, 라운드
            고정) → 주식운용실장 최종 결정(+claims 자진신고) → 결정론 게이트(체크리스트).
            차단/회피/보류 판정 종목은 target_codes 에서 투명하게 제외한다. 보유 종목은
            사후관리실장의 매도결정을 '보유 심의' 기록으로 재구성한다(추가 LLM 없음).
            어떤 실패도 사이클을 막지 않는다 — 실패 시 심의 기록만 생략(매매 파이프라인 불변)."""
            try:
                from agents import committee as cmt
                targets = list(getattr(cyc, "target_codes", None) or [])
                holdings = (cyc.stock_holdings if getattr(cyc, "stock_holdings", None) is not None
                            else getattr(cyc, "holdings", None)) or []
                sell_directives = getattr(cyc, "sell_directives", {}) or {}
                # 슬리브 매수가 있으면 주식 target/보유가 없어도 위원회를 연다(슬리브 찬반토론 위해).
                # 2026-07-29: 개회 판정도 *전체* 보유 기준 — 보유가 전부 슬리브 ETF 인 계정(모의)은
                # stock_holdings 가 [] 라 여기서 통째로 return 돼 '보유 심의' 카드가 안 떴다.
                if (not targets and not (getattr(cyc, "holdings", None) or [])
                        and not (getattr(cyc, "sleeve_buy_orders", None) or [])):
                    return
                name_map = getattr(cyc, "name_map", {}) or {}
                per_dart = getattr(cyc, "per_dart", {}) or {}
                q_scores = getattr(cyc, "_quant_scores", {}) or {}
                fr_map = getattr(cyc, "fundamental_research", {}) or {}
                macro_report = getattr(cyc, "macro_report", "") or ""
                _alloc = _parse_macro_stock_pct(macro_report)
                macro_view = " ".join(macro_report.replace("\n", " ").split())[:220]
                _min_qs = float(runtime.get("MIN_QUANT_SCORE", uid=self.uid) or 0) + float(getattr(self, "_regime_qs_bump", 0))  # 레짐 방어 엄선
                _cyc_ratio = float(runtime.get("MAX_CYCLE_BUDGET_RATIO", uid=self.uid) or 0.25) * float(getattr(self, "_regime_budget_mult", 1.0))
                _base_w = _cyc_ratio / max(1, len(targets) or 1)

                async def _prog(agent, msg):
                    await self._emit({"type": "agent_msg", "agent": agent, "message": msg})

                # 과거 매매 복기 주입(사장 지시 2026-07-31) — 같은 종목 청산 결과·교훈을 심의 브리핑에.
                def _tr_past(_code):
                    try:
                        from infra import trade_reflections
                        return trade_reflections.past_context(self.uid, _code)
                    except Exception:
                        return ""

                # 기업리서치팀장 (사장 지시 2026-07-21) — 마켓센티먼트팀장 다음·계량분석팀장 이후,
                # 개별 '기업' 매수 대상만(ETF·슬리브 제외) 최신 기업 분석자료를 종합해 위원회 평가에 추가한다.
                # fail-open: 실패해도 심의는 계속(실주문 파이프라인 불변).
                cyc.company_research = getattr(cyc, "company_research", {}) or {}
                _insight_map: Dict[str, str] = {}
                for code in targets:
                    name = name_map.get(code, code)
                    if not _is_company_code(code, name):
                        continue
                    try:
                        self.insight_analyst.reset_history()
                        _fr = fr_map.get(code) or {}
                        _fr_line = (f"판정 {_fr.get('verdict')} · 품질 {_fr.get('business_quality_score')}"
                                    f"·해자 {_fr.get('moat_score')}·경영 {_fr.get('management_score')}"
                                    f"·밸류 {_fr.get('valuation_margin_score')}") if _fr.get("verdict") else ""
                        _q_ex0 = _quant_ctx_for(getattr(cyc, "quant_report", ""), code, width=400)
                        _news0 = _extract_code_news(cyc.news_report, code, name)[:400]
                        _ins = await self.insight_analyst.think(
                            f"[기업 분석 대상] {name}({code})\n\n"
                            f"계량분석팀장 퀀트 요약:\n{_q_ex0 or '(요약 없음)'}\n\n"
                            f"마켓센티먼트팀장 뉴스 발췌:\n{_news0 or '(뉴스 없음)'}\n\n"
                            f"최근 DART 공시:\n{(per_dart.get(code, '') or '(공시 없음)')[:1200]}\n\n"
                            f"장기 펀더멘털 리서치 스냅샷: {_fr_line or '(스냅샷 없음)'}\n\n"
                            "위 자료로 이 기업의 투자 매력과 리스크를 종합한 '기업 분석 리포트'를 응답 형식대로 작성하십시오.")
                        _ins = (_ins or "").strip()
                        if _ins and not _ins.startswith("[기업리서치팀장 에러"):
                            _insight_map[code] = _ins
                            cyc.company_research[code] = _ins
                            await self._emit({"type": "agent_msg", "agent": "기업리서치팀장",
                                              "message": f"[{name}({code}) 기업 분석]\n{_ins}"})
                    except Exception as _ie:  # noqa: BLE001
                        logger.warning(f"[기업리서치] {code} 생략(fail-open): {_ie}")

                candidates, dropped = [], []   # dropped: (code, 사유)
                _DEBATE_CAP = 3   # LLM 토론은 상위 N종목까지 — 그 이하는 결정론 심의(게이트는 전 종목)
                # 보유 매도 심의도 같은 이유로 상한을 둔다(종목당 LLM 5턴). 초과분은 사후관리실장
                # 1차 판단을 그대로 쓴다 — 매도 자체가 누락되지는 않는다.
                _SELL_DEBATE_CAP = 4
                for i, code in enumerate(targets):
                    name = name_map.get(code, code)
                    _ins_ex = _insight_map.get(code, "")
                    rep = cmt.build_report(
                        code, name,
                        quant_line=(f"결정론 퀀트점수 {q_scores.get(code)}/10"
                                    if q_scores.get(code) is not None else ""),
                        per_dart=per_dart.get(code, ""),
                        news_excerpt=_extract_code_news(cyc.news_report, code, name)[:300],
                        fundamental=fr_map.get(code), insight_excerpt=_ins_ex)
                    q_ex = _quant_ctx_for(getattr(cyc, "quant_report", ""), code, width=300)

                    # ── 컴플라이언스실장 (사장 지시 2026-07-22) ──────────────────────
                    # 구 수탁자책임실 반려 기능 부활. 결정론 판정이며 **토론 이전**에 돌려
                    # 반려 종목엔 LLM 턴을 낭비하지 않는다. 어떤 실패에도 fail-open.
                    try:
                        _note = compliance_mod.honesty_note(rep.guardrail_log, rep.unverified_ratio)
                        if _note:
                            await self._emit({"type": "agent_msg",
                                              "agent": compliance_mod.NAME,
                                              "message": f"[{name}({code})] {_note}"})
                        _comp = compliance_mod.screen(code, name, sector=rep.sector,
                                                      dart_text=per_dart.get(code, ""))
                    except Exception as _ce:  # noqa: BLE001
                        logger.warning(f"[컴플라이언스] {code} 심사 생략(fail-open): {_ce}")
                        _comp = None
                    if _comp is not None and _comp.rejected:
                        _why = "; ".join(_comp.reasons)
                        await self._emit({"type": "agent_msg", "agent": compliance_mod.NAME,
                            "message": f"⛔ [{name}({code})] 컴플라이언스 반려 — {_why}\n"
                                       f"수탁자책임·정책 적합성 위반으로 매수 심의를 열지 않습니다."})
                        dropped.append((code, f"컴플라이언스 반려: {_why}"))
                        candidates.append({
                            "code": code, "name": name, "sector": rep.sector,
                            "quant_score": q_scores.get(code), "decision": "컴플라이언스 반려",
                            "engine": "deterministic", "report": rep.to_dict(),
                            "opinions": [], "dialogue": [],
                            "gate": {"verdict": cmt.BLOCK, "final_weight": 0.0,
                                     "checks": [c for c in _comp.to_dict()["checks"]]},
                            "compliance": _comp.to_dict(),
                        })
                        continue

                    if i < _DEBATE_CAP:
                        await self._emit({"type": "agent_msg", "agent": "주식운용실장",
                            "message": f"🏛️ [{name}({code})] 위원회 심의 소집 — 찬반토론 후 최종 확정합니다."})
                        opinions, dialogue, chief, llm_used = await cmt.deliberate_target(
                            code, name, rep, quant_score=q_scores.get(code),
                            quant_excerpt=q_ex,
                            news_excerpt=_extract_code_news(cyc.news_report, code, name)[:300],
                            macro_view=macro_view, insight_excerpt=_ins_ex,
                            past_context=_tr_past(code), progress=_prog)
                    else:
                        opinions, dialogue, chief, llm_used = await cmt.deliberate_target(
                            code, name, rep, quant_score=q_scores.get(code),
                            quant_excerpt=q_ex, news_excerpt="", macro_view=macro_view,
                            insight_excerpt=_ins_ex, past_context=_tr_past(code), progress=None)
                    gate = cmt.run_gate(rep, q_scores.get(code), chief["confidence"],
                                        min_quant_score=_min_qs, base_weight=_base_w)
                    if gate.verdict == cmt.BLOCK:
                        decision = "차단"
                        dropped.append((code, "게이트 차단: " + "; ".join(
                            c.detail for c in gate.checks if not c.passed)))
                    elif chief["stance"] != cmt.BUY:
                        decision = chief["stance"]
                        dropped.append((code, f"주식운용실장 심의 결정 {chief['stance']}"))
                    else:
                        decision = "매수" if gate.verdict == cmt.PASS_V else "매수(비중축소)"
                    candidates.append({
                        "code": code, "name": name, "sector": rep.sector,
                        "quant_score": q_scores.get(code), "decision": decision,
                        "engine": ("llm" if llm_used else "deterministic"),
                        "report": rep.to_dict(),
                        "opinions": opinions, "dialogue": dialogue,
                        "gate": gate.to_dict(),
                    })
                if dropped:
                    _dset = {c for c, _ in dropped}
                    cyc.target_codes = [c for c in targets if c not in _dset]
                    await self._emit({"type": "agent_msg", "agent": "리스크관리실장",
                        "message": "🚦 위원회 게이트 — 매수 제외: " + "; ".join(
                            f"{name_map.get(c, c)}({c}) — {r}" for c, r in dropped)
                            + f"\n최종 매수 대상: {', '.join(cyc.target_codes) or '없음'}"})

                # ── 슬리브(채권·원자재) 매수 위원회 심의 (사장 지시 2026-07-20) ──
                # 종전엔 채권·원자재 ETF 매수가 매니저 단독 판단으로 나갔다 — 주식과 동일하게
                # 매수 심사역↔보유 심사역 찬반토론 후 슬리브운용실장이 최종 확정한다.
                # 보류/회피 판정 ETF는 cyc.sleeve_buy_orders 에서 투명하게 제외한다.
                sleeve_orders = list(getattr(cyc, "sleeve_buy_orders", None) or [])
                sleeve_meta = getattr(cyc, "sleeve_buy_meta", {}) or {}
                if sleeve_orders:
                    _SLEEVE_DEBATE_CAP = 4   # LLM 토론은 상위 N건까지(그 이하는 매니저 결정 유지)
                    kept_orders, sleeve_cands, sleeve_dropped = [], [], []
                    for _j, _od in enumerate(sleeve_orders):
                        _sc = str(_od.get("ticker", "")).strip().upper()
                        _meta = sleeve_meta.get(_sc, {})
                        _slabel = _meta.get("sleeve_label", "자산배분")
                        _chief = _meta.get("chief_label", "운용실장")
                        _sname = _meta.get("name", _sc)
                        if _j < _SLEEVE_DEBATE_CAP:
                            await self._emit({"type": "agent_msg", "agent": _chief,
                                "message": f"🏛️ [{_sname}({_sc})] {_slabel} 매수 위원회 심의 소집 — 찬반토론 후 확정합니다."})
                            _dialogue, _decision, _llm_used = await cmt.deliberate_sleeve_buy(
                                _sc, _sname, sleeve_label=_slabel, chief_label=_chief,
                                macro_view=macro_view, manager_rationale=_meta.get("rationale", ""),
                                weight_ctx=_meta.get("weight_ctx", ""),
                                price=float(_od.get("price") or 0.0), progress=_prog)
                        else:
                            _dialogue, _llm_used = [], False
                            _decision = {"stance": cmt.BUY, "confidence": 0.6,
                                         "text": f"{_chief} 결정 유지 — 토론 캡 초과분(결정론)."}
                        if _decision["stance"] == cmt.BUY:
                            kept_orders.append(_od)
                            _sdec = "매수"
                        else:
                            sleeve_dropped.append((_sc, _sname, _decision["stance"]))
                            _sdec = _decision["stance"]
                        sleeve_cands.append({
                            "code": _sc, "name": _sname, "sector": _slabel, "track": _slabel,
                            "quant_score": None, "decision": _sdec,
                            "engine": ("llm" if _llm_used else "deterministic"),
                            "report": {"code": _sc, "name": _sname, "sector": _slabel,
                                       "facts": [], "red_flags": [],
                                       "credit_view": "자산배분 ETF — 개별 신용 스캔 대상 아님",
                                       "unverified_ratio": 0.0, "guardrail_log": []},
                            "opinions": [], "dialogue": _dialogue, "gate": {},
                        })
                    cyc.sleeve_buy_orders = kept_orders
                    if sleeve_dropped:
                        await self._emit({"type": "agent_msg", "agent": "리스크관리실장",
                            "message": "🚦 위원회 — 슬리브 매수 제외: " + "; ".join(
                                f"{n}({c}) — {st}" for c, n, st in sleeve_dropped)})
                    candidates.extend(sleeve_cands)

                # ── 보유 심의 (사장 지시 2026-07-29: 유지 심사역 ↔ 매도 심사역 찬반토론) ──
                # 종전엔 사후관리실장 단독 판단을 240자로 잘라 '재구성'만 했다. 이제 매수 심의와
                # 동일하게 토론을 돌리고, 그 최종 스탠스(유지|절반|전량)를 **실제 매도지시로
                # 되돌려 반영**한다. 심의 대상은 슬리브 ETF 포함 *전체* 보유 — 종전엔 주식만
                # (stock_holdings) 봐서 보유가 전부 채권·원자재 ETF인 계정(모의)에선 '보유 심의'
                # 카드가 통째로 비어 있었다.
                pm_view = getattr(cyc, "_pm_view", "") or ""
                _all_h = getattr(cyc, "holdings", None) or holdings
                # 토론 예산(_SELL_DEBATE_CAP)은 '결정이 갈릴 여지가 큰' 포지션부터 쓴다 —
                # ① 1차 판단이 '보유'가 아닌 것(실제 매도 후보) ② 평가손익이 나쁜 것 순.
                # 예산 밖 포지션은 사후관리실장 1차 판단을 그대로 쓴다(매도 누락 없음).
                _all_h = sorted(
                    _all_h,
                    key=lambda x: (str(sell_directives.get(str(x.get("code", "")).strip(), "보유")).strip()
                                   in _SELL_HOLD_WORDS,
                                   float(x.get("pnl_pct") or 0.0)))
                reviews = []
                _reviewed = 0
                for h in _all_h:
                    hc = str(h.get("code", "")).strip()
                    if not hc or hc not in sell_directives:
                        continue
                    directive = str(sell_directives.get(hc, "보유")).strip()
                    _nm_h = str(h.get("name", "") or hc)
                    _spec_h = sleeve_for_code(hc)
                    _chief = _spec_h.manager_name if _spec_h else "사후관리실장"
                    # 1차 판단 근거 — 사후관리실장 원문에서 이 종목 문단(절단 없이 넉넉히).
                    _view = _quant_ctx_for(pm_view, hc, width=1200) or ""
                    if not _view:
                        for _sent in re.split(r"(?<=[.다요])\s+", pm_view):
                            if hc in _sent or (_nm_h and _nm_h in _sent):
                                _view = _sent.strip(); break
                    _dialogue, _dec, _llm = [], None, False
                    if _reviewed < _SELL_DEBATE_CAP:
                        try:
                            from infra import position_thesis as _pth, sleeve_thesis as _sth
                            _th = ((_sth.get(self.uid, _spec_h.key, hc) or {}) if _spec_h
                                   else (_pth.get(self.uid, hc) or {}))
                            _hp = cycle_store.get_holding_period(hc, uid=self.uid) or {}
                            _planner_objection = ""
                            if _th:
                                from infra import fundamental_research_store as _frs
                                from infra import planner_sell_guard as _psg
                                _fr = (_frs.latest(self.uid, hc) or {}) if _spec_h is None else {}
                                _inv = ((_fr.get("thesis_invalidators") or [])
                                        if str(_fr.get("ts") or "") > str(_th.get("entry_ts") or "") else [])
                                _planner_objection = _psg.assess_objection(
                                    _th, h, hold_days=_hp.get("days_held"), invalidators=_inv).get("message", "")
                            _thesis_text = (f"진입사유: {_th.get('entry_reason') or '(없음)'} | "
                                            f"목표가: {_th.get('target_price') or '?'} | "
                                            f"손절가: {_th.get('stop_price') or '?'} | "
                                            f"계획보유: {_th.get('planned_hold_hours') or '?'}h") if _th else ""
                            _dialogue, _dec, _llm = await cmt.deliberate_position_sell(
                                hc, _nm_h, qty=int(h.get("qty") or 0),
                                pnl_pct=float(h.get("pnl_pct") or 0.0),
                                hold_days=_hp.get("days_held"),
                                manager_view=(_view or f"1차 매도지시: {directive}"),
                                thesis=_thesis_text,
                                quant_excerpt=_quant_ctx_for(getattr(cyc, "quant_report", ""), hc, width=300),
                                news_excerpt=_extract_code_news(cyc.news_report, hc, _nm_h)[:300],
                                macro_view=macro_view, chief_label=_chief,
                                fallback_directive=directive,
                                planner_objection=_planner_objection,
                                past_context=_tr_past(hc), progress=_prog)
                            _reviewed += 1
                        except Exception as _se:  # noqa: BLE001
                            logger.warning(f"[매도심의] {hc} 생략(fail-open): {_se}")
                            _dialogue, _dec = [], None
                    if _dec:
                        directive = _dec["stance"]
                        sell_directives[hc] = directive       # 토론 결과를 실제 매도지시로 반영
                        _reason = _dec["text"]
                    else:
                        _reason = _view or f"{_chief} 매도결정: {directive}"
                        _dialogue = [{"speaker": _chief, "role": "chief", "text": _reason,
                                      "stance": directive}]
                    decision = ("유지" if directive in _SELL_HOLD_WORDS
                                else ("매도" if directive in _SELL_ALL_WORDS else f"매도({directive})"))
                    reviews.append({
                        "code": hc, "name": _nm_h, "sector": "",
                        "pnl_pct": round(float(h.get("pnl_pct") or 0.0), 2),
                        "decision": decision, "reason": _reason,
                        "engine": ("llm" if _llm else "결정론"),
                        "dialogue": _dialogue,
                    })
                cyc.sell_directives = sell_directives

                if candidates or reviews:
                    cyc.committee = {
                        "generated_at": getattr(self.cycle_log, "started_at", ""),
                        "macro": {"view": macro_view, "equity_allocation": (_alloc if _alloc is not None else 0.7)},
                        "candidates": candidates,
                        "position_reviews": reviews,
                    }
            except Exception as _ce:
                logger.warning(f"[위원회] 심의 스테이지 실패 — fail-open(매매 파이프라인 불변): {_ce}")

    async def _apply_planner_sell_deferrals(self, cyc, holdings: List[Dict[str, Any]],
                                            sell_directives: Dict[str, Any]) -> Dict[str, Any]:
        """포트폴리오기획팀장의 강한 반론과 1회 보류권을 주문 조립 직전에 적용한다."""
        from infra import fundamental_research_store, planner_sell_guard, position_thesis, sleeve_thesis
        guarded = dict(sell_directives or {})
        records = []
        try:
            guard_uid = int(self.uid)
        except (TypeError, ValueError):
            # 일부 순수단위 테스트·진단 객체는 문자열 uid를 쓴다. 거래 계정 uid가 아니면 정책 저장을 생략한다.
            cyc.planner_sell_guard = records
            return guarded
        cycle_key = str(getattr(self.cycle_log, "started_at", "") or _current_hour_key_str())
        for holding in holdings or []:
            code = str(holding.get("code") or "").strip()
            key = code if code in guarded else code.upper()
            if not code or key not in guarded:
                continue
            directive = str(guarded.get(key) or "보유").strip()
            spec = sleeve_for_code(code)
            thesis = (sleeve_thesis.get(guard_uid, spec.key, code) if spec
                      else position_thesis.get(guard_uid, code)) or {}
            period = cycle_store.get_holding_period(code, uid=guard_uid) or {}
            invalidators = []
            research = (fundamental_research_store.latest(guard_uid, code) or {}) if not spec else {}
            # 진입 전에 이미 알고 있던 리스크는 진입 thesis에 반영된 배경이다. 진입 후 새로
            # 포착된 훼손 신호만 보류권의 하드 예외로 취급한다.
            if str(research.get("ts") or "") > str(thesis.get("entry_ts") or ""):
                invalidators = research.get("thesis_invalidators") or []
            try:
                objection = planner_sell_guard.assess_objection(
                    thesis, holding, hold_days=period.get("days_held"), invalidators=invalidators)
                final, status = planner_sell_guard.apply_one_cycle_deferral(
                    guard_uid, code, directive, objection, cycle_key=cycle_key)
            except Exception as exc:
                logger.warning(f"[planner-sell-guard] {code} 적용 실패, 원지시 유지: {exc}")
                continue
            guarded[key] = final
            records.append({"code": code, "original": directive, "final": final,
                            "status": status, "objection": objection})
            if status == "deferred":
                await self._emit({"type": "agent_msg", "agent": "포트폴리오기획팀장",
                                  "message": f"🛑 매도 1회 보류 — {objection.get('message', '')}"})
            elif status == "released":
                await self._emit({"type": "agent_msg", "agent": "포트폴리오기획팀장",
                                  "message": f"✅ {code} 다음 정기 사이클에도 매도 근거가 반복돼 "
                                             f"보류권 해제 — {directive} 지시를 통과시킵니다."})
            elif objection.get("hard_override") and directive not in _SELL_HOLD_WORDS:
                await self._emit({"type": "agent_msg", "agent": "포트폴리오기획팀장",
                                  "message": f"🚨 {objection.get('message', '')}"})
        cyc.planner_sell_guard = records
        cyc.sell_directives = guarded
        return guarded

    async def _cyc_stage_build_orders(self, cyc):
            market_open = cyc.market_open
            target_codes = cyc.target_codes
            candidate_codes = cyc.candidate_codes
            quant_report = cyc.quant_report
            news_report = cyc.news_report
            # C2 수정(사장 지시 2026-06-08): 주식 매도 트랙(_build_orders→_assemble_sell_orders 자동
            # 익절/손절/편중축소)에는 '채권 제외' 보유목록만 넘긴다. ENABLE_BOND_ETF ON 이면
            # finalize_sell 이 cyc.stock_holdings(주식만) 를 세팅한다. OFF/미설정이면 cyc.holdings
            # 전체를 그대로 써서 기존 동작 100% 불변. (cyc.holdings 원본은 채권 스테이지·후속 처리용으로 보존)
            holdings = getattr(cyc, "stock_holdings", None)
            if holdings is None:
                holdings = cyc.holdings
            sell_directives = cyc.sell_directives
            sell_directives = await self._apply_planner_sell_deferrals(
                cyc, getattr(cyc, "holdings", None) or holdings, sell_directives)
            _entry_dirs = cyc._entry_dirs
            _sell_prices = cyc._sell_prices
            # [7] ORDER DRAFT — assembled in Python: buys = 2패스 최종종목, sells = 사후관리실장 매도결정.
            #     The trader LLM is only invoked as a *fallback* when Python has nothing to act on.
            self.current_state = SwarmState.ORDER_DRAFTING
            await self._emit({"type":"status","state":"ORDER_DRAFTING","message":"주문 초안 (잔고 비율 기반 사이징)"})
            order_obj, price_map, buying_power = await self._build_orders(
                target_codes, candidate_codes, quant_report, news_report, holdings, sell_directives=sell_directives,
                market_open=market_open, entry_dirs=_entry_dirs, sell_prices=_sell_prices,
                quant_scores=getattr(cyc, "_quant_scores", None), quant_sigmas=getattr(cyc, "_quant_sigmas", None),
                macro_stock_pct=getattr(cyc, "_macro_stock_pct", None),
                systematic_policy=getattr(cyc, "systematic_policy", None))
            # 자산슬리브 트랙 합류(사장 지시 2026-06-09): 슬리브 주문을 주식 주문과 한 묶음으로 보내면
            # 이후 _cyc_stage_risk(validate_order_draft)·_cyc_stage_execute(KR/US 라우팅)가 자동 검증·집행한다.
            #  ① 슬리브 매수(자산배분, _cyc_stage_sleeves 산출)  ② 슬리브 매도(사후관리실장 종합결정 → 조립)
            order_obj["orders"].extend(getattr(cyc, "sleeve_buy_orders", []) or [])
            # 코드리뷰 #4(2026-06-09): 슬리브 매도 가격조회 실패(0/누락) 시 보유의 cur_price 로 폴백 —
            # 주식 매도 트랙과 달리 슬리브 매도는 sleeve_price_map 단일소스라 폴백이 없어, 사후관리실장이
            # 종합 결정한 매도가 가격실패로 조용히 누락되던 문제('주문 절대 스킵 금지' 위반) 방지.
            _sleeve_px = dict(getattr(cyc, "sleeve_price_map", None) or {})
            _hold_px = {str(h.get("code", "")).strip().upper(): float(h.get("cur_price") or 0.0)
                        for h in (getattr(cyc, "holdings", None) or [])}
            def _sleeve_price_lookup(_c):
                _p = float(_sleeve_px.get(_c) or 0.0)
                return _p if _p > 0 else _hold_px.get(str(_c).strip().upper(), 0.0)
            _sleeve_sells = _build_sleeve_sell_orders(
                sell_directives, getattr(cyc, "holdings", None) or [], _sleeve_price_lookup,
                pool=self._enabled_sleeve_codes())  # 코드리뷰 #5: 활성 슬리브만(OFF 슬리브는 주식 트랙)
            order_obj["orders"].extend(_sleeve_sells)
            # 두 매도 생산자(주식 트랙·슬리브 트랙)가 합류하는 유일한 지점 — 여기서 종목당 1건으로
            # 강제한다(2026-07-29 261220 이중매도). 위 stock_holdings 폴백 수정이 근본원인이지만,
            # 불변식은 합류점에서 지키는 게 맞다(새 매도 생산자가 늘어도 안전).
            order_obj["orders"], _dup_sells = dedupe_sell_orders(order_obj["orders"])
            if _dup_sells:
                _dup_msg = ", ".join(f"{t} x{q}" for t, q in _dup_sells)
                logger.warning(f"[매도중복 uid={self.uid}] 같은 종목 매도 주문 중복 제거: {_dup_msg}")
                await self._emit({"type": "agent_msg", "agent": "시스템",
                                  "message": f"🧹 매도 주문 중복 제거 — {_dup_msg} (보유 초과 매도 방지)"})
            # C1: 슬리브 가격을 price_map 에 합쳐야 validate_order_draft 가 슬리브 매수를 통과시킨다
            # (매수는 가격 미주입 시 price<=0 → 무조건 반려). 매도가 폴백 가격도 price_map 에 반영.
            # US 가격은 주식 price_map 과 동일 USD 단위(원시 us_last_price) — 환산은 guardrails 가 수행.
            price_map.update(getattr(cyc, "sleeve_price_map", {}) or {})
            for _so in _sleeve_sells:  # 폴백가로 조립된 매도도 price_map 에 실어 검증 일관성 유지
                price_map.setdefault(_so["ticker"], _so.get("price") or 0.0)
            order_draft = json.dumps(order_obj, ensure_ascii=False, indent=2)
            # 사장 피드백 2026-05-15 (3차): 트레이더 호출은 EXECUTION 이후로 이동.
            # 여기서는 시스템 상태 메시지로 "조립 완료, 리스크 검증 이동" 만 알린다.
            if order_obj["orders"]:
                _drafting_msg = f"주문 {len(order_obj['orders'])}건 조립 완료 → 리스크 검증으로 이관"
            else:
                _drafting_msg = "ℹ️ 잔고/한도 조건상 신규 주문 없음. " + ("; ".join(order_obj.get("sizing_notes") or []))[:240]
            await self._emit({"type":"status","state":"ORDER_DRAFTING","message": _drafting_msg})
            # 사장 지시 2026-06-16(투명성): 사이징 '제외' 사유(예산 초과·시세 결손 등)는 주문이
            # 있어도 대시보드에 실시간 발화 — OPS 가 '왜 안 샀는지'를 못 봐 '유동성 부족'으로
            # 환각하던 경로를 막는다(cycle 379 OPS#379 사례). 기존엔 final_report '다음 사이클
            # 유의'에만 묻혀 실시간 이벤트 스트림엔 안 보였다.
            _excluded_notes = [n for n in (order_obj.get("sizing_notes") or []) if "제외" in str(n)]
            if _excluded_notes:
                await self._emit({"type":"agent_msg","agent":"프롭트레이딩팀장",
                    "message": "🔎 매수 후보 제외 사유: " + " | ".join(str(n) for n in _excluded_notes)[:400]})
            self.cycle_log.log("DRAFT", "프롭트레이딩팀장", order_draft)
            cyc.order_obj = order_obj
            cyc.price_map = price_map
            cyc.buying_power = buying_power
            cyc.order_draft = order_draft

    async def _cyc_stage_risk(self, cyc):
            per_dart = cyc.per_dart
            buying_power = cyc.buying_power
            price_map = cyc.price_map
            order_draft = cyc.order_draft
            from tools.portfolio_risk import constrain_entries
            from infra.asset_sleeves import sleeve_for_code
            _draft = json.loads(order_draft) if isinstance(order_draft, str) else dict(order_draft)
            _policy_notes = []
            if getattr(cyc, "systematic_policy", None) is not None:
                from tools.stock_policy import cap_entries
                _draft["orders"], _policy_notes = cap_entries(
                    _draft.get("orders", []), cyc.systematic_policy, holdings=cyc.holdings or [],
                    prices=price_map, total_eval=float(buying_power.get("total_eval") or 0),
                    usdkrw=get_usdkrw(USDKRW_FALLBACK),
                    is_sleeve=lambda code: sleeve_for_code(code) is not None)
            _orders, _risk_notes = constrain_entries(
                _draft.get("orders", []), prices=price_map, buying_power=buying_power,
                holdings=cyc.holdings or [], approved_stock_codes=getattr(cyc, "target_codes", []),
                sigmas=getattr(cyc, "_quant_sigmas", {}),
                stop_loss_pct=runtime.get("STOP_LOSS_PCT", uid=self.uid),
                usdkrw=get_usdkrw(USDKRW_FALLBACK),
                is_sleeve=lambda code: sleeve_for_code(code) is not None)
            _risk_notes = _policy_notes + _risk_notes
            _draft["orders"] = _orders
            _draft.setdefault("sizing_notes", []).extend(_risk_notes)
            order_draft = json.dumps(_draft, ensure_ascii=False)
            cyc.order_obj = _draft
            cyc.order_draft = order_draft
            cyc.portfolio_risk_notes = _risk_notes
            # [8] RISK — 사장 지시 2026-05-14: 1차(결정론) + 2차(DART 공시) 통합 실행 후 **한 번에** 출력
            self.current_state = SwarmState.RISK_VALIDATION
            await self._emit({"type":"status","state":"RISK_VALIDATION","message":"리스크 검증 (결정론 + DART)"})
            self.validation_attempts = 1
            risk_result = validate_order_draft(order_draft, "", buying_power=buying_power, price_map=price_map, uid=self.uid)
            risk_approved = bool(risk_result["approved"])
            approved_orders = [r for r in risk_result["results"] if r["status"] == "APPROVED"]
            # 2차 DART 재심 — 매도(리스크 감소)는 통과, 매수만 공시 기반 재확인
            dart_resp_text = ""
            dart_vetoed: set = set()
            _buys = [r for r in approved_orders if (r.get("side") or "buy") != "sell"]
            if _buys:
                dart_vetoed, dart_resp_text = await self._dart_risk_review(_buys, per_dart)
                if dart_vetoed:
                    approved_orders = [r for r in approved_orders
                                       if not ((r.get("side") or "buy") != "sell" and str(r.get("ticker","")).upper() in dart_vetoed)]
                    risk_approved = bool(approved_orders)
            # 통합 메시지 — 1차 결과 + DART 재심을 한 번에
            unified_lines = ["🧮 [리스크관리실장 — 통합 검증 보고]\n",
                             "▶ 1차 결정론 검증 (편중도·MDD·예수금·예산·수량):", risk_result["report"], "\n▶ 포트폴리오 위험 예산:", "\n".join(_risk_notes) or "한도 내 주문 유지"]
            if _buys:
                unified_lines.append("\n▶ 2차 DART 공시 재심 (매수 한정):")
                unified_lines.append(dart_resp_text or "(DART 응답 없음)")
                if dart_vetoed:
                    unified_lines.append(f"\n⚠ DART 재심 반려: {', '.join(sorted(dart_vetoed))}")
            else:
                unified_lines.append("\n▶ 2차 DART 공시 재심: 매수 주문 없음 → 생략")
            unified_lines.append(f"\n▶ 최종 결과: {'✅ 승인' if risk_approved else '❌ 미승인'} ({len(approved_orders)}건)")
            unified_msg = "\n".join(unified_lines)
            self.cycle_log.log("RISK", "리스크관리실장", unified_msg)
            await self._emit({"type":"agent_msg","agent":"리스크관리실장","message":unified_msg})
            cyc.risk_result = risk_result
            cyc.risk_approved = risk_approved
            cyc.approved_orders = approved_orders
            # 사장 지시 2026-06-16(투명성): DART 2차 반려 종목 + 1차/2차 통합 보고를 cyc 에 보존 —
            # final_report·DB(risk_report)가 1차만 담아 DART 반려가 증발하던 결함 수정.
            cyc.dart_vetoed = dart_vetoed
            cyc.risk_report_full = unified_msg

    async def _cyc_stage_execute(self, cyc):
            session = cyc.session
            market_open = cyc.market_open
            _exec_N = cyc._exec_N
            holdings = cyc.holdings
            order_obj = cyc.order_obj
            order_draft = cyc.order_draft
            sell_directives = cyc.sell_directives
            name_map = cyc.name_map
            risk_result = cyc.risk_result
            risk_approved = cyc.risk_approved
            approved_orders = cyc.approved_orders
            # [9] EXECUTION — place KIS orders: sells(사후관리) 우선, 매수는 전략 N개까지. qty from sizing.
            self.current_state = SwarmState.EXECUTION
            exec_results: List[Dict] = []
            # 사장 지시 2026-05-14: 개장 사이클은 _exec_N(Pass2에서 확대됨)을 사용
            _max_trades = _exec_N if market_open else runtime.get("MAX_TRADES_PER_CYCLE", uid=self.uid); _max_qty = runtime.get("MAX_ORDER_QTY", uid=self.uid)
            if risk_approved and LIVE_TRADING and approved_orders:
                # I2 수정(사장 지시 2026-06-08): 채권 매수는 자산배분이라 매매빈도 cap(_max_trades)에서
                # 면제한다 — 주식 매수가 cap 을 채워도 채권 리밸런싱이 조용히 누락되지 않게('주문 절대 스킵
                # 금지'). ENABLE_BOND_ETF OFF 면 빈 풀 → build_exec_list = 기존 sells + buys[:cap](동작 불변).
                _bond_pool_for_cap = (all_sleeve_pool_codes()
                    if (bool(runtime.get("ENABLE_BOND_ETF", uid=self.uid))
                        or bool(runtime.get("ENABLE_COMMODITY_ETF", uid=self.uid)))
                    else set())
                _exec_list = build_exec_list(approved_orders, _max_trades if _max_trades is not None else 2, _bond_pool_for_cap)
                # 안전감시에서 직전에 접수한 전량매도를 정시 사이클이 중복 전송하지 않게 한다.
                _safety_recent = {c for c, ts in self._safety_last_order_at.items()
                                  if time.time() - ts < SAFETY_ORDER_COOLDOWN_SEC}
                _safety_dups = [str(r.get("ticker") or "").strip() for r in _exec_list
                                if (r.get("side") or "buy") == "sell"
                                and str(r.get("ticker") or "").strip() in _safety_recent]
                if _safety_dups:
                    _exec_list = [r for r in _exec_list
                                  if not ((r.get("side") or "buy") == "sell"
                                          and str(r.get("ticker") or "").strip() in _safety_recent)]
                    await self._emit({"type": "agent_msg", "agent": "리스크관리실장",
                                      "message": "🛡️ 안전감시에서 이미 접수한 매도 중복 제외 — "
                                                 + ", ".join(sorted(set(_safety_dups)))})
                # 사장 지시 2026-05-21: US 체결 재확인용 '주문 직전' 해외 보유 스냅샷.
                # cycle 의 holdings 는 kr_holdings() 라 US 가 없어, 이미 보유 중인 US 종목의
                # before_qty 가 0으로 잡혀 false-positive(미체결을 체결로 오판)가 날 수 있다.
                # US 주문이 있을 때만 1회 조회해 _reverify_fills baseline 에 합친다.
                _us_baseline: List[Dict] = []
                def _is_us_tk(_t: str) -> bool:
                    _t = (_t or "").strip()
                    return not (_is_kr_code(_t))
                if any(_is_us_tk(r.get("ticker")) for r in _exec_list):
                    try:
                        _us_baseline = await self.broker._overseas_holdings()
                    except Exception:
                        _us_baseline = []
                for r in _exec_list:
                    tk = str(r.get("ticker") or "").strip()
                    qty = max(1, int(r.get("qty") or 1))
                    if _max_qty and _max_qty > 0:
                        qty = min(qty, _max_qty)
                    is_kr = _is_kr_code(tk)
                    # 사장 피드백 2026-05-15 (#4): _build_orders에서 첨부한 entry_mode로 분기.
                    _emode = r.get("entry_mode") or "market"
                    _elimit = r.get("entry_limit")
                    _ewatch_pct = r.get("entry_watch_pct")
                    _side = r.get("side") or "buy"
                    # 사장 지시 2026-05-22: 매도는 계량분석팀장이 '매도가'를 숫자로 제시한 경우 그 지정가로,
                    # 아니면 시장가(즉시 청산). watch 모드는 매도에 적용하지 않는다.
                    if _side == "sell":
                        _emode = "limit" if (_emode == "limit" and _elimit and float(_elimit) > 0) else "market"
                    # An entry limit is the maximum approved buy price.
                    _pm = getattr(cyc, "price_map", None) or {}
                    _mkt_px = float(_pm.get(tk) or _pm.get(tk.upper()) or 0.0)
                    # 사장 정책 2026-06-22: 매도 지정가가 시장가 위면 '긴급청산(전량/손절/트레일링/편중)'
                    # 에 한해 시장가로 전환 — 미체결로 물량이 잠기는 걸 막는다(uid2 316140 지정가
                    # 32,500 > 시장가 30,000 → sellable 0 누적). 익절 목표가(자동 익절·절반·N주)는
                    # 의도된 지정가라 유지. buy_limit_below_market 의 매도 대칭(KR 도메스틱 누락분).
                    if sell_limit_above_market(_side, _emode, _elimit, _mkt_px) and \
                            _is_urgent_liquidation(r.get("reason", "")):
                        await self._emit({"type": "agent_msg", "agent": "사후관리실장",
                            "message": (f"⚠️ {tk} 긴급청산 매도 지정가 {_elimit} > 시장가 {_mkt_px:g} → "
                                        f"미체결·물량잠김 방지 위해 시장가 매도로 전환.")})
                        self.cycle_log.log("EXEC", "시스템",
                            f"{tk} 매도 지정가 {_elimit}>시장가 {_mkt_px}(긴급청산) → 시장가 전환")
                        _emode = "market"; _elimit = None
                    # 대기(watch) 모드 — 백그라운드 태스크 spawn, 즉시 체결 대기 카운트 X
                    if _emode == "watch" and _ewatch_pct is not None:
                        asyncio.create_task(self._entry_watch_task(
                            ticker=tk, qty=qty, market=("KR" if is_kr else "US"),
                            watch_pct=float(_ewatch_pct), baseline_holdings=list(holdings or []),
                            reason=r.get("reason", "")))
                        await self._emit({"type":"agent_msg","agent":"프롭트레이딩팀장",
                            "message": f"⏱ {tk} {qty}주 분봉 모니터링 시작 — {_ewatch_pct:+.1f}% 도달 시 시장가 매수 (최대 3시간, 장 마감 시 취소)"})
                        # exec_results에는 'pending watch'로 기록 (체결 카운트 X)
                        exec_results.append({"ticker": tk, "side": _side, "qty": qty,
                                             "result": f"watch task spawned ({_ewatch_pct:+.1f}%)",
                                             "accepted": False, "filled": False,
                                             "fill_note": "분봉 대기 중", "ok": False, "watch": True})
                        continue
                    # 지정가(limit) 모드 — KIS limit 주문 (장 마감 시 KIS가 자동 취소)
                    _price_for_kis = 0
                    if _emode == "limit" and _elimit and _elimit > 0:
                        _price_for_kis = float(_elimit) if not is_kr else int(_elimit)
                    od = OrderDraft(ticker=tk, side=_side, qty=qty,
                                    price_type=("limit" if _emode == "limit" and _price_for_kis else "market"),
                                    limit_price=(float(_price_for_kis) if _price_for_kis else None),
                                    market="KR" if is_kr else "US",
                                    reason="Arquant 자동매매 — 보수적 리스크 승인", approved=True)
                    # 사장 피드백 2026-05-15 (6차): 추정 체결가 → 실제 체결가 기록.
                    # 주문 직전 스냅샷 — before_qty/before_avg (매수 시 평균단가 차이로 정확한 체결가 역산용)
                    # 매도는 avg_price가 안 변하므로 주문 직전 cur_price를 시장가 매도 체결가로 사용 (KIS 시장가 = 직전체결가 근사).
                    before_qty = next((h.get("qty") for h in (holdings or []) if h.get("code") == tk), 0) or 0
                    before_avg = next((h.get("avg_price") for h in (holdings or []) if h.get("code") == tk), 0) or 0
                    # 사장 지시 2026-06-11(라이브 진단): KR 매도는 보유수량이 아닌 '매도가능수량
                    # (ord_psbl_qty)' 기준 — 미체결 매도 주문이 물량을 잠그면 hldg_qty>0 인데
                    # 매도가능 0 이라 'KIS 잔고내역이 없습니다'로 반복 거부된다(uid2 041830 사례).
                    # 사장 지시 2026-06-16: 매도가능 0(잠김)이면 '미체결 펜딩 매도가 실제로 있는지'
                    # 확인 — 있으면 kr_sell 이 그 펜딩을 취소해 풀 수 있으므로 보유 전량 시장가로
                    # 재청산(재가격+재시도)한다. 기존엔 무조건 보류+continue 라 kr_sell 을 영영 못 불러
                    # 영원히 잠겼다(375500 익절 종일 미체결). 펜딩이 없으면(결제/글리치 잠금) 보류.
                    if _side == "sell" and is_kr:
                        _sellable = next((h.get("sellable_qty") for h in (holdings or [])
                                          if h.get("code") == tk and h.get("sellable_qty") is not None), None)
                        if _sellable is not None and before_qty > 0:
                            _has_pending = False
                            if int(_sellable) <= 0:
                                try:
                                    _pend = await self.broker.kr_pending_orders(tk)
                                    _has_pending = any((r.get("sll_buy_dvsn_cd") or "").strip() == "01"
                                                       for r in (_pend or []))
                                except Exception as _pe:
                                    logger.warning(f"[매도잠금 {tk}] 펜딩 조회 실패 → 보류: {_pe}")
                                    _has_pending = False   # 조회 실패 → 무리한 시장가 금지(보류)
                            # 버그 C(2026-06-18): 펜딩 없는 잠김이 N사이클 연속이면 에스컬레이션.
                            from config import LOCKED_SELL_ESCALATE_AFTER as _ESC_AFTER
                            _lstreak_map = _load_locked_streak(self.uid)
                            _act, _val = _locked_sell_action(
                                _sellable, before_qty, qty, _has_pending,
                                locked_streak=int(_lstreak_map.get(tk, 0)), escalate_after=_ESC_AFTER)
                            if _act in ("proceed", "clamp", "reprice_market", "escalate_market"):
                                if _lstreak_map.pop(tk, None) is not None:   # 잠김 해소/시도 → 스트릭 리셋
                                    _save_locked_streak(self.uid, _lstreak_map)
                            if _act == "hold":
                                _lstreak_map[tk] = int(_lstreak_map.get(tk, 0)) + 1
                                _save_locked_streak(self.uid, _lstreak_map)
                                _lock_msg = (f"{tk} 매도 보류 — 보유 {before_qty}주 중 매도가능 0주 "
                                             f"(미체결 펜딩 없음 · 결제/제도 잠금 추정 — {_lstreak_map[tk]}/{_ESC_AFTER} 사이클, "
                                             f"임계 도달 시 강제 재청산)")
                                await self._emit({"type": "trade_failed", "message": f"⛔ {_lock_msg}",
                                                  "ticker": tk, "side": "sell", "qty": qty, "filled": False})
                                exec_results.append({"ticker": tk, "side": _side, "qty": qty,
                                                     "result": _lock_msg, "accepted": False, "filled": False,
                                                     "fill_note": "매도가능수량 0", "ok": False,
                                                     "fill_price": None, "fill_currency": "KRW",
                                                     "avg_cost": float(before_avg or 0.0)})
                                self.cycle_log.log("EXEC", "시스템", f"{tk} sell x{qty} → {_lock_msg}")
                                continue
                            if _act == "escalate_market":
                                qty = int(_val); od.qty = qty
                                od.price_type = "market"; od.limit_price = None
                                _emode = "market"; _price_for_kis = 0
                                # 모의는 thdt_sll_qty 일일카운터 quirk(익일 해소)라 INFO 로 순화, 실계정은
                                # 진짜 결제/제도 잠금일 수 있어 WARN(수동 확인) 보존 (사장 지시 2026-06-22).
                                _esc_sev, _esc_title, _esc_msg = _locked_sell_escalation_alert(
                                    tk, before_qty, _ESC_AFTER,
                                    is_mock=bool(getattr(self.broker, "is_mock", False)))
                                await self._emit({"type": "agent_msg", "agent": "시스템",
                                                  "message": f"{'🚨' if _esc_sev == 'WARN' else '🧪'} {_esc_msg}"})
                                try:
                                    notifier.alert(_esc_sev, _esc_title,
                                                   f"{tk}: {_esc_msg}", dedup_key=f"locked_sell_{tk}")
                                except Exception:
                                    pass
                                self.cycle_log.log("EXEC", "시스템", f"{tk} sell x{qty} → {_esc_msg}")
                            if _act == "reprice_market":
                                qty = int(_val); od.qty = qty
                                od.price_type = "market"; od.limit_price = None
                                _emode = "market"; _price_for_kis = 0
                                _rp_msg = (f"{tk} 매도 미체결로 물량 잠김 — 미체결 지정가 매도 취소 후 "
                                           f"보유 {before_qty}주 시장가 재청산(재가격+재시도)")
                                await self._emit({"type": "agent_msg", "agent": "시스템",
                                                  "message": f"🔁 {_rp_msg}"})
                                self.cycle_log.log("EXEC", "시스템", f"{tk} sell x{qty} → {_rp_msg}")
                            elif _act == "clamp":
                                await self._emit({"type": "agent_msg", "agent": "시스템",
                                    "message": f"✂️ {tk} 매도수량 클램프 {qty}→{_val}주 (매도가능수량 기준 — 잔여는 미체결 주문 잠금)"})
                                qty = int(_val); od.qty = qty
                    # 사장 지시 2026-06-16: 매수는 집행 직전 KIS 권위 매수가능수량으로 클램프 —
                    # 리스크 승인(bp.cash)과 실제 주문가능금액(증거금·D+2 미결제 반영)의 불일치로
                    # 마지막 매수가 '주문가능금액 초과' 거부되던 문제(uid1 241710) 방어. 정규장/US 만
                    # 적용(시간외 NXT 는 정규 매수가능과 기준이 달라 스킵), 모의/조회실패는 폴백(스킵).
                    if _side == "buy" and not is_kr_extended_hours(session):
                        _bq = None
                        try:
                            if is_kr:
                                _ps = await self.broker.kr_psbl_order(tk, _price_for_kis or _mkt_px or 0)
                                _bq = _ps.get("buy_qty") if _ps.get("ok") else None
                            else:
                                _ps = await self.broker.us_buying_power(tk, _mkt_px or _elimit or 0)
                                _bq = _ps.get("qty") if _ps.get("ok") else None
                        except Exception:
                            _bq = None
                        if _bq is not None:
                            _clamped = _clamp_qty_to_buyable(qty, _bq)
                            if _clamped < 1:
                                _bmsg = f"{tk} 매수 보류 — KIS 주문가능수량 0주(예산/증거금 부족 추정) → 다음 사이클 재시도"
                                await self._emit({"type": "trade_failed", "message": f"⛔ {_bmsg}",
                                                  "ticker": tk, "side": "buy", "qty": qty, "filled": False})
                                exec_results.append({"ticker": tk, "side": _side, "qty": qty,
                                                     "result": _bmsg, "accepted": False, "filled": False,
                                                     "fill_note": "KIS 주문가능금액 부족", "ok": False,
                                                     "fill_price": None,
                                                     "fill_currency": ("KRW" if is_kr else "USD"),
                                                     "avg_cost": 0.0})
                                self.cycle_log.log("EXEC", "시스템", f"{tk} buy x{qty} → {_bmsg}")
                                continue
                            if _clamped < qty:
                                await self._emit({"type": "agent_msg", "agent": "시스템",
                                    "message": f"✂️ {tk} 매수수량 클램프 {qty}→{_clamped}주 (KIS 주문가능수량 기준 — 주문가능금액 초과 방지)"})
                                qty = _clamped
                                od.qty = qty
                    sell_ref_price = 0.0
                    if _side == "sell":
                        if is_kr:
                            try: sell_ref_price = await self.broker.kr_last_price(tk)
                            except Exception: sell_ref_price = 0.0
                        else:
                            try: sell_ref_price = await self.broker.us_last_price(tk)
                            except Exception: sell_ref_price = 0.0
                    # 시간외(NXT) 세션이면 거래소 데코레이트 + 지정가 산정. 시세 결손 시 보류.
                    od, _nxt_skip = await self._finalize_kr_order_for_session(od, get_current_session())
                    if _nxt_skip:
                        await self._emit({"type":"trade_failed", "message": f"⚠️ {_nxt_skip}"})
                        exec_results.append({
                            "ticker": tk, "side": _side, "qty": qty,
                            "result": _nxt_skip, "accepted": False, "filled": False,
                            "fill_note": "시간외 주문 사전 보류", "ok": False,
                            "fill_price": None,
                            "fill_currency": ("KRW" if is_kr else "USD"),
                            "avg_cost": float(before_avg or 0.0),
                        })
                        self.cycle_log.log("EXEC", "시스템", f"{tk} {_side} x{qty} → {_nxt_skip}")
                        continue
                    # KIS throttles orders aggressively (초당 거래건수) — pace + one retry.
                    res = ""
                    for attempt in range(3):
                        await asyncio.sleep(1.5 if attempt == 0 else 2.5)
                        try:
                            res = await self.broker.place_order(od)
                        except Exception as e:
                            res = f"[주문 예외] {e}"
                        if not any(x in res for x in ("초당", "거래건수", "EGW", "유량", "TPS")):
                            break
                    accepted = all(bad not in res for bad in ("실패", "에러", "거부", "예외", "REJECT", "초당", "거래건수"))
                    # 사장 지시 2026-06-11: NXT 미상장 거부('종목정보가 없습니다…')는 영속 학습 —
                    # 이후 전 계정이 시간외(NXT) 세션에서 이 종목의 주문/매수 후보 자체를 건너뛴다.
                    if getattr(od, "exchange", "") == "NXT" and nxt_blacklist.looks_nxt_unsupported(res):
                        try:
                            if nxt_blacklist.record(od.ticker, note=res[:120]):
                                await self._emit({"type": "agent_msg", "agent": "시스템",
                                    "message": f"🚫 {od.ticker} NXT 거래불가 종목으로 학습(전 계정 공유) — "
                                               f"이후 시간외(NXT) 주문·매수 시도 제외, 정규장은 영향 없음"})
                        except Exception as _ne:
                            logger.warning(f"NXT 블랙리스트 기록 실패({od.ticker}): {_ne}")
                    # Fill confirmation — "주문 전송 완료" only means *accepted*, not *filled*. Re-read holdings.
                    filled = False; fill_note = ""; fill_price: Optional[float] = None
                    fill_qty = qty  # 원장·로그 기록용 실제 체결수량(기본=주문량, KR 확인 시 잔고 증분으로 교정)
                    after_qty = None  # KR 확인 후 잔고 — 부분체결 잔여분 폴링의 base 로 전달
                    after_avg = 0.0  # KR 체결 확인 시 갱신 — 매수 후 블렌딩 평단
                    if accepted and is_kr:
                        try:
                            await asyncio.sleep(2.0)
                            self.broker._acct_snap = None  # force fresh read post-order
                            after = await self.broker.kr_holdings()
                            after_qty = next((h.get("qty") for h in after if h.get("code") == tk), 0) or 0
                            after_avg = next((h.get("avg_price") for h in after if h.get("code") == tk), 0) or 0
                            if od.side == "buy" and after_qty > before_qty:
                                filled = True; fill_note = f"보유 {before_qty}→{after_qty}주 확인"
                                fill_qty, _rem = _settle_fill_qty("buy", before_qty, after_qty, qty)
                                # 사장 피드백 6차: 매수 체결가를 평균단가 차이로 정확히 역산.
                                # (after_avg × after_qty) - (before_avg × before_qty) = fill_price × buy_qty
                                buy_qty_actual = after_qty - before_qty
                                if buy_qty_actual > 0:
                                    fill_price = (after_avg * after_qty - before_avg * before_qty) / buy_qty_actual
                                    fill_note += f" · 체결가 {fill_price:,.0f}원"
                                if _rem > 0:
                                    fill_note += f" · ⚠ 부분체결 {fill_qty}/{qty}주(잔여 {_rem} 미체결 — 원장은 체결분만 반영, reconcile 감시)"
                            elif od.side == "sell" and after_qty < before_qty:
                                filled = True; fill_note = f"보유 {before_qty}→{after_qty}주 확인"
                                fill_qty, _rem = _settle_fill_qty("sell", before_qty, after_qty, qty)
                                # 사장 피드백 6차: 매도는 직전 last_price를 시장가 체결가로 사용.
                                if sell_ref_price > 0:
                                    fill_price = float(sell_ref_price)
                                    fill_note += f" · 체결가 ≈{fill_price:,.0f}원 (직전 호가)"
                                if _rem > 0:
                                    fill_note += f" · ⚠ 부분체결 {fill_qty}/{qty}주(잔여 {_rem} 미체결 — 원장은 체결분만 반영, reconcile 감시)"
                            else:
                                fill_note = f"체결 미확인(보유 {after_qty}주) — 접수만 완료, 호가 미체결 가능"
                        except Exception as _e:
                            fill_note = f"체결확인 조회 실패({_e})"
                    elif accepted and not is_kr:
                        # US: holdings 즉시 확인 어려움. 시장가 = 직전 last_price 사용.
                        if _side == "buy":
                            try: fill_price = await self.broker.us_last_price(tk)
                            except Exception: fill_price = None
                        else:
                            fill_price = sell_ref_price if sell_ref_price > 0 else None
                    # 사장 지시 2026-05-21: 누적 실매매 체결 카운트는 '즉시 체결이 확인된' 경우에만 +1.
                    # 접수(accepted)만으론 올리지 않는다 — 미확인 주문은 _poll_fills_until_confirmed 가
                    # 5분마다 반복 확인해, 실제 보유 변동이 잡히는 그 시점에 비로소 +1 한다(차감 로직 없음).
                    is_us = not is_kr
                    ok = filled  # 즉시 체결 확인된 주문만 카운트 (KR 2초 후 보유 확인 / US 는 즉시 확인 불가 → 폴링)
                    # 사장 지시 2026-05-19: 매수·매도 시점의 KIS 실매입평균가(pchs_avg_pric)를
                    # 거래에 직접 박는다. 매도는 주문 직전 평단(before_avg)=청산 로트의 실제
                    # 매수평단, 매수는 체결 후 블렌딩 평단(after_avg).
                    avg_cost = (float(before_avg or 0.0) if od.side == "sell"
                                else float(after_avg or before_avg or 0.0))
                    # 원장·로그엔 '실제 체결수량'(fill_qty)을 박는다 — 주문수량(qty)이 아닌 잔고 증분.
                    # order_qty 는 별도 보존(부분체결 추적·표시용).
                    rec = {"ticker": tk, "side": od.side, "qty": fill_qty, "order_qty": qty, "result": res,
                           "accepted": accepted, "filled": filled, "fill_note": fill_note, "ok": ok,
                           "fill_price": fill_price, "fill_currency": ("USD" if is_us else "KRW"),
                           "avg_cost": avg_cost, "_after_qty": after_qty}
                    exec_results.append(rec)
                    if ok:
                        self._trades_executed += 1
                        self._trade_log.append({"ts": _now_kst_iso(), **rec})
                        # 사장 지시 2026-06-11: 실거래 원장에 체결 반영 (KIS 집계 비의존 자산평가).
                        try:
                            trade_ledger.apply_fill(self.uid, ticker=tk,
                                                    side=("sell" if od.side == "sell" else "buy"), qty=fill_qty,
                                                    price=fill_price, ccy=("USD" if is_us else "KRW"),
                                                    avg_cost=avg_cost, note="exec_immediate")
                        except Exception as _le:
                            logger.warning(f"[원장 uid={self.uid}] 체결 반영 실패 {tk}: {_le}")
                        # 사장 지시 2026-05-28(우선순위 3): 매수 체결 직후 펀드기획팀장이 진입 thesis 작성·영속.
                        # fire-and-forget — LLM 호출이라 사이클 다음 단계를 막지 않게 background task.
                        if od.side == "buy":
                            asyncio.create_task(self._record_buy_thesis(rec, cyc))
                    # 모바일 알림 ①: 체결 신청(주문 접수 성공) — 즉시 체결이든 미확인이든 '접수'된 순간 1회.
                    if accepted:
                        await self._emit({"type": "order_submitted", "agent": "프롭트레이딩팀장",
                            "message": f"📨 {tk} {('매수' if od.side == 'buy' else '매도')} {qty}주 주문 접수 — 체결 확인 중",
                            "ticker": tk, "side": od.side, "qty": qty})
                    # 모바일 알림 ②: 체결 완료(즉시 확인) / 또는 주문 실패. 미확인(접수만)은 폴링이 추후 처리.
                    if filled or not accepted:
                        badge = _fill_badge(filled, getattr(self.broker, "is_mock", False))
                        await self._emit({"type": _trade_event_type(filled),
                            "message": f"{badge} — {res}" + (f" | {fill_note}" if fill_note else ""),
                            "ticker": tk, "side": od.side, "qty": qty, "filled": filled,
                            "fill_price": fill_price, "fill_currency": ("USD" if is_us else "KRW"),
                            "avg_cost": avg_cost,
                            "trades_total": self._trades_executed})
                    self.cycle_log.log("EXEC", "시스템", f"{tk} {od.side} x{qty} → {res} | {fill_note}")
                # 회로차단기 갱신(사장 지시 2026-06-17): 연속 US 매수실패 누적 / US 성공 시 0 리셋
                self._us_buy_fail_streak = _update_us_buy_fail_streak(
                    getattr(self, "_us_buy_fail_streak", 0), exec_results)
                # 사장 피드백 2026-05-15 (#15): 체결과 접수 분리 표기
                _filled_cnt = sum(1 for e in exec_results if e.get("filled"))
                _accepted_only = sum(1 for e in exec_results if e.get("accepted") and not e.get("filled"))
                _exec_msg = f"실행 완료 — 이번 사이클 체결 {_filled_cnt}건"
                if _accepted_only:
                    _exec_msg += f" (+ 접수만 {_accepted_only}건, 5분 후 재확인)"
                _exec_msg += f" / 금일 누적 체결 {self._trades_executed}건"
                await self._emit({"type":"execution_ready", "message": _exec_msg, "draft": order_draft})
                # ── 사장 지시 2026-05-21: 미체결 주문 5분마다 '반복' 체결 확인 백그라운드 태스크 spawn ──
                # 접수만 되고 즉시 체결이 확인 안 된 주문은 그 시점부터 5분마다 보유 변동을 재확인하다
                # 체결이 잡히면 그때 누적 카운트 +1 + '체결 확인됨' 보고. 그때까지는 조용히(메시지 없음).
                _unconfirmed = [e for e in exec_results if e.get("accepted") and not e.get("filled")]
                # 부분체결(즉시 일부만 체결)도 잔여분을 폴링으로 추적 — base/recorded 를 체결분으로 넘겨
                # 중복 없이 증분만 기록한다(2026-06-15 후속). _after_qty 가 있어야 base 산정 가능.
                _partials = [
                    {"ticker": e["ticker"], "side": e["side"], "qty": int(e.get("order_qty") or e.get("qty") or 0),
                     "_base_qty": int(e.get("_after_qty") or 0), "_base_avg": float(e.get("avg_cost") or 0),
                     "_recorded": int(e.get("qty") or 0)}
                    for e in exec_results
                    if e.get("filled") and e.get("_after_qty") is not None
                    and int(e.get("qty") or 0) < int(e.get("order_qty") or e.get("qty") or 0)
                ]
                _poll_list = _unconfirmed + _partials
                if _poll_list:
                    asyncio.create_task(self._poll_fills_until_confirmed(_poll_list, list(holdings or []) + _us_baseline, cyc=cyc))
            elif risk_approved and not LIVE_TRADING:
                await self._emit({"type":"execution_ready","message":"✅ 리스크 승인 (LIVE_TRADING=False — 실주문 생략)","draft":order_draft})
            else:
                await self._emit({"type":"execution_skipped","message":"리스크 미승인 — 실행 없음"})

            # [9.5] 프롭트레이딩팀장 체결 보고 — 결정론적 템플릿 (사장 피드백 2026-05-18)
            # 더 이상 LLM을 호출하지 않는다: 보고 내용이 전부 사실(종목·수량·체결여부·체결가·사유)
            # 이므로 고정 양식으로 조립하면 일관·정확·무비용. (체결 vs 접수 혼동도 구조적으로 차단.)
            try:
                def _nm(c: str) -> str:
                    c = str(c).strip()
                    return f"{c}({name_map[c]})" if c in name_map else c
                def _disp(tk: str) -> str:
                    try: return _nm(tk)            # name_map 있으면 코드(종목명)
                    except Exception: return tk
                def _price_str(e: Dict) -> str:
                    fp = e.get("fill_price")
                    if not fp or fp <= 0:
                        return ""
                    label = "체결가" if e.get("filled") else "참고가"
                    return f" · {label} {fp:,.0f}원" if (e.get("fill_currency") != "USD") else f" · {label} ${fp:,.2f}"
                # 사유 맵 — 매수: order_obj, 매도: 사후관리실장 directive
                _reason_by_tk: Dict[str, str] = {}
                for o in (order_obj.get("orders") or []):
                    _otk = str(o.get("ticker") or "").strip()
                    if _otk:
                        _reason_by_tk[_otk] = (o.get("reason") or "").strip()
                _HOLD = {"보유", "유지", "hold", "keep", "유보", "관망"}
                _sell_dir = {str(k).strip(): str(v).strip() for k, v in (sell_directives or {}).items()
                             if str(v).strip().lower() not in _HOLD}

                _buy_rows, _sell_rows = [], []
                for e in (exec_results or []):
                    _tk = str(e.get("ticker") or "").strip()
                    _qty = e.get("qty", 0)
                    _is_sell = (e.get("side") or "buy") == "sell"
                    if e.get("watch"):
                        _badge = "⏳ 분봉 대기"
                        _note = e.get("fill_note") or f"{e.get('result','')[:80]}"
                    elif e.get("filled"):
                        _badge = "✅ 체결확인"
                        _note = (e.get("fill_note") or "").strip()
                    elif e.get("accepted"):
                        _badge = "📨 접수(체결 확인 중)"
                        _note = "호가 미체결 가능 — 5분마다 보유 변동 반복 확인(체결 시 누적 +1)"
                    else:
                        _badge = "⚠️ 실패"
                        _note = (e.get("result") or "")[:100]
                    _reason = _reason_by_tk.get(_tk) or (
                        f"사후관리실장 판단 — {_sell_dir.get(_tk, '')}" if _is_sell else "")
                    _line = f"- {_badge} {_disp(_tk)} {_qty}주{_price_str(e)}"
                    if _note:
                        _line += f"\n   └ {_note}"
                    if _reason:
                        _line += f"\n   └ 사유: {_reason[:180]}"
                    (_sell_rows if _is_sell else _buy_rows).append(_line)

                _filled_cnt2 = sum(1 for e in (exec_results or []) if e.get("filled"))
                _accepted_only2 = sum(1 for e in (exec_results or []) if e.get("accepted") and not e.get("filled"))
                _watch_cnt = sum(1 for e in (exec_results or []) if e.get("watch"))

                _parts = ["🧾 [프롭트레이딩팀장 — 사이클 체결 보고]"]
                if not exec_results:
                    _why = "; ".join(order_obj.get("sizing_notes") or []) or (
                        "리스크 미승인" if not risk_approved else "잔고·한도 조건상 신규/청산 주문 없음")
                    _parts.append(f"\n▶ 이번 사이클 신규 체결 없음 — {_why[:240]}")
                else:
                    if _buy_rows:
                        _parts.append("\n▶ 매수\n" + "\n".join(_buy_rows))
                    if _sell_rows:
                        _parts.append("\n▶ 매도\n" + "\n".join(_sell_rows))
                _tail = f"\n▶ 집계 — 이번 사이클 체결 {_filled_cnt2}건"
                _extra = []
                if _accepted_only2: _extra.append(f"접수 {_accepted_only2}건")
                if _watch_cnt:     _extra.append(f"분봉대기 {_watch_cnt}건")
                if _extra:         _tail += f" (+ {' · '.join(_extra)})"
                _tail += f" / 금일 누적 체결 {self._trades_executed}건"
                _parts.append(_tail)
                _facts = "\n".join(_parts)      # 결정론 집계 — LLM 입력이자 실패 시 폴백
                _trader_msg = _facts

                # 프롭트레이딩팀장 배선 (사장 지시 2026-07-22).
                # 그동안 이 보고는 **코드가 조립한 템플릿을 팀장 명의로 내보내기만** 했고,
                # specialists.create_trader 로 정의된 페르소나는 사이클에서 한 번도 호출된 적이
                # 없었다(.think() 0회 — git 이력상 처음부터 배선 누락). 이제 실제로 말하게 한다.
                # · 수치는 결정론 집계(_tail)를 그대로 덧붙여 LLM 이 숫자를 지어낼 여지를 없앤다.
                # · 잔고는 이 스코프에 없으므로 '보유 현황 단정 금지'를 프롬프트로 못박는다
                #   (트레이더 시스템 프롬프트 절대규칙 6 과 충돌 방지).
                # · fail-open — 실패·빈응답이면 종전 템플릿을 그대로 쓴다.
                try:
                    _llm_rep = await self.trader.think(
                        "다음은 이번 사이클의 결정론 체결 집계입니다. 이 사실만 근거로 사장님께 올릴 "
                        "보고를 2~5문장 자연스러운 한국어 산문체로 작성하십시오.\n"
                        "- 주어진 수치 외에 새로운 숫자를 만들지 마십시오.\n"
                        "- 계좌 잔고 스냅샷은 이 보고의 범위가 아닙니다. 보유 수량·포지션 정리 여부를 "
                        "단정하지 말고, 아래 집계에 적힌 체결 사실만 서술하십시오.\n"
                        "- 집계 줄은 시스템이 따로 덧붙이므로 반복하지 마십시오.\n\n"
                        f"[이번 사이클 체결 집계]\n{_facts}")
                    _llm_rep = (_llm_rep or "").strip()
                    if _llm_rep and not _llm_rep.startswith("[프롭트레이딩팀장 에러"):
                        _trader_msg = f"🧾 [프롭트레이딩팀장 — 사이클 체결 보고]\n\n{_llm_rep}\n{_tail}"
                except Exception as _le:  # noqa: BLE001
                    logger.warning(f"프롭트레이딩팀장 LLM 보고 생략(결정론 폴백): {_le}")

                await self._emit({"type":"agent_msg","agent":"프롭트레이딩팀장", "message": _trader_msg})
                self.cycle_log.log("REPORT", "프롭트레이딩팀장", _trader_msg)
            except Exception as _te:
                logger.warning(f"트레이더 체결 보고 조립 실패: {_te}")

            # [9.7] 사이클 자기검증 — 체결률·거부를 ok:true 로 묻지 말고 지표·경고로 가시화 (2026-05-27 진단).
            try:
                _real = [e for e in (exec_results or []) if not e.get("watch")]
                _filled = sum(1 for e in _real if e.get("filled"))
                _acc = sum(1 for e in _real if e.get("accepted"))
                metrics.gauge("cycle_orders", float(len(_real)), uid=str(self.uid), session=str(session))
                metrics.gauge("cycle_filled", float(_filled), uid=str(self.uid), session=str(session))
                metrics.gauge("cycle_accepted", float(_acc), uid=str(self.uid), session=str(session))
                _health = cycle_health_warnings(exec_results)
                if _health:
                    metrics.incr("cycle_health_warning", by=float(len(_health)), uid=str(self.uid))
                    _wmsg = "⚠ 사이클 자기검증 경고:\n" + "\n".join(f"  • {w}" for w in _health)
                    await self._emit({"type": "agent_msg", "agent": "리스크관리실장", "message": _wmsg})
                    self.cycle_log.log("HEALTH", "시스템", _wmsg)
                    notifier.alert("WARN", "사이클 자기검증 경고", "; ".join(_health),
                                   dedup_key=f"cycle_health_{self.uid}")
            except Exception as _he:
                logger.warning(f"사이클 자기검증 실패(무시): {_he}")
            cyc.exec_results = exec_results

    async def _record_buy_thesis(self, rec: Dict, cyc) -> None:
        """매수 체결 직후 펀드기획팀장이 진입 thesis(목표가/손절가/계획 보유기간/사유)를 작성·영속.
        fire-and-forget — 실패해도 매매 흐름엔 영향 없음.
        사장 지시 2026-05-28(우선순위 3): 무계획 단타 매매 차단."""
        code = str(rec.get("ticker", "")).strip()
        if not code or rec.get("side") != "buy" or not rec.get("ok"):
            return
        # 슬리브 ETF 가드(사장 지시 2026-06-09): 채권·원자재 ETF 는 주식 thesis(목표가/손절가) 대상이
        # 아니다 — 슬리브 thesis(infra.sleeve_thesis, key=슬리브)에 계획 보유기간만 기록하고 종료.
        _sleeve_spec = sleeve_for_code(code)
        if _sleeve_spec is not None:
            try:
                from infra import sleeve_thesis
                if sleeve_thesis.get(self.uid, _sleeve_spec.key, code):
                    return  # 중복가드(동기 실행부 + 폴링 확정 경로)
                # 코드→듀레이션(채권 short/mid/long, 원자재 na) → 계획 보유기간
                _dur_map = {str(c).strip().upper(): d
                            for c, _n, d, *_ in (list(_sleeve_spec.pool_kr) + list(_sleeve_spec.pool_us))}
                _hold_by_dur = {"short": 72, "mid": 168, "long": 336, "na": 168}
                _dur = _dur_map.get(code.upper(), "na")
                _hold_h = _hold_by_dur.get(_dur, 168)
                fill_price = float(rec.get("fill_price") or rec.get("avg_cost") or 0.0)
                sleeve_thesis.record(self.uid, _sleeve_spec.key, code, {
                    "entry_ts": _now_kst_iso(),
                    "entry_price": fill_price,
                    "planned_hold_hours": _hold_h,
                    "entry_reason": f"{_sleeve_spec.macro_keyword} 자산배분",
                    "source_agent": "포트폴리오기획팀장",
                })
                # 진입 thesis 발화자는 포트폴리오기획팀장(진입 계획 전담). 매매판단은 슬리브 매니저.
                await self._emit({"type": "agent_msg", "agent": "포트폴리오기획팀장",
                    "message": (f"📌 [{_sleeve_spec.macro_keyword} 진입 thesis] {code} 체결가 {fill_price:,.2f} | "
                                f"계획 보유 {_hold_h}h ({_dur})")})
            except Exception as _be:
                logger.warning(f"[{_sleeve_spec.key}] thesis 기록 실패 {code}: {_be}")
            return
        try:
            from agents.specialists import parse_fund_plan
            from infra import position_thesis
            # 중복가드(2026-05-29): 동기 실행부와 폴링 확정 경로가 같은 종목을 중복 기록하지 않도록,
            # 이미 thesis 가 있으면 LLM 호출·재기록을 건너뛴다.
            if position_thesis.get(self.uid, code):
                return
            quant_brief = (getattr(cyc, "quant_report", "") or "")[:1200]
            order_obj = getattr(cyc, "order_obj", None) or {}
            buy_reason = ""
            for o in (order_obj.get("orders") or []):
                if str(o.get("ticker", "")).strip() == code and o.get("side") == "buy":
                    buy_reason = (o.get("reason") or "")[:300]
                    break
            # C-#5(2026-06-12): 코드 기준 정본 종목명(뉴스 LLM 의 이름 환각과 분리). 네트워크 호출은 스레드로.
            from tools.market_data import canonical_name
            try:
                auth_name = await asyncio.to_thread(canonical_name, code)
            except Exception:
                auth_name = ""
            disp_name = auth_name or code
            # B-#2(2026-06-12): 종목 무관한 news_report 앞부분(센티 리포트=삼성으로 시작) 통째 주입 금지 —
            # 이 종목 단락만 발췌(없으면 빈 값). 타종목 사유 오염(003490 thesis 에 삼성 사유) 차단.
            news_brief = _extract_code_news(getattr(cyc, "news_report", "") or "", code, auth_name)[:600]
            fundamental_brief = ""
            try:
                from infra import fundamental_research_store
                from tools.fundamental_rigor import format_research_for_prompt
                _fr = (getattr(cyc, "fundamental_research", {}) or {}).get(code) or fundamental_research_store.latest(self.uid, code)
                fundamental_brief = format_research_for_prompt(_fr)
            except Exception as _fre:
                logger.debug(f"[펀더멘털리서치] thesis 주입 생략 {code}: {_fre}")
            fill_price = float(rec.get("fill_price") or rec.get("avg_cost") or 0.0)
            ccy = rec.get("fill_currency") or "KRW"
            prompt = (f"[plan 모드]\n종목: {disp_name}({code})\n체결가: {fill_price:,.2f} {ccy}\n"
                      f"매수 사유(주식운용실장): {buy_reason}\n\n"
                      f"계량팀 리포트 발췌:\n{quant_brief}\n\n"
                      f"이 종목({disp_name}) 뉴스 발췌:\n{news_brief or '(이 종목 관련 뉴스 없음)'}\n\n"
                      f"장기 펀더멘털 리서치 참고(ai-berkshire 방식, advisory):\n{fundamental_brief or '(특이 리서치 없음)'}\n\n"
                      f"⚠️ 오직 이 종목({disp_name}/{code})의 매수 사유만 쓰십시오 — 다른 종목"
                      f"(예: 지수 1위 종목)의 뉴스·사유를 끌어오지 마십시오.\n"
                      f"이 매수의 thesis 를 4줄(목표가/손절가/계획 보유기간/진입 사유 요약)로만 응답하십시오.")
            text = await self.fund_planner.think(prompt)
            parsed = parse_fund_plan(text)
            thesis = {
                "entry_ts": _now_kst_iso(),
                "entry_price": fill_price,
                "fill_currency": ccy,
                "name": disp_name,
                "target_price": parsed.get("target_price"),
                "stop_price": parsed.get("stop_price"),
                "planned_hold_hours": parsed.get("planned_hold_hours"),
                "entry_reason": parsed.get("entry_reason") or buy_reason[:200],
                "source_agent": "포트폴리오기획팀장",
            }
            position_thesis.record(self.uid, code, thesis)
            await self._emit({"type": "agent_msg", "agent": "포트폴리오기획팀장",
                "message": (f"📌 [진입 thesis] {disp_name}({code}) 체결가 {fill_price:,.2f} {ccy}\n"
                            f"목표가 {parsed.get('target_price') or '?'} | 손절가 {parsed.get('stop_price') or '?'} | "
                            f"계획 보유 {parsed.get('planned_hold_hours') or '?'}h\n"
                            f"진입 사유: {parsed.get('entry_reason') or buy_reason[:150]}")})
            # Phase A 복기 기록(사장 지시 2026-07-31): 진입 스냅샷 + 벤치마크 진입 레벨을 pending 으로.
            # 전량 청산이 원장에서 확인되면 Phase B 가 수익률·알파·LLM 복기로 확정한다. fail-open.
            try:
                from infra import trade_reflections
                _cands = ((getattr(cyc, "committee", None) or {}).get("candidates") or [])
                _cmt_row = next((c for c in _cands if c.get("code") == code), {})
                _chief_op = next((o for o in (_cmt_row.get("opinions") or [])
                                  if o.get("role") == "chief"), {})
                _bench = await asyncio.to_thread(trade_reflections.bench_snapshot, code)
                trade_reflections.record_pending(self.uid, code, {
                    "name": disp_name, "entry_ts": thesis["entry_ts"],
                    "entry_price": fill_price, "ccy": ccy,
                    "qty": int(rec.get("qty") or 0),
                    "quant_score": (getattr(cyc, "_quant_scores", {}) or {}).get(code),
                    "alpha_tag": (getattr(cyc, "_quant_tags", {}) or {}).get(code),   # P2 알파 귀인
                    "committee_stance": _chief_op.get("stance") or _cmt_row.get("decision"),
                    "committee_confidence": _chief_op.get("confidence"),
                    "entry_reason": thesis.get("entry_reason") or "",
                    "target_price": thesis.get("target_price"),
                    "stop_price": thesis.get("stop_price"),
                    "planned_hold_hours": thesis.get("planned_hold_hours"),
                    "bench_name": _bench.get("bench_name"),
                    "bench_entry": _bench.get("level"),
                })
            except Exception as _tre:
                logger.warning(f"[복기] pending 기록 실패 {code}: {_tre}")
        except Exception as e:
            logger.warning(f"[펀드기획] thesis 기록 실패 {code}: {e}")

    def _sync_thesis_with_current_holdings(self, holdings, *, drop_foreign: bool = True) -> None:
        """전량 매도된 종목의 thesis 제거 (보유 0주가 된 코드 자동 정리).
        코드리뷰 #3(2026-06-09): broker._overseas_holdings()는 실패해도 예외 대신 []를 반환하므로,
        해외 조회가 비신뢰(drop_foreign=False)면 US(비-6자리) thesis 를 삭제 대상에서 제외한다 —
        US장 데이터결손 시 라이브 보유분의 계획 보유기간 thesis 가 통째로 오삭제되던 것 방지."""
        codes = [str(h.get("code", "")).strip() for h in (holdings or []) if int(h.get("qty") or 0) > 0]

        def _keep_with_foreign(stored_keys):
            # drop_foreign=False면 현재 저장된 US 코드 thesis 를 '보유 중'으로 취급해 삭제 회피.
            if drop_foreign:
                return codes
            return codes + [c for c in stored_keys if not _is_kr_code(c)]
        try:
            from infra import position_thesis
            removed = position_thesis.sync_with_holdings(
                self.uid, _keep_with_foreign(position_thesis.get_all(self.uid).keys()))
            if removed:
                logger.info(_thesis_prune_msg("펀드기획", removed))
        except Exception as e:
            logger.warning(f"[펀드기획] thesis 동기화 실패: {e}")
        # 슬리브 thesis(채권·원자재)도 동일하게 정리(전량 매도된 슬리브 ETF 의 sleeve_thesis 제거).
        try:
            from infra import sleeve_thesis
            for _spec in SLEEVES:
                removed_s = sleeve_thesis.sync_with_holdings(
                    self.uid, _spec.key, _keep_with_foreign(sleeve_thesis.get_all(self.uid, _spec.key).keys()))
                if removed_s:
                    logger.info(_thesis_prune_msg(_spec.key, removed_s))
        except Exception as e:
            logger.warning(f"[슬리브] thesis 동기화 실패: {e}")

    async def _emit_advisory_signals(self, cyc):
        """ROI 정량 신호(2026-06-15) 매 사이클 1회 산출·섀도우 보고 — 거래 동작 변경 없음.
        #3 레짐(LLM 매크로→de-risk score) · #2 퀀트 IC 확신도 · #5 데이터 품질(일봉·원장 괴리).
        실제 사이징/게이트 반영은 ENABLE_IC_SIZING·ENABLE_DILUTION_GATE 토글로 별도(기본 OFF)."""
        try:
            from tools.regime import regime_score_from_macro
            from tools.agent_scorecard import quant_confidence
            from tools.market_data import load_daily_csv
            from infra import data_quality
            rg = regime_score_from_macro(getattr(cyc, "macro_report", "") or "")
            conf, ic, n = quant_confidence(self.uid)
            cyc._regime = rg; cyc._quant_confidence = conf
            today = _now_kst_iso()[:10]
            issues: List[str] = []          # 로그·섀도우용 — 전체 가시성(자동보정·전이 괴리 포함)
            alert_issues: List[str] = []     # 푸시 알림용 — 자가치유/전이 노이즈 제외
            for h in (getattr(cyc, "stock_holdings", None) or [])[:12]:
                code = str(h.get("code", "")).strip()
                if not _is_kr_code(code):
                    continue
                df = load_daily_csv(code)
                if df is not None:
                    _ci = data_quality.csv_issues(code, df, today)
                    issues += _ci; alert_issues += _ci   # CSV 무결성은 즉시 알림
            try:
                snap = await self.broker.kr_account_snapshot()
                repaired = trade_ledger.repair_from_recent_partial_orders(
                    self.uid, snap.get("holdings") or [])
                if repaired:
                    logger.warning(f"[원장 uid={self.uid}] 부분체결 폴링 유실 보정: {repaired}")
                    issues += [f"원장 자동보정 — {x}" for x in repaired]   # 로그만(자가치유 성공 → 푸시 X)
                # 전이성 괴리(부분체결→폴링 사이, 다음 폴링이 채움) 푸시 노이즈 차단: 로그엔 전체,
                # 알림엔 threshold 사이클 연속 지속분만(2026-06-22 uid2 357870 오탐 14회 해소).
                _diffs = trade_ledger.reconcile(self.uid, snap.get("holdings") or [])
                issues += data_quality.ledger_drift_issues(_diffs)
                _dstreak = _load_drift_streak(self.uid)
                _persistent = data_quality.persistent_drift_issues(_diffs, _dstreak)
                _save_drift_streak(self.uid, _dstreak)
                alert_issues += data_quality.ledger_drift_issues(_persistent)
            except Exception:
                pass
            _ic = f"{ic:+.3f}·n{n}" if ic is not None else f"표본부족·n{n}"
            # 사장 지시 2026-07-20: 대시보드 표시엔 '지속(alert) 이슈'만 노출 — 부분체결 진행 중의
            # 전이성 원장 괴리(예: 357870 KIS 26 vs 원장 0, 다음 폴링이 채움)를 실시간 경고로
            # 띄우던 오탐 차단. 전체 이슈(전이 포함)는 아래 로그에만 남긴다.
            await self._emit({"type": "agent_msg", "agent": "운용지원실장",
                "message": (f"🧭 [정량 신호 — 섀도우/관찰] 레짐 {rg['regime']}(de-risk {rg['score']:.2f}) · "
                            f"퀀트 IC확신도 {conf:.2f}[{_ic}] · 데이터품질 이슈 {len(alert_issues)}건"
                            + (f" ⚠️ {alert_issues[0]}" if alert_issues else ""))})
            if issues:
                logger.warning(f"[데이터품질 uid={self.uid}] {len(issues)}건: {issues[:6]}")
            if alert_issues:
                try:
                    notifier.alert("WARN", "데이터 품질 이슈", "; ".join(alert_issues[:6]),
                                   dedup_key=f"dq_{self.uid}")
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"_emit_advisory_signals 실패(무시): {e}")

    async def _cyc_stage_report(self, cyc):
            session = cyc.session
            market_open = cyc.market_open
            news_articles = cyc.news_articles
            news_report = cyc.news_report
            index_report = cyc.index_report
            macro_report = cyc.macro_report
            quant_report = cyc.quant_report
            candidate_codes = cyc.candidate_codes
            target_codes = cyc.target_codes
            sell_directives = cyc.sell_directives
            holdings = cyc.holdings
            order_obj = cyc.order_obj
            buying_power = cyc.buying_power
            risk_result = cyc.risk_result
            risk_approved = cyc.risk_approved
            approved_orders = cyc.approved_orders
            exec_results = cyc.exec_results
            # [10] REPORT
            self.current_state = SwarmState.REPORT
            report = _build_cycle_final_report(
                exec_results, risk_result, (order_obj or {}).get("sizing_notes"),
                dart_vetoed=getattr(cyc, "dart_vetoed", None))
            self.cycle_log.final_report = report
            # 사장 지시 2026-05-28(우선순위 3): 사이클 종료 시 thesis 와 현재 보유 동기화 — 전량 매도된 종목 thesis 자동 제거.
            try:
                _sync_snap = await self.broker.kr_account_snapshot()
                _all_holdings = list(_sync_snap.get("holdings") or [])
                # US 보유까지 보장 위해 best-effort 합산. 코드리뷰 #3: 해외조회가 빈 결과면(실패 가능 —
                # _overseas_holdings 는 실패해도 [] 반환) US thesis 오삭제를 막도록 drop_foreign=False.
                _us = []
                try:
                    _us = list(await self.broker._overseas_holdings() or [])
                    _all_holdings += _us
                except Exception:
                    _us = []
                self._sync_thesis_with_current_holdings(_all_holdings, drop_foreign=bool(_us))
                # Phase B 복기 확정(사장 지시 2026-07-31): pending 종목이 보유에서 사라졌으면 청산으로 보고
                # 복기 확정을 백그라운드로 시도한다. US 조회가 비신뢰(_us 빈값)면 US 코드는 건너뛰고
                # (thesis 오삭제 방지와 동일 가드), 원장에 매도 체결이 없으면 resolve_position 이
                # pending 을 유지한다(KIS 결제 과도기 잔고 글리치 오발동 방지 — 이중 가드).
                try:
                    from infra import trade_reflections
                    _codes_now = {str(h.get("code", "")).strip() for h in (_all_holdings or [])
                                  if int(h.get("qty") or 0) > 0}
                    for _rc in trade_reflections.pending_codes(self.uid):
                        if _rc in _codes_now or (not _is_kr_code(_rc) and not _us):
                            continue
                        asyncio.create_task(
                            trade_reflections.resolve_position(self.uid, _rc, emit=self._emit))
                except Exception as _tre:
                    logger.warning(f"[복기] 청산 감지 스킵(fail-open): {_tre}")
            except Exception as _se:
                logger.warning(f"[펀드기획] 사이클 종료 thesis 동기화 스킵: {_se}")
            await self._emit({"type":"cycle_complete","report":report,"trades_total":self._trades_executed})
            self._cycle_history.append(self.cycle_log.to_dict())

            # ── QuantInSight 이식(2026-07-18): 위원회 기록에 체결 요약(portfolio) 채워넣기 ──
            _committee = getattr(cyc, "committee", None) or {}
            _committee["edge_evidence"] = getattr(cyc, "edge_evidence", {})
            _committee["systematic_policy"] = getattr(cyc, "systematic_policy", {})
            _committee["portfolio_risk"] = getattr(cyc, "portfolio_risk_notes", [])
            if _committee is not None:
                try:
                    _orders_c, _sells_c = [], []
                    _total = float(buying_power.get("total_eval") or 0.0) or 1.0
                    for e in (exec_results or []):
                        if not isinstance(e, dict):
                            continue
                        _side = str(e.get("side", "")).lower()
                        _row = {"code": e.get("ticker") or e.get("code") or "",
                                "name": e.get("name") or e.get("ticker") or e.get("code") or "",
                                "qty": e.get("qty"), "price": e.get("price"),
                                "ok": bool(e.get("ok", e.get("filled", False)))}
                        try:
                            _amt = float(e.get("qty") or 0) * float(e.get("price") or 0)
                            _row["amount"] = _amt
                            _row["weight"] = round(_amt / _total, 4)
                        except (TypeError, ValueError):
                            pass
                        (_sells_c if _side in ("sell", "매도") else _orders_c).append(_row)
                    _committee["portfolio"] = {
                        "orders": _orders_c, "sells": _sells_c,
                        "cash_weight": round(float(buying_power.get("cash") or 0.0) / _total, 4),
                        "cash_note": (f"예수금 {float(buying_power.get('cash') or 0.0):,.0f}원 · "
                                      f"총평가 {float(buying_power.get('total_eval') or 0.0):,.0f}원"),
                    }
                except Exception as _pe:
                    logger.warning(f"[위원회] portfolio 요약 실패(기록은 유지): {_pe}")

            # ── Persist cycle to SQLite (사장 지시 2026-05-14 — 백테스트/장기 분석용) ──
            new_cycle_id = None
            try:
                new_cycle_id = cycle_store.record_cycle({
                    "committee": _committee,
                    "uid": self.uid,
                    "started_at": self.cycle_log.started_at,
                    "ended_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
                    "session": session, "market_open": market_open,
                    "news_count": len(news_articles),
                    "candidate_codes": candidate_codes, "target_codes": target_codes,
                    "sell_directives": sell_directives,
                    "orders_planned": order_obj.get("orders", []),
                    "orders_executed": exec_results,
                    "risk_approved": risk_approved,
                    "risk_report": (getattr(cyc, "risk_report_full", None)
                                    or (risk_result.get("report", "") if isinstance(risk_result, dict) else ""))[:8000],
                    "macro_report": (macro_report or "")[:8000],
                    "quant_report": (quant_report or "")[:8000],
                    "news_report":  (news_report  or "")[:8000],
                    "final_report": (report or "")[:4000],
                    "bp_cash": float(buying_power.get("cash") or 0.0),
                    "bp_total_eval": float(buying_power.get("total_eval") or 0.0),
                    "bp_pnl_ratio": float(buying_power.get("pnl_ratio") or 0.0),
                })
                # holdings_history 갱신 — 보유기간 P&L 추적용
                try:
                    for h in (holdings or []):
                        cycle_store.upsert_holding_seen(h.get("code",""), int(h.get("qty") or 0), float(h.get("avg_price") or 0.0), uid=self.uid)
                    cycle_store.reconcile_holdings([h.get("code","") for h in (holdings or [])], uid=self.uid)
                except Exception: pass
            except Exception as _e:
                logger.warning(f"cycle_store 기록 실패: {_e}")

            # ── 운용지원실장 워커 spawn (사장 피드백 2026-05-18):
            #   더 이상 파이썬이 키워드로 도메인을 추측하지 않는다. 항상 운용지원실장(ops_support)
            #   1개만 spawn → 워커가 스스로 진단(Phase A) 후 ① 고칠 게 없으면 팀장 미호출·실장 선
            #   종료, ② 고칠 게 있으면 담당 팀장에게 구체 수정 지시를 위임(자식 프로세스 spawn).
            #   진단/위임/결과 메시지는 모두 워커가 [OPS#cycle] 마커로 직접 대시보드에 기록한다. ──
            if new_cycle_id:
                # 시간당 쓰로틀 2026-06-05: 매 사이클 spawn(낭비·크래시 무한반복) 폐지 — per-uid 마커로
                # 직전 ops 실행에서 OPS_THROTTLE_SEC(기본 1시간) 경과 시에만 spawn(시간당 적극 튜닝 케이던스).
                try:
                    from infra.ops_throttle import ops_due, read_last_run, write_last_run
                    from config import OPS_THROTTLE_SEC
                    _marker = _Path(self.equity_path).parent / ".ops_last_run"
                    _now_ts = time.time()
                    if ops_due(read_last_run(_marker), _now_ts, OPS_THROTTLE_SEC):
                        self._spawn_ops_support_worker(new_cycle_id, role="ops_support")
                        write_last_run(_marker, _now_ts)
                        await self._emit({"type":"agent_msg","agent":"운용지원실장",
                            "message": (f"🛠 [OPS#{new_cycle_id}] 직전 사이클을 백그라운드로 점검합니다 — "
                                        f"조정할 전략 파라미터가 있으면 바로 이어서 보고드리고, 바꿀 게 없으면 조용히 넘어갑니다 "
                                        f"(후속 메시지가 없으면 '이상 없음'으로 보시면 됩니다).")})
                    else:
                        logger.info(f"운용지원 워커 스킵 — 시간당 쓰로틀 미경과 (uid={self.uid})")
                except Exception as _e:
                    logger.warning(f"ops_support 워커 spawn 실패: {_e}")

    def stop(self):
        # 사장 보고 2026-06-10 회귀수정: 즉시중지(_stop_uid)가 task 를 cancel 하면 start_continuous
        # 말미의 current_state=STOPPED 설정이 스킵돼 상태가 stale(MONITORING/OFF_HOURS)로 남는다.
        # → get_status().current_state 가 STOPPED 가 아니어서 대시보드 상태배지·실행/중지 버튼이 어긋남.
        # stop()에서 권위상태를 STOPPED 로 직접 정정(이 메서드는 유저 중지 _stop_uid 에서만 호출).
        self._stop_event.set()
        self.current_state = SwarmState.STOPPED
    def get_status(self):
        _next_cycle_sec = max(0, int(PERIODIC_CYCLE_SEC - (time.time() - self._last_cycle_at))) if self._last_cycle_at else None
        return {"current_state":self.current_state.value,"session":get_current_session(),
            "time_kst":_now_kst().strftime("%H:%M:%S"),"is_trading":is_market_session_now(),
            "validation_attempts":self.validation_attempts,"cycle_history_count":len(self._cycle_history),
            "pending_news":len(self._pending_news),
            "next_cycle_sec":_next_cycle_sec,
            "trades_executed":self._trades_executed,"trade_log":self._trade_log[-5:],
            "live_trading":LIVE_TRADING,"strategy":runtime.active(uid=self.uid),
            "news_monitor":self.news_monitor.get_status(),
            "watchlist":list(INDEX_WATCHLIST.keys()),
            "schedule":{k:v["desc"] for k,v in SCHEDULE.items()}}
    def get_history(self): return self._cycle_history

# Phase 2 멀티테넌트: 전역 단일 스왐(get_swarm/_swarm) 폐지.
# 스왐은 유저별 UserContext.swarm(lazy ArquantOrchestrator(ctx))로 생성되며,
# 라이프사이클(start/stop)은 server/app.py 의 _start_uid/_stop_uid 가 관리한다.
def get_swarm():
    raise RuntimeError(
        "get_swarm() is retired in Phase 2 — use UserContext.swarm (per-uid). "
        "server/app.py routes by request.state.user_id → REGISTRY.get_or_create(uid).swarm.")

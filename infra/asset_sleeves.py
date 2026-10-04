"""자산슬리브 엔진 — 채권·원자재 등 매크로 자산배분 트랙을 단일 SleeveSpec 으로 일반화.

채권 트랙(2026-06-08)을 슬리브#1로 승격하고 원자재를 슬리브#2로 추가(2026-06-09).
main_swarm 의 채권 전용 순수 함수가 여기로 옮겨와 spec/keyword 인자를 받는 범용 함수가 된다
(동작 보존 — 채권=슬리브#1). 풀은 화이트리스트(LLM 티커 환각 방지) — 코드 오류=실주문 실패라
검증된 코드만 config 에 등재.

KR/US 비대칭 주의(CLAUDE.md): 매수/매도/세션풀/비중 전 경로에서 KR(6자리)·US(티커)를 동등
처리하고, KRW 한도와 USD 평가를 섞지 않는다(current_sleeve_weight 의 usdkrw 환산).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import config

# main_swarm 의 KR 세션/코드 판정과 동일(순환 import 회피 위해 로컬 정의 — 안정 상수).
_KR_SESSIONS = ("KR_TRADING", "KR_PRE_MARKET", "KR_AFTER_MARKET", "KR_CLOSE_REVIEW")


def _is_kr_code(code: Any) -> bool:
    """6자리 숫자면 KR, 아니면 US(티커). (main_swarm._is_kr_code 와 동일 규칙)"""
    s = str(code or "").strip()
    return s.isdigit() and len(s) == 6


@dataclass(frozen=True)
class SleeveSpec:
    """한 자산배분 트랙(채권/원자재 등)을 완전 기술하는 명세."""
    key: str                       # "bond" | "commodity" — thesis 파일·내부 키
    manager_name: str              # "채권운용실장" | "원자재운용실장" — 라우팅 키(한글)
    role: str                      # "bond_manager" | "commodity_manager" — 모델/토큰 키
    macro_keyword: str             # "채권" | "원자재" — 매크로% 파싱 단어
    decision_keyword: str          # "채권결정" | "원자재결정" — LLM 마지막 줄 파싱
    pool_kr: Tuple[tuple, ...]     # (code, name, duration, kind, fx)
    pool_us: Tuple[tuple, ...]
    enable_key: str                # 런타임 토글 키
    target_max_key: str            # 비중 상한 키
    band_key: str                  # 리밸런싱 데드존 키
    per_cycle_key: str             # 전용 사이클 매수 예산비율 키


BOND_SLEEVE = SleeveSpec(
    key="bond", manager_name="채권운용실장", role="bond_manager",
    macro_keyword="채권", decision_keyword="채권결정",
    pool_kr=tuple(config.BOND_ETF_POOL_KR), pool_us=tuple(config.BOND_ETF_POOL_US),
    enable_key="ENABLE_BOND_ETF", target_max_key="BOND_TARGET_MAX_PCT",
    band_key="BOND_REBALANCE_BAND_PCT", per_cycle_key="BOND_PER_CYCLE_RATIO",
)
COMMODITY_SLEEVE = SleeveSpec(
    key="commodity", manager_name="원자재운용실장", role="commodity_manager",
    macro_keyword="원자재", decision_keyword="원자재결정",
    pool_kr=tuple(config.COMMODITY_ETF_POOL_KR), pool_us=tuple(config.COMMODITY_ETF_POOL_US),
    enable_key="ENABLE_COMMODITY_ETF", target_max_key="COMMODITY_TARGET_MAX_PCT",
    band_key="COMMODITY_REBALANCE_BAND_PCT", per_cycle_key="COMMODITY_PER_CYCLE_RATIO",
)
SLEEVES: List[SleeveSpec] = [BOND_SLEEVE, COMMODITY_SLEEVE]


def get_sleeve(key: str) -> SleeveSpec:
    for s in SLEEVES:
        if s.key == key:
            return s
    raise KeyError(key)


def sleeve_codes(spec: SleeveSpec) -> set:
    """한 슬리브의 KR+US 풀 코드 집합(대문자)."""
    return {str(c).strip().upper() for c, *_ in (spec.pool_kr + spec.pool_us)}


def all_sleeve_pool_codes() -> set:
    """전 슬리브 풀 코드(KR+US) 합집합(대문자). 주식 매도 트랙에서 슬리브를 제외할 때 사용.
    세션 무관 전체를 쓰는 이유: 슬리브 ETF 코드는 어느 세션이든 그 자산군이며, 주식 자동
    익절/손절·사후관리 주식 매도가 절대 손대면 안 된다(반대편 시장 슬리브 보유분도 보호)."""
    out: set = set()
    for s in SLEEVES:
        out |= sleeve_codes(s)
    return out


def sleeve_for_code(code: str) -> Optional[SleeveSpec]:
    """코드가 속한 슬리브 spec(없으면 None). 가드레일 집중도 상한 선택에 사용."""
    cu = str(code or "").strip().upper()
    for s in SLEEVES:
        if cu in sleeve_codes(s):
            return s
    return None


# ── 매크로 자산배분% 파싱 ──────────────────────────────────────────────────────
def parse_macro_sleeve_pct(text: Optional[str], keyword: str) -> Optional[float]:
    """글로벌리서치팀장 매크로 보고에서 권고 '<keyword> X%' 비중을 분수(0.01)로 추출한다.
    '자산 배분 권고' 라인 우선(거기서 첫 매치 — 직전 괄호값 배제), 없으면 전체 첫 매치.
    마크다운 별표(채권 **4%**)도 허용. 못 찾으면 None(→ 호출부 fail-safe).
    keyword 예: '주식' | '채권' | '원자재' | '현금'."""
    if not text:
        return None
    s = str(text)
    pat = rf"{re.escape(keyword)}\s*\*{{0,2}}\s*(\d+(?:\.\d+)?)\s*%"
    anchor = s.find("자산 배분 권고")
    if anchor >= 0:
        m = re.search(pat, s[anchor:anchor + 200])
        if m:
            return float(m.group(1)) / 100.0
    m = re.search(pat, s)
    return float(m.group(1)) / 100.0 if m else None


# ── 세션 연동 풀 ───────────────────────────────────────────────────────────────
def sleeve_pool_for_session(spec: SleeveSpec, session, *, us_allowed: bool):
    """현재 세션에 매수 가능한 슬리브 ETF 풀(태그 튜플 리스트). KR 세션→KR 풀, US_TRADING→US 풀
    (미국장 활성 시만), 그 외→[]."""
    if session in _KR_SESSIONS:
        return list(spec.pool_kr)
    if session == "US_TRADING":
        return list(spec.pool_us) if us_allowed else []
    return []


# ── 보유 분리 / 비중 ──────────────────────────────────────────────────────────
def split_sleeve_holdings(holdings, sleeve_codes_set):
    """보유를 (비슬리브=주식, 슬리브ETF) 로 분리. sleeve_codes_set 은 대상 슬리브(들) 코드 집합."""
    pool = {str(c).strip().upper() for c in (sleeve_codes_set or [])}
    stocks, sleeve = [], []
    for h in (holdings or []):
        (sleeve if str(h.get("code", "")).strip().upper() in pool else stocks).append(h)
    return stocks, sleeve


def format_sleeve_holdings_block(sleeve_holdings) -> str:
    """슬리브 매니저(채권/원자재) 프롬프트용 현재 보유 정형 블록.
    버그 2026-06-12: 매니저 LLM 에 현재 보유·가격을 안 줘서 보유분을 '미보유'로 단정하고
    가격을 달러로 날조하던 환각(hh0908 137610·$가격) 차단용. 가격은 KRW(달러 표기 금지)."""
    rows = sleeve_holdings or []
    if not rows:
        return "현재 이 슬리브 보유 없음."
    lines = []
    for h in rows:
        code = str(h.get("code", "")).strip()
        name = (h.get("name") or code).strip()
        qty = int(float(h.get("qty") or 0))
        cur = float(h.get("cur_price") or 0.0)
        pnl = h.get("pnl_pct")
        bit = f"- {name}({code}): 보유 {qty:,}주 · 현재가 {cur:,.0f}원"
        if pnl is not None:
            bit += f" · 평가손익 {float(pnl):+.1f}%"
        lines.append(bit)
    return "\n".join(lines)


def current_sleeve_weight(holdings, total_eval_krw: float, pool_codes,
                          usdkrw: float = 1.0) -> float:
    """보유 중 슬리브 풀에 속한 것의 평가액 합 ÷ 총평가액. US ETF는 USDKRW 환산.
    총평가액 ≤ 0 이면 0.0(평가 불가)."""
    if not total_eval_krw or float(total_eval_krw) <= 0:
        return 0.0
    pool = {str(c).strip().upper() for c in (pool_codes or [])}
    s = 0.0
    for h in (holdings or []):
        code = str(h.get("code", "")).strip().upper()
        if code not in pool:
            continue
        val = float(h.get("qty") or 0.0) * float(h.get("cur_price") or 0.0)
        if not _is_kr_code(code):
            val *= float(usdkrw or 1.0)
        s += val
    return s / float(total_eval_krw)


# ── 사이징 ─────────────────────────────────────────────────────────────────────
def size_sleeve_action(rec_pct, cur_pct: float, total_eval_krw: float,
                       max_pct: float, band: float):
    """목표비중 추종. 반환 (action, notional_krw).
    action: 'skip'(권고없음) | 'hold'(데드존) | 'buy'(부족) | 'sell'(초과).
    주의(사장 결정 2026-06-09): action 은 *매수 예산 계산*에만 쓴다 — 매도 평가는 밴드와
    무관하게 매니저 LLM 이 항상 신호로 판단한다('%맞으니 보류' 금지)."""
    if rec_pct is None:
        return ("skip", 0.0)
    target = min(float(rec_pct), float(max_pct))
    diff = target - float(cur_pct or 0.0)
    if abs(diff) <= float(band):
        return ("hold", 0.0)
    notional = abs(diff) * float(total_eval_krw or 0.0)
    return ("buy" if diff > 0 else "sell", notional)


def should_execute_sleeve_buy(action, has_buy_directive: bool, has_sell_directive: bool):
    """매수 레그를 집행할지 결정한다(사장 지시 2026-06-12, Q4 — 비중 유지 회전 허용).

    데드존은 슬리브 '순(net)비중 drift' churn 을 막는 장치이지 '구성 교체(회전)'를 막는 게
    아니다. 종전엔 순비중이 밴드 안/위면 action=hold/sell → 매수 레그가 무조건 차단돼,
    'GLD 팔고 132030 사는' 순중립 회전조차 매수가 반려됐다(원자재 08시 사례).

    규칙:
      - action=='buy'(부족분): 매수 directive 가 있으면 평소대로 부족분 매수(회전 아님).
      - action in ('hold','sell','skip') + 매수 directive + 매도 directive 동시: **비중 유지 회전**
        → 매수 허용(순비중은 매도로 상쇄). 매수 예산은 호출부가 사이클 전용 예산까지 부여(Option B).
      - 그 외(매수 의견만/매도 의견만): 매수 집행 안 함.
    반환 (execute_buy: bool, is_rotation: bool)."""
    if action == "buy":
        return (bool(has_buy_directive), False)
    if has_buy_directive and has_sell_directive:
        return (True, True)
    return (False, False)


def cap_sleeve_buy_notional(notional_krw: float, total_eval_krw: float, cash_krw: float,
                            per_cycle_ratio: float, min_cash_buffer: float) -> float:
    """슬리브 *매수* notional 을 슬리브 전용 사이클 예산·예수금으로 cap(순수 함수).
    슬리브는 주식 MAX_CYCLE_BUDGET_RATIO 게이트가 면제이므로 여기서 cap 하지 않으면 무제한 매수가
    리스크검증(예수금만 봄)을 통과해버린다. 매도는 보유주수 기반이라 호출하지 않는다."""
    caps = [float(notional_krw or 0.0), float(total_eval_krw or 0.0) * float(per_cycle_ratio or 0.0)]
    _cash = float(cash_krw or 0.0)
    _mcb = float(min_cash_buffer or 0.0)
    if _cash > 0 and _mcb > 0:
        caps.append(_cash / _mcb)
    return min(caps)


# ── 결정 파싱 / 주문 조립 ──────────────────────────────────────────────────────
def normalize_sell_directive(d: str) -> str:
    """매도 지시 정규화 (사장 지시 2026-07-20) — LLM이 '1주 매도'를 '1매'/'5주'/'3매도'로 오기하면
    조립부의 isdigit() 판정에 걸려 매도가 **조용히 누락**되던 것 방지('실주문 누락 금지' 원칙).
    'N매'/'N주'/'N매도'/'N股' → 'N', '전량매도'→'전량', '절반매도'/'반'→'절반'. 매수/보유/전량/절반/
    순수숫자/매도는 그대로 통과."""
    s = str(d or "").strip()
    if not s:
        return s
    m = re.match(r"^(\d+)\s*(?:주|매도|매|股|shares?|share)?$", s, re.IGNORECASE)
    if m:
        return m.group(1)
    if s in ("전량매도", "전량 매도", "전부", "전부매도"):
        return "전량"
    if s in ("절반매도", "반", "반매도", "1/2"):
        return "절반"
    return s


def _sleeve_decision_line(text: str, keyword: str):
    """매니저 결정 줄에서 (code, directive, weight_or_None) 튜플들을 파싱하는 공통 이터레이터.
    형식: '<keyword>: 153130=매수:50, 357870=매수:30, 114260=보유'
    directive 는 콜론 앞(매수/보유/전량/절반/주수), weight 는 매수 시 배분 비중(정수/실수 %)."""
    kw_pat = re.escape(keyword)
    if keyword.endswith("결정"):
        kw_pat = re.escape(keyword[:-2]) + r"\s*결정"
    m = re.search(kw_pat + r"\s*[:：]\s*(.+)", text or "", re.IGNORECASE)
    if not m:
        return
    for part in re.split(r"[,，;]", m.group(1).splitlines()[0]):
        # directive 는 콜론(:／：)을 포함하지 않는 토큰, 그 뒤 선택적 ':비중'.
        mm = re.match(r"\s*([0-9]{6}|[A-Za-z]{1,5})\s*=\s*([^\s,，;:：]+)\s*(?:[:：]\s*(\d+(?:\.\d+)?))?",
                      part)
        if not mm:
            continue
        code = mm.group(1).upper()
        directive = normalize_sell_directive(mm.group(2).strip())  # '1매'→'1' 등(매수/보유는 통과)
        if directive == "보류":   # '보류'(매매 안 함)를 '보유'와 동의어로 정규화
            directive = "보유"
        weight = None
        if mm.group(3) is not None:
            try:
                weight = float(mm.group(3))
            except ValueError:
                weight = None
        yield code, directive, weight


def parse_sleeve_decisions(text: str, keyword: str, pool_codes) -> Dict[str, str]:
    """매니저의 '<keyword>: 148070=매수, TLT=보유' 한 줄 → {code: directive}.
    keyword 예: '채권결정' | '원자재결정'. ETF 풀 화이트리스트 밖 코드는 드롭(티커 환각 가드).
    '코드=매수:비중'의 비중 부분은 여기서 벗겨내고 directive 만 반환한다(비중은 parse_sleeve_weights)."""
    out: Dict[str, str] = {}
    pool = {str(c).strip().upper() for c in (pool_codes or [])}
    for code, directive, _w in _sleeve_decision_line(text, keyword):
        if code in pool:
            out[code] = directive
    return out


def parse_sleeve_weights(text: str, keyword: str, pool_codes) -> Dict[str, float]:
    """매수 종목의 배분 비중 '{code: weight%}' (사장 지시 2026-07-20 — 매니저 50/30/20 실제 반영).
    '코드=매수:비중' 에서 비중이 명시된 매수 항목만 담는다. 미명시면 빈 dict → 호출부가 균등 폴백."""
    out: Dict[str, float] = {}
    pool = {str(c).strip().upper() for c in (pool_codes or [])}
    for code, directive, w in _sleeve_decision_line(text, keyword):
        if code in pool and directive == "매수" and w is not None and w > 0:
            out[code] = w
    return out


def assemble_sleeve_orders(spec: SleeveSpec, action, notional_krw, directives, holdings,
                           price_lookup, usdkrw: float = 1.0, weights=None):
    """매니저 결정(파싱 결과)을 실제 주문 리스트로 조립한다(순수 함수, broker/LLM 무관).

    - action: size_sleeve_action 결과('buy'/'sell'/'hold'/'skip'). 'buy'/'sell'만 주문 생성.
    - notional_krw: 매수 시 분배할 총 원화 예산. 매도는 보유 주수 기반이라 무관.
    - directives: {code: directive}. directive ∈ {'매수','보유','전량','절반', 정수문자열}.
    - weights: {code: 비중%} (사장 지시 2026-07-20). 매수 종목 '전부'에 양수 비중이 있으면
      그 비중대로 예산을 분배하고, 일부만/전무면 **균등 분할**(기존 동작)로 폴백한다.
    - holdings: 현재 보유 슬리브 ETF 리스트([{code, qty, ...}]).
    - price_lookup(code) -> float: KR=원화, US=USD 원가격. 0/None이면 그 코드 스킵.
    - usdkrw: US ETF qty 계산용 환산율. order price 에는 **원가격**(원화 또는 USD)을 넣는다.
    - 반환: [{"ticker","side","qty","price","reason","entry_mode"}]."""
    reason = f"{spec.manager_name} 자산배분"
    orders = []
    if action == "buy":
        picks = [c for c, d in (directives or {}).items() if str(d).strip() == "매수"]
        if not picks:
            return orders
        total = float(notional_krw or 0.0)
        w = weights or {}
        _pw = [float(w.get(c) or 0.0) for c in picks]
        # 매수 종목 '전부'에 양수 비중이 지정됐을 때만 가중 분배 — 부분/전무면 균등 폴백(예측가능).
        _use_weights = bool(picks) and all(x > 0 for x in _pw) and sum(_pw) > 0
        _wsum = sum(_pw) if _use_weights else 0.0
        for _i, code in enumerate(picks):
            per_budget = (total * _pw[_i] / _wsum) if _use_weights else (total / len(picks))
            price = float(price_lookup(code) or 0.0)
            if price <= 0:
                continue
            unit_krw = price * float(usdkrw or 1.0) if not _is_kr_code(code) else price
            if unit_krw <= 0:
                continue
            qty = int(per_budget / unit_krw)
            if qty < 1:
                continue
            orders.append({"ticker": code, "side": "buy", "qty": qty,
                           "price": price, "reason": reason, "entry_mode": "market"})
    elif action == "sell":
        held = {str(h.get("code", "")).strip().upper(): int(float(h.get("qty") or 0))
                for h in (holdings or [])}
        for code, directive in (directives or {}).items():
            hold_qty = held.get(str(code).strip().upper(), 0)
            if hold_qty < 1:
                continue
            d = str(directive).strip()
            if d in ("전량", "매도"):
                qty = hold_qty
            elif d == "절반":
                qty = hold_qty // 2
            elif d.isdigit():
                qty = min(int(d), hold_qty)
            else:
                continue  # '보유'/알 수 없는 directive → 스킵
            if qty < 1:
                continue
            price = float(price_lookup(code) or 0.0)
            if price <= 0:
                continue
            orders.append({"ticker": code, "side": "sell", "qty": qty,
                           "price": price, "reason": reason, "entry_mode": "market"})
    return orders


def build_exec_list(approved_orders, max_trades, sleeve_pool_codes):
    """실행 리스트 = 매도 전부 + 슬리브 매수 전부(빈도 cap 면제) + 주식 매수[:cap].

    슬리브 매수는 자산배분(리밸런싱)이라 매매빈도 cap 적용 대상이 아니다 — 주식 매수가 cap 을
    채우면 슬리브 리밸런싱이 조용히 누락되던 버그('주문 절대 스킵 금지' 위반) 방지.
    매도는 cap 무관 전량 실행. 슬리브 코드 판별은 sleeve_pool_codes(대문자 집합) 기준."""
    pool = {str(c).strip().upper() for c in (sleeve_pool_codes or [])}
    sells = [r for r in approved_orders if (r.get("side") or "buy") == "sell"]
    buys = [r for r in approved_orders if (r.get("side") or "buy") != "sell"]
    sleeve_buys = [r for r in buys if str(r.get("ticker", "")).strip().upper() in pool]
    stock_buys = [r for r in buys if str(r.get("ticker", "")).strip().upper() not in pool]
    try:
        _cap = int(max_trades)
    except (TypeError, ValueError):
        _cap = 2
    if _cap < 0:
        _cap = 0
    return sells + sleeve_buys + stock_buys[:_cap]

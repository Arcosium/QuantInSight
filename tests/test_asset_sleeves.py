"""자산슬리브 엔진 — 채권·원자재 공통 SleeveSpec 범용 함수 (사장 지시 2026-06-09).

채권 트랙을 슬리브#1로 일반화하며 동작 보존을 핀(채권 전용 test_bond_*.py 와 동치).
"""
from infra.asset_sleeves import (
    SleeveSpec, SLEEVES, BOND_SLEEVE, COMMODITY_SLEEVE, get_sleeve,
    all_sleeve_pool_codes, sleeve_codes, sleeve_for_code,
    parse_macro_sleeve_pct, current_sleeve_weight, size_sleeve_action,
    cap_sleeve_buy_notional, assemble_sleeve_orders, parse_sleeve_decisions,
    sleeve_pool_for_session, split_sleeve_holdings, build_exec_list,
    should_execute_sleeve_buy,
)


# ── 비중유지 회전 (사장 지시 2026-06-12, Q4) ────────────────────────────────────
def test_rotation_buy_allowed_inside_band_when_sell_present():
    # 데드존(hold/sell)이어도 매도+매수 동시 제안이면 매수 집행(순비중은 매도로 상쇄).
    assert should_execute_sleeve_buy("hold", True, True) == (True, True)
    assert should_execute_sleeve_buy("sell", True, True) == (True, True)


def test_buy_action_normal_not_rotation():
    assert should_execute_sleeve_buy("buy", True, False) == (True, False)
    # 매수 action 인데 매수 directive 가 없으면 집행 안 함.
    assert should_execute_sleeve_buy("buy", False, True) == (False, False)


def test_no_rotation_when_only_buy_or_only_sell_in_band():
    # 데드존 + 매수 의견만(매도 없음) → 보류(기존 동작).
    assert should_execute_sleeve_buy("hold", True, False) == (False, False)
    # 데드존 + 매도만 → 매수 없음.
    assert should_execute_sleeve_buy("hold", False, True) == (False, False)
    assert should_execute_sleeve_buy("skip", False, False) == (False, False)


# ── 레지스트리 / spec ──────────────────────────────────────────────────────────
def test_bond_sleeve_registered():
    s = get_sleeve("bond")
    assert s.manager_name == "채권운용실장"
    assert s.role == "bond_manager"
    assert s.macro_keyword == "채권"
    assert s.decision_keyword == "채권결정"
    assert s.enable_key == "ENABLE_BOND_ETF"
    kr_codes = {c for c, *_ in s.pool_kr}
    assert {"153130", "114260", "148070"} <= kr_codes


def test_commodity_sleeve_registered():
    s = get_sleeve("commodity")
    assert s.manager_name == "원자재운용실장"
    assert s.role == "commodity_manager"
    assert s.macro_keyword == "원자재"
    assert s.decision_keyword == "원자재결정"


def test_two_sleeves_registered():
    assert {s.key for s in SLEEVES} == {"bond", "commodity"}


def test_all_pool_codes_union_upper():
    codes = all_sleeve_pool_codes()
    assert "153130" in codes and "TLT" in codes   # 채권
    assert "132030" in codes and "GLD" in codes    # 원자재
    assert "tlt" not in codes                       # 대문자 정규화


def test_sleeve_for_code():
    assert sleeve_for_code("148070") is BOND_SLEEVE
    assert sleeve_for_code("GLD") is COMMODITY_SLEEVE
    assert sleeve_for_code("005930") is None  # 주식은 슬리브 아님


# ── 매크로% 파싱 ───────────────────────────────────────────────────────────────
def test_parse_macro_pct_four_way():
    txt = "📈 자산 배분 권고: 주식 50% / 채권 25% / 원자재 15% / 현금 10%"
    assert parse_macro_sleeve_pct(txt, "주식") == 0.50
    assert parse_macro_sleeve_pct(txt, "채권") == 0.25
    assert parse_macro_sleeve_pct(txt, "원자재") == 0.15


def test_parse_macro_pct_anchor_excludes_prev():
    txt = ("자산 배분 권고: 주식 45% / 채권 25% / 원자재 20% / 현금 10% "
           "(직전: 주식 50% / 채권 30% / 원자재 5% / 현금 15%)")
    assert parse_macro_sleeve_pct(txt, "원자재") == 0.20  # 직전(5%) 아닌 현재(20%)
    assert parse_macro_sleeve_pct(txt, "채권") == 0.25


def test_parse_macro_pct_none_when_absent():
    assert parse_macro_sleeve_pct("아무 권고 없음", "채권") is None
    assert parse_macro_sleeve_pct(None, "채권") is None


# ── 사이징 ─────────────────────────────────────────────────────────────────────
def test_size_action_skip_when_no_rec():
    assert size_sleeve_action(None, 0.1, 1_000_000, 0.40, 0.03) == ("skip", 0.0)


def test_size_action_hold_inside_band():
    action, notional = size_sleeve_action(0.25, 0.24, 1_000_000, 0.40, 0.03)
    assert action == "hold" and notional == 0.0


def test_size_action_buy_below_band():
    action, notional = size_sleeve_action(0.30, 0.10, 1_000_000, 0.40, 0.03)
    assert action == "buy" and notional > 0


def test_size_action_sell_above_band():
    action, notional = size_sleeve_action(0.10, 0.30, 1_000_000, 0.40, 0.03)
    assert action == "sell" and notional > 0


def test_size_action_clamped_to_max():
    action, notional = size_sleeve_action(0.90, 0.10, 1_000_000, 0.40, 0.03)
    assert action == "buy"
    assert notional <= 0.40 * 1_000_000 + 1


# ── 예산 cap ───────────────────────────────────────────────────────────────────
def test_cap_uses_min_of_constraints():
    # notional 500k, per_cycle 15%*1M=150k, cash/buffer 1M/1.1≈909k → min=150k
    capped = cap_sleeve_buy_notional(500_000, 1_000_000, 1_000_000, 0.15, 1.1)
    assert abs(capped - 150_000) < 1


# ── 결정 파싱 ──────────────────────────────────────────────────────────────────
def test_parse_decisions_whitelist_drop_and_normalize():
    pool = {"148070", "TLT"}
    d = parse_sleeve_decisions("채권결정: 148070=매수, 999999=매수, TLT=보류", "채권결정", pool)
    assert d == {"148070": "매수", "TLT": "보유"}  # 풀밖 드롭 + 보류→보유


def test_parse_decisions_commodity_keyword():
    pool = {"GLD", "132030"}
    d = parse_sleeve_decisions("원자재결정: GLD=매수, 132030=보유", "원자재결정", pool)
    assert d == {"GLD": "매수", "132030": "보유"}


# ── 세션 풀 ────────────────────────────────────────────────────────────────────
def test_pool_for_session_kr_vs_us():
    kr = sleeve_pool_for_session(BOND_SLEEVE, "KR_TRADING", us_allowed=True)
    assert all(str(c).isdigit() for c, *_ in kr)
    us = sleeve_pool_for_session(BOND_SLEEVE, "US_TRADING", us_allowed=True)
    assert any(c == "TLT" for c, *_ in us)
    assert sleeve_pool_for_session(BOND_SLEEVE, "US_TRADING", us_allowed=False) == []
    assert sleeve_pool_for_session(BOND_SLEEVE, "OFF_HOURS", us_allowed=True) == []


# ── 비중 / 분리 ────────────────────────────────────────────────────────────────
def test_current_weight_usd_converted():
    holds = [{"code": "TLT", "qty": 10, "cur_price": 90.0}]  # USD
    w = current_sleeve_weight(holds, 1_350_000.0, {"TLT"}, usdkrw=1500.0)
    assert abs(w - (10 * 90 * 1500) / 1_350_000.0) < 1e-6


def test_split_sleeve_holdings():
    holds = [{"code": "005930"}, {"code": "148070"}, {"code": "GLD"}]
    stocks, sleeve = split_sleeve_holdings(holds, all_sleeve_pool_codes())
    assert {h["code"] for h in stocks} == {"005930"}
    assert {h["code"] for h in sleeve} == {"148070", "GLD"}


# ── 주문 조립 ──────────────────────────────────────────────────────────────────
def test_assemble_buy_splits_budget_and_reason():
    orders = assemble_sleeve_orders(
        BOND_SLEEVE, "buy", 100_000, {"148070": "매수"},
        [], price_lookup=lambda c: 50_000.0, usdkrw=1.0)
    assert len(orders) == 1
    o = orders[0]
    assert o["ticker"] == "148070" and o["side"] == "buy" and o["qty"] == 2
    assert o["reason"] == "채권운용실장 자산배분"


def test_assemble_commodity_reason():
    orders = assemble_sleeve_orders(
        COMMODITY_SLEEVE, "sell", 0, {"GLD": "전량"},
        [{"code": "GLD", "qty": 5}], price_lookup=lambda c: 200.0)
    assert orders[0]["reason"] == "원자재운용실장 자산배분"
    assert orders[0]["qty"] == 5


# ── 실행 리스트 ────────────────────────────────────────────────────────────────
def test_build_exec_list_sleeve_buys_exempt_from_cap():
    orders = [
        {"ticker": "005930", "side": "buy"}, {"ticker": "000660", "side": "buy"},
        {"ticker": "148070", "side": "buy"}, {"ticker": "005380", "side": "sell"},
    ]
    out = build_exec_list(orders, max_trades=1, sleeve_pool_codes={"148070"})
    tickers = [o["ticker"] for o in out]
    assert "005380" in tickers   # 매도 전부
    assert "148070" in tickers   # 슬리브 매수 cap 면제
    # 주식 매수는 cap=1 → 1개만
    assert len([t for t in tickers if t in ("005930", "000660")]) == 1

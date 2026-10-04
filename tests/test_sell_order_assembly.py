"""_assemble_sell_orders — 보유종목 → 매도 주문 조립 (KR·US 모두).

버그 2026-05-22 (2차): _build_orders 매도 루프가 6자리(KR) 코드만 처리하고 US 티커는
continue 로 건너뛰며 market="KR" 을 하드코딩했다. US 보유분이 매도 평가에 들어와도
(1차 수정) 실제 매도 주문이 조립되지 않아 체결될 수 없었다("주문 절대 스킵 금지" 위배).
이 테스트는 US 매도가 market="US" 주문으로 조립되고, KR 동작은 그대로이며, 통화 기준이
다른 편중축소(TRIM, KRW)는 US엔 적용되지 않음을 고정한다.
"""
from main_swarm import _assemble_sell_orders

_PARAMS = dict(enable_rebalance=True, take_profit_pct=10.0, stop_loss_pct=5.0,
               trim_over_ratio=True, conservative_ratio=0.2, per_stock_cap=500.0, total=10000.0)


def test_us_directive_full_sell_routes_to_us_market():
    holdings = [{"code": "XOM", "name": "EXXON", "qty": 3, "pnl_pct": -2.0, "cur_price": 110.0}]
    orders, _ = _assemble_sell_orders(holdings, {"XOM": "전량"}, **_PARAMS)
    assert len(orders) == 1
    o = orders[0]
    assert o["ticker"] == "XOM" and o["market"] == "US"
    assert o["side"] == "sell" and o["qty"] == 3 and o["price_type"] == "market"


def test_kr_directive_half_unchanged():
    holdings = [{"code": "039030", "name": "이오테크닉스", "qty": 4, "pnl_pct": 1.0, "cur_price": 200000.0}]
    orders, _ = _assemble_sell_orders(holdings, {"039030": "절반"}, **_PARAMS)
    assert len(orders) == 1
    o = orders[0]
    assert o["ticker"] == "039030" and o["market"] == "KR" and o["qty"] == 2


def test_us_unmentioned_stop_loss_auto_sells():
    # 사후관리실장이 언급 안 한 US 종목도 자동 손절 안전망이 걸려야 한다 (pnl -8% ≤ -5%).
    holdings = [{"code": "OXY", "name": "OCCIDENTAL", "qty": 5, "pnl_pct": -8.0, "cur_price": 50.0}]
    orders, _ = _assemble_sell_orders(holdings, {}, **_PARAMS)
    assert len(orders) == 1 and orders[0]["market"] == "US" and orders[0]["qty"] == 5


def test_us_trim_rule_skipped_currency_safety():
    # TLT 평가액 100×10=$1000 > per_stock_cap(500). KR이면 편중축소가 걸리지만 US엔 미적용
    # (KRW 한도와 USD 평가액을 섞으면 안 됨). pnl 중립이라 익절/손절도 없음 → 주문 0건.
    holdings = [{"code": "TLT", "name": "ISHARES", "qty": 10, "pnl_pct": 0.5, "cur_price": 100.0}]
    orders, _ = _assemble_sell_orders(holdings, {}, **_PARAMS)
    assert orders == []


def test_hold_directive_produces_no_order():
    holdings = [{"code": "XOM", "qty": 3, "pnl_pct": -2.0, "cur_price": 110.0}]
    orders, _ = _assemble_sell_orders(holdings, {"XOM": "보유"}, **_PARAMS)
    assert orders == []


def test_hard_take_profit_overrides_llm_hold_and_sells_all():
    """+20%인데 LLM이 보유/절반을 반복해 잔여주가 고착되는 일을 막는다."""
    holdings = [{"code": "214450", "name": "파마리서치", "qty": 2,
                 "pnl_pct": 20.4, "cur_price": 408500.0}]
    orders, _ = _assemble_sell_orders(holdings, {"214450": "보유"}, **_PARAMS)
    assert len(orders) == 1
    assert orders[0]["qty"] == 2
    assert "자동 익절" in orders[0]["reason"]


def test_hard_take_profit_ignores_llm_limit_price_and_uses_market():
    """실제 손익 안전망은 낡거나 환각된 LLM 지정가 때문에 미체결돼선 안 된다."""
    holdings = [{"code": "214450", "name": "파마리서치", "qty": 2,
                 "pnl_pct": 20.4, "cur_price": 408500.0}]
    orders, _ = _assemble_sell_orders(
        holdings, {"214450": "보유"},
        sell_prices={"214450": {"mode": "limit", "limit_price": 500000}},
        **_PARAMS)
    assert len(orders) == 1
    assert orders[0]["qty"] == 2 and orders[0]["price_type"] == "market"
    assert "entry_limit" not in orders[0]


def test_hard_stop_loss_overrides_llm_hold():
    holdings = [{"code": "005930", "qty": 3, "pnl_pct": -8.0, "cur_price": 70_000.0}]
    orders, _ = _assemble_sell_orders(holdings, {"005930": "보유"}, **_PARAMS)
    assert orders[0]["qty"] == 3 and "자동 손절" in orders[0]["reason"]


def test_disabling_rebalance_preserves_explicit_hold_escape_hatch():
    params = {**_PARAMS, "enable_rebalance": False}
    holdings = [{"code": "214450", "qty": 2, "pnl_pct": 20.4, "cur_price": 408500.0}]
    orders, _ = _assemble_sell_orders(holdings, {"214450": "보유"}, **params)
    assert orders == []

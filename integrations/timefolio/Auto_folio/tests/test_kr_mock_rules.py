from Auto_folio.autofolio.engine import AutoFolioEngine
from Auto_folio.autofolio.risk import validate_order_draft
from Auto_folio.autofolio.sizing import build_exec_list


def test_mock_account_trades_only_kr_regular_session():
    e = AutoFolioEngine()
    out = e.plan_cycle(
        target_codes=["005930"],
        holdings=[],
        buying_power={"cash": 10_000_000, "total_eval": 10_000_000, "pnl_ratio": 0, "ok": True},
        price_map={"005930": 70_000},
        session="KR_PRE_MARKET",
    )
    assert out["orders_planned"] == []
    assert "KRX 정규장만" in out["sizing_notes"][0]


def test_buy_order_uses_quantinsight_budget_caps():
    e = AutoFolioEngine()
    out = e.plan_cycle(
        target_codes=["005930", "000660"],
        holdings=[],
        buying_power={"cash": 10_000_000, "total_eval": 10_000_000, "pnl_ratio": 0, "ok": True},
        price_map={"005930": 70_000, "000660": 250_000},
        session="KR_TRADING",
    )
    assert out["approved_orders"]
    assert len(out["approved_orders"]) <= 2
    assert all(o["market"] == "KR" and o["exchange"] == "KRX" for o in out["approved_orders"])


def test_mdd_blocks_new_buys():
    risk = validate_order_draft(
        {"orders": [{"ticker": "005930", "side": "buy", "qty": 1, "reason": "테스트 매수"}]},
        buying_power={"cash": 10_000_000, "total_eval": 10_000_000, "pnl_ratio": -0.06, "ok": True},
        price_map={"005930": 70_000},
    )
    assert not risk["approved"]
    assert "MDD" in risk["report"]


def test_sells_are_prioritized_under_cycle_cap():
    orders = [
        {"ticker": "005930", "side": "buy", "qty": 1},
        {"ticker": "000660", "side": "sell", "qty": 1},
        {"ticker": "035420", "side": "buy", "qty": 1},
    ]
    out = build_exec_list(orders, 2)
    assert out[0]["side"] == "sell"
    assert len(out) == 2


def test_auto_take_profit_sell_before_buy():
    e = AutoFolioEngine()
    out = e.plan_cycle(
        target_codes=["000660"],
        holdings=[{"code": "005930", "qty": 10, "avg_price": 50_000, "cur_price": 60_000, "sellable_qty": 10}],
        buying_power={"cash": 10_000_000, "total_eval": 10_600_000, "pnl_ratio": 0, "ok": True},
        price_map={"005930": 60_000, "000660": 250_000},
        session="KR_TRADING",
    )
    assert out["orders_planned"][0]["side"] == "sell"
    assert out["orders_planned"][0]["ticker"] == "005930"

from types import SimpleNamespace
import asyncio
from tools.portfolio_risk import constrain_entries


def buy(code="005930", qty=1000):
    return {"ticker":code,"qty":qty,"side":"buy","reason":"위원회 심의 통과"}


def run(orders, **kw):
    args=dict(prices={"005930":1000,"000660":1000},
              buying_power={"ok":True,"cash":1000000,"total_eval":1000000},
              holdings=[],approved_stock_codes=["005930","000660"],usdkrw=1500)
    args.update(kw)
    return constrain_entries(orders,**args)


def test_existing_holding_and_duplicate_orders_share_one_risk_budget():
    orders,_=run([buy(),buy()], holdings=[{"code":"005930","qty":80,"cur_price":1000}])
    assert sum(o["qty"] for o in orders) == 13
    assert (80+sum(o["qty"] for o in orders))*1000*.0535 <= 5000


def test_higher_volatility_reduces_size_instead_of_manufacturing_expected_profit():
    low,_=run([buy()], sigmas={"005930":10})
    high,_=run([buy()], sigmas={"005930":150})
    assert 0 < high[0]["qty"] < low[0]["qty"] <= 100


def test_no_unreviewed_substitution_or_one_share_exception():
    assert run([buy()],approved_stock_codes=[])[0] == []
    assert run([buy()],prices={"005930":200000})[0] == []


def test_exits_do_not_release_unfilled_cash_and_reserve_covers_sleeves():
    sell=dict(buy(qty=500),side="sell")
    orders,_=run([sell,buy()],buying_power={"ok":True,"cash":105000,"total_eval":1000000},
                 is_sleeve=lambda _:True)
    assert orders[0] == sell
    assert orders[1]["qty"] == 4


def test_portfolio_risk_consumed_by_existing_positions():
    holdings=[{"code":str(100000+i),"qty":100,"cur_price":1000} for i in range(6)]
    assert run([buy()],holdings=holdings)[0] == []


def test_missing_valuation_blocks_entries_but_never_exits():
    sell=dict(buy(qty=2),side="sell")
    orders,_=run([sell,buy()],holdings=[{"code":"000660","qty":10,"cur_price":None}])
    assert orders == [sell]


def test_us_budget_uses_currency_conversion():
    orders,_=run([buy("AAPL")],prices={"AAPL":100},approved_stock_codes=["AAPL"])
    assert orders == []  # one $100 share exceeds the KRW risk allocation


def test_watch_and_invalid_limit_never_become_market_orders():
    assert run([dict(buy(),entry_mode="watch")])[0] == []
    assert run([dict(buy(),entry_mode="limit",entry_limit=None)])[0] == []
    limit,_=run([dict(buy(),entry_mode="limit",entry_limit=2000)])
    assert limit[0]["qty"]*2000*.0535 <= 5000


def test_us_explicit_limit_is_preserved_when_market_moves_away():
    from infra.kis_broker import KISBroker
    b=KISBroker({"kis_app_key":"test","kis_app_secret":"test",
                 "kis_account_no":"00000000-01",
                 "kis_base_url":"https://openapivts.koreainvestment.com:29443"})
    async def quote(_):return 200.0
    b.us_last_price=quote
    result=asyncio.run(b._overseas_order_body("AAPL",1,100.25,side="buy",excd="NASD"))
    assert result["OVRS_ORD_UNPR"] == "100.25"


def test_mock_nxt_order_does_not_request_real_quotes():
    from main_swarm import ArquantOrchestrator
    from infra.kis_broker import OrderDraft
    o=ArquantOrchestrator.__new__(ArquantOrchestrator)
    o.broker=SimpleNamespace(is_mock=True)
    draft=OrderDraft(ticker="005930",side="buy",qty=1,market="KR",reason="test order")
    _,reason=asyncio.run(o._finalize_kr_order_for_session(draft,"KR_PRE_MARKET"))
    assert "모의투자" in reason


def test_cycle_only_approves_and_reports_constrained_quantity(monkeypatch):
    import main_swarm
    from main_swarm import ArquantOrchestrator
    o=ArquantOrchestrator.__new__(ArquantOrchestrator)
    o.uid=12345
    o.cycle_log=SimpleNamespace(log=lambda *a:None)
    async def emit(_):pass
    async def review(*a):return set(),"공시 확인"
    o._emit=emit;o._dart_risk_review=review
    monkeypatch.setattr(main_swarm,"get_usdkrw",lambda *a:1500)
    c=SimpleNamespace(per_dart={},buying_power={"ok":True,"cash":1000000,"total_eval":1000000},
        price_map={"005930":1000},order_draft={"orders":[buy()]},target_codes=["005930"],
        holdings=[{"code":"005930","qty":80,"cur_price":1000}],_quant_sigmas={})
    asyncio.run(o._cyc_stage_risk(c))
    assert c.risk_approved
    assert c.approved_orders[0]["qty"] == 13
    assert c.order_obj["orders"][0]["qty"] == 13
    assert c.portfolio_risk_notes

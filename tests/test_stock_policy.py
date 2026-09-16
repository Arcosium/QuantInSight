import pandas as pd
import pytest
from tools import stock_policy as p


def policy():
    return {"status":"ready", "decision_id":"fixed", "entry_weights":{"005930":.04},
            "rows":[{"code":"005930","close":100,"score":.8,"weight":.04}]}


def test_existing_position_and_repeated_drafts_consume_the_target():
    orders=[{"ticker":"005930","qty":10,"side":"buy"}]*2
    kept,_=p.cap_entries(orders,policy(),holdings=[],prices={"005930":100},total_eval=10000)
    assert sum(x["qty"] for x in kept)==3
    kept,_=p.cap_entries(orders,policy(),holdings=[{"code":"005930","qty":4,"cur_price":100}],
                        prices={"005930":100},total_eval=10000)
    assert kept==[]


def test_foreign_and_unapproved_entries_blocked_exits_preserved():
    orders=[{"ticker":"005930","qty":1,"side":"sell"},{"ticker":"OTHER","qty":1,"side":"buy"}]
    kept,_=p.cap_entries(orders,policy(),holdings=[],prices={"OTHER":10},total_eval=10000,usdkrw=1400)
    assert kept==orders[:1]


def test_price_chase_is_not_filled_and_candidates_skip_funded_names():
    kept,_=p.cap_entries([{"ticker":"005930","qty":1}],policy(),holdings=[],prices={"005930":104},total_eval=10000)
    assert kept==[]
    assert p.candidates(policy(),[],10000)==["005930"]
    assert p.candidates(policy(),[{"code":"005930","qty":4,"cur_price":100}],10000)==[]


def test_nonfinite_quote_blocks_entry_without_crashing():
    kept,_=p.cap_entries([{"ticker":"005930","qty":1}],policy(),holdings=[],prices={"005930":float("inf")},total_eval=10000)
    assert kept==[]


def test_policy_freezes_five_sessions_and_never_reads_todays_bar(tmp_path,monkeypatch):
    dates=pd.bdate_range("2025-01-01", periods=100)
    histories={str(100000+c):pd.DataFrame({"open":[100+i*(1+c*.02) for i in range(100)],
        "high":[101+i*(1+c*.02) for i in range(100)],"low":[99+i*(1+c*.02) for i in range(100)],
        "close":[100+i*(1+c*.02) for i in range(100)],"volume":1e8},index=dates) for c in range(30)}
    monkeypatch.setattr(p,"DATA",tmp_path)
    monkeypatch.setattr(p,"histories",lambda market,today:{c:d[d.index<today] for c,d in histories.items()})
    a=p.plan(1,now=dates[90]); b=p.plan(1,now=dates[94]); c=p.plan(1,now=dates[95])
    assert a["as_of"]==str(dates[89].date())
    assert a["decision_id"]==b["decision_id"]
    assert c["decision_id"]!=a["decision_id"]
    assert sum(a["weights"].values())<=.40000001


def test_stale_policy_does_not_return_old_entry_weights(tmp_path,monkeypatch):
    monkeypatch.setattr(p,"DATA",tmp_path)
    monkeypatch.setattr(p,"histories",lambda *args:{})
    assert p.plan(1,now="2026-09-16")["weights"]=={}


def test_exit_proposals_respect_market_and_sleeves():
    pl=dict(policy(),market="KRX",weights={"005930":.04})
    held=[{"code":"005930","qty":10,"cur_price":100}, {"code":"000660","qty":2,"cur_price":100},
          {"code":"AAPL","qty":3,"cur_price":100}, {"code":"114260","qty":4,"cur_price":100}]
    assert p.exit_proposals(pl,held,total_eval=10000,usdkrw=1400,is_sleeve=lambda c:c=="114260")=={"005930":"6","000660":"전량"}


@pytest.mark.parametrize("mock",[True,False])
def test_actual_selection_stage_uses_same_policy_for_real_and_paper_without_news(monkeypatch,mock):
    import asyncio
    import main_swarm as m
    import config
    from types import SimpleNamespace as N
    from unittest.mock import AsyncMock,Mock
    monkeypatch.setattr(config,"SYSTEMATIC_POLICY_ENABLED",True)
    monkeypatch.setattr(m.runtime,"get",lambda *a,**k:0)
    monkeypatch.setattr(m,"get_usdkrw",lambda *a:1400)
    monkeypatch.setattr(m,"_resolve_candidate_codes",lambda text,**kw:["005930"] if "005930" in text else [])
    resolve=Mock(return_value=dict(policy(),exposure=.04,as_of="2026-09-15"))
    monkeypatch.setattr(p,"plan",resolve)
    o=m.ArquantOrchestrator.__new__(m.ArquantOrchestrator)
    o.uid=12345;o._emit=AsyncMock();o.cycle_log=N(log=Mock())
    o.orchestrator=N(think=AsyncMock(side_effect=AssertionError("LLM must not invent pilot candidates")))
    o.broker=N(is_mock=mock,kr_account_snapshot=AsyncMock(return_value={"buying_power":{"cash":10000,"total_eval":10000}}),
               kr_last_price=AsyncMock(return_value=100))
    c=N(session="KR_TRADING",market_open=False,news_articles=[],user_directive="",standing_directive_block="",
        _sell_only=True,_macro_buy_blocked=True,macro_report="",news_report="",index_facts="",holdings=[],holdings_str="",
        _budget_hint="",total_eval=10000)
    asyncio.run(o._cyc_stage_select(c))
    assert c.candidate_codes == ["005930"]
    assert resolve.call_count == 1
    assert not o.orchestrator.think.called

from datetime import datetime
from zoneinfo import ZoneInfo
from autofolio import timefolio_live as live


def test_weekend_never_submits():
    assert not live.market_open(datetime(2026,10,10,10,tzinfo=ZoneInfo('Asia/Seoul')))


def test_product_verdict_requires_explicit_clean_server_response():
    assert live.product_valid({'data':'','warnings':[]})
    for value in ({},{'data':''},{'warnings':[]},{'data':'진입불가','warnings':[]},{'data':'','warnings':['거래정지']},None):
        assert not live.product_valid(value)


def test_order_budget_obeys_cash_adv_and_position_limit():
    summary={'total_eval':1e9,'positions':[]}
    limits={'default':.15,'005930':.4}
    small=live.order_payload('000001','buy',.9,summary,10000,limits,1e8)
    assert small['qty']==100 and small['weight_pct']==.1
    limited=live.order_payload('000001','buy',.9,summary,10000,limits,1e12)
    assert limited['weight_pct']<15
    assert limited['qty']*10000/(1e9-limited['qty']*10000*.0015)<=.15
    summary['positions']=[{'ticker':'000002','qty':1,'value_krw':999000000,'weight_pct':99.9}]
    cash=live.order_payload('000001','buy',.9,summary,10000,limits,1e12)
    assert cash is None
    assert live.order_payload('000002','buy',.1,summary,10000,limits,1e12) is None


def test_sell_uses_actual_held_weight():
    summary={'total_eval':1e9,'positions':[{'ticker':'005930','qty':100,'value_krw':1e7,'weight_pct':1}]}
    out=live.order_payload('005930','sell',0,summary,100000,{'default':.15})
    assert out['qty']==100 and out['weight_pct']==1


def test_rebalance_does_not_sell_newly_bought_positions_again():
    plan={'sells':['A'],'buys':[{'symbol':'A'},{'symbol':'B'}],'rebalance':True}
    plan=live.remaining_plan(plan,{'A':10})
    assert plan['sells']==['A'] and len(plan['buys'])==2
    plan=live.remaining_plan(plan,{})
    assert plan['sells']==[] and len(plan['buys'])==2
    plan=live.remaining_plan(plan,{'A':20})
    assert plan['sells']==[] and plan['buys']==[{'symbol':'B'}]
    plan=live.remaining_plan(plan,{'A':20,'B':10})
    assert plan['sells']==[] and plan['buys']==[]


def test_uncertain_order_cannot_be_retried_by_another_model_same_day(monkeypatch,tmp_path):
    import sqlite3
    from contextlib import contextmanager
    @contextmanager
    def connection():
        with sqlite3.connect(tmp_path/'test.db') as db:yield db
    monkeypatch.setattr(live,'connect',connection)
    live.initialize()
    assert live.claim(1,'model-a','20261012','005930','buy')
    assert live.claim(1,'model-b','20261012','005930','buy') is None
    assert live.claim(2,'model-a','20261012','005930','buy')


def test_unknown_caps_limit_total_exposure_to_thirty_percent_after_fees():
    summary={'total_eval':1e9,'positions':[{'ticker':'000002','qty':1,'value_krw':295000000,'weight_pct':29.5}]}
    order=live.order_payload('005930','buy',.4,summary,10000,{'default':.15,'005930':.4},1e12)
    value=order['qty']*10000
    assert (295000000+value)/(1e9-value*.0015)<=.30


def test_signal_exit_preserves_kept_positions():
    from autofolio import execution
    g={'holding_sessions':1,'selection_count':2,'sell_rule':'signal_exit','max_weight':.1}
    records=[{'symbol':'A','score':2,'adv20':1e10},{'symbol':'B','score':1,'adv20':1e10}]
    plan=execution.build_plan(records,g,{'A':10,'C':10},{'A':100,'C':100},{'A':100,'C':100},True)
    pending=live.remaining_plan(plan,{'A':10,'C':10})
    assert pending['sells']==['C']
    assert [p['symbol'] for p in pending['buys']]==['B']


def test_uncertain_order_blocks_whole_account_across_days(monkeypatch,tmp_path):
    import sqlite3
    from contextlib import contextmanager
    @contextmanager
    def connection():
        with sqlite3.connect(tmp_path/'test.db') as db:yield db
    monkeypatch.setattr(live,'connect',connection)
    live.initialize()
    assert not live.unresolved(1)
    live.claim(1,'model-a','20261012','005930','buy')
    assert live.unresolved(1)
    assert not live.unresolved(2)


def test_replaced_deployment_cannot_submit_old_signal(monkeypatch,tmp_path):
    import sqlite3
    from contextlib import contextmanager
    @contextmanager
    def connection():
        with sqlite3.connect(tmp_path/'test.db') as db:yield db
    monkeypatch.setattr(live,'connect',connection)
    with connection() as db:
        db.executescript('''CREATE TABLE model_deployments(id,user_id,target,strategy_id,status);
          CREATE TABLE strategy_assignments(user_id,target,strategy_id,status);
          INSERT INTO model_deployments VALUES('d',1,'timefolio','s','ready');
          INSERT INTO strategy_assignments VALUES(1,'timefolio','s','timefolio_ready');''')
    assert live.current_deployment({'id':'d','user_id':1})
    assert not live.current_deployment({'id':'d','user_id':2})
    with connection() as db:db.execute("UPDATE strategy_assignments SET strategy_id='replacement'")
    assert not live.current_deployment({'id':'d','user_id':1})

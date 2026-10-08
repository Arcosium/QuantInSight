import pandas as pd
import pytest
from autofolio import execution,period


def genome(**values):
    return dict(holding_sessions=5,selection_count=2,**values)


def frame():
    return pd.DataFrame([dict(date=pd.Timestamp(day),symbol=symbol,open=10.,close=10.,
        high=100.,low=1.,tradable_buy=True,tradable_sell=True) for day in
        ['20260101','20260102','20260103','20260104'] for symbol in ['A','B']])


def signals():
    return pd.DataFrame([dict(date=pd.Timestamp(day),symbol=symbol,score=score,
        adv20=1e9,volatility=vol) for day in ['20260101','20260102','20260103','20260104']
        for symbol,score,vol in [('A',2.,.01),('B',1.,.03)]])


@pytest.fixture(autouse=True)
def calendar(monkeypatch):
    monkeypatch.setattr(period,'expected_dates',lambda *args:tuple(pd.date_range('2026-01-01',periods=4).strftime('%Y%m%d')))


def test_stops_use_close_and_only_fill_next_open():
    prices=frame();prices.loc[(prices.date==pd.Timestamp('20260102'))&(prices.symbol=='A'),'close']=8.
    prices.loc[(prices.date==pd.Timestamp('20260103'))&(prices.symbol=='A'),'open']=7.
    g=genome(sell_rule='stop_take',stop_loss=.1,take_profit=.2)
    book,_=execution.account(prices,signals(),g,'crypto','20260101','20260104')
    sells=[t for t in book['trades'] if t['side']=='sell']
    assert [(t['code'],t['date'],t['price']) for t in sells]==[('A','20260103',7.)]
    # Even wild intraday extremes never trigger same-bar fills.
    other=prices.copy();other['high']=1e6;other['low']=.0001
    assert execution.account(other,signals(),g,'crypto','20260101','20260104')[0]==book
    assert book['daily'][0]['nav']==book['initial_cash']
    assert all(d['cash']>=0 for d in book['daily'])


def test_sizing_and_selection():
    records=signals().iloc[:2].to_dict('records')
    inverse=execution.choose_picks(records,genome(position_sizing='inverse_volatility',max_weight=.8,gross_exposure=.8))
    assert inverse[0]['weight']==pytest.approx(.6) and inverse[1]['weight']==pytest.approx(.2)
    ranked=execution.choose_picks(records,genome(position_sizing='score',max_weight=1.))
    assert ranked[0]['weight']/ranked[1]['weight']==pytest.approx(2.)
    assert execution.choose_picks([dict(r,score=-1.) for r in records],genome(buy_rule='positive_score'))==[]
    quantiles=execution.choose_picks([dict(records[0],symbol=str(i),score=float(i)) for i in range(10)],genome(buy_rule='top_quantile'))
    assert [r['symbol'] for r in quantiles]==['9','8']


def test_signal_exit_retains_selected_and_default_rebalances():
    hold={'A':1.,'B':1.};marks={'A':10.,'B':10.};entry=marks.copy()
    records=[dict(symbol='A',score=1.,volatility=.01,adv20=1e9)]
    plan=execution.build_plan(records,genome(sell_rule='signal_exit'),hold,marks,entry,True)
    assert plan['sells']==['B']
    assert execution.build_plan(records,genome(),hold,marks,entry,True)['sells']==['A','B']
    assert execution.build_plan(records,genome(),hold,marks,entry,False)['sells']==[]


def test_open_fill_respects_exposure_fees_and_legacy_pending():
    quotes=frame().iloc[:2].set_index('symbol')
    hold={};marks={};entries={};g=genome(max_weight=.8,gross_exposure=.5)
    pending=[dict(symbol='A',adv20=1e12),dict(symbol='B',adv20=1e12)]
    cash,trades,unfilled=execution.fill_orders(1000.,hold,marks,entries,quotes,pending,g,'crypto','20260102')
    invested=sum(q*10 for q in hold.values());nav=cash+invested
    assert invested<=nav*.5+1e-8 and cash>=0 and not unfilled
    assert sum(t['fee'] for t in trades)==pytest.approx(1000-nav)
    # Changing the yet-unknown closing value cannot change an opening order.
    quotes['close']=999999.
    cash2,trades2,_=execution.fill_orders(1000.,{},{},{},quotes,pending,g,'crypto','20260102')
    assert cash==cash2 and trades==trades2


def test_unsellable_stop_is_retried():
    quotes=frame().iloc[:2].set_index('symbol');quotes.loc['A','tradable_sell']=False
    hold={'A':2.};marks={'A':8.};entries={'A':10.}
    cash,trades,retry=execution.fill_orders(100.,hold,marks,entries,quotes,
        {'sells':['A'],'buys':[]},genome(),'crypto','20260102')
    assert retry==['A'] and not trades and hold=={'A':2.} and cash==100.


def test_rebalance_interval_independent_of_prediction_target():
    g=genome(rebalance_sessions=1)
    book,_=execution.account(frame(),signals(),g,'crypto','20260101','20260104')
    assert {t['date'] for t in book['trades'] if t['side']=='sell'}=={'20260103','20260104'}


@pytest.mark.parametrize('rule',['rebalance','stop_take','signal_exit'])
def test_paper_and_backtest_share_identical_orders(rule):
    import numpy as np
    from autofolio import paper
    prices=frame();prices['ready']=True;prices['adv20']=1e9;prices['volatility']=.01
    prices.loc[(prices.date==pd.Timestamp('20260102'))&(prices.symbol=='A'),'close']=8.
    g=genome(sell_rule=rule,rebalance_sessions=1,position_sizing='score',max_weight=.2,gross_exposure=.5)
    backtest,_=execution.account(prices,signals(),g,'crypto','20260101','20260104')
    book=dict(not_before='20251231',initial=1e5,cash=1e5,nav=1e5,holdings={},trades=[],equity=[],last_day='',pending=[],rebalance=0)
    class Model:
        def predict(self,x):return np.array([2.,1.])
    for stamp,quotes in prices.groupby('date'):
        paper.step_book(book,quotes.set_index('symbol'),g,Model(),['open'],'crypto',stamp.strftime('%Y%m%d'),True,execution)
    assert book['trades']==backtest['trades']
    assert book['nav']==pytest.approx(backtest['daily'][-1]['nav'])

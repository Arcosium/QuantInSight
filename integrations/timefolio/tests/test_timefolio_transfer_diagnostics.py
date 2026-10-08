import numpy as np
from quant.timefolio_heatmap_transfer_diagnostics import cross_section, forward_outcomes, fixed_fill_costs


def test_ranking_is_separate_from_absolute_profit():
    scores=np.arange(20,dtype=float)
    out=cross_section(scores,np.linspace(-.20,-.01,20),np.ones(20,bool))
    assert out['rank_ic'] > .999
    assert out['top_return'] < 0 < out['top_excess']


def test_labels_start_at_next_execution_and_exclude_actions():
    p={'exec_price':np.array([[10.,20.,30.,40.],[10.,20.,30.,40.]]),
       'exec_count':np.full((2,4),26), 'split':np.ones((2,4))}
    p['split'][1,2]=2
    r=forward_outcomes(p,0,2)
    assert r[0] == 1. and np.isnan(r[1])
    assert forward_outcomes(p,2,2) is None


def test_fee_addback_keeps_fills_fixed():
    ix={'codes':['A'],'dates':['20260101','20260102']}
    p={'exec_price':np.array([[100.,100.]])}
    r={'daily':[{'date':d,'nav':1e9-2*(i+1)} for i,d in enumerate(ix['dates'])],
       'trades':[{'code':'A','date':d,'qty':1,'price':101 if i==0 else 99,
                  'fee':1,'side':'buy' if i==0 else 'sell'} for i,d in enumerate(ix['dates'])],
       'metrics':{'fees_krw':2,'slippage_krw':2}}
    got=fixed_fill_costs(r,p,ix)
    assert got['fixed_fill_cost_added_back']['total_return'] == 0
    assert got['cost_by_date'] == {'20260101':2,'20260102':2}

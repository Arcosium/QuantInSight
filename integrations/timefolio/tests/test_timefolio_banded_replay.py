import numpy as np
import pytest

from test_timefolio_heatmap import synthetic_panel
from quant.timefolio_heatmap_planned_replay import replay as original
from quant.timefolio_heatmap_banded_replay import replay


def setup():
    p,ix=synthetic_panel();scores=np.ones_like(p['close'])
    args=(p,ix,scores,ix['dates'][1],ix['dates'][2])
    kwargs=dict(rebalance=1,top_n=4,weight=.05,max_orders=10,return_trades=True,return_plans=True)
    return p,ix,scores,args,kwargs


def test_zero_band_reproduces_original_ledger_and_requests():
    _,_,_,args,kwargs=setup()
    assert replay(*args,**kwargs)==original(*args,**kwargs)


def test_band_skips_small_adjustments_but_keeps_initial_entries():
    _,ix,_,args,kwargs=setup();base=replay(*args,**kwargs);band=replay(*args,**kwargs,rebalance_band=.0005)
    assert any(t['date']==ix['dates'][2] for t in base['trades'])
    assert len(band['trades'])==4
    assert all(t['date']==ix['dates'][1] for t in band['trades'])


def test_full_exit_is_never_suppressed():
    _,ix,scores,args,kwargs=setup();scores[0,1]=-1
    result=replay(*args,**kwargs,rebalance_band=.02)
    assert any(t['date']==ix['dates'][2] and t['code']==ix['codes'][0] and t['side']=='sell' for t in result['trades'])


def test_small_rule_repair_bypasses_wide_band():
    p,ix,_,args,kwargs=setup();p['sector_cap'][:,1]=.049
    result=replay(*args,**kwargs,rebalance_band=.02)
    assert any(t['date']==ix['dates'][2] and t['side']=='sell' for t in result['trades'])
    assert result['daily'][-1]['gross']<.196


def test_opening_gap_breach_also_bypasses_band():
    p,ix,_,args,kwargs=setup();p['sector_cap'][:]=.051
    for key in ['o','exec_price','close']:p[key][0,2]=11000
    p['exec_high'][0,2]=11050;p['exec_low'][0,2]=10950
    result=replay(*args,**kwargs,rebalance_band=.02)
    assert any(t['date']==ix['dates'][2] and t['code']==ix['codes'][0] and t['side']=='sell' for t in result['trades'])


@pytest.mark.parametrize('band',[-.001,np.nan])
def test_invalid_band_is_rejected(band):
    *_,args,kwargs=setup()
    with pytest.raises(ValueError,match='rebalance_band'):replay(*args,**kwargs,rebalance_band=band)

from copy import deepcopy

import numpy as np
import pytest

from quant.timefolio_cnn_execution import freeze_rankings
from quant.timefolio_heatmap_locked_weighted_replay import replay
from test_timefolio_heatmap import synthetic_panel


def test_snapshot_uses_previous_signal_and_does_not_read_intermediate_or_future_scores():
    _,ix=synthetic_panel(days=8,names=3)
    scores=np.arange(24,dtype=float).reshape(3,8)
    original=scores.copy()
    frozen,anchors=freeze_rankings(scores,ix['dates'],ix['dates'][1],ix['dates'][-1],rebalance=3)
    np.testing.assert_array_equal(anchors,[0,0,0,3,3,3,6,-1])
    changed=scores.copy();changed[:,1:3]=-100;changed[:,4:6]=100;changed[:,6:]*=-1
    revised,_=freeze_rankings(changed,ix['dates'],ix['dates'][1],ix['dates'][-1],rebalance=3)
    np.testing.assert_equal(frozen[:,:6],revised[:,:6])
    np.testing.assert_equal(scores,original)
    daily,_=freeze_rankings(scores,ix['dates'],ix['dates'][1],ix['dates'][-1],rebalance=1)
    np.testing.assert_equal(daily,scores)


def test_missing_snapshot_does_not_borrow_a_later_prediction():
    _,ix=synthetic_panel(days=7,names=2)
    scores=np.ones((2,7));scores[:,0]=np.nan;scores[1,3]=np.nan
    frozen,anchors=freeze_rankings(scores,ix['dates'],ix['dates'][1],ix['dates'][-1],rebalance=3)
    assert np.isnan(frozen[:,:3]).all()
    assert np.isnan(frozen[1,3:6]).all()
    assert np.all(anchors[:6]<=np.arange(6))


def test_deferred_order_keeps_the_rebalance_ranking_and_avoids_an_unplanned_exit():
    p,ix=synthetic_panel(days=6,names=3)
    scores=np.tile([3.,2.,1.],(6,1)).T;scores[:,1:]=np.array([1.,2.,3.])[:,None]
    frozen,_=freeze_rankings(scores,ix['dates'],ix['dates'][1],ix['dates'][2],rebalance=5)
    kw=dict(rebalance=5,top_n=2,weight=.05,max_orders=1,rebalance_band=.005,return_trades=True)
    original=replay(p,ix,scores,ix['dates'][1],ix['dates'][2],**kw)
    held=replay(p,ix,frozen,ix['dates'][1],ix['dates'][2],**kw)
    assert original['metrics']['order_budget_deferred_requests']>0
    assert any(t['side']=='sell' and t['code']==ix['codes'][0] for t in original['trades'])
    assert all(t['side']=='buy' for t in held['trades'])
    assert {t['code'] for t in held['trades']}==set(ix['codes'][:2])
    assert all(sum(t['date']==date for t in held['trades'])<=1 for date in ix['dates'])


def test_frozen_scores_do_not_override_current_sector_repair_or_buy_exclusion():
    p,ix=synthetic_panel(days=6,names=3)
    scores=np.tile([3.,2.,1.],(6,1)).T
    frozen,_=freeze_rankings(scores,ix['dates'],ix['dates'][1],ix['dates'][2],rebalance=5)
    altered=deepcopy(p);altered['sector_cap'][0,1:]=.02
    result=replay(altered,ix,frozen,ix['dates'][1],ix['dates'][2],rebalance=5,
        top_n=1,weight=.05,max_orders=3,rebalance_band=.02,return_trades=True)
    assert any(t['date']==ix['dates'][2] and t['side']=='sell' for t in result['trades'])
    assert result['daily'][-1]['gross']<.021
    excluded=deepcopy(p);excluded['trade_allowed']=np.ones_like(p['eligible'])
    excluded['trade_allowed'][1,2]=False
    result=replay(excluded,ix,frozen,ix['dates'][1],ix['dates'][2],rebalance=5,
        top_n=2,weight=.05,max_orders=1,rebalance_band=.005,return_trades=True)
    assert not any(t['date']==ix['dates'][2] and t['side']=='buy' and t['code']==ix['codes'][1]
                   for t in result['trades'])


def test_invalid_date_axis_and_cadence_are_rejected():
    scores=np.ones((2,3))
    with pytest.raises(ValueError):
        freeze_rankings(scores,['20240103','20240102','20240104'],'20240102','20240104',rebalance=2)
    with pytest.raises(ValueError):
        freeze_rankings(scores,['20240102','20240103','20240104'],'20240102','20240104',rebalance=True)

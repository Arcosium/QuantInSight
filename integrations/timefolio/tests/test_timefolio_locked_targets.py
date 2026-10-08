from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
import pytest
from test_timefolio_heatmap import synthetic_panel
from quant.timefolio_heatmap_locked_targets import reserve_locked
from quant.timefolio_heatmap_locked_replay import replay,PARENT_SHA256
from quant.timefolio_heatmap_retention_replay import replay as previous


def inputs():
    return dict(desired=np.array([0,50,45]),qty=np.array([10,50,50]),locked=np.array([10,0,0]),
        prices=np.ones(3),planning_good=np.ones(3,bool),nav=1000.,sectors=np.zeros(3,int),
        sector_caps=np.full(3,.1),small=np.zeros(3,bool),stock_caps=np.full(3,.15),
        scores=np.array([0.,3.,2.]),codes=np.array(['a','b','c']),gross=.8)


def test_locked_floor_counts_against_sector_budget():
    args=inputs();before=deepcopy(args);out=reserve_locked(**args)
    np.testing.assert_array_equal(out,[10,50,35])
    for name,value in before.items():np.testing.assert_array_equal(args[name],value)
    assert out.sum()<=.95*.1*args['nav']


def test_cancel_purchases_before_selling_existing_positions():
    args=inputs();args['qty']=np.array([10,0,50]);out=reserve_locked(**args)
    np.testing.assert_array_equal(out,[10,40,45])


def test_uncorrectable_locked_excess_remains_visible_without_negative_shares():
    args=inputs();args['locked'][0]=args['qty'][0]=120
    out=reserve_locked(**args);np.testing.assert_array_equal(out,[120,0,0])
    assert out.sum()>.1*args['nav']


def test_smallcap_and_gross_include_locked_reserve():
    args=inputs();args['sector_caps'][:]=1;args['small'][:]=True;args['nav']=200.
    args['stock_caps'][:]=1;out=reserve_locked(**args)
    assert out.sum()<=.285*200 and out[0]==10
    args['small'][:]=False;args['gross']=.2;out=reserve_locked(**args)
    assert out.sum()<=.2*200 and out[0]==10


def test_no_locked_shortfall_preserves_targets_exactly():
    args=inputs();args['locked'][:]=0
    np.testing.assert_array_equal(reserve_locked(**args),args['desired'])


def test_parent_disabled_mode_and_no_action_mode_preserve_ledger():
    parent=Path(__file__).parents[1]/'quant/timefolio_heatmap_retention_replay.py'
    assert hashlib.sha256(parent.read_bytes()).hexdigest()==PARENT_SHA256
    p,ix=synthetic_panel(days=5,names=3);scores=np.ones(p['close'].shape)
    args=(p,ix,scores,ix['dates'][1],ix['dates'][-1]);kw=dict(return_trades=True,return_plans=True,rank_buffer=3)
    old=previous(*args,**kw);assert replay(*args,locked_repair=False,**kw)==old
    fixed=replay(*args,locked_repair=True,**kw);assert fixed['metrics'].pop('locked_target_adjusted_days')==0
    assert fixed==old


def action_case():
    p,ix=synthetic_panel(days=5,names=4);p['sector'][:]=0;p['sector_cap'][:]=.3;p['sector_cap'][:,1:]=.1
    p['split'][0,2]=2
    for key in ['close','o','exec_price','exec_high','exec_low']:p[key][0,2:]/=2
    scores=np.repeat(np.array([4.,3.,2.,1.])[:,None],5,axis=1);scores[:,1:]=np.array([0.,4.,3.,2.])[:,None]
    kw=dict(top_n=3,weight=.05,rank_buffer=0,rebalance=1,max_orders=10,return_trades=True,return_plans=True,
        action_release_dates={(ix['codes'][0],ix['dates'][2]):'20270101'})
    return p,ix,scores,kw


def test_replay_reroutes_sector_repair_while_preserving_locked_shares():
    p,ix,scores,kw=action_case();args=(p,ix,scores,ix['dates'][1],ix['dates'][2])
    old=previous(*args,**kw);new=replay(*args,locked_repair=True,**kw)
    assert old['daily'][-1]['gross']>.1 and new['daily'][-1]['gross']<.1
    assert old['daily'][0]==new['daily'][0]
    assert new['metrics']['locked_target_adjusted_days']==1
    bought=sum(t['qty'] for t in new['trades'] if t['code']==ix['codes'][0] and t['side']=='buy')
    sold=sum(t['qty'] for t in new['trades'] if t['code']==ix['codes'][0] and t['side']=='sell')
    assert sold<=bought and new['metrics']['unreleased_rights_positions']==1


def test_future_values_do_not_change_earlier_repair():
    p,ix,scores,kw=action_case();old=replay(p,ix,scores,ix['dates'][1],ix['dates'][-1],locked_repair=True,**kw)
    altered=deepcopy(p);changed=scores.copy();changed[:,3:]*=-50
    for key in ['close','exec_price','exec_volume','market_cap']:altered[key][:,3:]*=10
    new=replay(altered,ix,changed,ix['dates'][1],ix['dates'][-1],locked_repair=True,**kw)
    assert old['daily'][:2]==new['daily'][:2]
    assert [t for t in old['trades'] if t['date']<=ix['dates'][2]]==[t for t in new['trades'] if t['date']<=ix['dates'][2]]


def test_invalid_inventory_is_rejected():
    args=inputs();args['locked'][0]=11
    with pytest.raises(ValueError,match='inventory'):reserve_locked(**args)

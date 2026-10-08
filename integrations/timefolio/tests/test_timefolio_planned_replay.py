from copy import deepcopy

import numpy as np

from test_timefolio_heatmap import synthetic_panel
from quant.timefolio_heatmap_planned_replay import replay


def test_execution_window_prices_do_not_choose_same_day_plans():
    p,ix=synthetic_panel();scores=np.broadcast_to(np.arange(12)[:,None],p['close'].shape)
    kw=dict(rebalance=1,max_orders=3,return_plans=True,return_trades=True)
    before=replay(p,ix,scores,ix['dates'][1],ix['dates'][2],**kw)
    changed=deepcopy(p);changed['exec_price'][:3,2]=[13000,7000,12000]
    after=replay(changed,ix,scores,ix['dates'][1],ix['dates'][2],**kw)
    assert before['plans']==after['plans']
    assert before['daily'][0]==after['daily'][0]


def test_same_day_close_and_fractional_cash_cannot_fund_morning_orders():
    p,ix=synthetic_panel();p['split'][:,2]=1.05
    p['o'][:,2]/=1.05
    scores=np.broadcast_to(np.arange(12)[:,None],p['close'].shape)
    kw=dict(rebalance=1,max_orders=3,return_plans=True)
    before=replay(p,ix,scores,ix['dates'][1],ix['dates'][2],**kw)
    changed=deepcopy(p);changed['close'][:,2]*=2
    after=replay(changed,ix,scores,ix['dates'][1],ix['dates'][2],**kw)
    assert before['plans']==after['plans']
    assert before['metrics']['omitted_fractional_entitlement_value_proxy']>=0


def test_announced_listing_date_releases_bonus_despite_share_cancellations():
    p,ix=synthetic_panel();p['split'][:,3]=1.5
    for key in ['o','close','exec_price','exec_high','exec_low']:p[key][:,3:]/=1.5
    # Cancellation offsets the new issue, so a total-share threshold never fires.
    p['listed_shares'][:,3:]*=.9;p['sector_cap'][:,2:]=0
    release={(code,ix['dates'][3]):ix['dates'][5] for code in ix['codes']}
    result=replay(p,ix,np.ones_like(p['close']),ix['dates'][1],ix['dates'][5],
                  rebalance=1,action_release_dates=release,return_trades=True)
    assert result['metrics']['unreleased_rights_positions']==0
    assert any(t['side']=='sell' and t['date']==ix['dates'][5] for t in result['trades'])
    assert result['daily'][3]['holdings']>0 and result['daily'][4]['holdings']==0


def test_explicit_release_after_observed_data_stays_locked():
    p,ix=synthetic_panel();p['split'][:,3]=2;p['listed_shares'][:,3:]=10
    for key in ['o','close','exec_price','exec_high','exec_low']:p[key][:,3:]/=2
    p['sector_cap'][:,2:]=0
    release={(code,ix['dates'][3]):'20990101' for code in ix['codes']}
    result=replay(p,ix,np.ones_like(p['close']),ix['dates'][1],ix['dates'][5],
                  rebalance=1,action_release_dates=release,return_trades=True)
    assert result['metrics']['unreleased_rights_positions']>0

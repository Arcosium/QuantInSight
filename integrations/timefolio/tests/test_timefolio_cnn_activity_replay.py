from copy import deepcopy
from datetime import date,timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from quant.timefolio_cnn_activity_replay import ACTIVITY_PARENT_SHA256,replay
from quant.timefolio_heatmap_locked_weighted_replay import replay as frozen_replay
from quant.timefolio_heatmap_planned_audit import audit_fills,additional_checks
from quant.timefolio_cnn_account import turnover_windows


def fixture(days=35):
    dates=[];day=date(2026,4,1)
    while len(dates)<days:
        if day.weekday()<5:dates.append(day.strftime('%Y%m%d'))
        day+=timedelta(days=1)
    shape=(6,days)
    panel={k:np.full(shape,10000.) for k in ['o','close','exec_price','exec_high','exec_low']}
    panel['exec_high']+=50;panel['exec_low']-=50
    panel.update(exec_count=np.full(shape,30.),exec_volume=np.full(shape,1e7),
                 market_cap=np.full(shape,2e12),split=np.ones(shape),listed_shares=np.ones(shape),
                 sector_cap=np.full(shape,.5),sector=np.zeros(6,dtype=int),eligible=np.ones(shape,dtype=bool))
    index=dict(dates=dates,codes=[f'{i:06d}' for i in range(6)])
    scores=np.broadcast_to(np.array([10.,9.,1.,2.,3.,4.])[:,None],shape).copy()
    scores[2:,3:]=np.array([13.,12.,11.,10.5])[:,None]
    return panel,index,scores


@pytest.mark.parametrize('schedule',[False,True])
def test_disabled_activity_is_exactly_the_frozen_ledger(schedule):
    parent=Path(__file__).parents[1]/'quant/timefolio_heatmap_locked_weighted_replay.py'
    assert hashlib.sha256(parent.read_bytes()).hexdigest()==ACTIVITY_PARENT_SHA256
    panel,index,scores=fixture()
    kwargs=dict(rebalance=5,rank_buffer=5,rebalance_band=.005,return_trades=True,return_plans=True,
                max_orders=10,top_n=2,weight=.08,gross=.2)
    if schedule:kwargs.update(gross_schedule=np.full(35,.2),weight_schedule=np.full(35,.08))
    args=(panel,index,scores,index['dates'][1],index['dates'][-1])
    assert replay(*args,**kwargs)==frozen_replay(*args,**kwargs)


def policy_kwargs():
    return dict(rebalance=100,rank_buffer=5,rebalance_band=.005,return_trades=True,return_plans=True,
                max_orders=10,top_n=2,weight=.08,gross=.2,activity_policy=dict(intervene_after_low_weeks=0))


@pytest.mark.parametrize('intervene_after',[0,2])
def test_extra_orders_pay_costs_and_pass_independent_fill_and_exposure_audits(intervene_after):
    panel,index,scores=fixture()
    kwargs=policy_kwargs();kwargs['activity_policy']['intervene_after_low_weeks']=intervene_after
    result=replay(panel,index,scores,index['dates'][1],index['dates'][-1],**kwargs)
    assert result['metrics']['weekly_activity_proposal_pairs']>0
    assert any(not day['regular_rebalance'] and (day['proposal'] or {}).get('pairs') for day in result['activity_plans'])
    limits=audit_fills(panel,index,result,gross=.2)
    extra=additional_checks(panel,index,result,np.full(len(index['dates']),.2),max_orders=10)
    assert not limits['post_buy_limit_violations'] and not extra['additional_errors']
    assert limits['maximum_nav_reconstruction_error_krw']<.01
    assert result['daily'][-1]['nav']==pytest.approx(1e9-result['metrics']['fees_krw']-result['metrics']['slippage_krw'],abs=.01)
    assert result['daily'][-1]['nav']<1e9
    assert all(p['score_improvement']>0 for d in result['activity_plans'] for p in (d['proposal'] or {}).get('pairs',[]))


def test_same_day_execution_liquidity_and_close_cannot_change_activity_order_proposals():
    panel,index,scores=fixture()
    args=(index['dates'][1],index['dates'][-1]);kw=policy_kwargs()
    before=replay(panel,index,scores,*args,**kw)
    target=next(x['date'] for x in before['activity_plans'] if (x['proposal'] or {}).get('pairs'))
    d=index['dates'].index(target)
    changed=deepcopy(panel)
    changed['exec_price'][:,d]*=np.array([.97,1.01,1.02,.98,1.03,.99])
    changed['exec_volume'][:,d]=100.
    changed['close'][:,d]*=1.1
    after=replay(changed,index,scores,*args,**kw)
    assert [p for p in before['plans'] if p['date']<=target]==[p for p in after['plans'] if p['date']<=target]
    old=next(x for x in before['activity_plans'] if x['date']==target)
    new=next(x for x in after['activity_plans'] if x['date']==target)
    assert old['proposal']==new['proposal'] and old['requested_extra_notional']==new['requested_extra_notional']


def test_no_execution_volume_keeps_real_turnover_failure_despite_proposals():
    panel,index,scores=fixture();panel['exec_volume'][:,2:]=0.
    result=replay(panel,index,scores,index['dates'][1],index['dates'][-1],**policy_kwargs())
    assert result['metrics']['weekly_activity_requested_days']>0
    assert result['metrics']['weekly_activity_proposal_pairs']>0
    assert len(result['trades'])==2
    assert all(t['date']==index['dates'][1] for t in result['trades'])
    low=sum(w['turnover']<.05 for w in result['activity_actual_weeks'])
    windows=turnover_windows(result['daily'])
    assert low>=4 and windows[0]['low_turnover_weeks']==low and windows[0]['four_violation_screen_failed']


def test_zero_exposure_gate_cannot_be_silently_overridden_to_claim_compliance():
    panel,index,scores=fixture();kw=policy_kwargs();kw['gross_schedule']=np.zeros(len(index['dates']))
    result=replay(panel,index,scores,index['dates'][1],index['dates'][-1],**kw)
    assert result['metrics']['weekly_activity_requested_days']>0
    assert not result['trades'] and all(r['gross']==0 for r in result['daily'])
    assert turnover_windows(result['daily'])[0]['four_violation_screen_failed']


@pytest.mark.parametrize('intervene_after',[0,2])
def test_activity_result_can_be_persisted_without_custom_json_coercion(intervene_after):
    panel,index,scores=fixture();kwargs=policy_kwargs()
    kwargs['activity_policy']['intervene_after_low_weeks']=intervene_after
    result=replay(panel,index,scores,index['dates'][1],index['dates'][-1],**kwargs)
    assert result['metrics']['weekly_activity_proposal_pairs']>0
    assert type(result['metrics']['weekly_activity_requested_days']) is int
    restored=json.loads(json.dumps(result,allow_nan=False))
    assert restored['daily']==result['daily'] and restored['trades']==result['trades']
    assert restored['metrics']==result['metrics']

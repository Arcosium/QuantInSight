from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from quant.timefolio_cnn_activity_replay import replay as accepted_replay
from quant.timefolio_cnn_core_retention_replay import CORE_RETENTION_PARENT_SHA256,replay
from quant.timefolio_heatmap_planned_audit import audit_fills,additional_checks
from test_timefolio_cnn_activity_replay import fixture,policy_kwargs


def ordinary_kwargs():
    kw=policy_kwargs();kw['rebalance']=5
    return kw


@pytest.mark.parametrize('activity',[None,0,2])
def test_default_retention_preserves_complete_accepted_result(activity):
    parent=Path(__file__).parents[1]/'quant/timefolio_cnn_activity_replay.py'
    assert hashlib.sha256(parent.read_bytes()).hexdigest()==CORE_RETENTION_PARENT_SHA256
    panel,index,scores=fixture();kw=ordinary_kwargs()
    kw['activity_policy']=None if activity is None else dict(intervene_after_low_weeks=activity)
    args=(panel,index,scores,index['dates'][1],index['dates'][-1])
    assert replay(*args,**kw)==accepted_replay(*args,**kw)


@pytest.mark.parametrize('buffer',[0,5])
def test_separation_without_activity_preserves_all_trades_and_actual_core_holdings(buffer):
    panel,index,scores=fixture();kw=ordinary_kwargs();kw['activity_policy']=None;kw['rank_buffer']=buffer
    args=(panel,index,scores,index['dates'][1],index['dates'][-1])
    old=accepted_replay(*args,**kw);new=replay(*args,**kw,separate_activity_retention=True)
    assert old['daily']==new['daily'] and old['trades']==new['trades'] and old['plans']==new['plans']
    qty=Counter()
    for day in new['core_retention_decisions']:
        for trade in [t for t in new['trades'] if t['date']==day['date']]:
            qty[trade['code']]+=trade['qty'] if trade['side']=='buy' else -trade['qty']
        assert set(day['priority_after'])=={code for code,n in qty.items() if n>0}
        assert not day['supplemental_after']


def test_supplemental_new_position_does_not_gain_ordinary_retention_priority():
    panel,index,scores=fixture();kw=ordinary_kwargs()
    result=replay(panel,index,scores,index['dates'][1],index['dates'][-1],**kw,separate_activity_retention=True)
    first=next(d for d in result['core_retention_decisions'] if d['supplemental_after'])
    code=first['supplemental_after'][0]
    assert code not in first['priority_after']
    later=next(d for d in result['core_retention_decisions'] if d['date']>first['date'] and d['regular_rebalance'])
    assert code not in later['ordinary_admission_candidates'] and code not in later['priority_after']
    assert all(set(d['priority_after'])<=set(index['codes'][:2]) for d in result['core_retention_decisions'])
    original=accepted_replay(panel,index,scores,index['dates'][1],index['dates'][-1],**kw)
    old_topups=[t for t in original['trades'] if t['code']==code and t['side']=='buy' and t['date']>first['date']]
    assert old_topups and max(t['qty']*t['price'] for t in old_topups)>2e7
    limits=audit_fills(panel,index,result,gross=.2)
    additional=additional_checks(panel,index,result,np.full(len(index['dates']),.2),max_orders=10)
    assert not limits['post_buy_limit_violations'] and not additional['additional_errors']
    assert limits['maximum_nav_reconstruction_error_krw']<.01
    json.dumps(result,allow_nan=False)


def test_blocked_ordinary_entry_does_not_gain_priority():
    panel,index,scores=fixture();panel['exec_volume'][:,2:]=0.
    kw=ordinary_kwargs();kw['activity_policy']=None;kw['rank_buffer']=0
    result=replay(panel,index,scores,index['dates'][1],index['dates'][-1],**kw,separate_activity_retention=True)
    code=index['codes'][2]
    assert any(code in d['ordinary_admission_candidates'] for d in result['core_retention_decisions'])
    assert all(code not in d['priority_after'] for d in result['core_retention_decisions'])
    assert all(t['code']!=code for t in result['trades'])


def test_supplemental_position_can_be_admitted_by_later_ordinary_cnn_selection():
    panel,index,scores=fixture();kw=ordinary_kwargs()
    initial=replay(panel,index,scores,index['dates'][1],index['dates'][-1],**kw,separate_activity_retention=True)
    first=next(d for d in initial['core_retention_decisions'] if d['supplemental_after'])
    code=first['supplemental_after'][0]
    schedule=np.full(len(index['dates']),5,dtype=int)
    schedule[index['dates'].index(first['date']):]=0
    changed=replay(panel,index,scores,index['dates'][1],index['dates'][-1],**kw,
                   separate_activity_retention=True,rank_buffer_schedule=schedule)
    admission=next(d for d in changed['core_retention_decisions'] if d['date']>first['date'] and code in d['priority_after'])
    assert admission['regular_rebalance'] and code in admission['ordinary_admission_candidates']


def test_current_execution_and_close_do_not_change_current_priority_or_order_plan():
    panel,index,scores=fixture();kw=ordinary_kwargs()
    before=replay(panel,index,scores,index['dates'][1],index['dates'][-1],**kw,separate_activity_retention=True)
    target=next(d['date'] for d in before['core_retention_decisions'] if d['supplemental_after'])
    d=index['dates'].index(target);other=deepcopy(panel)
    other['exec_price'][:,d]*=.93;other['exec_volume'][:,d]=10.;other['close'][:,d]*=1.25
    after=replay(other,index,scores,index['dates'][1],index['dates'][-1],**kw,separate_activity_retention=True)
    assert [p for p in before['plans'] if p['date']<=target]==[p for p in after['plans'] if p['date']<=target]
    a=next(x for x in before['core_retention_decisions'] if x['date']==target)
    b=next(x for x in after['core_retention_decisions'] if x['date']==target)
    assert a['priority_before']==b['priority_before']
    assert a['ordinary_admission_candidates']==b['ordinary_admission_candidates']


def test_retention_mode_requires_an_explicit_boolean():
    panel,index,scores=fixture()
    with pytest.raises(ValueError,match='boolean'):
        replay(panel,index,scores,index['dates'][1],index['dates'][-1],separate_activity_retention=1)

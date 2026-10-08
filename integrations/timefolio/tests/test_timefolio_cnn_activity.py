from datetime import date,timedelta

import numpy as np
import pytest

from quant.timefolio_cnn_activity import WeeklyActivity,rank_improving_rotation


def weekdays(start,end):
    current=date.fromisoformat(start);last=date.fromisoformat(end);out=[]
    while current<=last:
        if current.weekday()<5:out.append(current.strftime('%Y%m%d'))
        current+=timedelta(days=1)
    return out


def test_only_actual_prior_fills_count_and_unfilled_proposals_do_not_pass_week():
    dates=weekdays('2024-01-01','2024-01-19');state=WeeklyActivity(dates)
    for day in dates[:13]:
        assert state.plan(day,1000.)['requested_extra_notional']==0
        state.observe(day,1000.,0.,0.)
    thursday=state.plan(dates[13],1000.)
    assert thursday['prior_low_weeks']==2 and thursday['remaining_sessions']==2
    assert thursday['requested_extra_notional']==pytest.approx(55.)
    state.observe(dates[13],1000.,0.,0.)  # A planned order was not filled.
    friday=state.plan(dates[14],1000.)
    assert friday['requested_extra_notional']==pytest.approx(110.)
    state.observe(dates[14],1000.,0.,0.)
    assert state.closed[-1]['turnover']==0 and len(state.closed)==3


def test_future_nav_is_an_estimate_and_actual_denominator_controls_final_result():
    dates=weekdays('2024-01-01','2024-01-05')
    state=WeeklyActivity(dates,intervene_after_low_weeks=0)
    for day in dates[:4]:state.observe(day,1000.,0.,0.)
    proposal=state.plan(dates[-1],1000.)
    assert proposal['requested_extra_notional']==pytest.approx(110.)
    state.observe(dates[-1],2000.,55.,55.)
    assert state.closed[0]['turnover']==pytest.approx(55/1200)
    assert state.closed[0]['turnover']<.05


def test_contest_budget_resets_and_boundary_week_counts_without_nav_reset():
    dates=weekdays('2024-02-28','2024-04-05');state=WeeklyActivity(dates,intervene_after_low_weeks=1)
    for day in dates:
        proposal=state.plan(day,1200.)
        if day.startswith('202403'):assert proposal['reason']=='outside_contest_window'
        if day.startswith('202404'):assert proposal['prior_low_weeks']==0
        state.observe(day,1200.,0.,0.)
    assert len(state.closed)==2 and state.closed[0]['sessions']==2
    assert state.closed[0]['window']==(2024,1) and state.closed[1]['window']==(2024,4)


def test_calendar_holidays_and_duplicate_closes_fail_closed():
    state=WeeklyActivity(['20240102','20240103'],intervene_after_low_weeks=0)
    assert state.plan('20240102',1000.)['requested_extra_notional']==pytest.approx(55.)
    with pytest.raises(ValueError,match='next known session'):state.observe('20240103',1000.,0.,0.)
    state.observe('20240102',1000.,20.,30.)
    assert state.plan('20240103',1000.)['requested_extra_notional']==pytest.approx(60.)
    with pytest.raises(ValueError,match='next known session'):state.observe('20240102',1000.,0.,0.)


def inputs():
    return dict(qty=np.array([100,100,0,0]),desired=np.array([100,100,0,0]),
        locked_qty=np.zeros(4,dtype=int),prices=np.ones(4),scores=np.array([.2,.6,.9,.1]),
        eligible=np.ones(4,dtype=bool),sectors=np.array([0,1,0,1]),market_cap=np.full(4,2e12),
        codes=np.array(['A','B','C','D']),nav=1000.,gross_ceiling=.8,requested_notional=100.)


def test_small_rank_improving_rotation_preserves_sector_cap_class_and_input():
    values=inputs();original=values['desired'].copy()
    desired,proof=rank_improving_rotation(**values)
    np.testing.assert_equal(desired,[80,100,20,0])
    np.testing.assert_equal(values['desired'],original)
    assert proof['extra_planned_notional']==40 and proof['unmet_planned_notional']==60
    assert desired.sum()==original.sum() and proof['pairs'][0]['score_improvement']>0


def test_locks_cash_gate_order_budget_and_missing_prices_do_not_create_fills():
    for change in [dict(locked_qty=np.array([100,0,0,0])),dict(gross_ceiling=0.),
                   dict(max_orders=1),dict(prices=np.array([1.,1.,np.nan,1.])),
                   dict(market_cap=np.array([2e12,2e12,5e11,2e12])),
                   dict(scores=np.array([1.,.6,.9,.1]))]:
        values=inputs();values.update(change)
        desired,proof=rank_improving_rotation(**values)
        np.testing.assert_equal(desired,values['desired'])
        assert not proof['pairs'] and proof['unmet_planned_notional']==100.


def test_base_order_requests_are_estimates_and_integer_rounding_cannot_add_gross():
    values=inputs();values['desired']=np.array([95,100,5,0]);values['requested_notional']=8.
    desired,proof=rank_improving_rotation(**values)
    np.testing.assert_equal(desired,values['desired'])
    assert proof['base_planned_notional']==10. and proof['estimated_only']
    values=inputs();values['prices']=np.array([3.,1.,7.,1.]);values['locked_qty'][0]=97
    desired,proof=rank_improving_rotation(**values)
    assert desired[0]>=97 and desired[2]==1
    assert desired@values['prices']<=values['qty']@values['prices']
    assert proof['unmet_planned_notional']>0

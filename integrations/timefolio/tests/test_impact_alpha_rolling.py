import numpy as np
import pandas as pd
import pytest

from quant.impact_alpha_features import FEATURE_COLUMNS
from quant.impact_alpha_rolling import windows,replay,pick_candidate,moving_block_ci,context_columns


def candidate(n=30,days=4,lower=.1,mean=1.,horizon=30):
    return {'horizon':horizon,'quantile':.97,'calibration':{
        'n':n,'trading_days':days,'mean_net_bps_ci95':[lower,2.],
        'mean_gross_bps':mean}}


def table():
    rows=[]
    for t in [2000,15000,40000]:
        rows.append({'day':'2026-09-01','event_ms':t-1000,'decision_ms':t,
          'entry_valid':True,'entry_ms':t+100,'entry_price':100.,'entry_quote_ms':t,
          'quantity':.1,'exit_valid_30':True,'exit_ms_30':t+30100,
          'exit_price_30':101.,'exit_quote_ms_30':t+30000})
    return pd.DataFrame(rows)


def test_daily_walk_forward_has_no_random_split_or_future_training():
    days=pd.date_range('2026-06-30',periods=90).strftime('%Y-%m-%d').tolist()
    folds=windows(days)
    assert len(folds)==62
    for fold in folds:
        assert len(fold['train'])==21 and len(fold['calibration'])==7
        assert max(fold['train'])<min(fold['calibration'])<fold['test']
        assert max(fold['calibration'])<fold['test']
        assert not set(fold['train'])&set(fold['calibration'])
    assert folds[1]['train'][0]==folds[0]['train'][1]


@pytest.mark.parametrize('c',[candidate(n=29),candidate(days=3),candidate(lower=0),candidate(lower=-1)])
def test_cost_gate_returns_cash_when_past_evidence_fails(c):
    assert pick_candidate([c]) is None


def test_gated_selector_uses_lower_bound_and_probe_is_explicitly_different():
    a=candidate(lower=.1,mean=50)
    b=candidate(lower=.2,mean=1,horizon=300)
    assert pick_candidate([a,b])==b
    assert pick_candidate([a,b],gated=False)==a


def test_replay_executes_next_trade_only_after_close_and_never_uses_return_to_enter():
    frame=table()
    rows,audit=replay(frame,[3,5,3],2,30)
    assert len(rows)==2 and audit['occupied']==1
    frame['exit_price_30']=90.
    losing,_=replay(frame,[3,5,3],2,30)
    assert losing.event_ms.tolist()==rows.event_ms.tolist()


def test_selected_missing_exit_raises_but_unselected_missing_exit_does_not():
    frame=table();frame.loc[0,'exit_valid_30']=False
    with pytest.raises(ValueError,match='Unpriced selected exit'):
        replay(frame,[3,0,0],2,30)
    rows,_=replay(frame,[0,0,3],2,30)
    assert len(rows)==1


def test_context_ablation_keeps_only_price_trade_and_clock_features():
    names=context_columns(FEATURE_COLUMNS)
    assert len(names)==14
    assert 'f_sign' in names and 'f_return_60s_bps' in names
    assert not any('refill' in n or 'depth' in n or 'impact' in n or 'obi' in n for n in names)


def test_moving_block_interval_handles_cash_days_and_all_cash():
    days=pd.date_range('2026-09-01',periods=14).strftime('%Y-%m-%d').tolist()
    rows=pd.DataFrame({'day':days[:7],'net_bps':[2.]*7})
    ci=moving_block_ci(rows,days)
    assert ci==pytest.approx([2.,2.])
    assert moving_block_ci(rows.iloc[:0],days) is None

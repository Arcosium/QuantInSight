from copy import deepcopy
import numpy as np
import pytest
from quant.timefolio_cnn_payoff_gate import walk_payoffs


def inputs():
    s=np.arange(1,31,dtype=int)
    x=np.column_stack([np.sin(s),s/30])
    y=np.column_stack([x[:,0]*.03-.01,x[:,1]*.02-.015,-x[:,0]*.01])
    folds=[dict(id='early',test_start=1,test_end=15),dict(id='late',test_start=15,test_end=31)]
    return x,y,s,s+3,folds


def test_unmatured_future_payoffs_cannot_change_a_fitted_forecast():
    x,y,s,end,folds=inputs()
    a=walk_payoffs(x,y,s,end,folds,days=31,minimum_rows=5)
    changed=y.copy();changed[end>=15]=1e8
    b=walk_payoffs(x,changed,s,end,folds,days=31,minimum_rows=5)
    np.testing.assert_array_equal(a['expected_net'],b['expected_net'])
    assert a['fits']==b['fits']
    assert a['fits'][1]['last_label_available']==14
    assert 12 not in a['fits'][1]['training_signals']  # label available exactly at cutoff15 is excluded.


def test_missing_target_uses_the_same_matured_rows_for_all_targets():
    x,y,s,end,folds=inputs();y[2,2]=np.nan
    out=walk_payoffs(x,y,s,end,folds,days=31,minimum_rows=5)
    fit=out['fits'][1]
    assert fit['matured_rows']==10 and 3 not in fit['training_signals']
    train=np.array(fit['training_signals'])-1
    mean=x[train].mean(0);scale=x[train].std(0);z=(x[train]-mean)/scale
    beta=np.linalg.solve(z.T@z+10*np.eye(2),z.T@(y[train]-y[train].mean(0)))
    np.testing.assert_allclose(out['expected_net'][15:31],(x[14:]-mean)/scale@beta+y[train].mean(0))


def test_cold_start_remains_invested_without_inventing_expectations():
    x,y,s,end,folds=inputs()
    out=walk_payoffs(x,y,s,end,folds,days=31,minimum_rows=60)
    assert out['cold_start'][s].all() and np.isnan(out['expected_net'][s]).all()
    assert (out['gross'][s]==.8).all()


def test_future_feature_availability_is_rejected():
    x,y,s,end,folds=inputs();available=s.copy();available[0]+=1
    with pytest.raises(ValueError,match='causal'):
        walk_payoffs(x,y,s,end,folds,days=31,feature_available=available)


def test_overlapping_or_missing_forecast_folds_are_rejected():
    x,y,s,end,folds=inputs();overlap=deepcopy(folds);overlap[1]['test_start']=14
    with pytest.raises(ValueError,match='Disjoint'):
        walk_payoffs(x,y,s,end,overlap,days=31)
    gap=deepcopy(folds);gap[1]['test_start']=16
    with pytest.raises(ValueError,match='Every feature'):
        walk_payoffs(x,y,s,end,gap,days=31)

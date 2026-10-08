from copy import deepcopy
import numpy as np
import pytest
import torch

from quant.timefolio_cnn_batch import job_command, policy_mode
from quant.timefolio_cnn_policy_replay import policy_schedule, FIXED_POLICY
from quant.timefolio_cnn_train import fit
from quant.timefolio_heatmap_locked_weighted_replay import replay
from test_timefolio_heatmap import synthetic_panel


def test_joint_selection_job_cannot_silently_fall_back_to_fixed_or_price_selection():
    spec=dict(dataset='dataset/manifest.json',output='results',account_panel='account_panel')
    job=dict(name='202401_joint',fold='202401',objective='listnet',seed=17,
             rebalance=5,rank_buffer=5,policy_selection='retention_grid')
    command=job_command(spec,job)
    assert command[command.index('--policy-selection')+1]=='retention_grid'
    assert command[command.index('--account-panel')+1]=='account_panel'
    for changed in [dict(account_panel=None),dict(account_panel='account_panel')]:
        bad=dict(job,rebalance=20) if changed['account_panel'] else job
        with pytest.raises(ValueError):policy_mode(bad,**changed)
    legacy=job_command(spec,{k:v for k,v in job.items() if k!='policy_selection'})
    assert '--policy-selection' not in legacy and '--account-panel' in legacy


def test_best_epoch_execution_policy_is_kept_instead_of_the_last_epoch_policy():
    torch.set_num_threads(1);rng=np.random.default_rng(17)
    x=rng.uniform(-1,1,(16,1,8,20)).astype(np.float32);y=np.zeros(16,np.float32)
    signal=np.repeat(np.arange(4),4);keys=np.tile(np.arange(4),4)
    cfg=dict(seed=17,width=2,dropout=.1,lr=.01,weight_decay=.001,epochs=3,
             top_k=2,temperature=.2,objective='mse',max_date_group=8)
    class Validator:
        sample_ids=np.flatnonzero(signal==2)
        provenance=dict(joint_epoch_and_policy_selection=True)
        def __init__(self):self.epoch=0
        def evaluate(self,scores):
            assert scores.shape==(4,) and np.isfinite(scores).all()
            # Exact profit ties preserve the earlier epoch even if its buffer
            # is larger. A later, infeasible high-profit epoch must lose.
            value=[.03,.03,.99][self.epoch];buffer=[20,5,50][self.epoch]
            self.epoch+=1
            return dict(net_return=value,turnover_screen_passed=self.epoch<3,
                        selected_policy=dict(FIXED_POLICY,rank_buffer=buffer))
    _,proof=fit(x,y,signal,keys,signal<2,signal==2,cfg,account_validator=Validator())
    assert proof['best_epoch']==1 and proof['selected_policy']['rank_buffer']==20
    assert proof['selected_policy_turnover_screen_passed'] is True
    assert proof['criterion']=='inner_account_net_profit_and_retention'


def test_constant_retention_schedule_preserves_every_fill_and_daily_balance():
    p,ix=synthetic_panel(days=12,names=24)
    scores=np.random.default_rng(17).normal(size=(24,12))
    kw=dict(rank_buffer=5,max_orders=10,rebalance_band=.005,return_trades=True)
    original=replay(p,ix,scores,ix['dates'][1],ix['dates'][-1],**kw)
    other=replay(p,ix,scores,ix['dates'][1],ix['dates'][-1],rank_buffer_schedule=np.full(12,5),**kw)
    assert other['metrics'].pop('rank_retention_buffer_schedule')==dict(minimum=5,maximum=5)
    assert original['metrics'].pop('rank_retention_buffer')==5
    assert other==original


def test_retention_change_is_next_session_only_and_preserves_existing_holdings():
    p,ix=synthetic_panel(days=8,names=4)
    scores=np.tile([4.,3.,2.,1.],(8,1)).T;scores[:,1:]=np.array([1.,4.,3.,2.])[:,None]
    schedule=np.full(8,50,dtype=int)
    kw=dict(top_n=1,max_orders=10,rebalance=1,return_trades=True,rank_buffer_schedule=schedule)
    original=replay(p,ix,scores,ix['dates'][1],ix['dates'][-1],**kw)
    schedule[2:]=0
    changed=replay(p,ix,scores,ix['dates'][1],ix['dates'][-1],**kw)
    assert changed['daily'][:2]==original['daily'][:2]
    assert any(t['date']==ix['dates'][3] and t['code']==ix['codes'][0] and t['side']=='sell'
               for t in changed['trades'])
    assert changed['daily'][2]['cash']!=1e9
    cadence=replay(p,ix,scores,ix['dates'][1],ix['dates'][-1],
        **dict(kw,rebalance=5))
    assert not any(t['side']=='sell' and t['date']<ix['dates'][6] for t in cadence['trades'])


def test_invalid_retention_schedules_are_rejected_before_replay():
    p,ix=synthetic_panel(days=8,names=4);scores=np.ones((4,8))
    for schedule in [np.full(8,5.),np.ones(8,bool),np.full(8,-1),np.full(7,5),np.full(8,np.nan)]:
        with pytest.raises(ValueError,match='rank_buffer_schedule'):
            replay(p,ix,scores,ix['dates'][1],ix['dates'][-1],rank_buffer_schedule=schedule)


def receipts():
    dates=np.array([f'202401{i:02}' for i in range(1,11)])
    folds=[dict(id='a',test_start=4,test_end=7),dict(id='b',test_start=7,test_end=10)]
    proofs=[]
    for fold,buffer in zip(folds,[20,50]):
        candidates=[dict(rank_buffer=b,net_return=.02 if b==buffer else .01,
                        turnover_screen_passed=True,last_account_mark_index=fold['test_start']-1)
                    for b in [5,20,50]]
        winner=next(c for c in candidates if c['rank_buffer']==buffer)
        policy=dict(FIXED_POLICY,rank_buffer=buffer)
        history=[dict(epoch=1,inner_account=dict(winner,candidate_policies=candidates,selected_policy=policy))]
        proofs.append(dict(fold=fold,config=dict(epochs=1,selected_policy=policy,
            account_selection=dict(joint_epoch_and_policy_selection=True,rank_buffers=[5,20,50])),
            selection=dict(best_epoch=1,criterion='inner_account_net_profit_and_retention',
                history=history,selected_policy=policy,selected_policy_turnover_screen_passed=True),
            refit=dict(best_epoch=1),last_training_label_index=fold['test_start']-2,
            last_selection_label_index=fold['test_start']-2,last_selection_account_mark_index=fold['test_start']-1))
    return dates,folds,proofs


def test_monthly_policy_is_indexed_by_signal_day_without_overwriting_prior_month():
    dates,folds,proof=receipts();schedule,covered,choices=policy_schedule(dates,folds,proof)
    np.testing.assert_array_equal(schedule,[5,5,5,5,20,20,20,50,50,50])
    np.testing.assert_array_equal(covered,[False]*4+[True]*6)
    assert choices[1]['last_selection_account_mark_index']==6
    changed=deepcopy(proof);changed[1]['last_selection_account_mark_index']=7
    with pytest.raises(ValueError,match='precede'):policy_schedule(dates,folds,changed)


def test_tampered_epoch_or_selected_buffer_cannot_be_used_by_the_account():
    dates,folds,proof=receipts()
    changed=deepcopy(proof);changed[0]['config']['selected_policy']['rank_buffer']=5
    with pytest.raises(ValueError):policy_schedule(dates,folds,changed)
    changed=deepcopy(proof);changed[0]['selection']['history'][0]['inner_account']['candidate_policies'][0]['net_return']=1.
    with pytest.raises(ValueError,match='constrained account profit'):policy_schedule(dates,folds,changed)
    with pytest.raises(ValueError):policy_schedule(dates,folds[::-1],proof[::-1])

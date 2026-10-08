import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from quant.timefolio_cnn_account import evaluate_daily
from quant.timefolio_cnn_account_selection import AccountValidator
from quant.timefolio_cnn_core_retention_replay import replay
from quant.timefolio_cnn_execution_selection import PairedAccountValidator, contest_policy
from quant.timefolio_cnn_paired_train import arm_selections
from quant.timefolio_cnn_train import fit


def fixture():
    n,d=12,85;dates=pd.bdate_range('2024-01-01',periods=d).strftime('%Y%m%d').tolist()
    price=np.full((n,d),10000.)
    p=dict(o=price.copy(),close=price.copy(),eligible=np.ones((n,d),bool),sector=np.arange(n),
           sector_cap=np.full((n,d),.5),market_cap=np.full((n,d),2e12),split=np.ones((n,d)),
           listed_shares=np.ones((n,d)),exec_price=price.copy(),exec_volume=np.full((n,d),1e7),
           exec_count=np.full((n,d),30.),exec_high=price.copy(),exec_low=price.copy())
    arrays=dict(dates=np.array(dates),signal_index=np.repeat(np.arange(d),n),
                security_key=np.tile(np.arange(n),d),eligible=np.ones(n*d,bool),returns=np.zeros(n*d))
    return p,dict(dates=dates,codes=[f'{i:06}' for i in range(n)]),arrays,dict(validation_start=10,test_start=70)


def validator(p,index,a,fold):
    return PairedAccountValidator(AccountValidator(p,index,a,fold,rebalance=5,rank_buffer=5))


def test_both_arms_stop_before_outer_and_ignore_future_label_availability():
    p,ix,a,f=fixture();v=validator(p,ix,a,f);scores=np.sin(np.arange(len(v.sample_ids)))
    before=v.evaluate(scores);a['returns'][:]=np.nan
    for key in ['o','close','exec_price','exec_high','exec_low']:p[key][:,70:]*=1000
    after=validator(p,ix,a,f)
    np.testing.assert_array_equal(v.sample_ids,after.sample_ids)
    assert before==after.evaluate(scores)
    for result in [before,before['paired_legacy_account']]:
        assert result['last_account_mark_index']==69 and result['end']==ix['dates'][69]
    assert v.signal.max()==58


def test_contest_selection_exactly_matches_final_executor_with_complete_policy():
    p,ix,a,f=fixture();v=validator(p,ix,a,f);scores=np.cos(np.arange(len(v.sample_ids))/12)
    observed=v.evaluate(scores)
    matrix=np.full((12,60),np.nan);matrix[v.key,v.signal]=scores
    expected=replay({k:x if k=='sector' else x[:,10:70] for k,x in p.items()},
                    dict(codes=ix['codes'],dates=ix['dates'][10:70]),matrix,ix['dates'][11],ix['dates'][69],
                    gross_schedule=np.full(60,.95),return_trades=True,**contest_policy())
    assert {k:observed[k] for k in ['sessions','net_return','sharpe','mdd']}==evaluate_daily(expected['daily'])['pooled']
    assert observed['fees_krw']==expected['metrics']['fees_krw']
    assert expected['metrics']['max_daily_orders']==10
    assert observed['contest_policy']['separate_activity_retention'] is True
    assert observed['net_return']<0  # Flat prices still pay costs.
    assert observed['paired_legacy_account']==v.legacy.evaluate(scores)


def test_missing_prints_remain_cash_and_invalid_predictions_are_rejected():
    p,ix,a,f=fixture();p['exec_count'][:]=0;v=validator(p,ix,a,f)
    result=v.evaluate(np.ones(len(v.sample_ids)))
    assert result['net_return']==0 and result['fees_krw']==0 and result['sharpe'] is None
    assert result['turnover_screen_passed'] is False
    with pytest.raises(ValueError):v.evaluate(np.full(len(v.sample_ids),np.nan))
    with pytest.raises(ValueError):v.evaluate(np.zeros(len(v.sample_ids)-1))


def test_arm_selection_keeps_feasibility_profit_and_earliest_tie_independent():
    rows=[]
    for epoch,(profit,passed,legacy) in enumerate([(.04,True,.03),(.50,False,.08),(.04,True,.08)],1):
        rows.append(dict(epoch=epoch,training_loss=1.,inner_account=dict(net_return=profit,
            turnover_screen_passed=passed,paired_legacy_account=dict(net_return=legacy,turnover_screen_passed=True))))
    shared=dict(history=rows,best_epoch=1,criterion='inner_account_net_profit');before=copy.deepcopy(shared)
    selections=arm_selections(shared)
    assert selections['contest']['best_epoch']==1 and selections['legacy']['best_epoch']==2
    assert shared==before
    assert all('paired_legacy_account' not in r['inner_account'] for x in selections.values() for r in x['history'])


def test_actual_small_cnn_fit_records_both_account_arms():
    torch.set_num_threads(1)
    p,ix,a,f=fixture();v=validator(p,ix,a,f);rng=np.random.default_rng(17)
    images=rng.uniform(-1,1,(len(a['returns']),1,8,20)).astype(np.float32)
    labels=rng.normal(0,.01,len(images)).astype(np.float32);s=a['signal_index']
    cfg=dict(seed=17,width=2,dropout=.1,lr=.001,weight_decay=.001,epochs=2,top_k=10,
             temperature=.2,objective='listnet',max_date_group=512)
    _,shared=fit(images,labels,s,a['security_key'],s<5,(s>=10)&(s<69),cfg,account_validator=v)
    selections=arm_selections(shared)
    assert set(selections)=={'legacy','contest'}
    assert len(shared['history'])==2
    assert all(row['inner_account']['last_account_mark_index']==69 for row in shared['history'])


@pytest.mark.parametrize('same_epoch',[False,True])
def test_worker_persists_two_arms_and_reuses_only_identical_refit_counts(tmp_path,monkeypatch,same_epoch):
    import quant.timefolio_cnn_paired_train as worker
    n,d=12,230;s=np.repeat(np.arange(d),n);ids=np.tile(np.arange(n),d)
    a=dict(images=np.empty((n*d,1,8,20),np.float32),returns=np.zeros(n*d),signal_index=s,
           label_end_index=s+2,security_key=ids,eligible=np.ones(n*d,bool))
    a['returns'][s>=210]=np.nan  # Unknown outcomes must not remove forecasts.
    fold=dict(id='202401',validation_start=140,test_start=210,test_end=220)
    manifest=dict(folds=[fold],paired_account_policy=contest_policy(),limitations=['synthetic test'],feature_rows=8)
    dataset=tmp_path/'manifest.json';dataset.write_text('{}')
    monkeypatch.setattr(worker,'load_dataset',lambda *args,**kwargs:(manifest,a))
    monkeypatch.setattr(worker,'load_paired_validator',lambda *args:SimpleNamespace(provenance={},last_mark_index=209))
    calls=[]
    def fake_fit(*args,epochs=None,**kwargs):
        calls.append(epochs)
        if epochs is not None:return torch.nn.Linear(1,1),dict(epochs=epochs)
        history=[dict(epoch=i,training_loss=1.,inner_account=dict(net_return=float(i),turnover_screen_passed=True,
            paired_legacy_account=dict(net_return=float(i if same_epoch else 3-i),turnover_screen_passed=True))) for i in [1,2]]
        return torch.nn.Linear(1,1),dict(best_epoch=2,history=history,criterion='inner_account_net_profit')
    monkeypatch.setattr(worker,'fit',fake_fit)
    monkeypatch.setattr(worker,'predict',lambda model,images,rows,**kwargs:np.ones(len(rows),np.float32))
    args=SimpleNamespace(dataset=dataset,output=tmp_path/'result',fold='202401',account_panel=tmp_path,
                         rebalance=5,rank_buffer=5,device='cpu',purpose='exploratory',objective='listnet',seed=17)
    worker.run(args)
    import json
    receipt=json.loads((args.output/'receipt.json').read_text())
    assert receipt['physical_refits']==(1 if same_epoch else 2)
    assert calls==([None,2] if same_epoch else [None,1,2])
    assert receipt['prediction_count']==120 and receipt['last_training_label_index']<210
    assert set(receipt['artifacts'])=={'model_contest.pt','model_legacy.pt','sample_ids.npy','scores_contest.npy','scores_legacy.npy'}
    assert receipt['logical_arm_models']==2
    for name,value in receipt['artifacts'].items():assert worker.sha(args.output/name)==value

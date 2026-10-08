import numpy as np
import pytest
from quant.timefolio_cnn_gate import basket_features, net_roundtrip, walk_gate


def test_basket_selection_uses_scores_and_stable_ids_without_realized_returns():
    c=np.ones((20,12))*100;c[-1]=101
    top,x=basket_features(c,np.ones(12),np.arange(12)[::-1])
    assert top.tolist()==list(range(11,1,-1)) and x.shape==(8,)
    assert x[0]==pytest.approx(.01) and x[2]==1 and x[7]==0
    _,scaled=basket_features(c*5,np.ones(12),np.arange(12)[::-1])
    np.testing.assert_allclose(x,scaled,atol=1e-12)


def test_net_label_charges_both_sides_and_cash_opportunity():
    expected=1.01*(1-.0005)*(1-.003)/((1+.0005)*(1+.001))-1
    assert net_roundtrip(.01)==pytest.approx(expected)
    assert net_roundtrip(.004)<0 and net_roundtrip(.01,cash_annual=.02)<expected


def test_gate_does_not_fit_unmatured_labels_or_future_feature_scaling():
    s=np.arange(100,201);x=np.tile(np.sin(s)[:,None],(1,8));y=np.sin(s)*.01
    fold=[dict(id='fixture',test_start=180,test_end=201)]
    args=dict(signal=s,label_end=s+6,train_end=s-1,selection_end=s-1,folds=fold,days=210)
    before=walk_gate(x,y,**args)
    y[s+6>=180]=100
    changed=x.copy();changed[(s<180)&(s+6>=180)]+=1e6
    after=walk_gate(changed,y,**args)
    np.testing.assert_array_equal(before[0],after[0])
    assert before[3][0]['matured_dates']==74 and before[3][0]['last_label_end']==179
    assert before[3]==after[3]


def test_gate_cold_start_and_zero_exposure_remain_explicit_requests():
    s=np.arange(100,121);x=np.zeros((21,8));y=np.zeros(21)
    expected,cold,exposure,fits=walk_gate(x,y,s,s+6,s-1,s-1,
        [dict(id='fixture',test_start=120,test_end=121)],days=125)
    assert cold[120] and np.isnan(expected[120]) and exposure['ridge_cash'][120]==.8
    assert exposure['cash'][120]==0 and fits[0]['cold_start']


def test_gate_selects_the_registered_seed_and_never_substitutes_another(tmp_path):
    import json
    from quant.timefolio_cnn_dataset import sha
    from quant.timefolio_cnn_gate import build
    dataset=tmp_path/'dataset';dataset.mkdir();results=tmp_path/'results';results.mkdir()
    signal=np.repeat([20,21],12);keys=np.tile(np.arange(12),2)
    dates=np.array([str(d).replace('-','') for d in np.arange('2024-01-01','2024-02-05',dtype='datetime64[D]')])
    arrays=dict(daily_ohlcv=np.ones((35,12,5))*100,returns=np.full(24,.01),
        signal_index=signal,label_end_index=signal+6,security_key=keys,
        eligible=np.ones(24,bool),dates=dates,codes=np.array([f'{i:06}' for i in range(12)]))
    specs={}
    for name,values in arrays.items():
        path=dataset/(name+'.npy');np.save(path,values)
        specs[name]=dict(path=path.name,sha256=sha(path))
    fold=dict(id='202401',test_start=20,test_end=22)
    manifest=dict(exploratory_ready=True,horizon=5,arrays=specs,folds=[fold])
    (dataset/'manifest.json').write_text(json.dumps(manifest))
    for seed in [17,29]:
        folder=results/f'202401_bce_{seed}';folder.mkdir()
        np.save(folder/'sample_ids.npy',np.arange(24))
        np.save(folder/'scores.npy',np.arange(24,dtype=np.float32)*(1 if seed==17 else -1))
        proof=dict(dataset_manifest_sha256=sha(dataset/'manifest.json'),fold=fold,
            config=dict(objective='bce',seed=seed),last_training_label_index=19,
            last_selection_label_index=19,artifacts={p.name:sha(p) for p in folder.iterdir()})
        (folder/'receipt.json').write_text(json.dumps(proof))
    default=build(dataset,results,tmp_path/'default','bce')
    alternate=build(dataset,results,tmp_path/'alternate','bce',seed=29)
    assert default['seed']==17 and alternate['seed']==29
    before=np.load(tmp_path/'default/scores.npy');after=np.load(tmp_path/'alternate/scores.npy')
    np.testing.assert_equal(before[20:22],-after[20:22])
    with pytest.raises(ValueError,match='Missing registered monthly model'):
        build(dataset,results,tmp_path/'missing','bce',seed=43)
    p=results/'202401_bce_29/receipt.json';proof=json.loads(p.read_text())
    proof['config']['seed']=17;p.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='Model identity'):
        build(dataset,results,tmp_path/'wrong','bce',seed=29)

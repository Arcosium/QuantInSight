import numpy as np
import pandas as pd
import pytest
from autofolio import learning


def genome(**kwargs):
    return dict({k:v[0] for k,v in learning.domains("kr").items()},**kwargs)

def panel(n=600):
    rng=np.random.default_rng(42);days=pd.bdate_range('2021-01-01',periods=n);frames=[]
    for i in range(3):
        c=100*np.exp(np.cumsum(rng.normal(.0001,.01,n)))
        frames.append(pd.DataFrame(dict(date=days,symbol=str(i),open=c,high=c*1.01,low=c*.99,close=c,volume=1e6,eligible=True,tradable_buy=True,tradable_sell=True,sector='x')))
    return pd.concat(frames,ignore_index=True)


def test_all_families_are_fitted_and_labels_mature():
    raw=panel();cutoff=pd.Timestamp('2022-10-01')
    for family in learning.domains('kr')['model']:
        g=learning.normalize(genome(model=family), 'kr');p,names=learning.features(raw,g,'kr')
        model,proof=learning.fit_model(p,names,cutoff,g)
        assert pd.Timestamp(proof['max_label_end'])<cutoff
        assert np.isfinite(model.predict(p[p.ready][names].tail(3).to_numpy())).all()
        assert proof['train_rows']>=100


def test_features_and_training_do_not_see_future():
    raw=panel();g=learning.normalize(genome(),'us');cutoff=pd.Timestamp('2022-10-01')
    p,names=learning.features(raw,g,'us');model,proof=learning.fit_model(p,names,cutoff,g)
    changed=raw.copy();changed.loc[changed.date>=cutoff,['open','high','low','close']]*=100
    q,_=learning.features(changed,g,'us');model2,proof2=learning.fit_model(q,names,cutoff,g)
    np.testing.assert_allclose(p[p.date<cutoff][names],q[q.date<cutoff][names],equal_nan=True)
    x=p[p.ready][names].head(10).to_numpy()
    np.testing.assert_allclose(model.predict(x),model2.predict(x))
    assert proof==proof2


def test_gaps_cannot_create_training_labels():
    raw=panel();raw=raw[~((raw.symbol=='0')&(raw.date==pd.Timestamp('2022-05-03')))]
    g=learning.normalize(genome(),'kr');p,_=learning.features(raw,g,'kr')
    row=p[(p.symbol=='0')&(p.date==pd.Timestamp('2022-05-02'))].iloc[0]
    assert pd.isna(row.target)


def test_trade_next_session_only_and_cash_conservation(monkeypatch):
    from autofolio import period
    monkeypatch.setattr(period,'expected_dates',lambda *args:('20260101','20260102','20260103'))
    p=pd.DataFrame([dict(date=pd.Timestamp(day),symbol='A',open=10,close=10,tradable_buy=True,tradable_sell=True) for day in ['20260101','20260102','20260103']])
    sig=pd.DataFrame([dict(date=pd.Timestamp('20260101'),symbol='A',score=1,adv20=1e9)])
    case,stale=learning.account(p,sig,learning.normalize(genome(),'crypto'),'crypto','20260101','20260103')
    assert [r['date'] for r in case['trades']]==['20260102']
    assert case['daily'][0]['nav']==case['initial_cash']
    assert case['daily'][1]['nav']==pytest.approx(case['initial_cash']-case['trades'][0]['fee'])
    assert case['daily'][1]['cash']>=0


def test_seed_normalization():
    assert learning.normalize(genome(seed=987654321),'kr')['seed']==987654321
    for invalid in [dict(model='arbitrary-code'),dict(seed=-5),dict(seed=True),dict(seed=1.5),dict(lookback=5.0),dict(feature_set='news'),dict(foo='bar')]:
        with pytest.raises(ValueError):learning.normalize(genome(**invalid),'kr')
    with pytest.raises(ValueError):learning.normalize({},'kr')


def test_trial_discards_models_and_retrain_retains_offline_model(tmp_path,monkeypatch):
    import json
    from autofolio import learning_input,model_recipe,period,config
    monkeypatch.setattr(config,'RUNS',tmp_path/'runs')
    raw=panel();data=tmp_path/'input.parquet';raw.to_parquet(data,index=False)
    monkeypatch.setattr(learning_input,'prepare',lambda market:raw)
    monkeypatch.setattr(learning_input,'descriptor',lambda market:dict(path=str(data),sha256=model_recipe.file_hash(data)))
    monkeypatch.setattr(period,'window',lambda:('20221003','20221031'))
    monkeypatch.setattr(period,'expected_dates',lambda *args:tuple(pd.bdate_range('2022-10-03','2022-10-31').strftime('%Y%m%d')))
    monkeypatch.setattr(period,'require_complete',lambda days,market:days)
    result=learning.evaluate(dict(market='us',definition=genome(),title='trained',user_id='test'),tmp_path/'trial')
    report=json.loads(result.read_text())
    assert report['model_proofs'] and report['weights_retained'] is False
    assert sorted(p.name for p in result.parent.iterdir())==['recipe.json','review.json']
    artifact=learning.retrain(report['recipe'],tmp_path/'deploy')
    assert artifact.exists()
    import joblib
    bundle=joblib.load(artifact)
    assert bundle['recipe_id']==report['recipe']['recipe_id']
    assert json.loads((artifact.parent/'deployment.json').read_text())['no_broker_orders'] is True


def test_timefolio_conservative_unknown_sector_cap_and_costs(monkeypatch):
    from autofolio import period
    monkeypatch.setattr(period,'expected_dates',lambda *args:('20260101','20260102','20260105'))
    p=pd.DataFrame([dict(date=pd.Timestamp(day),symbol=symbol,open=10,close=10,tradable_buy=True,tradable_sell=True,sector='UNKNOWN') for day in ['20260101','20260102','20260105'] for symbol in ['A','B','C']])
    sig=pd.DataFrame([dict(date=pd.Timestamp(day),symbol=symbol,score=1,adv20=1e12) for day in ['20260101','20260102'] for symbol in ['A','B','C']])
    g={k:v[0] for k,v in learning.domains('timefolio').items()}
    case,_=learning.account(p,sig,g,'timefolio','20260101','20260105')
    for row in case['daily']:assert row['nav']-row['cash']<=row['nav']*.1+.01
    buys=[r for r in case['trades'] if r['side']=='buy'];sells=[r for r in case['trades'] if r['side']=='sell']
    assert buys and sells
    assert buys[0]['fee']==pytest.approx(buys[0]['qty']*buys[0]['price']*.0015)
    assert sells[0]['fee']==pytest.approx(sells[0]['qty']*sells[0]['price']*.0035)
    assessment=learning.timefolio_assessment(case)
    assert assessment['competition_compliance_verified'] is False
    assert learning.domains('timefolio')['holding_sessions']==[1,3,5]

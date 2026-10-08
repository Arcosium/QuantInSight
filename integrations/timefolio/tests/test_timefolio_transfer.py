import numpy as np
from quant.timefolio_heatmap_transfer import transform, hypotheses


def fixture():
    rng=np.random.default_rng(47);n,t=40,100
    price=100*np.exp(np.cumsum(rng.normal(0,.01,(n,t)),axis=1))
    p=dict(close=price,exec_price=price.copy(),exec_count=np.full((n,t),26),
           split=np.ones((n,t)),eligible=np.ones((n,t),bool))
    return rng.normal(size=(n,t)),p,[f'2026{i:04d}' for i in range(101,201)]


def test_no_future_prices_or_scores_enter_any_variant():
    score,p,dates=fixture()
    for variant in ['smooth5','smooth10','cal20','cal60']:
        a,g,proof=transform(score,p,dates,variant)
        future={k:v.copy() for k,v in p.items()};future['close'][:,61:]*=5;future['exec_price'][:,61:]/=7
        changed=score.copy();changed[:,61:]*=-3
        b,h,_=transform(changed,future,dates,variant)
        np.testing.assert_array_equal(a[:,:61],b[:,:61])
        if g is not None:
            np.testing.assert_array_equal(g[:61],h[:61])
            assert set(g).issubset({.2,.6})
            assert any(r['fitted'] for r in proof['receipts'])
            assert all(r['last_observed_label_close']<=r['date'] for r in proof['receipts'] if r['fitted'])


def test_smoothing_uses_daily_ranks_and_current_eligibility():
    s=np.array([[1.,4.],[2.,3.],[3.,2.],[4.,1.]])
    p={'eligible':np.ones(s.shape,bool)}
    a,g,_=transform(s,p,['20260101','20260102'],'smooth5')
    np.testing.assert_allclose(a[:,1],.625)
    p['eligible'][0,1]=False
    b,_,_=transform(s,p,['20260101','20260102'],'smooth5')
    assert np.isnan(b[0,1]) and g is None


def test_registered_family_contains_original_and_shared_controls():
    rows=hypotheses(['one'])
    assert len(rows)==1344
    assert sum(ref is not None for _,_,ref in rows)==768
    assert any(label=='unchanged_original' for _,label,_ in rows)

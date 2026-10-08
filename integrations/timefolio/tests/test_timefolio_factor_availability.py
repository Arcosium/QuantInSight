import numpy as np
from quant.timefolio_heatmap_factor_residual import transform,VARIANTS


def fixture():
    rng=np.random.default_rng(200);s=rng.normal(size=(40,10))
    p=dict(eligible=np.ones(s.shape,bool),market_cap=rng.uniform(1,100,s.shape),
        vol20=rng.uniform(.01,.1,s.shape),sector=np.arange(40)%4)
    return s,p,[str(i) for i in range(10)]


def test_future_covariates_do_not_change_past_projection():
    score,p,dates=fixture();changed={k:v.copy() for k,v in p.items()}
    changed['market_cap'][:,5:]*=np.arange(1,41)[:,None]
    changed['vol20'][:,5:]*=np.arange(40,0,-1)[:,None]
    for v in VARIANTS:
        a,_,_=transform(score,p,dates,v);b,_,_=transform(score,changed,dates,v)
        np.testing.assert_array_equal(a[:,:5],b[:,:5])


def test_only_current_missing_covariates_exclude_scores():
    score,p,dates=fixture();p['market_cap'][:21,2]=np.nan;p['market_cap'][0,3]=np.nan
    result,_,proof=transform(score,p,dates,'remove_cap')
    assert np.isnan(result[:,2]).all() and not proof['receipts'][2]['projected']
    assert np.isnan(result[0,3]) and np.isfinite(result[1:,3]).all()
    assert np.isfinite(result[:,1]).all() and np.isfinite(result[:,4]).all()

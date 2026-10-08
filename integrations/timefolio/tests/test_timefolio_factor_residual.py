import numpy as np
from quant.timefolio_heatmap_factor_residual import residual,transform,VARIANTS


def test_projection_removes_added_factor_component():
    rng=np.random.default_rng(8);x=np.column_stack([np.ones(40),rng.normal(size=(40,2))]);y=rng.normal(size=40)
    a,proof=residual(y,x);b,_=residual(y + x @ np.array([3.,-5.,2.]),x)
    np.testing.assert_allclose(a,b,atol=1e-13)
    assert proof['orthogonality_error']<1e-12
    constant,_=residual(np.ones(40),np.ones((40,3)))
    np.testing.assert_allclose(constant,0,atol=1e-14)


def test_current_date_projection_is_causal_and_permutation_equivariant():
    rng=np.random.default_rng(98);score=rng.normal(size=(40,12))
    p=dict(eligible=np.ones(score.shape,bool),market_cap=rng.uniform(1,100,score.shape),
        vol20=rng.uniform(.01,.1,score.shape),sector=np.arange(40)%5)
    dates=[str(d) for d in range(12)];order=rng.permutation(40)
    for v in VARIANTS:
        a,g,proof=transform(score,p,dates,v)
        q={k:value[order] for k,value in p.items()};b,_,_=transform(score[order],q,dates,v)
        np.testing.assert_allclose(a[order],b,atol=1e-13)
        changed=score.copy();changed[:,6:]*=20
        c,_,_=transform(changed,p,dates,v)
        np.testing.assert_array_equal(a[:,:6],c[:,:6])
        assert g is None and not proof['uses_returns_as_fit_target']

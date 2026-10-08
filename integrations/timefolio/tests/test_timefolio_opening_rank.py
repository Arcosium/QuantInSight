import numpy as np
import pytest
from quant.timefolio_heatmap_opening_rank import transform,VARIANTS,hypotheses


def fixture():
    rng=np.random.default_rng(12);s=rng.normal(size=(40,10))
    p=dict(eligible=np.ones(s.shape,bool),close=rng.uniform(90,110,s.shape),
        o=rng.uniform(90,110,s.shape),vol20=rng.uniform(.01,.1,s.shape),
        exec_price=rng.uniform(90,110,s.shape),split=np.ones(s.shape))
    return (s,np.arange(10)),p,[f'{d:02d}' for d in range(10)]


def test_current_and_future_close_execution_and_actions_are_not_inputs():
    held,p,dates=fixture();q={k:v.copy() for k,v in p.items()}
    q['close'][:,5:]*=10;q['vol20'][:,5:]*=8;q['exec_price'][:]*=20;q['split'][:]*=2
    q['o'][:,6:]*=np.arange(1,41)[:,None]
    for v in VARIANTS:
        a,_=transform(held,p,dates,v);b,_=transform(held,q,dates,v)
        np.testing.assert_array_equal(a[:,:6],b[:,:6])


def test_current_open_affects_current_decision_but_not_previous_decisions():
    held,p,dates=fixture();q={k:v.copy() for k,v in p.items()}
    q['o'][:,5]=np.linspace(20,200,40)
    a,proof=transform(held,p,dates,'confirm50');b,_=transform(held,q,dates,'confirm50')
    np.testing.assert_array_equal(a[:,:5],b[:,:5]);assert not np.array_equal(a[:,5],b[:,5])
    assert proof['receipts'][4]['decision_date']=='05' and proof['receipts'][4]['forecast_origin_date']=='04'


def test_missing_open_does_not_remove_forecast_and_small_cross_section_has_no_overlay():
    held,p,dates=fixture();p['o'][0,5]=np.nan;p['o'][:21,6]=np.nan
    a,proof=transform(held,p,dates,'fade50');zero,_=transform(held,p,dates,'zero')
    assert np.isfinite(a[0,5]) and a[0,5]==zero[0,5]
    np.testing.assert_array_equal(a[:,6],zero[:,6]);assert not proof['receipts'][5]['overlay_available']
    np.testing.assert_array_equal(np.isfinite(a),np.isfinite(zero))


def test_zero_overlay_preserves_order_and_row_permutation():
    held,p,dates=fixture();zero,_=transform(held,p,dates,'zero')
    np.testing.assert_array_equal(np.argsort(zero[:,1:],axis=0),np.argsort(held[0][:,:-1],axis=0))
    order=np.arange(40)[::-1];q={k:v[order] for k,v in p.items()}
    for v in VARIANTS:
        a,_=transform(held,p,dates,v);b,_=transform((held[0][order],held[1]),q,dates,v)
        np.testing.assert_allclose(a[order],b,atol=1e-13,equal_nan=True)


def test_forecast_origin_may_be_old_but_never_future():
    held,p,dates=fixture();origins=held[1].copy();origins[4]=1
    _,proof=transform((held[0],origins),p,dates,'fade25')
    assert proof['receipts'][4]['forecast_origin_date']=='01'
    origins[4]=5
    with pytest.raises(AssertionError):transform((held[0],origins),p,dates,'fade25')


def test_every_source_seed_variant_and_comparator_is_retained():
    h=hypotheses(['case'+str(i) for i in range(9)])
    assert len(h)==11584 and len({k for k,_,_ in h})==4672
    for c in ['matching_untrained_opening','same_opening_nonimage','unchanged_original']:
        assert sum(label==c for _,label,_ in h)==2304

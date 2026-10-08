import numpy as np
import pytest

from quant.timefolio_cnn_context import context_rows, ranks


def sample():
    rng=np.random.default_rng(21)
    close=100*np.exp(np.cumsum(rng.normal(0,.01,(80,5)),axis=0))
    volume=rng.uniform(1,2,(80,5))*1e6
    adv=volume*close
    membership=np.ones(close.shape,bool)
    return close,volume,adv,membership,np.array([10,10,20,20,-1])


def test_context_is_invariant_to_future_prices_and_future_membership():
    c,v,a,m,s=sample();first,available=context_rows(c,v,a,m,s)
    c[50:]*=30;v[50:]*=100;a[50:]*=17;m[50:]=False
    second,other=context_rows(c,v,a,m,s)
    np.testing.assert_array_equal(first[:50],second[:50])
    np.testing.assert_array_equal(available[:50],other[:50])
    assert first.shape==(80,5,8) and np.isfinite(first).all()
    assert first.min()>=-1 and first.max()<=1


def test_cross_section_membership_ties_and_missing_sector_are_explicit():
    np.testing.assert_array_equal(ranks([7,7,7]),[0,0,0])
    np.testing.assert_array_equal(ranks([1,2,2,3]),[-1,0,0,1])
    c,v,a,m,s=sample();m[:,4]=False
    rows,available=context_rows(c,v,a,m,s)
    assert not rows[:,4].any() and not available[:,4].any()
    assert not rows[:20].any() and not available[:20].any()
    m[:,4]=True;rows,_=context_rows(c,v,a,m,s)
    assert not rows[:,4,5].any()
    with pytest.raises(ValueError):ranks([np.nan,0])

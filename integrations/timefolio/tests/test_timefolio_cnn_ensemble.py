import numpy as np
import pytest

from quant.timefolio_cnn_ensemble import mean_rank_percentiles


def test_equal_seed_weight_ties_missing_and_singletons():
    first=np.array([[10.,10.,30.,np.nan],[np.nan,7.,np.nan,np.nan],
                    [np.nan]*4])
    second=np.array([[3.,2.,1.,np.nan],[np.nan,-5.,np.nan,np.nan],
                     [np.nan]*4])
    third=np.array([[1.,1.,1.,np.nan],[np.nan,3.,np.nan,np.nan],
                    [np.nan]*4])
    original=first.copy()
    result=mean_rank_percentiles([first,second,third])
    np.testing.assert_allclose(result[0,:3],[20/36,16/36,18/36],rtol=1e-6)
    assert result[1,1]==.5
    np.testing.assert_array_equal(np.isnan(result),np.isnan(first))
    np.testing.assert_equal(first,original)


def test_rank_scale_invariance_and_no_cross_date_influence():
    first=np.array([[3.,1.,2.],[5.,2.,3.]])
    second=np.array([[1.,2.,3.],[7.,8.,9.]])
    result=mean_rank_percentiles([first,second])
    transformed=mean_rank_percentiles([first*10.+300.,second*2.-4.])
    np.testing.assert_equal(result,transformed)
    changed=second.copy();changed[1]=[-1.,-3.,-2.]
    np.testing.assert_equal(result[0],mean_rank_percentiles([first,changed])[0])
    np.testing.assert_equal(result,mean_rank_percentiles([second,first]))


def test_missing_seed_scores_or_invalid_geometry_are_not_silently_averaged():
    good=np.ones((2,3),dtype=float)
    missing=good.copy();missing[0,1]=np.nan
    with pytest.raises(ValueError,match='same observations'):
        mean_rank_percentiles([good,missing])
    for inputs in [[good],[good,good.T],[good,good.astype(int)],
                   [good,good*np.inf],[np.ones(3),np.ones(3)]]:
        with pytest.raises(ValueError):
            mean_rank_percentiles(inputs)

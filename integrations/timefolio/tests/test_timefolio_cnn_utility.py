import numpy as np
import pytest

from quant.timefolio_cnn_utility import utility_targets


def sample():
    daily=np.full((12,2,5),100.)
    daily[:,:,1]=110;daily[:,:,2]=95
    signals=np.array([0,2,8]);keys=np.array([0,1,0]);end=signals+4
    returns=np.array([0.,0.,np.nan])
    return daily,signals,keys,end,returns


def test_flat_prices_pay_full_roundtrip_and_downside_utility_is_separate():
    d,s,k,e,r=sample()
    out=utility_targets(d,s,k,e,r,horizon=3,penalty=.5)
    expected=(1-.0005)*(1-.003)/((1+.0005)*(1+.001))-1
    assert out['net_holding_returns'][0]==pytest.approx(expected)
    assert out['maximum_adverse_excursion'][0]==pytest.approx(.05)
    assert out['training_utility'][0]==pytest.approx(expected-.025)
    assert r[0]==0 and np.isnan(out['training_utility'][2])


def test_exit_day_low_and_later_outcomes_cannot_change_a_matured_target():
    d,s,k,e,r=sample();before=utility_targets(d,s,k,e,r,horizon=3,penalty=1.)
    d[e[0]:,0,2]=1.
    after=utility_targets(d,s,k,e,r,horizon=3,penalty=1.)
    assert before['training_utility'][0]==after['training_utility'][0]


def test_missing_low_does_not_hide_net_return_but_blocks_downside_label():
    d,s,k,e,r=sample();d[2,0,2]=np.nan
    net=utility_targets(d,s,k,e,r,horizon=3,penalty=0.)
    risk=utility_targets(d,s,k,e,r,horizon=3,penalty=.5)
    assert np.isfinite(net['training_utility'][0]) and np.isnan(risk['training_utility'][0])


def test_price_unit_scaling_and_mismatched_economic_labels():
    d,s,k,e,r=sample();before=utility_targets(d,s,k,e,r,horizon=3,penalty=.5)
    scaled=d.copy();scaled[:,:,:4]*=5
    after=utility_targets(scaled,s,k,e,r,horizon=3,penalty=.5)
    np.testing.assert_allclose(before['training_utility'],after['training_utility'],equal_nan=True)
    r[0]=.3
    with pytest.raises(ValueError,match='disagree'):
        utility_targets(d,s,k,e,r,horizon=3,penalty=.5)

import numpy as np
from quant.timefolio_cnn_controls import control_scores


def test_control_features_are_past_only_and_use_the_same_signal_membership():
    rng = np.random.default_rng(5)
    close = np.exp(rng.normal(4, .05, (60, 12))); adv = np.full(close.shape, 4e9)
    member = np.ones(close.shape, bool); member[:, 0] = False
    before = control_scores(close, adv, member)
    close[40:] *= 100; adv[40:] = 100; member[40:] = False
    after = control_scores(close, adv, member)
    for name in before:
        np.testing.assert_array_equal(before[name][:40], after[name][:40])
        assert np.isnan(before[name][:, 0]).all()


def test_price_based_controls_ignore_common_price_unit_rescaling():
    rng = np.random.default_rng(7)
    close = np.exp(rng.normal(5, .1, (40, 12))); adv = np.full(close.shape, 4e9)
    member = np.ones(close.shape, bool)
    first = control_scores(close, adv, member)
    second = control_scores(close*np.arange(1, 13), adv, member)
    for name in first:
        np.testing.assert_allclose(first[name], second[name], equal_nan=True, atol=1e-12)

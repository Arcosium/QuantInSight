import numpy as np
import pytest

from quant.timefolio_cnn_absolute_gate import absolute_basket_gate


def gate(rank, raw, keys=None, **kwargs):
    return absolute_basket_gate(rank, raw, np.arange(rank.shape[1]) if keys is None else keys,
                               probability_objective='bce', **kwargs)


def test_absolute_level_changes_cash_even_when_cross_sectional_ranking_is_identical():
    rank = np.tile(np.arange(12, dtype=float), (3, 1))
    raw = rank / 100 - 1
    before, after = gate(rank, raw), gate(rank, raw + 2)
    np.testing.assert_array_equal(before['gross_cash'], 0)
    np.testing.assert_array_equal(after['gross_cash'], .8)
    np.testing.assert_array_equal(before['gross_floor20'], .2)
    # Arbitrary ListNet score shifts must not change a BCE-based gate.
    np.testing.assert_array_equal(after['mean_probability'], gate(rank + 100, raw + 2)['mean_probability'])


def test_probability_is_for_ranker_selected_stocks_with_stable_security_ties():
    rank = np.ones((1, 12)); raw = np.full((1, 12), -2.); raw[:, :2] = 20.
    keys = np.arange(12)[::-1]
    result = gate(rank, raw, keys)
    assert result['mean_probability'][0] == pytest.approx(1 / (1 + np.exp(2)))
    order = np.array([9, 2, 1, 8, 5, 0, 7, 4, 10, 11, 3, 6])
    np.testing.assert_allclose(result['mean_probability'],
                              gate(rank[:, order], raw[:, order], keys[order])['mean_probability'])


def test_future_dates_do_not_change_any_prior_exposure_or_probability():
    rank = np.tile(np.arange(12, dtype=float), (6, 1)); raw = rank / 10 - 1
    before = gate(rank, raw)
    rank[4:] *= -1; raw[4:] = 999
    after = gate(rank, raw)
    for key in before:
        np.testing.assert_array_equal(before[key][:4], after[key][:4])


def test_extreme_logits_and_exact_neutral_probability_have_defined_behavior():
    rank = np.ones((3, 10)); raw = np.repeat([[-1e10], [0.], [1e10]], 10, axis=1)
    with np.errstate(over='raise', invalid='raise'):
        result = gate(rank, raw)
    np.testing.assert_array_equal(result['mean_probability'], [0., .5, 1.])
    np.testing.assert_array_equal(result['gross_cash'], [0., 0., .8])


def test_missing_whole_dates_are_not_backfilled_and_partial_membership_rejects():
    rank = np.ones((3, 12)); raw = rank.copy(); rank[0] = np.nan; raw[0] = np.nan
    result = gate(rank, raw)
    assert np.isnan(result['gross_cash'][0]) and result['forecast_count'][0] == 0
    raw[1, 0] = np.nan
    with pytest.raises(ValueError, match='exactly the same'):
        gate(rank, raw)
    rank[1, 0] = np.nan; rank[1, :4] = raw[1, :4] = np.nan
    with pytest.raises(ValueError, match='Incomplete basket'):
        gate(rank, raw)


def test_listwise_probability_source_invalid_axes_and_infinities_reject():
    rank = np.ones((3, 12)); raw = rank.copy(); keys = np.arange(12)
    with pytest.raises(ValueError, match='Only audited BCE'):
        absolute_basket_gate(rank, raw, keys, probability_objective='listnet')
    for invalid in [np.full((3, 12), np.inf), np.ones((3, 11))]:
        with pytest.raises(ValueError):
            gate(rank, invalid)
    with pytest.raises(ValueError):
        gate(rank, raw, np.zeros(12))
    with pytest.raises(ValueError):
        gate(rank, raw, top_k=True)


def test_account_uses_prior_close_gate_and_keeps_unfilled_shares():
    from quant.timefolio_heatmap_locked_weighted_replay import replay
    from test_timefolio_heatmap import synthetic_panel
    panel, index = synthetic_panel(days=5, names=3)
    rank = np.tile([3., 2., 1.], (5, 1)); raw = np.ones_like(rank)
    raw[1:] = -1
    schedule = gate(rank, raw, top_k=2)['gross_cash']
    # On the liquidation day, volume permits no fills: cash requests cannot
    # remove held shares or create sale proceeds outside the execution ledger.
    panel['exec_volume'][:, 2] = 0
    result = replay(panel, index, rank.T, index['dates'][1], index['dates'][3],
                    gross_schedule=schedule, top_n=2, return_trades=True, max_orders=10)
    buys = [t for t in result['trades'] if t['side'] == 'buy']
    sells = [t for t in result['trades'] if t['side'] == 'sell']
    assert buys and all(t['date'] == index['dates'][1] for t in buys)
    assert sells and all(t['date'] == index['dates'][3] for t in sells)
    assert result['daily'][1]['gross'] > 0
    assert result['daily'][2]['gross'] == 0

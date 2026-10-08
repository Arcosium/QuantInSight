import numpy as np
import pytest

from test_timefolio_heatmap import synthetic_panel
from quant.timefolio_heatmap_longonly_training import net_targets
from quant.timefolio_heatmap_longonly_evaluation import gated_scores, prior_family
from quant.timefolio_heatmap_planned_replay import replay


def test_after_cost_labels_match_cash_round_trip_and_preserve_missing():
    gross = np.array([-.1, 0., .004, .006, .5, np.nan])
    binary, net = net_targets(gross, 'binary')
    cash = 1_000_000.; entry = 10_000.; slip = .0005
    qty = cash/(entry*(1+slip)*1.001)
    proceeds = qty*entry*(1+gross)*(1-slip)*.997
    np.testing.assert_allclose(net, proceeds/cash-1, equal_nan=True)
    np.testing.assert_equal(binary, [0, 0, 0, 1, 1, np.nan])
    reg, _ = net_targets(gross, 'return')
    np.testing.assert_allclose(reg, np.clip(net/.10, -3, 3), rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize('mode,threshold', [('binary', .5), ('return', 0.)])
def test_cash_gate_distinguishes_missing_from_observed_nonpositive(mode, threshold):
    raw = np.array([[np.nan, threshold-.1, threshold+.1, threshold],
                    [np.nan, threshold-.2, threshold-.1, threshold]])
    score, schedule = gated_scores(raw, np.ones_like(raw, bool), mode, True)
    np.testing.assert_array_equal(schedule, [.8, 0., .8, 0.])
    assert np.isfinite(score).sum() == 1
    assert score[0, 2] == raw[0, 2]
    unchanged, rank_schedule = gated_scores(raw, np.ones_like(raw, bool), mode, False)
    np.testing.assert_equal(unchanged, raw)
    np.testing.assert_equal(rank_schedule, np.full(4, .8))


def test_ineligible_positive_forecast_does_not_prevent_cash():
    raw = np.array([[-.1], [1.]])
    _, schedule = gated_scores(raw, [[True], [False]], 'return', True)
    assert schedule[0] == 0


def test_all_negative_exits_between_rebalances_but_missing_does_not():
    p, ix = synthetic_panel(); raw = np.ones_like(p['close'])
    raw[:, 1] = np.nan; raw[:, 2] = -1
    scores, schedule = gated_scores(raw, p['eligible'], 'return', True)
    result = replay(p, ix, scores, ix['dates'][1], ix['dates'][3], rebalance=5,
                    top_n=4, weight=.05, max_orders=10, gross_schedule=schedule, return_trades=True)
    assert result['daily'][0]['holdings'] == 4
    assert result['daily'][1]['holdings'] == 4
    assert result['daily'][2]['holdings'] == 0
    assert not any(t['date'] == ix['dates'][2] for t in result['trades'])
    assert sum(t['side'] == 'sell' for t in result['trades']) == 4


def test_gate_is_causal_in_date_axis():
    p, _ = synthetic_panel(); raw = np.ones_like(p['close']); raw[:, :4] = -.1
    before = gated_scores(raw, p['eligible'], 'return', True)
    raw[:, 4:] = -100
    after = gated_scores(raw, p['eligible'], 'return', True)
    for a, b in zip(before, after): np.testing.assert_equal(a[..., :4], b[..., :4])

from copy import deepcopy

import numpy as np
import pytest

from test_timefolio_heatmap import synthetic_panel
from quant.timefolio_heatmap_locked_replay import replay as prior_replay
from quant.timefolio_heatmap_locked_weighted_replay import replay


def setup():
    p, ix = synthetic_panel(days=8, names=3)
    score = np.broadcast_to(np.array([3., 2., 1.])[:, None], p['close'].shape)
    return p, ix, score


def test_optional_schedule_and_constant_schedule_preserve_frozen_ledger():
    p, ix, score = setup()
    args = (p, ix, score, ix['dates'][1], ix['dates'][-1])
    kw = dict(locked_repair=True, top_n=3, weight=.05, max_orders=3, return_trades=True, rebalance_band=.0005)
    original = prior_replay(*args, **kw)
    assert replay(*args, **kw) == original
    scheduled = replay(*args, weight_schedule=np.full(8, .05), **kw)
    assert scheduled['metrics'].pop('mean_target_stock_weight') == pytest.approx(.05)
    assert scheduled == original


def test_schedule_uses_previous_close_and_does_not_read_future_values():
    p, ix, score = setup(); schedule = np.full(8, .05)
    args = (p, ix, score, ix['dates'][1], ix['dates'][-1])
    kw = dict(locked_repair=True, top_n=1, weight=.05, rebalance=1, return_trades=True, slip=0.)
    original = replay(*args, weight_schedule=schedule, **kw)
    schedule[3:] = .025
    result = replay(*args, weight_schedule=schedule, **kw)
    assert result['daily'][:3] == original['daily'][:3]
    assert [t for t in result['trades'] if t['date'] <= ix['dates'][3]] == [
        t for t in original['trades'] if t['date'] <= ix['dates'][3]]
    assert .024 < result['daily'][3]['gross'] < .026
    assert any(t['side'] == 'sell' and t['date'] == ix['dates'][4] for t in result['trades'])


def test_weight_reduction_binds_without_gross_ceiling_and_preserves_cadence():
    p, ix, score = setup(); before = deepcopy(p)
    schedule = np.full(8, .05); schedule[2:] = .025; original_schedule = schedule.copy()
    result = replay(p, ix, score, ix['dates'][1], ix['dates'][-1], top_n=1, weight=.05,
                    rebalance=5, weight_schedule=schedule, return_trades=True, slip=0.)
    assert .049 < result['daily'][4]['gross'] < .051
    assert .024 < result['daily'][5]['gross'] < .026
    assert {t['date'] for t in result['trades']} == {ix['dates'][1], ix['dates'][6]}
    for k in p: np.testing.assert_equal(p[k], before[k])
    np.testing.assert_equal(schedule, original_schedule)


@pytest.mark.parametrize('value', [np.zeros(8), np.full(8, -.01), np.full(8, .051),
                                  np.full(8, np.nan), np.full((3, 8), .05)])
def test_invalid_schedule_is_rejected(value):
    p, ix, score = setup()
    with pytest.raises(ValueError, match='weight_schedule'):
        replay(p, ix, score, ix['dates'][1], ix['dates'][-1], weight=.05, weight_schedule=value)


def test_weight_schedule_keeps_the_frozen_locked_share_repair():
    import hashlib
    from pathlib import Path
    from test_timefolio_locked_targets import action_case
    from quant.timefolio_heatmap_locked_weighted_replay import PARENT_SHA256
    parent = Path(__file__).parents[1] / 'quant/timefolio_heatmap_locked_replay.py'
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == PARENT_SHA256
    p, ix, score, kw = action_case()
    args = p, ix, score, ix['dates'][1], ix['dates'][-1]
    original = prior_replay(*args, locked_repair=True, **kw)
    result = replay(*args, locked_repair=True, weight_schedule=np.full(len(ix['dates']), .05), **kw)
    assert result['metrics'].pop('mean_target_stock_weight') == pytest.approx(.05)
    assert result == original

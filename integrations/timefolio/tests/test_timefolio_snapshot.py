import numpy as np
import pytest

from test_timefolio_heatmap import synthetic_panel
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_planned_replay import replay


def test_snapshot_anchors_execution_sessions_not_calendar_days():
    p, ix = synthetic_panel(days=15, names=2)
    raw = np.tile(np.arange(15), (2, 1)).astype(float)
    held, origins = snapshot_scores(raw, p['eligible'], ix['dates'], ix['dates'][1], ix['dates'][12], 5)
    np.testing.assert_array_equal(origins[:12], [0]*5+[5]*5+[10]*2)
    np.testing.assert_array_equal(held[0, :12], origins[:12])
    assert np.isnan(held[:, 12:]).all()


def test_future_and_between_refresh_scores_do_not_change_snapshot():
    p, ix = synthetic_panel(days=15, names=2); raw = np.ones_like(p['close'])
    before, _ = snapshot_scores(raw, p['eligible'], ix['dates'], ix['dates'][1], ix['dates'][12], 5)
    raw[:, 1:5] = 999; raw[:, 6:] = -999
    after, _ = snapshot_scores(raw, p['eligible'], ix['dates'], ix['dates'][1], ix['dates'][12], 5)
    np.testing.assert_array_equal(before[:, :10], after[:, :10])


def test_no_security_enters_using_a_score_that_was_ineligible_at_origin():
    p, ix = synthetic_panel(days=15, names=2); raw = np.ones_like(p['close'])
    p['eligible'][0, 0] = False
    held, _ = snapshot_scores(raw, p['eligible'], ix['dates'], ix['dates'][1], ix['dates'][12], 5)
    assert np.isnan(held[0, :5]).all()
    assert np.isfinite(held[0, 5:10]).all()


def test_no_eligible_session_preserves_original_missing_signal_behavior():
    p, ix = synthetic_panel(days=10, names=6); raw = np.ones_like(p['close'])
    p['eligible'][:, 1] = False; raw[:, 1] = np.nan
    held, _ = snapshot_scores(raw, p['eligible'], ix['dates'], ix['dates'][1], ix['dates'][3], 5)
    result = replay(p, ix, held, ix['dates'][1], ix['dates'][3], rebalance=1,
                    top_n=4, weight=.05, max_orders=10, return_trades=True)
    assert result['daily'][1]['holdings'] == result['daily'][0]['holdings']
    assert not any(t['date']==ix['dates'][2] for t in result['trades'])


def test_pending_orders_use_held_ranking_but_recheck_current_admission():
    p, ix = synthetic_panel(days=10, names=6)
    raw = np.tile(np.arange(6, 0, -1)[:, None], (1, 10)).astype(float)
    raw[:, 1:] = raw[::-1, 1:].copy()
    held, _ = snapshot_scores(raw, p['eligible'], ix['dates'], ix['dates'][1], ix['dates'][3], 5)
    args = dict(top_n=4, weight=.05, max_orders=1, return_trades=True)
    original = replay(p, ix, raw, ix['dates'][1], ix['dates'][3], **args)
    result = replay(p, ix, held, ix['dates'][1], ix['dates'][3], **args)
    def final_shares(account, code):
        return sum(t['qty']*(1 if t['side']=='buy' else -1) for t in account['trades'] if t['code']==code)
    assert final_shares(original, ix['codes'][0]) == 0
    assert final_shares(result, ix['codes'][0]) > 0
    p['trade_allowed'] = np.ones_like(p['eligible']); p['trade_allowed'][1, 2:] = False
    changed = replay(p, ix, held, ix['dates'][1], ix['dates'][3], **args)
    assert not any(t['side'] == 'buy' and t['code'] == ix['codes'][1] and t['date'] >= ix['dates'][2] for t in changed['trades'])


@pytest.mark.parametrize('interval', [0, -1, 1.5])
def test_invalid_interval_rejected(interval):
    p, ix = synthetic_panel()
    with pytest.raises(ValueError, match='interval'):
        snapshot_scores(p['close'], p['eligible'], ix['dates'], ix['dates'][1], ix['dates'][-1], interval)

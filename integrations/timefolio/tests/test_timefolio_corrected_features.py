from copy import deepcopy

import numpy as np
import pytest

from test_timefolio_heatmap import synthetic_panel
from quant.timefolio_heatmap_corrected_features import derive_features


def setup():
    p, _ = synthetic_panel(days=60, names=3)
    p['price_basis'] = np.ones_like(p['close'])
    p['krx_volume'] = p['v'].copy()
    p['traded_value'] = p['v'] * p['close']
    return p, np.ones_like(p['close'])


def test_split_correction_removes_artificial_loss_and_false_exclusion():
    p, coverage = setup(); p['close'][0, 30:] /= 2
    old = derive_features(p, p['split'], coverage)
    split = p['split'].copy(); split[0, 30] = 2
    new = derive_features(p, split, coverage)
    assert old['r1'][0, 30] == -.5 and new['r1'][0, 30] == 0
    assert not old['eligible'][0, 30:50].any()
    assert new['eligible'][0, 30:50].all()
    for key in old:
        np.testing.assert_array_equal(old[key][:, :30], new[key][:, :30])


def test_peer_ranks_and_context_are_recomputed_from_corrected_values():
    p, coverage = setup(); p['sector'][:] = 1
    p['close'][0, 30:] /= 2; p['close'][1, 30:] *= 1.1
    p['close'][2, 30:] *= .98
    split = p['split'].copy(); split[0, 30] = 2
    old = derive_features(p, p['split'], coverage); new = derive_features(p, split, coverage)
    assert new['sector_r5'][2, 30] == pytest.approx(.08 / 3)
    assert old['sector_r5'][2, 30] == pytest.approx(.08 / 2)
    assert new['market_r5'][2, 30] == pytest.approx(.08 / 3)
    assert old['rank_r5'][2, 30] != new['rank_r5'][2, 30]


def test_future_prices_actions_and_coverage_do_not_change_past_inputs():
    p, coverage = setup(); original = deepcopy(p)
    before = derive_features(p, p['split'], coverage)
    changed = deepcopy(p); changed['close'][:, 40:] *= 1.5
    changed['split'][0, 45] = 2; altered_coverage = coverage.copy(); altered_coverage[:, 50:] = 0
    after = derive_features(changed, changed['split'], altered_coverage)
    for key in before: np.testing.assert_array_equal(before[key][:, :40], after[key][:, :40])
    for key in original: np.testing.assert_array_equal(p[key], original[key])


@pytest.mark.parametrize('value', [0., -1., np.nan])
def test_invalid_action_ratio_is_rejected(value):
    p, coverage = setup(); split = p['split'].copy(); split[0, 30] = value
    with pytest.raises(ValueError, match='Positive finite'):
        derive_features(p, split, coverage)

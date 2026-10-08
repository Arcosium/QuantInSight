import numpy as np
import pytest
import torch

from quant.timefolio_heatmap_walkforward import continuous_targets
from quant.timefolio_heatmap_rank_training import pairwise_loss


def panel(returns):
    returns = np.asarray(returns, float); n = len(returns)
    close = np.full((n, 4), 100.); close[:, 2] *= 1 + returns
    return dict(close=close, exec_price=np.full_like(close, 100.), factor=np.ones_like(close),
                sector=np.repeat([10, 20], n // 2)), np.arange(n), np.zeros(n, int)


def test_absolute_target_preserves_between_sector_winners():
    p, ci, di = panel([.10, .11, .12, .13, -.10, -.09, -.08, -.07])
    absolute = continuous_targets(p, ci, di, 2, 'absolute')
    sector = continuous_targets(p, ci, di, 2, 'sector')
    np.testing.assert_allclose(absolute, (np.array([.10, .11, .12, .13, -.10, -.09, -.08, -.07])-.004)/.1)
    assert absolute[0] > absolute[7] and sector[0] < sector[7]
    p['sector'][:] = 99
    np.testing.assert_array_equal(continuous_targets(p, ci, di, 2, 'absolute'), absolute)


def test_absolute_label_respects_share_factor_entry_horizon_and_missing_price():
    p, ci, di = panel([0., 0., 0., 0.])
    p['close'][0, 2] = 55.; p['factor'][0, 2:] = 2.
    p['exec_price'][1, 1] = np.nan
    y = continuous_targets(p, ci, di, 2, 'absolute')
    assert y[0] == pytest.approx(.96)
    assert np.isnan(y[1])
    before = y.copy(); p['close'][:, 3] = 1e8
    np.testing.assert_array_equal(continuous_targets(p, ci, di, 2, 'absolute'), before)
    assert np.isnan(continuous_targets(p, ci, np.full(4, 3), 2, 'absolute')).all()


def test_clipped_absolute_tail_is_a_tie_not_an_invented_order():
    p, ci, di = panel([.40, .80, -.60, -.90])
    y = continuous_targets(p, ci, di, 2, 'absolute')
    np.testing.assert_array_equal(y, [3., 3., -3., -3.])
    score = torch.tensor([.2, -.4, .7, -.1])
    expected = torch.nn.functional.softplus(-(score[:2, None]-score[None, 2:])).mean()
    torch.testing.assert_close(pairwise_loss(score, torch.from_numpy(y)), expected)

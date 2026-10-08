import numpy as np
import pytest

from quant.timefolio_heatmap_daily_history import DAILY_ROWS, FIELDS, encode_daily_history
from quant.timefolio_heatmap_features import make_image


def fixture():
    rng = np.random.default_rng(1907)
    shape = (3, 34)
    p = {key: rng.uniform(.01, .8, shape).astype(np.float32) for key in FIELDS}
    p['v'] *= 1e7; p['v20'] *= 1e7; p['adv5'] *= 1e11
    p['market_cap'] *= 1e13
    p['factor'] = np.ones(shape, np.float32)
    p['close'] = rng.uniform(10000, 15000, shape).astype(np.float32)
    bars = np.empty((3, 34, 26, 6), np.float32)
    bars[..., :4] = p['close'][:, :, None, None]
    bars[..., 4] = rng.uniform(100, 10000, bars.shape[:3])
    bars[..., 5] = 15
    codes, days = np.array([0, 1, 2]), np.array([25, 29, 33])
    raw = np.stack([make_image(p, bars, c, d) for c, d in zip(codes, days)])
    return p, bars, codes, days, raw


def test_snapshot_is_exact_and_originals_are_unchanged():
    p, _, ci, di, raw = fixture()
    copies = {key: value.copy() for key, value in p.items()}; before = raw.copy()
    actual = encode_daily_history(raw, p, ci, di, 'snapshot')
    assert np.array_equal(actual, raw) and not np.shares_memory(actual, raw)
    assert np.array_equal(raw, before)
    assert all(np.array_equal(p[key], copies[key]) for key in p)


@pytest.mark.parametrize('mode,order', [('history', [0, 1, 2, 3, 4]), ('history_reverse', [3, 2, 1, 0, 4])])
def test_daily_blocks_match_independent_original_scalar_encoder(mode, order):
    p, bars, ci, di, raw = fixture()
    actual = encode_daily_history(raw, p, ci, di, mode)
    other = np.setdiff1d(np.arange(32), DAILY_ROWS)
    assert np.array_equal(actual[:, other], raw[:, other])
    assert np.array_equal(actual[:, :, -13:], raw[:, :, -13:])
    for row, (code, day) in enumerate(zip(ci, di)):
        for block, offset in enumerate(order):
            original = make_image(p, bars, code, day - 4 + offset)
            assert np.array_equal(actual[row, DAILY_ROWS, block * 13:(block + 1) * 13], original[DAILY_ROWS, -13:])


def test_reverse_preserves_per_row_information_and_latest_value():
    p, _, ci, di, raw = fixture()
    a = encode_daily_history(raw, p, ci, di, 'history')
    b = encode_daily_history(raw, p, ci, di, 'history_reverse')
    assert not np.array_equal(a, b)
    assert np.array_equal(np.sort(a, axis=-1), np.sort(b, axis=-1))
    assert np.array_equal(a[..., -1], b[..., -1])


def test_future_panel_values_do_not_change_any_input():
    p, _, ci, di, raw = fixture()
    before = encode_daily_history(raw, p, ci, di, 'history')
    for code, day in zip(ci, di):
        for key in FIELDS: p[key][code, day + 1:] = np.nan
    assert np.array_equal(encode_daily_history(raw, p, ci, di, 'history'), before)


def test_missing_past_features_match_original_neutral_encoding():
    p, bars, ci, di, raw = fixture()
    for key in ('r20', 'vol20', 'rank_r5'): p[key][ci[0], di[0] - 4] = np.nan
    actual = encode_daily_history(raw, p, ci, di, 'history')
    original = make_image(p, bars, ci[0], di[0] - 4)
    assert np.array_equal(actual[0, DAILY_ROWS, :13], original[DAILY_ROWS, -13:])
    assert (actual[0, [18, 19, 20], :13] == 128).all()


def test_wrong_axes_mode_and_snapshot_are_rejected():
    p, _, ci, di, raw = fixture()
    with pytest.raises(ValueError): encode_daily_history(raw, p, ci, di, 'unknown')
    with pytest.raises(ValueError): encode_daily_history(raw.astype(float), p, ci, di, 'history')
    with pytest.raises(ValueError): encode_daily_history(raw, p, ci.astype(float), di, 'history')
    with pytest.raises(ValueError): encode_daily_history(raw, p, ci, np.array([3, 4, 5]), 'history')
    changed = raw.copy(); changed[0, 16, 0] ^= 1
    with pytest.raises(AssertionError): encode_daily_history(changed, p, ci, di, 'history')

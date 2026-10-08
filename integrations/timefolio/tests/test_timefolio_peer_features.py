import numpy as np
import pytest

from quant.timefolio_heatmap_peer_features import peer_images


def fixture():
    raw = np.full((6, 32, 4), 128, np.uint8)
    raw[:, 0] = np.array([20, 40, 60, 80, 100, 120])[:, None]
    raw[:, 14] = 255
    return raw, np.arange(6), np.zeros(6, int), np.array([1, 1, 1, 1, 2, 2])


def test_context_excludes_self_and_small_sector_uses_other_market_stocks():
    raw, ci, di, sectors = fixture()
    result, counts, local = peer_images(raw, ci, di, sectors)
    np.testing.assert_array_equal(result[:, 0, 0], [60, 53, 47, 40, 64, 60])
    np.testing.assert_array_equal(counts, [3, 3, 3, 3, 5, 5])
    np.testing.assert_array_equal(local, [True, True, True, True, False, False])
    changed = raw.copy(); changed[0] = 200
    after, _, _ = peer_images(changed, ci, di, sectors)
    np.testing.assert_array_equal(result[0], after[0])
    assert not np.array_equal(result[1], after[1])


def test_missing_prints_are_not_prices_and_empty_slots_are_neutral():
    raw, ci, di, sectors = fixture()
    raw[1:, 14, 0] = 128; raw[1:, 0, 0] = 255; raw[1:, 28, 0] = 0
    result, _, _ = peer_images(raw, ci, di, sectors)
    assert result[0, 0, 0] == result[0, 28, 0] == 128
    assert result[1, 0, 0] == 20
    # A daily feature is still known when an individual intraday print is absent.
    raw[1:4, 16, 0] = [40, 60, 80]
    result, _, _ = peer_images(raw, ci, di, sectors)
    assert result[0, 16, 0] == 60


def test_future_samples_cannot_change_past_context_or_membership():
    raw, ci, di, sectors = fixture(); before = peer_images(raw, ci, di, sectors)
    extended = peer_images(np.concatenate([raw, 255 - raw]), np.tile(ci, 2),
                           np.r_[di, di + 1], sectors)
    for a, b in zip(before, extended): np.testing.assert_array_equal(a, b[:6])


def test_stock_order_and_sector_label_names_have_no_effect():
    raw, ci, di, sectors = fixture(); before = peer_images(raw, ci, di, sectors)
    perm = np.array([5, 2, 0, 4, 1, 3])
    after = peer_images(raw[perm], ci[perm], di[perm], np.array([99, 99, 99, 99, 7, 7]))
    for a, b in zip(before, after): np.testing.assert_array_equal(a[perm], b)


def test_single_image_has_no_peer_and_no_own_image_copy():
    raw, ci, di, sectors = fixture()
    result, counts, used = peer_images(raw[:1], ci[:1], di[:1], sectors)
    assert np.all(result == 128) and counts[0] == 0 and not used[0]


def test_duplicate_stock_date_is_rejected_instead_of_double_weighted():
    raw, ci, di, sectors = fixture(); ci[1] = ci[0]
    with pytest.raises(ValueError, match='Unique'):
        peer_images(raw, ci, di, sectors)

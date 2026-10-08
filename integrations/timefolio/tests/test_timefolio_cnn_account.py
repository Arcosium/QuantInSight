import numpy as np
import pytest

from quant.timefolio_cnn_account import (
    align_execution, anchored_cap, evaluate_daily, statistics, turnover_windows,
)


def test_adjusted_execution_preserves_notional_and_avoids_mixed_price_units():
    price, volume = align_execution([100., 500., 0.], [500., 500., 500.],
                                    [101., 505., 100.], [1000., 200., 100.])
    np.testing.assert_allclose(price[:2], [505., 505.])
    np.testing.assert_allclose(volume[:2], [200., 200.])
    np.testing.assert_allclose((price*volume)[:2], [101000., 101000.])
    assert np.isnan(price[2]) and np.isnan(volume[2])


def test_cap_proxy_respects_publication_and_cancels_adjustment_multiplier():
    dates = np.array(['20220101', '20220102', '20220103', '20220104'])
    raw = np.array([100., 110., 120., 130.]); adj = raw*5
    cap, anchor = anchored_cap(dates, raw, adj, 1e8, '20220102')
    scaled, _ = anchored_cap(dates, raw, adj*3, 1e8, '20220102')
    assert anchor == 1 and np.isnan(cap[0])
    np.testing.assert_allclose(cap[1:], raw[1:]*1e8)
    np.testing.assert_array_equal(cap, scaled)


def test_cash_fold_has_undefined_sharpe_and_cannot_pass_thresholds():
    assert statistics(np.zeros(30))['sharpe'] is None
    rows = [dict(date=f'202401{d:02}', nav=1e9) for d in range(1, 25)]
    result = evaluate_daily(rows)
    assert not result['numerical_record_threshold'] and not result['numerical_stop_threshold']


def test_monthly_returns_keep_prior_month_closing_nav():
    rows = [dict(date='20240130', nav=1.01e9), dict(date='20240131', nav=1.02e9),
            dict(date='20240201', nav=1.03e9), dict(date='20240202', nav=1.01e9)]
    result = evaluate_daily(rows)
    assert result['monthly_folds']['202402']['net_return'] == pytest.approx(1.01/1.02-1)
    assert result['pooled']['net_return'] == pytest.approx(.01)


def test_turnover_failure_count_resets_for_each_contest_window():
    rows = []
    for month in [1, 4]:
        for day in [1, 8, 15]:
            rows.append(dict(date=f'2024{month:02}{day:02}', nav=1e9, buy_value=0., sell_value=0.))
    result = turnover_windows(rows)
    assert len(result) == 2
    assert all(r['low_turnover_weeks'] == 3 and not r['four_violation_screen_failed'] for r in result)

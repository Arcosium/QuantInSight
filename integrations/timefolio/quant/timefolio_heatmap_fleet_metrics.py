"""Continuous monthly out-of-sample Sharpe, using the same 252-day convention."""
import numpy as np
from quant.timefolio_heatmap_paper_benchmark import nav_metrics

EXPECTED_MONTHS = [f'2026{month:02d}' for month in range(1, 10)]


def monthly_target(daily):
    dates = [row['date'] for row in daily]
    assert dates == sorted(set(dates)) and dates[-1] <= '20260923'
    values = np.asarray([row['nav'] for row in daily], float)
    months = sorted({date[:6] for date in dates})
    assert months == EXPECTED_MONTHS
    pooled = nav_metrics(values)
    folds = []
    for month in months:
        ids = [i for i, date in enumerate(dates) if date.startswith(month)]
        assert len(ids) >= 2
        previous_nav = values[ids[0] - 1] if ids[0] else 1e9
        metrics = nav_metrics(values[ids], initial=float(previous_nav))
        folds.append(dict(month=month, first_date=dates[ids[0]], last_date=dates[ids[-1]], **metrics))
    pooled_pass = pooled['sharpe_defined'] and pooled['sharpe'] > 3.
    folds_pass = all(r['sharpe_defined'] and r['sharpe'] > 1. for r in folds)
    return dict(pooled=pooled, folds=folds, minimum_fold_sharpe=min(r['sharpe'] for r in folds),
        pooled_sharpe_above_three=bool(pooled_pass), all_nine_folds_above_one=bool(folds_pass),
        practical_metrics_passed=bool(pooled_pass and folds_pass),
        undefined_folds=[r['month'] for r in folds if not r['sharpe_defined']],
        annual_observations=252, first_fold_initial_nav=1e9,
        later_folds_use_previous_close=True, independent_confirmation=False)

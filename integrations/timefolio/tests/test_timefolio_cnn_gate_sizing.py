"""Account-level checks for joint market-exposure and stock-weight schedules."""
import json
import numpy as np
import pandas as pd

from quant.timefolio_cnn_core_retention_replay import replay
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks


def fixture():
    dates = pd.bdate_range('2024-01-02', periods=18).strftime('%Y%m%d').tolist()
    codes = [str(600000 + i) for i in range(10)]
    shape = (len(codes), len(dates))
    prices = np.full(shape, 10000.)
    panel = dict(o=prices.copy(), close=prices.copy(), eligible=np.ones(shape, bool),
                 sector=np.arange(10), sector_cap=np.full(shape, .2),
                 market_cap=np.full(shape, 2e12), split=np.ones(shape), listed_shares=np.ones(shape),
                 exec_price=prices.copy(), exec_volume=np.full(shape, 1e8),
                 exec_count=np.full(shape, 30), exec_high=prices.copy(), exec_low=prices.copy())
    scores = np.broadcast_to(np.arange(10., 0., -1)[:, None], shape).copy()
    return panel, dict(dates=dates, codes=codes), scores


def run(panel, index, scores, gross, weights=None):
    return replay(panel, index, scores, index['dates'][1], index['dates'][-1],
                  gross_schedule=gross, weight_schedule=weights, gross=.8, weight=.08,
                  top_n=12, max_orders=10, rank_buffer=20, rebalance=5,
                  rebalance_band=.005, return_trades=True, separate_activity_retention=True,
                  activity_policy=dict(target_turnover=.055, intervene_after_low_weeks=2, last_sessions=2))


def assert_fills(panel, index, result, gross):
    audit = audit_fills(panel, index, result)
    extra = additional_checks(panel, index, result, gross, max_orders=10)
    assert not audit['post_buy_limit_violations'] and not extra['additional_errors']
    assert audit['maximum_nav_reconstruction_error_krw'] < .01
    json.dumps(result, allow_nan=False)


def test_low_exposure_can_preserve_ten_names_instead_of_three():
    panel, index, scores = fixture()
    gross = np.full(len(index['dates']), .2)
    fixed = run(panel, index, scores, gross)
    scaled = run(panel, index, scores, gross, np.full(len(gross), .02))
    assert fixed['daily'][0]['holdings'] == 3
    assert scaled['daily'][0]['holdings'] == 10
    for result in [fixed, scaled]:
        assert_fills(panel, index, result, gross)
        assert .19 < result['daily'][0]['gross'] <= .2


def test_full_exposure_schedule_reproduces_the_existing_account():
    panel, index, scores = fixture()
    gross = np.full(len(index['dates']), .8)
    old = run(panel, index, scores, gross)
    scaled = run(panel, index, scores, gross, np.full(len(gross), .08))
    for key in ['daily', 'trades', 'plans', 'core_retention_decisions', 'activity_plans', 'activity_actual_weeks']:
        assert old[key] == scaled[key]


def test_exposure_and_weights_are_lagged_and_repair_fills_obey_limits():
    panel, index, scores = fixture()
    gross = np.full(len(index['dates']), .8)
    reduced = gross.copy(); reduced[6:12] = .2
    baseline = run(panel, index, scores, gross, .08 * (gross / .8))
    candidate = run(panel, index, scores, reduced, .08 * (reduced / .8))
    cutoff = index['dates'][6]
    for key in ['daily', 'trades', 'plans']:
        assert [r for r in baseline[key] if r['date'] <= cutoff] == [r for r in candidate[key] if r['date'] <= cutoff]
    assert candidate['daily'][6]['date'] == index['dates'][7]
    # Sale costs reduce NAV: a 20% pre-cost target may close slightly above 20%.
    # Exact NAV and buy-time dynamic ceilings are independently checked below.
    assert .19 < candidate['daily'][6]['gross'] < .21
    assert candidate['daily'][6]['holdings'] == 10
    assert any(t['side'] == 'sell' and t['date'] == index['dates'][7] for t in candidate['trades'])
    assert_fills(panel, index, candidate, reduced)

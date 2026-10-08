import copy

import numpy as np
import pandas as pd
import pytest

from quant.impact_alpha_features import FEATURE_COLUMNS, build_day_frame


def market():
    times = np.arange(0, 100001, 100, dtype=np.int64)
    books = np.zeros((len(times), 2, 5, 2))
    for level in range(5):
        books[:, 0, level, 0] = 99.99-level*.01
        books[:, 1, level, 0] = 100.01+level*.01
        books[:, :, level, 1] = 20
    books[times >= 61100, 1, 0, 1] = 8
    books[times >= 61500, 1, 0, 1] = 12
    trades = pd.DataFrame([(55500, 1, 'Sell', .3, 99.99),
                           (61000, 1, 'Buy', 12., 100.01),
                           (61200, 1, 'Sell', 2., 99.99)],
                          columns=['ts', 'connection', 'side', 'quantity', 'price'])
    return {'times':times, 'books':books, 'epochs':np.ones(len(times), np.int64), 'trades':trades}


def build(data, **kwargs):
    return build_day_frame(data, '2026-09-01', 120000, horizons=(30,), **kwargs)


def only_features(frame):
    return frame[['day', 'event_ms', 'decision_ms', 'feature_max_book_ms',
                  'feature_max_trade_ms', *FEATURE_COLUMNS]]


def test_future_perturbation_and_truncation_preserve_features_and_rows():
    data = market()
    original, _ = build(data)
    changed = copy.deepcopy(data)
    changed['books'][changed['times'] > 62000, :, :, 0] += 10
    changed['books'][changed['times'] > 62000, :, :, 1] += 999
    changed['trades'] = pd.concat([changed['trades'], pd.DataFrame([
        (62300, 1, 'Buy', 50., 110.01)], columns=changed['trades'].columns)], ignore_index=True)
    perturbed, _ = build(changed)
    pd.testing.assert_frame_equal(only_features(original), only_features(perturbed))
    assert original.iloc[0].gross_bps_30 != perturbed.iloc[0].gross_bps_30
    clipped = {**data, **{key:data[key][data['times'] <= 62000] for key in ('times', 'books', 'epochs')}}
    clipped['trades'] = data['trades'][data['trades'].ts <= 62000]
    past, _ = build(clipped)
    pd.testing.assert_frame_equal(only_features(original), only_features(past))
    assert len(past) == 1 and not past.iloc[0].entry_valid
    assert not past.iloc[0].exit_valid_30 and np.isnan(past.iloc[0].gross_bps_30)


def test_exact_refill_impact_and_past_flow_definitions():
    data = market()
    frame, audit = build(data)
    row = frame.iloc[0]
    assert audit['actual_orders'] == 0
    assert row.f_log_refill_ratio_1s == pytest.approx(np.log1p(4/12))
    assert row.f_log_refill_ratio_100ms == 0
    assert row.f_refill_latency_seconds == .5 and row.f_refill_observed == 1
    assert row.f_consumption_fraction == .6
    assert row.f_anchor_depletion_100ms == .6 and row.f_anchor_depletion_1s == .4
    assert row.f_opposing_flow_share_1s == pytest.approx(2/14)
    assert row.f_log_activity_10s == pytest.approx(np.log1p(.3))
    assert row.f_buy_fraction_10s == 0
    assert row.f_return_60s_bps == 0 and row.f_volatility_60s_bps == 0
    assert row.feature_max_book_ms <= row.decision_ms
    assert row.feature_max_trade_ms <= row.decision_ms
    assert set(frame.filter(regex='^f_')) == set(FEATURE_COLUMNS)


def test_no_refill_uses_explicit_censor_flag():
    data = market()
    data['books'][:, 1, 0, 1] = 20
    row = build(data)[0].iloc[0]
    assert row.f_refill_latency_seconds == 1.1
    assert row.f_refill_observed == 0 and row.f_log_refill_ratio_1s == 0


@pytest.mark.parametrize('condition,reason', [
    ('short', 'missing_60s_history'), ('epoch', 'history_epoch_boundary'),
    ('stale', 'stale_history'), ('invalid', 'known_invalid_history'),
])
def test_incomplete_or_unreliable_past_history_is_rejected(condition, reason):
    data = market()
    keep = np.ones(len(data['times']), dtype=bool)
    if condition == 'short':
        keep = data['times'] >= 2000
    elif condition == 'epoch':
        data['epochs'][data['times'] >= 40000] = 2
    elif condition == 'stale':
        keep = (data['times'] < 20000) | (data['times'] >= 23000)
    else:
        data['invalid_intervals'] = [{'start_ms':35000, 'end_ms':35500}]
    for key in ('times', 'books', 'epochs'):
        data[key] = data[key][keep]
    frame, audit = build(data)
    assert frame.empty and audit['feature_excluded'][reason] == 1


def test_entry_after_observation_and_latency_walks_both_sides_without_mutation():
    data = market()
    at_entry = data['times'] == 62100
    at_exit = data['times'] == 92100
    data['books'][at_entry, 1, :2] = [[100.01, .05], [100.02, .05]]
    data['books'][at_exit, 0, :2] = [[99.99, .05], [99.98, .05]]
    before = data['books'].copy()
    row = build(data)[0].iloc[0]
    assert row.entry_ms == 62100 and row.exit_ms_30 == 92100
    assert row.entry_price == pytest.approx(100.015)
    assert row.exit_price_30 == pytest.approx(99.985)
    assert row.gross_bps_30 == pytest.approx((99.985/100.015-1)*1e4)
    assert row.entry_valid and row.exit_valid_30
    np.testing.assert_array_equal(before, data['books'])


def test_missing_future_exit_retains_features_and_open_position_flag():
    data = market()
    baseline, _ = build(data)
    for key in ('times', 'books', 'epochs'):
        data[key] = data[key][data['times'] <= 70000] if key != 'times' else data[key]
    data['times'] = data['times'][data['times'] <= 70000]
    frame, audit = build(data)
    pd.testing.assert_frame_equal(only_features(baseline), only_features(frame))
    assert len(frame) == 1 and frame.iloc[0].entry_valid
    assert not frame.iloc[0].exit_valid_30 and np.isnan(frame.iloc[0].exit_price_30)
    assert audit['execution']['exit_30_missing_quote'] == 1


def test_delayed_exit_waits_at_most_five_seconds_and_records_actual_timestamp():
    data = market()
    keep = (data['times'] < 90000) | (data['times'] >= 94000)
    for key in ('times', 'books', 'epochs'):
        data[key] = data[key][keep]
    frame, audit = build(data)
    assert frame.iloc[0].exit_ms_30 == 94000
    assert frame.iloc[0].exit_quote_ms_30 == 94000
    assert audit['execution']['exit_30_delayed_quote'] == 1


def test_insufficient_entry_and_exit_depth_remain_explicitly_invalid():
    data = market()
    data['books'][data['times'] == 62100, 1, :, 1] = .001
    frame, audit = build(data)
    assert len(frame) == 1 and not frame.iloc[0].entry_valid
    assert audit['execution']['entry_insufficient_depth'] == 1
    data = market()
    data['books'][data['times'] == 92100, 0, :, 1] = .001
    frame, audit = build(data)
    assert len(frame) == 1 and frame.iloc[0].entry_valid and not frame.iloc[0].exit_valid_30
    assert audit['execution']['exit_30_insufficient_depth'] == 1


def test_max_horizon_reservation_uses_known_schedule_not_last_observation():
    data = market()
    # Decision 62000 + 300000 + 5500 is beyond this externally fixed schedule.
    frame, audit = build_day_frame(data, '2026-09-01', 367499, horizons=(30, 300))
    assert frame.empty and audit['extractor']['excluded']['scheduled_end_buffer'] == 1
    frame, audit = build_day_frame(data, '2026-09-01', 367500, horizons=(30, 300))
    assert len(frame) == 1 and frame.iloc[0].exit_valid_30 and not frame.iloc[0].exit_valid_300
    assert audit['reserved_after_decision_ms'] == 305500


@pytest.mark.parametrize('kwargs', [dict(horizons=()), dict(horizons=(1, 1)),
                                   dict(horizons=(1.5,)), dict(quantity=0),
                                   dict(threshold=0), dict(latency=501)])
def test_bad_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        build_day_frame(market(), '2026-09-01', 120000, **kwargs)

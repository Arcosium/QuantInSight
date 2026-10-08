import copy

import numpy as np
import pandas as pd
import pytest

from quant.impact_alpha_state_features import (
    CONTEXT_COLUMNS, EVENT_FEATURE_COLUMNS, MICRO_COLUMNS,
    add_state_features, build_minute_quotes,
)

DAY = '2026-09-01'
START = int(pd.Timestamp(DAY, tz='UTC').timestamp()*1000)


def market(minutes=300):
    offset = np.arange(0, minutes*60000+1, 1000, dtype=np.int64)
    times = START+offset
    books = np.zeros((len(times), 2, 5, 2))
    trend = offset/60000*.001
    for level in range(5):
        books[:, 0, level, 0] = 99.99-level*.01+trend
        books[:, 1, level, 0] = 100.01+level*.01+trend
        books[:, :, level, 1] = .05
    return {'times':times, 'books':books, 'epochs':np.ones(len(times), np.int64)}


def event(minute, sign=1, refill=.2, absorption=0, notional=1000):
    row = dict(decision_ms=START+minute*60000, f_sign=sign,
               f_log_notional=np.log1p(notional), f_dominance=1.,
               f_log_refill_ratio_1s=np.log1p(refill), f_absorption=absorption,
               f_signed_impact_100ms_bps=1., f_signed_impact_1s_bps=.5,
               f_obi5=.2, f_log_bid_depth5=np.log1p(10),
               f_log_ask_depth5=np.log1p(20), f_spread_bps=2.)
    row['feature_max_book_ms'] = row['decision_ms']-100
    row['feature_max_trade_ms'] = row['decision_ms']-200
    return row


def test_full_minute_grid_uses_exact_asof_vwap_and_retains_invalid_rows():
    data = market(2)
    quotes = build_minute_quotes(data, DAY)
    assert len(quotes) == 1440 and quotes.decision_ms.diff().dropna().eq(60000).all()
    row = quotes.iloc[1]
    assert row.quote_valid and row.quote_age_ms == 0 and row.interval_valid
    assert row.buy_vwap == pytest.approx(100.016)
    assert row.sell_vwap == pytest.approx(99.986)
    assert row.buy_valid and row.sell_valid and row.quantity == .1
    assert not quotes.iloc[3].quote_valid and np.isnan(quotes.iloc[3].mid)
    assert not quotes.iloc[3].buy_valid
    assert not quotes.iloc[0].interval_valid


def test_minute_quote_age_boundary_and_no_future_asof():
    data = market(2)
    # Exactly one second old is allowed, even when the data end at that quote.
    cutoff = START+59000
    for key in ('times', 'books', 'epochs'):
        data[key] = data[key][data['times'] <= cutoff] if key != 'times' else data[key]
    data['times'] = data['times'][data['times'] <= cutoff]
    quotes = build_minute_quotes(data, DAY)
    assert quotes.iloc[1].quote_valid and quotes.iloc[1].quote_age_ms == 1000
    assert quotes.iloc[1].quote_ms <= quotes.iloc[1].decision_ms
    data = market(2)
    keep = (data['times'] < START+59000) | (data['times'] > START+60000)
    for key in ('times', 'books', 'epochs'):
        data[key] = data[key][keep]
    quotes = build_minute_quotes(data, DAY)
    assert not quotes.iloc[1].quote_valid and quotes.iloc[1].quote_age_ms == 2000


def test_known_invalid_quote_and_hidden_invalid_span_are_retained_and_masked():
    data = market(60)
    data['invalid_intervals'] = [
        {'start_ms':START+10*60000, 'end_ms':START+10*60000+100},
        {'start_ms':START+30*60000+100, 'end_ms':START+30*60000+200},
    ]
    quotes = build_minute_quotes(data, DAY)
    assert not quotes.iloc[10].quote_valid
    assert quotes.iloc[30].quote_valid and quotes.iloc[31].quote_valid
    assert not quotes.iloc[31].interval_valid
    states = add_state_features(quotes, pd.DataFrame())
    assert not states.iloc[31].state_valid and not states.iloc[45].state_valid
    assert states.iloc[46].state_valid


def test_epoch_change_between_valid_minute_quotes_resets_history():
    data = market(60)
    data['epochs'][data['times'] >= START+30*60000+1000] = 2
    quotes = build_minute_quotes(data, DAY)
    assert quotes.iloc[30].quote_valid and quotes.iloc[31].quote_valid
    assert not quotes.iloc[31].interval_valid
    states = add_state_features(quotes, pd.DataFrame())
    assert states.iloc[46].state_valid and states.iloc[46].continuous_minutes == 15


def test_future_quote_and_event_changes_leave_all_past_state_unchanged():
    data = market()
    events = pd.DataFrame([event(20), event(100, -1, 2, 1)])
    cutoff = START+80*60000
    original_quotes = build_minute_quotes(data, DAY)
    original = add_state_features(original_quotes, events)
    changed = copy.deepcopy(data)
    changed['books'][changed['times'] > cutoff, :, :, 0] += 50
    changed['books'][changed['times'] > cutoff, :, :, 1] += 100
    changed_events = events.copy()
    changed_events.loc[changed_events.decision_ms > cutoff, 'f_sign'] = 1
    changed_events['gross_bps_1800'] = 999999  # Not in the event input projection.
    changed_events['entry_valid'] = False
    future_changed = add_state_features(build_minute_quotes(changed, DAY), changed_events)
    past = original.decision_ms <= cutoff
    pd.testing.assert_frame_equal(original.loc[past], future_changed.loc[past])
    clipped = {key:value[data['times'] <= cutoff] for key, value in data.items()}
    truncated = add_state_features(build_minute_quotes(clipped, DAY), events[events.decision_ms <= cutoff])
    pd.testing.assert_frame_equal(original.loc[past], truncated.loc[past])


def test_exact_window_boundaries_pressure_formulas_and_staleness():
    quotes = build_minute_quotes(market(40), DAY)
    events = pd.DataFrame([event(15), event(20), event(20.5, -1, 2, 1, 3000)])
    states = add_state_features(quotes, events)
    row = states.iloc[20]
    assert row.event_count_5m == 1  # Lower window edge excluded; future event excluded.
    assert row.event_age_ms == 0
    assert row.f_low_refill_pressure_5m == pytest.approx(.8)
    assert row.f_impact_persistence_5m == pytest.approx(.5)
    row = states.iloc[21]
    assert row.event_count_5m == 2 and row.event_age_ms == 30000
    assert row.f_sign_balance_5m == 0
    assert row.f_net_notional_fraction_5m == pytest.approx(-.5)
    assert row.f_low_refill_pressure_5m == pytest.approx(.2)
    assert row.f_absorption_reversal_5m == pytest.approx(.75)
    assert row.f_impact_persistence_5m == pytest.approx(-.25)
    assert row.f_mean_obi5_5m == pytest.approx(.2)
    assert row.f_mean_log_depth5_5m == pytest.approx(np.log1p(30))


def test_missing_minute_prevents_bridge_but_long_unavailable_history_is_explicit():
    quotes = build_minute_quotes(market(), DAY)
    quotes = quotes.drop(index=100).reset_index(drop=True)
    states = add_state_features(quotes, pd.DataFrame([event(90), event(105)]))
    states = states.set_index('decision_ms')
    row = states.loc[START+101*60000]
    assert row.continuous_minutes == 0 and not row.state_valid
    assert row.event_count_120m == 0  # Previous-segment events are excluded.
    row = states.loc[START+116*60000]
    assert row.state_valid and row.f_history_15m == 1
    assert row.f_history_60m == 0 and row.f_history_240m == 0
    assert row.f_return_60m_bps == 0 and row.f_return_240m_bps == 0
    assert row.f_volatility_60m_bps == 0
    assert row.f_observed_window_fraction_120m == pytest.approx(15/120)
    assert np.isfinite(states.loc[:, CONTEXT_COLUMNS+MICRO_COLUMNS]).all().all()


def test_price_returns_and_volatility_have_correct_historical_window():
    quotes = build_minute_quotes(market(), DAY)
    states = add_state_features(quotes, pd.DataFrame())
    row = states.iloc[240]
    assert row.state_valid and row.f_history_240m == 1
    assert row.f_return_240m_bps == pytest.approx((100.24/100-1)*1e4)
    mids = quotes.mid.to_numpy()[180:241]
    expected = np.std((mids[1:]/mids[:-1]-1)*1e4, ddof=0)
    assert row.f_volatility_60m_bps == pytest.approx(expected)
    assert row.event_count_120m == 0 and np.isnan(row.event_age_ms)
    assert len(CONTEXT_COLUMNS) == 9 and len(MICRO_COLUMNS) == 34
    assert not set(CONTEXT_COLUMNS) & set(MICRO_COLUMNS)
    assert not any('gross' in key or 'exit' in key for key in CONTEXT_COLUMNS+MICRO_COLUMNS)


def test_insufficient_depth_preserves_quote_and_separate_execution_validity():
    data = market(1)
    data['books'][:, :, :, 1] = .001
    quotes = build_minute_quotes(data, DAY)
    assert quotes.iloc[1].quote_valid
    assert not quotes.iloc[1].buy_valid and not quotes.iloc[1].sell_valid
    assert np.isnan(quotes.iloc[1].buy_vwap)


def test_event_future_feature_timestamp_is_rejected():
    quotes = build_minute_quotes(market(30), DAY)
    row = event(20)
    row['feature_max_book_ms'] = row['decision_ms']+1
    with pytest.raises(ValueError, match='later than'):
        add_state_features(quotes, pd.DataFrame([row]))


def test_empty_inputs_preserve_schema_and_finite_feature_contract():
    quotes = build_minute_quotes(market(1), DAY).iloc[:0]
    states = add_state_features(quotes, pd.DataFrame(columns=EVENT_FEATURE_COLUMNS))
    assert states.empty and set(CONTEXT_COLUMNS+MICRO_COLUMNS).issubset(states)

"""Causal minute quotes and sparse burst-pressure state for offline research.

There are no orders, clients, credentials, future labels, or network calls.
``build_minute_quotes`` accepts the dictionary returned by ``load_day`` and
retains all 1440 UTC minute boundaries, including invalid observations. Quotes
are strictly as-of, at most 1000 ms old; no backward/forward filling is used.
VWAP walks up to 50 visible levels for 0.1 ETH, independently on each side.
``interval_valid`` also rules out epoch changes or known bad intervals between
the previous and current minute. An interval with no book update is not itself
declared corrupt: cache epoch/quality rules define known continuity breaks.

``add_state_features`` reads only the explicit event-feature whitelist below.
An event becomes observable at its decision timestamp, not its trade timestamp.
Trailing windows are (minute-window, minute], restricted to the current valid
quote segment. They summarize *selected, observed large bursts*, not all trade
activity. Missing events do not imply a quiet market. The existing event source
omits its scheduled final-30-minute label buffer and unsuitable past history;
this source-selection limitation remains even though its predictor values are
past-only. Counts, elapsed window coverage and age of the last observed event
are therefore exposed. Event age is diagnostic metadata, not a model input.

For event sign s, notional n, dominance d, observed 1 s refill r, absorption a,
and signed impacts i100 and i1, the fixed pressure formulas are:
  low refill = s*d*max(1-r, 0)
  absorption reversal = -s*a
  impact persistence = s*clip(i1/i100, 0, 1) if i100>0, else 0.
Their window values are notional-weighted means. Sign balance is count-weighted;
net-notional fraction is sum(s*n)/sum(n). OBI, log1p(total depth5), and spread
means are count-weighted. Empty-window summaries are zero with explicit count.

Price returns use 15/60/240 exact consecutive valid minutes. Volatility is the
population standard deviation of the last 60 one-minute simple returns (bp).
They never bridge a missing minute, invalid quote/span, epoch reset, or day
boundary. Unavailable long-context features are zero with explicit history
flags; ``state_valid`` requires only current validity and 15 continuous minutes,
so losing one old quote does not require discarding another four hours of data.
Only CONTEXT_COLUMNS and MICRO_COLUMNS are predictors. All such values are
finite; quote metadata and execution values may be NaN where invalid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.impact_alpha_backtest import fill_price, known_invalid

MINUTE_MS = 60000
WINDOWS = (5, 30, 120)
QUANTITY = .1
CONTEXT_COLUMNS = (
    'f_return_15m_bps', 'f_return_60m_bps', 'f_return_240m_bps',
    'f_volatility_60m_bps', 'f_history_15m', 'f_history_60m', 'f_history_240m',
    'f_utc_time_sin', 'f_utc_time_cos',
)
_WINDOW_NAMES = (
    'log_event_count', 'sign_balance', 'net_notional_fraction',
    'low_refill_pressure', 'absorption_reversal', 'impact_persistence',
    'mean_obi5', 'mean_log_depth5', 'mean_spread_bps', 'observed_window_fraction',
)
MICRO_COLUMNS = ('f_log_bid_depth5', 'f_log_ask_depth5', 'f_obi5', 'f_spread_bps') + tuple(
    f'f_{name}_{window}m' for window in WINDOWS for name in _WINDOW_NAMES
)
EVENT_FEATURE_COLUMNS = (
    'f_sign', 'f_log_notional', 'f_dominance', 'f_log_refill_ratio_1s',
    'f_absorption', 'f_signed_impact_100ms_bps', 'f_signed_impact_1s_bps',
    'f_obi5', 'f_log_bid_depth5', 'f_log_ask_depth5', 'f_spread_bps',
)
_QUOTE_COLUMNS = (
    'day', 'decision_ms', 'quote_ms', 'quote_age_ms', 'epoch', 'quote_valid',
    'interval_valid', 'bid', 'ask', 'mid', 'bid_depth5', 'ask_depth5', 'obi5',
    'spread_bps', 'buy_vwap', 'sell_vwap', 'buy_valid', 'sell_valid', 'quantity',
)


def build_minute_quotes(data, day):
    """Return one full UTC day of causal observations; never omit invalid rows."""
    start = pd.Timestamp(day)
    start = start.tz_localize('UTC') if start.tzinfo is None else start.tz_convert('UTC')
    if start != start.normalize():
        raise ValueError('day must identify a UTC midnight')
    start_ms = int(start.timestamp()*1000)
    targets = start_ms + np.arange(1440, dtype=np.int64)*MINUTE_MS
    times, books, epochs = data['times'], data['books'], data['epochs']
    if len(times) != len(books) or len(times) != len(epochs) or np.any(np.diff(times) < 0):
        raise ValueError('Matching ordered book/time/epoch arrays required')
    if books.ndim != 4 or books.shape[1] != 2 or books.shape[2] < 5 or books.shape[3] != 2:
        raise ValueError('At least five price/quantity levels per book side required')
    indices = np.searchsorted(times, targets, side='right')-1
    rows = []
    for minute, (target, index) in enumerate(zip(targets, indices, strict=True)):
        row = {name:np.nan for name in _QUOTE_COLUMNS}
        row.update(day=start.strftime('%Y-%m-%d'), decision_ms=int(target),
                   quote_valid=False, interval_valid=False, buy_valid=False,
                   sell_valid=False, quantity=QUANTITY)
        if index < 0:
            rows.append(row)
            continue
        quote_ms = int(times[index])
        row.update(quote_ms=quote_ms, quote_age_ms=int(target-quote_ms), epoch=int(epochs[index]))
        if (target-quote_ms > 1000 or quote_ms < start_ms
                or known_invalid(data, quote_ms, int(target))):
            rows.append(row)
            continue
        book = np.asarray(books[index, :, :50])
        if (not np.isfinite(book).all() or (book <= 0).any()
                or np.any(np.diff(book[0, :, 0]) >= 0)
                or np.any(np.diff(book[1, :, 0]) <= 0)
                or book[0, 0, 0] >= book[1, 0, 0]):
            rows.append(row)
            continue
        bid, ask = map(float, book[:, 0, 0])
        bid_depth, ask_depth = map(float, book[:, :5, 1].sum(axis=1))
        mid = (bid+ask)/2
        buy = fill_price(book[1], 'buy', QUANTITY)
        sell = fill_price(book[0], 'sell', QUANTITY)
        row.update(quote_valid=True, bid=bid, ask=ask, mid=mid,
                   bid_depth5=bid_depth, ask_depth5=ask_depth,
                   obi5=(bid_depth-ask_depth)/(bid_depth+ask_depth),
                   spread_bps=(ask-bid)/mid*1e4,
                   buy_vwap=float(buy) if buy is not None else np.nan,
                   sell_vwap=float(sell) if sell is not None else np.nan,
                   buy_valid=buy is not None, sell_valid=sell is not None)
        if minute and rows[-1]['quote_valid']:
            previous = int(indices[minute-1])
            row['interval_valid'] = bool(
                np.all(epochs[previous:index+1] == epochs[index])
                and not known_invalid(data, int(target-MINUTE_MS), int(target)))
        rows.append(row)
    return pd.DataFrame(rows, columns=_QUOTE_COLUMNS)


def _event_arrays(events):
    """Project first: execution flags, prices and outcome columns are ignored."""
    if events.empty:
        return np.empty(0, dtype=np.int64), np.zeros((0, 10))
    required = ['decision_ms', *EVENT_FEATURE_COLUMNS]
    missing = set(required)-set(events.columns)
    if missing:
        raise ValueError(f'Missing event features: {sorted(missing)}')
    selected = events.loc[:, required].sort_values('decision_ms', kind='stable')
    numeric = selected.to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError('Event predictors and decision timestamps must be finite')
    decision = selected.decision_ms.to_numpy(np.int64)
    if np.any(selected.decision_ms.to_numpy() != decision):
        raise ValueError('Event timestamps must be integer milliseconds')
    for column in ('feature_max_book_ms', 'feature_max_trade_ms'):
        if column in events:
            if np.any(events[column].to_numpy() > events.decision_ms.to_numpy()):
                raise ValueError('Event feature timestamp is later than its decision')
    sign = selected.f_sign.to_numpy(float)
    dominance = selected.f_dominance.to_numpy(float)
    absorption = selected.f_absorption.to_numpy(float)
    if (not np.isin(sign, (-1, 1)).all() or np.any((dominance < 0) | (dominance > 1))
            or not np.isin(absorption, (0, 1)).all()):
        raise ValueError('Invalid event sign, dominance or absorption flag')
    with np.errstate(over='ignore', invalid='ignore'):
        notional = np.expm1(selected.f_log_notional.to_numpy(float))
        refill = np.expm1(selected.f_log_refill_ratio_1s.to_numpy(float))
        depth = np.expm1(selected.f_log_bid_depth5.to_numpy(float)) + np.expm1(selected.f_log_ask_depth5.to_numpy(float))
    if (not np.isfinite([notional, refill, depth]).all()
            or np.any(notional <= 0) or np.any(refill < 0) or np.any(depth <= 0)):
        raise ValueError('Invalid event notional, refill or depth')
    initial = selected.f_signed_impact_100ms_bps.to_numpy(float)
    remaining = selected.f_signed_impact_1s_bps.to_numpy(float)
    persistence = np.clip(np.divide(remaining, initial, out=np.zeros(len(initial)), where=initial > 0), 0, 1)
    values = np.column_stack((
        np.ones(len(selected)), sign, notional, sign*notional,
        notional*sign*dominance*np.maximum(1-refill, 0), -notional*sign*absorption,
        notional*sign*persistence, selected.f_obi5.to_numpy(float), np.log1p(depth),
        selected.f_spread_bps.to_numpy(float),
    ))
    return decision, values


def add_state_features(quotes, events):
    """Return quote rows plus finite predictors, preserving missing observations.

    Call after concatenating daily quote frames. Missing row timestamps are not
    inserted or filled; instead, the next row starts a new continuity segment.
    The main backtest must retain the full minute grid for exact delayed fills.
    ``state_valid`` concerns predictor history, not entry/exit label availability.
    """
    missing = set(_QUOTE_COLUMNS)-set(quotes.columns)
    if missing:
        raise ValueError(f'Missing minute quote columns: {sorted(missing)}')
    frame = quotes.copy().sort_values('decision_ms', kind='stable').reset_index(drop=True)
    timestamps = frame.decision_ms.to_numpy(np.int64)
    if frame.decision_ms.isna().any() or np.any(frame.decision_ms.to_numpy() != timestamps):
        raise ValueError('Minute timestamps must be finite integer milliseconds')
    if np.any(np.diff(timestamps) <= 0) or np.any(timestamps % MINUTE_MS != 0):
        raise ValueError('Unique UTC minute timestamps required')
    valid = frame.quote_valid.to_numpy(bool)
    mids = frame.mid.to_numpy(float)
    if np.any(valid & (~np.isfinite(mids) | (mids <= 0))):
        raise ValueError('Valid quotes must have positive finite mid prices')
    previous_valid = np.r_[False, valid[:-1]] if len(frame) else np.empty(0, bool)
    consecutive = np.r_[False, np.diff(timestamps) == MINUTE_MS] if len(frame) else np.empty(0, bool)
    boundary = ~valid | ~previous_valid | ~consecutive | ~frame.interval_valid.to_numpy(bool)
    segments = np.cumsum(boundary)
    positions = np.arange(len(frame))
    starts = np.maximum.accumulate(np.where(boundary, positions, 0)) if len(frame) else positions
    continuous = np.where(valid, positions-starts, 0)
    frame['continuous_minutes'] = continuous
    frame['feature_max_book_ms'] = frame.quote_ms
    for window in (15, 60, 240):
        available = valid & (continuous >= window)
        value = np.zeros(len(frame))
        chosen = np.flatnonzero(available)
        value[chosen] = (mids[chosen]/mids[chosen-window]-1)*1e4
        frame[f'f_return_{window}m_bps'] = value
        frame[f'f_history_{window}m'] = available.astype(float)
    minute_returns = np.zeros(len(frame))
    joined = np.flatnonzero(valid & ~boundary)
    minute_returns[joined] = (mids[joined]/mids[joined-1]-1)*1e4
    # Segment start has no return. It becomes irrelevant once 60 full intervals
    # exist; unavailable outputs below are explicitly zero with the history flag.
    volatility = pd.Series(minute_returns).groupby(segments).rolling(60, min_periods=60).std(ddof=0)
    volatility = volatility.reset_index(level=0, drop=True).sort_index().to_numpy() if len(frame) else np.empty(0)
    frame['f_volatility_60m_bps'] = np.where(continuous >= 60, volatility, 0)
    phase = (timestamps % 86400000)/86400000*2*np.pi
    frame['f_utc_time_sin'], frame['f_utc_time_cos'] = np.sin(phase), np.cos(phase)
    for source, target in (('bid_depth5', 'f_log_bid_depth5'), ('ask_depth5', 'f_log_ask_depth5')):
        values = frame[source].to_numpy(float)
        if np.any(valid & (~np.isfinite(values) | (values <= 0))):
            raise ValueError('Valid quotes must have positive finite depth')
        frame[target] = np.log1p(np.where(valid, values, 0))
    for source, target in (('obi5', 'f_obi5'), ('spread_bps', 'f_spread_bps')):
        values = frame[source].to_numpy(float)
        if np.any(valid & ~np.isfinite(values)):
            raise ValueError('Valid quotes must have finite OBI and spread')
        frame[target] = np.where(valid, values, 0)

    event_times, event_values = _event_arrays(events)
    prefix = np.vstack((np.zeros((1, 10)), np.cumsum(event_values, axis=0)))
    right = np.searchsorted(event_times, timestamps, side='right')
    segment_times = timestamps[starts] if len(frame) else timestamps
    last = right-1
    seen = valid & (last >= 0)
    last_time = np.full(len(frame), np.nan)
    if len(event_times):
        seen &= event_times[np.maximum(last, 0)] >= segment_times
        last_time[seen] = event_times[last[seen]]
    frame['feature_max_event_ms'] = last_time
    frame['event_age_ms'] = timestamps-last_time
    for window in WINDOWS:
        cutoff = timestamps-window*MINUTE_MS
        left = np.searchsorted(event_times, cutoff, side='right')
        # Include events at segment start; exclude the ordinary open left edge.
        left = np.maximum(left, np.searchsorted(event_times, segment_times, side='left'))
        left = np.where(valid, left, right)
        sums = prefix[right]-prefix[left]
        count, weights = sums[:, 0], sums[:, 2]
        mean = lambda numerator, denominator: np.divide(numerator, denominator, out=np.zeros(len(frame)), where=denominator > 0)
        values = (
            np.log1p(count), mean(sums[:, 1], count), mean(sums[:, 3], weights),
            mean(sums[:, 4], weights), mean(sums[:, 5], weights), mean(sums[:, 6], weights),
            mean(sums[:, 7], count), mean(sums[:, 8], count), mean(sums[:, 9], count),
            np.minimum(continuous/window, 1),
        )
        for name, value in zip(_WINDOW_NAMES, values, strict=True):
            frame[f'f_{name}_{window}m'] = value
        frame[f'event_count_{window}m'] = count.astype(np.int64)
    frame['state_valid'] = valid & (continuous >= 15)
    if not np.isfinite(frame.loc[:, CONTEXT_COLUMNS+MICRO_COLUMNS].to_numpy()).all():
        raise ValueError('Non-finite state predictor')
    return frame

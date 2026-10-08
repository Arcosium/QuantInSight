"""Causal feature/label tables for offline rolling-window spot-alpha research.

No exchange client, credentials, service mutation, or order submission is used.
``data`` is the immutable dictionary returned by ``impact_backfill.load_day``.
The event universe is the existing 100 ms, >=80% one-direction quantity burst,
with fixed notional threshold and causal 11 s spacing. The existing extractor
observes refill for 1 s; no descriptive, future-coverage-filtered event CSV is
used. Scheduled capture/day end is supplied externally, never inferred from the
last available quote. Reserve max hold + 500 ms latency + 5 s exit grace.

Every ``f_`` column is available at ``decision_ms = event_ms + 1000``:
* event sign, log1p total 100 ms notional/quantity, and quantity dominance;
* log1p gross positive anchor-quantity changes / initial anchor execution at
  100 ms and 1 s. Only book changes at/after the first anchor trade count;
* first positive anchor refill delay from that trade, censored to 1.1 s with
  a separate observed flag; initial execution / pre-event anchor quantity;
* net anchor depletion at 100 ms and 1 s, max(1 - quantity/pre_quantity, 0);
* event-direction mid returns from pre-event mid to 100 ms and 1 s (bp),
  and giveback = signed 100 ms impact minus signed 1 s impact;
* opposite-direction quantity share and log1p total quantity over [t,t+1 s];
* log1p pre-event/current top-five bid/ask quantities, OBI=(bid-ask)/(bid+ask),
  and spread/mid in bp, plus the existing 1 s absorption-candidate flag;
* signed-in-clock-time (not event-direction) 10/60 s past mid returns and
  population standard deviation of 1 s simple mid returns in bp;
* log1p traded quantity and buy-quantity fraction in [t-10 s,t), with 0.5 for
  no activity, plus sine/cosine of UTC decision time of day.
Past price sampling is t-60 s,...,t-1 s,t-1 ms. Every as-of quote and the entire
60 s history through decision must have age <=1 s, stay in one epoch, and avoid
known invalid intervals. This deliberately stricter continuity rule prevents
sparse or stale history from masquerading as low volatility.

Labels use a long-only spot trade: actual ask-depth VWAP at decision+latency,
then bid-depth VWAP at entry+hold. These future quotes never select features or
remove rows. Missing entry/exit quotes or insufficient depth retain the row,
with false validity and NaN price/return. Exit quotes may wait at most 5 s, as
specified by the existing replay helper. Gross labels include spread/depth
cost, but no fees; the rolling runner must apply its explicitly stated fees.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from quant.impact_alpha_backtest import extract_signals, fill_price, known_invalid, quote_at


FEATURE_COLUMNS = (
    'f_sign', 'f_log_notional', 'f_log_quantity', 'f_dominance',
    'f_log_refill_ratio_100ms', 'f_log_refill_ratio_1s',
    'f_refill_latency_seconds', 'f_refill_observed', 'f_consumption_fraction',
    'f_anchor_depletion_100ms', 'f_anchor_depletion_1s',
    'f_signed_impact_100ms_bps', 'f_signed_impact_1s_bps', 'f_signed_giveback_bps',
    'f_opposing_flow_share_1s', 'f_log_flow_quantity_1s',
    'f_log_pre_bid_depth5', 'f_log_pre_ask_depth5',
    'f_log_bid_depth5', 'f_log_ask_depth5', 'f_pre_obi5', 'f_obi5',
    'f_pre_spread_bps', 'f_spread_bps', 'f_absorption',
    'f_return_10s_bps', 'f_return_60s_bps',
    'f_volatility_10s_bps', 'f_volatility_60s_bps',
    'f_log_activity_10s', 'f_buy_fraction_10s', 'f_utc_time_sin', 'f_utc_time_cos',
)
BASE_COLUMNS = (
    'day', 'event_ms', 'decision_ms', 'feature_max_book_ms', 'feature_max_trade_ms',
    'entry_ms', 'entry_price', 'entry_quote_ms', 'quantity', 'entry_valid',
)


def _label_columns(horizons):
    return tuple(f'{name}_{h}' for h in horizons for name in
                 ('exit_ms', 'exit_price', 'exit_quote_ms', 'exit_valid', 'gross_bps'))


def _book_summary(book):
    bid, ask = np.asarray(book[:, :5, 1]).sum(axis=1)
    best_bid, best_ask = book[0, 0, 0], book[1, 0, 0]
    mid = (best_bid + best_ask) / 2
    return float(bid), float(ask), float((bid-ask)/(bid+ask)), float((best_ask-best_bid)/mid*1e4)


def build_day_frame(data, day, known_end_ms, threshold=1000.0,
                    horizons=(30, 300, 1800), quantity=.1, latency=100):
    """Return all eligible past-observed rows and explicit future-label audits.

    ``known_end_ms`` must be the planned capture/day end, including when testing
    truncated data. Only ``FEATURE_COLUMNS`` may be supplied to a model. The
    metadata and validity columns contain execution/outcome information and
    must never be used as predictors or to redefine the event universe.
    """
    horizons = tuple(horizons)
    if not horizons or any(isinstance(h, bool) or int(h) != h or h <= 0 for h in horizons):
        raise ValueError('Horizons must be positive integer seconds')
    horizons = tuple(int(h) for h in horizons)
    if len(set(horizons)) != len(horizons):
        raise ValueError('Duplicate horizons')
    if not np.isfinite(quantity) or quantity <= 0:
        raise ValueError('Quantity must be positive and finite')
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Threshold must be positive and finite')
    if isinstance(latency, bool) or int(latency) != latency or not 0 <= latency <= 500:
        raise ValueError('Latency must be integer milliseconds between 0 and 500')
    times, books, epochs = data['times'], data['books'], data['epochs']
    if len(times) != len(books) or len(times) != len(epochs):
        raise ValueError('Matching book/time/epoch arrays required')

    # Existing extractor reserves 30 s + 500 ms + 5 s. Adjust only the known
    # schedule, never the observed last timestamp, for the requested max hold.
    adjusted_end = int(known_end_ms) - (max(horizons)-30)*1000
    signals, extraction_audit = extract_signals(data, threshold, adjusted_end)
    trades = data['trades'].sort_values('ts', kind='stable')
    tt = trades.ts.to_numpy(np.int64)
    tq = trades.quantity.to_numpy(float)
    tp = trades.price.to_numpy(float)
    sg = np.where(trades.side == 'Buy', 1, -1)
    rows, excluded, execution = [], Counter(), Counter()

    for signal in signals.itertuples():
        t, decision, sign = int(signal.event_ms), int(signal.decision_ms), int(signal.sign)
        pre = int(np.searchsorted(times, t, side='left')-1)
        end = int(np.searchsorted(times, decision, side='right')-1)
        targets = np.r_[t + np.arange(-60, 0, dtype=np.int64)*1000, t-1]
        history = np.searchsorted(times, targets, side='right')-1
        if history[0] < 0:
            excluded['missing_60s_history'] += 1
            continue
        start = int(history[0])
        if np.any(epochs[start:end+1] != epochs[pre]):
            excluded['history_epoch_boundary'] += 1
            continue
        if known_invalid(data, int(times[start]), decision):
            excluded['known_invalid_history'] += 1
            continue
        if (np.any(targets-times[history] > 1000) or decision-times[end] > 1000
                or np.any(np.diff(times[start:end+1]) > 1000)):
            excluded['stale_history'] += 1
            continue

        a = int(np.searchsorted(tt, t, side='left'))
        stop100 = int(np.searchsorted(tt, t+100, side='left'))
        stop = int(np.searchsorted(tt, decision, side='right'))
        before10 = int(np.searchsorted(tt, t-10000, side='left'))
        side = 1 if sign == 1 else 0
        price = float(books[pre, side, 0, 0])
        anchor = ((sg[a:stop100] == sign)
                  & np.isclose(tp[a:stop100], price, rtol=0, atol=1e-8))
        consumed = float(tq[a:stop100][anchor].sum())
        first_trade = int(tt[a:stop100][anchor][0])
        levels = np.asarray(books[pre:end+1, side])
        anchor_quantity = np.where(np.isclose(levels[:, :, 0], price, rtol=0, atol=1e-8),
                                   levels[:, :, 1], 0).sum(axis=1)
        path_times = times[pre:end+1]
        change_times = path_times[1:]
        increases = np.maximum(np.diff(anchor_quantity), 0)
        visible = change_times >= first_trade
        refills = visible & (increases > 0)
        added100 = float(increases[visible & (change_times <= t+100)].sum())
        ratio1 = float(signal.refill_ratio)
        first_refill = ((int(change_times[np.flatnonzero(refills)[0]])-first_trade)/1000
                        if refills.any() else 1.1)
        at100 = int(np.searchsorted(path_times, t+100, side='right')-1)
        pre_mid = float((books[pre, 0, 0, 0]+books[pre, 1, 0, 0])/2)
        hundred = pre+at100
        mid100 = float((books[hundred, 0, 0, 0]+books[hundred, 1, 0, 0])/2)
        mid1 = float((books[end, 0, 0, 0]+books[end, 1, 0, 0])/2)
        impact100, impact1 = sign*(mid100/pre_mid-1)*1e4, sign*(mid1/pre_mid-1)*1e4
        pre_bid, pre_ask, pre_obi, pre_spread = _book_summary(books[pre])
        bid, ask, obi, spread = _book_summary(books[end])
        mids = (books[history, 0, 0, 0]+books[history, 1, 0, 0])/2
        returns = (mids[1:]/mids[:-1]-1)*1e4
        volume1 = float(tq[a:stop].sum())
        volume10 = float(tq[before10:a].sum())
        bucket_qty = float(tq[a:stop100].sum())
        dominance = max(float(tq[a:stop100][sg[a:stop100] == 1].sum()),
                        float(tq[a:stop100][sg[a:stop100] == -1].sum()))/bucket_qty
        phase = (decision % 86400000)/86400000*2*np.pi
        features = dict(zip(FEATURE_COLUMNS, (
            sign, np.log1p((tq[a:stop100]*tp[a:stop100]).sum()), np.log1p(bucket_qty), dominance,
            np.log1p(added100/consumed), np.log1p(ratio1),
            first_refill, float(refills.any()), consumed/anchor_quantity[0],
            max(1-anchor_quantity[at100]/anchor_quantity[0], 0),
            max(1-anchor_quantity[-1]/anchor_quantity[0], 0),
            impact100, impact1, impact100-impact1,
            float(tq[a:stop][sg[a:stop] == -sign].sum())/volume1, np.log1p(volume1),
            np.log1p(pre_bid), np.log1p(pre_ask), np.log1p(bid), np.log1p(ask),
            pre_obi, obi, pre_spread, spread, float(signal.absorption),
            (mids[-1]/mids[-11]-1)*1e4, (mids[-1]/mids[0]-1)*1e4,
            np.std(returns[-10:], ddof=0), np.std(returns, ddof=0),
            np.log1p(volume10),
            float(tq[before10:a][sg[before10:a] == 1].sum())/volume10 if volume10 else .5,
            np.sin(phase), np.cos(phase),
        ), strict=True))
        if not np.isfinite(list(features.values())).all():
            excluded['nonfinite_features'] += 1
            continue

        entry_target = decision+int(latency)
        row = dict(day=str(day), event_ms=t, decision_ms=decision,
                   feature_max_book_ms=int(times[end]),
                   feature_max_trade_ms=int(tt[stop-1]) if stop else -1,
                   entry_ms=entry_target, entry_price=np.nan, entry_quote_ms=np.nan,
                   quantity=float(quantity), entry_valid=False, **features)
        entry = quote_at(data, entry_target)
        if entry is None:
            execution['entry_missing_quote'] += 1
        else:
            i, entry_ms = entry
            entry_price = fill_price(books[i, 1], 'buy', quantity)
            row.update(entry_ms=int(entry_ms), entry_quote_ms=int(times[i]))
            if entry_price is None:
                execution['entry_insufficient_depth'] += 1
            else:
                row.update(entry_price=float(entry_price), entry_valid=True)
        for h in horizons:
            row.update({f'exit_ms_{h}':np.nan, f'exit_price_{h}':np.nan,
                        f'exit_quote_ms_{h}':np.nan, f'exit_valid_{h}':False,
                        f'gross_bps_{h}':np.nan})
            if not row['entry_valid']:
                execution[f'exit_{h}_entry_invalid'] += 1
                continue
            due = row['entry_ms']+h*1000
            exit_quote = quote_at(data, due, allow_wait=True)
            if exit_quote is None:
                execution[f'exit_{h}_missing_quote'] += 1
                continue
            j, exit_ms = exit_quote
            exit_price = fill_price(books[j, 0], 'sell', quantity)
            row.update({f'exit_ms_{h}':int(exit_ms), f'exit_quote_ms_{h}':int(times[j])})
            if exit_price is None:
                execution[f'exit_{h}_insufficient_depth'] += 1
                continue
            row.update({f'exit_price_{h}':float(exit_price), f'exit_valid_{h}':True,
                        f'gross_bps_{h}':(exit_price/row['entry_price']-1)*1e4})
            execution[f'exit_{h}_valid'] += 1
            if exit_ms > due:
                execution[f'exit_{h}_delayed_quote'] += 1
        rows.append(row)

    frame = pd.DataFrame(rows, columns=BASE_COLUMNS+FEATURE_COLUMNS+_label_columns(horizons))
    audit = {
        'day':str(day), 'threshold_usdt':float(threshold), 'quantity':float(quantity),
        'latency_ms':int(latency), 'horizons_seconds':list(horizons),
        'known_end_ms':int(known_end_ms), 'reserved_after_decision_ms':max(horizons)*1000+5500,
        'extractor':extraction_audit, 'feature_excluded':dict(excluded),
        'feature_rows':len(frame), 'execution':dict(execution),
        'entry_valid':sum(bool(row['entry_valid']) for row in rows),
        'feature_count':len(FEATURE_COLUMNS), 'actual_orders':0,
    }
    return frame, audit

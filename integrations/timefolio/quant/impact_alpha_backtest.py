"""Offline, causal spot-alpha replay. No credentials, exchange clients or orders.

Refill is observed for one second BEFORE entry. Outcomes and future coverage
never choose signals. Existing descriptive-event CSVs are deliberately not used:
their 10/300-second eligibility conditions would leak future availability.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant.impact_backfill import load_day
from quant.impact_longitudinal import bursts, clean_json
from quant.market_impact import consume

RULES = ('flow_baseline', 'refill_only', 'breakout', 'absorption')
HOLDS = (5, 10, 30)
CAPITAL = 100000
TRADE_COLUMNS = ['day', 'rule', 'hold_seconds', 'latency_ms', 'event_ms',
                 'decision_ms', 'entry_ms', 'exit_ms', 'entry_price',
                 'exit_price', 'quantity', 'refill_ratio', 'depth',
                 'entry_quote_ms', 'exit_quote_ms', 'exit_delay_ms']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def known_invalid(data, start, end):
    return any(x['start_ms'] <= end and
               (x['end_ms'] is None or x['end_ms'] >= start)
               for x in data.get('invalid_intervals', []))


def extract_signals(data, threshold, known_end_ms):
    """Use only observations timestamped <= bucket start + 1000 ms."""
    times, books, epochs = data['times'], data['books'], data['epochs']
    if len(times) < 2 or np.any(np.diff(times) < 0):
        raise ValueError('Ordered books required')
    trades = data['trades'].sort_values('ts', kind='stable')
    tt = trades.ts.to_numpy(np.int64)
    tp = trades.price.to_numpy(float)
    tq = trades.quantity.to_numpy(float)
    sg = np.where(trades.side == 'Buy', 1, -1)
    candidates = bursts(trades)
    candidates = candidates[candidates.notional >= threshold]
    rows, excluded = [], Counter()
    last_event = -10**18
    for b in candidates.itertuples():
        t = int(b.bucket); decision = t+1000
        # End of day/capture is known before observing any price outcomes.
        # Reserve 30 s holding + max 500 ms latency + 5 s exit-data grace.
        if decision+35500 > known_end_ms:
            excluded['scheduled_end_buffer'] += 1; continue
        pre = int(np.searchsorted(times, t, side='left')-1)
        end = int(np.searchsorted(times, decision, side='right')-1)
        if pre < 0 or end < pre or decision > times[-1]:
            excluded['missing_past'] += 1; continue
        if t-times[pre] > 1000 or decision-times[end] > 1000:
            excluded['stale_signal_quote'] += 1; continue
        if epochs[pre] != epochs[end] or known_invalid(data, times[pre], decision):
            excluded['past_gap'] += 1; continue
        if t-last_event < 11000:
            excluded['causal_event_spacing'] += 1; continue
        last_event = t
        sign = int(b.sign); side = 1 if sign == 1 else 0
        price = float(books[pre, side, 0, 0])
        a = np.searchsorted(tt, t, side='left')
        b100 = np.searchsorted(tt, t+100, side='left')
        selected = (sg[a:b100] == sign) & np.isclose(tp[a:b100], price, rtol=0, atol=1e-8)
        consumed = float(tq[a:b100][selected].sum())
        if consumed <= 0:
            excluded['no_anchor_execution'] += 1; continue
        levels = np.asarray(books[pre:end+1, side])
        covered = levels[:, -1, 0] >= price-1e-8 if side == 1 else levels[:, -1, 0] <= price+1e-8
        if not covered.all():
            excluded['past_anchor_outside_book'] += 1; continue
        quantities = np.where(np.isclose(levels[:, :, 0], price, rtol=0, atol=1e-8),
                              levels[:, :, 1], 0).sum(axis=1)
        first_trade = int(tt[a:b100][selected][0])
        visible = times[pre+1:end+1] >= first_trade
        added = float(np.maximum(np.diff(quantities), 0)[visible].sum())
        ratio = added/consumed
        broken = bool((levels[:, 0, 0] > price+1e-8).any()) if side == 1 else bool((levels[:, 0, 0] < price-1e-8).any())
        stop = np.searchsorted(tt, decision, side='right')
        same = (sg[a:stop] == sign) & np.isclose(tp[a:stop], price, rtol=0, atol=1e-8)
        executed = float(tq[a:stop][same].sum())
        rows.append({'event_ms':t, 'decision_ms':decision, 'sign':sign,
                     'refill_ratio':ratio, 'depth':float(books[pre, side, :5, 1].sum()),
                     'absorption':bool(ratio >= 1 and not broken and executed >= quantities[0]),
                     'feature_book_max_ms':int(times[end]),
                     'feature_trade_max_ms':int(tt[stop-1]) if stop else -1})
    columns = ['event_ms','decision_ms','sign','refill_ratio','depth','absorption',
               'feature_book_max_ms','feature_trade_max_ms']
    return pd.DataFrame(rows, columns=columns), {'large_bursts':len(candidates),
                                               'signals':len(rows), 'excluded':dict(excluded)}


def rule_mask(signals, rule, thin, thick):
    buy = signals.sign == 1
    if rule == 'flow_baseline': return buy
    if rule == 'refill_only': return buy & (signals.refill_ratio < 1)
    if rule == 'breakout': return buy & (signals.refill_ratio < 1) & (signals.depth <= thin)
    if rule == 'absorption': return (signals.sign == -1) & signals.absorption & (signals.depth >= thick)
    raise ValueError('Unknown rule')


def quote_at(data, target, allow_wait=False):
    times = data['times']
    i = int(np.searchsorted(times, target, side='right')-1)
    if i >= 0 and target <= times[-1] and target-times[i] <= 1000 and not known_invalid(data, target, target):
        return i, target
    if not allow_wait: return None
    i = int(np.searchsorted(times, target, side='left'))
    if i < len(times) and times[i]-target <= 5000 and not known_invalid(data, times[i], times[i]):
        return i, int(times[i])
    return None


def fill_price(levels, side, quantity):
    # Existing ArcTrade depth consumer; copy the snapshot so replay data is immutable.
    book = {'bids':[], 'asks':[]}
    book['asks' if side == 'buy' else 'bids'] = np.asarray(levels).tolist()
    fill = consume(book, side, quantity)
    return fill['vwap'] if fill['unfilled'] <= 1e-10 else None


def simulate(data, signals, rule, hold, thin, thick, day, latency=100, quantity=.1):
    selected = signals.loc[rule_mask(signals, rule, thin, thick)]
    rows, audit = [], Counter()
    free_at = -10**18
    for signal in selected.itertuples():
        audit['triggered'] += 1
        if signal.decision_ms < free_at:
            audit['position_already_open'] += 1; continue
        target = int(signal.decision_ms+latency)
        entry = quote_at(data, target)
        if entry is None:
            audit['entry_cancelled_stale'] += 1; continue
        i, entry_ms = entry
        entry_price = fill_price(data['books'][i, 1], 'buy', quantity)
        if entry_price is None:
            audit['entry_cancelled_depth'] += 1; continue
        due = entry_ms+hold*1000
        exit_quote = quote_at(data, due, allow_wait=True)
        if exit_quote is None:
            # Never silently remove a position after seeing an unavailable exit.
            raise ValueError(f'Unpriced open position: {day} {rule} {hold}s entry={entry_ms}')
        j, exit_ms = exit_quote
        exit_price = fill_price(data['books'][j, 0], 'sell', quantity)
        if exit_price is None:
            raise ValueError(f'Insufficient exit depth: {day} {rule} entry={entry_ms}')
        if exit_ms > due: audit['delayed_exit_data'] += 1
        rows.append({'day':day,'rule':rule,'hold_seconds':hold,'latency_ms':latency,
                     'event_ms':int(signal.event_ms),'decision_ms':int(signal.decision_ms),
                     'entry_ms':entry_ms,'exit_ms':exit_ms,'entry_price':entry_price,
                     'exit_price':exit_price,'quantity':quantity,
                     'refill_ratio':signal.refill_ratio,'depth':signal.depth,
                     'entry_quote_ms':int(data['times'][i]),'exit_quote_ms':int(data['times'][j]),
                     'exit_delay_ms':exit_ms-due})
        free_at = exit_ms
    return pd.DataFrame(rows, columns=TRADE_COLUMNS), dict(audit)


def process_day(job):
    cache, output, threshold, thin, thick = job
    data = load_day(cache)
    day = data['manifest']['day_utc']
    known_end = int((pd.Timestamp(day, tz='UTC')+pd.Timedelta(days=1)).timestamp()*1000)
    signals, meta = extract_signals(data, threshold, known_end)
    frames = []; audits = {}
    for rule in RULES:
        for hold in HOLDS:
            frame, audit = simulate(data, signals, rule, hold, thin, thick, day)
            frames.append(frame); audits[f'{rule}_{hold}'] = audit
    destination = Path(output)/'days'/day
    destination.mkdir(parents=True, exist_ok=False)
    signals.to_csv(destination/'signals.csv.gz', index=False)
    nonempty = [frame for frame in frames if not frame.empty]
    combined = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame(columns=TRADE_COLUMNS)
    combined.to_csv(destination/'trades.csv.gz', index=False)
    meta.update(day=day, audit=audits, cache_manifest_sha256=sha(Path(cache)/'manifest.json'))
    (destination/'audit.json').write_text(json.dumps(clean_json(meta),indent=2))
    return day, len(signals), sum(len(x) for x in frames)


def summarize(output, reference, live_input=None, cache_root=None):
    from functools import partial
    from quant.impact_alpha_stats import summarize_trades as summarize_base, choose_horizon, add_net_columns
    summarize_trades = partial(summarize_base, initial_capital=CAPITAL)
    output = Path(output)
    directories = sorted((output/'days').iterdir())
    trades = pd.concat([pd.read_csv(p/'trades.csv.gz') for p in directories],ignore_index=True)
    cutoff = reference['calibration_end']
    train_days = [p.name for p in directories if p.name <= cutoff]
    eval_days = [p.name for p in directories if p.name > cutoff]
    train = trades[trades.day <= cutoff]; evaluation = trades[trades.day > cutoff]
    result = {'training_days':train_days,'evaluation_days':eval_days,'rules':{},
              'actual_orders':0,'mode':'offline_historical_replay',
              'limitations':['Evaluation dates were previously inspected for the descriptive study; this is not a fresh untouched holdout.',
                             'Historical quote replay omits our persistent price impact and feed/network delays beyond the assumed latency.',
                             '0.1 ETH fixed size, long-only spot, independent 100000 USDT accounts for each strategy; no compounding.',
                             'Fee is a constant quote-currency equivalent assumption, not the account fee schedule or historical tier changes.']}
    thin, thick = reference['regime_thresholds']['depth']['limits']
    selected = {}
    training_by_rule = {rule:{hold:summarize_trades(train[(train.rule==rule)&(train.hold_seconds==hold)],train_days)
                             for hold in HOLDS} for rule in RULES}
    breakout_hold = choose_horizon(training_by_rule['breakout'])['horizon_seconds']
    for rule in RULES:
        training = training_by_rule[rule]
        selection = choose_horizon(training); hold = selection['horizon_seconds']; selected[rule] = hold
        if rule in ('flow_baseline','refill_only'):
            hold = breakout_hold; selected[rule] = hold
            selection = {'horizon_seconds':hold,'selection_status':'control_uses_breakout_horizon',
                         'training_n':training[hold]['n'],'training_mean_net_bps':training[hold]['mean_net_bps']}
        rows = evaluation[(evaluation.rule==rule)&(evaluation.hold_seconds==hold)].copy()
        summary = summarize_trades(rows,eval_days)
        # Check fixed capital affordability with actual sequential cash, including fees.
        checked = add_net_columns(rows)
        if len(rows):
            cash_before = CAPITAL+np.r_[0,checked.net_pnl.to_numpy()[:-1].cumsum()]
            needed = rows.entry_price.to_numpy()*rows.quantity.to_numpy()*1.001
            if np.any(cash_before < needed): raise ValueError('Fixed capital exhausted; sizing replay needed')
        checked.to_csv(output/f'{rule}_evaluation_trades.csv',index=False)
        result['rules'][rule] = {'training':training,'selection':selection,'evaluation':summary,
             'fees':{str(fee):summarize_trades(rows,eval_days,fee_bps=fee) for fee in [0,1,5,10]},
             'extra_slippage_1bp_each_side':summarize_trades(rows,eval_days,extra_slippage_bps=1)}
    # Persist train-only selections before any live-period evaluation.
    (output/'selected_rules.json').write_text(json.dumps(clean_json(selected),indent=2))
    if live_input:
        from quant.impact_live_study import read_completed_source
        data, manifest, provenance = read_completed_source(live_input)
        known_end = int(datetime.fromisoformat(manifest['ended_utc']).timestamp()*1000)
        live_signals, live_meta = extract_signals(data,reference['notional_threshold'],known_end)
        day = datetime.fromisoformat(manifest['started_utc']).strftime('%Y-%m-%d')
        live_signals.to_csv(output/'live_signals.csv',index=False)
        for rule, hold in selected.items():
            rows,audit = simulate(data,live_signals,rule,hold,thin,thick,day)
            add_net_columns(rows).to_csv(output/f'{rule}_live_trades.csv',index=False)
            result['rules'][rule]['live'] = summarize_trades(rows,[day])
            result['rules'][rule]['live_audit'] = audit
        result['live_source'] = {'raw_sha256':provenance['raw_sha256'],'started_utc':manifest['started_utc'],
                                 'ended_utc':manifest['ended_utc'],'extraction':live_meta}
    # Latency sensitivity changes executions, never thresholds or selected horizons.
    if cache_root:
        stress = {rule:[] for rule in RULES}
        for directory in directories:
            if directory.name <= cutoff: continue
            data = load_day(Path(cache_root)/f'{directory.name}-ETHUSDT-depth50-gap0-v1')
            signals = pd.read_csv(directory/'signals.csv.gz')
            for rule, hold in selected.items():
                rows,_ = simulate(data,signals,rule,hold,thin,thick,directory.name,latency=500)
                stress[rule].append(rows)
        for rule in RULES:
            rows = pd.concat(stress[rule],ignore_index=True)
            add_net_columns(rows).to_csv(output/f'{rule}_latency500_trades.csv',index=False)
            result['rules'][rule]['latency500'] = summarize_trades(rows,eval_days)
    (output/'results.json').write_text(json.dumps(clean_json(result),ensure_ascii=False,indent=2))
    print(json.dumps({'phase':'complete','output':str(output),'selection':selected}),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--live-input',type=Path)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--summarize-only',action='store_true')
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    if 'longitudinal' in reference: reference = reference['longitudinal']
    if args.summarize_only:
        summarize(args.output,reference,args.live_input,args.cache_root); return
    args.output.mkdir(parents=True,exist_ok=False)
    caches = sorted(p for p in args.cache_root.iterdir() if (p/'manifest.json').exists())
    expected = reference['dates']
    caches = [p for p in caches if p.name[:10] in expected]
    if len(caches) != len(expected): raise ValueError('Missing daily caches')
    thin,thick = reference['regime_thresholds']['depth']['limits']
    protocol = {'created_utc':datetime.now(timezone.utc).isoformat(),'rules':RULES,'hold_candidates':HOLDS,
        'selection':'Highest training mean net bp, n>=30; tie shorter; no eval tuning; flow/refill controls use breakout horizon',
        'calibration_end':reference['calibration_end'],'threshold':reference['notional_threshold'],
        'thin_depth':thin,'thick_depth':thick,'quantity_eth':.1,'fee_bps_each_side':10,
        'latency_ms':100,'refill_observation_ms':1000,'capital_usdt':CAPITAL,
        'common_pool':'Past-only, same-price refill observable; 11-second causal spacing',
        'execution':'Ask VWAP entry / bid VWAP exit, known 1s quote age maximum; no overlapping positions',
        'boundary':'No new entry in final 35.5 seconds after decision of known UTC day/capture end',
        'source_reference_sha256':sha(args.reference),'code_sha256':sha(__file__),'actual_orders':0}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    jobs = [(str(p),str(args.output),reference['notional_threshold'],thin,thick) for p in caches]
    with ProcessPoolExecutor(max_workers=max(1,min(args.workers,4))) as pool:
        for day,n,count in pool.map(process_day,jobs):
            print(json.dumps({'phase':'daily','day':day,'signals':n,'scenario_trades':count}),flush=True)
    summarize(args.output,reference,args.live_input,args.cache_root)


if __name__ == '__main__': main()

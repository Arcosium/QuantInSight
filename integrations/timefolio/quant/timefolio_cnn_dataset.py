"""Freeze completed daily shards into an explicitly exploratory CNN package.

No source databases are opened. Unverified metadata is disclosed, not invented.
Signal-time selection is independent of future label availability and returns.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def encode_windows(windows):
    """Vectorized equivalent of core.heatmap for previously checked N,T,5."""
    a = np.asarray(windows, dtype=np.float64)
    o, h, l, c, v = a.transpose(2, 0, 1)
    prices = np.log(a[:, :, :4].transpose(0, 2, 1) / c[:, :1, None]) / .25
    volume = np.log((v + 1) / (np.median(v, axis=1, keepdims=True) + 1)) / 3
    gap = np.zeros_like(c); gap[:, 1:] = np.log(o[:, 1:] / c[:, :-1]) / .1
    rows = np.concatenate([prices, volume[:, None], (np.log(c / o) / .1)[:, None],
        (np.log(h / l) / .1)[:, None], gap[:, None]], axis=1)
    return np.clip(rows, -1, 1).astype(np.float32)[:, None]


def read_rows(path):
    with gzip.open(path, 'rt', newline='') as f:
        return list(csv.DictReader(f))


def select_samples(panel, value, *, window=20, maximum=512, threshold=3e9):
    """Selection uses trailing observed OHLCV and trailing liquidity only."""
    d, c, _ = panel.shape
    valid = (np.isfinite(panel).all(2) & (panel[:, :, :4] > 0).all(2)
        & (panel[:, :, 4] >= 0)
        & (panel[:, :, 2] <= np.minimum(panel[:, :, 0], panel[:, :, 3]))
        & (panel[:, :, 1] >= np.maximum(panel[:, :, 0], panel[:, :, 3])))
    eligible = np.zeros((d, c), bool)
    eligible[window-1:] = np.lib.stride_tricks.sliding_window_view(valid, window, axis=0).all(-1)
    eligible[window-1:] &= np.lib.stride_tricks.sliding_window_view(panel[:, :, 4], window, axis=0).sum(-1) > 0
    adv = np.full((d, c), np.nan)
    adv[4:] = np.lib.stride_tricks.sliding_window_view(value, 5, axis=0).mean(-1)
    eligible &= np.isfinite(adv) & (adv > threshold)
    for day in range(d):
        ids = np.flatnonzero(eligible[day])
        if len(ids) > maximum:
            keep = ids[np.argsort(-adv[day, ids], kind='stable')[:maximum]]
            eligible[day] = False; eligible[day, keep] = True
        # Defined on signal-time membership, never on finite future returns.
        if eligible[day].sum() < 10:
            eligible[day] = False
    return eligible, adv


def register_folds(dates, *, window, horizon, first_month='202401'):
    """An early partial month can supply OOF warmup without reducing fit history."""
    width = 40+horizon+1
    earliest = window-1+126+horizon+1+width
    result=[]
    for month in sorted({str(d)[:6] for d in dates if str(d)[:6] >= first_month}):
        days=np.flatnonzero(np.array([str(d).startswith(month) for d in dates]))
        first,end=max(int(days[0]),earliest),int(days[-1])+1
        if first < end:
            result.append(dict(id=month,validation_start=first-width,test_start=first,test_end=end,
                gate_warmup_only=month<'202401'))
    return result


def build(source, output, *, window=20, horizon=5, maximum=512, price_reference=None, cohort=None, first_month='202401'):
    source, output = Path(source), Path(output)
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise ValueError('Refusing to overwrite a frozen dataset')
    if window < 5 or horizon < 1:
        raise ValueError('Invalid research window/horizon')
    partial.mkdir(parents=True)
    started = time.monotonic()
    plan = json.loads((source/'source_plan.json').read_text())
    plan_hash = sha(source/'source_plan.json')
    shards = source/'daily_candidates'
    frozen = []
    # Snapshot completed receipts once, so an active extractor cannot expand it.
    expected = {r['code']:r for r in json.loads(Path(cohort).read_text())['receipts']} if cohort else None
    for p in sorted(shards.glob('*.json')):
        proof = json.loads(p.read_text()); data = p.with_suffix('.csv.gz')
        if expected is not None and proof['code'] not in expected:
            continue
        if proof['plan_sha256'] != plan_hash or sha(data) != proof['sha256']:
            raise ValueError('Extraction receipt mismatch')
        if expected is not None and proof['sha256'] != expected[proof['code']]['sha256']:
            raise ValueError('Requested matched cohort changed')
        frozen.append(dict(code=proof['code'], file=data.name, sha256=proof['sha256'],
            days=proof['days'], receipt_sha256=sha(p)))
    if not frozen:
        raise ValueError('No completed candidate shards')
    if expected is not None and {r['code'] for r in frozen} != set(expected):
        raise ValueError('Matched cohort is incomplete')
    (partial/'cohort.json').write_text(json.dumps(dict(frozen_at=time.time(),
        source_plan_sha256=plan_hash, source=str(source), receipts=frozen), indent=2)+'\n')
    # Samsung's observed sessions provide the candidate calendar, including
    # special sessions that still require independent reference reconciliation.
    calendar = read_rows(shards/'005930.csv.gz')
    dates = np.array(sorted({r['date'] for r in calendar if plan['start'] <= r['date'] <= plan['cutoff']}), dtype='U8')
    if len(dates) < window or dates[-1] > '20260923':
        raise ValueError('Invalid candidate calendar or reserved cutoff')
    codes = np.array([p['code'] for p in frozen], dtype='U6')
    date_index = {d: i for i, d in enumerate(dates)}
    panel = np.full((len(dates), len(codes), 5), np.nan, dtype=np.float64)
    value = np.full(panel.shape[:2], np.nan)
    for k, spec in enumerate(frozen):
        for r in read_rows(shards/spec['file']):
            if r['date'] not in date_index:
                continue
            i = date_index[r['date']]
            panel[i, k] = [float(r[n]) if r[n] else np.nan for n in ['open','high','low','close','volume']]
            value[i, k] = float(r['typical_value_proxy']) if r['typical_value_proxy'] else np.nan
    raw_panel = panel.copy() if price_reference else None
    reference_hashes = {}
    if price_reference:
        reference_root = Path(price_reference)
        for k, code in enumerate(codes):
            path, receipt = reference_root/(str(code)+'.json'), reference_root/(str(code)+'.receipt.json')
            if not path.exists() or not receipt.exists():
                raise ValueError('Matched daily reference is incomplete: '+str(code))
            digest = sha(path)
            if digest != json.loads(receipt.read_text())['sha256']:
                raise ValueError('Daily-reference receipt mismatch')
            reference = json.loads(path.read_text())
            if reference['code'] != code or reference['start'] != plan['start'] or reference['end'] != plan['cutoff']:
                raise ValueError('Daily reference crossed the registered cohort/interval')
            reference_hashes[str(code)] = digest
            panel[:, k] = np.nan
            for row in reference['rows']:
                if row['date'] not in date_index:
                    raise ValueError('Daily reference has a session outside the candidate calendar')
                panel[date_index[row['date']], k] = [row[n] for n in ['open','high','low','close','volume']]
        # Retain original time-specific turnover; adjusted historical price levels
        # must not change liquidity admission after a later corporate action.
        (partial/'price_reference_hashes.json').write_text(json.dumps(reference_hashes,indent=2)+'\n')
    eligible, adv = select_samples(panel, value, window=window, maximum=maximum)
    count = int(eligible.sum())
    if not count:
        raise ValueError('No liquid complete trailing windows')
    images = np.lib.format.open_memmap(partial/'images.npy', mode='w+', dtype='float32', shape=(count, 1, 8, window))
    signal, keys = np.empty(count, np.int32), np.empty(count, np.int32)
    returns = np.full(count, np.nan, np.float32)
    offset = 0
    for k in range(len(codes)):
        days = np.flatnonzero(eligible[:, k]); n = len(days)
        if not n:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(panel[:, k], window, axis=0).transpose(0, 2, 1)
        images[offset:offset+n] = encode_windows(windows[days-window+1])
        signal[offset:offset+n] = days; keys[offset:offset+n] = k
        matured = days + horizon + 1 < len(dates)
        selected = days[matured]
        entry, close = panel[selected+1, k, 0], panel[selected+horizon+1, k, 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            y = close/entry - 1
        y[(entry <= 0) | (close <= 0) | ~np.isfinite(y)] = np.nan
        returns[offset + np.flatnonzero(matured)] = y
        offset += n
    images.flush(); del images
    arrays = dict(returns=returns, signal_index=signal, label_end_index=signal+horizon+1,
        security_key=keys, eligible=np.ones(count, bool), dates=dates,
        codes=codes, daily_ohlcv=panel, daily_value_proxy=value, adv5_proxy=adv)
    if raw_panel is not None:
        arrays['raw_daily_ohlcv'] = raw_panel
    for name, array in arrays.items():
        np.save(partial/(name+'.npy'), array, allow_pickle=False)
    if any(len(str(d)) != 8 for d in dates):
        raise ValueError('Noncompact session date')
    folds = register_folds(dates,window=window,horizon=horizon,first_month=first_month)
    limitations = [
        'Exploratory raw-price study; not a Timefolio-compliant performance certificate.',
        'Cohort consists of completed extraction shards at freeze time; code-order and current-universe selection bias.',
        'Historical delistings, market cap, GICS and trading restrictions are not yet complete.',
        'Corporate actions and cash dividends are not reconciled; targets are raw open-to-open price returns.',
        'Candidate minute aggregates have unverified NXT provenance, special-session and auction discrepancies.',
        'Liquidity uses a minute typical-price turnover proxy; exact daily turnover is being collected.',
        'No order-book queue, fees, tax, sector caps or minimum weekly turnover in these model targets.',
        'Previously explored historical periods are development data, not an untouched final holdout.',
    ]
    full_source_cohort = set(codes.tolist()) == set(plan['codes'])
    if full_source_cohort:
        limitations[1] = 'All codes in the frozen source ledger are included; the ledger is still not a certified point-in-time universe including every delisted security.'
    if price_reference:
        limitations[3] = 'NAVER historical OHLCV supplies image and return prices; adjustment and dividend conventions are not yet certified against the full action ledger.'
        limitations[4] = 'NAVER venue/adjustment conventions remain under cross-provider audit; raw minute aggregates are separately retained for execution research.'
    manifest = dict(version=1, purpose='exploratory' if price_reference else 'data_diagnostic',
        exploratory_ready=bool(price_reference), training_ready=False,
        contest_certified=False, readiness=dict(venue=False, session_calendar=False,
            corporate_actions=False, point_in_time_universe=False, feature_availability=True,
            label_construction=True), limitations=limitations, folds=folds,
        start=str(dates[0]), cutoff=str(dates[-1]), sessions=len(dates), cohort_codes=len(codes),
        source_codes=len(plan['codes']), full_source_cohort=full_source_cohort,
        rows=count, finite_labels=int(np.isfinite(returns).sum()),
        window=window, horizon=horizon, top_k=10, maximum_per_date=maximum,
        label='open[t+H+1] / open[t+1] - 1; no future-value filtering of prediction membership',
        decision_time='After all signal-date reference bars are available; execution starts the following session.',
        calendar_code='005930', dataset_builder_sha256=sha(__file__),
        price_source='NAVER daily reference' if price_reference else 'candidate minute aggregates',
        liquidity_source='original minute typical-value proxy, never adjusted reference price times volume',
        matched_cohort_sha256=sha(cohort) if cohort else None,
        price_reference_sha256=sha(partial/'price_reference_hashes.json') if price_reference else None,
        source_plan_sha256=plan_hash, cohort_sha256=sha(partial/'cohort.json'),
        arrays={p.stem:dict(path=p.name, sha256=sha(p)) for p in sorted(partial.glob('*.npy'))},
        seconds=time.monotonic()-started)
    (partial/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    partial.rename(output)
    print(json.dumps({k:manifest[k] for k in ['sessions','cohort_codes','rows','finite_labels','seconds']}), flush=True)
    return manifest


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--price-reference',type=Path)
    ap.add_argument('--cohort',type=Path)
    ap.add_argument('--window',type=int,default=20)
    ap.add_argument('--horizon',type=int,default=5)
    ap.add_argument('--first-month',default='202401')
    args = ap.parse_args()
    build(args.source, args.output, window=args.window,horizon=args.horizon,
        price_reference=args.price_reference,cohort=args.cohort,first_month=args.first_month)

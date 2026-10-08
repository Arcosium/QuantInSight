"""Exploratory cash-account replay for the four-year CNN study.

Adjusted price units and incomplete historical metadata are explicit proxies.
This module never upgrades a proxy ledger to a certified contest result.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant.timefolio_cnn_dataset import sha
from quant.timefolio_heatmap_locked_weighted_replay import replay


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def anchored_cap(dates, raw_close, adjusted_close, shares, available_date):
    """Constant-share, price-evolved proxy; never backfill before publication.

    The adjustment multiplier cancels at the anchor. Later share issuance and
    cancellations are NOT reconstructed by this approximation.
    """
    dates, raw, adjusted = map(np.asarray, [dates, raw_close, adjusted_close])
    out = np.full(len(dates), np.nan)
    if not np.isfinite(shares) or shares <= 0:
        return out, None
    good = (dates >= available_date) & np.isfinite(raw) & (raw > 0) & np.isfinite(adjusted) & (adjusted > 0)
    if not good.any():
        return out, None
    anchor = int(np.flatnonzero(good)[0])
    out[anchor:] = shares * raw[anchor] * adjusted[anchor:] / adjusted[anchor]
    # A suspension carries the last known proxy; it cannot create buying eligibility.
    out = pd.Series(out).ffill().to_numpy()
    return out, anchor


def align_execution(raw_open, reference_open, raw_price, raw_volume):
    """Express price and volume in the same adjusted units, preserving notional."""
    ro, ref, price, volume = map(lambda x: np.asarray(x, dtype=float),
                                [raw_open, reference_open, raw_price, raw_volume])
    valid = np.isfinite(ro) & (ro > 0) & np.isfinite(ref) & (ref > 0)
    factor = np.divide(ref, ro, out=np.full(ro.shape, np.nan), where=valid)
    return price * factor, volume / factor


def build_panel(dataset, extraction, dart, sectors, output, *, kis=None):
    dataset, extraction, dart, sectors, output = map(Path, [dataset, extraction, dart, sectors, output])
    if output.exists():
        raise ValueError('Refusing to overwrite a frozen account panel')
    manifest = json.loads((dataset/'manifest.json').read_text())
    if (dataset/'RETIRED.json').exists() or manifest.get('exploratory_ready') is not True:
        raise ValueError('Consistent-price dataset required')
    arrays = {}
    for name in ['dates', 'codes', 'daily_ohlcv', 'raw_daily_ohlcv', 'adv5_proxy']:
        spec = manifest['arrays'][name]; p = (dataset/spec['path']).resolve()
        if not p.is_relative_to(dataset.resolve()) or sha(p) != spec['sha256']:
            raise ValueError('Dataset fingerprint changed')
        arrays[name] = np.load(p, allow_pickle=False, mmap_mode='r')
    dates, codes = arrays['dates'], arrays['codes']; d, n = len(dates), len(codes)
    daily = arrays['daily_ohlcv']; raw = arrays['raw_daily_ohlcv']
    mapping = {r['code'].removeprefix('A'): int(r['sector'])
               for r in json.loads(sectors.read_text())['securities']}
    sector = np.array([mapping.get(str(c), -1) for c in codes])
    cohort = json.loads((dataset/'cohort.json').read_text())
    receipts = {r['code']: r for r in cohort['receipts']}
    date_index = {str(v): i for i, v in enumerate(dates)}
    exec_price = np.full((d, n), np.nan); exec_volume = np.zeros((d, n)); count = np.zeros((d, n))
    market_cap = np.full((d, n), np.nan); provenance = []
    for j, code_value in enumerate(codes):
        code = str(code_value); source = extraction/'daily_candidates'/f'{code}.csv.gz'
        if sha(source) != receipts[code]['sha256']:
            raise ValueError('Minute-derived shard changed: '+code)
        prices = np.full(d, np.nan); volumes = np.zeros(d)
        with gzip.open(source, 'rt') as handle:
            for row in csv.DictReader(handle):
                t = date_index.get(row['date'])
                if t is None:
                    continue
                prices[t] = float(row['exec_price_proxy']) if row['exec_price_proxy'] else np.nan
                volumes[t] = float(row['exec_volume']); count[t, j] = int(row['exec_bars'])
        exec_price[:, j], exec_volume[:, j] = align_execution(raw[:, j, 0], daily[:, j, 0], prices, volumes)
        baseline = dart/f'{code}.json'
        proof = dict(code=code, sector_current=mapping.get(code), baseline_available=False,
                     execution_shard_sha256=sha(source))
        if baseline.exists():
            info = json.loads(baseline.read_text())
            common = [r for r in info.get('records', []) if r['share_class'] == '보통주'
                      and r['issued_shares'] is not None and r['issued_shares'] > 0]
            if len(common) == 1:
                record = common[0]; anchor_prices = raw[:, j, 3].copy()
                anchor_source = 'Minute-derived raw close; historical basis is not fully certified'
                reference = Path(kis)/code/'complete.json' if kis else None
                if reference is not None and reference.exists():
                    complete = json.loads(reference.read_text()); anchor_prices[:] = np.nan
                    for filename, digest in complete['pages'].items():
                        page = (reference.parent/filename).resolve()
                        if not page.is_relative_to(reference.parent.resolve()) or sha(page) != digest:
                            raise ValueError('KIS reference changed')
                        for row in json.loads(page.read_text())['bars']:
                            t = date_index.get(row['stck_bsop_date'])
                            if t is not None:
                                anchor_prices[t] = float(row['stck_clpr'])
                    proof['kis_receipt_sha256'] = sha(reference)
                    anchor_source = 'KIS venue J unadjusted daily close'
                cap, anchor = anchored_cap(dates, anchor_prices, daily[:, j, 3],
                                           record['issued_shares'], record['receipt_date'])
                market_cap[:, j] = cap
                proof.update(baseline_available=anchor is not None, baseline_sha256=sha(baseline),
                    baseline=record, anchor_source=anchor_source,
                    anchor_date=str(dates[anchor]) if anchor is not None else None)
        provenance.append(proof)
    observed = np.isfinite(daily[:, :, 3]) & (daily[:, :, 3] > 0)
    observed_count = observed.cumsum(axis=0)
    eligible = (observed & (observed_count >= 6) & (arrays['adv5_proxy'] > 3e9)
                & np.isfinite(market_cap) & (market_cap >= 1e11) & (sector[None, :] >= 0))
    panel = dict(o=daily[:, :, 0].T, close=daily[:, :, 3].T, eligible=eligible.T,
                 sector=sector, sector_cap=np.full((n, d), .1), market_cap=market_cap.T,
                 split=np.ones((n, d)), listed_shares=np.ones((n, d)),
                 exec_price=exec_price.T, exec_volume=exec_volume.T, exec_count=count.T,
                 # Conservative proxy: reject all opening-window fills beyond +/-29%,
                 # even when a real order might have filled away from the daily limit.
                 exec_high=exec_price.T, exec_low=exec_price.T)
    output.mkdir(parents=True)
    np.savez(output/'panel.npz', **panel)
    write(output/'index.json', dict(dates=dates.tolist(), codes=codes.tolist()))
    write(output/'provenance.json', provenance)
    limitations = [
        'Exploratory adjusted-unit account, not the actual historical share/entitlement ledger.',
        'The current Timefolio sector map is backcast. Historical sector changes are not certified.',
        'Uniform10% sector caps are the conservative floor, not reconstructed historical market weights.',
        'Market cap uses published2021 common issued shares and an anchored price path; later issuance/cancellation is missing.',
        'Current-universe and completed-cohort selection bias remains; delisted/new issuers are not complete.',
        'Risk/admin designations, cash dividends, locked rights and detailed special sessions remain incomplete.',
        '09:05--09:34 typical-price VWAP,25print requirement,5% participation and5bp slippage are execution proxies.',
        'All opening-window moves beyond+/-29% are conservatively blocked on the relevant order side.',
        'Adjusted share integer rounding and provider venue/adjustment conventions are not fully reconciled.',
        'The cash gate predicts a Top10 basket proxy; the constrained account can own a different basket.',
    ]
    proof = dict(dataset_manifest_sha256=sha(dataset/'manifest.json'), dates=d, codes=n,
        baseline_codes=sum(p['baseline_available'] for p in provenance),
        sector_codes=int((sector >= 0).sum()), eligible_security_dates=int(eligible.sum()),
        contest_certified=False, limitations=limitations,
        sector_source_sha256=sha(sectors), source_sha256=sha(__file__),
        artifacts={p.name: sha(p) for p in output.iterdir()})
    write(output/'receipt.json', proof)
    print(json.dumps({k: proof[k] for k in ['codes', 'baseline_codes', 'sector_codes', 'eligible_security_dates']}), flush=True)


def statistics(returns):
    r = np.asarray(returns, dtype=float)
    if not len(r) or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError('Finite non-bankrupt daily returns required')
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.
    nav = np.cumprod(1+r); peak = np.maximum.accumulate(np.r_[1., nav])[1:]
    return dict(sessions=len(r), net_return=float(nav[-1]-1),
                sharpe=float(r.mean()/sd*np.sqrt(252)) if sd > 1e-12 else None,
                mdd=float((nav/peak-1).min()))


def evaluate_daily(rows):
    nav = np.array([r['nav'] for r in rows]); dates = [r['date'] for r in rows]
    returns = nav/np.r_[1e9, nav[:-1]]-1
    months = sorted({d[:6] for d in dates})
    folds = {m: statistics(returns[[d.startswith(m) for d in dates]]) for m in months}
    pooled = statistics(returns); sharpes = [v['sharpe'] for v in folds.values()]
    def passes(total, minimum):
        return (pooled['sharpe'] is not None and pooled['sharpe'] > total
                and all(v is not None and v > minimum for v in sharpes))
    return dict(pooled=pooled, monthly_folds=folds,
                numerical_record_threshold=passes(2., 1.), numerical_stop_threshold=passes(3., 1.5))


def turnover_windows(rows):
    """Screen each two-month contest window separately on the continuous ledger.

    All intersecting weeks count conservatively. This is not a reset-NAV contest
    replay and does not certify boundary-week treatment.
    """
    frame = pd.DataFrame(rows); result = []
    for year in range(int(frame.date.min()[:4]), int(frame.date.max()[:4])+1):
        for month in [1, 4, 7, 10]:
            start = f'{year}{month:02}01'
            end = f'{year}{month+1:02}{calendar.monthrange(year, month+1)[1]}'
            part = frame[(frame.date >= start) & (frame.date <= end)].copy()
            if part.empty:
                continue
            week = pd.to_datetime(part.date).dt.to_period('W-SUN')
            sums = part.groupby(week).agg(nav=('nav', 'mean'), buys=('buy_value', 'sum'), sells=('sell_value', 'sum'))
            turnover = .5*(sums.buys+sums.sells)/sums.nav
            low = int((turnover < .05).sum())
            result.append(dict(start=start, end=end, weeks=len(turnover), low_turnover_weeks=low,
                four_violation_screen_failed=low >= 4,
                boundary_weeks_included=True, initial_NAV_was_not_reset=True))
    return result


def run_accounts(panel_path, gate_path, output, *, buffers=(0, 5)):
    panel_path, gate_path, output = map(Path, [panel_path, gate_path, output])
    if output.exists():
        raise ValueError('Refusing to overwrite account results')
    proofs = []
    for root in [panel_path, gate_path]:
        proof = json.loads((root/'receipt.json').read_text())
        for filename, digest in proof['artifacts'].items():
            p = (root/filename).resolve()
            if not p.is_relative_to(root.resolve()) or sha(p) != digest:
                raise ValueError('Account input fingerprint changed')
        proofs.append(proof)
    if proofs[0]['dataset_manifest_sha256'] != proofs[1]['dataset_manifest_sha256']:
        raise ValueError('Gate and account panel datasets differ')
    with np.load(panel_path/'panel.npz', allow_pickle=False) as z:
        panel = {k: z[k] for k in z.files}
    index = json.loads((panel_path/'index.json').read_text())
    scores = np.load(gate_path/'scores.npy', allow_pickle=False).T
    with np.load(gate_path/'gate.npz', allow_pickle=False) as z:
        gate = {k: z[k] for k in z.files}
    available = np.flatnonzero(np.isfinite(scores).any(axis=0))
    end = index['dates'][available[-1]]; start = '20240101'
    if end < start:
        raise ValueError('Only warmup forecasts are available')
    cases = [(name, buffer) for name in ['always', 'trend_floor20', 'ridge_cash', 'ridge_floor20']
             for buffer in buffers] + [('cash', 0)]
    plan = dict(objective=proofs[1]['objective'], cases=[dict(gate=n, rank_buffer=b) for n, b in cases],
        start=start, end=end, top_n=12, target_weight=.08, gross_ceiling=.8, rebalance_sessions=5,
        max_daily_orders=10, rebalance_band=.005, participation=.05, slippage=.0005,
        minimum_stock_cap=1e11, small_cap_limit=.3,
        sector_cap=float(panel['sector_cap'].flat[0]) if np.all(panel['sector_cap']==panel['sector_cap'].flat[0]) else None,
        sector_cap_min=float(panel['sector_cap'].min()), sector_cap_max=float(panel['sector_cap'].max()),
        sector_cap_source=proofs[0].get('sector_cap_method', 'Uniform10% conservative sector floor'),
        panel_receipt_sha256=sha(panel_path/'receipt.json'), gate_receipt_sha256=sha(gate_path/'receipt.json'),
        full_registered_OOF=proofs[1]['all_folds_complete'], contest_certified=False)
    output.mkdir(parents=True); write(output/'plan.json', plan)
    summary = []
    for name, buffer in cases:
        schedule = np.nan_to_num(gate['gross_'+name], nan=.8)
        result = replay(panel, index, scores, start, end, rebalance=5, gross_schedule=schedule,
                        top_n=12, weight=.08, gross=.8, max_orders=10,
                        rank_buffer=buffer, rebalance_band=.005, return_trades=True)
        evaluation = evaluate_daily(result['daily']); windows = turnover_windows(result['daily'])
        case = dict(gate=name, rank_buffer=buffer, **evaluation,
            metrics=result['metrics'], turnover_windows=windows, contest_certified=False,
            qualified_for_record=False, qualified_for_stop=False,
            qualification_reason='Historical metadata and execution proxies are not certified; numerical thresholds are reported separately.')
        # The old engine's whole-period four-week flag is not a contest-window verdict.
        case['metrics']['whole_period_four_week_turnover_stop'] = case['metrics'].pop('four_week_turnover_stop')
        case['metrics']['sharpe'] = evaluation['pooled']['sharpe']
        result['evaluation'] = case
        filename = f'{name}_buffer{buffer}.json'; write(output/filename, result)
        summary.append(dict(file=filename, sha256=sha(output/filename), **case))
        write(output/'progress.json', dict(completed=len(summary), planned=len(cases)))
        print(json.dumps(dict(gate=name, buffer=buffer, **evaluation['pooled']), allow_nan=False), flush=True)
    write(output/'summary.json', dict(plan=plan, cases=summary,
          limitations=proofs[0]['limitations'], source_sha256=sha(__file__),
          replay_source_sha256=sha(Path(__file__).with_name('timefolio_heatmap_locked_weighted_replay.py'))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('panel')
    for name in ['dataset', 'extraction', 'dart', 'sectors', 'output']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--kis')
    p = sub.add_parser('run')
    p.add_argument('--panel', required=True); p.add_argument('--gate', required=True); p.add_argument('--output', required=True)
    args = ap.parse_args(); values = vars(args); command = values.pop('command')
    if command == 'panel':
        build_panel(**values)
    else:
        run_accounts(values['panel'], values['gate'], values['output'])

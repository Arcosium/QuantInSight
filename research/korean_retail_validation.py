"""Korean retail validation, isolated from brokers and live collectors.

prepare freezes public market archives and recent provider snapshots.
run tests the original closed society and an explicitly DIFFERENT replay probe.
The probe reuses native routines/biases, masks unavailable fundamentals and
paper-settles against a funded passive outside investor at observed closes.
It forecasts NEXT-session labels; observed closes are inputs, not price forecasts.
No LLM, account records, broker API or service configuration is used.

python3 -m research.korean_retail_validation prepare
python3 -m research.korean_retail_validation run --run data/korean_retail/<run>
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from datetime import datetime
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from research.investor_society import BIAS_NAMES, Order, Resident, Society
from research.society_shocks import CoupledSociety
from research.news_swarm import ROOT, fetch_market, now, write_json

VERSION = 'korean-retail-validation-v1'
GROUPS = (('005930', '000660', '005380'), ('035420', '035720', '373220'),
          ('035900', '041510', '214150'), ('095340', '263750', '036930'))
CODES = tuple(code for group in GROUPS for code in group)
SEEDS = (7, 42)
FEATURES = ('ret1', 'ret5', 'vol10', 'relative_volume', 'flow1', 'flow5', 'market_ret1')
TARGETS = ('flow_proxy', 'entry_return')
ENGINES = ('train_mean', 'zero', 'persistence', 'linear', 'native_only',
           'native_no_bias_only', 'linear_native', 'linear_native_no_bias')
SELECTABLE = ('linear', 'native_only', 'native_no_bias_only', 'linear_native', 'linear_native_no_bias')
REFERENCES = {
    'krx': 'https://data.krx.co.kr/contents/MDC/MAIN/main.jspx',
    'korean_micro_evidence': 'https://www.kcmi.re.kr/common/downloadw?fgu=002002&fid=25260&fty=004003',
    'abm_validation': 'https://arxiv.org/abs/2206.09772',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def quote_basis_check(old, new):
    overlap = old.merge(new,on='date',suffixes=('_old','_new'))
    if not len(overlap):
        return {'overlap_dates':0,'suspect_adjustment_basis':False}
    ratio = overlap.close_new / overlap.close_old - 1
    median = float(np.median(abs(ratio)))
    fraction = float((abs(ratio) > .003).mean())
    return {'overlap_dates':len(overlap),'median_absolute_price_revision_pct':median*100,
            'dates_with_price_revision_over_0_3_pct_fraction':fraction,
            'suspect_adjustment_basis':bool(len(overlap)>=20 and median>.003 and fraction>=.5)}


def clean_frame(frame, kind, as_of):
    required = ['date', 'open', 'high', 'low', 'close', 'volume'] if kind == 'price' else ['date', 'inst_net', 'foreign_net']
    if any(c not in frame for c in required):
        raise ValueError('Missing public market fields')
    frame = frame.copy()
    frame['date'] = pd.to_datetime(frame.date, errors='coerce').dt.strftime('%Y-%m-%d')
    numeric = [c for c in frame if c != 'date']
    for c in numeric:
        frame[c] = pd.to_numeric(frame[c], errors='coerce')
    invalid = frame[required].isna().any(axis=1) | ~np.isfinite(frame[required[1:]]).all(axis=1)
    quality = {'raw_rows': len(frame), 'invalid_rows': int(invalid.sum())}
    frame = frame[~invalid & (frame.date < as_of)].copy()
    duplicates = frame.duplicated().sum()
    frame = frame.drop_duplicates()
    # Without vintage timestamps, conflicting rows cannot be ordered safely.
    conflicting = frame.groupby('date')[required[1:]].nunique().max(axis=1) > 1
    frame = frame[~frame.date.isin(conflicting[conflicting].index)]
    frame = frame.drop_duplicates('date').sort_values('date')
    if kind == 'price':
        valid = ((frame[['open', 'high', 'low', 'close']] > 0).all(axis=1) & (frame.volume > 0)
                 & (frame.high >= frame[['open', 'low', 'close']].max(axis=1))
                 & (frame.low <= frame[['open', 'high', 'close']].min(axis=1)))
    else:
        valid = pd.Series(True, index=frame.index)
    quality.update(exact_duplicate_rows_removed=int(duplicates),
                   conflicting_dates_quarantined=int(conflicting.sum()),
                   invalid_market_rows=int((~valid).sum()))
    frame = frame[valid].reset_index(drop=True)
    quality.update(clean_rows=len(frame), first_date=frame.date.min() if len(frame) else None,
                   last_date=frame.date.max() if len(frame) else None)
    return frame, quality


def prepare(fetch=True, start='2024-09-01'):
    run = ROOT / 'data' / 'korean_retail' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    (run / 'raw').mkdir(parents=True)
    (run / 'market').mkdir()
    as_of = now().date().isoformat()
    manifest = {'version': VERSION, 'created_at': now().isoformat(), 'as_of_exclusive': as_of,
                'start': start, 'codes': CODES, 'groups': GROUPS, 'people_per_replay': 1000,
                'seeds': SEEDS, 'split_fractions': [.6, .2, .2], 'split_by': 'label session, shared calendar',
                'targets': TARGETS, 'engines': ENGINES, 'validation_selection_candidates': SELECTABLE,
                'cost_bps_roundtrip': 20, 'bootstrap_blocks': [5, 10], 'bootstrap_draws': 2000,
                'primary_interval_coverage': .9875,
                'inference_family': 'four predeclared comparisons: native incremental and validation-selected vs linear, for two targets',
                'ridge_alpha': 1, 'references': REFERENCES,
                'micro_account_data_used': False, 'laya_or_llm_used': False,
                'primary_flow_definition': '-(institution_net_shares + foreign_net_shares) / next_session_volume',
                'primary_flow_is_exact_individual': False,
                'quote_basis_gate': 'reject >=20 overlap sessions with >=50% price revisions above 0.3% and median absolute revision above 0.3%',
                'data_quality_replacements_before_primary_fit': {'196170':'214150','247540':'095340'},
                'replacement_reason': 'persistent multiplicative price/volume revisions between archive and fresh chart; adjustment basis unverified; replacement chosen on data quality, not model scores',
                'limitations': [
                    'historical pseudo-out-of-sample; snapshots lack original observation/revision vintages',
                    '12 convenience stocks chosen from existing coverage; no point-in-time universe, delistings or representativeness claim',
                    'residual includes other corporations/foreigners; direct individual field is a recent secondary check',
                    'no account-level holdings, purchase prices, demographics or orders to identify individual biases',
                    'replay probe differs from original society: exogenous real quotes, passive outside investor, masked fundamentals/dividends, no simulated wage/consumption economy',
                    'paper fills at known closes are assumed liquidity, not observed individual fills',
                    'synthetic portfolio states and population distributions remain uncalibrated',
                    'trading scenario is next-open to same-day-close, fixed universe weights, hypothetical 20 bps roundtrip; not live executable alpha',
                    'no individual causal explanation or real-market intervention identification',
                ]}
    # Lock the protocol before reading outcomes or fitting anything.
    write_json(run / 'manifest.json', manifest)
    quality, source_files = {}, {}
    for code in (*CODES, '069500'):
        frames = {}
        for prefix, kind in [('daily', 'price'), ('investor', 'flow')]:
            path = ROOT / 'data' / f'{prefix}_{code}.csv'
            if path.exists():
                raw = path.read_bytes()
                (run / 'raw' / path.name).write_bytes(raw)
                source_files[path.name] = {'sha256': sha(raw), 'mtime_ns': path.stat().st_mtime_ns}
                frames[kind], q = clean_frame(pd.read_csv(BytesIO(raw)), kind, as_of)
                quality[f'{code}_{kind}_archive'] = q
        if fetch:
            try:
                prices, flows = fetch_market(code)
                prices.to_csv(run / 'raw' / f'fresh_prices_{code}.csv', index=False)
                flows.to_csv(run / 'raw' / f'fresh_flows_{code}.csv', index=False)
                flows = flows.rename(columns={'institution_shares': 'inst_net', 'foreign_shares': 'foreign_net'})
                for kind, incoming in [('price', prices), ('flow', flows)]:
                    old = frames.get(kind, pd.DataFrame(columns=incoming.columns))
                    if kind=='price':
                        basis = quote_basis_check(old,incoming)
                        quality[code+'_quote_basis'] = basis
                        if basis['suspect_adjustment_basis']:
                            raise ValueError('Unverified price adjustment basis')
                    overlap = old.merge(incoming, on='date', suffixes=('_old', '_new'))
                    fields = ['close', 'volume'] if kind == 'price' else ['inst_net', 'foreign_net']
                    revisions = int(np.logical_or.reduce([overlap[c+'_old'] != overlap[c+'_new'] for c in fields]).sum()) if len(overlap) else 0
                    # Fresh edition supersedes old in this NEW research snapshot only.
                    combined = pd.concat([old[~old.date.isin(incoming.date)], incoming], ignore_index=True)
                    frames[kind], q = clean_frame(combined, kind, as_of)
                    q['fresh_overlap_changed_dates'] = revisions
                    quality[f'{code}_{kind}_merged'] = q
                quality[code+'_fetch'] = {'status': 'ok', 'observed_at': now().isoformat()}
            except Exception as exc:
                quality[code+'_fetch'] = {'status': 'failed', 'error_type': type(exc).__name__}
        if code != '069500' and set(frames) != {'price', 'flow'}:
            raise ValueError('Required stock snapshot missing: '+code)
        for kind, frame in frames.items():
            frame.to_csv(run / 'market' / f'{kind}_{code}.csv', index=False)
    manifest['archive_sources'] = source_files
    manifest['source_sha256'] = {name: sha((ROOT / 'research' / name).read_bytes())
                               for name in ['korean_retail_validation.py', 'investor_society.py', 'society_shocks.py', 'news_swarm.py']}
    write_json(run / 'manifest.json', manifest)
    write_json(run / 'data_quality.json', quality)
    print(json.dumps({'run': str(run), 'codes': len(CODES)}), flush=True)
    return run


def load_markets(run):
    return {code: {kind: pd.read_csv(run / 'market' / f'{kind}_{code}.csv')
                   for kind in ['price', 'flow']} for code in CODES}


def make_panel(markets, calendar, market_prices, start):
    if market_prices is None:
        market = pd.concat([data['price'].set_index('date').reindex(calendar).close.pct_change(fill_method=None)
                            for data in markets.values()], axis=1).mean(axis=1)
    else:
        market = market_prices.set_index('date').reindex(calendar).close.pct_change(fill_method=None)
    rows, quality = [], {}
    for code, data in markets.items():
        p = data['price'].set_index('date').reindex(calendar)
        f = data['flow'].set_index('date').reindex(calendar)
        net = -(f.inst_net + f.foreign_net)
        invalid_flow = (f.inst_net.abs() > p.volume) | (f.foreign_net.abs() > p.volume) | (net.abs() > p.volume)
        net = net.mask(invalid_flow)
        ret = p.close.pct_change(fill_method=None)
        jumps = ret.abs() > .305
        # Adjustment status is unknown. Quarantine the jump and its lookback.
        clean_window = ~jumps.rolling(21, min_periods=1).max().astype(bool)
        q = pd.DataFrame({'code': code, 'date': calendar, 'label_date': pd.Series(calendar).shift(-1).values}, index=calendar)
        q['ret1'], q['ret5'] = ret, p.close / p.close.shift(5) - 1
        q['vol10'] = ret.rolling(10, min_periods=10).std(ddof=0)
        q['relative_volume'] = np.log(p.volume / p.volume.shift(1).rolling(20, min_periods=20).median())
        q['flow1'] = net / p.volume
        q['flow5'] = net.rolling(5, min_periods=5).sum() / p.volume.rolling(5, min_periods=5).sum()
        q['market_ret1'] = market
        q['entry_lag1'] = p.close / p.open - 1
        q['flow_proxy'] = net.shift(-1) / p.volume.shift(-1)
        q['entry_return'] = p.close.shift(-1) / p.open.shift(-1) - 1
        q['close_return'] = ret.shift(-1)
        q['individual_ratio'] = f['individual_shares'].shift(-1) / p.volume.shift(-1) if 'individual_shares' in f else np.nan
        q['raw_volume_next'] = p.volume.shift(-1)
        q = q[(q.date >= start) & clean_window & ~jumps.shift(-1, fill_value=False)]
        q = q.replace([np.inf, -np.inf], np.nan).dropna(subset=[*FEATURES, *TARGETS, 'close_return', 'label_date'])
        quality[code] = {'flow_rows_exceeding_total_volume': int(invalid_flow.sum()),
                         'unverified_price_jumps_quarantined': int(jumps.sum()), 'eligible_rows': len(q)}
        rows.append(q.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True), quality


def split_panel(panel):
    dates = sorted(panel.label_date.unique())
    if len(dates) < 150:
        raise ValueError('Fewer than 150 distinct label sessions')
    train_end = dates[max(1, int(len(dates) * .6)) - 1]
    validation_end = dates[max(2, int(len(dates) * .8)) - 1]
    out = panel.copy()
    out['split'] = np.where(out.label_date <= train_end, 'train', np.where(out.label_date <= validation_end, 'validation', 'test'))
    keep = out[out.split == 'train'].groupby('code').size()
    excluded = sorted(set(out.code) - set(keep[keep >= 60].index))
    out = out[~out.code.isin(excluded)].reset_index(drop=True)
    return out, {'train_label_end': train_end, 'validation_label_end': validation_end,
                 'test_label_start': out[out.split == 'test'].label_date.min(),
                 'excluded_insufficient_training_stocks': excluded,
                 'rows': out.groupby('split').size().to_dict(),
                 'dates': out.groupby('split').label_date.nunique().to_dict()}


class RetailReplay(CoupledSociety):
    """Diagnostic adapter, NOT an empirically calibrated price-forming society."""
    @classmethod
    def create(cls, people, seed, disabled=()):
        s = cls.from_checkpoint(Society(people, seed, disabled_biases=disabled).checkpoint(), seed)
        s.external = copy.deepcopy(s.residents[0])
        s.external.id = people
        s.external.cash = people * 200000
        s.external.shares = [people * 100] * 3
        s.external.cost_basis = [float(f.price) for f in s.firms]
        s.external.debt = s.external.interest_due = 0
        s.external.memories = []
        s.initial_cash += s.external.cash
        for j, f in enumerate(s.firms):
            f.supply += s.external.shares[j]
        s.validate_replay()
        return s

    def observation(self, a):
        obs = super().observation(a)
        for m in obs['market']:
            m.update(value_gap=0, dividend_yield_today=0,
                     firm_cash=None, operating_profit_ema=None)
        obs['unavailable_fields'] = ['fundamental_value', 'dividend', 'employer_accounts']
        return obs

    def validate_replay(self):
        self.residents.append(self.external)
        try:
            Society.validate(self)
        finally:
            self.residents.pop()

    def emit(self, prices):
        if len(prices) != 3 or min(prices) <= 0:
            raise ValueError('Three positive known quotes required')
        self.day += 1
        for f, price in zip(self.firms, prices):
            f.price = int(price)
        self.prices.append([int(x) for x in prices])
        orders = self.orders()
        buy = [sum(o.quantity for o in orders if o.sector == j and o.side == 'buy') for j in range(3)]
        sell = [sum(o.quantity for o in orders if o.sector == j and o.side == 'sell') for j in range(3)]
        # Conditional desired order pressure is emitted BEFORE any paper fills.
        pressure = [(b - s) / (b + s) if b + s else 0.0 for b, s in zip(buy, sell)]
        self.residents.append(self.external)
        try:
            for order in orders:
                price = int(prices[order.sector])
                if order.side == 'buy' and order.limit < price or order.side == 'sell' and order.limit > price:
                    continue
                opposite = Order(self.external.id, order.sector,
                                 'sell' if order.side == 'buy' else 'buy', order.quantity, price, -1, 'passive_external_paper_fill')
                if order.side == 'buy':
                    self.settle(order, opposite, order.quantity, price)
                else:
                    self.settle(opposite, order, order.quantity, price)
        finally:
            self.residents.pop()
        self.validate_replay()
        net = [sum(t['shares'] * ((t['buyer'] < self.external.id) - (t['seller'] < self.external.id))
                   for t in self.transactions if t['sector'] == j) for j in range(3)]
        record = {'pressure': pressure, 'buy': buy, 'sell': sell, 'paper_retail_net_shares': net,
                  'orders': len(orders), 'paper_fills': len(self.transactions),
                  'cash_error': 0, 'share_error': [0, 0, 0]}
        self.transactions.clear()
        self.decisions.clear()
        return record


def replay_job(run, group, seed, people, disabled):
    frames = [pd.read_csv(run / 'market' / f'price_{code}.csv').set_index('date') for code in group]
    manifest = json.loads((run / 'manifest.json').read_text())
    sessions = sorted(set.intersection(*(set(p.index) for p in frames)))
    sessions = [d for d in sessions if d >= manifest['start']]
    if not sessions:
        raise ValueError('No common quote sessions')
    s = RetailReplay.create(people, seed, disabled)
    original_quotes = [p.loc[sessions[0], 'close'] for p in frames]
    initial_prices = s.prices[0][:]
    records = []
    for date in sessions:
        known = [max(20, round(base * p.loc[date, 'close'] / initial))
                 for base, p, initial in zip(initial_prices, frames, original_quotes)]
        daily = s.emit(known)
        for j, code in enumerate(group):
            records.append({'date': date, 'code': code, 'seed': seed,
                            'regime': 'no_bias' if disabled else 'full',
                            'pressure': daily['pressure'][j], 'buy_shares': daily['buy'][j],
                            'sell_shares': daily['sell'][j], 'paper_retail_net_shares': daily['paper_retail_net_shares'][j],
                            'cash_error': 0, 'share_error': 0})
    return records


def autocorrelation(x, lag=1):
    x = np.asarray(x, dtype=float)
    if len(x) <= lag + 2 or np.std(x[:-lag]) == 0 or np.std(x[lag:]) == 0:
        return 0.0
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def moments(returns, volumes):
    r, v = np.asarray(returns, dtype=float), np.asarray(volumes, dtype=float)
    valid = np.isfinite(r) & np.isfinite(v)
    r, v = r[valid], v[valid]
    if len(r) < 30:
        raise ValueError('Insufficient moment sample')
    logv = np.log1p(v)
    correlation = np.corrcoef(r, logv)[0, 1] if np.std(r) and np.std(logv) else 0.0
    return {'return_std': float(np.std(r)), 'excess_kurtosis': float(stats.kurtosis(r, fisher=True, bias=False)) if np.std(r) else 0.0,
            'return_acf1': autocorrelation(r), 'abs_return_acf1': autocorrelation(abs(r)),
            'abs_return_acf5': autocorrelation(abs(r), 5), 'log_volume_acf1': autocorrelation(logv),
            'return_log_volume_correlation': float(correlation)}


def society_moment_job(people, seed, disabled, days=240, warmup=60):
    s = CoupledSociety.from_checkpoint(Society(people, seed, disabled_biases=disabled).checkpoint(), seed)
    rs, vs = [[], [], []], [[], [], []]
    for day in range(warmup + days):
        before, pos = s.prices[-1][:], len(s.transactions)
        row = s.step()
        if day < warmup:
            continue
        for j in range(3):
            rs[j].append(row['prices_cents'][j] / before[j] - 1)
            vs[j].append(sum(t['shares'] for t in s.transactions[pos:] if t['sector'] == j))
    return [{'regime': 'no_bias' if disabled else 'full', 'seed': seed, 'sector': j,
             'moments': moments(rs[j], vs[j])} for j in range(3)]


def engine_features(engine):
    if engine == 'linear':
        return list(FEATURES)
    if engine == 'native_only':
        return ['native_pressure']
    if engine == 'native_no_bias_only':
        return ['native_no_bias_pressure']
    if engine == 'linear_native':
        return [*FEATURES, 'native_pressure']
    if engine == 'linear_native_no_bias':
        return [*FEATURES, 'native_no_bias_pressure']
    return []


def fit_linear(frame, target, features, alpha=1.0):
    codes = sorted(frame.code.unique())
    raw = frame[features].to_numpy(float)
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale[scale == 0] = 1
    x = np.column_stack([np.ones(len(frame)), *[(frame.code == c).to_numpy(float) for c in codes[1:]], (raw-mean)/scale])
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ frame[target].to_numpy(float))
    return {'codes': codes, 'features': features, 'mean': mean.tolist(), 'scale': scale.tolist(), 'beta': beta.tolist()}


def predict_linear(fit, frame):
    raw = frame[fit['features']].to_numpy(float)
    x = np.column_stack([np.ones(len(frame)), *[(frame.code == c).to_numpy(float) for c in fit['codes'][1:]],
                         (raw-np.asarray(fit['mean']))/np.asarray(fit['scale'])])
    return x @ np.asarray(fit['beta'])


def fit_engine(frame, target, engine):
    if engine in {'train_mean', 'zero', 'persistence'}:
        return {'engine': engine, 'target': target, 'stock_means': frame.groupby('code')[target].mean().to_dict(),
                'global_mean': float(frame[target].mean())}
    return {'engine': engine, 'target': target, **fit_linear(frame, target, engine_features(engine))}


def predict_engine(fit, frame):
    if fit['engine'] == 'zero':
        return np.zeros(len(frame))
    if fit['engine'] == 'persistence':
        return frame['flow1' if fit['target'] == 'flow_proxy' else 'entry_lag1'].to_numpy(float)
    if fit['engine'] == 'train_mean':
        return frame.code.map(fit['stock_means']).fillna(fit['global_mean']).to_numpy(float)
    return predict_linear(fit, frame)


def predictive_metrics(y, prediction, reference):
    y, p, ref = np.asarray(y), np.asarray(prediction), np.asarray(reference)
    mse, refmse = np.mean((y-p)**2), np.mean((y-ref)**2)
    directions, pdirections = np.sign(y), np.sign(p)
    classes = np.unique(directions)
    balanced = np.mean([np.mean(pdirections[directions == c] == c) for c in classes])
    corr = np.corrcoef(y, p)[0, 1] if np.std(y) and np.std(p) else None
    return {'n': len(y), 'mse': float(mse), 'mae': float(np.mean(abs(y-p))),
            'oos_r2_vs_training_stock_mean': float(1-mse/refmse) if refmse else None,
            'direction_accuracy': float(np.mean(directions == pdirections)),
            'balanced_direction_accuracy': float(balanced), 'correlation': float(corr) if corr is not None else None}


def paired_block_interval(frame, target, candidate, baseline, block, draws=2000, coverage=.9875, seed=20261002):
    y = frame[target].to_numpy(float)
    delta = (y-frame[baseline].to_numpy(float))**2 - (y-frame[candidate].to_numpy(float))**2
    daily = pd.Series(delta, index=frame.label_date).groupby(level=0).mean().sort_index().to_numpy()
    if len(daily) < max(30, block*3):
        return {'dates': len(daily), 'estimable': False, 'block': block}
    rng = np.random.default_rng(seed + block)
    width = min(block, len(daily))
    starts = rng.integers(0, len(daily)-width+1, size=(draws, math.ceil(len(daily)/width)))
    indices = (starts[..., None] + np.arange(width)).reshape(draws, -1)[:, :len(daily)]
    sampled = daily[indices].mean(axis=1)
    tail = (1-coverage)/2
    low, high = np.quantile(sampled, [tail, 1-tail])
    return {'dates': len(daily), 'block': block, 'draws': draws, 'coverage': coverage,
            'estimable': True, 'mean_daily_mse_improvement': float(daily.mean()),
            'lower': float(low), 'upper': float(high), 'improvement_supported': bool(low > 0),
            'meaning': 'positive means lower squared error than baseline; cross-section kept together within each session'}


def evaluate_predictions(panel, run, manifest):
    train, validation, test = [panel[panel.split == s].copy() for s in ['train', 'validation', 'test']]
    report, selected, fits, heldout = {}, {}, {}, test.copy()
    calibration = pd.concat([train, validation], ignore_index=True)
    for target in TARGETS:
        val_scores = {}
        for engine in SELECTABLE:
            fit = fit_engine(train, target, engine)
            pred = predict_engine(fit, validation)
            val_scores[engine] = float(np.mean((validation[target].to_numpy()-pred)**2))
        chosen = min(SELECTABLE, key=lambda e: (val_scores[e], SELECTABLE.index(e)))
        selected[target] = {'engine': chosen, 'validation_mse': val_scores,
                            'rule': 'lowest validation MSE; refit train+validation once; no test selection'}
        reference = predict_engine(fit_engine(calibration, target, 'train_mean'), test)
        metrics = {}
        for engine in ENGINES:
            fit = fit_engine(calibration, target, engine)
            fits[target+'__'+engine] = fit
            pred = predict_engine(fit, test)
            column = target+'__'+engine
            heldout[column] = pred
            metrics[engine] = predictive_metrics(test[target].to_numpy(), pred, reference)
        intervals = {}
        for label, engine in [('native_incremental', 'linear_native'), ('validation_selected', chosen)]:
            intervals[label] = [paired_block_interval(heldout, target, target+'__'+engine,
                                                      target+'__linear', block, manifest['bootstrap_draws'], manifest['primary_interval_coverage'])
                                for block in manifest['bootstrap_blocks']]
        report[target] = {'metrics': metrics, 'intervals_vs_linear': intervals,
                          'native_incremental_supported': all(x.get('improvement_supported', False) for x in intervals['native_incremental']),
                          'validation_selected_improvement_supported': all(x.get('improvement_supported', False) for x in intervals['validation_selected'])}
    direct = heldout.dropna(subset=['individual_ratio']).copy()
    report['direct_individual_secondary'] = {'rows': len(direct), 'dates': direct.label_date.nunique(),
        'proxy_direction_agreement': float(np.mean(np.sign(direct.individual_ratio)==np.sign(direct.flow_proxy))) if len(direct) else None,
        'median_abs_proxy_ratio_difference': float(np.median(abs(direct.individual_ratio-direct.flow_proxy))) if len(direct) else None,
        'metrics': {e: predictive_metrics(direct.individual_ratio.to_numpy(), direct['flow_proxy__'+e].to_numpy(), direct['flow_proxy__train_mean'].to_numpy())
                    for e in ENGINES} if len(direct) else {},
        'note': 'models fit residual targets; this is transfer to exact individual net shares, not training on individual labels'}
    # Do not turn a sparse/incomplete cross-section into a portfolio silently.
    complete = heldout.groupby('label_date').code.nunique()
    dates = complete[complete == len(CODES)].index
    portfolio = heldout[heldout.label_date.isin(dates)].copy()
    cost = manifest['cost_bps_roundtrip']/10000
    scenarios = {}
    for name in ['always_long', 'linear', selected['entry_return']['engine'], 'linear_native']:
        active = np.ones(len(portfolio), dtype=bool) if name == 'always_long' else portfolio['entry_return__'+name].to_numpy() > 0
        daily = pd.Series(np.where(active, portfolio.entry_return.to_numpy()-cost, 0), index=portfolio.label_date).groupby(level=0).mean()
        nav = (1+daily).cumprod()
        scenarios[name] = {'dates': len(daily), 'trades': int(active.sum()),
                           'total_return_pct': float((nav.iloc[-1]-1)*100) if len(nav) else None,
                           'mean_daily_return_pct': float(daily.mean()*100) if len(daily) else None}
    report['illustrative_trading'] = {'complete_cross_section_dates': len(dates), 'cost_bps_roundtrip': manifest['cost_bps_roundtrip'],
                                     'execution': 'next open to same-session close; uninvested slots stay cash; equal weights across all 12 slots',
                                     'scenarios': scenarios, 'verified_executable_alpha': False}
    heldout.to_csv(run / 'heldout_predictions.csv', index=False)
    write_json(run / 'fitted_models.json', fits)
    write_json(run / 'selection.json', selected)
    return report, selected


def empirical_moment_rows(markets, train_end, test_start, start, calendar=None):
    records = []
    for code, data in markets.items():
        p = data['price'].set_index('date').sort_index()
        if calendar is not None:
            p = p.reindex(calendar)
        r = p.close.pct_change(fill_method=None)
        for split, mask in [('train', (p.index >= start) & (p.index <= train_end)), ('test', p.index >= test_start)]:
            valid = mask & (r.abs() <= .305) & r.notna()
            if valid.sum() >= 30:
                records.append({'code': code, 'split': split, 'moments': moments(r[valid].values, p.volume[valid].values)})
    return records


def moment_comparison(empirical, simulated):
    keys = list(empirical[0]['moments'])
    train = [x['moments'] for x in empirical if x['split']=='train']
    test = [x['moments'] for x in empirical if x['split']=='test']
    # These are descriptive dispersion bands, never a formal model confidence interval.
    floors = {'return_std': .001, 'excess_kurtosis': .5, 'return_acf1': .05,
              'abs_return_acf1': .05, 'abs_return_acf5': .05, 'log_volume_acf1': .05,
              'return_log_volume_correlation': .05}
    scale = {k: max(floors[k], float(np.quantile([x[k] for x in train], .75)-np.quantile([x[k] for x in train], .25))) for k in keys}
    output = {}
    for regime in ['full', 'no_bias']:
        sims = [x['moments'] for x in simulated if x['regime']==regime]
        comparisons = {}
        for k in keys:
            sm = float(np.median([x[k] for x in sims]))
            tm = float(np.median([x[k] for x in train]))
            lo, hi = np.quantile([x[k] for x in test], [.1,.9])
            comparisons[k] = {'simulated_median': sm, 'real_train_median': tm,
                              'real_test_median': float(np.median([x[k] for x in test])),
                              'real_test_stock_p10': float(lo), 'real_test_stock_p90': float(hi),
                              'within_test_stock_dispersion': bool(lo <= sm <= hi),
                              'train_scaled_error': abs(sm-tm)/scale[k]}
        output[regime] = {'moments': comparisons, 'training_distance': float(np.mean([v['train_scaled_error'] for v in comparisons.values()])),
                          'heldout_moments_inside_stock_p10_p90': sum(v['within_test_stock_dispersion'] for v in comparisons.values()),
                          'total_moments': len(keys)}
    selected = min(output, key=lambda c: output[c]['training_distance'])
    return {'regimes': output, 'training_selected_regime': selected,
            'method': 'regime selection from training distance only; test bands are cross-stock dispersion, not confidence intervals',
            'empirical_validity_established': False,
            'reason': 'matching some moments cannot identify human mechanisms or validate individual causal effects'}


def run_analysis(run, workers=4, smoke=False):
    manifest = json.loads((run / 'manifest.json').read_text())
    for name, value in manifest['source_sha256'].items():
        if sha((ROOT/'research'/name).read_bytes()) != value:
            raise ValueError('Research source changed after protocol freeze: '+name)
    if (run/'report.json').exists():
        raise ValueError('Completed runs are immutable; prepare a new run')
    quality_file=run/'data_quality.json'
    if quality_file.exists():
        quality=json.loads(quality_file.read_text())
        if any(q.get('suspect_adjustment_basis',False) for q in quality.values()):
            raise ValueError('Quote basis gate failed; do not mix unverified adjustment editions')
    people, seeds = (60, SEEDS[:1]) if smoke else (1000, SEEDS)
    settings = {'smoke_only': smoke, 'workers': workers, 'people_per_replay': people,
                'seeds': seeds, 'rules_only': True, 'retail_replay_state': 'same-group observed quotes; persistent native positions/cost bases/routines; passive funded external paper counterparty'}
    write_json(run/'run_settings.json', settings)
    markets = load_markets(run)
    calendar = sorted(markets['005930']['price'].date.unique())
    panel, panel_quality = make_panel(markets, calendar, None, manifest['start'])
    panel, split = split_panel(panel)
    split['market_feature'] = 'same-session equally-weighted sample close returns, not KOSPI index'
    write_json(run/'split.json', split)
    write_json(run/'panel_quality.json', panel_quality)
    empirical = empirical_moment_rows(markets, split['train_label_end'], split['test_label_start'], manifest['start'],calendar)
    write_json(run/'empirical_moments.json', empirical)
    # Structural falsification of the ORIGINAL closed resident-only society.
    original = {'aggregate_retail_net_shares_is_identically_zero': True,
                'real_nonzero_residual_fraction': float((panel.flow_proxy != 0).mean()),
                'direct_individual_nonzero_fraction': float((panel.individual_ratio.dropna()!=0).mean()) if panel.individual_ratio.notna().any() else None,
                'status': 'fails_real_aggregate_flow_structure',
                'reason': 'original matches only residents to each other; no institutions/foreigners to absorb a net retail flow'}
    records, simulated = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {}
        for group in GROUPS:
            for seed in seeds:
                for disabled in [(), BIAS_NAMES]:
                    jobs[pool.submit(replay_job, run, group, seed, people, disabled)] = ('replay', group, seed, bool(disabled))
        for seed in (7,42,99,2026) if not smoke else (7,):
            for disabled in [(), BIAS_NAMES]:
                jobs[pool.submit(society_moment_job, people, seed, disabled, 240 if not smoke else 60, 60 if not smoke else 20)] = ('original_moments', seed, bool(disabled))
        for future in as_completed(jobs):
            kind, *label = jobs[future]
            if kind == 'replay':
                records.extend(future.result())
            else:
                simulated.extend(future.result())
            print(json.dumps({'completed': kind, 'job': label}), flush=True)
            # Retain completed expensive jobs if a later job fails.
            if kind == 'replay':
                pd.DataFrame(records).to_csv(run/'replay_raw.csv', index=False)
            else:
                write_json(run/'simulated_moments.json', simulated)
    raw = pd.DataFrame(records)
    averaged = raw.groupby(['date','code','regime']).pressure.mean().unstack('regime').reset_index()
    averaged = averaged.rename(columns={'full':'native_pressure', 'no_bias':'native_no_bias_pressure'})
    count_before = len(panel)
    panel = panel.merge(averaged, on=['date','code'], how='inner').dropna(subset=['native_pressure','native_no_bias_pressure'])
    split['rows_lost_to_missing_replay_quotes'] = count_before-len(panel)
    split['paired_evaluation_rows'] = panel.groupby('split').size().to_dict()
    write_json(run/'split.json', split)
    panel.to_csv(run/'panel.csv',index=False)
    predictions, selection = evaluate_predictions(panel, run, manifest)
    price_stats = moment_comparison(empirical, simulated)
    report = {'version': VERSION, 'smoke_only': smoke, 'scope': '12 selected Korean equities; aggregate retail-associated flow and next-session price returns',
              'original_society_structure': original, 'original_price_statistics': price_stats,
              'replay_probe_predictions': predictions, 'selection': selection, 'split': split,
              'micro_behavior_validation': {'status':'unidentified_without_account_level_data',
                   'missing':['individual holdings and purchase prices','orders/trades by account','age/background and cash/income constraints'],
                   'not_validated':['holding periods','gain/loss-conditional sale propensity','individual trading frequency','persona distributions','human response to simulated trait interventions']},
              'price_causality_established': False, 'verified_real_market_alpha': False,
              'laya_or_llm_used': False, 'limitations': manifest['limitations'], 'references': REFERENCES}
    write_json(run/'report.json',report)
    print(json.dumps({'run': str(run), 'paired_rows':len(panel),'original_structure':original['status'],
                      'selected':{k:v['engine'] for k,v in selection.items()}}),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--offline',action='store_true'); prep.add_argument('--start',default='2024-09-01')
    run = sub.add_parser('run'); run.add_argument('--run',required=True); run.add_argument('--workers',type=int,default=4); run.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':
        prepare(not args.offline,args.start)
    else:
        if not 1 <= args.workers <= 4: parser.error('workers must be 1..4')
        run_analysis(Path(args.run),args.workers,args.smoke)


if __name__=='__main__':
    main()

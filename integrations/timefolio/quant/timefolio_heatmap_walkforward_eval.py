"""Causal portfolio comparison and exploratory, family-adjusted inference.

This is development evidence on reused history. A bootstrap cannot undo earlier
adaptive research choices or substitute for prospective independent validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_replay import INITIAL_NAV, replay
from quant.timefolio_heatmap_study import context, score_matrix
from quant.timefolio_heatmap_walkforward import DEST, SOURCE, monthly_folds


def market_regimes(p):
    """Lagged cap weights, observed close returns; output is indexed by signal day."""
    weights = np.zeros_like(p['market_cap'], dtype=float)
    weights[:, 1:] = p['market_cap'][:, :-1]
    ok = np.isfinite(weights) & (weights > 0) & np.isfinite(p['r1']) & (np.abs(p['r1']) <= .35)
    weights = np.where(ok, weights, 0.)
    numerator = (weights * np.nan_to_num(p['r1'])).sum(0)
    ret = np.divide(numerator, weights.sum(0), out=np.zeros(weights.shape[1]), where=weights.sum(0) > 0)
    level = np.cumprod(1 + ret)
    sma = pd.Series(level).rolling(20, min_periods=20).mean().to_numpy()
    vol = pd.Series(ret).rolling(20, min_periods=20).std().to_numpy() * np.sqrt(252)
    trend = np.where(np.isfinite(sma) & (level >= sma), .8, .4)
    volatility = np.clip(.12 / np.maximum(np.nan_to_num(vol, nan=.4), .001), .3, .8)
    return {'none': None, 'trend': trend, 'volatility': volatility}, ret


def per_date_rank(values, di):
    return pd.Series(values).groupby(di).rank(pct=True).to_numpy(np.float32)


def load_scores(dest, p, ix, ci, di, spec):
    """Monthly checkpoint origins also define the online selector's decision times."""
    scores, metadata = {}, {}
    folds = monthly_folds(ix['dates'], spec['evaluation_start'])
    for cfg in spec['configs']:
        values = np.full(len(ci), np.nan, np.float32); metadata[cfg['id']] = {}
        for fold in folds:
            path = dest / 'models' / cfg['id'] / (fold['month'] + '.json')
            row = json.loads(path.read_text())
            if row['last_refit_label'] >= row['first_execution']: raise AssertionError('Future label in refit')
            pred = np.load(path.with_suffix('.pred.npy'))
            expected = (di >= fold['start'] - 1) & (di < fold['end'] - 1)
            if not np.array_equal(np.isfinite(pred), expected): raise AssertionError('Prediction dates differ')
            values[expected] = pred[expected]; metadata[cfg['id']][fold['month']] = row
        scores[cfg['id']] = values
    neural = [c['id'] for c in spec['configs'] if c['architecture'] in ('cnn', 'split', 'tcn')]
    controls = [c['id'] for c in spec['configs'] if c['architecture'] in ('mlp', 'gbm')]
    ranks = {name: per_date_rank(scores[name], di) for name in scores}
    choices = []
    for name, pool, count in [('online_neural', neural, 1), ('online_neural3', neural, 3),
                              ('online_nonimage', controls, 1)]:
        out = np.full(len(ci), np.nan, np.float32)
        for fold in folds:
            selected = sorted(pool, key=lambda n: (-metadata[n][fold['month']]['common_inner_ic'], n))[:count]
            mask = (di >= fold['start'] - 1) & (di < fold['end'] - 1)
            out[mask] = np.mean([ranks[n][mask] for n in selected], axis=0)
            choices.append({'selector': name, 'month': fold['month'], 'first_execution': ix['dates'][fold['start']],
                            'chosen': selected, 'past_inner_ic': [metadata[n][fold['month']]['common_inner_ic'] for n in selected]})
        scores[name] = out
    matrices = {name: score_matrix(values, ci, di, p['close'].shape) for name, values in scores.items()}
    matrices.update(momentum5=p['r5'], momentum20=p['r20'], reversal1=-p['r1'], reversal5=-p['r5'],
                    sector_reversal5=-(p['r5']-p['sector_r5']), lowvol=-p['vol20'], liquidity=p['adv20'])
    return matrices, choices, neural + ['online_neural', 'online_neural3']


def block_indices(n, length, draws, rng):
    starts = rng.integers(0, n, (draws, (n + length - 1) // length))
    return ((starts[:, :, None] + np.arange(length)) % n).reshape(draws, -1)[:, :n]


def family_bootstrap(differences, *, block=5, draws=4000, seed=57):
    """One-sided, centred circular-block max-t test; all columns form ONE family.

    The denominator is each column's block-bootstrap standard error of its mean.
    This is a generic max-t correction, not Hansen's SPA or a PBO estimate.
    """
    a = np.asarray(differences, float)
    if a.ndim != 2 or len(a) < 2 or not np.isfinite(a).all(): raise ValueError('Finite day-by-strategy array required')
    rng = np.random.default_rng(seed); mean = a.mean(0); centered = a - mean
    boot = np.empty((draws, a.shape[1]))
    for offset in range(0, draws, 100):
        idx = block_indices(len(a), block, min(100, draws-offset), rng)
        boot[offset:offset+len(idx)] = centered[idx].mean(1)
    se = np.maximum(boot.std(0, ddof=1), 1e-12)
    maxima = np.max(boot / se, axis=1)
    observed = mean / se
    adjusted = (1 + (maxima[:, None] >= observed).sum(0)) / (draws + 1)
    marginal = (1 + (boot / se >= observed).sum(0)) / (draws + 1)
    critical = np.quantile(maxima, .95)
    return {'mean': mean, 'standard_error': se, 'adjusted_p': adjusted, 'marginal_p': marginal,
            'simultaneous_lower95': mean - critical * se,
            'critical_max_t': float(critical), 'block': block, 'draws': draws}


def calendar_blocks(daily):
    frame = pd.DataFrame(daily); nav = frame.nav.to_numpy()
    previous = np.r_[INITIAL_NAV, nav[:-1]]
    quarters = pd.to_datetime(frame.date).dt.to_period('Q').astype(str)
    return {q: float(np.prod(nav[quarters == q] / previous[quarters == q]) - 1) for q in quarters.unique()}


def freeze_evaluation(dest):
    spec = {'bootstrap_draws': 4000, 'bootstrap_seed': 57, 'blocks': [5,10],
            'family': 'all CNN/split/TCN and two causal online neural selectors x four policies x two comparators',
            'comparators': ['cash', 'online_nonimage_same_policy'], 'alpha': .05,
            'test': 'one-sided centred circular date-block max-t; arithmetic daily excess return',
            'nonimage_selector_pool': 'two MLP and four GBM; best common inner daily IC',
            'interpretation': 'exploratory; reused observations and all earlier adaptive trials remain limitations',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = dest/'evaluation_protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Evaluation protocol changed')
    if not path.exists(): atomic_json(path, spec)
    return spec


def evaluate(dest=DEST, source=SOURCE):
    dest, source = Path(dest), Path(source)
    evaluation_spec = freeze_evaluation(dest); spec = json.loads((dest/'protocol.json').read_text())
    p, ix, ci, di = context(source); schedules, market_returns = market_regimes(p)
    scores, choices, neural = load_scores(dest,p,ix,ci,di,spec)
    atomic_json(dest/'online_choices.json', choices)
    pd.DataFrame({'date':ix['dates'], 'market_proxy_return':market_returns,
                  'trend_gross':schedules['trend'], 'volatility_gross':schedules['volatility']}).to_csv(dest/'regimes.csv',index=False)
    output = dest/'portfolios'; output.mkdir(exist_ok=True)
    paths, rows, returns = {}, [], {}
    for policy in spec['policies']:
        panel = dict(p); panel['sector_cap'] = np.minimum(p['sector_cap'],policy['sector_cap'])
        for name, matrix in scores.items():
            key = name+'__'+policy['id']; path = output/(key+'.json')
            if path.exists(): result = json.loads(path.read_text())
            else:
                result = replay(panel,ix,matrix,spec['evaluation_start'],spec['evaluation_end'],
                                top_n=policy['top_n'],weight=policy['stock_weight'],gross=.8,
                                gross_schedule=schedules[policy['regime']],max_orders=20,return_trades=True)
                atomic_json(path,result)
            paths[key] = result; nav = np.array([r['nav'] for r in result['daily']])
            returns[key] = nav / np.r_[INITIAL_NAV,nav[:-1]] - 1
            blocks = calendar_blocks(result['daily'])
            row = dict(id=key, model=name,policy=policy['id'],**result['metrics'],**blocks,
                       positive_blocks=sum(v > 0 for v in blocks.values()))
            rows.append(row)
        print(json.dumps({'policy_complete': policy['id']}),flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(dest/'portfolio_summary.csv',index=False)
    family, columns = [], []
    for policy in spec['policies']:
        control = returns['online_nonimage__'+policy['id']]
        for name in neural:
            key = name+'__'+policy['id']
            for comparator, baseline in [('cash',0.),('nonimage',control)]:
                family.append((key,comparator)); columns.append(returns[key]-baseline)
    matrix = np.column_stack(columns); stats_rows = []
    for block in evaluation_spec['blocks']:
        boot = family_bootstrap(matrix,block=block,draws=evaluation_spec['bootstrap_draws'],seed=evaluation_spec['bootstrap_seed'])
        for i,(key,comparator) in enumerate(family):
            stats_rows.append({'id':key,'comparator':comparator,'block':block,
                               **{k:float(boot[k][i]) for k in ['mean','standard_error','adjusted_p','marginal_p','simultaneous_lower95']}})
    statistics = pd.DataFrame(stats_rows); statistics.to_csv(dest/'bootstrap.csv',index=False)
    candidates = []
    for row in rows:
        if row['model'] not in neural: continue
        inference = statistics[statistics.id == row['id']]
        gate = row['return'] > 0 and row['positive_blocks'] >= 2 and row['mdd'] > -.20 and not row['four_week_turnover_stop']
        if gate and (inference.adjusted_p < .05).all() and (inference.simultaneous_lower95 > 0).all():
            candidates.append(row['id'])
    result = {'status':'development_only', 'portfolios':len(rows),'family_hypotheses':len(family),
              'days':len(matrix),'candidate_gate_passed':candidates,'independent_confirmation':False,
              'top_return':frame.sort_values('return',ascending=False).head(10).replace({np.nan:None}).to_dict('records')}
    atomic_json(dest/'evaluation_summary.json',result)
    print(json.dumps({'evaluation_complete':True,'candidate_gate_passed':candidates,'portfolios':len(rows)}),flush=True)
    return result


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dest',type=Path,default=DEST);ap.add_argument('--source',type=Path,default=SOURCE)
    args=ap.parse_args();evaluate(args.dest,args.source)

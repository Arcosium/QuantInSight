"""Fixed rank-retention policies with matched controls and all seed outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_rank_seeds import DEST as SEED_ROOT, TRAIN_PRIOR, RANK_ROOT, prior_family as seed_prior_family
from quant.timefolio_heatmap_rank_evaluation import CASES
from quant.timefolio_heatmap_seed_evaluation import merge_months
from quant.timefolio_heatmap_action_amendment import amend_actions, release_audit
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_retention_replay import replay
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_stock_limits import historical_stock_caps, audit_pre_july_hynix
from quant.timefolio_heatmap_week_boundaries import assess_weeks
from quant.timefolio_heatmap_study import context, score_matrix
from quant.timefolio_heatmap_walkforward import SOURCE, DEST as MODEL_ROOT
from quant.timefolio_heatmap_walkforward_eval import load_scores, calendar_blocks, family_bootstrap

DEST = SOURCE.with_name('20260929_retention_v1')
MULTIPLIERS = [0, 1, 2]


def freeze():
    parent = json.loads((SEED_ROOT / 'protocol.json').read_text())
    for manifest in [parent['hashes'], json.loads((SEED_ROOT / 'forecast_hashes.json').read_text())]:
        for filename, expected in manifest.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                raise AssertionError('Seed study input changed')
    rank = json.loads((RANK_ROOT / 'protocol.json').read_text())
    files = [Path(__file__), SEED_ROOT / 'protocol.json', SEED_ROOT / 'joint_bootstrap.csv',
             SEED_ROOT / 'forecast_hashes.json', SEED_ROOT / 'cnn_ensemble.npy', SEED_ROOT / 'mlp_ensemble.npy',
             RANK_ROOT / 'protocol.json', SOURCE / 'panel.npz', SOURCE / 'panel_index.json', SOURCE / 'samples.npz']
    files += sorted((SEED_ROOT / 'models').glob('*/*.json'))
    files += sorted((SEED_ROOT / 'portfolios').glob('*.json'))
    files += sorted((RANK_ROOT / 'portfolios').glob('online_nonimage__*.json'))
    files += sorted((MODEL_ROOT / 'models').glob('*/*.json')) + sorted((MODEL_ROOT / 'models').glob('*/*.pred.npy'))
    files += [Path(__file__).with_name('timefolio_heatmap_' + name + '.py') for name in
              ['retention_replay', 'dated_replay', 'rank_seeds', 'rank_evaluation', 'h10_control',
               'concentration', 'consensus', 'snapshot', 'stock_limits', 'week_boundaries', 'action_amendment',
               'execution_amendment', 'planned_audit', 'seed_evaluation', 'walkforward_eval', 'walkforward',
               'study', 'features', 'replay', 'data']]
    spec = dict(parent=parent, rank_parent=rank, cases=CASES, buffer_multipliers=MULTIPLIERS,
                policy='At each original target-construction event, rank eligible finite original scores with ticker ties. Prioritise actual held positions within top_n+buffer, retaining their original relative rank; then remaining candidates in original rank. Existing target allocator and all execution limits remain. Buffer is multiplier*top_n.',
                observations='Only pre-fill actual quantities and the existing lagged score snapshot are used. The execution-date admission check is still applied before the ranking pool. Original score ordering still governs buys; sell ordering and daily budgets are unchanged.',
                baseline='buffer0 must reproduce all72 frozen accounts in every field: 64 seed-study accounts and8 original online_nonimage accounts',
                forecasts='No training. All CNN/MLP seeds17/29/43 and fixed equal-rank ensembles, plus original causal online_nonimage. No best-seed selection.',
                execution='top4/12 x3/10dailyfills x1/5session score refresh; target5%, rebalance5, NAV5bp band, gross80%, sector min(statutory,20%), small-cap30%, dated stock caps and completed-holiday-week turnover',
                portfolios='9score streams x8cases x3buffers=216 accounts, of which144 are new and72 baseline regressions',
                family='3296 retained hypotheses plus4CNN members x8cases x2positivebuffers x4comparators=3552',
                comparators=['cash', 'matched_mlp_same_buffer', 'online_nonimage_same_buffer', 'same_cnn_buffer0'],
                bootstrap=dict(blocks=[5, 10], draws=4000, seed=57, alpha=.025),
                gate='Only fixed CNN ensemble can qualify. All4comparators/bothblocks adjustedp<0.025 and simultaneous lower95>0; ensemble and all3CNN seeds at the same case/buffer must each have positive return, MDD>-20%, >=2positivequarters, fewer than4turnover failures. Stress and fresh confirmation still required.',
                reference='https://qlib.readthedocs.io/en/latest/component/strategy.html; conceptual reference for separating scores and turnover policy, not an exact TopkDropout implementation or profitability evidence',
                context='Adaptive development after seed-dependent H10 ranking results. No reserved fresh outcomes. Existing universe/action/order-book limitations remain; retention changes realised exposures as well as costs.',
                hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    DEST.mkdir(exist_ok=True); path = DEST / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec:
        raise RuntimeError('Retention protocol changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST / 'frozen_source'; folder.mkdir(exist_ok=True)
        for i, p in enumerate(files):
            if p.suffix == '.py':
                shutil.copy2(p, folder / (f'{i:03d}_' + p.name))
    return spec


def prior_family():
    family, differences, returns, _ = seed_prior_family()
    prior = pd.read_csv(SEED_ROOT / 'joint_bootstrap.csv')
    for key, comp, origin in prior[prior.origin == 'rank_seeds'][['id', 'comparator', 'origin']].drop_duplicates().itertuples(index=False, name=None):
        case = key.split('__', 1)[1]
        if comp == 'cash': baseline = 0.
        elif comp == 'matched_mlp': baseline = returns(SEED_ROOT, key.replace('cnn_', 'mlp_', 1))
        elif comp == 'online_nonimage': baseline = returns(RANK_ROOT, 'online_nonimage__' + case)
        else: raise AssertionError('Unknown prior comparator')
        family.append((key, comp, origin)); differences.append(returns(SEED_ROOT, key) - baseline)
    if len(family) != 3296 or len(set(family)) != 3296:
        raise AssertionError('Prior retention family incomplete')
    means = prior[prior.block == 5].set_index(['id', 'comparator', 'origin'])['mean']
    error = max(abs(float(np.mean(x)) - means.loc[key]) for key, x in zip(family, differences))
    if error > 1e-14: raise AssertionError('Previous returns changed')
    return family, differences, returns, error


def run():
    spec = freeze(); p, ix, ci, di = context(SOURCE)
    p, release = amend_actions(p, ix, spec['parent']['training_parent']['actions'])
    old, choices, _ = load_scores(MODEL_ROOT, p, ix, ci, di, spec['rank_parent']['base']['prior']['model_protocol'])
    scores = {'online_nonimage': old['online_nonimage']}; del old
    atomic_json(DEST / 'online_choices.json', choices)
    for arch in ['cnn', 'mlp']:
        for seed in [17, 29, 43]:
            folder = (TRAIN_PRIOR / 'models' / (arch + '_pairwise_h10') if seed == 17 else
                      SEED_ROOT / 'models' / (arch + f'_pairwise_h10_seed{seed}'))
            scores[arch + f'_rank_h10_seed{seed}'] = score_matrix(merge_months(folder, di, ix['dates']), ci, di, p['close'].shape)
        scores[arch + '_rank_h10_ensemble'] = np.load(SEED_ROOT / (arch + '_ensemble.npy'))
    if len(scores) != 9: raise AssertionError('Missing forecast stream')
    score_dir = DEST / 'scores'; score_dir.mkdir(exist_ok=True)
    for name, matrix in scores.items():
        path = score_dir / (name + '.npy')
        if path.exists():
            if not np.array_equal(matrix, np.load(path), equal_nan=True): raise AssertionError('Scores changed')
        else: np.save(path, matrix)
    atomic_json(DEST / 'score_hashes.json', {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(score_dir.glob('*.npy'))})
    p['sector_cap'] = np.minimum(p['sector_cap'], .20); caps = historical_stock_caps(ix['codes'], ix['dates'])
    dates = {date: d for d, date in enumerate(ix['dates'])}; folder = DEST / 'portfolios'; folder.mkdir(exist_ok=True)
    rows = []; audits = {}; regressions = []
    for model, raw in scores.items():
        snapshots = {r: snapshot_scores(raw, p['eligible'], ix['dates'], '20260101', '20260923', r) for r in [1, 5]}
        for case in CASES:
            alpha, origins = snapshots[case['refresh']]
            basekey = model + '__' + case['id']
            for multiplier in MULTIPLIERS:
                buffer = multiplier * case['top_n']; key = basekey + f'__buffer{buffer}'; path = folder / (key + '.json')
                if path.exists(): result = json.loads(path.read_text())
                else:
                    result = replay(p, ix, alpha, '20260101', '20260923', top_n=case['top_n'], weight=.05,
                                    max_orders=case['max_orders'], rebalance=5, rebalance_band=.0005,
                                    return_trades=True, planning_price='open', action_release_dates=release,
                                    stock_cap_schedule=caps, rank_buffer=buffer)
                    for trade in result['trades']:
                        d = dates[trade['signal_date']]; origin = origins[d]
                        if not 0 <= origin <= d: raise AssertionError('Future retention score')
                        trade['portfolio_score_date'] = ix['dates'][origin]
                    result['metrics'].update(assess_weeks(result)); atomic_json(path, result)
                if buffer == 0:
                    root = RANK_ROOT if model == 'online_nonimage' else SEED_ROOT
                    original = json.loads((root / 'portfolios' / (basekey + '.json')).read_text())
                    if result != original: raise AssertionError('Zero buffer changed baseline account')
                    regressions.append(dict(id=key, all_fields_equal=True))
                check = audit_fills(p, ix, result)
                check.update(additional_checks(p, ix, result, None, max_orders=case['max_orders']))
                check['announced_action_errors'] = release_audit(p, ix, result, release)
                check['dated_hynix_audit'] = audit_pre_july_hynix(p, ix, result)
                check['snapshot_origin_errors'] = []
                for trade in result['trades']:
                    d = dates[trade['signal_date']]; origin = origins[d]
                    if not 0 <= origin <= d or trade['portfolio_score_date'] != ix['dates'][origin]:
                        check['snapshot_origin_errors'].append(trade['date'])
                if (check['post_buy_limit_violations'] or check['additional_errors'] or check['announced_action_errors']
                        or check['dated_hynix_audit']['post_buy_limit_errors'] or check['snapshot_origin_errors']
                        or check['maximum_nav_reconstruction_error_krw'] > .01):
                    atomic_json(DEST / 'failed_audit.json', dict(id=key, audit=check)); raise AssertionError('Retention audit failed')
                audits[key] = check; quarters = calendar_blocks(result['daily'])
                rows.append(dict(id=key, model=model, case=case['id'], buffer=buffer, multiplier=multiplier,
                                 **result['metrics'], **quarters, positive_blocks=sum(v > 0 for v in quarters.values())))
        print(json.dumps(dict(model=model, portfolios=len(rows))), flush=True)
    if len(regressions) != 72 or len(rows) != 216: raise AssertionError('Retention accounts incomplete')
    atomic_json(DEST / 'baseline_regression.json', regressions); atomic_json(DEST / 'independent_audit.json', audits)
    frame = pd.DataFrame(rows); frame.to_csv(DEST / 'portfolio_summary.csv', index=False)
    family, differences, returns, error = prior_family()
    atomic_json(DEST / 'prior_family_reconstruction.json', dict(hypotheses=len(family), maximum_mean_error=error))
    new = frame[frame.model.str.startswith('cnn_') & (frame.buffer > 0)]
    for row in new.to_dict('records'):
        suffix = '__' + row['case'] + f"__buffer{row['buffer']}"
        for comp, baseline in [('cash', 0.),
                               ('matched_mlp_same_buffer', returns(DEST, row['model'].replace('cnn_', 'mlp_', 1) + suffix)),
                               ('online_nonimage_same_buffer', returns(DEST, 'online_nonimage' + suffix)),
                               ('same_cnn_buffer0', returns(DEST, row['model'] + '__' + row['case'] + '__buffer0'))]:
            family.append((row['id'], comp, 'retention')); differences.append(returns(DEST, row['id']) - baseline)
    if len(family) != 3552 or len(set(family)) != 3552: raise AssertionError('Retention family changed')
    statistics = []
    for block in [5, 10]:
        result = family_bootstrap(np.column_stack(differences), block=block, draws=4000, seed=57)
        for i, (key, comp, origin) in enumerate(family):
            statistics.append(dict(id=key, comparator=comp, origin=origin, block=block,
                                   **{k: float(result[k][i]) for k in ['mean', 'standard_error', 'adjusted_p', 'marginal_p', 'simultaneous_lower95']}))
    stats = pd.DataFrame(statistics); stats.to_csv(DEST / 'joint_bootstrap.csv', index=False); candidates = []
    for row in new[new.model == 'cnn_rank_h10_ensemble'].to_dict('records'):
        infer = stats[(stats.id == row['id']) & (stats.origin == 'retention')]
        members = new[(new.case == row['case']) & (new.buffer == row['buffer'])]
        if len(infer) != 8 or len(members) != 4: raise AssertionError('Missing robustness comparisons')
        stable = ((members['return'] > 0) & (members.mdd > -.20) & (members.positive_blocks >= 2) & (~members.four_week_turnover_stop)).all()
        if stable and (infer.adjusted_p < .025).all() and (infer.simultaneous_lower95 > 0).all(): candidates.append(row['id'])
    atomic_json(DEST / 'evaluation_summary.json', dict(status='development_only', portfolios=216, new_portfolios=144,
                baseline_regression_accounts=72, joint_hypotheses=len(family), robust_candidate_gate_passed=candidates,
                independent_confirmation=False))
    print(json.dumps(dict(complete=True, robust_candidates=candidates)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['freeze', 'run']); args = ap.parse_args()
    if args.action == 'freeze': freeze()
    else: run()

"""Top-rank objective evaluation with uniform-loss and untrained controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_toprank_training import DEST as TRAIN_ROOT, DATA_ROOT, CONFIGS, EVALUATION_PLAN
from quant.timefolio_heatmap_absolute_evaluation import DEST as PRIOR, prior_family as absolute_prior_family
from quant.timefolio_heatmap_initialisation_probe import DEST as UNTRAINED
from quant.timefolio_heatmap_seed_evaluation import merge_months
from quant.timefolio_heatmap_consensus import rank_consensus
from quant.timefolio_heatmap_action_amendment import amend_actions, release_audit
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_retention_replay import replay
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_stock_limits import historical_stock_caps, audit_pre_july_hynix
from quant.timefolio_heatmap_week_boundaries import assess_weeks
from quant.timefolio_heatmap_study import context, score_matrix
from quant.timefolio_heatmap_walkforward import SOURCE, AlternativeNet, predict
from quant.timefolio_heatmap_walkforward_eval import calendar_blocks, family_bootstrap

DEST = SOURCE.with_name('20260929_toprank_evaluation_v1')
LEGACY = [f'{a}{middle}{m}' for middle in ['_rank_h10_', '_absolute_rank_h10_']
          for a in ['cnn', 'mlp'] for m in ['seed17', 'seed29', 'seed43', 'ensemble']] + ['online_nonimage']
UNTRAINED_NAMES = [f'{a}_untrained_{m}' for a in ['cnn', 'mlp'] for m in ['seed17', 'seed29', 'seed43', 'ensemble']]


def freeze():
    training = json.loads((TRAIN_ROOT / 'protocol.json').read_text())
    if training['evaluation_plan'] != EVALUATION_PLAN: raise AssertionError('Top-rank evaluation plan changed')
    for manifest in [training['hashes'], json.loads((DATA_ROOT / 'data_hashes.json').read_text()),
                     json.loads((PRIOR / 'protocol.json').read_text())['hashes'],
                     json.loads((PRIOR / 'score_hashes.json').read_text()),
                     json.loads((UNTRAINED / 'protocol.json').read_text())['hashes'],
                     json.loads((UNTRAINED / 'output_hashes.json').read_text())]:
        for filename, sha in manifest.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != sha: raise AssertionError('Top-rank evaluation dependency changed')
    files = [Path(__file__), TRAIN_ROOT / 'protocol.json', DATA_ROOT / 'data_hashes.json',
             PRIOR / 'joint_bootstrap.csv', PRIOR / 'protocol.json', PRIOR / 'validation_complete.json', UNTRAINED / 'protocol.json',
             UNTRAINED / 'validation_complete.json', UNTRAINED / 'output_hashes.json']
    for name in LEGACY:
        files.append(PRIOR / 'scores' / (name + '.npy'))
        files += sorted((PRIOR / 'portfolios').glob(name + '__*.json'))
    files += [UNTRAINED / 'scores' / (name + '.npy') for name in UNTRAINED_NAMES]
    files += [Path(__file__).with_name('timefolio_heatmap_' + n + '.py') for n in
              ['toprank_training', 'toprank', 'absolute_evaluation', 'absolute_training', 'peer_evaluation', 'peer_training', 'corrected_evaluation', 'corrected_training',
               'retention', 'retention_replay', 'rank_seeds', 'rank_evaluation', 'h10_control', 'concentration',
               'consensus', 'snapshot', 'stock_limits', 'week_boundaries', 'action_amendment', 'execution_amendment',
               'planned_audit', 'seed_evaluation', 'walkforward_eval', 'walkforward', 'study', 'features', 'replay', 'data']]
    spec = dict(training=training, plan=EVALUATION_PLAN,
                legacy='408 matched absolute/sector and online accounts must reproduce every field.',
                hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    DEST.mkdir(exist_ok=True); path = DEST / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Top-rank evaluation protocol changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST / 'frozen_source'; folder.mkdir(exist_ok=True)
        for i, p in enumerate(files):
            if p.suffix == '.py': shutil.copy2(p, folder / (f'{i:03d}_' + p.name))
    return spec


def prior_family():
    family, differences, returns, _ = absolute_prior_family()
    prior = pd.read_csv(PRIOR / 'joint_bootstrap.csv')
    added = prior[prior.origin == 'absolute_target'][['id', 'comparator', 'origin']].drop_duplicates()
    for key, comp, origin in added.itertuples(index=False, name=None):
        model, suffix = key.split('__', 1)
        if comp == 'cash': baseline = 0.
        elif comp == 'matched_absolute_mlp': baseline = returns(PRIOR, 'mlp' + key[3:])
        elif comp == 'online_nonimage': baseline = returns(PRIOR, 'online_nonimage__' + suffix)
        elif comp == 'matched_sector_cnn': baseline = returns(PRIOR, 'cnn_rank_h10_' + model.split('_')[-1] + '__' + suffix)
        elif comp == 'same_absolute_cnn_buffer0': baseline = returns(PRIOR, key.rsplit('__buffer', 1)[0] + '__buffer0')
        else: raise AssertionError('Unknown retained absolute comparator')
        family.append((key, comp, origin)); differences.append(returns(PRIOR, key) - baseline)
    if len(family) != 5344 or len(set(family)) != 5344: raise AssertionError('Top-rank prior family incomplete')
    means = prior[prior.block == 5].set_index(['id', 'comparator', 'origin'])['mean']
    error = max(abs(float(np.mean(x)) - means.loc[key]) for key, x in zip(family, differences))
    if error > 1e-14: raise AssertionError('Retained absolute means changed')
    return family, differences, returns, error


def run():
    spec = freeze(); torch.set_num_threads(4)
    complete = json.loads((TRAIN_ROOT / 'training_complete.json').read_text())
    if complete['folds'] != 108 or complete['newly_fitted'] != 108: raise AssertionError('Top-rank training incomplete')
    hashes = json.loads((TRAIN_ROOT / 'forecast_hashes.json').read_text())
    if len(hashes) != 108: raise AssertionError('Missing top-rank forecasts')
    for filename, sha in hashes.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != sha: raise AssertionError('Top-rank forecast changed')
    atomic_json(DEST / 'forecast_hashes.json', hashes)
    p, ix, ci, di = context(DATA_ROOT)
    actions = spec['training']['parents']['sector']['data_protocol']['parent']['training_parent']['actions']
    p, release = amend_actions(p, ix, actions)
    x = torch.from_numpy(np.array(np.load(DATA_ROOT / 'images_raw.npy', mmap_mode='r')[:, None], copy=True))
    matrices, reproduced = {}, []
    for cfg in CONFIGS:
        folder = TRAIN_ROOT / 'models' / cfg['id']; path = folder / '202609.pt'
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['config'] != cfg: raise AssertionError('Top-rank checkpoint configuration changed')
        model = AlternativeNet(cfg['architecture']); model.load_state_dict(saved['state_dict'])
        expected = np.load(path.with_suffix('.pred.npy')); ids = np.flatnonzero(np.isfinite(expected))
        error = float(np.max(np.abs(predict(model, x, ids) - expected[ids]))); del model
        if error > 1e-6: raise AssertionError('Top-rank checkpoint reinference failed')
        reproduced.append(dict(model=cfg['id'], rows=len(ids), maximum_error=error))
        matrices[cfg['id']] = score_matrix(merge_months(folder, di, ix['dates']), ci, di, p['close'].shape)
    del x
    for arch in ['cnn', 'mlp']:
        for target in ['sector', 'absolute']:
            prefix = arch + '_toprank_' + target + '_h10_'
            members = {k: v for k, v in matrices.items() if k.startswith(prefix)}
            if len(members) != 3: raise AssertionError('Missing top-rank seed')
            matrices[prefix + 'ensemble'] = rank_consensus(members, {k: arch for k in members}, p['eligible'], 'mean')
    for name in LEGACY: matrices[name] = np.load(PRIOR / 'scores' / (name + '.npy'))
    for name in UNTRAINED_NAMES: matrices[name] = np.load(UNTRAINED / 'scores' / (name + '.npy'))
    if len(matrices) != 41: raise AssertionError('Incomplete top-rank score grid')
    atomic_json(DEST / 'checkpoint_reproduction.json', reproduced)
    score_dir = DEST / 'scores'; score_dir.mkdir(exist_ok=True)
    for name, values in matrices.items(): np.save(score_dir / (name + '.npy'), values)
    atomic_json(DEST / 'score_hashes.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(score_dir.glob('*.npy'))})
    p['sector_cap'] = np.minimum(p['sector_cap'], .20); caps = historical_stock_caps(ix['codes'], ix['dates'])
    dates = {d: j for j, d in enumerate(ix['dates'])}; rows, audits, regressions = [], {}, []
    folder = DEST / 'portfolios'; folder.mkdir(exist_ok=True)
    for name, matrix in matrices.items():
        held = {r: snapshot_scores(matrix, p['eligible'], ix['dates'], '20260101', '20260923', r) for r in [1, 5]}
        variant = 'legacy' if name in LEGACY else ('untrained' if name in UNTRAINED_NAMES else name.split('_')[2])
        for case in EVALUATION_PLAN['cases']:
            alpha, origins = held[case['refresh']]
            for multiplier in EVALUATION_PLAN['buffer_multipliers']:
                buffer = multiplier * case['top_n']; key = name + '__' + case['id'] + f'__buffer{buffer}'
                path = folder / (key + '.json')
                if path.exists(): raise RuntimeError('Do not overwrite a completed top-rank account')
                result = replay(p, ix, alpha, '20260101', '20260923', top_n=case['top_n'], weight=.05,
                                max_orders=case['max_orders'], rebalance=5, rebalance_band=.0005,
                                return_trades=True, planning_price='open', action_release_dates=release,
                                stock_cap_schedule=caps, rank_buffer=buffer)
                for trade in result['trades']:
                    d = dates[trade['signal_date']]; origin = origins[d]
                    if not 0 <= origin <= d: raise AssertionError('Future top-rank score')
                    trade['portfolio_score_date'] = ix['dates'][origin]
                result['metrics'].update(assess_weeks(result)); atomic_json(path, result)
                if name in LEGACY:
                    if result != json.loads((PRIOR / 'portfolios' / path.name).read_text()): raise AssertionError('Matched legacy account changed')
                    regressions.append(dict(id=key, all_fields_equal=True))
                check = audit_fills(p, ix, result); check.update(additional_checks(p, ix, result, None, max_orders=case['max_orders']))
                check['announced_action_errors'] = release_audit(p, ix, result, release)
                check['dated_hynix_audit'] = audit_pre_july_hynix(p, ix, result); check['snapshot_origin_errors'] = []
                for trade in result['trades']:
                    d = dates[trade['signal_date']]; origin = origins[d]
                    if not 0 <= origin <= d or trade['portfolio_score_date'] != ix['dates'][origin]: check['snapshot_origin_errors'].append(trade['date'])
                if (check['post_buy_limit_violations'] or check['additional_errors'] or check['announced_action_errors']
                        or check['dated_hynix_audit']['post_buy_limit_errors'] or check['snapshot_origin_errors']
                        or check['maximum_nav_reconstruction_error_krw'] > .01):
                    atomic_json(DEST / 'failed_audit.json', dict(id=key, audit=check)); raise AssertionError('Top-rank account audit failed')
                audits[key] = check; quarters = calendar_blocks(result['daily'])
                rows.append(dict(id=key, model=name, variant=variant, case=case['id'], buffer=buffer, multiplier=multiplier,
                                 **result['metrics'], **quarters, positive_blocks=sum(v > 0 for v in quarters.values())))
        print(json.dumps(dict(model=name, portfolios=len(rows))), flush=True)
    if len(rows) != 984 or len(regressions) != 408: raise AssertionError('Top-rank account grid incomplete')
    atomic_json(DEST / 'independent_audit.json', audits); atomic_json(DEST / 'baseline_regression.json', regressions)
    frame = pd.DataFrame(rows); frame.to_csv(DEST / 'portfolio_summary.csv', index=False)
    family, differences, returns, error = prior_family()
    atomic_json(DEST / 'prior_family_reconstruction.json', dict(hypotheses=len(family), maximum_mean_error=error))
    new = frame[frame.model.str.startswith('cnn_toprank_')]
    for row in new.to_dict('records'):
        suffix = '__' + row['case'] + f"__buffer{row['buffer']}"; member = row['model'].rsplit('_', 1)[1]
        target = row['variant']
        uniform = 'cnn_' + ('absolute_' if target == 'absolute' else '') + 'rank_h10_' + member
        comparisons = [('cash', 0.), ('matched_mlp_same_target', returns(DEST, row['model'].replace('cnn_', 'mlp_', 1) + suffix)),
                       ('online_nonimage', returns(DEST, 'online_nonimage' + suffix)),
                       ('matched_uniform_cnn', returns(DEST, uniform + suffix)),
                       ('matched_untrained_cnn', returns(DEST, 'cnn_untrained_' + member + suffix))]
        if row['buffer'] > 0: comparisons.append(('same_toprank_cnn_buffer0', returns(DEST, row['model'] + '__' + row['case'] + '__buffer0')))
        for comp, baseline in comparisons:
            family.append((row['id'], comp, 'toprank_objective')); differences.append(returns(DEST, row['id']) - baseline)
    if len(family) != 6432 or len(set(family)) != 6432: raise AssertionError('Top-rank joint family differs')
    statistics = []
    for block in [5, 10]:
        result = family_bootstrap(np.column_stack(differences), block=block, draws=4000, seed=57)
        for i, (key, comp, origin) in enumerate(family):
            statistics.append(dict(id=key, comparator=comp, origin=origin, block=block,
                                   **{k: float(result[k][i]) for k in ['mean', 'standard_error', 'adjusted_p', 'marginal_p', 'simultaneous_lower95']}))
    stats = pd.DataFrame(statistics); stats.to_csv(DEST / 'joint_bootstrap.csv', index=False); candidates = []
    for row in new[new.model.str.endswith('_ensemble')].to_dict('records'):
        infer = stats[(stats.id == row['id']) & (stats.origin == 'toprank_objective')]
        members = new[(new.case == row['case']) & (new.buffer == row['buffer']) & (new.variant == row['variant'])]
        if len(infer) != 2 * (5 + (row['buffer'] > 0)) or len(members) != 4: raise AssertionError('Incomplete top-rank comparisons')
        stable = ((members['return'] > 0) & (members.mdd > -.20) & (members.positive_blocks >= 2) & (~members.four_week_turnover_stop)).all()
        if stable and (infer.adjusted_p < .025).all() and (infer.simultaneous_lower95 > 0).all(): candidates.append(row['id'])
    atomic_json(DEST / 'evaluation_summary.json', dict(status='development_only', portfolios=984, new_portfolios=576, new_toprank_portfolios=384, new_untrained_portfolios=192,
                legacy_regressions=408, joint_hypotheses=len(family), robust_candidate_gate_passed=candidates, independent_confirmation=False))
    print(json.dumps(dict(complete=True, robust_candidates=candidates)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['freeze', 'run']); args = ap.parse_args()
    if args.action == 'freeze': freeze()
    else: run()

"""Evaluate corrected inputs with the full seed/retention grid and old family."""
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
from quant.timefolio_heatmap_corrected_features import DEST as DATA_ROOT
from quant.timefolio_heatmap_corrected_training import DEST as TRAIN_ROOT, CONFIGS
from quant.timefolio_heatmap_retention import DEST as PRIOR, prior_family as retention_prior_family, MULTIPLIERS
from quant.timefolio_heatmap_rank_evaluation import CASES
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

DEST = SOURCE.with_name('20260929_corrected_evaluation_v1')


def freeze():
    training = json.loads((TRAIN_ROOT / 'protocol.json').read_text())
    for filename, value in training['hashes'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != value: raise AssertionError('Training dependency changed')
    for filename, value in json.loads((DATA_ROOT / 'data_hashes.json').read_text()).items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != value: raise AssertionError('Corrected data changed')
    files = [Path(__file__), TRAIN_ROOT / 'protocol.json', PRIOR / 'protocol.json', PRIOR / 'joint_bootstrap.csv',
             PRIOR / 'scores' / 'online_nonimage.npy', DATA_ROOT / 'data_hashes.json']
    files += sorted((PRIOR / 'portfolios').glob('online_nonimage__*.json'))
    files += [Path(__file__).with_name('timefolio_heatmap_' + n + '.py') for n in
              ['corrected_training', 'corrected_features', 'retention', 'retention_replay', 'rank_seeds',
               'rank_evaluation', 'h10_control', 'concentration', 'consensus', 'snapshot', 'stock_limits',
               'week_boundaries', 'action_amendment', 'execution_amendment', 'planned_audit', 'seed_evaluation',
               'walkforward_eval', 'walkforward', 'study', 'features', 'replay', 'data']]
    spec = dict(training=training, cases=CASES, buffer_multipliers=MULTIPLIERS,
                accounts='216 accounts: all6 corrected CNN/MLP forecasts,2fixed rank ensembles and the fixed original online_nonimage,8cases,3buffers.24legacy online accounts must reproduce exactly;192 corrected-model accounts.',
                family='3552 old hypotheses retained;4CNN members x8cases x(3comparators at buffer0 +4comparators at each of2positive buffers)=352 new;3904 total',
                comparators='cash, matched corrected MLP at same case/buffer, fixed original online_nonimage at same case/buffer; positive buffers also same corrected CNN at buffer0',
                interpretation='Known input correction with matched corrected MLP. Legacy online comparator remains fixed. Reused development dates; no source-based or data-based selection by profit.',
                gate=training['gate'], bootstrap=dict(blocks=[5,10], draws=4000, seed=57, alpha=.025),
                hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    DEST.mkdir(exist_ok=True); path = DEST / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Corrected evaluation protocol changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST / 'frozen_source'; folder.mkdir(exist_ok=True)
        for i, p in enumerate(files):
            if p.suffix == '.py': shutil.copy2(p, folder / (f'{i:03d}_' + p.name))
    return spec


def prior_family():
    family, differences, returns, _ = retention_prior_family()
    prior = pd.read_csv(PRIOR / 'joint_bootstrap.csv')
    added = prior[prior.origin == 'retention'][['id', 'comparator', 'origin']].drop_duplicates()
    for key, comp, origin in added.itertuples(index=False, name=None):
        suffix = key.split('__', 1)[1]
        if comp == 'cash': baseline = 0.
        elif comp == 'matched_mlp_same_buffer': baseline = returns(PRIOR, key.replace('cnn_', 'mlp_', 1))
        elif comp == 'online_nonimage_same_buffer': baseline = returns(PRIOR, 'online_nonimage__' + suffix)
        elif comp == 'same_cnn_buffer0': baseline = returns(PRIOR, key.rsplit('__buffer', 1)[0] + '__buffer0')
        else: raise AssertionError('Unknown retained comparator')
        family.append((key, comp, origin)); differences.append(returns(PRIOR, key) - baseline)
    if len(family) != 3552 or len(set(family)) != 3552: raise AssertionError('Retained corrected family incomplete')
    means = prior[prior.block == 5].set_index(['id', 'comparator', 'origin'])['mean']
    error = max(abs(float(np.mean(x)) - means.loc[key]) for key, x in zip(family, differences))
    if error > 1e-14: raise AssertionError('Retained means differ')
    return family, differences, returns, error


def run():
    spec = freeze(); torch.set_num_threads(4)
    complete = json.loads((TRAIN_ROOT / 'training_complete.json').read_text())
    if complete['folds'] != 54: raise AssertionError('Training incomplete')
    hashes = json.loads((TRAIN_ROOT / 'forecast_hashes.json').read_text())
    if len(hashes) != 54: raise AssertionError('Missing corrected forecasts')
    for filename, value in hashes.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != value: raise AssertionError('Corrected forecast changed')
    atomic_json(DEST / 'forecast_hashes.json', hashes)
    p, ix, ci, di = context(DATA_ROOT)
    p, release = amend_actions(p, ix, spec['training']['data_protocol']['parent']['training_parent']['actions'])
    matrices = {}; reproduced = []
    x = torch.from_numpy(np.array(np.load(DATA_ROOT / 'images_raw.npy', mmap_mode='r')[:, None], copy=True))
    for cfg in CONFIGS:
        folder = TRAIN_ROOT / 'models' / cfg['id']; checkpoint = folder / '202609.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['config'] != cfg: raise AssertionError('Corrected checkpoint config changed')
        model = AlternativeNet(cfg['architecture']); model.load_state_dict(saved['state_dict'])
        expected = np.load(checkpoint.with_suffix('.pred.npy')); ids = np.flatnonzero(np.isfinite(expected))
        error = float(np.max(np.abs(predict(model, x, ids) - expected[ids]))); del model
        if error > 1e-6: raise AssertionError('Corrected checkpoint reinference failed')
        reproduced.append(dict(model=cfg['id'], rows=len(ids), maximum_error=error))
        name = cfg['architecture'] + f"_rank_h10_seed{cfg['seed']}"
        matrices[name] = score_matrix(merge_months(folder, di, ix['dates']), ci, di, p['close'].shape)
    del x
    for arch in ['cnn', 'mlp']:
        members = {k: v for k, v in matrices.items() if k.startswith(arch + '_')}
        if len(members) != 3: raise AssertionError('Missing corrected seed')
        matrices[arch + '_rank_h10_ensemble'] = rank_consensus(members, {k: arch for k in members}, p['eligible'], 'mean')
    matrices['online_nonimage'] = np.load(PRIOR / 'scores' / 'online_nonimage.npy')
    atomic_json(DEST / 'checkpoint_reproduction.json', reproduced)
    score_dir = DEST / 'scores'; score_dir.mkdir(exist_ok=True)
    for name, matrix in matrices.items(): np.save(score_dir / (name + '.npy'), matrix)
    atomic_json(DEST / 'score_hashes.json', {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(score_dir.glob('*.npy'))})
    p['sector_cap'] = np.minimum(p['sector_cap'], .20); caps = historical_stock_caps(ix['codes'], ix['dates'])
    dates = {d: j for j, d in enumerate(ix['dates'])}; rows = []; audits = {}; regressions = []
    folder = DEST / 'portfolios'; folder.mkdir(exist_ok=True)
    for name, matrix in matrices.items():
        held = {r: snapshot_scores(matrix, p['eligible'], ix['dates'], '20260101', '20260923', r) for r in [1, 5]}
        for case in CASES:
            alpha, origins = held[case['refresh']]
            for multiplier in MULTIPLIERS:
                buffer = multiplier * case['top_n']; key = name + '__' + case['id'] + f'__buffer{buffer}'
                path = folder / (key + '.json')
                if path.exists(): result = json.loads(path.read_text())
                else:
                    result = replay(p, ix, alpha, '20260101', '20260923', top_n=case['top_n'], weight=.05,
                                    max_orders=case['max_orders'], rebalance=5, rebalance_band=.0005,
                                    return_trades=True, planning_price='open', action_release_dates=release,
                                    stock_cap_schedule=caps, rank_buffer=buffer)
                    for trade in result['trades']:
                        d = dates[trade['signal_date']]; origin = origins[d]
                        if not 0 <= origin <= d: raise AssertionError('Future corrected score')
                        trade['portfolio_score_date'] = ix['dates'][origin]
                    result['metrics'].update(assess_weeks(result)); atomic_json(path, result)
                if name == 'online_nonimage':
                    if result != json.loads((PRIOR / 'portfolios' / path.name).read_text()): raise AssertionError('Legacy online account changed')
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
                    atomic_json(DEST / 'failed_audit.json', dict(id=key, audit=check)); raise AssertionError('Corrected account audit failed')
                audits[key] = check; quarters = calendar_blocks(result['daily'])
                rows.append(dict(id=key, model=name, case=case['id'], buffer=buffer, multiplier=multiplier,
                                 **result['metrics'], **quarters, positive_blocks=sum(v > 0 for v in quarters.values())))
        print(json.dumps(dict(model=name, portfolios=len(rows))), flush=True)
    if len(rows) != 216 or len(regressions) != 24: raise AssertionError('Corrected accounts incomplete')
    atomic_json(DEST / 'independent_audit.json', audits); atomic_json(DEST / 'baseline_regression.json', regressions)
    frame = pd.DataFrame(rows); frame.to_csv(DEST / 'portfolio_summary.csv', index=False)
    family, differences, returns, error = prior_family()
    atomic_json(DEST / 'prior_family_reconstruction.json', dict(hypotheses=len(family), maximum_mean_error=error))
    new = frame[frame.model.str.startswith('cnn_')]
    for row in new.to_dict('records'):
        suffix = '__' + row['case'] + f"__buffer{row['buffer']}"
        comparisons = [('cash', 0.), ('matched_mlp_same_buffer', returns(DEST, row['model'].replace('cnn_', 'mlp_', 1) + suffix)),
                       ('online_nonimage_same_buffer', returns(DEST, 'online_nonimage' + suffix))]
        if row['buffer'] > 0: comparisons.append(('same_cnn_buffer0', returns(DEST, row['model'] + '__' + row['case'] + '__buffer0')))
        for comp, baseline in comparisons:
            family.append((row['id'], comp, 'corrected_features')); differences.append(returns(DEST, row['id']) - baseline)
    if len(family) != 3904 or len(set(family)) != 3904: raise AssertionError('Corrected family differs')
    statistics = []
    for block in [5, 10]:
        result = family_bootstrap(np.column_stack(differences), block=block, draws=4000, seed=57)
        for i, (key, comp, origin) in enumerate(family):
            statistics.append(dict(id=key, comparator=comp, origin=origin, block=block,
                                   **{k: float(result[k][i]) for k in ['mean', 'standard_error', 'adjusted_p', 'marginal_p', 'simultaneous_lower95']}))
    stats = pd.DataFrame(statistics); stats.to_csv(DEST / 'joint_bootstrap.csv', index=False); candidates = []
    for row in new[new.model == 'cnn_rank_h10_ensemble'].to_dict('records'):
        infer = stats[(stats.id == row['id']) & (stats.origin == 'corrected_features')]
        members = new[(new.case == row['case']) & (new.buffer == row['buffer'])]
        if len(infer) != (8 if row['buffer'] else 6) or len(members) != 4: raise AssertionError('Incomplete corrected comparisons')
        stable = ((members['return'] > 0) & (members.mdd > -.20) & (members.positive_blocks >= 2) & (~members.four_week_turnover_stop)).all()
        if stable and (infer.adjusted_p < .025).all() and (infer.simultaneous_lower95 > 0).all(): candidates.append(row['id'])
    atomic_json(DEST / 'evaluation_summary.json', dict(status='development_only', portfolios=216, corrected_model_portfolios=192,
                unchanged_legacy_accounts=24, joint_hypotheses=len(family), robust_candidate_gate_passed=candidates, independent_confirmation=False))
    print(json.dumps(dict(complete=True, robust_candidates=candidates)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['freeze', 'run']); a = ap.parse_args()
    if a.action == 'freeze': freeze()
    else: run()

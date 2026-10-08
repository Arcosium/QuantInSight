"""Matched relation-model accounts with the full retained development family."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from quant.timefolio_heatmap_stock_relation_training import (
    DEST as TRAIN_ROOT, PRIOR, DATA_ROOT, CONFIGS, parent_ready, check_hashes, digest,
)
from quant.timefolio_heatmap_stock_relation import StockRelationNet, predict_by_date
from quant.timefolio_heatmap_adjustment_band import (
    MODELS as LEGACY, CASES, BUFFERS, prior_family as band_prior_family, new_family as band_new_family,
    model_info as legacy_info,
)
from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_action_amendment import amend_actions, release_audit
from quant.timefolio_heatmap_consensus import rank_consensus
from quant.timefolio_heatmap_seed_evaluation import merge_months
from quant.timefolio_heatmap_study import context, score_matrix
from quant.timefolio_heatmap_walkforward import monthly_folds
from quant.timefolio_heatmap_walkforward_eval import calendar_blocks, family_bootstrap
from quant.timefolio_heatmap_retention_replay import replay
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_stock_limits import historical_stock_caps, audit_pre_july_hynix
from quant.timefolio_heatmap_week_boundaries import assess_weeks

DEST = TRAIN_ROOT.with_name('20260929_stock_relation_evaluation_v1')
MEMBERS = ['seed17', 'seed29', 'seed43', 'ensemble']
TRAINED = [f'{arch}_relation_{relation}_h10_{member}' for arch in ['cnn', 'mlp']
           for relation in ['self', 'mean', 'attention'] for member in MEMBERS]
UNTRAINED = [f'cnn_relation_{relation}_untrained_{member}' for relation in ['self', 'mean', 'attention'] for member in MEMBERS]
MODELS = TRAINED + UNTRAINED + LEGACY


def model_info(model):
    if model in LEGACY: return dict(**legacy_info(model), relation='legacy')
    if model not in TRAINED + UNTRAINED: raise ValueError('Unknown relation model')
    arch, _, relation, objective, member = model.split('_')
    return dict(architecture=arch, relation=relation, objective='relation' if objective == 'h10' else 'relation_untrained',
                target='absolute' if objective == 'h10' else 'control', member=member)


def account_id(model, case, buffer):
    if model not in MODELS or case not in {c['id'] for c in CASES} or buffer not in BUFFERS:
        raise ValueError('Unknown relation account')
    return f'{model}__{case}__buffer{buffer}__band5bp'


def comparison_ids(model, case, buffer):
    if model not in TRAINED or not model.startswith('cnn_'): raise ValueError('Trained relation CNN required')
    info = model_info(model); key = lambda name: account_id(name, case, buffer)
    pairs = [('cash', None), ('matched_mlp', key('mlp' + model[3:])),
             ('matched_untrained_cnn', key(model.replace('_h10_', '_untrained_', 1))),
             ('matched_uniform_cnn', key('cnn_absolute_rank_h10_' + info['member'])),
             ('online_nonimage', key('online_nonimage'))]
    if info['relation'] in ['mean', 'attention']:
        pairs.append(('matched_self_cnn', key(f"cnn_relation_self_h10_{info['member']}")))
    if info['relation'] == 'attention':
        pairs.append(('matched_mean_cnn', key(f"cnn_relation_mean_h10_{info['member']}")))
    return pairs


def new_family():
    family = []
    for model in TRAINED:
        if not model.startswith('cnn_'): continue
        for case in CASES:
            for buffer in BUFFERS:
                key = account_id(model, case['id'], buffer)
                family.extend((key, comp, reference) for comp, reference in comparison_ids(model, case['id'], buffer))
    if len(family) != 576 or len({(key, comp) for key, comp, _ in family}) != 576:
        raise AssertionError('Incomplete relation comparison grid')
    return family


def prior_family():
    family, differences, returns, _ = band_prior_family()
    stored = pd.read_csv(PRIOR / 'joint_bootstrap.csv')
    added = []
    for key, comp, reference in band_new_family():
        record = (key, comp, 'adjustment_band'); added.append(record); family.append(record)
        differences.append(returns(PRIOR, key) - (returns(PRIOR, reference) if reference is not None else 0.))
    if set(added) != set(stored[stored.origin == 'adjustment_band'][['id', 'comparator', 'origin']].itertuples(index=False, name=None)):
        raise AssertionError('Retained band comparison set changed')
    if len(family) != 11168 or len(set(family)) != 11168: raise AssertionError('Incomplete relation prior family')
    means = stored[stored.block == 5].set_index(['id', 'comparator', 'origin'])['mean']
    error = max(abs(float(np.mean(value)) - means.loc[key]) for key, value in zip(family, differences))
    if error > 1e-14: raise AssertionError('Retained band means changed')
    return family, differences, returns, error


def freeze():
    parent_ready(); training = json.loads((TRAIN_ROOT / 'protocol.json').read_text())
    parent = json.loads((PRIOR / 'protocol.json').read_text())
    for manifest in [training['hashes'], parent['hashes'], json.loads((PRIOR / 'score_hashes.json').read_text()),
                     json.loads((DATA_ROOT / 'data_hashes.json').read_text())]: check_hashes(manifest)
    files = [Path(__file__), TRAIN_ROOT / 'protocol.json', PRIOR / 'protocol.json', PRIOR / 'validation_complete.json',
             PRIOR / 'score_hashes.json', PRIOR / 'joint_bootstrap.csv', DATA_ROOT / 'data_hashes.json']
    files += [Path(__file__).with_name('timefolio_heatmap_' + name + '.py') for name in
              ['stock_relation', 'stock_relation_training', 'adjustment_band', 'retention_replay', 'snapshot', 'stock_limits',
               'planned_audit', 'action_amendment', 'week_boundaries', 'seed_evaluation', 'consensus', 'walkforward_eval',
               'walkforward', 'study', 'replay', 'data']]
    for model in LEGACY:
        files.append(PRIOR / 'scores' / (model + '.npy'))
        files += [PRIOR / 'portfolios' / (account_id(model, case['id'], buffer) + '.json') for case in CASES for buffer in BUFFERS]
    spec = dict(training=training, models=MODELS, cases=CASES, buffers=BUFFERS,
                new_comparisons=len(new_family()), hashes={str(path.resolve()): digest(path) for path in files})
    DEST.mkdir(exist_ok=True); path = DEST / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Frozen relation evaluation changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST / 'frozen_source'; folder.mkdir(exist_ok=True)
        for i, source in enumerate(files):
            if source.suffix == '.py': shutil.copy2(source, folder / (f'{i:03d}_' + source.name))
    return spec


def candidate_ids(frame, stats):
    trained = frame[(frame.architecture == 'cnn') & (frame.objective == 'relation')]
    added = stats[stats.origin == 'stock_relation']; selected = []
    for row in trained[trained.member == 'ensemble'].to_dict('records'):
        group = trained[(trained.relation == row['relation']) & (trained.case == row['case']) & (trained.buffer == row['buffer'])]
        infer = added[added.id == row['id']]; expected = comparison_ids(row['model'], row['case'], row['buffer'])
        if len(group) != 4 or set(group.member) != set(MEMBERS): raise AssertionError('Incomplete relation seeds')
        if len(infer) != len(expected) * 2 or set(infer[['comparator', 'block']].itertuples(index=False, name=None)) != {(comp, b) for comp, _ in expected for b in [5, 10]}:
            raise AssertionError('Incomplete relation comparisons')
        stable = ((group['return'] > 0) & (group.mdd > -.2) & (group.positive_blocks >= 2) & (~group.four_week_turnover_stop)).all()
        if stable and (infer.adjusted_p < .025).all() and (infer.simultaneous_lower95 > 0).all(): selected.append(row['id'])
    return selected


def run():
    spec = freeze(); torch.set_num_threads(4)
    complete = json.loads((TRAIN_ROOT / 'training_complete.json').read_text())
    if complete['folds'] != 162 or complete['newly_fitted'] != 162: raise RuntimeError('Incomplete relation training')
    preflight = json.loads((DEST / 'preflight.json').read_text())
    if (preflight['prior_hypotheses'], preflight['joint_hypotheses']) != (11168, 11744): raise RuntimeError('Complete relation preflight')
    hashes = json.loads((TRAIN_ROOT / 'forecast_hashes.json').read_text()); check_hashes(hashes)
    if len(hashes) != 162: raise AssertionError('Missing relation forecasts')
    atomic_json(DEST / 'forecast_hashes.json', hashes)
    p, ix, ci, di = context(DATA_ROOT)
    if ix['dates'][-1] != '20260923': raise AssertionError('Reserved date boundary changed')
    p, release = amend_actions(p, ix, spec['training']['actions'])
    x = torch.from_numpy(np.array(np.load(DATA_ROOT / 'images_raw.npy', mmap_mode='r')[:, None], copy=True))
    matrices, reproduced = {}, []
    for cfg in CONFIGS:
        folder = TRAIN_ROOT / 'models' / cfg['id']; path = folder / '202609.pt'
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['config'] != cfg: raise AssertionError('Relation checkpoint configuration changed')
        model = StockRelationNet(cfg['architecture'], cfg['relation']); model.load_state_dict(saved['state_dict'])
        expected = np.load(path.with_suffix('.pred.npy')); ids = np.flatnonzero(np.isfinite(expected))
        error = float(np.max(np.abs(predict_by_date(model, x, di, ids) - expected[ids]))); del model
        if error > 1e-6: raise AssertionError('Relation checkpoint reinference failed')
        reproduced.append(dict(model=cfg['id'], rows=len(ids), maximum_error=error))
        matrices[cfg['id']] = score_matrix(merge_months(folder, di, ix['dates']), ci, di, p['close'].shape)
    untrained_dir = DEST / 'untrained_models'; untrained_dir.mkdir(exist_ok=True)
    predict_mask = np.zeros(len(di), bool)
    for fold in monthly_folds(ix['dates']): predict_mask |= (di >= fold['start'] - 1) & (di < fold['end'] - 1)
    for relation in ['self', 'mean', 'attention']:
        for seed in [17, 29, 43]:
            torch.manual_seed(seed); model = StockRelationNet('cnn', relation)
            name = f'cnn_relation_{relation}_untrained_seed{seed}'; path = untrained_dir / (name + '.pt')
            if path.exists(): raise RuntimeError('Do not overwrite untrained relation checkpoint')
            torch.save(dict(architecture='cnn', relation=relation, seed=seed, optimizer_steps=0, state_dict=model.state_dict()), path)
            values = np.full(len(di), np.nan, np.float32)
            values[predict_mask] = predict_by_date(model, x, di, np.flatnonzero(predict_mask)); del model
            matrices[name] = score_matrix(values, ci, di, p['close'].shape)
    del x
    for arch, objective in [('cnn', 'h10'), ('mlp', 'h10'), ('cnn', 'untrained')]:
        for relation in ['self', 'mean', 'attention']:
            prefix = f'{arch}_relation_{relation}_{objective}_'
            members = {name: value for name, value in matrices.items() if name.startswith(prefix)}
            if len(members) != 3: raise AssertionError('Incomplete relation ensemble')
            matrices[prefix + 'ensemble'] = rank_consensus(members, {name: arch for name in members}, p['eligible'], 'mean')
    for name in LEGACY: matrices[name] = np.load(PRIOR / 'scores' / (name + '.npy'))
    if set(matrices) != set(MODELS) or len(matrices) != 93: raise AssertionError('Incomplete relation streams')
    atomic_json(DEST / 'checkpoint_reproduction.json', reproduced)
    atomic_json(DEST / 'untrained_hashes.json', {str(path): digest(path) for path in sorted(untrained_dir.glob('*.pt'))})
    score_dir = DEST / 'scores'; score_dir.mkdir(exist_ok=True)
    for name, values in matrices.items():
        path = score_dir / (name + '.npy')
        if path.exists(): raise RuntimeError('Do not overwrite relation scores')
        np.save(path, values)
    atomic_json(DEST / 'score_hashes.json', {str(path): digest(path) for path in sorted(score_dir.glob('*.npy'))})
    p['sector_cap'] = np.minimum(p['sector_cap'], .20); caps = historical_stock_caps(ix['codes'], ix['dates'])
    dates = {date: d for d, date in enumerate(ix['dates'])}; rows, audits, regressions = [], {}, []
    folder = DEST / 'portfolios'; folder.mkdir(exist_ok=True)
    for name in MODELS:
        matrix = matrices.pop(name); info = model_info(name)
        held = {r: snapshot_scores(matrix, p['eligible'], ix['dates'], '20260101', '20260923', r) for r in [1, 5]}
        for case in CASES:
            alpha, origins = held[case['refresh']]
            for buffer in BUFFERS:
                key = account_id(name, case['id'], buffer); path = folder / (key + '.json')
                if path.exists(): raise RuntimeError('Do not overwrite relation account')
                result = replay(p, ix, alpha, '20260101', '20260923', top_n=12, weight=.05, max_orders=case['max_orders'],
                                rebalance=5, rebalance_band=.0005, return_trades=True, planning_price='open',
                                action_release_dates=release, stock_cap_schedule=caps, rank_buffer=buffer)
                for trade in result['trades']:
                    d = dates[trade['signal_date']]; origin = origins[d]
                    if not 0 <= origin <= d: raise AssertionError('Future relation score')
                    trade['portfolio_score_date'] = ix['dates'][origin]
                result['metrics'].update(assess_weeks(result)); atomic_json(path, result)
                if name in LEGACY:
                    if result != json.loads((PRIOR / 'portfolios' / path.name).read_text()): raise AssertionError('Legacy relation account changed')
                    regressions.append(dict(id=key, all_fields_equal=True))
                check = audit_fills(p, ix, result); check.update(additional_checks(p, ix, result, None, max_orders=case['max_orders']))
                check['announced_action_errors'] = release_audit(p, ix, result, release)
                check['dated_hynix_audit'] = audit_pre_july_hynix(p, ix, result)
                check['snapshot_origin_errors'] = [trade['date'] for trade in result['trades']
                    if trade['portfolio_score_date'] != ix['dates'][origins[dates[trade['signal_date']]]]]
                if any(check[n] for n in ['post_buy_limit_violations', 'additional_errors', 'announced_action_errors', 'snapshot_origin_errors']) or check['dated_hynix_audit']['post_buy_limit_errors'] or check['maximum_nav_reconstruction_error_krw'] > .01:
                    atomic_json(DEST / 'failed_audit.json', dict(id=key, audit=check)); raise AssertionError('Relation account audit failed')
                audits[key] = check; quarters = calendar_blocks(result['daily'])
                rows.append(dict(id=key, model=name, **info, case=case['id'], buffer=buffer, band_bp=5,
                                 **result['metrics'], **quarters, positive_blocks=sum(v > 0 for v in quarters.values())))
        print(json.dumps(dict(model=name, portfolios=len(rows))), flush=True)
    if len(rows) != 744 or len(regressions) != 456: raise AssertionError('Incomplete relation accounts')
    atomic_json(DEST / 'independent_audit.json', audits); atomic_json(DEST / 'baseline_regression.json', regressions)
    frame = pd.DataFrame(rows); frame.to_csv(DEST / 'portfolio_summary.csv', index=False)
    family, differences, returns, error = prior_family()
    atomic_json(DEST / 'prior_family_reconstruction.json', dict(hypotheses=len(family), maximum_mean_error=error))
    for key, comp, reference in new_family():
        family.append((key, comp, 'stock_relation'))
        differences.append(returns(DEST, key) - (returns(DEST, reference) if reference is not None else 0.))
    matrix = np.column_stack(differences)
    if matrix.shape != (179, 11744) or len(set(family)) != 11744: raise AssertionError('Incomplete relation family')
    records = []
    for block in [5, 10]:
        stats = family_bootstrap(matrix, block=block, draws=4000, seed=57)
        for i, (key, comp, origin) in enumerate(family):
            records.append(dict(id=key, comparator=comp, origin=origin, block=block,
                               **{name: float(stats[name][i]) for name in ['mean', 'standard_error', 'adjusted_p', 'marginal_p', 'simultaneous_lower95']}))
    stats = pd.DataFrame(records); stats.to_csv(DEST / 'joint_bootstrap.csv', index=False)
    selected = candidate_ids(frame, stats)
    atomic_json(DEST / 'evaluation_summary.json', dict(status='development_only', portfolios=744, new_portfolios=288,
                legacy_regressions=456, joint_hypotheses=11744, new_hypotheses=576, robust_candidate_gate_passed=selected,
                independent_confirmation=False))
    print(json.dumps(dict(complete=True, robust_candidates=selected)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'run'])
    if parser.parse_args().action == 'freeze': freeze()
    else: run()

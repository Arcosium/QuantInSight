"""Wider continuing-position adjustment bands on all matched frozen signals."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from quant.timefolio_heatmap_action_amendment import amend_actions, release_audit
from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_feasible_training import CONFIGS, DATA_ROOT, EVALUATION_PLAN
from quant.timefolio_heatmap_feasible_evaluation import DEST as PRIOR, LEGACY, prior_family as feasible_prior_family
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_retention_replay import replay
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_stock_limits import historical_stock_caps, audit_pre_july_hynix
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward_eval import calendar_blocks, family_bootstrap
from quant.timefolio_heatmap_week_boundaries import assess_weeks

DEST = PRIOR.with_name('20260929_adjustment_band_v1')
PLAN_RECORD = Path(__file__).resolve().parents[1] / '_workspace/timefolio_heatmap_walkforward_v3/adjustment_band_next_plan.json'
MODELS = ([c['id'] for c in CONFIGS] +
          [f'{a}_feasible_{t}_h10_ensemble' for a in ['cnn', 'mlp'] for t in ['sector', 'absolute']] + LEGACY)
CASES = [case for case in EVALUATION_PLAN['cases'] if case['top_n'] == 12]
BUFFERS = [12, 24]
BANDS = [5, 25, 50, 100]


def model_info(model):
    if model not in MODELS: raise ValueError('Unknown frozen model')
    if model == 'online_nonimage': return dict(architecture='online', objective='control', target='control', member='online')
    arch = model.split('_')[0]; member = model.rsplit('_', 1)[1]
    if '_untrained_' in model: return dict(architecture=arch, objective='untrained', target='control', member=member)
    objective = 'feasible' if '_feasible_' in model else 'toprank' if '_toprank_' in model else 'uniform'
    return dict(architecture=arch, objective=objective, target='absolute' if '_absolute_' in model else 'sector', member=member)


def account_id(model, case, buffer, band):
    if model not in MODELS or case not in {c['id'] for c in CASES} or buffer not in BUFFERS or band not in BANDS:
        raise ValueError('Unknown adjustment-band account')
    return f'{model}__{case}__buffer{buffer}__band{band}bp'


def comparison_ids(model, case, buffer, band):
    info = model_info(model)
    if info['architecture'] != 'cnn' or info['objective'] not in ['uniform', 'toprank', 'feasible'] or band == 5:
        raise ValueError('New-band trained CNN comparison required')
    key = lambda name, bp=band: account_id(name, case, buffer, bp)
    uniform = 'cnn_' + ('absolute_' if info['target'] == 'absolute' else '') + 'rank_h10_' + info['member']
    pairs = [('cash', None), ('matched_mlp', key('mlp' + model[3:])),
             ('matched_untrained_cnn', key('cnn_untrained_' + info['member'])),
             ('online_nonimage', key('online_nonimage')), ('same_cnn_band5bp', key(model, 5))]
    if info['objective'] in ['toprank', 'feasible']: pairs.append(('matched_uniform_cnn', key(uniform)))
    if info['objective'] == 'feasible':
        pairs.append(('matched_toprank_cnn', key(model.replace('_feasible_', '_toprank_', 1))))
    return pairs


def new_family():
    result = []
    for model in MODELS:
        info = model_info(model)
        if info['architecture'] != 'cnn' or info['objective'] not in ['uniform', 'toprank', 'feasible']: continue
        for case in CASES:
            for buffer in BUFFERS:
                for band in BANDS[1:]:
                    key = account_id(model, case['id'], buffer, band)
                    result.extend((key, comp, reference) for comp, reference in comparison_ids(model, case['id'], buffer, band))
    if len(result) != 3456 or len({(key, comp) for key, comp, _ in result}) != 3456:
        raise AssertionError('Adjustment-band comparison grid incomplete')
    return result


def parent_ready():
    validation = json.loads((PRIOR / 'validation_complete.json').read_text())
    if validation['status'] != 'numerical_validation_complete' or validation['portfolios'] != 1368:
        raise RuntimeError('Finish independent feasible-account validation before adjustment-band work')
    summary = json.loads((PRIOR / 'evaluation_summary.json').read_text())
    if summary['portfolios'] != 1368 or summary['joint_hypotheses'] != 7712:
        raise RuntimeError('Unexpected parent evaluation')
    if summary['robust_candidate_gate_passed']:
        raise RuntimeError('Prioritize parent candidate stress and fresh confirmation')
    return validation


def freeze():
    parent_ready()
    planned = json.loads(PLAN_RECORD.read_text())
    expected = dict(models=MODELS, cases=CASES, buffer_multipliers=[1, 2], reference_band_nav_fraction=.0005,
                    new_bands_nav_fraction=[.0025, .005, .01], total_accounts=1824, new_accounts=1368,
                    legacy_accounts=456, prior_hypotheses=7712, new_hypotheses=3456, joint_hypotheses=11168)
    for key, value in expected.items():
        if planned[key] != value: raise AssertionError('Registered adjustment-band plan changed: ' + key)
    parent = json.loads((PRIOR / 'protocol.json').read_text())
    manifests = [planned['evidence_hashes'], parent['hashes'], json.loads((PRIOR / 'score_hashes.json').read_text()),
                 json.loads((DATA_ROOT / 'data_hashes.json').read_text())]
    def digest(path):
        with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
    for manifest in manifests:
        for filename, sha in manifest.items():
            if digest(filename) != sha: raise AssertionError('Adjustment-band input changed')
    files = [Path(__file__), PLAN_RECORD, PRIOR / 'protocol.json', PRIOR / 'validation_complete.json',
             PRIOR / 'evaluation_summary.json', PRIOR / 'joint_bootstrap.csv', PRIOR / 'score_hashes.json', DATA_ROOT / 'data_hashes.json']
    files += [Path(__file__).with_name('timefolio_heatmap_' + name + '.py') for name in
              ['feasible_evaluation', 'feasible_training', 'retention_replay', 'planned_audit', 'action_amendment',
               'stock_limits', 'week_boundaries', 'snapshot', 'walkforward_eval', 'replay', 'study', 'data']]
    files += [PRIOR / 'scores' / (name + '.npy') for name in MODELS]
    for model in MODELS:
        for case in CASES:
            for buffer in BUFFERS:
                files.append(PRIOR / 'portfolios' / (f"{model}__{case['id']}__buffer{buffer}.json"))
    spec = dict(plan=planned, parent=parent, models=MODELS, cases=CASES, buffers=BUFFERS, bands_bp=BANDS,
                comparison_count=len(new_family()),
                bootstrap_memory=dict(days=179, hypotheses=11168, sample_batch_bytes=100 * 179 * 11168 * 8,
                                      stored_bootstrap_mean_bytes=4000 * 11168 * 8,
                                      implementation='Unchanged family_bootstrap with100draw sample chunks; safety supervisor enforces8GiB.'),
                hashes={str(path.resolve()): digest(path) for path in files})
    DEST.mkdir(exist_ok=True); path = DEST / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Adjustment-band protocol changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST / 'frozen_source'; folder.mkdir(exist_ok=True)
        for i, source in enumerate(files):
            if source.suffix == '.py': shutil.copy2(source, folder / (f'{i:03d}_' + source.name))
    return spec


def prior_family():
    family, differences, returns, _ = feasible_prior_family()
    stored = pd.read_csv(PRIOR / 'joint_bootstrap.csv')
    for key, comp in stored[stored.origin == 'feasible_objective'][['id', 'comparator']].drop_duplicates().itertuples(index=False, name=None):
        model, suffix = key.split('__', 1); member = model.rsplit('_', 1)[1]; target = model.split('_')[2]
        if comp == 'cash': baseline = 0.
        elif comp == 'matched_mlp_same_target': baseline = returns(PRIOR, 'mlp' + key[3:])
        elif comp == 'online_nonimage': baseline = returns(PRIOR, 'online_nonimage__' + suffix)
        elif comp == 'matched_uniform_cnn':
            uniform = 'cnn_' + ('absolute_' if target == 'absolute' else '') + 'rank_h10_' + member
            baseline = returns(PRIOR, uniform + '__' + suffix)
        elif comp == 'matched_toprank_cnn': baseline = returns(PRIOR, key.replace('_feasible_', '_toprank_', 1))
        elif comp == 'matched_untrained_cnn': baseline = returns(PRIOR, 'cnn_untrained_' + member + '__' + suffix)
        elif comp == 'same_feasible_cnn_buffer0': baseline = returns(PRIOR, key.rsplit('__buffer', 1)[0] + '__buffer0')
        else: raise AssertionError('Unknown retained feasible comparator')
        family.append((key, comp, 'feasible_objective')); differences.append(returns(PRIOR, key) - baseline)
    if len(family) != 7712 or len(set(family)) != 7712: raise AssertionError('Incomplete adjustment-band prior family')
    means = stored[stored.block == 5].set_index(['id', 'comparator', 'origin'])['mean']
    error = max(abs(float(np.mean(value)) - means.loc[key]) for key, value in zip(family, differences))
    if error > 1e-14: raise AssertionError('Retained feasible means changed')
    return family, differences, returns, error


def candidate_ids(frame, stats):
    trained = frame[(frame.architecture == 'cnn') & frame.objective.isin(['uniform', 'toprank', 'feasible']) & (frame.band_bp != 5)]
    added = stats[stats.origin == 'adjustment_band']; candidates = []
    for row in trained[trained.member == 'ensemble'].to_dict('records'):
        members = trained[(trained.objective == row['objective']) & (trained.target == row['target']) & (trained.case == row['case'])
                          & (trained.buffer == row['buffer']) & (trained.band_bp == row['band_bp'])]
        infer = added[added.id == row['id']]
        expected = comparison_ids(row['model'], row['case'], row['buffer'], row['band_bp'])
        if set(members.member) != {'seed17', 'seed29', 'seed43', 'ensemble'} or len(members) != 4:
            raise AssertionError('Incomplete matched band seeds')
        if set(infer[['comparator', 'block']].itertuples(index=False, name=None)) != {(c, b) for c, _ in expected for b in [5, 10]} or len(infer) != len(expected) * 2:
            raise AssertionError('Incomplete adjustment-band comparisons')
        stable = ((members['return'] > 0) & (members.mdd > -.20) & (members.positive_blocks >= 2) & (~members.four_week_turnover_stop)).all()
        if stable and (infer.adjusted_p < .025).all() and (infer.simultaneous_lower95 > 0).all(): candidates.append(row['id'])
    return candidates


def run():
    spec = freeze()
    preflight = json.loads((DEST / 'preflight.json').read_text())
    if preflight['prior_hypotheses'] != 7712 or preflight['joint_hypotheses'] != 11168:
        raise RuntimeError('Complete comparison preflight before running accounts')
    p, ix, _, _ = context(DATA_ROOT); assert ix['dates'][-1] == '20260923'
    actions = spec['parent']['training']['parents']['sector']['data_protocol']['parent']['training_parent']['actions']
    p, release = amend_actions(p, ix, actions); p['sector_cap'] = np.minimum(p['sector_cap'], .20)
    caps = historical_stock_caps(ix['codes'], ix['dates']); dates = {date: d for d, date in enumerate(ix['dates'])}
    folder = DEST / 'portfolios'; folder.mkdir(exist_ok=True); score_dir = DEST / 'scores'; score_dir.mkdir(exist_ok=True)
    rows, audits, regressions, score_hashes = [], {}, [], {}
    for model in MODELS:
        score_path = score_dir / (model + '.npy'); shutil.copy2(PRIOR / 'scores' / score_path.name, score_path)
        with score_path.open('rb') as stream: score_hashes[str(score_path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
        matrix = np.load(score_path); info = model_info(model)
        snapshots = {r: snapshot_scores(matrix, p['eligible'], ix['dates'], '20260101', '20260923', r) for r in [1, 5]}
        for case in CASES:
            alpha, origins = snapshots[case['refresh']]
            for buffer in BUFFERS:
                for band in BANDS:
                    key = account_id(model, case['id'], buffer, band); path = folder / (key + '.json')
                    if path.exists(): raise RuntimeError('Do not overwrite an adjustment-band account')
                    result = replay(p, ix, alpha, '20260101', '20260923', top_n=12, weight=.05, max_orders=case['max_orders'],
                                    rank_buffer=buffer, rebalance=5, rebalance_band=band / 10000., return_trades=True,
                                    planning_price='open', action_release_dates=release, stock_cap_schedule=caps)
                    for trade in result['trades']:
                        d = dates[trade['signal_date']]; origin = origins[d]
                        if not 0 <= origin <= d: raise AssertionError('Future adjustment-band score')
                        trade['portfolio_score_date'] = ix['dates'][origin]
                    result['metrics'].update(assess_weeks(result)); atomic_json(path, result)
                    if band == 5:
                        prior_key = f"{model}__{case['id']}__buffer{buffer}"
                        if result != json.loads((PRIOR / 'portfolios' / (prior_key + '.json')).read_text()):
                            raise AssertionError('Reference5bp account changed')
                        regressions.append(dict(id=key, parent_id=prior_key, all_fields_equal=True))
                    check = audit_fills(p, ix, result); check.update(additional_checks(p, ix, result, None, max_orders=case['max_orders']))
                    check['announced_action_errors'] = release_audit(p, ix, result, release)
                    check['dated_hynix_audit'] = audit_pre_july_hynix(p, ix, result)
                    check['snapshot_origin_errors'] = [trade['date'] for trade in result['trades']
                        if trade['portfolio_score_date'] != ix['dates'][origins[dates[trade['signal_date']]]]]
                    if any(check[name] for name in ['post_buy_limit_violations', 'additional_errors', 'announced_action_errors', 'snapshot_origin_errors']) or check['dated_hynix_audit']['post_buy_limit_errors'] or check['maximum_nav_reconstruction_error_krw'] > .01:
                        atomic_json(DEST / 'failed_audit.json', dict(id=key, audit=check)); raise AssertionError('Adjustment-band fill audit failed')
                    audits[key] = check; quarters = calendar_blocks(result['daily'])
                    rows.append(dict(id=key, model=model, **info, case=case['id'], buffer=buffer, band_bp=band,
                                     **result['metrics'], **quarters, positive_blocks=sum(v > 0 for v in quarters.values())))
        print(json.dumps(dict(model=model, portfolios=len(rows))), flush=True)
    if len(rows) != 1824 or len(regressions) != 456: raise AssertionError('Adjustment-band account grid incomplete')
    atomic_json(DEST / 'score_hashes.json', score_hashes); atomic_json(DEST / 'independent_audit.json', audits)
    atomic_json(DEST / 'baseline_regression.json', regressions)
    frame = pd.DataFrame(rows); frame.to_csv(DEST / 'portfolio_summary.csv', index=False)
    family, differences, returns, error = prior_family()
    atomic_json(DEST / 'prior_family_reconstruction.json', dict(hypotheses=len(family), maximum_mean_error=error))
    for key, comp, reference in new_family():
        family.append((key, comp, 'adjustment_band'))
        differences.append(returns(DEST, key) - (returns(DEST, reference) if reference is not None else 0.))
    matrix = np.column_stack(differences)
    if matrix.shape != (179, 11168) or len(set(family)) != 11168: raise AssertionError('Adjustment-band joint family differs')
    records = []
    for block in [5, 10]:
        statistics = family_bootstrap(matrix, block=block, draws=4000, seed=57)
        for i, (key, comp, origin) in enumerate(family):
            records.append(dict(id=key, comparator=comp, origin=origin, block=block,
                               **{name: float(statistics[name][i]) for name in ['mean', 'standard_error', 'adjusted_p', 'marginal_p', 'simultaneous_lower95']}))
    stats = pd.DataFrame(records); stats.to_csv(DEST / 'joint_bootstrap.csv', index=False)
    candidates = candidate_ids(frame, stats)
    atomic_json(DEST / 'evaluation_summary.json', dict(status='development_only', portfolios=1824, new_portfolios=1368,
                legacy_regressions=456, joint_hypotheses=11168, new_hypotheses=3456, robust_candidate_gate_passed=candidates,
                independent_confirmation=False))
    print(json.dumps(dict(complete=True, robust_candidates=candidates)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'run'])
    if parser.parse_args().action == 'freeze': freeze()
    else: run()

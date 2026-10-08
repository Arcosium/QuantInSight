"""Registered daily-summary history comparisons under two sector ceilings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from quant.timefolio_heatmap_daily_history_training import DEST as TRAIN, FEATURES, CONFIGS, EVALUATION_PLAN, feature_path
from quant.timefolio_heatmap_sector_ceiling import (DEST as PRIOR, MODES, sector_panel,
    prior_family as ceiling_prior_family, new_family as ceiling_new_family)
from quant.timefolio_heatmap_stock_relation_evaluation import (
    DEST as RAW_SOURCE, MEMBERS, CASES, BUFFERS, model_info as legacy_info)
from quant.timefolio_heatmap_stock_relation_training import DATA_ROOT, digest, check_hashes
from quant.timefolio_heatmap_initialisation_probe import DEST as INITIAL, state_digest
from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_seed_evaluation import merge_months
from quant.timefolio_heatmap_consensus import rank_consensus
from quant.timefolio_heatmap_study import context, score_matrix
from quant.timefolio_heatmap_walkforward import AlternativeNet, predict
from quant.timefolio_heatmap_action_amendment import amend_actions, release_audit
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_retention_replay import replay
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_stock_limits import historical_stock_caps, audit_pre_july_hynix
from quant.timefolio_heatmap_week_boundaries import assess_weeks
from quant.timefolio_heatmap_walkforward_eval import calendar_blocks, family_bootstrap

DEST = PRIOR.with_name('20260929_daily_history_evaluation_v1')
TOKENS = {'history': 'history', 'reverse': 'history_reverse'}
TRAINED = [f'{arch}_daily_{token}_h10_{member}' for arch in ['cnn', 'mlp'] for token in TOKENS for member in MEMBERS]
UNTRAINED = [f'cnn_daily_{token}_untrained_{member}' for token in TOKENS for member in MEMBERS]
LEGACY = [f'{arch}_absolute_rank_h10_{member}' for arch in ['cnn', 'mlp'] for member in MEMBERS]
LEGACY += ['cnn_untrained_' + member for member in MEMBERS] + ['online_nonimage']
MODELS = TRAINED + UNTRAINED + LEGACY


def model_info(model):
    if model in LEGACY: return dict(legacy_info(model), encoding='snapshot')
    if model not in TRAINED + UNTRAINED: raise ValueError('Known daily-history model required')
    arch, _, token, objective, member = model.split('_')
    return dict(architecture=arch, encoding=TOKENS[token], objective='daily_history' if objective == 'h10' else 'untrained',
                target='absolute' if objective == 'h10' else 'none', member=member)


def account_id(model, case, buffer, ceiling):
    if model not in MODELS or case not in {row['id'] for row in CASES} or buffer not in BUFFERS or ceiling not in MODES:
        raise ValueError('Registered daily-history account required')
    return f'{model}__{case}__buffer{buffer}__band5bp__sector_{ceiling}'


def legacy_path(model, case, buffer, ceiling):
    if model not in LEGACY: return None
    key = account_id(model, case, buffer, ceiling)
    if ceiling == 'research20': return RAW_SOURCE / 'portfolios' / (key.rsplit('__sector_', 1)[0] + '.json')
    if model.startswith('cnn_absolute_rank_h10_') or model == 'online_nonimage':
        return PRIOR / 'portfolios' / (key + '.json')
    return None


def comparison_ids(model, case, buffer, ceiling):
    if model not in TRAINED or not model.startswith('cnn_'): raise ValueError('Trained daily-history CNN required')
    _, _, token, _, member = model.split('_')
    key = lambda name: account_id(name, case, buffer, ceiling)
    other = 'reverse' if token == 'history' else 'history'
    result = [('cash', None), ('matched_mlp', key('mlp' + model[3:])),
        ('matched_untrained_cnn', key(f'cnn_daily_{token}_untrained_{member}')),
        ('matched_snapshot_cnn', key(f'cnn_absolute_rank_h10_{member}')),
        ('other_daily_cnn', key(f'cnn_daily_{other}_h10_{member}')),
        ('online_nonimage', key('online_nonimage'))]
    if ceiling == 'formula': result.append(('same_cnn_research20', account_id(model, case, buffer, 'research20')))
    return result


def new_family():
    result = []
    for model in TRAINED:
        if not model.startswith('cnn_'): continue
        for case in CASES:
            for buffer in BUFFERS:
                for ceiling in MODES:
                    key = account_id(model, case['id'], buffer, ceiling)
                    result.extend((key, comp, reference) for comp, reference in comparison_ids(model, case['id'], buffer, ceiling))
    if len(result) != 832 or len({(key, comp) for key, comp, _ in result}) != 832:
        raise AssertionError('Incomplete daily-history comparisons')
    return result


def prior_family():
    family, differences, returns, _ = ceiling_prior_family()
    stored = pd.read_csv(PRIOR / 'joint_bootstrap.csv'); added = []
    for key, comp, reference in ceiling_new_family():
        record = (key, comp, 'sector_ceiling'); family.append(record); added.append(record)
        differences.append(returns(PRIOR, key) - (returns(PRIOR, reference) if reference is not None else 0.))
    expected = stored[stored.origin == 'sector_ceiling'][['id', 'comparator', 'origin']].itertuples(index=False, name=None)
    if set(added) != set(expected) or len(family) != 12416 or len(set(family)) != 12416:
        raise AssertionError('Incomplete retained sector-ceiling family')
    means = stored[stored.block == 5].set_index(['id', 'comparator', 'origin'])['mean']
    error = max(abs(float(np.mean(value)) - means.loc[key]) for key, value in zip(family, differences))
    if error > 1e-14: raise AssertionError('Retained sector-ceiling means changed')
    return family, differences, returns, error


def freeze():
    training = json.loads((TRAIN / 'protocol.json').read_text())
    if training['evaluation_plan'] != EVALUATION_PLAN: raise AssertionError('Training and evaluation plans differ')
    for manifest in [training['hashes'], json.loads((PRIOR / 'protocol.json').read_text())['hashes'],
                     json.loads((RAW_SOURCE / 'score_hashes.json').read_text()),
                     json.loads((INITIAL / 'output_hashes.json').read_text())]: check_hashes(manifest)
    files = [Path(__file__), TRAIN / 'protocol.json', PRIOR / 'protocol.json', PRIOR / 'validation_complete.json',
             PRIOR / 'joint_bootstrap.csv', RAW_SOURCE / 'score_hashes.json', INITIAL / 'output_hashes.json']
    for model in LEGACY: files.append(RAW_SOURCE / 'scores' / (model + '.npy'))
    for seed in [17, 29, 43]: files.append(INITIAL / 'models' / f'cnn_untrained_seed{seed}.pt')
    for model in MODELS:
        for case in CASES:
            for buffer in BUFFERS:
                for ceiling in MODES:
                    path = legacy_path(model, case['id'], buffer, ceiling)
                    if path is not None: files.append(path)
    files += [Path(__file__).with_name('timefolio_heatmap_' + name + '.py') for name in
        ['daily_history_training', 'sector_ceiling', 'stock_relation_evaluation', 'initialisation_probe', 'seed_evaluation',
         'consensus', 'retention_replay', 'snapshot', 'stock_limits', 'planned_audit', 'action_amendment', 'week_boundaries',
         'walkforward_eval', 'walkforward', 'study', 'data']]
    files.append(Path(__file__).resolve().parents[1] / 'tests/test_timefolio_daily_history_evaluation.py')
    files = sorted(set(path.resolve() for path in files))
    spec = dict(training=training, plan=EVALUATION_PLAN, models=MODELS, cases=CASES, buffers=BUFFERS,
                untrained='Same original seed-specific AlternativeNet CNN weights; all feature rows inferred, zero optimizer steps. No outcome-selected signs.',
                hashes={str(path): digest(path) for path in files})
    DEST.mkdir(exist_ok=True); path = DEST / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Frozen daily-history evaluation changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST / 'frozen_source'; folder.mkdir(exist_ok=True)
        for i, source in enumerate(files):
            if source.suffix == '.py': shutil.copy2(source, folder / (f'{i:03d}_' + source.name))
    return spec


def candidate_ids(frame, stats):
    trained = frame[(frame.architecture == 'cnn') & (frame.objective == 'daily_history')]
    selected = []
    for row in trained[trained.member == 'ensemble'].to_dict('records'):
        group = trained[(trained.encoding == row['encoding']) & (trained.case == row['case']) &
                        (trained.buffer == row['buffer']) & (trained.ceiling == row['ceiling'])]
        infer = stats[(stats.id == row['id']) & (stats.origin == 'daily_history')]
        expected = {(comp, block) for comp, _ in comparison_ids(row['model'], row['case'], row['buffer'], row['ceiling']) for block in [5, 10]}
        if len(group) != 4 or set(group.member) != set(MEMBERS): raise AssertionError('Incomplete daily-history seeds')
        if len(infer) != len(expected) or set(infer[['comparator', 'block']].itertuples(index=False, name=None)) != expected:
            raise AssertionError('Incomplete daily-history candidate comparisons')
        stable = ((group['return'] > 0) & (group.mdd > -.2) & (group.positive_blocks >= 2) & (~group.four_week_turnover_stop)).all()
        if stable and (infer.adjusted_p < .025).all() and (infer.simultaneous_lower95 > 0).all(): selected.append(row['id'])
    return selected


def forecasts(panel, ix, ci, di):
    complete = json.loads((TRAIN / 'training_complete.json').read_text())
    if complete['folds'] != 108 or complete['newly_fitted'] != 108: raise AssertionError('Daily-history training unfinished')
    for name, count in [('forecast_hashes.json', 108), ('model_artifact_hashes.json', 324)]:
        manifest = json.loads((TRAIN / name).read_text()); assert len(manifest) == count; check_hashes(manifest)
    matrices = {}; checkpoints = []; untrained = []
    folder = DEST / 'untrained_models'; folder.mkdir(exist_ok=True)
    for token, encoding in TOKENS.items():
        x = torch.from_numpy(np.array(np.load(FEATURES / f'images_{encoding}.npy', mmap_mode='r')[:, None], copy=True))
        for cfg in [row for row in CONFIGS if row['encoding'] == encoding]:
            model_root = TRAIN / 'models' / cfg['id']; path = model_root / '202609.pt'
            saved = torch.load(path, map_location='cpu', weights_only=True); assert saved['config'] == cfg
            model = AlternativeNet(cfg['architecture']); model.load_state_dict(saved['state_dict'])
            expected = np.load(path.with_suffix('.pred.npy')); ids = np.flatnonzero(np.isfinite(expected))
            error = float(np.max(np.abs(predict(model, x, ids) - expected[ids]))); assert error <= 1e-6
            checkpoints.append(dict(model=cfg['id'], rows=len(ids), maximum_error=error))
            matrices[cfg['id']] = score_matrix(merge_months(model_root, di, ix['dates']), ci, di, panel['close'].shape)
        for seed in [17, 29, 43]:
            name = f'cnn_daily_{token}_untrained_seed{seed}'
            saved = torch.load(INITIAL / 'models' / f'cnn_untrained_seed{seed}.pt', map_location='cpu', weights_only=True)
            torch.manual_seed(seed); model = AlternativeNet('cnn')
            for key, value in model.state_dict().items(): assert torch.equal(value, saved['state_dict'][key])
            before = state_digest(model); values = predict(model, x, np.arange(len(ci)))
            assert np.isfinite(values).all() and state_digest(model) == before
            path = folder / (name + '.pt'); assert not path.exists()
            torch.save(dict(architecture='cnn', encoding=encoding, seed=seed, optimizer_steps=0, state_dict=model.state_dict()), path)
            matrices[name] = score_matrix(values, ci, di, panel['close'].shape)
            untrained.append(dict(model=name, rows=len(ci), optimizer_steps=0, original_weights_exact=True, weights_unchanged=True))
        del x
    for prefix in [f'{arch}_daily_{token}_h10_' for arch in ['cnn', 'mlp'] for token in TOKENS] + [f'cnn_daily_{token}_untrained_' for token in TOKENS]:
        seeds = {prefix + member: matrices[prefix + member] for member in MEMBERS[:3]}
        matrices[prefix + 'ensemble'] = rank_consensus(seeds, {name: prefix.split('_')[0] for name in seeds}, panel['eligible'], 'mean')
    for model in LEGACY: matrices[model] = np.load(RAW_SOURCE / 'scores' / (model + '.npy'))
    assert set(matrices) == set(MODELS) and len(matrices) == 37
    atomic_json(DEST / 'checkpoint_reproduction.json', checkpoints); atomic_json(DEST / 'untrained_reproduction.json', untrained)
    atomic_json(DEST / 'untrained_hashes.json', {str(path): digest(path) for path in sorted(folder.glob('*.pt'))})
    return matrices


def run():
    spec = freeze(); torch.set_num_threads(4)
    preflight = json.loads((DEST / 'preflight.json').read_text())
    if (preflight['prior_hypotheses'], preflight['joint_hypotheses']) != (12416, 13248): raise RuntimeError('Daily-history preflight required')
    p, ix, ci, di = context(DATA_ROOT); assert ix['dates'][-1] == '20260923'
    matrices = forecasts(p, ix, ci, di)
    p, release = amend_actions(p, ix, spec['training']['actions'])
    panels = {mode: sector_panel(p, mode) for mode in MODES}; caps = historical_stock_caps(ix['codes'], ix['dates'])
    dates = {date: day for day, date in enumerate(ix['dates'])}
    folder = DEST / 'portfolios'; folder.mkdir(exist_ok=True); scores = DEST / 'scores'; scores.mkdir(exist_ok=True)
    rows, audits, regressions = [], {}, []
    for model in MODELS:
        path = scores / (model + '.npy'); assert not path.exists(); np.save(path, matrices[model])
        if model in LEGACY: assert digest(path) == digest(RAW_SOURCE / 'scores' / path.name)
        held = {refresh: snapshot_scores(matrices[model], p['eligible'], ix['dates'], '20260101', '20260923', refresh) for refresh in [1, 5]}
        for case in CASES:
            alpha, origins = held[case['refresh']]
            for buffer in BUFFERS:
                for ceiling in MODES:
                    panel = panels[ceiling]; key = account_id(model, case['id'], buffer, ceiling)
                    path = folder / (key + '.json'); assert not path.exists()
                    result = replay(panel, ix, alpha, '20260101', '20260923', top_n=12, weight=.05,
                        max_orders=case['max_orders'], rank_buffer=buffer, rebalance=5, rebalance_band=.0005,
                        return_trades=True, planning_price='open', action_release_dates=release, stock_cap_schedule=caps)
                    for trade in result['trades']:
                        day = dates[trade['signal_date']]; origin = origins[day]; assert 0 <= origin <= day
                        trade['portfolio_score_date'] = ix['dates'][origin]
                    result['metrics'].update(assess_weeks(result)); atomic_json(path, result)
                    prior = legacy_path(model, case['id'], buffer, ceiling)
                    if prior is not None:
                        assert result == json.loads(prior.read_text()); regressions.append(dict(id=key, prior=str(prior), all_fields_exact=True))
                    audit = audit_fills(panel, ix, result); audit.update(additional_checks(panel, ix, result, None, max_orders=case['max_orders']))
                    audit['announced_action_errors'] = release_audit(panel, ix, result, release)
                    audit['dated_hynix_audit'] = audit_pre_july_hynix(panel, ix, result)
                    audit['snapshot_origin_errors'] = [trade['date'] for trade in result['trades']
                        if trade['portfolio_score_date'] != ix['dates'][origins[dates[trade['signal_date']]]]]
                    if any(audit[name] for name in ['post_buy_limit_violations', 'additional_errors', 'announced_action_errors', 'snapshot_origin_errors']) or audit['dated_hynix_audit']['post_buy_limit_errors'] or audit['maximum_nav_reconstruction_error_krw'] > .01:
                        atomic_json(DEST / 'failed_audit.json', dict(id=key, audit=audit)); raise AssertionError('Daily-history account audit failed')
                    audits[key] = audit; quarters = calendar_blocks(result['daily'])
                    rows.append(dict(id=key, model=model, **model_info(model), case=case['id'], buffer=buffer, band_bp=5,
                        ceiling=ceiling, **result['metrics'], **quarters, positive_blocks=sum(value > 0 for value in quarters.values())))
        print(json.dumps(dict(model=model, portfolios=len(rows))), flush=True)
    assert len(rows) == 592 and len(regressions) == 144
    atomic_json(DEST / 'score_hashes.json', {str(path): digest(path) for path in sorted(scores.glob('*.npy'))})
    atomic_json(DEST / 'independent_audit.json', audits); atomic_json(DEST / 'baseline_regression.json', regressions)
    frame = pd.DataFrame(rows); frame.to_csv(DEST / 'portfolio_summary.csv', index=False)
    family, differences, returns, error = prior_family()
    atomic_json(DEST / 'prior_family_reconstruction.json', dict(prior_hypotheses=len(family), maximum_mean_error=error))
    for key, comp, reference in new_family():
        family.append((key, comp, 'daily_history'))
        differences.append(returns(DEST, key) - (returns(DEST, reference) if reference is not None else 0.))
    matrix = np.column_stack(differences); assert matrix.shape == (179, 13248) and len(set(family)) == 13248
    records = []
    for block in [5, 10]:
        stats = family_bootstrap(matrix, block=block, draws=4000, seed=57)
        for i, (key, comp, origin) in enumerate(family):
            records.append(dict(id=key, comparator=comp, origin=origin, block=block,
                **{name: float(stats[name][i]) for name in ['mean', 'standard_error', 'adjusted_p', 'marginal_p', 'simultaneous_lower95']}))
    stats = pd.DataFrame(records); stats.to_csv(DEST / 'joint_bootstrap.csv', index=False)
    selected = candidate_ids(frame, stats); check_hashes(spec['hashes'])
    atomic_json(DEST / 'evaluation_summary.json', dict(status='development_only', portfolios=592, new_portfolios=448,
        legacy_regressions=144, joint_hypotheses=13248, new_hypotheses=832, robust_candidate_gate_passed=selected,
        independent_confirmation=False, full_contest_compliance_certified=False))
    print(json.dumps(dict(complete=True, robust_candidates=selected)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'run'])
    if parser.parse_args().action == 'freeze': freeze()
    else: run()

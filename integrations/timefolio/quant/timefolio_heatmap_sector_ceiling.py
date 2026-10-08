"""Compare the extra20% research sector ceiling with the frozen rule formula.

No new fitting. The formula mode retains dated stock, small-cap, gross, admission
and execution constraints, including the original5% target-sector headroom.
Historical sector mapping remains an approximation, not a compliance certificate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from quant.timefolio_heatmap_stock_relation_evaluation import (
    DEST as PRIOR, TRAINED, UNTRAINED, MEMBERS, CASES, BUFFERS, model_info,
    account_id as parent_account_id, comparison_ids as parent_comparisons,
    prior_family as relation_prior_family, new_family as relation_new_family,
)
from quant.timefolio_heatmap_stock_relation_training import DATA_ROOT, digest, check_hashes
from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_action_amendment import amend_actions, release_audit
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_retention_replay import replay
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_stock_limits import historical_stock_caps, audit_pre_july_hynix
from quant.timefolio_heatmap_week_boundaries import assess_weeks
from quant.timefolio_heatmap_walkforward_eval import calendar_blocks, family_bootstrap

DEST = PRIOR.with_name('20260929_sector_ceiling_v1')
PLAN_RECORD = Path(__file__).resolve().parents[1] / '_workspace/timefolio_heatmap_walkforward_v3/sector_ceiling_next_plan.json'
MODES = ['research20', 'formula']
MODELS = TRAINED + UNTRAINED + [f'cnn_absolute_rank_h10_{member}' for member in MEMBERS] + ['online_nonimage']


def sector_panel(panel, mode):
    if mode not in MODES: raise ValueError('Registered sector mode required')
    caps = np.asarray(panel['sector_cap'])
    if not np.isfinite(caps).all() or (caps <= 0).any(): raise ValueError('Finite positive frozen sector limits required')
    out = dict(panel)
    # Formula may exceed1 when a sector is over half the market; cash/gross
    # ceilings still prohibit leverage. Do not reinterpret it as a target.
    out['sector_cap'] = np.minimum(caps, .20) if mode == 'research20' else caps.copy()
    return out


def account_id(model, case, buffer, mode):
    if model not in MODELS or mode not in MODES: raise ValueError('Known sector-ceiling account required')
    return parent_account_id(model, case, buffer) + '__sector_' + mode


def comparison_ids(model, case, buffer):
    if model not in TRAINED or not model.startswith('cnn_'): raise ValueError('Trained relation CNN required')
    pairs = [(comp, reference + '__sector_formula' if reference is not None else None)
             for comp, reference in parent_comparisons(model, case, buffer)]
    pairs.append(('same_cnn_research20', account_id(model, case, buffer, 'research20')))
    return pairs


def new_family():
    family = []
    for model in TRAINED:
        if not model.startswith('cnn_'): continue
        for case in CASES:
            for buffer in BUFFERS:
                key = account_id(model, case['id'], buffer, 'formula')
                family.extend((key, comp, reference) for comp, reference in comparison_ids(model, case['id'], buffer))
    if len(family) != 672 or len({(key, comp) for key, comp, _ in family}) != 672:
        raise AssertionError('Incomplete sector-ceiling comparisons')
    return family


def parent_ready():
    review = json.loads((PRIOR / 'validation_complete.json').read_text())
    if review['status'] != 'numerical_validation_complete' or review['portfolios'] != 744:
        raise RuntimeError('Finish main relation-model validation')
    summary = json.loads((PRIOR / 'evaluation_summary.json').read_text())
    if (summary['portfolios'], summary['joint_hypotheses']) != (744, 11744): raise RuntimeError('Unexpected relation parent')
    if summary['robust_candidate_gate_passed'] or review['candidates']:
        raise RuntimeError('Prioritize parent candidate stress and fresh confirmation')


def prior_family():
    family, differences, returns, _ = relation_prior_family()
    stored = pd.read_csv(PRIOR / 'joint_bootstrap.csv'); added = []
    for key, comp, reference in relation_new_family():
        record = (key, comp, 'stock_relation'); family.append(record); added.append(record)
        differences.append(returns(PRIOR, key) - (returns(PRIOR, reference) if reference is not None else 0.))
    if set(added) != set(stored[stored.origin == 'stock_relation'][['id', 'comparator', 'origin']].itertuples(index=False, name=None)):
        raise AssertionError('Retained relation family changed')
    if len(family) != 11744 or len(set(family)) != 11744: raise AssertionError('Incomplete sector-ceiling prior family')
    means = stored[stored.block == 5].set_index(['id', 'comparator', 'origin'])['mean']
    error = max(abs(float(np.mean(value)) - means.loc[key]) for key, value in zip(family, differences))
    if error > 1e-14: raise AssertionError('Retained relation means changed')
    return family, differences, returns, error


def freeze():
    parent_ready(); plan = json.loads(PLAN_RECORD.read_text())
    expected = dict(models=MODELS, modes=MODES, cases=CASES, buffers=BUFFERS, accounts=656, new_accounts=328,
                    legacy_accounts=328, prior_hypotheses=11744, new_hypotheses=672, joint_hypotheses=12416, new_fits=0)
    for name, value in expected.items():
        if plan[name] != value: raise AssertionError('Registered sector-ceiling plan changed: ' + name)
    parent = json.loads((PRIOR / 'protocol.json').read_text())
    for manifest in [plan['evidence_hashes'], parent['hashes'], json.loads((PRIOR / 'score_hashes.json').read_text()),
                     json.loads((DATA_ROOT / 'data_hashes.json').read_text())]: check_hashes(manifest)
    files = [Path(__file__), PLAN_RECORD, PRIOR / 'protocol.json', PRIOR / 'validation_complete.json',
             PRIOR / 'evaluation_summary.json', PRIOR / 'joint_bootstrap.csv', PRIOR / 'score_hashes.json', DATA_ROOT / 'data_hashes.json']
    files += [Path(__file__).with_name('timefolio_heatmap_' + name + '.py') for name in
              ['stock_relation_evaluation', 'stock_relation_training', 'retention_replay', 'snapshot', 'stock_limits',
               'planned_audit', 'action_amendment', 'week_boundaries', 'walkforward_eval', 'replay', 'study', 'data']]
    files.append(Path(__file__).resolve().parents[1] / 'tests/test_timefolio_sector_ceiling.py')
    for model in MODELS:
        files.append(PRIOR / 'scores' / (model + '.npy'))
        files += [PRIOR / 'portfolios' / (parent_account_id(model, case['id'], buffer) + '.json') for case in CASES for buffer in BUFFERS]
    spec = dict(parent=parent, plan=plan, models=MODELS, modes=MODES,
                hashes={str(path.resolve()): digest(path) for path in files})
    DEST.mkdir(exist_ok=True); path = DEST / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Frozen sector-ceiling protocol changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST / 'frozen_source'; folder.mkdir(exist_ok=True)
        for i, source in enumerate(files):
            if source.suffix == '.py': shutil.copy2(source, folder / (f'{i:03d}_' + source.name))
    return spec


def candidate_ids(frame, stats):
    new = frame[(frame.ceiling == 'formula') & (frame.architecture == 'cnn') & (frame.objective == 'relation')]
    selected = []; added = stats[stats.origin == 'sector_ceiling']
    for row in new[new.member == 'ensemble'].to_dict('records'):
        members = new[(new.relation == row['relation']) & (new.case == row['case']) & (new.buffer == row['buffer'])]
        infer = added[added.id == row['id']]; expected = comparison_ids(row['model'], row['case'], row['buffer'])
        if len(members) != 4 or set(members.member) != set(MEMBERS): raise AssertionError('Incomplete sector-ceiling seeds')
        if len(infer) != len(expected) * 2 or set(infer[['comparator', 'block']].itertuples(index=False, name=None)) != {(comp, b) for comp, _ in expected for b in [5, 10]}:
            raise AssertionError('Incomplete sector-ceiling comparisons')
        stable = ((members['return'] > 0) & (members.mdd > -.2) & (members.positive_blocks >= 2) & (~members.four_week_turnover_stop)).all()
        if stable and (infer.adjusted_p < .025).all() and (infer.simultaneous_lower95 > 0).all(): selected.append(row['id'])
    return selected


def run():
    spec = freeze(); preflight = json.loads((DEST / 'preflight.json').read_text())
    if (preflight['prior_hypotheses'], preflight['joint_hypotheses']) != (11744, 12416): raise RuntimeError('Complete sector-ceiling preflight')
    p, ix, _, _ = context(DATA_ROOT)
    if ix['dates'][-1] != '20260923': raise AssertionError('Reserved outcome boundary changed')
    p, release = amend_actions(p, ix, spec['parent']['training']['actions'])
    panels = {mode: sector_panel(p, mode) for mode in MODES}; caps = historical_stock_caps(ix['codes'], ix['dates'])
    dates = {date: d for d, date in enumerate(ix['dates'])}; rows, audits, regressions = [], {}, []
    folder = DEST / 'portfolios'; folder.mkdir(exist_ok=True); scores = DEST / 'scores'; scores.mkdir(exist_ok=True)
    for model in MODELS:
        path = scores / (model + '.npy')
        if path.exists(): raise RuntimeError('Do not overwrite sector-ceiling scores')
        shutil.copy2(PRIOR / 'scores' / path.name, path); matrix = np.load(path); info = model_info(model)
        held = {r: snapshot_scores(matrix, p['eligible'], ix['dates'], '20260101', '20260923', r) for r in [1, 5]}
        for case in CASES:
            alpha, origins = held[case['refresh']]
            for buffer in BUFFERS:
                for mode in MODES:
                    panel = panels[mode]; key = account_id(model, case['id'], buffer, mode); path = folder / (key + '.json')
                    if path.exists(): raise RuntimeError('Do not overwrite sector-ceiling account')
                    result = replay(panel, ix, alpha, '20260101', '20260923', top_n=12, weight=.05, max_orders=case['max_orders'],
                        rank_buffer=buffer, rebalance=5, rebalance_band=.0005, return_trades=True, planning_price='open',
                        action_release_dates=release, stock_cap_schedule=caps)
                    for trade in result['trades']:
                        d = dates[trade['signal_date']]; origin = origins[d]
                        if not 0 <= origin <= d: raise AssertionError('Future sector-ceiling score')
                        trade['portfolio_score_date'] = ix['dates'][origin]
                    result['metrics'].update(assess_weeks(result)); atomic_json(path, result)
                    if mode == 'research20':
                        prior_key = parent_account_id(model, case['id'], buffer)
                        if result != json.loads((PRIOR / 'portfolios' / (prior_key + '.json')).read_text()): raise AssertionError('Research20 account changed')
                        regressions.append(dict(id=key, parent_id=prior_key, all_fields_equal=True))
                    check = audit_fills(panel, ix, result); check.update(additional_checks(panel, ix, result, None, max_orders=case['max_orders']))
                    check['announced_action_errors'] = release_audit(panel, ix, result, release)
                    check['dated_hynix_audit'] = audit_pre_july_hynix(panel, ix, result)
                    check['snapshot_origin_errors'] = [trade['date'] for trade in result['trades']
                        if trade['portfolio_score_date'] != ix['dates'][origins[dates[trade['signal_date']]]]]
                    if any(check[name] for name in ['post_buy_limit_violations', 'additional_errors', 'announced_action_errors', 'snapshot_origin_errors']) or check['dated_hynix_audit']['post_buy_limit_errors'] or check['maximum_nav_reconstruction_error_krw'] > .01:
                        atomic_json(DEST / 'failed_audit.json', dict(id=key, audit=check)); raise AssertionError('Sector-ceiling account audit failed')
                    audits[key] = check; quarters = calendar_blocks(result['daily'])
                    rows.append(dict(id=key, model=model, **info, case=case['id'], buffer=buffer, band_bp=5, ceiling=mode,
                        **result['metrics'], **quarters, positive_blocks=sum(v > 0 for v in quarters.values())))
        print(json.dumps(dict(model=model, portfolios=len(rows))), flush=True)
    if len(rows) != 656 or len(regressions) != 328: raise AssertionError('Incomplete sector-ceiling grid')
    atomic_json(DEST / 'score_hashes.json', {str(path): digest(path) for path in sorted(scores.glob('*.npy'))})
    atomic_json(DEST / 'independent_audit.json', audits); atomic_json(DEST / 'baseline_regression.json', regressions)
    frame = pd.DataFrame(rows); frame.to_csv(DEST / 'portfolio_summary.csv', index=False)
    family, differences, returns, error = prior_family()
    atomic_json(DEST / 'prior_family_reconstruction.json', dict(prior_hypotheses=len(family), maximum_mean_error=error))
    for key, comp, reference in new_family():
        family.append((key, comp, 'sector_ceiling'))
        differences.append(returns(DEST, key) - (returns(DEST, reference) if reference is not None else 0.))
    matrix = np.column_stack(differences)
    if matrix.shape != (179, 12416) or len(set(family)) != 12416: raise AssertionError('Incomplete sector-ceiling family')
    records = []
    for block in [5, 10]:
        stats = family_bootstrap(matrix, block=block, draws=4000, seed=57)
        for i, (key, comp, origin) in enumerate(family):
            records.append(dict(id=key, comparator=comp, origin=origin, block=block,
                **{name: float(stats[name][i]) for name in ['mean', 'standard_error', 'adjusted_p', 'marginal_p', 'simultaneous_lower95']}))
    stats = pd.DataFrame(records); stats.to_csv(DEST / 'joint_bootstrap.csv', index=False); selected = candidate_ids(frame, stats)
    check_hashes(spec['hashes'])
    atomic_json(DEST / 'evaluation_summary.json', dict(status='development_only', portfolios=656, new_portfolios=328,
        legacy_regressions=328, joint_hypotheses=12416, new_hypotheses=672, robust_candidate_gate_passed=selected,
        independent_confirmation=False, full_contest_compliance_certified=False))
    print(json.dumps(dict(complete=True, robust_candidates=selected)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'run'])
    if parser.parse_args().action == 'freeze': freeze()
    else: run()

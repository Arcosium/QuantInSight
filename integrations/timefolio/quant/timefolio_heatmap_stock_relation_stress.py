"""Conditional execution stress for independently reviewed relation candidates.

This freezes adverse execution checks before relation account outcomes exist.
It cannot replace a failed statistical gate, choose a favorable stress scenario,
open reserved data, or claim independent confirmation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant.timefolio_heatmap_stock_relation_evaluation import DEST as EVAL, comparison_ids, candidate_ids, MEMBERS
from quant.timefolio_heatmap_stock_relation_training import DATA_ROOT, digest, check_hashes
from quant.timefolio_heatmap_action_amendment import amend_actions, release_audit
from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_retention_replay import replay
from quant.timefolio_heatmap_stock_limits import historical_stock_caps, audit_pre_july_hynix
from quant.timefolio_heatmap_week_boundaries import assess_weeks
from quant.timefolio_heatmap_walkforward_eval import calendar_blocks
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks

DEST = EVAL.with_name('20260929_stock_relation_stress_v1')
SCENARIOS = [dict(id='base', slip=.0005, participation=.05, order_ceiling=10),
             dict(id='slippage10bp', slip=.001, participation=.05, order_ceiling=10),
             dict(id='slippage25bp', slip=.0025, participation=.05, order_ceiling=10),
             dict(id='participation1pct', slip=.0005, participation=.01, order_ceiling=10),
             dict(id='orders3', slip=.0005, participation=.05, order_ceiling=3),
             dict(id='combined', slip=.0025, participation=.01, order_ceiling=3)]


def split_account(key):
    parts = key.split('__')
    if len(parts) != 6 or parts[-1] != 'band5bp': raise ValueError('Registered base account required')
    model = parts[0]; case = '__'.join(parts[1:4]); buffer = int(parts[4].removeprefix('buffer'))
    comparisons = comparison_ids(model, case, buffer)
    if not model.endswith('_ensemble'): raise ValueError('Fixed CNN ensemble required')
    return model, case, buffer, comparisons


def candidate_accounts(candidate):
    model, case, buffer, _ = split_account(candidate); suffix = f'__{case}__buffer{buffer}__band5bp'
    accounts = set()
    for member in MEMBERS:
        name = model.removesuffix('ensemble') + member; accounts.add(name + suffix)
        accounts.update(reference for _, reference in comparison_ids(name, case, buffer) if reference is not None)
    return sorted(accounts)


def execution_options(key, scenario):
    if scenario not in SCENARIOS: raise ValueError('Registered execution scenario required')
    parts = key.split('__')
    if len(parts) != 6 or parts[1] != 'n12' or parts[-1] != 'band5bp': raise ValueError('Registered stress account required')
    budget = int(parts[2].removeprefix('orders')); refresh = int(parts[3].removeprefix('refresh'))
    buffer = int(parts[4].removeprefix('buffer'))
    if budget not in [3, 10] or refresh not in [1, 5] or buffer not in [12, 24]: raise ValueError('Unknown base policy')
    return dict(top_n=12, weight=.05, rebalance=5, rebalance_band=.0005, rank_buffer=buffer,
                max_orders=min(budget, scenario['order_ceiling']), slip=scenario['slip'], participation=scenario['participation'])


def freeze_plan():
    files = [Path(__file__), EVAL / 'protocol.json', DATA_ROOT / 'data_hashes.json',
             Path(__file__).resolve().parents[1] / 'tests/test_timefolio_stock_relation_stress.py']
    files += [Path(__file__).with_name('timefolio_heatmap_' + name + '.py') for name in
              ['stock_relation_evaluation', 'retention_replay', 'snapshot', 'stock_limits', 'planned_audit',
               'week_boundaries', 'action_amendment', 'walkforward_eval', 'study', 'data']]
    spec = dict(scenarios=SCENARIOS, scope='Conditional adverse-execution development check; no new significance claim or independent confirmation.',
                trigger='Main numerical validation, same nonempty candidate set in evaluation and independent audit, complete unmodified evidence.',
                population='Every qualifying fixed CNN ensemble, its three seeds and every registered matched comparator under the same six scenarios. Never a best-return fallback.',
                reference='Base scenario must reproduce all account fields exactly. Three-order base policies intentionally duplicate the orders3 case; duplicates are not extra evidence.',
                gate='For every scenario, candidate ensemble and three seeds each require positive return,MDD>-20%,>=2positive quarters,<4turnover failures. Ensemble mean daily excess vs every noncash comparator must remain positive. No favorable scenario or seed selection.',
                independent_followup='A passed stress check only permits separately preregistered fresh confirmation. One reserved session is an operational check, not statistical replication. Do not open reserved outcomes from this module.',
                selection_context='Full11744 development comparisons remain retained; these scenarios cannot rescue a failed parent gate. Separate news147 family unchanged.',
                hashes={str(path.resolve()): digest(path) for path in files})
    DEST.mkdir(exist_ok=True); path = DEST / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Frozen stress plan changed')
    if not path.exists(): atomic_json(path, spec)
    return spec


def reviewed_candidates(root=EVAL):
    """Refuse unfinished, failed, empty, inconsistent or altered parent evidence."""
    root = Path(root); validation = json.loads((root / 'validation_complete.json').read_text())
    if validation['status'] != 'numerical_validation_complete' or validation['portfolios'] != 744:
        raise RuntimeError('Main numerical review required before candidate stress')
    summary = json.loads((root / 'evaluation_summary.json').read_text())
    audit = json.loads((root / 'effect_review.json').read_text())
    candidates = summary['robust_candidate_gate_passed']
    if candidates != audit['independently_recomputed_candidates'] or candidates != validation['candidates']:
        raise RuntimeError('Inconsistent reviewed candidate set')
    if not candidates: raise RuntimeError('No qualifying candidate; do not stress an apparent winner')
    if len(set(candidates)) != len(candidates): raise RuntimeError('Duplicate reviewed candidates')
    required = ['evaluation_summary.json', 'effect_review.json', 'joint_bootstrap.csv', 'portfolio_summary.csv', 'closing_relation_followup.json']
    evidence = validation['main_review_evidence_hashes']
    if not all(str((root / name).resolve()) in evidence for name in required): raise RuntimeError('Incomplete main-review evidence')
    check_hashes(evidence)
    check_hashes(json.loads((root / 'protocol.json').read_text())['hashes'])
    check_hashes(json.loads((root / 'score_hashes.json').read_text()))
    for key in candidates: split_account(key)
    frame = pd.read_csv(root / 'portfolio_summary.csv'); stats = pd.read_csv(root / 'joint_bootstrap.csv')
    if sorted(candidate_ids(frame, stats)) != sorted(candidates): raise RuntimeError('Parent gate no longer reproduces')
    return candidates


def stress_gate(frame, candidates):
    result = []
    for candidate in candidates:
        model, case, buffer, _ = split_account(candidate); suffix = f'__{case}__buffer{buffer}__band5bp'
        basic, excess = [], []
        for scenario in SCENARIOS:
            rows = frame[frame.scenario == scenario['id']].set_index('parent_id', verify_integrity=True)
            for member in MEMBERS:
                key = model.removesuffix('ensemble') + member + suffix; row = rows.loc[key]
                passed = bool(row['return'] > 0 and row['mdd'] > -.2 and row['positive_blocks'] >= 2 and not row['four_week_turnover_stop'])
                basic.append(dict(scenario=scenario['id'], member=member, passed=passed))
            own = rows.loc[candidate]
            for comp, reference in comparison_ids(model, case, buffer):
                if reference is None: continue
                difference = float(own['daily_mean'] - rows.loc[reference]['daily_mean'])
                excess.append(dict(scenario=scenario['id'], comparator=comp, mean_daily_excess=difference, passed=difference > 0))
        result.append(dict(candidate=candidate, stress_passed=all(r['passed'] for r in basic + excess), seed_basics=basic,
                           ensemble_comparisons=excess, independent_confirmation=False))
    return result


def run():
    # Check eligibility before creating any output or reading market panels.
    candidates = reviewed_candidates(); spec = freeze_plan(); check_hashes(spec['hashes'])
    registration = DEST / 'registered_candidates.json'
    if registration.exists(): raise RuntimeError('Do not repeat a registered candidate stress study')
    accounts = sorted({key for candidate in candidates for key in candidate_accounts(candidate)})
    files = [EVAL / name for name in ['validation_complete.json', 'evaluation_summary.json', 'effect_review.json', 'score_hashes.json']]
    files += [EVAL / 'portfolios' / (key + '.json') for key in accounts]
    hashes = {str(path): digest(path) for path in files}
    atomic_json(registration, dict(candidates=candidates, accounts=accounts, scenarios=SCENARIOS, hashes=hashes))
    p, ix, _, _ = context(DATA_ROOT)
    if ix['dates'][-1] != '20260923': raise AssertionError('Reserved date boundary changed')
    protocol = json.loads((EVAL / 'protocol.json').read_text())
    p, release = amend_actions(p, ix, protocol['training']['actions']); p['sector_cap'] = np.minimum(p['sector_cap'], .20)
    caps = historical_stock_caps(ix['codes'], ix['dates']); dates = {date: d for d, date in enumerate(ix['dates'])}
    folder = DEST / 'portfolios'; folder.mkdir(exist_ok=True); rows, audits, regressions = [], {}, []
    for key in accounts:
        model = key.split('__', 1)[0]; refresh = int(key.split('__')[3].removeprefix('refresh'))
        matrix = np.load(EVAL / 'scores' / (model + '.npy'))
        alpha, origins = snapshot_scores(matrix, p['eligible'], ix['dates'], '20260101', '20260923', refresh)
        for scenario in SCENARIOS:
            name = key + '__stress_' + scenario['id']; path = folder / (name + '.json')
            if path.exists(): raise RuntimeError('Do not overwrite candidate stress account')
            options = execution_options(key, scenario)
            result = replay(p, ix, alpha, '20260101', '20260923', **options, return_trades=True,
                            planning_price='open', action_release_dates=release, stock_cap_schedule=caps)
            for trade in result['trades']:
                d = dates[trade['signal_date']]; origin = origins[d]
                if not 0 <= origin <= d: raise AssertionError('Future stress score')
                trade['portfolio_score_date'] = ix['dates'][origin]
            result['metrics'].update(assess_weeks(result))
            if scenario['id'] == 'base':
                if result != json.loads((EVAL / 'portfolios' / (key + '.json')).read_text()): raise AssertionError('Stress base differs')
                regressions.append(key)
            check = audit_fills(p, ix, result, slip=options['slip'])
            check.update(additional_checks(p, ix, result, None, **{name: options[name] for name in ['slip', 'participation', 'max_orders']}))
            check['announced_action_errors'] = release_audit(p, ix, result, release)
            check['dated_hynix_audit'] = audit_pre_july_hynix(p, ix, result)
            if any(check[name] for name in ['post_buy_limit_violations', 'additional_errors', 'announced_action_errors']) or check['dated_hynix_audit']['post_buy_limit_errors'] or check['maximum_nav_reconstruction_error_krw'] > .01:
                atomic_json(DEST / 'failed_audit.json', dict(id=name, audit=check)); raise AssertionError('Stress account audit failed')
            atomic_json(path, result); audits[name] = check; quarters = calendar_blocks(result['daily'])
            rows.append(dict(id=name, parent_id=key, scenario=scenario['id'], **result['metrics'], **quarters,
                             positive_blocks=sum(v > 0 for v in quarters.values())))
        print(json.dumps(dict(parent=key, stress_accounts=len(rows))), flush=True)
    if len(rows) != len(accounts) * len(SCENARIOS) or len(regressions) != len(accounts): raise AssertionError('Incomplete stress grid')
    check_hashes(hashes); check_hashes(spec['hashes'])
    frame = pd.DataFrame(rows); frame.to_csv(DEST / 'portfolio_summary.csv', index=False)
    atomic_json(DEST / 'independent_audit.json', audits); atomic_json(DEST / 'baseline_regression.json', regressions)
    decisions = stress_gate(frame, candidates)
    atomic_json(DEST / 'stress_summary.json', dict(status='development_stress_main_review_pending', decisions=decisions,
                stress_passed=[r['candidate'] for r in decisions if r['stress_passed']], independent_confirmation=False,
                reserved_outcomes_read=False, full_contest_compliance_certified=False))
    print(json.dumps(dict(stress_accounts=len(rows), stress_passed=sum(r['stress_passed'] for r in decisions))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'run'])
    if parser.parse_args().action == 'freeze': freeze_plan()
    else: run()

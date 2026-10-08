"""Full geometry cohort under the locked-inventory repair, without new fits."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant import timefolio_heatmap_lab_evaluation as geometry
from quant import timefolio_heatmap_feature_evaluation as previous
from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_stock_relation_training import DATA_ROOT, digest, check_hashes
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_sector_ceiling import sector_panel
from quant.timefolio_heatmap_action_amendment import amend_actions, release_audit
from quant.timefolio_heatmap_stock_limits import historical_stock_caps, audit_pre_july_hynix
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_week_boundaries import assess_weeks
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_walkforward_eval import calendar_blocks
from quant.timefolio_heatmap_bounded_bootstrap import family_bootstrap
from quant.timefolio_heatmap_locked_replay import replay

PRIOR = previous.DEST
SOURCE = geometry.DEST
DEST = SOURCE.with_name('20260929_locked_cohort_evaluation_v1')
MODELS, TRAINED, MEMBERS = geometry.MODELS, geometry.TRAINED, geometry.MEMBERS
CASES, BUFFERS, MODES = geometry.CASES, geometry.BUFFERS, geometry.MODES
account_id, model_info, comparison_ids = geometry.account_id, geometry.model_info, geometry.comparison_ids


def account_grid():
    return [(m, c, b, s) for m in MODELS for c in CASES for b in BUFFERS for s in MODES]


def new_family():
    """Prediction contrasts and paired rule effects have distinct origins."""
    predictive = [(key, comp, 'locked_geometry', DEST, ref) for key, comp, ref in geometry.new_family()]
    paired = [(account_id(m, c['id'], b, s), 'same_account_old_rule', 'locked_rule_effect',
               SOURCE, account_id(m, c['id'], b, s)) for m, c, b, s in account_grid()]
    result = predictive + paired
    assert len(result) == len({row[:3] for row in result}) == 13136
    return result


def prior_family():
    family, differences, returns, _ = previous.prior_family()
    for key, comp, ref in previous.new_family():
        family.append((key, comp, 'feature_lab'))
        differences.append(returns(PRIOR, key) - (returns(PRIOR, ref) if ref else 0.))
    means = pd.read_csv(PRIOR / 'joint_bootstrap.csv').query('block == 5').set_index(['id', 'comparator', 'origin'])['mean']
    assert set(family) == set(means.index) and len(family) == 43808
    error = max(abs(float(np.mean(a)) - means.loc[key]) for key, a in zip(family, differences))
    assert error < 1e-14
    return family, differences, returns, error


def candidate_ids(frame, stats):
    # Qualification still requires predictive value under the new rule.
    # The paired execution-rule effect is retained in the joint correction,
    # but beating one's own old rule is not a predictive qualification gate.
    selected = stats[stats.origin == 'locked_geometry'].copy()
    selected['origin'] = 'geometry_lab'
    return geometry.candidate_ids(frame, selected)


def same_ledger(result, old):
    # The repaired replay always adds a diagnostic counter, even if no target
    # changed. Exclude only that documented field when comparing ledgers.
    metrics = dict(result['metrics'])
    metrics.pop('locked_target_adjusted_days')
    return dict(result, metrics=metrics) == old


def freeze():
    parent = json.loads((PRIOR / 'protocol.json').read_text())
    original = json.loads((SOURCE / 'protocol.json').read_text())
    check_hashes(parent['hashes']); check_hashes(original['hashes'])
    assert parent['plan']['joint_hypotheses'] == 43808
    files = [Path(__file__).resolve(), PRIOR / 'protocol.json', SOURCE / 'protocol.json']
    files += [Path(__file__).with_name('timefolio_heatmap_' + n + '.py') for n in
              ['lab_evaluation', 'feature_evaluation', 'locked_replay', 'locked_targets', 'retention_replay',
               'bounded_bootstrap', 'stock_limits', 'action_amendment', 'planned_audit', 'snapshot',
               'week_boundaries', 'sector_ceiling', 'stock_relation_training', 'walkforward_eval', 'study', 'data']]
    files += [Path(__file__).resolve().parents[1] / 'tests' / 'test_timefolio_locked_evaluation.py']
    plan = dict(accounts=2576, new_gpu_fits=0, old_mode_exact_regressions=2576,
                new_predictive_hypotheses=10560, paired_rule_effect_hypotheses=2576,
                prior_hypotheses=43808, joint_hypotheses=56944,
                blocks=[5, 10], draws=4000, seed=57, family_alpha=.025)
    spec = dict(plan=plan, training=original['training'], models=MODELS, cases=CASES, buffers=BUFFERS,
                execution=dict(locked_repair=True), bootstrap_scratch_ceiling_bytes=268435456,
                prerequisites=[str(SOURCE / 'validation_complete.json'), str(PRIOR / 'validation_complete.json')],
                candidate_gate='Original geometry gate under repaired execution; paired rule effects are explanatory only.',
                rule_selection='Fixed correction to locked inventory; no return-selected toggles or parameters.',
                scope='Development only; retain all comparisons and all geometry accounts, including untrained controls.',
                hashes={str(p.resolve()): digest(p) for p in files})
    DEST.mkdir(exist_ok=True)
    path = DEST / 'protocol.json'
    if path.exists(): assert json.loads(path.read_text()) == spec
    else: atomic_json(path, spec)
    assert len(account_grid()) == 2576 and len(new_family()) == 13136
    atomic_json(DEST / 'preflight.json', dict(plan, prior_main_review_required=True, reserved_outcomes_read=False))
    return spec


def run():
    spec = freeze()
    for root in [SOURCE, PRIOR]:
        review = json.loads((root / 'validation_complete.json').read_text())
        assert review['status'] == 'numerical_validation_complete'
        check_hashes(review['main_review_evidence_hashes'])
    check_hashes(json.loads((SOURCE / 'score_hashes.json').read_text()))
    panel, ix, _, _ = context(DATA_ROOT)
    assert len(ix['dates']) == 255 and ix['dates'][-1] == '20260923'
    panel, release = amend_actions(panel, ix, spec['training']['actions'])
    panels = {mode: sector_panel(panel, mode) for mode in MODES}
    caps = historical_stock_caps(ix['codes'], ix['dates']); days = {d: i for i, d in enumerate(ix['dates'])}
    folder = DEST / 'portfolios'; folder.mkdir(exist_ok=True)
    scores = DEST / 'scores'; scores.mkdir(exist_ok=True)
    rows, audits, regressions, evidence = [], {}, [], {}
    for model in MODELS:
        source = SOURCE / 'scores' / (model + '.npy'); evidence[str(source)] = digest(source)
        matrix = np.load(source); target = scores / source.name
        assert not target.exists(); np.save(target, matrix); assert digest(target) == digest(source)
        held = {r: snapshot_scores(matrix, panel['eligible'], ix['dates'], '20260101', '20260923', r) for r in [1, 5]}
        for case in CASES:
            alpha, origins = held[case['refresh']]
            for buffer in BUFFERS:
                for ceiling in MODES:
                    key = account_id(model, case['id'], buffer, ceiling)
                    output = folder / (key + '.json'); assert not output.exists()
                    old_path = SOURCE / 'portfolios' / output.name; evidence[str(old_path)] = digest(old_path)
                    old = json.loads(old_path.read_text()); results = {}
                    for enabled in [False, True]:
                        result = replay(panels[ceiling], ix, alpha, '20260101', '20260923', top_n=12, weight=.05,
                                        max_orders=case['max_orders'], rank_buffer=buffer, rebalance=5,
                                        rebalance_band=.0005, return_trades=True, planning_price='open',
                                        action_release_dates=release, stock_cap_schedule=caps, locked_repair=enabled)
                        for trade in result['trades']:
                            d = days[trade['signal_date']]; assert 0 <= origins[d] <= d
                            trade['portfolio_score_date'] = ix['dates'][origins[d]]
                        result['metrics'].update(assess_weeks(result)); results[enabled] = result
                    assert results[False] == old
                    result = results[True]; atomic_json(output, result)
                    regressions.append(dict(id=key, old_mode_all_fields_exact=True,
                                            new_mode_ledger_equal_excluding_repair_counter=same_ledger(result, old),
                                            locked_target_adjusted_days=result['metrics']['locked_target_adjusted_days']))
                    audit = audit_fills(panels[ceiling], ix, result)
                    audit.update(additional_checks(panels[ceiling], ix, result, None, max_orders=case['max_orders']))
                    audit['announced_action_errors'] = release_audit(panels[ceiling], ix, result, release)
                    audit['dated_hynix_audit'] = audit_pre_july_hynix(panels[ceiling], ix, result)
                    audit['snapshot_origin_errors'] = [t['date'] for t in result['trades'] if t['portfolio_score_date'] != ix['dates'][origins[days[t['signal_date']]]]]
                    assert not any(audit[k] for k in ['post_buy_limit_violations', 'additional_errors', 'announced_action_errors', 'snapshot_origin_errors'])
                    assert not audit['dated_hynix_audit']['post_buy_limit_errors'] and audit['maximum_nav_reconstruction_error_krw'] < .01
                    audits[key] = audit; quarters = calendar_blocks(result['daily'])
                    rows.append(dict(id=key, model=model, **model_info(model), case=case['id'], buffer=buffer,
                                     band_bp=5, ceiling=ceiling, **result['metrics'], **quarters,
                                     positive_blocks=sum(v > 0 for v in quarters.values())))
        print(json.dumps(dict(model=model, portfolios=len(rows))), flush=True)
    assert len(rows) == len(regressions) == 2576
    check_hashes(evidence); atomic_json(DEST / 'source_artifact_hashes.json', evidence)
    atomic_json(DEST / 'score_hashes.json', {str(p): digest(p) for p in sorted(scores.glob('*.npy'))})
    atomic_json(DEST / 'baseline_regression.json', regressions); atomic_json(DEST / 'independent_audit.json', audits)
    frame = pd.DataFrame(rows); frame.to_csv(DEST / 'portfolio_summary.csv', index=False)
    family, differences, returns, error = prior_family()
    atomic_json(DEST / 'prior_family_reconstruction.json', dict(prior_hypotheses=len(family), maximum_mean_error=error))
    for key, comp, origin, root, ref in new_family():
        family.append((key, comp, origin))
        differences.append(returns(DEST, key) - (returns(root, ref) if ref else 0.))
    matrix = np.column_stack(differences); assert matrix.shape == (179, 56944) and len(set(family)) == 56944
    records = []
    for block in [5, 10]:
        stats = family_bootstrap(matrix, block=block, draws=4000, seed=57)
        for i, (key, comp, origin) in enumerate(family):
            records.append(dict(id=key, comparator=comp, origin=origin, block=block,
                                **{n: float(stats[n][i]) for n in ['mean', 'standard_error', 'adjusted_p', 'marginal_p', 'simultaneous_lower95']}))
    stats = pd.DataFrame(records); stats.to_csv(DEST / 'joint_bootstrap.csv', index=False)
    selected = candidate_ids(frame, stats); check_hashes(spec['hashes']); check_hashes(evidence)
    atomic_json(DEST / 'evaluation_summary.json', dict(status='development_only', portfolios=2576, model_fits=0,
                old_mode_exact_regressions=2576, unchanged_accounts=sum(r['new_mode_ledger_equal_excluding_repair_counter'] for r in regressions),
                joint_hypotheses=56944, new_hypotheses=13136, robust_candidate_gate_passed=selected,
                independent_confirmation=False, full_contest_compliance_certified=False))
    print(json.dumps(dict(complete=True, robust_candidates=selected)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'run'])
    if parser.parse_args().action == 'freeze': freeze()
    else: run()

"""Full prior-family correction for the frozen cash-control experiment."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from quant.timefolio_heatmap_cash_accounts import hypotheses, units, identity, MEMBERS, CONTROLS
from quant.timefolio_heatmap_fleet_accounts import policy_grid
from quant.timefolio_heatmap_fleet_inference import new_family
from quant.timefolio_heatmap_fleet_audit_helpers import count_bootstrap
from quant.timefolio_heatmap_fleet_metrics import monthly_target
from quant.timefolio_heatmap_fleet_thresholds import classify
from quant.timefolio_heatmap_parallel_bootstrap import family_bootstrap
from quant.timefolio_heatmap_gpu_worker import digest, write

ORIGIN = 'cash_control_v1'


def read(p):
    return json.loads(p.read_text())


def nav_returns(result):
    values = np.asarray([1e9] + [r['nav'] for r in result['daily']])
    assert len(values) == 180 and np.isfinite(values).all() and (values > 0).all()
    return values[1:] / values[:-1] - 1


def prior_matrix(root):
    for name, sha in read(root / 'input_hashes.json').items(): assert digest(root / name) == sha
    family = [tuple(r) for r in read(root / 'prior_family.json')]
    old = np.load(root / 'prior_differences.npy', allow_pickle=False)
    assert old.shape == (179, 68720) and len(family) == 68720
    returns = {}
    for source in sorted((root / 'results').iterdir()):
        ids = read(source / 'account_ids.json')
        values = np.load(source / 'daily_returns.npy', allow_pickle=False)
        assert values.shape == (179, 128)
        for i, key in enumerate(ids):
            assert key not in returns
            returns[key] = values[:, i]
    for row in read(root / 'online.json'):
        returns[row['summary']['id']] = np.asarray(row['daily_returns'])
    assert len(returns) == 19984
    additions = new_family()
    matrix = np.empty((179, 122096))
    matrix[:, :68720] = old
    for i, (key, comparator, ref) in enumerate(additions, 68720):
        matrix[:, i] = returns[key] - (returns[ref] if ref else 0.)
        family.append((key, comparator, 'fleet_factorial'))
    assert len(family) == matrix.shape[1] == 122096
    return family, matrix, returns


def candidates(cases, frame, stats, p_column='adjusted_p'):
    accounts = frame.set_index('id').to_dict('index')
    tests = {key: group for key, group in stats[stats.origin == ORIGIN].groupby('id', sort=False)}
    selected, basic = [], []
    expected = {(label, block) for label in ['cash', 'unchanged_baseline',
                'same_exposure_nonimage', 'matching_untrained_pipeline'] for block in [5, 10]}
    for case in cases:
        for policy in policy_grid():
            for control in CONTROLS:
                ids = [identity(case, 'trained', m, policy, control) for m in MEMBERS]
                rows = [accounts[key] for key in ids]
                stable = all(r['return'] > 0 and r['mdd'] > -.2 and r['positive_blocks'] >= 2
                             and not r['four_week_turnover_stop'] for r in rows)
                key = ids[-1]
                group = tests[key]
                assert len(group) == 8 and set(group[['comparator', 'block']].itertuples(index=False, name=None)) == expected
                if stable: basic.append(key)
                if stable and (group[p_column] < .025).all() and (group.simultaneous_lower95 > 0).all():
                    selected.append(key)
    return selected, basic


def run(prior, root, accounts, output):
    assert not output.exists(); output.mkdir(parents=True)
    registration = read(root / 'execution_registration.json')
    done = read(accounts / 'complete.json')
    assert not done['stopped'] and done['remaining'] == 0 and len(done['completed']) == 72
    assert not (accounts / 'STOP.json').exists()
    for name, sha in read(root / 'input_hashes.json').items(): assert digest(root / name) == sha
    assert digest(prior / 'input_hashes.json') == registration['prior_family_manifest_sha256']
    family, old, returns = prior_matrix(prior)
    rows, records, stops, proofs = [], [], [], {}
    maximum_nav_error = 0.
    closing_unplanned = 0
    for unit in units(registration['cases']):
        source = accounts / 'units' / unit['id']
        complete = read(source / 'complete.json')
        assert complete['accounts'] == 224 and complete['completed'] and complete['baseline_parities'] == 16
        assert digest(source / 'summary.json') == complete['summary_sha256']
        proofs[unit['id']] = digest(source / 'complete.json')
        for row in read(source / 'summary.json'):
            key = row['id']; folder = source / key
            receipt = read(folder / 'complete.json')
            assert receipt['independent_account_arithmetic_passed']
            for name, sha in receipt['hashes'].items(): assert digest(folder / name) == sha
            result = read(folder / 'portfolio.json')
            expected = monthly_target(result['daily'])
            assert expected == read(folder / 'monthly.json')
            classified = classify(expected)
            assert all(row[k] == v for k, v in classified.items())
            assert key not in returns
            returns[key] = nav_returns(result)
            checked = read(folder / 'audit.json')
            maximum_nav_error = max(maximum_nav_error, checked['maximum_nav_reconstruction_error_krw'],
                                    checked['closing']['exposure']['maximum_nav_error'])
            closing_unplanned += len(checked['closing']['unplanned_exceptions'])
            if classified['retain_candidate']: records.append(dict(id=key, **classified, monthly=expected))
            if classified['stop_search_and_validate']: stops.append(dict(id=key, **classified, monthly=expected))
            rows.append(row)
    assert len(rows) == registration['new_heatmap_accounts'] + registration['new_paired_nonimage_accounts'] == 16128
    assert maximum_nav_error < .01
    additions = [tuple(r) for r in read(root / 'hypotheses.json')]
    assert additions == hypotheses(registration['cases']) and len(additions) == 36288
    matrix = np.empty((179, registration['joint_hypotheses']))
    matrix[:, :old.shape[1]] = old
    for i, (key, comparator, ref) in enumerate(additions, old.shape[1]):
        matrix[:, i] = returns[key] - (returns[ref] if ref else 0.)
        family.append((key, comparator, ORIGIN))
    assert matrix.shape == (179, 158384) and len(set(family)) == len(family) == 158384
    # Preserve every searched difference for subsequent cohorts, including controls.
    np.save(output / 'joint_differences.npy', matrix)
    write(output / 'family.json', family)
    frame = pd.DataFrame(rows); frame.to_csv(output / 'portfolio_summary.csv', index=False)
    write(output / 'retained_candidates.json', records)
    write(output / 'stop_candidates.json', stops)
    write(output / 'account_review.json', dict(accounts=len(rows), source_unit_hashes=proofs,
        maximum_nav_error=maximum_nav_error, unplanned_closing_events=closing_unplanned,
        all_baseline_ledgers_reproduced=1152, full_contest_compliance_certified=False))
    statistical = []
    for block in [5, 10]:
        started = time.monotonic()
        report = family_bootstrap(matrix, block=block, draws=4000, seed=57, workers=8)
        for i, (key, comp, origin) in enumerate(family):
            statistical.append(dict(id=key, comparator=comp, origin=origin, block=block,
                **{n: float(report[n][i]) for n in
                   ['mean', 'standard_error', 'adjusted_p', 'marginal_p', 'simultaneous_lower95']}))
        write(output / f'block{block}_receipt.json', dict(seconds=time.monotonic()-started,
            hypotheses=len(family), draws=4000, critical_max_t=report['critical_max_t']))
    stats = pd.DataFrame(statistical); stats.to_csv(output / 'joint_bootstrap.csv', index=False)
    reports, bounds = count_bootstrap(family, matrix.T, stats)
    bounds.to_csv(output / 'bootstrap_roundoff_bounds.csv', index=False)
    selected, basic = candidates(registration['cases'], frame, stats)
    conservative, basic_again = candidates(registration['cases'], frame, bounds, 'adjusted_high')
    assert selected == conservative and basic == basic_again
    write(output / 'evaluation_summary.json', dict(at=time.time(), status='main_review_pending',
        accounts=len(rows), new_hypotheses=len(additions), joint_hypotheses=len(family),
        record_accounts=len(records), stop_accounts=len(stops), all_seed_basic_cases=len(basic),
        statistical_candidates=selected, current_target_and_statistical_candidates=sorted(
            set(selected) & {r['id'] for r in stops}), independent_bootstrap=reports,
        reused_development_only=True, independent_confirmation=False,
        full_contest_compliance_certified=False, reserved_outcomes_read=False, orders_submitted=False))
    write(output / 'artifact_hashes.json', {p.name:digest(p) for p in output.iterdir() if p.is_file()})
    write(output / 'complete.json', dict(status='main_review_pending',
        input_manifest_sha256=digest(root / 'input_hashes.json'), source_sha256=digest(Path(__file__)),
        prior_family_hypotheses=122096, new_hypotheses=36288, joint_hypotheses=158384))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['prior', 'root', 'accounts', 'output']: p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); run(a.prior, a.root, a.accounts, a.output)

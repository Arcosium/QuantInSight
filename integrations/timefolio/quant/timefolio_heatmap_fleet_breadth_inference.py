"""Preserve every breadth comparison; resample the full family when candidates exist."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from quant.timefolio_heatmap_fleet_breadth import hypotheses, identity, MEMBERS, policies
from quant.timefolio_heatmap_fleet_breadth_accounts import units
from quant.timefolio_heatmap_fleet_accounts import account_id, policy_grid
from quant.timefolio_heatmap_fleet_audit_helpers import count_bootstrap
from quant.timefolio_heatmap_fleet_metrics import monthly_target
from quant.timefolio_heatmap_fleet_thresholds import classify
from quant.timefolio_heatmap_parallel_bootstrap import family_bootstrap
from quant.timefolio_heatmap_gpu_worker import digest, write

ORIGIN = 'fleet_breadth_v1'


def read(p):
    return json.loads(p.read_text())


def nav_returns(result):
    values = np.asarray([1e9] + [r['nav'] for r in result['daily']])
    assert len(values) == 180 and np.isfinite(values).all() and (values > 0).all()
    return values[1:] / values[:-1] - 1


def prior_matrix(prior, root):
    complete=read(prior/'complete.json');assert complete['joint_hypotheses']==176496
    registration=read(root/'execution_registration.json')
    assert digest(prior/'artifact_hashes.json')==registration['prior_inference_manifest_sha256']
    for name,sha in read(prior/'artifact_hashes.json').items():assert digest(prior/name)==sha
    family=[tuple(row) for row in read(prior/'family.json')]
    matrix=np.load(prior/'joint_differences.npy',allow_pickle=False)
    assert matrix.shape==(179,176496) and len(family)==176496
    returns={}
    for portfolio in (root/'sources').glob('*/portfolios/*.json'):
        assert portfolio.stem not in returns
        returns[portfolio.stem]=nav_returns(read(portfolio))
    assert len(returns)==1152
    return family,matrix,returns


def should_resample(records, stops):
    # No result with either target flag can take the deferred branch.
    return bool(records or stops)


def seal(root, output, *, resampling_performed):
    write(output/'artifact_hashes.json',{p.name:digest(p) for p in output.iterdir() if p.is_file()})
    write(output/'complete.json',dict(status='main_review_pending',
        input_manifest_sha256=digest(root/'input_hashes.json'),source_sha256=digest(Path(__file__)),
        prior_family_hypotheses=176496,new_hypotheses=8688,joint_hypotheses=185184,
        resampling_performed=resampling_performed,final_validation_passed=False))


def candidates(cases, frame, stats, p_column='adjusted_p'):
    accounts=frame.set_index('id').to_dict('index')
    tests={key:group for key,group in stats[stats.origin==ORIGIN].groupby('id',sort=False)}
    expected={(label,block) for label in ['cash','matching_untrained_breadth','same_breadth_nonimage','original_top12'] for block in [5,10]}
    selected,basic=[],[]
    for case in cases:
        for policy in policies():
            ids=[identity(f'{case}_trained_{member}',policy) for member in MEMBERS]
            rows=[accounts[key] for key in ids]
            stable=all(r['return']>0 and r['mdd']>-.2 and r['positive_blocks']>=2 and not r['four_week_turnover_stop'] for r in rows)
            key=ids[-1];group=tests[key]
            assert len(group)==len(expected) and set(group[['comparator','block']].itertuples(index=False,name=None))==expected
            if stable:basic.append(key)
            if stable and (group[p_column]<.025).all() and (group.simultaneous_lower95>0).all():selected.append(key)
    return selected,basic


def run(prior, root, accounts, output):
    assert not output.exists(); output.mkdir(parents=True)
    registration = read(root / 'execution_registration.json')
    design = read(root / 'design_registration.json')
    cases = design['source_cases']
    done = read(accounts / 'complete.json')
    assert not done['stopped'] and done['remaining'] == 0 and len(done['completed']) == 73
    assert not (accounts / 'STOP.json').exists() and not Path(registration['shared_stop']).exists()
    for name, sha in read(root / 'input_hashes.json').items(): assert digest(root / name) == sha
    assert read(prior / 'complete.json')['input_manifest_sha256'] == registration['prior_family_manifest_sha256']
    family, old, returns = prior_matrix(prior, root)
    assert digest(root/'design_registration.json')==registration['design_sha256']
    rows, records, stops, proofs = [], [], [], {}
    maximum_nav_error = 0.
    closing_unplanned = 0
    for unit in units(cases):
        source = accounts / 'units' / unit['id']
        complete = read(source / 'complete.json')
        assert complete['accounts'] == 48 and complete['completed']
        assert complete['original_accounts_reproduced']==16
        assert digest(source/'source_score.json')==complete['source_score_sha256']
        proof=read(source/'source_score.json')
        model='online_nonimage' if unit['nonimage'] else 'flt_'+unit['id']
        expected_sha=registration['online_score_sha256'] if unit['nonimage'] else digest(root/'sources'/unit['case']/'scores'/(model+'.npy'))
        assert proof['model']==model and proof['source_score_sha256']==expected_sha and proof['no_new_fit']
        assert digest(source/'baseline_parity.json')==complete['baseline_parity_sha256']
        parities=read(source/'baseline_parity.json')
        assert len(parities)==16 and {p['id'] for p in parities}=={account_id(model,p) for p in policy_grid()}
        assert max(p['maximum_absolute_float_error'] for p in parities)<.01
        if unit['nonimage']:assert max(p['saved_return_maximum_error'] for p in parities)<1e-12
        assert digest(source / 'summary.json') == complete['summary_sha256']
        proofs[unit['id']] = digest(source / 'complete.json')
        for row in read(source / 'summary.json'):
            key = row['id']; folder = source / key
            receipt = read(folder / 'complete.json')
            assert receipt['independent_account_arithmetic_passed'] and receipt['original_account_parity_passed']
            for name, sha in receipt['hashes'].items(): assert digest(folder / name) == sha
            result = read(folder / 'portfolio.json')
            expected = monthly_target(result['daily'])
            assert expected == read(folder / 'monthly.json')
            classified = classify(expected)
            assert all(row[k] == v for k, v in classified.items())
            assert key not in returns
            returns[key] = nav_returns(result)
            checked = read(folder/'audit.json')
            assert not checked['additional_errors'] and not checked['post_buy_limit_violations']
            assert not checked['announced_action_errors'] and not checked['dated_hynix_audit']['post_buy_limit_errors']
            maximum_nav_error = max(maximum_nav_error, checked['maximum_nav_reconstruction_error_krw'],
                                    checked['closing']['exposure']['maximum_nav_error'])
            closing_unplanned += len(checked['closing']['unplanned_exceptions'])
            if classified['retain_candidate']: records.append(dict(id=key, **classified, monthly=expected))
            if classified['stop_search_and_validate']: stops.append(dict(id=key, **classified, monthly=expected))
            rows.append(row)
    assert len(rows) == registration['accounts'] == 3504
    assert {r['id'] for r in rows} == {r[0] for r in hypotheses(cases)}
    assert maximum_nav_error < .01
    additions = [tuple(r) for r in read(root / 'hypotheses.json')]
    assert additions == hypotheses(cases) and len(additions) == 8688
    matrix = np.empty((179, registration['joint_hypotheses']))
    matrix[:, :old.shape[1]] = old
    for i, (key, comparator, ref) in enumerate(additions, old.shape[1]):
        matrix[:, i] = returns[key] - (returns[ref] if ref else 0.)
        family.append((key, comparator, ORIGIN))
    assert matrix.shape == (179, 185184) and len(set(family)) == len(family) == 185184
    # Preserve every searched difference for subsequent cohorts, including controls.
    np.save(output / 'joint_differences.npy', matrix)
    write(output / 'family.json', family)
    frame = pd.DataFrame(rows); frame.to_csv(output / 'portfolio_summary.csv', index=False)
    write(output / 'retained_candidates.json', records)
    write(output / 'stop_candidates.json', stops)
    write(output / 'account_review.json', dict(accounts=len(rows), source_unit_hashes=proofs,
        maximum_nav_error=maximum_nav_error, unplanned_closing_events=closing_unplanned,
        all_original_score_sources_verified=True,all_original_account_parities_passed=True, full_contest_compliance_certified=False))
    if not should_resample(records, stops):
        write(output/'evaluation_summary.json',dict(at=time.time(),status='main_review_pending',
            accounts=len(rows),new_hypotheses=len(additions),joint_hypotheses=len(family),
            record_accounts=0,stop_accounts=0,resampling_performed=False,
            statistical_candidates=None,statistical_validation='deferred_no_record_or_stop_candidate',
            all_comparison_returns_preserved=True,independent_confirmation=False,
            full_contest_compliance_certified=False,reserved_outcomes_read=False,orders_submitted=False))
        seal(root,output,resampling_performed=False)
        return
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
    selected, basic = candidates(cases, frame, stats)
    conservative, basic_again = candidates(cases, frame, bounds, 'adjusted_high')
    assert selected == conservative and basic == basic_again
    write(output / 'evaluation_summary.json', dict(at=time.time(), status='main_review_pending',
        accounts=len(rows), new_hypotheses=len(additions), joint_hypotheses=len(family),resampling_performed=True,
        record_accounts=len(records), stop_accounts=len(stops), all_seed_basic_cases=len(basic),
        statistical_candidates=selected, current_target_and_statistical_candidates=sorted(
            set(selected) & {r['id'] for r in stops}), independent_bootstrap=reports,
        reused_development_only=True, independent_confirmation=False,
        full_contest_compliance_certified=False, reserved_outcomes_read=False, orders_submitted=False))
    seal(root,output,resampling_performed=True)



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['prior', 'root', 'accounts', 'output']: p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); run(a.prior, a.root, a.accounts, a.output)

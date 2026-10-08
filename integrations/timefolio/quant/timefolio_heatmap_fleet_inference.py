"""All registered factorial comparisons, retaining the complete prior family."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from quant.timefolio_heatmap_fleet_lab import cases, MEMBERS, comparison_cases
from quant.timefolio_heatmap_fleet_accounts import account_id, policy_grid
from quant.timefolio_heatmap_gpu_worker import digest, write
from quant.timefolio_heatmap_streamed_bootstrap import family_bootstrap
from quant.timefolio_heatmap_fleet_audit_helpers import count_bootstrap

ORIGIN = 'fleet_factorial'


def comparison_ids(cfg, member, policy):
    model = f"flt_{cfg['id']}_trained_{member}"
    result = [('cash', None),
        ('own_case_untrained', account_id(f"flt_{cfg['id']}_untrained_{member}", policy)),
        ('online', account_id('online_nonimage', policy))]
    result += [(label, account_id(f'flt_{other}_trained_{member}', policy))
               for label, other in comparison_cases(cfg)]
    if policy['ceiling'] == 'formula':
        result.append(('same_model_research20', account_id(model, dict(policy, ceiling='research20'))))
    return result


def new_family():
    family = [(account_id(f"flt_{cfg['id']}_trained_{member}", policy), label, reference)
              for cfg in cases() for member in MEMBERS for policy in policy_grid()
              for label, reference in comparison_ids(cfg, member, policy)]
    assert len(family) == len({(key, label) for key, label, _ in family}) == 53376
    return family


def model_info(model):
    if model == 'online_nonimage':
        return dict(architecture='nonimage', encoding='none', geometry='none',
                    member='online', objective='online', target='none')
    body, objective, member = model.rsplit('_', 2)
    assert body.startswith('flt_') and objective in ['trained', 'untrained'] and member in MEMBERS
    cfg = next(c for c in cases() if c['id'] == body[4:])
    return dict(architecture=cfg['kind'], encoding='history', geometry=cfg['id'], member=member,
                objective=ORIGIN if objective == 'trained' else 'untrained',
                target='absolute' if objective == 'trained' else 'none')


def candidate_ids(frame, stats, *, p_column='adjusted_p'):
    accounts = frame.set_index('id').to_dict('index')
    tested = {key: group for key, group in stats[stats.origin == ORIGIN].groupby('id', sort=False)}
    selected, basic = [], []
    for cfg in cases():
        for policy in policy_grid():
            ids = [account_id(f"flt_{cfg['id']}_trained_{m}", policy) for m in MEMBERS]
            rows = [accounts[key] for key in ids]
            stable = all(r['return'] > 0 and r['mdd'] > -.2 and r['positive_blocks'] >= 2
                         and not r['four_week_turnover_stop'] for r in rows)
            key = ids[-1]; tests = tested[key]
            expected = {(label, block) for label, _ in comparison_ids(cfg, 'ensemble', policy) for block in [5, 10]}
            assert len(tests) == len(expected)
            assert set(tests[['comparator', 'block']].itertuples(index=False, name=None)) == expected
            if stable: basic.append(key)
            if stable and (tests[p_column] < .025).all() and (tests.simultaneous_lower95 > 0).all():
                selected.append(key)
    return selected, basic


def run(root, output):
    assert not output.exists(); output.mkdir(parents=True)
    registered = json.loads((root / 'registration.json').read_text())
    assert registered['joint_hypotheses'] == 122096
    hashes = json.loads((root / 'input_hashes.json').read_text())
    for name, sha in hashes.items(): assert digest(root / name) == sha
    family = [tuple(v) for v in json.loads((root / 'prior_family.json').read_text())]
    old = np.load(root / 'prior_differences.npy', allow_pickle=False)
    assert len(family) == old.shape[1] == 68720 and old.shape[0] == 179
    returns = {}; rows = []; metric_passed = []
    for cfg in cases():
        source = root / 'results' / cfg['id']
        completed = json.loads((source / 'complete.json').read_text())
        assert completed['case'] == cfg['id'] and completed['accounts'] == 128
        ids = json.loads((source / 'account_ids.json').read_text())
        values = np.load(source / 'daily_returns.npy', allow_pickle=False)
        assert values.shape == (179, 128) and len(set(ids)) == 128
        for i, key in enumerate(ids):
            assert key not in returns; returns[key] = values[:, i]
        for row in json.loads((source / 'summary.json').read_text()):
            rows.append(dict(row, **model_info(row['model']), band_bp=5,
                case=f"n12__orders{row['max_orders']}__refresh{row['refresh']}"))
        for row in json.loads((source / 'paper_proximity.json').read_text()):
            if '_trained_ensemble__' in row['id'] and row['monthly_target']['practical_metrics_passed']:
                metric_passed.append(row['id'])
    online = json.loads((root / 'online.json').read_text())
    assert len(online) == 16
    for row in online:
        assert row['summary']['id'] not in returns
        returns[row['summary']['id']] = np.asarray(row['daily_returns'])
        rows.append(dict(row['summary'], **model_info('online_nonimage')))
    assert len(returns) == len(rows) == 19984
    frame = pd.DataFrame(rows); frame.to_csv(output / 'portfolio_summary.csv', index=False)
    new = new_family()
    matrix = np.empty((179, 122096))
    matrix[:, :68720] = old
    for i, (key, comp, reference) in enumerate(new, 68720):
        matrix[:, i] = returns[key] - (returns[reference] if reference else 0.)
        family.append((key, comp, ORIGIN))
    assert len(set(family)) == len(family) == matrix.shape[1] == 122096
    write(output / 'family.json', family)
    records = []
    for block in [5, 10]:
        started = time.monotonic()
        report = family_bootstrap(matrix, block=block, draws=4000, seed=57)
        for i, (key, comp, origin) in enumerate(family):
            records.append(dict(id=key, comparator=comp, origin=origin, block=block,
                **{n: float(report[n][i]) for n in
                   ['mean', 'standard_error', 'adjusted_p', 'marginal_p', 'simultaneous_lower95']}))
        write(output / f'block{block}_receipt.json', dict(seconds=time.monotonic()-started,
            hypotheses=122096, draws=4000, critical_max_t=report['critical_max_t']))
    stats = pd.DataFrame(records); stats.to_csv(output / 'joint_bootstrap.csv', index=False)
    # An independent day-count multiplication checks every retained hypothesis.
    reports, bounds = count_bootstrap(family, matrix.T, stats)
    bounds.to_csv(output / 'bootstrap_roundoff_bounds.csv', index=False)
    selected, basic = candidate_ids(frame, stats)
    conservative, basic_again = candidate_ids(frame, bounds, p_column='adjusted_high')
    assert selected == conservative and basic == basic_again
    for name, sha in hashes.items(): assert digest(root / name) == sha
    write(output / 'evaluation_summary.json', dict(status='numerical_checks_complete_main_review_pending',
        portfolios=19984, joint_hypotheses=122096, new_hypotheses=53376,
        all_seed_basic_cases=len(basic), robust_candidate_gate_passed=selected,
        practical_metric_passed=sorted(metric_passed),
        practical_and_statistical_passed=sorted(set(selected) & set(metric_passed)),
        independent_bootstrap=reports, reused_development_only=True,
        independent_confirmation=False, full_contest_compliance_certified=False,
        reserved_outcomes_read=False, orders_submitted=False))
    write(output / 'artifact_hashes.json', {p.name: digest(p) for p in output.iterdir() if p.is_file()})
    write(output / 'complete.json', dict(status='main_review_pending', portfolios=19984,
        hypotheses=122096, input_manifest_sha256=digest(root / 'input_hashes.json')))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); args = ap.parse_args()
    run(args.root, args.output)

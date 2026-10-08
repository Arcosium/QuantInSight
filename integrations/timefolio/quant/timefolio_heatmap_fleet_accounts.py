"""Portable account evaluation using the existing frozen research functions."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from quant.timefolio_heatmap_gpu_worker import digest, verify, write, predict
from quant.timefolio_heatmap_lab_models import LabNet
from quant.timefolio_heatmap_fleet_lab import cases, SEEDS, MEMBERS
from quant.timefolio_heatmap_study import score_matrix
from quant.timefolio_heatmap_gpu_merge import merge_months
from quant.timefolio_heatmap_consensus import rank_consensus
from quant.timefolio_heatmap_action_amendment import amend_actions, release_audit
from quant.timefolio_heatmap_stock_limits import historical_stock_caps, audit_pre_july_hynix
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_locked_replay import replay
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_week_boundaries import assess_weeks
from quant.timefolio_heatmap_walkforward_eval import calendar_blocks


def policy_grid():
    return [dict(max_orders=o, refresh=r, buffer=b, ceiling=s)
            for o in [3, 10] for r in [1, 5] for b in [12, 24] for s in ['research20', 'formula']]


def account_id(model, policy):
    return (f"{model}__n12__orders{policy['max_orders']}__refresh{policy['refresh']}"
            f"__buffer{policy['buffer']}__band5bp__sector_{policy['ceiling']}")


def load_market(root):
    plan = json.loads((root / 'plan.json').read_text())
    ix = json.loads((root / 'panel_index.json').read_text())
    assert len(ix['dates']) == 255 and ix['dates'][0] == '20250908' and ix['dates'][-1] == '20260923'
    p = dict(np.load(root / 'panel.npz', allow_pickle=False))
    p['eligible'] = p['eligible'].astype(bool); p['sector'] = p['sector'].astype(int)
    p, release = amend_actions(p, ix, plan['actions'])
    samples = np.load(root / 'samples.npz', allow_pickle=False)
    return p, ix, release, samples['ci'], samples['di']


def one_account(p, ix, release, caps, held, model, policy):
    alpha, origins = held[policy['refresh']]
    panel = dict(p)
    panel['sector_cap'] = np.minimum(p['sector_cap'], .2) if policy['ceiling'] == 'research20' else p['sector_cap'].copy()
    result = replay(panel, ix, alpha, '20260101', '20260923', top_n=12, weight=.05,
        max_orders=policy['max_orders'], rank_buffer=policy['buffer'], rebalance=5,
        rebalance_band=.0005, return_trades=True, planning_price='open',
        action_release_dates=release, stock_cap_schedule=caps, locked_repair=True)
    dates = {d: i for i, d in enumerate(ix['dates'])}
    for trade in result['trades']:
        day = dates[trade['signal_date']]; origin = origins[day]
        assert 0 <= origin <= day
        trade['portfolio_score_date'] = ix['dates'][origin]
    result['metrics'].update(assess_weeks(result))
    audit = audit_fills(panel, ix, result)
    audit.update(additional_checks(panel, ix, result, None, max_orders=policy['max_orders']))
    audit['announced_action_errors'] = release_audit(panel, ix, result, release)
    audit['dated_hynix_audit'] = audit_pre_july_hynix(panel, ix, result)
    audit['snapshot_origin_errors'] = [t['date'] for t in result['trades']
        if t['portfolio_score_date'] != ix['dates'][origins[dates[t['signal_date']]]]]
    assert not any(audit[k] for k in ['post_buy_limit_violations', 'additional_errors',
        'announced_action_errors', 'snapshot_origin_errors'])
    assert not audit['dated_hynix_audit']['post_buy_limit_errors']
    assert audit['maximum_nav_reconstruction_error_krw'] < .01
    blocks = calendar_blocks(result['daily'])
    summary = dict(id=account_id(model, policy), model=model, **policy, **result['metrics'],
                   **blocks, positive_blocks=sum(v > 0 for v in blocks.values()))
    return result, audit, summary


def compare_nested(actual, expected, path='', errors=None):
    """Discrete decisions must be exact; only arithmetic roundoff is tolerated."""
    if errors is None: errors = dict(maximum_absolute_float_error=0., floats=0)
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for k in expected: compare_nested(actual[k], expected[k], path + '/' + str(k), errors)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected)): compare_nested(a, b, path + '/' + str(i), errors)
    elif isinstance(expected, float):
        delta = abs(actual - expected)
        # Financial absolute values can be1e9; tolerance remains below one cent.
        assert np.isclose(actual, expected, rtol=1e-13, atol=1e-10, equal_nan=False), (path, delta)
        assert delta < .01, (path, delta)
        errors['maximum_absolute_float_error'] = max(errors['maximum_absolute_float_error'], delta)
        errors['floats'] += 1
    else:
        assert type(actual) is type(expected) and actual == expected, path
    return errors


def parity(root, out):
    verify(root); torch.set_num_threads(1)
    fixture = json.loads((root / 'parity_fixture.json').read_text())
    p, ix, release, _, _ = load_market(root)
    caps = historical_stock_caps(ix['codes'], ix['dates']); reports = []
    start = time.monotonic()
    for model, fixture_rows in fixture.items():
        matrix = np.load(root / 'parity_scores' / (model + '.npy'), allow_pickle=False)
        held = {r: snapshot_scores(matrix, p['eligible'], ix['dates'], '20260101', '20260923', r) for r in [1, 5]}
        for policy, expected, expected_audit in fixture_rows:
            result, audit, _ = one_account(p, ix, release, caps, held, model, policy)
            reports.append(dict(id=account_id(model, policy), ledger=compare_nested(result, expected),
                audit=compare_nested(audit, expected_audit), decisions_exact=True))
    assert len(reports) == 32
    write(out, dict(status='remote_parity_passed', accounts=32, reports=reports,
        seconds=time.monotonic() - start, manifest_sha256=digest(root / 'manifest.json'),
        numpy=np.__version__, torch=str(torch.__version__)))


def evaluate(root, case):
    verify(root); torch.set_num_threads(1)
    plan = json.loads((root / 'plan.json').read_text())
    cfg = next(c for c in cases() if c['id'] == case)
    out = root / 'accounts' / case
    assert not out.exists(); out.mkdir(parents=True)
    p, ix, release, ci, di = load_market(root)
    caps = historical_stock_caps(ix['codes'], ix['dates'])
    raw = np.load(root / 'images_history.npy', mmap_mode='r')
    x = torch.from_numpy(np.array(raw[:, None], copy=True))
    matrices = {}; checks = []
    for seed in SEEDS:
        name = f'{case}_seed{seed}'; job = root / 'jobs' / name
        done = json.loads((job / 'complete.json').read_text())
        assert done['package_sha256'] == digest(root / 'manifest.json') and done['folds'] == 9
        hashes = json.loads((job / 'artifact_hashes.json').read_text())
        for rel, sha in hashes.items(): assert digest(job / rel) == sha
        model_root = job / 'models' / name
        masks = np.load(root / f"labels_h{cfg['horizon']}_w0_v20.npz")
        for fold in plan['folds']:
            month = fold['month']; r = json.loads((model_root / (month + '.json')).read_text())
            assert r['fold'] == fold and r['refit_epochs'] == cfg['epochs'] and not r['inner_validation_used']
            assert r['refit_fit']['validation_rows'] == 0
            assert r['refit_fit']['training_rows'] == int(masks[month + '_rf'].sum())
            assert max(di[masks[month + '_rf']] + cfg['horizon']) < fold['start']
        model = LabNet(cfg)
        saved = torch.load(model_root / '202609.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(saved['state_dict'])
        expected = np.load(model_root / '202609.pred.npy'); ids = np.flatnonzero(np.isfinite(expected))
        actual = predict(model, x, ids)
        np.testing.assert_allclose(actual, expected[ids], rtol=1e-4, atol=1e-5)
        matrices[f'flt_{case}_trained_seed{seed}'] = score_matrix(merge_months(model_root, di, ix['dates']), ci, di, p['close'].shape)
        torch.manual_seed(seed); fresh = LabNet(cfg)
        initial = torch.load(job / 'untrained.pt', map_location='cpu', weights_only=True)
        assert fresh.state_dict().keys() == initial.keys()
        assert all(torch.equal(v, initial[k]) for k, v in fresh.state_dict().items())
        values = np.load(job / 'untrained.pred.npy')
        probes = np.unique(np.linspace(0, len(di) - 1, 384, dtype=int))
        observed = predict(fresh, x, probes)
        np.testing.assert_allclose(observed, values[probes], rtol=1e-4, atol=1e-5)
        matrices[f'flt_{case}_untrained_seed{seed}'] = score_matrix(values, ci, di, p['close'].shape)
        checks.append(dict(model=name, trained_rows=len(ids), maximum_prediction_error=float(np.max(np.abs(actual - expected[ids]))),
            untrained_rows=len(probes), initial_weights_exact=True,
            maximum_untrained_prediction_error=float(np.max(np.abs(observed - values[probes])))))
    del x
    for objective in ['trained', 'untrained']:
        prefix = f'flt_{case}_{objective}_'
        members = {prefix + m: matrices[prefix + m] for m in MEMBERS[:3]}
        matrices[prefix + 'ensemble'] = rank_consensus(members, {m: cfg['kind'] for m in members}, p['eligible'], 'mean')
    (out / 'portfolios').mkdir(); (out / 'scores').mkdir()
    rows = []; audits = {}
    for model, matrix in matrices.items():
        np.save(out / 'scores' / (model + '.npy'), matrix)
        held = {r: snapshot_scores(matrix, p['eligible'], ix['dates'], '20260101', '20260923', r) for r in [1, 5]}
        for policy in policy_grid():
            result, audit, row = one_account(p, ix, release, caps, held, model, policy)
            write(out / 'portfolios' / (row['id'] + '.json'), result)
            rows.append(row); audits[row['id']] = audit
    assert len(rows) == 128
    write(out / 'summary.json', rows); write(out / 'audits.json', audits)
    write(out / 'checkpoint_reproduction.json', checks)
    verify(root)
    hashes = {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write(out / 'artifact_hashes.json', hashes)
    write(out / 'complete.json', dict(case=case, accounts=128, status='raw_accounts_audited',
        package_sha256=digest(root / 'manifest.json'), main_review_required=True,
        closing_followup_required=True, family_inference_pending=True,
        independent_confirmation=False, orders_submitted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--case'); parser.add_argument('--parity-output', type=Path)
    args = parser.parse_args()
    if args.parity_output: parity(args.root, args.parity_output)
    else: evaluate(args.root, args.case)

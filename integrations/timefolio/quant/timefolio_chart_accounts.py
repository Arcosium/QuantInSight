"""Chart forecasts with the unchanged long-only account and closing audits."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from quant.timefolio_chart_lab import cases, SEEDS, MEMBERS
from quant.timefolio_chart_models import ChartNet
from quant.timefolio_heatmap_gpu_worker import digest, verify, write, predict
from quant.timefolio_heatmap_fleet_accounts import load_market, policy_grid, account_id
from quant.timefolio_heatmap_study import score_matrix
from quant.timefolio_heatmap_gpu_merge import merge_months
from quant.timefolio_heatmap_consensus import rank_consensus
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_stock_limits import historical_stock_caps
from quant.timefolio_heatmap_cash_accounts import simulate, audit, publish_threshold
from quant.timefolio_heatmap_fleet_metrics import monthly_target


def matrices(root, case, raw, ix, ci, di):
    cfg = next(c for c in cases() if c['id'] == case); result = {}; checks = []
    images = root / f"images_{cfg['view']}.npy"
    image_receipt = json.loads(images.with_suffix('.json').read_text())
    assert digest(images) == image_receipt['sha256']
    x = torch.from_numpy(np.array(np.load(images, mmap_mode='r')[:, None], copy=True))
    masks = np.load(root / f"labels_h{cfg['horizon']}_w0_v20.npz")
    plan = json.loads((root / 'plan.json').read_text())
    for seed in SEEDS:
        name = f'{case}_seed{seed}'; job = root / 'jobs' / name
        done = json.loads((job / 'complete.json').read_text())
        assert done['package_sha256'] == digest(root / 'manifest.json') and done['folds'] == 9
        runtime = json.loads((job / 'runtime.json').read_text()); assert runtime['images_sha256'] == image_receipt['sha256']
        for rel, sha in json.loads((job / 'artifact_hashes.json').read_text()).items(): assert digest(job / rel) == sha
        models = job / 'models' / name
        for fold in plan['folds']:
            month = fold['month']; r = json.loads((models / (month + '.json')).read_text())
            assert r['fold'] == fold and r['refit_epochs'] == cfg['epochs'] and not r['inner_validation_used']
            assert r['refit_fit']['validation_rows'] == 0
            assert r['refit_fit']['training_rows'] == int(masks[month + '_rf'].sum())
            assert max(di[masks[month + '_rf']] + cfg['horizon']) < fold['start']
        model = ChartNet(cfg)
        model.load_state_dict(torch.load(models / '202609.pt', map_location='cpu', weights_only=True)['state_dict'])
        expected = np.load(models / '202609.pred.npy'); ids = np.flatnonzero(np.isfinite(expected))
        actual = predict(model, x, ids); np.testing.assert_allclose(actual, expected[ids], rtol=1e-4, atol=1e-5)
        result[f'chart_{case}_trained_seed{seed}'] = score_matrix(merge_months(models, di, ix['dates']), ci, di, raw['close'].shape)
        torch.manual_seed(seed); fresh = ChartNet(cfg)
        initial = torch.load(job / 'untrained.pt', map_location='cpu', weights_only=True)
        assert fresh.state_dict().keys() == initial.keys()
        assert all(torch.equal(v, initial[k]) for k, v in fresh.state_dict().items())
        values = np.load(job / 'untrained.pred.npy'); probes = np.unique(np.linspace(0, len(di)-1, 384, dtype=int))
        observed = predict(fresh, x, probes); np.testing.assert_allclose(observed, values[probes], rtol=1e-4, atol=1e-5)
        result[f'chart_{case}_untrained_seed{seed}'] = score_matrix(values, ci, di, raw['close'].shape)
        checks.append(dict(model=name, trained_rows=len(ids), initial_weights_exact=True,
            maximum_prediction_error=float(np.max(np.abs(actual-expected[ids]))),
            untrained_rows=len(probes), maximum_untrained_prediction_error=float(np.max(np.abs(observed-values[probes])))))
    for objective in ['trained', 'untrained']:
        prefix = f'chart_{case}_{objective}_'
        members = {prefix+m: result[prefix+m] for m in MEMBERS[:3]}
        result[prefix+'ensemble'] = rank_consensus(members, {m:'cnn' for m in members}, raw['eligible'], 'mean')
    return result, checks


def evaluate(root, case):
    root = Path(root); verify(root); torch.set_num_threads(1)
    output = root / 'accounts' / case
    if output.exists(): raise RuntimeError('Account output already exists')
    output.mkdir(parents=True); (output / 'portfolios').mkdir(); (output / 'scores').mkdir(); (output / 'audits').mkdir()
    raw, ix, release, ci, di = load_market(root); caps = historical_stock_caps(ix['codes'], ix['dates'])
    if case == 'online_nonimage':
        streams, checks = {'online_nonimage': np.load(root / 'online_nonimage.npy')}, []
    else:
        streams, checks = matrices(root, case, raw, ix, ci, di)
    write(output / 'checkpoint_reproduction.json', checks); rows = []
    for model, matrix in streams.items():
        np.save(output / 'scores' / (model + '.npy'), matrix)
        held = {r:snapshot_scores(matrix, raw['eligible'], ix['dates'], '20260101', '20260923', r) for r in [1,5]}
        for policy in policy_grid():
            if (root / 'STOP.json').exists():
                write(output / 'stopped.json', dict(accounts=len(rows), user_target_stop=True)); return
            key = account_id(model, policy)
            panel, result, plans = simulate(raw, ix, release, caps, held, policy)
            # Persist the complete ledger before publishing the threshold. Even
            # a rule-audit failure must not erase a user-requested record.
            write(output / 'portfolios' / (key + '.json'), result)
            monthly = monthly_target(result['daily']); verdict = publish_threshold(root, key, monthly)
            report = audit(raw, panel, ix, release, caps, result, plans, key, policy, None)
            write(output / 'audits' / (key + '.json'), report)
            row = dict(id=key, model=model, case=case, **policy, **result['metrics'],
                       monthly=monthly, **verdict)
            rows.append(row); write(output / 'summary.json', rows)
            write(output / 'progress.json', dict(accounts=len(rows), at=time.time()))
    hashes = {str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write(output / 'artifact_hashes.json', hashes)
    expected = 16 if case == 'online_nonimage' else 128
    assert len(rows) == expected
    write(output / 'complete.json', dict(case=case, accounts=expected,
        package_sha256=digest(root / 'manifest.json'), main_review_required=True,
        closing_audits_complete=True, family_inference_pending=True, independent_confirmation=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--case',required=True)
    a=p.parse_args(); evaluate(a.root,a.case)

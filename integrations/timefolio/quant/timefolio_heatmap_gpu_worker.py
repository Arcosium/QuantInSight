"""Portable, device-matched training; no credentials or live data access."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def verify(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, sha in manifest.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or digest(path) != sha:
            raise AssertionError('Package input hash mismatch: ' + name)
    return manifest


def reference(root):
    spec = importlib.util.spec_from_file_location('heatmap_reference', Path(root) / 'reference.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def predict(model, x, indices):
    model.eval(); values = []
    with torch.no_grad():
        for start in range(0, len(indices), 512):
            out = model(x[indices[start:start + 512]].float().div(127.5).sub(1))
            values.append(out.cpu().numpy())
    return np.concatenate(values) if values else np.array([], np.float32)


def train(ref, x, y, train_mask, val, cfg, di, *, epochs=None):
    """Original date-balanced optimizer with explicit device transfers only."""
    seed = cfg['seed']; torch.manual_seed(seed)
    model = ref.AlternativeNet(cfg['architecture']).to(x.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=.001, foreach=False)
    target = torch.as_tensor(np.nan_to_num(y), dtype=torch.float32, device=x.device)
    groups = ref.date_groups(di, train_mask); va = np.flatnonzero(val)
    if not groups or max(map(len, groups)) > 512: raise ValueError('Invalid date groups')
    if not np.isfinite(y[train_mask | val]).all(): raise ValueError('Nonfinite selected label')
    if cfg['objective'] != 'pairwise': raise ValueError('Registered pairwise objective required')
    best, state, best_epoch, history = -np.inf, None, 0, []
    for epoch in range(epochs or cfg['epochs']):
        model.train(); losses = []
        for group in np.random.default_rng(seed + epoch).permutation(len(groups)):
            ids = groups[group]; opt.zero_grad(set_to_none=True)
            out = model(x[ids].float().div(127.5).sub(1))
            loss = ref.pairwise_loss(out, target[ids])
            if not torch.isfinite(loss): raise AssertionError('Nonfinite training loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
            losses.append(float(loss.detach()))
        metric = 0.
        if len(va):
            values = np.full(len(y), np.nan); values[va] = predict(model, x, va)
            metric = ref.daily_ic(y, values, val, di)
        history.append(dict(epoch=epoch + 1, loss=float(np.mean(losses)), inner_ic=metric))
        if not len(va) or metric > best:
            best, best_epoch = metric, epoch + 1
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, dict(best_epoch=best_epoch, history=history, training_rows=int(train_mask.sum()),
                      validation_rows=int(val.sum()), training_dates=len(groups))


def configure(device):
    torch.set_num_threads(2)
    if device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is required, no CPU fallback')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def benchmark(root, out):
    """Synthetic labels only; timing never selects market outcomes."""
    verify(root); ref = reference(root); configure('cuda')
    raw = np.array(np.load(Path(root) / 'images_snapshot.npy', mmap_mode='r')[:200, None])
    y = np.linspace(-1, 1, 200, dtype=np.float32); di = np.repeat(np.arange(4), 50)
    cfg = dict(architecture='cnn', objective='pairwise', seed=17, lr=.0007, epochs=1)
    tr = np.ones(200, bool); val = np.zeros(200, bool); timing = {}
    for device in ['cpu', 'cuda']:
        x = torch.from_numpy(raw.copy()).to(device)
        for repetition in range(2):
            start = time.monotonic(); model, fit = train(ref, x, y, tr, val, cfg, di)
            if device == 'cuda': torch.cuda.synchronize()
            elapsed = time.monotonic() - start
        timing[device] = elapsed
        assert np.isfinite(predict(model, x, np.arange(200))).all()
    write(Path(out) / 'benchmark.json', dict(seconds=timing, cpu_over_cuda=timing['cpu'] / timing['cuda'],
        gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), cuda=torch.version.cuda, synthetic_labels=True))


def run(root, out, seed, device='cuda'):
    root, out = Path(root), Path(out); verify(root); configure(device)
    if out.exists(): raise RuntimeError('Do not overwrite a worker output')
    out.mkdir(parents=True); ref = reference(root)
    plan = json.loads((root / 'plan.json').read_text())
    if seed not in plan['seeds']: raise ValueError('Unknown shard')
    data = np.load(root / 'labels_masks.npz', allow_pickle=False)
    y, di = data['y'], data['di']; start = time.time()
    runtime = dict(seed=seed, device=device, gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
        torch=str(torch.__version__), numpy=np.__version__, python=platform.python_version(),
        cuda=torch.version.cuda, package_sha256=digest(root / 'manifest.json'), started_at=start)
    write(out / 'runtime.json', runtime)
    if device == 'cuda': benchmark(root, out)
    records = []
    for architecture in plan['architectures']:
        for encoding in plan['encodings']:
            raw = np.load(root / ('images_' + encoding + '.npy'), mmap_mode='r')
            x = torch.from_numpy(np.array(raw[:, None], copy=True)).to(device)
            cfg = dict(id=f'{architecture}_dailygpu_{encoding}_h10_seed{seed}', architecture=architecture,
                objective='pairwise', target='absolute', horizon=10, seed=seed, lr=.0007, epochs=6, encoding=encoding)
            folder = out / 'models' / cfg['id']; folder.mkdir(parents=True)
            for fold in plan['folds']:
                month = fold['month']; masks = {k: data[month + '_' + k] for k in ['tr', 'va', 'rf', 'pr']}
                if not np.isfinite(y[masks['tr'] | masks['va'] | masks['rf']]).all(): raise AssertionError('Labels')
                tick = time.monotonic()
                model, fit = train(ref, x, y, masks['tr'], masks['va'], cfg, di); del model
                model, _ = train(ref, x, y, masks['rf'], np.zeros(len(y), bool), cfg, di, epochs=fit['best_epoch'])
                forecast = np.full(len(y), np.nan, np.float32)
                forecast[masks['pr']] = predict(model, x, np.flatnonzero(masks['pr']))
                if not np.array_equal(np.isfinite(forecast), masks['pr']): raise AssertionError('Prediction coverage')
                path = folder / month
                torch.save(dict(config=cfg, state_dict={k: v.cpu() for k, v in model.state_dict().items()}), path.with_suffix('.pt'))
                np.save(path.with_suffix('.pred.npy'), forecast); del model
                row = dict(config=cfg, fold=fold, fit=fit, refit_rows=int(masks['rf'].sum()),
                    boundaries=plan['boundaries'][month], seconds=time.monotonic() - tick)
                write(path.with_suffix('.json'), row)
                records.append(dict(model=cfg['id'], month=month, seconds=row['seconds']))
                write(out / 'progress.json', dict(completed=len(records), expected=54, latest=records[-1]))
                print(json.dumps(records[-1]), flush=True)
            del x
    if len(records) != 54: raise AssertionError('Incomplete worker shard')
    verify(root)
    artifacts = {str(p.relative_to(out)): digest(p) for p in sorted((out / 'models').glob('*/*'))}
    if len(artifacts) != 162: raise AssertionError('Artifact cardinality')
    write(out / 'artifact_hashes.json', artifacts)
    write(out / 'complete.json', dict(folds=len(records), seconds=time.time() - start,
        package_sha256=runtime['package_sha256'], seed=seed, evaluation_complete=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    a = parser.parse_args(); run(a.root, a.out, a.seed, a.device)

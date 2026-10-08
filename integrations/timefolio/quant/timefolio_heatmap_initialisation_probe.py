"""Outcome-free untrained network controls for the corrected KRX images."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_corrected_features import DEST as DATA_ROOT
from quant.timefolio_heatmap_study import context, score_matrix
from quant.timefolio_heatmap_consensus import rank_consensus
from quant.timefolio_heatmap_walkforward import SOURCE, AlternativeNet, predict

DEST = SOURCE.with_name('20260929_initialisation_probe_v1')
CONFIGS = [dict(id=f'{arch}_untrained_seed{seed}', architecture=arch, seed=seed)
           for arch in ['cnn', 'mlp'] for seed in [17, 29, 43]]


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        a = value.detach().cpu().numpy()
        digest.update(name.encode()); digest.update(str(a.dtype).encode()); digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def freeze():
    data = json.loads((DATA_ROOT / 'protocol.json').read_text())
    for manifest in [data['hashes'], json.loads((DATA_ROOT / 'data_hashes.json').read_text())]:
        for filename, sha in manifest.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != sha: raise AssertionError('Probe input changed')
    files = [Path(__file__), DATA_ROOT / 'protocol.json', DATA_ROOT / 'data_hashes.json']
    files += [Path(__file__).with_name('timefolio_heatmap_' + n + '.py') for n in
              ['walkforward', 'study', 'features', 'consensus', 'data']]
    spec = dict(configs=CONFIGS,
                inputs='Same corrected raw one-channel32x65 images. No new labels, account outcomes, or reserved dates.',
                model='Original AlternativeNet CNN/MLP with exact torch.manual_seed used by trained controls. Eval mode; no optimiser, backward pass, fitted centering, or outcome-selected sign.',
                output='All6 raw untrained forecasts and same-date equal-rank ensemble per architecture. No inversion of unpromising scores; preserve original output signs and all seeds.',
                diagnostic='Compare prediction ranks with frozen trained counterparts; this is an input/initialisation diagnostic, not a performance claim or candidate gate. Any portfolio test needs a separate protocol and retained hypothesis family.',
                hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    DEST.mkdir(exist_ok=True); path = DEST / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Initialisation probe specification changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST / 'frozen_source'; folder.mkdir(exist_ok=True)
        for i, p in enumerate(files):
            if p.suffix == '.py': shutil.copy2(p, folder / (f'{i:03d}_' + p.name))
    return spec


def run():
    spec = freeze(); torch.set_num_threads(4)
    p, ix, ci, di = context(DATA_ROOT)
    if ix['dates'][-1] != '20260923': raise AssertionError('Probe date boundary changed')
    raw = np.load(DATA_ROOT / 'images_raw.npy', mmap_mode='r')
    x = torch.from_numpy(np.array(raw[:, None], copy=True)); ids = np.arange(len(ci))
    folder = DEST / 'models'; folder.mkdir(exist_ok=True); scores = DEST / 'scores'; scores.mkdir(exist_ok=True)
    matrices, checks = {}, []
    for cfg in CONFIGS:
        path = folder / (cfg['id'] + '.pt')
        if path.exists() or path.with_suffix('.pred.npy').exists(): raise RuntimeError('Do not overwrite probe forecasts')
        torch.manual_seed(cfg['seed']); model = AlternativeNet(cfg['architecture']); initial = state_digest(model)
        forecast = predict(model, x, ids)
        if not np.isfinite(forecast).all() or state_digest(model) != initial: raise AssertionError('Probe inference changed weights or returned invalid values')
        torch.save(dict(config=cfg, state_dict=model.state_dict()), path); np.save(path.with_suffix('.pred.npy'), forecast)
        matrices[cfg['id']] = score_matrix(forecast, ci, di, p['close'].shape)
        # Batch composition must not affect this feed-forward eval-mode model.
        subset = np.arange(0, min(len(ids), 1024), 37)
        separate = predict(model, x, subset)
        error = float(np.max(np.abs(separate - forecast[subset])))
        if error > 1e-6: raise AssertionError('Probe depends on inference batch composition')
        torch.manual_seed(cfg['seed']); rebuilt = AlternativeNet(cfg['architecture'])
        if state_digest(rebuilt) != initial: raise AssertionError('Initial weights are not reproducible')
        del model, rebuilt
        checks.append(dict(model=cfg['id'], rows=len(forecast), weights_unchanged=True, initial_sha256=initial,
                           independent_initialisation_exact=True, different_batch_maximum_error=error))
        print(json.dumps(checks[-1]), flush=True)
    for arch in ['cnn', 'mlp']:
        members = {k: v for k, v in matrices.items() if k.startswith(arch + '_')}
        matrices[arch + '_untrained_ensemble'] = rank_consensus(members, {k: arch for k in members}, p['eligible'], 'mean')
    if len(matrices) != 8: raise AssertionError('Probe score streams incomplete')
    for name, matrix in matrices.items(): np.save(scores / (name + '.npy'), matrix)
    outputs = sorted(folder.glob('*.pt')) + sorted(folder.glob('*.pred.npy')) + sorted(scores.glob('*.npy'))
    atomic_json(DEST / 'output_hashes.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs})
    atomic_json(DEST / 'inference_complete.json', dict(models=6, score_streams=8, samples=len(ci), checks=checks,
                frozen_dependencies=len(spec['hashes']), portfolio_evaluation=False, candidate_promotion=False))
    print(json.dumps(dict(inference_complete=True, models=6, portfolio_evaluation=False)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['freeze', 'run']); args = ap.parse_args()
    if args.action == 'freeze': freeze()
    else: run()

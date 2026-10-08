"""Remote-only, date-balanced chart CNN training with fixed monthly purging."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from quant import timefolio_heatmap_gpu_worker as engine
from quant.timefolio_chart_lab import cases, SEEDS
from quant.timefolio_chart_models import ChartNet


class SearchStopped(RuntimeError): pass


def train(ref, x, y, mask, di, cfg, root):
    torch.manual_seed(cfg['seed']); model = ChartNet(cfg).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                                 weight_decay=cfg['weight_decay'], foreach=False)
    groups = ref.date_groups(di, mask)
    assert groups and max(map(len, groups)) <= 512 and np.isfinite(y[mask]).all()
    target = torch.as_tensor(np.nan_to_num(y), dtype=torch.float32, device=x.device)
    history = []
    for epoch in range(cfg['epochs']):
        model.train(); losses = []
        for group in np.random.default_rng(cfg['seed'] + epoch).permutation(len(groups)):
            if (root / 'STOP.json').exists(): raise SearchStopped('User target reached')
            ids = groups[group]; optimizer.zero_grad(set_to_none=True)
            output = model(x[ids].float().div(127.5).sub(1))
            if cfg['objective'] == 'pairwise': loss = ref.pairwise_loss(output, target[ids])
            elif cfg['objective'] == 'mse': loss = nn.functional.mse_loss(output, target[ids])
            elif cfg['objective'] == 'bce':
                loss = nn.functional.binary_cross_entropy_with_logits(output, (target[ids] > 0).float())
            else: raise ValueError('Unknown objective')
            if not torch.isfinite(loss): raise AssertionError('Nonfinite loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch + 1, loss=float(np.mean(losses))))
    return model, dict(best_epoch=cfg['epochs'], history=history, training_rows=int(mask.sum()),
                      training_dates=len(groups), validation_rows=0)


def run(root, out, case, seed):
    root, out = Path(root), Path(out); engine.verify(root); engine.configure('cuda')
    plan = json.loads((root / 'plan.json').read_text())
    assert plan['cases'] == cases() and seed in SEEDS
    template = next(c for c in cases() if c['id'] == case)
    cfg = dict(template, id=f'{case}_seed{seed}', seed=seed)
    if out.exists(): raise RuntimeError('Existing job output requires review')
    out.mkdir(parents=True); model_root = out / 'models' / cfg['id']; model_root.mkdir(parents=True)
    ref = engine.reference(root)
    data = np.load(root / f"labels_h{cfg['horizon']}_w0_v20.npz", allow_pickle=False)
    y, di = data['y'], data['di']
    image = root / f"images_{cfg['view']}.npy"
    receipt = json.loads(image.with_suffix('.json').read_text()); assert engine.digest(image) == receipt['sha256']
    x = torch.from_numpy(np.array(np.load(image, mmap_mode='r')[:, None], copy=True)).to('cuda')
    started = time.time(); sha = engine.digest(root / 'manifest.json')
    engine.write(out / 'runtime.json', dict(config=cfg, gpu=torch.cuda.get_device_name(),
        torch=str(torch.__version__), cuda=torch.version.cuda, numpy=np.__version__, device='cuda',
        package_sha256=sha, images_sha256=receipt['sha256'], started_at=started))
    torch.manual_seed(seed); untrained = ChartNet(cfg).cuda()
    values = engine.predict(untrained, x, np.arange(len(y))); assert np.isfinite(values).all()
    torch.save({k: v.cpu() for k, v in untrained.state_dict().items()}, out / 'untrained.pt')
    np.save(out / 'untrained.pred.npy', values); del untrained
    for i, fold in enumerate(plan['folds']):
        month = fold['month']; mask = data[month + '_rf']; prediction = data[month + '_pr']; tick = time.monotonic()
        assert max(di[mask] + cfg['horizon']) < fold['start']
        model, fit = train(ref, x, y, mask, di, cfg, root)
        forecast = np.full(len(y), np.nan, np.float32)
        forecast[prediction] = engine.predict(model, x, np.flatnonzero(prediction))
        assert np.array_equal(np.isfinite(forecast), prediction)
        path = model_root / month
        torch.save(dict(config=cfg, state_dict={k: v.cpu() for k, v in model.state_dict().items()}), path.with_suffix('.pt'))
        del model; np.save(path.with_suffix('.pred.npy'), forecast)
        engine.write(path.with_suffix('.json'), dict(config=cfg, fold=fold, fit=None, refit_fit=fit,
            selection_rule='fixed', refit_epochs=cfg['epochs'], inner_validation_used=False,
            boundaries=plan['boundaries'][str(cfg['horizon'])][month],
            **plan['boundaries'][str(cfg['horizon'])][month], seconds=time.monotonic() - tick))
        engine.write(out / 'progress.json', dict(model=cfg['id'], completed=i + 1, expected=9, month=month))
        print(json.dumps(dict(model=cfg['id'], month=month, completed=i + 1, seconds=time.monotonic() - tick)), flush=True)
    hashes = {str(p.relative_to(out)): engine.digest(p) for p in model_root.iterdir()}
    hashes.update({n: engine.digest(out / n) for n in ['untrained.pt', 'untrained.pred.npy']})
    assert len(hashes) == 29
    engine.write(out / 'artifact_hashes.json', hashes)
    engine.write(out / 'complete.json', dict(folds=9, model=cfg['id'], package_sha256=sha,
        seconds=time.time() - started, maximum_gpu_memory_bytes=torch.cuda.max_memory_allocated(), evaluation_complete=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--case', required=True)
    p.add_argument('--seed', type=int, required=True); a = p.parse_args()
    try: run(a.root, a.out, a.case, a.seed)
    except SearchStopped:
        engine.write(a.out / 'stopped.json', dict(user_target_stop=True, complete=False))

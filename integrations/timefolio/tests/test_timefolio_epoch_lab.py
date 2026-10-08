from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from quant.timefolio_heatmap_epoch_lab import cases, fit_fold
from quant.timefolio_heatmap_gpu_worker import train
from quant.timefolio_heatmap_rank_training import date_groups, pairwise_loss


def inputs():
    rng = np.random.default_rng(921)
    x = torch.from_numpy(rng.integers(0, 256, (36, 1, 32, 65), dtype=np.uint8))
    y = rng.normal(size=36).astype(np.float32); di = np.repeat(np.arange(6), 6)
    masks = dict(tr=di < 2, va=(di >= 2) & (di < 4), rf=di < 5)
    cfg = dict(cases()[2], architecture='cnn', objective='pairwise', seed=17)
    return x, y, di, masks, cfg


def tiny_ref(metric):
    return SimpleNamespace(AlternativeNet=lambda _: nn.Sequential(nn.Flatten(), nn.Linear(2080, 1), nn.Flatten(0)),
        date_groups=date_groups, pairwise_loss=pairwise_loss, daily_ic=metric)


def test_fixed_training_ignores_inner_partition_and_never_calls_selection_metric():
    x, y, di, masks, cfg = inputs()
    def forbidden(*args): raise AssertionError('Fixed epochs must not select with validation')
    ref = tiny_ref(forbidden)
    a, report = fit_fold(ref, x, y, masks, cfg, di)
    changed = dict(masks, tr=np.zeros(36, bool), va=np.ones(36, bool))
    b, other = fit_fold(ref, x, y, changed, cfg, di)
    assert report == other and report['fit'] is None and report['refit_epochs'] == 3
    for k, v in a.state_dict().items(): torch.testing.assert_close(v, b.state_dict()[k], atol=0, rtol=0)


def test_selected_duration_restarts_refit_at_the_first_best_epoch():
    x, y, di, masks, cfg = inputs(); cfg.update(epoch_rule='inner_ic', epochs=3)
    values = iter([.1, .2, .2]); ref = tiny_ref(lambda *args: next(values))
    actual, report = fit_fold(ref, x, y, masks, cfg, di)
    expected, _ = train(ref, x, y, masks['rf'], np.zeros(36, bool), cfg, di, epochs=2)
    assert report['refit_epochs'] == 2 and report['inner_validation_used']
    assert report['fit']['training_rows'] == 12 and report['refit_fit']['training_rows'] == 30
    for k, v in actual.state_dict().items(): torch.testing.assert_close(v, expected.state_dict()[k], atol=0, rtol=0)


def test_registered_grid_is_matched_by_architecture():
    configs = cases(); assert len(configs) == 10
    for kind in ['cnn', 'mlp']:
        rows = [c for c in configs if c['kind'] == kind]
        assert {c['epochs'] for c in rows if c['epoch_rule'] == 'fixed'} == {1,3,6,12}
        assert len([c for c in rows if c['epoch_rule'] == 'inner_ic' and c['epochs'] == 6]) == 1

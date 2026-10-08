import numpy as np
import pytest
import torch

import quant.timefolio_heatmap_toprank_training as training


def sample():
    rng = np.random.default_rng(71)
    x = torch.from_numpy(rng.integers(0, 256, (80, 1, 32, 8), dtype=np.uint8))
    y = np.tile(np.linspace(-1, 1, 20), 4).astype(np.float32)
    di = np.repeat(np.arange(4), 20); ci = np.tile(np.arange(20)[::-1], 4)
    cfg = dict(architecture='mlp', objective='toprank', top_k=12, seed=17, lr=.0007, epochs=2)
    return x, y, di, ci, di < 2, di == 2, cfg


def test_excluded_future_data_cannot_change_fit_or_epoch_choice():
    x, y, di, ci, train, val, cfg = sample()
    first, fit = training.train_toprank(x, y, train, val, cfg, di, ci)
    changed = x.clone(); changed[di == 3] = 255 - changed[di == 3]
    label = y.copy(); label[di == 3] = np.nan
    second, other = training.train_toprank(changed, label, train, val, cfg, di, ci)
    assert fit == other
    for key, value in first.state_dict().items(): assert torch.equal(value, second.state_dict()[key])


def test_optimizer_receives_only_one_training_date_and_stable_security_keys(monkeypatch):
    x, y, di, ci, train, val, cfg = sample()
    y = (y + di * 10).astype(np.float32)  # Preserve the actual target dtype.
    actual = training.top_rank_loss; seen = []

    def checked(scores, target, *, top_k, security_keys):
        values = target.detach().numpy()
        matching = [d for d in [0, 1] if np.array_equal(values, y[di == d])]
        assert len(matching) == 1
        assert top_k == 12
        np.testing.assert_array_equal(security_keys.numpy(), ci[di == matching[0]])
        seen.append(matching[0])
        return actual(scores, target, top_k=top_k, security_keys=security_keys)

    monkeypatch.setattr(training, 'top_rank_loss', checked)
    _, fit = training.train_toprank(x, y, train, val, cfg, di, ci)
    assert sorted(seen) == [0, 0, 1, 1]
    assert fit['training_dates'] == 2 and fit['training_rows'] == 40 and fit['validation_rows'] == 20


def test_refit_uses_requested_epochs_and_ignores_unused_labels():
    x, y, di, ci, train, _, cfg = sample()
    y[~train] = np.nan
    _, fit = training.train_toprank(x, y, train, np.zeros(len(y), bool), cfg, di, ci, epochs=3)
    assert fit['best_epoch'] == 3 and len(fit['history']) == 3
    assert all(h['inner_ic'] == 0 for h in fit['history'])


def test_overlap_and_duplicate_security_keys_fail():
    x, y, di, ci, train, val, cfg = sample()
    with pytest.raises(ValueError, match='overlap'):
        training.train_toprank(x, y, train, train, cfg, di, ci)
    ci[:] = 1
    with pytest.raises(ValueError, match='unique'):
        training.train_toprank(x, y, train, val, cfg, di, ci)


def test_only_objective_and_top_k_differ_from_matched_configuration():
    from quant.timefolio_heatmap_absolute_training import CONFIGS as absolute
    from quant.timefolio_heatmap_corrected_training import CONFIGS as sector
    for cfg in training.CONFIGS:
        ref = next(c for c in (absolute if cfg['target'] == 'absolute' else sector)
                   if c['architecture'] == cfg['architecture'] and c['seed'] == cfg['seed'])
        actual = {k: v for k, v in cfg.items() if k not in ['id', 'objective', 'top_k']}
        expected = {k: v for k, v in ref.items() if k not in ['id', 'objective']}
        assert actual == expected
        _, reference_id = training.reference(cfg)
        assert reference_id == ref['id']

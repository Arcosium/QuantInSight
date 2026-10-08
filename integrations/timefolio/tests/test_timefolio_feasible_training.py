import copy

import numpy as np
import pytest
import torch

import quant.timefolio_heatmap_feasible_training as training


def sample():
    rng = np.random.default_rng(71); n = 24
    x = torch.from_numpy(rng.integers(0, 256, (4 * n, 1, 32, 8), dtype=np.uint8))
    y = np.tile(np.linspace(-1, 1, n), 4).astype(np.float32)
    di = np.repeat(np.arange(4), n); ci = np.tile(np.arange(n)[::-1], 4)
    cfg = dict(architecture='mlp', objective='feasible_pair', pool_n=12, random_draws=16,
               pool_seed=317, seed=17, lr=.0007, epochs=2)
    index = dict(dates=['20260105', '20260106', '20260107', '20260108', '20260109'],
                 codes=[f'{i:06d}' for i in range(n)])
    p = dict(eligible=np.ones((n, 5), bool), trade_allowed=np.ones((n, 5), bool),
             sector=np.repeat([10, 20, 30, 40], 6), market_cap=np.full((n, 5), 2e12),
             sector_cap=np.tile([.10, .14, .18, .20, .20], (n, 1)))
    caps = np.full((n, 5), .15)
    return x, y, di, ci, di < 2, di == 2, cfg, p, index, caps


def fit(values, *, epochs=None):
    x, y, di, ci, train, val, cfg, p, index, caps = values
    return training.train_feasible(x, y, train, val, cfg, di, ci, p, index, caps, epochs=epochs)


def test_unused_future_features_labels_and_metadata_cannot_change_fit():
    values = sample(); first, record = fit(values)
    changed = copy.deepcopy(values)
    x, y, di, ci, tr, va, cfg, p, index, caps = changed
    x[di == 3] = 255 - x[di == 3]; y[di == 3] = np.nan
    p['sector_cap'][:, 2:] = .01; p['market_cap'][:, 2:] = 1e11
    p['eligible'][:, 2:] = False; p['trade_allowed'][:, 3:] = False
    caps[:, 3:] = .01
    second, other = fit(changed)
    assert record == other
    for key, value in first.state_dict().items(): assert torch.equal(value, second.state_dict()[key])


def test_observed_signal_date_sector_limits_affect_fitted_parameters():
    values = sample(); first, _ = fit(values)
    changed = copy.deepcopy(values); changed[7]['sector_cap'][:, :2] = .20
    second, _ = fit(changed)
    assert any(not torch.equal(value, second.state_dict()[key]) for key, value in first.state_dict().items())


def test_optimizer_pool_sees_only_training_labels_and_matching_metadata(monkeypatch):
    values = sample(); x, y, di, ci, train, val, cfg, p, index, caps = values
    y[:] = (y + 10 * di).astype(np.float32)
    actual = training.feasible_pool; seen = []

    def checked(predicted, target, **kwargs):
        dates = [d for d in [0, 1] if np.array_equal(target, y[di == d])]
        assert len(dates) == 1
        d = dates[0]; c = ci[di == d]
        assert kwargs['seed'] == 317 + int(index['dates'][d])
        np.testing.assert_array_equal(kwargs['sector_caps'], np.minimum(p['sector_cap'][c, d], .2))
        np.testing.assert_array_equal(kwargs['market_cap'], p['market_cap'][c, d])
        np.testing.assert_array_equal(kwargs['stock_caps'], caps[c, d + 1])
        np.testing.assert_array_equal(kwargs['codes'], np.asarray(index['codes'])[c])
        seen.append(d)
        return actual(predicted, target, **kwargs)

    monkeypatch.setattr(training, 'feasible_pool', checked)
    _, record = fit(values)
    assert sorted(seen) == [0, 0, 1, 1]
    assert record['training_dates'] == 2 and record['training_rows'] == 48 and record['validation_rows'] == 24


def test_metadata_uses_signal_values_and_next_execution_stock_policy():
    x, y, di, ci, tr, va, cfg, p, ix, caps = sample()
    caps[:, 1] = .08; caps[:, 2] = .04
    ids = np.flatnonzero(di == 1)
    actual = training.metadata_for_date(p, ix, ci, di, ids, caps)
    np.testing.assert_array_equal(actual['stock_caps'], np.full(24, .04))
    np.testing.assert_array_equal(actual['sector_caps'], np.full(24, .14))
    assert actual['signal_date'] == 20260106
    with pytest.raises(ValueError, match='One training date'):
        training.metadata_for_date(p, ix, ci, di, np.array([0, 24]), caps)
    p['trade_allowed'][ci[ids[0]], 2] = False
    with pytest.raises(ValueError, match='inadmissible'):
        training.metadata_for_date(p, ix, ci, di, ids, caps)


def test_refit_uses_requested_epochs_without_validation_pool():
    values = list(sample()); values[5] = np.zeros(len(values[1]), bool)
    values[1][~values[4]] = np.nan
    _, record = fit(values, epochs=3)
    assert record['best_epoch'] == 3 and len(record['history']) == 3
    assert all(row['inner_ic'] == 0 for row in record['history'])


def test_only_objective_options_change_from_matched_uniform_configuration():
    from quant.timefolio_heatmap_absolute_training import CONFIGS as absolute
    from quant.timefolio_heatmap_corrected_training import CONFIGS as sector
    for cfg in training.CONFIGS:
        old = next(c for c in (absolute if cfg['target'] == 'absolute' else sector)
                   if c['architecture'] == cfg['architecture'] and c['seed'] == cfg['seed'])
        assert {k: v for k, v in cfg.items() if k not in ['id', 'objective', 'pool_n', 'random_draws', 'pool_seed']} == {
            k: v for k, v in old.items() if k not in ['id', 'objective']}
        assert training.reference(cfg)[1] == old['id']

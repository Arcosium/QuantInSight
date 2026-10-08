import json
import numpy as np
import pytest
import torch

from quant.timefolio_cnn_train import fit, groups, load_dataset, predict
from quant.timefolio_cnn_train import sha


def test_candidate_dataset_is_refused_before_loading_arrays(tmp_path):
    p = tmp_path/'manifest.json'
    p.write_text(json.dumps({'training_ready':False}))
    with pytest.raises(ValueError, match='readiness'):
        load_dataset(p)


def test_retired_dataset_cannot_enter_even_exploratory_training(tmp_path):
    p=tmp_path/'manifest.json';p.write_text('{}')
    (tmp_path/'RETIRED.json').write_text('{"reason":"provider basis mismatch"}')
    with pytest.raises(ValueError,match='retired'):
        load_dataset(p,purpose='exploratory')


def test_exploration_requires_explicit_purpose_and_disclosed_limitations(tmp_path):
    arrays = dict(images=np.zeros((1, 1, 8, 20), np.float32), returns=np.zeros(1),
        signal_index=np.zeros(1, np.int32), label_end_index=np.ones(1, np.int32),
        security_key=np.zeros(1, np.int32), eligible=np.ones(1, bool), dates=np.array(['20240101']))
    specs = {}
    for key, a in arrays.items():
        file = tmp_path/(key+'.npy'); np.save(file, a)
        specs[key] = dict(path=file.name, sha256=sha(file))
    metadata = dict(training_ready=False, exploratory_ready=True, contest_certified=False,
        limitations=['Historical corporate actions unresolved'],
        readiness=dict(feature_availability=True, label_construction=True), arrays=specs)
    p = tmp_path/'manifest.json'; p.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='readiness'):
        load_dataset(p)
    m, a = load_dataset(p, purpose='exploratory')
    assert m['contest_certified'] is False and a['images'].shape == (1, 1, 8, 20)
    metadata['limitations'] = []; p.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='limitations'):
        load_dataset(p, purpose='exploratory')


def test_duplicate_security_cannot_count_twice_in_ranking():
    with pytest.raises(ValueError, match='Duplicate'):
        groups(np.array([1, 1]), np.ones(2, bool), np.array([20, 20]))


def test_fit_uses_only_train_and_validation_not_outer_outcomes():
    torch.set_num_threads(1)
    rng = np.random.default_rng(17)
    x = rng.uniform(-1, 1, (24, 1, 8, 20)).astype(np.float32)
    y = rng.normal(0, .01, 24).astype(np.float32)
    signal = np.repeat(np.arange(6), 4); keys = np.tile(np.arange(4), 6)
    tr, va = signal < 3, signal == 3
    cfg = dict(seed=17, width=2, dropout=.1, lr=.01, weight_decay=.001,
        epochs=2, top_k=2, temperature=.2, objective='listnet', max_date_group=8)
    model, receipt = fit(x, y, signal, keys, tr, va, cfg)
    before = predict(model, x, np.flatnonzero(signal >= 4), device='cpu')
    y[signal >= 4] = 100
    again, other = fit(x, y, signal, keys, tr, va, cfg)
    after = predict(again, x, np.flatnonzero(signal >= 4), device='cpu')
    assert receipt == other and receipt['training_dates'] == 3 and receipt['validation_dates'] == 1
    np.testing.assert_array_equal(before, after)
    assert len(before) == 8 and np.isfinite(before).all()

import gc
import io

import joblib
import numpy as np
import pytest
import torch

from autofolio import neural_models as neural


@pytest.fixture
def tiny(monkeypatch):
    # Train the exact family implementations with small widths for CI only.
    monkeypatch.setitem(neural.SIZES, 'compact', dict(mlp=16, residual=16, lstm=16, gru=16, tcn=16, transformer=16))
    return np.random.default_rng(42)


@pytest.mark.parametrize('family', neural.FAMILIES)
def test_all_architectures_train_restore_and_reproduce(family, tiny):
    x = tiny.normal(size=(30, 12))
    y = 100 + x[:, -1] * .1
    records = []
    model = neural.NeuralRegressor(model=family, input_features=3, sequence_length=4,
                                   epoch_callback=records.append)
    model.fit(x[:24], y[:24], validation_data=(x[24:], y[24:]))
    predicted = model.predict(x[24:])
    proof = model.training_proof_
    assert np.isfinite(predicted).all() and np.all(np.abs(predicted - 100) < 1)
    assert proof['train_rows'] == 24 and proof['validation_rows'] == 6
    assert proof['device'] == 'cpu' and proof['threads'] == 1
    assert 3 <= proof['epochs_completed'] <= 10
    assert len(records) == proof['epochs_completed']
    assert all(record['rows_seen'] == 24 for record in records)
    assert proof['restored_best_validation']
    assert model.parameter_count_ == sum(p.numel() for p in model.network_.parameters())
    np.testing.assert_allclose(model.feature_mean_, x[:24].mean(axis=0))
    assert abs(float(model.target_mean_) - y[:24].mean()) < 1e-12
    replica = neural.NeuralRegressor(model=family, input_features=3, sequence_length=4)
    replica.fit(x[:24], y[:24], validation_data=(x[24:], y[24:]))
    np.testing.assert_array_equal(predicted, replica.predict(x[24:]))
    buffer = io.BytesIO(); joblib.dump(model, buffer); buffer.seek(0)
    recovered = joblib.load(buffer)
    assert recovered.epoch_callback is None
    np.testing.assert_array_equal(predicted, recovered.predict(x[24:]))


@pytest.mark.parametrize('family', neural.FAMILIES)
def test_actual_large_architectures_have_ten_million_parameters(family):
    # Meta tensors audit real shapes without allocating six large models.
    with torch.device('meta'):
        model = neural.Network(family, input_features=7, sequence_length=60, model_size='large')
    count = sum(p.numel() for p in model.parameters())
    assert 10_000_000 <= count <= 30_000_000, (family, count)
    del model; gc.collect()


def test_no_subsampling_and_train_only_scaling(tiny):
    x = np.ones((30101, 1))
    y = np.full(30101, 5.)
    model = neural.NeuralRegressor(input_features=1)
    model.fit(x, y, validation_data=(np.array([[1000.]]), np.array([1000.])))
    assert model.training_proof_['train_rows'] == 30101
    assert all(r['rows_seen'] == 30101 for r in model.training_proof_['history'])
    assert not model.training_proof_['row_subsampling']
    np.testing.assert_array_equal(model.feature_mean_, [1.])
    np.testing.assert_array_equal(model.feature_scale_, [1.])
    assert model.target_mean_ == 5 and model.target_scale_ == 1
    assert model.predict(np.empty((0, 1))).shape == (0,)


def test_invalid_inputs_and_unfitted_prediction(tiny):
    model = neural.NeuralRegressor(input_features=2)
    with pytest.raises(ValueError, match='not been fitted'):
        model.predict(np.ones((2, 2)))
    with pytest.raises(ValueError, match='finite'):
        model.fit(np.array([[1., np.nan], [2., 1.]]), np.ones(2))
    with pytest.raises(ValueError, match='time-major'):
        model.fit(np.ones((2, 3)), np.ones(2))
    with pytest.raises(ValueError, match='validation'):
        model.fit(np.ones((2, 2)), np.ones(2), validation_data=(np.ones((2, 2)), np.array([1., np.nan])))
    with pytest.raises(ValueError, match='configuration'):
        neural.NeuralRegressor(epochs=1)

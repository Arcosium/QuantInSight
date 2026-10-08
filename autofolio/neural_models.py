"""Deterministic CPU neural regressors for purged chronological experiments.

Input is time-major: [oldest fields..., ..., newest fields...]. The caller
owns the validation boundary and must purge labels crossing that boundary.
No row subsampling, external downloads, GPU allocation or disk checkpoints.
"""
import copy
import math
import time

import numpy as np
import torch
from torch import nn

FAMILIES = ('neural_mlp', 'residual_mlp', 'lstm', 'gru', 'tcn', 'transformer')
# Widths produce roughly 1–3M / 10–30M parameters for the supported inputs.
SIZES = {
    'compact': dict(mlp=768, residual=512, lstm=384, gru=448, tcn=384, transformer=256),
    'large': dict(mlp=2048, residual=1536, lstm=1024, gru=1152, tcn=1024, transformer=768),
}


class ResidualDense(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.block = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                   nn.Linear(width, width))

    def forward(self, x):
        return x + self.block(x)


class CausalConv(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.trim = 2 * dilation
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation, padding=self.trim)

    def forward(self, x):
        return self.conv(x)[:, :, :-self.trim]


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.layers = nn.Sequential(CausalConv(width, dilation), nn.GELU(),
                                    CausalConv(width, dilation), nn.GELU())

    def forward(self, x):
        return x + self.layers(x)


class Network(nn.Module):
    def __init__(self, model, input_features, sequence_length, model_size):
        super().__init__()
        self.model, self.features, self.steps = model, input_features, sequence_length
        widths = SIZES[model_size]
        if model == 'neural_mlp':
            width = widths['mlp']
            depth = 3 if model_size == 'compact' else 4
            layers = [nn.Linear(input_features * sequence_length, width), nn.GELU()]
            for _ in range(depth - 1):
                layers.extend([nn.Linear(width, width), nn.GELU()])
            self.body = nn.Sequential(*layers)
        elif model == 'residual_mlp':
            width = widths['residual']
            self.body = nn.Sequential(nn.Linear(input_features * sequence_length, width), nn.GELU(),
                                      *[ResidualDense(width) for _ in range(3)])
        elif model in ('lstm', 'gru'):
            width = widths[model]
            cell = nn.LSTM if model == 'lstm' else nn.GRU
            self.body = cell(input_features, width, num_layers=2, batch_first=True)
        elif model == 'tcn':
            width = widths['tcn']
            self.body = nn.Sequential(nn.Conv1d(input_features, width, 1),
                                      TemporalBlock(width, 1), TemporalBlock(width, 2))
        else:
            width = widths['transformer']
            self.projection = nn.Linear(input_features, width)
            position = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
            frequency = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.) / width))
            encoding = torch.zeros(sequence_length, width)
            encoding[:, 0::2] = torch.sin(position * frequency)
            encoding[:, 1::2] = torch.cos(position * frequency)
            self.register_buffer('position', encoding)
            layer = nn.TransformerEncoderLayer(width, nhead=8, dim_feedforward=width * 4,
                                               dropout=0., batch_first=True, activation='gelu')
            self.body = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            # PyTorch clones its prototype layer; initialize each independently.
            for block in self.body.layers:
                for value in block.parameters():
                    if value.ndim > 1:
                        nn.init.xavier_uniform_(value)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        if self.model in ('neural_mlp', 'residual_mlp'):
            hidden = self.body(x)
        else:
            x = x.reshape(-1, self.steps, self.features)
            if self.model in ('lstm', 'gru'):
                hidden = self.body(x)[0][:, -1]
            elif self.model == 'tcn':
                hidden = self.body(x.transpose(1, 2))[:, :, -1]
            else:
                hidden = self.body(self.projection(x) + self.position)[..., -1, :]
        return self.head(hidden).squeeze(-1)


def _moments(values, batch_size=4096):
    """Streaming stable moments, avoiding a normalized copy of the full input."""
    total, mean, second = 0, None, None
    for offset in range(0, len(values), batch_size):
        batch = np.asarray(values[offset:offset + batch_size], dtype=np.float64)
        if not np.isfinite(batch).all():
            raise ValueError('Neural inputs must be finite')
        size = len(batch)
        local = batch.mean(axis=0)
        variance = np.square(batch - local).sum(axis=0)
        if total == 0:
            mean, second = local, variance
        else:
            delta = local - mean
            second += variance + delta * delta * total * size / (total + size)
            mean += delta * size / (total + size)
        total += size
    std = np.sqrt(second / total)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError('Neural input scale overflow')
    return mean, np.where(std > 1e-12, std, 1.)


class NeuralRegressor:
    """A fitted object supports predict and joblib without retaining train data."""
    def __init__(self, model='neural_mlp', input_features=7, sequence_length=1,
                 model_size='compact', epochs=10, batch_size=128,
                 learning_rate=1e-3, seed=20261008, epoch_callback=None):
        if model not in FAMILIES or model_size not in SIZES:
            raise ValueError('Unsupported neural architecture')
        if (type(input_features) is not int or type(sequence_length) is not int
                or input_features < 1 or not 1 <= sequence_length <= 256):
            raise ValueError('Invalid neural input shape')
        if (type(epochs) is not int or epochs not in (10, 30) or type(batch_size) is not int
                or batch_size not in (128, 256) or learning_rate not in (3e-4, 1e-3)
                or type(seed) is not int or not 1 <= seed <= 2147483647):
            raise ValueError('Unsupported neural training configuration')
        self.model, self.input_features, self.sequence_length = model, input_features, sequence_length
        self.model_size, self.epochs, self.batch_size = model_size, epochs, batch_size
        self.learning_rate, self.seed, self.epoch_callback = learning_rate, seed, epoch_callback

    def _array(self, x):
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != self.input_features * self.sequence_length:
            raise ValueError('Neural input must be a time-major flattened matrix')
        return x

    def _tensor(self, x):
        values = np.asarray((x - self.feature_mean_) / self.feature_scale_, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError('Neural inputs must be finite')
        return torch.from_numpy(values)

    def _validation_loss(self, x, y):
        self.network_.eval()
        total = 0.
        with torch.inference_mode():
            for offset in range(0, len(x), self.batch_size):
                stop = offset + self.batch_size
                target = torch.from_numpy(np.asarray((y[offset:stop] - self.target_mean_) / self.target_scale_, dtype=np.float32))
                error = torch.square(self.network_(self._tensor(x[offset:stop])) - target).sum().item()
                if not math.isfinite(error):
                    raise ValueError('Non-finite neural validation loss')
                total += error
        return total / len(x)

    def fit(self, X, y, validation_data=None):
        started = time.monotonic()
        X, y = self._array(X), np.asarray(y)
        if y.ndim != 1 or len(y) != len(X) or len(y) < 2:
            raise ValueError('Insufficient or mismatched neural targets')
        self.feature_mean_, self.feature_scale_ = _moments(X)
        self.target_mean_, self.target_scale_ = _moments(y)
        validation = None
        if validation_data is not None:
            vx, vy = self._array(validation_data[0]), np.asarray(validation_data[1])
            if vy.ndim != 1 or len(vy) != len(vx) or not len(vy):
                raise ValueError('Invalid neural validation data')
            validation = vx, vy
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        generator = np.random.default_rng(self.seed)
        # Do not initialize CUDA or alter any academic GPU process.
        torch.random.default_generator.manual_seed(self.seed)
        with torch.device('cpu'):
            self.network_ = Network(self.model, self.input_features, self.sequence_length, self.model_size)
        self.parameter_count_ = sum(p.numel() for p in self.network_.parameters())
        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        best, best_epoch, weights, stale, history = math.inf, 0, None, 0, []
        for epoch in range(1, self.epochs + 1):
            self.network_.train()
            ordering = generator.permutation(len(X))
            total = 0.
            for offset in range(0, len(X), self.batch_size):
                indices = ordering[offset:offset + self.batch_size]
                batch = self._tensor(X[indices])
                target = torch.from_numpy(np.asarray((y[indices] - self.target_mean_) / self.target_scale_, dtype=np.float32))
                optimizer.zero_grad(set_to_none=True)
                loss = torch.square(self.network_(batch) - target).mean()
                if not torch.isfinite(loss).item():
                    raise ValueError('Non-finite neural training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network_.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                total += loss.item() * len(indices)
            training_loss = total / len(X)
            validation_loss = self._validation_loss(*validation) if validation is not None else None
            score = validation_loss if validation_loss is not None else training_loss
            if score < best - 1e-8:
                best, best_epoch, stale = score, epoch, 0
                if validation is not None:
                    weights = copy.deepcopy(self.network_.state_dict())
            else:
                stale += 1
            record = dict(epoch=epoch, train_mse=training_loss, validation_mse=validation_loss,
                          seconds=time.monotonic() - started, rows_seen=len(X))
            history.append(record)
            if self.epoch_callback is not None:
                self.epoch_callback(dict(record))
            if validation is not None and epoch >= 3 and stale >= 5:
                break
        if weights is not None:
            self.network_.load_state_dict(weights)
        self.network_.eval()
        self.training_proof_ = dict(architecture=self.model, model_size=self.model_size,
            parameter_count=self.parameter_count_, epochs_requested=self.epochs, epochs_completed=len(history),
            best_epoch=best_epoch if validation is not None else len(history),
            train_rows=len(X), validation_rows=len(validation[0]) if validation is not None else 0,
            seconds=time.monotonic() - started, device='cpu', threads=1, seed=self.seed,
            best_validation_mse=best if validation is not None else None,
            validation_mse_unit='standardized_target', restored_best_validation=weights is not None,
            row_subsampling=False, early_stopping=validation is not None and len(history) < self.epochs,
            history=history)
        return self

    def predict(self, X):
        if not hasattr(self, 'network_'):
            raise ValueError('Neural estimator has not been fitted')
        X = self._array(X)
        out = np.empty(len(X), dtype=np.float64)
        self.network_.eval()
        with torch.inference_mode():
            for offset in range(0, len(X), self.batch_size):
                prediction = self.network_(self._tensor(X[offset:offset + self.batch_size])).numpy()
                out[offset:offset + self.batch_size] = prediction * self.target_scale_ + self.target_mean_
        if not np.isfinite(out).all():
            raise ValueError('Non-finite neural predictions')
        return out

    def __getstate__(self):
        state = self.__dict__.copy()
        state['epoch_callback'] = None
        network = state.pop('network_', None)
        if network is not None:
            state['_weights'] = {key: value.detach().cpu().numpy().copy() for key, value in network.state_dict().items()}
        return state

    def __setstate__(self, state):
        weights = state.pop('_weights', None)
        self.__dict__.update(state)
        if weights is not None:
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            with torch.device('cpu'):
                self.network_ = Network(self.model, self.input_features, self.sequence_length, self.model_size)
            self.network_.load_state_dict({key: torch.from_numpy(value) for key, value in weights.items()})
            self.network_.eval()

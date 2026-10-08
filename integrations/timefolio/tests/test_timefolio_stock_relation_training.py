import copy
import json

import numpy as np
import pytest
import torch

from quant import timefolio_heatmap_stock_relation_training as study
from quant.timefolio_heatmap_stock_relation import StockRelationNet, date_contexts
from quant.timefolio_heatmap_rank_training import pairwise_loss


def fixture():
    rng = np.random.default_rng(912)
    x = torch.from_numpy(rng.integers(0, 256, (12, 1, 32, 65), dtype=np.uint8))
    days = np.repeat([10, 20, 30], 4)
    y = np.array([1., np.nan, -.5, np.nan] * 3)
    train = np.arange(12) < 4; train &= np.isfinite(y)
    val = (days == 20) & np.isfinite(y)
    cfg = dict(study.CONFIGS[15], epochs=3)
    assert cfg['architecture'] == 'mlp' and cfg['relation'] == 'attention'
    return x, y, days, train, val, cfg


def assert_states_equal(left, right):
    assert left.state_dict().keys() == right.state_dict().keys()
    for key, value in left.state_dict().items(): torch.testing.assert_close(value, right.state_dict()[key], atol=0, rtol=0)


def test_one_step_matches_full_context_manual_loss_not_filtered_input():
    x, y, days, train, val, cfg = fixture(); val[:] = False
    actual, report = study.train_relation(x, y, train, val, cfg, days, epochs=1)
    torch.manual_seed(cfg['seed']); expected = StockRelationNet(cfg['architecture'], cfg['relation'])
    opt = torch.optim.AdamW(expected.parameters(), lr=cfg['lr'], weight_decay=.001)
    expected.train(); scores = expected(x[:4].float().div(127.5).sub(1))
    target = torch.as_tensor(y[train], dtype=torch.float32)
    loss = pairwise_loss(scores[[0, 2]], target)
    loss.backward(); torch.nn.utils.clip_grad_norm_(expected.parameters(), 5., error_if_nonfinite=True); opt.step()
    assert_states_equal(actual, expected)
    assert report['training_rows'] == 2 and report['training_context_rows'] == 4
    assert report['history'][0]['loss'] == pytest.approx(float(loss.detach()))
    assert report['training_dates'] == 1


def test_unsupervised_labels_and_other_date_features_do_not_change_training():
    x, y, days, train, val, cfg = fixture(); val[:] = False
    left, report = study.train_relation(x, y, train, val, cfg, days, epochs=2)
    changed_y = y.copy(); changed_y[~train] = np.linspace(-1e5, 1e5, (~train).sum())
    changed_x = x.clone(); changed_x[4:] = 255 - changed_x[4:]
    right, other = study.train_relation(changed_x, changed_y, train, val, cfg, days, epochs=2)
    assert_states_equal(left, right); assert report == other


def test_same_date_unsupervised_features_remain_in_training_context():
    x, y, days, train, val, cfg = fixture(); val[:] = False
    left, _ = study.train_relation(x, y, train, val, cfg, days, epochs=1)
    x[1] = 255 - x[1]
    right, _ = study.train_relation(x, y, train, val, cfg, days, epochs=1)
    assert any(not torch.equal(value, right.state_dict()[key]) for key, value in left.state_dict().items())


def test_validation_preserves_peers_and_best_epoch_refit_restarts(monkeypatch):
    x, y, days, train, val, cfg = fixture()
    requests = []; original = study.predict_by_date
    def checked(model, images, dates, ids):
        groups = date_contexts(dates, np.isin(np.arange(len(dates)), ids))
        assert len(groups) == 1
        np.testing.assert_array_equal(groups[0][0], [4, 5, 6, 7])
        np.testing.assert_array_equal(ids, [4, 6])
        requests.append(ids.tolist())
        return original(model, images, dates, ids)
    monkeypatch.setattr(study, 'predict_by_date', checked)
    metrics = iter([.2, .9, .1]); monkeypatch.setattr(study, 'daily_ic', lambda *args: next(metrics))
    chosen, report = study.train_relation(x, y, train, val, cfg, days)
    assert report['best_epoch'] == 2 and len(requests) == 3
    assert report['validation_context_rows'] == 4 and report['validation_rows'] == 2
    refit, fit = study.train_relation(x, y, train, np.zeros(len(y), bool), cfg, days, epochs=2)
    assert_states_equal(chosen, refit); assert fit['best_epoch'] == 2


@pytest.mark.parametrize('issue', ['nonfinite', 'same_date', 'nonbool', 'no_train', 'float_image', 'zero_epoch', 'bool_epoch', 'objective'])
def test_invalid_training_inputs_fail_before_updates(issue):
    x, y, days, train, val, cfg = fixture(); epochs = 1
    if issue == 'nonfinite': y[0] = np.nan
    elif issue == 'same_date': val[:] = False; val[1] = True; y[1] = 0.
    elif issue == 'nonbool': train = train.astype(int)
    elif issue == 'no_train': train[:] = False
    elif issue == 'float_image': x = x.float()
    elif issue == 'zero_epoch': epochs = 0
    elif issue == 'bool_epoch': epochs = True
    elif issue == 'objective': cfg['objective'] = 'huber'
    with pytest.raises(ValueError): study.train_relation(x, y, train, val, cfg, days, epochs=epochs)


def test_parent_guard_preserves_candidate_priority(monkeypatch, tmp_path):
    monkeypatch.setattr(study, 'PRIOR', tmp_path)
    validation = dict(status='incomplete', portfolios=1824)
    summary = dict(portfolios=1824, joint_hypotheses=11168, robust_candidate_gate_passed=[])
    (tmp_path / 'validation_complete.json').write_text(json.dumps(validation))
    (tmp_path / 'evaluation_summary.json').write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match='validation'): study.parent_ready()
    validation['status'] = 'numerical_validation_complete'
    (tmp_path / 'validation_complete.json').write_text(json.dumps(validation))
    summary['robust_candidate_gate_passed'] = ['candidate']
    (tmp_path / 'evaluation_summary.json').write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match='Prioritize'): study.parent_ready()
    summary['robust_candidate_gate_passed'] = []
    (tmp_path / 'evaluation_summary.json').write_text(json.dumps(summary))
    study.parent_ready()

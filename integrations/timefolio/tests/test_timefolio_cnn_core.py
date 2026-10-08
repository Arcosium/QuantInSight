import numpy as np
import pytest
import torch

from quant.timefolio_cnn_core import CashGate, RankCNN, date_loss, heatmap, purged_masks


def test_heatmap_is_price_scale_invariant_and_rejects_missing_or_impossible_bars():
    a = np.array([[100, 103, 99, 102, 500], [103, 105, 101, 104, 600]], dtype=float)
    b = a.copy(); b[:, :4] *= 7
    np.testing.assert_allclose(heatmap(a), heatmap(b), atol=1e-7)
    for bad in [np.nan, 0, 106]:
        b = a.copy(); b[0, 2] = bad
        with pytest.raises(ValueError): heatmap(b)


@pytest.mark.parametrize('objective', ['listnet', 'topk_ce'])
def test_rank_losses_prefer_correct_order_and_are_permutation_invariant(objective):
    y = torch.tensor([-.03, -.01, .02, .04])
    good = torch.tensor([-2., -1., 1., 2.], requires_grad=True)
    loss = date_loss(good, y, objective, top_k=2)
    assert loss < date_loss(-good, y, objective, top_k=2)
    p = torch.tensor([3, 1, 0, 2])
    torch.testing.assert_close(loss, date_loss(good[p], y[p], objective, top_k=2))
    loss.backward()
    assert torch.isfinite(good.grad).all()


def test_topk_cutoff_ties_receive_equal_gradient_and_outsiders_are_penalized():
    score = torch.zeros(4, requires_grad=True)
    date_loss(score, torch.tensor([.1, .1, .1, -.1]), 'topk_ce', top_k=2).backward()
    assert torch.all(score.grad[:3] < 0) and score.grad[3] > 0
    torch.testing.assert_close(score.grad[:3], score.grad[:1].expand(3))


def test_purge_uses_actual_exit_and_does_not_hide_unlabelled_outer_predictions():
    s = np.arange(50); e = s + 6
    valid = np.ones(50, bool); valid[40:] = False
    masks = purged_masks(s, e, valid, validation_start=20, test_start=35, test_end=50)
    assert s[masks['train']].max() == 13
    assert s[masks['validation']].min() == 20 and s[masks['validation']].max() == 28
    assert s[masks['refit']].max() == 28
    np.testing.assert_array_equal(s[masks['prediction']], np.arange(35, 50))


def gate_data():
    s = np.arange(100, 200)
    return np.column_stack([np.sin(s), np.cos(s)]), np.sin(s) * .01, dict(
        signal_dates=s, feature_available_dates=s, label_available_dates=s+6,
        stage1_train_label_end=s-1, stage1_selection_label_end=s-1, fit_cutoff=180)


def test_gate_future_outcomes_and_scaling_cannot_change_past_fit():
    x, y, kw = gate_data(); first = CashGate.fit(x, y, **kw)
    future = kw['label_available_dates'] >= kw['fit_cutoff']
    x[future] += 1e6; y[future] += 1e6
    second = CashGate.fit(x, y, **kw)
    assert first.fitted_rows == 74
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.coefficient, second.coefficient)
    assert first.intercept == second.intercept


@pytest.mark.parametrize('field', ['feature_available_dates', 'stage1_train_label_end', 'stage1_selection_label_end'])
def test_gate_rejects_lookahead_or_in_sample_stage1_predictions(field):
    x, y, kw = gate_data(); kw[field] = kw['signal_dates'] + 1
    with pytest.raises(ValueError): CashGate.fit(x, y, **kw)


def test_gate_rejects_retrospective_prediction():
    x, y, kw = gate_data(); gate = CashGate.fit(x, y, **kw)
    with pytest.raises(ValueError):
        gate.predict(x[:1], signal_dates=np.array([179]), feature_available_dates=np.array([179]))
    assert np.isfinite(gate.predict(x[:1], signal_dates=np.array([180]), feature_available_dates=np.array([180]))).all()


def test_cnn_forward_backward_supports_both_registered_lookbacks():
    torch.set_num_threads(1); torch.manual_seed(17)
    model = RankCNN(width=4)
    for lookback in [20, 60]:
        out = model(torch.randn(3, 1, 8, lookback))
        assert out.shape == (3,)
        date_loss(out, torch.tensor([-.01, .01, .02]), 'listnet').backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())

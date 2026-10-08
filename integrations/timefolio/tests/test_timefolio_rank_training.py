import numpy as np
import pytest
import torch

from quant.timefolio_heatmap_rank_training import pairwise_loss, date_groups


def test_pairwise_gradient_improves_the_winner_without_requiring_absolute_level():
    score = torch.tensor([0., 0.], requires_grad=True); target = torch.tensor([1., 0.])
    loss = pairwise_loss(score, target); loss.backward()
    assert float(loss.detach()) == pytest.approx(np.log(2))
    np.testing.assert_allclose(score.grad.numpy(), [-.5, .5])
    improved = score.detach()-.1*score.grad
    assert pairwise_loss(improved, target) < loss
    assert pairwise_loss(improved+10, target) == pytest.approx(float(pairwise_loss(improved, target)), abs=1e-6)


def test_pairs_are_invariant_to_row_permutation_and_monotone_target_mapping():
    score = torch.tensor([.3, -.8, .5, .9]); target = torch.tensor([2., 2., -1., 3.])
    perm = torch.tensor([2, 0, 3, 1])
    a = pairwise_loss(score, target)
    torch.testing.assert_close(a, pairwise_loss(score[perm], target[perm]))
    torch.testing.assert_close(a, pairwise_loss(score, target*4+7))


def test_exact_ties_have_zero_loss_and_finite_zero_gradient():
    score = torch.tensor([-.8, .2, .9], requires_grad=True)
    loss = pairwise_loss(score, torch.ones(3)); loss.backward()
    assert loss == 0
    torch.testing.assert_close(score.grad, torch.zeros(3))


def test_date_groups_never_mix_dates_or_admit_unselected_rows():
    dates = np.array([3, 1, 3, 2, 1, 2]); mask = np.array([True, True, False, True, True, False])
    groups = date_groups(dates, mask)
    assert [g.tolist() for g in groups] == [[1, 4], [3], [0]]
    for g in groups: assert len(np.unique(dates[g])) == 1 and mask[g].all()
    dates = np.r_[dates, 999]; mask = np.r_[mask, False]
    assert [g.tolist() for g in date_groups(dates, mask)] == [[1, 4], [3], [0]]


@pytest.mark.parametrize('score,target', [([0., float('nan')], [1., 0.]), ([1., 0.], [float('inf'), 0.]), ([1.], [0., 1.])])
def test_invalid_ranking_arrays_are_rejected(score, target):
    with pytest.raises(ValueError): pairwise_loss(torch.tensor(score), torch.tensor(target))


def test_future_rows_cannot_change_fitted_parameters_or_epoch_selection():
    from quant.timefolio_heatmap_rank_training import train_daywise
    rng = np.random.default_rng(51)
    x = torch.from_numpy(rng.integers(0, 256, (40, 1, 32, 8), dtype=np.uint8))
    y = np.tile(np.linspace(-1, 1, 10), 4).astype(np.float32)
    di = np.repeat(np.arange(4), 10); train, val = di < 2, di == 2
    cfg = dict(architecture='mlp', objective='pairwise', seed=17, lr=.0007, epochs=1)
    first, first_fit = train_daywise(x, y, train, val, cfg, di)
    changed_x = x.clone(); changed_x[di == 3] = 255-changed_x[di == 3]
    changed_y = y.copy(); changed_y[di == 3] = 100
    second, second_fit = train_daywise(changed_x, changed_y, train, val, cfg, di)
    assert first_fit == second_fit
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key])

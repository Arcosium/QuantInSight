import numpy as np
import pytest
import torch
from scipy.stats import rankdata

from quant.timefolio_heatmap_toprank import ndcg_pair_weights, top_rank_loss


def test_swap_weights_match_direct_ndcg_recomputation_with_a_cutoff():
    target = np.array([.2, -.1, .4, .1, .1, .5])
    score = np.array([.9, .7, -.3, .5, .1, -.2])
    gains = 2 ** (4 * (rankdata(target) - 1) / (len(target) - 1)) - 1
    discount = np.r_[1/np.log2(np.arange(2, 5)), np.zeros(3)]
    ideal = np.sort(gains)[::-1] @ discount
    order = np.argsort(-score); base = gains[order] @ discount / ideal
    expected = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if target[i] <= target[j]: continue
            swapped = order.copy(); a, b = np.where(order == i)[0][0], np.where(order == j)[0][0]
            swapped[a], swapped[b] = swapped[b], swapped[a]
            expected[i, j] = abs(gains[swapped] @ discount / ideal - base)
    actual = ndcg_pair_weights(torch.tensor(score), torch.tensor(target), top_k=3)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-15)
    assert actual[5, 2] == 0  # Both model positions are below the cutoff.
    assert actual[5, 1] > 0   # A swap crosses the cutoff.


def test_gradient_matches_detached_lambdas_and_pushes_winners_up():
    score = torch.tensor([.4, -.7, .1, .9], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([.3, .6, -.5, .1], dtype=torch.float64)
    weights = ndcg_pair_weights(score, target, top_k=2)
    assert not weights.requires_grad
    margin = score.detach()[:, None] - score.detach()[None, :]
    lambdas = -torch.sigmoid(-margin) * weights / weights.sum()
    expected = lambdas.sum(1) - lambdas.sum(0)
    loss = top_rank_loss(score, target, top_k=2); loss.backward()
    torch.testing.assert_close(score.grad, expected)
    assert abs(float(score.grad.sum())) < 1e-14
    assert score.grad[1] < 0 and score.grad[2] > 0
    assert top_rank_loss(score.detach() - 1e-4*score.grad, target, top_k=2) < loss


def test_tied_predictions_use_security_keys_and_survive_row_permutation():
    score = torch.tensor([0., 0., 0., 0.], requires_grad=True)
    target = torch.tensor([3., 0., 2., 1.]); keys = torch.tensor([30, 10, 40, 20])
    permutation = torch.tensor([2, 0, 3, 1]); inverse = torch.argsort(permutation)
    before = ndcg_pair_weights(score, target, top_k=2, security_keys=keys)
    after = ndcg_pair_weights(score[permutation], target[permutation], top_k=2, security_keys=keys[permutation])
    torch.testing.assert_close(before, after[inverse][:, inverse])
    loss = top_rank_loss(score, target, top_k=2, security_keys=keys); loss.backward()
    assert torch.isfinite(score.grad).all() and torch.any(score.grad != 0)


@pytest.mark.parametrize('n', [1, 4])
def test_equal_targets_have_zero_loss_and_zero_finite_gradient(n):
    scores = torch.arange(n, dtype=torch.float32, requires_grad=True)
    loss = top_rank_loss(scores, torch.ones(n), top_k=99); loss.backward()
    assert loss == 0; torch.testing.assert_close(scores.grad, torch.zeros(n))


def test_common_shifts_and_monotone_target_mapping_preserve_weights():
    scores = torch.tensor([.2, -.1, .8, .5]); target = torch.tensor([2., 1., 4., 3.])
    before = ndcg_pair_weights(scores, target, top_k=2)
    after = ndcg_pair_weights(scores + 3, target ** 3, top_k=2)
    torch.testing.assert_close(before, after)


@pytest.mark.parametrize('top_k', [0, -1, True, 1.5])
def test_invalid_cutoff_is_rejected(top_k):
    with pytest.raises(ValueError): top_rank_loss(torch.zeros(2), torch.ones(2), top_k=top_k)


def test_missing_values_duplicate_security_keys_and_oversized_dates_are_rejected():
    with pytest.raises(ValueError): top_rank_loss(torch.tensor([0., float('nan')]), torch.ones(2))
    with pytest.raises(ValueError): top_rank_loss(torch.zeros(2), torch.tensor([float('inf'), 0.]))
    with pytest.raises(ValueError): top_rank_loss(torch.zeros(2), torch.ones(2), security_keys=torch.tensor([1, 1]))
    with pytest.raises(ValueError): top_rank_loss(torch.zeros(513), torch.ones(513))

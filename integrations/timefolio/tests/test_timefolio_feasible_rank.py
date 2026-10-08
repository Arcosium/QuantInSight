import numpy as np
import pytest
import torch

from quant.timefolio_heatmap_feasible_rank import feasible_pool, feasible_pair_loss
from quant.timefolio_heatmap_replay import targets


def fixture():
    n = 24
    return dict(predicted=np.arange(n, 0, -1.), target=np.sin(np.arange(n)),
                eligible=np.ones(n, bool), sector=np.repeat([10, 20, 30, 40], 6),
                sector_caps=np.full(n, .10), market_cap=np.full(n, 5e11),
                codes=np.array([f'{i:06d}' for i in range(n)]), stock_caps=np.full(n, .15))


def test_pool_respects_sector_smallcap_stock_gross_and_admission():
    args = fixture(); args['eligible'][0] = False; args['stock_caps'][7] = .025
    pool = feasible_pool(**args, seed=51)
    assert 1 <= len(pool) <= 18
    assert (pool >= 0).all() and (pool[:, 0] == 0).all()
    assert (pool <= args['stock_caps'] + 1e-12).all()
    assert (pool.sum(1) <= .285 + 1e-12).all()
    assert ((pool > 0).sum(1) <= 12).all()
    for sector in [10, 20, 30, 40]:
        assert (pool[:, args['sector'] == sector].sum(1) <= .095 + 1e-12).all()


def test_prediction_and_label_targets_match_existing_builder():
    args = fixture(); pool = feasible_pool(**args, random_draws=0)
    for i, name in enumerate(['predicted', 'target']):
        expected = targets(args[name], args['eligible'], args['sector'], args['sector_caps'],
                           args['market_cap'], args['codes'], top_n=12, weight=.05, gross=.8)
        np.testing.assert_array_equal(pool[i], expected)
    # A selected security outside the raw top12 enters the same feasible pool.
    assert (pool[0, 12:] > 0).any()


def test_random_pool_is_ticker_stable_under_row_permutation():
    args = fixture(); args['predicted'][:] = 0
    original = feasible_pool(**args, seed=81)
    perm = np.random.default_rng(9).permutation(24)
    reordered = feasible_pool(**{k: v[perm] for k, v in args.items()}, seed=81)
    np.testing.assert_allclose(original[:, perm], reordered, atol=1e-14, rtol=0)


def test_pair_gradient_matches_manual_feasible_weight_differences():
    score = torch.tensor([.1, -.2, .3], dtype=torch.float64, requires_grad=True)
    label = torch.tensor([2., -1., .5], dtype=torch.float64, requires_grad=True)
    pool = torch.tensor([[.1, .0, .2], [.0, .2, .1], [.2, .1, .0]], dtype=torch.float64, requires_grad=True)
    loss = feasible_pair_loss(score, label, pool); loss.backward()
    utility = pool.detach().numpy() @ label.detach().numpy() / .3
    predicted = pool.detach().numpy() @ score.detach().numpy() / .3
    expected = np.zeros(3); losses = []
    for i in range(3):
        for j in range(3):
            if utility[i] <= utility[j]: continue
            difference = predicted[i] - predicted[j]
            losses.append(np.logaddexp(0, -difference))
            expected -= (pool[i].detach().numpy() - pool[j].detach().numpy()) / (.3 * (1 + np.exp(difference)))
    expected /= len(losses)
    np.testing.assert_allclose(score.grad.numpy(), expected, atol=1e-14)
    assert float(loss.detach()) == pytest.approx(np.mean(losses))
    assert label.grad is None and pool.grad is None


def test_a_good_stock_only_matters_where_feasible_portfolios_differ():
    score = torch.zeros(3, requires_grad=True); label = torch.tensor([4., 2., -1.])
    pool = torch.tensor([[.1, .1, 0.], [.1, 0., .1]])
    feasible_pair_loss(score, label, pool).backward()
    assert score.grad[0] == 0 and score.grad[1] < 0 and score.grad[2] > 0


def test_equal_utility_and_duplicate_solutions_do_not_invent_signal():
    args = fixture(); args['eligible'][:] = False
    pool = feasible_pool(**args)
    assert pool.shape == (1, 24) and not pool.any()
    score = torch.ones(24, requires_grad=True); label = torch.arange(24, dtype=torch.float32)
    loss = feasible_pair_loss(score, label, torch.from_numpy(pool)); loss.backward()
    assert loss == 0 and not score.grad.any()
    score = torch.tensor([1., 2.], requires_grad=True)
    loss = feasible_pair_loss(score, torch.ones(2), torch.tensor([[.1, 0.], [0., .1]]))
    loss.backward(); assert loss == 0 and not score.grad.any()


@pytest.mark.parametrize('field,value', [('sector_caps', np.linspace(.1, .2, 24)), ('stock_caps', np.zeros(24)), ('predicted', np.full(24, np.nan)), ('codes', np.full(24, 'same'))])
def test_invalid_pool_metadata_rejected(field, value):
    args = fixture(); args[field] = value
    with pytest.raises(ValueError): feasible_pool(**args)


def test_different_gross_levels_cannot_be_ranked_by_common_output_bias():
    score = torch.tensor([.3, -.4, .1], dtype=torch.float64)
    label = torch.tensor([2., -1., .5], dtype=torch.float64)
    pool = torch.tensor([[.1, .0, .1], [.0, .2, .1], [.1, .1, .0]], dtype=torch.float64)
    original = feasible_pair_loss(score, label, pool)
    torch.testing.assert_close(original, feasible_pair_loss(score + 7, label + 4, pool))
    # Rescaling one row changes its raw gross but not its invested-part utility.
    rescaled = pool.clone(); rescaled[0] *= 2
    torch.testing.assert_close(original, feasible_pair_loss(score, label, rescaled))


def test_constant_labels_with_unequal_fractional_weights_have_no_gradient():
    score = torch.tensor([.1, -.2, .3], dtype=torch.float64, requires_grad=True)
    pool = torch.tensor([[.04, .05, .0099], [.1, .0, .027], [.01, .025, .03]], dtype=torch.float64)
    loss = feasible_pair_loss(score, torch.ones(3, dtype=torch.float64), pool)
    loss.backward()
    assert loss == 0 and not score.grad.any()

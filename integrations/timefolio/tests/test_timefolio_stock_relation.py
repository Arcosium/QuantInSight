import numpy as np
import pytest
import torch

from quant.timefolio_heatmap_stock_relation import StockRelationNet, date_contexts, predict_by_date


def network(architecture='cnn', relation='attention'):
    torch.manual_seed(17)
    return StockRelationNet(architecture, relation).eval()


@pytest.mark.parametrize('architecture', ['cnn', 'mlp'])
def test_relation_ablations_have_identical_initial_parameters(architecture):
    models = [network(architecture, relation) for relation in ['self', 'mean', 'attention']]
    reference = models[0].state_dict()
    for model in models[1:]:
        assert reference.keys() == model.state_dict().keys()
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in reference.items())


@pytest.mark.parametrize('architecture', ['cnn', 'mlp'])
@pytest.mark.parametrize('relation', ['self', 'mean', 'attention'])
def test_stock_permutation_is_equivariant(architecture, relation):
    model = network(architecture, relation)
    x = torch.rand(6, 1, 32, 65) * 2 - 1; permutation = torch.tensor([3, 0, 5, 1, 4, 2])
    with torch.no_grad():
        assert torch.allclose(model(x)[permutation], model(x[permutation]), atol=2e-6, rtol=1e-6)


@pytest.mark.parametrize('architecture', ['cnn', 'mlp'])
def test_only_peer_modes_react_to_other_stock_features(architecture):
    torch.manual_seed(991); x = torch.rand(4, 1, 32, 65) * 2 - 1
    changed = x.clone(); changed[1:] = .9
    for relation in ['self', 'mean', 'attention']:
        model = network(architecture, relation)
        with torch.no_grad():
            first, other = model(x)[0], model(changed)[0]
        if relation == 'self': assert torch.equal(first, other)
        else: assert abs(float(first - other)) > 1e-5


@pytest.mark.parametrize('architecture', ['cnn', 'mlp'])
def test_single_stock_context_has_same_behavior_for_all_modes(architecture):
    x = torch.rand(1, 1, 32, 65) * 2 - 1
    with torch.no_grad(): values = [network(architecture, mode)(x) for mode in ['self', 'mean', 'attention']]
    assert torch.equal(values[0], values[1]) and torch.equal(values[0], values[2])


def test_label_or_requested_mask_never_filters_context_constituents():
    days = np.array([10, 11, 10, 11, 10, 11]); requested = np.array([True, False, False, False, True, False])
    groups = date_contexts(days, requested)
    assert len(groups) == 1 and np.array_equal(groups[0][0], [0, 2, 4]) and np.array_equal(groups[0][1], [0, 2])
    requested[4] = False
    assert np.array_equal(date_contexts(days, requested)[0][0], [0, 2, 4])


@pytest.mark.parametrize('relation', ['self', 'mean', 'attention'])
def test_prediction_never_mixes_dates_and_preserves_context_for_partial_requests(relation):
    model = network('cnn', relation); x = torch.randint(0, 256, (7, 1, 32, 65), dtype=torch.uint8)
    days = np.array([10, 11, 10, 11, 10, 11, 11]); requested = np.array([4, 0])
    before = predict_by_date(model, x, days, requested)
    changed = x.clone(); changed[np.flatnonzero(days == 11)] = 0
    assert np.array_equal(before, predict_by_date(model, changed, days, requested))
    complete = predict_by_date(model, x, days, np.arange(len(x)))
    assert np.array_equal(before, complete[requested])
    with torch.no_grad(): direct = model(x[[0, 2, 4]].float().div(127.5).sub(1)).numpy()
    assert np.array_equal(before, direct[[2, 0]])


@pytest.mark.parametrize('architecture', ['cnn', 'mlp'])
def test_attention_has_finite_gradients_through_every_parameter(architecture):
    model = network(architecture).train(); x = torch.rand(5, 1, 32, 65) * 2 - 1
    score = model(x); loss = torch.nn.functional.softplus(score[1:] - score[0]).mean(); loss.backward()
    assert torch.isfinite(loss)
    for parameter in model.parameters(): assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_invalid_batches_and_indices_are_rejected():
    model = network(); x = torch.zeros(2, 1, 32, 65, dtype=torch.uint8); days = np.array([0, 0])
    for ids in [np.array([0, 0]), np.array([-1]), np.array([2]), np.array([.5])]:
        with pytest.raises(ValueError): predict_by_date(model, x, days, ids)
    with pytest.raises(ValueError): predict_by_date(model, x.float(), days, np.array([0]))
    with pytest.raises(ValueError): date_contexts(np.zeros(513, int), np.ones(513, bool))
    with pytest.raises(ValueError): date_contexts(np.array([1., 2.]), np.array([True, False]))
    with pytest.raises(ValueError): model(torch.zeros(2, 1, 32, 64))
    with pytest.raises(ValueError): model(x)
    with pytest.raises(ValueError): model(torch.full((2, 1, 32, 65), float('nan')))

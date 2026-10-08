import pytest

from quant.timefolio_heatmap_fleet_breadth import policies, hypotheses, identity, reference


def test_breadth_keeps_target_sum_and_order_constraints():
    for policy in policies():
        assert policy['top_n'] * policy['weight'] == pytest.approx(12 * .05)
        assert 0 < policy['weight'] < .05
        assert policy['max_orders'] in [3, 10]
        assert policy['buffer'] / policy['top_n'] == policy['reference_buffer'] / 12


def test_reference_keeps_seed_objective_and_original_policy():
    policy = next(p for p in policies() if p['top_n'] == 30 and p['buffer'] == 60)
    ref = reference('flt_example_trained_seed29', policy)
    assert ref.startswith('flt_example_trained_seed29__n12__')
    assert '__buffer24__' in ref
    assert '__weight200bp__' in identity('example_trained_seed29', policy)


def test_all_controls_and_prior_comparisons_are_retained():
    family = hypotheses([f'case{i}' for i in range(9)])
    assert len(family) == 8688
    assert len({key for key, _, _ in family}) == 3504
    online = {key for key, _, _ in family if key.startswith('width_online_nonimage__')}
    assert len(online) == 48  # Shared control accounts must not be multiplied by source/seed.
    assert all(ref in online for _, label, ref in family if label == 'same_breadth_nonimage')
    assert len({ref for _, label, ref in family if label == 'original_top12'}) == 576

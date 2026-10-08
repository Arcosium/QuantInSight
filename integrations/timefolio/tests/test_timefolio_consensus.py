import numpy as np
import pytest

from quant.timefolio_heatmap_consensus import rank_consensus


def fixture():
    a = np.array([[1., 3.], [2., 2.], [3., 1.]])
    return {'a': a, 'b': a*100+5, 'c': -a}, {'a': 'cnn', 'b': 'cnn', 'c': 'tcn'}, np.ones(a.shape, bool)


def test_consensus_is_invariant_to_member_scale_and_input_order():
    matrices, groups, eligible = fixture()
    before = rank_consensus(matrices, groups, eligible, 'mean')
    changed = {'c': matrices['c']*.001, 'b': matrices['b']*10-20, 'a': matrices['a']+1000}
    np.testing.assert_array_equal(before, rank_consensus(changed, groups, eligible, 'mean'))
    np.testing.assert_allclose(before[:, 0], [5/9, 2/3, 7/9])


def test_balanced_weights_architectures_not_number_of_variants():
    matrices, groups, eligible = fixture()
    balanced = rank_consensus(matrices, groups, eligible, 'balanced')
    np.testing.assert_allclose(balanced, 2/3)
    assert not np.allclose(balanced, rank_consensus(matrices, groups, eligible, 'mean'))


def test_median_uses_middle_rank_and_shared_missingness_is_preserved():
    matrices, groups, eligible = fixture()
    eligible[:, 1] = False
    value = rank_consensus(matrices, groups, eligible, 'median')
    np.testing.assert_allclose(value[:, 0], [1/3, 2/3, 1.])
    assert np.isnan(value[:, 1]).all()


def test_future_columns_cannot_change_current_consensus():
    matrices, groups, eligible = fixture()
    before = rank_consensus(matrices, groups, eligible, 'balanced')
    for a in matrices.values(): a[:, 1] = [9999, -999, 0]
    after = rank_consensus(matrices, groups, eligible, 'balanced')
    np.testing.assert_array_equal(before[:, 0], after[:, 0])


def test_inconsistent_member_coverage_is_rejected():
    matrices, groups, eligible = fixture(); matrices['a'][0, 0] = np.nan
    with pytest.raises(ValueError, match='coverage'):
        rank_consensus(matrices, groups, eligible, 'mean')


def test_no_input_arrays_are_modified():
    matrices, groups, eligible = fixture(); before = {n: a.copy() for n, a in matrices.items()}
    rank_consensus(matrices, groups, eligible, 'median')
    for name in matrices: np.testing.assert_array_equal(matrices[name], before[name])

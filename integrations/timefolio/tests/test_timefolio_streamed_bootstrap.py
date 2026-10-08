import numpy as np
import pytest
from quant.timefolio_heatmap_streamed_bootstrap import family_bootstrap
from quant.timefolio_heatmap_walkforward_eval import family_bootstrap as original


def test_joint_correction_keeps_cross_chunk_competitors_and_duplicate_ties():
    a = np.random.default_rng(219).normal(0, .01, (179, 23))
    a[:, 0] = 0; a[:, 1] = .001; a[:, 2] = a[:, 19]
    for layout in [a, np.asfortranarray(a)]:
        for block in [5, 10]:
            expected = original(layout, block=block, draws=237, seed=57)
            for size in [1, 3, 8, 1024]:
                actual = family_bootstrap(layout, block=block, draws=237, seed=57,
                    column_batch=size, scratch_bytes=179*8*min(size,23)*3)
                assert set(actual) == set(expected)
                for key in expected:
                    np.testing.assert_array_equal(actual[key], expected[key])


def test_column_budget_changes_storage_without_dropping_hypotheses():
    a = np.random.default_rng(314).normal(0, .01, (71, 87))
    expected = original(a, block=5, draws=201, seed=57)
    actual = family_bootstrap(a, block=5, draws=201, seed=57,
                             bootstrap_bytes=201*8*2, scratch_bytes=71*8*2)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_invalid_budgets_fail_before_bootstrap_allocation():
    a = np.ones((10, 5))
    for kwargs in [dict(draws=1), dict(block=0), dict(column_batch=0),
                   dict(scratch_bytes=1), dict(bootstrap_bytes=1)]:
        with pytest.raises(ValueError): family_bootstrap(a, **kwargs)


def test_single_original_column_and_strided_input_preserve_reductions():
    base = np.random.default_rng(8).normal(0, .01, (180, 8))
    for a in [base[:, :1].copy(), base[::2, ::-1]]:
        expected = original(a, block=10, draws=203, seed=57)
        actual = family_bootstrap(a, block=10, draws=203, seed=57, column_batch=1)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])

import numpy as np
import pytest

from quant.timefolio_cnn_diagnostics import date_metrics


def test_missing_winner_is_not_replaced_by_a_known_outcome():
    row = date_metrics(np.array([4., 3., 2., 1.]), np.array([np.nan, .2, .3, .4]), np.arange(4), top_k=2)
    assert row['selected_keys'] == [0, 1] and row['missing_selected_outcomes'] == 1
    assert row['topk_raw_h5'] is None and row['topk_excess_h5'] is None


def test_tied_predictions_have_no_fake_rank_ic_and_stable_tie_break():
    row = date_metrics(np.ones(4), np.arange(4)/10, np.array([3, 2, 1, 0]), top_k=2)
    assert row['selected_keys'] == [0, 1] and row['rank_ic'] is None
    assert row['topk_raw_h5'] == pytest.approx(.25)
    row = date_metrics(np.arange(4), np.arange(4)/10, np.arange(4), top_k=2)
    assert row['rank_ic'] == pytest.approx(1)

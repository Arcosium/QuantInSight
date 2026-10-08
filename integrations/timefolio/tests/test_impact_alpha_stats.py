"""Synthetic accounting and uncertainty checks, with no market connectivity."""

import json

import numpy as np
import pandas as pd
import pytest

from quant.impact_alpha_stats import add_net_columns, choose_horizon, summarize_trades


def trades(rows):
    return pd.DataFrame(rows, columns=["day", "entry_ms", "exit_ms", "entry_price", "exit_price", "quantity"])


def test_exact_round_trip_costs_and_copy_semantics():
    frame = trades([("2026-01-01", 0, 1000, 100, 110, 2)])
    out = add_net_columns(frame, fee_bps=10, extra_slippage_bps=2)
    assert out.iloc[0]["gross_pnl"] == pytest.approx(20)
    assert out.iloc[0]["fee_cost"] == pytest.approx(.42)
    assert out.iloc[0]["extra_slippage_cost"] == pytest.approx(.084)
    assert out.iloc[0]["net_pnl"] == pytest.approx(19.496)
    assert out.iloc[0]["gross_bps"] == pytest.approx(1000)
    assert out.iloc[0]["net_bps"] == pytest.approx(974.8)
    assert "net_pnl" not in frame


def test_break_even_fee_uses_aggregate_actual_turnover_not_average_bps():
    frame = trades([("2026-01-01", 0, 1, 100, 110, 1),
                    ("2026-01-02", 2, 3, 1000, 1001, 5)])
    result = summarize_trades(frame, ["2026-01-01", "2026-01-02"], extra_slippage_bps=2)
    expected = 15 / (210 + 10005) * 1e4 - 2
    assert result["break_even_fee_per_side_bps"] == pytest.approx(expected)
    assert expected != pytest.approx(result["mean_gross_bps"] / 2)
    assert add_net_columns(frame, fee_bps=expected, extra_slippage_bps=2)["net_pnl"].sum() == pytest.approx(0, abs=1e-12)


def test_cluster_bootstrap_pools_trade_counts_and_includes_zero_trade_days():
    days = [f"2026-01-{i:02}" for i in range(1, 7)]
    frame = trades([(days[0], 0, 1, 100, 100.1, 1),
                    (days[0], 2, 3, 100, 100.2, 1),
                    (days[0], 4, 5, 100, 100.3, 1),
                    (days[1], 6, 7, 100, 101, 1),
                    (days[2], 8, 9, 100, 99, 1),
                    (days[3], 10, 11, 100, 100.8, 1)])
    repetitions, seed = 257, 71
    result = summarize_trades(frame, days, fee_bps=0, repetitions=repetitions, seed=seed)
    draws = np.random.default_rng(seed).integers(0, len(days), size=(repetitions, len(days)))
    counts = np.array([3, 1, 1, 1, 0, 0])[draws].sum(axis=1)
    sums = np.array([60, 100, -100, 80, 0, 0])[draws].sum(axis=1)
    valid = counts > 0
    expected = np.quantile(sums[valid] / counts[valid], [.025, .975])
    assert result["mean_net_bps"] == pytest.approx(140 / 6)
    assert result["mean_net_bps_ci95"] == pytest.approx(expected)
    assert result["mean_gross_bps_ci95"] == pytest.approx(expected)
    assert result["calendar_days"] == 6 and result["trading_days"] == 4
    assert result["bootstrap_valid_repetitions"] == int(valid.sum())


def test_zero_trades_and_single_day_intervals_are_undefined_and_json_safe():
    empty = summarize_trades(pd.DataFrame(), ["2026-01-01", "2026-01-02"])
    assert empty["n"] == 0 and empty["net_pnl"] == 0
    for name in ("mean_gross_bps", "mean_net_bps", "profit_factor", "win_rate",
                 "mean_net_bps_ci95", "mean_gross_bps_ci95", "break_even_fee_per_side_bps"):
        assert empty[name] is None
    assert empty["closed_trade_max_drawdown_pct"] == 0
    frame = trades([("2026-01-01", 0, 1, 100, 101, 1)])
    single = summarize_trades(frame, ["2026-01-01"])
    assert single["mean_net_bps_ci95"] is None and single["mean_gross_bps_ci95"] is None
    json.dumps({"empty": empty, "single": single}, allow_nan=False)


def test_realized_drawdown_includes_initial_loss_and_exit_time_order():
    # Input ordering is deliberately reversed. Fixed capital is never resized.
    frame = trades([("2026-01-01", 5, 6, 100, 80, 100),
                    ("2026-01-01", 1, 2, 100, 90, 100),
                    ("2026-01-01", 3, 4, 100, 120, 100)])
    result = summarize_trades(frame, ["2026-01-01"], fee_bps=0)
    # Closed equity 10,000 -> 9,000 -> 11,000 -> 9,000.
    assert result["closed_trade_max_drawdown_pct"] == pytest.approx(2000 / 11000 * 100)
    assert result["net_pnl"] == -1000
    assert result["cumulative_return_pct"] == -10
    assert result["profit_factor"] == pytest.approx(2000 / 3000)
    initial_loss = summarize_trades(frame.iloc[[1]], ["2026-01-01"], fee_bps=0)
    assert initial_loss["closed_trade_max_drawdown_pct"] == 10


def test_simultaneous_exits_do_not_create_artificial_order_dependent_drawdown():
    frame = trades([("2026-01-01", 0, 1, 100, 90, 100),
                    ("2026-01-01", 0, 1, 100, 110, 100)])
    assert summarize_trades(frame, ["2026-01-01"], fee_bps=0)["closed_trade_max_drawdown_pct"] == 0


def test_empty_cluster_draws_are_omitted_not_filled_with_zero_return():
    frame = trades([("2026-01-01", 0, 1, 100, 110, 1)])
    result = summarize_trades(frame, ["2026-01-01", "2026-01-02"], fee_bps=0, repetitions=1000)
    assert result["mean_net_bps_ci95"] == pytest.approx([1000, 1000])
    assert 650 < result["bootstrap_valid_repetitions"] < 850


def test_training_selection_requires_30_trades_and_breaks_ties_toward_shorter():
    result = choose_horizon({1: {"n": 29, "mean_net_bps": 999},
                             5: {"n": 30, "mean_net_bps": -2},
                             10: {"n": 90, "mean_net_bps": -2},
                             60: {"n": 200, "mean_net_bps": -3}})
    assert result == {"horizon_seconds": 5, "selection_status": "selected",
                      "training_n": 30, "training_mean_net_bps": -2}
    assert choose_horizon({5: {"n": 29, "mean_net_bps": 10}})["selection_status"] == "insufficient"
    assert choose_horizon({})["horizon_seconds"] == 5
    assert choose_horizon({10: {"n": 30, "mean_net_bps": 1}})["horizon_seconds"] == 10


@pytest.mark.parametrize("column,value", [("quantity", 0), ("entry_price", -1), ("exit_price", float("nan")), ("exit_ms", -1)])
def test_invalid_accounting_inputs_are_rejected(column, value):
    frame = trades([("2026-01-01", 0, 1, 100, 110, 1)])
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        summarize_trades(frame, ["2026-01-01"])


def test_days_cannot_be_missing_or_duplicate():
    frame = trades([("2026-01-01", 0, 1, 100, 110, 1)])
    with pytest.raises(ValueError, match="missing from all_days"):
        summarize_trades(frame, [])
    with pytest.raises(ValueError, match="unique"):
        summarize_trades(frame, ["2026-01-01", "2026-01-01"])

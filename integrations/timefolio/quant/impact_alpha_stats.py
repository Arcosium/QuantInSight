"""Offline, long-only alpha statistics; no market or order APIs are used."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "day", "entry_ms", "exit_ms", "entry_price", "exit_price", "quantity",
)


def _nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def add_net_columns(
    frame: pd.DataFrame,
    fee_bps: float = 10,
    extra_slippage_bps: float = 0,
) -> pd.DataFrame:
    """Copy trades and charge fees/slippage on both actual trade notionals.

    Slippage is an additional cost assumption, separate from any spread or
    book-walking cost already embedded in entry_price and exit_price.
    """
    fee_bps = _nonnegative_finite(fee_bps, "fee_bps")
    extra_slippage_bps = _nonnegative_finite(extra_slippage_bps, "extra_slippage_bps")
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing and not frame.empty:
        raise ValueError(f"missing trade columns: {sorted(missing)}")
    out = frame.copy()
    for column in missing:
        out[column] = pd.Series(index=out.index, dtype="object" if column == "day" else float)
    for column in REQUIRED_COLUMNS[1:]:
        out[column] = pd.to_numeric(out[column], errors="raise").astype(float)
        if not np.isfinite(out[column].to_numpy()).all():
            raise ValueError(f"{column} must be finite")
    if out["day"].isna().any():
        raise ValueError("day must not be missing")
    out["day"] = out["day"].astype(str)
    if (out[["entry_price", "exit_price", "quantity"]] <= 0).any().any():
        raise ValueError("prices and quantities must be positive")
    if (out["exit_ms"] < out["entry_ms"]).any():
        raise ValueError("exit_ms must be at or after entry_ms")

    out["entry_notional"] = out["entry_price"] * out["quantity"]
    out["turnover"] = (out["entry_price"] + out["exit_price"]) * out["quantity"]
    out["gross_pnl"] = out["quantity"] * (out["exit_price"] - out["entry_price"])
    out["fee_cost"] = fee_bps / 1e4 * out["turnover"]
    out["extra_slippage_cost"] = extra_slippage_bps / 1e4 * out["turnover"]
    out["net_pnl"] = out["gross_pnl"] - out["fee_cost"] - out["extra_slippage_cost"]
    out["gross_bps"] = (out["exit_price"] - out["entry_price"]) / out["entry_price"] * 1e4
    out["net_bps"] = out["net_pnl"] / out["entry_notional"] * 1e4
    return out


def _daily_cluster_intervals(
    trades: pd.DataFrame, days: list[str], repetitions: int, seed: int,
) -> tuple[list[float] | None, list[float] | None, int]:
    """Resample complete days, then divide pooled bps sums by pooled counts.

    Empty days stay in the sampling population. Draws containing no trades
    cannot define a mean, so only those draws are omitted from the quantiles.
    The interval is pointwise and makes no multiple-testing adjustment.
    """
    if len(days) < 2 or trades.empty or repetitions == 0:
        return None, None, 0
    grouped = trades.groupby("day", sort=False).agg(
        count=("net_bps", "size"), net_sum=("net_bps", "sum"), gross_sum=("gross_bps", "sum"),
    ).reindex(days, fill_value=0)
    draws = np.random.default_rng(seed).integers(0, len(days), size=(repetitions, len(days)))
    counts = grouped["count"].to_numpy()[draws].sum(axis=1)
    valid = counts > 0
    if not valid.any():
        return None, None, 0
    intervals = []
    for name in ("net_sum", "gross_sum"):
        means = grouped[name].to_numpy()[draws].sum(axis=1)[valid] / counts[valid]
        intervals.append([float(value) for value in np.quantile(means, [0.025, 0.975])])
    return intervals[0], intervals[1], int(valid.sum())


def summarize_trades(
    frame: pd.DataFrame,
    all_days: Iterable[str],
    fee_bps: float = 10,
    extra_slippage_bps: float = 0,
    initial_capital: float = 10000,
    repetitions: int = 2000,
    seed: int = 928,
) -> dict:
    """Summarize fixed-size closed trades and UTC-day cluster uncertainty.

    Returns plain JSON values. Undefined means, win rates, profit factors and
    confidence intervals are None. Equity includes starting capital and P&L
    realized at each distinct exit timestamp. Drawdown is closed-trade only;
    it does not measure adverse movement while a position remains open.
    """
    initial_capital = float(initial_capital)
    if not math.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be finite and positive")
    if not isinstance(repetitions, (int, np.integer)) or repetitions < 0:
        raise ValueError("repetitions must be a nonnegative integer")
    days = [str(day) for day in all_days]
    if len(set(days)) != len(days):
        raise ValueError("all_days must contain unique UTC dates")
    trades = add_net_columns(frame, fee_bps, extra_slippage_bps)
    unexpected_days = set(trades["day"]).difference(days)
    if unexpected_days:
        raise ValueError(f"trade dates missing from all_days: {sorted(unexpected_days)}")
    n = len(trades)
    pnl = float(trades["net_pnl"].sum())
    positive_pnl = float(trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum())
    negative_pnl = -float(trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum())
    closed_pnl = trades.groupby("exit_ms", sort=True)["net_pnl"].sum().to_numpy()
    equity = np.r_[initial_capital, initial_capital + np.cumsum(closed_pnl)]
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    net_ci, gross_ci, valid_repetitions = _daily_cluster_intervals(trades, days, repetitions, seed)
    turnover = float(trades["turnover"].sum())
    # This is the maximum additional per-side fee after the specified extra
    # slippage, so the net P&L is exactly zero when that fee is substituted.
    fee_budget = float((trades["gross_pnl"] - trades["extra_slippage_cost"]).sum())
    return {
        "n": int(n),
        "calendar_days": len(days),
        "trading_days": int(trades["day"].nunique()),
        "mean_gross_bps": float(trades["gross_bps"].mean()) if n else None,
        "mean_net_bps": float(trades["net_bps"].mean()) if n else None,
        "gross_pnl": float(trades["gross_pnl"].sum()),
        "net_pnl": pnl,
        "win_rate": float((trades["net_pnl"] > 0).mean()) if n else None,
        "profit_factor": positive_pnl / negative_pnl if negative_pnl > 0 else None,
        "cumulative_return_pct": pnl / initial_capital * 100,
        "closed_trade_max_drawdown_pct": float(drawdown.max() * 100),
        "mean_net_bps_ci95": net_ci,
        "mean_gross_bps_ci95": gross_ci,
        "ci_method": "pointwise UTC-day cluster percentile bootstrap; zero-trade days included",
        "bootstrap_repetitions": int(repetitions),
        "bootstrap_valid_repetitions": valid_repetitions,
        "break_even_fee_per_side_bps": fee_budget / turnover * 1e4 if turnover > 0 else None,
        "initial_capital": initial_capital,
        "fee_per_side_bps": float(fee_bps),
        "extra_slippage_per_side_bps": float(extra_slippage_bps),
    }


def choose_horizon(training_results: Mapping[int, Mapping]) -> dict:
    """Select using training-only mean net bps, requiring at least 30 trades.

    Evaluation results are intentionally not accepted as a separate input.
    The caller is responsible for supplying only the training partition.
    """
    eligible = []
    for horizon, result in training_results.items():
        horizon = int(horizon)
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        score = result.get("mean_net_bps")
        if int(result.get("n", 0)) >= 30 and score is not None and math.isfinite(float(score)):
            eligible.append((float(score), -horizon, result))
    if eligible:
        score, negative_horizon, result = max(eligible, key=lambda item: (item[0], item[1]))
        return {"horizon_seconds": -negative_horizon, "selection_status": "selected",
                "training_n": int(result["n"]), "training_mean_net_bps": score}
    fallback = training_results.get(5, {})
    score = fallback.get("mean_net_bps")
    return {"horizon_seconds": 5, "selection_status": "insufficient",
            "training_n": int(fallback.get("n", 0)),
            "training_mean_net_bps": float(score) if score is not None and math.isfinite(float(score)) else None}

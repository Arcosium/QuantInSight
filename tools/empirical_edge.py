"""Cost-aware research evidence from completed, non-overlapping observations.

This is a historical estimate, not an LLM forecast or a profit guarantee.
Shadow research only: this unvalidated estimate never admits or sizes live orders.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EdgeEvidence:
    eligible: bool = False
    reason: str = "insufficient_history"
    samples: int = 0
    horizon_days: int = 5
    gross_mean_pct: float | None = None
    net_mean_pct: float | None = None
    conservative_net_pct: float | None = None
    win_rate: float | None = None
    cost_pct: float = 0.0
    as_of: str = ""
    regime: str = ""

    def to_dict(self):
        return asdict(self)


def estimate_edge(daily, *, cost_pct: float, min_net_edge_pct: float = 0.0,
                  horizon: int = 5, min_samples: int = 12, as_of=None) -> EdgeEvidence:
    """Past signal at close -> next open entry -> close after `horizon` days.

    Only observations matching today's 20/60-day trend state are used. Sample
    windows never overlap; historical features only see their own past. A
    zero-return prior (20 samples) and one standard error penalize uncertainty.
    Today's unfinished bar is excluded even if the caller forgot to remove it.
    """
    if horizon < 1 or min_samples < 2 or not math.isfinite(cost_pct) or cost_pct < 0:
        raise ValueError("invalid evidence parameters")
    if not math.isfinite(min_net_edge_pct):
        raise ValueError("invalid edge threshold")
    base = dict(horizon_days=horizon, cost_pct=cost_pct)
    def result(**values):
        return EdgeEvidence(**(base | values))
    if daily is None or not {"date", "open", "close"}.issubset(daily.columns):
        return result()
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    today = pd.Timestamp(as_of or pd.Timestamp.now(tz="Asia/Seoul").date()).tz_localize(None).normalize()
    df = df[df.date < today].drop_duplicates("date", keep="last").sort_values("date")
    if len(df) < 60 + horizon * min_samples:
        return result()
    base["as_of"] = df.date.iloc[-1].strftime("%Y-%m-%d")
    if (today - df.date.iloc[-1]).days > 7:
        return result(reason="stale_prices")
    close = pd.to_numeric(df.close, errors="coerce").to_numpy(dtype=float)
    opens = pd.to_numeric(df.open, errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(close) & (close > 0)) or not np.all(np.isfinite(opens) & (opens > 0)):
        return result(reason="invalid_prices")
    # Unadjusted splits/reverse splits must not look like trading alpha.
    if np.any(np.abs(close[1:] / close[:-1] - 1) > .45):
        return result(reason="corporate_action_or_bad_price")
    s = pd.Series(close)
    ma20, ma60 = s.rolling(20).mean().to_numpy(), s.rolling(60).mean().to_numpy()
    trend = (close > ma60) & (ma20 > ma60)
    base["regime"] = "uptrend" if trend[-1] else "defensive"
    if not trend[-1]:
        return result(reason="defensive_regime")
    returns = []
    # Anchor at the first valid signal. Do not resample to maximize performance.
    for i in range(59, len(df) - horizon, horizon):
        if trend[i]:
            returns.append((close[i + horizon] / opens[i + 1] - 1) * 100)
    n = len(returns)
    if n < min_samples:
        return result(samples=n)
    arr = np.asarray(returns)
    gross = float(arr.mean())
    shrunk = gross * n / (n + 20)
    net = shrunk - cost_pct
    lower = net - float(arr.std(ddof=1) / math.sqrt(n))
    allowed = lower >= min_net_edge_pct
    return result(eligible=allowed, reason="positive_net_edge" if allowed else "unproven_net_edge",
                  samples=n, gross_mean_pct=gross, net_mean_pct=net,
                  conservative_net_pct=lower, win_rate=float((arr > cost_pct).mean()))


def evaluate_candidates(codes, load_daily, *, min_net_edge_pct=0.0):
    """Costs are explicit conservative assumptions, not advertised fee quotes."""
    evidence = {}
    for code in dict.fromkeys(codes):
        try:
            evidence[code] = estimate_edge(load_daily(code),
                cost_pct=0.35 if str(code).isdigit() else 0.70,
                min_net_edge_pct=min_net_edge_pct).to_dict()
        except Exception:
            evidence[code] = EdgeEvidence(reason="data_error").to_dict()
    return evidence


def rank_eligible(codes, evidence):
    return sorted((c for c in dict.fromkeys(codes) if evidence.get(c, {}).get("eligible")),
                  key=lambda c: evidence[c]["conservative_net_pct"], reverse=True)

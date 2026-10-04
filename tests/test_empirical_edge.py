import numpy as np
import pandas as pd
from tools.empirical_edge import estimate_edge, rank_eligible


def prices(drift=.002):
    dates = pd.bdate_range(end="2026-09-15", periods=420)
    close = 100 * np.exp(np.arange(len(dates)) * drift)
    return pd.DataFrame({"date": dates, "open": close, "close": close})


def test_positive_trend_must_cover_cost_and_uncertainty():
    df = prices()
    assert estimate_edge(df, cost_pct=.1, as_of="2026-09-16").eligible
    assert not estimate_edge(df, cost_pct=2, as_of="2026-09-16").eligible


def test_volatility_and_target_are_not_profit_evidence():
    df = prices(-.003)
    assert not estimate_edge(df, cost_pct=.35, as_of="2026-09-16").eligible
    assert not estimate_edge(df.tail(80), cost_pct=.35, as_of="2026-09-16").eligible
    assert not estimate_edge(None, cost_pct=.35).eligible


def test_unfinished_and_future_data_cannot_change_decision():
    df = prices()
    expected = estimate_edge(df, cost_pct=.35, as_of="2026-09-16")
    extra = pd.DataFrame({"date": pd.to_datetime(["2026-09-16", "2027-01-01"]),
                          "open": [99999, 1], "close": [99999, 1]})
    assert estimate_edge(pd.concat([df, extra]), cost_pct=.35, as_of="2026-09-16") == expected
    assert expected.samples <= (len(df) - 60) // 5


def test_stale_invalid_and_split_prices_block():
    df = prices()
    assert estimate_edge(df, cost_pct=.35, as_of="2026-10-01").reason == "stale_prices"
    df.loc[100, "close"] *= 10
    assert estimate_edge(df, cost_pct=.35, as_of="2026-09-16").reason == "corporate_action_or_bad_price"
    df.loc[100, "close"] = np.nan
    assert estimate_edge(df, cost_pct=.35, as_of="2026-09-16").reason == "invalid_prices"


def test_committee_cannot_reintroduce_unverified_names():
    evidence = {"A": {"eligible": True, "conservative_net_pct": 1},
                "B": {"eligible": True, "conservative_net_pct": 2},
                "C": {"eligible": False}}
    assert rank_eligible(["A", "B", "A", "C", "invented"], evidence) == ["B", "A"]

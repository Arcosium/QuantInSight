"""IS/OS/ROS 롤링 — 선택은 OS(직전 분기)만, 판정은 ROS 합산만."""
import numpy as np
import pandas as pd

from tools.policy_validation import walk_forward


def _series(idx, mu, seed):
    return pd.Series(np.random.default_rng(seed).normal(mu, .01, len(idx)), index=idx)


def test_pick_uses_previous_quarter_only():
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    q = idx.to_period("Q")
    # 5일 주기는 1분기에만 좋고, 20일 주기는 2분기부터 좋다 → 2분기 ROS 는 5일(1분기 OS 기준)을 써야 한다.
    r5 = pd.Series(np.where(q == pd.Period("2024Q1"), .004, -.002), index=idx) + _series(idx, 0, 1)
    r20 = pd.Series(np.where(q == pd.Period("2024Q1"), -.002, .004), index=idx) + _series(idx, 0, 2)
    r10 = _series(idx, 0, 3)
    res = walk_forward({5: r5, 10: r10, 20: r20}, _series(idx, 0, 4))
    picks = {f["ROS"]: f["pick_every"] for f in res["folds"]}
    assert picks["2024Q2"] == 5 and picks["2024Q3"] == 20 and picks["2024Q4"] == 20
    assert res["live_every"] == 20


def test_noise_is_not_validated():
    idx = pd.bdate_range("2023-01-01", "2025-12-31")
    res = walk_forward({e: _series(idx, 0, e) for e in (5, 10, 20)}, _series(idx, 0, 9))
    assert res["validated"] is False

"""stat-arb OU/s-score 자체검증 — 합성 평균회귀 시계열로 반감기 회복·부호 확인.

python3.12 -m pytest tests/test_statarb.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                    # noqa: E402
from core.analytics import ou_score              # noqa: E402


def _ar1(b, n, sigma=1e-3, seed=0):
    """X_{t+1}=b·X_t+ξ 합성 (평균 0 근방 평균회귀)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = b * x[t - 1] + rng.normal(0, sigma)
    return x


def test_recovers_halflife():
    b = 0.8                                   # 반감기 = ln2/(-ln b) ≈ 3.1 스텝
    o = ou_score(_ar1(b, 300))
    assert o is not None
    theo = math.log(2) / (-math.log(b))
    assert abs(o["half_life"] - theo) < 1.2   # 추정 반감기 ≈ 이론값
    assert o["sig_ok"] is True                # 빠른 평균회귀 → 적격


def test_oversold_gives_negative_s():
    x = _ar1(0.8, 300)
    x[-1] -= 6.0 * x.std()                     # 마지막을 크게 끌어내림(과매도)
    o = ou_score(x)
    assert o is not None
    assert o["s"] < -1.25                       # 진입 문턱 아래
    assert o["exp_ret"] > 0                     # 롱 후보는 기대반전 양수


def test_random_walk_rejected():
    # 단위근(b≈1) 랜덤워크는 평균회귀가 아니므로 적격이 아니어야 한다.
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 1e-3, 300))
    o = ou_score(x)
    assert o is None or o["sig_ok"] is False


def test_short_series_none():
    assert ou_score(np.zeros(config.STATARB_MIN_OBS - 1)) is None


def test_market_efficiency():
    from core.analytics import market_efficiency
    assert market_efficiency([0.01] * 10) > 0.99         # 완전추세(한 방향)
    assert market_efficiency([0.01, -0.01] * 10) < 0.05  # 완전횡보(왕복)
    assert market_efficiency([]) == 0.0                  # 빈 입력 안전


if __name__ == "__main__":                     # ponytail: 프레임워크 없이도 돈다
    test_recovers_halflife()
    test_oversold_gives_negative_s()
    test_random_walk_rejected()
    test_short_series_none()
    test_market_efficiency()
    print("test_statarb OK")

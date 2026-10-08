"""analytics 검증 — 잔차(시장중립화)가 시장 성분을 실제로 걷어내는지.

2026-07-30 정리: 그랜저 스캔·전이엔트로피·EngineState streak 테스트 4건을 제거했다.
7/28 stat-arb 전환 때 대상 함수가 엔진에서 사라져 AttributeError 로만 죽고 있었다.
2026-08-07: 팩터가 고유포트폴리오로 바뀌어(crypto 이식) 두 경로를 모두 검증한다.
s-score/OU 쪽 검증은 tests/test_statarb.py.

python3 -m pytest tests/test_analytics.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                     # noqa: E402
from core import analytics                        # noqa: E402


def _panel(n_codes=40, n_bars=800, seed=1):
    """공통 시장 팩터 하나에 종목별 베타·고유노이즈를 얹은 가격 패널 + 그 팩터."""
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 0.005, n_bars)
    beta = rng.uniform(0.5, 1.8, n_codes)
    R = np.outer(m, beta) + rng.normal(0, 5e-4, (n_bars, n_codes))
    codes = [f"{i:06d}" for i in range(n_codes)]
    idx = [f"20260701{900 + i:04d}" for i in range(n_bars)]
    return pd.DataFrame(100 * np.exp(np.cumsum(R, axis=0)), columns=codes, index=idx), m


def _max_abs_corr(resid, m):
    """잔차 각 열과 **진짜 시장 팩터**의 상관 최대값. 잔차 평균과의 상관보다 강한 검사다."""
    mm = m[1:len(resid) + 1]
    return max(abs(np.corrcoef(resid[:, i], mm)[0, 1]) for i in range(resid.shape[1]))


def test_residuals_remove_market_eigen():
    """고유포트폴리오 경로(기본). 시장 성분이 잔차에서 사라져야 한다.

    ⚠ 종목 수를 3개 같은 소수로 잡으면 **거짓 실패한다** — 3종목에서 2팩터를 빼면 잔차
    공간이 1차원이라 서로 완전 공선이 된다. crypto 쪽 테스트에서도 같은 함정을 밟았다.
    """
    prices, m = _panel()
    saved, config.USE_EIGEN_FACTORS = config.USE_EIGEN_FACTORS, 1
    try:
        resid, codes = analytics.to_residuals(prices)
    finally:
        config.USE_EIGEN_FACTORS = saved
    assert len(codes) == prices.shape[1]
    assert _max_abs_corr(resid, m) < 0.2, _max_abs_corr(resid, m)


def test_residuals_remove_market_legacy_two_factor():
    """구 2팩터(KOSPI·KOSDAQ 등가중평균) 경로도 계속 동작해야 한다 — 되돌릴 길이다."""
    prices, m = _panel()
    markets = {c: ("KOSPI" if i % 2 else "KOSDAQ") for i, c in enumerate(prices.columns)}
    saved, config.USE_EIGEN_FACTORS = config.USE_EIGEN_FACTORS, 0
    try:
        resid, _ = analytics.to_residuals(prices, markets)
    finally:
        config.USE_EIGEN_FACTORS = saved
    assert _max_abs_corr(resid, m) < 0.2, _max_abs_corr(resid, m)


def test_day_boundary_rows_dropped():
    """일 경계를 넘는 수익률 행은 빠져야 한다 — 밤샘 갭은 전 종목 공통이라 상관을 부풀린다."""
    prices, _ = _panel(n_codes=20, n_bars=100)
    idx = [f"20260701{900 + i:04d}" for i in range(50)] + \
          [f"20260702{900 + i:04d}" for i in range(50)]
    prices.index = idx
    resid, _ = analytics.to_residuals(prices)
    assert resid.shape[0] == 98, resid.shape      # 100행 → 99수익률 − 경계 1


def test_sector_demean_zeroes_group_sum():
    """섹터/클러스터 동시점 디민이 걸리면 같은 군의 잔차 합은 0 이다."""
    prices, _ = _panel(n_codes=20, n_bars=400)
    codes = list(prices.columns)
    sectors = {c: ("군1" if i < 10 else "군2") for i, c in enumerate(codes)}
    saved, config.USE_SECTOR_FACTOR = config.USE_SECTOR_FACTOR, 1
    try:
        resid, _ = analytics.to_residuals(prices, sectors=sectors)
    finally:
        config.USE_SECTOR_FACTOR = saved
    assert abs(resid[:, :10].sum(axis=1)).max() < 1e-9
    assert abs(resid[:, 10:].sum(axis=1)).max() < 1e-9

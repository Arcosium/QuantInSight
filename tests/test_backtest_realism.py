"""백테스트 엔진 신뢰성 — 토요일 주간리뷰 입력 정직성 수정 (2026-06-15).

두 결함을 고친다:
  ① 샤프 부호 모순: 산술평균 기반이라 변동성 드래그 구간에서 '음수 복리수익 + 양수 샤프'가
     나와, 돈 잃는 설정을 '위험조정 양호'로 오인하게 만든다 → 로그수익 기반으로 부호 일치.
  ② 진입필터 무반응: MAX_BUY_VOLATILITY_PCT 등 결정론 매수필터가 params 로 전달되는데도
     엔진이 SMA 돌파만 보고 무시 → ops 가 필터를 조여도 백테스트가 반응 안 함(피드백 단절).
"""
import math

import config
from backtest.engine import (
    load_prices,
    run_backtest,
    _metrics,
    passes_entry_filters,
)


# ── ① 샤프 부호 일치 ──────────────────────────────────────────────────────
def test_sharpe_negative_under_volatility_drag():
    """산술평균은 양수지만 복리 수익이 음수인 곡선 → 샤프도 음수여야 한다(부호 일치).

    [100,150,90,135,81]: 단순수익 +50/-40/+50/-40% → 산술평균 +5%(구버전이면 샤프>0, 버그).
    복리로는 -19% → 로그수익 평균 음수 → 샤프 음수가 정직하다.
    """
    curve = [100.0, 150.0, 90.0, 135.0, 81.0]
    m = _metrics("t", curve[0], curve[-1], curve, trades=4, wins=2)
    assert m["total_return_pct"] < 0
    assert m["sharpe_like"] < 0, "음수 복리수익인데 샤프가 양수면 안 됨(변동성 드래그 모순)"


def test_sharpe_positive_for_steady_growth():
    curve = [100.0 * (1.01 ** k) for k in range(30)]
    m = _metrics("t", curve[0], curve[-1], curve, trades=3, wins=3)
    assert m["total_return_pct"] > 0
    assert m["sharpe_like"] > 0


# ── ② 진입필터 honored ───────────────────────────────────────────────────
def _calm_series():
    """연환산 변동성 ≈ 28% (생성 가능·라이브 기본선 100% 미만)."""
    c = [100.0]
    for k in range(40):
        c.append(c[-1] * (1 + (0.02 if k % 2 == 0 else -0.015)))
    return c


def _wild_series():
    """연환산 변동성 ≈ 200%+ (라이브 매수필터 기본선 초과)."""
    c = [100.0]
    for k in range(40):
        c.append(c[-1] * (1 + (0.15 if k % 2 == 0 else -0.12)))
    return c


def test_entry_filter_blocks_extreme_volatility_by_default():
    """파라미터 미설정(off)이라도 라이브 퀀트 프롬프트 기본선(변동성<100%)을 적용해 차단."""
    wild = _wild_series()
    assert passes_entry_filters(wild, 40, 20, {}) is False


def test_entry_filter_honors_strict_volatility_param():
    """MAX_BUY_VOLATILITY_PCT 를 조이면 기본선을 통과하던 종목도 차단(파라미터 honored)."""
    calm = _calm_series()
    assert passes_entry_filters(calm, 40, 20, {}) is True
    assert passes_entry_filters(calm, 40, 20, {"MAX_BUY_VOLATILITY_PCT": 1.0}) is False


def test_backtest_responds_to_volatility_filter():
    """엔진이 실제로 필터를 적용 — 변동성 상한을 조이면 매매 수가 줄어든다(무반응=구버그)."""
    prices = load_prices()
    assert prices, "data/daily_*.csv 필요"
    loose = dict(config.STRATEGY_DEFAULTS)
    loose["MAX_BUY_VOLATILITY_PCT"] = 0.0          # off → 라이브 기본선 100%
    strict = dict(config.STRATEGY_DEFAULTS)
    strict["MAX_BUY_VOLATILITY_PCT"] = 20.0        # 연환산 변동성 20% 미만만 진입
    r_loose = run_backtest(loose, prices)
    r_strict = run_backtest(strict, prices)
    assert r_strict["trades"] < r_loose["trades"]

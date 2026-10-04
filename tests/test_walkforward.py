"""워크포워드 백테스트 하니스 — 롤링 아웃오브샘플 성과 안정성 (2026-06-15 ROI#1).

단일 백테스트 한 숫자는 '한 시기의 운'일 수 있다(과적합 증상). 히스토리를 연속 윈도우로 쪼개
각 구간을 평가하고, 구간별 분포(일관성·최악 구간)를 집계해 성과가 견고한지 본다.
"""
import config
from backtest.walkforward import walk_forward, _aggregate


def _synth(days=90):
    # 2종목 × days 일, 완만한 우상향 + 노이즈
    out = {}
    for code, base in (("AAA", 100.0), ("BBB", 50.0)):
        rows = []
        px = base
        for i in range(days):
            px = px * (1 + (0.01 if i % 3 else -0.008))
            d = f"2026-{1+i//30:02d}-{1+i%30:02d}"
            rows.append({"date": d, "close": px, "high": px * 1.01, "low": px * 0.99})
        out[code] = rows
    return out


def test_returns_windows_and_aggregate():
    res = walk_forward(config.STRATEGY_DEFAULTS, _synth(90), test_days=20, warmup_days=20)
    assert "windows" in res and "aggregate" in res
    assert res["aggregate"]["n_windows"] >= 1
    assert len(res["windows"]) == res["aggregate"]["n_windows"]


def test_aggregate_consistency_metric_bounded():
    res = walk_forward(config.STRATEGY_DEFAULTS, _synth(90), test_days=20, warmup_days=20)
    agg = res["aggregate"]
    assert 0.0 <= agg["pct_positive"] <= 1.0
    assert set(agg) >= {"mean_return_pct", "median_return_pct", "pct_positive",
                        "worst_return_pct", "worst_mdd_pct", "n_windows"}


def test_too_short_history_no_windows():
    res = walk_forward(config.STRATEGY_DEFAULTS, _synth(10), test_days=20, warmup_days=20)
    assert res["aggregate"]["n_windows"] == 0


def test_aggregate_pure_function():
    wins = [{"return_pct": 5.0, "mdd_pct": -3.0}, {"return_pct": -2.0, "mdd_pct": -8.0},
            {"return_pct": 3.0, "mdd_pct": -1.0}]
    agg = _aggregate(wins)
    assert agg["n_windows"] == 3
    assert agg["pct_positive"] == round(2 / 3, 3)        # 3건 중 2건 양(+)
    assert agg["worst_return_pct"] == -2.0
    assert agg["worst_mdd_pct"] == -8.0

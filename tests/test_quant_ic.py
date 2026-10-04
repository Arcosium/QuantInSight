"""P1 — 퀀트점수 지표 IC 측정 (2026-08-02).

핵심 계약 3개: ① IC 분포 통계, ② 가중치 부호 vs 실측 IC 불일치 감지,
③ 룩어헤드 없이 실제 CSV 로 돌아간다(투자자 CSV 를 현재 시점에서 읽지 않는다).
"""
import pytest

from backtest.quant_ic import _ic_stats, sign_conflicts, summary_lines, run_ic


def test_ic_stats_ir_and_hit_rate():
    st = _ic_stats([0.1, -0.1, 0.2, 0.0])
    assert st["n_dates"] == 4 and st["hit_rate"] == 0.5
    assert st["ic_mean"] == pytest.approx(0.05, abs=1e-9)
    assert st["ir"] == pytest.approx(st["ic_mean"] / st["ic_std"], abs=1e-3)


def test_ic_stats_empty_is_honest():
    assert _ic_stats([]) == {"ic_mean": None, "ic_std": None, "ir": None, "hit_rate": None, "n_dates": 0}


def test_sign_conflict_flags_only_meaningful_ones():
    res = {"weights_used": {"mom": 12, "rsi": 5, "vwap": 8, "cmf": 0},
           "by_axis": {
               "mom": {"h20": {"ic_mean": -0.05, "n_dates": 20}},    # 부호 불일치 → 적발
               "rsi": {"h20": {"ic_mean": -0.001, "n_dates": 20}},   # 잡음 수준 → 보류
               "vwap": {"h20": {"ic_mean": -0.09, "n_dates": 3}},    # 표본 부족 → 보류
               "cmf": {"h20": {"ic_mean": -0.20, "n_dates": 20}},    # 가중치 0 → 대상 아님
           }}
    flags = sign_conflicts(res)
    assert len(flags) == 1 and flags[0].startswith("mom:")


def test_summary_lines_reports_unavailable_reason():
    assert "불가" in summary_lines({"available": False, "reason": "no_daily_csv"})[0]


def test_run_ic_on_real_csv_excludes_unreproducible_axes():
    """CSV 가 있으면 실제로 돌고, 재현 불가 축(flow·leadlag)은 절대 결과에 없어야 한다.
    (compute_quant_indicators 에 investor=[] 를 넘기지 않으면 현재 수급 CSV 를 읽어 룩어헤드가 된다.)"""
    r = run_ic(max_names=40, dates=3, step=5, min_names=10)
    if not r.get("available"):
        pytest.skip(f"일봉 CSV 부족: {r.get('reason')}")
    assert not ({"flow", "leadlag"} & set(r["by_axis"]))
    assert not ({"flow", "leadlag"} & set(r["weights_used"]))
    assert r["names_per_date"] >= 10

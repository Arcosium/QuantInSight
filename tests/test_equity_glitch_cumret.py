"""_equity_points — 결제 글리치 carry-forward + 누적수익 기반(R6/R7).

사장 지시 2026-05-22: 보유 종목 변동이 없는데 총평가가 비정상 급변하면(KIS 결제 글리치)
직전 값을 유지해 누적수익·MDD·그래프에 가짜 스파이크가 끼지 않게 한다. 보유가 실제로
바뀐 시점의 변동은 정상이므로 그대로 둔다.
"""
from main_swarm import _equity_points, performance_kpis

_H1 = {"005930": 1}


def _pt(ts, total, holdings=_H1):
    return {"ts": ts, "total_eval": total, "holdings": holdings, "ok": True}


def test_glitch_carried_forward_when_holdings_unchanged():
    raw = [
        _pt("2026-05-22 10:00:00", 1_000_000),
        _pt("2026-05-22 10:05:00", 600_000),   # 보유 동일인데 -40% → 글리치 → 1,000,000 유지
        _pt("2026-05-22 10:10:00", 1_010_000),
    ]
    vals = [v for _dt, v, _p in _equity_points(raw)]
    assert vals == [1_000_000, 1_000_000, 1_010_000]


def test_real_change_kept_when_holdings_changed():
    raw = [
        _pt("2026-05-22 10:00:00", 1_000_000, {"005930": 1}),
        _pt("2026-05-22 10:05:00", 600_000, {"005930": 0}),  # 보유 변동(매도) → 정상, 그대로
    ]
    vals = [v for _dt, v, _p in _equity_points(raw)]
    assert vals == [1_000_000, 600_000]


def test_normal_small_move_kept():
    raw = [
        _pt("2026-05-22 10:00:00", 1_000_000),
        _pt("2026-05-22 10:05:00", 1_030_000),  # +3% (가격 변동) → 유지
    ]
    vals = [v for _dt, v, _p in _equity_points(raw)]
    assert vals == [1_000_000, 1_030_000]


def test_mdd_excludes_glitch():
    raw = [
        _pt("2026-05-22 10:00:00", 1_000_000),
        _pt("2026-05-22 10:05:00", 500_000),   # 글리치 -50% → 제외
        _pt("2026-05-22 10:10:00", 1_005_000),
    ]
    k = performance_kpis(raw_equity=raw, trades=[])
    # 글리치가 MDD에 반영되면 -50% 가 찍힘 — carry-forward 로 거의 0 이어야 한다
    assert k["mdd_pct"] > -1.0


def test_transient_glitch_caught_when_holdings_missing():
    """강화(사장 지시 2026-05-24): 보유 스냅샷이 누락된 포인트라도, 직후 값이 직전으로
    되돌아오는 '일시 스파이크'면 글리치로 보고 직전 값을 유지한다."""
    raw = [
        {"ts": "2026-05-22 10:00:00", "total_eval": 1_000_000, "holdings": _H1, "ok": True},
        {"ts": "2026-05-22 10:05:00", "total_eval": 400_000, "ok": True},   # holdings 누락 + -60% 스파이크
        {"ts": "2026-05-22 10:10:00", "total_eval": 1_010_000, "holdings": _H1, "ok": True},
    ]
    vals = [v for _dt, v, _p in _equity_points(raw)]
    assert vals == [1_000_000, 1_000_000, 1_010_000]


def test_sustained_move_kept_when_holdings_missing():
    """반대로, 보유 미상이어도 되돌아오지 않는 '지속 변동'은 실제 손익으로 보존한다
    (며칠에 걸친 +변동을 글리치로 동결하면 안 됨)."""
    raw = [
        {"ts": "2026-05-18 10:00:00", "total_eval": 1_000_000, "ok": True},
        {"ts": "2026-05-19 10:00:00", "total_eval": 1_300_000, "ok": True},  # +30% 지속(되돌림 없음)
        {"ts": "2026-05-20 10:00:00", "total_eval": 1_350_000, "ok": True},
    ]
    vals = [v for _dt, v, _p in _equity_points(raw)]
    assert vals == [1_000_000, 1_300_000, 1_350_000]


def test_spike_with_holdings_change_removed():
    """사장 보고 2026-09-25: 모의 스파이크는 보유목록 변동과 자주 겹친다 — 되돌아오면 글리치."""
    raw = [
        _pt("2026-09-24 23:06:00", 38_246_000, {"KO": 4}),
        _pt("2026-09-24 23:12:00", 40_380_000, {"KO": 4, "AAPL": 1}),   # +5.6% 후 복귀
        _pt("2026-09-24 23:17:00", 38_230_000, {"KO": 4, "AAPL": 1}),
    ]
    vals = [v for _dt, v, _p in _equity_points(raw)]
    assert vals == [38_246_000, 38_246_000, 38_230_000]


def test_multi_point_spike_removed():
    """해외분이 3폴 연속 빠졌다가 돌아온 경우(uid2 9/10 23:05~23:17)."""
    raw = [
        _pt("2026-09-10 23:00:00", 46_269_000),
        _pt("2026-09-10 23:05:00", 26_174_000),
        _pt("2026-09-10 23:10:00", 26_156_000),
        _pt("2026-09-10 23:17:00", 25_116_000),
        _pt("2026-09-10 23:22:00", 46_178_000),
    ]
    vals = [v for _dt, v, _p in _equity_points(raw)]
    assert vals == [46_269_000] * 4 + [46_178_000]


def test_sustained_move_with_same_holdings_not_frozen():
    """종전 '보유 동일이면 즉시 직전값 유지'는 되돌림을 안 봐서 실제 변동을 동결했다."""
    raw = [
        _pt("2026-09-10 10:00:00", 1_000_000),
        _pt("2026-09-10 10:05:00", 1_120_000),   # +12% 지속, 보유 동일
        _pt("2026-09-10 10:10:00", 1_125_000),
        _pt("2026-09-10 10:15:00", 1_130_000),
        _pt("2026-09-10 10:20:00", 1_128_000),
    ]
    vals = [v for _dt, v, _p in _equity_points(raw)]
    assert vals == [1_000_000, 1_120_000, 1_125_000, 1_130_000, 1_128_000]

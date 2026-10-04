"""자산곡선 정직성 — 리시드/허수정정 단차는 매매손실로 안 친다 (2026-06-17).

배경: 허수(047810) 제거로 ledger_eval 이 -310,749 단차로 떨어지면, 자산곡선 cum_pnl 이
이를 '매매손실'로 표시해 사장이 -30만원 실손실로 오인했다. 실제론 장부 정정(가치 자체가
허수였음). reconcile_adj 가 박힌 포인트의 단차는 cum_pnl 누적에서 carry-forward 한다 —
표시 평가액(adj_total_eval)은 실제값으로 내려가되, 수익선은 가짜 손실을 안 찍는다.
"""
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from main_swarm import get_equity_series, performance_kpis

KST = ZoneInfo("Asia/Seoul")


def _write(tmp_path, raw):
    f = tmp_path / "equity_curve.json"
    f.write_text(json.dumps(raw), encoding="utf-8")
    return f


def test_reconcile_drop_excluded_from_cum_pnl(tmp_path):
    raw = [
        {"ts": "2026-06-15 10:00:00", "total_eval": 6539048, "ledger_eval": 6539048, "ok": True},
        # 허수 제거 단차 -310,749 (장부 정정) — 매매손실 아님
        {"ts": "2026-06-16 10:00:00", "total_eval": 6228299, "ledger_eval": 6228299,
         "reconcile_adj": -310749, "ok": True},
        {"ts": "2026-06-17 10:00:00", "total_eval": 6230000, "ledger_eval": 6230000, "ok": True},
    ]
    out = get_equity_series(_write(tmp_path, raw), view="daily")
    cum = [p["cum_pnl"] for p in out]
    val = [p["adj_total_eval"] for p in out]
    assert cum[0] == 0
    assert abs(cum[1]) < 1.0                       # 정정 단차는 손실로 안 찍힘 (≈0)
    assert abs(cum[2] - 1701) < 1.0                # 정정 후 실제 변동(6230000-6228299)만 반영
    assert abs(val[1] - 6228299) < 1.0             # 표시 평가액은 실제값으로 정직하게 내려감


def test_eq_change_buckets_exclude_reconcile_adj():
    """KPI '평가금액 변동'(eq_all/week/month_chg)도 cum_pnl 처럼 reconcile_adj(장부정정)를
    손실로 안 친다. 현재 평가액(current)은 실제값 그대로 둔다. (2026-06-18 버그 A)"""
    raw = [
        {"ts": "2026-06-11 10:00:00", "total_eval": 6624058, "ledger_eval": 6624058, "ok": True},
        # 허수 제거 장부정정 -310,749 (06-17) — 매매손실 아님
        {"ts": "2026-06-17 09:27:00", "total_eval": 6228299, "ledger_eval": 6228299,
         "reconcile_adj": -310749, "ok": True},
        {"ts": "2026-06-18 16:00:00", "total_eval": 6183317, "ledger_eval": 6183317, "ok": True},
    ]
    now = datetime(2026, 6, 18, 17, 0, tzinfo=KST)
    kpi = performance_kpis(raw_equity=raw, trades=[], now=now)
    # 현재 평가액은 실제값 그대로 (정정 반영된 실제 6,183,317)
    assert abs(kpi["current"] - 6183317) < 1
    # 전체/주/월 변동은 -310,749 장부정정을 제외 → 실제 매매변동 ≈ -129,992
    #   (raw 단순차 6,183,317-6,624,058 = -440,741, 정정 +310,749 보정)
    assert abs(kpi["eq_all_chg"] - (-129992)) < 2
    assert abs(kpi["eq_week_chg"] - (-129992)) < 2
    assert abs(kpi["eq_month_chg"] - (-129992)) < 2


def test_no_reconcile_adj_behaves_as_before(tmp_path):
    # reconcile_adj 없으면 기존 동작(cum_pnl = ledger_eval - baseline) 그대로
    raw = [
        {"ts": "2026-06-15 10:00:00", "total_eval": 1_000_000, "ledger_eval": 1_000_000, "ok": True},
        {"ts": "2026-06-16 10:00:00", "total_eval": 1_050_000, "ledger_eval": 1_050_000, "ok": True},
    ]
    out = get_equity_series(_write(tmp_path, raw), view="daily")
    cum = [p["cum_pnl"] for p in out]
    assert cum == [0, 50_000]

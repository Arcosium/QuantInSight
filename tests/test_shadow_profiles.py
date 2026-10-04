"""P3 — 섀도우 프로파일 비교 (2026-08-02).

모의 계정들이 서로 다른 파라미터로 병렬 운용되는 것을 저비용 알파풀로 쓴다.
계약: 입출금 보정 · 날짜별 마지막 관측 · 공통일 부족 시 상관 None(무음 금지) ·
base 대비 '더 높은 샤프 + 낮은 상관'만 채택 후보로 지목.
"""
import json

import infra.shadow_profiles as sp


def _write(tmp_path, uid, rows):
    d = tmp_path / "data" / str(uid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "equity_curve.json").write_text(json.dumps(rows), encoding="utf-8")


def _curve(uid, tmp_path, vals, *, flow=0.0):
    _write(tmp_path, uid, [{"ts": f"2026-07-{10 + i:02d} 09:00:00", "total_eval": v,
                            "external_flow_cum": flow} for i, v in enumerate(vals)])


def test_daily_equity_adjusts_flow_and_keeps_last_of_day(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "PROJECT_ROOT", tmp_path)
    _write(tmp_path, 1, [
        {"ts": "2026-07-10 09:00:00", "total_eval": 1000.0, "external_flow_cum": 0.0},
        {"ts": "2026-07-10 15:00:00", "total_eval": 1100.0, "external_flow_cum": 0.0},   # 같은 날 → 마지막만
        {"ts": "2026-07-11 15:00:00", "total_eval": 2100.0, "external_flow_cum": 1000.0},  # 입금 1000 보정
    ])
    assert sp.daily_equity(1) == {"2026-07-10": 1100.0, "2026-07-11": 1100.0}


def test_compare_needs_two_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "PROJECT_ROOT", tmp_path)
    _curve(1, tmp_path, [100, 101, 102, 103, 104, 105])
    r = sp.compare()
    assert r["available"] is False and "부족" in r["reason"]


def test_compare_reports_correlation_and_diversifier(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("infra.profile_overrides.load", lambda uid: {"MIN_QUANT_SCORE": 6 + uid})
    _curve(7, tmp_path, [100, 101, 100, 101, 100, 101, 100])          # 톱니 — 낮은 샤프
    _curve(2, tmp_path, [100, 102, 104, 106, 108, 110, 112])          # 우상향 — 높은 샤프
    r = sp.compare(base_uid=7)
    assert r["available"] and set(r["profiles"]) == {"7", "2"}
    assert r["profiles"]["2"]["sharpe_like"] > r["profiles"]["7"]["sharpe_like"]
    assert r["corr_days"]["2|7"] == 6
    d = r["diversifiers"]
    assert len(d) == 1 and d[0].startswith("uid 2:")
    assert "MIN_QUANT_SCORE 13→8" in d[0]          # 파라미터 차이가 함께 보고된다
    assert "프로파일 간 상관" in " ".join(sp.summary_lines(r))


def test_correlation_none_when_common_days_short(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "PROJECT_ROOT", tmp_path)
    _write(tmp_path, 1, [{"ts": f"2026-07-{10 + i:02d} 09:00:00", "total_eval": 100 + i,
                          "external_flow_cum": 0.0} for i in range(6)])
    _write(tmp_path, 2, [{"ts": f"2026-08-{10 + i:02d} 09:00:00", "total_eval": 100 + i,
                          "external_flow_cum": 0.0} for i in range(6)])   # 겹치는 날 없음
    r = sp.compare()
    assert r["correlation"]["1|2"] is None and r["corr_days"]["1|2"] == 0

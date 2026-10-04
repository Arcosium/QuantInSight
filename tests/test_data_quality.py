"""데이터 품질 모니터 — silent corruption 자동 탐지 (2026-06-15 ROI#5).

이번 세션 버그(당일봉 고정·부분체결 오기록·원장 괴리) 후속: 일봉 이상(말미 0거래량 잔존·중복
날짜·큰 갭·stale)과 KIS↔원장 수량 괴리를 매 사이클 자동 점검해 조용한 오염을 표면화한다.
"""
import pandas as pd
from infra.data_quality import csv_issues, ledger_drift_issues


def _df(dates, vols):
    return pd.DataFrame({"date": pd.to_datetime(dates), "close": [100]*len(dates), "volume": vols})


def test_clean_csv_no_issues():
    df = _df(["2026-06-10", "2026-06-11", "2026-06-12"], [1000, 1100, 1200])
    assert csv_issues("005930", df, today_str="2026-06-15") == []


def test_trailing_zero_volume_flagged():
    df = _df(["2026-06-11", "2026-06-12", "2026-06-15"], [1000, 1100, 0])
    issues = csv_issues("005930", df, today_str="2026-06-15")
    assert any("거래량 0" in i or "0거래량" in i for i in issues)


def test_duplicate_date_flagged():
    df = _df(["2026-06-11", "2026-06-12", "2026-06-12"], [1000, 1100, 1200])
    issues = csv_issues("005930", df, today_str="2026-06-15")
    assert any("중복" in i for i in issues)


def test_stale_data_flagged():
    df = _df(["2026-05-01", "2026-05-02"], [1000, 1100])  # 6주 전 stale
    issues = csv_issues("005930", df, today_str="2026-06-15")
    assert any("stale" in i.lower() or "오래" in i for i in issues)


def test_ledger_drift_surfaced():
    diffs = ["462870: KIS 314주 vs 원장 8주"]
    out = ledger_drift_issues(diffs)
    assert len(out) == 1 and "462870" in out[0]
    assert ledger_drift_issues([]) == []

"""원장 괴리 전이 오탐 알림 억제 — 2026-06-22.

배경(월 06-22 모의 uid2): 부분체결 직후~5분 폴링 사이의 '전이 상태'에 reconcile 이 KIS>원장
괴리(357870: KIS 226 vs 원장 162)를 잡아 매 사이클 푸시 알림을 냈다. 폴링이 잔량을 채워 곧
자가치유되는데도 사장님께 알림 노이즈가 갔다(06-19 이후 14회). 사이클 연속 '지속된' 괴리만
알림으로 올리고, 1회성 전이 괴리는 거른다(로그는 유지).
"""
from infra.data_quality import persistent_drift_issues


def test_first_sighting_suppressed():
    streak = {}
    out = persistent_drift_issues(["357870: KIS 226주 vs 원장 162주"], streak, threshold=2)
    assert out == []                 # 1회성 → 알림 억제
    assert streak == {"357870": 1}   # 연속 카운트만 기록


def test_second_consecutive_surfaces():
    streak = {"357870": 1}
    out = persistent_drift_issues(["357870: KIS 226주 vs 원장 162주"], streak, threshold=2)
    assert out == ["357870: KIS 226주 vs 원장 162주"]   # 2사이클 연속 → 알림
    assert streak == {"357870": 2}


def test_resolved_drift_resets_streak():
    streak = {"357870": 1}
    out = persistent_drift_issues([], streak, threshold=2)   # 폴링이 채워 괴리 사라짐
    assert out == []
    assert streak == {}              # 리셋(다음 전이가 다시 1부터)


def test_different_ticker_independent():
    streak = {"357870": 1}
    out = persistent_drift_issues(["003070: KIS 472주 vs 원장 372주"], streak, threshold=2)
    assert out == []                 # 새 종목 첫 등장 → 억제
    assert streak == {"003070": 1}   # 357870 은 이번에 없어 리셋, 003070 신규


def test_threshold_one_surfaces_immediately():
    streak = {}
    out = persistent_drift_issues(["003070: KIS 472주 vs 원장 372주"], streak, threshold=1)
    assert out == ["003070: KIS 472주 vs 원장 372주"]

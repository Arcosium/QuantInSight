"""운용지원실장 anti-oscillation — 직전 조정을 되감는 진동 차단 (버그 B, 2026-06-18).

OPS#455(14:07) PER_ORDER_BUDGET_RATIO 0.3→1.0, OPS#457(15:09) 1.0→0.3 으로 1시간 내 되감기.
같은 키를 window 내 반대 방향으로 되감는 변경은 진동으로 보고 보류한다(목표지향 같은방향 조정은 허용).
"""
from infra.ops_support_worker import filter_oscillating_overrides


def test_blocks_reversal_within_window():
    last = {"PER_ORDER_BUDGET_RATIO": {"from": 0.3, "to": 1.0, "ts": "2026-06-18 14:07:00"}}
    kept, dropped = filter_oscillating_overrides(
        {"PER_ORDER_BUDGET_RATIO": 0.3}, last, now_ts="2026-06-18 15:09:00", window_sec=7200)
    assert kept == {}
    assert "PER_ORDER_BUDGET_RATIO" in dropped


def test_allows_same_direction_change():
    last = {"K": {"from": 0.3, "to": 0.5, "ts": "2026-06-18 14:00:00"}}
    kept, dropped = filter_oscillating_overrides(
        {"K": 0.7}, last, now_ts="2026-06-18 15:00:00", window_sec=7200)
    assert kept == {"K": 0.7} and dropped == {}   # 0.5→0.7 은 0.3→0.5 와 같은 방향(추세 강화)


def test_allows_reversal_after_window():
    last = {"K": {"from": 0.3, "to": 1.0, "ts": "2026-06-18 09:00:00"}}
    kept, _ = filter_oscillating_overrides(
        {"K": 0.3}, last, now_ts="2026-06-18 15:00:00", window_sec=3600)
    assert kept == {"K": 0.3}   # 6시간 경과 → 레짐 변화 가능, 허용


def test_new_key_passes():
    kept, dropped = filter_oscillating_overrides(
        {"K": 1.0}, {}, now_ts="2026-06-18 15:00:00", window_sec=7200)
    assert kept == {"K": 1.0} and dropped == {}


def test_bool_and_nonnumeric_pass_through():
    last = {"ENABLE_X": {"from": True, "to": False, "ts": "2026-06-18 14:50:00"}}
    kept, dropped = filter_oscillating_overrides(
        {"ENABLE_X": True}, last, now_ts="2026-06-18 15:00:00", window_sec=7200)
    assert kept == {"ENABLE_X": True}   # 진동 판정은 수치형만(불리언은 통과)

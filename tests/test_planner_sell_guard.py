from datetime import datetime

from infra import planner_sell_guard as guard
from infra import user_paths


def _thesis():
    return {
        "entry_ts": "2026-08-27 09:00:00",
        "entry_price": 100.0,
        "target_price": 110.0,
        "stop_price": 93.0,
        "planned_hold_hours": 48,
        "entry_reason": "실적 개선",
    }


def _holding(price=101.0):
    return {"code": "005930", "name": "테스트", "qty": 3,
            "cur_price": price, "pnl_pct": price - 100.0}


def test_planned_position_gets_strong_objection():
    result = guard.assess_objection(
        _thesis(), _holding(), hold_days=0.2,
        now=datetime(2026, 8, 27, 12, 0, 0))
    assert result["active"] is True
    assert result["hard_override"] is False
    assert "강한 매도 반대" in result["message"]


def test_stop_target_and_new_invalidator_bypass_deferral():
    now = datetime(2026, 8, 27, 12, 0, 0)
    assert guard.assess_objection(_thesis(), _holding(92), now=now)["hard_override"] is True
    assert guard.assess_objection(_thesis(), _holding(111), now=now)["hard_override"] is True
    result = guard.assess_objection(_thesis(), _holding(), invalidators=["high:거래정지"], now=now)
    assert result["hard_override"] is True


def test_first_sell_deferred_same_cycle_stays_deferred_next_cycle_releases(tmp_path, monkeypatch):
    monkeypatch.setattr(user_paths, "_DATA_DIR", tmp_path)
    objection = guard.assess_objection(
        _thesis(), _holding(), now=datetime(2026, 8, 27, 12, 0, 0))
    final1, state1 = guard.apply_one_cycle_deferral(
        99, "005930", "전량", objection,
        now=datetime(2026, 8, 27, 12, 0, 0), cycle_key="cycle-1")
    final_same, state_same = guard.apply_one_cycle_deferral(
        99, "005930", "전량", objection,
        now=datetime(2026, 8, 27, 12, 1, 0), cycle_key="cycle-1")
    final2, state2 = guard.apply_one_cycle_deferral(
        99, "005930", "전량", objection,
        now=datetime(2026, 8, 27, 13, 0, 0), cycle_key="cycle-2")
    assert (final1, state1) == ("보유", "deferred")
    assert (final_same, state_same) == ("보유", "deferred")
    assert (final2, state2) == ("전량", "released")


def test_hold_clears_pending_deferral(tmp_path, monkeypatch):
    monkeypatch.setattr(user_paths, "_DATA_DIR", tmp_path)
    objection = guard.assess_objection(
        _thesis(), _holding(), now=datetime(2026, 8, 27, 12, 0, 0))
    guard.apply_one_cycle_deferral(98, "005930", "절반", objection, cycle_key="cycle-1")
    final, state = guard.apply_one_cycle_deferral(98, "005930", "보유", objection, cycle_key="cycle-2")
    assert (final, state) == ("보유", "cleared")

"""P4 — 파라미터 조정 이력을 운용지원실장 프롬프트에 주입 (2026-08-02).

filter_oscillating_overrides 는 '적용' 단계에서 되감기를 드롭하지만, LLM 은 이력을 몰라
매 사이클 같은 제안을 재생산했다. 이력 블록을 프롬프트에 넣어 재제안 자체를 줄인다.
"""
from infra.ops_support_worker import tuning_trail, _tuning_trail_block, build_prompt


def test_trail_groups_by_key_in_time_order(monkeypatch):
    import infra.profile_overrides as po
    monkeypatch.setattr(po, "load_proposals", lambda uid: [
        {"ts": "2026-07-28 06:00:00", "overrides_applied": {"MIN_QUANT_SCORE": 6.0}},
        {"ts": "2026-07-30 06:00:00", "overrides_applied": {"MIN_QUANT_SCORE": 6.5, "STOP_LOSS_PCT": 5.0}},
        {"ts": "2026-08-01 06:00:00", "overrides_applied": {}},          # 변경 없음 run 은 무시
    ])
    t = tuning_trail(7)
    assert [v for _, v in t["MIN_QUANT_SCORE"]] == [6.0, 6.5]
    assert len(t["STOP_LOSS_PCT"]) == 1


def test_flags_round_trip():
    """과거 값으로 되돌아온 키에만 왕복 경고 — 같은 방향 추세 조정은 경고하지 않는다."""
    flip = _tuning_trail_block({"A": [("07-28 06:00", 6.0), ("07-30 06:00", 6.5), ("08-01 06:00", 6.0)]})
    assert "⚠️왕복" in flip
    trend = _tuning_trail_block({"A": [("07-28 06:00", 6.0), ("07-30 06:00", 6.5), ("08-01 06:00", 7.0)]})
    assert "⚠️왕복" not in trend


def test_empty_trail_adds_nothing():
    assert _tuning_trail_block({}) == ""
    assert "최근 파라미터 조정 이력" not in build_prompt({"target_cycle": {"started_at": "t"}})


def test_build_prompt_includes_trail():
    p = build_prompt({"target_cycle": {"started_at": "t"},
                      "tuning_trail": {"MIN_QUANT_SCORE": [("07-28 06:00", 6.0), ("07-30 06:00", 6.5)]}})
    assert "최근 파라미터 조정 이력" in p and "MIN_QUANT_SCORE" in p

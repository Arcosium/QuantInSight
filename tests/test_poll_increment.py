"""부분체결 잔여분 다단계 폴링 — 매 폴링 '증분만' 기록하고 목표 도달까지 추적 (2026-06-15 후속).

①의 후속: 부분체결(예: 84주 중 33주)이 나면 잔여 51주를 5분 폴링이 계속 추적하되,
이미 기록한 33주를 다시 세지 않도록(이동 base + 누적 recorded) 증분만 원장에 반영한다.
"""
from main_swarm import _poll_increment


def test_full_fill_one_poll():
    # base 0 → after 314, 목표 314 → 314 기록·완료
    rec, base, recorded, done = _poll_increment("buy", base_qty=0, after_qty=314, target=314, recorded=0)
    assert (rec, base, recorded, done) == (314, 314, 314, True)


def test_two_stage_partial_no_double_count():
    # 1차: base 0 → after 33 (목표 84) → 33 기록, 미완료
    r1, b1, rec1, d1 = _poll_increment("buy", 0, 33, 84, 0)
    assert (r1, b1, rec1, d1) == (33, 33, 33, False)
    # 2차: base 33 → after 84 → 추가 51만 기록(33 재계산 없음), 완료
    r2, b2, rec2, d2 = _poll_increment("buy", b1, 84, 84, rec1)
    assert (r2, b2, rec2, d2) == (51, 84, 84, True)


def test_no_change_records_zero():
    r, b, rec, d = _poll_increment("buy", 33, 33, 84, 33)
    assert (r, b, rec, d) == (0, 33, 33, False)


def test_sell_increment():
    # 매도 40주, base 40 → after 10 → 30 기록, 잔여 10 미완료
    r, b, rec, d = _poll_increment("sell", 40, 10, 40, 0)
    assert (r, b, rec, d) == (30, 10, 30, False)


def test_caps_at_target():
    # 잔고가 목표보다 더 늘어도(타 매수 합산) 목표분만 기록·완료
    r, b, rec, d = _poll_increment("buy", 0, 500, 314, 0)
    assert (r, rec, d) == (314, 314, True)

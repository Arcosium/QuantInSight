"""US 매수 회로차단기 — 사장 지시 2026-06-17.

배경(uid1 hh09080 06/16 야간): USD 예수금 0/마이너스인데 매크로 '주식 95%' 압력에 트레이더가
매 사이클 US 종목을 재선정 → 사이징 재초안 → KIS '주문가능금액 초과' 거부를 6사이클 반복.
운용지원실장(파라미터 튜닝 전용)은 코드 버그라 못 막았고, '같은 실패 N회 → 중단' 같은
사이클 간 상태가 전무했다.

이 회로차단기는 Fix B(클램프 거래소 교정)가 1차로 거른 뒤에도, 조회마저 실패해 못 살 주문이
반복 거부될 때의 2차 안전망이다. 자가치유: US 매도/매수 성공(USD 확보 신호) 시 즉시 리셋.
"""
from main_swarm import (_is_us_buy_fail, _us_activity_freed_cash,
                        _update_us_buy_fail_streak, _us_buy_circuit_open)


def _fail(tk):
    return {"ticker": tk, "market": "US", "side": "buy", "accepted": False,
            "filled": False, "result": f"[US매수 실패] {tk} → 주문가능금액을 초과 했습니다"}


def _buy_ok(tk):
    return {"ticker": tk, "market": "US", "side": "buy", "accepted": True, "filled": False,
            "result": f"[US매수] {tk} → 주문 전송 완료 되었습니다"}


def _sell_ok(tk):
    return {"ticker": tk, "market": "US", "side": "sell", "accepted": True, "filled": False,
            "result": f"[US매도] {tk} → 주문 전송 완료 되었습니다"}


# ── 실패 식별 ──
def test_detects_us_buy_amount_exceeded_fail():
    assert _is_us_buy_fail(_fail("JNJ")) is True


def test_kr_buy_fail_not_counted():
    assert _is_us_buy_fail({"ticker": "005930", "market": "KR", "side": "buy",
                            "accepted": False, "result": "주문가능금액을 초과 했습니다"}) is False


def test_accepted_us_buy_not_a_fail():
    assert _is_us_buy_fail(_buy_ok("DAL")) is False


# ── 스트릭 갱신 ──
def test_streak_increments_on_pure_fail_cycle():
    # PG·ARKX 둘 다 실패, US 성공 0 → +2
    assert _update_us_buy_fail_streak(0, [_fail("PG"), _fail("ARKX")]) == 2


def test_streak_resets_on_us_sell_even_with_fail():
    # 매도 성공이 있으면(USD 확보) 같은 사이클에 실패가 있어도 리셋 — 로테이션 정상
    assert _update_us_buy_fail_streak(3, [_sell_ok("WAL"), _fail("KEY")]) == 0


def test_streak_resets_on_us_buy_success():
    assert _update_us_buy_fail_streak(2, [_buy_ok("ANET"), _fail("DHI")]) == 0


def test_streak_unchanged_when_no_us_activity():
    assert _update_us_buy_fail_streak(2, [{"ticker": "005930", "market": "KR",
                                           "side": "buy", "accepted": True}]) == 2


# ── 회로 개방 판정 ──
def test_circuit_opens_at_threshold():
    assert _us_buy_circuit_open(2, 2) is True
    assert _us_buy_circuit_open(3, 2) is True


def test_circuit_closed_below_threshold():
    assert _us_buy_circuit_open(1, 2) is False


def test_threshold_zero_disables_breaker():
    assert _us_buy_circuit_open(99, 0) is False

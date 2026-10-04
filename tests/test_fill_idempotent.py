"""apply_fill 멱등 — 사장 지시 2026-06-16.

배경: 즉시체결(exec_immediate)된 매도가, 같은 종목을 직전 사이클에서 미체결로 주문해둔
폴링(poll_confirm)에 cross-cycle 로 재계상되어 원장에 중복 반영됐다(uid2 375500: 63주가
exec_immediate + poll_confirm 둘 다 → 보유 126 전량 차감, 실제 매도는 63). 폴링은 'baseline
대비 보유 감소 = 내 체결'로 가정하는데, 다른 사이클의 매도가 그 감소를 흡수하기 때문.
_is_duplicate_fill 은 최근 윈도우 내 동일 (ticker,side,qty,~price) 체결을 중복으로 차단한다.
부분체결 누적은 _poll_increment 가 '증분(다른 qty)'만 기록하므로 같은 qty 중복만 걸린다.
"""
from infra.trade_ledger import _is_duplicate_fill


def test_duplicate_within_window():
    fills = [{"ts": "2026-06-16 10:07:06", "ticker": "375500", "side": "sell", "qty": 63, "price": 89900.0}]
    assert _is_duplicate_fill(fills, "375500", "sell", 63, 89900.0, "2026-06-16 10:07:28") is True


def test_different_qty_not_duplicate():
    # 부분체결 증분(다른 qty)은 중복이 아니다
    fills = [{"ts": "2026-06-16 10:07:06", "ticker": "375500", "side": "sell", "qty": 63, "price": 89900.0}]
    assert _is_duplicate_fill(fills, "375500", "sell", 20, 89900.0, "2026-06-16 10:07:28") is False


def test_outside_window_not_duplicate():
    fills = [{"ts": "2026-06-16 10:00:00", "ticker": "375500", "side": "sell", "qty": 63, "price": 89900.0}]
    assert _is_duplicate_fill(fills, "375500", "sell", 63, 89900.0, "2026-06-16 10:20:00") is False


def test_different_ticker_not_duplicate():
    fills = [{"ts": "2026-06-16 10:07:06", "ticker": "AAA", "side": "sell", "qty": 63, "price": 89900.0}]
    assert _is_duplicate_fill(fills, "375500", "sell", 63, 89900.0, "2026-06-16 10:07:28") is False


def test_different_side_not_duplicate():
    fills = [{"ts": "2026-06-16 10:07:06", "ticker": "375500", "side": "buy", "qty": 63, "price": 89900.0}]
    assert _is_duplicate_fill(fills, "375500", "sell", 63, 89900.0, "2026-06-16 10:07:28") is False


def test_empty_fills_not_duplicate():
    assert _is_duplicate_fill([], "375500", "sell", 63, 89900.0, "2026-06-16 10:07:28") is False


def test_price_near_match_is_duplicate():
    # 직전호가 미세차이(0.1% 이내)는 동일 체결로 본다
    fills = [{"ts": "2026-06-16 10:07:06", "ticker": "375500", "side": "sell", "qty": 63, "price": 89900.0}]
    assert _is_duplicate_fill(fills, "375500", "sell", 63, 89950.0, "2026-06-16 10:07:28") is True


# ── repair vs poll_confirm 부분체결 이중계상 (2026-06-19) ─────────────────────
# 근본버그: repair_from_recent_partial_orders 가 부분체결 잔여분을 추정가(fill_price)로 기록한 뒤,
# 살아있는 background 폴링이 같은 잔여분을 '실제 체결가'로 또 기록 → 같은 qty 인데 가격만 달라
# (86300 vs 86800) 0.1% 게이트를 통과해 이중계상. 원장이 매도 과다기록으로 KIS 아래로 떨어져 고착
# (uid2 161890: 매도 247주 기록·보유 229뿐·KIS 65). repair 가 관여된 쌍은 가격 게이트를 면제한다.
def test_repair_then_poll_confirm_same_qty_dedup_despite_price_diff():
    fills = [{"ts": "2026-06-19 11:13:39", "ticker": "161890", "side": "sell",
              "qty": 83, "price": 86300.0, "note": "repair_partial_restart"}]
    assert _is_duplicate_fill(fills, "161890", "sell", 83, 86800.0,
                              "2026-06-19 11:18:51", note="poll_confirm") is True


def test_repair_as_current_fill_dedup_against_poll():
    # 순서 반대(poll 먼저 기록 → repair 시도)도 대칭으로 중복 차단
    fills = [{"ts": "2026-06-19 11:13:39", "ticker": "161890", "side": "sell",
              "qty": 83, "price": 86800.0, "note": "poll_confirm"}]
    assert _is_duplicate_fill(fills, "161890", "sell", 83, 86300.0,
                              "2026-06-19 11:18:51", note="repair_partial_restart") is True


def test_non_repair_pair_keeps_price_gate():
    # repair 무관 쌍은 기존 가격 게이트 유지(가격차 0.1% 초과 → 별개 체결)
    fills = [{"ts": "2026-06-19 11:13:39", "ticker": "161890", "side": "sell",
              "qty": 83, "price": 86300.0, "note": "exec_immediate"}]
    assert _is_duplicate_fill(fills, "161890", "sell", 83, 86800.0,
                              "2026-06-19 11:18:51", note="poll_confirm") is False


def test_repair_pair_still_needs_same_qty():
    # repair 면제는 가격만 — qty 가 다르면(정상 부분체결 증분) 여전히 중복 아님
    fills = [{"ts": "2026-06-19 11:13:39", "ticker": "161890", "side": "sell",
              "qty": 83, "price": 86300.0, "note": "repair_partial_restart"}]
    assert _is_duplicate_fill(fills, "161890", "sell", 20, 86800.0,
                              "2026-06-19 11:18:51", note="poll_confirm") is False

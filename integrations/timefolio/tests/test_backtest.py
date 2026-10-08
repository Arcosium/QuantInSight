"""web/backtest.py — 신호 실현손익 + 지도 페어 적중률.

계약:
  1. 숏에도 원화 손익이 붙는다(엔진과 같은 사이징). 예전엔 항상 null 이었다.
  2. 롱·숏이 **한 번의 계산**으로 함께 나온다 → 화면이 두 번 그려지지 않는다(깜빡임).
  3. 지도 승률은 엔진 진입 규칙(LEADER_JUMP / FOLLOWER_FLAT / lag)을 그대로 복제한다.
"""
from datetime import datetime, timedelta

import config
from web import backtest as bt


def _ts(minute):
    return (datetime(2026, 7, 10, 9, 0) + timedelta(minutes=minute)).isoformat()


def _bars(pairs):
    """[(분, 종가)] → {'YYYYMMDDHHMM': 종가}"""
    base = datetime(2026, 7, 10, 9, 0)
    return {bt.minute_key(base + timedelta(minutes=m)): c for m, c in pairs}


def test_long_uses_logged_pnl_and_computes_krw():
    sigs = [
        {"ts": _ts(0), "kind": "BUY", "follower": "003230", "leader": "069620",
         "qty": 3, "price": 1000.0},
        {"ts": _ts(5), "kind": "SELL", "follower": "003230", "leader": "069620",
         "qty": 3, "price": 1010.0, "pnl": 0.01},
    ]
    out = bt.resolve_signals(sigs, {})["signals"]
    sell = out[1]
    assert sell["trade_type"] == "L" and sell["resolved"] is True
    assert sell["ret"] == 0.01
    assert sell["krw"] == (1010.0 - 1000.0) * 3
    assert "ret" not in out[0]                     # BUY 는 진입일 뿐 손익이 없다


def test_short_gets_krw_pnl_with_engine_sizing():
    """숏은 실제 주문이 없다 — 엔진의 롱 사이징(ORDER_NOTIONAL // 진입가)으로 가상 수량을 잡는다."""
    entry, exit_px, lag = 100_000.0, 98_000.0, 3
    sigs = [{"ts": _ts(0), "kind": "SHORT_SIGNAL", "follower": "090430",
             "leader": "087010", "lag": lag, "price": entry}]
    closes = {"090430": _bars([(lag, exit_px)])}

    s = bt.resolve_signals(sigs, closes)["signals"][0]
    qty = max(1, int(config.ORDER_NOTIONAL // entry))
    assert s["trade_type"] == "S" and s["resolved"] is True
    assert abs(s["ret"] - (entry - exit_px) / entry) < 1e-12      # 하락 = 이익
    assert s["krw"] == (entry - exit_px) * qty > 0
    assert s["krw"] is not None, "숏 원화 손익이 여전히 비어 있다"


def test_short_without_bars_is_marked_unresolved_not_dropped():
    sigs = [{"ts": _ts(0), "kind": "SHORT_SIGNAL", "follower": "090430",
             "leader": "087010", "lag": 5, "price": 1000.0}]
    res = bt.resolve_signals(sigs, {"090430": {}})
    assert res["unresolved"] == 1
    assert res["signals"][0]["resolved"] is False
    assert "ret" not in res["signals"][0]


def test_short_rising_price_is_a_loss():
    sigs = [{"ts": _ts(0), "kind": "SHORT_SIGNAL", "follower": "090430",
             "leader": "087010", "lag": 1, "price": 1000.0}]
    s = bt.resolve_signals(sigs, {"090430": _bars([(1, 1050.0)])})["signals"][0]
    assert s["ret"] < 0 and s["krw"] < 0


def test_longs_and_shorts_resolve_in_one_pass():
    """롱과 숏이 같은 응답에 함께 담긴다 — 프론트가 두 번 렌더할 이유가 없어진다."""
    sigs = [
        {"ts": _ts(0), "kind": "BUY", "follower": "A", "leader": "L", "qty": 1, "price": 100.0},
        {"ts": _ts(1), "kind": "SHORT_SIGNAL", "follower": "B", "leader": "L", "lag": 1, "price": 200.0},
        {"ts": _ts(2), "kind": "SELL", "follower": "A", "leader": "L", "qty": 1, "price": 110.0, "pnl": 0.1},
    ]
    out = bt.resolve_signals(sigs, {"B": _bars([(2, 190.0)])})["signals"]
    resolved = [s for s in out if s.get("resolved")]
    assert {s["trade_type"] for s in resolved} == {"L", "S"}
    assert all(s.get("krw") is not None for s in resolved)


def test_fifo_pairing_when_two_buys_precede_two_sells():
    sigs = [
        {"ts": _ts(0), "kind": "BUY", "follower": "A", "leader": "L", "qty": 1, "price": 100.0},
        {"ts": _ts(1), "kind": "BUY", "follower": "A", "leader": "L", "qty": 1, "price": 200.0},
        {"ts": _ts(2), "kind": "SELL", "follower": "A", "leader": "L", "qty": 1, "price": 110.0},
        {"ts": _ts(3), "kind": "SELL", "follower": "A", "leader": "L", "qty": 1, "price": 190.0},
    ]
    out = [s for s in bt.resolve_signals(sigs, {})["signals"] if s["kind"] == "SELL"]
    assert out[0]["entry_price"] == 100.0 and out[0]["krw"] == 10.0     # 먼저 산 걸 먼저 판다
    assert out[1]["entry_price"] == 200.0 and out[1]["krw"] == -10.0


# --------------------------------------------------------------------------- #
# 지도 승률
# --------------------------------------------------------------------------- #

def test_pair_hit_rate_counts_follow_through_as_a_win():
    lag = 2
    jump = config.LEADER_JUMP
    # 1분: 선행주가 +jump 초과 급등. 그때 후행주는 평평(FOLLOWER_FLAT 이내).
    leader = _bars([(0, 100.0), (1, 100.0 * (1 + jump * 1.5))])
    follower = _bars([(0, 50.0), (1, 50.0), (1 + lag, 51.0)])   # lag 뒤 상승 = 맞춤

    r = bt.pair_hit_rate({"L": leader, "F": follower}, "L", "F", lag)
    assert r == {"wins": 1, "losses": 0, "n": 1, "win_rate": 1.0}


def test_pair_hit_rate_counts_reversal_as_a_loss():
    lag, jump = 2, config.LEADER_JUMP
    leader = _bars([(0, 100.0), (1, 100.0 * (1 + jump * 1.5))])
    follower = _bars([(0, 50.0), (1, 50.0), (1 + lag, 49.0)])   # lag 뒤 하락 = 틀림

    r = bt.pair_hit_rate({"L": leader, "F": follower}, "L", "F", lag)
    assert r["wins"] == 0 and r["losses"] == 1 and r["win_rate"] == 0.0


def test_pair_hit_rate_ignores_events_where_follower_already_moved():
    """엔진은 후행주가 이미 움직였으면 진입하지 않는다 — 승률도 그 케이스를 세면 안 된다."""
    lag, jump = 2, config.LEADER_JUMP
    leader = _bars([(0, 100.0), (1, 100.0 * (1 + jump * 1.5))])
    follower = _bars([(0, 50.0), (1, 50.0 * (1 + config.FOLLOWER_FLAT * 3)), (1 + lag, 60.0)])

    assert bt.pair_hit_rate({"L": leader, "F": follower}, "L", "F", lag)["n"] == 0


def test_pair_hit_rate_ignores_moves_below_the_jump_threshold():
    lag, jump = 2, config.LEADER_JUMP
    leader = _bars([(0, 100.0), (1, 100.0 * (1 + jump * 0.5))])   # 문턱 미달
    follower = _bars([(0, 50.0), (1, 50.0), (1 + lag, 55.0)])

    assert bt.pair_hit_rate({"L": leader, "F": follower}, "L", "F", lag)["n"] == 0


def test_pair_hit_rate_handles_short_side_direction():
    lag, jump = 1, config.LEADER_JUMP
    leader = _bars([(0, 100.0), (1, 100.0 * (1 - jump * 1.5))])    # 급락
    follower = _bars([(0, 50.0), (1, 50.0), (1 + lag, 49.0)])      # 따라 하락 = 맞춤

    assert bt.pair_hit_rate({"L": leader, "F": follower}, "L", "F", lag)["wins"] == 1


def test_pair_hit_rate_without_samples_returns_none():
    r = bt.pair_hit_rate({}, "L", "F", 1)
    assert r["n"] == 0 and r["win_rate"] is None


def test_returns_skip_non_adjacent_minutes():
    """장 마감 경계처럼 분봉이 끊긴 구간은 '1분 수익률' 이 아니다 — 가짜 급등으로 세면 안 된다."""
    jump = config.LEADER_JUMP
    leader = _bars([(0, 100.0), (30, 100.0 * (1 + jump * 5))])     # 30분 점프
    follower = _bars([(0, 50.0), (30, 50.0), (31, 60.0)])

    assert bt.pair_hit_rate({"L": leader, "F": follower}, "L", "F", 1)["n"] == 0

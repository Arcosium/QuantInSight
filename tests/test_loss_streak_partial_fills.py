"""사장 지시 2026-07-22: 연속 손절 회로차단이 **부분체결**에 오발동하던 버그.

실측(uid2): 148070 매도 1건이 호가에 잘려 13조각으로 체결됐는데 조각마다 손실이 찍혀
streak 18 이 됐다(합계 -72,098원 = 1억 계좌의 -0.07%). 임계값을 넘겨 신규 매수가 전면 차단.
가드의 취지는 '전략이 연속으로 깨진다'이지 '주문이 잘게 잘렸다'가 아니다.

불변식: 연속된 같은 종목 매도는 **한 거래**로 합산해 센다. 단, 사이에 그 종목 매수가 있으면
되산 뒤 다시 판 별개 거래이므로 묶음을 끊는다.
"""
import pytest

from infra import trade_ledger


@pytest.fixture
def led(tmp_path, monkeypatch):
    monkeypatch.setattr(trade_ledger, "_DATA_DIR", tmp_path)

    def _write(fills):
        trade_ledger.save(1, {"version": 1, "positions": {}, "fills": fills})
    return _write


def _sell(tkr, realized, ts="2026-07-22 10:00:00"):
    return {"ts": ts, "ticker": tkr, "side": "sell", "qty": 1, "realized": realized}


def _buy(tkr, ts="2026-07-22 09:00:00"):
    return {"ts": ts, "ticker": tkr, "side": "buy", "qty": 1}


def test_partial_fills_of_one_exit_count_as_one_trade(led):
    """실제 사고 재현 — 한 종목 13조각 부분체결은 손절 1회여야 한다(종전 13회)."""
    led([_sell("148070", -x) for x in (398, 3981, 4324, 2786, 1179, 17688, 5896,
                                       1915, 4599, 9731, 8779, 1890, 8428)])
    assert trade_ledger.recent_loss_streak(1) == 1


def test_distinct_tickers_still_count_separately(led):
    """서로 다른 종목의 연속 손절은 그대로 N회 — 가드 본래 기능은 살아 있어야 한다."""
    led([_sell("A", -100), _sell("B", -100), _sell("C", -100)])
    assert trade_ledger.recent_loss_streak(1) == 3


def test_rebuy_between_sells_breaks_the_group(led):
    """되산 뒤 다시 판 것은 별개 거래 — 같은 종목이어도 2회로 센다."""
    led([_sell("A", -100), _buy("A"), _sell("A", -100)])
    assert trade_ledger.recent_loss_streak(1) == 2


def test_profitable_group_stops_the_streak(led):
    """조각 합이 이익이면 거기서 중단 — 조각 중 일부가 손실이어도 마찬가지."""
    led([_sell("A", -100), _sell("B", +500), _sell("B", -100), _sell("C", -100)])
    # 역순: C(-100)=1회 → B 묶음(+400) → 이익이므로 중단
    assert trade_ledger.recent_loss_streak(1) == 1


def test_group_summing_to_loss_counts_once(led):
    """조각 합이 손실이면 1회로 세고 계속 거슬러 올라간다."""
    led([_sell("A", -100), _sell("B", +50), _sell("B", -300)])
    # 역순: B 묶음(-250)=1회 → A(-100)=2회
    assert trade_ledger.recent_loss_streak(1) == 2


def test_buy_of_other_ticker_does_not_break_group(led):
    """다른 종목 매수는 이 종목 매도 묶음과 무관하다."""
    led([_sell("A", -100), _buy("ZZZ"), _sell("A", -100)])
    assert trade_ledger.recent_loss_streak(1) == 1


def test_no_sells_is_zero(led):
    led([_buy("A"), _buy("B")])
    assert trade_ledger.recent_loss_streak(1) == 0


def test_missing_ledger_is_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(trade_ledger, "_DATA_DIR", tmp_path)
    assert trade_ledger.recent_loss_streak(999) == 0

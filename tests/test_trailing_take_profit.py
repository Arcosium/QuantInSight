"""트레일링 익절 (고회전 수익화·비대칭 청산, 2026-06-18).

보유 고점 대비 TRAILING_TAKE_PROFIT_PCT 만큼 되밀리면 매도 — 승자를 길게 가져가되 차익은 보호.
승자에만 적용(고점 평가손익 ≥ trailing). 손실 포지션 하락은 손절(STOP_LOSS_PCT)의 영역.
종목별 고점(peak_price·peak_pnl)은 peaks dict 에 영속(사이클 간 유지·재시작 내성).
"""
from main_swarm import _assemble_sell_orders


def _call(holdings, **kw):
    base = dict(enable_rebalance=True, take_profit_pct=12.0, stop_loss_pct=7.0,
                trim_over_ratio=False, conservative_ratio=0.15, per_stock_cap=0, total=1_000_000)
    base.update(kw)
    return _assemble_sell_orders(holdings, {}, **base)


def test_trailing_fires_on_retrace():
    peaks = {"AAA": {"peak_price": 112.0, "peak_pnl": 12.0}}
    h = [{"code": "AAA", "qty": 10, "pnl_pct": 6.0, "cur_price": 106.0, "name": "AAA"}]
    orders, _ = _call(h, trailing_pct=5.0, peaks=peaks)
    assert len(orders) == 1
    assert orders[0]["side"] == "sell" and orders[0]["qty"] == 10
    assert "트레일링" in orders[0]["reason"]


def test_trailing_no_fire_small_retrace():
    peaks = {"AAA": {"peak_price": 112.0, "peak_pnl": 12.0}}
    # 현재 +10%·110원: 되밀림 (112-110)/112 = 1.8% < 5% → 매도 안 함
    h = [{"code": "AAA", "qty": 10, "pnl_pct": 10.0, "cur_price": 110.0, "name": "AAA"}]
    orders, _ = _call(h, trailing_pct=5.0, peaks=peaks)
    assert orders == []


def test_trailing_only_winners_not_losers():
    # 한 번도 이익 안 난 포지션(peak_pnl 0)·현재 손실 → 트레일링 발동 금지(손절의 영역)
    peaks = {"AAA": {"peak_price": 100.0, "peak_pnl": 0.0}}
    h = [{"code": "AAA", "qty": 10, "pnl_pct": -3.0, "cur_price": 97.0, "name": "AAA"}]
    orders, _ = _call(h, trailing_pct=5.0, peaks=peaks)
    assert orders == []


def test_trailing_off_by_default():
    peaks = {}
    h = [{"code": "AAA", "qty": 10, "pnl_pct": 6.0, "cur_price": 106.0, "name": "AAA"}]
    orders, _ = _call(h, trailing_pct=0.0, peaks=peaks)
    assert orders == []  # trailing_pct=0 → off


def test_peak_updates_from_current_holdings():
    peaks = {}
    h = [{"code": "AAA", "qty": 10, "pnl_pct": 8.0, "cur_price": 108.0, "name": "AAA"}]
    _call(h, trailing_pct=5.0, peaks=peaks)
    assert peaks["AAA"]["peak_pnl"] == 8.0
    assert peaks["AAA"]["peak_price"] == 108.0


def test_take_profit_still_takes_priority():
    # 고정 익절(≥12%)이 트레일링보다 우선 — 둘 다 충족이면 익절 사유로 매도
    peaks = {"AAA": {"peak_price": 115.0, "peak_pnl": 15.0}}
    h = [{"code": "AAA", "qty": 10, "pnl_pct": 13.0, "cur_price": 113.0, "name": "AAA"}]
    orders, _ = _call(h, trailing_pct=5.0, peaks=peaks)
    assert len(orders) == 1
    assert "익절" in orders[0]["reason"]

"""부분체결 수량 정산 — 주문수량이 아닌 '실제 잔고 증분'을 기록 (2026-06-15).

버그: after_qty>before_qty면(1주만 늘어도) filled=True + 원장/로그에 주문수량을 기록 →
부분체결이 전량체결로 둔갑(012510 주문84·실제33). 수정: 실제 증분만 기록, 잔여는 폴링.
"""
from main_swarm import _settle_fill_qty


def test_buy_full_fill():
    filled, remaining = _settle_fill_qty("buy", before_qty=0, after_qty=314, order_qty=314)
    assert filled == 314 and remaining == 0


def test_buy_partial_fill():
    # 84주 주문, 잔고 0→33 → 33 체결, 51 잔여
    filled, remaining = _settle_fill_qty("buy", before_qty=0, after_qty=33, order_qty=84)
    assert filled == 33 and remaining == 51


def test_sell_partial_fill():
    # 40주 매도, 잔고 40→10 → 30 체결, 10 잔여
    filled, remaining = _settle_fill_qty("sell", before_qty=40, after_qty=10, order_qty=40)
    assert filled == 30 and remaining == 10


def test_no_fill_returns_zero():
    filled, remaining = _settle_fill_qty("buy", before_qty=5, after_qty=5, order_qty=10)
    assert filled == 0 and remaining == 10


def test_delta_capped_at_order_qty():
    # 잔고 증분이 주문량 초과(타 매수 합산 등) → 주문량으로 캡(과대기록 방지)
    filled, remaining = _settle_fill_qty("buy", before_qty=0, after_qty=500, order_qty=314)
    assert filled == 314 and remaining == 0

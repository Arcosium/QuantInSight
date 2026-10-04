"""매수수량을 KIS 권위 매수가능수량으로 클램프 — 사장 지시 2026-06-16.

배경: 리스크 승인은 bp.cash 기반인데 KIS 실제 주문가능금액(ord_psbl_cash·증거금·D+2 미결제
반영)은 더 작을 수 있어, 다종목 순차 매수의 마지막 주문이 '주문가능금액을 초과 했습니다'로
거부된다(uid1 cycle380 241710 7주). 집행 직전 KIS 매수가능수량으로 클램프하면 전량 실패
대신 가능한 만큼 체결되고, 0 이면 보류(조용한 드롭 아님 — 사유 발화 후 다음 사이클 재시도).
"""
from main_swarm import _clamp_qty_to_buyable


def test_clamp_reduces_to_buyable():
    # 241710 사례: 7주 주문, KIS 매수가능 3주 → 3주
    assert _clamp_qty_to_buyable(7, 3) == 3


def test_no_clamp_when_enough():
    assert _clamp_qty_to_buyable(2, 10) == 2


def test_none_buyable_keeps_original():
    # 조회 실패(None) → 원래 수량 유지(폴백 — 조회 실패로 매수를 막지 않음)
    assert _clamp_qty_to_buyable(5, None) == 5


def test_zero_buyable_returns_zero():
    # 매수가능 0 → 0 (호출부가 보류 + 사유 발화)
    assert _clamp_qty_to_buyable(7, 0) == 0


def test_negative_buyable_floored_to_zero():
    assert _clamp_qty_to_buyable(7, -1) == 0

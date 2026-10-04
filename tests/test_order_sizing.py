"""`main_swarm._affordable_one_share` — 최소 1주 매수 허용 판정.

기본 정책(보수형 B): 리스크관리실장 예수금 게이트와 동일하게
`price × MIN_CASH_BUFFER ≤ cash` 일 때만 1주를 살 수 있다고 본다
(주문 초안이 곧바로 리스크 단에서 반려되는 모순 방지).
fixture 가 MIN_CASH_BUFFER=1.10 으로 고정한다.
"""
import pytest

import main_swarm as ms


@pytest.mark.parametrize("price,cash", [(0, 1000), (-1, 1000), (100, 0), (100, -5)])
def test_non_positive_inputs_are_unaffordable(price, cash):
    assert ms._affordable_one_share(price, cash, total=cash) is False


def test_affordable_when_price_times_buffer_within_cash():
    # 100 × 1.10 = 110 ≤ 200  → 매수 가능
    assert ms._affordable_one_share(100.0, 200.0, total=1000.0) is True


def test_not_affordable_inside_buffer_band():
    # price ≤ cash 이지만 price × 1.10 > cash → 보수적으로 거른다 (KIS 거절 예방)
    assert ms._affordable_one_share(100.0, 105.0, total=1000.0) is False


def test_boundary_is_floating_point_conservative():
    # 100 × 1.10 == 110.00000000000001 (IEEE754) 이므로 cash 가 정확히 110 이면
    # 미세하게 거른다 — 경계에서 보수적으로 동작함을 고정한다.
    assert ms._affordable_one_share(100.0, 110.0, total=1000.0) is False
    # cash 에 아주 작은 여유만 있어도 통과.
    assert ms._affordable_one_share(100.0, 110.01, total=1000.0) is True


# ── 사장 결정 2026-06-16: 고가주 1주가 '사이클 매수예산'을 초과하면 매수하지 않는다 ──
# 배경: `_affordable_one_share` 가 예수금만 보고 사이클예산을 무시해, 고가주(AMD $551 등)를
# 사이징은 1주 허용하나 리스크 가드레일(MAX_CYCLE_BUDGET_RATIO)이 예산 초과로 반려 →
# 매 사이클 골랐다 반려되는 데드존 + final_report '예수금 이내 → 매수 가능' 모순 메시지.
# cycle_remaining 을 주면 그 안일 때만 허용해 사이징과 가드레일을 일치시킨다.

def test_rejects_one_share_exceeding_cycle_budget():
    # AMD 사례(원화 환산): 1주 839,147원 · 예수금 2,083,243원(이내) · 사이클 잔여예산 729,135원(초과)
    assert ms._affordable_one_share(
        839_147, 2_083_243, total=6_020_649, cycle_remaining=729_135) is False


def test_allows_one_share_within_cycle_budget():
    # 1주가 예수금·사이클 잔여예산 모두 이내면 허용
    assert ms._affordable_one_share(
        500_000, 2_083_243, total=6_020_649, cycle_remaining=729_135) is True


def test_cycle_budget_omitted_keeps_legacy_behavior():
    # cycle_remaining 미전달/None 이면 사이클예산 체크 생략 — 기존(예수금 전용) 동작 보존(하위호환)
    assert ms._affordable_one_share(100.0, 200.0, total=1000.0) is True
    assert ms._affordable_one_share(100.0, 200.0, total=1000.0, cycle_remaining=None) is True


def test_cycle_budget_does_not_override_cash_gate():
    # 사이클 잔여예산이 충분해도 예수금 게이트(price×buffer ≤ cash)는 그대로 — 예수금 부족이면 거른다
    assert ms._affordable_one_share(
        100.0, 105.0, total=1000.0, cycle_remaining=10_000) is False

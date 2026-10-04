"""매크로 목표 향한 예산 플로어 (2026-06-15, 사장 선택).

문제: 매크로가 주식 비중을 공격적으로 올려도(55→65%) ops 가 토요일에 깎은 per-order(5%)·
per-cycle(20%) 예산 컷이 배치를 묶어, 한 사이클에 9%만 매수되고 목표 수렴이 느리다.
수정: 매크로 목표 > 현재 주식비중(여력 있음)이면 예산 비율에 최소 플로어를 적용해 ops 컷이
배치를 과도하게 묶지 못하게 한다. 여력 없으면 원값 유지(방어 의도 보존).
"""
from main_swarm import apply_macro_deploy_floor

P = {"MACRO_DEPLOY_FLOOR_ENABLED": True,
     "PER_ORDER_BUDGET_FLOOR_RATIO": 0.10,
     "MAX_CYCLE_BUDGET_FLOOR_RATIO": 0.30}


def test_floors_applied_when_room_toward_target():
    po, cyc = apply_macro_deploy_floor(0.05, 0.20, macro_target_pct=0.65,
                                       current_stock_ratio=0.09, params=P)
    assert po == 0.10
    assert cyc == 0.30


def test_unchanged_when_no_room():
    po, cyc = apply_macro_deploy_floor(0.05, 0.20, macro_target_pct=0.10,
                                       current_stock_ratio=0.30, params=P)
    assert (po, cyc) == (0.05, 0.20)


def test_unchanged_when_room_below_gap():
    # 목표 0.12 vs 현재 0.10 → 여력 2%p (<5%p 트리거) → 컷 유지(방어)
    po, cyc = apply_macro_deploy_floor(0.05, 0.20, macro_target_pct=0.12,
                                       current_stock_ratio=0.10, params=P)
    assert (po, cyc) == (0.05, 0.20)


def test_never_lowers_existing_higher_ratios():
    po, cyc = apply_macro_deploy_floor(0.50, 0.50, macro_target_pct=0.65,
                                       current_stock_ratio=0.09, params=P)
    assert (po, cyc) == (0.50, 0.50)


def test_disabled_is_noop():
    po, cyc = apply_macro_deploy_floor(0.05, 0.20, macro_target_pct=0.65,
                                       current_stock_ratio=0.09,
                                       params={"MACRO_DEPLOY_FLOOR_ENABLED": False})
    assert (po, cyc) == (0.05, 0.20)


def test_unknown_macro_is_noop():
    po, cyc = apply_macro_deploy_floor(0.05, 0.20, macro_target_pct=None,
                                       current_stock_ratio=0.09, params=P)
    assert (po, cyc) == (0.05, 0.20)

"""1주 매수 제외 사유 메시지 정확화 — 사장 지시 2026-06-17(11시 US 사이클 점검).

배경: cyc430 final_report 가 "UAL ... 예수금 $0.00/사이클 잔여예산 $175.29 한도 초과 → 제외"
로 떴는데, UAL $119.46 < 잔여예산 $175.29 라 '예산 한도 초과'는 사실과 다르다(진짜 binding
은 예수금 $0.00). binding 한도를 정확히 표기한다.
"""
from main_swarm import _one_share_exclude_reason


def test_cash_is_binding():
    # 1주 가격 > 예수금 → 예수금 부족
    assert _one_share_exclude_reason(price=119.46, cash=0.0, cyc_rem=175.29) == "예수금 부족"


def test_cycle_budget_is_binding():
    # 예수금은 충분한데 사이클 잔여예산이 1주 가격 미만 → 사이클 예산 초과
    assert _one_share_exclude_reason(price=119.46, cash=5000.0, cyc_rem=50.0) == "사이클 예산 초과"


def test_both_short_reports_cash_first():
    # 둘 다 모자라면 예수금을 우선 사유로(자금이 근본)
    assert _one_share_exclude_reason(price=119.46, cash=10.0, cyc_rem=10.0) == "예수금 부족"

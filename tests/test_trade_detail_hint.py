"""매도 거래내역 '인용 평가손익'(pnl_pct_hint)이 권위 실현손익과 부호가 반대면 숨긴다 —
사장 지시 2026-06-16.

배경: 재매수가 미결제(D+2)인 동안 KIS 평단이 일시 옛값으로 잡혀, 매도 시점 인용 평가손익이
실제와 반대 부호로 기록된다(uid1 375500: 인용 +12.9% vs 권위 실현 -2.7%/-12,500원). 거래내역에
이 인용이 그대로 보이면 손실을 이익으로 오해한다. 권위 실현손익과 부호가 충돌하면 인용을
None 으로 만들어 표시에서 빼고 권위값만 남긴다.
"""
from main_swarm import _hint_conflicts_authority


def test_conflict_opposite_signs():
    # uid1 375500: 인용 +12.9% vs 권위 실현 -2.7% → 충돌(오해 유발)
    assert _hint_conflicts_authority(12.9, -2.7) is True
    assert _hint_conflicts_authority(-2.7, 12.9) is True


def test_no_conflict_same_sign():
    assert _hint_conflicts_authority(5.0, 3.0) is False
    assert _hint_conflicts_authority(-2.0, -8.0) is False


def test_none_no_conflict():
    # 둘 중 하나라도 미상이면 판단 보류(인용 유지)
    assert _hint_conflicts_authority(None, -2.7) is False
    assert _hint_conflicts_authority(12.9, None) is False


def test_zero_no_conflict():
    # 0% 는 부호 판정 대상이 아니다(충돌 아님)
    assert _hint_conflicts_authority(0, -2.7) is False
    assert _hint_conflicts_authority(12.9, 0) is False

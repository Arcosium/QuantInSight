"""KR 매도가능수량 0 잠금 해소 — 사장 지시 2026-06-16.

배경: 지정가 매도가 시세 위라 미체결되면 그 주문이 보유물량을 잠가(ord_psbl_qty=0) 다음 사이클
매도가 '매도가능 0'으로 보류된다. 기존 가드는 그냥 보류+continue 라 kr_sell(펜딩 취소 후 신규
전송)을 영영 호출하지 않아 영원히 잠긴다(375500 DL이앤씨 +11.3% 익절이 종일 미체결로 막힘).

사장 요구: "매도 체결이 안된 걸 알았으면 다시 매도가격을 정하던가, 시도하지 말던가."
→ 미체결 펜딩 매도가 실제로 있으면 '펜딩 취소 후 시장가 재청산'(재가격+재시도),
   펜딩이 없으면(결제/글리치 잠금) '보류'(무리한 시장가 금지, 정직).
"""
from main_swarm import _locked_sell_action


def test_sellable_unknown_proceeds():
    # 매도가능 조회 실패(None) → 주문 막지 않음(기존 폴백)
    assert _locked_sell_action(None, 63, 63, False) == ("proceed", 63)


def test_no_holdings_proceeds():
    assert _locked_sell_action(0, 0, 10, True) == ("proceed", 10)


def test_sellable_covers_qty_proceeds():
    assert _locked_sell_action(63, 63, 63, False) == ("proceed", 63)


def test_sellable_partial_clamps():
    # 일부만 매도가능 → 그만큼으로 클램프
    assert _locked_sell_action(20, 63, 63, False) == ("clamp", 20)


def test_locked_with_pending_reprices_to_market():
    # 매도가능 0 + 미체결 펜딩 매도 존재 → 펜딩 취소 후 보유 전량 시장가 재청산
    assert _locked_sell_action(0, 63, 63, True) == ("reprice_market", 63)


def test_locked_without_pending_holds():
    # 매도가능 0 + 펜딩 없음 → 결제/글리치 잠금 추정 → 보류(무리한 시장가 금지)
    assert _locked_sell_action(0, 63, 63, False) == ("hold", None)


# ── 버그 C(2026-06-18): 잠긴 손절 매도 무한루프 → N사이클 후 에스컬레이션 ──

def test_locked_no_pending_holds_initially():
    # 잠김 초반(스트릭 < 임계)엔 결제 해소 기회를 주며 보류
    assert _locked_sell_action(0, 100, 100, False, locked_streak=0, escalate_after=3) == ("hold", None)
    assert _locked_sell_action(0, 100, 100, False, locked_streak=2, escalate_after=3) == ("hold", None)


def test_locked_no_pending_escalates_after_streak():
    # 임계 도달(N사이클 연속 잠김) → 강제 시장가 재청산 시도(무한 보류 차단·정직 표면화)
    assert _locked_sell_action(0, 100, 100, False, locked_streak=3, escalate_after=3) == ("escalate_market", 100)


def test_pending_still_reprices_regardless_of_streak():
    # 펜딩 존재면 스트릭 무관 항상 reprice(기존 동작 우선)
    assert _locked_sell_action(0, 100, 100, True, locked_streak=9, escalate_after=3) == ("reprice_market", 100)

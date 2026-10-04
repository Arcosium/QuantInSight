"""매수 진입 지정가 시장성 가드 (사장 지시 2026-06-12).

LLM(주식운용실장/계량분석팀장)이 시장가보다 낮은 매수 진입가를 지정하면 KIS limit 매수가
접수만 되고 체결되지 않아 매수가 조용히 증발한다('주문 절대 스킵 금지' 위반).
uid1 cyc312 RKLB: 진입가 105 vs 시장가 112.83 → accepted/filled=False 사례.
buy_limit_below_market 가 이 상황을 감지하면 호출부가 시장가로 전환한다.
"""
from main_swarm import buy_limit_below_market


def test_buy_limit_below_market_detected():
    # RKLB 실제 사례: 지정가 105 < 시장가 112.83 → 미체결 위험.
    assert buy_limit_below_market("buy", "limit", 105.0, 112.83) is True


def test_buy_limit_at_or_above_market_ok():
    # 지정가가 시장가 이상이면 시장성 지정가 → 즉시 체결, 손대지 않음.
    assert buy_limit_below_market("buy", "limit", 120.0, 112.83) is False
    assert buy_limit_below_market("buy", "limit", 112.83, 112.83) is False


def test_market_order_unaffected():
    assert buy_limit_below_market("buy", "market", None, 112.0) is False


def test_sell_unaffected():
    # 매도 지정가는 이 가드 대상이 아니다(시장가/지정가 매도는 별도 의도).
    assert buy_limit_below_market("sell", "limit", 105.0, 112.0) is False


def test_no_market_price_does_not_touch():
    # 시장가를 모르면(0/None) 함부로 전환하지 않는다(보수적).
    assert buy_limit_below_market("buy", "limit", 105.0, 0.0) is False
    assert buy_limit_below_market("buy", "limit", 105.0, None) is False


def test_no_limit_price_safe():
    assert buy_limit_below_market("buy", "limit", None, 112.0) is False
    assert buy_limit_below_market("buy", "limit", 0.0, 112.0) is False

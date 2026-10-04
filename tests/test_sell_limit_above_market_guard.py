"""매도 지정가가 시장가보다 위면(=미체결·물량잠김) 긴급청산 한정 시장가 전환 — 2026-06-22.

배경(월 06-22 모의 uid2 316140 진단): 사후관리실장이 '전량 손절'(-5.4%)을 시장가 30,000원
위인 지정가 32,500원으로 제시 → KIS limit 매도가 접수만 되고 영원히 미체결 → 보유물량이 잠겨
(sellable 0) 매 사이클 더 잠긴다. 매수 경로엔 buy_limit_below_market(시장가보다 낮은 매수 →
시장가 전환) 가드가 있는데 매도 경로엔 대칭 가드가 없던 buy/sell 비대칭 잠복 버그.

사장 정책 2026-06-22: '긴급청산(전량/손절/트레일링/편중)'만 시장가 전환. '익절 목표가'(시장가
위에서 더 받으려 대기하는 의도된 지정가)는 그대로 유지한다.
"""
from main_swarm import sell_limit_above_market, _is_urgent_liquidation


# ── sell_limit_above_market: 순수 기하(매수 가드 buy_limit_below_market 의 대칭) ──
def test_sell_limit_above_market_true():
    # 매도 지정가 32,500 > 시장가 30,000 → 미체결 위험 표면화
    assert sell_limit_above_market("sell", "limit", 32500, 30000) is True


def test_sell_limit_at_or_below_market_false():
    # 시장가 이하 매도 지정가는 즉시 체결 → 손대지 않음
    assert sell_limit_above_market("sell", "limit", 29000, 30000) is False
    assert sell_limit_above_market("sell", "limit", 30000, 30000) is False


def test_buy_side_untouched():
    assert sell_limit_above_market("buy", "limit", 32500, 30000) is False


def test_market_mode_untouched():
    assert sell_limit_above_market("sell", "market", 32500, 30000) is False


def test_no_market_price_conservative_false():
    # 시세 미확보(0/None)면 판단 보류 — 주문 손대지 않음(보수)
    assert sell_limit_above_market("sell", "limit", 32500, 0) is False
    assert sell_limit_above_market("sell", "limit", 32500, None) is False


# ── _is_urgent_liquidation: 긴급청산 지시 판별(reason 문자열 기반) ──
def test_urgent_full_liquidation():
    assert _is_urgent_liquidation("시프트업 사후관리실장 매도 판단 — 전량 (보유 100주, 평가손익 -5.2%)") is True


def test_urgent_stop_loss():
    assert _is_urgent_liquidation("자동 손절 — 평가손익 -5.3% ≤ -5%") is True


def test_urgent_trailing():
    assert _is_urgent_liquidation("트레일링 익절 — 고점 +8.0% → 현재 +3.1% (고점가 대비 -4.5% 되밀림)") is True


def test_urgent_concentration_trim():
    assert _is_urgent_liquidation("편중 축소 — 비중 24.1% > 20% 한도") is True


def test_not_urgent_half():
    assert _is_urgent_liquidation("사후관리실장 매도 판단 — 절반 (보유 4주, 평가손익 +6.1%)") is False


def test_not_urgent_auto_take_profit():
    # 자동 익절(목표 도달)은 의도된 지정가 목표 → 긴급 아님(지정가 유지)
    assert _is_urgent_liquidation("자동 익절 — 평가손익 +9.2% ≥ +8%") is False


def test_not_urgent_numeric_partial():
    assert _is_urgent_liquidation("사후관리실장 매도 판단 — 3주") is False

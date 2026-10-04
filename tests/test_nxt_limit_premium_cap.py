"""NXT 시간외 지정가 — 정규 전일종가 기준 프리미엄 상한 캡 (2026-06-15).

버그: NXT 프리마켓 매수가 'NXT시세 × (1+슬리피지)'라 얇은 NXT 프리미엄을 추종 →
003490 을 정규 종가 26,600 대비 +4.9% 인 27,900 에 체결, 보유는 정규가로 마킹돼 즉시
-4.65% + 토요일 SL 3% 와 겹쳐 조기 손절. 수정: ref_price(정규 전일종가) 대비 매수 상한·
매도 하한을 max_premium_pct 로 캡. ref_price 없으면 기존 동작 유지(하위호환).
"""
from infra.kis_broker import compute_nxt_limit_price, round_to_tick


def test_backward_compatible_without_ref():
    """ref_price 미지정 시 기존 동작(NXT시세 × (1+슬리피지))."""
    assert compute_nxt_limit_price(27776, side="buy", slippage_pct=0.5) == round_to_tick(27776 * 1.005)


def test_buy_capped_at_ref_premium():
    """NXT가 정규 종가보다 크게 프리미엄(+4.4%)일 때 매수 지정가를 ref×(1+캡)으로 제한."""
    capped = compute_nxt_limit_price(27776, side="buy", slippage_pct=0.5,
                                     ref_price=26600, max_premium_pct=1.5)
    assert capped == round_to_tick(26600 * 1.015)
    assert capped < round_to_tick(27776 * 1.005), "프리미엄 추종을 막아야 한다"


def test_buy_not_capped_when_near_ref():
    """NXT가 정규 종가 근처면 캡이 안 걸리고 기존 슬리피지 밴드 사용."""
    px = compute_nxt_limit_price(26700, side="buy", slippage_pct=0.5,
                                 ref_price=26600, max_premium_pct=1.5)
    assert px == round_to_tick(26700 * 1.005)


def test_sell_floored_at_ref_discount():
    """매도는 ref×(1−캡) 아래로 못 내려가게(과소 매도 방지)."""
    px = compute_nxt_limit_price(25000, side="sell", slippage_pct=0.5,
                                 ref_price=26600, max_premium_pct=1.5)
    assert px == round_to_tick(26600 * 0.985)

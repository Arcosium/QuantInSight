"""US 매수 사이징의 USD 현금 기준 — 사장 지시 2026-06-17.

버그(uid1 hh09080 06/16 야간): US 매수 사이징이 USD 매수가능액을 `KRW예수금 ÷ 환율`로
합성했다. 이는 '국내 현금 전액이 통합증거금으로 US 매수에 쓰인다'는 과대 가정이라, 실제
USD 예수금이 0/마이너스인 계좌(cash_usd=-$1,282)에 매 사이클 phantom 1주 초안을 만들고
KIS 가 '주문가능금액을 초과 했습니다'로 거부했다.

교정: KIS 실제 USD 주문가능액(us_buying_power.usd, ok=True)을 우선 쓰고, 조회 실패(mock·
에러)시에만 KRW÷환율 합성으로 폴백한다(주문 드롭 금지 원칙 — 폴백 시엔 클램프가 최종 방어).
"""
from main_swarm import _us_buy_usd_basis


def test_uses_real_usd_when_available():
    # KIS 가 USD 주문가능 $1,500 보고 → 그 값을 그대로 사용(KRW 합성 무시)
    assert _us_buy_usd_basis(True, 1500.0, krw_cash=2_635_671.0, krw_per_usd=1500.0) == 1500.0


def test_zero_real_usd_blocks_draft():
    # USD 가용 0 → 0 (phantom 매수 차단). KRW 가 많아도 합성 안 함.
    assert _us_buy_usd_basis(True, 0.0, krw_cash=2_635_671.0, krw_per_usd=1500.0) == 0.0


def test_negative_real_usd_floored_to_zero():
    # USD 예수금 마이너스 계좌(uid1 실제) → 0
    assert _us_buy_usd_basis(True, -50.0, krw_cash=3_000_000.0, krw_per_usd=1500.0) == 0.0


def test_query_failed_falls_back_to_krw_synthesis():
    # 조회 실패(ok=False, 모의·에러) → KRW÷환율 폴백(기존 동작 보존, 클램프가 최종 방어)
    got = _us_buy_usd_basis(False, 0.0, krw_cash=1_500_000.0, krw_per_usd=1500.0)
    assert abs(got - 1000.0) < 1e-6

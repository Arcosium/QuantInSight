"""KR 호가단위 반올림 + NXT 시간외 지정가(밴드) 산정 — 순수 함수."""
from infra.kis_broker import kr_tick_size, round_to_tick, ceil_to_tick, compute_nxt_limit_price

def test_tick_size_bands():
    assert kr_tick_size(1500)   == 1
    assert kr_tick_size(3000)   == 5
    assert kr_tick_size(12000)  == 10
    assert kr_tick_size(45000)  == 50
    assert kr_tick_size(150000) == 100
    assert kr_tick_size(300000) == 500
    assert kr_tick_size(800000) == 1000

def test_round_to_tick_snaps_to_valid():
    assert round_to_tick(12345) == 12340     # 호가단위 10 → 가장 가까운 유효호가
    assert round_to_tick(12346) == 12350
    assert round_to_tick(45070) == 45050      # 호가단위 50


def test_ceil_to_tick_always_aligns_upward():
    assert ceil_to_tick(27196) == 27200       # 2만~5만원: 50원 단위
    assert ceil_to_tick(12341) == 12350       # 5천~2만원: 10원 단위
    assert ceil_to_tick(70000) == 70000       # 이미 유효하면 불변
    assert ceil_to_tick(0) == 0               # 시장가 표식 보존


def test_ceil_to_tick_handles_price_band_boundaries():
    assert ceil_to_tick(4999) == 5000
    assert ceil_to_tick(19999) == 20000
    assert ceil_to_tick(499999) == 500000

def test_buy_limit_adds_band_and_snaps():
    # last=10000 → 호가단위 10(5천~2만). +0.5% = 10050 → 단위 10에 이미 정합
    px = compute_nxt_limit_price(10000, side="buy", slippage_pct=0.5)
    assert px == 10050        # 10000×1.005=10050
    assert isinstance(px, int)

def test_sell_limit_subtracts_band():
    px = compute_nxt_limit_price(10000, side="sell", slippage_pct=0.5)
    assert px == 9950          # 10000×0.995=9950

def test_zero_last_returns_zero():
    assert compute_nxt_limit_price(0, side="buy", slippage_pct=0.5) == 0

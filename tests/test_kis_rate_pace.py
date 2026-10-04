"""KIS 호출 간격 적응적 자가완화 — 사장 지시 2026-06-17(11시 US 사이클 점검).

배경: US 세션 내내 '초당 거래건수 초과(EGW00201)' 거부가 수십 회 폭주했다. 재전송으로
자가복구는 되나, 고정 _min_interval(과거 실전 0.06s≈15TPS)이 KIS 실측 한도를 넘는 버스트
구간에서 매 호출이 거부→재전송→로그폭주를 반복한다.
교정: rate-limit 거부를 만나면 _min_interval 을 곱셈 상향(상한까지)해 버스트 동안 스스로
간격을 벌리고, 거부 없이 호출이 흐르면 base 로 점감 복귀한다(평소 속도 유지).
"""
import asyncio
import time

from infra.kis_broker import KISBroker


def _broker():
    return KISBroker({"kis_app_key": "k", "kis_app_secret": "s",
                      "kis_account_no": "1234567801",
                      "kis_base_url": "https://openapi.koreainvestment.com:9443"})


def test_rate_limit_bumps_interval_capped():
    """거부를 거듭 만나면 간격이 상향되되 _RATE_MAX_INTERVAL 을 넘지 않는다."""
    b = _broker()
    base = b._rate_base
    b._note_rate_limited()
    assert b._min_interval > base
    for _ in range(50):
        b._note_rate_limited()
    assert b._min_interval <= b._RATE_MAX_INTERVAL + 1e-9


def test_interval_decays_back_to_base():
    """거부가 그치면(_decay 가 반복되면) 간격이 base 로 복귀한다(아래로 안 뚫음)."""
    b = _broker()
    base = b._rate_base
    for _ in range(5):
        b._note_rate_limited()
    assert b._min_interval > base
    for _ in range(200):
        b._decay_interval()
    assert abs(b._min_interval - base) < 1e-9


def test_decay_noop_when_at_base():
    """base 상태에서 점감은 무동작(과도한 가속 방지)."""
    b = _broker()
    base = b._rate_base
    b._decay_interval()
    assert b._min_interval == base


def test_mock_rate_limit_never_shortens_conservative_one_second_base():
    b = KISBroker({"kis_app_key": "mk", "kis_app_secret": "s",
                   "kis_account_no": "1234567801",
                   "kis_base_url": "https://openapivts.koreainvestment.com:29443"})
    assert b._rate_base == 1.0
    b._note_rate_limited()
    assert b._min_interval > 1.0, "거부 후에는 1초 기본 간격보다 더 느려져야 한다"


def test_same_appkey_brokers_share_one_pacer():
    async def run():
        b1, b2 = _broker(), _broker()
        b1._min_interval = b2._min_interval = 0.04
        started = time.monotonic()
        await asyncio.gather(b1._pace(), b2._pace(), b1._pace())
        return time.monotonic() - started

    elapsed = asyncio.run(run())
    assert elapsed >= 0.07, f"인스턴스가 달라도 3호출이 공유 직렬화돼야 함 ({elapsed:.3f}s)"


def test_us_quote_direct_path_also_uses_pacer():
    class Resp:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def json(self): return {"rt_cd": "0", "output": {"last": "123.45"}}

    class Session:
        def get(self, *args, **kwargs): return Resp()

    b = _broker()
    b._us_excd_cache = {}
    calls = {"pace": 0}
    async def token(force=False): return "tok"
    async def session(): return Session()
    async def pace(): calls["pace"] += 1
    b.token, b._s, b._pace = token, session, pace

    out = asyncio.run(b._us_price_raw("AAPL"))
    assert out["last"] == "123.45"
    assert calls["pace"] == 1

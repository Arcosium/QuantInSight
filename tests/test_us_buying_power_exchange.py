"""us_buying_power(TTTS3007R)가 자동판별 거래소를 사용 — 사장 지시 2026-06-17.

배경(uid1 hh09080 06/16 야간 US 세션 디버깅): 매수 직전 KIS 권위 클램프
(_clamp_orders_to_psbl·실행부)가 us_buying_power(code, price, None) 로 호출 →
OVRS_EXCG_CD 가 'NASD' 로 고정된다. WHR/KEY/NEE/DHI/PG/JNJ/XOM 등 NYSE 종목을
나스닥으로 조회하면 KIS 가 '상품이 없습니다'(rt_cd≠0)로 거부 → ok=False →
'조회 실패 시 주문 드롭 금지' 원칙에 따라 클램프가 스킵 → 부풀린 수량이 그대로
KIS 로 전송되어 '주문가능금액을 초과 했습니다'로 거부됐다(라이브 조회로 확정:
WHR NASD=ok=False '상품이 없습니다', NYS/NYSE=ok=True qty=0).

실제 주문(_overseas_order_body)은 _us_excd_cache(시세 프로브가 자동판별한 NYSE)를
쓰는데 클램프만 NASD 라 비대칭. 교정: us_buying_power 가 excg 미지정 시
_us_excd_cache(→excd_to_excg)의 자동판별 거래소를 쓰고, 명시값도 정규화한다.
"""
import asyncio
from infra.kis_broker import KISBroker


def _broker():
    return KISBroker({"kis_app_key": "k", "kis_app_secret": "s",
                      "kis_account_no": "1234567801",
                      "kis_base_url": "https://openapi.koreainvestment.com:9443"})


def _ok_resp(params, captured):
    captured.update(params)

    async def _f(path, tr_id, p):
        captured.clear(); captured.update(p)
        return {"rt_cd": "0", "output": {"ord_psbl_frcr_amt": "0",
                                         "max_ord_psbl_qty": "0", "exrt": "1300"}}
    return _f


def test_buying_power_uses_cached_exchange_not_nasd():
    """캐시에 NYSE 종목이면 OVRS_EXCG_CD=NYSE 로 조회(NASD 고정 금지)."""
    b = _broker()
    b._us_excd_cache["WHR"] = "NYS"   # 시세 프로브 자동판별
    captured = {}
    b._get_json = _ok_resp({}, captured)
    r = asyncio.run(b.us_buying_power("WHR", 40.0, None))
    assert captured.get("OVRS_EXCG_CD") == "NYSE"
    assert r["ok"] is True


def test_buying_power_explicit_excd_normalized():
    """NAS/NYS/AMS(excd 포맷)로 줘도 주문코드(excg)로 정규화한다."""
    b = _broker()
    captured = {}
    b._get_json = _ok_resp({}, captured)
    asyncio.run(b.us_buying_power("AAPL", 300.0, "NAS"))
    assert captured.get("OVRS_EXCG_CD") == "NASD"


def test_buying_power_no_cache_probes_then_falls_back_nasd():
    """캐시도 없고 프로브도 거래소를 못 채우면 NASD 안전폴백(기존 동작 보존)."""
    b = _broker()

    async def _no_probe(tk):
        return 0.0   # 프로브 실패 → 캐시 안 채움
    b.us_last_price = _no_probe
    captured = {}
    b._get_json = _ok_resp({}, captured)
    asyncio.run(b.us_buying_power("ZZZZ", 1.0, None))
    assert captured.get("OVRS_EXCG_CD") == "NASD"


def test_buying_power_probes_when_cache_miss():
    """캐시 미스면 us_last_price 1회 프로브로 거래소를 확보해 사용한다."""
    b = _broker()

    async def _probe(tk):
        b._us_excd_cache[tk.upper()] = "AMS"   # 프로브가 AMEX 로 판별
        return 12.34
    b.us_last_price = _probe
    captured = {}
    b._get_json = _ok_resp({}, captured)
    asyncio.run(b.us_buying_power("ARKX", 12.34, None))
    assert captured.get("OVRS_EXCG_CD") == "AMEX"

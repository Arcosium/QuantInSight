"""us_buying_power 가 통합증거금 주문가능액을 읽는다 — 사장 지시 2026-06-17(11시 US 사이클 점검).

배경: uid1(hh09080)은 통합증거금이 활성인데도 US 신규매수가 전부 '예수금 $0 → 제외'됐다.
라이브 TTTS3007R raw 출력(UAL $119, NYSE):
    ord_psbl_frcr_amt = "0.00"      ← 순수 USD 현금(우리가 읽던 값) = $0
    ovrs_ord_psbl_amt = "1657.94"   ← 해외 주문가능금액(통합증거금 반영)
    frcr_ord_psbl_amt1= "2037.05"   ← 외화 주문가능금액1
    max_ord_psbl_qty  = "13"        ← UAL $119 에 13주까지 주문 가능
근본: KIS 는 통합증거금 매수력을 ovrs_ord_psbl_amt/max_ord_psbl_qty 로 정확히 주는데,
us_buying_power 가 ord_psbl_frcr_amt(순수 USD 현금=$0)만 읽어 사이징이 전부 제외했다
(어젯밤 거래소 NASD 고정과 별개의 '잘못된 필드' 비대칭 버그).
교정: usd 는 ovrs_ord_psbl_amt → frcr_ord_psbl_amt1 → ord_psbl_frcr_amt 순으로 첫 양수.
"""
import asyncio
from infra.kis_broker import KISBroker


def _broker():
    return KISBroker({"kis_app_key": "k", "kis_app_secret": "s",
                      "kis_account_no": "1234567801",
                      "kis_base_url": "https://openapi.koreainvestment.com:9443"})


def _stub(output):
    async def _f(path, tr_id, p):
        return {"rt_cd": "0", "output": output}
    return _f


def test_uses_overseas_order_possible_amount_when_usd_cash_zero():
    """순수 USD 현금=0 이지만 통합증거금 ovrs_ord_psbl_amt=1657.94 → usd 는 1657.94."""
    b = _broker()
    b._us_excd_cache["UAL"] = "NYS"
    b._get_json = _stub({"ord_psbl_frcr_amt": "0.00", "ovrs_ord_psbl_amt": "1657.94",
                         "frcr_ord_psbl_amt1": "2037.051001", "max_ord_psbl_qty": "13",
                         "exrt": "1513.5"})
    r = asyncio.run(b.us_buying_power("UAL", 119.0, None))
    assert r["ok"] is True
    assert abs(r["usd"] - 1657.94) < 0.01
    assert r["qty"] == 13


def test_falls_back_to_frcr1_when_overseas_amount_absent():
    """ovrs_ord_psbl_amt 가 없거나 0 이면 frcr_ord_psbl_amt1 사용."""
    b = _broker()
    b._us_excd_cache["UAL"] = "NYS"
    b._get_json = _stub({"ord_psbl_frcr_amt": "0.00", "ovrs_ord_psbl_amt": "0",
                         "frcr_ord_psbl_amt1": "2037.05", "max_ord_psbl_qty": "13"})
    r = asyncio.run(b.us_buying_power("UAL", 119.0, None))
    assert abs(r["usd"] - 2037.05) < 0.01


def test_falls_back_to_usd_cash_when_only_field_present():
    """통합증거금 필드가 전무하면(모의 등) 기존 ord_psbl_frcr_amt 보존."""
    b = _broker()
    b._us_excd_cache["AAPL"] = "NAS"
    b._get_json = _stub({"ord_psbl_frcr_amt": "5000.00", "max_ord_psbl_qty": "10"})
    r = asyncio.run(b.us_buying_power("AAPL", 200.0, None))
    assert abs(r["usd"] - 5000.0) < 0.01


def test_rt_cd_failure_still_returns_not_ok():
    """rt_cd≠0 거부는 기존대로 ok=False(회귀 방지)."""
    b = _broker()
    b._us_excd_cache["WHR"] = "NYS"

    async def _fail(path, tr_id, p):
        return {"rt_cd": "1", "msg1": "상품이 없습니다"}
    b._get_json = _fail
    r = asyncio.run(b.us_buying_power("WHR", 40.0, None))
    assert r["ok"] is False and r["usd"] == 0.0

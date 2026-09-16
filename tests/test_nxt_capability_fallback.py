"""NXT 주문 미지원 응답 → _nxt_supported=False 고정, 이후 시간외 스킵 판정."""
import asyncio
from infra.kis_broker import KISBroker

_CREDS = {"kis_app_key": "K", "kis_app_secret": "S",
          "kis_account_no": "12345678-01",
          "kis_base_url": "https://openapi.koreainvestment.com:9443"}

class _Resp:
    def __init__(self, p): self._p = p
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def json(self): return self._p

class _Sess:
    def __init__(self, payload): self.payload = payload
    def get(self, url, headers=None, params=None):
        return _Resp({"rt_cd": "0", "output": [], "ctx_area_fk100": "", "ctx_area_nk100": ""})
    def post(self, url, headers=None, json=None):
        return _Resp(self.payload)

def _broker(monkeypatch, payload, is_mock=True):
    creds = dict(_CREDS)
    if is_mock:
        creds["kis_base_url"] = "https://openapivts.koreainvestment.com:29443"
    b = KISBroker(creds)
    async def _tok(): return "T"
    async def _s(): return _Sess(payload)
    monkeypatch.setattr(b, "token", _tok)
    monkeypatch.setattr(b, "_s", _s)
    return b

def test_initial_supported_is_none():
    b = KISBroker(_CREDS)
    assert b.nxt_supported() is None        # 미탐 — 1회 시도 허용

def test_unsupported_response_trips_flag(monkeypatch):
    b = _broker(monkeypatch, {"rt_cd": "1", "msg_cd": "40570000",
                              "msg1": "모의투자 미지원 거래소입니다"})
    asyncio.run(b.kr_buy("005930", 1, price=70000, exchange="NXT"))
    assert b.nxt_supported() is False

def test_real_mock_rejection_message_trips_flag(monkeypatch):
    # 2026-06-08 라이브검증으로 확인된 모의서버 실제 거부 메시지(msg_cd=41050000) — 회귀 고정
    b = _broker(monkeypatch, {"rt_cd": "1", "msg_cd": "41050000",
                              "msg1": "모의투자에서 대체거래소 서비스를 제공하지 않습니다."})
    asyncio.run(b.kr_buy("005930", 1, price=70000, exchange="NXT"))
    assert b.nxt_supported() is False

def test_success_sets_supported_true(monkeypatch):
    b = _broker(monkeypatch, {"rt_cd": "0", "msg1": "정상처리 되었습니다"})
    asyncio.run(b.kr_buy("005930", 1, price=70000, exchange="NXT"))
    assert b.nxt_supported() is True

def test_krx_orders_do_not_set_nxt_flag(monkeypatch):
    b = _broker(monkeypatch, {"rt_cd": "0", "msg1": "정상"})
    asyncio.run(b.kr_buy("005930", 1, price=70000))   # KRX
    assert b.nxt_supported() is None                  # NXT 무관 주문은 플래그 불변

def test_ordinary_nxt_rejection_does_not_trip(monkeypatch):
    # 잔고부족 등 '지원은 되나 거부' → 미지원으로 오판하면 안 됨
    b = _broker(monkeypatch, {"rt_cd": "1", "msg_cd": "40310000",
                              "msg1": "주문가능금액을 초과하였습니다"})
    asyncio.run(b.kr_buy("005930", 1, price=70000, exchange="NXT"))
    assert b.nxt_supported() is not False             # None 유지(지원 여부 미확정)

"""정규장(KRX)=구 TR 무변경 회귀 + 시간외(NXT)=신 TR + EXCG_ID_DVSN_CD."""
import asyncio
from infra.kis_broker import KISBroker, OrderDraft

class _Resp:
    def __init__(self, p): self._p = p
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def json(self): return self._p

class _Sess:
    def __init__(self): self.posts = []
    def get(self, url, headers=None, params=None):
        return _Resp({"rt_cd": "0", "output": [], "ctx_area_fk100": "", "ctx_area_nk100": ""})
    def post(self, url, headers=None, json=None):
        self.posts.append({"url": url, "tr_id": (headers or {}).get("tr_id"), "json": json})
        return _Resp({"rt_cd": "0", "msg1": "정상처리 되었습니다"})

_CREDS = {"kis_app_key": "K", "kis_app_secret": "S",
          "kis_account_no": "12345678-01",
          "kis_base_url": "https://openapi.koreainvestment.com:9443"}

def _broker(monkeypatch, sess, is_mock=False):
    b = KISBroker(_CREDS)
    b.is_mock = is_mock
    async def _tok(): return "T"
    async def _s(): return sess
    monkeypatch.setattr(b, "token", _tok)
    monkeypatch.setattr(b, "_s", _s)
    return b

def test_krx_buy_unchanged_legacy_tr(monkeypatch):
    s = _Sess(); b = _broker(monkeypatch, s)
    asyncio.run(b.kr_buy("005930", 1, price=70000))           # exchange 기본 KRX
    o = [p for p in s.posts if p["url"].endswith("/order-cash")][0]
    assert o["tr_id"] == "TTTC0802U"
    assert "EXCG_ID_DVSN_CD" not in o["json"]
    assert o["json"]["ORD_DVSN"] == "00"

def test_krx_sell_unchanged_legacy_tr(monkeypatch):
    s = _Sess(); b = _broker(monkeypatch, s)
    asyncio.run(b.kr_sell("005930", 1, price=0))              # 시장가
    o = [p for p in s.posts if p["url"].endswith("/order-cash")][0]
    assert o["tr_id"] == "TTTC0801U"
    assert "EXCG_ID_DVSN_CD" not in o["json"]
    assert o["json"]["ORD_DVSN"] == "01"

def test_nxt_buy_new_tr_and_excg(monkeypatch):
    s = _Sess(); b = _broker(monkeypatch, s)
    asyncio.run(b.kr_buy("005930", 1, price=70000, exchange="NXT"))
    o = [p for p in s.posts if p["url"].endswith("/order-cash")][0]
    assert o["tr_id"] == "TTTC0012U"
    assert o["json"]["EXCG_ID_DVSN_CD"] == "NXT"
    assert o["json"]["ORD_DVSN"] == "00"
    assert o["json"]["ORD_UNPR"] == "70000"

def test_nxt_sell_new_tr_and_sll_type(monkeypatch):
    s = _Sess(); b = _broker(monkeypatch, s)
    asyncio.run(b.kr_sell("005930", 1, price=69000, exchange="NXT"))
    o = [p for p in s.posts if p["url"].endswith("/order-cash")][0]
    assert o["tr_id"] == "TTTC0011U"
    assert o["json"]["EXCG_ID_DVSN_CD"] == "NXT"
    assert o["json"]["SLL_TYPE"] == "01"
    assert o["json"]["ORD_DVSN"] == "00"

def test_nxt_mock_tr_conversion(monkeypatch):
    s = _Sess(); b = _broker(monkeypatch, s, is_mock=True)
    asyncio.run(b.kr_buy("005930", 1, price=70000, exchange="NXT"))
    o = [p for p in s.posts if p["url"].endswith("/order-cash")][0]
    assert o["tr_id"] == "VTTC0012U"

def test_place_order_routes_exchange(monkeypatch):
    s = _Sess(); b = _broker(monkeypatch, s)
    od = OrderDraft(ticker="005930", side="buy", qty=1, price_type="limit",
                    limit_price=70000, market="KR", exchange="NXT", approved=True)
    asyncio.run(b.place_order(od))
    o = [p for p in s.posts if p["url"].endswith("/order-cash")][0]
    assert o["tr_id"] == "TTTC0012U"
    assert o["json"]["EXCG_ID_DVSN_CD"] == "NXT"


def test_kr_buy_invalid_limit_is_rounded_up_to_exchange_tick(monkeypatch):
    s = _Sess(); b = _broker(monkeypatch, s)
    asyncio.run(b.kr_buy("003540", 1, price=27196))
    o = [p for p in s.posts if p["url"].endswith("/order-cash")][0]
    assert o["json"]["ORD_DVSN"] == "00"
    assert o["json"]["ORD_UNPR"] == "27200"


def test_kr_sell_invalid_limit_is_rounded_up_to_exchange_tick(monkeypatch):
    s = _Sess(); b = _broker(monkeypatch, s)
    asyncio.run(b.kr_sell("003540", 1, price=27196))
    o = [p for p in s.posts if p["url"].endswith("/order-cash")][0]
    assert o["json"]["ORD_DVSN"] == "00"
    assert o["json"]["ORD_UNPR"] == "27200"

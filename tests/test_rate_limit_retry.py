"""KIS rate-limit('초당 거래건수를 초과하였습니다') 거부 시 자동 재전송 — 주문 드롭 금지.

배경(2026-05-28 로그 리뷰): uid2 cycle 34 에서 FCX 시장가 매수가 '초당 거래건수를 초과하였습니다'
로 거부됐는데 재시도가 없어 주문이 통째로 드롭됐다(주문 절대 스킵 금지 위반). rate-limit 거부는
rt_cd≠0(미체결 확정)이라 재전송이 안전하다 — 간격을 두고 몇 차례 재시도한다.
"""
import asyncio
from infra.kis_broker import KISBroker


def _broker():
    b = object.__new__(KISBroker)
    b._RATE_LIMIT_BACKOFF_SEC = 0.0  # 테스트 빠르게
    b._rate_base = 0.06; b._min_interval = 0.06  # 적응 페이싱(2026-06-17) 상태

    async def fake_token(force=False):
        return "tok"
    async def fake_pace():
        return None
    b.token = fake_token
    b._pace = fake_pace
    return b


def test_rate_limited_order_is_retried_then_succeeds():
    b = _broker()
    calls = {"n": 0}

    async def make_request(tok):
        calls["n"] += 1
        if calls["n"] < 3:
            return {"rt_cd": "1", "msg_cd": "EGW00201",
                    "msg1": "초당 거래건수를 초과하였습니다"}
        return {"rt_cd": "0", "msg1": "주문 전송 완료"}

    d = asyncio.run(b._authed_json(make_request))
    assert d["rt_cd"] == "0", "rate-limit 거부 후 재시도로 결국 성공해야 한다"
    assert calls["n"] == 3, "거부 2회 + 성공 1회"


def test_rate_limit_gives_up_after_max_but_does_not_loop_forever():
    b = _broker()
    calls = {"n": 0}

    async def make_request(tok):
        calls["n"] += 1
        return {"rt_cd": "1", "msg_cd": "EGW00201",
                "msg1": "초당 거래건수를 초과하였습니다"}

    d = asyncio.run(b._authed_json(make_request))
    assert d["rt_cd"] != "0"
    assert calls["n"] == 4, "최초 1회 + 재시도 3회 후 포기(무한루프 금지)"


def test_normal_response_not_retried():
    b = _broker()
    calls = {"n": 0}

    async def make_request(tok):
        calls["n"] += 1
        return {"rt_cd": "0", "msg1": "주문 전송 완료"}

    d = asyncio.run(b._authed_json(make_request))
    assert d["rt_cd"] == "0"
    assert calls["n"] == 1, "정상 응답은 재시도 안 함"

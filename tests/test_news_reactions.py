"""News-only reception provenance, bounded failure, and trading isolation."""
import asyncio
import copy
import json
import time
from types import SimpleNamespace

import aiohttp
import pytest
from tools import news_reactions as nr

ARTICLE = {"title": "실적 공시", "summary": "회사가 실적을 발표했다.",
           "link": "https://n.news.naver.com/article/015/0005338465"}


def result():
    return {"voices": [{"id": pid, "reading": "확인할 필요가 있다.",
                        "counterpoint": "실적의 지속성은 미확인.", "evidence": ["N1"]}
                       for pid in nr.PERSONAS]}


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    nr._CACHE.clear()
    monkeypatch.setattr(nr, "_LOCK", asyncio.Lock())


def test_rejects_unverified_evidence_observed_opinion_and_price_scores():
    d = result()
    d["voices"][0]["evidence"] = ["C1"]
    with pytest.raises(ValueError):
        nr._validate(d, {"N1"})
    d = result()
    d["observed"] = [{"theme": "실제 여론이라고 주장", "evidence": ["N1"]}]
    with pytest.raises(ValueError):
        nr._validate(d, {"N1"})
    d = result()
    d["voices"][0]["price_score"] = 0.9
    with pytest.raises(ValueError):
        nr._validate(d, {"N1"})


def test_two_rounds_news_only_no_external_collection_and_shared_cache(monkeypatch):
    calls = []

    def forbidden_http(*args, **kwargs):
        raise AssertionError("Reception must not fetch comments or external data")

    async def turn(model, payload):
        calls.append(copy.deepcopy(payload))
        return result()

    monkeypatch.setattr(aiohttp, "ClientSession", forbidden_http)
    monkeypatch.setattr(nr, "_turn", turn)

    async def run():
        article = {**ARTICLE, "comments": ["사용하면 안 되는 댓글"]}
        return await asyncio.gather(*(nr.analyze_reception([article], model="local",
                                      disclosures="검증할 공시") for _ in range(2)))

    a, b = asyncio.run(run())
    assert a == b and a["status"] == "ok" and a["rounds"] == 2
    assert len(calls) == 2 and calls[1]["prior_voices"] == result()["voices"]
    assert set(calls[0]) == {"round", "personas", "news", "disclosures"}
    assert calls[0]["disclosures"] == "검증할 공시"
    assert "사용하면 안 되는 댓글" not in json.dumps(calls, ensure_ascii=False)
    assert not {"observations", "observed", "comments"} & set(a)
    rendered = nr.format_reception(a)
    assert "뉴스 예상 반응" in rendered and "실제 댓글은 수집하지 않으며" in rendered
    assert ARTICLE["link"] in rendered


def test_failed_second_round_retains_verified_first_round(monkeypatch):
    async def turn(model, payload):
        if payload["round"] == 2:
            raise ValueError("Unverifiable model output")
        return result()

    monkeypatch.setattr(nr, "_turn", turn)
    r = asyncio.run(nr.analyze_reception([ARTICLE], model="local"))
    assert r["status"] == "partial" and r["rounds"] == 1 and r["voices"]
    assert "실제 여론은 미확인" in nr.format_reception(r)


def test_deadline_cancels_long_local_retry_and_external_cancellation_propagates(monkeypatch):
    monkeypatch.setattr(nr, "DEADLINE_SEC", 0.01)

    async def hang(model, payload):
        await asyncio.sleep(30)

    monkeypatch.setattr(nr, "_turn", hang)
    start = time.monotonic()
    r = asyncio.run(nr.analyze_reception([ARTICLE], model="local"))
    assert time.monotonic() - start < 1
    assert r["status"] == "unavailable" and r["error_type"] == "TimeoutError"
    assert "실제 여론은 미확인" in nr.format_reception(r)

    async def cancel(model, payload):
        raise asyncio.CancelledError()

    monkeypatch.setattr(nr, "_turn", cancel)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(nr.analyze_reception([ARTICLE], model="local"))


def test_news_stage_keeps_reactions_out_of_trading_evidence(monkeypatch):
    import main_swarm as ms
    factual = "기사로 확인된 실적 공시입니다. " * 5
    synthetic = "가상의 독자 해석은 불확실합니다."
    events, prompts = [], []

    async def think(prompt):
        prompts.append(prompt)
        return factual

    async def emit(event):
        events.append(event)

    async def shared(kind, fingerprint, compute):
        return await compute()

    async def snapshot():
        return {"buying_power": {}}

    async def reaction(*args, **kw):
        return {"status": "ok", "rounds": 2}

    monkeypatch.setattr(nr, "analyze_reception", reaction)
    monkeypatch.setattr(nr, "format_reception", lambda r: synthetic)
    monkeypatch.setattr(ms, "_news_cache", {"value": "", "session": "", "ts": 0})
    monkeypatch.setattr(ms, "_macro_cache", {"value": "기존 매크로 분석입니다. " * 5, "ts": time.time()})
    obj = SimpleNamespace(uid=None, news_analyst=SimpleNamespace(model="local", think=think),
        cycle_log=SimpleNamespace(log=lambda *a: None), _emit=emit, _emit_news_activity=emit,
        _shared_or_compute=shared, broker=SimpleNamespace(kr_account_snapshot=snapshot))
    cyc = SimpleNamespace(news_articles=[ARTICLE], market_open=False, session="US_TRADING",
        formatted_news="뉴스 기사", index_facts="검증 지수", dart_report="공시", fresh_news=True, holdings=[])
    asyncio.run(ms.ArquantOrchestrator._cyc_stage_news_macro(obj, cyc))
    assert cyc.news_report == factual and synthetic not in cyc.news_report
    assert all(synthetic not in p for p in prompts)
    assert any(e.get("message") == synthetic and "news_reception" in e for e in events)

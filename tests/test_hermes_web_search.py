"""Macro research(deep_research) — 서버제어 브라우저 파이프라인(ArachneControl) 동작.

2026-06-29 사장 지시: Hermes 에이전트(웹툴 서브프로세스) 제거 → Arachne companion
(/research/gather) 수집 + 로컬 LLM 합성으로 대체. 아래는 새 경로 검증.
"""
import pytest

from tools import global_search


@pytest.mark.asyncio
async def test_deep_research_synthesizes_from_gathered_evidence(monkeypatch):
    """gather 근거가 합성 프롬프트에 들어가고 LLM 합성 결과를 반환한다."""
    async def fake_gather(query, timeout_sec):
        return {
            "results": [{"title": "연준", "url": "https://ex/1", "content": "6월 금리 동결"}],
            "extra": [{"title": "참고", "url": "https://ex/2", "snippet": "점도표 상향"}],
        }
    monkeypatch.setattr(global_search, "_gather", fake_gather)

    captured = {}

    async def fake_chat(**kwargs):
        captured["user"] = kwargs["messages"][-1]["content"]
        captured["thinking"] = kwargs.get("thinking")
        return {"choices": [{"message": {"content": "합성된 리서치"}}]}
    monkeypatch.setattr("infra.local_llm_client.chat_completion", fake_chat)

    out = await global_search.deep_research("연준 전망", model="qwen-plain")
    assert out == "합성된 리서치"
    # 수집 근거가 프롬프트에 주입됐는지 + 합성은 reasoning OFF
    assert "6월 금리 동결" in captured["user"]
    assert captured["thinking"] is False


@pytest.mark.asyncio
async def test_deep_research_fail_open_when_no_evidence(monkeypatch):
    """수집(gather) 실패 시 LLM 호출 없이 빈 문자열(fail-open)."""
    async def empty_gather(query, timeout_sec):
        return None
    monkeypatch.setattr(global_search, "_gather", empty_gather)

    called = {"v": False}

    async def fake_chat(**kwargs):
        called["v"] = True
        return {"choices": [{"message": {"content": "x"}}]}
    monkeypatch.setattr("infra.local_llm_client.chat_completion", fake_chat)

    assert await global_search.deep_research("시장") == ""
    assert called["v"] is False


@pytest.mark.asyncio
async def test_long_instruction_query_is_distilled_to_search_terms(monkeypatch):
    """지시문형 긴 질의는 짧은 검색어로 증류해 gather 한다(2026-08-24).

    main_swarm 이 2,600자 분석 지시문을 그대로 넘겨 Bing 질의가 상시 0건이던 사고의 회귀."""
    long_query = ("현재 한국 증시 시황을 종합·심층 분석해 주세요. 다음 4가지 관점:\n"
                  "1. 외국인 수급\n2. 반도체 업황\n3. 환율\n4. 금리\n" + "컨텍스트 " * 100)
    gathered = []

    async def fake_gather(query, timeout_sec):
        gathered.append(query)
        return {"results": [{"title": "t", "url": f"https://ex/{len(gathered)}",
                             "content": "근거 본문"}], "extra": []}
    monkeypatch.setattr(global_search, "_gather", fake_gather)

    async def fake_chat(**kwargs):
        user = kwargs["messages"][-1]["content"]
        if "질의 3개로 줄여라" in user:
            return {"choices": [{"message": {"content": "코스피 외국인 수급\n반도체 업황 전망\n원달러 환율 금리"}}]}
        return {"choices": [{"message": {"content": "합성된 리서치"}}]}
    monkeypatch.setattr("infra.local_llm_client.chat_completion", fake_chat)

    out = await global_search.deep_research(long_query, model="qwen-plain")
    assert out == "합성된 리서치"
    assert gathered == ["코스피 외국인 수급", "반도체 업황 전망", "원달러 환율 금리"]
    assert all(len(q) <= 80 for q in gathered)


@pytest.mark.asyncio
async def test_short_query_passes_through_without_distillation(monkeypatch):
    """짧은 한 줄 질의는 증류 LLM 콜 없이 그대로 검색한다."""
    gathered = []

    async def fake_gather(query, timeout_sec):
        gathered.append(query)
        return {"results": [{"title": "t", "url": "https://ex/1", "content": "근거"}], "extra": []}
    monkeypatch.setattr(global_search, "_gather", fake_gather)
    chat_calls = []

    async def fake_chat(**kwargs):
        chat_calls.append(kwargs["messages"][-1]["content"])
        return {"choices": [{"message": {"content": "합성"}}]}
    monkeypatch.setattr("infra.local_llm_client.chat_completion", fake_chat)

    assert await global_search.deep_research("연준 금리 전망", model="qwen-plain") == "합성"
    assert gathered == ["연준 금리 전망"]
    assert len(chat_calls) == 1          # 합성 1콜뿐 — 증류 콜 없음

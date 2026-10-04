"""Macro research via the server-controlled browser pipeline (ArachneControl).

(2026-06-29 사장 지시) Hermes 에이전트(웹툴 서브프로세스)를 제거하고, 서버가 헤드리스
브라우저를 원격 제어해 수집하는 **Arachne 리서치 companion**(기본 127.0.0.1:8771)을 쓴다.
흐름: 질의 → companion `/research/gather`(Bing 검색 + 본문 추출) → 근거(evidence)를
로컬 LLM(OpenAI 호환)으로 한국어 합성. 외부 유료 API·Hermes 불필요. 실패는 fail-open("").
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import aiohttp

logger = logging.getLogger("RESEARCH")

# 서버제어 브라우저 리서치 companion (ArachneControl). 로컬 전용 기본.
RESEARCH_URL = os.getenv("ARQUANT_RESEARCH_URL", "http://127.0.0.1:8771").rstrip("/")
RESEARCH_K = int(os.getenv("ARQUANT_RESEARCH_K", "6"))
RESEARCH_FETCH_K = int(os.getenv("ARQUANT_RESEARCH_FETCH_K", "4"))
# 출처별 본문 인용 상한(합성 프롬프트 비대화 방지)
_PER_SOURCE_CHARS = int(os.getenv("ARQUANT_RESEARCH_SRC_CHARS", "2500"))


# 검색엔진에 넣을 수 있는 질의 길이 상한 — 이보다 길면 지시문으로 보고 증류한다.
_QUERY_MAX = 60


async def _distill_queries(query: str, model: str, timeout_sec: int) -> list[str]:
    """지시문형 질의 → 짧은 검색어 2~3개.

    호출부(main_swarm)가 2,600자짜리 분석 지시문을 그대로 넘기는데, 이것이 Bing 질의로
    URL 인코딩돼 상시 0건이었다(2026-08-24 감사에서 확인 — '리서치 근거가 비었습니다'의
    정체). 짧은 질의는 그대로 쓰고, 긴 지시문만 LLM 1콜(thinking OFF)로 줄인다."""
    q = (query or "").strip()
    if len(q) <= _QUERY_MAX and "\n" not in q:
        return [q]
    from infra.local_llm_client import chat_completion, response_text
    try:
        r = await chat_completion(
            api_key="", model=model,
            messages=[{"role": "system", "content": "너는 검색 질의 설계자다."},
                      {"role": "user", "content":
                       "아래 리서치 요청의 핵심을 검색엔진용 한국어 질의 3개로 줄여라. "
                       "질의는 한 줄에 하나씩, 각 8단어 이내로만 출력하라. 설명 금지.\n\n" + q[:2000]}],
            max_tokens=200, temperature=0.2, timeout_sec=min(timeout_sec, 60), thinking=False)
        lines = [ln.strip(" -•*\"'") for ln in (response_text(r) or "").splitlines()]
        out = [ln for ln in lines if ln and len(ln) <= 80][:3]
        if out:
            return out
    except Exception as exc:
        logger.warning("질의 증류 실패(첫 줄 폴백): %s", exc)
    return [q.splitlines()[0][:_QUERY_MAX]]


async def _gather(query: str, timeout_sec: int) -> Optional[dict]:
    """Arachne companion 에서 검색+본문추출 근거를 수집한다."""
    url = f"{RESEARCH_URL}/research/gather"
    payload = {"query": query, "k": RESEARCH_K, "fetch_k": RESEARCH_FETCH_K}
    try:
        timeout = aiohttp.ClientTimeout(total=min(timeout_sec, 120))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    logger.warning("리서치 gather HTTP %s", resp.status)
                    return None
                return await resp.json()
    except Exception as exc:
        logger.warning("리서치 gather 실패: %s", exc)
        return None


def _format_evidence(data: dict) -> str:
    """수집 근거를 출처 표기가 있는 프롬프트 블록으로 변환."""
    blocks: list[str] = []
    idx = 1
    for r in data.get("results", []) or []:
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        content = (r.get("content") or "").strip()
        if not (content or title):
            continue
        blocks.append(
            f"[출처{idx}] {title}\nURL: {url}\n본문발췌: {content[:_PER_SOURCE_CHARS]}\n"
        )
        idx += 1
    # 본문 추출은 안 됐지만 검색에 잡힌 항목은 제목/스니펫만 보조 제공
    for r in data.get("extra", []) or []:
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        snip = (r.get("snippet") or "").strip()
        if not title:
            continue
        blocks.append(f"[참고{idx}] {title}\nURL: {url}\n요약: {snip}\n")
        idx += 1
    return "\n".join(blocks)


async def deep_research(
    query: str,
    max_tokens: int = 8000,
    model: Optional[str] = None,
    timeout_sec: int = 180,
    api_key: Optional[str] = None,
) -> str:
    """서버제어 브라우저로 근거를 모은 뒤 로컬 LLM 으로 한국어 매크로 리서치를 합성한다.

    ``api_key``/``model`` 인자는 이전 호출부 호환용. ``model`` 미지정 시 macro_researcher
    배정 모델을 쓰되, 합성은 reasoning OFF(빠르고 빈응답 버그 회피)로 수행한다."""
    from config import LOCAL_LLM_MODEL
    from infra.admin_config import resolve_model
    from infra.local_llm_client import chat_completion, response_text, split_thinking

    selected_model = model or resolve_model("macro_researcher", LOCAL_LLM_MODEL)
    real_model, _ = split_thinking(selected_model)  # 합성은 thinking OFF로 강제

    # 지시문 → 검색어 증류 후 질의별 수집, url 기준 dedup 병합.
    merged: dict = {"results": [], "extra": []}
    seen: set[str] = set()
    for q in await _distill_queries(query, real_model, timeout_sec):
        data = await _gather(q, timeout_sec)
        for key in ("results", "extra"):
            for r in (data or {}).get(key) or []:
                u = (r.get("url") or "").split("#")[0]
                if u and u in seen:
                    continue
                seen.add(u)
                merged[key].append(r)
    evidence = _format_evidence(merged)
    if not evidence.strip():
        logger.warning("리서치 근거가 비었습니다(검색/추출 0건).")
        return ""

    system = (
        "당신은 금융시장 리서치 전문가입니다. 아래 [수집 근거]는 방금 웹에서 수집한 "
        "최신 자료입니다. 이 근거에 기반해서만 한국어로 분석하세요. "
        "각 핵심 주장에 출처(기관/매체)와 가능하면 날짜를 명시하고, 근거에 없는 사실을 "
        "단정하지 마세요. 근거가 빈약한 부분은 '확인 불가'로 표시하세요."
    )
    user = f"[질의]\n{query}\n\n[수집 근거]\n{evidence}"

    try:
        result = await chat_completion(
            api_key="",
            model=real_model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max(512, int(max_tokens)),
            temperature=0.3,
            timeout_sec=min(timeout_sec, 300),
            thinking=False,
        )
    except Exception as exc:
        logger.warning("리서치 합성(LLM) 실패: %s", exc)
        return ""
    text = response_text(result)
    if not text:
        logger.warning("리서치 합성이 빈 응답을 반환했습니다.")
    return text

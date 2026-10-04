"""Async client for the local OpenAI-compatible LLM server.

No cloud-provider API key is accepted or sent.  ``+thinking`` is an internal
model suffix that is converted to the common llama.cpp/OpenAI-compatible
thinking request fields immediately before dispatch.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import aiohttp

logger = logging.getLogger("LOCAL_LLM")

# Thinking ON 을 나타내는 가상 접미사(실제 모델명엔 없음 — 전송 전 제거).
_THINK_SUFFIX = "+thinking"

# 로컬 OpenAI 호환 서버가 재기동 중이거나 일시적으로 바쁠 수 있으므로 백오프 재시도한다.
# 400/401/402/422 같은 영구 오류는 재시도하지 않고 즉시 실패한다.
_RETRYABLE_STATUS = (429, 500, 503)
_MAX_ATTEMPTS = 3
# 사장 지시 2026-07-22 상향 (900→1800 / 1800→3600).
# 이 두 값은 config.AGENT_MAX_TOKENS 의 pro 티어 예산과 **짝을 이뤄야 한다** — 예산이 타임아웃보다
# 길면 예산을 다 쓰는 생성이 구조적으로 완주하지 못하고, 재시도는 토큰 0부터 다시 시작해 그레이스만
# 태우다 최종 실패한다(2026-07-22 관측: 24h 재시도 13회·최종 실패 4회, 전부 thinking ON 에이전트).
#   실측 디코드 53 t/s · pro 예산 40000 → 755초 소요.
#   TIMEOUT 1800 = 2.4배 여유(GPU 경합·프롬프트 길이 변동 흡수).
#   GRACE   3600 = 타임아웃의 2배 — 모델 재기동 중 끊긴 경우 재시도 1회를 보장한다.
# 예산을 다시 올릴 일이 있으면 여기 타임아웃도 같이 올릴 것.
_LOCAL_LLM_GRACE_SEC = int(os.environ.get("LOCAL_LLM_GRACE_SEC", "3600"))
_LOCAL_LLM_TIMEOUT_SEC = int(os.environ.get("LOCAL_LLM_TIMEOUT_SEC", "1800"))


class LocalLLMError(RuntimeError):
    pass


# ── 슬러그/백엔드 판별 ────────────────────────────────────────────────────────

def split_thinking(model: str) -> Tuple[str, bool]:
    """'+thinking' 가상 접미사를 떼어 (실제_슬러그, reasoning_on) 으로 분리한다."""
    m = model or ""
    if m.endswith(_THINK_SUFFIX):
        return m[: -len(_THINK_SUFFIX)], True
    return m, False


# ── 요청 조립(순수 함수 — I/O 없음, 단위테스트 대상) ──────────────────────────

def build_request(
    *,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    thinking: Optional[bool],
    response_format: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    """로컬 OpenAI 호환 서버의 (url, headers, payload)를 반환한다.

    ``api_key`` 인자는 기존 호출부와의 호환을 위해 남겨두지만 사용하지 않는다.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
    }
    from config import LOCAL_LLM_BASE_URL
    real_model, suffix_thinking = split_thinking(model)
    reasoning_on = bool(thinking) if thinking is not None else suffix_thinking
    payload["model"] = real_model
    # llama.cpp accepts chat_template_kwargs; reasoning is also understood by
    # several OpenAI-compatible servers. Sending both preserves the existing
    # thinking/non-thinking agent split without an external provider.
    payload["chat_template_kwargs"] = {"enable_thinking": reasoning_on}
    payload["reasoning"] = {"enabled": reasoning_on}
    url = f"{LOCAL_LLM_BASE_URL}/chat/completions"
    if response_format:
        payload["response_format"] = response_format
    headers = {"Content-Type": "application/json"}
    return url, headers, payload


# ── HTTP + 백오프 재시도(공유) ────────────────────────────────────────────────

def _is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


async def _post_with_retry(url: str, headers: Dict[str, str],
                           payload: Dict[str, Any], timeout_sec: int) -> Dict[str, Any]:
    last_err: Optional[str] = None
    is_local = _is_local_url(url)
    grace_deadline = asyncio.get_running_loop().time() + _LOCAL_LLM_GRACE_SEC
    attempt = 0
    while True:
        try:
            effective_timeout_sec = (
                max(int(timeout_sec), _LOCAL_LLM_TIMEOUT_SEC)
                if is_local
                else int(timeout_sec)
            )
            timeout = aiohttp.ClientTimeout(total=effective_timeout_sec)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if not isinstance(data, dict) or not data.get("choices"):
                            raise LocalLLMError(
                                f"LLM 응답 형식 오류: {str(data)[:300]}")
                        return data
                    body = await response.text()
                    last_err = f"HTTP {response.status}: {body[:300]}"
                    # 영구오류 또는 마지막 시도 → 즉시 실패
                    if response.status not in _RETRYABLE_STATUS:
                        raise LocalLLMError(f"LLM HTTP {response.status}: {body[:600]}")
                    if not is_local and attempt >= _MAX_ATTEMPTS - 1:
                        raise LocalLLMError(
                            f"LLM HTTP {response.status} ({_MAX_ATTEMPTS}회 재시도 실패): {body[:300]}")
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            if not is_local and attempt >= _MAX_ATTEMPTS - 1:
                raise LocalLLMError(
                    f"LLM 연결/timeout 실패({_MAX_ATTEMPTS}회): {last_err}")
        attempt += 1
        if is_local:
            remaining = grace_deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise LocalLLMError(
                    f"로컬 LLM 재기동 대기 초과({_LOCAL_LLM_GRACE_SEC}s): {last_err}")
            _wait = min(1.5 * (2 ** min(attempt - 1, 5)), 15.0, remaining)
            logger.warning("로컬 LLM 대기 %d회 (%s) — %.1fs 후 재요청",
                           attempt, last_err, _wait)
        else:
            if attempt >= _MAX_ATTEMPTS:
                raise LocalLLMError(
                    f"LLM 연결/timeout 실패({_MAX_ATTEMPTS}회): {last_err}")
            _wait = 1.5 * (2 ** (attempt - 1))
            logger.warning("LLM 재시도 %d/%d (%s) — %.1fs 후 재요청",
                           attempt, _MAX_ATTEMPTS, last_err, _wait)
        await asyncio.sleep(_wait)
    raise LocalLLMError(f"LLM 호출 실패: {last_err}")


async def chat_completion(
    *,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.3,
    timeout_sec: int = 300,
    thinking: Optional[bool] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    url, headers, payload = build_request(
        api_key=api_key, model=model, messages=messages, max_tokens=max_tokens,
        temperature=temperature, thinking=thinking, response_format=response_format,
    )
    return await _post_with_retry(url, headers, payload, timeout_sec)


_MD_EMPHASIS_RE = re.compile(r"\*\*+")


def strip_markdown_emphasis(text: str) -> str:
    """LLM 응답의 마크다운 굵게 마커(`**`)를 제거한다 (사장 지시 — 재발 2026-07-29).

    채팅 UI는 마크다운을 렌더하지 않아 `**농심**` 이 그대로 노출된다. 전 에이전트 프롬프트에
    금지를 박아뒀지만 로컬 모델이 계속 무시하므로, 출력 단계에서 결정론적으로 지운다.
    `**` 런 자체만 지우므로 본문 글자는 하나도 손대지 않는다."""
    return _MD_EMPHASIS_RE.sub("", text or "")


def response_text(data: Dict[str, Any]) -> str:
    choice = (data.get("choices") or [{}])[0] or {}
    message = choice.get("message") or {}
    return strip_markdown_emphasis(str(message.get("content") or "")).strip()

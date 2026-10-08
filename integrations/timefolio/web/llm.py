"""OpenAI 호환 LLM 호출기.

API 키는 프로세스 환경변수에서만 읽고 요청·응답 로그에 남기지 않는다.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import config


class LLMConfigError(RuntimeError):
    pass


def provider_status() -> dict:
    return {
        "openai": {"configured": bool(config.OPENAI_API_KEY), "model": config.OPENAI_MODEL},
        "gemini": {"configured": bool(config.GEMINI_API_KEY), "model": config.GEMINI_MODEL},
        "deepseek": {"configured": bool(config.DEEPSEEK_API_KEY), "model": config.DEEPSEEK_MODEL},
        "local": {"configured": True, "model": config.LOCAL_LLM_MODEL},
        "default": config.LLM_PROVIDER,
    }


def _resolve(provider: str | None) -> tuple[str, str, str, str]:
    requested = (provider or config.LLM_PROVIDER or "auto").strip().lower()
    choices = {
        "openai": (config.OPENAI_BASE_URL, config.OPENAI_API_KEY, config.OPENAI_MODEL),
        "gemini": (config.GEMINI_BASE_URL, config.GEMINI_API_KEY, config.GEMINI_MODEL),
        "deepseek": (config.DEEPSEEK_BASE_URL, config.DEEPSEEK_API_KEY, config.DEEPSEEK_MODEL),
        "local": (config.LOCAL_LLM_BASE_URL, "", config.LOCAL_LLM_MODEL),
    }
    if requested == "auto":
        for name in ("openai", "gemini", "deepseek"):
            base, key, model = choices[name]
            if key:
                return name, base, key, model
        requested = "local"
    if requested not in choices:
        raise LLMConfigError(f"지원하지 않는 AI 제공사입니다: {requested}")
    base, key, model = choices[requested]
    if requested != "local" and not key:
        raise LLMConfigError(f"{requested.upper()} API 키가 설정되지 않았습니다")
    return requested, base, key, model


def complete(prompt: str, *, provider: str | None = None, json_mode: bool = False,
             max_tokens: int = 8000) -> tuple[str, dict]:
    name, base, key, model = _resolve(provider)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if name == "openai":
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read()).get("error", {}).get("message", "")
        except Exception:
            pass
        raise RuntimeError(f"{name} 요청 실패({exc.code})" + (f": {detail}" if detail else "")) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{name} 연결 실패: {exc.reason}") from exc
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"{name} 응답 형식을 읽지 못했습니다") from exc
    return content, {"provider": name, "model": model}

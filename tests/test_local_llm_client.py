"""Local LLM migration contract: no provider key and preserved thinking split."""
from infra.local_llm_client import build_request


def test_local_request_has_no_authorization_and_uses_configured_base(monkeypatch):
    monkeypatch.setattr("config.LOCAL_LLM_BASE_URL", "http://local.test:8080/v1")
    ignored_legacy_argument = "ignored"
    url, headers, payload = build_request(
        api_key=ignored_legacy_argument,
        model="local-model", messages=[], max_tokens=10, temperature=0.0,
        thinking=False, response_format=None,
    )
    assert url == "http://local.test:8080/v1/chat/completions"
    assert "Authorization" not in headers
    assert payload["model"] == "local-model"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["reasoning"] == {"enabled": False}


def test_thinking_suffix_is_converted_to_local_request_flag(monkeypatch):
    monkeypatch.setattr("config.LOCAL_LLM_BASE_URL", "http://local.test/v1")
    _, _, payload = build_request(
        api_key="", model="local-model+thinking", messages=[], max_tokens=10,
        temperature=0.0, thinking=None, response_format=None,
    )
    assert payload["model"] == "local-model"
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["reasoning"] == {"enabled": True}

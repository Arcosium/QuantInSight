import pytest

from web import llm


def test_missing_remote_key_is_rejected(monkeypatch):
    monkeypatch.setattr(llm.config, "OPENAI_API_KEY", "")
    with pytest.raises(llm.LLMConfigError):
        llm._resolve("openai")


def test_auto_selects_configured_provider_without_exposing_key(monkeypatch):
    monkeypatch.setattr(llm.config, "OPENAI_API_KEY", "")
    monkeypatch.setattr(llm.config, "GEMINI_API_KEY", "configured-secret")
    monkeypatch.setattr(llm.config, "DEEPSEEK_API_KEY", "")
    name, _base, key, _model = llm._resolve("auto")
    assert name == "gemini"
    assert key == "configured-secret"
    assert "configured-secret" not in json_text(llm.provider_status())


def json_text(value):
    return str(value)

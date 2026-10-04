from agents.base_agent import BaseAgent
import pytest


def test_agent_uses_injected_local_model_override():
    import config
    inj = {"model_overrides": {"quant_analyst": config.LOCAL_LLM_MODEL_THINKING}}
    a = BaseAgent(name="t", role="quant_analyst", system_prompt="p",
                  model_key="quant_analyst", injection=inj)
    assert a.api_key == ""
    assert a.model == config.LOCAL_LLM_MODEL_THINKING


def test_agent_falls_back_to_config_when_no_injection():
    # No injection bundle → legacy behaviour (reads config). Must not crash.
    a = BaseAgent(name="t", role="quant_analyst", system_prompt="p",
                  model_key="quant_analyst")
    assert isinstance(a.model, str) and a.model


@pytest.mark.asyncio
async def test_agent_forwards_per_call_generation_limits(monkeypatch):
    seen = {}

    async def fake_chat_completion(**kwargs):
        seen.update(kwargs)
        return {"choices": [{"message": {"content": "후보종목: 삼성전자(005930)"}}]}

    monkeypatch.setattr("infra.local_llm_client.chat_completion", fake_chat_completion)
    a = BaseAgent(name="t", role="quant_analyst", system_prompt="p",
                  model_key="quant_analyst")
    reply = await a.think("후보를 고르라", max_tokens=2500,
                          timeout_sec=120, thinking=False)

    assert "005930" in reply
    assert seen["max_tokens"] == 2500
    assert seen["timeout_sec"] == 120
    assert seen["thinking"] is False

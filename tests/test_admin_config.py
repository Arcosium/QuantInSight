"""admin_config — ADMIN 탭의 전역(전체 계정) 설정 스토어 (사장 지시 2026-05-22).

각 에이전트·뉴스 크롤러 모델 오버라이드 + 뉴스 크롤 주기를 data/admin_config.json 에 저장,
모든 계정에 적용. 모델은 재시작 반영, 크롤 주기는 즉시 반영.
"""
import importlib
from pathlib import Path

import pytest

import infra.admin_config as ac
import config


@pytest.fixture(autouse=True)
def _isolate_admin_config(tmp_path, monkeypatch):
    """실 data/admin_config.json 격리 (2026-07-22).

    admin_config._PATH 는 QUANTINSIGHT_DATA_DIR 을 타지 않는 하드코딩 경로라, 격리 없이
    pytest 를 돌리면 아래 _reset() 이 **사장의 라이브 ADMIN 설정**(에이전트 모델 오버라이드·
    뉴스 크롤 주기)을 실제로 초기화해 버린다. _PATH 는 _read()/set_config() 가 호출 시점에
    읽는 모듈 전역이므로 여기 한 곳만 갈아끼우면 전 경로가 tmp 로 간다."""
    monkeypatch.setattr(ac, "_PATH", tmp_path / "admin_config.json")


def _reset():
    ac.set_config(model_overrides={}, news_crawl_interval_sec=0)


def test_model_override_roundtrip():
    _reset()
    ac.set_config(model_overrides={"quant_analyst": config.LOCAL_LLM_MODEL_THINKING})
    assert ac.get_model_override("quant_analyst") == config.LOCAL_LLM_MODEL_THINKING
    assert ac.get_model_override("macro_analyst") == ""  # 미설정
    _reset()


def test_empty_value_removes_override():
    _reset()
    ac.set_config(model_overrides={"trader": "  "})
    assert ac.get_model_override("trader") == ""  # 빈/공백 → 오버라이드 없음
    _reset()


def test_crawl_interval_default_and_override():
    _reset()
    assert ac.news_crawl_interval(900) == 900   # 0/미설정 → 기본값
    ac.set_config(news_crawl_interval_sec=600)
    assert ac.news_crawl_interval(900) == 600
    _reset()
    assert ac.news_crawl_interval(900) == 900


def test_base_agent_uses_override():
    _reset()
    ac.set_config(model_overrides={"quant_analyst": config.LOCAL_LLM_MODEL_THINKING})
    from agents.base_agent import BaseAgent
    a = BaseAgent(name="t", role="quant_analyst", model_key="quant_analyst", system_prompt="x")
    assert a.model == config.LOCAL_LLM_MODEL_THINKING
    _reset()
    b = BaseAgent(name="t2", role="quant_analyst", model_key="quant_analyst", system_prompt="x")
    assert b.model != config.LOCAL_LLM_MODEL_THINKING  # 기본값으로 복귀


def test_foreign_model_override_is_rejected():
    _reset()
    ac.set_config(model_overrides={"quant_analyst": "foreign-model"})
    assert ac.get_model_override("quant_analyst") == ""

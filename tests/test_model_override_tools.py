"""매크로 리서치 tool 모델 오버라이드 (사장 지시 2026-05-30).

버그: deep_research(매크로)는 BaseAgent 가 아니라 tool 함수라 config.MODEL_ASSIGNMENTS 만 읽고
ADMIN 모델 오버라이드(admin_config)를 '무시'했다. → 사장이 대시보드에서 모델을 바꿔도 늘 config
기본값으로 동작. resolve_model 로 통일해 수정.
(사장 지시 2026-06-04: 뉴스 분류기(llm_classify_articles) 폐지 — 관련 오버라이드 테스트 제거.)
"""
import config
import pytest
from infra import admin_config


@pytest.fixture
def tmp_admin(monkeypatch, tmp_path):
    # 실제 data/admin_config.json 을 건드리지 않도록 임시 경로로
    monkeypatch.setattr(admin_config, "_PATH", tmp_path / "admin_config.json")
    return admin_config


def test_resolve_model_prefers_override(tmp_admin):
    tmp_admin.set_config(model_overrides={"macro_researcher": config.LOCAL_LLM_MODEL})
    assert tmp_admin.resolve_model("macro_researcher", "fallback") == config.LOCAL_LLM_MODEL


def test_resolve_model_falls_back_to_config(tmp_admin):
    tmp_admin.set_config(model_overrides={})  # 오버라이드 없음
    # config 기본값으로 폴백
    assert tmp_admin.resolve_model("macro_researcher", "fallback") == \
        config.MODEL_ASSIGNMENTS["macro_researcher"]


def test_resolve_model_default_when_unknown_key(tmp_admin):
    tmp_admin.set_config(model_overrides={})
    assert tmp_admin.resolve_model("no_such_key", config.LOCAL_LLM_MODEL) == \
        config.LOCAL_LLM_MODEL

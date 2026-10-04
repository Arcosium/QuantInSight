"""포트폴리오기획팀장 = 강한 매도 반론 + 고정 1회 보류 정책."""
import agents.specialists as specialists
import config
from agents.specialists import create_fund_planner, format_thesis_reminder


def test_legacy_veto_function_stays_removed():
    """과거 무기한 거부 함수는 되살리지 않고, 새 보류 정책은 별도 모듈에 둔다."""
    assert not hasattr(specialists, "apply_thesis_veto"), \
        "apply_thesis_veto 는 폐지됐어야 한다 (강제 거부권 금지, 권고만 허용)"


def test_veto_config_keys_removed():
    """거부권 토글·노이즈밴드 설정 키가 제거됐다 (상수·튜너블 양쪽)."""
    assert not hasattr(config, "THESIS_VETO_ENABLED")
    assert not hasattr(config, "THESIS_NOISE_BAND_PCT")
    assert "THESIS_VETO_ENABLED" not in config.STRATEGY_TUNABLE_KEYS
    assert "THESIS_NOISE_BAND_PCT" not in config.STRATEGY_TUNABLE_KEYS
    # 기본값에도 잔재가 없어야 한다.
    assert "THESIS_VETO_ENABLED" not in config.STRATEGY_DEFAULTS
    assert "THESIS_NOISE_BAND_PCT" not in config.STRATEGY_DEFAULTS


def test_reminder_exposes_strong_sell_objection():
    theses = {
        "006400": {
            "entry_ts": "2026-06-08 09:00:00",
            "entry_price": 100000.0,
            "target_price": 110000.0,
            "stop_price": 93000.0,
            "planned_hold_hours": 48,
            "entry_reason": "2차전지 비중 확대",
            "source_agent": "포트폴리오기획팀장",
        },
    }
    holdings = [{"code": "006400", "name": "삼성SDI", "qty": 1, "cur_price": 101000.0}]
    out = format_thesis_reminder(theses, holdings, now_iso="2026-06-08 11:00:00")
    assert "강한 매도 반론" in out
    assert "한 사이클 보류" in out


def test_fund_planner_persona_has_bounded_deferral():
    a = create_fund_planner(injection={"uid": 1})
    assert "거부권" not in a.system_prompt
    assert "한 사이클 보류" in a.system_prompt
    assert "다음 정기 사이클" in a.system_prompt
    # plan 모드(목표·손절·계획기간 작성)는 그대로 유지된다.
    assert "목표가" in a.system_prompt and "손절가" in a.system_prompt

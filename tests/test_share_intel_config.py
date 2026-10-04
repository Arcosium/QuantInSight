"""ADMIN 인텔리전스 공유 설정 키 — 기본값 + 튜너블 카탈로그 등재."""
import config

def test_share_flags_defaults():
    assert config.SHARE_MARKET_INTELLIGENCE is True
    # 2026-07-22: 120초는 생산자 게시보다 짧아 소비자가 매번 자체계산으로 떨어졌다.
    assert config.SHARE_PRODUCER_WAIT_SEC == 420
    assert config.SHARE_STALE_OK_SEC == 30 * 60

def test_share_keys_in_tunable_catalog():
    for k in ("SHARE_MARKET_INTELLIGENCE", "SHARE_PRODUCER_WAIT_SEC", "SHARE_STALE_OK_SEC"):
        assert k in config.STRATEGY_TUNABLE_KEYS, f"{k} missing from STRATEGY_TUNABLE_KEYS"
        assert k in config.STRATEGY_KEY_META, f"{k} missing from STRATEGY_KEY_META"
        assert k in config.STRATEGY_KEY_EFFECT, f"{k} missing from STRATEGY_KEY_EFFECT"

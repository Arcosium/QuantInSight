"""공통 픽스처.

실거래 검증 로직(`guardrails._check_single_order`, `main_swarm._affordable_one_share`)은
``import runtime; runtime.get(KEY)`` 로 활성 전략 프리셋 값을 읽는다. 테스트가 디스크에
저장된 전략 상태(`data/strategy_state.json`)에 의존하면 비결정적이 되므로, 여기서
``runtime.get`` 을 **고정된 보수형 한도 맵**으로 패치한다. 두 모듈 모두 모듈 객체를
import 하므로 `runtime.get` 한 곳만 패치하면 양쪽에 적용된다.
"""
import os
import sys
import tempfile
from pathlib import Path

# 테스트 격리: runtime 등 일부 모듈은 import 시점에 data/ 를 생성·시드한다(실 data/ 오염).
# **프로젝트 모듈 import 이전에** 프로세스 전역 데이터 디렉터리를 임시 경로로 지정한다.
os.environ.setdefault("QUANTINSIGHT_DATA_DIR", tempfile.mkdtemp(prefix="qis-test-data-"))

import pytest
from infra import auth_store as _auth_store

# 저장소 루트를 import path 에 추가 (pytest를 어디서 호출하든 동작).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 테스트가 가정하는 결정론적 한도 (config.py 의 'balanced' 프리셋과 동일값).
FIXED_LIMITS = {
    "CONSERVATIVE_MDD": 0.05,
    "CONSERVATIVE_STOCK_RATIO": 0.15,
    "MIN_CASH_BUFFER": 1.10,
    "MAX_CYCLE_BUDGET_RATIO": 0.25,
}


@pytest.fixture(autouse=True)
def isolate_fx_cache(tmp_path, monkeypatch):
    """2026-09-25: 잔고 계산 테스트(test_overseas_settled_curve 등)가 가짜 기준환율 1500 을
    set_usdkrw 로 라이브 data/usdkrw_fx.json 에 썼다. 9/16 재시작 뒤 모든 달러 평가가 1,500원에
    고정된 원인이다(실제 약 1,356원). 모든 테스트에서 캐시 경로와 메모리 값을 격리한다."""
    import tools.market_data as _md
    monkeypatch.setattr(_md, "_FX_CACHE_PATH", tmp_path / "usdkrw_fx.json")
    monkeypatch.setattr(_md, "_LAST_FX", 0.0)


@pytest.fixture(autouse=True)
def fixed_runtime_limits(monkeypatch):
    """모든 테스트에서 runtime.get 을 고정 한도로 대체 (없는 키는 config 폴백)."""
    import runtime as _rt

    real_get = _rt.get

    def _patched(key, default=None, uid=None):
        if key in FIXED_LIMITS:
            return FIXED_LIMITS[key]
        return real_get(key, default, uid=uid)

    monkeypatch.setattr(_rt, "get", _patched)
    return FIXED_LIMITS


@pytest.fixture
def isolate_orchestrator_data(tmp_path, monkeypatch):
    """오케스트레이터를 **전체 생성**하는 테스트 전용 격리.

    ArquantOrchestrator.__init__/메서드는 auth_store.init()(→ 스키마 마이그레이션)·
    user_paths.equity_path(→ data/<uid>/)·api_cost rollup 을 건드린다. 격리가 없으면
    `pytest` 실행만으로 실 credential DB 가 마이그레이션되거나 운영 데이터가 오염된다
    (test_orchestrator_ctx·test_ceo_directive_routing 에서 실제 회귀 이력 있음).
    strategy_history 는 conftest 상단의 QUANTINSIGHT_DATA_DIR(runtime) 로 이미 격리됨."""
    from infra import user_paths as _user_paths
    from agents import base_agent as _base_agent
    for _n, _v in [("_DATA_DIR", tmp_path), ("_DB_PATH", tmp_path / "auth.db"),
                   ("_FERNET_KEY_PATH", tmp_path / ".fernet.key"),
                   ("_AUDIT_PATH", tmp_path / "auth_audit.log"),
                   ("_INITED", False), ("_FERNET", None),
                   ("_FERNET_RAW", None), ("_BIDX_KEY", None)]:
        monkeypatch.setattr(_auth_store, _n, _v, raising=False)
    monkeypatch.setattr(_user_paths, "_DATA_DIR", tmp_path, raising=False)
    monkeypatch.setattr(_base_agent, "_COST_ROLLUP_PATH",
                        tmp_path / "api_cost_rollup.json", raising=False)


@pytest.fixture
def fresh_auth(tmp_path, monkeypatch):
    """격리된 임시 DB/키로 auth_store 초기화 (모듈 글로벌 리셋). 인증 테스트 공용."""
    for _n, _v in [("_DATA_DIR", tmp_path), ("_DB_PATH", tmp_path / "auth.db"),
                   ("_FERNET_KEY_PATH", tmp_path / ".fernet.key"),
                   ("_AUDIT_PATH", tmp_path / "auth_audit.log"),
                   ("_INITED", False), ("_FERNET", None),
                   ("_FERNET_RAW", None), ("_BIDX_KEY", None)]:
        monkeypatch.setattr(_auth_store, _n, _v, raising=False)
    _auth_store.init()
    return _auth_store

"""TASK A 회귀 테스트: Coresight admin 전용 게이트 + Coresight 수신함 pending→approve→standing_directive.

커버 범위:
  - 비관리자 query_coresight → 빈/거부 결과 (데이터 누출 없음)
  - 관리자 query_coresight → 실제 RAG 경로 진입 (설정/파일 없어도 빈 결과, 크래시 없음)
  - pending enqueue + admin approve → standing_directive 로 전환 (해당 uid 전용)
  - reject → 큐에서 제거
  - uid 간 격리 (A의 pending 이 B에 보이지 않음)
  - 비관리자 endpoint → 403
"""
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


# ─── 공통 픽스처 ────────────────────────────────────────────────────────────────

@pytest.fixture
def isolated_inbox(tmp_path, monkeypatch):
    """coresight_inbox 의 profiles 경로를 tmp_path 로 격리."""
    import infra.coresight_inbox as ci
    monkeypatch.setattr(ci, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(ci, "_DATA_DIR", tmp_path)
    return ci


@pytest.fixture
def isolated_directives(tmp_path, monkeypatch):
    """standing_directives 의 profiles 경로를 같은 tmp_path 로 격리."""
    import infra.standing_directives as sd
    monkeypatch.setattr(sd, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(sd, "_DATA_DIR", tmp_path)
    return sd


@pytest.fixture
def no_external_coresight(monkeypatch):
    """query_coresight 의 **외부 소스**(ArcAI.ve 위키 HTTP :8080 · arcembed 임베딩 :8765)를 끊는다.

    .env 의 CORESIGHT_RAG_MODE=hybrid 때문에 운영에선 _rank_hybrid 가 돈다. 이 테스트들이
    검증하려는 건 'admin 게이트 통과 후 CORESIGHT_PATH 의 매칭 파일이 결과에 나온다'인데,
    라이브 위키가 떠 있으면 위키 가중치(2.0) 항목들이 top_k 를 채워 tmp 파일이 밀려나
    결과가 **네트워크 상태에 따라** 달라졌다(전체 실행에서 실패한 원인).
    hybrid 경로는 그대로 태우되(운영 모드 유지) 외부 소스만 '미가용'으로 만들면,
    코드의 fail-soft 폴백까지 함께 검증되면서 결정론적이 된다."""
    import tools.coresight_rag as cr
    monkeypatch.setattr(cr, "_wiki_search", lambda *a, **k: [])
    monkeypatch.setattr(cr, "_arcembed", lambda: None)
    return cr


@pytest.fixture
def admin_uid():
    return 1


@pytest.fixture
def nonadmin_uid():
    return 2


def _patch_admin(monkeypatch, admin_uid_value: int):
    """infra.auth_store.is_admin を패치해 admin_uid_value만 True로 반환."""
    import infra.auth_store as _as

    def _fake_is_admin(uid):
        return uid == admin_uid_value

    monkeypatch.setattr(_as, "is_admin", _fake_is_admin)


# ─── A-1: query_coresight — 비관리자 deny ──────────────────────────────────────

@pytest.mark.asyncio
async def test_query_coresight_nonadmin_returns_benign(monkeypatch):
    """비관리자 활성 계정 → 빈/거부 문자열 반환, 데이터 누출 없음."""
    import tools.coresight_rag as cr

    # 활성 계정을 비관리자로 패치 (Phase 2: _is_admin_active(uid) 시그니처)
    monkeypatch.setattr(cr, "_is_admin_active", lambda *a, **k: False)

    result = await cr.query_coresight("테스트 쿼리")
    assert "[Coresight] 비활성" in result
    # 실제 데이터 내용이 포함되지 않아야 한다
    assert "검색 결과" not in result


@pytest.mark.asyncio
async def test_query_coresight_nonadmin_no_exception(monkeypatch):
    """비관리자 호출 시 예외 발생 없음 (fail-soft)."""
    import tools.coresight_rag as cr

    monkeypatch.setattr(cr, "_is_admin_active", lambda *a, **k: False)

    # 예외 없이 문자열 반환되어야 한다
    try:
        result = await cr.query_coresight("아무 쿼리")
    except Exception as e:
        pytest.fail(f"query_coresight raised unexpectedly for non-admin: {e}")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_query_coresight_admin_path_no_crash(monkeypatch, tmp_path, no_external_coresight):
    """관리자 활성 계정 → admin 경로 진입. Coresight 파일 없어도 크래시 없음."""
    import tools.coresight_rag as cr

    monkeypatch.setattr(cr, "_is_admin_active", lambda *a, **k: True)
    # CORESIGHT_PATH 를 빈 tmp 디렉토리로 패치
    import config as _cfg
    monkeypatch.setattr(_cfg, "CORESIGHT_PATH", str(tmp_path))

    result = await cr.query_coresight("전략 조회")
    assert isinstance(result, str)
    assert "[Coresight] 비활성" not in result  # 관리자니까 거부 메시지 아님


@pytest.mark.asyncio
async def test_query_coresight_admin_finds_matching_file(monkeypatch, tmp_path,
                                                         no_external_coresight):
    """관리자 활성 계정 + 매칭 파일 → 검색 결과 반환."""
    import tools.coresight_rag as cr
    import config as _cfg

    monkeypatch.setattr(cr, "_is_admin_active", lambda *a, **k: True)
    monkeypatch.setattr(_cfg, "CORESIGHT_PATH", str(tmp_path))

    # 매칭 파일 생성
    (tmp_path / "test_strategy.json").write_text(
        json.dumps({"summary": "달러 MMF 전략 분석 결과"}), encoding="utf-8"
    )

    result = await cr.query_coresight("달러 전략")
    assert "Coresight 검색 결과" in result
    # title 추출 규칙: filename.split("_", 1)[-1] → "test_strategy.json" → "strategy"
    assert "strategy" in result


# ─── A-2: _is_admin_active / _get_active_uid 오류 내성 ─────────────────────────

@pytest.mark.asyncio
async def test_query_coresight_credentials_error_is_denied(monkeypatch):
    """credentials 조회 실패 시 deny-by-default (비활성 반환)."""
    import tools.coresight_rag as cr

    # _is_admin_active 가 예외를 던져도 query_coresight 는 크래시하지 않아야 한다
    def _raise(*a, **k):
        raise RuntimeError("credentials unavailable")

    monkeypatch.setattr(cr, "_is_admin_active", _raise)

    result = await cr.query_coresight("쿼리")
    assert "[Coresight] 비활성" in result


# ─── A-3: coresight_inbox — 비관리자 deny ──────────────────────────────────────

def test_scan_and_enqueue_nonadmin_returns_zero(isolated_inbox, monkeypatch):
    """비관리자 uid → scan_and_enqueue 는 0 반환 (no-op, fail-soft)."""
    ci = isolated_inbox
    import infra.auth_store as _as
    monkeypatch.setattr(_as, "is_admin", lambda uid: False)

    result = ci.scan_and_enqueue(uid=999)
    assert result == 0


def test_list_pending_nonadmin_returns_empty(isolated_inbox, monkeypatch):
    """비관리자 uid → list_pending 은 [] 반환."""
    ci = isolated_inbox
    import infra.auth_store as _as
    monkeypatch.setattr(_as, "is_admin", lambda uid: False)

    result = ci.list_pending(uid=999)
    assert result == []


def test_approve_nonadmin_returns_false(isolated_inbox, monkeypatch):
    """비관리자 uid → approve 는 False 반환."""
    ci = isolated_inbox
    import infra.auth_store as _as
    monkeypatch.setattr(_as, "is_admin", lambda uid: False)

    result = ci.approve(uid=999, item_id="someitem")
    assert result is False


def test_reject_nonadmin_returns_false(isolated_inbox, monkeypatch):
    """비관리자 uid → reject 는 False 반환."""
    ci = isolated_inbox
    import infra.auth_store as _as
    monkeypatch.setattr(_as, "is_admin", lambda uid: False)

    result = ci.reject(uid=999, item_id="someitem")
    assert result is False


# ─── A-4: scan_and_enqueue — 신호 탐지 ─────────────────────────────────────────

def _make_coresight_file(directory: Path, name: str, data: dict) -> Path:
    p = directory / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_scan_enqueues_valid_directive(isolated_inbox, monkeypatch, tmp_path):
    """타입+instruction 있는 파일 → pending 에 enqueue."""
    ci = isolated_inbox
    import infra.auth_store as _as
    import config as _cfg

    admin_uid = 10
    monkeypatch.setattr(_as, "is_admin", lambda uid: uid == admin_uid)
    monkeypatch.setattr(_cfg, "CORESIGHT_PATH", str(tmp_path))

    _make_coresight_file(tmp_path, "sig1.json", {
        "type": "directive",
        "instruction": "달러 자산 비중 70% 유지 전략 실행",
        "ts": "2026-05-19 10:00:00"
    })

    count = ci.scan_and_enqueue(admin_uid)
    assert count == 1

    pending = ci.list_pending(admin_uid)
    assert len(pending) == 1
    assert "달러 자산 비중" in pending[0]["text"]
    assert pending[0]["status"] == "pending"
    assert pending[0]["label"] == "Coresight 제안(미승인)"


def test_scan_ignores_noninvestment_file(isolated_inbox, monkeypatch, tmp_path):
    """투자 로직과 무관한 파일 → enqueue 안 함 (fail-closed)."""
    ci = isolated_inbox
    import infra.auth_store as _as
    import config as _cfg

    admin_uid = 11
    monkeypatch.setattr(_as, "is_admin", lambda uid: uid == admin_uid)
    monkeypatch.setattr(_cfg, "CORESIGHT_PATH", str(tmp_path))

    _make_coresight_file(tmp_path, "misc.json", {
        "type": "log",
        "content": "서버 시작 완료 2026-05-19"
    })

    count = ci.scan_and_enqueue(admin_uid)
    assert count == 0
    assert ci.list_pending(admin_uid) == []


def test_scan_idempotent(isolated_inbox, monkeypatch, tmp_path):
    """같은 파일 두 번 스캔 → 1건만 enqueue (멱등)."""
    ci = isolated_inbox
    import infra.auth_store as _as
    import config as _cfg

    admin_uid = 12
    monkeypatch.setattr(_as, "is_admin", lambda uid: uid == admin_uid)
    monkeypatch.setattr(_cfg, "CORESIGHT_PATH", str(tmp_path))

    _make_coresight_file(tmp_path, "dup.json", {
        "type": "signal",
        "instruction": "포트폴리오 리밸런싱 실행"
    })

    ci.scan_and_enqueue(admin_uid)
    ci.scan_and_enqueue(admin_uid)

    pending = ci.list_pending(admin_uid)
    assert len(pending) == 1


# ─── A-5: approve → standing_directive ─────────────────────────────────────────

def test_approve_converts_to_standing_directive(isolated_inbox, isolated_directives, monkeypatch, tmp_path):
    """admin approve → standing_directive 로 전환. uid 전용."""
    ci = isolated_inbox
    sd = isolated_directives
    import infra.auth_store as _as
    import config as _cfg

    # 두 픽스처가 같은 tmp_path 공유되도록 profiles 경로 맞추기
    from pathlib import Path as P
    monkeypatch.setattr(ci, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(sd, "_PROFILES_DIR", tmp_path / "profiles")

    admin_uid = 20
    monkeypatch.setattr(_as, "is_admin", lambda uid: uid == admin_uid)
    monkeypatch.setattr(_cfg, "CORESIGHT_PATH", str(tmp_path))

    _make_coresight_file(tmp_path, "approve_test.json", {
        "type": "directive",
        "instruction": "달러 MMF 핵심 축으로 설정"
    })

    ci.scan_and_enqueue(admin_uid)
    pending = ci.list_pending(admin_uid)
    assert len(pending) == 1
    item_id = pending[0]["id"]

    # 승인
    ok = ci.approve(admin_uid, item_id)
    assert ok is True

    # standing_directive 에 추가되었는지 확인
    directives = sd.load(admin_uid)
    assert any("달러 MMF 핵심 축" in d["text"] for d in directives)
    assert any("[Coresight 유래]" in d["text"] for d in directives)


def test_approve_updates_status(isolated_inbox, monkeypatch, tmp_path):
    """approve 후 item status 가 'approved' 로 변경된다."""
    ci = isolated_inbox
    import infra.auth_store as _as
    import config as _cfg
    import infra.standing_directives as sd

    monkeypatch.setattr(sd, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(ci, "_PROFILES_DIR", tmp_path / "profiles")

    admin_uid = 21
    monkeypatch.setattr(_as, "is_admin", lambda uid: uid == admin_uid)
    monkeypatch.setattr(_cfg, "CORESIGHT_PATH", str(tmp_path))

    _make_coresight_file(tmp_path, "st.json", {
        "type": "directive",
        "instruction": "섹터 로테이션 전략 적용"
    })
    ci.scan_and_enqueue(admin_uid)
    pending = ci.list_pending(admin_uid)
    item_id = pending[0]["id"]

    ci.approve(admin_uid, item_id)

    # list_pending 은 approved 항목을 제외해야 한다
    still_pending = ci.list_pending(admin_uid)
    assert all(i["id"] != item_id for i in still_pending)


# ─── A-6: reject ─────────────────────────────────────────────────────────────────

def test_reject_removes_from_queue(isolated_inbox, monkeypatch, tmp_path):
    """reject → 큐에서 영구 제거."""
    ci = isolated_inbox
    import infra.auth_store as _as
    import config as _cfg

    admin_uid = 30
    monkeypatch.setattr(_as, "is_admin", lambda uid: uid == admin_uid)
    monkeypatch.setattr(_cfg, "CORESIGHT_PATH", str(tmp_path))

    _make_coresight_file(tmp_path, "rej.json", {
        "type": "signal",
        "instruction": "리스크 헤지 포지션 취득"
    })
    ci.scan_and_enqueue(admin_uid)
    pending = ci.list_pending(admin_uid)
    assert len(pending) == 1
    item_id = pending[0]["id"]

    ok = ci.reject(admin_uid, item_id)
    assert ok is True
    assert ci.list_pending(admin_uid) == []


def test_reject_nonexistent_returns_false(isolated_inbox, monkeypatch):
    """없는 item_id reject → False."""
    ci = isolated_inbox
    import infra.auth_store as _as
    monkeypatch.setattr(_as, "is_admin", lambda uid: True)

    ok = ci.reject(uid=50, item_id="nonexistent_id_xyz")
    assert ok is False


# ─── A-7: uid 간 격리 ─────────────────────────────────────────────────────────────

def test_pending_isolation_between_uids(isolated_inbox, monkeypatch, tmp_path):
    """uid A의 pending 이 uid B에 보이지 않아야 한다."""
    ci = isolated_inbox
    import infra.auth_store as _as
    import config as _cfg

    uid_a, uid_b = 40, 41
    monkeypatch.setattr(_as, "is_admin", lambda uid: uid in (uid_a, uid_b))
    monkeypatch.setattr(_cfg, "CORESIGHT_PATH", str(tmp_path))

    _make_coresight_file(tmp_path, "iso.json", {
        "type": "directive",
        "instruction": "A용 포트폴리오 전략"
    })

    # A만 스캔
    ci.scan_and_enqueue(uid_a)

    # B 에는 보이지 않아야 한다
    assert ci.list_pending(uid_b) == []
    assert len(ci.list_pending(uid_a)) == 1


def test_approve_only_affects_own_uid(isolated_inbox, isolated_directives, monkeypatch, tmp_path):
    """uid A가 승인해도 uid B의 standing_directive 에 반영되지 않는다."""
    ci = isolated_inbox
    sd = isolated_directives
    import infra.auth_store as _as
    import config as _cfg

    monkeypatch.setattr(ci, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(sd, "_PROFILES_DIR", tmp_path / "profiles")

    uid_a, uid_b = 50, 51
    monkeypatch.setattr(_as, "is_admin", lambda uid: uid in (uid_a, uid_b))
    monkeypatch.setattr(_cfg, "CORESIGHT_PATH", str(tmp_path))

    _make_coresight_file(tmp_path, "iso2.json", {
        "type": "directive",
        "instruction": "A 전용 자산 배분 전략"
    })

    ci.scan_and_enqueue(uid_a)
    pending_a = ci.list_pending(uid_a)
    ci.approve(uid_a, pending_a[0]["id"])

    # B 의 standing_directive 는 비어있어야 한다
    assert sd.load(uid_b) == []
    # A 의 standing_directive 에는 있어야 한다
    assert any("A 전용 자산 배분" in d["text"] for d in sd.load(uid_a))


# ─── A-8: app.py 엔드포인트 — 비관리자 403 ─────────────────────────────────────

@pytest.fixture
def test_client(monkeypatch):
    """FastAPI TestClient. 인증 미들웨어 우회를 위해 state.user_id 를 직접 설정."""
    from fastapi.testclient import TestClient
    from server.app import app
    return TestClient(app, raise_server_exceptions=False)


def _inject_user(client, uid: int):
    """요청마다 state.user_id 를 주입하기 위한 미들웨어 패치 대신,
    session lookup을 패치해 특정 uid 로 인증되게 한다."""
    import infra.auth_store as _as

    def _fake_lookup(token):
        if token == "test_token":
            return uid
        return None

    return _fake_lookup


def test_coresight_pending_endpoint_403_for_nonadmin(test_client, monkeypatch):
    """비관리자 uid → GET /api/coresight/pending → 403."""
    import infra.auth_store as _as

    nonadmin_uid = 100
    from infra import cloudflare_access
    monkeypatch.setattr(cloudflare_access, "verify", lambda token: (nonadmin_uid, 9999999999))
    monkeypatch.setattr(_as, "is_admin", lambda uid: False)

    resp = test_client.get("/api/coresight/pending",
                           headers={"Cf-Access-Jwt-Assertion": "verified-in-test"})
    assert resp.status_code == 403


def test_coresight_approve_endpoint_403_for_nonadmin(test_client, monkeypatch):
    """비관리자 uid → POST /api/coresight/approve → 403."""
    import infra.auth_store as _as

    nonadmin_uid = 101
    from infra import cloudflare_access
    monkeypatch.setattr(cloudflare_access, "verify", lambda token: (nonadmin_uid, 9999999999))
    monkeypatch.setattr(_as, "is_admin", lambda uid: False)

    resp = test_client.post("/api/coresight/approve",
                            json={"item_id": "abc"},
                            headers={"Cf-Access-Jwt-Assertion": "verified-in-test"})
    assert resp.status_code == 403


def test_coresight_reject_endpoint_403_for_nonadmin(test_client, monkeypatch):
    """비관리자 uid → POST /api/coresight/reject → 403."""
    import infra.auth_store as _as

    nonadmin_uid = 102
    from infra import cloudflare_access
    monkeypatch.setattr(cloudflare_access, "verify", lambda token: (nonadmin_uid, 9999999999))
    monkeypatch.setattr(_as, "is_admin", lambda uid: False)

    resp = test_client.post("/api/coresight/reject",
                            json={"item_id": "abc"},
                            headers={"Cf-Access-Jwt-Assertion": "verified-in-test"})
    assert resp.status_code == 403


def test_coresight_pending_endpoint_admin_ok(test_client, monkeypatch):
    """admin uid → GET /api/coresight/pending → 200 (빈 목록도 ok)."""
    import infra.auth_store as _as
    import infra.coresight_inbox as ci
    from pathlib import Path
    import tempfile

    admin_uid = 200
    from infra import cloudflare_access
    monkeypatch.setattr(cloudflare_access, "verify", lambda token: (admin_uid, 9999999999))
    monkeypatch.setattr(_as, "is_admin", lambda uid: uid == admin_uid)
    # tmp profiles dir
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setattr(ci, "_PROFILES_DIR", Path(td) / "profiles")
        resp = test_client.get("/api/coresight/pending",
                               headers={"Cf-Access-Jwt-Assertion": "verified-in-test"})
    assert resp.status_code == 200
    data = resp.json()
    assert "pending" in data


def test_coresight_endpoints_require_auth(test_client):
    """인증 없는 요청 → 401 (세션 쿠키 없음)."""
    resp = test_client.get("/api/coresight/pending")
    assert resp.status_code == 401

"""프로필 엔드포인트 — 세션 인증·비번변경·지시 CRUD·탈퇴."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    import infra.auth_store as a
    for n, v in [("_DATA_DIR", tmp_path), ("_DB_PATH", tmp_path / "auth.db"),
                 ("_FERNET_KEY_PATH", tmp_path / ".fernet.key"),
                 ("_AUDIT_PATH", tmp_path / "audit.log"),
                 ("_INITED", False), ("_FERNET", None),
                 ("_FERNET_RAW", None), ("_BIDX_KEY", None)]:
        monkeypatch.setattr(a, n, v, raising=False)
    a.init()
    uid = a.upsert_user("u1", "OldPassw0rd!!", "AK", "AS", "5012345601",
                         "https://openapi.koreainvestment.com:9443")
    import infra.standing_directives as sd
    monkeypatch.setattr(sd, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(sd, "_DATA_DIR", tmp_path)  # tombstone 격리 — 실 data/ 오염 방지
    import server.app as app_mod
    monkeypatch.setattr(app_mod, "_PROFILES_DIR", tmp_path / "profiles")  # rmtree 격리 — 실 profiles/<uid> 삭제 방지
    # 2026-06-11 참사 재발 방지: 탈퇴 라우트의 _decommission_uid 가 user_paths.user_dir(uid) 를
    # rmtree 한다 — tmp 격리 없이는 '실' data/<uid>(=data/1)가 통째로 삭제됐다.
    from infra import user_paths
    monkeypatch.setattr(user_paths, "_DATA_DIR", tmp_path / "data")
    c = TestClient(app_mod.app)
    monkeypatch.setattr(app_mod.cloudflare_access, "verify", lambda token: (uid, 9999999999))
    c.headers.update({"Cf-Access-Jwt-Assertion": "verified-in-test"})
    return c, a, uid


def test_password_change_unauth_401():
    import server.app as app_mod
    c = TestClient(app_mod.app)
    assert c.post("/api/profile/password",
                  json={"current": "x", "new": "y"}).status_code == 401


def test_password_change_flow(client):
    c,a,uid=client
    assert c.post("/api/profile/password",json={"current":"x","new":"y"}).status_code==404


def test_directives_crud(client):
    c, a, uid = client
    assert c.get("/api/profile/directives").json()["directives"] == []
    assert c.post("/api/profile/directives",
                  json={"text": "달러 비중 확대"}).status_code == 200
    lst = c.get("/api/profile/directives").json()["directives"]
    assert len(lst) == 1 and lst[0]["text"] == "달러 비중 확대"
    did = lst[0]["id"]
    assert c.delete(f"/api/profile/directives/{did}").status_code == 200
    assert c.get("/api/profile/directives").json()["directives"] == []


def test_delete_account_requires_password(client):
    c,a,uid=client
    assert c.post("/api/profile/delete_account",json={"password":"unused"}).status_code==404
    assert a.get_user_credentials(uid) is not None


@pytest.fixture
def admin_client(tmp_path, monkeypatch):
    import infra.auth_store as a
    for n, v in [("_DATA_DIR", tmp_path), ("_DB_PATH", tmp_path / "auth.db"),
                 ("_FERNET_KEY_PATH", tmp_path / ".fernet.key"),
                 ("_AUDIT_PATH", tmp_path / "audit.log"),
                 ("_INITED", False), ("_FERNET", None),
                 ("_FERNET_RAW", None), ("_BIDX_KEY", None)]:
        monkeypatch.setattr(a, n, v, raising=False)
    a.init()
    uid = a.upsert_user("hh09080", "AdminPassw0rd!!", "AK", "AS", "5012345601",
                         "https://openapi.koreainvestment.com:9443", is_admin=True)
    import infra.standing_directives as sd
    monkeypatch.setattr(sd, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(sd, "_DATA_DIR", tmp_path)  # tombstone 격리 — 실 data/ 오염 방지
    import server.app as app_mod
    monkeypatch.setattr(app_mod, "_PROFILES_DIR", tmp_path / "profiles")  # rmtree 격리 — 실 profiles/<uid> 삭제 방지
    # 2026-06-11 참사 재발 방지: 탈퇴 라우트의 _decommission_uid 가 user_paths.user_dir(uid) 를
    # rmtree 한다 — tmp 격리 없이는 '실' data/<uid>(=data/1)가 통째로 삭제됐다.
    from infra import user_paths
    monkeypatch.setattr(user_paths, "_DATA_DIR", tmp_path / "data")
    c = TestClient(app_mod.app)
    monkeypatch.setattr(app_mod.cloudflare_access, "verify", lambda token: (uid, 9999999999))
    c.headers.update({"Cf-Access-Jwt-Assertion": "verified-in-test"})
    return c, a, uid


def test_admin_cannot_self_delete(admin_client):
    c,a,uid=admin_client
    assert c.post("/api/profile/delete_account",json={}).status_code==404
    assert a.get_user_credentials(uid) is not None


# ── Fix 3: /api/profile/credentials tests ────────────────────────────────────

async def _stub_validate_kis_ok(app_key, app_secret, base_url):
    return True, "ok"



async def _stub_validate_kis_fail(app_key, app_secret, base_url):
    return False, "bad"


def test_credentials_partial_update_does_not_clobber_other_creds(client, monkeypatch):
    """POST only kis_account_no — other credential fields must be unchanged."""
    import server.app as app_mod
    monkeypatch.setattr(app_mod, "_validate_kis", _stub_validate_kis_ok)
    c, a, uid = client
    resp = c.post("/api/profile/credentials", json={"kis_account_no": "NEW123"})
    assert resp.status_code == 200, resp.text
    stored = a.get_user_credentials(uid)
    assert stored["kis_account_no"] == "NEW123"
    assert stored["kis_app_key"] == "AK"
    assert stored["kis_app_secret"] == "AS"


def test_credentials_validation_failure_blocks_save(client, monkeypatch):
    """If _validate_kis returns failure, no save should occur."""
    import server.app as app_mod
    monkeypatch.setattr(app_mod, "_validate_kis", _stub_validate_kis_fail)
    c, a, uid = client
    resp = c.post("/api/profile/credentials", json={"kis_app_key": "X"})
    assert resp.status_code == 400, resp.text
    stored = a.get_user_credentials(uid)
    assert stored["kis_app_key"] == "AK"  # unchanged


def test_credentials_strips_whitespace(client, monkeypatch):
    """Whitespace around credential fields must be stripped before saving."""
    import server.app as app_mod
    monkeypatch.setattr(app_mod, "_validate_kis", _stub_validate_kis_ok)
    c, a, uid = client
    resp = c.post("/api/profile/credentials", json={"kis_account_no": "  PADDED  "})
    assert resp.status_code == 200, resp.text
    stored = a.get_user_credentials(uid)
    assert stored["kis_account_no"] == "PADDED"

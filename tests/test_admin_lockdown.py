"""ADMIN = hh09080 영구·단독 — 승격 거부 + 강등 거부 + 부팅 스윕."""
import time
import infra.auth_store as A


def _fresh_db(tmp_path, monkeypatch):
    db = tmp_path / "auth.db"
    # auth_store 는 모듈 상수 _DB_PATH 를 _connect()/init() 가 호출 시점에 읽으므로
    # 이 patch 만으로 깨끗한 temp DB 를 사용한다. _INITED 리셋으로 재-init 강제.
    monkeypatch.setattr(A, "_DB_PATH", str(db), raising=False)
    monkeypatch.setattr(A, "_INITED", False, raising=False)
    return db


def _mk_user(conn, username, is_admin=0):
    now = time.time()
    conn.execute(
        "INSERT INTO users (username, password_enc, password_hash, kis_app_key_enc, "
        "kis_app_secret_enc, kis_account_no_enc, kis_base_url, "
        "dart_key_enc, label, created_at, last_login_at, last_validated_at, is_admin) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (username, "", "h", "", "", "", "", "", username, now, now, 0.0, is_admin))


def test_reject_promote_non_admin_username(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch); A.init()
    with A._DB_LOCK, A._connect() as conn:
        _mk_user(conn, "alice")
        uid = conn.execute("SELECT id FROM users WHERE username='alice'").fetchone()[0]
    assert A.set_admin(uid, True) is False
    assert A.is_admin(uid) is False


def test_reject_demote_hh09080(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch); A.init()
    with A._DB_LOCK, A._connect() as conn:
        _mk_user(conn, "hh09080", is_admin=1)
        uid = conn.execute("SELECT id FROM users WHERE username='hh09080'").fetchone()[0]
    assert A.set_admin(uid, False) is False
    assert A.is_admin(uid) is True


def test_boot_sweep_demotes_stray_admin(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch); A.init()
    with A._DB_LOCK, A._connect() as conn:
        _mk_user(conn, "hh09080", is_admin=0)
        _mk_user(conn, "mallory", is_admin=1)
    A._INITED = False
    A.init()
    with A._DB_LOCK, A._connect() as conn:
        rows = {r[0]: r[1] for r in conn.execute("SELECT username, is_admin FROM users")}
    assert rows["hh09080"] == 1
    assert rows["mallory"] == 0

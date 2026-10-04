"""
ArQuant v1.0 — Auth & Credential Store (사장 피드백 2026-05-16)

Cloudflare Access 제거 → 앱 자체 로그인. 인증/세션 로직은 HYFE_IQC 의 월드퀀트
계정 로그인 방식을 참조한다:
  - SQLite + cryptography.Fernet 대칭 암호화 (KIS·로컬 LLM·DART·계좌번호 암호화)
  - 비밀번호는 argon2id 해시로 저장 (password_hash); blind-index(HMAC) 컬럼으로 계정 복구 지원
  - 불투명 세션 토큰 secrets.token_urlsafe(32), 7일 TTL, 만료 시 자동 삭제

사장 피드백 2026-05-16 (2차): 로그인 정체성을 **사용자가 정한 아이디/비밀번호**로 변경.
  - 등록: 아이디(중복 불가) + 비밀번호(10자 이상·특수문자 1개 이상) + API 키들
  - 로그인: 아이디 + 비밀번호만
  - KIS App Key/Secret·로컬 LLM·DART·계좌번호/Base URL 은 저장되는 자격증명
    (정체성이 아님 — 시스템 구동에 사용)

여러 계정 등록 가능(멀티). 스왐은 단일 프로세스라 로그인한 계정이 봇을 장악.
"""
from __future__ import annotations

import json as _json
import logging
import os
import secrets
import sqlite3
import hashlib
import hmac as _hmac
import threading
import time
from datetime import datetime as _dt, timezone as _tz
from pathlib import Path
from typing import Any, Dict, List, Optional

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHash, VerificationError
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes as _hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

logger = logging.getLogger("AUTH")

_DATA_DIR = Path(__file__).parent.parent / "data"
_DATA_DIR.mkdir(exist_ok=True)
_DB_PATH = _DATA_DIR / "arquant_auth.db"
_FERNET_KEY_PATH = _DATA_DIR / ".fernet.key"
_AUDIT_PATH = _DATA_DIR / "auth_audit.log"   # *.log → .gitignore 로 추적 제외

SESSION_COOKIE = "arquant_session"
SESSION_TTL_SEC = 7 * 24 * 60 * 60  # 7일

_DB_LOCK = threading.RLock()
_FERNET: Optional[Fernet] = None
_FERNET_RAW: Optional[bytes] = None  # _ensure_fernet 에서 _FERNET 와 함께 세팅 — 둘은 항상 동기 유지
_BIDX_KEY: Optional[bytes] = None

# 비밀번호 정책 (사장 피드백 2026-05-16 2차)
PW_MIN_LEN = 10

# 사장 피드백 2026-05-18: 멀티테넌트 안전 — '코드 변경'은 전체 유저 공유 소스에
# 영향을 주므로 ADMIN 계정만 실제 소스 수정+재시작(전체 반영) 권한을 가진다.
# 비관리자 계정은 ops_support_worker가 샌드박스 모드로 동작(프로필 한정 오버라이드).
# .env 시드 계정도 부팅 시 자동으로 ADMIN 으로 표시된다(bootstrap_from_env).
ADMIN_USERNAMES = frozenset({"hh09080"})
VIEWER_MODE = "viewer"
TRADING_MODE = "trading"

# ── 통합 계정 프로필 (사장 지시 2026-07-20) ────────────────────────────────────
# 한 로그인 계정(마스터 행) 아래에 매매 프로필(서브 행)들을 owner_id 로 연결한다.
# 각 프로필 행은 기존처럼 독립 uid·data/<uid>·스웜 루프를 가지며, 로그인 유저는
# active_profile_uid 토글로 보기/제어 대상을 전환한다. 프로필 종류(kind)는 별도
# 컬럼 없이 행의 account_mode·자격증명·base_url 에서 파생한다(상태 이원화 방지).
PROFILE_KIS_REAL = "kis_real"
PROFILE_KIS_PAPER = "kis_paper"
PROFILE_KIND_ORDER = (PROFILE_KIS_REAL, PROFILE_KIS_PAPER)
PROFILE_KIND_LABELS = {PROFILE_KIS_REAL: "KIS 실전투자",
                       PROFILE_KIS_PAPER: "KIS 모의투자"}
# 서브 프로필 행의 username 접미사 — 직접 로그인 불가 식별자. 가입 시 "::" 포함 금지.
PROFILE_USERNAME_SEP = "::"
_PROFILE_SUFFIX = {PROFILE_KIS_REAL: "real", PROFILE_KIS_PAPER: "paper"}


def normalize_account_mode(mode: str) -> str:
    mode = (mode or "").strip().lower()
    if mode == VIEWER_MODE:
        return VIEWER_MODE
    if mode not in (TRADING_MODE, ""):
        raise ValueError("지원하지 않는 계정 모드입니다.")
    return TRADING_MODE


def password_policy_error(pw: str) -> Optional[str]:
    """정책 위반 시 사람이 읽을 메시지, 통과면 None.
    규칙: 10자 이상 + 특수문자(영숫자 아닌 문자) 1개 이상."""
    pw = pw or ""
    if len(pw) < PW_MIN_LEN:
        return f"비밀번호는 최소 {PW_MIN_LEN}자 이상이어야 합니다."
    if not any((not c.isalnum()) and (not c.isspace()) for c in pw):
        return "비밀번호에 특수문자를 1개 이상 포함해야 합니다."
    return None


class FernetKeyLost(RuntimeError):
    """data/.fernet.key 가 사라졌는데 이미 등록 계정이 있는 위험 상태.
    새 키를 만들면 **모든 유저**의 암호화 자격증명이 영구 복호 불능이 되므로
    절대 자동 재생성하지 않고 운영자에게 키 복구를 요구한다 (멀티유저 계정 보호)."""


def _existing_user_count() -> int:
    """Fernet 없이도 안전하게 호출 가능한 가벼운 카운트 (encrypt/decrypt 미사용)."""
    if not _DB_PATH.exists():
        return 0
    try:
        c = sqlite3.connect(_DB_PATH, timeout=5)
        try:
            row = c.execute("SELECT COUNT(*) FROM users").fetchone()
            return int(row[0]) if row else 0
        finally:
            c.close()
    except sqlite3.Error:
        return 0  # users 테이블 자체가 없으면 신규 상태로 간주


# ─── Fernet ───────────────────────────────────────────────────────────────────
def _ensure_fernet() -> None:
    global _FERNET, _FERNET_RAW
    if _FERNET is not None:
        return
    key = (os.environ.get("ARQUANT_FERNET_KEY", "").strip() or "").encode("utf-8") or None
    if not key:
        if _FERNET_KEY_PATH.exists():
            key = _FERNET_KEY_PATH.read_bytes().strip()
        else:
            # 사장 피드백 2026-05-16 (3차): 멀티유저 계정 내구성 — 키가 사라졌는데
            # 이미 계정이 있으면 재생성 금지 (재생성 시 전 유저 자격증명 복호 불능).
            n = _existing_user_count()
            if n > 0:
                logger.critical(
                    "FERNET 키 분실: %s 없음 + 등록 계정 %d개 존재 → 자동 재생성 거부. "
                    "백업된 data/.fernet.key 를 복구하거나 ARQUANT_FERNET_KEY 로 주입하십시오. "
                    "(새 키 생성 시 모든 유저 계정이 영구 복호 불능)",
                    _FERNET_KEY_PATH, n)
                raise FernetKeyLost(
                    f"암호화 키(data/.fernet.key) 분실 — 등록 계정 {n}개 보호를 위해 "
                    f"자동 재생성을 거부했습니다. 키 백업을 복구한 뒤 재시작하세요.")
            key = Fernet.generate_key()
            _FERNET_KEY_PATH.write_bytes(key)
            os.chmod(_FERNET_KEY_PATH, 0o600)
            logger.info("Fernet 키 신규 생성(신규 환경): %s (0600)", _FERNET_KEY_PATH)
    _FERNET_RAW = key
    _FERNET = Fernet(key)


def encrypt(text: str) -> str:
    _ensure_fernet()
    return _FERNET.encrypt((text or "").encode("utf-8")).decode("ascii")


def decrypt(token: str) -> str:
    _ensure_fernet()
    if not token:
        return ""
    try:
        return _FERNET.decrypt(token.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError) as e:
        logger.warning("Fernet 복호 실패(%s) — 빈 문자열 반환", type(e).__name__)
        return ""


def _norm(v: str) -> str:
    """블라인드 인덱스 정규화. 저장·조회 양쪽 모두 bidx()→_norm() 을 거치므로
    정규화는 대칭적이다(매칭 보장은 이 대칭성에서 나온다 — upsert_user 가
    개별 키를 strip 한다고 가정하지 말 것)."""
    return (v or "").strip()


def _bidx_key() -> bytes:
    """Fernet 키에서 HKDF-SHA256 으로 파생한 블라인드 인덱스 전용 키.
    별도 시크릿을 두지 않아 운영 부담이 없다(단, Fernet 키에 결합됨)."""
    global _BIDX_KEY
    if _BIDX_KEY is not None:
        return _BIDX_KEY
    _ensure_fernet()
    if not _FERNET_RAW:
        raise RuntimeError("blind-index 키 파생 불가 — Fernet 키 없음")
    _BIDX_KEY = HKDF(algorithm=_hashes.SHA256(), length=32, salt=None,
                     info=b"arquant-bidx-v1").derive(_FERNET_RAW)
    return _BIDX_KEY


def bidx(value: str) -> str:
    """결정론적 블라인드 인덱스(HMAC-SHA256 hex) — 평문 노출 없이 동치 조회용."""
    return _hmac.new(_bidx_key(), _norm(value).encode("utf-8"),
                     hashlib.sha256).hexdigest()


def audit(event: str, *, username: Optional[str], ip: Optional[str],
          outcome: str, detail: str = "") -> None:
    """인증 감사 로그(JSONL). 절대 키/자격증명 값을 detail 에 넣지 말 것."""
    try:
        rec = {"ts": _dt.now(tz=_tz.utc).isoformat(), "event": event,
               "username": username or "", "ip": ip or "", "outcome": outcome,
               "detail": detail}
        with open(_AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        logger.warning("auth audit 기록 실패(event=%s)", event)


_PH = PasswordHasher()  # argon2id, library defaults (tune later if needed)


def hash_password(pw: str) -> str:
    """평문 비밀번호 → argon2id 해시 문자열."""
    return _PH.hash(pw or "")


def verify_pw_hash(stored_hash: str, pw: str) -> bool:
    """저장된 argon2 해시와 평문 일치 검증. 불일치/빈 해시/손상 해시는 False."""
    if not stored_hash:
        return False
    try:
        return _PH.verify(stored_hash, pw or "")
    except (VerificationError, InvalidHash):  # VerifyMismatchError ⊂ VerificationError
        return False


# ─── DB ───────────────────────────────────────────────────────────────────────
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


_INITED = False


def init() -> None:
    global _INITED
    if _INITED:
        return
    with _DB_LOCK, _connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_enc TEXT NOT NULL,   -- DEPRECATED: 항상 '' (비밀번호는 password_hash=argon2id). 하위호환 위해 컬럼만 유지
                kis_app_key_enc TEXT NOT NULL,
                kis_app_secret_enc TEXT NOT NULL,
                kis_account_no_enc TEXT NOT NULL,
                kis_base_url TEXT NOT NULL,
                dart_key_enc TEXT NOT NULL DEFAULT '',
                label TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                last_login_at REAL NOT NULL,
                last_validated_at REAL NOT NULL DEFAULT 0
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )"""
        )
        # ── 마이그레이션: is_admin 컬럼 (사장 피드백 2026-05-18) ──
        # CREATE TABLE IF NOT EXISTS 는 기존 DB에 컬럼을 추가하지 못하므로 ALTER 로 보강.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        # 2026-07: 로컬 LLM 전환으로 per-user LLM 키(구 DeepSeek/OpenRouter) 폐기.
        # 남아있던 llm_key_enc/bidx (및 더 옛 이름) 컬럼을 기존 DB에서 제거.
        for _dead in ("llm_key_enc", "llm_key_bidx",
                      "open" + "router_key_enc", "open" + "router_key_bidx",
                      "deep" + "seek_api_key_enc", "deep" + "seek_api_key_bidx"):
            if _dead in cols:
                try:
                    conn.execute(f"ALTER TABLE users DROP COLUMN {_dead}")
                    cols.discard(_dead)
                    logger.info("auth_store 마이그레이션: users.%s 컬럼 제거(로컬 LLM 전환)", _dead)
                except Exception as e:  # noqa: BLE001
                    logger.warning("auth_store 마이그레이션: %s 제거 실패 — %s", _dead, e)
        if "is_admin" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
            logger.info("auth_store 마이그레이션: users.is_admin 컬럼 추가")
        if "account_mode" not in cols:
            conn.execute(
                "ALTER TABLE users ADD COLUMN account_mode TEXT NOT NULL DEFAULT 'trading'")
            logger.info("auth_store 마이그레이션: users.account_mode 컬럼 추가")
        # 사장 지시 2026-06-08: ADMIN = hh09080 영구·단독. 화이트리스트는 승격하고
        # 그 외 stray is_admin=1 행은 강등(부팅 스윕).
        if ADMIN_USERNAMES:
            qs = ",".join("?" * len(ADMIN_USERNAMES))
            conn.execute(f"UPDATE users SET is_admin=1 WHERE username IN ({qs})",
                         tuple(ADMIN_USERNAMES))
            conn.execute(f"UPDATE users SET is_admin=0 WHERE username NOT IN ({qs})",
                         tuple(ADMIN_USERNAMES))
        else:
            conn.execute("UPDATE users SET is_admin=0")
        for _c in ("password_hash", "kis_app_key_bidx",
                   "kis_app_secret_bidx",
                   "kis_account_no_bidx"):
            if _c not in cols:
                conn.execute(
                    f"ALTER TABLE users ADD COLUMN {_c} TEXT NOT NULL DEFAULT ''")
                logger.info("auth_store 마이그레이션: users.%s 컬럼 추가", _c)
        # 통합 계정 프로필 (사장 지시 2026-07-20): owner_id=마스터 uid(0=독립 로그인 행),
        # active_profile_uid=현재 토글된 프로필 uid(0=자기 자신/자동).
        for _c, _ddl in (("owner_id", "INTEGER NOT NULL DEFAULT 0"),
                         ("active_profile_uid", "INTEGER NOT NULL DEFAULT 0")):
            if _c not in cols:
                conn.execute(f"ALTER TABLE users ADD COLUMN {_c} {_ddl}")
                logger.info("auth_store 마이그레이션: users.%s 컬럼 추가", _c)
    _INITED = True


# ─── User CRUD ────────────────────────────────────────────────────────────────
def username_exists(username: str) -> bool:
    init()
    with _DB_LOCK, _connect() as conn:
        row = conn.execute("SELECT 1 FROM users WHERE username=?",
                           ((username or "").strip(),)).fetchone()
    return row is not None


def upsert_user(username: str, password: str, kis_app_key: str, kis_app_secret: str,
                kis_account_no: str, kis_base_url: str,
                dart_key: str = "", label: str = "", is_admin: bool = False,
                account_mode: str = TRADING_MODE) -> int:
    """username 기준 upsert. 비밀번호는 argon2id 해시로만 저장(password_enc 미사용),
    복구용 블라인드 인덱스도 함께 기록. user_id 반환.

    is_admin: 신규 생성 시에만 적용. 기존 ADMIN 강등 안 함."""
    init()
    now = time.time()
    username = (username or "").strip()
    base_url = (kis_base_url or "https://openapi.koreainvestment.com:9443").strip()
    label = (label or username).strip()
    account_mode = normalize_account_mode(account_mode)
    vals = dict(
        password_hash=hash_password(password),
        kis_app_key_enc=encrypt(kis_app_key),
        kis_app_secret_enc=encrypt(kis_app_secret),
        kis_account_no_enc=encrypt(kis_account_no),
        kis_base_url=base_url,
        dart_key_enc=encrypt(dart_key) if (dart_key or "").strip() else "",
        label=label,
        kis_app_key_bidx=bidx(kis_app_key),
        kis_app_secret_bidx=bidx(kis_app_secret),
        kis_account_no_bidx=bidx(kis_account_no),
    )
    with _DB_LOCK, _connect() as conn:
        row = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
        if row:
            uid = int(row["id"])
            conn.execute(
                """UPDATE users SET password_hash=?, password_enc='',
                   kis_app_key_enc=?, kis_app_secret_enc=?,
                   kis_account_no_enc=?, kis_base_url=?, dart_key_enc=?, label=?,
                   kis_app_key_bidx=?, kis_app_secret_bidx=?,
                   kis_account_no_bidx=?,
                   account_mode=?, last_login_at=?, last_validated_at=? WHERE id=?""",
                (vals["password_hash"], vals["kis_app_key_enc"], vals["kis_app_secret_enc"],
                 vals["kis_account_no_enc"], vals["kis_base_url"],
                 vals["dart_key_enc"], vals["label"], vals["kis_app_key_bidx"],
                 vals["kis_app_secret_bidx"],
                 vals["kis_account_no_bidx"], account_mode, now, now, uid),
            )
            return uid
        adm = 1 if (is_admin or username in ADMIN_USERNAMES) else 0
        cur = conn.execute(
            """INSERT INTO users (username, password_enc, password_hash,
               kis_app_key_enc, kis_app_secret_enc,
               kis_account_no_enc, kis_base_url, dart_key_enc, label,
               kis_app_key_bidx, kis_app_secret_bidx,
               kis_account_no_bidx,
               is_admin, account_mode, created_at, last_login_at, last_validated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (username, "", vals["password_hash"], vals["kis_app_key_enc"],
             vals["kis_app_secret_enc"],
             vals["kis_account_no_enc"], vals["kis_base_url"], vals["dart_key_enc"],
             vals["label"], vals["kis_app_key_bidx"], vals["kis_app_secret_bidx"],
             vals["kis_account_no_bidx"],
             adm, account_mode, now, now, now),
        )
        return int(cur.lastrowid)


def _row_to_creds(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": int(row["id"]),
        "username": row["username"],
        "password": decrypt(row["password_enc"]),
        "password_hash": row["password_hash"] if "password_hash" in row.keys() else "",
        "kis_app_key": decrypt(row["kis_app_key_enc"]),
        "kis_app_secret": decrypt(row["kis_app_secret_enc"]),
        "kis_account_no": decrypt(row["kis_account_no_enc"]),
        "kis_base_url": row["kis_base_url"],
        "dart_key": decrypt(row["dart_key_enc"]) if row["dart_key_enc"] else "",
        "label": row["label"],
        "is_admin": bool(row["is_admin"]) if "is_admin" in row.keys() else False,
        "account_mode": (row["account_mode"] if "account_mode" in row.keys() else TRADING_MODE),
        "owner_id": int(row["owner_id"]) if "owner_id" in row.keys() and row["owner_id"] else 0,
        "active_profile_uid": (int(row["active_profile_uid"])
                               if "active_profile_uid" in row.keys() and row["active_profile_uid"] else 0),
    }


def find_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    init()
    with _DB_LOCK, _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username=?",
                           ((username or "").strip(),)).fetchone()
    return _row_to_creds(row) if row else None


def get_user_credentials(user_id: int) -> Optional[Dict[str, Any]]:
    init()
    with _DB_LOCK, _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (int(user_id),)).fetchone()
    return _row_to_creds(row) if row else None


def find_username_by_factors(kis_account_no: str, kis_app_secret: str) -> Optional[str]:
    """한투 계좌번호 + 한투 App Secret 이 모두 일치하는 단일 유저 아이디 반환.
    블라인드 인덱스 조회 — 전체 복호 없음."""
    if not (_norm(kis_account_no) and _norm(kis_app_secret)):
        return None
    init()
    a, b = bidx(kis_account_no), bidx(kis_app_secret)
    with _DB_LOCK, _connect() as conn:
        row = conn.execute(
            "SELECT id, username, owner_id FROM users WHERE kis_account_no_bidx=? AND "
            "kis_app_secret_bidx=?", (a, b)).fetchone()
    if not row:
        return None
    # 통합 계정: KIS 자격증명이 서브 프로필 행에 있으면 로그인 가능한 마스터 아이디를 돌려준다.
    owner_id = int(row["owner_id"]) if ("owner_id" in row.keys() and row["owner_id"]) else 0
    if owner_id:
        owner = get_user_credentials(owner_id)
        return owner["username"] if owner else None
    return row["username"]


def reset_password_by_factors(username: str, kis_account_no: str,
                              kis_app_secret: str, new_password: str) -> bool:
    """정책 먼저 검사(위반→ValueError, 팩터 정오 무관 — enum 오라클 차단).
    그다음 아이디+계좌번호+App Secret 완전 일치 시에만 재설정. 불일치 시 False."""
    perr = password_policy_error(new_password or "")
    if perr:
        raise ValueError(perr)
    if not (_norm(kis_account_no) and _norm(kis_app_secret)):
        return False
    init()
    a, b = bidx(kis_account_no), bidx(kis_app_secret)
    with _DB_LOCK, _connect() as conn:
        # 통합 계정: 팩터(KIS 계좌+Secret)는 서브 프로필 행에 있을 수 있다 —
        # 팩터 행을 찾고, 그 행(독립) 또는 그 마스터의 username 이 일치할 때
        # '로그인 행'(마스터)의 비밀번호를 재설정한다.
        frow = conn.execute(
            "SELECT id, username, owner_id FROM users WHERE kis_account_no_bidx=? AND "
            "kis_app_secret_bidx=?", (a, b)).fetchone()
        if not frow:
            return False
        owner_id = int(frow["owner_id"]) if ("owner_id" in frow.keys() and frow["owner_id"]) else 0
        login_row_id, login_name = int(frow["id"]), frow["username"]
        if owner_id:
            orow = conn.execute("SELECT id, username FROM users WHERE id=?",
                                (owner_id,)).fetchone()
            if not orow:
                return False
            login_row_id, login_name = int(orow["id"]), orow["username"]
        if login_name != _norm(username):
            return False
        conn.execute("UPDATE users SET password_hash=?, password_enc='' WHERE id=?",
                     (hash_password(new_password), login_row_id))
    return True


def is_admin(user_id: Optional[int]) -> bool:
    """ADMIN 계정 여부. 미상/없음/오류는 모두 False (안전한 기본값 — 비관리자 취급).

    이 한 줄이 멀티테넌트 안전의 핵심: 판단 불가 시 절대 전체 소스 변경 권한을
    부여하지 않는다(default-deny)."""
    if user_id is None:
        return False
    try:
        init()
        with _DB_LOCK, _connect() as conn:
            row = conn.execute("SELECT is_admin FROM users WHERE id=?",
                               (int(user_id),)).fetchone()
        return bool(row and row["is_admin"])
    except Exception as e:
        logger.warning("is_admin 조회 실패(user_id=%s): %s — 비관리자로 처리", user_id, e)
        return False


def is_viewer(user_id: Optional[int]) -> bool:
    if user_id is None:
        return False
    try:
        init()
        with _DB_LOCK, _connect() as conn:
            row = conn.execute("SELECT account_mode FROM users WHERE id=?",
                               (int(user_id),)).fetchone()
        return bool(row and row["account_mode"] == VIEWER_MODE)
    except Exception as e:
        logger.warning("is_viewer 조회 실패(user_id=%s): %s", user_id, e)
        return False




def admin_view_uid() -> Optional[int]:
    """관전 계정이 읽을 대상 uid. 사장 지시 2026-07-20: 관전 기본값은 ADMIN 의
    **KIS 모의투자** 프로필이다(실계좌 노출 방지). 모의 프로필이 없으면 ADMIN 본인 행."""
    init()
    with _DB_LOCK, _connect() as conn:
        row = conn.execute(
            "SELECT id FROM users WHERE username=? AND is_admin=1 LIMIT 1",
            (next(iter(ADMIN_USERNAMES)),)).fetchone()
    if not row:
        return None
    admin_uid = int(row["id"])
    for p in list_profiles(admin_uid):
        if p["kind"] == PROFILE_KIS_PAPER:
            return int(p["uid"])
    return admin_uid


# ─── 통합 계정 프로필 계층 (사장 지시 2026-07-20) ─────────────────────────────
def profile_kind_of(creds: Dict[str, Any]) -> Optional[str]:
    """행(자격증명 dict)의 프로필 종류. 매매 자격이 없는 로그인 전용/관전 행은 None."""
    if not creds:
        return None
    mode = creds.get("account_mode") or TRADING_MODE
    if mode != TRADING_MODE:
        return None
    if (creds.get("kis_app_key") or "") and (creds.get("kis_account_no") or ""):
        return (PROFILE_KIS_PAPER if _is_mock_url(creds.get("kis_base_url") or "")
                else PROFILE_KIS_REAL)
    return None


def list_profiles(login_uid: int) -> List[Dict[str, Any]]:
    """로그인 계정이 토글할 수 있는 매매 프로필 목록(kind 순서 고정).
    마스터 행 자신도 매매 자격이 있으면 프로필로 포함한다."""
    init()
    login_uid = int(login_uid)
    with _DB_LOCK, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM users WHERE id=? OR owner_id=? ORDER BY id",
            (login_uid, login_uid)).fetchall()
    out: List[Dict[str, Any]] = []
    active = 0
    for r in rows:
        c = _row_to_creds(r)
        if int(c["id"]) == login_uid:
            active = int(c.get("active_profile_uid") or 0)
        kind = profile_kind_of(c)
        if not kind:
            continue
        if any(p["kind"] == kind for p in out):
            continue  # kind 당 1개 — 중복 행은 첫 행(작은 uid) 우선
        out.append({"uid": int(c["id"]), "kind": kind,
                    "label": PROFILE_KIND_LABELS.get(kind, kind),
                    "kis_account_no_masked": _mask(c.get("kis_account_no") or "", 4),
                    "is_mock": _is_mock_url(c.get("kis_base_url") or "")})
    out.sort(key=lambda p: PROFILE_KIND_ORDER.index(p["kind"]))
    valid_uids = {p["uid"] for p in out}
    eff = active if active in valid_uids else (out[0]["uid"] if out else login_uid)
    for p in out:
        p["is_active"] = (p["uid"] == eff)
    return out


def resolve_profile_uid(login_uid: int) -> int:
    """로그인 uid → 현재 토글된(활성) 매매 프로필 uid. 프로필이 없으면 자기 자신."""
    profs = list_profiles(int(login_uid))
    for p in profs:
        if p.get("is_active"):
            return int(p["uid"])
    return int(login_uid)


def set_active_profile(login_uid: int, profile_uid: int) -> bool:
    """토글 전환 — profile_uid 가 본인 소유 프로필일 때만 저장. 성공 True."""
    profs = list_profiles(int(login_uid))
    if int(profile_uid) not in {p["uid"] for p in profs}:
        return False
    init()
    with _DB_LOCK, _connect() as conn:
        conn.execute("UPDATE users SET active_profile_uid=? WHERE id=?",
                     (int(profile_uid), int(login_uid)))
    return True


def owned_profile_uids(login_uid: int) -> List[int]:
    """owner_id 로 연결된 서브 프로필 uid 목록 (마스터 자신 제외) — 탈퇴/삭제 정리용."""
    init()
    with _DB_LOCK, _connect() as conn:
        rows = conn.execute("SELECT id FROM users WHERE owner_id=?",
                            (int(login_uid),)).fetchall()
    return [int(r["id"]) for r in rows]


def login_uid_of(uid: int) -> int:
    """프로필 uid → 로그인(마스터) uid. 독립 행이면 자기 자신."""
    c = get_user_credentials(int(uid))
    return int(c["owner_id"]) if c and c.get("owner_id") else int(uid)


def create_subprofile(owner_uid: int, kind: str, *, kis_app_key: str = "",
                      kis_app_secret: str = "", kis_account_no: str = "",
                      kis_base_url: str = "") -> int:
    """마스터 계정 아래 매매 프로필 행 생성. 서브 행은 직접 로그인 불가
    (username 은 '<owner>::<suffix>' + 랜덤 비밀번호). 생성된 uid 반환."""
    owner = get_user_credentials(int(owner_uid))
    if not owner:
        raise ValueError(f"owner uid={owner_uid} 없음")
    if owner.get("owner_id"):
        raise ValueError("서브 프로필 아래에 프로필을 만들 수 없습니다.")
    suffix = _PROFILE_SUFFIX.get(kind)
    if not suffix:
        raise ValueError(f"알 수 없는 프로필 종류: {kind}")
    base_name = f"{owner['username']}{PROFILE_USERNAME_SEP}{suffix}"
    name = base_name
    n = 2
    while username_exists(name):
        name = f"{base_name}{n}"
        n += 1
    uid = upsert_user(
        username=name, password=secrets.token_urlsafe(24) + "!",
        kis_app_key=kis_app_key, kis_app_secret=kis_app_secret,
        kis_account_no=kis_account_no,
        kis_base_url=(kis_base_url or "https://openapi.koreainvestment.com:9443"),
        label=f"{owner['username']} · {PROFILE_KIND_LABELS.get(kind, kind)}",
        account_mode=TRADING_MODE)
    with _DB_LOCK, _connect() as conn:
        conn.execute("UPDATE users SET owner_id=? WHERE id=?", (int(owner_uid), uid))
    return uid


def set_account_mode(user_id: int, mode: str) -> None:
    mode = normalize_account_mode(mode)
    init()
    with _DB_LOCK, _connect() as conn:
        conn.execute("UPDATE users SET account_mode=? WHERE id=?", (mode, int(user_id)))


def set_admin(user_id: int, is_admin_flag: bool) -> bool:
    """ADMIN 권한 부여/회수. 사장 지시 2026-06-08: ADMIN 은 hh09080 영구·단독.
    - hh09080 외 계정 승격 거부, hh09080 강등 거부 (모두 return False + 경고, 예외 없음).
    """
    init()
    with _DB_LOCK, _connect() as conn:
        row = conn.execute("SELECT username FROM users WHERE id=?", (int(user_id),)).fetchone()
        if not row:
            return False
        uname = row["username"]
        if is_admin_flag and uname not in ADMIN_USERNAMES:
            logger.warning("set_admin 거부: %s 는 ADMIN 화이트리스트(hh09080) 밖 — 승격 불가", uname)
            return False
        if not is_admin_flag and uname in ADMIN_USERNAMES:
            logger.warning("set_admin 거부: %s 는 시드 ADMIN — 강등 불가", uname)
            return False
        conn.execute("UPDATE users SET is_admin=? WHERE id=?",
                     (1 if is_admin_flag else 0, int(user_id)))
    return True


def verify_password(username: str, password: str) -> Optional[Dict[str, Any]]:
    """argon2id 검증. 미마이그레이션(legacy) 행이면 복호-비교 후 즉시 해시로 승격.
    성공 시 자격증명 dict, 실패 시 None."""
    u = find_user_by_username(username)
    if not u:
        return None
    if u.get("owner_id"):
        # 통합 계정의 서브 프로필 행은 직접 로그인 불가 — 마스터 계정으로 로그인해 토글로 전환.
        return None
    stored = u.get("password_hash") or ""
    if stored:
        if not verify_pw_hash(stored, password or ""):
            return None
        try:
            if _PH.check_needs_rehash(stored):
                _set_password_hash(u["id"], hash_password(password or ""), clear_enc=True)
        except Exception:
            logger.warning("argon2 rehash 점검 실패(user_id=%s)", u.get("id"))
        return u
    # legacy: password_hash 없음 → 복호-비교 후 해시 승격(1회성)
    if (u.get("password") or "") and u["password"] == (password or ""):
        _set_password_hash(u["id"], hash_password(password or ""), clear_enc=True)
        return u
    return None


def touch_login(user_id: int) -> None:
    init()
    with _DB_LOCK, _connect() as conn:
        conn.execute("UPDATE users SET last_login_at=? WHERE id=?", (time.time(), int(user_id)))


def _set_password_hash(user_id: int, new_hash: str, clear_enc: bool = True) -> None:
    init()
    with _DB_LOCK, _connect() as conn:
        if clear_enc:
            conn.execute("UPDATE users SET password_hash=?, password_enc='' WHERE id=?",
                         (new_hash, int(user_id)))
        else:
            conn.execute("UPDATE users SET password_hash=? WHERE id=?",
                         (new_hash, int(user_id)))


def migrate_passwords_and_bidx() -> Dict[str, int]:
    """부팅 1회 실행(멱등). password_enc→argon2 해시 승격 + 누락된 블라인드
    인덱스 백필. 이미 마이그레이션된 행은 건너뜀. {'pw':n,'bidx':m} 반환.

    복호 실패(InvalidToken 또는 기타 예외) 시 해당 행을 건너뜀 — 데이터 훼손 방지.
    FernetKeyLost 는 _ensure_fernet 에서 루프 진입 전에 발생하므로 여기서 잡지 않음."""
    init()
    _ensure_fernet()  # 키 분실이면 여기서 FernetKeyLost — per-row except 에 삼켜지지 않게 선제 발생
    stats = {"pw": 0, "bidx": 0, "acct_bidx": 0}
    with _DB_LOCK, _connect() as conn:
        rows = conn.execute(
            "SELECT id, password_enc, password_hash, "
            "kis_app_key_enc, kis_app_secret_enc, "
            "kis_app_key_bidx, kis_app_secret_bidx, "
            "kis_account_no_enc, kis_account_no_bidx FROM users"
        ).fetchall()
        for r in rows:
            try:
                updates: Dict[str, Any] = {}
                did_pw = False
                did_bidx = False
                did_acct = False

                # ── 비밀번호 승격 경로 ──────────────────────────────────────
                if not (r["password_hash"] or "") and (r["password_enc"] or ""):
                    dec_pw = decrypt(r["password_enc"])
                    if not dec_pw:
                        # 복호 실패 — 행 보존, 스킵 (데이터 훼손 방지)
                        logger.error(
                            "auth 마이그레이션: user_id=%s password_enc 복호 실패 — 행 보존(스킵)",
                            r["id"])
                        continue
                    updates["password_hash"] = hash_password(dec_pw)
                    updates["password_enc"] = ""
                    did_pw = True

                # ── bidx 백필 경로 ─────────────────────────────────────────
                if not (r["kis_app_key_bidx"] or ""):
                    enc_key = r["kis_app_key_enc"] or ""
                    enc_secret = r["kis_app_secret_enc"] or ""
                    if enc_key and enc_secret:
                        dec_key = decrypt(enc_key)
                        dec_secret = decrypt(enc_secret)
                        if not (dec_key and dec_secret):
                            logger.error(
                                "auth 마이그레이션: user_id=%s KIS enc 복호 실패 — bidx 백필 스킵",
                                r["id"])
                            continue
                        updates["kis_app_key_bidx"] = bidx(dec_key)
                        updates["kis_app_secret_bidx"] = bidx(dec_secret)
                        did_bidx = True
                    # enc 자체가 비어있는 경우(빈 enc) — bidx 백필 대상 아님, 통과

                if not (r["kis_account_no_bidx"] or ""):
                    enc_acct = r["kis_account_no_enc"] or ""
                    if enc_acct:
                        dec_acct = decrypt(enc_acct)
                        if dec_acct:
                            updates["kis_account_no_bidx"] = bidx(dec_acct)
                            did_acct = True
                        else:
                            logger.error(
                                "auth 마이그레이션: user_id=%s kis_account_no_enc "
                                "복호 실패 — 계좌bidx 백필 스킵(행 보존)", r["id"])

                # 여기까지 오면 updates 는 안전하게 쓸 수 있는 값들만 포함
                if updates:
                    sets = ",".join(f"{k}=?" for k in updates)
                    conn.execute(f"UPDATE users SET {sets} WHERE id=?",
                                 (*updates.values(), int(r["id"])))
                if did_pw:
                    stats["pw"] += 1
                if did_bidx:
                    stats["bidx"] += 1
                if did_acct:
                    stats["acct_bidx"] += 1
            except Exception as e:
                # FernetKeyLost 는 위 _ensure_fernet() 선제 호출에서 발생 — 여기 도달 안 함
                logger.error(
                    "auth 마이그레이션 행 실패 user_id=%s: %s — 스킵", r["id"], e)
                continue
    if stats["pw"] or stats["bidx"] or stats["acct_bidx"]:
        logger.info("auth 마이그레이션 완료: 해시승격 %d, bidx백필 %d, 계좌bidx백필 %d",
                    stats["pw"], stats["bidx"], stats["acct_bidx"])
    return stats


def _mask(s: str, keep: int = 4) -> str:
    s = s or ""
    return (s[:keep] + "…" + s[-2:]) if len(s) > keep + 2 else "…"


def list_accounts() -> List[Dict[str, Any]]:
    """등록된 계정 목록 (민감값 마스킹) — 계정 전환 UI용. 아이디는 그대로 노출(식별용)."""
    init()
    with _DB_LOCK, _connect() as conn:
        rows = conn.execute(
            "SELECT id, username, kis_account_no_enc, label, last_login_at, is_admin "
            "FROM users ORDER BY last_login_at DESC"
        ).fetchall()
    return [{
        "id": int(r["id"]),
        "username": r["username"],
        "label": r["label"],
        "kis_account_no_masked": _mask(decrypt(r["kis_account_no_enc"]), keep=4),
        "last_login_at": r["last_login_at"],
        "is_admin": bool(r["is_admin"]),
    } for r in rows]


def delete_user(user_id: int) -> bool:
    """유저 행 + 해당 세션 완전 삭제. 성공 True, 없는 uid False.
    (SQLite FK CASCADE 는 PRAGMA 미설정이라 세션을 명시 삭제한다.)"""
    init()
    with _DB_LOCK, _connect() as conn:
        row = conn.execute("SELECT id FROM users WHERE id=?",
                           (int(user_id),)).fetchone()
        if not row:
            return False
        conn.execute("DELETE FROM sessions WHERE user_id=?", (int(user_id),))
        conn.execute("DELETE FROM users WHERE id=?", (int(user_id),))
    return True


def _is_mock_url(url: str) -> bool:
    # NOTE: 모의/실전 판별 규칙은 infra/kis_broker.py 의 is_mock 과 동일 — 한쪽 변경 시 양쪽 동기화 필요
    u = url or ""
    return ("openapivts" in u) or (":29443" in u)


def list_members() -> List[Dict[str, Any]]:
    """ADMIN 회원 현황(읽기 전용). 민감값 비노출. is_mock 은 Base URL 파생."""
    init()
    with _DB_LOCK, _connect() as conn:
        rows = conn.execute(
            "SELECT id, username, label, kis_base_url, account_mode, created_at, last_login_at, "
            "is_admin, owner_id FROM users ORDER BY created_at ASC").fetchall()
    return [{
        "id": int(r["id"]), "username": r["username"],
        "label": (r["label"] if "label" in r.keys() else "") or "",
        "created_at": r["created_at"], "last_login_at": r["last_login_at"],
        "is_admin": bool(r["is_admin"]),
        "account_mode": (r["account_mode"] if "account_mode" in r.keys() else TRADING_MODE),
        "is_mock": _is_mock_url(r["kis_base_url"]),
        "owner_id": (int(r["owner_id"]) if "owner_id" in r.keys() and r["owner_id"] else 0),
    } for r in rows]


# 회원관리 표시용 짧은 기능명 (사장 지시 2026-07-21).
PROFILE_KIND_LABELS_SHORT = {PROFILE_KIS_REAL: "실전", PROFILE_KIS_PAPER: "모의"}


def admin_member_overview() -> List[Dict[str, Any]]:
    """ADMIN 회원관리용 — 로그인 계정(owner_id=0)만 보이고, 통합 계정의 서브 프로필은
    마스터에 접혀 '활성 기능(실전/모의)' 목록으로 표시된다(사장 지시 2026-07-21).
    관전 계정은 functions=['관전'].

    is_mock: 이 회원에게 **실전 매매 프로필이 없으면** True(= 모의/관전 전용). 계정 통합
    (2026-07-21) 전에는 /api/admin/members 가 list_members() 를 태워 회원행의 Base URL 하나로
    is_mock 을 내려줬는데, 통합 개편에서 이 라우트를 admin_member_overview() 로 갈아끼우며
    키가 통째로 사라졌다 — 회귀다. 구 대시보드(server/static/legacy.html loadMembers())가
    아직 m.is_mock 으로 '모의/실거래' 배지를 그리므로(undefined → 전원 '실거래' 오표시)
    되살린다. 단, 통합 계정은 실전+모의를 동시에 가질 수 있어 '회원행 URL' 이 아니라
    '실전 프로필 보유 여부'로 판정한다(현행 UI 는 functions 배지를 쓴다)."""
    init()
    with _DB_LOCK, _connect() as conn:
        rows = conn.execute(
            "SELECT id, username, account_mode, is_admin FROM users "
            "WHERE owner_id=0 OR owner_id IS NULL ORDER BY id ASC").fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        uid = int(r["id"])
        mode = (r["account_mode"] if "account_mode" in r.keys() else TRADING_MODE)
        kinds: set = set()
        if mode == VIEWER_MODE:
            funcs = ["관전"]
        else:
            kinds = {p["kind"] for p in list_profiles(uid)}
            funcs = [PROFILE_KIND_LABELS_SHORT[k] for k in PROFILE_KIND_ORDER if k in kinds]
            if not funcs:
                funcs = ["관전"]   # 매매 프로필이 하나도 없으면 실질 관전
        out.append({"id": uid, "username": r["username"],
                    "is_admin": bool(r["is_admin"]), "functions": funcs,
                    "is_mock": PROFILE_KIS_REAL not in kinds})
    return out


def change_password(user_id: int, current: str, new_password: str) -> bool:
    """현재 비번 검증 → 신규 정책 검사 → argon2 해시 갱신.
    현재 비번 불일치/정책 위반 → ValueError."""
    init()
    creds = get_user_credentials(int(user_id))
    if not creds:
        raise ValueError("계정을 찾을 수 없습니다.")
    if not verify_password(creds["username"], current or ""):
        raise ValueError("현재 비밀번호가 일치하지 않습니다.")
    perr = password_policy_error(new_password or "")
    if perr:
        raise ValueError(perr)
    _set_password_hash(int(user_id), hash_password(new_password), clear_enc=True)
    return True


def update_credentials(user_id: int, *,
                        kis_app_key: Optional[str] = None,
                        kis_app_secret: Optional[str] = None,
                        kis_account_no: Optional[str] = None,
                        kis_base_url: Optional[str] = None) -> bool:
    """제공된 자격증명만 갱신(None=미변경). 변경분 enc + bidx 동시 재계산."""
    init()
    sets, params = [], []
    if kis_app_key is not None:
        sets += ["kis_app_key_enc=?", "kis_app_key_bidx=?"]
        params += [encrypt(kis_app_key), bidx(kis_app_key)]
    if kis_app_secret is not None:
        sets += ["kis_app_secret_enc=?", "kis_app_secret_bidx=?"]
        params += [encrypt(kis_app_secret), bidx(kis_app_secret)]
    if kis_account_no is not None:
        sets += ["kis_account_no_enc=?", "kis_account_no_bidx=?"]
        params += [encrypt(kis_account_no), bidx(kis_account_no)]
    if kis_base_url is not None:
        sets += ["kis_base_url=?"]
        params += [(kis_base_url or "").strip()]
    if not sets:
        return False
    params.append(int(user_id))
    with _DB_LOCK, _connect() as conn:
        conn.execute(f"UPDATE users SET {','.join(sets)} WHERE id=?", tuple(params))
    return True


# ─── Sessions ─────────────────────────────────────────────────────────────────
def create_session(user_id: int) -> str:
    init()
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _DB_LOCK, _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?,?,?,?)",
            (token, int(user_id), now, now + SESSION_TTL_SEC),
        )
    return token


def lookup_session(token: str) -> Optional[int]:
    if not token:
        return None
    init()
    with _DB_LOCK, _connect() as conn:
        row = conn.execute("SELECT user_id, expires_at FROM sessions WHERE token=?",
                           (token,)).fetchone()
        if not row:
            return None
        if float(row["expires_at"]) < time.time():
            conn.execute("DELETE FROM sessions WHERE token=?", (token,))
            return None
    return int(row["user_id"])


def delete_session(token: str) -> None:
    if not token:
        return
    init()
    with _DB_LOCK, _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token=?", (token,))


# ─── Bootstrap from .env (사장 피드백 2026-05-16) ───────────────────────────────
def bootstrap_from_env() -> Optional[int]:
    """등록 계정이 없고 .env 에 KIS 정보 + ARQUANT_BOOTSTRAP_USER/PASS 가 있으면
    사장님 프로필 1개를 생성. 비밀번호는 .env(=gitignore)에만 두어 소스/깃에 노출 안 함.
    이미 계정이 있으면 None."""
    init()
    with _DB_LOCK, _connect() as conn:
        n = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    if n:
        return None
    try:
        from config import KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT_NO, KIS_BASE_URL, OPENDART_API_KEY
    except Exception as e:
        logger.warning("bootstrap_from_env: config 로드 실패 %s", e)
        return None
    bu = os.getenv("ARQUANT_BOOTSTRAP_USER", "").strip()
    bp = os.getenv("ARQUANT_BOOTSTRAP_PASS", "").strip()
    if not (bu and bp):
        logger.warning("bootstrap_from_env: ARQUANT_BOOTSTRAP_USER/PASS 미설정 — 시드 생략 "
                       "(로그인 아이디/비밀번호는 사용자가 정해야 함)")
        return None
    if not (KIS_APP_KEY and KIS_APP_SECRET and KIS_ACCOUNT_NO):
        logger.warning("bootstrap_from_env: .env 필수 KIS 정보 누락 — 시드 생략")
        return None
    perr = password_policy_error(bp)
    if perr:
        logger.warning("bootstrap_from_env: ARQUANT_BOOTSTRAP_PASS 정책 위반 — %s", perr)
        return None
    uid = upsert_user(
        username=bu, password=bp,
        kis_app_key=KIS_APP_KEY, kis_app_secret=KIS_APP_SECRET,
        kis_account_no=KIS_ACCOUNT_NO,
        kis_base_url=KIS_BASE_URL, dart_key=OPENDART_API_KEY or "",
        label="사장님 (.env 시드 · ADMIN)", is_admin=True,
    )
    logger.info("bootstrap_from_env: .env 기준 ADMIN 프로필 생성 user_id=%s username=%s", uid, bu)
    return uid

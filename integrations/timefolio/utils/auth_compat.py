"""ArQuant `infra.auth_store`가 없는 ArcTrade 독립 실행용 최소 대체 구현.

- hash_password / verify_pw_hash: stdlib scrypt — 독립 상태 파일 전용.
- encrypt / decrypt: 독립 모드에선 사이트 자격증명을 저장하지 않는다(순수 모의거래엔 불필요).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets


def password_policy_error(password: str) -> str | None:
    if len(password or "") < 8:
        return "비밀번호는 8자 이상이어야 합니다."
    return None


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt((password or "").encode(), salt=salt, n=2**14, r=8, p=1)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def verify_pw_hash(stored: str, password: str) -> bool:
    try:
        scheme, salt_b64, digest_b64 = (stored or "").split("$", 2)
        if scheme != "scrypt":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        digest = hashlib.scrypt((password or "").encode(), salt=salt, n=2**14, r=8, p=1)
        return hmac.compare_digest(digest, expected)
    except Exception:
        return False


def encrypt(value: str) -> str:
    return ""


def decrypt(value: str) -> str:
    return ""

"""Single-owner authentication using signed Cloudflare Access application JWTs."""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import re

import jwt


class AccessNotConfigured(RuntimeError):
    pass


def settings():
    path = Path(os.getenv("QIS_ACCESS_CONFIG", "/home/arcosium/vault/QuantInSight/cloudflare_access.json"))
    try:
        cfg = json.loads(path.read_text())
        domain = str(cfg["team_domain"])
        if not re.fullmatch(r"[a-zA-Z0-9-]+\.cloudflareaccess\.com", domain):
            raise ValueError("invalid issuer")
        if not cfg.get("audience") or not cfg.get("owner_email") or int(cfg["owner_uid"]) <= 0:
            raise ValueError("incomplete configuration")
        return cfg
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AccessNotConfigured("Cloudflare Access 설정을 확인하세요") from exc


@lru_cache(maxsize=2)
def key_client(domain):
    return jwt.PyJWKClient(f"https://{domain}/cdn-cgi/access/certs", cache_keys=True,
                           lifespan=3600, timeout=5)


def verify(token):
    """Never accept a session cookie, email header or unverified token payload."""
    if not token or len(token) > 32768:
        raise PermissionError("Cloudflare Access 인증이 필요합니다")
    cfg = settings()
    try:
        key = key_client(cfg["team_domain"]).get_signing_key_from_jwt(token).key
        payload = jwt.decode(token, key, algorithms=["RS256"], audience=cfg["audience"],
            issuer=f"https://{cfg['team_domain']}", leeway=15,
            options={"require":["exp", "iat", "iss", "aud", "sub", "email"]})
    except jwt.PyJWTError as exc:
        raise PermissionError("Cloudflare Access 인증을 확인할 수 없습니다") from exc
    if str(payload.get("email", "")).casefold() != cfg["owner_email"].casefold():
        raise PermissionError("본인 계정만 접속할 수 있습니다")
    return int(cfg["owner_uid"]), float(payload["exp"])

"""Per-uid runtime context + registry — the multi-tenant isolation core.

Each UserContext holds one user's decrypted credentials and lazily builds that
user's KISBroker and ArquantOrchestrator. The registry keeps one context per uid
for the life of the process. Nothing here mutates global config — isolation comes
from injection, not from rewriting shared globals.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from infra import auth_store
from infra import user_paths

logger = logging.getLogger("USER_CTX")


class UserContext:
    def __init__(self, creds: Dict[str, Any]):
        self.uid: int = int(creds["id"])
        self.creds: Dict[str, Any] = creds
        self.is_admin: bool = bool(creds.get("is_admin"))
        self.paths = user_paths
        self._broker = None   # lazy — built on first access
        self._swarm = None    # lazy — built on first access
        self.task = None      # asyncio.Task | None (this uid's trading loop)

    @property
    def broker(self):
        if self._broker is None:
            if self.creds.get("account_mode", auth_store.TRADING_MODE) != auth_store.TRADING_MODE:
                raise ValueError("KIS 거래 프로필만 브로커를 생성할 수 있습니다.")
            from infra.kis_broker import KISBroker
            self._broker = KISBroker(self.creds,
                                     token_path=user_paths.token_path(self.uid))
        return self._broker

    @property
    def swarm(self):
        if self._swarm is None:
            from main_swarm import ArquantOrchestrator
            self._swarm = ArquantOrchestrator(self)
        return self._swarm

    def reset(self) -> None:
        """Drop broker/swarm so they rebuild (e.g. after a credentials update)."""
        self._broker = None
        self._swarm = None


class UserRegistry:
    def __init__(self):
        self._ctx: Dict[int, UserContext] = {}
        self._lock = threading.RLock()

    def get_or_create(self, uid: int) -> UserContext:
        uid = int(uid)
        with self._lock:
            ctx = self._ctx.get(uid)
            if ctx is not None:
                return ctx
            creds = auth_store.get_user_credentials(uid)
            if not creds:
                raise ValueError(f"user_id={uid} 자격증명 없음 — 컨텍스트 생성 불가")
            ctx = UserContext(creds)
            self._ctx[uid] = ctx
            logger.info("UserContext 생성 uid=%s label=%s admin=%s",
                        uid, creds.get("label"), ctx.is_admin)
            return ctx

    def get(self, uid: int) -> Optional[UserContext]:
        return self._ctx.get(int(uid))

    def all_contexts(self) -> Dict[int, UserContext]:
        return dict(self._ctx)

    def drop(self, uid: int) -> None:
        with self._lock:
            self._ctx.pop(int(uid), None)


REGISTRY = UserRegistry()

"""Persistent advisory fundamental research snapshots.

This store is deliberately separate from order/risk state. Research generated
from ai-berkshire-style workflows is advisory unless another deterministic gate
explicitly consumes it.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("FUNDAMENTAL_RESEARCH")
DB_PATH = Path(__file__).parent.parent / "data" / "fundamental_research.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_snapshots (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    uid                      INTEGER,
    cycle_started_at         TEXT,
    ts                       TEXT,
    code                     TEXT NOT NULL,
    name                     TEXT,
    source                   TEXT,
    verdict                  TEXT,
    business_quality_score   REAL,
    valuation_margin_score   REAL,
    moat_score               REAL,
    management_score         REAL,
    thesis_invalidators      TEXT,
    financial_checks         TEXT,
    memo                     TEXT
);
CREATE INDEX IF NOT EXISTS idx_frs_uid_code ON research_snapshots(uid, code);
CREATE INDEX IF NOT EXISTS idx_frs_ts ON research_snapshots(ts);
"""

_lock = threading.RLock()
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, isolation_level=None)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(_SCHEMA)
    return _conn


def record_snapshot(row: Dict[str, Any]) -> Optional[int]:
    cols = (
        "uid", "cycle_started_at", "ts", "code", "name", "source", "verdict",
        "business_quality_score", "valuation_margin_score", "moat_score", "management_score",
        "thesis_invalidators", "financial_checks", "memo",
    )
    vals = []
    for c in cols:
        v = row.get(c)
        if c in ("thesis_invalidators", "financial_checks") and v is not None:
            try:
                v = json.dumps(v, ensure_ascii=False)
            except Exception:
                v = None
        vals.append(v)
    try:
        with _lock:
            cur = _get_conn().execute(
                f"INSERT INTO research_snapshots ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                vals,
            )
            return cur.lastrowid
    except Exception as e:
        logger.warning("record_snapshot 실패: %s", e)
        return None


def _decode_row(r: sqlite3.Row) -> Dict[str, Any]:
    d = dict(r)
    for c in ("thesis_invalidators", "financial_checks"):
        if d.get(c):
            try:
                d[c] = json.loads(d[c])
            except Exception:
                pass
    return d


def latest(uid: Optional[int], code: str) -> Optional[Dict[str, Any]]:
    try:
        args: List[Any] = [str(code).strip()]
        where = "code=?"
        if uid is not None:
            where += " AND uid=?"
            args.append(int(uid))
        with _lock:
            row = _get_conn().execute(
                f"SELECT * FROM research_snapshots WHERE {where} ORDER BY id DESC LIMIT 1",
                tuple(args),
            ).fetchone()
        return _decode_row(row) if row else None
    except Exception as e:
        logger.warning("latest 실패: %s", e)
        return None


def list_snapshots(uid: Optional[int] = None, since: Optional[str] = None,
                   limit: int = 500) -> List[Dict[str, Any]]:
    try:
        where, args = [], []
        if uid is not None:
            where.append("uid=?")
            args.append(int(uid))
        if since:
            where.append("ts>=?")
            args.append(since)
        wsql = ("WHERE " + " AND ".join(where)) if where else ""
        args.append(int(limit))
        with _lock:
            rows = _get_conn().execute(
                f"SELECT * FROM research_snapshots {wsql} ORDER BY id DESC LIMIT ?",
                tuple(args),
            ).fetchall()
        return [_decode_row(r) for r in rows]
    except Exception as e:
        logger.warning("list_snapshots 실패: %s", e)
        return []


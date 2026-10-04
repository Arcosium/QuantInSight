"""
Arquant v1.0 — Cycle persistence (SQLite).
Stores one row per analysis cycle so the dashboard / backtests / audits have
durable history beyond the 4000-entry claude_response.json cap.

Schema kept intentionally flat — JSON columns hold the variable-shape parts
(orders list, risk verdicts, sell directives, equity snapshot) so we can grow
without migrations. Indexes on (started_at, session) cover the common queries.

Single-writer pattern (the orchestrator thread). Concurrent read is fine.
"""
from __future__ import annotations
import json, sqlite3, threading, time, logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("CYCLE_STORE")
KST = timezone(timedelta(hours=9))
DB_PATH = Path(__file__).parent.parent / "data" / "cycles.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT    NOT NULL,           -- 'YYYY-MM-DD HH:MM:SS' KST
    ended_at     TEXT,
    session      TEXT,                       -- KR_TRADING / KR_PRE_MARKET / US_TRADING / OFF_HOURS
    market_open  INTEGER DEFAULT 0,          -- 1 if cycle was triggered by market_open
    news_count   INTEGER DEFAULT 0,
    candidate_codes TEXT,                    -- JSON list
    target_codes    TEXT,                    -- JSON list (final 매수)
    sell_directives TEXT,                    -- JSON {code: directive}
    orders_planned  TEXT,                    -- JSON list of order dicts
    orders_executed TEXT,                    -- JSON list with broker results
    risk_approved   INTEGER DEFAULT 0,
    risk_report     TEXT,
    macro_report    TEXT,
    quant_report    TEXT,
    news_report     TEXT,
    final_report    TEXT,
    bp_cash         REAL,
    bp_total_eval   REAL,
    bp_pnl_ratio    REAL,
    error           TEXT,
    uid             INTEGER,
    committee       TEXT                        -- JSON — QuantInSight 이식 위원회 심의 기록 (2026-07-18)
);
CREATE INDEX IF NOT EXISTS idx_cycles_started ON cycles(started_at);
CREATE INDEX IF NOT EXISTS idx_cycles_session ON cycles(session);
CREATE INDEX IF NOT EXISTS idx_cycles_uid ON cycles(uid);

CREATE TABLE IF NOT EXISTS holdings_history (
    uid         INTEGER NOT NULL DEFAULT 0,   -- Phase2: 계정별 보유기간 격리 (2026-06-15)
    code        TEXT NOT NULL,
    first_seen  TEXT NOT NULL,               -- 'YYYY-MM-DD HH:MM:SS' KST
    last_seen   TEXT NOT NULL,
    last_qty    INTEGER,
    last_avg    REAL,
    PRIMARY KEY (uid, code, first_seen)
);
CREATE INDEX IF NOT EXISTS idx_hh_code ON holdings_history(uid, code);
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
        # Idempotent migration: pre-Phase2 DBs lack the uid column.
        cols = {r[1] for r in _conn.execute("PRAGMA table_info(cycles)").fetchall()}
        if "uid" not in cols:
            _conn.execute("ALTER TABLE cycles ADD COLUMN uid INTEGER")
            _conn.execute("CREATE INDEX IF NOT EXISTS idx_cycles_uid ON cycles(uid)")
        # QuantInSight 이식(2026-07-18): 위원회 심의 기록 컬럼 — 기존 DB 무이전 호환.
        if "committee" not in cols:
            _conn.execute("ALTER TABLE cycles ADD COLUMN committee TEXT")
        # 보유기간 per-uid 격리(2026-06-15): 기존 비-uid 테이블은 계정 간 오염되어 있고
        # derived 데이터(매 사이클 재적재)이므로 재생성한다. PRAGMA 결과가 비면(테이블 미존재) 스킵.
        hh = {r[1] for r in _conn.execute("PRAGMA table_info(holdings_history)").fetchall()}
        if hh and "uid" not in hh:
            _conn.execute("DROP TABLE holdings_history")
            _conn.executescript(_SCHEMA)
    return _conn

def _now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")

def _j(obj) -> Optional[str]:
    if obj is None:
        return None
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return None

def record_cycle(meta: Dict[str, Any]) -> Optional[int]:
    """Persist one cycle. `meta` may include any of the column keys above; missing → NULL.
    Returns the inserted row id, or None on failure."""
    cols = ("started_at","ended_at","session","market_open","news_count",
            "candidate_codes","target_codes","sell_directives","orders_planned",
            "orders_executed","risk_approved","risk_report","macro_report",
            "quant_report","news_report","final_report","bp_cash","bp_total_eval",
            "bp_pnl_ratio","error","uid","committee")
    vals = []
    for c in cols:
        v = meta.get(c)
        if c in ("candidate_codes","target_codes","sell_directives","orders_planned","orders_executed","committee"):
            v = _j(v)
        elif c in ("market_open","risk_approved"):
            v = 1 if v else 0
        vals.append(v)
    try:
        with _lock:
            conn = _get_conn()
            cur = conn.execute(
                f"INSERT INTO cycles ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
                vals)
            return cur.lastrowid
    except Exception as e:
        logger.warning(f"record_cycle 실패: {e}")
        return None

def list_cycles(limit: int = 50, offset: int = 0, uid: Optional[int] = None) -> List[Dict]:
    """Newest cycles first, paged. Returns raw rows (JSON fields un-parsed to keep wire cost low).
    `uid` filters to one user's cycles (Phase 2); None returns all."""
    try:
        where = "WHERE uid=?" if uid is not None else ""
        args = ([int(uid)] if uid is not None else []) + [int(limit), int(offset)]
        with _lock:
            rows = _get_conn().execute(
                f"SELECT * FROM cycles {where} ORDER BY id DESC LIMIT ? OFFSET ?",
                tuple(args)).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"list_cycles 실패: {e}")
        return []

def list_cycles_summary(limit: int = 100, offset: int = 0, uid: Optional[int] = None) -> List[Dict]:
    """사이클 탭 히스토리용 경량 목록 — 무거운 리포트 본문 없이 id/시각/세션 + 체결·심의 유무만.
    (QuantInSight 이식 UI 2026-07-18: 목록은 이걸로, 상세는 get_cycle 로.)"""
    try:
        where = "WHERE uid=?" if uid is not None else ""
        args = ([int(uid)] if uid is not None else []) + [int(limit), int(offset)]
        with _lock:
            rows = _get_conn().execute(
                f"SELECT id, started_at, ended_at, session, market_open, error, "
                f"orders_executed, sell_directives, committee IS NOT NULL AS has_committee "
                f"FROM cycles {where} ORDER BY id DESC LIMIT ? OFFSET ?",
                tuple(args)).fetchall()
        out: List[Dict] = []
        for r in rows:
            d = dict(r)
            buys = sells = 0
            try:
                for e in json.loads(d.get("orders_executed") or "[]"):
                    side = str((e or {}).get("side", "")).lower()
                    if side in ("sell", "매도"):
                        sells += 1
                    else:
                        buys += 1
            except Exception:
                pass
            out.append({"id": d["id"], "started_at": d["started_at"], "ended_at": d["ended_at"],
                        "session": d["session"], "market_open": d["market_open"],
                        "error": bool(d.get("error")), "orders": buys, "sells": sells,
                        "has_committee": bool(d.get("has_committee"))})
        return out
    except Exception as e:
        logger.warning(f"list_cycles_summary 실패: {e}")
        return []

def get_cycle(cycle_id: int) -> Optional[Dict]:
    try:
        with _lock:
            r = _get_conn().execute("SELECT * FROM cycles WHERE id=?", (int(cycle_id),)).fetchone()
        return dict(r) if r else None
    except Exception:
        return None

# ── Holdings history (used for 보유 기간 P&L) ────────────────────────────
def upsert_holding_seen(code: str, qty: int, avg_price: float, uid: int = 0):
    """Mark a code as 'seen with this qty' right now (계정별). First-time codes get a first_seen row;
    subsequent observations update last_seen/last_qty/last_avg.
    When qty falls to 0 (사이클이 매도 후 보이지 않음) the row is closed at last_seen and
    a new row starts on the next purchase — keeps 매수 시점이 의미 있는 단위로 유지됨."""
    code = (code or "").strip()
    if not code:
        return
    uid = int(uid or 0)
    now = _now_kst()
    try:
        with _lock:
            conn = _get_conn()
            row = conn.execute(
                "SELECT first_seen, last_qty FROM holdings_history WHERE uid=? AND code=? "
                "ORDER BY first_seen DESC LIMIT 1", (uid, code)).fetchone()
            if row and (row["last_qty"] or 0) > 0:
                # active position — just update last_seen
                conn.execute(
                    "UPDATE holdings_history SET last_seen=?, last_qty=?, last_avg=? "
                    "WHERE uid=? AND code=? AND first_seen=?",
                    (now, int(qty), float(avg_price or 0.0), uid, code, row["first_seen"]))
            else:
                # fresh position (first time, or after a full sell) → open new row
                conn.execute(
                    "INSERT OR REPLACE INTO holdings_history(uid,code,first_seen,last_seen,last_qty,last_avg) "
                    "VALUES (?,?,?,?,?,?)",
                    (uid, code, now, now, int(qty), float(avg_price or 0.0)))
    except Exception as e:
        logger.warning(f"upsert_holding_seen({code}) 실패: {e}")

def mark_position_closed(code: str, uid: int = 0):
    """Called when a holding disappears (full sell). Sets last_qty=0 so the next purchase
    opens a fresh row — distinct holding-period from the previous one. (계정별)"""
    code = (code or "").strip()
    if not code: return
    uid = int(uid or 0)
    try:
        with _lock:
            conn = _get_conn()
            row = conn.execute(
                "SELECT first_seen FROM holdings_history WHERE uid=? AND code=? "
                "ORDER BY first_seen DESC LIMIT 1", (uid, code)).fetchone()
            if row:
                conn.execute(
                    "UPDATE holdings_history SET last_qty=0, last_seen=? "
                    "WHERE uid=? AND code=? AND first_seen=?",
                    (_now_kst(), uid, code, row["first_seen"]))
    except Exception as e:
        logger.warning(f"mark_position_closed({code}) 실패: {e}")

def get_holding_period(code: str, uid: int = 0) -> Optional[Dict]:
    """Latest active holding period for `code` (계정별), or None.
    Returns {'first_seen','last_seen','days_held'}."""
    try:
        with _lock:
            row = _get_conn().execute(
                "SELECT first_seen, last_seen, last_qty, last_avg FROM holdings_history "
                "WHERE uid=? AND code=? AND last_qty>0 ORDER BY first_seen DESC LIMIT 1",
                (int(uid or 0), (code or "").strip())).fetchone()
        if not row:
            return None
        try:
            # first_seen은 KST 문자열 — 서버가 UTC라도 KST로 통일해 일수 계산 (timezone 일관성)
            fs = datetime.strptime(row["first_seen"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
            now = datetime.now(KST)
            days = max(0.0, (now - fs).total_seconds() / 86400.0)
        except Exception:
            days = 0.0
        return {"first_seen": row["first_seen"], "last_seen": row["last_seen"],
                "qty": row["last_qty"], "avg_price": row["last_avg"], "days_held": days}
    except Exception:
        return None

def reconcile_holdings(current_codes: List[str], uid: int = 0):
    """Close out this account's active rows whose code isn't in its current holdings (계정별).
    2026-06-15: uid 없이 전 계정 종목을 닫아 다른 계정 보유기간을 훼손하던 버그 수정."""
    uid = int(uid or 0)
    try:
        with _lock:
            conn = _get_conn()
            active = conn.execute(
                "SELECT DISTINCT code FROM holdings_history WHERE uid=? AND last_qty>0", (uid,)).fetchall()
            now_set = set((c or "").strip() for c in current_codes)
            for r in active:
                if r["code"] not in now_set:
                    mark_position_closed(r["code"], uid=uid)
    except Exception as e:
        logger.warning(f"reconcile_holdings 실패: {e}")

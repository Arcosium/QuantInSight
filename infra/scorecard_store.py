"""에이전트 예측 신호 영속화 (사장 지시 2026-06-04 ④ — 성과 귀인용).
data/scorecard.db 의 agent_signals 테이블에 사이클별·종목별 예측을 전향적으로 적재.
cycle_store.py 의 단일-라이터 sqlite 패턴을 따른다. thesis/sell/risk 컬럼은 예약(nullable)."""
from __future__ import annotations
import json, sqlite3, threading, logging
from pathlib import Path
from typing import Optional, Dict, List, Any

logger = logging.getLogger("SCORECARD_STORE")
DB_PATH = Path(__file__).parent.parent / "data" / "scorecard.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_signals (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    uid              INTEGER,
    cycle_started_at TEXT,
    ts               TEXT,
    code             TEXT NOT NULL,
    name             TEXT,
    news_sentiment   REAL,
    quant_score      INTEGER,
    det_breakdown    TEXT,          -- JSON
    thesis_verdict   TEXT,          -- 예약(향후)
    sell_decision    TEXT,          -- 예약
    risk_verdict     TEXT,          -- 예약
    fundamental_verdict       TEXT,
    business_quality_score    REAL,
    moat_score                REAL,
    management_score          REAL,
    valuation_margin_score    REAL,
    thesis_invalidators       TEXT   -- JSON
);
CREATE INDEX IF NOT EXISTS idx_sig_uid ON agent_signals(uid);
CREATE INDEX IF NOT EXISTS idx_sig_code ON agent_signals(code);
CREATE INDEX IF NOT EXISTS idx_sig_ts ON agent_signals(ts);
"""

_lock = threading.RLock()
_conn: Optional[sqlite3.Connection] = None

_OPTIONAL_COLUMNS = {
    "fundamental_verdict": "TEXT",
    "business_quality_score": "REAL",
    "moat_score": "REAL",
    "management_score": "REAL",
    "valuation_margin_score": "REAL",
    "thesis_invalidators": "TEXT",
}


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, isolation_level=None)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(_SCHEMA)
        _ensure_optional_columns(_conn)
    return _conn


def _ensure_optional_columns(conn: sqlite3.Connection) -> None:
    """Migrate older scorecard.db files in place."""
    try:
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(agent_signals)").fetchall()}
        for col, typ in _OPTIONAL_COLUMNS.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE agent_signals ADD COLUMN {col} {typ}")
    except Exception as e:
        logger.warning(f"scorecard optional column migration 실패: {e}")


def record_signal(sig: Dict[str, Any]) -> Optional[int]:
    """한 종목·한 사이클의 에이전트 예측을 적재. 누락 키 → NULL. Returns row id or None."""
    cols = ("uid", "cycle_started_at", "ts", "code", "name", "news_sentiment",
            "quant_score", "det_breakdown", "thesis_verdict", "sell_decision", "risk_verdict",
            "fundamental_verdict", "business_quality_score", "moat_score", "management_score",
            "valuation_margin_score", "thesis_invalidators")
    vals = []
    for c in cols:
        v = sig.get(c)
        if c in ("det_breakdown", "thesis_invalidators") and v is not None:
            try:
                v = json.dumps(v, ensure_ascii=False)
            except Exception:
                v = None
        vals.append(v)
    try:
        with _lock:
            cur = _get_conn().execute(
                f"INSERT INTO agent_signals ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})", vals)
            return cur.lastrowid
    except Exception as e:
        logger.warning(f"record_signal 실패: {e}")
        return None


def list_signals(uid: Optional[int] = None, since: Optional[str] = None, limit: int = 5000) -> List[Dict]:
    """최신순 신호. uid/since(ts >=) 필터. det_breakdown 은 dict 로 파싱해 반환."""
    try:
        where, args = [], []
        if uid is not None:
            where.append("uid=?"); args.append(int(uid))
        if since:
            where.append("ts>=?"); args.append(since)
        wsql = ("WHERE " + " AND ".join(where)) if where else ""
        args.append(int(limit))
        with _lock:
            rows = _get_conn().execute(
                f"SELECT * FROM agent_signals {wsql} ORDER BY id DESC LIMIT ?", tuple(args)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for json_col in ("det_breakdown", "thesis_invalidators"):
                if not d.get(json_col):
                    continue
                try:
                    d[json_col] = json.loads(d[json_col])
                except Exception:
                    pass
            out.append(d)
        return out
    except Exception as e:
        logger.warning(f"list_signals 실패: {e}")
        return []

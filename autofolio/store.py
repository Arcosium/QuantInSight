import contextlib
import json
import sqlite3
import time
from .config import DATA, DB


@contextlib.contextmanager
def connect():
    DATA.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA busy_timeout=30000')
    try:
        yield db
        db.commit()
    finally:
        db.close()


def initialize():
    with connect() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS strategies (
          id TEXT PRIMARY KEY, title TEXT NOT NULL, family TEXT NOT NULL,
          cohort TEXT NOT NULL, payload TEXT NOT NULL, source TEXT NOT NULL,
          updated REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS strategies_cohort ON strategies(cohort);
        CREATE TABLE IF NOT EXISTS sources (
          path TEXT PRIMARY KEY, mtime REAL, size INTEGER, digest TEXT, status TEXT
        );
        CREATE TABLE IF NOT EXISTS jobs (
          id TEXT PRIMARY KEY, genome TEXT NOT NULL, model_id TEXT NOT NULL,
          generation INTEGER NOT NULL, parents TEXT NOT NULL, operator TEXT NOT NULL,
          status TEXT NOT NULL, pid INTEGER, created REAL NOT NULL, started REAL,
          finished REAL, error TEXT, result TEXT
        );
        CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, timestamp REAL, kind TEXT, body TEXT);
        ''')
        for k,v in {'enabled': True, 'concurrency': 2, 'generation': 0}.items():
            db.execute('INSERT OR IGNORE INTO settings VALUES (?,?)', (k,json.dumps(v)))


def setting(key, default=None):
    with connect() as db:
        r = db.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone()
    return json.loads(r['value']) if r else default


def set_setting(key, value):
    with connect() as db:
        db.execute('INSERT INTO settings VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                   (key,json.dumps(value)))


def event(kind, body):
    with connect() as db:
        db.execute('INSERT INTO events(timestamp,kind,body) VALUES (?,?,?)',
                   (time.time(),kind,json.dumps(body,ensure_ascii=False)))


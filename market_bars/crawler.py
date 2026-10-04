"""모듈 A — 실시간 분봉 크롤러 + SQLite 롤링 버퍼.

수집원: 네이버 금융 siseJson API (분봉: 종가/누적거래량만 유효).
  https://api.finance.naver.com/siseJson.naver?symbol=005930&requestType=1
      &startTime=YYYYMMDDHHMM&endTime=YYYYMMDDHHMM&timeframe=minute

- 매분 02~05초 사이 분산 호출 (서버 차단 방지, 스펙 §모듈A)
- 최근 BUFFER_DAYS 영업일치만 유지하는 롤링 버퍼 (SQLite WAL)
- 종목별 누락 분봉은 읽기 시점에 전분 종가로 ffill → 행렬 차원 불일치 방지
- 사이클 연속 3회 실패 → notify 후 다음 분 재시도 (fail-safe, 시스템 다운 금지)
"""
import ast
import csv
import logging
import random
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import requests

from . import bars_config as config
from . import market_time as mt
from .notify import notify

log = logging.getLogger("quantinsight.bars")

SISE_URL = ("https://api.finance.naver.com/siseJson.naver?symbol={code}"
            "&requestType=1&startTime={start}&endTime={end}&timeframe=minute")
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36",
           "Referer": "https://finance.naver.com"}

_ROW_RE = re.compile(r'\["(\d{12})"')


# ── 저장소 ────────────────────────────────────────────────────────
def open_db(path=None):
    conn = sqlite3.connect(str(path or config.DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS bars(
        code TEXT NOT NULL, ts TEXT NOT NULL,
        close REAL NOT NULL, vol_cum REAL,
        PRIMARY KEY(code, ts))""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bars_ts ON bars(ts)")
    return conn


def upsert_bars(conn, code, rows):
    """rows: [(ts_str, close, vol_cum)]"""
    conn.executemany(
        "INSERT OR REPLACE INTO bars(code, ts, close, vol_cum) VALUES(?,?,?,?)",
        [(code, ts, c, v) for ts, c, v in rows])
    conn.commit()


def purge_old(conn, keep_days=None):
    """오래된 분봉 삭제 — **기본값은 '삭제 안 함'** 이다.

    분봉 아카이브는 영구 보존한다(사장 지시 2026-07-14). 지나간 분봉은 다시 살 수 없고 쌓일수록
    자산이 된다. 예전엔 이 함수가 BUFFER_DAYS(3일)로 DB를 지워, 학습창을 늘리려 해도 데이터가
    이미 없는 상태였다. 저장(RETAIN_DAYS)과 학습창(BUFFER_DAYS)은 이제 완전히 별개다.

    RETAIN_DAYS > 0 으로 명시할 때만 그 일수를 남기고 지운다.
    """
    keep_days = keep_days if keep_days is not None else config.RETAIN_DAYS
    if not keep_days or keep_days <= 0:
        return                                  # 영구 보존 (기본)
    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT substr(ts,1,8) FROM bars ORDER BY 1 DESC")]
    if len(dates) > keep_days:
        cutoff = dates[keep_days - 1]
        conn.execute("DELETE FROM bars WHERE substr(ts,1,8) < ?", (cutoff,))
        conn.commit()
        log.info("rolling purge: < %s 삭제 (보유일 %s)", cutoff, dates[:keep_days])


# ── 유니버스 ──────────────────────────────────────────────────────
def load_universe():
    """data/universe.csv → [(code, name)]. 없으면 tools.build_universe 로 생성."""
    if not config.UNIVERSE_CSV.exists():
        log.info("universe.csv 없음 — 네이버 시총 상위에서 생성")
        from .build_universe import build
        build()
    with open(config.UNIVERSE_CSV, newline="", encoding="utf-8") as f:
        rows = [(r["code"], r["name"]) for r in csv.DictReader(f)]
    return rows[:config.UNIVERSE_SIZE]


# ── 수집 ─────────────────────────────────────────────────────────
def parse_sise(text):
    """siseJson 응답 → [(ts, close, vol_cum)] 시간 오름차순.
    분봉 응답의 시가/고가/저가는 null — 종가·누적거래량만 쓴다."""
    try:
        data = ast.literal_eval(text.replace("null", "None").strip())
    except (ValueError, SyntaxError):
        return []
    out = []
    for row in data:
        if not isinstance(row, list) or not row or not _ROW_RE.match(f'["{row[0]}"'):
            continue
        ts, close, vol = str(row[0]), row[4], row[5]
        if len(ts) != 12 or close is None:
            continue
        out.append((ts, float(close), float(vol or 0)))
    out.sort(key=lambda r: r[0])
    return out


def fetch_minutes(session, code, start, end):
    """분봉 조회. 실패 시 예외 전파(호출자가 카운트)."""
    url = SISE_URL.format(code=code, start=start, end=end)
    r = session.get(url, headers=HEADERS, timeout=config.REQUEST_TIMEOUT)
    r.raise_for_status()
    rows = parse_sise(r.text)
    if not rows:
        raise ValueError(f"{code}: 빈 응답")
    return rows


def backfill(conn, codes, days=None, workers=None):
    """최근 N 영업일 분봉 전체 백필 (기동 시 1회)."""
    days = days or config.BUFFER_DAYS
    tdays = mt.recent_trading_days(days)
    start = tdays[0].strftime("%Y%m%d") + "0900"
    end = tdays[-1].strftime("%Y%m%d") + "1531"
    session = requests.Session()
    ok = fail = 0

    def _one(code):
        rows = fetch_minutes(session, code, start, end)
        return code, rows

    with ThreadPoolExecutor(max_workers=workers or config.CRAWL_WORKERS) as ex:
        futs = {ex.submit(_one, c): c for c in codes}
        for fut in as_completed(futs):
            code = futs[fut]
            try:
                _, rows = fut.result()
                upsert_bars(conn, code, rows)
                ok += 1
            except Exception as e:
                fail += 1
                log.warning("backfill %s 실패: %s", code, e)
    log.info("backfill 완료: ok=%d fail=%d (%s~%s)", ok, fail, start, end)
    purge_old(conn)
    return ok, fail


class Crawler:
    """매분 02~05초에 유니버스 전체의 최근 분봉을 증분 수집."""

    def __init__(self, codes, db_path=None):
        self.codes = list(codes)
        self.db_path = db_path or config.DB_PATH
        self.consec_fail_cycles = 0
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def crawl_cycle(self, conn, session):
        """직전 ~15분 구간을 재조회(지연 확정치 보정 포함). 성공 종목 수 반환."""
        now = mt.now_kst()
        start = (now - timedelta(minutes=15)).strftime("%Y%m%d%H%M")
        end = now.strftime("%Y%m%d%H%M")
        ok = 0
        errs = []

        def _one(code):
            time.sleep(random.uniform(0, 1.0))  # 종목간 미세 분산
            return code, fetch_minutes(session, code, start, end)

        with ThreadPoolExecutor(max_workers=config.CRAWL_WORKERS) as ex:
            futs = {ex.submit(_one, c): c for c in self.codes}
            for fut in as_completed(futs):
                code = futs[fut]
                try:
                    _, rows = fut.result()
                    upsert_bars(conn, code, rows)
                    ok += 1
                except Exception as e:
                    errs.append((code, str(e)[:80]))
        if errs:
            log.warning("cycle 실패 %d종목 (예: %s)", len(errs), errs[:3])
        return ok

    def run_forever(self):
        """메인 수집 루프. 장중에만 돌고, 실패해도 죽지 않는다."""
        conn = open_db(self.db_path)
        session = requests.Session()
        log.info("crawler 시작: %d 종목", len(self.codes))
        last_purge_day = None
        while not self._stop.is_set():
            now = mt.now_kst()
            if not mt.in_crawl_session(now):
                if self._stop.wait(30):
                    break
                continue
            # 매분 02~05초 사이 시작 시점으로 정렬
            target_sec = random.uniform(config.CRAWL_SEC_MIN, config.CRAWL_SEC_MAX)
            nxt = now.replace(second=0, microsecond=0) + timedelta(minutes=1, seconds=target_sec)
            if self._stop.wait(max(0.0, (nxt - mt.now_kst()).total_seconds())):
                break
            try:
                ok = self.crawl_cycle(conn, session)
                if ok == 0:
                    self.consec_fail_cycles += 1
                    if self.consec_fail_cycles >= config.CRAWL_FAIL_LIMIT:
                        notify(f"크롤링 {self.consec_fail_cycles}사이클 연속 전멸 — "
                               "네트워크/차단 점검 필요. 다음 분 재시도.")
                else:
                    self.consec_fail_cycles = 0
            except Exception as e:  # fail-safe: 어떤 예외도 루프를 죽이지 않음
                self.consec_fail_cycles += 1
                log.error("crawl_cycle 예외: %s", e)
                if self.consec_fail_cycles >= config.CRAWL_FAIL_LIMIT:
                    notify(f"크롤러 연속 예외 {self.consec_fail_cycles}회: {e}")
            day = mt.now_kst().date()
            if day != last_purge_day:
                purge_old(conn)
                last_purge_day = day
        conn.close()
        log.info("crawler 종료")

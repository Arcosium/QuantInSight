"""데이터 레이어 — CryptoBars 아카이브를 가격·거래대금 패널로 읽는다.

원본은 `CryptoBars/data/export/parquet/<BASE>.parquet` (1분봉, 2023-01-01~, **상장폐지 종목 포함**).
팀에 전달한 것과 **같은 파일**을 읽는다 — 분석 결과가 팀 데이터와 어긋나지 않게.

1분봉 전수는 8.3억 행이라 매번 훑을 수 없다. 목표 주기로 접어(bucket) 롱포맷으로 캐시하고,
필요할 때 피벗한다. 상장폐지 종목이 섞여 있어 **listwise 삭제는 금물**이다 —
어떤 종목은 2024년에 사라지고 어떤 종목은 2025년에 생긴다. 창 단위로 생존자를 고른다.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("crypto.data")

_SHARED = Path(os.environ.get("ARCTRADE_MINUTE_DATA_DIR", str(Path(__file__).resolve().parents[2] / "CryptoBars/data")))
SRC = Path(os.environ.get("CRYPTO_SRC", str(_SHARED / "export")))
CACHE = Path(__file__).resolve().parent.parent / "data" / "crypto"
CACHE.mkdir(parents=True, exist_ok=True)

MS_MIN = 60_000


def _con():
    import duckdb
    c = duckdb.connect()
    c.execute("PRAGMA threads=%d" % max(2, (os.cpu_count() or 4) - 2))
    c.execute("PRAGMA memory_limit='32GB'")
    c.execute("PRAGMA disable_progress_bar")
    c.execute("SET temp_directory='%s'" % (CACHE / "duckdb_tmp"))
    return c


def universe() -> pd.DataFrame:
    """coverage.csv — base 별 출처·기간·행수·상장폐지 여부. 백필이 만든 감사표."""
    df = pd.read_csv(SRC / "metadata" / "coverage.csv")
    df["delisted"] = df["delisted"].astype(bool)
    return df.set_index("base")


def _files(bases=None) -> str:
    """DuckDB read_parquet 인자(리터럴). bases 를 주면 그 파일만 읽는다(스캔량이 곧 시간).
    base 이름엔 한자·기호가 섞여 있다(`龙虾`, `币安人生`) — 반드시 따옴표로 감쌀 것."""
    paths = ([str(SRC / "parquet" / "*.parquet")] if not bases
             else [str(SRC / "parquet" / f"{b}.parquet") for b in bases])
    lit = ",".join("'" + p.replace("'", "''") + "'" for p in paths)
    return lit if len(paths) == 1 else "[" + lit + "]"


def bars(freq_min: int, bases=None, start_ms=None, end_ms=None) -> pd.DataFrame:
    """롱포맷 OHLCV 패널: ts(버킷 시작 UTC ms) · base · open/high/low/close · qv · n.

    `n` = 그 버킷 안에 실제로 존재한 1분봉 수. 결측(무거래) 판정에 쓴다 — 크립토
    마이크로캡은 몇 분씩 체결이 없어서, 이걸 안 보면 '가격이 안 변했다'와
    '거래가 없었다'를 구분 못 한다.
    """
    b = freq_min * MS_MIN
    src = _files(bases)
    where = []
    if start_ms is not None:
        where.append(f"ts >= {int(start_ms)}")
    if end_ms is not None:
        where.append(f"ts < {int(end_ms)}")
    w = ("WHERE " + " AND ".join(where)) if where else ""
    q = f"""
      SELECT (ts // {b}) * {b} AS ts, base,
             arg_min(open, ts) AS open, max(high) AS high, min(low) AS low,
             arg_max(close, ts) AS close,
             sum(quote_volume) AS qv, count(*)::INT AS n
      FROM read_parquet({src}, union_by_name=true) {w}
      GROUP BY 1, 2
    """
    with _con() as c:
        return c.execute(q).df()


def cached_bars(freq_min: int, rebuild=False) -> pd.DataFrame:
    """전 종목 패널 캐시. 클러스터링이 창을 옮겨가며 수십 번 읽으므로 한 번 구워둔다."""
    p = CACHE / f"bars_{freq_min}m.parquet"
    if rebuild or not p.exists():
        log.info("캐시 생성 %s — 1분봉 전수를 %d분으로 접는다(수 분 걸림)", p.name, freq_min)
        df = bars(freq_min)
        tmp = p.with_suffix(".tmp")
        df.to_parquet(tmp, compression="zstd", index=False)
        tmp.replace(p)
        log.info("캐시 완료: %s (%.0fM행)", p.name, len(df) / 1e6)
        return df
    return pd.read_parquet(p)


def pivot(df: pd.DataFrame, col="close") -> pd.DataFrame:
    """롱포맷 → 행=ts, 열=base 행렬. 결측은 채우지 않는다(생존 판정은 호출자 몫)."""
    return df.pivot(index="ts", columns="base", values=col).sort_index()


def alive(px: pd.DataFrame, min_cover=0.95) -> list[str]:
    """창 전 구간 살아 있던 종목 — 관측 비율이 min_cover 이상.

    상관행렬을 PSD 로 유지하려면 pairwise 결측이 없어야 한다. 상폐/신규 종목은
    이 단계에서 창별로 자연히 갈린다(전 기간 공통집합을 잡으면 생존편향이 되살아난다).
    """
    return list(px.columns[px.notna().mean() >= min_cover])


LIVE = Path(os.environ.get("CRYPTO_LIVE", str(_SHARED)))


def live_bars(freq_min: int, hours: int) -> pd.DataFrame:
    """**지금 시각까지의** 최근 몇 시간을 CryptoBars 실시간 저장소에서 읽는다.

    export/ 는 팀에 전달한 시점의 스냅샷이라 라이브엔 못 쓴다. 정본 `history`(월 파일)와
    오늘 치 버퍼 `bars`(일 파일)를 합쳐 읽는다 — CryptoBars 는 자정에 버퍼를 history 로
    접으므로 최근 48시간은 두 곳에 걸쳐 있다. 겹치는 (ts, base) 는 history 를 우선한다.
    """
    b = freq_min * MS_MIN
    lo = int(time.time() * 1000) - hours * 3_600_000
    h, d = LIVE / "history" / "**" / "*.parquet", LIVE / "bars" / "**" / "*.parquet"
    q = f"""
      WITH u AS (
        SELECT ts, base, open, high, low, close, quote_volume, 0 AS src
        FROM read_parquet('{h}', union_by_name=true) WHERE ts >= {lo}
        UNION ALL
        SELECT ts, base, open, high, low, close, quote_volume, 1 AS src
        FROM read_parquet('{d}', hive_partitioning=1) WHERE ts >= {lo}
      ), d AS (
        SELECT * FROM (SELECT *, row_number() OVER (PARTITION BY ts, base ORDER BY src) rn FROM u)
        WHERE rn = 1
      )
      SELECT (ts // {b}) * {b} AS ts, base, arg_min(open, ts) AS open, max(high) AS high,
             min(low) AS low, arg_max(close, ts) AS close,
             sum(quote_volume) AS qv, count(*)::INT AS n
      FROM d GROUP BY 1, 2
    """
    with _con() as c:
        return c.execute(q).df()


def log_returns(px: pd.DataFrame) -> pd.DataFrame:
    """로그수익률. 24/7 이라 일 경계 제거가 없다 — KRX 판과 다른 유일한 지점.
    남은 구멍은 직전가 ffill 후 0 수익률이 된다(무거래를 '변동 없음'으로 취급)."""
    p = px.ffill()
    return np.log(p / p.shift(1)).iloc[1:].replace([np.inf, -np.inf], np.nan).fillna(0.0)

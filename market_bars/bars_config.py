"""분봉 크롤러 상수 (구 Lag_Trading config.py 발췌 이관).

DB/유니버스 경로는 QuantInSight 의 config 값과 같은 환경변수를 읽어 **크롤러가 쓰는 곳과
lead-lag/정량 운용안가 읽는 곳이 항상 일치**하도록 한다(기본값 = QuantInSight/data/).
"""
import os
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _env(key, default, cast=str):
    v = os.environ.get(key)
    return cast(v) if v not in (None, "") else default


# 경로 — config.LEADLAG_BARS_DB / MARKET_UNIVERSE_CSV 와 동일 env 를 공유
DB_PATH = Path(os.getenv("LEADLAG_BARS_DB", str(_DATA_DIR / "bars.db")))
UNIVERSE_CSV = Path(os.getenv("MARKET_UNIVERSE_CSV", str(_DATA_DIR / "universe.csv")))

# 유니버스 규모(KOSPI 시총 상위 200 + KOSDAQ 상위 150 = 350)
UNIVERSE_KOSPI = _env("LAG_UNIVERSE_KOSPI", 200, int)
UNIVERSE_KOSDAQ = _env("LAG_UNIVERSE_KOSDAQ", 150, int)
UNIVERSE_SIZE = _env("LAG_UNIVERSE_SIZE", UNIVERSE_KOSPI + UNIVERSE_KOSDAQ, int)

# 수집/보존
BUFFER_DAYS = _env("LAG_BUFFER_DAYS", 20, int)     # 기동 백필 창(영업일)
RETAIN_DAYS = _env("LAG_RETAIN_DAYS", 0, int)      # 0 = 분봉 영구 보존(기본)
CRAWL_WORKERS = _env("LAG_CRAWL_WORKERS", 8, int)  # 분당 수집 스레드 수
CRAWL_SEC_MIN, CRAWL_SEC_MAX = 2, 5                # 매분 02~05초 사이 분산 호출
CRAWL_FAIL_LIMIT = 3                               # 연속 전멸 N회 → 알림 후 다음 분 재시도
REQUEST_TIMEOUT = 7                                # HTTP 타임아웃(초)

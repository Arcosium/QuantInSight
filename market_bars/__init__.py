"""분봉 크롤러 — KOSPI200+KOSDAQ150 (구 Lag_Trading core.crawler 이관, 사장 지시 2026-07-21).

Lag_Trading 프로젝트(그랜저 선행-후행 매매 전략)는 폐기됐지만, 그 크롤러가 수집하던
분봉 데이터는 QuantInSight 의 두 기능이 계속 소비한다:
  · tools/leadlag.py  — 30분 지평 선행-후행 신호(계량 팩터)
  · tools/stock_policy.py — 정량 운용안의 유니버스

그래서 크롤러만 이 자기완결 패키지로 이관해 유지한다(죽은 그랜저 엔진 analytics/strategy 는 폐기).
quantinsight.service 기동 시 `start_background()` 로 백그라운드 스레드 1회 시작 — 별도 systemd
유닛 없이 트레이딩 서버 프로세스 안에서 돈다. 장중(평일 09:00~15:35)에만 수집하고, 어떤 예외도
루프를 죽이지 않는다(fail-safe).
"""
import logging
import threading

log = logging.getLogger("quantinsight.bars")

_started = False
_lock = threading.Lock()


def start_background():
    """분봉 크롤러를 데몬 스레드로 1회 시작(중복 호출은 무시). 서버 기동 훅에서 호출."""
    import os
    if os.getenv("QIS_INTERNAL_BAR_CRAWLER", "0") != "1":
        log.info("분봉 수집은 CryptoBars 공통 수집기가 담당합니다")
        return
    global _started
    with _lock:
        if _started:
            return
        _started = True

    def _run():
        try:
            from . import crawler
            codes = [c for c, _ in crawler.load_universe()]
            if not codes:
                log.warning("bar crawler: 유니버스가 비어 크롤러를 시작하지 않음")
                return
            conn = crawler.open_db()
            n = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
            if n < len(codes) * 100:            # 이관 직후엔 bars.db 가 이미 차 있어 스킵됨
                log.info("bar crawler: 버퍼 부족(%d rows) — 백필 시작", n)
                crawler.backfill(conn, codes)
            conn.close()
            crawler.Crawler(codes).run_forever()
        except Exception as e:                  # 크롤러가 죽어도 트레이딩 서버는 계속
            log.error("bar crawler 스레드 종료(예외): %s", e)

    t = threading.Thread(target=_run, name="bar-crawler", daemon=True)
    t.start()
    log.info("bar crawler 백그라운드 스레드 시작(%d 종목 유니버스 로드 예정)", 0)

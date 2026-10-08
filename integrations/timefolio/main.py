"""ArcTrade 오케스트레이터 — 멀티프로세스 3-프로세스 아키텍처.

  [Data]    core.crawler  : 매분 02~05초 네이버 분봉 수집 → SQLite 롤링 버퍼
  [Engine]  core.analytics: 매분 전이 행렬(그랜저) 연산 → 지도 큐 전달
  [Exec]    core.strategy : 지도 소비 + 진입/청산 (모의 기본, LIVE 는 config)

사용:
  python3 main.py                  # 라이브 루프 (장중 자동 동작)
  python3 main.py --backfill       # 최근 3영업일 분봉 백필만 하고 종료
  python3 main.py --once           # 백필 가정 하에 엔진 1사이클 + 지도 출력 (dry-run)
"""
import argparse
import multiprocessing as mp
import signal
import sys
import time

import config
from utils.logging_setup import setup


def _data_proc(stop_event):
    from core import crawler
    log = setup("data")
    codes = [c for c, _ in crawler.load_universe()]
    conn = crawler.open_db()
    # 기동 시 버퍼가 비었으면 자동 백필
    n = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    if n < len(codes) * 100:
        log.info("버퍼 부족(%d rows) — 백필 시작", n)
        crawler.backfill(conn, codes)
    conn.close()
    cr = crawler.Crawler(codes)

    def _stopper():
        stop_event.wait()
        cr.stop()
    import threading
    threading.Thread(target=_stopper, daemon=True).start()
    cr.run_forever()


def _engine_proc(map_queue, stop_event):
    import os
    try:
        os.nice(config.ENGINE_NICE)   # Pool 자식에 상속 — 버스트 때 타 서비스 우선
    except OSError:
        pass
    from core.analytics import engine_loop
    engine_loop(map_queue, stop_event)


def _exec_proc(map_queue, stop_event):
    from core.strategy import trading_loop
    trading_loop(map_queue, stop_event)


def run_live():
    log = setup("main")
    ctx = mp.get_context("fork")
    stop_event = ctx.Event()
    map_queue = ctx.Queue(maxsize=4)

    def _spawn(name):
        target = {"data": _data_proc, "engine": _engine_proc, "exec": _exec_proc}[name]
        args = (stop_event,) if name == "data" else (map_queue, stop_event)
        # daemon 금지(기본값 유지): engine 이 DART·lead-lag 캐시를 백그라운드 스레드로
        # 돌리고, 자식을 만들 여지를 남긴다("daemonic processes are not allowed to have children").
        p = ctx.Process(target=target, args=args, name=f"lag-{name}")
        p.start()
        return p

    # QIS market_bars 가 이미 bars.db 를 채우면 자체 크롤러(data)를 끈다(LAG_RUN_CRAWLER=0).
    names = ("data", "engine", "exec") if config.RUN_CRAWLER else ("engine", "exec")
    procs = {name: _spawn(name) for name in names}
    log.info("%d-프로세스 기동: %s", len(procs), {k: p.pid for k, p in procs.items()})

    # 시그널 핸들러는 async-safe 해야 한다: mp.Event.set()/logging 은 내부 락을
    # 잡아 메인스레드가 같은 락 위에서 인터럽트되면 self-deadlock (futex 행 —
    # 2026-07-02 스모크에서 재현). 락 없는 bool 플래그만 세운다.
    shutdown = {"flag": False}

    def _sig(_s, _f):
        shutdown["flag"] = True
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        while not shutdown["flag"]:
            for name, p in procs.items():
                if not p.is_alive():           # fail-safe: 죽은 프로세스 재기동
                    log.error("%s 프로세스 사망(exit=%s) — 재기동", name, p.exitcode)
                    procs[name] = _spawn(name)
            for _ in range(10):                # 0.5s 단위로 플래그 반응
                if shutdown["flag"]:
                    break
                time.sleep(0.5)
    finally:
        log.info("종료 신호 — 프로세스 정리")
        stop_event.set()
        for p in procs.values():
            p.join(timeout=10)
        for p in procs.values():
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
            if p.is_alive():
                p.kill()
        log.info("종료 완료")


def run_backfill():
    from core import crawler
    setup("backfill")
    codes = [c for c, _ in crawler.load_universe()]
    conn = crawler.open_db()
    crawler.backfill(conn, codes)
    conn.close()


def run_once():
    import json
    from core import crawler
    from core.analytics import run_cycle
    log = setup("once")
    codes = [c for c, _ in crawler.load_universe()]
    conn = crawler.open_db()
    out = run_cycle(conn, codes)
    conn.close()
    if not out:
        log.error("사이클 실패 — 먼저 --backfill 을 실행했는지 확인")
        return 1
    print(json.dumps({"n_codes": out["n_codes"], "window": out["window"],
                      "n_eligible": out["n_eligible"], "n_candidates": out["n_candidates"],
                      "top_candidates": out["candidates"][:10]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ArcTrade orchestrator")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    if a.backfill:
        run_backfill()
    elif a.once:
        sys.exit(run_once())
    else:
        run_live()

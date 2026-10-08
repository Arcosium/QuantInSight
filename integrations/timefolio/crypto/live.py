"""라이브 루프 — CryptoBars 실시간 수집분 위에서 s-score 를 돌리고 장부를 굴린다.

24/7 이라 장 시간 판정이 없다. 그냥 계속 돈다.

  · 매 CRY_BAR_MIN×step 분마다 스냅샷 → 진입/청산
  · 클러스터 라벨은 `data/crypto/clusters.csv` 를 하루 1회 다시 읽는다
    (cluster.py 를 크론으로 주기 재적합하면 라벨이 갱신된다)
  · 백테스트와 **같은 engine.snapshot** 을 쓴다

🔒 **실주문은 아직 없다.** PaperLedger 만 있고 CRY_LIVE=1 이면 즉시 예외로 끊는다.
거래소 주문은 사장 승인 후에 붙인다 — 인증 클라이언트는 MultiEX 에 이미 있으므로
거기 어댑터를 연결하면 되지, 여기서 키를 새로 다루지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import time
from pathlib import Path

import pandas as pd

from crypto import data as D, engine

log = logging.getLogger("crypto.live")
STATE = D.CACHE
TRADES, POSITIONS = STATE / "paper_trades.csv", STATE / "paper_positions.json"
CLUSTERS = STATE / "clusters.csv"


class PaperLedger:
    """모의 장부. 체결가 = 결정 시점 종가 ± 슬리피지 절반씩(왕복 SLIP_RT 를 반으로 나눈다)."""

    def __init__(self, p=engine.P):
        self.p = p
        self.pos = json.loads(POSITIONS.read_text()) if POSITIONS.exists() else {}
        if not TRADES.exists():
            with open(TRADES, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["ts", "action", "base", "side", "price", "s",
                                        "half_life", "ret", "reason"])

    def _row(self, *a):
        with open(TRADES, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(a)

    def open(self, o, side, now):
        px = o["price"] * (1 + side * self.p.SLIP_RT / 2)
        self.pos[o["base"]] = {"side": side, "entry": px, "t": now,
                               "hl": o["half_life"], "s": o["s"]}
        self._row(now, "OPEN", o["base"], side, round(px, 8), o["s"], o["half_life"], "", "")
        log.info("진입 %s %s @ %.6g (s=%+.2f, 반감기 %.0f봉)",
                 "롱" if side > 0 else "숏", o["base"], px, o["s"], o["half_life"])

    def close(self, base, price, reason, now):
        p0 = self.pos.pop(base)
        px = price * (1 - p0["side"] * self.p.SLIP_RT / 2)
        ret = (px / p0["entry"] - 1.0) * p0["side"] - self.p.FEE_RT
        self._row(now, "CLOSE", base, p0["side"], round(px, 8), "", "", round(ret, 6), reason)
        log.info("청산 %s @ %.6g (%s) 순손익 %+.3f%%", base, px, reason, ret * 100)

    def save(self):
        tmp = POSITIONS.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.pos, ensure_ascii=False))
        tmp.replace(POSITIONS)


def load_clusters():
    """clusters.csv → (bases, labels dict). 없으면 클러스터 없이(고유포트폴리오만) 돈다."""
    if not CLUSTERS.exists():
        log.warning("clusters.csv 없음 — 클러스터 디민 없이 돈다 (python -m crypto.cluster 로 생성)")
        return {}
    df = pd.read_csv(CLUSTERS)
    return dict(zip(df["base"], df["cluster"]))


def cycle(led: PaperLedger, clusters, p=engine.P):
    """한 사이클: 최근 봉 로드 → 스냅샷 → 청산 → 진입."""
    hours = int(p.WINDOW_BARS * p.BAR_MIN / 60 * 1.5) + 2
    panel = D.live_bars(p.BAR_MIN, hours)
    if panel.empty:
        log.warning("실시간 봉 없음 — CryptoBars 수집기 확인")
        return
    px = D.pivot(panel, "close")
    cols = [c for c in px.columns if not clusters or c in clusters]
    px = px[cols].iloc[-(p.WINDOW_BARS + 1):]
    keep = list(px.columns[px.notna().mean() >= 0.9])
    if len(keep) < 20:
        log.warning("유효 종목 %d개 — 건너뜀", len(keep))
        return
    labels = [clusters[c] for c in keep] if clusters else None
    now = int(px.index[-1])
    snap = engine.snapshot(px[keep], labels, updated=now, p=p)
    if not snap:
        return
    sc = snap["scores"]

    for base in list(led.pos):
        po = led.pos[base]
        o = sc.get(base)
        if not o:
            continue
        held = (now - po["t"]) / (p.BAR_MIN * 60_000)
        why = None
        if abs(o["s"]) < p.STATARB_S_EXIT:
            why = "REVERT"
        elif o["s"] * po["side"] > p.STATARB_S_STOP:
            why = "SSTOP"
        elif po["hl"] > 0 and held >= 2 * po["hl"]:
            why = "TIMEOUT"
        elif (o["price"] / po["entry"] - 1.0) * po["side"] <= -0.05:
            why = "SL"
        if why:
            led.close(base, o["price"], why, now)

    room = p.MAX_POSITIONS - sum(1 for v in led.pos.values() if v["side"] > 0)
    for side, book in ((1, snap["longs"]), (-1, snap["shorts"])):
        n = room
        for o in book:
            if n <= 0:
                break
            if o["base"] not in led.pos:
                led.open(o, side, now)
                n -= 1
    led.save()
    log.info("사이클: %d종목 · 적격 %d · 롱후보 %d 숏후보 %d · 보유 %d",
             snap["n_codes"], snap["n_eligible"], snap["n_longs_raw"],
             snap["n_shorts_raw"], len(led.pos))


def loop(step_bars=4, p=engine.P):
    if os.environ.get("CRY_LIVE") == "1":
        raise NotImplementedError(
            "실주문 미연결. 거래소 어댑터는 사장 승인 후 MultiEX 인증 클라이언트로 붙인다.")
    led, clusters, loaded = PaperLedger(p), load_clusters(), time.time()
    stop = {"f": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("f", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("f", True))
    period = p.BAR_MIN * step_bars * 60
    while not stop["f"]:
        t0 = time.monotonic()
        try:
            if time.time() - loaded > 86_400:            # 클러스터 라벨 일 1회 재적재
                clusters, loaded = load_clusters(), time.time()
            cycle(led, clusters, p)
        except Exception as e:                            # 어떤 예외도 루프를 죽이지 않는다
            log.error("사이클 예외: %s", e, exc_info=True)
        for _ in range(int(max(5.0, period - (time.monotonic() - t0)))):
            if stop["f"]:
                break
            time.sleep(1)
    log.info("종료")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="크립토 stat-arb 라이브 루프(모의)")
    ap.add_argument("--once", action="store_true", help="1사이클만 (검증)")
    ap.add_argument("--step", type=int, default=4, help="몇 봉마다 결정할지")
    a = ap.parse_args()
    if a.once:
        cycle(PaperLedger(), load_clusters())
    else:
        loop(a.step)

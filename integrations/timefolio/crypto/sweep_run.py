"""문턱·속도 격자 1회 실행 — "총이익이 비용선을 어디서 넘는가"를 재는 게 목적.

비용을 0 으로 두고 돌린다. 그러면 `gross_avg_bps` 가 곧 **손익분기 왕복비용(bp)** 이다.
참고선: 바이낸스 USDⓈ-M 테이커 왕복 10bp · 메이커 왕복 4bp.

  python3 -m crypto.sweep_run --from 2026-02-01
"""
import argparse
import copy
import json
import logging

import pandas as pd

from crypto import backtest, engine

GRID = [
    # (이름, 진입문턱, 잔차창(봉), 결정주기(봉))  — 15분봉 기준
    ("기준 s1.25 · 창24h · 1시간마다", 1.25, 96, 4),
    ("고확신 s2.50 · 창24h · 1시간마다", 2.50, 96, 4),
    ("느림 s1.25 · 창72h · 4시간마다", 1.25, 288, 16),
    ("느림+고확신 s2.50 · 창72h · 4시간마다", 2.50, 288, 16),
]

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2026-02-01")
    a = ap.parse_args()
    st = int(pd.Timestamp(a.start, tz="UTC").timestamp() * 1000)
    rows = []
    for name, s_entry, win, step in GRID:
        p = copy.copy(engine.P)
        p.STATARB_S_ENTRY, p.WINDOW_BARS = s_entry, win
        p.FEE_RT = p.SLIP_RT = 0.0              # 비용 0 → gross 가 곧 손익분기 비용
        m = backtest.run(step_bars=step, start=st, p=p)
        m["config"] = name
        rows.append(m)
        print(f"{name:<38} 거래 {m.get('n',0):>6} · 총이익 {m.get('gross_avg_bps',0):>6.2f}bp "
              f"= 손익분기 비용 · 보유 {m.get('avg_hold_bars',0):>5.1f}봉 · "
              f"승률 {m.get('win_gross_pct',0):>4.1f}%", flush=True)
    (backtest.D.CACHE / "sweep.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

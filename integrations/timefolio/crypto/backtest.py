"""워크포워드 백테스트 — 라이브가 매 결정시점에 하는 일을 과거에 그대로 돌린다.

**라이브와 같은 함수(engine.snapshot)를 쓴다.** 백테스트 전용 로직을 따로 쓰면 그건
전략이 아니라 백테스트를 검증하는 것이다. 여기 있는 건 장부 관리와 비용 계산뿐이다.

무-lookahead 3중 방어:
  1) 스냅샷은 창의 마지막 행까지만 본다 (engine.snapshot)
  2) 클러스터·스크리닝을 refit_days 마다 **그 시점 이전 데이터로만** 다시 적합한다
     — 클러스터를 전 기간으로 한 번 적합하면 미래 섹터 구조가 새어 들어온다
  3) 체결가는 결정 봉의 **종가**, 즉 결정한 그 순간의 가격이다

⚠ 자금조달료(funding)는 데이터에 없다. 달러중립이라 상당 부분 상쇄되지만 정확히 0 은
아니다. `--funding-bps-day` 로 최악을 가정해 스트레스를 걸 수 있다.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import time

# config 가 BLAS 를 1스레드로 묶는다(KRX 엔진은 프로세스 병렬이라 그게 맞다). 백테스트는
# 반대로 매 결정마다 N×N eigh 를 한 번씩 도는 단일 프로세스라 스레드가 곧 속도다.
# numpy 가 로드되기 전에 걸어야 하고, config 는 setdefault 라 여기서 정한 값이 이긴다.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, str(max(2, (os.cpu_count() or 4) - 2)))

import numpy as np
import pandas as pd

from crypto import cluster, data as D, engine, screen

log = logging.getLogger("crypto.backtest")
DAY_MS = 86_400_000


def _fit_universe(hour_panel, upto_ms, win_days, k, n_factors):
    """결정시점 이전 hour 데이터만으로 스크리닝 + 클러스터 적합 → {bases, labels, spreads}.

    2023년 초엔 상장 종목이 160개도 안 돼 k=160 이 성립하지 않는다. k 를 반씩 낮춰가며
    가능한 최대를 쓴다 — 이걸 안 두면 초반 8개월이 **거래 0건으로 조용히 사라진다**
    (예외가 안 나서 '실패 0건'으로 통과해버리는, CryptoBars BUSD 사고와 같은 종류).
    """
    sl = hour_panel[(hour_panel["ts"] >= upto_ms - win_days * DAY_MS) & (hour_panel["ts"] < upto_ms)]
    if sl.empty:
        return None
    ks = tuple(sorted({max(10, k >> i) for i in range(5)}, reverse=True))
    f = cluster.fit(sl, 60, n_factors, ks=ks)
    if f is None:
        return None
    got = next((x for x in ks if x in f["labels"]), None)
    if got is None:
        return None
    return {"bases": f["bases"], "labels": f["labels"][got], "k": got,
            "spreads": f["stats"]["cs_spread"].to_dict()}


def run(freq=15, k=160, refit_days=30, step_bars=4, cluster_win_days=180,
        start=None, end=None, funding_bps_day=0.0, use_clusters=True, p=engine.P, dump=None):
    """워크포워드 시뮬. 반환: 지표 dict + 자산곡선.

    step_bars: 몇 봉마다 결정할지. 15분봉 × 4 = 1시간마다.
    포지션은 다리당 동일 명목, 롱·숏 같은 수 = 달러중립.
    """
    t0 = time.time()
    hour = D.cached_bars(60)
    panel = D.cached_bars(freq)
    if start:
        panel = panel[panel["ts"] >= start]
    if end:
        panel = panel[panel["ts"] < end]
    px_all = D.pivot(panel, "close")
    # 체결은 **다음 봉 시가**로 한다. 신호를 만든 그 봉의 종가로 체결하면 결정과 체결이
    # 같은 시각이라 1봉짜리 lookahead 가 된다 — stat-arb 백테스트가 부풀려지는 전형적 자리다.
    op_all = D.pivot(panel, "open").reindex_like(px_all)
    idx = px_all.index.to_numpy()
    log.info("패널 %d봉 × %d종목 (%s ~ %s)", len(idx), px_all.shape[1],
             pd.Timestamp(idx[0], unit="ms").date(), pd.Timestamp(idx[-1], unit="ms").date())

    W = p.WINDOW_BARS
    fund_bar = funding_bps_day / 10_000.0 * (freq / 1440.0)
    uni, next_refit = None, -1
    positions, trades, marks = {}, [], []
    equity, leg_w = 1.0, 1.0 / (2 * p.MAX_POSITIONS)

    ncol = {b: j for j, b in enumerate(px_all.columns)}
    nxt = op_all.to_numpy(np.float64)             # 다음 봉 시가 조회용(체결가)

    def fill(base, t):
        """t 에 결정 → t+1 봉 시가에 체결. 시가가 없으면 체결 불가(그 결정은 버린다)."""
        j = ncol.get(base)
        if j is None or t + 1 >= len(idx):
            return 0.0
        v = nxt[t + 1, j]
        return float(v) if v > 0 else 0.0

    for t in range(W + 1, len(idx) - 1, step_bars):
        now = int(idx[t])
        if now >= next_refit:                       # 클러스터·유니버스 재적합(과거 데이터만)
            u = _fit_universe(hour, now, cluster_win_days, k, p.N_FACTORS)
            if u:
                uni = u
                log.info("%s 유니버스 재적합: %d종목 · k=%d",
                         pd.Timestamp(now, unit="ms").date(), len(u["bases"]), u["k"])
            next_refit = now + refit_days * DAY_MS
        if not uni:
            continue
        cols = [b for b in uni["bases"] if b in px_all.columns]
        win = px_all[cols].iloc[t - W:t + 1]
        keep = win.notna().mean() >= 0.9
        cols = list(win.columns[keep])
        if len(cols) < 20:
            continue
        pos_in = {b: i for i, b in enumerate(uni["bases"])}
        labels = np.array([uni["labels"][pos_in[b]] for b in cols]) if use_clusters else None
        snap = engine.snapshot(win[cols], labels, uni["spreads"], updated=now, p=p)
        if not snap:
            continue
        sc = snap["scores"]

        # ── 청산 먼저 ──
        for b in list(positions):
            po = positions[b]
            o = sc.get(b)
            exit_px = fill(b, t)                  # 청산도 다음 봉 시가
            if not (exit_px > 0):
                continue
            held = t - po["t"]
            raw = (exit_px / po["entry"] - 1.0) * po["side"]
            s = o["s"] if o else None
            why = None
            if s is not None and abs(s) < p.STATARB_S_EXIT:
                why = "REVERT"
            elif s is not None and s * po["side"] > p.STATARB_S_STOP:
                why = "SSTOP"            # 반대로 더 벌어짐(롱인데 s 가 더 음수)
            elif po["hl"] > 0 and held >= 2 * po["hl"]:
                why = "TIMEOUT"
            elif raw <= -0.05:
                why = "SL"
            if why:
                net = raw - po["cost"] - fund_bar * held
                # 진입 시점의 **기대** 반전폭을 같이 남긴다. "기대치가 비용을 넘을 때만 들어간다"는
                # 문턱 전략이 성립하려면 exp_ret 이 실현 수익을 예측해야 하는데, 그게 사실인지
                # 아닌지를 재려면 쌍(기대, 실현)이 필요하다. 없으면 문턱을 아무리 올려도 헛수고다.
                trades.append({"base": b, "side": po["side"], "ret": raw, "net": net,
                               "held_bars": held, "reason": why,
                               "exp_ret": po["exp"], "s0": po["s"], "cost": po["cost"],
                               "day": str(pd.Timestamp(now, unit="ms").date())})
                equity += net * leg_w
                del positions[b]

        # ── 진입: 달러중립으로 양다리 같은 수 ──
        room = p.MAX_POSITIONS - sum(1 for v in positions.values() if v["side"] > 0)
        for side, book in ((1, snap["longs"]), (-1, snap["shorts"])):
            n = room
            for o in book:
                if n <= 0:
                    break
                b = o["base"]
                if b in positions:
                    continue
                entry = fill(b, t)
                if not (entry > 0):
                    continue
                positions[b] = {"side": side, "entry": entry, "t": t, "hl": o["half_life"],
                                "cost": o["cost"], "s": o["s"], "exp": o["exp_ret"]}
                n -= 1
        marks.append({"ts": now, "equity": equity, "n_pos": len(positions),
                      "n_long_cand": snap["n_longs_raw"], "n_short_cand": snap["n_shorts_raw"]})

    if dump and trades:
        # 거래 원장을 남긴다. 요약 통계를 하나 더 보려고 12분짜리 백테스트를 다시 돌리는 건
        # 낭비다 — 연도별·사유별 슬라이스는 이 파일로 사후에 얼마든지 한다.
        pd.DataFrame(trades).to_parquet(D.CACHE / dump, compression="zstd", index=False)
        log.info("거래 원장 %d건 → %s", len(trades), dump)
    return _report(trades, marks, freq, step_bars, time.time() - t0)


def _report(trades, marks, freq, step_bars, elapsed):
    if not trades:
        return {"n": 0, "error": "거래 0건", "elapsed_min": round(elapsed / 60, 1)}
    tr = pd.DataFrame(trades)
    eq = pd.DataFrame(marks).set_index("ts")["equity"]
    dd = (eq / eq.cummax() - 1.0).min()
    step_min = freq * step_bars
    per_year = 365 * 24 * 60 / step_min
    d = eq.diff().dropna()
    sharpe = float(d.mean() / d.std() * np.sqrt(per_year)) if d.std() > 1e-12 else 0.0
    days = max(1.0, (eq.index[-1] - eq.index[0]) / DAY_MS)
    return {
        "n": len(tr), "days": round(days),
        "final_equity": round(float(eq.iloc[-1]), 4),
        "ann_return_pct": round((float(eq.iloc[-1]) - 1.0) * 365 / days * 100, 2),
        "sharpe": round(sharpe, 2), "max_dd_pct": round(float(dd) * 100, 2),
        "win_gross_pct": round(float((tr["ret"] > 0).mean() * 100), 1),
        "win_net_pct": round(float((tr["net"] > 0).mean() * 100), 1),
        "gross_avg_bps": round(float(tr["ret"].mean() * 1e4), 2),
        # 거래당 총이익은 표준오차 없이는 못 읽는다. 크립토 3시간 수익률의 산포가 수백 bp 라
        # 거래 1,000건이어도 SE 가 10bp 수준이다 — t<2 면 '엣지 있음'이라고 말하면 안 된다.
        "gross_sd_bps": round(float(tr["ret"].std() * 1e4), 1),
        "gross_t": round(float(tr["ret"].mean() / (tr["ret"].std() / np.sqrt(len(tr)))), 2),
        "net_avg_bps": round(float(tr["net"].mean() * 1e4), 2),
        "avg_hold_bars": round(float(tr["held_bars"].mean()), 1),
        "long_net_bps": round(float(tr[tr.side > 0]["net"].mean() * 1e4), 2),
        "short_net_bps": round(float(tr[tr.side < 0]["net"].mean() * 1e4), 2),
        "reasons": tr["reason"].value_counts().to_dict(),
        "trades_per_day": round(len(tr) / days, 1),
        "elapsed_min": round(elapsed / 60, 1),
        **_edge_gate_check(tr),
    }


def _edge_gate_check(tr: pd.DataFrame) -> dict:
    """**"기대 반전폭 > 비용일 때만 진입"이 성립하는지** 재는 진단.

    2026-08-07 실측(1년·3,234거래): 기대-실현 상관 0.10, 기대 5분위 [50 · 77 · 105 · 149 · 340]bp
    에 대해 실현 [−0.5 · −6.8 · 8.7 · 8.9 · 23.0]bp.
    → **exp_ret 은 수준이 15배 부풀려져 있고 순위만 쓸모가 있다.** 그래서 `exp_ret >= cost`
    문턱은 무력하다 — 최저 분위조차 기대치가 50bp 라 비용 12bp 를 자동으로 통과한다.
    쓰려면 절대값이 아니라 **분위**로 잘라야 한다. 분위별 t 를 같이 내는 이유가 그것이다
    (상위 분위만 남기면 표본이 5분의 1로 줄어 유의성이 먼저 죽는다).
    """
    if "exp_ret" not in tr or tr["exp_ret"].nunique() < 5:
        return {}
    q = pd.qcut(tr["exp_ret"], 5, labels=False, duplicates="drop")
    g = tr.groupby(q)["ret"]
    n, mean, sd = g.count(), g.mean(), g.std()
    return {"exp_vs_real_corr": round(float(tr["exp_ret"].corr(tr["ret"])), 4),
            "exp_bps_by_quintile": [round(v * 1e4, 1) for v in tr.groupby(q)["exp_ret"].mean()],
            "real_bps_by_exp_quintile": [round(v * 1e4, 1) for v in mean],
            "real_t_by_exp_quintile": [round(float(m / (s / np.sqrt(c))), 2)
                                       for m, s, c in zip(mean, sd, n)],
            "n_by_exp_quintile": [int(v) for v in n]}


def sweep(grid, **kw):
    """진입 문턱 × 비용 격자. 전략이 **어디서 비용선을 넘는지**를 보는 게 목적이다.

    총이익은 좋은데 순이익이 음수인 상태에선 파라미터를 하나씩 만지는 것보다
    "몇 bp 아래여야 사는가"를 먼저 아는 게 빠르다.
    """
    out = []
    for s_entry, cost_bps in grid:
        p = copy.copy(engine.P)
        p.STATARB_S_ENTRY = s_entry
        p.FEE_RT, p.SLIP_RT = cost_bps / 1e4, 0.0
        m = run(p=p, **kw)
        m.update({"s_entry": s_entry, "cost_bps": cost_bps})
        out.append(m)
        log.info("s_entry=%.2f cost=%dbp → 거래 %s · 총 %s bp · 순 %s bp · Sharpe %s",
                 s_entry, cost_bps, m.get("n"), m.get("gross_avg_bps"),
                 m.get("net_avg_bps"), m.get("sharpe"))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="크립토 stat-arb 워크포워드 백테스트")
    ap.add_argument("--freq", type=int, default=15, help="봉 주기(분)")
    ap.add_argument("--k", type=int, default=160, help="클러스터 수(cluster.py 가 고른 값)")
    ap.add_argument("--step", type=int, default=4, help="몇 봉마다 결정할지")
    ap.add_argument("--refit-days", type=int, default=30)
    ap.add_argument("--from", dest="start", default=None, help="YYYY-MM-DD")
    ap.add_argument("--funding-bps-day", type=float, default=0.0)
    ap.add_argument("--ablation", action="store_true", help="클러스터 디민 on/off 비교")
    ap.add_argument("--sweep", action="store_true", help="진입문턱 × 비용 격자")
    a = ap.parse_args()
    st = int(pd.Timestamp(a.start, tz="UTC").timestamp() * 1000) if a.start else None
    kw = dict(freq=a.freq, k=a.k, refit_days=a.refit_days, step_bars=a.step,
              start=st, funding_bps_day=a.funding_bps_day)
    if a.sweep:
        g = [(s, c) for s in (1.25, 1.75, 2.25, 3.0) for c in (0, 4, 12)]
        print(json.dumps(sweep(g, **kw), ensure_ascii=False, indent=2))
    else:
        print("\n=== 클러스터 디민 ON ===")
        print(json.dumps(run(**kw), ensure_ascii=False, indent=2))
        if a.ablation:
            print("\n=== 클러스터 디민 OFF (고유포트폴리오만) ===")
            print(json.dumps(run(use_clusters=False, **kw), ensure_ascii=False, indent=2))

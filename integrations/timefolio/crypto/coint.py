"""페어 후보 검증 — `pairs.csv` 가 무작위 쌍보다 실제로 나은가.

팀에 후보 2,583쌍을 넘겼는데, 그게 아무 쌍이나 뽑은 것보다 낫다는 근거는 아직 없다.
**상관이 높다는 것과 공적분한다는 것은 다른 물건이다.** 클러스터가 상관 기준으로 뭉치고
표본 밖에서 유지된다는 것까지는 쟀지만(ARI 0.533), 그게 공적분으로 이어지는지는 안 쟀다.
이어지지 않으면 후보 목록은 무작위 추출과 다를 게 없고, 그럼 바로 알려야 한다.

설계 — 어떤 것도 같은 데이터로 두 번 쓰지 않는다:
  창 A (과거 180일)  클러스터를 짓고 **후보 쌍**을 뽑는다
  창 B (직후 180일)  거기서 Engle-Granger 공적분 검정 → 완전한 표본 밖
  대조군            같은 종목 풀에서 **서로 다른 클러스터**끼리 무작위로 같은 수만큼

세 가지를 본다:
  1) 기각률   후보군이 대조군보다 높은가 (BH FDR 보정 전/후)
  2) 반감기   공적분해도 너무 느리면 자본이 묶이고 너무 빠르면 비용에 먹힌다
  3) **진폭 대 비용**  이게 진짜다. 공적분해도 스프레드가 왕복비용보다 얕으면 못 먹는다.
     페어는 다리가 둘이라 비용도 두 배다 — 왕복 = 2다리 × 2회 × (수수료+슬리피지).

  python3 -m crypto.coint
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

from crypto import cluster, data as D, engine

log = logging.getLogger("crypto.coint")
DAY_MS = 86_400_000


def bh_reject(pvals: np.ndarray, alpha=0.05) -> np.ndarray:
    """Benjamini-Hochberg. 2,583개를 그냥 5%로 검정하면 관계가 없어도 129개가 유의하다."""
    n = len(pvals)
    if not n:
        return np.zeros(0, bool)
    order = np.argsort(pvals)
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = pvals[order] <= thresh
    out = np.zeros(n, bool)
    if passed.any():
        out[order[:np.flatnonzero(passed)[-1] + 1]] = True
    return out


def control_pairs(bases, labels, n, seed=0):
    """대조군: **서로 다른 클러스터**에서 무작위로 뽑은 쌍. 후보군과 같은 개수.

    같은 종목 풀에서 뽑으므로 유동성·변동성 조건은 후보군과 같다 — 차이는 오직
    '클러스터가 같은가' 하나다. 그래야 클러스터링의 기여만 분리해서 잴 수 있다.
    """
    rng = np.random.default_rng(seed)
    lab = dict(zip(bases, labels))
    seen, out = set(), []
    guard = 0
    while len(out) < n and guard < n * 50:
        guard += 1
        a, b = rng.choice(len(bases), 2, replace=False)
        x, y = bases[a], bases[b]
        if lab[x] == lab[y] or (x, y) in seen:
            continue
        seen.add((x, y))
        out.append((x, y))
    return out


def test_pairs(px: pd.DataFrame, pairs, p=engine.P):
    """창 B 가격으로 Engle-Granger. 반환: pvalue · 반감기(봉) · 스프레드 σ · 진폭/비용."""
    from statsmodels.tsa.stattools import coint

    lp = np.log(px.ffill().bfill())
    rows, resid_cols, keep = [], [], []
    for a, b in pairs:
        if a not in lp.columns or b not in lp.columns:
            continue
        y, x = lp[a].to_numpy(), lp[b].to_numpy()
        if not (np.isfinite(y).all() and np.isfinite(x).all()):
            continue
        try:
            tstat, pval, _ = coint(y, x, trend="c", autolag="AIC")
        except (ValueError, np.linalg.LinAlgError):
            continue
        xc = x - x.mean()
        beta = float(xc @ (y - y.mean()) / (xc @ xc)) if xc @ xc > 1e-18 else 0.0
        e = y - y.mean() - beta * (x - x.mean())     # 헤지 잔차 = 스프레드
        rows.append({"a": a, "b": b, "pvalue": pval, "tstat": tstat, "beta": beta})
        resid_cols.append(e)
        keep.append(True)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # 스프레드의 OU 적합 — 잔차 '수준' 자체가 평균회귀 대상이라 그대로 넣는다
    q = engine.P.__class__(**{**vars(p), "STATARB_MIN_OBS": 50})
    V = engine.ou_scores(np.column_stack(resid_cols), q)
    df["half_life"] = V["half_life"]
    df["sigma"] = V["sigma_eq"]
    df["ou_ok"] = V["ok"] & V["sig_ok"]
    # 진폭: 2σ 에서 진입해 평균으로 돌아오면 2σ 를 먹는다(로그수익 단위).
    # 비용: 다리 2개 × 진입·청산 2회 × (수수료+슬리피지).
    cost = 4.0 * (p.FEE_RT + p.SLIP_RT)
    df["edge_bps"] = df["sigma"] * 2.0 * 1e4
    df["cost_bps"] = cost * 1e4
    df["pays"] = df["edge_bps"] > df["cost_bps"]
    return df


def summarize(df: pd.DataFrame, name: str, alpha=0.05) -> dict:
    if df.empty:
        return {"name": name, "n": 0}
    rej = df["pvalue"] <= alpha
    bh = bh_reject(df["pvalue"].to_numpy(), alpha)
    good = bh & df["ou_ok"].to_numpy() & df["pays"].to_numpy()
    return {"name": name, "n": len(df),
            "reject_5pct": round(float(rej.mean()) * 100, 1),
            "reject_bh": round(float(bh.mean()) * 100, 1),
            "n_bh": int(bh.sum()),
            "median_half_life_bars": round(float(df.loc[bh, "half_life"].median()), 1) if bh.any() else None,
            "median_edge_bps": round(float(df.loc[bh, "edge_bps"].median()), 1) if bh.any() else None,
            "cost_bps": round(float(df["cost_bps"].iloc[0]), 1),
            "pays_pct": round(float(df.loc[bh, "pays"].mean()) * 100, 1) if bh.any() else 0.0,
            "tradeable": int(good.sum())}


def run(win_days=180, k=160, n_factors=2, alpha=0.05, seed=0, freq_min=60):
    t0 = time.time()
    panel = D.cached_bars(freq_min)
    hi = int(panel["ts"].max())
    b_lo = hi - win_days * DAY_MS
    a_lo = b_lo - win_days * DAY_MS
    A = panel[(panel["ts"] >= a_lo) & (panel["ts"] < b_lo)]
    B = panel[panel["ts"] >= b_lo]
    log.info("창A %s~%s (후보 생성) · 창B %s~%s (검정)",
             pd.Timestamp(a_lo, unit="ms").date(), pd.Timestamp(b_lo, unit="ms").date(),
             pd.Timestamp(b_lo, unit="ms").date(), pd.Timestamp(hi, unit="ms").date())

    fa = cluster.fit(A, freq_min, n_factors, ks=(k,))
    if fa is None or k not in fa["labels"]:
        return {"error": "창A 클러스터 실패"}
    lab = fa["labels"][k]
    cand = cluster._pairs(fa, k)
    cand = cand[~cand["same_asset_flag"]]                 # 같은 자산 쌍은 애초에 제외
    ctrl = control_pairs(fa["bases"], lab, len(cand), seed)
    log.info("후보 %d쌍 · 대조 %d쌍 (창A 종목 %d개)", len(cand), len(ctrl), len(fa["bases"]))

    pxB = D.pivot(B, "close")
    alive = set(D.alive(pxB, 0.95))
    pxB = pxB[[c for c in pxB.columns if c in alive]]
    cp = [(a, b) for a, b in zip(cand["a"], cand["b"]) if a in alive and b in alive]
    kp = [(a, b) for a, b in ctrl if a in alive and b in alive]
    log.info("창B 생존 후 후보 %d쌍 · 대조 %d쌍 — 검정 시작", len(cp), len(kp))

    dc = test_pairs(pxB, cp)
    dk = test_pairs(pxB, kp)
    out = {"window_a": f"{pd.Timestamp(a_lo, unit='ms').date()}~{pd.Timestamp(b_lo, unit='ms').date()}",
           "window_b": f"{pd.Timestamp(b_lo, unit='ms').date()}~{pd.Timestamp(hi, unit='ms').date()}",
           "k": k, "alpha": alpha,
           "candidate": summarize(dc, "후보(클러스터 내부)", alpha),
           "control": summarize(dk, "대조(무작위 교차)", alpha),
           "elapsed_min": round((time.time() - t0) / 60, 1)}
    # 설정을 파일명에 박는다. 처음엔 고정 이름이라 n_factors=0 시험 실행이 본 결과를 조용히
    # 덮어썼고, 그 상태로 팀 드라이브에 올라갔다(2026-08-08). 설정이 바뀌면 파일도 달라야 한다.
    tag = f"f{n_factors}k{k}"
    if not dc.empty:
        dc.sort_values("pvalue").to_csv(D.CACHE / f"coint_pairs_{tag}.csv", index=False)
    (D.CACHE / f"coint_{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="페어 후보 vs 무작위 쌍 공적분 검증(표본 밖)")
    ap.add_argument("--win-days", type=int, default=180)
    ap.add_argument("--k", type=int, default=160)
    a = ap.parse_args()
    r = run(a.win_days, a.k)
    print(json.dumps(r, ensure_ascii=False, indent=2))

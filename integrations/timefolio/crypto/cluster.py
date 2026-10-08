"""전수 상관 → 클러스터링 → **안정성 검증**. 페어 후보를 만드는 단계.

왜 클러스터링부터 하나: 1,057종목이면 558,096쌍이다. 여기 Engle-Granger 를 전수로 돌리면
5% 유의수준에서 거짓 양성만 2만 8천 쌍 나온다. 클러스터 내부만 검정하면 후보가 두 자릿수 배
줄어 다중검정이 감당 가능해진다. 클러스터링은 목적이 아니라 **검정 횟수를 줄이는 도구**다.

  0) 창(기본 180일) 단위로 자르고 그 창을 완주한 종목만 — 상폐·신규가 섞여 있어
     전 기간 공통집합을 잡으면 애써 없앤 생존편향이 되살아난다
  1) 유동성·가짜유동성 필터 (screen.py) — **창 시작 시점 정보만** 쓰므로 lookahead 없음
  2) 고유포트폴리오 회귀로 시장 모드 제거 (factors.residualize)
  3) 잔차 상관 → Marchenko-Pastur 잡음 클리핑 (factors.rmt_clip)
  4) d=√(2(1-ρ)) 계층 클러스터링
  5) 창 t 라벨을 창 t+1 에 적용해 **밖에서도 유지되는가** 측정
     · ARI(t, t+1)          라벨 자체의 재현성
     · oos_within − oos_all 창 t 클러스터가 창 t+1 에서도 초과 상관을 갖는가 ← 이게 본 지표

ARI 가 0.3 을 못 넘거나 oos 초과상관이 0 근처면 그 설정은 노이즈를 나눈 것이다.

사용:
  python3 -m crypto.cluster --report          # 스크리닝 분포만 (문턱 정하기 전에)
  python3 -m crypto.cluster                   # 전수 실행 + 안정성
  python3 -m crypto.cluster --rebuild-cache   # 1분봉 재집계부터
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from crypto import data as D
from core import factors
from crypto import screen

log = logging.getLogger("crypto.cluster")
OUT = D.CACHE
KS = (5, 10, 15, 20, 30, 40, 60, 80, 120, 160, 200, 250)
DAY_MS = 86_400_000
# k 선택 규칙(결과를 보고 고르면 그건 튜닝이 아니라 자기기만이라 먼저 못박는다):
# 세 하한을 모두 통과한 k 중 표본밖 초과상관이 가장 큰 것.
MIN_PAIRS = 300          # 후보 쌍이 이보다 적으면 k 를 아무리 키워도 쓸 수 없다
MIN_COVERED = 0.60       # 종목의 60% 는 3인 이상 클러스터에 남아야 한다(바스켓 최소 인원)
MIN_ARI = 0.30           # **평균이 아니라 최악의 창**에서도 이만큼은 재현돼야 한다


def windows(lo: int, hi: int, win_days: int, step_days: int):
    """[lo, hi) 를 겹치는 창으로 자른다. 마지막 창은 hi 에 붙인다(최신 데이터를 버리지 않게)."""
    w, s = win_days * DAY_MS, step_days * DAY_MS
    out = []
    t = lo
    while t + w <= hi:
        out.append((t, t + w))
        t += s
    if not out or out[-1][1] < hi - s // 2:
        out.append((hi - w, hi))
    return out


def fit(panel: pd.DataFrame, freq_min: int, n_factors=2, ks=KS, thresholds=None):
    """한 창 → {bases, labels{k: array}, resid, corr, stats}. 못 돌면 None."""
    st = screen.stats(panel, freq_min)
    keep = st.index[screen.passes(st, **(thresholds or {}))]
    px = D.pivot(panel, "close")[list(keep)]
    keep = [c for c in px.columns if c in set(D.alive(px))]
    if len(keep) < 20:
        return None
    px = px[keep].ffill().bfill()
    R = D.log_returns(px).to_numpy(np.float64)
    C, Z = factors.corr_linkage(R, n_factors)          # KRX 와 공유하는 수학(core/factors.py)

    from scipy.cluster.hierarchy import fcluster
    labels = {k: fcluster(Z, k, criterion="maxclust") for k in ks if k < len(keep)}
    return {"bases": keep, "labels": labels, "corr": C,
            "stats": st.loc[keep], "n_screened": int(len(st)), "n_kept": len(keep)}


def within_excess(C: np.ndarray, labels: np.ndarray) -> float:
    """클러스터 내부 평균상관 − 전체 평균상관. 클러스터가 정말 뭉쳐 있으면 양수."""
    iu = np.triu_indices_from(C, 1)
    all_mean = float(C[iu].mean())
    same = labels[iu[0]] == labels[iu[1]]
    return float(C[iu][same].mean() - all_mean) if same.any() else 0.0


def oos_excess(prev, cur, k: int, n_shuffle=20, seed=0):
    """창 t 라벨을 창 t+1 상관행렬에 적용한 초과상관. **표본 밖 검증**이라 이게 본 지표다.

    라벨을 섞은 귀무값도 같이 낸다. k 를 키우면 클러스터가 작아져 초과상관이 저절로
    올라가는 성질이 있어서(작은 집단의 평균은 원래 더 튄다), 귀무 대비로 봐야 한다.
    """
    common = [b for b in prev["bases"] if b in set(cur["bases"])]
    if len(common) < 20 or k not in prev["labels"]:
        return float("nan"), float("nan")
    pi = {b: i for i, b in enumerate(prev["bases"])}
    ci = {b: i for i, b in enumerate(cur["bases"])}
    lab = np.array([prev["labels"][k][pi[b]] for b in common])
    sub = cur["corr"][np.ix_([ci[b] for b in common], [ci[b] for b in common])]
    rng = np.random.default_rng(seed)
    null = np.mean([within_excess(sub, rng.permutation(lab)) for _ in range(n_shuffle)])
    return within_excess(sub, lab), float(null)


def shape(labels: np.ndarray) -> dict:
    """클러스터 크기 분포 — k 를 키우다 보면 대부분이 1인분(singleton)이 되어 지표가 무의미해진다.

    covered = 크기 3 이상 클러스터에 들어간 **종목** 비율. 초과상관은 k 를 키울수록 저절로
    오르므로(작은 집단일수록 평균이 튄다) 이 값으로 제동을 건다. 바스켓 전략에는 최소 3명이
    필요하다 — 2명짜리는 서로의 평균이라 디민하면 둘 다 0 이 된다.
    """
    sizes = np.bincount(labels)[1:]
    sizes = sizes[sizes > 0]
    return {"n_pairs": int((sizes * (sizes - 1) // 2).sum()),
            "singleton_frac": round(float((sizes == 1).sum() / len(sizes)), 3),
            "covered": round(float(sizes[sizes >= 3].sum() / sizes.sum()), 3),
            "size_p50": int(np.median(sizes)), "size_max": int(sizes.max())}


def run(freq_min=60, win_days=180, step_days=90, n_factors=2, ks=KS,
        thresholds=None, rebuild=False):
    t0 = time.time()
    panel = D.cached_bars(freq_min, rebuild=rebuild)
    lo, hi = int(panel["ts"].min()), int(panel["ts"].max()) + freq_min * 60_000
    wins = windows(lo, hi, win_days, step_days)
    log.info("창 %d개 (%d일 창 / %d일 보폭) · 패널 %.1fM행", len(wins), win_days, step_days,
             len(panel) / 1e6)

    fits, meta = [], []
    for i, (a, b) in enumerate(wins, 1):
        sl = panel[(panel["ts"] >= a) & (panel["ts"] < b)]
        f = fit(sl, freq_min, n_factors, ks, thresholds)
        tag = f"{pd.Timestamp(a, unit='ms'):%Y-%m-%d}~{pd.Timestamp(b, unit='ms'):%Y-%m-%d}"
        if f is None:
            log.warning("창 %s 실패(종목 부족)", tag)
            continue
        f["tag"] = tag
        fits.append(f)
        meta.append({"window": tag, "screened": f["n_screened"], "kept": f["n_kept"]})
        log.info("[%d/%d] %s · 스크린 통과 %d/%d", i, len(wins), tag, f["n_kept"], f["n_screened"])

    from sklearn.metrics import adjusted_rand_score
    stab = {}
    for k in ks:
        aris, oos, null, wex = [], [], [], []
        for p, c in zip(fits, fits[1:]):
            if k not in p["labels"] or k not in c["labels"]:
                continue
            common = [x for x in p["bases"] if x in set(c["bases"])]
            if len(common) < 20:
                continue
            pi = {b: i for i, b in enumerate(p["bases"])}
            ci = {b: i for i, b in enumerate(c["bases"])}
            aris.append(adjusted_rand_score([p["labels"][k][pi[b]] for b in common],
                                            [c["labels"][k][ci[b]] for b in common]))
            o, n = oos_excess(p, c, k)
            oos.append(o)
            null.append(n)
            wex.append(within_excess(c["corr"], c["labels"][k]))
        if not aris:
            continue
        stab[k] = {"ari": round(float(np.mean(aris)), 3), "ari_min": round(float(np.min(aris)), 3),
                   "oos_excess_corr": round(float(np.nanmean(oos)), 4),
                   "oos_null": round(float(np.nanmean(null)), 4),
                   "oos_net": round(float(np.nanmean(oos) - np.nanmean(null)), 4),
                   "in_excess_corr": round(float(np.mean(wex)), 4),
                   "n_pairs_of_windows": len(aris), **shape(fits[-1]["labels"][k])}

    ok = {k: v for k, v in stab.items()
          if v["n_pairs"] >= MIN_PAIRS and v["covered"] >= MIN_COVERED
          and v["ari_min"] >= MIN_ARI}
    best = max(ok, key=lambda k: ok[k]["oos_net"]) if ok else None
    res = {"freq_min": freq_min, "win_days": win_days, "step_days": step_days,
           "n_factors": n_factors, "windows": meta, "stability": stab, "best_k": best,
           "elapsed_min": round((time.time() - t0) / 60, 1)}

    if fits and best:
        last = fits[-1]
        lab = last["labels"][best]
        df = last["stats"].copy()
        df.insert(0, "cluster", lab)
        df["window"] = last["tag"]
        uni = D.universe()
        df["delisted"] = uni["delisted"].reindex(df.index).fillna(False)
        df["source"] = uni["source"].reindex(df.index)
        df.sort_values(["cluster", "adv_usd"], ascending=[True, False]).to_csv(OUT / "clusters.csv")
        _pairs(last, best).to_csv(OUT / "pairs.csv", index=False)
        doc = Path(__file__).resolve().parent / "PAIRS.md"     # 산출물 옆에 설명서를 붙여 보낸다
        if doc.exists():
            (OUT / "PAIRS.md").write_bytes(doc.read_bytes())
        res["n_all_pairs"] = len(last["bases"]) * (len(last["bases"]) - 1) // 2
        res["n_cluster_pairs"] = int(sum(np.bincount(lab)[1:] * (np.bincount(lab)[1:] - 1) // 2))
    (OUT / "stability.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    return res


def _stem(s: str) -> str:
    """'1000PEPE'·'10000000AIDOGE' → 'PEPE'·'AIDOGE'. 배수 접두사만 떼어낸 몸통."""
    return s.lstrip("0123456789") or s


def _same_asset(a: str, b: str) -> bool:
    """같은 자산의 다른 표기인가 — 배수 티커(1000PEPE/PEPE)나 한쪽이 다른 쪽을 품는 경우.

    이런 쌍의 스프레드는 통계적 차익거래가 아니라 **상환·단위 아비트라지**다. 평균회귀가
    자명하게 성립하므로 상관 상위를 싹쓸이하는데, 정작 스프레드가 비용 밑이라 못 먹는다.
    자동 판정이라 완벽하지 않다(LAYER/SOLAYER 처럼 이름만 겹치는 별개 토큰도 걸린다) —
    **플래그일 뿐 자동 제외는 하지 않는다.** 눈으로 확인하고 빼는 건 사용자 몫.
    """
    x, y = _stem(a).upper(), _stem(b).upper()
    return x == y or (len(x) >= 4 and x in y) or (len(y) >= 4 and y in x)


def _pairs(f, k) -> pd.DataFrame:
    """최신 창의 클러스터 내부 쌍 → 공적분 검정에 넘길 후보표(잔차상관 내림차순).

    받는 쪽이 다시 조인하지 않아도 되게 **두 다리의 제약조건을 쌍 단위로 접어서** 싣는다.
    유동성·비용은 항상 나쁜 다리가 결정하므로 min/max 로 접는 게 맞다.
    """
    lab, C, bs, st = f["labels"][k], f["corr"], f["bases"], f["stats"]
    size = pd.Series(lab, index=bs).map(pd.Series(lab).value_counts())
    iu = np.triu_indices_from(C, 1)
    same = lab[iu[0]] == lab[iu[1]]
    i, j = iu[0][same], iu[1][same]
    a = [bs[x] for x in i]
    b = [bs[y] for y in j]
    g = lambda col, ks: st[col].reindex(ks).to_numpy()      # noqa: E731
    return (pd.DataFrame({
        "a": a, "b": b, "cluster": lab[i], "cluster_size": size.reindex(a).to_numpy(),
        "resid_corr": C[i, j].round(4),
        # 쌍이 실제로 감당해야 하는 제약 — 나쁜 다리 기준
        "adv_usd_min": np.minimum(g("adv_usd", a), g("adv_usd", b)).round(0),
        "spread_max": np.maximum(g("cs_spread", a), g("cs_spread", b)).round(5),
        "vol_a": g("vol", a).round(3), "vol_b": g("vol", b).round(3),
        "wash_z_max": np.maximum(g("wash_z", a), g("wash_z", b)).round(2),
        # ⚠ 검정 전에 눈으로 확인할 것 — 같은 자산이면 공적분이 자명하고 못 먹는다
        "same_asset_flag": [_same_asset(x, y) for x, y in zip(a, b)],
    }).sort_values("resid_corr", ascending=False))


def report(freq_min=60, win_days=180, rebuild=False):
    """스크리닝 지표 분포만 출력 — 문턱을 감으로 정하지 않기 위한 사전 조사."""
    panel = D.cached_bars(freq_min, rebuild=rebuild)
    hi = int(panel["ts"].max())
    sl = panel[panel["ts"] >= hi - win_days * DAY_MS]
    st = screen.stats(sl, freq_min)
    q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    print(f"\n=== 최근 {win_days}일 스크리닝 분포 ({len(st)}종목) ===")
    print(st[["cover", "adv_usd", "amihud", "cs_spread", "zero_vol", "vol", "wash_z"]]
          .describe(percentiles=q).to_string(float_format=lambda x: f"{x:,.4g}"))
    print(f"\n기본 문턱 통과: {int(screen.passes(st).sum())}/{len(st)}")
    for name, m in [("거래대금<$1M/일", st["adv_usd"] < 1e6),
                    ("스프레드>0.4%", st["cs_spread"] > 0.004),
                    ("무변동봉>35%", st["zero_vol"] > 0.35),
                    ("변동성<20%", st["vol"] < 0.2),
                    ("wash_z>2", st["wash_z"] > 2.0)]:
        print(f"  {name:<16} {int(m.fillna(False).sum()):>5}종목 탈락")
    print("\nwash_z 상위 15 (거래대금 대비 가격충격이 비정상적으로 작음):")
    print(st.nlargest(15, "wash_z")[["adv_usd", "amihud", "wash_z", "zero_vol", "vol"]]
          .to_string(float_format=lambda x: f"{x:,.4g}"))
    return st


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="크립토 전수 상관 클러스터링 + 안정성 검증")
    ap.add_argument("--freq", type=int, default=60, help="봉 주기(분). 1분봉은 Epps 효과로 유동성 순위표가 된다")
    ap.add_argument("--win-days", type=int, default=180)
    ap.add_argument("--step-days", type=int, default=90)
    ap.add_argument("--factors", type=int, default=2)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--rebuild-cache", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(a.freq, a.win_days, a.rebuild_cache)
    else:
        r = run(a.freq, a.win_days, a.step_days, a.factors, rebuild=a.rebuild_cache)
        print(json.dumps({k: v for k, v in r.items() if k != "windows"},
                         ensure_ascii=False, indent=2))

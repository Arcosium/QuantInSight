"""정량 운용안(trend-rank) IS/OS/ROS 롤링 검증 (사장 지시 2026-09-25).

사장 규칙: 폴드는 3개월 롤링, **선택은 OS 에서만, 판정은 ROS 에서만**.
- 후보(사전 등록): 재조정 주기 5·10·20거래일 모멘텀. 비교 기준은 거래대금 상위 50 균등가중.
- 폴드 k 의 ROS 수익 = 직전 폴드(OS)에서 Sharpe 가 가장 높았던 주기의 폴드 k 일수익.
  IS(OS 이전 전 구간)는 규칙을 고정했으므로 적합에 쓰지 않고 참고로만 보고한다.
- 합격: ROS 합산 Sharpe ≥ 1.0, p ≤ 0.05, 같은 날짜 기준선보다 높은 Sharpe.
  합격 전에는 실계좌가 이 운용안으로 신규 매수하지 않는다(main_swarm 게이트).

실행: python3 -m tools.policy_validation   (결과 → ~/vault/QuantInSight/research/stock_policy_rolling.json)
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path.home()/"vault"/"QuantInSight"/"research"/"stock_policy_rolling.json"
EVERY = (5, 10, 20)
MIN_SHARPE, MAX_P = 1.0, .05


def _stats(r):
    r = np.asarray(r, dtype=float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return {"sharpe": None, "t": None, "p": None, "ret_pct": None, "mdd_pct": None, "days": len(r)}
    t = r.mean()/r.std(ddof=1)*math.sqrt(len(r))
    nav = np.cumprod(1+r)
    return {"sharpe": float(r.mean()/r.std(ddof=1)*math.sqrt(252)), "t": float(t),
            "p": float(math.erfc(abs(t)/math.sqrt(2))),                 # 양측, 정규근사
            "ret_pct": float((nav[-1]-1)*100), "mdd_pct": float((nav/np.maximum.accumulate(nav)-1).min()*100),
            "days": len(r)}


def walk_forward(returns: dict, bench: pd.Series):
    """returns: {every: pd.Series(일수익)} → 폴드별 IS/OS/ROS 표와 ROS 합산 판정."""
    idx = bench.index
    quarters = sorted(set(idx.to_period("Q")))
    folds, ros, ros_bench = [], [], []
    for os_q, ros_q in zip(quarters, quarters[1:]):
        os_mask, ros_mask, is_mask = idx.to_period("Q") == os_q, idx.to_period("Q") == ros_q, idx.to_period("Q") < os_q
        os_sh = {e: _stats(s[os_mask])["sharpe"] for e, s in returns.items()}
        if any(v is None for v in os_sh.values()):
            continue
        pick = max(EVERY, key=lambda e: (os_sh[e], -e))        # 동률이면 회전이 적은 긴 주기
        r = returns[pick][ros_mask]
        ros.append(r); ros_bench.append(bench[ros_mask])
        folds.append({"ROS": str(ros_q), "OS": str(os_q), "pick_every": pick,
                      "IS_sharpe": _stats(returns[pick][is_mask])["sharpe"] if is_mask.sum() > 20 else None,
                      "OS_sharpe": os_sh, "ROS_stats": _stats(r),
                      "ROS_bench_sharpe": _stats(bench[ros_mask])["sharpe"]})
    pooled = _stats(pd.concat(ros)) if ros else _stats([])
    pooled_bench = _stats(pd.concat(ros_bench)) if ros_bench else _stats([])
    ok = bool(pooled["sharpe"] is not None and pooled["sharpe"] >= MIN_SHARPE and pooled["p"] <= MAX_P
              and pooled["sharpe"] > (pooled_bench["sharpe"] or 0))
    # 다음 분기에 쓸 주기 = 마지막 완결 폴드(OS)에서 고른 값
    last_q = quarters[-1] if quarters else None
    live_every = 5
    if last_q is not None:
        m = idx.to_period("Q") == last_q
        sh = {e: _stats(s[m])["sharpe"] for e, s in returns.items()}
        if all(v is not None for v in sh.values()):
            live_every = max(EVERY, key=lambda e: (sh[e], -e))
    return {"folds": folds, "ROS_pooled": pooled, "ROS_bench_pooled": pooled_bench,
            "validated": ok, "live_every": live_every}


def run(market="KRX", start="2023-01-01"):
    from arcmarket.systematic import features, replay
    from tools.stock_policy import histories
    hist = histories(market, pd.Timestamp.now().normalize())
    frame = features(hist)
    curves = {e: replay(hist, frame, start=start, every=e) for e in EVERY}
    curves["bench"] = replay(hist, frame, start=start, method="equal_weight")
    def daily(c):
        s = pd.Series([x["v"] for x in c["curve"]], index=pd.to_datetime([x["d"] for x in c["curve"]]))
        return s.pct_change().dropna()
    rets = {e: daily(curves[e]) for e in EVERY}
    bench = daily(curves["bench"])
    res = walk_forward(rets, bench)
    res.update({"market": market, "start": start, "end": str(bench.index[-1].date()),
                "universe": len(hist), "fee_per_side": .002,
                "full_period": {str(e): curves[e]["metrics"] for e in EVERY} | {"bench": curves["bench"]["metrics"]},
                "rule": f"선택=OS(직전 분기) Sharpe 최대 주기, 판정=ROS 합산 Sharpe≥{MIN_SHARPE}·p≤{MAX_P}·기준선 초과",
                "caveats": ["유니버스가 현재 상장 종목(universe.csv) 기준이라 생존 편향이 있다 — 결과가 실제보다 좋게 나온다.",
                            "p 는 일수익 t 검정의 정규근사(자기상관 보정 없음)."],
                "created_at": datetime.now().isoformat()})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return res


def load():
    try:
        return json.loads(OUT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


if __name__ == "__main__":
    r = run()
    for f in r["folds"]:
        s = f["ROS_stats"]
        print(f"ROS {f['ROS']} pick={f['pick_every']:>2}d  sharpe={s['sharpe'] and round(s['sharpe'], 2)}  "
              f"ret={s['ret_pct'] and round(s['ret_pct'], 2)}%  bench={f['ROS_bench_sharpe'] and round(f['ROS_bench_sharpe'], 2)}")
    print("ROS pooled", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["ROS_pooled"].items()})
    print("bench pooled", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["ROS_bench_pooled"].items()})
    print("validated", r["validated"], "live_every", r["live_every"])

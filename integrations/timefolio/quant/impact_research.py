"""Reproducible descriptive study of public trades and displayed liquidity.

Predeclared pilot: 100 ms directional bursts, upper-quartile notional, 11 s
spacing, 10 s follow-up. No parameter is selected using the response sign.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = np.round(np.arange(0.1, 10.01, 0.1), 1)
METHOD = {
    "burst_ms": 100, "direction_dominance": 0.8, "large_quantile": 0.75,
    "event_spacing_ms": 11000, "followup_seconds": 10,
    "max_book_age_ms": 1000, "depletion_fraction": 0.2,
    "recovery_sustain_seconds": 0.3, "bootstrap_repetitions": 2000,
    "bootstrap_seed": 928, "regime_quantiles": [1/3, 2/3],
    "matching_quantity_ratio": 1.5, "matching_time_minutes": 10,
}


def reconstruct(raw_path):
    counters = Counter()
    bids, asks = {}, {}
    active_conn, last_u, last_ts, epoch = None, -1, -1, 0
    times, epochs, conns, levels, trades = [], [], [], [], []
    seen_trades = set()
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            msg, conn = row["message"], row["segment"]
            topic = msg.get("topic", "")
            if topic.startswith("orderbook."):
                d = msg["data"]
                if msg["type"] == "snapshot":
                    bids, asks = {}, {}
                    active_conn, last_u, last_ts = conn, -1, -1
                    epoch += 1
                    counters["snapshots"] += 1
                if active_conn != conn:
                    counters["delta_without_snapshot"] += 1
                    continue
                ts = msg.get("cts")
                if ts is None:
                    counters["missing_matching_timestamp"] += 1
                    continue
                if d["u"] <= last_u or ts < last_ts:
                    counters["duplicate_or_reordered_books"] += 1
                    continue
                last_u, last_ts = d["u"], ts
                for target, key in ((bids, "b"), (asks, "a")):
                    for price, quantity in d[key]:
                        p, q = float(price), float(quantity)
                        if q == 0:
                            target.pop(p, None)
                        else:
                            target[p] = q
                b = sorted(bids.items(), reverse=True)[:50]
                a = sorted(asks.items())[:50]
                bids, asks = dict(b), dict(a)
                if len(b) < 50 or len(a) < 50 or b[0][0] >= a[0][0]:
                    counters["invalid_or_shallow_books"] += 1
                    epoch += 1
                    continue
                times.append(ts)
                epochs.append(epoch)
                conns.append(conn)
                levels.append(np.asarray([b, a], dtype=np.float64))
            elif topic.startswith("publicTrade."):
                for trade in msg["data"]:
                    if trade["i"] in seen_trades:
                        counters["duplicate_trades"] += 1
                        continue
                    seen_trades.add(trade["i"])
                    if trade.get("BT") or trade.get("RPI"):
                        counters["block_or_rpi_trades_excluded"] += 1
                        continue
                    trades.append((int(trade["T"]), conn, trade["S"], float(trade["v"]), float(trade["p"])))
    if not levels or not trades:
        raise ValueError("분석 가능한 호가와 체결이 모두 필요합니다.")
    times = np.asarray(times, dtype=np.int64)
    order = np.argsort(times, kind="stable")
    books = np.stack(levels)[order]
    data = {"times": times[order], "epochs": np.asarray(epochs)[order],
            "connections": np.asarray(conns)[order], "books": books,
            "trades": pd.DataFrame(trades, columns=["ts", "connection", "side", "quantity", "price"]),
            "quality": dict(counters)}
    return data


def _curve_stats(values, groups=None, seed=928):
    """Resample whole one-minute clusters, preserving within-minute dependence."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"n": 0, "mean": [], "lower": [], "upper": [], "clusters": 0}
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups if groups is not None else np.arange(len(values)))
    unique = np.unique(groups)
    means = np.nanmean(values, axis=0)
    if len(unique) < 2:
        return {"n": len(values), "mean": means.tolist(), "lower": [None]*values.shape[1],
                "upper": [None]*values.shape[1], "clusters": len(unique)}
    blocks = [values[groups == key] for key in unique]
    sums = np.asarray([np.nansum(b, axis=0) for b in blocks])
    ns = np.asarray([np.sum(np.isfinite(b), axis=0) for b in blocks])
    selected = rng.integers(0, len(blocks), size=(METHOD["bootstrap_repetitions"], len(blocks)))
    numer, denom = sums[selected].sum(axis=1), ns[selected].sum(axis=1)
    samples = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)
    return {"n": len(values), "mean": means.tolist(),
            "lower": np.nanpercentile(samples, 2.5, axis=0).tolist(),
            "upper": np.nanpercentile(samples, 97.5, axis=0).tolist(), "clusters": len(unique)}


def first_sustained(values, threshold, *, above=True, after_index=0):
    condition = np.asarray(values) >= threshold if above else np.asarray(values) <= threshold
    # 0.3 s sustained requires the first point plus three following 0.1 s points.
    for i in range(after_index, len(values)-3):
        if condition[i:i+4].all():
            return float(HORIZONS[i])
    return None


def recovery_summary(times, count):
    observed = [t for t in times if t is not None]
    # All observations have the same follow-up; nonrecoveries remain in denominator.
    median = sorted(observed)[math.ceil(count/2)-1] if count and len(observed) >= math.ceil(count/2) else None
    return {"n": count, "recovered": len(observed), "censored": count-len(observed),
            "fraction": len(observed)/count if count else None,
            "median_seconds": median,
            "recovered_only_median_seconds": float(np.median(observed)) if observed else None}


def analyse(data, manifest):
    times, books = data["times"], data["books"]
    epochs, conns = data["epochs"], data["connections"]
    trades = data["trades"].copy()
    trades["bucket"] = trades.ts // 100 * 100
    trades["notional"] = trades.quantity * trades.price
    trades["buy_qty"] = np.where(trades.side == "Buy", trades.quantity, 0)
    trades["sell_qty"] = np.where(trades.side == "Sell", trades.quantity, 0)
    burst = trades.groupby(["connection", "bucket"], as_index=False).agg(
        quantity=("quantity", "sum"), notional=("notional", "sum"),
        buy=("buy_qty", "sum"), sell=("sell_qty", "sum"), trades=("ts", "size"))
    burst["dominance"] = np.maximum(burst.buy, burst.sell)/burst.quantity
    burst["sign"] = np.where(burst.buy >= burst.sell, 1, -1)
    burst["dominant_qty"] = np.maximum(burst.buy, burst.sell)
    directional = burst[burst.dominance >= METHOD["direction_dominance"]].sort_values("bucket")
    threshold = float(directional.notional.quantile(METHOD["large_quantile"]))
    candidates = directional[directional.notional >= threshold]
    mid = (books[:, 0, 0, 0]+books[:, 1, 0, 0])/2
    spread = (books[:, 1, 0, 0]-books[:, 0, 0, 0])/mid*1e4
    skip = Counter()
    events = []
    last_event = -np.inf
    for row in candidates.itertuples():
        t0 = int(row.bucket)
        pre = int(np.searchsorted(times, t0, side="left")-1)
        before1 = int(np.searchsorted(times, t0-1000, side="right")-1)
        future = np.searchsorted(times, t0+(HORIZONS*1000).astype(int), side="right")-1
        if before1 < 0 or future[-1] >= len(times) or t0+10000 > times[-1]:
            skip["window_boundary"] += 1
            continue
        if conns[pre] != row.connection or epochs[before1] != epochs[future[-1]]:
            skip["connection_or_snapshot_boundary"] += 1
            continue
        ages = t0+(HORIZONS*1000).astype(int)-times[future]
        if t0-times[pre] > METHOD["max_book_age_ms"] or ages.max() > METHOD["max_book_age_ms"] or t0-1000-times[before1] > METHOD["max_book_age_ms"]:
            skip["stale_book"] += 1
            continue
        if t0-last_event < METHOD["event_spacing_ms"]:
            skip["overlapping_event"] += 1
            continue
        side_idx = 1 if row.sign == 1 else 0
        pre_levels = books[pre, side_idx, :5]
        low, high = pre_levels[:, 0].min(), pre_levels[:, 0].max()
        p = books[future, side_idx, :, 0]
        q = books[future, side_idx, :, 1]
        # Anchored price band, never re-label a deeper level as the original level.
        coverage = p[:, -1] >= high if side_idx == 1 else p[:, -1] <= low
        fixed = np.sum(np.where((p >= low) & (p <= high), q, 0), axis=1)
        fixed[~coverage] = np.nan
        if not coverage.all():
            skip["anchor_outside_observed_depth"] += 1
            continue
        depth = float(pre_levels[:, 1].sum())
        normalized = fixed/depth
        rolling_depth = books[future, side_idx, :5, 1].sum(axis=1)/depth
        response = row.sign*(mid[future]-mid[pre])/mid[pre]*1e4
        spread_path = (books[future,1,0,0]-books[future,0,0,0])/mid[pre]*1e4
        widened = bool(spread_path[0] > spread[pre]+1e-8)
        spread_recovery = first_sustained(spread_path, float(spread[pre])+1e-8, above=False, after_index=1) if widened else None
        depleted = bool(normalized[0] <= 1-METHOD["depletion_fraction"])
        t90 = first_sustained(normalized, .9, after_index=1) if depleted else None
        t50gap = first_sustained(normalized, (1+normalized[0])/2, after_index=1) if depleted else None
        events.append({"ts": t0, "minute": t0//60000, "side": "buy" if row.sign == 1 else "sell",
                       "quantity": float(row.dominant_qty), "notional": float(row.notional),
                       "depth": depth, "pre_spread_bps": float(spread[pre]),
                       "pre_return_bps": float(row.sign*(mid[pre]-mid[before1])/mid[before1]*1e4),
                       "impact": response, "depth_fixed": normalized, "depth_rolling": rolling_depth,
                       "spread": spread_path, "depleted": depleted, "t90": t90,
                       "spread_widened": widened, "spread_recovery_seconds": spread_recovery,
                       "t50_gap": t50gap, "book_index": pre})
        last_event = t0
    if len(events) < 4:
        raise ValueError(f"비중복 사건 {len(events)}건: 최소 4건 필요. 수집 기간을 늘리세요.")
    depth_limits = np.quantile([e["depth"] for e in events], METHOD["regime_quantiles"])
    for e in events:
        e["regime"] = "thin" if e["depth"] <= depth_limits[0] else "thick" if e["depth"] >= depth_limits[1] else "middle"

    def curve(selected, key):
        return _curve_stats([e[key] for e in selected], [e["minute"] for e in selected])

    groups = {}
    for name in ["all", "thin", "middle", "thick"]:
        selected = events if name == "all" else [e for e in events if e["regime"] == name]
        dep = [e for e in selected if e["depleted"]]
        wide = [e for e in selected if e["spread_widened"]]
        groups[name] = {"n": len(selected), "median_quantity": float(np.median([e["quantity"] for e in selected])) if selected else None,
                        "median_depth": float(np.median([e["depth"] for e in selected])) if selected else None,
                        "impact": curve(selected, "impact"), "spread": curve(selected, "spread"),
                        "pre_spread_mean_bps": float(np.mean([e["pre_spread_bps"] for e in selected])) if selected else None,
                        "spread_recovery": recovery_summary([e["spread_recovery_seconds"] for e in wide],len(wide)),
                        "depth_fixed": curve(dep, "depth_fixed"), "depth_rolling": curve(dep, "depth_rolling"),
                        "t90": recovery_summary([e["t90"] for e in dep], len(dep)),
                        "t50_gap": recovery_summary([e["t50_gap"] for e in dep], len(dep))}
    # Same taker side, quantity ratio <= 1.5, <= 10 min apart, no re-use.
    thin = [e for e in events if e["regime"] == "thin"]
    thick = [e for e in events if e["regime"] == "thick"]
    used, pairs = set(), []
    for e in thin:
        options = [(abs(math.log(e["quantity"]/other["quantity"])), i, other)
                   for i, other in enumerate(thick)
                   if i not in used and other["side"] == e["side"]
                   and max(other["quantity"], e["quantity"])/min(other["quantity"], e["quantity"]) <= 1.5
                   and abs(other["ts"]-e["ts"]) <= 600000]
        if options:
            _, i, other = min(options, key=lambda x: x[0])
            used.add(i)
            pairs.append((e, other))
    matched = {"pairs": len(pairs), "difference": _curve_stats([a["impact"]-b["impact"] for a,b in pairs]),
               "thin": _curve_stats([a["impact"] for a,b in pairs]),
               "thick": _curve_stats([b["impact"] for a,b in pairs]),
               "quantity_ratio_median": float(np.median([max(a["quantity"],b["quantity"])/min(a["quantity"],b["quantity"]) for a,b in pairs])) if pairs else None,
               "inference": "같은 방향·비슷한 규모를 짝지은 탐색적 비교. 인과효과 추정 아님."}
    # Peak location is descriptive; no forced exponential fit or permanent-impact claim.
    average = np.asarray(groups["all"]["impact"]["mean"])
    peak = int(np.argmax(average))
    half = first_sustained(average, average[peak]/2, above=False, after_index=peak+1) if average[peak] > 0 else None
    decay = {"peak_bps": float(average[peak]), "peak_at_seconds": float(HORIZONS[peak]),
             "half_level_at_seconds": half,
             "half_decay_seconds": half-float(HORIZONS[peak]) if half is not None else None,
             "end_bps": float(average[-1])}
    snapshots = []
    # Typical pre-event depth, chosen without post-event returns.
    for regime in ["thin", "thick"]:
        for side in ["buy", "sell"]:
            pool = [e for e in events if e["regime"] == regime and e["side"] == side]
            if not pool:
                continue
            target = np.median([e["depth"] for e in pool])
            event = min(pool, key=lambda e: abs(e["depth"]-target))
            idx = event["book_index"]
            snapshots.append({"id": f"{regime}-{side}", "regime": regime, "event_side": side,
                              "timestamp_ms": int(times[idx]), "event_timestamp_ms": event["ts"],
                              "bids": books[idx, 0].tolist(), "asks": books[idx, 1].tolist()})
    calibration = {}
    for name in ["all", "thin", "thick"]:
        stat = groups[name]["t50_gap"]
        value = stat["median_seconds"]
        usable = stat["n"] >= 8 and value is not None and value > .1
        calibration[name] = {"half_time_seconds": round(value-.1, 3) if usable else None,
                             "depletion_events": stat["n"], "recovered": stat["recovered"],
                             "source": "fixed-band net-depth deficit half recovery, median" if usable else "insufficient or censored observations",
                             "empirical": usable}
    result = {"schema_version": 1, "symbol": manifest["symbol"], "venue": "Bybit spot",
              "started_utc": datetime.fromtimestamp(times[0]/1000, timezone.utc).isoformat(),
              "ended_utc": datetime.fromtimestamp(times[-1]/1000, timezone.utc).isoformat(),
              "duration_seconds": float((times[-1]-times[0])/1000),
              "book_updates": len(times), "trades": len(trades), "bursts": len(burst),
              "directional_bursts": len(directional), "large_candidates": len(candidates),
              "large_notional_threshold": threshold, "quality": data["quality"],
              "exclusions": dict(skip), "method": METHOD, "regime_depth_thresholds": depth_limits.tolist(),
              "horizons_seconds": HORIZONS.tolist(), "groups": groups, "matched": matched,
              "decay": decay, "snapshots": snapshots, "calibration": calibration,
              "limitations": ["한 종목·30분 내외의 관찰 연구이며 실제 내 주문의 인과효과가 아님",
                              "호가 변화에는 신규 주문·취소·체결이 섞이며 신규 공급량을 따로 식별하지 않음",
                              "표시되지 않는 RPI·숨은 유동성은 분석에서 제외",
                              "고정 가격 구간의 잔량 회복과 현재 최우선 주변 유동성 회복은 서로 다름",
                              "10초 이후의 가격·잔량 변화와 영구 충격은 판단하지 않음",
                              "신뢰구간은 1분 군집 부트스트랩의 점별 95% 구간이며 짝 비교는 쌍 단위 탐색 구간"]}
    return result, events


def export_artifacts(result, events, output):
    output.mkdir(parents=True, exist_ok=False)
    (output/"study.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    records = []
    for e in events:
        r = {k:v for k,v in e.items() if k not in ["impact", "depth_fixed", "depth_rolling", "spread"]}
        for h in [.1, .5, 1, 2, 5, 10]:
            j = int(round(h*10))-1
            r[f"impact_{h:g}s_bps"] = float(e["impact"][j])
            r[f"depth_{h:g}s_ratio"] = float(e["depth_fixed"][j])
        records.append(r)
    pd.DataFrame(records).to_csv(output/"events.csv", index=False)
    chart(result, output)


def chart(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#b4b9bd", "axes.labelcolor": "#242a2b",
                         "text.color": "#242a2b", "xtick.color": "#51585a", "ytick.color": "#51585a",
                         "font.size": 10, "axes.unicode_minus": False, "figure.dpi": 150})
    t = result["horizons_seconds"]
    accent, ink = "#9b3c28", "#242a2b"

    def draw(ax, stats, label, color, linestyle="-"):
        if not stats["n"]:
            return
        ax.plot(t, stats["mean"], label=label, color=color, lw=1.8, ls=linestyle)
        if stats["lower"] and stats["lower"][0] is not None:
            ax.fill_between(t, stats["lower"], stats["upper"], color=color, alpha=.08, linewidth=0)
        ax.grid(axis="y", alpha=.16, linewidth=.6)
        ax.set_xlabel("체결 묶음 시작 후 경과 시간 (초)")

    fig, ax = plt.subplots(figsize=(8.6, 4.0), layout="constrained")
    g = result["groups"]["all"]
    draw(ax, g["depth_fixed"], f"원래 가격 구간 · {g['depth_fixed']['n']}건", accent)
    draw(ax, g["depth_rolling"], "매 시점 최우선 5단계", ink, "--")
    ax.axhline(.9, ls=":", color="#777777", lw=1, label="사건 전 잔량의 90%")
    ax.set_ylabel("사건 전 잔량 대비 비율")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(output/"04_refill.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.0), layout="constrained")
    for name, label, color in [("thin", "얇은 호가", accent), ("thick", "두꺼운 호가", ink)]:
        g = result["groups"][name]
        draw(ax, g["impact"], f"{label} · {g['n']}건", color)
    ax.axhline(0, color="#999999", lw=.8)
    ax.set_ylabel("방향을 맞춘 mid price 변화 (bp)")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(output/"05_regime.png"); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.0), sharex=True, layout="constrained")
    draw(axes[0], result["groups"]["all"]["impact"], "전체 큰 체결", accent)
    axes[0].set_ylabel("mid price 변화 (bp)")
    axes[0].axhline(0, color="#999999", lw=.8)
    axes[0].set_xlabel("")
    draw(axes[1], result["groups"]["all"]["spread"], "스프레드", ink)
    axes[1].axhline(result["groups"]["all"]["pre_spread_mean_bps"], color="#777777", ls=":", lw=1, label="사건 전 평균")
    axes[1].legend(frameon=False,fontsize=9)
    axes[1].set_ylabel("스프레드 (bp)")
    fig.savefig(output/"06_decay.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.0), layout="constrained")
    matched = result["matched"]
    draw(ax, matched["difference"], f"얇음 − 두꺼움 · {matched['pairs']}쌍", accent)
    ax.axhline(0, color="#999999", lw=.8)
    ax.set_ylabel("짝 비교 가격 반응 차이 (bp)")
    if matched["pairs"]:
        ax.legend(frameon=False, fontsize=9)
    else:
        ax.text(.5, .5, "조건에 맞는 비교 쌍 없음", transform=ax.transAxes, ha="center")
    fig.savefig(output/"05_matched.png"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.input/"manifest.json").read_text())
    if not manifest.get("complete"):
        raise SystemExit("완료된 수집본만 분석합니다.")
    data = reconstruct(args.input/"raw.jsonl.gz")
    result, events = analyse(data, manifest)
    result["raw_sha256"] = hashlib.sha256((args.input/"raw.jsonl.gz").read_bytes()).hexdigest()
    result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    import platform
    result["software"] = {"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__}
    result["source_manifest"] = manifest
    result["generated_utc"] = datetime.now(timezone.utc).isoformat()
    export_artifacts(result, events, args.output)
    print(json.dumps({"events": len(events), "depleted": result["groups"]["all"]["t90"],
                      "matched_pairs": result["matched"]["pairs"], "decay": result["decay"],
                      "calibration": result["calibration"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

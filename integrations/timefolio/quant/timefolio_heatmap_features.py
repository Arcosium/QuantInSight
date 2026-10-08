"""Causal heatmap features: only completed KRX sessions enter a signal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant.timefolio_heatmap_data import ROOT, atomic_json

FEATURE_NAMES = [
    "open/last", "high/last", "low/last", "close/last", "bar return", "bar range", "opening gap", "intraday return",
    "slot relative volume", "slot relative value", "volume share", "volume surprise", "ADV5 headroom", "illiquidity proxy", "bar coverage", "ADV20 rank",
    "return 1d", "return 5d", "return 20d", "volatility 20d", "return 5d rank", "volatility rank", "sector return 5d", "market return 5d",
    "market cap", "small cap flag", "sector cap", "capacity burden", "upper limit distance", "lower limit distance", "session time", "zero volume",
]


def rolling(a, width, method="mean", min_periods=None):
    frame = pd.DataFrame(a.T)
    return getattr(frame.rolling(width, min_periods=min_periods or width), method)().to_numpy().T


def lag(a, n):
    out = np.full_like(a, np.nan, dtype=float)
    out[:, n:] = a[:, :-n]
    return out


def rank(a):
    return pd.DataFrame(a).rank(axis=0, pct=True).to_numpy(dtype=np.float32)


def restore_price_basis(ohlcv, scales):
    """Undo the vendor's retrospective adjustment using that day's exchange close.

    Only a per-session multiplicative basis conversion is performed. The KRX daily
    OHLC cross-check and reconstructed volume coverage are audited separately.
    """
    out = np.array(ohlcv, dtype=np.float32, copy=True)
    out[..., :4] *= np.asarray(scales)[:, None, None]
    out[..., 4] /= np.asarray(scales)[:, None]
    return out


def infer_actions(basis, regular, raw_close):
    """Avoid false actions from truncated sessions (e.g. the CSAT late close).

    An incomplete session may establish only a large simple split/consolidation,
    rounded to its integer ratio. It never establishes a small rights adjustment.
    Updating the reference basis on an inferred event prevents double counting.
    """
    actions = np.ones_like(basis)
    reference = np.full(basis.shape[0], np.nan)
    previous_close = np.full(basis.shape[0], np.nan)
    for d in range(basis.shape[1]):
        b, c = basis[:, d], raw_close[:, d]
        valid = np.isfinite(b) & (b > 0)
        r = reference / b
        with np.errstate(divide="ignore", invalid="ignore"):
            simple = np.where(r >= 1, np.maximum(1, np.rint(r)), 1 / np.maximum(1, np.rint(1 / r)))
        large = ((r > 1.8) | (r < .56)) & (np.abs(r / simple - 1) < .08)
        proposed = np.where(regular[:, d], r, np.where(large, simple, 1.))
        adjusted_move = proposed * c / previous_close
        accept = valid & (np.abs(np.log(proposed)) > .002) & (adjusted_move > .5) & (adjusted_move < 1.5)
        actions[accept, d] = proposed[accept]
        reference[accept & ~regular[:, d]] /= proposed[accept & ~regular[:, d]]
        reference[valid & regular[:, d]] = b[valid & regular[:, d]]
        known = np.isfinite(c) & (c > 0)
        previous_close[known] = c[known]
    return np.round(actions, 6)


def prepare(root=ROOT):
    root = Path(root)
    dates = json.loads((root / "calendar.json").read_text())
    raw = pd.read_parquet(root / "daily_minutes.parquet")
    meta = pd.read_parquet(root / "historical_krx.parquet")
    snapshot = json.loads((root / "timefolio_sectors.json").read_text())
    sectors = {x["code"]: int(x["sector"]) for x in snapshot["securities"]}
    # The collector's universe excludes most non-common instruments; Timefolio membership
    # provides a second check. Historical delistings still cause survivor bias.
    codes = sorted(set(raw.code) & set(sectors))
    n, nd = len(codes), len(dates)
    merged = meta.merge(raw, on=["code", "date"], how="left", validate="one_to_one")
    def pivot(col):
        return merged.pivot(index="code", columns="date", values=col).reindex(index=codes, columns=dates).to_numpy(dtype=float)
    p = {k: pivot(k) for k in ["o", "c", "v", "regular", "exec_price", "exec_volume", "exec_high", "exec_low", "exec_count",
                                    "market_cap", "traded_value", "krx_close", "listed_shares", "krx_volume"]}
    p["sector"] = np.array([sectors[c] for c in codes], dtype=int)
    basis = np.divide(p["krx_close"], p["c"], out=np.full((n, nd), np.nan), where=p["c"] > 0)
    p["price_basis"] = basis
    for key in ["o", "c", "exec_price", "exec_high", "exec_low"]:
        p[key] *= basis
    for key in ["v", "exec_volume"]:
        p[key] /= basis
    # Infer price-basis changes without inventing a price crash at a rights/split event.
    # This is not a full corporate-action ledger: event type/cash dividends remain unknown.
    p["split"] = infer_actions(basis, p["regular"] == 1, p["krx_close"])
    p["factor"] = np.cumprod(p["split"], axis=1)
    p["close"] = np.where(p["krx_close"] > 0, p["krx_close"], p["c"])
    adjusted = p["close"] * p["factor"]
    p["r1"] = adjusted / lag(adjusted, 1) - 1
    p["r5"] = adjusted / lag(adjusted, 5) - 1
    p["r20"] = adjusted / lag(adjusted, 20) - 1
    p["vol20"] = rolling(p["r1"], 20, "std")
    p["adv5"] = rolling(p["traded_value"], 5)
    p["adv20"] = rolling(p["traded_value"], 20)
    p["v20"] = rolling(p["v"], 20)
    # A signal at today's close may use today's completed KRX turnover.
    good = (p["regular"] == 1) & (p["close"] > 0)
    volume_coverage = p["v"] / np.maximum(p["krx_volume"], 1)
    basis_valid = np.isfinite(basis) & (basis > 0) & (volume_coverage >= .85) & (volume_coverage <= 1.02)
    recent_good = rolling((good & basis_valid).astype(float), 20) >= .95
    no_jump = rolling((np.abs(p["r1"]) > .35).astype(float), 20) == 0
    eligible = good & basis_valid & recent_good & no_jump & (p["market_cap"] >= 1e11) & (p["adv5"] > 3e9)
    eligible &= np.isfinite(p["vol20"]) & np.isfinite(p["r20"])
    liquid_rank = pd.DataFrame(np.where(eligible, p["adv20"], np.nan)).rank(axis=0, ascending=False, method="first").to_numpy()
    p["eligible"] = eligible & (liquid_rank <= 200)
    p["rank_r5"] = rank(np.where(eligible, p["r5"], np.nan))
    p["rank_vol"] = rank(np.where(eligible, p["vol20"], np.nan))
    p["rank_adv"] = rank(np.where(eligible, p["adv20"], np.nan))
    p["sector_r5"] = np.full((n, nd), np.nan)
    p["sector_cap"] = np.zeros((n, nd))
    # Denominator includes all Timefolio-mapped common shares in the historical KRX file,
    # not merely the 200 selected names. Today's GICS mapping is a static approximation.
    m = meta.copy()
    m["sector"] = m.code.map(sectors)
    cap = m.dropna(subset=["sector"]).pivot_table(index="sector", columns="date", values="market_cap", aggfunc="sum").reindex(columns=dates)
    weights = cap / cap.sum(axis=0)
    for sec in np.unique(p["sector"]):
        ix = p["sector"] == sec
        sector_ret = pd.DataFrame(np.where(eligible[ix], p["r5"][ix], np.nan)).mean(axis=0).to_numpy()
        p["sector_r5"][ix] = sector_ret
        p["sector_cap"][ix] = np.maximum(.1, 2 * weights.loc[sec].to_numpy())
    market = pd.DataFrame(np.where(eligible, p["r5"], np.nan)).mean(axis=0).to_numpy()
    p["market_r5"] = np.broadcast_to(market, (n, nd)).copy()
    # Memmapped storage keeps peak memory bounded and can be transferred without a DB.
    bars = np.lib.format.open_memmap(root / "bars15.npy", mode="w+", dtype="float32", shape=(n, nd, 26, 6))
    bars[:] = np.nan
    index = pd.MultiIndex.from_product([dates, range(26)], names=["date", "slot"])
    for i, code in enumerate(codes):
        d = pd.read_parquet(root / "bar_shards" / f"{code}.parquet")
        a = d.set_index(["date", "slot"]).reindex(index)[["o", "h", "l", "c", "v", "count"]].to_numpy(np.float32)
        bars[i] = restore_price_basis(a.reshape(nd, 26, 6), basis[i])
        bars[i, p["regular"][i] != 1] = np.nan
    bars.flush()
    # Preserve circuit-breaker/session holes as masked pixels rather than invent prices.
    # All configurations share the same maximum-window coverage criterion.
    coverage = np.isfinite(bars[..., 3]).mean(axis=2)
    p["eligible"] &= rolling(coverage, 10) >= .85
    np.savez_compressed(root / "panel.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in p.items()})
    atomic_json(root / "panel_index.json", {"dates": dates, "codes": codes, "features": FEATURE_NAMES})
    audit = {"start": dates[0], "end": dates[-1], "sessions": nd, "codes": n,
             "minute_security_days": len(raw), "eligible_signal_rows": int(p["eligible"].sum()),
             "median_daily_eligible": float(np.median(p["eligible"].sum(axis=0)[20:])),
             "inferred_corporate_actions": int((p["split"] != 1).sum()),
             "unexplained_daily_moves_gt35pct": int((np.abs(p["r1"]) > .35).sum()),
             "vendor_adjusted_security_days": int((good & (np.abs(basis - 1) > .002)).sum()),
             "failed_price_volume_basis_days": int((good & ~basis_valid).sum()),
             "price_basis": "KRX official close / archived adjusted close; prices multiply, share volumes divide. Intraday shape is preserved.",
             "gics_as_of": snapshot["as_of"], "gics_is_historical": False,
             "missing_rules": ["historical warning/caution/administrative designations", "historical GICS changes", "exact order-book depth and queue delays", "exact corporate-action types, cash dividends and rights listing dates"],
             "corporate_action_proxy": "Vendor basis changes infer entitlements; extra shares stay locked until KRX listed-share counts reach the inferred post-event count.",
             "survivorship": "Archive starts from a current/common-stock collector universe; delisted coverage is incomplete.",
             "promotion": "Research only; cannot certify exact historical contest eligibility."}
    atomic_json(root / "data_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


def load_panel(root=ROOT):
    root = Path(root)
    ix = json.loads((root / "panel_index.json").read_text())
    p = dict(np.load(root / "panel.npz"))
    p["eligible"] = p["eligible"].astype(bool)
    p["sector"] = p["sector"].astype(int)
    return p, ix, np.load(root / "bars15.npy", mmap_mode="r")


def make_image(p, bars, code_i, day_i, window=5, frequency=30):
    start = day_i - window + 1
    w = np.array(bars[code_i, start:day_i + 1], copy=True)
    factor = p["factor"][code_i, start:day_i + 1]
    w[..., :4] *= factor[:, None, None]
    w[..., 4] /= factor[:, None]
    if frequency == 30:
        pair = w.reshape(window, 13, 2, 6)
        w = np.stack([pair[:, :, 0, 0], pair[..., 1].max(2), pair[..., 2].min(2), pair[:, :, 1, 3],
                      pair[..., 4].sum(2), pair[..., 5].sum(2)], axis=-1)
    slots = w.shape[1]
    flat = w.reshape(-1, 6)
    close = flat[:, 3]
    out = np.zeros((32, len(flat)), dtype=np.float32)
    out[:4] = (np.log(flat[:, :4] / close[-1]) / .2).T
    previous_bar = np.r_[close[0], close[:-1]]
    out[4] = np.log(close / previous_bar) / .03
    out[5] = (flat[:, 1] / flat[:, 2] - 1) / .04
    prior = p["close"][code_i, start - 1:day_i] * p["factor"][code_i, start - 1:day_i]
    opening = w[:, 0, 0]
    out[6] = np.repeat((opening / prior - 1) / .1, slots)
    out[7] = (close / np.repeat(opening, slots) - 1) / .15
    # A historical slot baseline is frozen before the displayed window starts.
    history = np.array(bars[code_i, max(0, start - 20):start], copy=True)
    hf = p["factor"][code_i, max(0, start - 20):start]
    history[..., 4] /= hf[:, None]
    history[..., :4] *= hf[:, None, None]
    hv = np.nanmean(history[..., 4], axis=0)
    ha = np.nanmean(history[..., 4] * history[..., 3], axis=0)
    if frequency == 30:
        hv, ha = hv.reshape(13, 2).sum(1), ha.reshape(13, 2).sum(1)
    vol, amount = flat[:, 4], flat[:, 4] * flat[:, 3]
    out[8] = np.log((vol + 1) / (np.tile(hv, window) + 1)) / 3
    out[9] = np.log((amount + 1) / (np.tile(ha, window) + 1)) / 3
    out[10] = vol / np.maximum(np.repeat(np.nansum(w[..., 4], axis=1), slots), 1) * slots / 3
    ci, di = code_i, day_i
    out[11] = np.log((p["v"][ci, di] + 1) / (p["v20"][ci, di] + 1)) / 2
    out[12] = np.log10(max(p["adv5"][ci, di], 1) / 3e9) / 3
    out[13] = np.abs(out[4]) / np.maximum(amount / 1e8, 1)
    out[14] = flat[:, 5] / frequency
    out[15] = p["rank_adv"][ci, di] * 2 - 1
    for row, key, scale in [(16, "r1", .1), (17, "r5", .2), (18, "r20", .4), (19, "vol20", .05),
                             (22, "sector_r5", .2), (23, "market_r5", .2)]:
        out[row] = p[key][ci, di] / scale
    out[20] = p["rank_r5"][ci, di] * 2 - 1
    out[21] = p["rank_vol"][ci, di] * 2 - 1
    out[24] = np.log10(max(p["market_cap"][ci, di], 1) / 1e11) / 4
    out[25] = 1 if p["market_cap"][ci, di] < 1e12 else -1
    out[26] = p["sector_cap"][ci, di]
    out[27] = 8e7 / max(p["adv5"][ci, di] * .05, 1)
    relative = close / np.repeat(prior, slots) - 1
    out[28], out[29] = (.3 - relative) / .6, (.3 + relative) / .6
    out[30] = np.tile(np.linspace(-1, 1, slots), window)
    out[31] = np.where(vol > 0, -1, 1)
    return np.rint((np.clip(np.nan_to_num(out), -1, 1) + 1) * 127.5).astype(np.uint8)


def labels(p, ci, di, horizon, target="relative"):
    n = p["close"].shape[1]
    fwd = np.full(len(ci), np.nan)
    valid = di + horizon < n
    c, d = ci[valid], di[valid]
    entry = p["exec_price"][c, d + 1] * p["factor"][c, d + 1]
    final = p["close"][c, d + horizon] * p["factor"][c, d + horizon]
    fwd[valid] = final / entry - 1
    if target == "relative":
        median = pd.Series(fwd).groupby(di).transform("median").to_numpy()
        y = (fwd - median > .004).astype(np.float32)
    else:
        y = (fwd > .004).astype(np.float32)
    y[~np.isfinite(fwd)] = np.nan
    return y, fwd


def build_images(root=ROOT):
    root = Path(root)
    p, ix, bars = load_panel(root)
    ci, di = np.where(p["eligible"])
    order = np.lexsort((ci, di)); ci, di = ci[order], di[order]
    np.savez_compressed(root / "samples.npz", ci=ci, di=di)
    for frequency, window in [(30, 3), (30, 5), (30, 10), (15, 3), (15, 5), (15, 10)]:
        path = root / f"images_f{frequency}_w{window}.npy"
        if path.exists():
            continue
        a = np.lib.format.open_memmap(path, mode="w+", dtype="uint8", shape=(len(ci), 32, window * 390 // frequency))
        for j, (c, d) in enumerate(zip(ci, di)):
            a[j] = make_image(p, bars, c, d, window, frequency)
        a.flush()
        print(json.dumps({"images": path.name, "samples": len(ci), "shape": list(a.shape)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["panel", "images", "all"])
    ap.add_argument("--root", type=Path, default=ROOT); a = ap.parse_args()
    if a.command in ("panel", "all"): prepare(a.root)
    if a.command in ("images", "all"): build_images(a.root)

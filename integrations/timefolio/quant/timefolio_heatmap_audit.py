"""Independently reconstruct saved fills to audit cash, NAV and weight limits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quant.timefolio_heatmap_data import ROOT, atomic_json


def audit_fills(p, index, result, *, slip=.0005, gross=.8, sector_floor=False):
    codes, dates = index["codes"], index["dates"]
    code_index = {code: i for i, code in enumerate(codes)}
    date_index = {date: i for i, date in enumerate(dates)}
    qty = np.zeros(len(codes), dtype=np.int64)
    marks = np.zeros(len(codes)); cash = 1e9
    sector = p["sector"]
    members = {s: np.flatnonzero(sector == s) for s in np.unique(sector)}
    stock_cap = np.array([.4 if c == "005930" else .3 if c == "000660" else .15 for c in codes])
    events = {}
    for trade in result["trades"]:
        events.setdefault(trade["date"], []).append(trade)
    violations, stock_breaches = [], []
    nav_error = 0.; minimum_cash = 1e9; maximum_stock_weight = 0.; buy_checks = 0
    for daily in result["daily"]:
        date = daily["date"]; d = date_index[date]
        ratio = p["split"][:, d]
        new_qty = qty * ratio
        cash += float(np.sum(np.where(qty > 0, (new_qty - np.floor(new_qty)) * np.nan_to_num(p["close"][:, d]), 0)))
        qty = np.floor(new_qty + 1e-7).astype(np.int64)
        marks = np.divide(marks, ratio, out=marks.copy(), where=ratio > 0)
        opening = p["exec_price"][:, d].astype(float)
        good = np.isfinite(opening) & (opening > 0) & (p["exec_count"][:, d] >= 25)
        marks[good] = opening[good]
        previous = p["close"][:, d - 1] / ratio
        cap_now = p["market_cap"][:, d - 1] * np.divide(marks, previous, out=np.ones(len(codes)), where=previous > 0)
        small = cap_now < 1e12
        caps = np.full(len(codes), .1) if sector_floor else p["sector_cap"][:, d - 1]
        for trade in events.get(date, []):
            i = code_index[trade["code"]]; q = trade["qty"]
            buying = trade["side"] == "buy"
            qty[i] += q if buying else -q
            cash += (-q * trade["price"] if buying else q * trade["price"]) - trade["fee"]
            minimum_cash = min(minimum_cash, cash)
            if not buying:
                continue
            buy_checks += 1
            values = qty * marks; nav = cash + values.sum()
            failed = []
            if cash < -.01 or np.any(qty < 0): failed.append("cash_or_short")
            if "trade_allowed" in p and not p["trade_allowed"][i, d]: failed.append("designation")
            if np.any(values > stock_cap * nav + 1): failed.append("individual")
            if values[small].sum() > .3 * nav + 1: failed.append("small_cap")
            if values.sum() > gross * nav + 1: failed.append("gross")
            if any(values[m].sum() > caps[m[0]] * nav + 1 for m in members.values()): failed.append("sector")
            if failed: violations.append({"date": date, "code": trade["code"], "limits": failed})
        close = p["close"][:, d]
        known = np.isfinite(close) & (close > 0)
        marks[known] = close[known]
        values = qty * marks; nav = cash + values.sum()
        nav_error = max(nav_error, abs(nav - daily["nav"]))
        maximum_stock_weight = max(maximum_stock_weight, float(np.max(values / nav)))
        breached = np.flatnonzero(values > stock_cap * nav + 1)
        stock_breaches += [{"date": date, "code": codes[i], "weight": float(values[i] / nav), "limit": float(stock_cap[i])} for i in breached]
    return {"buy_checks": buy_checks, "post_buy_limit_violations": violations,
            "closing_individual_breaches": stock_breaches, "maximum_stock_weight": maximum_stock_weight,
            "minimum_cash_krw": minimum_cash, "maximum_nav_reconstruction_error_krw": nav_error,
            "scope": "All positions after each executed buy, same historical metadata and execution marks; missing designations/order book remain unaudited."}


def main(root):
    from quant.timefolio_heatmap_study import context, score_matrix
    from quant.timefolio_heatmap_replay import replay
    root = Path(root); p, ix, ci, di = context(root)
    checks = {}
    for path in sorted((root / "screen").glob("*.pred.npy")):
        config = json.loads(path.with_name(path.name.removesuffix(".pred.npy") + ".json").read_text())["config"]
        scores = score_matrix(np.load(path), ci, di, p["close"].shape)
        result = replay(p, ix, scores, "20260401", "20260630", rebalance=config["rebalance"], return_trades=True)
        checks[config["id"]] = audit_fills(p, ix, result)
    holdout = root / "holdout.json"
    if holdout.exists():
        result = json.loads(holdout.read_text())
        if "paths" in result:
            checks["holdout"] = {k: audit_fills(p, ix, v) for k, v in result["paths"].items()}
    atomic_json(root / "independent_weight_audit.json", checks)
    total = [v for k, v in checks.items() if k != "holdout"] + list(checks.get("holdout", {}).values())
    print(json.dumps({"portfolios": len(total), "buy_checks": sum(v["buy_checks"] for v in total),
                      "violating_buys": sum(len(v["post_buy_limit_violations"]) for v in total),
                      "individual_breach_days": sum(len(v["closing_individual_breaches"]) for v in total),
                      "maximum_nav_error_krw": max(v["maximum_nav_reconstruction_error_krw"] for v in total)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=ROOT)
    main(parser.parse_args().root)

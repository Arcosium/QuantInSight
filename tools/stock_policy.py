"""Systematic research desk for the institutional trading committee.

No CNN dependency. A fixed relative-strength/low-volatility hypothesis supplies
the candidate list and target weights. Committee vetoes and existing exits stay
in force. This pilot never enables a real brokerage profile.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import pandas as pd
from arcmarket.systematic import VERSION, features, select

DATA = Path(__file__).resolve().parents[1]/"data"
SHARED = Path(os.getenv("QIS_SHARED_KRX_DAILY", Path.home()/"vault"/"CryptoBars"/"data"/"KRX"/"daily_cnn"))
STOCK_BUDGET_CAP = .40
_lock = threading.Lock()


def _number(value):
    try:
        n = float(value or 0)
        return n if math.isfinite(n) else 0.
    except (ValueError, TypeError):
        return 0.


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        temp = f.name
    os.replace(temp, path)


def histories(market, today):
    if market == "KRX":
        with (SHARED.parent/"universe.csv").open() as f:
            codes = {r["code"] for r in csv.DictReader(f)}
        out = {p.stem: pd.read_parquet(p) for p in SHARED.glob("*.parquet") if p.stem in codes}
    else:
        folder = SHARED.parents[1]/"USA"/"daily_policy"
        status_path = folder/"status.json"
        info = json.loads(status_path.read_text()) if status_path.exists() else {}
        codes = set(info.get("codes", []))
        if info.get("ok", 0) < .9*max(1, info.get("requested", 0)):
            return {}
        out = {p.stem: pd.read_parquet(p) for p in folder.glob("*.parquet") if p.stem in codes}
    return {c: d[d.index < today] for c, d in out.items() if len(d[d.index < today]) >= 61}


def plan(uid, market="KRX", *, now=None):
    """Persist the same targets for five completed sessions, with daily checks."""
    today = pd.Timestamp(now or pd.Timestamp.now(tz="Asia/Seoul").date()).tz_localize(None).normalize()
    path = DATA/str(int(uid))/"stock_policy"/f"stock_policy_{market}.json"
    with _lock:
        data = histories(market, today)
        frame = features(data)
        base = {"version": VERSION, "market": market, "mode": "systematic", "weights": {}, "rows": [],
                "status": "insufficient_data", "reason": "자료 부족", "max_stock_budget": STOCK_BUDGET_CAP,
                "validated": False, "roles": {"research": "수치로 후보·비중 제안", "committee": "공시·뉴스 위험 심의",
                                             "risk": "기존 보유 포함 예산 제한", "execution": "선택 계정 주문·체결 추적"}}
        if frame.empty:
            return base
        dates = sorted(pd.Timestamp(x) for x in frame.date.unique())
        asof = dates[-1]; day = frame[frame.date == asof]
        # US caches do not represent a full point-in-time universe: show coverage.
        coverage = len(day)/max(1, len(data))
        base.update(as_of=str(asof.date()), coverage=coverage, histories=len(data))
        if coverage < .90 or (today-asof).days > 4:
            base.update(status="stale_or_incomplete", reason="자료 갱신·범위 확인 필요")
            return base
        old = json.loads(path.read_text()) if path.exists() else {}
        age = sum(t > pd.Timestamp(old.get("as_of", "1900-01-01")) for t in dates)
        if old.get("version") == VERSION and age < 5:
            result = dict(old, mode=base["mode"], roles=base["roles"],
                          checked_as_of=str(asof.date()), coverage=coverage)
        else:
            raw = select(day, min_adv=1e9 if market=="KRX" else 1e7)
            if raw["reason"] == "insufficient_universe":
                return base
            scale = STOCK_BUDGET_CAP/.8
            result = {**base, **raw, "weights": {c: w*scale for c, w in raw["weights"].items()},
                      "exposure": raw["exposure"]*scale, "status": "ready", "as_of": str(asof.date()),
                      "created_at": pd.Timestamp.now(tz="UTC").isoformat(), "reason": "5거래일 고정 후보",
                      "round_trip_cost_assumption_pct": .4 if market=="KRX" else .7}
            for r in result["rows"]:
                r["weight"] *= scale
            result["decision_id"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:16]
            # Immutable decision history allows later comparisons without rewriting.
            _write(path.parent/f"stock_policy_{market}_{result['decision_id']}.json", result)
            _write(path, result)
        # Reject fresh purchases after trend failure; keep original targets auditable.
        healthy = set(day.loc[day.valid & day.trend & (day.r5 < .2), "code"])
        result["entry_weights"] = {c: w for c, w in result["weights"].items() if c in healthy}
        result["trend_exit_codes"] = sorted(set(day.loc[day.valid & ~day.trend, "code"]))
        return result


def candidates(policy, holdings, total_eval, *, usdkrw=0):
    weights = policy.get("entry_weights", policy.get("weights", {}))
    held = {str(h.get("code")): h for h in holdings}
    ranked = []
    for row in policy.get("rows", []):
        c = row["code"]
        if c not in weights:
            continue
        h = held.get(c, {})
        # The broker assembler deliberately does not pyramid held stocks.
        if float(h.get("qty") or 0) > 0:
            continue
        fx = 1 if c.isdigit() else usdkrw
        value = float(h.get("qty") or 0)*float(h.get("cur_price") or 0)*fx
        if total_eval > 0 and value < total_eval*weights[c]*.85:
            ranked.append(c)
    return ranked[:5]


def cap_entries(orders, policy, *, holdings, prices, total_eval, usdkrw=0, is_sleeve=lambda c: False):
    """Plan + current holdings cap entries. Exits and sleeve routing are unchanged.

    Repeated cycles cannot buy the same target repeatedly. >3% price chasing is
    rejected; the frozen signal can be reassessed at the next scheduled review.
    """
    out, notes, exposure = [], [], {}
    total_eval = _number(total_eval)
    def fx(c):
        return 1. if c.isdigit() else _number(usdkrw)
    for h in holdings:
        c = str(h.get("code", ""))
        qty = _number(h.get("qty")); price = _number(h.get("cur_price"))
        if qty > 0 and (price <= 0 or fx(c) <= 0):
            total_eval = 0
        exposure[c] = exposure.get(c, 0)+qty*price*fx(c)
    used = sum(v for c, v in exposure.items() if not is_sleeve(c))
    weights = policy.get("entry_weights", policy.get("weights", {})) if policy.get("status") == "ready" else {}
    references = {r["code"]: r["close"] for r in policy.get("rows", [])}
    for order in orders:
        c = str(order.get("ticker", ""))
        if order.get("side", "buy").lower() == "sell" or is_sleeve(c):
            out.append(dict(order)); continue
        px = _number(prices.get(c))
        ref = _number(references.get(c))
        cap = min(max(0., total_eval*weights.get(c, 0)-exposure.get(c, 0)),
                  max(0., total_eval*STOCK_BUDGET_CAP-used))
        limit = _number(order.get("entry_limit")) if order.get("entry_mode") == "limit" else 0
        price = max(px, limit)*fx(c)
        requested = int(_number(order.get("qty")))
        qty = max(0, min(requested, int(cap/(price*1.007)) if price > 0 else 0))
        if not all(math.isfinite(x) for x in (px, ref, total_eval, price)) or ref <= 0 or px > ref*1.03:
            qty = 0
        if qty:
            out.append(dict(order, qty=qty, strategy_id=VERSION, decision_id=policy.get("decision_id")))
            exposure[c] = exposure.get(c, 0)+qty*price; used += qty*price
        if qty != requested:
            notes.append(f"{c}: 정량 운용안·기존 비중·추격 제한으로 {order.get('qty', 0)}주 → {qty}주")
    return out, notes


def exit_proposals(policy, holdings, *, total_eval, usdkrw=0, is_sleeve=lambda c: False):
    """Aftercare receives explicit rotation/trim proposals from the frozen plan."""
    if policy.get("status") != "ready" or total_eval <= 0:
        return {}
    targets = {c:w for c,w in policy.get("weights", {}).items() if c not in policy.get("trend_exit_codes", [])}
    result = {}
    for h in holdings:
        code = str(h.get("code", ""))
        if is_sleeve(code) or ((policy.get("market") == "KRX") != code.isdigit()):
            continue
        qty = int(h.get("qty") or 0)
        price = float(h.get("cur_price") or 0)*(1 if code.isdigit() else usdkrw)
        if not qty or price <= 0:
            continue
        target = total_eval*targets.get(code, 0)
        if not target:
            result[code] = "전량"
        elif qty*price > target*1.25:
            sell = max(0, qty-int(target/price))
            if sell:
                result[code] = str(sell)
    return result


def status(uid):
    folder = DATA/str(int(uid))/"stock_policy"
    plans = []
    for market in ("KRX", "USA"):
        path = folder/f"stock_policy_{market}.json"
        if path.exists():
            plans.append(json.loads(path.read_text()))
    path = Path.home()/"vault"/"QuantInSight"/"research"/"stock_policy_baseline.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    return {"plans": plans, "baseline": {"scope": report.get("scope"), "start": report.get("start"),
             "end": report.get("end"), "metrics": {k:v["metrics"] for k,v in report.get("paths", {}).items()}},
            "mode": "systematic", "validated": False}

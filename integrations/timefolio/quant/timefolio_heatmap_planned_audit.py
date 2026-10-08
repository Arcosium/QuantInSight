"""Independent audits for the separate planning-time-corrected replay.

The frozen v3 auditors are retained separately. This variant reconstructs the
explicitly declared zero fractional-entitlement cash policy, without changing
order prices, post-buy exposure checks, or the source market data.
"""
from __future__ import annotations
from collections import Counter
import numpy as np


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
        # Match the declared conservative ledger: no fractional entitlement cash.
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



def additional_checks(p, ix, result, schedule, *, participation=.05, slip=.0005, max_orders=20):
    codes = {code:i for i,code in enumerate(ix['codes'])}; dates = {date:i for i,date in enumerate(ix['dates'])}
    grouped = {}; errors = []; qty = np.zeros(len(codes)); cash = 1e9; marks = np.zeros(len(codes))
    for t in result['trades']: grouped.setdefault(t['date'],[]).append(t)
    max_count = 0; sector_exposure = {int(s):[] for s in np.unique(p['sector'])}
    for row in result['daily']:
        d = dates[row['date']]; ratio = p['split'][:,d]; new = qty*ratio
        # Fractional entitlements are omitted from spendable cash and NAV.
        qty = np.floor(new+1e-7); marks = np.divide(marks,ratio,out=marks.copy(),where=ratio>0)
        price = p['exec_price'][:,d].astype(float); available = np.isfinite(price)&(price>0)&(p['exec_count'][:,d]>=25)
        marks[available] = price[available]
        trades = grouped.get(row['date'],[]); max_count=max(max_count,len(trades))
        if len(trades)>max_orders: errors.append({'date':row['date'],'kind':'daily_order_budget'})
        volume = Counter()
        for t in trades:
            i=codes[t['code']]; buy=t['side']=='buy'; q=t['qty']; volume[i]+=q
            if t['signal_date'] != ix['dates'][d-1] or not (q>0 and int(q)==q):
                errors.append({'date':row['date'],'kind':'signal_or_quantity'})
            expected=price[i]*(1+slip if buy else 1-slip)
            if not np.isclose(t['price'],expected,rtol=1e-10,atol=1e-8): errors.append({'date':row['date'],'kind':'fill_price'})
            fee=q*t['price']*(.001 if buy else .003)
            if not np.isclose(t['fee'],fee,rtol=1e-10,atol=1e-6): errors.append({'date':row['date'],'kind':'fee'})
            qty[i]+=q if buy else -q; cash+=(-q*t['price'] if buy else q*t['price'])-t['fee']
            if cash<-.01 or np.any(qty<0): errors.append({'date':row['date'],'kind':'cash_or_short'})
            value=qty@marks; nav=cash+value
            ceiling=.8 if schedule is None else schedule[d-1]
            if buy and value>ceiling*nav+1: errors.append({'date':row['date'],'kind':'dynamic_gross'})
        for i,q in volume.items():
            if q>np.floor(p['exec_volume'][i,d]*participation): errors.append({'date':row['date'],'kind':'participation'})
        close=p['close'][:,d]; good=np.isfinite(close)&(close>0); marks[good]=close[good]
        values=qty*marks; nav=cash+values.sum()
        for sec in sector_exposure: sector_exposure[sec].append(float(values[p['sector']==sec].sum()/nav))
    return {'additional_errors':errors,'observed_max_daily_orders':max_count,
            'mean_account_sector_weights':{str(s):float(np.mean(v)) for s,v in sector_exposure.items()}}


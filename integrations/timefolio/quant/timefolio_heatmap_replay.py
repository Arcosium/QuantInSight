"""Long-only, cash-account replay with explicit Timefolio rule approximations.

No account/API integration. Optional point-in-time designations block purchases;
order queues and corporate-action details remain approximations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INITIAL_NAV = 1_000_000_000.
BUY_FEE, SELL_FEE = .001, .003


def targets(scores, eligible, sector, caps, market_cap, codes, *, top_n=12, weight=.08, gross=.8):
    """Greedy diversified target; the caller supplies only signal-time metadata."""
    out = np.zeros(len(scores))
    sectors, small, total, held = {}, 0., 0., 0
    # Stable ticker tie-breaking makes seeds/baselines reproducible.
    order = np.lexsort((np.asarray(codes), -np.nan_to_num(scores, nan=-np.inf)))
    for i in order:
        if not eligible[i] or not np.isfinite(scores[i]) or held >= top_n:
            continue
        statutory = .4 if codes[i] == "005930" else .3 if codes[i] == "000660" else .15
        sec = int(sector[i])
        remaining = .95 * caps[i] - sectors.get(sec, 0.)
        w = min(weight, statutory, gross - total, remaining)
        is_small = market_cap[i] < 1e12
        if is_small:
            w = min(w, .285 - small)
        if w <= .005:
            continue
        out[i] = w
        total += w; held += 1
        sectors[sec] = sectors.get(sec, 0.) + w
        small += w if is_small else 0.
    return out


def perf(nav, initial=INITIAL_NAV):
    a = np.asarray(nav, dtype=float)
    r = a / np.r_[initial, a[:-1]] - 1
    peak = np.maximum.accumulate(np.r_[initial, a])[1:]
    return {"return": float(a[-1] / initial - 1) if len(a) else 0.,
            "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 and r.std(ddof=1) > 0 else 0.,
            "mdd": float(np.min(a / peak - 1)) if len(a) else 0.,
            "daily_mean": float(r.mean()) if len(a) else 0.}


def replay(p, index, scores, start, end, *, rebalance=5, slip=.0005, participation=.05,
           sector_floor=False, top_n=12, weight=.08, gross=.8, return_trades=False,
           gross_schedule=None, max_orders=None):
    """Signals at t close, integer fills at t+1 09:05--09:34 proxy VWAP.

    Volume participation limits fills, never candidate selection. Fees reduce cash;
    buy headroom includes fee-induced NAV shrinkage. Unsold shares carry forward.
    Weekly turnover is reported; a strategy with four violations fails selection.
    """
    dates, codes = index["dates"], index["codes"]
    days = [i for i, d in enumerate(dates) if start <= d <= end and i > 0]
    n = len(codes); qty = np.zeros(n, dtype=np.int64); cash = INITIAL_NAV
    locked_qty = np.zeros(n, dtype=np.int64)
    release_shares = np.full(n, np.inf)
    marks = np.zeros(n); rows, trades = [], []
    commission = slip_paid = buy_total = sell_total = 0.
    partial = blocked = price_breach = action_events = 0
    deferred_orders = 0; pending = False
    if gross_schedule is not None:
        gross_schedule = np.asarray(gross_schedule, dtype=float)
        if gross_schedule.shape != (len(dates),) or not np.all(np.isfinite(gross_schedule)):
            raise ValueError("gross_schedule must contain one finite value per signal date")
        if np.any((gross_schedule < 0) | (gross_schedule > gross)):
            raise ValueError("gross_schedule must stay between zero and the gross ceiling")
    if max_orders is not None and (not isinstance(max_orders, int) or max_orders < 1):
        raise ValueError("max_orders must be a positive integer")
    sectors = p["sector"]
    stock_caps = np.array([.4 if c == "005930" else .3 if c == "000660" else .15 for c in codes])
    sector_members = {s: np.where(sectors == s)[0] for s in np.unique(sectors)}
    for k, d in enumerate(days):
        target_gross = gross if gross_schedule is None else float(gross_schedule[d - 1])
        filled_orders = 0
        old_values = qty * marks
        old_nav = cash + old_values.sum()
        known_caps = np.full(n, .1) if sector_floor else p["sector_cap"][:, d - 1]
        repair = any(old_values[m].sum() > known_caps[m[0]] * old_nav + 1 for m in sector_members.values())
        repair |= old_values[p["market_cap"][:, d - 1] < 1e12].sum() > .3 * old_nav + 1
        repair |= np.any(old_values > stock_caps * old_nav + 1)
        if gross_schedule is not None:
            # Two percentage points of price drift avoid daily mechanical churn.
            repair |= old_values.sum() > (target_gross + .02) * old_nav + 1
        # Reconstruct only split-like corporate actions supported by both shares and prices.
        ratio = p["split"][:, d]
        had = qty > 0
        action_events += int((had & (np.abs(ratio - 1) > .002)).sum())
        previous_qty = qty.copy()
        newq = qty * ratio
        frac = newq - np.floor(newq)
        cash += float(np.sum(np.where(had, frac * np.nan_to_num(p["close"][:, d]), 0)))
        qty = np.floor(newq + 1e-7).astype(np.int64)
        expanded = ratio > 1.002
        locked_qty = np.floor(locked_qty * np.minimum(ratio, 1)).astype(np.int64)
        locked_qty[expanded] += qty[expanded] - previous_qty[expanded]
        release_shares[expanded] = p["listed_shares"][expanded, d - 1] * ratio[expanded]
        released = p["listed_shares"][:, d] >= release_shares * .999
        locked_qty[released] = 0; release_shares[released] = np.inf
        marks = np.divide(marks, ratio, out=marks.copy(), where=ratio > 0)
        opening = p["exec_price"][:, d].astype(float)
        good = np.isfinite(opening) & (opening > 0) & (p["exec_count"][:, d] >= 25)
        marks[good] = opening[good]
        # Only held positions use stale marks when no print is available.
        nav = cash + float(qty @ marks)
        buys = sells = 0.
        eligible = p["eligible"][:, d - 1].copy()
        if "trade_allowed" in p:
            # Broker-side admission is checked on the execution date, including
            # designations effective overnight. It never uses the next day's prices.
            eligible &= p["trade_allowed"][:, d].astype(bool)
        signal_available = np.any(np.isfinite(scores[:, d - 1]) & p["eligible"][:, d - 1])
        liquidate = gross_schedule is not None and target_gross == 0
        if (k % rebalance == 0 or repair or pending) and (signal_available or liquidate):
            pending = False
            cap = np.full(n, .1) if sector_floor else p["sector_cap"][:, d - 1].astype(float)
            old_cap = p["market_cap"][:, d - 1].astype(float)
            w = targets(scores[:, d - 1], eligible, sectors, cap,
                        old_cap, codes, top_n=top_n, weight=weight, gross=target_gross)
            desired = np.floor(np.divide(w * nav, opening, out=np.zeros(n), where=good)).astype(np.int64)
            desired[~good] = qty[~good]  # Orders cannot erase halted holdings.
            previous = p["close"][:, d - 1] / ratio
            locked = np.isclose(p["exec_high"][:, d], p["exec_low"][:, d], rtol=0, atol=.001)
            up = locked & (opening / previous >= 1.29)
            down = locked & (opening / previous <= .71)
            capacity = np.floor(np.nan_to_num(p["exec_volume"][:, d]) * participation).astype(np.int64)
            for side in ("sell", "buy"):
                requested = np.maximum(qty - desired, 0) if side == "sell" else np.maximum(desired - qty, 0)
                if side == "sell": requested = np.minimum(requested, qty - locked_qty)
                candidates = np.where(requested > 0)[0]
                if side == "buy":
                    candidates = candidates[np.argsort(-scores[candidates, d - 1], kind="stable")]
                elif max_orders is not None:
                    candidates = candidates[np.argsort(-(requested[candidates] * marks[candidates]), kind="stable")]
                for i in candidates:
                    if max_orders is not None and filled_orders >= max_orders:
                        deferred_orders += 1; pending = True
                        continue
                    if not good[i] or (up[i] if side == "buy" else down[i]):
                        blocked += 1
                        continue
                    q = min(int(requested[i]), int(capacity[i]))
                    if q < requested[i]: partial += 1
                    if q <= 0: continue
                    price = opening[i] * (1 + slip if side == "buy" else 1 - slip)
                    fee = BUY_FEE if side == "buy" else SELL_FEE
                    if side == "buy":
                        nav = cash + float(qty @ marks)
                        pos_value = qty * marks
                        sec_ix = sector_members[sectors[i]]
                        # Approximate current capitalisation using previous close shares.
                        cap_now = old_cap * np.divide(marks, previous, out=np.ones(n), where=previous > 0)
                        if not np.isfinite(cap_now[i]) or cap_now[i] < 1e11:
                            blocked += 1
                            continue
                        small = cap_now < 1e12
                        statutory = .4 if codes[i] == "005930" else .3 if codes[i] == "000660" else .15
                        # Fee/impact reduce NAV by loss per share relative to its mark.
                        loss = price * (1 + fee) - marks[i]
                        limits = [q, cash / (price * (1 + fee)),
                                  (statutory * nav - pos_value[i]) / (marks[i] + statutory * loss),
                                  (cap[i] * nav - pos_value[sec_ix].sum()) / (marks[i] + cap[i] * loss),
                                  (target_gross * nav - pos_value.sum()) / (marks[i] + target_gross * loss)]
                        if small[i]:
                            limits.append((.3 * nav - pos_value[small].sum()) / (marks[i] + .3 * loss))
                        elif loss > 0:
                            limits.append((.3 * nav - pos_value[small].sum()) / (.3 * loss))
                        if loss > 0:
                            others = np.arange(n) != i
                            limits.append(float(np.min((stock_caps[others] * nav - pos_value[others]) /
                                                       (stock_caps[others] * loss))) if others.any() else np.inf)
                            for sec, members in sector_members.items():
                                if sec != sectors[i]:
                                    other_cap = cap[members[0]]
                                    limits.append((other_cap * nav - pos_value[members].sum()) / (other_cap * loss))
                        # Do not add to pre-existing sector/small-cap breaches elsewhere.
                        if pos_value[small].sum() > .3 * nav + 1:
                            limits.append(0)
                        q = max(0, int(np.floor(min(limits))))
                    if q <= 0: continue
                    notional = q * price
                    filled_orders += 1
                    if side == "buy":
                        cash -= notional * (1 + fee); qty[i] += q; buys += notional
                    else:
                        cash += notional * (1 - fee); qty[i] -= q; sells += notional
                    commission += notional * fee
                    slip_paid += q * opening[i] * slip
                    if return_trades:
                        trades.append({"date": dates[d], "signal_date": dates[d - 1], "code": codes[i],
                                       "side": side, "qty": q, "price": float(price), "fee": float(notional * fee)})
        closing = p["close"][:, d]
        valid_close = np.isfinite(closing) & (closing > 0)
        marks[valid_close] = closing[valid_close]
        values = qty * marks
        nav = cash + values.sum()
        if cash < -.01 or np.any(qty < 0):
            raise AssertionError("Replay violated long-only/cash accounting")
        caps = np.full(n, .1) if sector_floor else p["sector_cap"][:, d - 1]
        breached = any(values[m].sum() > caps[m[0]] * nav + 1 for m in sector_members.values())
        breached |= values[p["market_cap"][:, d] < 1e12].sum() > .3 * nav + 1
        breached |= np.any(values > stock_caps * nav + 1)
        price_breach += int(breached)
        buy_total += buys; sell_total += sells
        rows.append({"date": dates[d], "nav": float(nav), "cash": float(cash), "gross": float(values.sum() / nav),
                     "holdings": int((qty > 0).sum()), "buy_value": float(buys), "sell_value": float(sells)})
    frame = pd.DataFrame(rows)
    week = pd.to_datetime(frame.date).dt.to_period("W-SUN")
    weekly = frame.groupby(week).agg(nav=("nav", "mean"), buys=("buy_value", "sum"), sells=("sell_value", "sum"), days=("date", "size"))
    weekly["turnover"] = .5 * (weekly.buys + weekly.sells) / weekly.nav
    # First/last truncated weeks are reported separately; all other weeks count,
    # including exchange holiday weeks shorter than five sessions.
    full = weekly.iloc[1:-1] if len(weekly) > 2 else weekly.iloc[:0]
    metrics = {**perf(frame.nav), "mean_gross": float(frame.gross.mean()), "mean_holdings": float(frame.holdings.mean()),
               "fees_krw": float(commission), "slippage_krw": float(slip_paid),
               "buy_value": float(buy_total), "sell_value": float(sell_total),
               "weekly_turnover_mean": float(weekly.turnover.mean()),
               "low_turnover_weeks": int((full.turnover < .05).sum()), "assessed_full_weeks": len(full),
               "four_week_turnover_stop": bool((full.turnover < .05).sum() >= 4),
               "partial_fills": partial, "blocked_fills": blocked, "closing_weight_breach_days": price_breach,
               "inferred_actions_while_held": action_events, "unreleased_rights_positions": int((locked_qty > 0).sum()),
               "terminal_liquidation_reserve": float(np.sum(qty * marks) * (SELL_FEE + slip)),
               "rule_certified": False}
    if max_orders is not None:
        metrics["order_budget_deferred_requests"] = deferred_orders
        metrics["max_daily_orders"] = max_orders
    if gross_schedule is not None:
        metrics["mean_target_gross"] = float(np.mean(gross_schedule[np.asarray(days) - 1]))
    # Mark-to-market return is the contest measure. Liquidation reserve is disclosed.
    return {"metrics": metrics, "daily": rows,
            "weekly": [{"week": str(i), "turnover": float(r.turnover), "days": int(r.days)} for i, r in weekly.iterrows()],
            "trades": trades}

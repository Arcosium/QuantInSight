"""Causal weekly activity accounting and small rank-improving order proposals.

These helpers never certify compliance or execute orders. They use a known
session calendar, past actual fills, and current permitted planning prices.
Future closing NAV and opening-window liquidity are deliberately absent.
"""
from datetime import datetime

import numpy as np


class WeeklyActivity:
    """Track the same intersecting two-month contest weeks as the account audit."""

    def __init__(self, dates, *, target_turnover=.055, intervene_after_low_weeks=2,
                 last_sessions=2):
        self.dates = tuple(dates)
        if (not self.dates or list(self.dates) != sorted(set(self.dates))
                or not np.isfinite(target_turnover) or not .05 <= target_turnover <= .10
                or type(intervene_after_low_weeks) is not int or not 0 <= intervene_after_low_weeks <= 3
                or type(last_sessions) is not int or not 1 <= last_sessions <= 5):
            raise ValueError('Ordered session dates and bounded weekly policy required')
        self.target = float(target_turnover)
        self.intervene_after = intervene_after_low_weeks
        self.last_sessions = last_sessions
        self.position = 0
        self.keys = {}
        self.groups = {}
        self.observed = {}
        self.closed = []
        for date in self.dates:
            parsed = datetime.strptime(date, '%Y%m%d')
            if parsed.strftime('%Y%m%d') != date:
                raise ValueError('Canonical YYYYMMDD dates required')
            if parsed.month % 3 == 0:
                self.keys[date] = None
                continue
            window = (parsed.year, (parsed.month - 1) // 3 * 3 + 1)
            key = (window, parsed.isocalendar()[:2])
            self.keys[date] = key
            self.groups.setdefault(key, []).append(date)

    def _check_date(self, date):
        if self.position >= len(self.dates) or date != self.dates[self.position]:
            raise ValueError('Process exactly the next known session; no skipped or duplicate closes')

    def plan(self, date, planning_nav):
        self._check_date(date)
        if not np.isfinite(planning_nav) or planning_nav <= 0:
            raise ValueError('Positive observed planning NAV required')
        key = self.keys[date]
        if key is None:
            return dict(date=date,requested_extra_notional=0.,reason='outside_contest_window',
                        estimated_only=True)
        past = self.observed.get(key, [])
        sessions = self.groups[key]
        remaining = len(sessions) - len(past)
        assert sessions[len(past)] == date
        low = sum(row['window'] == key[0] and row['turnover'] < .05 for row in self.closed)
        # Future NAV is unknown. Carry today's planning NAV only as an estimate.
        estimated_mean_nav = (sum(row['nav'] for row in past) + remaining * planning_nav) / len(sessions)
        realized = sum(row['gross_notional'] for row in past)
        deficit = max(0., 2 * self.target * estimated_mean_nav - realized)
        trigger = low >= self.intervene_after and remaining <= self.last_sessions and deficit > 0
        return dict(date=date,window=key[0],week_sessions=len(sessions),remaining_sessions=remaining,
                    prior_low_weeks=low,realized_week_notional=realized,
                    estimated_week_mean_nav=estimated_mean_nav,estimated_week_notional_deficit=deficit,
                    requested_extra_notional=deficit / remaining if trigger else 0.,
                    reason='activity_budget_due' if trigger else 'budget_or_calendar_not_due',
                    estimated_only=True)

    def observe(self, date, closing_nav, buy_value, sell_value):
        self._check_date(date)
        if (not all(np.isfinite(x) for x in [closing_nav,buy_value,sell_value])
                or closing_nav <= 0 or buy_value < 0 or sell_value < 0):
            raise ValueError('Positive actual closing NAV and nonnegative actual fills required')
        key = self.keys[date]
        if key is not None:
            rows = self.observed.setdefault(key, [])
            rows.append(dict(date=date,nav=float(closing_nav),gross_notional=float(buy_value+sell_value)))
            if date == self.groups[key][-1]:
                mean_nav = sum(row['nav'] for row in rows) / len(rows)
                turnover = .5 * sum(row['gross_notional'] for row in rows) / mean_nav
                self.closed.append(dict(window=key[0],first_date=rows[0]['date'],last_date=date,
                                        sessions=len(rows),turnover=turnover,actual_closes_and_fills=True))
        self.position += 1


def rank_improving_rotation(qty, desired, locked_qty, prices, scores, eligible,
                            sectors, market_cap, codes, *, nav, gross_ceiling,
                            requested_notional, target_weight=.08, max_orders=10,
                            max_pair_weight=.02):
    """Propose paired reductions/additions; real fills and all limits stay external.

Only untouched, unlocked held shares fund new names in the same sector and
same small/large-cap class. A new name must outrank its funding position.
No more than target_weight is requested per new name. Neither side uses
future execution prices, volume, or realized returns. Unmet demand is retained.
"""
    qty, desired, locked = [np.asarray(x) for x in [qty,desired,locked_qty]]
    prices, scores, eligible, sectors, market_cap, codes = [np.asarray(x) for x in
        [prices,scores,eligible,sectors,market_cap,codes]]
    shape = qty.shape
    if (len(shape) != 1 or not shape[0] or any(x.shape != shape for x in
            [desired,locked,prices,scores,eligible,sectors,market_cap,codes])
            or any(not np.issubdtype(x.dtype,np.integer) or np.any(x<0) for x in [qty,desired,locked])
            or np.any(locked>qty) or eligible.dtype != bool or np.isinf(scores).any()
            or len(set(codes.tolist())) != shape[0]):
        raise ValueError('Matching unique security axes and valid long-only quantities required')
    if (not all(np.isfinite(x) for x in [nav,gross_ceiling,requested_notional,target_weight,max_pair_weight])
            or nav <= 0 or not 0 <= gross_ceiling <= 1 or requested_notional < 0
            or not 0 < max_pair_weight <= target_weight <= .15
            or type(max_orders) is not int or max_orders < 1):
        raise ValueError('Invalid bounded activity proposal policy')
    out = desired.copy()
    good = np.isfinite(prices) & (prices>0)
    # Existing base orders get credit only as proposals, never as realized fills.
    base_notional = float(np.sum(np.abs(out[good]-qty[good])*prices[good]))
    remaining = max(0., float(requested_notional)-base_notional)
    orders = int(np.count_nonzero(out != qty))
    pairs = []
    if gross_ceiling == 0 or remaining == 0 or orders + 2 > max_orders:
        return out,dict(base_planned_notional=base_notional,extra_planned_notional=0.,
                        unmet_planned_notional=remaining,pairs=pairs,estimated_only=True)
    valid = good & np.isfinite(scores) & np.isfinite(market_cap) & (market_cap>=1e11) & (sectors>=0)
    donors = np.flatnonzero(valid & (qty>locked) & (out==qty))
    buyers = np.flatnonzero(valid & eligible & (qty==0) & (out==0))
    donors = donors[np.lexsort((codes[donors],scores[donors]))]
    buyers = buyers[np.lexsort((codes[buyers],-scores[buyers]))]
    used = set()
    for seller in donors:
        if remaining <= 0 or orders + 2 > max_orders:
            break
        choices = [int(i) for i in buyers if i not in used and sectors[i]==sectors[seller]
                   and (market_cap[i]<1e12)==(market_cap[seller]<1e12) and scores[i]>scores[seller]]
        if not choices:
            continue
        buyer = choices[0]
        budget = min(remaining/2, max_pair_weight*nav, target_weight*nav,
                     float((qty[seller]-locked[seller])*prices[seller]))
        sold = min(int(qty[seller]-locked[seller]),int(np.floor(budget/prices[seller])))
        bought = int(np.floor(sold*prices[seller]/prices[buyer]))
        if sold <= 0 or bought <= 0:
            continue
        out[seller] -= sold
        out[buyer] += bought
        notional = sold*prices[seller]+bought*prices[buyer]
        pairs.append(dict(seller=str(codes[seller]),buyer=str(codes[buyer]),sell_qty=sold,buy_qty=bought,
                          planned_notional=float(notional),score_improvement=float(scores[buyer]-scores[seller])))
        used.add(buyer);orders += 2;remaining = max(0.,remaining-notional)
    return out,dict(base_planned_notional=base_notional,
                    extra_planned_notional=float(sum(p['planned_notional'] for p in pairs)),
                    unmet_planned_notional=remaining,pairs=pairs,estimated_only=True)

"""Portable unchanged copies of reviewed independent audit functions."""
from collections import Counter, defaultdict
import numpy as np
import pandas as pd


def inspect_account(panel, ix, result, key, mode, stock_caps):
    assert mode in ['research20', 'formula'], 'Unknown account ceiling'
    codes = {code: c for c, code in enumerate(ix['codes'])}; dates = {date: d for d, date in enumerate(ix['dates'])}
    sectors, first, inverse = np.unique(panel['sector'], return_index=True, return_inverse=True)
    qty = np.zeros(len(codes), np.int64); marks = np.zeros(len(codes)); cash = 1e9
    trades = defaultdict(list)
    for trade in result['trades']: trades[trade['date']].append(trade)
    events, category_days = [], Counter(); maximum_error, reproduced = 0., 0
    for row in result['daily']:
        d = dates[row['date']]; ratio = panel['split'][:, d]
        qty = np.floor(qty * ratio + 1e-7).astype(np.int64)
        marks = np.divide(marks, ratio, out=marks.copy(), where=ratio > 0)
        opening = panel['exec_price'][:, d].astype(float)
        good = np.isfinite(opening) & (opening > 0) & (panel['exec_count'][:, d] >= 25); marks[good] = opening[good]
        for trade in trades[row['date']]:
            sign = 1 if trade['side'] == 'buy' else -1
            qty[codes[trade['code']]] += sign * trade['qty']; cash -= sign * trade['qty'] * trade['price'] + trade['fee']
        close = panel['close'][:, d]; good = np.isfinite(close) & (close > 0); marks[good] = close[good]
        values = qty * marks; nav = cash + values.sum()
        assert nav > 0 and cash >= -.01 and (qty >= 0).all()
        maximum_error = max(maximum_error, abs(nav - row['nav']))
        sector_values = np.bincount(inverse, weights=values, minlength=len(sectors))
        official = panel['sector_cap'][first, d - 1]
        policy = np.minimum(official, .20) if mode == 'research20' else official
        small_value = values[panel['market_cap'][:, d] < 1e12].sum()
        flags = dict(policy_sector=sector_values > policy * nav + 1,
                     official_sector=sector_values > official * nav + 1,
                     small_cap=small_value > .3 * nav + 1,
                     individual=np.any(values > stock_caps[:, d] * nav + 1), gross=values.sum() > .8 * nav + 1)
        reproduced += int(np.any(flags['policy_sector']) or flags['small_cap'] or flags['individual'])
        for category, flag in flags.items():
            if not np.any(flag): continue
            category_days[category] += 1
            if category.endswith('sector'):
                limits = official if category == 'official_sector' else policy
                for s in np.flatnonzero(flag):
                    events.append(dict(id=key, ceiling=mode, date=row['date'], category=category,
                        sector=int(sectors[s]), weight=float(sector_values[s] / nav), limit=float(limits[s])))
            else:
                value = small_value if category == 'small_cap' else values.sum()
                events.append(dict(id=key, ceiling=mode, date=row['date'], category=category,
                    weight=float(value / nav) if category != 'individual' else None))
    assert maximum_error < .01 and reproduced == result['metrics']['closing_weight_breach_days']
    return dict(id=key, ceiling=mode, maximum_nav_error=maximum_error, reproduced_breach_days=reproduced,
                category_days=dict(category_days), account_days=len(result['daily'])), events


def event_key(event, date=None):
    return event['id'], date or event['date'], event['category'], event.get('sector')


def matching_sale(trade, event, p, codes, prior_day):
    if trade['side'] != 'sell': return False
    c = codes[trade['code']]
    if event['category'] == 'official_sector': return p['sector'][c] == event['sector']
    assert event['category'] == 'small_cap'
    return p['market_cap'][c, prior_day] < 1e12


def screen_account(events, result, p, ix):
    above = {event_key(e) for e in events}; assert len(above) == len(events), 'Duplicate closing event'
    assert len({e['id'] for e in events}) <= 1
    dates = {date: d for d, date in enumerate(ix['dates'])}; codes = {code: c for c, code in enumerate(ix['codes'])}
    daily_dates = {row['date'] for row in result['daily']}; trades = defaultdict(list)
    for trade in result['trades']: trades[trade['date']].append(trade)
    screen = []
    for event in events:
        d = dates[event['date']]
        if d + 1 == len(ix['dates']):
            screen.append(dict(**event, next_date=None, filled_relevant_sales=None, status='terminal_unobserved')); continue
        next_date = ix['dates'][d + 1]; assert next_date in daily_dates
        sells = sum(matching_sale(t, event, p, codes, d) for t in trades[next_date])
        status = ('still_above_with_relevant_sale' if sells else 'still_above_no_relevant_sale') if event_key(event, next_date) in above else 'within_limit_next_close'
        screen.append(dict(**event, next_date=next_date, filled_relevant_sales=int(sells), status=status))
    return screen


def plan_checks(event, plans, p, ix, official_caps):
    dates = {date: d for d, date in enumerate(ix['dates'])}; codes = {code: c for c, code in enumerate(ix['codes'])}
    d = dates[event['next_date']]; assert d == dates[event['date']] + 1
    relevant = [plan for plan in plans if plan['date'] == event['next_date'] and matching_sale(plan, event, p, codes, d - 1)]
    checked = []
    for plan in relevant:
        c = codes[plan['code']]; price = float(p['exec_price'][c, d]); volume = float(p['exec_volume'][c, d])
        assert not np.isinf(volume) and (not np.isfinite(volume) or volume >= 0)
        # Preserve the original array dtype and multiplication used by the ledger.
        capacity = int(np.floor(np.nan_to_num(p['exec_volume'][:, d]) * .05)[c])
        count = float(p['exec_count'][c, d]); assert not np.isinf(count) and (not np.isfinite(count) or count >= 0)
        checked.append(dict(**plan, execution_price_missing=not bool(np.isfinite(price) and price > 0),
                            execution_volume=volume if np.isfinite(volume) else None, available_capacity=capacity,
                            print_count=int(count) if np.isfinite(count) else None))
    detail = dict(**event, sell_plans=checked)
    if not checked and event['category'] == 'official_sector':
        limits = official_caps[p['sector'] == event['sector'], d - 1]
        assert len(limits) and np.isfinite(limits).all() and np.all(limits == limits[0])
        detail.update(next_official_limit=float(limits[0]), prior_weight_within_next_limit=bool(event['weight'] <= limits[0]))
    return detail


def classify_plans(p, ix, plans, trades, budget):
    """Follow the frozen order processing order; expose coarse binding stages."""
    dates = {date: d for d, date in enumerate(ix['dates'])}
    codes = {code: c for c, code in enumerate(ix['codes'])}
    fills = {(t['date'], t['code'], t['side']): t for t in trades}
    assert len(fills) == len(trades)
    assert not np.isinf(p['exec_volume']).any()
    assert np.all(np.nan_to_num(p['exec_volume'], nan=0.) >= 0)
    capacities = np.floor(np.nan_to_num(p['exec_volume']) * .05).astype(np.int64)
    seen, done, rows = set(), Counter(), []
    assert [r['date'] for r in plans] == sorted(r['date'] for r in plans)
    for plan in plans:
        key = plan['date'], plan['code'], plan['side']; assert key not in seen; seen.add(key)
        date, code, side = key; d, c = dates[date], codes[code]
        requested = int(plan['requested_qty']); filled = int(fills.get(key, {}).get('qty', 0))
        price = float(p['exec_price'][c, d]); prints = float(p['exec_count'][c, d]); volume = float(p['exec_volume'][c, d])
        assert requested > 0 and 0 <= filled <= requested
        assert not np.isinf(volume) and (not np.isfinite(volume) or volume >= 0)
        capacity = int(capacities[c, d])
        previous = float(p['close'][c, d - 1] / p['split'][c, d])
        locked = bool(np.isclose(p['exec_high'][c, d], p['exec_low'][c, d], rtol=0, atol=.001))
        limited = locked and ((price / previous >= 1.29) if side == 'buy' else (price / previous <= .71))
        ready = np.isfinite(price) and price > 0 and prints >= 25
        if done[date] >= budget: stage = 'order_budget'
        elif not ready: stage = 'missing_execution_price_or_prints'
        elif limited: stage = 'price_limit'
        elif capacity <= 0: stage = 'zero_volume_capacity'
        elif filled == requested: stage = 'filled_as_requested'
        elif capacity < requested and filled == capacity: stage = 'volume_capacity'
        else: stage = 'portfolio_or_cash_headroom'
        if stage in ['order_budget', 'missing_execution_price_or_prints', 'price_limit', 'zero_volume_capacity']:
            assert filled == 0, (key, stage, filled)
        if stage == 'portfolio_or_cash_headroom': assert side == 'buy' and filled < min(requested, capacity)
        planning_price = float(plan['planning_price']); assert np.isfinite(planning_price) and planning_price > 0
        rows.append(dict(date=date, code=code, side=side, requested_qty=requested, filled_qty=filled,
                         requested_at_planning_price=requested * planning_price, filled_at_planning_price=filled * planning_price,
                         unfilled_at_planning_price=(requested - filled) * planning_price, stage=stage,
                         filled_orders_before_plan=done[date], available_capacity=capacity))
        if filled: done[date] += 1
        assert done[date] <= budget
    assert set(fills) <= seen and sum(done.values()) == len(trades)
    return rows


def summarize(screens, details):
    result = {}
    for category in ['official_sector', 'small_cap']:
        selected = [e for e in screens if e['category'] == category]
        unresolved = [e for e in details if e['category'] == category]
        plans = [plan for e in unresolved for plan in e['sell_plans']]
        no_plans = [e for e in unresolved if not e['sell_plans']]
        result[category] = dict(events=len(selected), accounts=len({e['id'] for e in selected}),
            maximum_weight=max((e['weight'] for e in selected), default=None),
            next_close_counts=dict(Counter(e['status'] for e in selected)), events_without_sell_plan=len(no_plans),
            unfilled_sell_plans=len(plans), all_unfilled_plans_missing_price_and_capacity=all(p['execution_price_missing'] and p['available_capacity'] == 0 for p in plans))
        if category == 'official_sector':
            result[category]['all_no_plan_prior_weights_within_next_limit'] = all(e['prior_weight_within_next_limit'] for e in no_plans)
    return result


def count_bootstrap(family, differences, stats, *, draws=4000, seed=57):
    """Count each sampled day, then multiply; no resampled return tensor."""
    a = np.column_stack(differences); assert np.isfinite(a).all() and a.shape[1] == len(family)
    mean = a.mean(0); centered = a - mean; reports, bounds = [], []
    for block in [5, 10]:
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, len(a), size=(draws, (len(a) + block - 1) // block))
        indices = ((starts[:, :, None] + np.arange(block)) % len(a)).reshape(draws, -1)[:, :len(a)]
        counts = np.stack([np.bincount(row, minlength=len(a)) for row in indices])
        boot = counts @ centered / len(a); se = np.maximum(boot.std(0, ddof=1), 1e-12)
        maxima = (boot / se).max(1); observed = mean / se
        expected = dict(mean=mean, standard_error=se,
                        adjusted_p=(1 + (maxima[:, None] >= observed).sum(0)) / (draws + 1),
                        marginal_p=(1 + (boot / se >= observed).sum(0)) / (draws + 1),
                        simultaneous_lower95=mean - np.quantile(maxima, .95) * se)
        stored = stats[stats.block == block].reset_index(drop=True)
        assert list(stored[['id', 'comparator', 'origin']].itertuples(index=False, name=None)) == family
        errors = {key: float(np.max(np.abs(value - stored[key].to_numpy()))) for key, value in expected.items()}
        for key in ['mean', 'standard_error', 'simultaneous_lower95']: assert errors[key] < 1e-13
        eps = np.finfo(np.float64).eps
        tolerance = 512 * eps * np.maximum(np.max(np.abs(a), axis=0), 1e-300)
        t_tolerance = 512 * eps * (np.max(np.abs(maxima)) + np.abs(observed) + 1)
        marginal_low = (1 + (boot > mean + tolerance).sum(0)) / (draws + 1)
        marginal_high = (1 + (boot >= mean - tolerance).sum(0)) / (draws + 1)
        adjusted_low = (1 + (maxima[:, None] > observed + t_tolerance).sum(0)) / (draws + 1)
        adjusted_high = (1 + (maxima[:, None] >= observed - t_tolerance).sum(0)) / (draws + 1)
        assert np.all(stored.marginal_p >= marginal_low - 1e-12) and np.all(stored.marginal_p <= marginal_high + 1e-12)
        assert np.all(stored.adjusted_p >= adjusted_low - 1e-12) and np.all(stored.adjusted_p <= adjusted_high + 1e-12)
        for j, (key, comp, origin) in enumerate(family):
            bounds.append(dict(id=key, comparator=comp, origin=origin, block=block,
                               marginal_low=float(marginal_low[j]), marginal_high=float(marginal_high[j]),
                               adjusted_low=float(adjusted_low[j]), adjusted_high=float(adjusted_high[j]),
                               simultaneous_lower95=float(expected['simultaneous_lower95'][j])))
        reports.append(dict(block=block, draws=draws, maximum_point_errors=errors,
                            marginal_roundoff_tie_columns=int((marginal_high > marginal_low).sum()),
                            adjusted_roundoff_tie_columns=int((adjusted_high > adjusted_low).sum()),
                            recorded_p_values_within_roundoff_bounds=True))
    return reports, pd.DataFrame(bounds)

SOURCE_FUNCTIONS = {'inspect_account': {'source': 'audit_sector_ceiling_closing.py', 'source_sha256': 'cda11bd10e22a1d17b1a81200361fb6ff1ec23b6e60942a12e7323632fc9454f'}, 'event_key': {'source': 'audit_adjustment_band_closing.py', 'source_sha256': '7d250e97c082ebb77a5e2b68d8cb385a2aaddb6b23cc7e85f4311cd687a6ede4'}, 'matching_sale': {'source': 'audit_adjustment_band_closing.py', 'source_sha256': '7d250e97c082ebb77a5e2b68d8cb385a2aaddb6b23cc7e85f4311cd687a6ede4'}, 'screen_account': {'source': 'audit_adjustment_band_closing.py', 'source_sha256': '7d250e97c082ebb77a5e2b68d8cb385a2aaddb6b23cc7e85f4311cd687a6ede4'}, 'plan_checks': {'source': 'audit_adjustment_band_closing.py', 'source_sha256': '7d250e97c082ebb77a5e2b68d8cb385a2aaddb6b23cc7e85f4311cd687a6ede4'}, 'classify_plans': {'source': 'audit_execution_gap.py', 'source_sha256': 'fda598c0523e66f0845b9e1ef3aa2b1863e311dceec9d2be2e9a1684b2f52683'}, 'summarize': {'source': 'audit_stock_relation_closing.py', 'source_sha256': 'ab9645391418214ca1474fdd9066058f967e59855d537b0e09b8079e7208a1c8'}, 'count_bootstrap': {'source': 'audit_stock_relation_run.py', 'source_sha256': '733fdb1d2d8188a1a1b312d670faa6bce015185518780d89c30c149a45d9ef3b'}}

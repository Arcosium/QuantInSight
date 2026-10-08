import calendar
import math
import statistics


def normalized_date(value):
    s = ''.join(c for c in str(value) if c.isdigit())
    return s[:8]


def ledger(case, start=None, end=None):
    rows = sorted((r for r in case.get('daily', []) if isinstance(r,dict)
                   and 'nav' in r and 'date' in r), key=lambda r: normalized_date(r['date']))
    initial = float(case.get('initial_cash', case.get('initial_nav', 1e9)))
    result = []
    previous = initial
    for row in rows:
        d = normalized_date(row['date'])
        nav = float(row['nav'])
        if not math.isfinite(nav) or nav <= 0 or previous <= 0:
            raise ValueError('Invalid account NAV')
        if (start is None or d >= start) and (end is None or d <= end):
            result.append(dict(row, date=d, previous_nav=previous, net_daily_return=nav/previous-1))
        previous = nav
    return result


def statistics_for(rows):
    if not rows:
        raise ValueError('Empty account evaluation')
    daily = [r['net_daily_return'] for r in rows]
    anchor = rows[0]['previous_nav']
    monthly = {}
    for r in rows:
        key = r['date'][:6]
        monthly.setdefault(key,[]).append(r)
    months = [dict(month=k, net_return=v[-1]['nav']/v[0]['previous_nav']-1,
                   start_nav=v[0]['previous_nav'], end_nav=v[-1]['nav']) for k,v in sorted(monthly.items())]
    deviation = statistics.stdev(daily) if len(daily)>1 else 0
    sharpe = statistics.mean(daily)/deviation*math.sqrt(252) if deviation>0 else None
    peak, mdd = anchor, 0.
    for r in rows:
        peak = max(peak,r['nav'])
        mdd = min(mdd,r['nav']/peak-1)
    losses = [m['net_return'] for m in months if m['net_return'] < 0]
    return dict(net_return=rows[-1]['nav']/anchor-1, negative_months=len(losses),
                months=len(months), start=rows[0]['date'], end=rows[-1]['date'],
                sessions=len(rows), sharpe=sharpe, mdd=mdd,
                mean_loss_month=statistics.mean(losses) if losses else 0,
                worst_month=min(m['net_return'] for m in months), monthly=months)


def evaluation_scope(case):
    for metrics in [case.get(k) for k in ['registered33','registered','evaluation','full_history_metrics','metrics']]+[case]:
        if not isinstance(metrics,dict):
            continue
        folds = metrics.get('monthly_folds')
        if isinstance(folds,dict) and folds:
            months = sorted(normalized_date(k)[:6] for k in folds)
            if all(len(m)==6 for m in months):
                y,m = int(months[-1][:4]),int(months[-1][4:])
                return months[0]+'01',months[-1]+f'{calendar.monthrange(y,m)[1]:02}',metrics
    return None,None,None


def pareto_front(rows, return_key='net_return', loss_key='negative_months'):
    valid = [r for r in rows if isinstance(r.get(return_key),(int,float)) and math.isfinite(r[return_key])
             and isinstance(r.get(loss_key),(int,float)) and math.isfinite(r[loss_key])]
    front = []
    for p in valid:
        dominated = any(q[return_key]>=p[return_key] and q[loss_key]<=p[loss_key]
                        and (q[return_key]>p[return_key] or q[loss_key]<p[loss_key]) for q in valid)
        if not dominated:
            front.append(p)
    return sorted(front,key=lambda r:(r[loss_key],r[return_key],r.get('id','')))

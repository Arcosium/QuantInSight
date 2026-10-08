"""Anchored 24-month IS / 9-month OS / 3-month ROS; only OS selects models."""
import calendar
import math
import statistics
from datetime import date, datetime, timedelta
from .period import window
from .metrics import statistics_for

PROTOCOL = '24is_9os_3ros_v1'


def shift_months(day, count):
    total=day.year*12+day.month-1+count;year,month=divmod(total,12)
    return date(year,month+1,min(day.day,calendar.monthrange(year,month+1)[1]))


def periods(start=None,end=None):
    if not start or not end:start,end=window()
    finish=datetime.strptime(end,'%Y%m%d').date()+timedelta(days=1)
    os_start=shift_months(finish,-12);ros_start=shift_months(finish,-3)
    return {'is':dict(start=start,end=(os_start-timedelta(days=1)).strftime('%Y%m%d'),months=24),
            'os':dict(start=os_start.strftime('%Y%m%d'),end=(ros_start-timedelta(days=1)).strftime('%Y%m%d'),months=9),
            'ros':dict(start=ros_start.strftime('%Y%m%d'),end=end,months=3)}


def segment_metrics(rows,start,end,months):
    selected=[r for r in rows if start<=r['date']<=end]
    if not selected:return None
    result=statistics_for(selected)
    finish=datetime.strptime(end,'%Y%m%d').date()+timedelta(days=1)
    boundaries=[shift_months(finish,-i).strftime('%Y%m%d') for i in range(months,-1,-1)]
    monthly=[]
    for left,right in zip(boundaries[:-1],boundaries[1:]):
        values=[r for r in selected if left<=r['date']<right]
        if not values:return None
        monthly.append(dict(month=left[:6],start=left,end=(datetime.strptime(right,'%Y%m%d').date()-timedelta(days=1)).strftime('%Y%m%d'),
                            net_return=values[-1]['nav']/values[0]['previous_nav']-1))
    losses=[r['net_return'] for r in monthly if r['net_return']<0]
    result.update(mean_loss_month=statistics.mean(losses) if losses else 0, worst_month=min(r['net_return'] for r in monthly),months=months,monthly=monthly,negative_months=sum(r['net_return']<0 for r in monthly),
                  start=selected[0]['date'],end=selected[-1]['date'])
    return result


def performance(rows,start=None,end=None):
    return {name:segment_metrics(rows,**span) for name,span in periods(start,end).items()}


def selection_metrics(summary):
    if summary.get('evaluation_protocol')!=PROTOCOL:return None
    metric=(summary.get('performance') or {}).get('os')
    if not isinstance(metric,dict) or metric.get('months')!=9:return None
    if not all(isinstance(metric.get(k),(int,float)) and math.isfinite(metric[k]) for k in ('net_return','negative_months','mdd')):return None
    return metric


def valid_window(span):
    try:
        start,end=span
        finish=datetime.strptime(end,'%Y%m%d').date()+timedelta(days=1)
        return start==shift_months(finish,-36).strftime('%Y%m%d')
    except (ValueError,TypeError):return False


def valid_result(summary):
    from .period import using_window, accepts
    span=summary.get('evaluation_window')
    if not valid_window(span):return False
    with using_window(*span):return accepts(summary)

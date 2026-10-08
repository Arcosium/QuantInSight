"""One shared window: the last 36 completed calendar months in Seoul."""
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo


def window(today=None):
    today = today or datetime.now(ZoneInfo('Asia/Seoul')).date()
    end = today.replace(day=1) - timedelta(days=1)
    start = date(today.year - 3, today.month, 1)
    return start.strftime('%Y%m%d'), end.strftime('%Y%m%d')


def accepts(summary, today=None):
    start, end = window(today)
    # Exact trading calendar is verified when a result is produced. These bounds
    # reject old and partial windows, including a missing final trading week.
    return (summary.get('months') == 36 and
            start <= summary.get('start', '') <= start[:6] + '10' and
            end[:6] + '20' <= summary.get('end', '') <= end)

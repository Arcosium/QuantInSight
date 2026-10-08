"""Public-news availability gates and causal, shared daily activity features."""
from datetime import datetime,timedelta,timezone
from functools import lru_cache
from pathlib import Path
import json
import sqlite3
from .briefing import NEWS_DB,_date,_category
from .period import window


@lru_cache(maxsize=16)
def _news_status(start,end,bucket):
    try:
        with sqlite3.connect(NEWS_DB.resolve().as_uri()+'?mode=ro',uri=True,timeout=2) as db:
            first,last,total=db.execute('SELECT min(collected_at),max(collected_at),count(*) FROM articles').fetchone()
        first,last=_date(first),_date(last)
        ready=bool(first and last and first.strftime('%Y%m%d')<=start and last.strftime('%Y%m%d')>=end)
        return dict(ready=ready,start=first.date().isoformat() if first else None,end=last.date().isoformat() if last else None,articles=total,
                    message='뉴스 활동량 · 수집 시각 이후에만 사용' if ready else '뉴스 수집 이력이 IS 학습 구간을 덮지 않습니다. 부족한 과거 뉴스를 0으로 채워 학습하지 않습니다.')
    except (OSError,sqlite3.Error,ValueError):
        return dict(ready=False,start=None,end=None,articles=0,message='뉴스 원본 수집 이력 확인 필요')


def news_status(market=None):
    import time
    return dict(_news_status(*window(),int(time.time()//300)))


def prepare_news(market):
    """Counts only, not language sentiment; midnight UTC shifts to the next date."""
    import pandas as pd
    from .config import RUNS
    if not news_status(market)['ready']:raise ValueError(news_status(market)['message'])
    start,end=window();path=RUNS/'research_inputs'/f'{start}-{end}'/f'{market}-news-activity.parquet'
    # Multiple research workers may request the same immutable snapshot.
    import fcntl
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _write_news(market,start,end,path)


def _write_news(market,start,end,path):
    import pandas as pd
    if path.exists():return path
    counts={}
    with sqlite3.connect(NEWS_DB.resolve().as_uri()+'?mode=ro',uri=True,timeout=2) as db:
        for title,published,collected in db.execute('SELECT title,published_at,collected_at FROM articles'):
            collected=_date(collected);published=_date(published)
            if not collected:continue
            known=max(collected,published) if published else collected
            day=(known.astimezone(timezone.utc)+timedelta(days=1)).strftime('%Y%m%d')
            category=_category(title or '')
            if day>end or category not in (market,'world') and not (market=='timefolio' and category=='kr'):continue
            counts[day]=counts.get(day,0)+1
    if not counts:raise ValueError('해당 시장의 시점 정렬된 뉴스 없음')
    first=min(counts)
    rows=pd.DataFrame({'date':pd.date_range(pd.Timestamp(first),pd.Timestamp(end))})
    rows['news_count']=rows.date.dt.strftime('%Y%m%d').map(counts).fillna(0).astype('float32')
    rows['news_count_7']=rows.news_count.rolling(7,min_periods=1).mean()
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');rows.to_parquet(tmp,index=False);tmp.replace(path)
    return path


def news_descriptor(market):
    from .model_recipe import file_hash
    path=prepare_news(market)
    return dict(path=str(path.resolve()),sha256=file_hash(path),kind='causal_public_news_activity')

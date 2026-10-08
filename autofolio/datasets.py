"""Read-only dataset inventory. Periods describe observed records, never file mtimes."""
import csv
import datetime as dt
import re
import sqlite3
import threading
import time
from pathlib import Path
from .config import HOME, ROOT

_LOCK = threading.Lock()
_CACHE = None
_CACHE_AT = 0
CACHE_SECONDS = 300


def date_value(value):
    if value is None: return None
    text = str(value)
    if re.fullmatch(r'(?:19|20)\d{6}(?:\d{4}|\d{6})?', text):
        try: return dt.datetime.strptime(text[:8], '%Y%m%d').date().isoformat()
        except ValueError: return None
    if isinstance(value, (int, float)) or re.fullmatch(r'\d{10,19}', text):
        n = float(value)
        while n > 1e11: n /= 1000
        try: return dt.datetime.fromtimestamp(n, dt.timezone.utc).date().isoformat()
        except (ValueError, OverflowError, OSError): return None
    try: return dt.date.fromisoformat(text[:10]).isoformat()
    except ValueError: return None


def sqlite_period(path, table, column, group=None, budget=3):
    # Identifiers are fixed by this module, never accepted from HTTP input.
    deadline = time.monotonic() + budget
    with sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True, timeout=.2) as db:
        db.execute('PRAGMA query_only=ON')
        db.set_progress_handler(lambda: int(time.monotonic() > deadline), 2000)
        if not group:
            lo = db.execute(f'SELECT min("{column}") FROM "{table}"').fetchone()[0]
            hi = db.execute(f'SELECT max("{column}") FROM "{table}"').fetchone()[0]
            values = (lo, hi)
        else:
            # Skip through the compound primary key instead of scanning every bar.
            periods, last = [], ''
            while True:
                if time.monotonic() > deadline: raise TimeoutError()
                row = db.execute(f'SELECT "{group}" FROM "{table}" WHERE "{group}">? ORDER BY "{group}" LIMIT 1', (last,)).fetchone()
                if row is None: break
                last = row[0]
                lo = db.execute(f'SELECT "{column}" FROM "{table}" WHERE "{group}"=? ORDER BY "{column}" LIMIT 1',(last,)).fetchone()
                hi = db.execute(f'SELECT "{column}" FROM "{table}" WHERE "{group}"=? ORDER BY "{column}" DESC LIMIT 1',(last,)).fetchone()
                if lo and hi: periods.append((lo[0], hi[0]))
            values = (min(p[0] for p in periods), max(p[1] for p in periods)) if periods else (None,None)
        return tuple(date_value(v) for v in values)


def parquet_period(path, partitioned=False, budget=12):
    import pyarrow.parquet as pq
    deadline = time.monotonic() + budget
    files = [path] if path.is_file() else []
    # Monthly partition names select only boundary partitions; footer statistics
    # then establish actual observed dates. Empty marker files do not count.
    for file in (() if path.is_file() else (path.glob('*/*.parquet') if partitioned else path.glob('*.parquet'))):
        if time.monotonic() > deadline: raise TimeoutError()
        files.append(file)
    monthly = [(re.search(r'(\d{4}-\d{2})(?:\.parquet)$', f.name), f) for f in files]
    if files and all(m for m, _ in monthly):
        keys = [m.group(1) for m, _ in monthly]
        bounds = {min(keys), max(keys)}
        files = [f for m, f in monthly if m.group(1) in bounds]
    periods=[]
    for file in files:
        if time.monotonic() > deadline: raise TimeoutError()
        metadata = pq.ParquetFile(file).metadata
        for j in range(metadata.num_row_groups):
            rg=metadata.row_group(j)
            for i in range(rg.num_columns):
                col=rg.column(i)
                if col.path_in_schema in ('ts','t','date','timestamp','datetime'):
                    st=col.statistics
                    if st and st.has_min_max:
                        lo,hi=date_value(st.min),date_value(st.max)
                        if lo and hi: periods.append((lo,hi))
                    else: raise ValueError('날짜 통계 없음')
    return union(periods)


def union(periods):
    valid=[(a,b) for a,b in periods if a and b]
    return (min(a for a,b in valid),max(b for a,b in valid)) if valid else (None,None)


def csv_period(path, column='date'):
    with path.open(encoding='utf-8-sig',newline='') as f:
        dates=[date_value(r.get(column)) for r in csv.DictReader(f)]
    dates=[v for v in dates if v]
    return (min(dates),max(dates)) if dates else (None,None)


def csv_folder_period(path, pattern):
    # Daily history files are small; read boundary lines rather than full rows.
    periods=[]
    deadline=time.monotonic()+8
    for file in path.glob(pattern):
        if time.monotonic()>deadline: raise TimeoutError()
        with file.open('rb') as f:
            header=f.readline().decode('utf-8-sig').strip().split(',')
            if 'date' not in header: continue
            index=header.index('date')
            first=f.readline().decode().strip().split(',')
            f.seek(0,2); size=f.tell(); f.seek(max(0,size-4096))
            lines=f.read().decode().strip().splitlines()
            last=lines[-1].split(',') if lines else []
            if len(first)>index and len(last)>index:
                periods.append((date_value(first[index]),date_value(last[index])))
    return union(periods)


def panel_period(path):
    import numpy as np
    values=np.load(path/'dates.npy',mmap_mode='r',allow_pickle=False)
    return date_value(min(values).item()),date_value(max(values).item())


def financial_period(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        years=[int(r['기준연도']) for r in csv.DictReader(f) if r.get('기준연도','').isdigit()]
    return (f'{min(years)}-01-01',f'{max(years)}-12-31') if years else (None,None)


def source(name,path,reader,note=''):
    found=path.exists()
    start=end=None
    status='확인 필요'
    if found:
        try:
            start,end=reader(path)
            if start and end:status='확인됨'
        except (OSError, ValueError, KeyError, sqlite3.Error, TimeoutError, ImportError): pass
    return dict(name=name,available=found,updated=path.stat().st_mtime if found else None,
                start=start,end=end,period_status=status,note=note)


def inventory():
    vault=HOME/'vault'; bars=vault/'CryptoBars/data'; daily=vault/'QuantInSight/data'
    specs=[
      ('한국 분봉','KRX 1분 시세 · 현재 수집과 과거 OHLCV',[
       ('KRX OHLCV',bars/'KRX/bars_ohlc.db',lambda p:sqlite_period(p,'bars','ts','code'),''),
       ('KRX 수집 시세',bars/'KRX/bars.db',lambda p:sqlite_period(p,'bars','ts'),'종가·누적 거래량'),
       ('KRX 과거 OHLCV',bars/'KRX/bars_toss.db',lambda p:sqlite_period(p,'bars','ts','code'),'')]),
      ('미국 분봉','미국 주식 1분 OHLCV', [('미국 1분',bars/'USA/1m',lambda p:parquet_period(p,True),'UTC 기준')]),
      ('크립토 분봉','거래소별 1분 OHLCV · 현재 수집과 과거 자료',[
       ('현재 수집',bars/'bars',lambda p:parquet_period(p,True),'UTC 기준'),
       ('과거 자료',bars/'history',lambda p:parquet_period(p,True),'UTC 기준')]),
      ('NXT 분봉','넥스트레이드 1분 OHLCV',[('NXT',bars/'NXT/bars.db',lambda p:sqlite_period(p,'bars','ts'),'')]),
      ('미국 일봉','미국 주식 일봉 · 정책 대상 종목과 보유 시세',[
       ('정책 대상 일봉',bars/'USA/daily_policy',parquet_period,''),
       ('미국 보유 일봉',daily,lambda p:csv_folder_period(p,'daily_US_*.csv'),'')]),
      ('뉴스','증권 뉴스 · RSS',[('뉴스 수집 기록',HOME/'projects/lib/data/arcnews.db',lambda p:sqlite_period(p,'articles','collected_at'),'발행일 형식이 달라 수집일 기준으로 표시')]),
      ('공시·재무','DART 기업 재무 · 회계연도 기준',[('기업 재무',ROOT/'integrations/timefolio/quant/financials_5y.csv',financial_period,'일별 관측이 아닌 회계연도 범위')]),
      ('한국 일봉','한국 주식 수정 주가·거래량', [('한국 보유 일봉',daily,lambda p:csv_folder_period(p,'daily_[0-9]*.csv'),'')]),
      ('시장 분석','종목·수급 사건 분석', [('시장 분석 사건',vault/'HYFE/9.28/market_data/analysis-20260928-revision-03',lambda p:union([csv_period(p/'same_price_events.csv','ts'),csv_period(p/'regime_events.csv','ts')]),'사건 발생 시각 기준')]),
    ]
    result=[]
    for name,description,definitions in specs:
        sources=[source(*spec) for spec in definitions]
        start,end=union([(s['start'],s['end']) for s in sources])
        unknown=any(s['available'] and not s['start'] for s in sources)
        result.append(dict(name=name,description=description,available=any(s['available'] for s in sources),
            updated=max((s['updated'] for s in sources if s['updated']),default=None),start=start,end=end,
            period_status='일부 확인 필요' if unknown and start else ('확인됨' if start else '확인 필요'),sources=sources,
            coverage_note='종목·거래소별 기간은 다릅니다. 표시 범위는 확인된 자료의 합집합이며 전 기간 연속 보유를 뜻하지 않습니다.'))
    return dict(datasets=result)


def datasets():
    global _CACHE,_CACHE_AT
    with _LOCK:
        if _CACHE is None or time.monotonic()-_CACHE_AT>CACHE_SECONDS:
            _CACHE=inventory();_CACHE_AT=time.monotonic()
        return _CACHE

"""Shared, frozen OHLCV inputs; never duplicate a dataset for each model trial."""
import argparse
import hashlib
import json
from pathlib import Path
from .config import HOME, RUNS
from .period import window, expected_dates, using_window, last_complete_date


def paths(market):
    from .research_input import paths as stocks
    if market != 'crypto': return stocks('kr' if market == 'timefolio' else market)
    folder = RUNS/'research_inputs'/('-'.join(window()))
    return folder/'crypto-learning.parquet', folder/'crypto-learning.audit.json'


def status(market):
    data, audit = paths(market)
    try: info = json.loads(audit.read_text())
    except (OSError, ValueError): info = {}
    return dict(ready=bool(info.get('ready') and data.exists() and info.get('window') == list(window()) and (market!='crypto' or info.get('learning_input_version')==3)),
                message=info.get('message', '최근 36개월 학습 입력 준비 필요'),
                start=info.get('start'), end=info.get('end'), symbols=info.get('symbols', 0))


def descriptor(market):
    data, audit = paths(market)
    from .model_recipe import file_hash
    info=json.loads(audit.read_text())
    return dict(path=str(data.resolve()),sha256=file_hash(data),audit_path=str(audit.resolve()),audit_sha256=file_hash(audit),
                window=info.get('window'),symbols=info.get('symbols'),limitations=info.get('limitations',[]))


def validate_minutes(frame,previous_ts=None):
    """Reject overlapping venue bars, malformed OHLCV and unordered exports."""
    import numpy as np
    if frame.empty:return previous_ts
    if frame.ts.isna().any() or not frame.ts.is_monotonic_increasing or frame.ts.duplicated().any():
        raise ValueError('크립토 원본 분봉 중복/정렬 오류')
    if previous_ts is not None and frame.ts.iloc[0]<=previous_ts:raise ValueError('크립토 원본 분봉 경계 중복/정렬 오류')
    values=frame[['open','high','low','close','quote_volume']].to_numpy(float)
    if not np.isfinite(values).all() or (values[:,:4]<=0).any() or (values[:,4]<0).any():raise ValueError('크립토 원본 분봉 OHLCV 오류')
    if (frame.high<frame[['open','close','low']].max(axis=1)).any() or (frame.low>frame[['open','close','high']].min(axis=1)).any():raise ValueError('크립토 원본 분봉 OHLC 범위 오류')
    return int(frame.ts.iloc[-1])


def canonical_minutes(parts, reject_invalid=False):
    """Same history-first merge as CryptoBars/export.py, with deterministic ties."""
    import pandas as pd
    joined=pd.concat(parts,ignore_index=True)
    # Stable sorting preserves sorted file order for history/history ties.
    joined=joined.sort_values(['_source','ts'],kind='stable')
    duplicates=int(joined.duplicated('ts').sum())
    joined=joined.drop_duplicates('ts',keep='first').sort_values('ts').reset_index(drop=True)
    rejected=0
    if reject_invalid:
        joined,rejected=valid_minutes(joined)
    validate_minutes(joined)
    joined.attrs['rejected_rows']=rejected
    return joined,duplicates


def valid_minutes(frame):
    """Drop malformed minute rows; never repair OHLCV or missing quote volume."""
    import numpy as np
    values=frame[['open','high','low','close','quote_volume']].to_numpy(float)
    valid=(np.isfinite(values).all(axis=1)&(values[:,:4]>0).all(axis=1)&(values[:,4]>=0)
           &frame.ts.notna().to_numpy()
           &(frame.high>=frame[['open','close','low']].max(axis=1)).to_numpy()
           &(frame.low<=frame[['open','close','high']].min(axis=1)).to_numpy())
    return frame.loc[valid].copy(),int((~valid).sum())


def daily_minutes(frame):
    import pandas as pd
    d=frame.copy();d['date']=pd.to_datetime(d.ts,unit='ms').dt.floor('D')
    return d.groupby('date').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('quote_volume','sum'),minutes=('ts','nunique'))


def input_fingerprints(files):
    return [dict(path=str(file),size=file.stat().st_size,mtime_ns=file.stat().st_mtime_ns) for file in sorted(set(files))]


def symbol_cache_paths(symbol):
    folder=RUNS/'research_inputs/crypto_daily_cache/v3';folder.mkdir(parents=True,exist_ok=True)
    name=hashlib.sha256(symbol.encode()).hexdigest()
    return folder/(name+'.parquet'),folder/(name+'.json')


def source_signature(refs,end):
    # Source identity includes precise modification time and length. The actual
    # derived parquet is independently content-hashed before every reuse.
    return hashlib.sha256(json.dumps(dict(version=3,end=str(end),sources=refs),sort_keys=True).encode()).hexdigest()


def cached_symbol(symbol,signature=None):
    import pyarrow.parquet as pq
    from .model_recipe import file_hash
    data,audit=symbol_cache_paths(symbol)
    try:
        meta=json.loads(audit.read_text())
        if (signature is not None and meta['source_signature']!=signature) or meta['content_sha256']!=file_hash(data):return None
        return pq.ParquetFile(data).read(use_threads=False).to_pandas(),meta
    except (OSError,ValueError,KeyError):return None


def save_symbol(symbol,frame,meta):
    from .model_recipe import file_hash
    data,audit=symbol_cache_paths(symbol)
    temp=data.with_suffix('.tmp');frame.to_parquet(temp,index=False)
    meta=dict(meta,content_sha256=file_hash(temp));temp.replace(data)
    pending=audit.with_suffix('.tmp');pending.write_text(json.dumps(meta,ensure_ascii=False));pending.replace(audit)
    return meta


def prepare_symbol(folder,months,live_by_month,end):
    import pandas as pd
    import pyarrow.parquet as pq
    symbol=folder.name.split('=',1)[1]
    monthly={month:(sorted(folder.glob('part-'+month+'*.parquet')),live_by_month.get(month,[])) for month in months}
    refs=input_fingerprints([file for pair in monthly.values() for files in pair for file in files])
    signature=source_signature(refs,end)
    cached=cached_symbol(symbol,signature)
    if cached is not None:
        d,meta=cached
    else:
        previous=cached_symbol(symbol)
        old_frame,old_meta=previous if previous else (None,{})
        def source_month(ref):
            path=Path(ref['path'])
            return path.name[5:12] if path.name.startswith('part-') else path.parent.name[5:12] if path.parent.name.startswith('date=') else ''
        parts=[];reused_frames=[];rejected={};duplicates_total=0;duplicate_by_month={}
        for month,(history,live) in monthly.items():
            first=pd.Timestamp(month+'-01');stop=first+pd.offsets.MonthBegin(1);minute_parts=[]
            current_refs=[r for r in refs if source_month(r)==month]
            old_refs=[r for r in old_meta.get('source_references',[]) if source_month(r)==month]
            if previous and current_refs and current_refs==old_refs and month in old_meta.get('duplicate_by_month',{}):
                saved=old_frame[(old_frame.date>=first)&(old_frame.date<stop)]
                if not saved.empty:reused_frames.append(saved.copy())
                if old_meta.get('rejected_rows',{}).get(month):rejected[month]=old_meta['rejected_rows'][month]
                duplicate_by_month[month]=old_meta['duplicate_by_month'][month]
                duplicates_total+=duplicate_by_month[month]
                continue
            for priority,files in enumerate((history,live)):
                for file in files:
                    for batch in pq.ParquetFile(file).iter_batches(batch_size=131072,columns=['ts','base','open','high','low','close','quote_volume'],use_threads=False):
                        d=batch.to_pandas();d=d[(d.base==symbol)&(d.ts>=first.value//10**6)&(d.ts<stop.value//10**6)].copy()
                        if not d.empty:d['_source']=priority;minute_parts.append(d)
            if not minute_parts:continue
            d,duplicates=canonical_minutes(minute_parts,reject_invalid=True);duplicates_total+=duplicates
            duplicate_by_month[month]=duplicates
            if d.attrs['rejected_rows']:rejected[month]=d.attrs['rejected_rows']
            if not d.empty:parts.append(daily_minutes(d))
        if parts:
            d=pd.concat(parts).reset_index()
            d['symbol']=symbol;d['sector']='CRYPTO';d['eligible']=d.minutes.ge(1200)&d.volume.gt(0)
            d['tradable_buy']=d['tradable_sell']=d.eligible;d=d.drop(columns='minutes')
        else:
            d=pd.DataFrame(columns=['date','open','high','low','close','volume','symbol','sector','eligible','tradable_buy','tradable_sell'])
        if reused_frames:d=pd.concat(([d] if not d.empty else [])+reused_frames,ignore_index=True).sort_values('date').reset_index(drop=True)
        if refs!=input_fingerprints([Path(r['path']) for r in refs]):raise ValueError('크립토 입력 파일 변경 감지; 준비 재시도 필요')
        meta=save_symbol(symbol,d,dict(symbol=symbol,source_signature=signature,source_references=refs,rejected_rows=rejected,duplicate_minutes=duplicates_total,duplicate_by_month=duplicate_by_month))
    return symbol,d,meta,refs,int(cached is not None)


def ordered_symbols(folders,months,live_by_month,end,workers):
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    # At most workers futures in flight, returned in canonical symbol order.
    folders=iter(folders)
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='crypto-input') as pool:
        pending=deque()
        for _ in range(workers):
            folder=next(folders,None)
            if folder is not None:pending.append(pool.submit(prepare_symbol,folder,months,live_by_month,end))
        while pending:
            result=pending.popleft().result()
            folder=next(folders,None)
            if folder is not None:pending.append(pool.submit(prepare_symbol,folder,months,live_by_month,end))
            yield result


def prepare(market):
    with using_window(*window()):
        return _prepare(market)


def _prepare(market):
    import pandas as pd
    import pyarrow.parquet as pq
    if market != 'crypto':
        from .research_input import prepare as stocks
        return stocks('kr' if market == 'timefolio' else market)
    target, audit = paths(market)
    target.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with target.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if status(market)['ready']: return pq.ParquetFile(target).read(use_threads=False).to_pandas()
        # Canonical history is updated nightly; the export is only an old delivery.
        # Bound memory to one symbol-month and preserve history over live overlaps.
        source=HOME/'vault/CryptoBars/data'; end=pd.Timestamp(window()[1])+pd.Timedelta(days=1)
        months=[x.strftime('%Y-%m') for x in pd.date_range('2023-01-01',end-pd.Timedelta(days=1),freq='MS')]
        live_by_month={}
        for folder in sorted((source/'bars').glob('date=*')):
            try:day=pd.Timestamp(folder.name.split('=',1)[1])
            except ValueError:continue
            if day>=end:continue
            live_by_month.setdefault(day.strftime('%Y-%m'),[]).extend(sorted(folder.glob('*.parquet')))
        frames=[];sources={};duplicate_minutes=0;rejected_rows={};cache_hits=0
        folders=sorted((source/'history').glob('base=*'))
        from .store import setting
        workers=min(3,max(1,int(setting('cpu_cores',4))-1))
        for number,result in enumerate(ordered_symbols(folders,months,live_by_month,end,workers),1):
            symbol,d,meta,refs,hit=result
            sources.update({r['path']:r for r in refs});cache_hits+=hit
            duplicate_minutes+=meta['duplicate_minutes']
            if meta['rejected_rows']:rejected_rows[symbol]=meta['rejected_rows']
            if not d.empty:frames.append(d)
            if number%25==0 or number==len(folders):
                print(json.dumps(dict(phase='crypto_daily_input',workers=workers,processed=number,total=len(folders),cache_hits=cache_hits,rejected_rows=sum(sum(x.values()) for x in rejected_rows.values()))),flush=True)
        if not frames: raise ValueError('크립토 원본 분봉 없음')
        out=pd.concat(frames,ignore_index=True).sort_values(['symbol','date']).reset_index(drop=True)
        completed=min(end,pd.Timestamp(last_complete_date('crypto'))+pd.Timedelta(days=1))
        out=out[out.date<completed].copy()
        from .research_input import validate_frame
        for _, group in out.groupby('symbol',sort=False): validate_frame(group)
        counts=out[out.eligible].groupby(out.date.dt.strftime('%Y%m%d')).symbol.nunique()
        missing=[day for day in expected_dates('crypto',*window()) if counts.get(day,0)<20]
        repair_info={}
        if missing:
            from .crypto_repair import repair
            closed_missing=[day for day in missing if pd.Timestamp(day)<completed]
            out,repair_info=repair(out,closed_missing,source)
            counts=out[out.eligible].groupby(out.date.dt.strftime('%Y%m%d')).symbol.nunique()
            missing=[day for day in expected_dates('crypto',*window()) if counts.get(day,0)<20]
        info=dict(learning_input_version=3,ready=not missing,window=list(window()),symbols=int(out.symbol.nunique()),start=str(out.date.min().date()),end=str(out.date.max().date()),
                  daily_repair=repair_info,source_files=list(sources.values()),duplicate_minutes=duplicate_minutes,rejected_rows=rejected_rows,cache_hits=cache_hits,source_priority='history then live; stable file and row order within source',volume_unit='quote USD not base units', missing_dates=missing,
                  limitations=['현 보유 아카이브 유니버스의 수집 선택 편향', '잘못된 분봉은 제외하고 보간하지 않음; UTC 일봉의 유효 분봉 1200분 미만은 신호 제외', '뉴스·공시 시점 정렬 미지원'],
                  message='최근 36개월 크립토 학습 입력 준비 완료' if not missing else f'크립토 입력 부족: {len(missing)}일')
        if not missing:
            temp=target.with_suffix('.tmp');out.to_parquet(temp,index=False)
            from .model_recipe import file_hash
            info['snapshot_hash']=file_hash(temp);temp.replace(target)
        audit.write_text(json.dumps(info,ensure_ascii=False,indent=2))
        if missing: raise ValueError(info['message'])
        return out


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--market',choices=['kr','us','crypto','timefolio'],required=True)
    args=parser.parse_args();prepare(args.market);print(json.dumps(status(args.market),ensure_ascii=False))

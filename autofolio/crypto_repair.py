"""Verified exchange daily candles for globally incomplete crypto days.

Collector minute files remain untouched. Exact symbol/venue and neighboring
complete-day OHLCV agreement are required; fetched candles are never interpolated.
"""
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from .config import RUNS


def repair(out,missing,source):
    import pandas as pd
    import numpy as np
    import pyarrow.parquet as pq
    fixes=[];receipts=[];rejected=0
    for day in missing:
        date=pd.Timestamp(day);start=date-pd.Timedelta(days=1);stop=date+pd.Timedelta(days=2)
        folder=RUNS/'research_inputs/crypto_daily_repair'/day;folder.mkdir(parents=True,exist_ok=True)
        groups=list(out.groupby('symbol',sort=True))
        for index,(symbol,rows) in enumerate(groups,1):
            rows=rows.set_index('date');current=rows.loc[date] if date in rows.index else None
            if current is not None and bool(current.eligible):continue
            neighbors=[d for d in [start,date+pd.Timedelta(days=1)] if d in rows.index and rows.loc[d,'eligible']]
            if not neighbors:continue
            receipt=folder/(hashlib.sha256(str(symbol).encode()).hexdigest()+'.json')
            try:
                if receipt.exists():record=json.loads(receipt.read_text());candles=record['candles']
                else:
                    file=source/'history'/('base='+symbol)/('part-'+date.strftime('%Y-%m')+'.parquet')
                    if not file.exists():continue
                    venues=pq.ParquetFile(file).read(columns=['venue','symbol'],use_threads=False).to_pandas().drop_duplicates()
                    if len(venues)!=1 or venues.iloc[0].venue!='binance':continue
                    native=str(venues.iloc[0].symbol)
                    if not native.isascii() or not native.isalnum():continue
                    params=dict(symbol=native,interval='1d',startTime=start.value//10**6,endTime=stop.value//10**6-1,limit=3)
                    url='https://fapi.binance.com/fapi/v1/klines?'+urllib.parse.urlencode(params)
                    with urllib.request.urlopen(url,timeout=10) as response:candles=json.loads(response.read(100000))
                    record=dict(symbol=symbol,venue='binance',native_symbol=native,url=url,fetched_at=time.time(),candles=candles)
                    time.sleep(.05)
                observed={pd.Timestamp(int(c[0]),unit='ms'):dict(open=float(c[1]),high=float(c[2]),low=float(c[3]),close=float(c[4]),volume=float(c[7])) for c in candles}
                if date not in observed:continue
                for neighbor in neighbors:
                    actual=observed[neighbor]
                    for field in ['open','high','low','close','volume']:
                        reference=float(rows.loc[neighbor,field]);tol=1e-5 if field=='volume' else 1e-7
                        if not np.isclose(actual[field],reference,rtol=tol,atol=1e-8):raise ValueError('neighbor basis mismatch')
                values=observed[date]
                from .research_input import validate_frame
                validate_frame(pd.DataFrame([dict(date=date,**values)]))
                if values['volume']<=0:continue
                fix=dict(date=date,symbol=symbol,sector='CRYPTO',eligible=True,tradable_buy=True,tradable_sell=True,**values)
                record['verification']='same Binance futures instrument; neighboring complete-day OHLCV matched'
                receipt.write_text(json.dumps(record,ensure_ascii=False));receipts.append(str(receipt));fixes.append(fix)
            except Exception:rejected+=1
            if index%50==0:print(json.dumps(dict(phase='crypto_daily_repair',date=day,checked=index,total=len(groups),verified=len(fixes),rejected=rejected)),flush=True)
    if fixes:
        extra=pd.DataFrame(fixes);keys=pd.MultiIndex.from_frame(extra[['symbol','date']])
        out=out[~pd.MultiIndex.from_frame(out[['symbol','date']]).isin(keys)]
        out=pd.concat([out,extra],ignore_index=True).sort_values(['symbol','date']).reset_index(drop=True)
    return out,dict(verified_daily_candles=len(fixes),receipt_paths=receipts,rejected=rejected,method='native exchange daily OHLCV; adjacent complete days verified; original minute files unchanged')

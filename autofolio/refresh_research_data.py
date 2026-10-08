"""Acquire a consistent US daily basis, then prepare one bounded stock snapshot."""
import argparse
import json
import os
import sys
from pathlib import Path
from .config import HOME,RUNS
from .period import window


def refresh(market):
    from .research_input import prepare,status
    if status(market)['ready']:return status(market)
    if market=='us':
        import pandas as pd
        sys.path.insert(0,str(HOME/'projects/lib'))
        import arcmarket
        start,end=window();warmup=str(int(start[:4])-1)+start[4:]
        folder=RUNS/'research_inputs/us_extended_daily';folder.mkdir(parents=True,exist_ok=True)
        originals=sorted((HOME/'vault/CryptoBars/data/USA/daily_policy').glob('*.parquet'))
        fetched=[];failed=[]
        for source in originals:
            destination=folder/source.name
            if destination.exists():
                cached=pd.read_parquet(destination)
                if cached.index.min()<=pd.Timestamp(warmup)+pd.Timedelta(days=7) and cached.index.max()>=pd.Timestamp(end):continue
            frame=arcmarket.us_daily(source.stem,start=warmup,end=end,adjusted=True)
            if frame is None or frame.empty:failed.append(source.stem);continue
            temporary=destination.with_suffix('.parquet.tmp');frame.to_parquet(temporary);os.replace(temporary,destination)
            fetched.append(source.stem)
            print(json.dumps(dict(market=market,symbol=source.stem,rows=len(frame)),ensure_ascii=False),flush=True)
        (folder/'source.json').write_text(json.dumps(dict(provider='arcmarket/yfinance adjusted daily',window=[start,end],warmup=warmup,fetched=fetched,failed=failed),indent=2))
    prepare(market)
    return status(market)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--market',choices=['kr','us'],required=True)
    print(json.dumps(refresh(parser.parse_args().market),ensure_ascii=False),flush=True)

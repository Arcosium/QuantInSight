"""Refresh stock inputs independently of the retired web scheduler."""
import argparse
import json
import os
import sys
from pathlib import Path
from .config import HOME,RUNS
from .period import window,using_window


def fetch_kr(code, destination):
    # Reuse the same provider/basis without touching the shared collector files.
    import importlib.util
    from .config import ARC
    sys.path.insert(0,str(HOME/'projects/lib'))
    spec=importlib.util.spec_from_file_location('_quantinsight_daily',ARC/'quant/krx_cnn_data.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.fetch(code,destination=destination)


def refresh_kr():
    import pandas as pd
    import tempfile
    from .research_input import ROOTS,validate_frame
    from .period import expected_dates,last_complete_date
    start,end=window()
    expected=expected_dates('kr',start,min(end,last_complete_date('kr')))
    folder=RUNS/'research_inputs/kr_extended_daily';folder.mkdir(parents=True,exist_ok=True)
    fetched=[];failed={};cached=[]
    for source in sorted(ROOTS['kr'].glob('*.parquet')):
        destination=folder/source.name
        current=destination if destination.exists() else source
        required={expected[-1]} if expected else set()
        try:
            frame=validate_frame(pd.read_parquet(current))
            existing=set(frame.date.dt.strftime('%Y%m%d'))
            required.update(existing.intersection(expected))
            if required.issubset(existing):
                cached.append(source.stem);continue
        except (ValueError,KeyError,OSError):pass
        # Validate before promotion; a bad response must not replace good input.
        with tempfile.NamedTemporaryFile(dir=folder,suffix='.parquet.tmp',delete=False) as temp:
            try:
                fetch_kr(source.stem,Path(temp.name))
                frame=validate_frame(pd.read_parquet(temp.name))
                dates=set(frame.date.dt.strftime('%Y%m%d'))
                # A recent listing need not have existed three years ago. The
                # aggregate panel gate checks full-window breadth separately.
                if not required.issubset(dates):raise ValueError('incomplete_provider_history')
                os.replace(temp.name,destination)
                fetched.append(source.stem)
            except Exception as exc:
                failed[source.stem]=type(exc).__name__+': '+str(exc)[:120]
            finally:
                Path(temp.name).unlink(missing_ok=True)
        print(json.dumps(dict(market='kr',completed=len(fetched)+len(failed)+len(cached),fetched=len(fetched),failed=len(failed))),flush=True)
    receipt=dict(provider='NAVER daily chart',window=[start,end],fetched=fetched,cached=cached,failed=failed)
    temporary=folder/'source.json.tmp';temporary.write_text(json.dumps(receipt,ensure_ascii=False,indent=2));os.replace(temporary,folder/'source.json')


def refresh(market):
    import fcntl
    folder=RUNS/'research_inputs';folder.mkdir(parents=True,exist_ok=True)
    with (folder/f'.{market}-refresh.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with using_window(*window()):return _refresh(market)


def _refresh(market):
    from .research_input import prepare,status
    if status(market)['ready']:return status(market)
    if market=='kr':refresh_kr()
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

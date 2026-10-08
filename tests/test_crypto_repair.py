import json
import pandas as pd
from autofolio import crypto_repair


def test_exchange_daily_repair_requires_neighbor_agreement(tmp_path,monkeypatch):
    monkeypatch.setattr(crypto_repair,'RUNS',tmp_path)
    source=tmp_path/'source/history/base=BTC';source.mkdir(parents=True)
    pd.DataFrame({'venue':['binance'],'symbol':['BTCUSDT']}).to_parquet(source/'part-2026-09.parquet',index=False)
    days=pd.date_range('2026-09-01',periods=3)
    frame=pd.DataFrame([dict(date=d,symbol='BTC',sector='CRYPTO',open=100.,high=110.,low=90.,close=105.,volume=1000.,eligible=i!=1,tradable_buy=i!=1,tradable_sell=i!=1) for i,d in enumerate(days)])
    candles=[[int(d.value//10**6),100,110,90,105,1,0,1000] for d in days]
    class Response:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def read(self,*a):return json.dumps(candles).encode()
    monkeypatch.setattr(crypto_repair.urllib.request,'urlopen',lambda *a,**k:Response())
    monkeypatch.setattr(crypto_repair.time,'sleep',lambda _:None)
    out,info=crypto_repair.repair(frame,['20260902'],tmp_path/'source')
    assert info['verified_daily_candles']==1 and out.eligible.all()
    # Cached raw receipts still undergo basis checks, never unconditional trust.
    altered=frame.copy();altered.loc[0,'close']=106
    out,info=crypto_repair.repair(altered,['20260902'],tmp_path/'source')
    assert info['verified_daily_candles']==0 and not out.iloc[1].eligible

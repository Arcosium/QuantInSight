import pandas as pd
from autofolio import paper,config,research_input


def test_paper_reads_refreshed_daily_tail_not_retired_collector(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'RUNS',tmp_path)
    stale=tmp_path/'old';stale.mkdir()
    monkeypatch.setattr(research_input,'ROOTS',{'kr':stale})
    base=pd.DataFrame(dict(date=pd.to_datetime(['2026-10-07']),symbol='005930',sector='UNKNOWN',eligible=True,tradable_buy=True,tradable_sell=True,open=100.,high=101.,low=99.,close=100.,volume=100.))
    cols=['date','open','high','low','close','volume']
    base[cols].to_parquet(stale/'005930.parquet',index=False)
    fresh=pd.concat([base,base.assign(date=pd.Timestamp('2026-10-08'))])
    folder=tmp_path/'research_inputs/kr_extended_daily';folder.mkdir(parents=True)
    fresh[cols].to_parquet(folder/'005930.parquet',index=False)
    result=paper.stock_prices(base,'kr')
    assert result.date.max()==pd.Timestamp('2026-10-08')
    assert len(result)==2
    fresh['close']=200.
    fresh['high']=201.
    fresh[cols].to_parquet(folder/'005930.parquet',index=False)
    assert len(paper.stock_prices(base,'kr'))==1  # reject incompatible basis

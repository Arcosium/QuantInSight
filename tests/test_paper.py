import json
import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import numpy as np
from autofolio import paper,store,research,learning,model_recipe,research_input

class Model:
    def predict(self,values):return np.ones(len(values))


def test_paper_never_replays_pre_activation_fills(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DATA',tmp_path);monkeypatch.setattr(store,'DB',tmp_path/'state.sqlite')
    store.initialize();research.initialize();paper.initialize()
    folder=tmp_path/'deploy';folder.mkdir();artifact=folder/'model.joblib';artifact.write_bytes(b'only-test')
    root=tmp_path/'quotes';root.mkdir();monkeypatch.setattr(research_input,'ROOTS',{'kr':root})
    today=pd.Timestamp(datetime.datetime.now(ZoneInfo('Asia/Seoul')).date())
    days=pd.bdate_range(end=today-pd.Timedelta(days=5),periods=100)
    raw=pd.DataFrame(dict(date=days,symbol='005930',sector='x',eligible=True,tradable_buy=True,tradable_sell=True,open=100.,high=101.,low=99.,close=100.,volume=1e7))
    data=tmp_path/'snapshot.parquet';raw.to_parquet(data,index=False)
    raw[['date','open','high','low','close','volume']].to_parquet(root/'005930.parquet',index=False)
    recipe=dict(code=model_recipe.code_hashes(),definition={k:v[0] for k,v in learning.domains('kr').items()},input={'path':str(data)})
    (folder/'recipe.json').write_text(json.dumps(recipe));(folder/'deployment.json').write_text(json.dumps({'model_sha256':model_recipe.file_hash(artifact)}))
    dep=dict(id='d',user_id=1,target='kr-paper',artifact=str(artifact))
    with patch('autofolio.model_recipe.verify',return_value=True),patch('joblib.load',return_value={'model':Model()}):
        paper.advance(dep)
        with store.connect() as db:book=json.loads(db.execute('SELECT body FROM paper_books').fetchone()[0])
        assert book['pending'] and book['trades']==[]
        # A delayed historical quote on activation day must not backfill an order.
        new=raw.iloc[-1:].copy();new['date']=today-pd.Timedelta(days=2)
        pd.concat([raw,new])[['date','open','high','low','close','volume']].to_parquet(root/'005930.parquet',index=False)
        paper.advance(dep)
        with store.connect() as db:book=json.loads(db.execute('SELECT body FROM paper_books').fetchone()[0])
        assert book['trades']==[] and book['cash']==book['initial']

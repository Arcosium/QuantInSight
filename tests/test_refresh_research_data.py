import json
import pandas as pd
from autofolio import refresh_research_data as refresh, research_input


def frame(dates):
    return pd.DataFrame(dict(date=pd.to_datetime(dates),open=10.,high=11.,low=9.,close=10.,volume=100.))


def setup(monkeypatch,tmp_path):
    source=tmp_path/'source';source.mkdir()
    monkeypatch.setattr(research_input,'ROOTS',{'kr':source})
    monkeypatch.setattr(refresh,'RUNS',tmp_path/'runs')
    from autofolio import period
    monkeypatch.setattr(period,'expected_dates',lambda *args:('20261007','20261008'))
    monkeypatch.setattr(period,'last_complete_date',lambda *args:'20261008')
    return source,tmp_path/'runs/research_inputs/kr_extended_daily'


def test_missing_day_fetched_without_modifying_collector(monkeypatch,tmp_path):
    source,dest=setup(monkeypatch,tmp_path)
    original=source/'005930.parquet';frame(['2026-10-07']).to_parquet(original)
    before=original.read_bytes();calls=[]
    def fetch(code,path):
        calls.append(code);frame(['2026-10-07','2026-10-08']).to_parquet(path)
    monkeypatch.setattr(refresh,'fetch_kr',fetch)
    refresh.refresh_kr()
    assert original.read_bytes()==before
    assert len(pd.read_parquet(dest/original.name))==2
    refresh.refresh_kr()
    assert calls==['005930']  # closed day, not calendar end, is the watermark


def test_bad_provider_response_preserves_previous_file(monkeypatch,tmp_path):
    source,dest=setup(monkeypatch,tmp_path);dest.mkdir(parents=True)
    name='005930.parquet';frame(['2026-10-07']).to_parquet(source/name)
    frame(['2026-10-07']).to_parquet(dest/name);before=(dest/name).read_bytes()
    monkeypatch.setattr(refresh,'fetch_kr',lambda code,path:frame(['2026-10-07']).to_parquet(path))
    refresh.refresh_kr()
    assert (dest/name).read_bytes()==before
    assert '005930' in json.loads((dest/'source.json').read_text())['failed']
    assert not list(dest.glob('*.tmp'))


def test_recent_listing_can_refresh_without_inventing_prelisting_prices(monkeypatch,tmp_path):
    source,dest=setup(monkeypatch,tmp_path)
    from autofolio import period
    monkeypatch.setattr(period,'expected_dates',lambda *args:('20261006','20261007','20261008'))
    frame(['2026-10-07']).to_parquet(source/'NEW.parquet')
    monkeypatch.setattr(refresh,'fetch_kr',lambda code,path:frame(['2026-10-07','2026-10-08']).to_parquet(path))
    refresh.refresh_kr()
    assert pd.read_parquet(dest/'NEW.parquet').date.min()==pd.Timestamp('2026-10-07')


def test_kr_refresh_precedes_prepare_and_ready_is_noop(monkeypatch,tmp_path):
    monkeypatch.setattr(refresh,'RUNS',tmp_path);calls=[]
    monkeypatch.setattr(research_input,'status',lambda market:dict(ready=False))
    monkeypatch.setattr(refresh,'refresh_kr',lambda:calls.append('fetch'))
    monkeypatch.setattr(research_input,'prepare',lambda market:calls.append('prepare'))
    refresh.refresh('kr');assert calls==['fetch','prepare']
    monkeypatch.setattr(research_input,'status',lambda market:dict(ready=True))
    calls.clear();refresh.refresh('kr');assert not calls


def test_candidate_view_reports_missing_input_without_rewriting_running_trial(monkeypatch,tmp_path):
    from autofolio import store,research
    monkeypatch.setattr(store,'DATA',tmp_path)
    monkeypatch.setattr(store,'DB',tmp_path/'state.sqlite3')
    research.initialize()
    with store.connect() as db:
        for identity,state in [('a'*20,'queued'),('b'*20,'running')]:
            db.execute("INSERT INTO alpha_candidates(id,user_id,market,title,definition,provider,created,status) VALUES (?,1,'timefolio','test','{}','local',1,?)",(identity,state))
    monkeypatch.setattr(research,'protocol',lambda market:dict(ready=False,message='missing 20261008'))
    records={r['id']:r for r in research.candidates(1,'timefolio')}
    assert records['a'*20]['status']=='waiting_data'
    assert records['a'*20]['message']=='missing 20261008'
    assert records['b'*20]['status']=='running'
    with store.connect() as db:
        assert db.execute('SELECT status FROM alpha_candidates WHERE id=?',('a'*20,)).fetchone()[0]=='queued'

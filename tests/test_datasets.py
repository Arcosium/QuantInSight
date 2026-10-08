import sqlite3
from unittest.mock import patch
from autofolio import datasets as module


def test_compound_index_period(tmp_path):
    path=tmp_path/'bars.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE bars(code TEXT,ts TEXT,PRIMARY KEY(code,ts))')
        db.executemany('INSERT INTO bars VALUES (?,?)', [('A','2025-02-01'),('A','2025-09-07'),('B','2023-03-04'),('B','2026-10-08')])
    assert module.sqlite_period(path,'bars','ts','code')==('2023-03-04','2026-10-08')
    with sqlite3.connect(path) as db: assert db.execute('SELECT count(*) FROM bars').fetchone()[0]==4


def test_union_and_unknown_source(tmp_path):
    assert module.union([(None,None),('2025-01-01','2025-02-01'),('2024-01-01','2026-01-01')])==('2024-01-01','2026-01-01')
    path=tmp_path/'unknown';path.touch()
    row=module.source('자료',path,lambda p:(None,None))
    assert row['available'] and row['start'] is None and row['period_status']=='확인 필요'
    assert str(tmp_path) not in str(row)


def test_cache(monkeypatch):
    monkeypatch.setattr(module,'_CACHE',None)
    with patch.object(module,'inventory',return_value={'datasets':[]}) as build:
        assert module.datasets()==module.datasets()
        assert build.call_count==1
        monkeypatch.setattr(module,'_CACHE_AT',-10000)
        module.datasets()
        assert build.call_count==2


def test_parquet_boundary_metadata(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    p=tmp_path/'symbol';p.mkdir()
    for month,dates in [('2023-01',[1672704000000]),('2026-10',[1791331200000])]:
        pq.write_table(pa.table({'ts':dates}),p/f'part-{month}.parquet')
    assert module.parquet_period(tmp_path,True)==('2023-01-03','2026-10-07')


def test_dates():
    assert module.date_value(20231004)=='2023-10-04'
    assert module.date_value('2026-10-08T12:00:00')=='2026-10-08'
    assert module.date_value('202609160800')=='2026-09-16'
    assert module.date_value('20250908090000')=='2025-09-08'
    assert module.date_value('bad') is None

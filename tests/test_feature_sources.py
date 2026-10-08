import sqlite3
import pandas as pd
import pytest
from autofolio import feature_sources,config


def test_news_uses_later_publication_or_collection_and_freezes(tmp_path,monkeypatch):
    path=tmp_path/'news.sqlite';db=sqlite3.connect(path)
    db.execute('CREATE TABLE articles(title,published_at,collected_at)')
    db.executemany('INSERT INTO articles VALUES(?,?,?)',[
        ('미국 증시','2020-01-01T00:00:00+00:00','2020-01-01T00:00:00+00:00'),
        ('미국 지표','2023-01-03T10:00:00+00:00','2023-01-08T10:00:00+00:00'),
        ('미국 발표','2023-01-09T10:00:00+00:00','2023-01-08T10:00:00+00:00'),
        ('비트코인 소식','2023-01-08T10:00:00+00:00','2023-01-08T10:00:00+00:00'),
        ('미국 최신','2023-01-10T10:00:00+00:00','2023-01-10T10:00:00+00:00')])
    db.commit();db.close()
    monkeypatch.setattr(feature_sources,'NEWS_DB',path)
    monkeypatch.setattr(feature_sources,'window',lambda:('20230101','20230110'))
    monkeypatch.setattr(config,'RUNS',tmp_path/'runs');feature_sources._news_status.cache_clear()
    try:
        assert feature_sources.news_status('us')['ready']
        saved=feature_sources.prepare_news('us');rows=pd.read_parquet(saved).set_index('date')
        assert rows.loc['2023-01-08','news_count']==0
        assert rows.loc['2023-01-09','news_count']==1
        assert rows.loc['2023-01-10','news_count']==1
        digest=feature_sources.news_descriptor('us')['sha256']
        db=sqlite3.connect(path);db.execute('INSERT INTO articles VALUES(?,?,?)',('미국 수정','2023-01-07','2023-01-07'));db.commit();db.close()
        assert feature_sources.news_descriptor('us')['sha256']==digest
    finally:feature_sources._news_status.cache_clear()


def test_short_news_history_is_not_zero_filled(tmp_path,monkeypatch):
    path=tmp_path/'news.sqlite';db=sqlite3.connect(path)
    db.execute('CREATE TABLE articles(title,published_at,collected_at)')
    db.execute('INSERT INTO articles VALUES(?,?,?)',('미국 증시','2026-10-08','2026-10-08'));db.commit();db.close()
    monkeypatch.setattr(feature_sources,'NEWS_DB',path)
    monkeypatch.setattr(feature_sources,'window',lambda:('20231008','20261007'))
    feature_sources._news_status.cache_clear()
    try:
        assert not feature_sources.news_status('us')['ready']
        with pytest.raises(ValueError,match='이력'):feature_sources.prepare_news('us')
    finally:feature_sources._news_status.cache_clear()

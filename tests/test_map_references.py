import json
import sqlite3
from datetime import datetime, timezone

import pytest
from autofolio import map_references as m


def test_timefolio_diamond_requires_connected_current_model(monkeypatch,tmp_path):
    path=tmp_path/'store.sqlite3';monkeypatch.setattr(m.config,'DB',path)
    with sqlite3.connect(path) as db:
        db.executescript('''CREATE TABLE strategy_assignments(user_id,target,strategy_id,status);
          CREATE TABLE model_deployments(id,user_id,target,strategy_id,status);
          CREATE TABLE paper_books(user_id,market,deployment);
          CREATE TABLE timefolio_model_state(user_id,body);
          CREATE TABLE strategies(id,payload);
          INSERT INTO strategy_assignments VALUES(1,'timefolio','s','timefolio_ready');
          INSERT INTO model_deployments VALUES('d',1,'timefolio','s','ready');
          INSERT INTO strategies VALUES('s','{}');''')
        db.execute('INSERT INTO timefolio_model_state VALUES(1,?)',(json.dumps(dict(deployment='other',connected=True)),))
    assert m._assignments({'id':1},[{'id':'s'}])=={}
    with sqlite3.connect(path) as db:db.execute('UPDATE timefolio_model_state SET body=?',(json.dumps(dict(deployment='d',connected=True)),))
    assert m._assignments({'id':1},[{'id':'s'}])=={'s':['timefolio']}
    assert m._assignments({'id':2},[{'id':'s'}])=={}


def test_fold_overlap_and_gaps_remain_explicit(tmp_path):
    folds = [{'2025-01-01': 1, '2025-01-02': 1.1, '2025-01-03': 1.21},
             {'2025-01-02': 1, '2025-01-03': .9},
             {'2025-03-01': 1, '2025-03-02': 1.2},
             {'2025-05-01': 1, '2025-05-02': .8}]
    for i, daily in enumerate(folds):
        (tmp_path / f'ens_heatf10_direct_4h_s{i}_cohort_H84.json').write_text(json.dumps({'daily': daily}))
    values = m._fold_returns(tmp_path, 'cnn_equal')
    assert list(values) == ['20250102', '20250103', '20250302', '20250502']
    assert values['20250103'] == pytest.approx(-.1)
    assert m.measure(values)['months'] == 3


def test_signal_nav_uses_final_mark_and_never_invents_dates(tmp_path):
    def mark(day, hour, nav):
        ts = int(datetime(2025, 1, day, hour, tzinfo=timezone.utc).timestamp() * 1000)
        return json.dumps(dict(ts=ts, equity=nav))
    path = tmp_path / 'signals.jsonl'
    path.write_text('\n'.join([mark(1, 0, 1), mark(2, 0, 1.1), mark(2, 20, 1.2), mark(4, 0, .9)]))
    values = m._signal_returns(path)
    assert values == pytest.approx({'20250102': .2, '20250104': -.25})
    assert m.measure(values)['net_return'] == pytest.approx(-.1)
    path.write_text(mark(1, 0, 1))
    with pytest.raises(ValueError):
        m._signal_returns(path)


def test_missing_references_never_get_fake_zero_coordinates(monkeypatch, tmp_path):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    m._crypto_points.cache_clear()
    points = m._crypto_points(1)
    assert len(points) == 5 and len({p['id'] for p in points}) == 5
    assert all(not p['available'] and p['net_return'] is None and p['negative_months'] is None for p in points)
    m._crypto_points.cache_clear()


def test_shared_operational_markers_do_not_leak_to_other_users(monkeypatch):
    points = [dict(id='reference-crypto-cnn', operational=True, applied_targets=[])]
    monkeypatch.setattr(m, '_crypto_points', lambda _: points)
    monkeypatch.setattr(m, '_assignments', lambda *_: {})
    monkeypatch.setattr(m, '_outside_points', lambda *_: [])
    admin = m.references('crypto', dict(id=1, role='admin'), [])
    member = m.references('crypto', dict(id=2, role='user'), [])
    assert admin['points'][0]['applied_targets'] == ['crypto-paper']
    assert member['points'][0]['applied_targets'] == []
    assert points[0]['applied_targets'] == []
    assert m.references('kr', dict(id=1, role='admin'), [])['points'] == []


def test_assignment_requires_own_visible_ready_current_book(monkeypatch, tmp_path):
    path = tmp_path / 'store.sqlite3'
    monkeypatch.setattr(m.config, 'DB', path)
    with sqlite3.connect(path) as db:
        db.executescript('''CREATE TABLE strategy_assignments(user_id,target,strategy_id,status);
            CREATE TABLE model_deployments(id,user_id,target,strategy_id,status);
            CREATE TABLE paper_books(user_id,market,deployment);''')
        cases = [
            (1, 'kr-paper', 'active', 'paper_ready', 'ready', True),
            (2, 'us-paper', 'other-user', 'paper_ready', 'ready', True),
            (1, 'us-paper', 'waiting-book', 'paper_ready', 'ready', False),
            (1, 'crypto-paper', 'training', 'retraining', 'training', True),
            (1, 'crypto-paper', 'invisible', 'paper_ready', 'ready', True),
        ]
        for i, (uid, target, strategy, assigned, status, book) in enumerate(cases):
            db.execute('INSERT INTO strategy_assignments VALUES(?,?,?,?)', (uid, target, strategy, assigned))
            db.execute('INSERT INTO model_deployments VALUES(?,?,?,?,?)', (str(i), uid, target, strategy, status))
            if book:
                db.execute('INSERT INTO paper_books VALUES(?,?,?)', (uid, target.removesuffix('-paper'), str(i)))
    visible = [dict(id=x) for x in ('active', 'other-user', 'waiting-book', 'training')]
    assert m._assignments(dict(id=1), visible) == {'active': ['kr-paper']}
    assert m._assignments(dict(id=2), visible) == {'other-user': ['us-paper']}
    assert m._assignments(dict(id=1), [dict(id='active', owner_id=2)]) == {}


def test_applied_older_strategy_stays_visible_without_becoming_rankable(monkeypatch, tmp_path):
    path = tmp_path / 'store.sqlite3'
    monkeypatch.setattr(m.config, 'DB', path)
    with sqlite3.connect(path) as db:
        db.executescript('''CREATE TABLE strategy_assignments(user_id,target,strategy_id,status);
            CREATE TABLE model_deployments(id,user_id,target,strategy_id,status);
            CREATE TABLE paper_books(user_id,market,deployment);
            CREATE TABLE strategies(id,payload);
            INSERT INTO strategy_assignments VALUES(1,'kr-paper','older','paper_ready');
            INSERT INTO model_deployments VALUES('deployment',1,'kr-paper','older','ready');
            INSERT INTO paper_books VALUES(1,'kr','deployment');''')
        row = dict(title='older strategy', owner_id=1, market='kr', net_return=.1,
                   negative_months=3, months=36, start='20230101', end='20251231', sessions=750)
        db.execute('INSERT INTO strategies VALUES(?,?)', ('older', json.dumps(row)))
    points = m._outside_points('kr', dict(id=1), [])
    assert len(points) == 1 and points[0]['available'] and points[0]['reference']
    assert points[0]['applied_targets'] == ['kr-paper']
    assert points[0]['start'] == '20230101'
    assert m._outside_points('kr', dict(id=1), [dict(id='older')]) == []
    assert m._outside_points('us', dict(id=1), []) == []
    assert m._outside_points('kr', dict(id=2), []) == []
    row['owner_id'] = 2
    with sqlite3.connect(path) as db:
        db.execute('UPDATE strategies SET payload=?', (json.dumps(row),))
    assert m._outside_points('kr', dict(id=1), []) == []

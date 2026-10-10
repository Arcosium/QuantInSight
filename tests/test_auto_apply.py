import json
from contextlib import contextmanager
import pytest
from fastapi import HTTPException
from autofolio import auto_apply as a, deployment, store, research, auth


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(store, 'DATA', tmp_path)
    monkeypatch.setattr(store, 'DB', tmp_path/'state.sqlite3')
    monkeypatch.setenv('QUANTINSIGHT_AUTH_DIR', str(tmp_path/'auth'))
    store.initialize();research.initialize();deployment.initialize();a.initialize()
    with auth.connect() as db:
        db.execute("INSERT INTO users VALUES(1,'owner','unused','admin',0)")
        db.execute("INSERT INTO users VALUES(2,'member','unused','user',0)")
    monkeypatch.setattr(a, 'accepts', lambda r: r.get('months') == 36)
    def add(identity, ret, loss, owner=1, market='kr', months=36):
        payload=dict(id=identity,title=identity,owner_id=owner,market=market,months=months,
                     net_return=ret,negative_months=loss,cohort='same',genome={'engine':'learned_v1'})
        from autofolio.evaluation import PROTOCOL
        payload.update(evaluation_protocol=PROTOCOL,performance={'os':dict(months=9,net_return=ret,negative_months=loss,mdd=-.1)})
        source=tmp_path/(identity+'.json');source.write_text(json.dumps({'recipe':{'recipe_id':identity}}))
        with store.connect() as db:
            db.execute('INSERT INTO strategies VALUES(?,?,?,?,?,?,?)',(identity,identity,'ridge','same',json.dumps(payload),str(source),1))
        return payload
    return add


def test_default_off_and_timefolio_live_never_enabled(setup):
    assert not a.state(1,'kr')['enabled']
    with pytest.raises(HTTPException):a.configure({'id':1},'timefolio',True)
    with pytest.raises(HTTPException):a.configure({'id':1},'kis-live',True)
    assert a.state(1,'kr')['target']=='kr-paper'


def test_only_new_own_current_pareto_is_queued_and_old_assignment_kept(setup):
    setup('old',.1,10)
    with store.connect() as db:db.execute("INSERT INTO strategy_assignments VALUES(1,'kr-paper','old',1,'paper_ready')")
    a.configure({'id':1},'kr',True)
    a.tick()
    with store.connect() as db:assert not db.execute('SELECT * FROM model_deployments').fetchall()
    setup('new',.2,8);setup('dominated',.1,11);setup('other-user',2,0,owner=2);setup('short',3,0,months=12);setup('other-market',4,0,market='us')
    a.tick();a.tick()
    with store.connect() as db:
        rows=db.execute('SELECT * FROM model_deployments').fetchall()
        assert len(rows)==1 and rows[0]['strategy_id']=='new' and rows[0]['target']=='kr-paper'
        assert db.execute('SELECT strategy_id FROM strategy_assignments').fetchone()[0]=='old'
        assert db.execute('SELECT revision FROM auto_apply_jobs').fetchone()[0]==1
    a.configure({'id':1},'kr',False)
    with store.connect() as db:assert db.execute('SELECT status FROM model_deployments').fetchone()[0]=='cancelled'
    with pytest.raises(ValueError):
        with a.activation(rows[0]['id']):pass


def test_disable_revision_blocks_inflight_even_after_reenable(setup):
    setup('old',.1,5);a.configure({'id':1},'kr',True)
    setup('new',.2,4);a.tick()
    with store.connect() as db:
        identity=db.execute('SELECT id FROM model_deployments').fetchone()[0]
        db.execute("UPDATE model_deployments SET status='training'")
    a.configure({'id':1},'kr',False);a.configure({'id':1},'kr',True)
    with pytest.raises(ValueError):
        with a.activation(identity):pass
    with a.activation('manual-deployment'):pass


def test_multiple_new_frontiers_choose_return_then_loss(setup):
    a.configure({'id':1},'kr',True)
    setup('low-risk',.1,0);setup('high-return',.5,10)
    a.tick()
    with store.connect() as db:assert db.execute('SELECT strategy_id FROM model_deployments').fetchone()[0]=='high-return'


def test_auto_validation_failure_keeps_old_assignment_and_removes_new_weights(setup, monkeypatch, tmp_path):
    from autofolio import learning
    monkeypatch.setattr(deployment, 'RUNS', tmp_path)
    setup('old',.1,5);a.configure({'id':1},'kr',True);setup('new',.2,4);a.tick()
    with store.connect() as db:
        identity=db.execute('SELECT id FROM model_deployments').fetchone()[0]
        db.execute("INSERT INTO strategy_assignments VALUES(1,'kr-paper','old',1,'paper_ready')")
    def retrain(source, dest):
        dest.mkdir(parents=True);p=dest/'model.joblib';p.write_bytes(b'test')
        a.configure({'id':1},'kr',False)
        return p
    monkeypatch.setattr(learning,'retrain',retrain)
    with pytest.raises(ValueError):deployment.run(identity)
    assert not (tmp_path/'deployments'/'1'/identity/'model.joblib').exists()
    with store.connect() as db:assert db.execute('SELECT strategy_id FROM strategy_assignments').fetchone()[0]=='old'


def test_verified_auto_model_replaces_assignment_only_at_activation(setup, monkeypatch, tmp_path):
    from autofolio import learning
    monkeypatch.setattr(deployment, 'RUNS', tmp_path)
    setup('old', .1, 5)
    with store.connect() as db:db.execute("INSERT INTO strategy_assignments VALUES(1,'kr-paper','old',1,'paper_ready')")
    a.configure({'id':1}, 'kr', True);setup('new', .2, 4);a.tick()
    with store.connect() as db:identity=db.execute('SELECT id FROM model_deployments').fetchone()[0]
    def retrain(source, dest):
        with store.connect() as db:assert db.execute('SELECT strategy_id FROM strategy_assignments').fetchone()[0]=='old'
        dest.mkdir(parents=True);artifact=dest/'model.joblib';artifact.write_bytes(b'checked');return artifact
    monkeypatch.setattr(learning, 'retrain', retrain)
    deployment.run(identity)
    with store.connect() as db:
        assert db.execute('SELECT strategy_id,status FROM strategy_assignments').fetchone()[:]==('new','paper_ready')
        assert db.execute('SELECT status FROM model_deployments').fetchone()[0]=='ready'


def test_automatic_policy_endpoints_require_authentication():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app=FastAPI();app.include_router(a.router)
    client=TestClient(app)
    assert client.get('/api/research/kr/auto-apply').status_code==401
    assert client.post('/api/research/kr/auto-apply',json={'enabled':True}).status_code in (401,403)


def test_timefolio_policy_requires_own_connection(setup,monkeypatch):
    monkeypatch.setattr(auth,'get_connection',lambda uid,kind: {'configured':True} if uid==1 else None)
    assert a.configure({'id':1},'timefolio',True)['enabled']
    with pytest.raises(HTTPException):a.configure({'id':2},'timefolio',True)


def test_transient_queue_failure_does_not_consume_candidate(setup,monkeypatch):
    a.configure({'id':1},'kr',True);setup('new',.2,4)
    original=deployment.request_retrain
    def unavailable(*args,**kwargs):raise ValueError('temporary unavailable')
    monkeypatch.setattr(deployment,'request_retrain',unavailable);a.tick()
    with store.connect() as db:
        assert 'new' not in json.loads(db.execute('SELECT seen FROM auto_apply_policies').fetchone()[0])
    monkeypatch.setattr(deployment,'request_retrain',original);a.tick()
    with store.connect() as db:assert db.execute('SELECT strategy_id FROM model_deployments').fetchone()[0]=='new'

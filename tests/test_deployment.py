import json
from unittest.mock import patch
import pytest
from autofolio import deployment,store,research

@pytest.fixture
def db(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DATA',tmp_path);monkeypatch.setattr(store,'DB',tmp_path/'state.sqlite')
    store.initialize();research.initialize();deployment.initialize()
    source=tmp_path/'review.json';source.write_text(json.dumps(dict(recipe=dict(recipe_id='sealed'))))
    with store.connect() as con:con.execute('INSERT INTO strategies VALUES(?,?,?,?,?,?,?)',('s','title','model','cohort','{}',str(source),1))
    return source


def test_no_live_target_and_no_unverified_contest(db):
    with pytest.raises(ValueError):deployment.request_retrain(1,'s','kis-live')
    with pytest.raises(ValueError):deployment.request_retrain(1,'s','timefolio')


def test_durable_retraining_queue_deduplicates_and_separates_users(db):
    first=deployment.request_retrain(1,'s','kr-paper')
    assert first==deployment.request_retrain(1,'s','kr-paper')
    assert first!=deployment.request_retrain(2,'s','kr-paper')
    with store.connect() as con:
        assert con.execute('SELECT count(*) FROM model_deployments').fetchone()[0]==2
        assert con.execute('SELECT status FROM strategy_assignments WHERE user_id=1').fetchone()[0]=='retraining'


def test_missing_recipe_cannot_be_applied(db):
    db.write_text('{}')
    with pytest.raises(ValueError):deployment.request_retrain(1,'s','us-paper')


def test_timefolio_retraining_separate_from_historical_certification(db,monkeypatch):
    from autofolio import auth,contest_validation
    monkeypatch.setattr(auth,'get_connection',lambda *_:{'configured':True})
    assessment=dict(status='missing',rule_profile={'contest':13},competition_compliance_verified=False,summary='past evidence missing')
    monkeypatch.setattr(contest_validation,'report_assessment',lambda _:assessment)
    assert deployment.request_retrain(1,'s','timefolio')
    assessment['status']='failed'
    with pytest.raises(ValueError,match='적용 불가'):deployment.request_retrain(1,'s','timefolio')

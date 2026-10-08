import json
import pytest
from unittest.mock import patch
from autofolio import labs,research,store,learning

@pytest.fixture
def db(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DATA',tmp_path);monkeypatch.setattr(store,'DB',tmp_path/'state.sqlite')
    store.initialize();research.initialize()
    monkeypatch.setattr(research,'protocol',lambda m:dict(ready=True,message='ready'))
    store.set_setting('labs_v2_enabled',True)
    for market in labs.MARKETS:labs.control(market,True)


def test_all_four_labs_train_and_continue_after_local_failure(db):
    with patch('autofolio.providers.generate',side_effect=RuntimeError('offline')):
        for market in labs.MARKETS:
            labs.plan(1,market);labs.plan(1,market)
    with store.connect() as database:
        rows=database.execute('SELECT * FROM alpha_candidates').fetchall()
        lineage=database.execute('SELECT * FROM lab_lineage').fetchall()
    assert len(rows)==16 and len(lineage)==16
    for row in rows:learning.normalize(json.loads(row['definition']),row['market'])
    assert all(s['generation']==2 for s in labs.status(1)['markets'].values())
    labs.control('crypto',False)
    with patch('autofolio.providers.generate') as ai:labs.plan(1,'crypto');ai.assert_not_called()
    assert labs.status(1)['markets']['crypto']['phase']=='stopped'


def test_real_parent_crossovers_and_fresh_reproducible_seeds():
    domain=learning.domains('kr');g={k:v[0] for k,v in domain.items()}
    pool=[dict(id='winner',genome=g)]
    a=labs.offspring(domain,pool,'kr',5);b=labs.offspring(domain,pool,'kr',6)
    assert a==labs.offspring(domain,pool,'kr',5)
    assert {x[0]['seed'] for x in a}.isdisjoint(x[0]['seed'] for x in b)
    assert all('winner' in p and op=='crossover_mutation' for _,p,op in a)


def test_local_ai_valid_configuration_is_recorded(db):
    g={k:v[0] for k,v in learning.domains('kr').items()}
    with patch('autofolio.providers.generate',return_value={'genomes':[g]}):labs.plan(1,'kr')
    with store.connect() as database:
        assert database.execute("SELECT count(*) FROM lab_lineage WHERE operator='local_ai'").fetchone()[0]==1
    assert labs.state('kr')['local_ai'] is True


def test_control_is_persistent_per_market_and_admin_only(db,monkeypatch):
    import secrets
    from fastapi.testclient import TestClient
    from autofolio import auth
    from autofolio.app import app
    monkeypatch.setenv('QUANTINSIGHT_AUTH_DIR',str(store.DATA/'auth'))
    password=secrets.token_urlsafe(24);auth.bootstrap_admin('owner',password)
    client=TestClient(app);headers={'X-Requested-With':'QuantInSight'}
    client.post('/api/auth/login',headers=headers,json={'username':'owner','password':password})
    assert client.post('/api/research/crypto/control',headers=headers,json={'enabled':False}).status_code==200
    assert not labs.enabled('crypto') and labs.enabled('kr')
    assert client.post('/api/research/crypto/control',headers=headers,json={'enabled':True}).status_code==200
    assert labs.enabled('crypto')
    client.post('/api/auth/register',headers=headers,json={'username':'member','password':password})
    client.post('/api/auth/login',headers=headers,json={'username':'member','password':password})
    assert client.post('/api/research/crypto/control',headers=headers,json={'enabled':False}).status_code==403
    assert labs.enabled('crypto')

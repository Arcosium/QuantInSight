import secrets
OWNER_PASSWORD = secrets.token_urlsafe(24)
MEMBER_PASSWORD = secrets.token_urlsafe(24)
TEST_KEY = secrets.token_urlsafe(24)

import json
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from autofolio import auth,providers,research
from autofolio.app import app
from autofolio.period import require_complete,expected_dates,window
H={'X-Requested-With':'QuantInSight'}

@pytest.fixture
def member(tmp_path,monkeypatch):
    monkeypatch.setenv('QUANTINSIGHT_AUTH_DIR',str(tmp_path/'auth'))
    auth.bootstrap_admin('owner',OWNER_PASSWORD)
    c=TestClient(app)
    c.post('/api/auth/register',headers=H,json=dict(username='member',password=MEMBER_PASSWORD))
    c.post('/api/auth/login',headers=H,json=dict(username='member',password=MEMBER_PASSWORD))
    return c

def test_members_cannot_read_owner_accounts_or_change_limits(member):
    for mode in ['kis-live','kis-paper','timefolio']:
        with patch('autofolio.controls.subprocess.run') as run:
            r=member.get('/api/accounts/'+mode)
            assert r.status_code==200 and not r.json()['connected'] and not r.json()['holdings']
            run.assert_not_called()
    assert member.post('/api/resources',headers=H,json=dict(cpu_cores=1,memory_gb=2,parallel=1)).status_code==403
    assert member.get('/api/status').status_code==403
    assert member.get('/api/logs').status_code==403

def test_member_cannot_fall_back_to_local_model(member):
    with patch('autofolio.providers.generate') as call:
        assert member.post('/api/seed',headers=H,json=dict(strategy='차트 이미지 전략',market='kr')).status_code==422
        call.assert_not_called()

@pytest.mark.parametrize('provider',['openai','anthropic','gemini','deepseek','openrouter'])
def test_provider_wire_format(provider):
    req=providers.build_request(provider,TEST_KEY,'model-v1','schema','strategy')
    assert req.full_url.startswith('https://')
    assert TEST_KEY not in req.full_url
    body=json.loads(req.data)
    assert body
    if provider=='anthropic':assert req.get_header('X-api-key')==TEST_KEY
    elif provider=='gemini':assert req.get_header('X-goog-api-key')==TEST_KEY
    else:assert req.get_header('Authorization')=='Bearer '+TEST_KEY

def test_calendar_rejects_missing_middle_and_final_session():
    days=expected_dates('us',*window())
    require_complete(days,'us')
    with pytest.raises(ValueError):require_complete(days[:-1],'us')
    with pytest.raises(ValueError):require_complete(days[:80]+days[81:],'us')

def test_scope_hides_another_member_strategy():
    assert not research.visible(dict(owner_id=2,market='kr'),dict(id=3),'kr')
    assert not research.visible(dict(owner_id=3,market='us'),dict(id=3),'kr')
    assert research.visible(dict(owner_id=3,market='kr'),dict(id=3),'kr')

def test_member_seed_uses_only_own_key(member):
    member.post('/api/auth/credentials',headers=H,json=dict(provider='deepseek',model='chosen-model',api_key=TEST_KEY))
    definition={k:v[0] for k,v in research.STOCK_DOMAINS.items()}
    with patch('autofolio.providers.generate',return_value={'genomes':[definition],'unsupported':[]}) as call,patch('autofolio.research.save_candidates',return_value=['candidate']):
        r=member.post('/api/seed',headers=H,json=dict(strategy='차트 이미지 전략',market='kr'))
        assert r.status_code==200
        assert call.call_args.args[:3]==('deepseek',TEST_KEY,'chosen-model')

def test_apply_preserves_market_and_member_scope(member):
    days=expected_dates('us',*window())
    summary=dict(owner_id=2,market='us',months=36,start=days[0],end=days[-1],sessions=len(days))
    with patch('autofolio.market_routes.strategy_case',return_value=(summary,{})):
        options=member.get('/api/strategy/test/targets').json()['targets']
        assert [o['id'] for o in options]==['us-paper']
        assert member.post('/api/strategy/test/apply',headers=H,json={'target':'kis-live'}).status_code==403
        with patch('autofolio.market_routes.connect') as db:
            r=member.post('/api/strategy/test/apply',headers=H,json={'target':'us-paper'})
            assert r.status_code==200 and r.json()['status']=='awaiting_signal'
            db.assert_called_once()
        summary['owner_id']=3
        assert member.get('/api/strategy/test/targets').status_code==404

def test_apply_rejects_incomplete_period(member):
    summary=dict(owner_id=2,market='us',months=33,start='20240102',end='20260930')
    with patch('autofolio.market_routes.strategy_case',return_value=(summary,{})),patch('autofolio.market_routes.connect') as db:
        assert member.post('/api/strategy/test/apply',headers=H,json={'target':'us-paper'}).status_code==409
        db.assert_not_called()

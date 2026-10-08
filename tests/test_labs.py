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


def expanded_domain():
    domain=learning.domains('kr')
    domain.update(model=['ridge','extra_trees','hist_gradient_boosting',*labs.NEURAL_MODELS],
                  model_size=['compact','large'],sequence_length=[20,60],epochs=[10,30],
                  batch_size=[128,256],learning_rate=[.0003,.001])
    return domain


def test_neural_round_robin_preserves_classical_and_covers_sizes(db):
    store.set_setting('neural_research_enabled',True)
    domain=expanded_domain()
    # Old classical parents have no sequence/model-size genes.
    old={k:v[0] for k,v in domain.items() if k not in ('model_size','sequence_length','epochs','batch_size','learning_rate')}
    seen=set()
    for generation in range(1,13):
        proposals=labs.offspring(domain,[dict(id='old-parent',genome=old)],'kr',generation,2)
        assert proposals[0][0]['model'] not in labs.NEURAL_MODELS
        neural=proposals[1][0]
        assert neural['model'] in labs.NEURAL_MODELS
        assert 'old-parent' in proposals[1][1]
        seen.add((neural['model'],neural['model_size']))
    assert seen=={(model,size) for model in labs.NEURAL_MODELS for size in ('compact','large')}
    store.set_setting('neural_research_enabled',False)
    assert all(g['model'] not in labs.NEURAL_MODELS for g,_,_ in labs.offspring(domain,[],'kr',1))


def test_busy_neural_slot_keeps_classical_queue_moving():
    rows=[dict(id='old-classical',definition=json.dumps({'model':'ridge'})),
          dict(id='old-neural',definition=json.dumps({'model':'lstm'})),
          dict(id='new-neural',definition=json.dumps({'model':'transformer'}))]
    assert research.pick_candidate(rows,True)['id']=='old-neural'
    assert research.pick_candidate(rows,False)['id']=='old-classical'
    assert research.pick_candidate(rows[1:],False) is None


def test_neural_memory_and_shared_deployment_guard(db):
    from autofolio import deployment
    deployment.initialize()
    store.set_setting('neural_research_enabled',True)
    store.set_setting('memory_gb',8)
    assert research.neural_slot_available(8*2**30)
    assert not research.neural_slot_available(5*2**30)
    store.set_setting('memory_gb',4)
    assert not research.neural_slot_available(8*2**30)
    store.set_setting('memory_gb',8)
    with store.connect() as database:
        database.execute("INSERT INTO strategies VALUES('neural','n','n','c',?,'source',0)",(json.dumps({'genome':{'model':'gru'}}),))
        database.execute("INSERT INTO model_deployments VALUES('dep',1,'kr-paper','neural','source','training',0,0,NULL,'')")
    assert not research.neural_slot_available(8*2**30)
    with store.connect() as database:
        database.execute("UPDATE model_deployments SET status='ready'")
        database.execute("INSERT INTO alpha_candidates VALUES('running',1,'us','n',?,'p',0,'running','',NULL,NULL)",(json.dumps({'model':'tcn'}),))
    assert not research.neural_slot_available(8*2**30)
    with store.connect() as database:database.execute("UPDATE alpha_candidates SET status='done'")
    assert research.neural_slot_available(8*2**30)


def test_progress_reads_only_bounded_candidate_file(db,tmp_path,monkeypatch):
    from autofolio import config
    monkeypatch.setattr(config,'RUNS',tmp_path)
    identity='a'*20;path=tmp_path/'market_experiments'/identity/'progress.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(epoch=3,epochs=10,model='gru',parameters=1000000,secret='no',nested={})))
    assert research.training_progress(identity)==dict(epoch=3,epochs=10,model='gru',parameters=1000000)
    assert research.training_progress('../bad') is None
    path.write_text('x'*20000)
    assert research.training_progress(identity) is None


def test_scheduler_keeps_market_rotation_while_neural_slot_busy(db,tmp_path,monkeypatch):
    import hashlib
    import shutil
    import subprocess
    from types import SimpleNamespace
    from autofolio import config,worker
    from autofolio.period import window
    monkeypatch.setattr(config,'RUNS',tmp_path)
    monkeypatch.setattr(worker,'memory_available',lambda:16*2**30)
    monkeypatch.setattr(worker,'storage_usage',lambda:0)
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=100*2**30))
    for key in ('_PROCESSES','_INPUT_PROCESSES','_PLANNERS','_DEPLOYMENTS','_PAPER','_INPUT_CHECK'):
        monkeypatch.setattr(research,key,{})
    store.set_setting('research_campaign_owner',1)
    store.set_setting('neural_research_enabled',True)
    store.set_setting('memory_gb',8)
    labs.control('crypto',False);labs.control('timefolio',False)
    started=[]
    def spawn(args,**kwargs):
        started.append(args[-1]);return SimpleNamespace(poll=lambda:None)
    monkeypatch.setattr(subprocess,'Popen',spawn)
    identities={}
    with store.connect() as database:
        database.execute("INSERT INTO alpha_candidates VALUES('running-nn',1,'crypto','n',?,'p',0,'running','',NULL,NULL)",(json.dumps({'model':'lstm'}),))
        for order,(market,model) in enumerate([('kr','gru'),('kr','ridge'),('us','ridge')]):
            definition=json.dumps({'model':model},sort_keys=True)
            identity=hashlib.sha256(f'1:{market}:{window()}:{definition}'.encode()).hexdigest()[:20]
            identities[(market,model)]=identity
            database.execute("INSERT INTO alpha_candidates VALUES(?,1,?,?,?,'p',?,'queued','',NULL,NULL)",
                             (identity,market,model,definition,order))
    research.tick(2)
    assert started==[identities[('kr','ridge')],identities[('us','ridge')]]
    assert store.setting('lab_cursor')==2
    with store.connect() as database:
        assert database.execute('SELECT status FROM alpha_candidates WHERE id=?',(identities[('kr','gru')],)).fetchone()[0]=='queued'
    # Once the lightweight queue drains, a blocked neural candidate must not
    # prevent a new classical planning round in either market.
    monkeypatch.setattr(research,'_PROCESSES',{})
    with store.connect() as database:
        database.execute("UPDATE alpha_candidates SET status='done' WHERE id IN (?,?)",(identities[('kr','ridge')],identities[('us','ridge')]))
    started.clear()
    research.tick(2)
    assert started==['kr','us']
    assert set(research._PLANNERS)=={'kr','us'}


def test_slow_neural_trial_bounds_queue_without_skipping_architectures(db,monkeypatch):
    store.set_setting('neural_research_enabled',True)
    domain=expanded_domain()
    monkeypatch.setattr(research,'domains',lambda _:domain)
    # This test isolates scheduling; the learner owns configuration validation.
    monkeypatch.setattr(research,'normalize',lambda value,market:dict(value))
    proposed={k:v[0] for k,v in domain.items()}
    proposed.update(model='transformer',epochs=30)
    seen=[]
    with patch('autofolio.providers.generate',return_value={'genomes':[proposed]}):
        for turn in range(12):
            for _ in range(3):
                labs.plan(1,'kr')
                assert labs.neural_population('kr')==(1,turn+1)
            with store.connect() as database:
                neural=[r for r in database.execute("SELECT id,definition FROM alpha_candidates WHERE status='queued'")
                        if research.is_neural(r['definition'])]
                assert len(neural)==1
                genome=json.loads(neural[0]['definition'])
                assert genome['epochs']==30  # AI refines the reserved neural trial.
                seen.append((genome['model'],genome['model_size']))
                database.execute("UPDATE alpha_candidates SET status='done' WHERE id=?",(neural[0]['id'],))
    assert seen==[(model,size) for size in ('compact','large') for model in labs.NEURAL_MODELS]
    assert labs.state('kr')['generation']==36
    assert labs.state('kr')['neural_cursor']==12


def test_parent_selection_uses_only_os_and_skips_legacy(db,monkeypatch):
    import sys
    from types import SimpleNamespace
    # Isolate the planner boundary from the evaluation module's date accounting.
    monkeypatch.setitem(sys.modules,'autofolio.evaluation',SimpleNamespace(
        selection_metrics=lambda row:(row.get('performance') or {}).get('os')))
    genome={'model':'ridge'}
    with store.connect() as database:
        for identity,os_return,ros_return in [('a',.1,10),('b',.2,-.9),('legacy',None,999)]:
            payload={'net_return':999,'negative_months':0,'mdd':0}
            if os_return is not None:
                payload['performance']={'os':dict(net_return=os_return,negative_months=1,mdd=-.1),
                                        'ros':dict(net_return=ros_return)}
            database.execute("INSERT INTO alpha_candidates VALUES(?,1,'kr','n',?,?,0,'done','',?,NULL)",
                             (identity,json.dumps(genome),labs.PROVIDER,identity))
            database.execute("INSERT INTO strategies VALUES(?,'n','f','c',?,?,0)",
                             (identity,json.dumps(payload),identity))
    parents=labs.parents(1,'kr')
    assert [p['id'] for p in parents]==['b','a']
    assert parents[0]['net_return']==.2
    assert set(parents[0])=={'id','genome','net_return','negative_months','mdd'}


def test_saved_candidate_keeps_original_window_after_daily_roll(db):
    from autofolio.period import using_window
    with using_window('20231008','20261007'):
        genome={k:v[0] for k,v in learning.domains('kr').items()}
        identity=research.save_candidates(1,'kr',[genome],labs.PROVIDER)[0]
    with store.connect() as database:
        row=dict(database.execute('SELECT * FROM alpha_candidates WHERE id=?',(identity,)).fetchone())
    with using_window('20231009','20261008'):
        assert research.candidate_window(row)==('20231008','20261007')
        assert research.candidates(1,'kr')[0]['evaluation_window']==['20231008','20261007']
        with pytest.raises(ValueError):research.candidate_window(dict(row,evaluation_window=None))
        with pytest.raises(ValueError):research.candidate_window(dict(row,evaluation_window='["20231009","20261008"]'))

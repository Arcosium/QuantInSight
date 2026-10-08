"""Market-scoped research definitions and strict 36-month evidence gates."""
import hashlib
import json
import time
from functools import lru_cache
from fastapi import HTTPException
from .config import HOME
from .period import window
from .store import connect

STUDY=HOME/'projects/2026_파경_학술제'
BASE=STUDY/'_workspace/2026-10-02_CNN学習'
SOURCES={
 'kr':BASE/'kr_current_v10_full_native_basis_and_primary_owned_events_v1/every_original_source_quote_native_pair_audit.parquet',
 'us':BASE/'us_current_v10_all_source_quote_reference_comparison_v1/every_original_current_US_quote_and_all_reference_diagnostics.parquet',
}
STOCK_DOMAINS={'representation':['momentum','reversal','trend_momentum','lowvol_momentum','candle_hog','gasf','recurrence','mtf','fourier','image_rank_ensemble'],
 'selection_count':[10,20,30], 'holding_sessions':[5,14,28], 'ridge_alpha':[1.,10.,100.],
 'mode':['rank_only','regime_gate'], 'window':[20]}
CRYPTO_DOMAINS={'representation':['heatf_cnn'], 'window':[60], 'holding_bars':[84],
 'bar_hours':[4], 'selection_fraction':[0.1], 'weighting':['equal','event']}


def initialize():
    with connect() as db:
        db.executescript('''CREATE TABLE IF NOT EXISTS alpha_candidates (
          id TEXT PRIMARY KEY,user_id INTEGER NOT NULL,market TEXT NOT NULL,title TEXT NOT NULL,
          definition TEXT NOT NULL,provider TEXT NOT NULL,created REAL NOT NULL,status TEXT NOT NULL,
          message TEXT NOT NULL DEFAULT '',result TEXT);
          CREATE TABLE IF NOT EXISTS lab_lineage (id TEXT PRIMARY KEY,market TEXT NOT NULL,generation INTEGER NOT NULL,parents TEXT NOT NULL,operator TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS strategy_assignments (
          user_id INTEGER NOT NULL,target TEXT NOT NULL,strategy_id TEXT NOT NULL,updated REAL NOT NULL,
          status TEXT NOT NULL,PRIMARY KEY(user_id,target));''')
        if 'evaluation_window' not in {r['name'] for r in db.execute('PRAGMA table_info(alpha_candidates)')}:
            import sqlite3
            try:db.execute('ALTER TABLE alpha_candidates ADD COLUMN evaluation_window TEXT')
            except sqlite3.OperationalError:
                if 'evaluation_window' not in {r['name'] for r in db.execute('PRAGMA table_info(alpha_candidates)')}:raise


def domains(market):
    if market not in ('kr','us','crypto','timefolio'):raise HTTPException(422,'시장을 확인해 주세요.')
    from .learning import domains as model_domains
    return model_domains(market)


def normalize(value,market):
    if isinstance(value,dict) and 'model' in value:
        from .learning import normalize as model_normalize
        return model_normalize(value,market)
    if market=='timefolio':
        from .genetics import normalize as old
        return old(value)
    domain=STOCK_DOMAINS if market in SOURCES else CRYPTO_DOMAINS
    if not isinstance(value,dict) or set(value)!=set(domain):raise ValueError('Invalid candidate fields')
    if any(isinstance(value[k],bool) or value[k] not in values for k,values in domain.items()):raise ValueError('Invalid candidate values')
    return value


@lru_cache(maxsize=12)
def _source_range(market,stamp):
    import pyarrow.parquet as pq
    metadata=pq.ParquetFile(SOURCES[market]).metadata
    bounds=[]
    for row in range(metadata.num_row_groups):
        group=metadata.row_group(row)
        for index in range(group.num_columns):
            column=group.column(index)
            if column.path_in_schema=='date':
                stat=column.statistics
                if not stat or not stat.has_min_max:raise ValueError('검증 시세 날짜 메타데이터 없음')
                bounds.append((stat.min,stat.max))
    return min(a for a,b in bounds).strftime('%Y%m%d'),max(b for a,b in bounds).strftime('%Y%m%d')



def protocol(market):
    from .learning_input import status as input_status
    from .labs import THEMES
    from .evaluation import periods
    start,end=window();current=input_status(market)
    if market=='crypto' and not current.get('ready'):
        from .config import RUNS
        try:
            with (RUNS/'crypto-input-refresh.log').open('rb') as log:
                log.seek(0,2);size=log.tell();log.seek(max(0,size-8192));lines=log.read().decode(errors='replace').splitlines()
            for line in reversed(lines):
                if line.startswith('{'):
                    progress=json.loads(line)
                    if progress.get('phase')=='crypto_daily_input':
                        current['message']=f"원본 분봉 정리 {progress['processed']:,}/{progress['total']:,}종목 · 유효 봉으로 36개월 확인"
                        break
        except (OSError,ValueError,KeyError):pass
    return dict(market=market,start=start,end=end,months=36,
                description=THEMES[market]+' · 가격·거래량 시계열 · IS 24 / OS 9 / ROS 3개월',
                evaluation_periods=periods(),selection_scope='os',
                source_start=current.get('source_start') or current.get('start'),
                source_end=current.get('source_end') or current.get('end'),
                ready=current.get('ready',False),message=current.get('message','학습 입력 준비'),
                model_policy='실험 가중치 삭제 · 학습 조건과 결과 보관 · 적용 시 재학습')


def save_candidates(uid,market,genomes,provider):
    from .period import using_window
    with using_window(*window()):
        return _save_candidates(uid,market,genomes,provider)


def _save_candidates(uid,market,genomes,provider):
    initialize();p=protocol(market);saved=[];span=window()
    with connect() as db:
        for value in genomes:
            g=normalize(value,market)
            body=json.dumps(g,sort_keys=True)
            identity=hashlib.sha256(f'{uid}:{market}:{span}:{body}'.encode()).hexdigest()[:20]
            title=(str(g.get('model') or g.get('representation') or g.get('family'))+' · '+str(g.get('selection_count') or g.get('top_n') or '상·하위 10%'))
            if 'model' in g:title+=f" · {g['feature_set']} · {g['holding_sessions']}일"
            elif market in SOURCES:title+=f" · {g['holding_sessions']}일 · {g['mode']} · α{g['ridge_alpha']:g}"
            if g.get('model_size'):title+=f" · {g['model_size']} · 시퀀스 {g['sequence_length']} · {g['epochs']} epochs"
            status='queued' if p['ready'] else 'waiting_data'
            message=p['message']
            if g.get('news_mode','off')!='off':
                from .feature_sources import news_status
                news=news_status(market)
                if not news['ready']:
                    status='waiting_features';message=news['message']
            added=db.execute('INSERT OR IGNORE INTO alpha_candidates(id,user_id,market,title,definition,provider,created,status,message,evaluation_window) VALUES(?,?,?,?,?,?,?,?,?,?)',
                       (identity,uid,market,title,body,provider,time.time(),status,message,json.dumps(span))).rowcount
            if added:saved.append(identity)
    return saved


def candidates(uid,market):
    initialize()
    with connect() as db:rows=db.execute('SELECT id,title,definition,provider,created,status,message,evaluation_window FROM alpha_candidates WHERE user_id=? AND market=? ORDER BY created DESC LIMIT 100',(uid,market)).fetchall()
    return [dict(r,definition=json.loads(r['definition']),evaluation_window=json.loads(r['evaluation_window']) if r['evaluation_window'] else None,progress=training_progress(r['id'])) for r in rows]


def candidate_window(row):
    """The saved ID must authenticate the exact dates used to create the recipe."""
    from .period import using_window
    span=json.loads(row['evaluation_window']) if row.get('evaluation_window') else list(window())
    if not isinstance(span,(list,tuple)) or len(span)!=2 or any(not isinstance(x,str) or len(x)!=8 for x in span):raise ValueError('Invalid candidate window')
    with using_window(*span):
        expected=hashlib.sha256(f"{row['user_id']}:{row['market']}:{tuple(span)}:{row['definition']}".encode()).hexdigest()[:20]
        if row['id']!=expected:raise ValueError('Candidate window does not match identity')
    return tuple(span)


def refresh_queued_window(row):
    """Only not-yet-started trials move to today's window; running trials stay sealed."""
    from .period import using_window
    with using_window(*window()):return _refresh_queued_window(row)


def _refresh_queued_window(row):
    if row.get('status') not in ('queued','waiting_data','waiting_features'):return row
    if candidate_window(row)==window():return row
    save_candidates(row['user_id'],row['market'],[json.loads(row['definition'])],row['provider'])
    body=json.dumps(normalize(json.loads(row['definition']),row['market']),sort_keys=True)
    identity=hashlib.sha256(f"{row['user_id']}:{row['market']}:{window()}:{body}".encode()).hexdigest()[:20]
    with connect() as db:
        db.execute("UPDATE alpha_candidates SET status='retired',message='새 평가 기간의 후보로 이동' WHERE id=? AND status IN ('queued','waiting_data','waiting_features')",(row['id'],))
        db.execute('INSERT OR IGNORE INTO lab_lineage SELECT ?,market,generation,parents,operator FROM lab_lineage WHERE id=?',(identity,row['id']))
        current=db.execute('SELECT * FROM alpha_candidates WHERE id=?',(identity,)).fetchone()
    return dict(current) if current and current['status']=='queued' else None


def training_progress(identity):
    from .config import RUNS
    if len(identity)!=20 or any(c not in '0123456789abcdef' for c in identity):return None
    path=RUNS/'market_experiments'/identity/'progress.json'
    try:
        if path.stat().st_size>16384:return None
        data=json.loads(path.read_text())
        return {k:v for k,v in data.items() if k in ('phase','model','parameters','epoch','epochs',
                'train_rows','validation_rows','elapsed_seconds','quarter') and isinstance(v,(str,int,float))}
    except (OSError,ValueError,TypeError,AttributeError):return None


def is_neural(definition):
    from .labs import NEURAL_MODELS
    try:
        value=json.loads(definition) if isinstance(definition,str) else definition
        return value.get('model') in NEURAL_MODELS
    except (ValueError,TypeError,AttributeError):return False


def deployment_neural(row):
    """A deployment uses the stored recipe, not whichever model the planner prefers."""
    from pathlib import Path
    try:
        payload=json.loads(row['payload']) if row.get('payload') else {}
        genome=payload.get('genome') or payload.get('definition')
        if genome:return is_neural(genome)
        source=Path(row['source'])
        if source.stat().st_size>8_000_000:return True  # Unknown large recipe is conservative.
        report=json.loads(source.read_text())
        return is_neural(report.get('definition') or report.get('genome') or (report.get('recipe') or {}).get('definition'))
    except (OSError,ValueError,KeyError,TypeError):return True


def neural_slot_available(memory_free):
    from .store import setting
    if not setting('neural_research_enabled',False) or int(setting('memory_gb',8))<6 or memory_free<6*2**30:return False
    with connect() as db:
        running=db.execute("SELECT definition FROM alpha_candidates WHERE status='running'").fetchall()
        deployments=db.execute("SELECT d.source,s.payload FROM model_deployments d LEFT JOIN strategies s ON s.id=d.strategy_id WHERE d.status='training'").fetchall()
    return not any(is_neural(r['definition']) for r in running) and not any(deployment_neural(dict(r)) for r in deployments)


def pick_candidate(rows,neural_available):
    """Reserve the next free neural slot for the oldest neural trial in this market."""
    neural=[r for r in rows if is_neural(r['definition'])]
    classical=[r for r in rows if not is_neural(r['definition'])]
    return (neural[0] if neural_available and neural else classical[0] if classical else None)


def visible(summary,person,market=None):
    return (summary.get('owner_id') in (None,person['id']) and (market is None or summary.get('market','timefolio')==market))

_PROCESSES={}
_INPUT_PROCESSES={}
_PLANNERS={}
_DEPLOYMENTS={}
_PAPER={}
_PAPER_AT=0
_INPUT_CHECK={}


def tick(capacity):
    """Fair four-lab scheduler; every preparation/planner/trainer shares the cap."""
    global _PAPER_AT
    import os,subprocess,sys,shutil
    from .config import ROOT,RUNS
    from .store import setting,set_setting,event
    from . import labs
    from .worker import memory_available,storage_usage
    for jobs in (_PROCESSES,_INPUT_PROCESSES,_PLANNERS,_DEPLOYMENTS,_PAPER):
        for identity,process in list(jobs.items()):
            if process.poll() is not None:
                if jobs is _PROCESSES:
                    with connect() as db:db.execute("UPDATE alpha_candidates SET status='failed',message='실험 프로세스 종료' WHERE id=? AND status='running'",(identity,))
                if jobs is _DEPLOYMENTS:
                    with connect() as db:db.execute("UPDATE model_deployments SET status='failed',message='재학습 프로세스 종료' WHERE id=? AND status='training'",(identity,))
                del jobs[identity]
    count=sum(map(len,(_PROCESSES,_INPUT_PROCESSES,_PLANNERS,_DEPLOYMENTS,_PAPER)))
    available=max(0,capacity-count)
    used=storage_usage()
    blocked=used>int(setting('storage_budget_gb',8))*2**30 or shutil.disk_usage(RUNS).free<20*2**30 or memory_available()<4*2**30
    set_setting('market_guard',dict(blocked=blocked,used_gb=round(used/2**30,3)))
    if blocked:return count
    uid=setting('research_campaign_owner')
    if uid is None:return count
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    def spawn(args,path):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as output:return subprocess.Popen([sys.executable,'-m',*args],cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
    from .deployment import initialize as deployment_init
    deployment_init()
    if available:
        with connect() as db:pending_rows=db.execute("SELECT d.*,s.payload FROM model_deployments d LEFT JOIN strategies s ON s.id=d.strategy_id WHERE d.status='queued' ORDER BY d.created LIMIT 100").fetchall()
        neural_available=neural_slot_available(memory_available())
        pending=next((r for r in pending_rows if not deployment_neural(dict(r)) or neural_available),None)
        if pending:
            identity=pending['id']
            with connect() as db:db.execute("UPDATE model_deployments SET status='training',message='동일 조건 모델 재학습 중' WHERE id=?",(identity,))
            _DEPLOYMENTS[identity]=spawn(['autofolio.deployment',identity],RUNS/'deployments'/f'{identity}.log');available-=1
    if available and not _PAPER and time.time()-_PAPER_AT>300:
        _PAPER_AT=time.time()
        with connect() as db:ready=db.execute("SELECT 1 FROM model_deployments WHERE status='ready' AND target IN ('kr-paper','us-paper','crypto-paper') LIMIT 1").fetchone()
        if ready:
            _PAPER['tick']=spawn(['autofolio.paper'],RUNS/'paper.log');available-=1
    cursor=int(setting('lab_cursor',0))%4
    for market in labs.MARKETS[cursor:]+labs.MARKETS[:cursor]:
        if not labs.enabled(market) or available<=0:continue
        if not protocol(market)['ready']:
            shared='kr' if market=='timefolio' else market
            preparing={'kr' if m=='timefolio' else m for m in _INPUT_PROCESSES}
            if shared in preparing:continue
            if market not in _INPUT_PROCESSES and time.time()-_INPUT_CHECK.get(market,0)>300:
                _INPUT_CHECK[market]=time.time()
                module='autofolio.learning_input' if market=='crypto' else 'autofolio.refresh_research_data'
                input_market='kr' if market=='timefolio' else market
                _INPUT_PROCESSES[market]=spawn([module,'--market',input_market],RUNS/f'{market}-input-refresh.log');available-=1
                event('research_input',dict(market=market,message='36개월 학습 입력 준비'))
                set_setting('lab_cursor',(labs.MARKETS.index(market)+1)%4)
            continue
        with connect() as db:
            db.execute("UPDATE alpha_candidates SET status='queued',message='36개월 평가 대기' WHERE market=? AND status='waiting_data'",(market,))
            pending_rows=db.execute("SELECT * FROM alpha_candidates WHERE status='queued' AND market=? AND json_extract(definition,'$.model') IS NOT NULL ORDER BY created LIMIT 256",(market,)).fetchall()
        waiting=pick_candidate(pending_rows,neural_slot_available(memory_available()))
        if waiting:
            row=dict(waiting);identity=row['id']
            try:
                row=refresh_queued_window(row)
                if row is None:continue
                identity=row['id'];span=candidate_window(row)
            except (ValueError,TypeError):
                with connect() as db:db.execute("UPDATE alpha_candidates SET status='retired',message='평가 기간 변경' WHERE id=?",(identity,))
                continue
            with connect() as db:db.execute("UPDATE alpha_candidates SET status='running',message='36개월 순차 모델 학습·계좌 평가' WHERE id=?",(identity,))
            try:_PROCESSES[identity]=spawn(['autofolio.market_runner',identity],RUNS/'market_experiments'/identity/'run.log')
            except OSError:
                with connect() as db:db.execute("UPDATE alpha_candidates SET status='failed',message='실험 시작 실패' WHERE id=?",(identity,))
                raise
            event('job_started',dict(job=identity,market=market,message=row['title']))
            available-=1;set_setting('lab_cursor',(labs.MARKETS.index(market)+1)%4)
        # A long neural fit must not idle the lightweight slot. The planner caps
        # outstanding neural proposals and replenishes classical trials only.
        elif market not in _PLANNERS and time.time()>labs.state(market).get('retry_after',0):
            _PLANNERS[market]=spawn(['autofolio.labs',str(uid),market],RUNS/f'{market}-planner.log')
            available-=1;set_setting('lab_cursor',(labs.MARKETS.index(market)+1)%4)
    return sum(map(len,(_PROCESSES,_INPUT_PROCESSES,_PLANNERS,_DEPLOYMENTS,_PAPER)))

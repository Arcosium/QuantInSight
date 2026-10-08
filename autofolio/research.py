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
STOCK_DOMAINS={'representation':['candle_hog','gasf','recurrence','mtf','fourier','image_rank_ensemble'],
 'selection_count':[10,20,30], 'holding_sessions':[14], 'ridge_alpha':[1.,10.,100.],
 'mode':['rank_only'], 'window':[20]}
CRYPTO_DOMAINS={'representation':['heatf_cnn'], 'window':[60], 'holding_bars':[84],
 'bar_hours':[4], 'selection_fraction':[0.1], 'weighting':['equal','event']}


def initialize():
    with connect() as db:
        db.executescript('''CREATE TABLE IF NOT EXISTS alpha_candidates (
          id TEXT PRIMARY KEY,user_id INTEGER NOT NULL,market TEXT NOT NULL,title TEXT NOT NULL,
          definition TEXT NOT NULL,provider TEXT NOT NULL,created REAL NOT NULL,status TEXT NOT NULL,
          message TEXT NOT NULL DEFAULT '',result TEXT);
          CREATE TABLE IF NOT EXISTS strategy_assignments (
          user_id INTEGER NOT NULL,target TEXT NOT NULL,strategy_id TEXT NOT NULL,updated REAL NOT NULL,
          status TEXT NOT NULL,PRIMARY KEY(user_id,target));''')


def domains(market):
    if market in ('kr','us'):return STOCK_DOMAINS
    if market=='crypto':return CRYPTO_DOMAINS
    from .genetics import DOMAINS
    if market=='timefolio':return DOMAINS
    raise HTTPException(422,'시장을 확인해 주세요.')


def normalize(value,market):
    if market=='timefolio':
        from .genetics import normalize as old
        return old(value)
    domain=domains(market)
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
    start,end=window()
    description='학술제 · 차트 이미지·CNN 비교 · 성숙한 라벨만 학습 · 다음 거래일 시가 체결'
    source_start=source_end=None
    if market in SOURCES:
        source=SOURCES[market]
        if source.exists():source_start,source_end=_source_range(market,source.stat().st_mtime_ns)
        from .period import expected_dates
        ready=bool(source_start and source_start<start and source_end>=expected_dates(market,start,end)[-1])
        message='36개월 연속 평가 준비' if ready else '학술제 검증 원본 갱신 대기 · 최근 36개월을 채운 결과만 게시'
    elif market=='crypto':
        description='HYFE_QTPA · HeatF CNN · 4시간 봉 · 14일 롱숏 코호트'
        ready=False
        message='최근 36개월 순차 학습·검증 준비 · 논문 모의매매는 기존 고정 모델 유지'
    else:
        description='타임폴리오 · 롱온리 계좌 · 비용·회전율 반영'
        ready=False;message='최근 36개월 전체 거래일 입력 갱신 대기'
    return dict(market=market,start=start,end=end,months=36,description=description,source_start=source_start,source_end=source_end,ready=ready,message=message)


def save_candidates(uid,market,genomes,provider):
    initialize();p=protocol(market);saved=[]
    with connect() as db:
        for value in genomes:
            g=normalize(value,market)
            body=json.dumps(g,sort_keys=True)
            identity=hashlib.sha256(f'{uid}:{market}:{window()}:{body}'.encode()).hexdigest()[:20]
            title=(str(g.get('representation') or g.get('family'))+' · '+str(g.get('selection_count') or g.get('top_n') or '상·하위 10%'))
            status='queued' if p['ready'] else 'waiting_data' if market in SOURCES else 'waiting_validation'
            added=db.execute('INSERT OR IGNORE INTO alpha_candidates(id,user_id,market,title,definition,provider,created,status,message) VALUES(?,?,?,?,?,?,?,?,?)',
                       (identity,uid,market,title,body,provider,time.time(),status,p['message'])).rowcount
            if added:saved.append(identity)
    return saved


def candidates(uid,market):
    initialize()
    with connect() as db:rows=db.execute('SELECT id,title,definition,provider,created,status,message FROM alpha_candidates WHERE user_id=? AND market=? ORDER BY created DESC LIMIT 100',(uid,market)).fetchall()
    return [dict(r,definition=json.loads(r['definition'])) for r in rows]


def visible(summary,person,market=None):
    return (summary.get('owner_id') in (None,person['id']) and (market is None or summary.get('market','timefolio')==market))

_PROCESSES={}
_LAST_CHECK=0

def tick(capacity):
    """Run market evaluations inside the same bounded worker cgroup."""
    global _LAST_CHECK
    import os,subprocess,sys
    from .config import ROOT,RUNS
    for identity,process in list(_PROCESSES.items()):
        if process.poll() is not None:
            with connect() as db:db.execute("UPDATE alpha_candidates SET status='failed',message='실험 프로세스 종료' WHERE id=? AND status='running'",(identity,))
            del _PROCESSES[identity]
    if time.time()-_LAST_CHECK>300:
        for market in ('kr','us'):
            if protocol(market)['ready']:
                with connect() as db:db.execute("UPDATE alpha_candidates SET status='queued',message='36개월 평가 대기' WHERE market=? AND status='waiting_data'",(market,))
        _LAST_CHECK=time.time()
    available=max(0,capacity-len(_PROCESSES))
    if available:
        with connect() as db:waiting=db.execute("SELECT id FROM alpha_candidates WHERE status='queued' AND market IN ('kr','us') ORDER BY created LIMIT ?",(available,)).fetchall()
        for row in waiting:
            identity=row['id'];directory=RUNS/'market_experiments'/identity;directory.mkdir(parents=True,exist_ok=True)
            with (directory/'run.log').open('a') as output:
                child=subprocess.Popen([sys.executable,'-m','autofolio.market_runner',identity],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,
                    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=''))
            _PROCESSES[identity]=child
            with connect() as db:db.execute("UPDATE alpha_candidates SET status='running',message='36개월 순차 학습·계좌 평가' WHERE id=?",(identity,))
    return len(_PROCESSES)

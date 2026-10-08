"""Owner-only controls, served behind the existing Cloudflare Access policy."""
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit
from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, Field
from .config import HOME, ROOT
from .store import connect, setting, set_setting, event
from .genetics import DOMAINS, normalize, fingerprint, model_fingerprint

router = APIRouter()


def mutation_guard(request):
    # All remote traffic enters via owner-only Access. Reject cross-site forms
    # and cross-origin script requests; local requests need the same header.
    if request.headers.get('x-requested-with') != 'QuantInSight':
        raise HTTPException(403, '이 화면에서 다시 요청해 주세요.')
    origin = request.headers.get('origin')
    if origin and urlsplit(origin).netloc != request.headers.get('host'):
        raise HTTPException(403, '다른 사이트의 요청은 허용하지 않습니다.')


class Resources(BaseModel):
    cpu_cores: int = Field(ge=1, le=64)
    memory_gb: int = Field(ge=2, le=128)
    parallel: int = Field(ge=1, le=8)


@router.post('/api/resources')
def resources(value: Resources, request: Request):
    mutation_guard(request)
    physical = os.cpu_count() or 1
    total = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemTotal:'))) / 2**20
    if value.cpu_cores > physical or value.memory_gb > total * .8:
        raise HTTPException(422, '호스트 CPU 수와 전체 메모리의 80% 이내로 설정해 주세요.')
    env = dict(os.environ, XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}',
               DBUS_SESSION_BUS_ADDRESS=f'unix:path=/run/user/{os.getuid()}/bus')
    command = ['systemctl', '--user', 'set-property', 'autofolio-worker.service',
               f'CPUQuota={value.cpu_cores*100}%', f'MemoryMax={value.memory_gb}G']
    try:
        subprocess.run(command, env=env, check=True, capture_output=True, timeout=15)
    except (subprocess.SubprocessError, OSError):
        raise HTTPException(503, '실제 자원 한도를 적용하지 못했습니다.')
    for key, val in [('cpu_cores', value.cpu_cores), ('memory_gb', value.memory_gb), ('concurrency', value.parallel)]:
        set_setting(key, val)
    event('resources_updated', value.model_dump())
    return value.model_dump()


@router.get('/api/logs')
def logs(after: int = Query(0, ge=0)):
    with connect() as db:
        found = db.execute('SELECT id,timestamp,kind,body FROM events WHERE id>? ORDER BY id LIMIT 300', (after,)).fetchall()
    names = {'job_started':'실험 시작', 'worker_started':'탐색 시작', 'catalogue_refresh':'결과 갱신',
             'resources_updated':'자원 한도 변경', 'seed_generated':'전략 생성', 'training':'모델 학습',
             'backtesting':'계좌 평가', 'failed':'실험 실패', 'model_cache_hit':'학습 결과 재사용',
             'validation_required':'추가 검증 대기'}
    events=[]
    labels={'generation':'세대','quarter':'학습 분기','phase':'평가 계좌','iteration':'진행','total':'전체','count':'후보','cpu_cores':'CPU','memory_gb':'메모리','parallel':'동시 실행','start':'시작일','end':'종료일','months':'개월','error':'오류','message':''}
    for row in found:
        body=json.loads(row['body'])
        values=[(labels[k]+' '+str(v)).strip() for k,v in body.items() if k in labels]
        if body.get('job') or body.get('id'):values.insert(0,str(body.get('job') or body['id'])[:8])
        events.append(dict(id=row['id'],timestamp=row['timestamp'],message=' · '.join([names.get(row['kind'],row['kind'])]+values)))
    return dict(events=events)


class Seed(BaseModel):
    strategy: str = Field(min_length=3, max_length=4000)


@router.post('/api/seed')
def seed(value: Seed, request: Request):
    mutation_guard(request)
    prompt = ('사용자가 입력한 전략을 중심으로 계좌 전략 후보 5개를 설계하세요. '
              '아래 유전자 도메인 밖 조건은 구현할 수 없으므로 unsupported에 적으세요. '
              '모든 유전자를 포함한 JSON {"genomes":[...],"unsupported":[]}만 출력하세요. '
              '평가 기간은 최근 36개월로 고정입니다. '+json.dumps(DOMAINS,ensure_ascii=False))
    payload = dict(model=os.environ.get('AUTOFOLIO_LOCAL_MODEL','arc-local'),
        messages=[dict(role='system',content=prompt),dict(role='user',content=value.strategy)],
        stream=False, max_tokens=3000, chat_template_kwargs=dict(enable_thinking=False), response_format=dict(type='json_object'))
    req = urllib.request.Request('http://127.0.0.1:11434/v1/chat/completions',
        data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=180) as response: result=json.load(response)
        content=result['choices'][0]['message']['content'].strip()
        if content.startswith('```') and content.endswith('```'):
            content=content.split('\n',1)[1].rsplit('```',1)[0].strip()
        parsed=json.loads(content)
        if parsed.get('unsupported'):
            raise HTTPException(422,'현재 탐색기가 지원하지 않는 조건: '+', '.join(map(str,parsed['unsupported'])))
        genomes=[normalize(g) for g in parsed['genomes'][:5]]
        if not genomes: raise ValueError('No genomes')
    except HTTPException: raise
    except Exception:
        raise HTTPException(503,'로컬 AI가 유효한 전략을 생성하지 못했습니다. 전략 조건을 바꿔 다시 시도해 주세요.')
    count=0
    with connect() as db:
        for g in genomes:
            count+=db.execute("INSERT OR IGNORE INTO jobs(id,genome,model_id,generation,parents,operator,status,created) VALUES (?,?,?,?,?,?,'queued',?)",
                (fingerprint(g),json.dumps(g),model_fingerprint(g,'pending'),int(setting('generation',0))+1,'[]','local_ai',time.time())).rowcount
    set_setting('seed_genomes',genomes)
    event('seed_generated',dict(count=count,provider='local'))
    return dict(count=count)


@router.get('/api/datasets')
def datasets():
    vault=HOME/'vault'
    definitions=[
        ('한국 분봉','KRX · 1분 OHLCV',vault/'CryptoBars/data/KRX/bars_ohlc.db'),
        ('한국 시세','KRX · 분봉과 거래일',vault/'CryptoBars/data/KRX/bars.db'),
        ('미국 분봉','미국 주식 · 1분 OHLCV',vault/'CryptoBars/data/USA/1m'),
        ('크립토 분봉','거래소별 1분 OHLCV',vault/'CryptoBars/data/bars'),
        ('크립토 과거 분봉','종목별 과거 1분 OHLCV',vault/'CryptoBars/data/history'),
        ('NXT 분봉','넥스트레이드 · 1분 OHLCV',vault/'CryptoBars/data/NXT/bars.db'),
        ('한국 장기 분봉','토스 · 과거 1분 OHLCV',vault/'CryptoBars/data/KRX/bars_toss.db'),
        ('미국 일봉 정책','상장 이력 · 종목 변경',vault/'CryptoBars/data/USA/daily_policy'),
        ('뉴스','증권 뉴스 · RSS',HOME/'projects/lib/data/arcnews.db'),
        ('공시·재무','DART · 기업 재무',ROOT/'integrations/timefolio/quant/financials_5y.csv'),
        ('한국 일봉','수정 주가 · 거래량',vault/'QuantInSight/data'),
        ('연구 입력','확정 계좌 패널 · 학습 입력',vault/'ArcTrade/timefolio_cnn_4y/20261002_v1/context_full_w20_h5_v1/dataset'),
        ('시장 분석','종목·수급 분석 자료',vault/'HYFE/9.28/market_data/analysis-20260928-revision-03'),
    ]
    result=[]
    for name,description,path in definitions:
        available=path.exists()
        result.append(dict(name=name,description=description,available=available,
                           updated=path.stat().st_mtime if available else None))
    return dict(datasets=result)


@router.get('/api/accounts/{mode}')
def account(mode: str):
    if mode not in ['timefolio','kis-live','kis-paper']:raise HTTPException(404)
    result=dict(connected=False,equity=[],message='계좌 연결 확인 필요',total_eval=None,cash=None,pnl_ratio=None,holdings=[],trades=[])
    if mode=='timefolio':
        try:
            with urllib.request.urlopen('http://127.0.0.1:8620/api/autofolio/summary',timeout=8) as r: source=json.load(r)
            portfolio=(source.get('account') or {}).get('portfolio') or {}
            result.update(connected=False,message='저장 장부 · 사이트 연결 미확인',total_eval=portfolio.get('total_eval'),cash=portfolio.get('cash'),pnl_ratio=(portfolio['unrealized_pnl_pct']/100 if portfolio.get('unrealized_pnl_pct') is not None else None))
            for p in portfolio.get('positions',[]):
                result['holdings'].append(dict(code=p.get('ticker') or p.get('code'),name=p.get('name'),qty=p.get('qty',0),value=p.get('value'),pnl_ratio=(p['pnl_pct']/100 if p.get('pnl_pct') is not None else None)))
            with urllib.request.urlopen('http://127.0.0.1:8620/api/autofolio/trades?limit=50',timeout=8) as r: trades=json.load(r)
            with urllib.request.urlopen('http://127.0.0.1:8620/api/autofolio/equity',timeout=8) as r: eq=json.load(r)
            base=eq.get('initial_cash')
            if base and base>0:
                result['equity']=[dict(date=p['ts_kst'][:10].replace('-',''),net_return=p['total_eval']/base-1) for p in eq.get('equity',[]) if p.get('total_eval') is not None]
                result['pnl_ratio']=(source.get('performance') or {}).get('eq_all_pct')
                if result['pnl_ratio'] is not None:result['pnl_ratio']/=100
            result['trades']=[dict(date=t.get('date') or t.get('timestamp') or t.get('ts'),code=t.get('ticker'),name=t.get('name'),side=t.get('side'),qty=t.get('qty'),price=t.get('price')) for t in trades.get('trades',[])]
        except (OSError,ValueError,TypeError):result.update(message='계좌 조회 실패',connected=False)
    else:
        import sys
        kis=ROOT/'integrations/kis'
        if not kis.exists():kis=HOME/'projects/QuantInSight'
        try:
            process=subprocess.run([sys.executable,str(ROOT/'autofolio/account_bridge.py'),str(kis),mode],
                                   cwd=kis,capture_output=True,text=True,timeout=90,check=True)
            result.update(json.loads(process.stdout))
        except (subprocess.SubprocessError,OSError,ValueError):result['message']='계좌 조회 실패'
    return result

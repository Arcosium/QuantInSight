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
    from .auth import admin_required
    admin_required(request)
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
def logs(request:Request,after: int = Query(0, ge=0)):
    from .auth import admin_required
    admin_required(request)
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
    market: str = Field(default='kr',pattern='^(kr|us|crypto|timefolio)$')


@router.post('/api/seed')
def seed(value: Seed, request: Request):
    mutation_guard(request)
    from .auth import user,get_credentials,throttle
    from . import providers,research
    person=user(request)
    throttle(request,'seed:'+str(person['id']))
    credential=get_credentials(person['id']) if person['role']!='admin' else None
    if person['role']!='admin' and not credential:raise HTTPException(422,'먼저 API Key와 모델 ID를 저장해 주세요.')
    provider=credential['provider'] if credential else 'local'
    prompt=('입력 전략을 중심으로 서로 다른 후보 최대 5개를 설계하세요. 지원 범위 밖은 unsupported에 쓰세요. '
            'JSON {"genomes":[...],"unsupported":[]}만 출력하세요. 시장 '+value.market+
            ', 최근 36개월 고정, 미래 데이터 사용 금지. 모든 필수 유전자와 허용값: '+json.dumps(research.domains(value.market),ensure_ascii=False))
    prompt+=' 반환 예시: '+json.dumps({'genomes':[{k:v[0] for k,v in research.domains(value.market).items()}],'unsupported':[]},ensure_ascii=False)+' genomes 원소는 객체이며 중첩 배열을 사용하지 마세요.'
    result=providers.generate(provider,credential['api_key'] if credential else '',credential['model'] if credential else os.environ.get('AUTOFOLIO_LOCAL_MODEL','arc-local'),prompt,value.strategy)
    try:
        if result.get('unsupported'):raise HTTPException(422,'지원하지 않는 조건: '+', '.join(map(str,result['unsupported']))[:500])
        raw=result['genomes']
        if isinstance(raw,list) and len(raw)==1 and isinstance(raw[0],list):raw=raw[0]
        genomes=[research.normalize(g,value.market) for g in raw[:5]]
        if not genomes:raise ValueError()
    except (KeyError,ValueError,TypeError):raise HTTPException(422,'AI가 지원 범위에 맞는 전략을 반환하지 않았습니다.')
    ids=research.save_candidates(person['id'],value.market,genomes,provider)
    if value.market=='timefolio' and person['role']=='admin':
        with connect() as db:
            for g in genomes:
                db.execute("INSERT OR IGNORE INTO jobs(id,genome,model_id,generation,parents,operator,status,created) VALUES (?,?,?,?,?,?,'queued',?)",
                    (fingerprint(g),json.dumps(g),model_fingerprint(g,'pending'),int(setting('generation',0))+1,'[]',provider,time.time()))
        set_setting('seed_genomes',genomes)
    return dict(count=len(ids),ids=ids,message=research.protocol(value.market)['message'])


@router.get('/api/datasets')
def datasets():
    from .datasets import datasets as catalogue
    return catalogue()


_ACCOUNT_CACHE={}

@router.get('/api/accounts/{mode}')
def account(mode: str,request:Request):
    from .auth import user,get_connection
    import sys
    person=user(request)
    if mode not in ['timefolio','kis-live','kis-paper']:raise HTTPException(404)
    result=dict(connected=False,equity=[],message='계좌 연결 정보를 등록해 주세요.',total_eval=None,cash=None,pnl_ratio=None,holdings=[],trades=[])
    credential=get_connection(person['id'],mode)
    if not credential and (person['role']!='admin' or mode=='timefolio'):return result
    import hashlib
    key=(person['id'],mode,hashlib.sha256(json.dumps(credential,sort_keys=True).encode()).hexdigest())
    cached=_ACCOUNT_CACHE.get(key)
    if cached and time.time()-cached[0]<60:return cached[1]
    try:
        if mode=='timefolio':
            command=[sys.executable,str(ROOT/'autofolio/timefolio_bridge.py')]
            payload=credential
        else:
            command=[sys.executable,str(ROOT/'autofolio/account_bridge.py'),str(ROOT),mode]
            payload=dict(user_id=person['id'],credentials=credential) if credential else None
            if payload:command.append('--member')
        process=subprocess.run(command,input=json.dumps(payload),cwd=ROOT,capture_output=True,text=True,timeout=100,check=True)
        result.update(json.loads(process.stdout))
        result['updated']=time.time()
    except (subprocess.SubprocessError,OSError,ValueError):result['message']='계좌 조회 실패'
    _ACCOUNT_CACHE[key]=(time.time(),result)
    return result

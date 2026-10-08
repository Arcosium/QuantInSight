"""Market labs, retraining requests and stopped live-trading targets."""
import json
import time
from fastapi import APIRouter,Request,HTTPException
from pydantic import BaseModel
from .auth import user,admin_required,mutation_guard,get_connection
from . import research,labs
from .store import connect,set_setting
from .period import accepts
from .catalogue import strategy_case
router=APIRouter()
TARGETS={'kr-paper':('kr','한국주식 페이퍼매매'), 'kis-live':('kr','한투 실매매 · 정지'),
         'us-paper':('us','미국주식 페이퍼매매'),'crypto-paper':('crypto','크립토 페이퍼매매'),
         'timefolio':('timefolio','타임폴리오 13회 모의매매')}

@router.get('/api/research/{market}')
def status(market:str,request:Request):
    research.domains(market);person=user(request)
    return dict(protocol=research.protocol(market),candidates=research.candidates(person['id'],market),
                campaign=labs.status(person['id']) if person['role']=='admin' else None)

class Control(BaseModel):
    enabled:bool

@router.post('/api/research/{market}/control')
def control(market:str,value:Control,request:Request):
    mutation_guard(request);person=admin_required(request);research.domains(market)
    set_setting('research_campaign_owner',person['id']);set_setting('labs_v2_enabled',True)
    labs.control(market,value.enabled)
    return dict(market=market,enabled=value.enabled,message='연속 연구 시작' if value.enabled else '중지됨 · 실행 중 실험은 완료 후 종료')

@router.get('/api/strategy/{identity}/targets')
def targets(identity:str,request:Request):
    person=user(request)
    try:summary,_=strategy_case(identity)
    except (KeyError,ValueError):raise HTTPException(404,'전략을 찾을 수 없습니다.')
    if not research.visible(summary,person):raise HTTPException(404)
    market=summary.get('market','timefolio')
    return dict(targets=[dict(id=k,label=label+(' · 규칙 검증 대기' if k=='timefolio' and not summary.get('contest_certified') else ''),available=k!='kis-live' and (k!='timefolio' or (summary.get('contest_certified',False) and bool(get_connection(person['id'],k)))))
                         for k,(m,label) in TARGETS.items() if m==market])

class Apply(BaseModel):
    target:str

@router.post('/api/strategy/{identity}/apply')
def apply(identity:str,value:Apply,request:Request):
    mutation_guard(request);person=user(request)
    if value.target=='kis-live':raise HTTPException(409,'실매매 정지 상태입니다. 전략을 적용할 수 없습니다.')
    options=targets(identity,request)['targets']
    if not any(t['id']==value.target and t['available'] for t in options):raise HTTPException(403,'이 계좌에 적용할 권한이 없습니다.')
    summary,_=strategy_case(identity)
    if not accepts(summary):raise HTTPException(409,'최근 36개월 평가를 통과한 전략만 적용할 수 있습니다.')
    from .deployment import request_retrain
    try:record=request_retrain(person['id'],identity,value.target)
    except ValueError as exc:raise HTTPException(409,str(exc))
    return dict(target=value.target,strategy_id=identity,status='retraining',message='동일 조건 재학습 요청 · 검증 후 페이퍼 운용 연결',deployment_id=record)

@router.get('/api/assignments')
def assignments(request:Request):
    with connect() as db:rows=db.execute('SELECT target,strategy_id,status,updated FROM strategy_assignments WHERE user_id=?',(user(request)['id'],)).fetchall()
    return dict(assignments=[dict(r) for r in rows])

@router.get('/api/paper/{market}')
def paper(market:str,request:Request):
    if market not in ('kr','us','crypto'):raise HTTPException(404)
    from .paper import snapshot
    return snapshot(user(request)['id'],market)

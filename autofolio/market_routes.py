"""Market research and explicit strategy-to-account assignments."""
import json
import time
from fastapi import APIRouter,Request,HTTPException
from pydantic import BaseModel,Field
from .auth import user,admin_required,mutation_guard,get_connection
from . import research
from .store import connect
from .period import accepts
from .catalogue import strategy_case
router=APIRouter()
TARGETS={'kis-live':('kr','한투 (실매매)'), 'kis-paper':('kr','한투 (모의매매)'),
         'us-paper':('us','미국주식 모의매매'),'crypto-paper':('crypto','크립토 모의매매'),
         'timefolio':('kr','타임폴리오 13회')}

@router.get('/api/research/{market}')
def status(market:str,request:Request):
    research.domains(market)
    person=user(request)
    from .campaign import status as campaign_status
    return dict(protocol=research.protocol(market),candidates=research.candidates(person['id'],market),campaign=campaign_status(person['id']) if person['role']=='admin' else None)

@router.get('/api/strategy/{identity}/targets')
def targets(identity:str,request:Request):
    person=user(request)
    try:summary,_=strategy_case(identity)
    except (KeyError,ValueError):raise HTTPException(404,'전략을 찾을 수 없습니다.')
    if not research.visible(summary,person):raise HTTPException(404)
    market=summary.get('market','timefolio')
    if market=='timefolio':market='kr'
    return dict(targets=[dict(id=k,label=label,available=person['role']=='admin' or k in ('us-paper','crypto-paper') or bool(get_connection(person['id'],k)))
                         for k,(m,label) in TARGETS.items() if m==market])

class Apply(BaseModel):
    target:str

@router.post('/api/strategy/{identity}/apply')
def apply(identity:str,value:Apply,request:Request):
    mutation_guard(request);person=user(request)
    options=targets(identity,request)['targets']
    if not any(t['id']==value.target and t['available'] for t in options):raise HTTPException(403,'이 계좌에 적용할 권한이 없습니다.')
    summary,_=strategy_case(identity)
    if not accepts(summary):raise HTTPException(409,'최근 36개월 평가를 통과한 전략만 적용할 수 있습니다.')
    # Binding is explicit. Keep the account stopped until its execution adapter
    # has verified current signals; selecting a backtest cannot fabricate fills.
    with connect() as db:
        db.execute('INSERT INTO strategy_assignments VALUES(?,?,?,?,?) ON CONFLICT(user_id,target) DO UPDATE SET strategy_id=excluded.strategy_id,updated=excluded.updated,status=excluded.status',
                   (person['id'],value.target,identity,time.time(),'awaiting_signal'))
    return dict(target=value.target,strategy_id=identity,status='awaiting_signal',message='전략 선택 저장 · 최신 운용 신호 검증 대기')

@router.get('/api/assignments')
def assignments(request:Request):
    with connect() as db:rows=db.execute('SELECT target,strategy_id,status,updated FROM strategy_assignments WHERE user_id=?',(user(request)['id'],)).fetchall()
    return dict(assignments=[dict(r) for r in rows])

@router.get('/api/paper/us')
def us_paper(request:Request):
    records=assignments(request)['assignments'];active=next((r for r in records if r['target']=='us-paper'),None)
    return dict(connected=False,message='최신 운용 신호 검증 대기' if active else '적용한 전략 없음',currency='USD',equity=[],holdings=[],trades=[],total_eval=None,cash=None,pnl_ratio=None,assignment=active)

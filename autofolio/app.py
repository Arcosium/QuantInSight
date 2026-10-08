"""Authenticated market research and user-scoped account views."""
import asyncio
import contextlib
import calendar
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from .config import ROOT, RUNS, ARC
from .store import initialize, connect, setting
from .catalogue import rows, strategy_case
from .metrics import pareto_front, ledger, statistics_for, normalized_date, evaluation_scope

initialize()
from .auth import user,admin_required,AuthMiddleware,router as auth_router
from . import research
research.initialize()
@contextlib.asynccontextmanager
async def lifespan(app):
    async def automatic_deployments():
        from .auto_apply import tick
        from .store import event
        while True:
            try:
                await asyncio.to_thread(tick)
            except Exception as exc:
                event('auto_apply_error',dict(message='자동 적용 점검 실패',error=str(exc)[:180]))
            await asyncio.sleep(30)
    task=asyncio.create_task(automatic_deployments())
    yield
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

app=FastAPI(title='QuantInSight',docs_url=None,redoc_url=None,lifespan=lifespan)
app.add_middleware(AuthMiddleware)
app.include_router(auth_router)
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')


@app.middleware('http')
async def headers(request,call_next):
    if request.url.hostname=='autofolio.ai-ve.uk':
        return RedirectResponse('https://quantinsight.ai-ve.uk'+request.url.path+('?' + request.url.query if request.url.query else ''),status_code=308)
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='same-origin'
    response.headers.setdefault('Content-Security-Policy',"default-src 'self'; script-src 'self' https://static.cloudflareinsights.com; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-src 'self'; frame-ancestors 'none'")
    return response


@app.get('/')
@app.get('/strategy/{identity}')
def index(identity=None):return FileResponse(ROOT/'static/index.html')


@app.get('/api/leaderboard')
def leaderboard(request:Request,cohort: str|None=None,market:str=Query("kr",pattern="^(kr|us|crypto|timefolio)$")):
    from .evaluation import selection_metrics
    all_rows=[]
    for r in rows():
        if not research.visible(r,user(request),market):continue
        span=qualified_window(r);metric=selection_metrics(r)
        if not span or not metric:continue
        if setting('labs_v2_enabled',False) and (r.get('genome') or {}).get('engine')!='learned_v1':continue
        all_rows.append(dict(r,**{k:v for k,v in metric.items() if k!='monthly'},
                             evaluation_window=list(span),evaluation_months=36,selection_scope='os'))
    cohorts={}
    for r in all_rows:
        cohorts.setdefault(r['cohort'],dict(id=r['cohort'],start=r['start'],end=r['end'],sessions=r['sessions'],months=r['months'],evaluation_window=r['evaluation_window'],count=0))['count']+=1
    cohorts=sorted(cohorts.values(),key=lambda c:(c['evaluation_window'][1],c['count'],c['sessions']),reverse=True)
    selected=cohort if cohort in {c['id'] for c in cohorts} else (cohorts[0]['id'] if cohorts else None)
    chosen=[r for r in all_rows if r['cohort']==selected]
    front={r['id'] for r in pareto_front(chosen)}
    screened=[r for r in chosen if r.get('rule_screen_pass')]
    valid_front={r['id'] for r in pareto_front(screened)}
    points=[{k:v for k,v in r.items() if k not in ['cases','source_digest','limitations','genome']} |
            dict(pareto=r['id'] in front,screened_pareto=r['id'] in valid_front) for r in chosen]
    points.sort(key=lambda r:(-r['net_return'],r['negative_months']))
    from .map_references import references
    mapped=references(market,user(request),chosen)
    for point in points:point['applied_targets']=mapped['assignments'].get(point['id'],[])
    return dict(cohort=selected,cohorts=cohorts,strategies=points,total=len(all_rows),
                references=mapped['points'],
                frontier=[r['id'] for r in pareto_front(chosen)],
                screened_frontier=[r['id'] for r in pareto_front(screened)],
                period_label=((' ~ '.join([chosen[0]['start'],chosen[0]['end']])+' · OS 9개월') if chosen else 'OS 9개월 · 첫 검증 대기'), comparison='OS 9개월',selection_scope='os',protocol=research.protocol(market))


def qualified_window(summary):
    """A complete, pinned 36-month result remains inspectable after the daily roll."""
    from .period import window
    from .evaluation import valid_result
    span=summary.get('evaluation_window') or window()
    try:
        return tuple(span) if valid_result(summary) and span[1]<=window()[1] else None
    except (ValueError,TypeError,IndexError):return None


def selected_book(identity,phase):
    try:
        summary,case=strategy_case(identity,phase)
        if not qualified_window(summary):raise HTTPException(410,'완전한 36개월 평가 기록이 아닙니다.')
        return summary,case
    except KeyError:raise HTTPException(404,'전략을 찾을 수 없습니다.')
    except ValueError:raise HTTPException(400,'시작일을 확인해 주세요.')
    except (OSError,TypeError,IndexError):raise HTTPException(409,'원본 결과를 확인할 수 없습니다.')


def period_rows(case,months,span=None,scope='all'):
    if span:
        from .evaluation import periods
        bounds=dict(start=span[0],end=span[1]) if scope=='all' else periods(*span)[scope]
        return ledger(case,bounds['start'],bounds['end'])
    start,end,_=evaluation_scope(case)
    selected=ledger(case,start,end)
    if months and selected:
        keys=sorted({r['date'][:6] for r in selected})[-months:]
        selected=[r for r in selected if r['date'][:6] in keys]
    return selected


@app.get('/api/strategy/{identity}')
def detail(identity: str,request:Request,phase:int=Query(0,ge=0),months:int=Query(36,ge=36,le=36),symbol:str=Query('',max_length=12),scope:str=Query('os',pattern='^(is|os|ros|all)$')):
    summary,case=selected_book(identity,phase)
    if not research.visible(summary,user(request)):raise HTTPException(404,'전략을 찾을 수 없습니다.')
    from .evaluation import periods,performance,segment_metrics
    span=qualified_window(summary)
    selected=period_rows(case,months,span,scope)
    if not selected:raise HTTPException(409,'이 기간에 계좌 장부가 없습니다.')
    split=performance(period_rows(case,months,span),*span)
    metrics=segment_metrics(selected,*span,36) if scope=='all' else split[scope]
    if not metrics:raise HTTPException(409,'선택 구간의 완전한 평가 기록이 없습니다.')
    trades=[t for t in case['trades'] if selected[0]['date']<=normalized_date(t.get('date',''))<=selected[-1]['date']]
    symbols=Counter(str(t.get('code','')) for t in trades)
    filtered=[t for t in trades if not symbol or t.get('code')==symbol]
    flow=defaultdict(lambda:dict(buy=0.,sell=0.,buy_count=0,sell_count=0,fees=0.))
    for t in filtered:
        f=flow[normalized_date(t['date'])]
        side=t.get('side')
        if side not in ['buy','sell']:continue
        f[side]+=float(t['qty'])*float(t['price'])
        f[side+'_count']+=1
        f['fees']+=float(t.get('fee',0))
    daily=[dict(date=r['date'],net_return=r['nav']/selected[0]['previous_nav']-1,
                nav=r['nav'],cash=r.get('cash'),gross=r.get('gross'),**flow[r['date']]) for r in selected]
    fills=[dict(date=normalized_date(t['date']),price=t['price'],side=t['side'],qty=t['qty'],code=t['code']) for t in filtered] if symbol else []
    return dict(summary=summary,phase=phase,metrics=metrics,daily=daily,trade_count=len(filtered),
                scope=scope,selection_scope='os',performance=split,evaluation_periods=periods(*span),
                symbols=[dict(code=k,count=v) for k,v in sorted(symbols.items())],fills=fills,
                total_fees=sum(float(t.get('fee',0)) for t in trades),
                trades_note='체결 가격에는 슬리피지가 포함됩니다. 수수료는 별도입니다.')


@app.get('/api/strategy/{identity}/trades')
def transactions(identity: str,request:Request,phase:int=Query(0,ge=0),months:int=Query(36,ge=36,le=36),
                 symbol:str=Query('',max_length=12),offset:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=200),scope:str=Query('os',pattern='^(is|os|ros|all)$')):
    summary,case=selected_book(identity,phase)
    if not research.visible(summary,user(request)):raise HTTPException(404,'전략을 찾을 수 없습니다.')
    selected=period_rows(case,months,qualified_window(summary),scope)
    if not selected:return dict(total=0,trades=[])
    trades=[t for t in case['trades'] if selected[0]['date']<=normalized_date(t.get('date',''))<=selected[-1]['date']
            and (not symbol or t.get('code')==symbol)]
    trades.sort(key=lambda t:normalized_date(t['date']),reverse=True)
    return dict(total=len(trades),trades=trades[offset:offset+limit],offset=offset,limit=limit)


@app.get('/api/status')
def status(request:Request):
    admin_required(request)
    with connect() as db:
        counts=dict((r['status'],r['n']) for r in db.execute('SELECT status,count(*) n FROM jobs GROUP BY status'))
        coverage=dict((r['status'],r['n']) for r in db.execute('SELECT status,count(*) n FROM sources GROUP BY status'))
        active=[dict(r) for r in db.execute("SELECT id,generation,operator,status,started,genome FROM jobs WHERE status IN ('running','queued') ORDER BY created LIMIT 10")]
        recent=[dict(r) for r in db.execute('SELECT id,generation,operator,status,finished,error FROM jobs ORDER BY created DESC LIMIT 8')]
    for r in active:
        r['genome']=json.loads(r['genome'])
        path=RUNS/'experiments'/r['id']/'progress.json'
        try:r['progress']=json.loads(path.read_text())
        except (OSError,ValueError):r['progress']=None
    heartbeat=setting('heartbeat',{})
    return dict(market_running=setting('market_running',0),enabled=setting('enabled'),worker_alive=time.time()-heartbeat.get('timestamp',0)<60,
                heartbeat=heartbeat,generation=setting('generation',0),jobs=counts,active=active,recent=recent,
                coverage=coverage,baseline_verified=bool(setting('baseline_verified')),
                validation_candidate=setting('validation_candidate'),
                storage_guard=setting('storage_guard',{}),
                resources=dict(cpu_cores=setting('cpu_cores',4),memory_gb=setting('memory_gb',8),parallel=setting('concurrency',2),cloud=False),
                research_only=True,orders_enabled=False)


from .controls import router as controls_router
app.include_router(controls_router)

from .market_routes import router as market_router
from .crypto_view import router as crypto_router
app.include_router(market_router)
app.include_router(crypto_router)

from .briefing import router as briefing_router
from .auto_apply import router as auto_apply_router
app.include_router(briefing_router)
app.include_router(auto_apply_router)

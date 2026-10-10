"""User-scoped RFM 13 model execution with durable, fail-closed order intents.

Historical certification is separate from the contest server's current product
and account validation. An ambiguous submission is never automatically retried.
"""
import json
import math
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo
from .store import connect


def initialize():
    with connect() as db:
        db.execute('CREATE TABLE IF NOT EXISTS timefolio_model_state(user_id INTEGER PRIMARY KEY,body TEXT NOT NULL,updated REAL NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS timefolio_model_orders(identity TEXT PRIMARY KEY,user_id INTEGER NOT NULL,deployment TEXT NOT NULL,day TEXT NOT NULL,symbol TEXT NOT NULL,side TEXT NOT NULL,status TEXT NOT NULL,updated REAL NOT NULL)')


def snapshot(uid):
    initialize()
    with connect() as db:r=db.execute('SELECT body,updated FROM timefolio_model_state WHERE user_id=?',(uid,)).fetchone()
    return dict(json.loads(r['body']),updated=r['updated']) if r else dict(status='waiting',message='모델 연결 대기')


def save(uid,**body):
    with connect() as db:db.execute('INSERT INTO timefolio_model_state VALUES(?,?,?) ON CONFLICT(user_id) DO UPDATE SET body=excluded.body,updated=excluded.updated',(uid,json.dumps(body,ensure_ascii=False),time.time()))
    return body


def market_open(now):
    from .period import expected_dates
    day=now.strftime('%Y%m%d')
    return (now.weekday()<5 and day in expected_dates('kr',day,day)
            and (9,1)<=(now.hour,now.minute)<(15,15))


def select_contest(browser):
    selector=browser.page.locator('select').first
    options=selector.locator('option').evaluate_all('(els)=>els.map(o=>({text:o.text,value:o.value,selected:o.selected}))')
    selected=next((o for o in options if re.search(r'(?<!\d)13\s*회',o['text']) and '종료' not in o['text']),None)
    if not selected:raise ValueError('활성 13회 대회 없음')
    if not selected['selected']:
        selector.select_option(selected['value']);browser.page.wait_for_timeout(1800);browser._wait_nav(browser.page)
    if selector.input_value()!=selected['value']:raise ValueError('13회 계좌 선택 불일치')
    return selected['value']


def signal(dep,now):
    import joblib
    import pandas as pd
    import pyarrow.parquet as pq
    from .model_recipe import verify,file_hash,saved_learner
    from .paper import stock_prices
    from .period import last_complete_date
    folder=Path(dep['artifact']).parent
    recipe=json.loads((folder/'recipe.json').read_text());verify(recipe)
    if recipe['market']!='timefolio':raise ValueError('타임폴리오 학습 모델이 아닙니다')
    proof=json.loads((folder/'deployment.json').read_text())
    if file_hash(dep['artifact'])!=proof['model_sha256']:raise ValueError('운용 가중치 해시 변경')
    artifact=joblib.load(dep['artifact']);learner=saved_learner(recipe)
    base=pq.ParquetFile(recipe['input']['path']).read(use_threads=False).to_pandas()
    prices=stock_prices(base,'kr');last=last_complete_date('kr',now)
    prices=prices[prices.date<=pd.Timestamp(last)]
    g=recipe['definition'];aux=recipe.get('auxiliary_inputs',[])
    frame,names=learner.features(prices,g,'timefolio',aux[0]['path']) if aux else learner.features(prices,g,'timefolio')
    latest=frame.date.max()
    if latest.strftime('%Y%m%d')!=last:raise ValueError('최근 확정 거래일 연구 시세 부족')
    rows=frame[frame.date.eq(latest)&frame.ready].copy()
    if rows.empty:raise ValueError('예측 가능한 최신 시세 없음')
    rows['score']=artifact['model'].predict(rows[names].to_numpy())
    from . import execution
    execution=getattr(learner,'_EXECUTION_MODULE',None) or execution
    return rows.to_dict('records'),g,execution,last


def remaining_plan(plan,hold):
    """Keep buys after exits while removing only fills already seen in the book."""
    sells=[symbol for symbol in plan['sells'] if symbol in hold]
    buys=[pick for pick in plan['buys'] if pick['symbol'] not in hold or pick['symbol'] in sells]
    return dict(plan,sells=sells,buys=buys)


def order_payload(symbol,side,weight,summary,price,limits,adv=0,gross_exposure=1.):
    nav=float(summary.get('total_eval') or 0)
    if not math.isfinite(nav) or nav<=0:raise ValueError('계좌 평가액 없음')
    positions=summary.get('positions',[])
    held=next((p for p in positions if p['ticker']==symbol),None)
    if side=='sell':
        if not held:return None
        return dict(ticker=symbol,side=side,qty=int(held['qty']),weight_pct=float(held['weight_pct']),submit=True)
    if held:return None
    cash=nav-sum(float(p['value_krw']) for p in positions)
    # Unknown current market caps: treat every holding as small-cap (30% total).
    gross_exposure=min(gross_exposure,.30)
    exposure=max(0.,nav*gross_exposure-(nav-cash))/(1+gross_exposure*.0015)
    limit=min(weight,limits.get(symbol,limits['default']))
    budget=min(nav*limit/(1+limit*.0015),max(0,cash)/1.0015,adv*.01,exposure)
    if not math.isfinite(budget) or budget<=0:return None
    qty=math.floor(budget/price) if math.isfinite(price) and price>0 else 0
    if qty<=0:return None
    return dict(ticker=symbol,side=side,qty=qty,weight_pct=math.floor(qty*price/nav*10000)/100,submit=True)


def claim(uid,dep,day,symbol,side):
    identity=f'{uid}:{day}:{symbol}:{side}'
    with connect() as db:
        cur=db.execute("INSERT OR IGNORE INTO timefolio_model_orders VALUES(?,?,?,?,?,?,'submitting',?)",(identity,uid,dep,day,symbol,side,time.time()))
    return identity if cur.rowcount else None


def unresolved(uid):
    with connect() as db:
        return bool(db.execute("SELECT 1 FROM timefolio_model_orders WHERE user_id=? AND status IN ('submitting','review_required') LIMIT 1",(uid,)).fetchone())


def current_deployment(dep):
    with connect() as db:
        return bool(db.execute('''SELECT 1 FROM model_deployments d JOIN strategy_assignments a
            ON a.user_id=d.user_id AND a.target=d.target AND a.strategy_id=d.strategy_id
            WHERE d.id=? AND d.user_id=? AND d.target='timefolio' AND d.status='ready'
            AND a.status='timefolio_ready' ''',(dep['id'],dep['user_id'])).fetchone())


def advance(dep,now=None):
    import fcntl
    from .config import RUNS
    folder=RUNS/'timefolio_locks';folder.mkdir(parents=True,exist_ok=True)
    with (folder/(str(int(dep['user_id']))+'.lock')).open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(status='busy',message='타임폴리오 실행 중')
        try:return _advance(dep,now)
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def _advance(dep,now=None):
    from .auth import get_connection
    from .contest_rules_evidence import profile
    now=now or datetime.now(ZoneInfo('Asia/Seoul'));uid=dep['user_id'];initialize()
    rules=profile();day=now.strftime('%Y%m%d')
    base=dict(deployment=dep['id'],strategy_id=dep['strategy_id'],competition_compliance_verified=False)
    if not rules or not rules['start']<=day<=rules['end']:
        return save(uid,**base,status='blocked',message='13회 유효 규정 또는 대회 기간 확인 필요')
    if unresolved(uid):
        return save(uid,**base,status='review_required',message='이전 주문 접수 여부 확인 필요 · 추가 주문 정지')
    credentials=get_connection(uid,'timefolio')
    if not credentials:return save(uid,**base,status='blocked',message='타임폴리오 계정 연결 필요')
    records,g,execution,signal_day=signal(dep,now)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'integrations/timefolio'))
    from Auto_folio.autofolio.timefolio_browser import TimefolioBrowser,TimefolioCredentials
    with TimefolioBrowser(headless=True,live_enabled=market_open(now)) as browser:
        if not browser.login(TimefolioCredentials(credentials['username'],credentials['password'])).get('logged_in'):
            raise ValueError('타임폴리오 로그인 실패')
        contest=select_contest(browser);summary=browser.scrape_summary()
        if not summary.get('total_eval'):raise ValueError('대회 평가액 확인 실패')
        base.update(contest_id=contest,signal_day=signal_day,connected=True)
        positions=summary.get('positions',[])
        hold={p['ticker']:p['qty'] for p in positions};marks={p['ticker']:p['last_price'] for p in positions};entries={p['ticker']:p['avg_price'] for p in positions}
        previous=snapshot(uid)
        from .period import expected_dates
        last_rebalance=previous.get('last_rebalance')
        elapsed=len(expected_dates('kr',last_rebalance,signal_day))-1 if last_rebalance and last_rebalance<=signal_day else 0
        rebalance=previous.get('deployment')!=dep['id'] or not last_rebalance or elapsed>=execution.policy(g)['rebalance_sessions']
        if previous.get('deployment')==dep['id'] and previous.get('signal_day')==signal_day and previous.get('pending_plan'):
            plan=previous['pending_plan']
        else:plan=execution.build_plan(records,g,hold,marks,entries,rebalance)
        plan=remaining_plan(plan,hold)
        base['pending_plan']=plan
        base.update(last_rebalance=last_rebalance,planned_buys=len(plan['buys']),planned_sells=len(plan['sells']))
        if not market_open(now):return save(uid,**base,status='waiting_session',message='13회 모델 연결 완료 · 다음 거래시간 대기')
        if browser.list_working_orders():return save(uid,**base,status='waiting_orders',message='기존 미체결 주문 정리 대기')
        # Sell first. Any submission ends this cycle; re-read actual balances next time.
        prices={r['symbol']:float(r['close']) for r in records}
        candidates=[(s,'sell',0,0) for s in plan['sells']]+[(p['symbol'],'buy',p['weight'],p['adv20']) for p in plan['buys']]
        for symbol,side,weight,adv in candidates:
            payload=order_payload(symbol,side,weight,summary,prices.get(symbol,marks.get(symbol,0)),rules['position_limits'],adv,execution.policy(g)['gross_exposure'])
            if not payload or payload['weight_pct']<=0:continue
            # Preparing a ticket queries the contest's current product validator.
            validation={}
            def capture(response):
                if urlsplit(response.url).path=='/api/Portfolio/ValidateProduct':
                    try:validation.update(response.json())
                    except Exception:pass
            browser.page.on('response',capture)
            prepared=browser.place_order(dict(payload,submit=False))
            browser.page.remove_listener('response',capture);browser._close_ticket(browser.page)
            if not prepared.get('prepared') or not product_valid(validation):continue
            if not market_open(datetime.now(ZoneInfo('Asia/Seoul'))):
                return save(uid,**base,status='waiting_session',message='거래시간 종료 · 다음 거래시간 대기')
            if unresolved(uid):
                return save(uid,**base,status='review_required',message='이전 주문 접수 여부 확인 필요 · 추가 주문 정지')
            # Deployment activation uses this same lock. A model superseded
            # while logging in or planning must never submit its old signal.
            from .auto_apply import lock as activation_lock
            with activation_lock():
                if not current_deployment(dep):
                    return dict(status='superseded',message='후속 모델로 전환됨 · 이전 주문 취소')
                identity=claim(uid,dep['id'],day,symbol,side)
                if not identity:continue
                result=browser.place_order(payload)
            status='filled' if result.get('filled') else 'accepted' if result.get('accepted') else 'review_required'
            with connect() as db:db.execute('UPDATE timefolio_model_orders SET status=?,updated=? WHERE identity=?',(status,time.time(),identity))
            return save(uid,**base,status=status,message='13회 모델 주문 '+status,last_order=dict(symbol=symbol,side=side,status=status))
        if plan.get('rebalance') and not plan['sells'] and not plan['buys']:
            base['last_rebalance']=signal_day; base['pending_plan']=None
        return save(uid,**base,status='connected',message='13회 모델 운용 연결 · 현재 주문 조건 대기')


def product_valid(value):
    """Fail closed until a positive server product verdict is available."""
    return isinstance(value,dict) and value.get('data')=='' and value.get('warnings')==[]


def run():
    initialize()
    with connect() as db:rows=db.execute("SELECT d.* FROM model_deployments d JOIN strategy_assignments s ON s.user_id=d.user_id AND s.target=d.target AND s.strategy_id=d.strategy_id WHERE d.status='ready' AND d.target='timefolio'").fetchall()
    for row in rows:
        try:advance(dict(row))
        except Exception as exc:save(row['user_id'],deployment=row['id'],strategy_id=row['strategy_id'],status='blocked',message='타임폴리오 운용 확인 필요: '+type(exc).__name__)


if __name__=='__main__':run()

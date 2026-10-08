"""Own forward-only stock paper accounts, with no brokerage API dependency."""
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from .store import connect


def initialize():
    with connect() as db:db.execute('''CREATE TABLE IF NOT EXISTS paper_books(
        user_id INTEGER NOT NULL,market TEXT NOT NULL,deployment TEXT NOT NULL,
        body TEXT NOT NULL,updated REAL NOT NULL,PRIMARY KEY(user_id,market))''')


def snapshot(uid,market):
    initialize();target=market+'-paper'
    from .deployment import initialize as deployments
    deployments()
    with connect() as db:
        row=db.execute('SELECT body,updated FROM paper_books WHERE user_id=? AND market=?',(uid,market)).fetchone()
        assignment=db.execute('SELECT * FROM strategy_assignments WHERE user_id=? AND target=?',(uid,target)).fetchone()
        dep=db.execute('SELECT status,message FROM model_deployments WHERE user_id=? AND target=? ORDER BY updated DESC LIMIT 1',(uid,target)).fetchone()
    result=dict(connected=False,message=dep['message'] if dep else '적용한 전략 없음',currency='KRW' if market=='kr' else 'USD',equity=[],holdings=[],trades=[],total_eval=None,cash=None,pnl_ratio=None,assignment=dict(assignment) if assignment else None)
    if row:
        book=json.loads(row['body']);result.update(connected=True,message=book.get('message','자체 페이퍼 장부'),equity=book['equity'],holdings=list(book['holdings'].values()),trades=list(reversed(book['trades'][-200:])),total_eval=book['nav'],cash=book['cash'],pnl_ratio=book['nav']/book['initial']-1,updated=row['updated'])
    return result


def advance(dep):
    """Record a signal now; only a later observed session may fill it.

    Initial activation never replays historical trades into live paper equity.
    Daily bars are processed after the session closes; execution is a simulated
    next-open price, not an intraday fill feed.
    """
    import joblib
    import pandas as pd
    import pyarrow.parquet as pq
    from pathlib import Path
    from .learning import features
    from .research_input import ROOTS,validate_frame
    from .model_recipe import verify,file_hash
    market=dep['target'].removesuffix('-paper')
    if market not in ('kr','us','crypto'):return
    folder=Path(dep['artifact']).parent;recipe=json.loads((folder/'recipe.json').read_text());verify(recipe)
    proof=json.loads((folder/'deployment.json').read_text())
    if file_hash(dep['artifact'])!=proof['model_sha256']:raise ValueError('운용 가중치 해시 변경')
    from .model_recipe import saved_learner
    learner=saved_learner(recipe)
    features=learner.features
    from . import execution as current_execution
    execution=getattr(learner,'_EXECUTION_MODULE',None) or current_execution
    artifact=joblib.load(dep['artifact']);g=recipe['definition']
    base=pq.ParquetFile(recipe['input']['path']).read(use_threads=False).to_pandas()
    local=datetime.now(ZoneInfo('UTC' if market=='crypto' else 'Asia/Seoul' if market=='kr' else 'America/New_York'))
    if market=='crypto':
        prices=crypto_prices(base,local)
    else:
        frames=[]
        # Preserve historical adjustment basis; incompatible providers are excluded.
        for symbol,old in base.groupby('symbol',sort=False):
            path=ROOTS[market]/(str(symbol).split('@')[0]+'.parquet')
            if not path.exists():frames.append(old);continue
            fresh=validate_frame(pq.ParquetFile(path).read(use_threads=False).to_pandas())
            overlap=fresh.merge(old[['date','close']],on='date',suffixes=('_n','_o')).tail(20)
            if overlap.empty or ((overlap.close_n/overlap.close_o-1).abs()>.005).any():frames.append(old);continue
            newer=fresh[fresh.date>old.date.max()].copy()
            for k in ['symbol','sector','eligible']:newer[k]=old.iloc[-1][k]
            newer['tradable_buy']=newer['tradable_sell']=newer.volume.gt(0)
            frames.extend([old,newer])
        prices=pd.concat(frames,ignore_index=True)
    cutoff=pd.Timestamp(local.date())
    if market=='crypto' or local.hour<(16 if market=='kr' else 17):cutoff-=pd.Timedelta(days=1)
    prices=prices[prices.date<=cutoff];p,names=features(prices,g,market)
    if p.empty:return
    latest=p.date.max();day=latest.strftime('%Y%m%d')
    with connect() as db:r=db.execute('SELECT body,deployment FROM paper_books WHERE user_id=? AND market=?',(dep['user_id'],market)).fetchone()
    book=json.loads(r['body']) if r else dict(not_before=local.strftime('%Y%m%d'),initial=1e8 if market=='kr' else 1e5,cash=1e8 if market=='kr' else 1e5,nav=1e8 if market=='kr' else 1e5,holdings={},trades=[],equity=[],last_day='',pending=[],rebalance=0)
    if book['last_day']>=day:return
    same_deployment=bool(r and r['deployment']==dep['id'])
    if not same_deployment:
        book['not_before']=local.strftime('%Y%m%d');book['pending']=[];book['rebalance']=0
    if same_deployment:
        # Catch up each newly observed session, never jump an old signal to the latest open.
        observed=sorted(p.loc[(p.date.dt.strftime('%Y%m%d')>book['last_day']) &
            (p.date.dt.strftime('%Y%m%d')>=book.get('not_before',day)),'date'].unique())
        if not observed:observed=[latest]
    else:observed=[latest]
    for stamp in observed:
        stamp=pd.Timestamp(stamp);day=stamp.strftime('%Y%m%d')
        quotes=p[p.date.eq(stamp)].set_index('symbol')
        step_book(book,quotes,g,artifact['model'],names,market,day,same_deployment,execution)
    book['message']='자체 페이퍼 운용 · 일봉 확정 후 다음 시가 체결 모의 · 증권사 주문 없음'
    initialize()
    with connect() as db:db.execute('INSERT INTO paper_books VALUES(?,?,?,?,?) ON CONFLICT(user_id,market) DO UPDATE SET deployment=excluded.deployment,body=excluded.body,updated=excluded.updated',(dep['user_id'],market,dep['id'],json.dumps(book,ensure_ascii=False),time.time()))


def step_book(book,quotes,g,model,names,market,day,same_deployment,execution):
    """Advance one completed session using the exact research execution policy."""
    hold={symbol:h['qty'] for symbol,h in book['holdings'].items()}
    marks={symbol:h['value']/h['qty'] for symbol,h in book['holdings'].items()}
    entries={symbol:h['entry'] for symbol,h in book['holdings'].items()}
    unfilled=[]
    if same_deployment and book['pending'] and book['last_day']<day and day>book.get('not_before',day):
        pending=book['pending']
        book['cash'],trades,unfilled=execution.fill_orders(book['cash'],hold,marks,entries,quotes,pending,g,market,day)
        book['trades'].extend(trades)
        if isinstance(pending,list) or pending.get('rebalance'):
            book['rebalance']=execution.policy(g)['rebalance_sessions']
        book['pending']=[]
    import math
    for symbol in hold:
        if symbol in quotes.index:
            px=float(quotes.loc[symbol,'close'])
            if math.isfinite(px) and px>0:marks[symbol]=px
    book['holdings']={symbol:dict(code=symbol,name=symbol,qty=qty,entry=entries[symbol],
        value=qty*marks[symbol],pnl_ratio=marks[symbol]/entries[symbol]-1) for symbol,qty in hold.items()}
    book['nav']=book['cash']+sum(h['value'] for h in book['holdings'].values())
    rebalance=not same_deployment or book['rebalance']<=1
    test=quotes[quotes.ready].copy();records=[]
    if len(test) and rebalance:
        test['score']=model.predict(test[names].to_numpy())
        records=test.reset_index().to_dict('records')
    book['pending']=execution.build_plan(records,g,hold,marks,entries,rebalance)
    book['pending']['sells']=list(dict.fromkeys(unfilled+book['pending']['sells']))
    book['rebalance']=max(0,book['rebalance']-1);book['last_day']=day
    book['equity'].append(dict(date=day,net_return=book['nav']/book['initial']-1))


def crypto_prices(base,now):
    """Shared completed-day crypto tail for deployed paper models, no exchange API."""
    import pandas as pd
    import pyarrow.parquet as pq
    from .config import HOME,RUNS
    from .learning_input import canonical_minutes
    cutoff=pd.Timestamp(now.date())
    cache=RUNS/'research_inputs/live_crypto'/f"{base.date.max():%Y%m%d}-{cutoff:%Y%m%d}.parquet"
    if cache.exists():return pd.concat([base,pq.ParquetFile(cache).read(use_threads=False).to_pandas()],ignore_index=True)
    start=base.date.max()+pd.Timedelta(days=1);frames=[]
    source=HOME/'vault/CryptoBars/data/history'
    months=pd.period_range(start,cutoff-pd.Timedelta(days=1),freq='M') if start<cutoff else []
    for symbol in sorted(base.symbol.unique()):
        parts=[]
        for month in months:
            for file in sorted((source/('base='+symbol)).glob('part-'+str(month)+'*.parquet')):
                d=pq.ParquetFile(file).read(columns=['ts','base','open','high','low','close','quote_volume'],use_threads=False).to_pandas()
                d=d[(d.ts>=start.value//10**6)&(d.ts<cutoff.value//10**6)].copy();d['_source']=0
                if len(d):parts.append(d)
        if not parts:continue
        try:d,_=canonical_minutes(parts,reject_invalid=True)
        except ValueError:continue
        d['date']=pd.to_datetime(d.ts,unit='ms').dt.floor('D')
        d=d.groupby('date').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('quote_volume','sum'),minutes=('ts','nunique')).reset_index()
        d['symbol']=symbol;d['sector']='CRYPTO';d['eligible']=d.minutes.ge(1200)&d.volume.gt(0);d['tradable_buy']=d['tradable_sell']=d.eligible
        frames.append(d.drop(columns='minutes'))
    if not frames:return base
    tail=pd.concat(frames,ignore_index=True);cache.parent.mkdir(parents=True,exist_ok=True)
    temp=cache.with_suffix('.tmp');tail.to_parquet(temp,index=False);temp.replace(cache)
    return pd.concat([base,tail],ignore_index=True)


def run():
    from .deployment import initialize as deployments
    deployments();initialize()
    with connect() as db:rows=db.execute("SELECT d.* FROM model_deployments d JOIN strategy_assignments s ON s.user_id=d.user_id AND s.target=d.target AND s.strategy_id=d.strategy_id WHERE d.status='ready' AND d.target IN ('kr-paper','us-paper','crypto-paper')").fetchall()
    for row in rows:
        try:advance(dict(row))
        except Exception as exc:
            from .store import event
            event('paper_failed',dict(message=str(exc)[:200],market=row['target'].split('-')[0]))

if __name__=='__main__':run()

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
    artifact=joblib.load(dep['artifact']);g=recipe['definition']
    from .model_recipe import saved_learner
    features=saved_learner(recipe).features
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
    latest=p.date.max();day=latest.strftime('%Y%m%d');quotes=p[p.date.eq(latest)].set_index('symbol')
    with connect() as db:r=db.execute('SELECT body,deployment FROM paper_books WHERE user_id=? AND market=?',(dep['user_id'],market)).fetchone()
    book=json.loads(r['body']) if r else dict(not_before=local.strftime('%Y%m%d'),initial=1e8 if market=='kr' else 1e5,cash=1e8 if market=='kr' else 1e5,nav=1e8 if market=='kr' else 1e5,holdings={},trades=[],equity=[],last_day='',pending=[],rebalance=0)
    if book['last_day']>=day:return
    import math
    cost=.002 if market=='kr' else .001
    if r and r['deployment']==dep['id'] and book['pending'] and book['last_day']<day and day>book.get('not_before',day):
        for symbol,h in list(book['holdings'].items()):
            if symbol not in quotes.index or not quotes.loc[symbol,'tradable_sell']:continue
            px=float(quotes.loc[symbol,'open']);qty=h['qty'];fee=px*qty*cost;book['cash']+=px*qty-fee
            book['trades'].append(dict(date=day,code=symbol,side='sell',qty=qty,price=px,fee=fee));del book['holdings'][symbol]
        for pick in book['pending']:
            symbol=pick['symbol']
            if symbol not in quotes.index or symbol in book['holdings'] or not quotes.loc[symbol,'tradable_buy']:continue
            px=float(quotes.loc[symbol,'open']);budget=min(book['cash']/(1+cost),book['nav']*min(.1,1/g['selection_count']),pick['adv20']*.01);qty=budget/px if market=='crypto' else math.floor(budget/px)
            if qty<=0:continue
            fee=px*qty*cost;book['cash']-=px*qty+fee;book['holdings'][symbol]=dict(code=symbol,name=symbol,qty=qty,entry=px,value=qty*px,pnl_ratio=0)
            book['trades'].append(dict(date=day,code=symbol,side='buy',qty=qty,price=px,fee=fee))
        book['pending']=[];book['rebalance']=g['holding_sessions']
    for symbol,h in book['holdings'].items():
        if symbol in quotes.index:
            px=float(quotes.loc[symbol,'close']);h.update(value=px*h['qty'],pnl_ratio=px/h['entry']-1)
    book['nav']=book['cash']+sum(h['value'] for h in book['holdings'].values())
    test=quotes[quotes.ready].copy()
    if len(test) and (not r or r['deployment']!=dep['id'] or book['rebalance']<=1):
        test['score']=artifact['model'].predict(test[names].to_numpy())
        book['pending']=[dict(symbol=s,adv20=float(row.adv20)) for s,row in test.sort_values('score',ascending=False).head(g['selection_count']).iterrows()]
    book['rebalance']=max(0,book['rebalance']-1);book['last_day']=day
    book['equity'].append(dict(date=day,net_return=book['nav']/book['initial']-1))
    book['message']='자체 페이퍼 운용 · 일봉 확정 후 다음 시가 체결 모의 · 증권사 주문 없음'
    initialize()
    with connect() as db:db.execute('INSERT INTO paper_books VALUES(?,?,?,?,?) ON CONFLICT(user_id,market) DO UPDATE SET deployment=excluded.deployment,body=excluded.body,updated=excluded.updated',(dep['user_id'],market,dep['id'],json.dumps(book,ensure_ascii=False),time.time()))


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

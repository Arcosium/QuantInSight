"""Bounded learned-model evolution with causal, quarterly walk-forward evaluation.

Each trial fits real estimators, records its recipe, then releases all weights.
Retraining creates an offline artifact only, never broker orders.
"""
import gc
import json
from pathlib import Path
import hashlib
_LOADED_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def domains(market):
    if market not in ('kr','us','crypto','timefolio'):raise ValueError('지원하지 않는 연구소')
    return dict(model=['ridge','extra_trees','hist_gradient_boosting'],feature_set=['price','price_volume','price_risk','all'],lookback=[5,10,20,40],holding_sessions=[1,3,5] if market=='timefolio' else [5,10,20],selection_count=[5,10,20],training_days=[365,730],seed=[20261008],regularization=[.1,1.,10.],max_depth=[3,5,8])


def normalize(g,market):
    limits=domains(market)
    if not isinstance(g,dict) or set(g)-set(limits)-{'engine'} or set(limits)-set(g):
        raise ValueError('학습 설정 필드가 누락되거나 지원하지 않는 필드가 있음')
    if 'engine' in g and g['engine']!='learned_v1':raise ValueError('지원하지 않는 학습 엔진')
    out={}
    for key,values in limits.items():
        value=g[key]
        if isinstance(value,bool):raise ValueError('잘못된 학습 설정 형식')
        if key=='seed':
            if not isinstance(value,int) or not 1<=value<=2147483647:raise ValueError('학습 seed 범위 오류')
        elif key in ('lookback','holding_sessions','selection_count','training_days','max_depth'):
            if not isinstance(value,int) or value not in values:raise ValueError('지원하지 않는 학습 설정: '+key)
        elif key=='regularization':
            if not isinstance(value,(int,float)) or value not in values:raise ValueError('지원하지 않는 정규화 계수')
        elif not isinstance(value,str) or value not in values:raise ValueError('지원하지 않는 학습 설정: '+key)
        out[key]=value
    out['engine']='learned_v1'
    return out


def features(prices,g,market):
    import numpy as np
    import pandas as pd
    p=prices.sort_values(['symbol','date']).reset_index(drop=True).copy()
    grouped=p.groupby('symbol',sort=False);w=g['lookback']
    p['return_1']=grouped.close.pct_change(fill_method=None)
    p['return_w']=grouped.close.pct_change(w,fill_method=None)
    p['range']=(p.high-p.low)/p.close
    p['body']=(p.close-p.open)/p.open
    p['volatility']=p.groupby('symbol').return_1.transform(lambda s:s.rolling(w).std())
    p['trend']=p.close/grouped.close.transform(lambda s:s.rolling(w).mean())-1
    p['volume_ratio']=p.volume/grouped.volume.transform(lambda s:s.rolling(w).mean())-1
    value=p.volume if market=='crypto' else p.volume*p.close
    p['adv20']=value.groupby(p.symbol).transform(lambda s:s.rolling(20).mean())
    names=['return_1','return_w','range','body','trend']
    if g['feature_set'] in ('price_volume','all'):names+=['volume_ratio']
    if g['feature_set'] in ('price_risk','all'):names+=['volatility']
    h=g['holding_sessions'];entry=grouped.open.shift(-1);exit_price=grouped.open.shift(-h-1)
    p['target']=exit_price/entry-1
    p['label_end']=grouped.date.shift(-h-1)
    # Labels spanning missing exchange sessions are forbidden.
    calendar=pd.DatetimeIndex(sorted(p.date.unique()));ordinal=pd.Series(calendar.get_indexer(p.date),index=p.index)
    past=ordinal-ordinal.groupby(p.symbol).shift(max(w,20))
    future=ordinal.groupby(p.symbol).shift(-h-1)-ordinal
    p['ready']=p.eligible.astype(bool)&past.eq(max(w,20))&np.isfinite(p[names]).all(axis=1)
    executable=grouped.tradable_buy.shift(-1).eq(True)&grouped.tradable_sell.shift(-h-1).eq(True)
    p.loc[~future.eq(h+1)|~executable,'target']=np.nan
    return p,names


def estimator(g):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
    if g['model']=='ridge':return make_pipeline(StandardScaler(),Ridge(alpha=g['regularization']))
    if g['model']=='extra_trees':return ExtraTreesRegressor(n_estimators=48,max_depth=g['max_depth'],min_samples_leaf=20,n_jobs=1,random_state=g['seed'])
    return HistGradientBoostingRegressor(max_iter=60,max_depth=g['max_depth'],l2_regularization=g['regularization'],early_stopping=False,random_state=g['seed'])


def training_rows(p,first,g):
    import numpy as np
    import pandas as pd
    rows=np.flatnonzero(p.ready & np.isfinite(p.target)&p.label_end.lt(first)&p.date.ge(first-pd.Timedelta(days=g['training_days'])))
    if len(rows)<100:raise ValueError('인과적 모델 학습 표본 부족')
    # Deterministic bounded training, independent of target magnitudes.
    if len(rows)>30000:rows=np.sort(np.random.default_rng(g['seed']).choice(rows,30000,replace=False))
    return rows


def fit_model(p,names,first,g):
    import pandas as pd
    from threadpoolctl import threadpool_limits
    rows=training_rows(p,first,g);model=estimator(g)
    with threadpool_limits(limits=1):model.fit(p.iloc[rows][names].to_numpy(),p.iloc[rows].target.to_numpy())
    proof=dict(train_start=str(p.iloc[rows].date.min().date()),train_end=str(p.iloc[rows].date.max().date()),max_label_end=str(p.iloc[rows].label_end.max().date()),model_as_of=str(first.date()),train_rows=len(rows))
    assert pd.Timestamp(proof['max_label_end'])<first
    return model,proof


def walk_forward(p,names,g,start,end):
    import pandas as pd
    from threadpoolctl import threadpool_limits
    first=pd.Timestamp(start);final=pd.Timestamp(end)+pd.Timedelta(days=1)
    bounds=sorted(set([first,final,*pd.date_range(first,final,freq='QS')]))
    signals=[];proofs=[]
    for first,stop in zip(bounds[:-1],bounds[1:]):
        model,proof=fit_model(p,names,first,g)
        test=p[p.ready&p.date.ge(first)&p.date.lt(stop)].copy()
        if test.empty:raise ValueError('분기 평가 표본 부족')
        with threadpool_limits(limits=1):test['score']=model.predict(test[names].to_numpy())
        signals.append(test[['date','symbol','score','adv20']]);proof.update(test_start=str(first.date()),test_end_exclusive=str(stop.date()));proofs.append(proof)
        del model,test;gc.collect()
    return pd.concat(signals,ignore_index=True),proofs


def account(prices,signals,g,market,start,end):
    """Close-t signals, next-session open fills, long-only cash account.

    Missing prices never manufacture fills. Unsellable holdings stay marked at
    their last known close, explicitly flagged in the report.
    """
    import math
    import pandas as pd
    from .period import expected_dates
    days=expected_dates(market,start,end);bydate={d:g.set_index('symbol') for d,g in prices.groupby(prices.date.dt.strftime('%Y%m%d'))}
    sig={d:g.sort_values(['score','symbol'],ascending=[False,True]) for d,g in signals.groupby(signals.date.dt.strftime('%Y%m%d'))}
    initial=1e9 if market=='timefolio' else 1e8 if market=='kr' else 1e5
    cash=float(initial);hold={};marks={};daily=[];trades=[];stale=0
    buy_cost=.0015 if market=='timefolio' else .002 if market=='kr' else .001
    sell_cost=.0035 if market=='timefolio' else buy_cost
    for index,day in enumerate(days):
        book=bydate.get(day)
        if book is None:raise ValueError('평가일 시세 누락')
        if index>0 and (index-1)%g['holding_sessions']==0:
            for symbol,qty in list(hold.items()):
                if symbol not in book.index or not bool(book.loc[symbol,'tradable_sell']):continue
                price=float(book.loc[symbol,'open']);fee=qty*price*sell_cost;cash+=qty*price-fee
                trades.append(dict(date=day,code=symbol,side='sell',qty=qty,price=price,fee=fee));del hold[symbol]
            picks=sig.get(days[index-1]);equity=cash+sum(q*marks[s] for s,q in hold.items())
            if picks is not None:
                for row in picks.head(g['selection_count']).itertuples():
                    if row.symbol in hold or row.symbol not in book.index or not bool(book.loc[row.symbol,'tradable_buy']):continue
                    price=float(book.loc[row.symbol,'open']);budget=min(cash/(1+buy_cost),equity*min(.1,1/g['selection_count']),float(row.adv20)*.01)
                    if market=='timefolio':budget=min(budget,timefolio_headroom(cash,hold,marks,book,row.symbol,buy_cost))
                    qty=budget/price if market=='crypto' else math.floor(budget/price)
                    if qty<=0:continue
                    fee=qty*price*buy_cost;cash-=qty*price+fee;hold[row.symbol]=qty;marks[row.symbol]=price
                    trades.append(dict(date=day,code=row.symbol,side='buy',qty=qty,price=price,fee=fee))
        for symbol in hold:
            if symbol in book.index:marks[symbol]=float(book.loc[symbol,'close'])
            else:stale+=1
        nav=cash+sum(q*marks[s] for s,q in hold.items());daily.append(dict(date=day,nav=nav,cash=cash,positions=len(hold)))
    return dict(initial_cash=initial,daily=daily,trades=trades),stale


def timefolio_headroom(cash,hold,marks,book,symbol,buy_cost):
    """Conservative proxy until dated sector/size/designation data is audited.

    Existing Timefolio study: 10% floor sector cap, small-cap basket 30%,
    exposure80%. Missing sector treats every unknown as the same sector;
    missing size treats every name as small, never as exempt.
    """
    values={s:q*float(book.loc[s,'open'] if s in book.index else marks[s]) for s,q in hold.items()}
    nav=cash+sum(values.values())
    def sector(s):
        if s not in book.index or 'sector' not in book.columns:return 'UNKNOWN'
        value=book.loc[s,'sector']
        return 'UNKNOWN' if value is None or str(value) in ('','nan','UNKNOWN') else str(value)
    used=sum(value for s,value in values.items() if sector(s)==sector(symbol))
    # Include fee-induced NAV reduction in each limit's headroom.
    limits=[(.1*nav-used)/(1+.1*buy_cost),(.3*nav-sum(values.values()))/(1+.3*buy_cost),(.8*nav-sum(values.values()))/(1+.8*buy_cost)]
    return max(0.,min(limits))


def timefolio_assessment(case):
    import pandas as pd
    d=pd.DataFrame(case['daily']);d['date']=pd.to_datetime(d.date)
    d['week']=d.date.dt.to_period('W')
    trades=pd.DataFrame(case['trades'])
    if trades.empty:d['traded']=0.
    else:
        trades['date']=pd.to_datetime(trades.date);trades['value']=trades.qty*trades.price
        d['traded']=d.date.map(trades.groupby('date').value.sum()).fillna(0.)
    weekly=d.groupby('week').agg(nav=('nav','mean'),traded=('traded','sum'),days=('date','size'))
    weekly['turnover']=.5*weekly.traded/weekly.nav
    full=weekly.iloc[1:-1];low=int(full.turnover.lt(.05).sum())
    return dict(profile='conservative_contest_proxy',sector_cap=.1,unknown_sector_policy='shared UNKNOWN sector',unknown_size_policy='all treated as small-cap',small_cap_total=.3,max_gross=.8,buy_fee=.001,sell_fee=.003,slippage_proxy=.0005,
                min_weekly_turnover=.05,low_turnover_weeks=low,four_week_turnover_stop=low>=4,
                historical_designations_verified=False,competition_compliance_verified=False,
                weekly=[dict(week=str(i),turnover=float(row.turnover),days=int(row.days)) for i,row in weekly.iterrows()])


def evaluate(candidate,destination):
    import pandas as pd
    from . import learning_input,model_recipe
    from .period import window,require_complete
    from .metrics import ledger,statistics_for
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    market=candidate['market'];definition=candidate['definition'];g=normalize(json.loads(definition) if isinstance(definition,str) else definition,market)
    code=model_recipe.code_hashes()
    if code['learning.py']!=_LOADED_SHA256:raise ValueError('실험 시작 중 코드 변경 감지; 새 버전으로 다시 실행해 주세요.')
    model_recipe.preserve_code()
    prices=learning_input.prepare(market);p,names=features(prices,g,market);start,end=window()
    signals,proofs=walk_forward(p,names,g,start,end)
    case,stale=account(p,signals,g,market,start,end);require_complete([r['date'] for r in case['daily']],market)
    recipe=model_recipe.seal(dict(version=1,market=market,definition=g,fields=names,input=learning_input.descriptor(market),libraries=model_recipe.environment(),code=code,evaluation_start=start,evaluation_end=end,folds=proofs,deploy_as_of=str((pd.Timestamp(end)+pd.Timedelta(days=1)).date()),weights_retained=False,execution=dict(signal='session close',fill='next session open',long_only=True,max_name_weight=.1,adv_participation=.01,timefolio_profile='conservative_contest_proxy' if market=='timefolio' else None),target='next-open to holding_sessions+1 open total price return'))
    report=dict(title=candidate.get('title',g['model']),family='인공지능 학습 · '+g['model'],market=market,owner_id=candidate.get('user_id'),definition=g,model_proofs=proofs,recipe=recipe,evaluation_months=36,cases=[case],no_broker_orders=True,independent_holdout=False,weights_retained=False,competition_compliance_verified=False,
      limitations=['반복 탐색에 사용한 36개월 개발 평가이며 독립 홀드아웃 아님','현재 보유 데이터 유니버스의 생존·수집 선택 편향','기업행사·상장폐지·체결·대회 세부 규칙 미감사; 적용 전 별도 검증 필요','롱 전용; 뉴스·공시 시점 정렬 미지원; 가격·거래량 파생 필드만 사용',f'가격 누락 보유 종목을 마지막 종가로 평가한 종목일 {stale}회'],metrics=statistics_for(ledger(case,start,end)))
    if market=='timefolio':report['contest_constraints']=timefolio_assessment(case)
    (destination/'recipe.json').write_text(json.dumps(recipe,ensure_ascii=False,allow_nan=False,indent=2))
    path=destination/'review.json';path.write_text(json.dumps(report,ensure_ascii=False,allow_nan=False));return path


def retrain(recipe,destination):
    """Return retained offline artifact for the sealed terminal training cut-off.

    recipe may be a dict, recipe.json path, or review.json path. Exact-fold replay
    is available via the stored folds; deployment refits at deploy_as_of.
    """
    import pandas as pd
    import pyarrow.parquet as pq
    import joblib
    from . import model_recipe
    if not isinstance(recipe,dict):recipe=json.loads(Path(recipe).read_text())
    recipe=recipe.get('recipe',recipe);model_recipe.verify(recipe)
    learner=model_recipe.saved_learner(recipe)
    g=learner.normalize(recipe['definition'],recipe['market'])
    p,names=learner.features(pq.ParquetFile(recipe['input']['path']).read(use_threads=False).to_pandas(),g,recipe['market'])
    model,proof=learner.fit_model(p,names,pd.Timestamp(recipe['deploy_as_of']),g)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    path=destination/'model.joblib';joblib.dump(dict(model=model,fields=names,recipe_id=recipe['recipe_id'],proof=proof),path)
    (destination/'recipe.json').write_text(json.dumps(recipe,ensure_ascii=False,indent=2))
    (destination/'deployment.json').write_text(json.dumps(dict(recipe_id=recipe['recipe_id'],model_sha256=model_recipe.file_hash(path),training=proof,no_broker_orders=True,status='offline_retrained'),ensure_ascii=False,indent=2))
    return path

"""Bounded learned-model evolution with causal, quarterly walk-forward evaluation.

Each trial fits real estimators, records its recipe, then releases all weights.
Retraining creates an offline artifact only, never broker orders.
"""
import gc
import json
from pathlib import Path
import hashlib
_LOADED_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


NEURAL_MODELS=('neural_mlp','residual_mlp','lstm','gru','tcn','transformer')
_NEURAL_MODULE=None
_EXECUTION_MODULE=None
_PROGRESS=None


def domains(market):
    if market not in ('kr','us','crypto','timefolio'):raise ValueError('지원하지 않는 연구소')
    return dict(model=['ridge','extra_trees','hist_gradient_boosting',*NEURAL_MODELS],
        feature_set=['price','price_volume','price_risk','all','ohlc','volume'],lookback=[5,10,20,40],
        holding_sessions=[1,3,5] if market=='timefolio' else [5,10,20],selection_count=[5,10,20],
        training_days=[180,365,730,1095],seed=[20261008],regularization=[.1,1.,10.],max_depth=[3,5,8],
        model_size=['compact','large'],sequence_length=[20,60],epochs=[10,30],batch_size=[128,256],learning_rate=[.0003,.001],
        retrain_months=[1,3,6],buy_rule=['top_k','positive_score','top_quantile'],sell_rule=['rebalance','stop_take','signal_exit'],
        position_sizing=['equal','inverse_volatility','score'],stop_loss=[.05,.1,.2],take_profit=[.1,.2,.4],
        max_weight=[.05,.1,.2],gross_exposure=[.5,.8,1.],rebalance_sessions=[1,5,10,20],
        target_kind=['return','risk_adjusted','direction'],news_mode=['off','activity'])


def normalize(g,market):
    limits=domains(market)
    required={'model','feature_set','lookback','holding_sessions','selection_count','training_days','seed','regularization','max_depth'}
    if not isinstance(g,dict) or set(g)-set(limits)-{'engine'} or required-set(g):raise ValueError('학습 설정 필드 누락 또는 미지원')
    if 'engine' in g and g['engine']!='learned_v1':raise ValueError('지원하지 않는 학습 엔진')
    defaults=dict(model_size='compact',sequence_length=20,epochs=10,batch_size=128,learning_rate=.001,
        retrain_months=3,buy_rule='top_k',sell_rule='rebalance',position_sizing='equal',stop_loss=.1,take_profit=.2,
        max_weight=.1,gross_exposure=1.,rebalance_sessions=g['holding_sessions'],target_kind='return',news_mode='off')
    out={}
    for key,values in limits.items():
        value=g.get(key,defaults.get(key))
        if isinstance(value,bool):raise ValueError('잘못된 학습 설정 형식')
        if key=='seed':
            if not isinstance(value,int) or not 1<=value<=2147483647:raise ValueError('학습 seed 범위 오류')
        elif key in ('lookback','holding_sessions','selection_count','training_days','max_depth','sequence_length','epochs','batch_size','retrain_months','rebalance_sessions'):
            allowed=values+([g['holding_sessions']] if key=='rebalance_sessions' else [])
            if not isinstance(value,int) or value not in allowed:raise ValueError('지원하지 않는 학습 설정: '+key)
        elif key in ('regularization','learning_rate','stop_loss','take_profit','max_weight','gross_exposure'):
            if not isinstance(value,(int,float)) or value not in values:raise ValueError('지원하지 않는 숫자 설정: '+key)
        elif not isinstance(value,str) or value not in values:raise ValueError('지원하지 않는 학습 설정: '+key)
        out[key]=value
    if out['model'] not in NEURAL_MODELS:
        for key in ('model_size','sequence_length','epochs','batch_size','learning_rate'):out.pop(key)
    out['engine']='learned_v1'
    return out


def features(prices,g,market,news_path=None):
    import numpy as np
    import pandas as pd
    p=prices.sort_values(['symbol','date']).reset_index(drop=True).copy()
    grouped=p.groupby('symbol',sort=False);w=g['lookback'];previous=grouped.close.shift(1)
    p['return_1']=grouped.close.pct_change(fill_method=None)
    p['return_w']=grouped.close.pct_change(w,fill_method=None)
    p['range']=(p.high-p.low)/p.close;p['body']=(p.close-p.open)/p.open
    p['volatility']=p.groupby('symbol').return_1.transform(lambda s:s.rolling(w).std())
    p['trend']=p.close/grouped.close.transform(lambda s:s.rolling(w).mean())-1
    p['volume_ratio']=p.volume/grouped.volume.transform(lambda s:s.rolling(w).mean())-1
    p['volume_change']=grouped.volume.pct_change(fill_method=None)
    p['open_gap']=p.open/previous-1;p['high_move']=p.high/previous-1;p['low_move']=p.low/previous-1
    value=p.volume if market=='crypto' else p.volume*p.close
    p['adv20']=value.groupby(p.symbol).transform(lambda s:s.rolling(20).mean())
    p['volume_rank']=p.groupby('date').adv20.rank(pct=True)
    names=['return_1','return_w','range','body','trend']
    if g['feature_set'] in ('price_volume','all'):names+=['volume_ratio']
    if g['feature_set'] in ('price_risk','all'):names+=['volatility']
    if g['feature_set']=='ohlc':names=['return_1','return_w','open_gap','high_move','low_move','body','range']
    if g['feature_set']=='volume':names=['volume_ratio','volume_change','volume_rank','return_1']
    if g['feature_set']=='all':names+=['open_gap','high_move','low_move','volume_rank','volume_change']
    if g.get('news_mode','off')!='off':
        import pyarrow.parquet as pq
        if news_path is None:
            from .feature_sources import prepare_news
            news_path=prepare_news(market)
        news=pq.ParquetFile(news_path).read(use_threads=False).to_pandas()
        p=p.merge(news,on='date',how='left',validate='many_to_one').sort_values(['symbol','date']).reset_index(drop=True)
        grouped=p.groupby('symbol',sort=False);names+=['news_count','news_count_7']
    h=g['holding_sessions'];entry=grouped.open.shift(-1);exit_price=grouped.open.shift(-h-1)
    p['target']=exit_price/entry-1
    if g.get('target_kind')=='direction':p['target']=np.sign(p.target)
    elif g.get('target_kind')=='risk_adjusted':p['target']=p.target/(p.volatility.clip(lower=.001)*np.sqrt(h))
    p['label_end']=grouped.date.shift(-h-1)
    calendar=pd.DatetimeIndex(sorted(p.date.unique()));ordinal=pd.Series(calendar.get_indexer(p.date),index=p.index)
    history=max(w,20)
    if g['model'] in NEURAL_MODELS:
        steps=g['sequence_length'];base=p[names].astype('float32');lagged=[]
        for lag in range(steps-1,-1,-1):
            values=base.groupby(p.symbol,sort=False).shift(lag) if lag else base
            lagged.append(values.rename(columns={name:f'{name}_lag{lag}' for name in names}))
        wide=pd.concat(lagged,axis=1);names=list(wide.columns);p=pd.concat([p,wide],axis=1);history+=steps-1
        del lagged,wide,base,values
        gc.collect()
    past=ordinal-ordinal.groupby(p.symbol).shift(history)
    future=ordinal.groupby(p.symbol).shift(-h-1)-ordinal
    p['ready']=p.eligible.astype(bool)&past.eq(history)&np.isfinite(p[names]).all(axis=1)
    executable=grouped.tradable_buy.shift(-1).eq(True)&grouped.tradable_sell.shift(-h-1).eq(True)
    p.loc[~future.eq(h+1)|~executable,'target']=np.nan
    return p,names


def estimator(g,n_features=None):
    if g['model'] in NEURAL_MODELS:
        module=_NEURAL_MODULE
        if module is None:
            from . import neural_models as module
        return module.NeuralRegressor(model=g['model'],input_features=n_features//g['sequence_length'],
            sequence_length=g['sequence_length'],model_size=g['model_size'],epochs=g['epochs'],
            batch_size=g['batch_size'],learning_rate=g['learning_rate'],seed=g['seed'])
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
    if g['model'] not in NEURAL_MODELS and len(rows)>30000:rows=np.sort(np.random.default_rng(g['seed']).choice(rows,30000,replace=False))
    return rows


def fit_model(p,names,first,g):
    import time
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    started=time.monotonic();rows=training_rows(p,first,g);model=estimator(g,len(names));validation=[]
    if g['model'] in NEURAL_MODELS:
        days=sorted(p.iloc[rows].date.unique());split=pd.Timestamp(days[max(1,int(len(days)*.85))])
        validation=rows[p.iloc[rows].date.ge(split).to_numpy()]
        rows=rows[p.iloc[rows].label_end.lt(split).to_numpy()]
        if len(rows)<100 or len(validation)<20:raise ValueError('시간 분리 신경망 학습·검증 표본 부족')
        def epoch_progress(record):
            if _PROGRESS:_PROGRESS(dict(phase='training',model=g['model'],quarter=str(first.date()),parameters=model.parameter_count_,epochs=g['epochs'],train_rows=len(rows),validation_rows=len(validation),**record))
        model.epoch_callback=epoch_progress
        model.fit(p.iloc[rows][names].to_numpy(dtype=np.float32),p.iloc[rows].target.to_numpy(),
                  validation_data=(p.iloc[validation][names].to_numpy(dtype=np.float32),p.iloc[validation].target.to_numpy()))
    else:
        with threadpool_limits(limits=1):model.fit(p.iloc[rows][names].to_numpy(),p.iloc[rows].target.to_numpy())
    proof=dict(train_start=str(p.iloc[rows].date.min().date()),train_end=str(p.iloc[rows].date.max().date()),max_label_end=str(p.iloc[rows].label_end.max().date()),model_as_of=str(first.date()),train_rows=len(rows),
               input_fields=len(names),fit_seconds=time.monotonic()-started,model=g['model'],validation_rows=len(validation))
    if len(validation):
        proof.update(neural=model.training_proof_,validation_start=str(p.iloc[validation].date.min().date()),
            validation_end=str(p.iloc[validation].date.max().date()),max_validation_label_end=str(p.iloc[validation].label_end.max().date()))
        assert pd.Timestamp(proof['max_label_end'])<pd.Timestamp(proof['validation_start'])
        assert pd.Timestamp(proof['max_validation_label_end'])<first
    assert pd.Timestamp(proof['max_label_end'])<first
    return model,proof


def walk_forward(p,names,g,start,end):
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from .evaluation import periods
    first=pd.Timestamp(start);final=pd.Timestamp(end)+pd.Timedelta(days=1)
    spans=periods(start,end);os_start=pd.Timestamp(spans['os']['start']);ros_start=pd.Timestamp(spans['ros']['start'])
    steps=g.get('retrain_months',3);edges=[];current=first
    while current<ros_start:
        edges.append(current);current+=pd.DateOffset(months=steps)
    bounds=sorted(set([first,final,*edges,*[x for x in (os_start,ros_start) if first<x<final]]))
    signals=[];proofs=[]
    for first,stop in zip(bounds[:-1],bounds[1:]):
        model,proof=fit_model(p,names,first,g)
        test=p[p.ready&p.date.ge(first)&p.date.lt(stop)].copy()
        if test.empty:raise ValueError('순차 평가 표본 부족')
        with threadpool_limits(limits=1):test['score']=model.predict(test[names].to_numpy())
        signals.append(test[['date','symbol','score','adv20','volatility']]);proof.update(test_start=str(first.date()),test_end_exclusive=str(stop.date()));proofs.append(proof)
        del model,test;gc.collect()
    return pd.concat(signals,ignore_index=True),proofs


def account(prices,signals,g,market,start,end):
    module=_EXECUTION_MODULE
    if module is None:
        from . import execution as module
    return module.account(prices,signals,g,market,start,end,headroom=timefolio_headroom if market=='timefolio' else None)


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
    # A three-year count of low-turnover weeks is not one contest's violation
    # count. The shared validator restricts the check to the actual edition.
    from .contest_validation import assess
    from .contest_rules_evidence import profile
    return assess(case,rules=profile())


def evaluate(candidate,destination):
    from .period import using_window,window
    value=candidate.get('evaluation_window')
    span=json.loads(value) if isinstance(value,str) else value or window()
    global _PROGRESS,_NEURAL_MODULE,_EXECUTION_MODULE
    previous=(_NEURAL_MODULE,_EXECUTION_MODULE)
    try:
        with using_window(*span):return _evaluate(candidate,destination)
    finally:
        _PROGRESS=None
        _NEURAL_MODULE,_EXECUTION_MODULE=previous


def _evaluate(candidate,destination):
    import time
    import pandas as pd
    from . import learning_input,model_recipe
    from .period import window,require_complete
    from .metrics import ledger
    from .evaluation import PROTOCOL,performance,segment_metrics
    global _PROGRESS,_NEURAL_MODULE,_EXECUTION_MODULE
    started=time.monotonic();destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    market=candidate['market'];definition=candidate['definition'];g=normalize(json.loads(definition) if isinstance(definition,str) else definition,market)
    code=model_recipe.code_hashes()
    if code['learning.py']!=_LOADED_SHA256:raise ValueError('실험 시작 중 코드 변경 감지; 새 버전으로 다시 실행해 주세요.')
    model_recipe.preserve_code()
    # Bind this trial to the archived dependencies before long-running fits.
    # A deployment while training cannot change its later execution policy.
    _EXECUTION_MODULE=model_recipe.saved_component({'code':code},'execution.py')
    _NEURAL_MODULE=model_recipe.saved_component({'code':code},'neural_models.py') if g['model'] in NEURAL_MODELS else None
    prices=learning_input.prepare(market)
    inputs=learning_input.descriptor(market)
    auxiliary=[]
    if g.get('news_mode')!='off':
        from .feature_sources import news_descriptor
        auxiliary.append(news_descriptor(market))
    def progress(value):
        payload=dict(value,elapsed_seconds=time.monotonic()-started)
        temp=destination/'progress.tmp';temp.write_text(json.dumps(payload,ensure_ascii=False));temp.replace(destination/'progress.json')
        if 'epoch' in value:
            from .store import event
            event('training',dict(job=candidate.get('id'),market=market,message=f"{g['model']} {value['quarter']} · epoch {value['epoch']}/{g['epochs']} · {value['parameters']:,}계수 · {value['train_rows']:,}행"))
    _PROGRESS=progress;progress(dict(phase='features',model=g['model']))
    p,names=features(prices,g,market,auxiliary[0]['path'] if auxiliary else None);start,end=window()
    signals,proofs=walk_forward(p,names,g,start,end)
    case,stale=account(p,signals,g,market,start,end);require_complete([r['date'] for r in case['daily']],market)
    books=ledger(case,start,end);metrics=segment_metrics(books,start,end,36)
    recipe=model_recipe.seal(dict(version=2,market=market,definition=g,fields=names,input=inputs,auxiliary_inputs=auxiliary,libraries=model_recipe.environment(neural=g['model'] in NEURAL_MODELS),code=code,
      evaluation_start=start,evaluation_end=end,evaluation_protocol=PROTOCOL,selection_scope='os',folds=proofs,
      deploy_as_of=str((pd.Timestamp(end)+pd.Timedelta(days=1)).date()),weights_retained=False,
      execution=dict(signal='session close',fill='next session open',long_only=True,config={k:g[k] for k in ('buy_rule','sell_rule','position_sizing','max_weight','gross_exposure','rebalance_sessions','stop_loss','take_profit')},adv_participation=.01),target=g['target_kind']))
    report=dict(title=candidate.get('title',g['model']),family='인공지능 학습 · '+g['model'],market=market,owner_id=candidate.get('user_id'),definition=g,model_proofs=proofs,recipe=recipe,evaluation_months=36,
      evaluation_window=[start,end],evaluation_protocol=PROTOCOL,performance=performance(books,start,end),selection_scope='os',
      training_summary=dict(model=g['model'],model_size=g.get('model_size','classical'),parameters=max((x.get('neural',{}).get('parameter_count',0) for x in proofs),default=0),
                            folds=len(proofs),seconds=time.monotonic()-started,max_train_rows=max(x['train_rows'] for x in proofs),fields=len(names),device='cpu'),
      cases=[case],no_broker_orders=True,independent_holdout=False,weights_retained=False,competition_compliance_verified=False,
      limitations=['IS 24개월 · OS 9개월 선발 · ROS 3개월은 탐색·선발에 미사용','OS는 반복 탐색용이며 독립 검증 아님; ROS도 과거 노출 가능성이 있어 독립 인증하지 않음',
        '현재 보유 데이터 유니버스의 생존·수집 선택 편향','기업행사·상장폐지·체결·대회 세부 규칙 미감사','롱 전용; 뉴스는 충분한 시점 정렬 이력이 확보된 경우만 사용',f'가격 누락 보유 종목을 마지막 종가로 평가한 종목일 {stale}회'],metrics=metrics)
    if market=='timefolio':
        from .contest_validation import report_assessment
        report['contest_validation']=report_assessment(report)
        report['contest_constraints']=report['contest_validation']
        report['competition_compliance_verified']=report['contest_validation']['competition_compliance_verified']
    (destination/'recipe.json').write_text(json.dumps(recipe,ensure_ascii=False,allow_nan=False,indent=2))
    path=destination/'review.json';path.write_text(json.dumps(report,ensure_ascii=False,allow_nan=False));progress(dict(phase='completed',**report['training_summary']));_PROGRESS=None;return path


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
    raw=pq.ParquetFile(recipe['input']['path']).read(use_threads=False).to_pandas()
    auxiliary=recipe.get('auxiliary_inputs',[])
    p,names=learner.features(raw,g,recipe['market'],auxiliary[0]['path']) if auxiliary else learner.features(raw,g,recipe['market'])
    model,proof=learner.fit_model(p,names,pd.Timestamp(recipe['deploy_as_of']),g)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    path=destination/'model.joblib';joblib.dump(dict(model=model,fields=names,recipe_id=recipe['recipe_id'],proof=proof),path)
    (destination/'recipe.json').write_text(json.dumps(recipe,ensure_ascii=False,indent=2))
    (destination/'deployment.json').write_text(json.dumps(dict(recipe_id=recipe['recipe_id'],model_sha256=model_recipe.file_hash(path),training=proof,no_broker_orders=True,status='offline_retrained'),ensure_ascii=False,indent=2))
    return path

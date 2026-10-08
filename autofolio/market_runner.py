"""Causal, bounded development backtests on a frozen 36-month input snapshot."""
import importlib.util
import json
import sys
from .config import RUNS
from .store import connect,event
from . import research
from .period import window,require_complete

BASELINES={'momentum','reversal','trend_momentum','lowvol_momentum'}


def load_study():
    script=research.STUDY/'_workspace/2026-10-08_chart_vision_screen_v1/run_chart_image_screen.py'
    spec=importlib.util.spec_from_file_location('qis_study_images',script)
    study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
    return study


def price_features(prices):
    """All information is available at the signal day's close."""
    import numpy as np
    import pandas as pd
    p=prices.copy()
    grouped=p.groupby('symbol',sort=False)
    p['r20']=grouped.close.pct_change(20,fill_method=None)
    p['r5']=grouped.close.pct_change(5,fill_method=None)
    p['r1']=grouped.close.pct_change(fill_method=None)
    p['vol20']=p.groupby('symbol',sort=False).r1.transform(lambda s:s.rolling(20).std())
    p['mean60']=grouped.close.transform(lambda s:s.rolling(60).mean())
    value=p[['open','high','low','close']].mean(axis=1)*p.volume
    p['adv20']=value.groupby(p.symbol,sort=False).transform(lambda s:s.rolling(20).mean())
    p['execution_adv20']=p.adv20
    # Require contiguous exchange sessions, rather than concatenating across gaps.
    all_dates=pd.DatetimeIndex(sorted(p.date.unique()))
    ordinal=pd.Series(all_dates.get_indexer(p.date),index=p.index)
    span=ordinal-ordinal.groupby(p.symbol,sort=False).shift(60)
    p['history_ready']=span.eq(60)&np.isfinite(p[['r20','r5','vol20','mean60','adv20']]).all(axis=1)&p.vol20.gt(0)
    p['eligible']=p.eligible.astype(bool)&p.history_ready
    usable=p[p.eligible]
    breadth=usable.assign(above=usable.close>usable.mean60).groupby('date').above.mean()
    p['regime_on']=p.date.map(breadth).fillna(0).ge(.5)
    return p


def baseline_signals(prices,definition):
    import numpy as np
    name=definition['representation']
    scores={'momentum':prices.r20,'reversal':-prices.r5,
            'trend_momentum':prices.r20,'lowvol_momentum':prices.r20/prices.vol20.clip(lower=.001)}
    signal=prices[['date','symbol','sector','adv20','eligible','regime_on']].copy()
    signal['score']=scores[name]
    if name=='trend_momentum':signal['eligible']&=prices.close>prices.mean60
    signal=signal[np.isfinite(signal.score)&np.isfinite(signal.adv20)]
    if definition['mode']=='regime_gate':signal['eligible']&=signal.regime_on
    return signal


def chart_signals(prices,g,study,candidate):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    start,end=window()
    # Module instance is local to this isolated process. Never edit source research.
    study.HORIZON=g['holding_sessions']
    y,ends,history=study.add_labels_and_history(prices)
    eligible=prices.eligible.to_numpy()&history
    bounds=pd.date_range(pd.Timestamp(start),pd.Timestamp(end)+pd.Timedelta(days=1),freq='QS')
    if not len(bounds) or bounds[0]!=pd.Timestamp(start):bounds=bounds.insert(0,pd.Timestamp(start))
    final=pd.Timestamp(end)+pd.Timedelta(days=1)
    if bounds[-1]!=final:bounds=bounds.append(pd.DatetimeIndex([final]))
    signals=[];proofs=[]
    for first,stop in zip(bounds[:-1],bounds[1:]):
        train=np.flatnonzero(eligible&np.isfinite(y)&(ends<first.value)&(prices.date>=first-pd.Timedelta(days=730)))
        if not len(train):raise ValueError('분기 학습 표본 부족')
        rng=np.random.default_rng(20261008)
        groups=prices.iloc[train].groupby('date').indices
        train=np.concatenate([train[np.sort(rng.choice(ix,min(250,len(ix)),replace=False))] for ix in groups.values()])
        test=np.flatnonzero(eligible&(prices.date>=first)&(prices.date<stop))
        if not len(test):raise ValueError('분기 평가 표본 부족')
        event('training',dict(job=candidate['id'],quarter=str(first.date()),market=candidate['market']))
        print(json.dumps(dict(quarter=str(first.date()),train=len(train),test=len(test))),flush=True)
        feats=study.image_features(prices,np.concatenate([train,test]))
        names=list(feats) if g['representation']=='image_rank_ensemble' else [g['representation']]
        dates=prices.iloc[train].date.to_numpy();target=y[train].astype(float)
        target-=pd.Series(target).groupby(pd.Series(dates)).transform('mean').to_numpy()
        counts=pd.Series(dates).value_counts();weights=np.array([1/counts[d] for d in dates])
        predictions=[]
        for name in names:
            scaler=StandardScaler().fit(feats[name][:len(train)])
            model=Ridge(alpha=g['ridge_alpha']).fit(scaler.transform(feats[name][:len(train)]),target,sample_weight=weights)
            pred=model.predict(scaler.transform(feats[name][len(train):]))
            predictions.append(study.rank_score(prices.iloc[test].date,pred))
        sig=prices.iloc[test][['date','symbol','sector','adv20','regime_on']].copy()
        sig['score']=np.mean(predictions,axis=0);sig['eligible']=True;sig['model_as_of']=first
        if g['mode']=='regime_gate':sig['eligible']&=sig.regime_on
        signals.append(sig)
        proofs.append(dict(test_start=str(first.date()),test_end_exclusive=str(stop.date()),max_label_end=str(pd.Timestamp(int(ends[train].max())).date()),train_rows=len(train)))
        # Do not retain fitted weights or repeated feature matrices between quarters.
        del feats,scaler,model
    return pd.concat(signals,ignore_index=True),proofs


def stock_evaluate(candidate,destination):
    import pandas as pd
    from .research_input import prepare,status as input_status
    from .metrics import statistics_for,ledger
    study=load_study();market=candidate['market'];g=research.normalize(json.loads(candidate['definition']),market)
    start,end=window();prices=prepare(market)
    prices=price_features(prices)
    days=sorted(prices.loc[(prices.date>=pd.Timestamp(start))&(prices.date<=pd.Timestamp(end)),'date'].dt.strftime('%Y%m%d').unique())
    require_complete(days,market)
    if g['representation'] in BASELINES:signals=baseline_signals(prices,g);proofs=[]
    else:signals,proofs=chart_signals(prices,g,study,candidate)
    signals=signals[(signals.date>=pd.Timestamp(start))&(signals.date<=pd.Timestamp(end))]
    evaluation=prices[(prices.date>=pd.Timestamp(start))&(prices.date<=pd.Timestamp(end))]
    cost=.002 if market=='kr' else .001
    config=study.Config(market=market.upper(),initial_cash=1e8 if market=='kr' else 1e5,
        selection_count=g['selection_count'],holding_sessions=g['holding_sessions'],buy_cost=cost,sell_cost=cost,mode='rank_only')
    event('backtesting',dict(job=candidate['id'],start=start,end=end,months=36))
    daily,trades,_=study.backtest(signals,evaluation,config)
    daily['date']=pd.to_datetime(daily.date).dt.strftime('%Y%m%d')
    require_complete(daily.date.tolist(),market)
    transactions=[dict(date=pd.Timestamp(r.date).strftime('%Y%m%d'),code=r.symbol,side=r.side.lower(),qty=r.shares,price=r.price,fee=r.fee) for r in trades.itertuples()]
    case=dict(initial_cash=config.initial_cash,daily=daily.to_dict('records'),trades=transactions)
    source=input_status(market)
    report=dict(title=candidate['title'],family='학술제 확장 · '+('가격·위험 비교' if g['representation'] in BASELINES else '차트 이미지'),
        market=market,owner_id=candidate['user_id'],definition=g,model_proofs=proofs,evaluation_months=36,cases=[case],
        no_broker_orders=True,independent_holdout=False,input_snapshot=source,
        limitations=['최근 36개월 개발 실험이며 반복 탐색으로 독립 검증이 아님',
        '현재 보유 종목 집합을 사용하므로 생존·수집 선택 편향이 있음',
        '수정 시세·유동성의 근사 체결이며 완전한 기업행사·상장폐지 계좌 감사가 아님',
        '학술제 원본의 동결된 검증 구간과 별개인 QuantInSight 확장 실험'])
    report['metrics']=statistics_for(ledger(case,start,end))
    (destination/'review.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False))
    return destination/'review.json'


def run(identity):
    with connect() as db:r=db.execute('SELECT * FROM alpha_candidates WHERE id=?',(identity,)).fetchone()
    if not r:raise ValueError('Unknown candidate')
    candidate=dict(r);destination=RUNS/'market_experiments'/identity;destination.mkdir(parents=True,exist_ok=True)
    try:
        if 'model' in json.loads(candidate['definition']):
            from . import learning
            fit=learning.fit_model
            def logged_fit(p,names,first,g):
                event('training',dict(job=identity,market=candidate['market'],quarter=str(first.date()),message=g['model']+' 모델 학습'))
                return fit(p,names,first,g)
            learning.fit_model=logged_fit
            path=learning.evaluate(candidate,destination)
        else:
            if candidate['market'] not in ('kr','us'):raise ValueError('지원하지 않는 시장')
            path=stock_evaluate(candidate,destination)
        from .catalogue import ingest_file
        status,count=ingest_file(path,path.stat())
        if status!='indexed' or count!=1:raise ValueError('36개월 계좌 검증 실패')
        with connect() as db:db.execute("UPDATE alpha_candidates SET status='done',result=?,message='36개월 평가 완료' WHERE id=?",(str(path.resolve()),identity))
        event('market_completed',dict(job=identity,market=candidate['market'],message='36개월 평가 완료'))
    except Exception as exc:
        with connect() as db:db.execute("UPDATE alpha_candidates SET status='failed',message=? WHERE id=?",(str(exc)[:300],identity))
        event('failed',dict(job=identity,error=str(exc)[:300]))
        raise

if __name__=='__main__':run(sys.argv[1])

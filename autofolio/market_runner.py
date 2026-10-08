"""Isolated rolling study adapter. Source studies and collectors stay read-only."""
import importlib.util
import json
import sys
import time
from pathlib import Path
from .config import RUNS
from .store import connect,event
from . import research
from .period import window,require_complete


def stock_evaluate(candidate,destination):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    script=research.STUDY/'_workspace/2026-10-08_chart_vision_screen_v1/run_chart_image_screen.py'
    spec=importlib.util.spec_from_file_location('qis_study_images',script)
    study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
    market=candidate['market'];g=json.loads(candidate['definition']);code=market.upper()
    start,end=window();prices=study.load_prices(code)
    days=sorted(prices.loc[(prices.date>=pd.Timestamp(start))&(prices.date<=pd.Timestamp(end)),'date'].dt.strftime('%Y%m%d').unique())
    require_complete(days,market)
    y,ends,history=study.add_labels_and_history(prices);eligible=study.allowed_rows(prices,code)&history
    prices['adv20']=prices.groupby('symbol',sort=False).apply(lambda d:(d[['open','high','low','close']].mean(axis=1)*d.volume).rolling(20).mean(),include_groups=False).reset_index(level=0,drop=True).reindex(prices.index)
    boundaries=pd.date_range(pd.Timestamp(start),pd.Timestamp(end)+pd.Timedelta(days=1),freq='QS')
    if not len(boundaries) or boundaries[0]!=pd.Timestamp(start):boundaries=boundaries.insert(0,pd.Timestamp(start))
    final=pd.Timestamp(end)+pd.Timedelta(days=1)
    if boundaries[-1]!=final:boundaries=boundaries.append(pd.DatetimeIndex([final]))
    signals=[];proofs=[]
    for first,stop in zip(boundaries[:-1],boundaries[1:]):
        train=np.flatnonzero(eligible&np.isfinite(y)&(ends<first.value))
        # Deterministic sampling depends only on dates/rows, never on their returns.
        rng=np.random.default_rng(20261008)
        groups=prices.iloc[train].groupby('date').indices
        train=np.concatenate([train[np.sort(rng.choice(ix,min(250,len(ix)),replace=False))] for ix in groups.values()])
        test=np.flatnonzero(eligible&(prices.date>=first)&(prices.date<stop))
        if not len(train) or not len(test):raise ValueError('학습·평가 표본 부족')
        event('training',dict(job=candidate['id'],quarter=str(first.date()),market=market))
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
            np.savez(destination/f'{first:%Y%m}_{name}.npz',mean=scaler.mean_,scale=scaler.scale_,coef=model.coef_,intercept=model.intercept_)
        score=np.mean(predictions,axis=0)
        sig=prices.iloc[test][['date','symbol','sector','adv20']].copy();sig['score']=score;sig['eligible']=True;sig['model_as_of']=first
        signals.append(sig)
        proofs.append(dict(test_start=str(first.date()),test_end_exclusive=str(stop.date()),max_label_end=str(pd.Timestamp(int(ends[train].max())).date()),train_rows=len(train)))
    all_signals=pd.concat(signals,ignore_index=True)
    config=study.Config(market=code,initial_cash=1e8 if market=='kr' else 1e5,selection_count=g['selection_count'],holding_sessions=g['holding_sessions'],buy_cost=.002 if market=='kr' else .001,sell_cost=.002 if market=='kr' else .001,mode=g['mode'])
    daily,trades,_=study.backtest(all_signals,prices[(prices.date>=pd.Timestamp(start))&(prices.date<=pd.Timestamp(end))],config)
    daily['date']=pd.to_datetime(daily.date).dt.strftime('%Y%m%d')
    require_complete(daily.date.tolist(),market)
    transactions=[dict(date=str(r.date).replace('-',''),code=r.symbol,side=r.side.lower(),qty=r.shares,price=r.price,fee=r.fee) for r in trades.itertuples()]
    report=dict(title=candidate['title'],family='학술제 차트 이미지',market=market,owner_id=candidate['user_id'],definition=g,model_proofs=proofs,
                evaluation_months=36,cases=[dict(initial_cash=config.initial_cash,daily=daily.to_dict('records'),trades=transactions)],
                no_broker_orders=True,independent_holdout=False,source_reference=str(research.SOURCES[market]),
                limitations=['학술제 방법을 최근 36개월 순차 학습으로 재평가한 개발 결과','검증 시세의 기업행사·섹터·체결 한계 유지'])
    (destination/'review.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False))
    return destination/'review.json'


def run(identity):
    with connect() as db:r=db.execute('SELECT * FROM alpha_candidates WHERE id=?',(identity,)).fetchone()
    if not r:raise ValueError('Unknown candidate')
    candidate=dict(r);destination=RUNS/'market_experiments'/identity;destination.mkdir(parents=True,exist_ok=True)
    try:
        if candidate['market'] not in ('kr','us'):raise ValueError('크립토 36개월 순차 모델 검증이 필요합니다.')
        path=stock_evaluate(candidate,destination)
        from .catalogue import ingest_file
        status,count=ingest_file(path,path.stat())
        if status!='indexed' or count!=1:raise ValueError('36개월 계좌 검증 실패')
        with connect() as db:db.execute("UPDATE alpha_candidates SET status='done',result=?,message='36개월 평가 완료' WHERE id=?",(str(path),identity))
    except Exception as exc:
        with connect() as db:db.execute("UPDATE alpha_candidates SET status='failed',message=? WHERE id=?",(str(exc)[:300],identity))
        raise

if __name__=='__main__':run(sys.argv[1])

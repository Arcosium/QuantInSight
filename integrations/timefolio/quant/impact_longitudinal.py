"""Deadline-bounded, day-clustered study of archived Bybit spot books/trades.

No order endpoints. The 100 ms public-trade bucket is an observed flow event,
not an identified investor order. Long responses are descriptive associations.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant.impact_extension import refill_path

GRID = np.unique(np.r_[np.arange(1, 101)/10, np.arange(11, 61), np.arange(65, 301, 5)])
POINTS = [.1, 1, 5, 10, 30, 60, 300]
POINT_INDEX = {h: int(np.flatnonzero(np.isclose(GRID, h))[0]) for h in POINTS}
RULES = {
    'burst_ms': 100, 'direction_dominance': .8, 'large_quantile': .75,
    'short_spacing_ms': 11000, 'long_spacing_ms': 301000,
    'short_seconds': 10, 'long_seconds': 300, 'lookback_seconds': 10,
    'refill_classification_seconds': 1, 'high_refill_ratio': 1,
    'depletion_fraction': .2, 'depth_recovery_fraction': .9,
    'recovery_sustain_seconds': .3, 'matching_quantity_ratio': 1.1,
    'matching_time_ms': 600000, 'bootstrap_repetitions': 2000, 'seed': 928,
    'clock': 'book cts / trade timestamp, milliseconds',
    'archive_update_policy': 'forward-hold between contiguous delta updates; epoch resets cannot be crossed',
    'inference': 'UTC-day cluster bootstrap; intervals are descriptive pointwise intervals',
}


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def bursts(trades):
    t = trades.copy()
    t['bucket'] = t.ts // 100 * 100
    t['notional'] = t.quantity*t.price
    t['buy'] = np.where(t.side == 'Buy', t.quantity, 0.)
    t['sell'] = np.where(t.side == 'Sell', t.quantity, 0.)
    b = t.groupby('bucket', sort=True).agg(quantity=('quantity', 'sum'),
            notional=('notional', 'sum'), buy=('buy', 'sum'), sell=('sell', 'sum'))
    b['dominance'] = b[['buy', 'sell']].max(axis=1)/b.quantity
    b['sign'] = np.where(b.buy >= b.sell, 1, -1)
    b['dominant_qty'] = b[['buy', 'sell']].max(axis=1)
    return b[b.dominance >= RULES['direction_dominance']].reset_index()


def training_notional_threshold(root, days, symbol):
    """Use only calibration dates; validation dates never set thresholds."""
    values = []
    for day in days:
        frame = pd.read_csv(root/f'{symbol}_{day}.csv.gz')
        required = {'timestamp','price','volume','side','rpi','id'}
        if not required.issubset(frame):
            raise ValueError(f'{day}: unexpected trade schema')
        if frame.id.duplicated().any():
            raise ValueError(f'{day}: duplicate trade ids in calibration input')
        day_start = int(pd.Timestamp(day, tz='UTC').timestamp()*1000)
        if not frame.side.isin(['buy','sell']).all() or not frame.rpi.isin([0,1]).all():
            raise ValueError(f'{day}: unknown side or RPI flag')
        numeric = frame[['timestamp','price','volume']].to_numpy(float)
        if not np.isfinite(numeric).all() or (frame.price <= 0).any() or (frame.volume <= 0).any():
            raise ValueError(f'{day}: invalid trade values')
        frame = frame[(frame.rpi == 0) & (frame.timestamp >= day_start)
                      & (frame.timestamp < day_start+86400000)].copy()
        frame = frame.rename(columns={'timestamp':'ts','volume':'quantity'})
        frame['side'] = frame.side.str.title()
        values.append(bursts(frame).notional.to_numpy())
    combined = np.concatenate(values)
    return float(np.quantile(combined, .75)), len(combined)


def sustained_start(values, threshold, above=False, step=.1):
    """Onset of >=0.3 s sustained condition, first tested at t=0.2 s.

    NaN breaks a run. Four samples 0.1 s apart span 0.3 s, not 0.4 s.
    """
    a = np.asarray(values, float)
    ok = np.isfinite(a) & ((a >= threshold) if above else (a <= threshold))
    if len(ok) < 5:
        return None
    run = ok[:-3] & ok[1:-2] & ok[2:-1] & ok[3:]
    run[0] = False
    ix = np.flatnonzero(run)
    return float((ix[0]+1)*step) if len(ix) else None


def same_price_record(books, times, pre, last, event_ms, side_index, trade_times,
                      trade_prices, trade_qty, trade_signs, sign):
    """Fixed individual pre-event best price; future prices are never relabelled."""
    price = float(books[pre, side_index, 0, 0])
    start = np.searchsorted(trade_times, event_ms, side='left')
    end = np.searchsorted(trade_times, event_ms+100, side='left')
    same_initial = (trade_signs[start:end] == sign) & np.isclose(trade_prices[start:end], price, rtol=0, atol=1e-8)
    consumed = float(trade_qty[start:end][same_initial].sum())
    if consumed <= 0:
        return {'refill_eligible':False, 'refill_exclusion':'no_initial_trade_at_pre_best'}
    ix = np.arange(pre, last+1)
    level = np.asarray(books[pre:last+1, side_index])
    coverage = (price <= level[:, -1, 0]+1e-8) if side_index == 1 else (price >= level[:, -1, 0]-1e-8)
    if not coverage.all():
        return {'refill_eligible':False, 'refill_exclusion':'original_price_outside_retained_depth'}
    quantities = np.where(np.isclose(level[:,:,0],price,rtol=0,atol=1e-8),level[:,:,1],0).sum(axis=1)
    a = np.searchsorted(trade_times, times[pre], side='right')
    b = np.searchsorted(trade_times, times[last], side='right')
    selected = (trade_signs[a:b] == sign) & np.isclose(trade_prices[a:b],price,rtol=0,atol=1e-8)
    executions = np.column_stack((trade_times[a:b][selected], trade_qty[a:b][selected]))
    first_trade = int(trade_times[start:end][same_initial][0])
    path = refill_path(times[ix], quantities, executions, first_trade)
    one = int(np.searchsorted(times[ix],event_ms+1000,side='right')-1)
    ratio1 = float(path['increase'][one]/consumed)
    held = level[:one+1,0,0]
    broken = bool((held > price+1e-8).any()) if side_index == 1 else bool((held < price-1e-8).any())
    filled_one = float(executions[executions[:,0] <= event_ms+1000,1].sum())
    return {
        'refill_eligible':True, 'refill_exclusion':'', 'anchor_price':price,
        'consumed_initial':consumed, 'anchor_pre_quantity':float(quantities[0]),
        'refill_ratio_1s':ratio1, 'refill_ratio_10s':float(path['increase'][-1]/consumed),
        'refill_adjusted_10s':float(path['adjusted'][-1]/consumed),
        'refill_first_seconds':None if path['first_refill_ms'] is None else (path['first_refill_ms']-first_trade)/1000,
        'refill_cycles':path['cycles'], 'refill_high':ratio1 >= 1,
        'absorption_candidate':ratio1 >= 1 and filled_one >= quantities[0] and not broken,
    }


def extract_day(data, day, notional_threshold, tick_size=.01):
    """Short events every >=11 s; separate >=301 s long-window cohort.

    Long eligibility does not remove otherwise valid short-window observations.
    Update age sensitivity is recorded separately from known sequence gaps.
    """
    times, books = data['times'], data['books']
    epochs = data['epochs']
    if len(times) < 2 or np.any(np.diff(times) < 0):
        raise ValueError('ordered matching-engine book timestamps required')
    trades = data['trades'].sort_values('ts',kind='stable')
    tt=trades.ts.to_numpy(np.int64); tp=trades.price.to_numpy(float)
    tq=trades.quantity.to_numpy(float); sg=np.where(trades.side == 'Buy',1,-1)
    buy_cs=np.r_[0,np.cumsum(np.where(sg==1,tq,0))]
    sell_cs=np.r_[0,np.cumsum(np.where(sg==-1,tq,0))]
    mid=(books[:,0,0,0]+books[:,1,0,0])/2
    spread=(books[:,1,0,0]-books[:,0,0,0])
    depths=np.asarray(books[:,:,:5,1].sum(axis=2))
    start_ms=int(pd.Timestamp(day,tz='UTC').timestamp()*1000)
    gtimes=np.arange(start_ms,start_ms+86400000,100,dtype=np.int64)
    gi=np.searchsorted(times,gtimes,side='right')-1
    safe=np.maximum(gi,0)
    gmid=np.asarray(mid[safe]); gepoch=np.asarray(epochs[safe])
    gage=gtimes-times[safe]
    candidate=bursts(trades)
    directional_count=len(candidate)
    candidate=candidate[candidate.notional >= notional_threshold]
    rows=[]; curves=[]; post_curves=[]; exclusions=Counter()
    last_short=-10**18; last_long=-10**18
    for b in candidate.itertuples():
        t=int(b.bucket); k=(t-start_ms)//100
        pre=int(np.searchsorted(times,t,side='left')-1)
        if pre < 0 or k < 100 or k+100 >= len(gtimes) or t+10000 > times[-1]:
            exclusions['short_or_history_day_boundary']+=1; continue
        ep=epochs[pre]
        if gi[k-100] < 0 or gepoch[k-100]!=ep or gepoch[k+100]!=ep:
            exclusions['short_or_history_epoch_boundary']+=1; continue
        if t-last_short < RULES['short_spacing_ms']:
            exclusions['short_overlap']+=1; continue
        last_short=t
        sign=int(b.sign); side=1 if sign==1 else 0
        future=k+(GRID*10).round().astype(int)
        valid=(future < len(gtimes)) & (t+GRID*1000 <= times[-1])
        future=np.minimum(future,len(gtimes)-1)
        valid &= gepoch[future]==ep
        impact=sign*(gmid[future]-mid[pre])/mid[pre]*1e4
        impact[~valid]=np.nan
        dense10=sign*(gmid[k+1:k+101]-mid[pre])/mid[pre]*1e4
        valid300=bool(valid[-1])
        long_primary=valid300 and t-last_long >= RULES['long_spacing_ms']
        if long_primary:last_long=t
        hidx=gi[k-100:k]
        hprice=mid[hidx]
        volatility=float(np.std(np.diff(np.log(hprice)))*1e4)
        at0=np.searchsorted(tt,t-10000,side='left'); at1=np.searchsorted(tt,t,side='left')
        activity=float(buy_cs[at1]-buy_cs[at0]+sell_cs[at1]-sell_cs[at0])
        bid,ask=depths[pre]
        obi=float((bid-ask)/(bid+ask))
        row={'day':day,'ts':t,'side':'buy' if sign==1 else 'sell','sign':sign,
             'quantity':float(b.dominant_qty),'notional':float(b.notional),
             'depth':float(depths[pre,side]),'spread_ticks':int(round(spread[pre]/tick_size)),
             'volatility':volatility,'activity':activity,'obi':obi,'directional_obi':sign*obi,
             'valid300':valid300,'long_primary':long_primary,
             'strict_age_10s':bool(max(t-times[pre],gage[k-100:k+101].max()) <= 1000),
             'strict_age_300s':bool(valid300 and max(t-times[pre],gage[k-100:k+3001].max()) <= 1000),
             'initial_positive':bool(dense10[0]>1e-9),
             'half_time_10s':sustained_start(dense10,dense10[0]/2) if dense10[0]>1e-9 else None,
             'return_time_10s':sustained_start(dense10,0) if dense10[0]>1e-9 else None}
        for h,j in POINT_INDEX.items():row[f'impact_{h:g}s']=float(impact[j])
        # Event-flow end means last dominant-side trade within the 100 ms bucket.
        a=np.searchsorted(tt,t,side='left'); c=np.searchsorted(tt,t+100,side='left')
        dominant_times=tt[a:c][sg[a:c]==sign]
        end_t=int(dominant_times[-1])
        targets=end_t+(GRID*1000).round().astype(int)
        post_ix=np.searchsorted(times,targets,side='right')-1
        post_valid=(targets <= times[-1]) & (epochs[post_ix]==ep)
        post=sign*(mid[post_ix]-mid[pre])/mid[pre]*1e4
        post[~post_valid]=np.nan
        row['event_end_ms']=end_t
        # Same-price observation ends at 10 s and is not conditioned on 300 s coverage.
        row.update(same_price_record(books,times,pre,int(gi[k+100]),t,side,tt,tp,tq,sg,sign))
        for h in [10,60,300]:
            target=t+h*1000
            if target > times[-1] or not valid[POINT_INDEX[h]]:
                row[f'opposite_share_{h}s']=None; continue
            a=np.searchsorted(tt,t+100,side='left'); c=np.searchsorted(tt,target,side='right')
            buys=buy_cs[c]-buy_cs[a]; sells=sell_cs[c]-sell_cs[a]
            other=sells if sign==1 else buys
            row[f'opposite_share_{h}s']=float(other/(buys+sells)) if buys+sells else None
            row[f'opposite_present_{h}s']=bool(other>0)
        if long_primary:
            long_ix=gi[k+1:k+3001]
            dense=sign*(mid[long_ix]-mid[pre])/mid[pre]*1e4
            row['half_time_300s']=sustained_start(dense,dense[0]/2) if row['initial_positive'] else None
            row['return_time_300s']=sustained_start(dense,0) if row['initial_positive'] else None
            sp=spread[long_ix]
            row['spread_widened']=bool(sp[0]>spread[pre]+1e-8)
            row['spread_recovery_time']=sustained_start(sp,spread[pre]+1e-8) if row['spread_widened'] else None
            low=float(books[pre,side,:5,0].min()); high=float(books[pre,side,:5,0].max())
            p=np.asarray(books[long_ix,side,:,0]);q=np.asarray(books[long_ix,side,:,1])
            coverage=(p[:,-1]>=high) if side==1 else (p[:,-1]<=low)
            fixed=np.where((p>=low)&(p<=high),q,0).sum(axis=1)/row['depth']
            fixed[~coverage]=np.nan
            row['depth_full_coverage']=bool(coverage.all())
            row['depth_depleted']=bool(np.isfinite(fixed[0]) and fixed[0]<=.8)
            row['depth_recovery_time']=sustained_start(fixed,.9,above=True) if coverage.all() and row['depth_depleted'] else None
        rows.append(row);curves.append(impact);post_curves.append(post)
    return pd.DataFrame(rows),np.asarray(curves),np.asarray(post_curves),{
        'day':day,'book_updates':len(times),'trades':len(trades),'directional_bursts':directional_count,
        'large_candidates':len(candidate),'events':len(rows),'exclusions':dict(exclusions),
        'quality':data['quality'],'start_ms':int(times[0]),'end_ms':int(times[-1]),
    }


def curve_stats(values, days, repetitions=2000):
    """Event-weighted point estimates, whole-day resampling for uncertainty."""
    a=np.asarray(values,float)
    if len(a)==0:return {'n':0,'days':0,'mean':[],'lower':[],'upper':[]}
    if a.ndim==1:a=a[:,None]
    days=np.asarray(days);unique=np.unique(days)
    sums=np.array([np.nansum(a[days==d],axis=0) for d in unique])
    counts=np.array([np.isfinite(a[days==d]).sum(axis=0) for d in unique])
    denom=counts.sum(axis=0)
    mean=np.divide(sums.sum(axis=0),denom,out=np.full(a.shape[1],np.nan),where=denom>0)
    if len(unique)<2:
        lower=upper=np.full(a.shape[1],np.nan)
    else:
        rng=np.random.default_rng(RULES['seed'])
        # Multinomial day counts avoid creating a repetitions x days x horizons tensor.
        weights=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=repetitions)
        num=weights@sums;den=weights@counts
        samples=np.divide(num,den,out=np.full_like(num,np.nan,dtype=float),where=den>0)
        with np.errstate(all='ignore'):
            lower,upper=np.nanpercentile(samples,[2.5,97.5],axis=0)
    return clean_json({'n':len(a),'days':len(unique),'valid_n':denom,'mean':mean,'lower':lower,'upper':upper})


def assign_regimes(frame, calibration_end):
    train=frame.day <= calibration_end
    if not train.any():raise ValueError('no calibration events')
    thresholds={}
    for field,quantiles in [('depth',[.2,.8]),('volatility',[1/3,2/3]),('activity',[1/3,2/3])]:
        lo,hi=frame.loc[train,field].quantile(quantiles).to_numpy()
        frame[field+'_regime']=np.where(frame[field]<=lo,'Low',np.where(frame[field]>=hi,'High','Normal'))
        thresholds[field]={'limits':[float(lo),float(hi)],'quantiles':quantiles}
    frame['spread_regime']=np.where(frame.spread_ticks==1,'1 tick',np.where(frame.spread_ticks==2,'2 tick','3+ tick'))
    frame['obi_bin']=pd.cut(frame.directional_obi,[-1.000001,-.2,.2,1.000001],right=False,labels=['negative','neutral','positive']).astype(str)
    frame['split']=np.where(train,'calibration','validation')
    return thresholds


def matched_pairs(frame, field, low, high, *, obi_control=False):
    """No reuse, same day/direction, <=10 min, quantity ratio <=1.1."""
    pairs=[]
    for (_,side),group in frame.groupby(['day','side'],sort=True):
        a=group[group[field]==low].sort_values('ts')
        b=group[group[field]==high].sort_values('ts')
        used=set(); bt=b.ts.to_numpy();bq=b.quantity.to_numpy();bi=b.index.to_numpy();bo=b.obi_bin.to_numpy()
        for row in a.itertuples():
            left=np.searchsorted(bt,row.ts-600000,'left');right=np.searchsorted(bt,row.ts+600000,'right')
            if left==right:continue
            positions=np.arange(left,right)
            ratios=np.maximum(row.quantity/bq[positions],bq[positions]/row.quantity)
            ok=ratios<=1.1+1e-12
            if obi_control:ok &= bo[positions]==row.obi_bin
            choices=[(abs(np.log(row.quantity/bq[p])),abs(row.ts-bt[p]),int(bi[p])) for p in positions[ok] if int(bi[p]) not in used]
            if choices:
                _,_,index=min(choices);used.add(index);pairs.append((row.Index,index))
    return pairs


def recovery_table(frame, time_column, eligibility):
    pool=frame.loc[eligibility]
    if len(pool)==0:return {'n':0,'recovered_by':{},'median_seconds':None}
    t=pd.to_numeric(pool[time_column],errors='coerce').to_numpy()
    counts={str(h):int((t+.3 <= h+1e-8).sum()) for h in [10,30,60,300]}
    observed=np.sort(t[np.isfinite(t)])
    median=float(observed[int(np.ceil(len(pool)/2))-1]) if len(observed)>=np.ceil(len(pool)/2) else None
    return {'n':len(pool),'recovered_by':counts,'censored_at_300':len(pool)-counts['300'],
            'median_seconds':median,'recovered_only_median_seconds':float(np.median(observed)) if len(observed) else None}


def pair_summary(frame, curves, pairs):
    if not pairs:return {'pairs':0,'days':0,'difference':curve_stats([],[])}
    a=np.array([p[0] for p in pairs]); b=np.array([p[1] for p in pairs])
    result=curve_stats(curves[a]-curves[b],frame.loc[a,'day'].to_numpy())
    ratios=np.maximum(frame.loc[a,'quantity'].to_numpy()/frame.loc[b,'quantity'].to_numpy(),
                      frame.loc[b,'quantity'].to_numpy()/frame.loc[a,'quantity'].to_numpy())
    return {'pairs':len(pairs),'days':result['days'],'difference':result,
            'quantity_ratio_median':float(np.median(ratios))}


def summarize_cohort(frame, curves, post_curves, mask):
    pool=frame.loc[mask]
    def stats(selected, values=curves):
        return curve_stats(values[selected.index.to_numpy()],selected.day.to_numpy())
    long=pool[pool.long_primary]
    refill=pool[pool.refill_eligible]
    observed=refill.refill_first_seconds.notna()
    r={'eligible':len(refill),'excluded':pool.loc[~pool.refill_eligible,'refill_exclusion'].value_counts().to_dict(),
       'observed':int(observed.sum()),'not_observed':int((~observed).sum()),
       'first_refill_median_observed_only':refill.loc[observed,'refill_first_seconds'].median(),
       'repeated_cycles':int((refill.refill_cycles>=2).sum()),
       'absorption_candidates':int(refill.absorption_candidate.fillna(False).sum()),'groups':{},'matched':{}}
    after=curves-curves[:,POINT_INDEX[1],None]
    after[:,GRID<1]=np.nan
    for side in ['buy','sell']:
        for high in [True,False]:
            selected=refill[(refill.side==side)&(refill.refill_high==high)]
            r['groups'][side+('_high' if high else '_low')]={
                'n':len(selected),'median_consumed':selected.consumed_initial.median(),
                'after_1s':stats(selected,after)}
        selected=refill[refill.side==side]
        pairs=matched_pairs(selected,'refill_high',True,False)
        r['matched'][side]=pair_summary(frame,after,pairs)
    regimes={}
    for field in ['depth','volatility','activity','spread']:
        labels=['1 tick','2 tick','3+ tick'] if field=='spread' else ['Low','Normal','High']
        pairing=matched_pairs(pool,field+'_regime',labels[0],labels[-1])
        regimes[field]={'counts':pool[field+'_regime'].value_counts().to_dict(),
                        'groups':{label:stats(pool[pool[field+'_regime']==label]) for label in labels},
                        'matched':pair_summary(frame,curves,pairing)}
    controlled=matched_pairs(pool,'depth_regime','Low','High',obi_control=True)
    cells=[]
    for category in ['negative','neutral','positive']:
        for depth in ['Low','High']:
            selected=pool[(pool.obi_bin==category)&(pool.depth_regime==depth)]
            cells.append({'obi_bin':category,'depth':depth,'n':len(selected),
                          'mean_5s_bps':selected['impact_5s'].mean()})
    positive=long.initial_positive
    half=recovery_table(long,'half_time_300s',positive)
    returned=recovery_table(long,'return_time_300s',positive)
    spread_recovery=recovery_table(long,'spread_recovery_time',long.spread_widened.fillna(False))
    depth_mask=long.depth_full_coverage.fillna(False)&long.depth_depleted.fillna(False)
    depth_recovery=recovery_table(long,'depth_recovery_time',depth_mask)
    return clean_json({'events':len(pool),'days':pool.day.nunique(),
       'direction_counts':pool.side.value_counts().to_dict(),
       'impact_short':stats(pool),'refill':r,'regimes':regimes,'obi_cells':cells,
       'depth_matched_with_obi':pair_summary(frame,curves,controlled),
       'resiliency':{'cohort_n':len(long),'cohort_days':long.day.nunique(),
          'impact':stats(long),'after_flow_end':stats(long,post_curves),
          'initial_positive':int(positive.sum()),'half_recovery':half,'return_to_start':returned,
          'spread_recovery':spread_recovery,'depth_recovery':depth_recovery,
          'depth_full_coverage':int(long.depth_full_coverage.fillna(False).sum()),
          'positive_at_300':int((long['impact_300s']>1e-9).sum()),
          'opposite_counts':{str(h):int(long[f'opposite_present_{h}s'].fillna(False).sum()) for h in [10,60,300]},
          'strict_age_n':int(long.strict_age_300s.sum()),
          'strict_age_impact':stats(long[long.strict_age_300s])},
       'strict_age_short_n':int(pool.strict_age_10s.sum()),
       'strict_age_short_impact':stats(pool[pool.strict_age_10s])})


def summarize_run(output, calibration_end, notional_threshold, training_bursts):
    manifests=[];frames=[];curves=[];post=[]
    for directory in sorted((output/'days').iterdir()):
        if not (directory/'manifest.json').exists():continue
        manifests.append(json.loads((directory/'manifest.json').read_text()))
        frames.append(pd.read_csv(directory/'events.csv'))
        with np.load(directory/'curves.npz',allow_pickle=False) as values:
            curves.append(values['impact']);post.append(values['after_end'])
    frame=pd.concat(frames,ignore_index=True)
    impact=np.concatenate(curves);after_end=np.concatenate(post)
    thresholds=assign_regimes(frame,calibration_end)
    frame.to_csv(output/'events.csv',index=False)
    result={'schema_version':2,'symbol':'ETHUSDT','venue':'Bybit spot',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'dates':[m['day'] for m in manifests],'calibration_end':calibration_end,
        'book_updates':sum(m['book_updates'] for m in manifests),'trades':sum(m['trades'] for m in manifests),
        'events':len(frame),'days':len(manifests),'method':RULES,
        'notional_threshold':notional_threshold,'training_directional_bursts':training_bursts,
        'regime_thresholds':thresholds,'horizons_seconds':GRID.tolist(),
        'daily':manifests,'cohorts':{},
        'actual_own_orders_observed':0,
        'source':{'history_page':'https://www.bybit.com/en/derivative-activity/history-data',
            'orderbook':'https://quote-saver.bycsi.com/orderbook/spot/ETHUSDT/',
            'trades':'https://public.bybit.com/spot/ETHUSDT/',
            'orderbook_depth':200,'retained_depth':50,'nominal_book_push_ms':100},
        'limitations':[
            '동일 가격의 양의 표시 잔량 변화는 총 신규 공급을 완전히 식별하지 못한다.',
            '백필 200단계 피드는 약100ms, 기존50단계 수집은20ms이므로 첫 refill 속도와 누적 증가의 직접 비교에 주의한다.',
            '300초 코호트는 사건 간격301초 이상이다. 다른 참여자의 체결은 계속 발생하므로 인과적인 주문 효과는 아니다.',
            '호가 변화 때 전송되는 delta의 연속 update ID를 검증해 갱신 사이 상태를 유지했다. 1초 제한 민감도를 별도로 제시한다.',
            '시간 분리는 고정 문턱의 별도 기간 재확인이다. 수익성 백테스트나 외부 시장 검증이 아니다.',
            '여러 지표의 점별95%구간은 탐색적 비교이며 다중검정을 보정한 확증 결과가 아니다.',
            '실제 본인 주문 기록은 없으며 파일럿 관측0건이다.']}
    for name,mask in [('all',np.ones(len(frame),bool)),('calibration',frame.split=='calibration'),('validation',frame.split=='validation')]:
        result['cohorts'][name]=summarize_cohort(frame,impact,after_end,mask)
    result['source_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result=clean_json(result)
    (output/'study.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return result


def prepare_worker(root,day):
    from quant.impact_backfill import prepare_day
    data=prepare_day(root,day,max_gap_ms=0)
    return day,data['cache_dir']


def main():
    from concurrent.futures import ProcessPoolExecutor,as_completed
    from quant.impact_backfill import load_day
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--start',required=True);parser.add_argument('--end',required=True)
    parser.add_argument('--calibration-end',required=True)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.output.exists():parser.error('output already exists; use a new run directory')
    days=[d.strftime('%Y-%m-%d') for d in pd.date_range(args.start,args.end)]
    train=[d for d in days if d<=args.calibration_end]
    if not train or len(train)==len(days):parser.error('both calibration and validation dates required')
    if not 1<=args.workers<=3:parser.error('workers must be 1..3')
    threshold,count=training_notional_threshold(args.root,train,'ETHUSDT')
    args.output.mkdir(parents=True);(args.output/'days').mkdir()
    run={'dates':days,'calibration_end':args.calibration_end,'notional_threshold':threshold,
         'training_directional_bursts':count,'method':RULES,'started_utc':datetime.now(timezone.utc).isoformat()}
    (args.output/'run.json').write_text(json.dumps(run,ensure_ascii=False,indent=2))
    print(json.dumps({'phase':'threshold','USDT':threshold,'training_bursts':count}),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(prepare_worker,args.root,day) for day in days]
        for future in as_completed(futures):
            day,cache=future.result()
            print(json.dumps({'phase':'analyse','day':day,'cache':cache}),flush=True)
            data=load_day(cache)
            rows,curves,post,meta=extract_day(data,day,threshold)
            destination=args.output/'days'/day;destination.mkdir()
            rows.to_csv(destination/'events.csv',index=False)
            np.savez_compressed(destination/'curves.npz',impact=curves,after_end=post)
            (destination/'manifest.json').write_text(json.dumps(clean_json(meta),ensure_ascii=False,indent=2))
            print(json.dumps({'phase':'day_done','day':day,'events':len(rows),'long':int(rows.long_primary.sum())}),flush=True)
            del data,rows,curves,post
    print(json.dumps({'phase':'summarize'}),flush=True)
    result=summarize_run(args.output,args.calibration_end,threshold,count)
    print(json.dumps({'phase':'complete','days':result['days'],'events':result['events'],
                     'long':result['cohorts']['all']['resiliency']['cohort_n']}),flush=True)


if __name__=='__main__':main()

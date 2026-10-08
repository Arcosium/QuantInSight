import numpy as np
import pandas as pd
import pytest

from quant.impact_longitudinal import (
    GRID, POINT_INDEX, extract_day, sustained_start, curve_stats,
    assign_regimes, matched_pairs, recovery_table,
)


def fixture_day():
    start=int(pd.Timestamp('2026-09-01',tz='UTC').timestamp()*1000)
    times=start+np.arange(0,801000,100)
    books=np.zeros((len(times),2,50,2))
    for j in range(50):
        books[:,0,j,0]=99.99-j*.01
        books[:,1,j,0]=100.01+j*.01
        books[:,:,j,1]=10
    records=[(start+t,1,'Buy',2.,100.01) for t in range(20000,799000,11000)]
    return {'times':times,'books':books,'epochs':np.ones(len(times),np.int64),
            'connections':np.ones(len(times),np.int64),'quality':{},
            'trades':pd.DataFrame(records,columns=['ts','connection','side','quantity','price'])}


def test_short_and_long_windows_keep_separate_denominators_and_constant_price():
    rows,curves,post,meta=extract_day(fixture_day(),'2026-09-01',100)
    assert len(rows)>50
    assert rows.long_primary.sum()==2
    long=rows[rows.long_primary]
    assert np.diff(long.ts).min()>=301000
    assert (rows['impact_10s']==0).all()
    assert not rows.initial_positive.any()
    assert (rows.valid300==False).any()
    assert rows.loc[~rows.valid300,'impact_300s'].isna().all()
    assert np.all(curves[np.isfinite(curves)]==0)
    assert not rows.refill_high.any()


def test_epoch_reset_blocks_long_path_without_discarding_short_event():
    data=fixture_day()
    data['epochs'][2500:]=2
    rows,_,_,_=extract_day(data,'2026-09-01',100)
    first=rows.iloc[0]
    assert first.ts==data['times'][200]
    assert first['impact_10s']==0
    assert not first.valid300
    assert pd.isna(first['impact_300s'])


def test_sustained_recovery_requires_elapsed_duration_and_nan_breaks_run():
    assert sustained_start([2.,1.,1.,1.,1.],1.)==pytest.approx(.2)
    assert sustained_start([2.,1.,1.,1.],1.) is None
    assert sustained_start([2.,1.,np.nan,1.,1.,1.],1.) is None
    frame=pd.DataFrame({'t':[9.8,9.7,None]})
    r=recovery_table(frame,'t',np.ones(3,bool))
    assert r['recovered_by']['10']==1  # 9.8 + 0.3 cannot be confirmed by t=10.
    assert r['recovered_by']['30']==2
    assert r['censored_at_300']==1


def test_thresholds_are_frozen_before_validation():
    frame=pd.DataFrame({'day':['2026-09-01']*5+['2026-09-02']*5,
        'depth':[1,2,3,4,5]+[100]*5,'volatility':[0,1,2,3,4]+[99]*5,
        'activity':[1,2,3,4,5]+[100]*5,'spread_ticks':[1]*10,'directional_obi':[0.]*10})
    thresholds=assign_regimes(frame,'2026-09-01')
    assert thresholds['depth']['limits']==pytest.approx([1.8,4.2])
    assert (frame.loc[frame.split=='validation','depth_regime']=='High').all()


def test_matching_no_reuse_and_obi_support():
    f=pd.DataFrame({'day':['a']*5,'side':['buy']*5,'ts':[1000,2000,3000,4000,5000],
        'quantity':[10,10,10.5,50,10],'regime':['Low','Low','High','High','High'],
        'obi_bin':['negative','negative','positive','negative','negative']})
    pairs=matched_pairs(f,'regime','Low','High',obi_control=True)
    assert pairs==[(0,4)]
    unconstrained=matched_pairs(f,'regime','Low','High')
    assert len(unconstrained)==2
    assert len(set(b for a,b in unconstrained))==2


def test_uncertainty_resamples_days_not_individual_observations():
    a=np.array([[0.],[0.],[0.],[10.]])
    s=curve_stats(a,['first']*3+['second'],repetitions=1000)
    assert s['n']==4 and s['days']==2
    assert s['mean']==pytest.approx([2.5])
    assert s['lower']==pytest.approx([0.])
    assert s['upper']==pytest.approx([10.])
    one=curve_stats(a,['only']*4)
    assert one['lower']==[None] and one['upper']==[None]

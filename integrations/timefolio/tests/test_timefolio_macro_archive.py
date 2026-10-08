from quant.timefolio_heatmap_macro_archive import parse_allocation, first_available, daily_signals


def record(available, digest='a', stock=.6):
    return dict(available_at=available,report_sha256=digest,stock_pct=stock,cash_pct=.1,session='KR_TRADING')


def test_only_current_explicit_allocation_is_accepted():
    assert parse_allocation('현재 주식 80%. 자산 배분 권고: 주식 **55%** / 채권 30% / 현금 15% (직전: 주식 80% / 현금 20%)') == {'stock_pct':.55,'cash_pct':.15}
    assert parse_allocation('현재 주식 80%, 현금 20%') is None
    assert parse_allocation('자산 배분 권고: 채권 80% (직전: 주식 80%)') is None
    assert parse_allocation('자산 배분 권고: 주식 110% / 현금 10%') is None
    assert parse_allocation('자산 배분 권고: 주식 80% / 현금 30%') is None
    assert parse_allocation('자산 배분 권고: 주식 20% → 50% / 현금 50%') is None
    assert parse_allocation('자산 배분 권고: 기존 주식 20% / 현금 80%') is None


def test_cached_repetitions_do_not_refresh_original_availability():
    records=[record('2026-05-22T15:00:00+09:00'),record('2026-05-18T10:00:00+09:00')]
    assert len(first_available(records)) == 1
    rows=daily_signals(records,['20260518','20260522'])
    assert rows[0]['available'] and not rows[1]['available']


def test_close_cutoff_and_future_append_invariance():
    records=[record('2026-05-18T15:30:01+09:00'),record('2026-05-19T15:00:00+09:00','b',.3)]
    before=daily_signals(records,['20260518','20260519'])
    assert not before[0]['available'] and before[1]['stock_pct']==.3
    records.append(record('2026-05-20T09:00:00+09:00','c',.8))
    assert daily_signals(records,['20260518','20260519'])==before


def test_utc_timestamp_is_compared_in_kst():
    rows=daily_signals([record('2026-05-18T06:30:00+00:00')],['20260518'])
    assert rows[0]['available'] and rows[0]['age_hours']==0


def test_overlay_bounds_fallback_smoothing_and_future_invariance():
    from copy import deepcopy
    from datetime import datetime
    import numpy as np
    import pytest
    from test_timefolio_heatmap import synthetic_panel
    from quant.timefolio_heatmap_macro_overlay import schedules_from_records
    p,ix=synthetic_panel(days=35);dates=ix['dates']
    records=[]
    for i,date in enumerate(dates):
        stamp=datetime.strptime(date,'%Y%m%d').strftime('%Y-%m-%dT15:30:00+09:00')
        records.append(dict(date=date,signal_cutoff=stamp,report_available_at=stamp,
                            available=i>=1,age_hours=0,stock_pct=0. if i==1 else 1.))
    schedules,available=schedules_from_records(p,dates,records)
    assert schedules['macro_stock'][0]==.6
    assert schedules['macro_stock'][1]==.3 and schedules['macro_stock'][2]==.8
    assert schedules['macro_smooth'][1]==.6
    assert schedules['macro_smooth'][2]==pytest.approx(.45)
    changed=deepcopy(records)
    for row in changed[20:]:row['stock_pct']=.1
    later,_=schedules_from_records(p,dates,changed)
    for name in schedules:
        np.testing.assert_array_equal(schedules[name][:20],later[name][:20])
        assert np.all((schedules[name]>=.3)&(schedules[name]<=.8))
    changed[1]['report_available_at']='2099-01-01T15:30:00+09:00'
    with pytest.raises(ValueError,match='Future'):schedules_from_records(p,dates,changed)

import numpy as np
import pandas as pd
import pytest

from quant.impact_alpha_state import execution_labels,replay,choose,block_interval


def minutes(n=1440):
    ts=pd.date_range('2026-08-01',periods=n,freq='min',tz='UTC')
    return pd.DataFrame({'day':ts.strftime('%Y-%m-%d'),'decision_ms':ts.as_unit('ms').asi8,
                         'quote_ms':ts.as_unit('ms').asi8-100,'state_valid':True,
                         'buy_vwap':100.,'sell_vwap':99.,'buy_valid':True,'sell_valid':True})


def test_future_label_uses_delayed_entry_and_end_of_known_day():
    f=minutes();f.loc[1,'buy_vwap']=101.;f.loc[61,'sell_vwap']=102.
    out=execution_labels(f)
    assert out.loc[0,'label_60']==pytest.approx((102/101-1)*10000)
    assert out.loc[0,'label_exit_ms_60']==f.loc[61,'decision_ms']
    assert out.loc[1193,'entry_allowed'] and not out.loc[1194,'entry_allowed']
    f.loc[1,'buy_valid']=False
    invalid=execution_labels(f)
    assert len(invalid)==len(f) and np.isnan(invalid.loc[0,'label_60'])


def test_missing_calendar_row_cannot_silently_shift_execution():
    with pytest.raises(ValueError,match='every calendar minute'):
        execution_labels(minutes().drop(index=5))


def test_delayed_execution_nonoverlap_and_no_future_price_entry_selection():
    f=execution_labels(minutes());score=np.full(len(f),30.)
    a,_=replay(f,score,21,60,'fixed')
    f['sell_vwap']=50.
    b,_=replay(f,score,21,60,'fixed')
    assert a.entry_ms.tolist()==b.entry_ms.tolist()
    assert (a.entry_ms-a.decision_ms==60000).all()
    assert (a.exit_ms-a.entry_ms==60*60000).all()
    assert np.all(a.entry_ms.to_numpy()[1:]>=a.exit_ms.to_numpy()[:-1])


def test_reverse_exit_waits_minimum_holding_and_next_minute_execution():
    f=execution_labels(minutes());scores=np.full(len(f),np.nan)
    scores[0]=30.;scores[5]=-10.;scores[16]=-3.
    rows,_=replay(f,scores,21,60,'reverse15')
    assert len(rows)==1
    assert rows.iloc[0].entry_ms==f.loc[1,'decision_ms']
    assert rows.iloc[0].exit_ms==f.loc[17,'decision_ms']
    assert rows.iloc[0].exit_reason=='forecast_reversal'


def test_missing_selected_exit_waits_then_fails_instead_of_disappearing():
    f=execution_labels(minutes());scores=np.full(len(f),np.nan);scores[0]=30.
    f.loc[61:63,'sell_valid']=False
    rows,_=replay(f,scores,21,60,'fixed')
    assert rows.iloc[0].exit_wait_minutes==3
    f.loc[61:66,'sell_valid']=False
    with pytest.raises(ValueError,match='Unpriced selected exit'):
        replay(f,scores,21,60,'fixed')


def test_invalid_entry_rejected_at_execution_time():
    f=execution_labels(minutes());scores=np.full(len(f),np.nan);scores[0]=30.
    f.loc[1,'buy_valid']=False
    rows,audit=replay(f,scores,21,60,'fixed')
    assert rows.empty and audit['rejected_entries']==1


def test_calibration_selector_requires_positive_money_and_breadth():
    def c(n=14,days=4,pnl=1.,hold=60):
        return {'hold':hold,'quantile':.8,'exit_mode':'fixed',
                'calibration':{'n':n,'trading_days':days,'net_pnl':pnl}}
    assert choose([c(n=13),c(days=3),c(pnl=0)]) is None
    assert choose([c(pnl=1),c(pnl=2,hold=240)])['hold']==240


def test_block_ci_does_not_make_all_cash_a_positive_result():
    assert block_interval(np.zeros(62))==[0.,0.]
    assert block_interval(np.full(62,2.))==pytest.approx([2.,2.])

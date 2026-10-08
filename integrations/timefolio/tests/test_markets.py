import math
import pytest
from web.markets import trend_backtest,validate,bars


def series():
    return [{"time":i*60,"close":100+i*.05,"open":100+i*.05} for i in range(150)]


def test_costs_reduce_performance_and_flat_market_never_invents_profit():
    assert trend_backtest(series(),cost_pct=2)['return_pct'] < trend_backtest(series(),cost_pct=0)['return_pct']
    flat=[dict(r,open=100,close=100) for r in series()]
    assert trend_backtest(flat)['return_pct']==0
    assert trend_backtest(flat)['trades']==0


def test_signal_cannot_trade_before_next_bar():
    rows=[dict(r,open=100,close=100) for r in series()]
    rows[-1]['close']=10000
    result=trend_backtest(rows)
    assert result['trades']==0 and result['return_pct']==0


@pytest.mark.parametrize('m,s',[('NXT','../x'),('KRX','AAPL'),('USA','../secret'),('FAKE','BTC')])
def test_market_and_symbol_validation(m,s):
    with pytest.raises(ValueError):validate(m,s)


def test_bad_cost_and_missing_data_rejected():
    with pytest.raises(ValueError):trend_backtest(series(),cost_pct=math.nan)
    with pytest.raises(ValueError):trend_backtest(series()[:60])

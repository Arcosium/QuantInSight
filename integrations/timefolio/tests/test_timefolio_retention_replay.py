from copy import deepcopy

import numpy as np
import pytest

from test_timefolio_heatmap import synthetic_panel
from quant.timefolio_heatmap_dated_replay import replay as original
from quant.timefolio_heatmap_retention_replay import replay, retention_scores
from quant.timefolio_heatmap_stock_limits import historical_stock_caps


def setup():
    p, ix = synthetic_panel(days=6, names=3)
    scores = np.tile([3., 2., 1.], (6, 1)).T
    scores[:, 1:] = np.array([2., 3., 1.])[:, None]
    kw = dict(top_n=1, weight=.05, rebalance=1, rebalance_band=.0005,
              max_orders=3, return_trades=True, return_plans=True)
    return p, ix, scores, kw


def test_zero_buffer_preserves_entire_account():
    p, ix, scores, kw = setup()
    caps = historical_stock_caps(ix['codes'], ix['dates'])
    args = (p, ix, scores, ix['dates'][1], ix['dates'][-1])
    assert replay(*args, stock_cap_schedule=caps, **kw) == original(*args, stock_cap_schedule=caps, **kw)


def test_near_rank_change_retains_actual_position_and_reduces_fees():
    p, ix, scores, kw = setup()
    args = (p, ix, scores, ix['dates'][1], ix['dates'][2])
    before = replay(*args, **kw); after = replay(*args, rank_buffer=1, **kw)
    assert any(t['side'] == 'sell' for t in before['trades'])
    assert len(after['trades']) == 1
    assert after['trades'][0]['code'] == ix['codes'][0]
    assert after['metrics']['fees_krw'] < before['metrics']['fees_krw']


def test_position_outside_buffer_is_sold():
    p, ix, scores, kw = setup(); scores[:, 1] = [1., 3., 2.]
    result = replay(p, ix, scores, ix['dates'][1], ix['dates'][2], rank_buffer=1, **kw)
    assert any(t['side'] == 'sell' and t['code'] == ix['codes'][0] for t in result['trades'])


def test_unfilled_wish_is_not_treated_as_a_holding():
    p, ix, scores, kw = setup(); p['exec_count'][0, 1] = 0
    result = replay(p, ix, scores, ix['dates'][1], ix['dates'][2], rank_buffer=1, **kw)
    assert result['daily'][0]['holdings'] == 0
    assert {t['code'] for t in result['trades']} == {ix['codes'][1]}


def test_execution_date_buy_exclusion_is_not_overridden():
    p, ix, scores, kw = setup(); p['trade_allowed'] = np.ones_like(p['eligible'])
    p['trade_allowed'][0, 2] = False
    result = replay(p, ix, scores, ix['dates'][1], ix['dates'][2], rank_buffer=2, **kw)
    assert any(t['date'] == ix['dates'][2] and t['side'] == 'sell' and t['code'] == ix['codes'][0]
               for t in result['trades'])
    assert not any(t['date'] == ix['dates'][2] and t['side'] == 'buy' and t['code'] == ix['codes'][0]
                   for t in result['trades'])


def test_sector_repair_bypasses_retention_and_weight_band():
    p, ix, scores, kw = setup(); p['sector_cap'][0, 1:] = .03
    kw.update(rebalance=5, rebalance_band=.02)
    result = replay(p, ix, scores, ix['dates'][1], ix['dates'][2], rank_buffer=2, **kw)
    assert any(t['date'] == ix['dates'][2] and t['side'] == 'sell' for t in result['trades'])
    assert result['daily'][-1]['gross'] < .03


def test_dated_individual_repair_bypasses_retention():
    p, ix, scores, kw = setup(); ix['codes'] = ['000660', '005930', '000001']
    ix['dates'] = ['20260622', '20260623', '20260624', '20260625', '20260626', '20260629']
    p['sector_cap'][:] = .8; p['close'][0, 1:] = 12000
    for key in ['o', 'exec_price', 'exec_high', 'exec_low']:
        p[key][0, 2:] *= 1.2
    kw.update(weight=.14, rebalance=5, rebalance_band=.02)
    result = replay(p, ix, scores, ix['dates'][1], ix['dates'][2], rank_buffer=2,
                    stock_cap_schedule=historical_stock_caps(ix['codes'], ix['dates']), **kw)
    assert any(t['date'] == ix['dates'][2] and t['side'] == 'sell' for t in result['trades'])
    assert result['daily'][-1]['gross'] <= .15


def test_future_changes_do_not_change_earlier_fills_or_nav():
    p, ix, scores, kw = setup()
    first = replay(p, ix, scores, ix['dates'][1], ix['dates'][-1], rank_buffer=1, **kw)
    altered = deepcopy(p); changed = scores.copy(); changed[:, 3:] *= -100
    altered['eligible'][:, 4:] = False; altered['exec_volume'][:, 4:] = 0
    second = replay(altered, ix, changed, ix['dates'][1], ix['dates'][-1], rank_buffer=1, **kw)
    assert first['daily'][:3] == second['daily'][:3]
    assert [t for t in first['trades'] if t['date'] < ix['dates'][4]] == [t for t in second['trades'] if t['date'] < ix['dates'][4]]


def test_smallcap_repair_is_not_suppressed_by_retained_positions():
    p, ix = synthetic_panel(days=5, names=6)
    p['sector_cap'][:] = .8; p['market_cap'][:, 1:] = 5e11
    scores = np.broadcast_to(np.arange(6., 0., -1.)[:, None], p['close'].shape)
    result = replay(p, ix, scores, ix['dates'][1], ix['dates'][2], top_n=6, weight=.08,
                    rebalance=5, rebalance_band=.02, rank_buffer=6, max_orders=10, return_trades=True)
    assert result['daily'][0]['gross'] > .4
    assert any(t['date'] == ix['dates'][2] and t['side'] == 'sell' for t in result['trades'])
    assert result['daily'][-1]['gross'] <= .3


def test_ties_and_eligibility_use_ticker_order_without_mutation():
    scores = np.array([1., 1., 1., np.nan]); original_scores = scores.copy()
    eligible = np.array([True, True, False, True]); held = np.array([True, False, True, True])
    codes = ['000002', '000001', '000003', '000004']
    ranked = retention_scores(scores, eligible, held, codes, top_n=1, rank_buffer=1)
    assert ranked[0] > ranked[1] and np.isnan(ranked[2:]).all()
    np.testing.assert_array_equal(scores, original_scores)
    no_holdings = retention_scores(scores, eligible, np.zeros(4, bool), codes, top_n=1, rank_buffer=1)
    assert no_holdings[1] > no_holdings[0]


@pytest.mark.parametrize('buffer', [-1, 1.5, True])
def test_invalid_buffer_is_rejected(buffer):
    p, ix, scores, kw = setup()
    with pytest.raises(ValueError, match='rank_buffer'):
        replay(p, ix, scores, ix['dates'][1], ix['dates'][2], rank_buffer=buffer, **kw)

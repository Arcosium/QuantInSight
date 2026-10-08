from copy import deepcopy

import numpy as np
import pytest

from test_timefolio_heatmap import synthetic_panel
from quant.timefolio_heatmap_banded_replay import replay as prior_replay
from quant.timefolio_heatmap_dated_replay import replay
from quant.timefolio_heatmap_stock_limits import historical_stock_caps, audit_pre_july_hynix


def setup():
    p, ix = synthetic_panel(days=5, names=3)
    ix['codes'] = ['000660', '005930', '000001']
    ix['dates'] = ['20260629', '20260630', '20260701', '20260702', '20260703']
    p['sector_cap'][:] = .8
    score = np.broadcast_to(np.array([3., 2., 1.])[:, None], p['close'].shape)
    return p, ix, score


def test_limit_changes_on_execution_date_and_samsung_is_unchanged():
    p, ix, score = setup(); caps = historical_stock_caps(ix['codes'], ix['dates'])
    np.testing.assert_equal(caps[0], [.15, .15, .30, .30, .30])
    np.testing.assert_equal(caps[1], np.full(5, .4))
    result = replay(p, ix, score, ix['dates'][1], ix['dates'][2], rebalance=1,
                    top_n=1, weight=.25, stock_cap_schedule=caps, return_trades=True)
    assert result['daily'][0]['gross'] <= .15
    assert .20 < result['daily'][1]['gross'] <= .30
    assert not audit_pre_july_hynix(p, ix, result)['post_buy_limit_errors']


def test_independent_audit_detects_old_early_exception():
    p, ix, score = setup()
    result = prior_replay(p, ix, score, ix['dates'][1], ix['dates'][1], top_n=1, weight=.25, return_trades=True)
    assert audit_pre_july_hynix(p, ix, result)['post_buy_limit_errors']


def test_default_reproduces_frozen_banded_ledger():
    p, ix, score = setup(); args = (p, ix, score, ix['dates'][1], ix['dates'][3])
    kw = dict(top_n=3, weight=.05, max_orders=3, return_trades=True, rebalance_band=.0005)
    assert replay(*args, **kw) == prior_replay(*args, **kw)


def test_future_limits_cannot_change_earlier_trades():
    p, ix, score = setup(); caps = historical_stock_caps(ix['codes'], ix['dates'])
    args = (p, ix, score, ix['dates'][1], ix['dates'][3]); kw = dict(top_n=2, weight=.25, rebalance=1, return_trades=True)
    before = replay(*args, stock_cap_schedule=caps, **kw); caps[:, 3:] = .05
    after = replay(*args, stock_cap_schedule=caps, **kw)
    assert before['daily'][:2] == after['daily'][:2]
    assert [t for t in before['trades'] if t['date'] < ix['dates'][3]] == [t for t in after['trades'] if t['date'] < ix['dates'][3]]


def test_known_price_breach_repairs_even_with_wide_adjustment_band():
    p, ix, score = setup(); ix['dates'] = ['20260622', '20260623', '20260624', '20260625', '20260626']
    p['close'][0, 1:] = 12000
    for key in ['o', 'exec_price', 'exec_high', 'exec_low']: p[key][0, 2:] *= 1.2
    caps = historical_stock_caps(ix['codes'], ix['dates'])
    result = replay(p, ix, score, ix['dates'][1], ix['dates'][2], rebalance=5, top_n=1, weight=.14,
                    stock_cap_schedule=caps, rebalance_band=.02, return_trades=True)
    assert any(t['date']==ix['dates'][2] and t['side']=='sell' for t in result['trades'])
    assert result['daily'][1]['gross'] <= .15


@pytest.mark.parametrize('value', [0., np.nan, 1.01])
def test_invalid_limits_are_rejected(value):
    p, ix, score = setup(); caps = np.full(p['close'].shape, value)
    with pytest.raises(ValueError, match='stock_cap_schedule'):
        replay(p, ix, score, ix['dates'][1], ix['dates'][2], stock_cap_schedule=caps)

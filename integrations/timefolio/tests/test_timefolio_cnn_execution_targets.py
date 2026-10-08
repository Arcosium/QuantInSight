import numpy as np
import pytest

from quant.timefolio_cnn_execution_targets import execution_targets
from quant.timefolio_cnn_core_retention_replay import replay


def panel(days=6):
    shape = (1, days)
    p = {name: np.full(shape, 1000., dtype=float) for name in
         ['o', 'close', 'exec_price', 'exec_high', 'exec_low']}
    p.update(eligible=np.ones(shape, bool), sector=np.array([0]),
             sector_cap=np.full(shape, .1), market_cap=np.full(shape, 2e12),
             split=np.ones(shape), listed_shares=np.ones(shape),
             exec_volume=np.full(shape, 1e8), exec_count=np.full(shape, 30))
    index = dict(codes=['000001'], dates=[f'2024010{i+1}' for i in range(days)])
    return p, index


def target(p, *, horizon=3, economic=0.):
    return execution_targets(p, ['000001'], np.array([0]), np.array([0]),
                             np.array([horizon+1]), np.array([economic]), horizon=horizon)


def test_partial_fill_pnl_and_unfilled_cash_match_one_position_replay():
    p, idx = panel(); p['exec_volume'][0, 1] = 20000
    p['close'][0, 3] = 1100
    out = target(p)
    result = replay(p, idx, np.ones((1, 6)), idx['dates'][1], idx['dates'][3],
                    rebalance=100, return_trades=True)
    assert out['requested_qty'][0] == 80000
    assert out['filled_qty'][0] == 1000
    assert out['entry_cash_spent'][0] == pytest.approx(1000*1000*1.0005*1.001)
    assert out['account_return'][0] == pytest.approx(result['metrics']['return'], abs=1e-15)
    assert out['position_utility'][0] == pytest.approx((1100000-1000*1000*1.0005*1.001)/8e7)
    assert out['close_gross'][0] == pytest.approx(.1)


@pytest.mark.parametrize('reason', ['prints', 'limit', 'admission', 'cap', 'volume'])
def test_unfillable_request_stays_a_zero_cash_label(reason):
    p, _ = panel()
    if reason == 'prints': p['exec_count'][0, 1] = 24
    if reason == 'limit':
        for name in ['exec_price', 'exec_high', 'exec_low']: p[name][0, 1] = 1300
    if reason == 'admission':
        p['trade_allowed'] = np.ones((1, 6), bool); p['trade_allowed'][0, 1] = False
    if reason == 'cap': p['market_cap'][0, 0] = 1e10
    if reason == 'volume': p['exec_volume'][0, 1] = 0
    out = target(p)
    assert out['filled_qty'][0] == 0
    assert out['position_utility'][0] == 0
    assert np.isfinite(out['close_gross'][0])


def test_sector_headroom_includes_fee_shrink_and_can_spend_above_plan():
    p, idx = panel()
    # Open-to-window jump is below the upper-limit proxy; tight sector cap binds.
    p['sector_cap'][:] = .081
    for name in ['exec_price', 'exec_high', 'exec_low', 'close']: p[name][0, 1] = 1280
    out = target(p, horizon=1)
    result = replay(p, idx, np.ones((1, 6)), idx['dates'][1], idx['dates'][1], return_trades=True)
    q = out['filled_qty'][0]; cost = 1280*1.0005*1.001
    assert q == np.floor(.081*1e9/(1280+.081*(cost-1280)))
    assert out['entry_cash_spent'][0] > 8e7
    assert q*1280/(1e9-q*(cost-1280)) <= .081
    assert out['account_return'][0] == pytest.approx(result['metrics']['return'], abs=1e-15)


def test_terminal_mark_uses_stale_or_window_prices_and_never_future_outcomes():
    p, _ = panel(); p['close'][0, 3] = np.nan; p['exec_price'][0, 3] = 1030
    a = target(p)
    assert a['terminal_mark'][0] == 1030
    p['exec_count'][0, 3] = 0; p['close'][0, 2] = 1020
    b = target(p)
    assert b['terminal_mark'][0] == 1020
    p['close'][0, 4:] = 1e9; p['exec_volume'][0, 4:] = 0
    c = target(p)
    for name in b: np.testing.assert_array_equal(b[name], c[name])


def test_parent_missing_labels_and_conservative_availability_are_preserved():
    p, _ = panel(); out = target(p, economic=np.nan)
    assert all(np.isnan(a[0]) for a in out.values())
    with pytest.raises(ValueError, match='axes'):
        execution_targets(p, ['000001'], np.array([0]), np.array([0]), np.array([3]),
                          np.array([0.]), horizon=3)
    p['split'][0, 2] = 2
    with pytest.raises(ValueError, match='action ledger'): target(p)


def test_sector_double_market_weight_above_one_keeps_stock_target_and_caps():
    p, idx = panel(); p['sector_cap'][:] = 1.28
    out = target(p, horizon=1)
    result = replay(p, idx, np.ones((1, 6)), idx['dates'][1], idx['dates'][1], return_trades=True)
    assert out['filled_qty'][0] == 80000
    assert out['account_return'][0] == pytest.approx(result['metrics']['return'], abs=1e-15)

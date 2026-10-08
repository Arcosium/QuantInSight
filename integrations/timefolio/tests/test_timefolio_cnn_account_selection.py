import numpy as np
import pandas as pd
import pytest
import torch

from quant.timefolio_cnn_account_selection import AccountValidator, selection_key
from quant.timefolio_cnn_train import fit


def fixture():
    n, d = 12, 85
    dates = pd.bdate_range('2024-01-01', periods=d).strftime('%Y%m%d').tolist()
    price = np.full((n, d), 10000.)
    p = dict(o=price.copy(), close=price.copy(), eligible=np.ones((n, d), bool),
             sector=np.arange(n), sector_cap=np.full((n, d), .1), market_cap=np.full((n, d), 2e12),
             split=np.ones((n, d)), listed_shares=np.ones((n, d)), exec_price=price.copy(),
             exec_volume=np.full((n, d), 1e7), exec_count=np.full((n, d), 30.),
             exec_high=price.copy(), exec_low=price.copy())
    a = dict(dates=np.array(dates), signal_index=np.repeat(np.arange(d), n),
             security_key=np.tile(np.arange(n), d), eligible=np.ones(n*d, bool),
             returns=np.zeros(n*d))
    index = dict(dates=dates, codes=[f'{i:06}' for i in range(n)])
    fold = dict(validation_start=10, test_start=70)
    return p, index, a, fold


def test_account_stops_before_outer_and_ignores_future_label_availability():
    p, index, a, fold = fixture()
    first = AccountValidator(p, index, a, fold)
    scores = np.ones(len(first.sample_ids))
    before = first.evaluate(scores)
    a['returns'][:] = np.nan
    for name in ['o', 'close', 'exec_price', 'exec_high', 'exec_low']:
        p[name][:, fold['test_start']:] *= 100
    second = AccountValidator(p, index, a, fold)
    np.testing.assert_array_equal(first.sample_ids, second.sample_ids)
    assert before == second.evaluate(scores)
    assert before['end'] == index['dates'][69] and before['last_account_mark_index'] == 69
    assert a['signal_index'][first.sample_ids].max() == 68


def test_flat_price_account_loses_fees_and_cannot_pass_low_turnover_screen():
    p, index, a, fold = fixture()
    val = AccountValidator(p, index, a, fold)
    result = val.evaluate(np.ones(len(val.sample_ids)))
    assert result['net_return'] < 0 and result['fees_krw'] > 0 and result['slippage_krw'] > 0
    assert result['turnover_screen_passed'] is False
    assert result['contest_certified'] is False
    assert result['terminal_liquidation_reserve'] > 0


def test_profit_is_primary_within_turnover_survivors_and_invalid_cash_cannot_win():
    winner = dict(net_return=.20, turnover_screen_passed=True)
    assert selection_key(winner) > selection_key(dict(net_return=.05, turnover_screen_passed=True))
    assert selection_key(winner) > selection_key(dict(net_return=.80, turnover_screen_passed=False))
    with pytest.raises(ValueError):
        selection_key(dict(net_return=np.nan, turnover_screen_passed=True))


def test_epoch_selection_uses_account_profit_and_records_selected_ledger():
    torch.set_num_threads(1)
    rng = np.random.default_rng(17)
    x = rng.uniform(-1, 1, (16, 1, 8, 20)).astype(np.float32)
    y = rng.normal(0, .01, 16).astype(np.float32)
    signal = np.repeat(np.arange(4), 4); keys = np.tile(np.arange(4), 4)
    cfg = dict(seed=17, width=2, dropout=.1, lr=.01, weight_decay=.001,
               epochs=3, top_k=2, temperature=.2, objective='mse', max_date_group=8)
    class Validator:
        sample_ids = np.flatnonzero(signal == 2)
        def __init__(self): self.epoch = 0
        def evaluate(self, scores):
            assert len(scores) == 4 and np.isfinite(scores).all()
            value = [.03, .02, .90][self.epoch]
            self.epoch += 1
            return dict(net_return=value, turnover_screen_passed=self.epoch < 3)
    _, receipt = fit(x, y, signal, keys, signal < 2, signal == 2, cfg,
                     account_validator=Validator())
    assert receipt['best_epoch'] == 1
    assert receipt['criterion'] == 'inner_account_net_profit'
    assert len(receipt['history']) == 3

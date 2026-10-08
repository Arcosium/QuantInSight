import json
import pytest
from autofolio import paper_benchmark as b


def test_overlap_uses_later_fold_and_never_fills_gap(tmp_path):
    folds = [
        {'2025-01-01': 1, '2025-01-02': 1.1, '2025-01-03': 1.21},
        {'2025-01-02': 1, '2025-01-03': .9},
        {'2025-03-01': 1, '2025-03-02': 1.2},
        {'2025-05-01': 1, '2025-05-02': .8},
    ]
    for i, daily in enumerate(folds):
        (tmp_path / f'ens_heatf10_direct_4h_s{i}_cohort_H84_wev.json').write_text(json.dumps({'daily': daily}))
    returns, _ = b.paper_returns(tmp_path)
    assert list(returns) == ['20250102', '20250103', '20250302', '20250502']
    assert returns['20250103'] == pytest.approx(-.1)
    result = b.measure(returns)
    assert result['net_return'] == pytest.approx(1.1*.9*1.2*.8-1)
    assert result['months'] == 3 and result['negative_months'] == 2


def test_comparison_uses_daily_returns_not_gap_jump(monkeypatch):
    monkeypatch.setattr(b, 'paper_returns', lambda: ({'20250102': .1, '20250104': -.1}, []))
    case = {'initial_cash': 100, 'daily': [dict(date='20250101', nav=100),
        dict(date='20250102', nav=110), dict(date='20250103', nav=220), dict(date='20250104', nav=198)]}
    monkeypatch.setattr(b, 'strategy_case', lambda *_: ({}, case))
    result = b.comparison([dict(id='test', title='test', phase_count=1)])
    assert result['points'][1]['net_return'] == pytest.approx(-.01)
    assert result['superior'] == 0
    monkeypatch.setattr(b, 'paper_returns', lambda: ({'20250105': .1}, []))
    assert b.comparison([dict(id='test', title='test', phase_count=1)])['skipped'] == 1


def test_missing_source_is_explicit(monkeypatch):
    def missing():
        raise FileNotFoundError
    monkeypatch.setattr(b, 'paper_returns', missing)
    assert b.comparison([])['available'] is False

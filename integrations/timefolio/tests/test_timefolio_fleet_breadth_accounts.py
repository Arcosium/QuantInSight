import numpy as np

from quant import timefolio_heatmap_fleet_breadth_accounts as runner
from quant.timefolio_heatmap_fleet_breadth import identity, policies, hypotheses


def test_unit_plan_covers_each_account_once_and_shares_nonimage():
    cases = [f'case{i}' for i in range(9)]
    units = runner.units(cases)
    keys = [identity(unit['id'], policy) for unit in units for policy in policies()]
    assert len(units) == 73 and sum(u['nonimage'] for u in units) == 1
    assert len(keys) == len(set(keys)) == 3504
    assert set(keys) == {key for key, _, _ in hypotheses(cases)}


def test_new_width_reaches_replay_without_relaxing_limits(monkeypatch):
    calls = []
    def fake_replay(panel, ix, scores, start, end, **kwargs):
        calls.append((panel, kwargs))
        return {'metrics': {}, 'trades': [{'signal_date': '20260101'}], 'plans': ['plan']}
    monkeypatch.setattr(runner.engine, 'replay', fake_replay)
    monkeypatch.setattr(runner, 'assess_weeks', lambda result: {})
    raw = {'sector_cap': np.array([[.1], [.3]])}
    policy = next(p for p in policies() if p['top_n'] == 40 and p['buffer'] == 80 and p['ceiling'] == 'research20')
    panel, result, plans = runner.simulate(raw, {'dates': ['20260101']}, 'release', 'caps',
        {policy['refresh']: (np.zeros((2, 1)), [0])}, policy)
    assert calls[0][1]['top_n'] == 40 and calls[0][1]['weight'] == .015
    assert calls[0][1]['rank_buffer'] == 80 and calls[0][1]['locked_repair'] is True
    assert calls[0][1]['stock_cap_schedule'] == 'caps' and calls[0][1]['action_release_dates'] == 'release'
    assert calls[0][1]['max_orders'] == policy['max_orders']
    assert 'gross_schedule' not in calls[0][1] and 'weight_schedule' not in calls[0][1]
    np.testing.assert_array_equal(panel['sector_cap'], [[.1], [.2]])
    np.testing.assert_array_equal(raw['sector_cap'], [[.1], [.3]])
    assert result['trades'][0]['portfolio_score_date'] == '20260101'
    assert plans == ['plan'] and result['plans'] == []

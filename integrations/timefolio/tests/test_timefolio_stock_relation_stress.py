import json

import numpy as np
import pandas as pd
import pytest

from quant import timefolio_heatmap_stock_relation_stress as study
from test_timefolio_heatmap import synthetic_panel

CANDIDATE = 'cnn_relation_attention_h10_ensemble__n12__orders10__refresh1__buffer24__band5bp'


def test_population_keeps_all_seeds_and_controls_and_never_relaxes_budget():
    accounts = study.candidate_accounts(CANDIDATE)
    assert len(accounts) == 25 and len(set(accounts)) == 25
    assert 'cnn_relation_attention_h10_seed43__n12__orders10__refresh1__buffer24__band5bp' in accounts
    assert 'mlp_relation_attention_h10_seed29__n12__orders10__refresh1__buffer24__band5bp' in accounts
    for key in accounts:
        for scenario in study.SCENARIOS:
            options = study.execution_options(key, scenario)
            assert options['max_orders'] in [3, 10] and options['slip'] >= .0005 and options['participation'] <= .05
            smaller = study.execution_options(key.replace('__orders10__', '__orders3__'), scenario)
            assert smaller['max_orders'] == 3
    with pytest.raises(ValueError): study.candidate_accounts(CANDIDATE.replace('_ensemble__', '_seed17__'))
    with pytest.raises(ValueError): study.candidate_accounts(CANDIDATE.replace('cnn_', 'mlp_', 1))


def frame_fixture():
    rows = []
    for key in study.candidate_accounts(CANDIDATE):
        for scenario in study.SCENARIOS:
            rows.append(dict(parent_id=key, scenario=scenario['id'], **{'return': .1}, mdd=-.1,
                positive_blocks=2, four_week_turnover_stop=False, daily_mean=.002 if key == CANDIDATE else .001))
    return pd.DataFrame(rows)


def test_gate_requires_all_adverse_scenarios_and_all_seeds():
    frame = frame_fixture(); result = study.stress_gate(frame, [CANDIDATE])[0]
    assert result['stress_passed'] and len(result['seed_basics']) == 24 and len(result['ensemble_comparisons']) == 36
    bad = frame.copy(); mask = bad.parent_id.str.contains('cnn_relation_attention_h10_seed43') & (bad.scenario == 'combined')
    bad.loc[mask, 'return'] = 0.
    assert not study.stress_gate(bad, [CANDIDATE])[0]['stress_passed']
    bad = frame.copy(); bad.loc[(bad.parent_id == CANDIDATE) & (bad.scenario == 'slippage25bp'), 'daily_mean'] = .001
    assert not study.stress_gate(bad, [CANDIDATE])[0]['stress_passed']


def test_gate_refuses_missing_or_duplicate_controls():
    frame = frame_fixture()
    with pytest.raises(KeyError): study.stress_gate(frame.iloc[1:], [CANDIDATE])
    with pytest.raises(ValueError): study.stress_gate(pd.concat([frame, frame.iloc[:1]]), [CANDIDATE])


@pytest.mark.parametrize('condition', ['missing', 'unfinished', 'no_candidate', 'inconsistent', 'missing_evidence'])
def test_review_guard_never_promotes_a_failed_or_unreviewed_winner(tmp_path, condition):
    if condition == 'missing':
        with pytest.raises(FileNotFoundError): study.reviewed_candidates(tmp_path)
        return
    chosen = [] if condition == 'no_candidate' else [CANDIDATE]
    validation = dict(status='incomplete' if condition == 'unfinished' else 'numerical_validation_complete',
                      portfolios=744, candidates=chosen, main_review_evidence_hashes={})
    (tmp_path / 'validation_complete.json').write_text(json.dumps(validation))
    (tmp_path / 'evaluation_summary.json').write_text(json.dumps(dict(robust_candidate_gate_passed=chosen)))
    (tmp_path / 'effect_review.json').write_text(json.dumps(dict(independently_recomputed_candidates=[] if condition == 'inconsistent' else chosen)))
    with pytest.raises(RuntimeError): study.reviewed_candidates(tmp_path)


def test_guard_failure_does_not_create_outputs_or_read_market_panel(monkeypatch, tmp_path):
    monkeypatch.setattr(study, 'DEST', tmp_path / 'untouched')
    def reject(): raise RuntimeError('No qualifying candidate')
    monkeypatch.setattr(study, 'reviewed_candidates', reject)
    monkeypatch.setattr(study, 'context', lambda *args: pytest.fail('Market data read before review'))
    with pytest.raises(RuntimeError): study.run()
    assert not study.DEST.exists()


@pytest.mark.parametrize('scenario', study.SCENARIOS, ids=lambda s: s['id'])
def test_stress_execution_and_auditor_share_prices_volume_and_order_budget(scenario):
    p, ix = synthetic_panel(days=4, names=8)
    p['sector_cap'][:] = .8; p['market_cap'][:] = 2e12; p['exec_volume'][:] = 20000.
    scores = np.broadcast_to(np.arange(8., 0., -1.)[:, None], p['close'].shape)
    options = study.execution_options(CANDIDATE, scenario)
    result = study.replay(p, ix, scores, ix['dates'][1], ix['dates'][2], **options, return_trades=True, planning_price='open')
    assert result['trades']
    audit = study.audit_fills(p, ix, result, slip=options['slip'])
    check = study.additional_checks(p, ix, result, None, **{n: options[n] for n in ['slip', 'participation', 'max_orders']})
    assert not audit['post_buy_limit_violations'] and audit['maximum_nav_reconstruction_error_krw'] < .01
    assert not check['additional_errors'] and check['observed_max_daily_orders'] <= options['max_orders']
    assert all(t['qty'] <= np.floor(20000 * options['participation']) for t in result['trades'])
    buys = [t for t in result['trades'] if t['side'] == 'buy']
    assert all(t['price'] == pytest.approx(10000 * (1 + options['slip'])) for t in buys)


def test_base_scenario_is_all_fields_identical_to_original_call():
    p, ix = synthetic_panel(days=4, names=8); p['sector_cap'][:] = .8
    scores = np.broadcast_to(np.arange(8., 0., -1.)[:, None], p['close'].shape)
    original = study.replay(p, ix, scores, ix['dates'][1], ix['dates'][2], top_n=12, weight=.05, max_orders=10,
        rebalance=5, rebalance_band=.0005, rank_buffer=24, return_trades=True, planning_price='open')
    stress = study.replay(p, ix, scores, ix['dates'][1], ix['dates'][2], **study.execution_options(CANDIDATE, study.SCENARIOS[0]),
        return_trades=True, planning_price='open')
    assert original == stress

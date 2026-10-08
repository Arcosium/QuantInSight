import json

import numpy as np
import pandas as pd
import pytest

from quant import timefolio_heatmap_adjustment_band as study
from test_timefolio_heatmap import synthetic_panel


CASE = 'n12__orders3__refresh1'


def test_full_grid_and_every_matched_control_exist_without_self_comparisons():
    accounts = {study.account_id(model, case['id'], buffer, band) for model in study.MODELS
                for case in study.CASES for buffer in study.BUFFERS for band in study.BANDS}
    assert len(accounts) == 1824 and len(study.MODELS) == 57
    family = study.new_family(); assert len(family) == 3456
    counts = dict(uniform=0, toprank=0, feasible=0)
    for key, comparator, reference in family:
        counts[study.model_info(key.split('__', 1)[0])['objective']] += 1
        assert key in accounts and key != reference
        assert reference is None if comparator == 'cash' else reference in accounts
    assert counts == dict(uniform=960, toprank=1152, feasible=1344)


def test_absolute_feasible_controls_keep_target_seed_case_and_band():
    model = 'cnn_feasible_absolute_h10_seed29'
    actual = dict(study.comparison_ids(model, CASE, 24, 50))
    suffix = '__n12__orders3__refresh1__buffer24__band50bp'
    assert actual == {
        'cash': None,
        'matched_mlp': 'mlp_feasible_absolute_h10_seed29' + suffix,
        'matched_untrained_cnn': 'cnn_untrained_seed29' + suffix,
        'online_nonimage': 'online_nonimage' + suffix,
        'same_cnn_band5bp': model + '__n12__orders3__refresh1__buffer24__band5bp',
        'matched_uniform_cnn': 'cnn_absolute_rank_h10_seed29' + suffix,
        'matched_toprank_cnn': 'cnn_toprank_absolute_h10_seed29' + suffix,
    }
    sector = dict(study.comparison_ids('cnn_toprank_sector_h10_seed43', CASE, 12, 25))
    assert sector['matched_uniform_cnn'] == 'cnn_rank_h10_seed43__n12__orders3__refresh1__buffer12__band25bp'
    assert len(study.comparison_ids('cnn_rank_h10_ensemble', CASE, 12, 100)) == 5
    with pytest.raises(ValueError): study.comparison_ids('cnn_untrained_ensemble', CASE, 12, 25)


def fixture_candidates():
    rows = []
    for member in ['seed17', 'seed29', 'seed43', 'ensemble']:
        model = 'cnn_feasible_sector_h10_' + member
        rows.append(dict(id=study.account_id(model, CASE, 12, 25), model=model, **study.model_info(model),
                         case=CASE, buffer=12, band_bp=25, **{'return': .1}, mdd=-.1,
                         positive_blocks=2, four_week_turnover_stop=False))
    key = rows[-1]['id']
    stats = [dict(id=key, comparator=comp, block=block, origin='adjustment_band', adjusted_p=.01, simultaneous_lower95=.0001)
             for comp, _ in study.comparison_ids(rows[-1]['model'], CASE, 12, 25) for block in [5, 10]]
    return pd.DataFrame(rows), pd.DataFrame(stats), key


def test_gate_requires_all_seeds_and_all_comparisons_in_both_blocks():
    frame, stats, key = fixture_candidates()
    assert study.candidate_ids(frame, stats) == [key]
    changed = frame.copy(); changed.loc[changed.member == 'seed43', 'return'] = -.01
    assert study.candidate_ids(changed, stats) == []
    changed = stats.copy(); changed.loc[changed.index[-1], 'adjusted_p'] = .026
    assert study.candidate_ids(frame, changed) == []
    with pytest.raises(AssertionError, match='comparisons'): study.candidate_ids(frame, stats.iloc[:-1])
    with pytest.raises(AssertionError, match='seeds'): study.candidate_ids(frame[frame.member != 'seed43'], stats)


def test_parent_guard_stops_before_unreviewed_outcomes_or_candidate_followup(monkeypatch, tmp_path):
    monkeypatch.setattr(study, 'PRIOR', tmp_path)
    with pytest.raises(FileNotFoundError): study.parent_ready()
    (tmp_path / 'validation_complete.json').write_text(json.dumps(dict(status='incomplete', portfolios=1368)))
    with pytest.raises(RuntimeError, match='validation'): study.parent_ready()
    (tmp_path / 'validation_complete.json').write_text(json.dumps(dict(status='numerical_validation_complete', portfolios=1368)))
    summary = dict(portfolios=1368, joint_hypotheses=7712, robust_candidate_gate_passed=['parent_candidate'])
    (tmp_path / 'evaluation_summary.json').write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match='Prioritize'): study.parent_ready()
    summary['robust_candidate_gate_passed'] = []
    (tmp_path / 'evaluation_summary.json').write_text(json.dumps(summary))
    assert study.parent_ready()['portfolios'] == 1368


@pytest.mark.parametrize('band', [25, 50, 100])
def test_wider_band_preserves_entries_and_smallcap_repair_at_planned_weight(band):
    p, ix = synthetic_panel(days=4, names=8)
    p['sector_cap'][:] = .8; p['market_cap'][:, 1:] = 5e11
    scores = np.broadcast_to(np.arange(8., 0., -1.)[:, None], p['close'].shape)
    result = study.replay(p, ix, scores, ix['dates'][1], ix['dates'][2], top_n=12, weight=.05,
                          max_orders=10, rebalance=5, rank_buffer=12, rebalance_band=band / 10000.,
                          planning_price='open', return_trades=True)
    assert result['daily'][0]['gross'] > .35
    assert any(t['side'] == 'sell' and t['date'] == ix['dates'][2] for t in result['trades'])
    assert result['daily'][-1]['gross'] <= .3
    assert max(sum(t['date'] == date for t in result['trades']) for date in ix['dates']) <= 10


@pytest.mark.parametrize('band', [25, 50, 100])
def test_wider_band_skips_small_continuing_adjustment_without_changing_entry(band):
    p, ix = synthetic_panel(days=4, names=1)
    p['sector_cap'][:] = .8
    for name in ['o', 'c', 'close', 'exec_price']:
        p[name][:, 2:] = 10200.
    p['exec_high'][:, 2:] = 10250.; p['exec_low'][:, 2:] = 10150.
    scores = np.ones(p['close'].shape)
    args = dict(top_n=12, weight=.05, max_orders=10, rebalance=1, rank_buffer=12,
                planning_price='open', return_trades=True)
    base = study.replay(p, ix, scores, ix['dates'][1], ix['dates'][2], rebalance_band=.0005, **args)
    wide = study.replay(p, ix, scores, ix['dates'][1], ix['dates'][2], rebalance_band=band / 10000., **args)
    assert base['trades'][0] == wide['trades'][0]
    assert len(base['trades']) == 2 and len(wide['trades']) == 1
    assert base['trades'][1]['side'] == 'sell'

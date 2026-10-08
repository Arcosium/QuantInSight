from collections import Counter
import json

import numpy as np
import pandas as pd
import pytest

from quant import timefolio_heatmap_sector_ceiling as study
from test_timefolio_heatmap import synthetic_panel

CASE = 'n12__orders10__refresh1'


def test_complete_grid_preserves_all_matched_comparators():
    keys = {study.account_id(model, case['id'], buffer, mode) for model in study.MODELS
            for case in study.CASES for buffer in study.BUFFERS for mode in study.MODES}
    assert len(study.MODELS) == 41 and len(keys) == 656
    family = study.new_family(); assert len(family) == 672
    counts = Counter()
    for key, comp, reference in family:
        counts[study.model_info(key.split('__', 1)[0])['relation']] += 1
        assert key in keys and key != reference
        assert reference is None if comp == 'cash' else reference in keys
        if reference is not None:
            assert reference.endswith('__sector_research20' if comp == 'same_cnn_research20' else '__sector_formula')
    assert counts == dict(self=192, mean=224, attention=256)


def test_sector_panel_changes_only_extra_ceiling_and_preserves_original():
    p, _ = synthetic_panel(days=4, names=3)
    p['sector_cap'][:] = np.array([.1, .4, 1.2])[:, None]; original = p['sector_cap'].copy()
    base = study.sector_panel(p, 'research20'); formula = study.sector_panel(p, 'formula')
    np.testing.assert_array_equal(base['sector_cap'], np.minimum(original, .2))
    np.testing.assert_array_equal(formula['sector_cap'], original)
    for name in p:
        if name != 'sector_cap': assert base[name] is formula[name] is p[name]
    base['sector_cap'][:] = .01; formula['sector_cap'][:] = .02
    np.testing.assert_array_equal(p['sector_cap'], original)
    with pytest.raises(ValueError): study.sector_panel(p, 'unrestricted')
    p['sector_cap'][0, 0] = np.nan
    with pytest.raises(ValueError): study.sector_panel(p, 'formula')


@pytest.mark.parametrize('smallcap', [False, True])
def test_formula_sector_mode_still_enforces_cash_stock_and_smallcap_constraints(smallcap):
    p, ix = synthetic_panel(days=4, names=12); p['sector'][:] = 45; p['sector_cap'][:] = 1.2
    p['market_cap'][:] = 5e11 if smallcap else 2e12
    scores = np.broadcast_to(np.arange(12., 0., -1.)[:, None], p['close'].shape)
    results = {}
    for mode in study.MODES:
        panel = study.sector_panel(p, mode)
        result = study.replay(panel, ix, scores, ix['dates'][1], ix['dates'][2], top_n=12, weight=.05,
            max_orders=10, rebalance=5, rebalance_band=.0005, rank_buffer=24, return_trades=True, planning_price='open')
        check = study.audit_fills(panel, ix, result)
        assert not check['post_buy_limit_violations'] and check['maximum_nav_reconstruction_error_krw'] < .01
        assert all(r['cash'] >= 0 and r['gross'] <= (.3 if smallcap else .8) for r in result['daily'])
        results[mode] = result
    assert results['formula']['daily'][0]['gross'] > results['research20']['daily'][0]['gross']
    assert results['research20']['daily'][0]['gross'] <= .20


def test_formula_mode_reacts_to_tighter_dated_sector_limit():
    p, ix = synthetic_panel(days=4, names=12); p['sector'][:] = 45
    p['sector_cap'][:] = .4; p['sector_cap'][:, 1:] = .1
    scores = np.broadcast_to(np.arange(12., 0., -1.)[:, None], p['close'].shape)
    panel = study.sector_panel(p, 'formula')
    result = study.replay(panel, ix, scores, ix['dates'][1], ix['dates'][2], top_n=12, weight=.05, max_orders=10,
        rebalance=5, rebalance_band=.0005, rank_buffer=24, return_trades=True, planning_price='open')
    assert result['daily'][0]['gross'] > .3 and result['daily'][1]['gross'] <= .1
    assert any(t['side'] == 'sell' and t['date'] == ix['dates'][2] for t in result['trades'])
    assert not study.audit_fills(panel, ix, result)['post_buy_limit_violations']


def fixture():
    rows = []
    for member in study.MEMBERS:
        model = 'cnn_relation_attention_h10_' + member
        rows.append(dict(id=study.account_id(model, CASE, 24, 'formula'), model=model, **study.model_info(model),
            case=CASE, buffer=24, ceiling='formula', **{'return': .1}, mdd=-.1, positive_blocks=2, four_week_turnover_stop=False))
    key = rows[-1]['id']
    stats = pd.DataFrame([dict(id=key, comparator=comp, origin='sector_ceiling', block=block, adjusted_p=.01, simultaneous_lower95=.0001)
        for comp, _ in study.comparison_ids(rows[-1]['model'], CASE, 24) for block in [5, 10]])
    return pd.DataFrame(rows), stats, key


def test_gate_needs_old_ceiling_comparison_and_every_seed():
    frame, stats, key = fixture(); assert study.candidate_ids(frame, stats) == [key]
    changed = stats.copy(); changed.loc[changed.comparator == 'same_cnn_research20', 'adjusted_p'] = .025
    assert study.candidate_ids(frame, changed) == []
    changed = frame.copy(); changed.loc[changed.member == 'seed29', 'return'] = 0.
    assert study.candidate_ids(changed, stats) == []
    with pytest.raises(AssertionError, match='comparisons'): study.candidate_ids(frame, stats.iloc[:-1])


def test_parent_guard_preserves_confirmation_priority_and_leaves_no_output(monkeypatch, tmp_path):
    parent, dest = tmp_path / 'parent', tmp_path / 'never_created'; parent.mkdir()
    monkeypatch.setattr(study, 'PRIOR', parent); monkeypatch.setattr(study, 'DEST', dest)
    with pytest.raises(FileNotFoundError): study.freeze()
    assert not dest.exists()
    review = dict(status='incomplete', portfolios=744, candidates=[])
    (parent / 'validation_complete.json').write_text(json.dumps(review))
    with pytest.raises(RuntimeError, match='validation'): study.freeze()
    review['status'] = 'numerical_validation_complete'; (parent / 'validation_complete.json').write_text(json.dumps(review))
    (parent / 'evaluation_summary.json').write_text(json.dumps(dict(portfolios=744, joint_hypotheses=11744, robust_candidate_gate_passed=['candidate'])))
    with pytest.raises(RuntimeError, match='Prioritize'): study.freeze()
    assert not dest.exists()

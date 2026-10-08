from collections import Counter

import pandas as pd
import pytest

from quant import timefolio_heatmap_stock_relation_evaluation as study

CASE = 'n12__orders3__refresh1'


def test_registered_grid_and_all_matched_controls_exist():
    accounts = {study.account_id(model, case['id'], buffer) for model in study.MODELS for case in study.CASES for buffer in study.BUFFERS}
    assert len(accounts) == 744 and len(study.MODELS) == 93 and len(study.LEGACY) == 57
    assert len(study.TRAINED) == 24 and len(study.UNTRAINED) == 12
    family = study.new_family(); assert len(family) == 576
    count = Counter()
    for key, comp, reference in family:
        count[study.model_info(key.split('__', 1)[0])['relation']] += 1
        assert key in accounts and key != reference
        assert reference is None if comp == 'cash' else reference in accounts
    assert count == dict(self=160, mean=192, attention=224)


def test_attention_comparators_match_seed_policy_and_architecture():
    name = 'cnn_relation_attention_h10_seed29'
    suffix = '__n12__orders3__refresh1__buffer24__band5bp'
    assert dict(study.comparison_ids(name, CASE, 24)) == dict(
        cash=None, matched_mlp='mlp_relation_attention_h10_seed29' + suffix,
        matched_untrained_cnn='cnn_relation_attention_untrained_seed29' + suffix,
        matched_uniform_cnn='cnn_absolute_rank_h10_seed29' + suffix,
        online_nonimage='online_nonimage' + suffix,
        matched_self_cnn='cnn_relation_self_h10_seed29' + suffix,
        matched_mean_cnn='cnn_relation_mean_h10_seed29' + suffix)
    with pytest.raises(ValueError): study.comparison_ids('cnn_relation_attention_untrained_seed29', CASE, 24)
    with pytest.raises(ValueError): study.comparison_ids('mlp_relation_self_h10_seed17', CASE, 12)


def fixture():
    rows = []
    for member in study.MEMBERS:
        name = 'cnn_relation_attention_h10_' + member
        rows.append(dict(id=study.account_id(name, CASE, 12), model=name, **study.model_info(name),
                         case=CASE, buffer=12, **{'return': .1}, mdd=-.1, positive_blocks=2, four_week_turnover_stop=False))
    key = rows[-1]['id']
    stats = [dict(id=key, comparator=comp, origin='stock_relation', block=block, adjusted_p=.01, simultaneous_lower95=.0001)
             for comp, _ in study.comparison_ids(rows[-1]['model'], CASE, 12) for block in [5, 10]]
    return pd.DataFrame(rows), pd.DataFrame(stats), key


@pytest.mark.parametrize('field,value', [('return', 0.), ('mdd', -.2), ('positive_blocks', 1), ('four_week_turnover_stop', True)])
def test_one_failed_seed_rejects_ensemble(field, value):
    frame, stats, key = fixture(); assert study.candidate_ids(frame, stats) == [key]
    frame.loc[frame.member == 'seed29', field] = value
    assert study.candidate_ids(frame, stats) == []


def test_gate_requires_complete_blocks_and_all_relational_controls():
    frame, stats, key = fixture()
    assert study.candidate_ids(frame, stats) == [key]
    changed = stats.copy(); changed.loc[changed.index[-1], 'adjusted_p'] = .025
    assert study.candidate_ids(frame, changed) == []
    changed = stats.copy(); changed.loc[changed.index[-1], 'simultaneous_lower95'] = 0.
    assert study.candidate_ids(frame, changed) == []
    with pytest.raises(AssertionError, match='comparisons'): study.candidate_ids(frame, stats.iloc[:-1])
    with pytest.raises(AssertionError, match='comparisons'): study.candidate_ids(frame, pd.concat([stats, stats.iloc[:1]]))
    with pytest.raises(AssertionError, match='seeds'): study.candidate_ids(frame[frame.member != 'seed43'], stats)

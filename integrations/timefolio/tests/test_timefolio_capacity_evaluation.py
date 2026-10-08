import pandas as pd
import pytest
from quant import timefolio_heatmap_capacity_evaluation as study


def test_grid_retains_full_input_controls_and_every_seed():
    accounts = {study.account_id(m, c['id'], b, s) for m in study.MODELS
                for c in study.CASES for b in study.BUFFERS for s in study.MODES}
    assert len(study.CONFIGS) == 8 and len(study.MODELS) == 65
    assert len(accounts) == 1040 and len(study.new_family()) == 4032
    for key, label, other in study.new_family():
        assert key in accounts
        if other:
            assert other in accounts and other != key
            if label != 'same_model_research20':
                assert key.split('__', 1)[1] == other.split('__', 1)[1]
        info = study.model_info(key.split('__')[0])
        labels = {c for c, _ in study.comparison_ids(key.split('__')[0],
                  study.CASES[0]['id'], study.BUFFERS[0], 'formula')}
        assert info['geometry'] not in labels
        assert set(study.CONTROLS) - {info['geometry']} <= labels


def test_candidate_requires_all_comparators_and_seeds():
    case = study.CASES[0]['id']; buffer = study.BUFFERS[0]; rows = []
    for member in study.MEMBERS:
        model = 'cap_cnn_width96_trained_' + member
        rows.append(dict(id=study.account_id(model, case, buffer, 'formula'), model=model,
            case=case, buffer=buffer, ceiling='formula', **study.model_info(model),
            **{'return': .1, 'mdd': -.1, 'positive_blocks': 3, 'four_week_turnover_stop': False}))
    frame = pd.DataFrame(rows); row = rows[-1]
    stats = pd.DataFrame([dict(id=row['id'], origin='capacity_lab', comparator=c, block=b,
        adjusted_p=.01, simultaneous_lower95=.001)
        for c, _ in study.comparison_ids(row['model'], case, buffer, 'formula') for b in [5, 10]])
    assert study.candidate_ids(frame, stats) == [row['id']]
    for comparator in ['full_linear', 'full_mlp128', 'summary_mlp']:
        changed = stats.copy(); changed.loc[changed.comparator == comparator, 'adjusted_p'] = .1
        assert study.candidate_ids(frame, changed) == []
        with pytest.raises(AssertionError):
            study.candidate_ids(frame, stats[stats.comparator != comparator])
    frame.loc[0, 'return'] = -.1
    assert study.candidate_ids(frame, stats) == []

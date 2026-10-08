import json

from quant.timefolio_heatmap_cash_accounts import hypotheses, units, publish_threshold, can_start


def monthly(pooled, low):
    return dict(pooled=dict(sharpe=pooled, sharpe_defined=True),
        folds=[dict(month=f'2026{m:02d}', sharpe=low, sharpe_defined=True) for m in range(1, 10)],
        annual_observations=252, later_folds_use_previous_close=True)


def test_stop_is_strict_persistent_and_preserves_every_record(tmp_path):
    assert can_start(tmp_path)
    assert not publish_threshold(tmp_path, 'boundary', monthly(2., 1.1))['retain_candidate']
    assert not (tmp_path/'candidates').exists()
    assert publish_threshold(tmp_path, 'record', monthly(2.5, 1.1))['retain_candidate']
    assert can_start(tmp_path)
    publish_threshold(tmp_path, 'stop_boundary', monthly(3.1, 1.5))
    assert can_start(tmp_path)
    publish_threshold(tmp_path, 'first', monthly(3.1, 1.6))
    assert not can_start(tmp_path)
    publish_threshold(tmp_path, 'second', monthly(3.2, 1.7))
    assert json.loads((tmp_path/'STOP.json').read_text())['id'] == 'first'
    assert len(list((tmp_path/'candidates').glob('*.json'))) == 4


def test_manifest_covers_all_members_and_paired_controls():
    rows = hypotheses(['case_a', 'case_b'])
    assert len(units(['case_a', 'case_b'])) == 16
    assert len(rows) == 2 * 4032
    keys = {r[0] for r in rows}
    assert len(keys) == 2 * 1792
    for key, label, reference in rows:
        if label in ['matching_untrained_pipeline', 'same_exposure_nonimage']:
            assert reference in keys
        if label == 'matching_untrained_pipeline':
            assert '_trained_' in key and '_untrained_' in reference
        assert len(key+'.json') <= 255

import pytest
from quant.timefolio_cnn_sector_sensitivity import snapshot_limits


def snapshot():
    sectors = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    weights = [1.02, 2.88, 18.3, 5.53, 1.94, 4.99, 7.28, 55.07, 2.46, .51, .03]
    return dict(requested_date='2026-08-20', sectors=[dict(sector=str(s), market_weight_percent=w) for s, w in zip(sectors, weights)])


def test_sector_weights_are_percentages_and_actual_formula_can_exceed100percent():
    result = snapshot_limits(snapshot())
    assert result[10] == .1 and result[20] == pytest.approx(.366)
    assert result[45] == pytest.approx(1.1014)


def test_empty_duplicate_or_nonpercentage_snapshot_is_rejected():
    for transform in [lambda r: [], lambda r: r[:-1]+[r[0]],
                      lambda r: [dict(v, market_weight_percent=v['market_weight_percent']/100) for v in r]]:
        data = snapshot(); data['sectors'] = transform(data['sectors'])
        with pytest.raises(ValueError):
            snapshot_limits(data)


def test_reserved_future_sector_snapshot_is_rejected():
    data = snapshot(); data['requested_date'] = '2026-10-02'
    with pytest.raises(ValueError, match='reserved'):
        snapshot_limits(data)

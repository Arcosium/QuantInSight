import numpy as np
import pytest
import torch
from quant.timefolio_heatmap_daily_history import DAILY_ROWS, FIELDS, daily_values, quantize
from quant.timefolio_heatmap_context_spacing import encode
from quant.timefolio_heatmap_context_spacing_models import LabNet, cases, evaluation_cases, feature_control
from quant.timefolio_heatmap_feature_models import cases as feature_cases, LabNet as FeatureNet


def fixture():
    panel = {k: np.ones((2, 40), dtype=np.float32) for k in FIELDS}
    for k in ['r1', 'r5', 'r20', 'vol20', 'sector_r5', 'market_r5']:
        panel[k] *= np.arange(40, dtype=np.float32) / 100
    panel['market_cap'] *= 2e12; panel['adv5'] *= 6e9
    panel['rank_adv'][:] = np.arange(40, dtype=np.float32) / 40
    ci, di = np.array([0, 1]), np.array([20, 25])
    history = np.arange(2 * 32 * 65, dtype=np.uint8).reshape(2, 32, 65)
    history[:, DAILY_ROWS] = np.repeat(quantize(daily_values(panel, ci, di[:, None] + np.arange(-4, 1))), 13, axis=2)
    return panel, ci, di, history


def test_spacing_uses_actual_archived_dates_and_preserves_intraday():
    panel, ci, di, history = fixture()
    for stride in [1, 2, 4]:
        out = encode(history, panel, ci, di, stride, 'constant')
        keep = [r for r in range(32) if r not in DAILY_ROWS]
        np.testing.assert_array_equal(out[:, 0, keep], history[:, keep])
        np.testing.assert_array_equal(out[:, 0, :, -13:], history[:, :, -13:])
        assert np.all(out[:, 1] == 255)
        for slot, lag in enumerate(range(-4 * stride, 1, stride)):
            expected = quantize(daily_values(panel, ci, (di + lag)[:, None]))
            np.testing.assert_array_equal(out[:, 0, DAILY_ROWS, slot*13:(slot+1)*13], np.repeat(expected, 13, axis=2))
        if stride == 1:
            np.testing.assert_array_equal(out[:, 0], history)


def test_missingness_is_explicit_and_future_data_cannot_change_images():
    panel, ci, di, history = fixture()
    panel['rank_adv'][0, 12] = np.nan
    out = encode(history, panel, ci, di, 4, 'available')
    assert (out[0, 1, 15, 26:39] == 0).all()
    assert (out[0, 0, 15, 26:39] == 128).all()
    for key in FIELDS:
        for c, d in zip(ci, di):
            panel[key][c, d+1:] = -999999
    np.testing.assert_array_equal(out, encode(history, panel, ci, di, 4, 'available'))


def test_invalid_axes_and_out_of_range_history_rejected():
    panel, ci, di, history = fixture()
    with pytest.raises(ValueError): encode(history, panel, ci, di, 3, 'constant')
    with pytest.raises(ValueError): encode(history, panel, ci, np.array([15, 25]), 4, 'constant')
    with pytest.raises(ValueError): encode(history, panel, ci.astype(float), di, 1, 'constant')


def test_models_exactly_match_existing_two_channel_controls():
    torch.set_num_threads(1)
    original = {c['id']: c for c in feature_cases()}
    assert len(cases()) == 8 and len(evaluation_cases()) == 12
    x = torch.zeros((3, 2, 32, 65))
    for cfg in evaluation_cases():
        torch.manual_seed(17); model = LabNet(cfg).eval()
        refcfg = original[cfg['kind'] + '_mask_' + cfg['availability_mask']]
        torch.manual_seed(17); ref = FeatureNet(refcfg).eval()
        for key, value in model.state_dict().items():
            assert torch.equal(value, ref.state_dict()[key])
        assert torch.equal(model(x), ref(x))
        assert (feature_control(cfg) is not None) == (cfg['stride'] == 1)

import numpy as np
import json

from quant.timefolio_cnn_dataset import sha
from quant.timefolio_cnn_market_gate import market_records, walk_market_gate, derive_gate


def test_market_basket_membership_and_features_do_not_use_future_targets():
    days, stocks = 35, 4
    price = np.ones((days, stocks, 5)); price[:, :, :4] *= np.arange(days)[:,None,None]+100
    arrays = dict(daily_ohlcv=price, adv5_proxy=np.tile([4.,3.,2.,1.], (days,1)),
        signal_index=np.array([25]*stocks),security_key=np.arange(stocks),
        eligible=np.ones(stocks,bool),returns=np.array([.05,.03,.01,-.01]))
    tradable = np.ones((days,stocks),bool)
    first = market_records(arrays,tradable,horizon=5,top_k=2)[0]
    arrays['daily_ohlcv'][26:,:,:4] *= 10
    arrays['returns'][0] = np.nan
    second = market_records(arrays,tradable,horizon=5,top_k=2)[0]
    assert first['features'] == second['features']
    assert first['selected_security_keys'] == second['selected_security_keys'] == [0,1]
    assert second['net_excess'] is None  # Never replace a suspended target with a surviving stock.


def test_market_fit_excludes_unmatured_outcomes_and_later_features():
    rng = np.random.default_rng(17)
    records = [dict(signal=i,label_end=i+6,features=rng.normal(size=4).tolist(),
                    net_excess=float(rng.normal(.005,.02))) for i in range(140)]
    folds = [dict(id='test',test_start=110,test_end=130)]
    original = walk_market_gate(records,folds,days=140)
    for r in records:
        if r['label_end'] >= 110:r['net_excess'] = 9000.
        if r['signal'] > 110:r['features'] = [9000.]*4
    changed = walk_market_gate(records,folds,days=140)
    assert changed[0][110] == original[0][110]
    assert changed[3] == original[3]
    assert original[3][0]['last_label_end'] < 110


def test_missing_market_labels_keep_forecast_dates_and_cold_start_explicit():
    records = [dict(signal=i,label_end=i+6,features=[float(i),0.,.5,.01],
                    net_excess=None) for i in range(80)]
    expected,cash,floor,fits = walk_market_gate(records,[dict(id='x',test_start=70,test_end=80)],days=80)
    assert np.isnan(expected[70:]).all()
    assert (cash[70:]==.8).all() and (floor[70:]==.8).all()
    assert fits[0]['cold_start'] and fits[0]['matured_dates']==0


def test_derived_market_gate_preserves_cnn_and_control_schedules(tmp_path):
    original, market, output = [tmp_path/n for n in ['original','market','output']]
    original.mkdir();market.mkdir()
    np.save(original/'scores.npy',np.arange(21).reshape(7,3))
    original_arrays={n:np.full(7,v) for n,v in [('gross_always',.8),('gross_trend_floor20',.2),
        ('gross_cash',0.),('gross_ridge_cash',.8),('gross_ridge_floor20',.8)]}
    np.savez(original/'gate.npz',**original_arrays,features=np.ones((3,8)))
    np.savez(market/'market_gate.npz',expected_net=np.zeros(7),cold_start=np.zeros(7,bool),
        gross_ridge_cash=np.zeros(7),gross_ridge_floor20=np.full(7,.2))
    np.savez(market/'market_records.npz',features=np.ones((5,4)),signal_index=np.arange(5),
        net_excess=np.ones(5),label_end=np.arange(5)+6)
    for root in [original,market]:
        (root/'receipt.json').write_text(json.dumps(dict(dataset_manifest_sha256='same',
            observations=5,cold_start_folds=[],artifacts={p.name:sha(p) for p in root.iterdir()})))
    parent_hash=sha(original/'receipt.json')
    derive_gate(original,market,output)
    assert sha(original/'receipt.json')==parent_hash
    assert sha(original/'scores.npy')==sha(output/'scores.npy')
    with np.load(output/'gate.npz') as z:
        assert z['features'].shape==(5,4)
        for name in ['gross_always','gross_trend_floor20','gross_cash']:
            np.testing.assert_array_equal(z[name],original_arrays[name])
        assert (z['gross_ridge_cash']==0).all()

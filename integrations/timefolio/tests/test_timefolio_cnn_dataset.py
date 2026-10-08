import csv
from datetime import date,timedelta
import gzip
import json
import numpy as np

from quant.timefolio_cnn_core import heatmap
from quant.timefolio_cnn_dataset import build, encode_windows, select_samples, sha, register_folds
from quant.timefolio_cnn_core import purged_masks


def test_vectorized_images_match_reference_for_different_price_and_volume_scales():
    rng = np.random.default_rng(4)
    p = np.exp(rng.normal(8, .1, (11, 20)))
    a = np.stack([p, p*1.03, p*.94, p*.98, rng.integers(0, 100000, p.shape)], -1)
    actual = encode_windows(a)
    expected = np.stack([heatmap(w) for w in a])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_later_common_price_rescaling_cancels_within_observed_image():
    p=np.linspace(100,130,20)
    a=np.stack([p,p*1.02,p*.98,p*1.01,np.arange(20)+10],-1)[None]
    changed=a.copy();changed[:,:,:4]*=5
    np.testing.assert_allclose(encode_windows(a),encode_windows(changed),rtol=1e-6,atol=1e-7)


def test_warmup_registration_preserves_126_fit_and_40_validation_dates():
    dates=np.array([(date(2022,11,23)+timedelta(days=i)).strftime('%Y%m%d') for i in range(500)])
    for window in [20,60]:
        for horizon in [5,10]:
            folds=register_folds(dates,window=window,horizon=horizon,first_month='202211')
            first=folds[0];s=np.arange(window-1,len(dates));end=s+horizon+1
            masks=purged_masks(s,end,np.ones(len(s),bool),validation_start=first['validation_start'],test_start=first['test_start'],test_end=first['test_end'])
            assert masks['train'].sum()>=126 and masks['validation'].sum()==40
            assert end[masks['train']].max()<first['validation_start']
            assert end[masks['validation']].max()<first['test_start']


def test_signal_membership_does_not_see_future_prices_or_missing_exit():
    p = np.ones((50, 12, 5))*100
    p[:, :, 1] = 110; p[:, :, 2] = 90
    v = np.ones((50, 12))*4e9
    first, _ = select_samples(p, v)
    p[30:] = np.nan; v[30:] = np.nan
    second, _ = select_samples(p, v)
    np.testing.assert_array_equal(first[:30], second[:30])
    assert first[29].sum() == 12


def test_liquidity_is_past_only_and_ties_keep_stable_code_order():
    p = np.ones((50, 15, 5))*100
    v = np.ones((50, 15))*4e9
    v[25:, 14] = 20e9
    selected, _ = select_samples(p, v, maximum=10)
    assert np.flatnonzero(selected[24]).tolist() == list(range(10))
    assert selected[25, 14] and not selected[25, 9]
    p[21, 0, 0] = np.nan
    selected, _ = select_samples(p, v, maximum=10)
    assert not selected[30, 0]


def test_reference_replacement_preserves_raw_turnover_cohort_and_execution_panel(tmp_path):
    source=tmp_path/'source';shards=source/'daily_candidates';shards.mkdir(parents=True)
    reference=tmp_path/'reference';reference.mkdir()
    codes=['005930']+[f'{i:06d}' for i in range(10)]
    days=[(date(2024,1,1)+timedelta(days=i)).strftime('%Y%m%d') for i in range(65)]
    plan=dict(codes=codes,start='20221123',cutoff='20260923')
    (source/'source_plan.json').write_text(json.dumps(plan))
    for code in codes:
        rows=[dict(code=code,date=d,open=100,high=110,low=90,close=100,volume=1e8,
            typical_value_proxy=4e9 if code!='000009' else 2e9) for d in days]
        path=shards/(code+'.csv.gz')
        with gzip.open(path,'wt',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (shards/(code+'.json')).write_text(json.dumps(dict(code=code,days=len(days),sha256=sha(path),plan_sha256=sha(source/'source_plan.json'))))
        daily=[dict(date=d,open=500,high=550,low=450,close=500,volume=1e8) for d in days]
        p=reference/(code+'.json');p.write_text(json.dumps(dict(code=code,start=plan['start'],end=plan['cutoff'],rows=daily)))
        (reference/(code+'.receipt.json')).write_text(json.dumps(dict(sha256=sha(p))))
    first=tmp_path/'first';second=tmp_path/'second'
    raw_proof=build(source,first)
    assert not raw_proof['exploratory_ready']
    proof=build(source,second,price_reference=reference,cohort=first/'cohort.json')
    for name in ['signal_index','security_key','eligible','adv5_proxy','daily_value_proxy','images','returns']:
        np.testing.assert_allclose(np.load(first/(name+'.npy')),np.load(second/(name+'.npy')),equal_nan=True)
    np.testing.assert_array_equal(np.load(first/'daily_ohlcv.npy'),np.load(second/'raw_daily_ohlcv.npy'))
    keys=np.load(second/'security_key.npy');names=np.load(second/'codes.npy')
    assert '000009' not in names[keys]  # adjusted price must not admit the illiquid code
    assert proof['price_reference_sha256'] and proof['exploratory_ready'] and not proof['contest_certified']

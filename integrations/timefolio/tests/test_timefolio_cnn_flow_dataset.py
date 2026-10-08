import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quant.timefolio_cnn_dataset import sha
from quant.timefolio_cnn_flow_context import FEATURES
from quant.timefolio_cnn_flow_dataset import build


@pytest.fixture(autouse=True)
def fixture_disk_margin(monkeypatch):
    monkeypatch.setattr('quant.timefolio_cnn_flow_dataset.shutil.disk_usage',
        lambda path: SimpleNamespace(free=20*1024**3))


def fixture(tmp_path):
    parent=tmp_path/'parent';parent.mkdir();account=tmp_path/'account';account.mkdir();flow=tmp_path/'flow';flow.mkdir()
    dates=np.array(['20230102','20230103','20230104','20230105','20230106','20230109'])
    codes=np.array(['000001','000002'])
    arrays={'dates':dates,'codes':codes,'images':np.arange(3*16*4,dtype=np.float32).reshape(3,1,16,4)/200,
        'signal_index':np.array([3,4,5]),'security_key':np.array([0,1,0]),'eligible':np.ones(3,bool),
        'returns':np.array([.1,-.2,np.nan]),'label_end_index':np.array([5,6,7])}
    for name,value in arrays.items():np.save(parent/(name+'.npy'),value,allow_pickle=False)
    manifest={'feature_rows':16,'exploratory_ready':True,'window':4,'rows':3,'folds':[{'id':'202301'}],
        'limitations':['Synthetic fixture only'],'arrays':{name:{'path':name+'.npy','sha256':sha(parent/(name+'.npy'))} for name in arrays}}
    (parent/'manifest.json').write_text(json.dumps(manifest))
    (account/'index.json').write_text(json.dumps({'dates':dates.tolist(),'codes':codes.tolist()}))
    np.savez(account/'panel.npz',marker=np.array([17]))
    (account/'receipt.json').write_text(json.dumps({'dataset_manifest_sha256':sha(parent/'manifest.json'),
        'artifacts':{name:sha(account/name) for name in ['index.json','panel.npz']}}))
    rows=np.zeros((6,2,6),np.float32);rows[:,:,0]=.5;rows[:,:,4:]=1;rows[0,:,4:]=-1
    for name,value in [('dates',dates),('codes',codes),('rows',rows)]:np.save(flow/(name+'.npy'),value)
    source=Path(__file__).parents[1]/'quant'
    (flow/'receipt.json').write_text(json.dumps({'parent_manifest_sha256':sha(parent/'manifest.json'),
        'lag_sessions':1,'features':FEATURES,'composed_image_height':22,
        'source_module_sha256':sha(source/'timefolio_cnn_flow_context.py'),
        'rank_module_sha256':sha(source/'timefolio_cnn_context.py'),
        'artifacts':{name+'.npy':sha(flow/(name+'.npy')) for name in ['dates','codes','rows']}}))
    return parent,account,flow


def test_three_datasets_preserve_unlabeled_predictions_labels_and_account_bytes(tmp_path):
    parent,account,flow=fixture(tmp_path);out=tmp_path/'derived'
    before={p.name:sha(p) for p in parent.iterdir()}
    build(parent,account,flow,out)
    original=np.load(parent/'images.npy');results={}
    for mode in ['padded','missingness','flow']:
        dataset=out/mode/'dataset';m=json.loads((dataset/'manifest.json').read_text())
        assert m['feature_rows']==22 and m['flow_mode']==mode and m['rows']==3
        for name in ['returns','eligible','signal_index','security_key','label_end_index']:
            assert sha(dataset/(name+'.npy'))==sha(parent/(name+'.npy'))
        assert np.isnan(np.load(dataset/'returns.npy')[-1]) and np.load(dataset/'eligible.npy')[-1]
        results[mode]=np.load(dataset/'images.npy');np.testing.assert_array_equal(results[mode][:,:,:16],original)
        proof=json.loads((out/mode/'account_panel/receipt.json').read_text())
        assert proof['dataset_manifest_sha256']==sha(dataset/'manifest.json')
        for name in ['index.json','panel.npz']:assert sha(out/mode/'account_panel'/name)==sha(account/name)
    assert not results['padded'][:,:,16:].any() and not results['missingness'][:,:,16:20].any()
    np.testing.assert_array_equal(results['missingness'][:,:,20:],results['flow'][:,:,20:])
    assert results['flow'][:,:,16:20].any()
    assert {p.name:sha(p) for p in parent.iterdir()}==before


def test_changed_flow_or_account_identity_is_rejected_before_writing(tmp_path):
    parent,account,flow=fixture(tmp_path);out=tmp_path/'derived'
    p=account/'receipt.json';value=json.loads(p.read_text());value['dataset_manifest_sha256']='wrong';p.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='Account panel'):build(parent,account,flow,out)
    assert not out.exists()
    value['dataset_manifest_sha256']=sha(parent/'manifest.json');p.write_text(json.dumps(value))
    np.save(flow/'rows.npy',np.ones((6,2,6),np.float32))
    with pytest.raises(ValueError,match='artifact changed'):build(parent,account,flow,out)
    assert not out.exists()

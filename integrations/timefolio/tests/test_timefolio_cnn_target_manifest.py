import json

import numpy as np
import pytest

from quant.timefolio_cnn_train import load_dataset, sha
from quant.timefolio_cnn_utility import build


def test_utility_derivation_preserves_economic_labels_and_prediction_membership(tmp_path):
    parent=tmp_path/'parent';parent.mkdir();panel=tmp_path/'panel';panel.mkdir()
    daily=np.full((30,2,5),100.,dtype=float);daily[:,:,1]=105;daily[:,:,2]=95
    signal=np.array([20,21,28],dtype=np.int32)
    arrays=dict(images=np.zeros((3,1,8,20),np.float32),daily_ohlcv=daily,
                returns=np.array([0.,0.,np.nan]),signal_index=signal,label_end_index=signal+4,
                security_key=np.array([0,1,0],np.int32),eligible=np.ones(3,bool),
                dates=np.array([f'202401{i:02}' for i in range(1,31)]))
    specs={}
    for name,array in arrays.items():
        p=parent/(name+'.npy');np.save(p,array);specs[name]=dict(path=p.name,sha256=sha(p))
    manifest=dict(exploratory_ready=True,contest_certified=False,limitations=['Unit-fixture prices'],
                  readiness=dict(feature_availability=True,label_construction=True),arrays=specs,
                  horizon=3,rows=3)
    (parent/'manifest.json').write_text(json.dumps(manifest))
    (parent/'cohort.json').write_text('{}')
    (panel/'receipt.json').write_text(json.dumps(dict(dataset_manifest_sha256=sha(parent/'manifest.json'),artifacts={},limitations=['Unit-fixture panel'])))
    original=sha(parent/'returns.npy');output=tmp_path/'derived'
    build(parent,panel,output,penalty=.5)
    proof,loaded=load_dataset(output/'dataset/manifest.json',purpose='exploratory')
    assert proof['training_target']=='training_utility'
    assert sha(parent/'returns.npy')==sha(output/'dataset/returns.npy')==original
    np.testing.assert_array_equal(loaded['eligible'],arrays['eligible'])
    np.testing.assert_allclose(loaded['returns'],arrays['returns'],equal_nan=True)
    assert (loaded['training_utility'][:2]<0).all() and np.isnan(loaded['training_utility'][2])
    assert loaded['eligible'][2]  # unlabelled future outcome remains predictable
    np.save(output/'dataset/training_utility.npy',np.zeros(3))
    with pytest.raises(ValueError,match='hash mismatch'):
        load_dataset(output/'dataset/manifest.json',purpose='exploratory')

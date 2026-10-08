"""Flow geometry, model gradients and time boundaries; no profit claims."""
import json

import numpy as np
import pytest
import torch

import quant.timefolio_cnn_core as staged_core
from quant.timefolio_cnn_train import fit, load_dataset, predict, sha

def test_22row_forward_backpropagates_into_flow_and_missingness_rows():
    torch.set_num_threads(1);torch.manual_seed(17)
    model=staged_core.RankCNN(width=2,dropout=0)
    image=torch.rand(3,1,22,20,requires_grad=True)
    model(image).sum().backward()
    assert image.grad[:,:,16:20].abs().sum()>0
    assert image.grad[:,:,20:22].abs().sum()>0


def test_loader_requires_explicit_22row_geometry_and_retains_unmatured_sample(tmp_path):
    arrays=dict(images=np.zeros((2,1,22,20),np.float32),returns=np.array([.01,np.nan]),
        signal_index=np.array([0,1]),label_end_index=np.array([2,3]),security_key=np.array([0,1]),
        eligible=np.ones(2,bool),dates=np.array(['20240102','20240103']))
    specs={}
    for name,array in arrays.items():
        path=tmp_path/(name+'.npy');np.save(path,array);specs[name]=dict(path=path.name,sha256=sha(path))
    manifest=dict(exploratory_ready=True,contest_certified=False,limitations=['Synthetic fixture'],
        readiness=dict(feature_availability=True,label_construction=True),arrays=specs)
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='geometry'):load_dataset(path,purpose='exploratory')
    manifest['feature_rows']=22;path.write_text(json.dumps(manifest))
    _,loaded=load_dataset(path,purpose='exploratory')
    assert loaded['images'].shape==(2,1,22,20) and loaded['eligible'][-1] and np.isnan(loaded['returns'][-1])


def test_fit_and_checkpoint_with_22rows_ignore_outer_labels(tmp_path):
    torch.set_num_threads(1)
    rng=np.random.default_rng(17)
    images=rng.uniform(-1,1,(24,1,22,20)).astype(np.float32)
    returns=rng.normal(0,.01,24).astype(np.float32)
    signal=np.repeat(np.arange(6),4);keys=np.tile(np.arange(4),6)
    config=dict(seed=17,width=2,dropout=.1,lr=.01,weight_decay=.001,epochs=2,
        top_k=2,temperature=.2,objective='listnet',max_date_group=8,feature_rows=22)
    model,receipt=fit(images,returns,signal,keys,signal<3,signal==3,config)
    ids=np.flatnonzero(signal>=4);first=predict(model,images,ids,device='cpu')
    returns[signal>=4]=100
    second,other=fit(images,returns,signal,keys,signal<3,signal==3,config)
    np.testing.assert_array_equal(first,predict(second,images,ids,device='cpu'));assert receipt==other
    path=tmp_path/'model.pt';torch.save(dict(config=config,state_dict=model.state_dict()),path)
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    restored=staged_core.RankCNN(width=2,dropout=.1);restored.load_state_dict(checkpoint['state_dict']);restored.eval()
    assert checkpoint['config']['feature_rows']==22
    np.testing.assert_array_equal(first,predict(restored,images,ids,device='cpu'))

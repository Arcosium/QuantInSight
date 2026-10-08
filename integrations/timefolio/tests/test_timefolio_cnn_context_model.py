import json

import numpy as np
import pytest
import torch

from quant.timefolio_cnn_core import RankCNN
from quant.timefolio_cnn_train import load_dataset, sha


def test_context_rows_participate_in_cnn_gradient():
    torch.set_num_threads(1);torch.manual_seed(17)
    model=RankCNN(width=2,dropout=0)
    image=torch.rand(3,1,16,20,requires_grad=True)
    score=model(image)
    assert score.shape==(3,) and torch.isfinite(score).all()
    score.sum().backward()
    assert image.grad[:,:,8:].abs().sum()>0


def test_expanded_image_rows_require_a_matching_manifest_declaration(tmp_path):
    arrays=dict(images=np.zeros((2,1,16,20),np.float32),returns=np.zeros(2),
                signal_index=np.zeros(2,np.int32),label_end_index=np.ones(2,np.int32),
                security_key=np.arange(2,dtype=np.int32),eligible=np.ones(2,bool),
                dates=np.array(['20240102','20240103']))
    specs={}
    for name,array in arrays.items():
        file=tmp_path/(name+'.npy');np.save(file,array)
        specs[name]=dict(path=file.name,sha256=sha(file))
    manifest=dict(exploratory_ready=True,contest_certified=False,limitations=['Current-sector proxy'],
                  readiness=dict(feature_availability=True,label_construction=True),arrays=specs)
    file=tmp_path/'manifest.json';file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='geometry'):load_dataset(file,purpose='exploratory')
    manifest['feature_rows']=16;file.write_text(json.dumps(manifest))
    _,loaded=load_dataset(file,purpose='exploratory')
    assert loaded['images'].shape==(2,1,16,20)

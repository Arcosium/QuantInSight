from copy import deepcopy
import numpy as np
import pytest
import torch
from test_timefolio_heatmap import synthetic_panel
from quant.timefolio_heatmap_daily_history import DAILY_ROWS,daily_values
from quant.timefolio_heatmap_feature_ablation import GROUPS,VARIANTS,availability,transform
from quant.timefolio_heatmap_feature_models import cases,LabNet
from quant.timefolio_heatmap_lab_models import LabNet as BaseNet


def test_groups_partition_every_row_and_ablation_does_not_mutate_source():
    assert sorted(sum(GROUPS.values(),[]))==list(range(32))
    raw=np.random.default_rng(57).integers(0,256,(3,32,65),dtype=np.uint8);before=raw.copy()
    np.testing.assert_array_equal(transform(raw,'full')[:,0],raw)
    for group in ['price','flow','momentum','context','constraints']:
        out=transform(raw,'no_'+group)[:,0];rows=GROUPS[group]
        assert np.all(out[:,rows]==128)
        others=[i for i in range(32) if i not in rows];np.testing.assert_array_equal(out[:,others],raw[:,others])
    np.testing.assert_array_equal(raw,before)


def test_restricted_inputs_and_latest_day_have_no_hidden_other_values():
    raw=np.full((2,32,65),255,np.uint8)
    for name,rows in [('price_only',GROUPS['price']),('flow_only',GROUPS['flow']),('daily_only',list(DAILY_ROWS))]:
        out=transform(raw,name)[:,0];keep=set(rows+[30]);assert np.all(out[:,list(keep)]==255)
        assert np.all(out[:,[i for i in range(32) if i not in keep]]==128)
    out=transform(raw,'last_day')[:,0];assert np.all(out[:,:,:52]==128);np.testing.assert_array_equal(out[:,:,52:],raw[:,:,52:])


def test_missingness_is_exact_prequantization_and_future_invariant():
    p,_=synthetic_panel(days=35,names=2);p['rank_r5'][0,22]=np.nan;p['v20'][1,23]=np.nan
    ci=np.array([0,1]);di=np.array([25,25]);mask=availability(p,ci,di)
    expected=np.repeat(np.isfinite(daily_values(p,ci,di[:,None]+np.arange(-4,1))).astype(np.uint8)*255,13,axis=2)
    np.testing.assert_array_equal(mask[:,DAILY_ROWS],expected)
    others=[i for i in range(32) if i not in DAILY_ROWS];assert np.all(mask[:,others]==255)
    future=deepcopy(p)
    for value in future.values():
        if value.ndim==2 and value.dtype!=bool:value[:,26:]=np.nan
    np.testing.assert_array_equal(availability(future,ci,di),mask)
    assert (mask==0).any()


def test_availability_only_changes_extra_channel_and_observed_control_matches():
    raw=np.random.default_rng(57).integers(0,256,(2,32,65),dtype=np.uint8);mask=np.full_like(raw,255)
    constant=transform(raw,'mask_constant');np.testing.assert_array_equal(transform(raw,'mask_available',mask),constant)
    mask[0,20,:13]=0;out=transform(raw,'mask_available',mask)
    np.testing.assert_array_equal(out[:,0],raw);np.testing.assert_array_equal(out[:,1],mask)
    assert np.count_nonzero(out!=constant)==13


def test_row_center_and_unknown_inputs():
    raw=np.broadcast_to(np.arange(32,dtype=np.uint8)[None,:,None],(1,32,65)).copy()
    assert np.all(transform(raw,'row_center')==128)
    values=np.random.default_rng(19).integers(20,200,(2,32,65),dtype=np.uint8)
    np.testing.assert_array_equal(transform(values,'row_center'),transform(values+10,'row_center'))
    for i,row in enumerate(values[0]):
        expected=np.array([round(min(255,max(0,int(v)-sum(map(int,row))/65+127.5))) for v in row],np.uint8)
        np.testing.assert_array_equal(transform(values,'row_center')[0,0,i],expected)
    with pytest.raises(ValueError):transform(raw,'unknown')
    with pytest.raises(ValueError):transform(raw,'mask_available')


def test_one_channel_model_exact_and_two_channel_controls_match_initialization():
    torch.set_num_threads(1);configs=cases();assert len(configs)==28
    for kind in ['cnn','mlp']:
        cfg=next(c for c in configs if c['id']==kind+'_full')
        torch.manual_seed(17);a=BaseNet(cfg);torch.manual_seed(17);b=LabNet(cfg)
        for k,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[k])
        cfgs=[next(c for c in configs if c['id']==kind+'_'+v) for v in ['mask_constant','mask_available']]
        torch.manual_seed(17);a=LabNet(cfgs[0]);torch.manual_seed(17);b=LabNet(cfgs[1])
        for k,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[k])
        x=torch.randn(4,2,32,65);y=a(x);assert y.shape==(4,);y.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in a.parameters())

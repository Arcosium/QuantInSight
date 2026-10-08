import torch
from quant.timefolio_heatmap_capacity_models import LabNet, cases
from quant.timefolio_heatmap_lab_models import LabNet as Original


def test_existing_controls_retain_weights_outputs_and_gradients():
    x=torch.linspace(-1,1,2*32*65).reshape(2,1,32,65)
    for cfg in [c for c in cases() if c['id'] in ['cnn_width8','cnn_width32','summary_mlp']]:
        torch.manual_seed(17);before=Original(cfg)
        torch.manual_seed(17);after=LabNet(cfg)
        assert all(torch.equal(v,after.base.state_dict()[k]) for k,v in before.state_dict().items())
        before.eval();after.eval()
        a=before(x);b=after(x);torch.testing.assert_close(a,b,rtol=0,atol=0)
        a.sum().backward();b.sum().backward()
        for p,q in zip(before.parameters(),after.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)


def test_full_input_linear_can_distinguish_order_preserving_the_summary():
    cfg=next(c for c in cases() if c['kind']=='full_linear');model=LabNet(cfg)
    with torch.no_grad():
        model.full[1].weight.zero_();model.full[1].bias.zero_();model.full[1].weight[0,0]=1
    a=torch.zeros(1,1,32,65);a[0,0,0,0]=1
    b=torch.zeros_like(a);b[0,0,0,1]=1
    assert torch.equal(a.mean(-1),b.mean(-1)) and torch.equal(a.std(-1),b.std(-1))
    assert torch.equal(a[:,:,:,-1],b[:,:,:,-1])
    assert model(a).item()==1 and model(b).item()==0


def test_all_registered_models_have_finite_gradients_and_fixed_capacity():
    x=torch.linspace(-1,1,2*32*65).reshape(2,1,32,65);counts={}
    for cfg in cases():
        torch.manual_seed(17);model=LabNet(cfg);out=model(x)
        assert out.shape==(2,) and torch.isfinite(out).all()
        out.square().mean().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        counts[cfg['id']]=sum(p.numel() for p in model.parameters())
    assert counts['full_linear']==2081
    assert counts['cnn_width8']<counts['cnn_width32']<counts['cnn_width64']<counts['cnn_width96']
    assert 700000<counts['cnn_width96']<900000
    assert counts['summary_mlp']<counts['full_mlp128']<counts['full_mlp256']

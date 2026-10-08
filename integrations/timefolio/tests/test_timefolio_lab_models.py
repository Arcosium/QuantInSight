import unittest
import torch
from torch import nn
from quant.timefolio_heatmap_lab_models import LabNet,TemporalPool,cases
from quant.timefolio_heatmap_walkforward import AlternativeNet

class LabModelTests(unittest.TestCase):
    def test_all_registered_shapes_and_gradients(self):
        torch.set_num_threads(1);self.assertEqual(len(cases()),20)
        for cfg in cases():
            model=LabNet(cfg);x=torch.randn(3,1,32,65,requires_grad=True);y=model(x)
            self.assertEqual(tuple(y.shape),(3,));y.sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all(),cfg['id'])

    def test_control_weights_and_predictions_match_original(self):
        for kind in ['cnn','mlp','split','tcn']:
            cfg=next(c for c in cases() if c['kind']==kind)
            torch.manual_seed(17);actual=LabNet(cfg).eval()
            torch.manual_seed(17);expected=AlternativeNet(kind).eval()
            self.assertEqual(set(actual.state_dict()),set(expected.state_dict()))
            for key,value in actual.state_dict().items():self.assertTrue(torch.equal(value,expected.state_dict()[key]))
            x=torch.randn(8,1,32,65);torch.testing.assert_close(actual(x),expected(x),atol=1e-6,rtol=1e-5)

    def test_temporal_pool_matches_forward_and_gradient(self):
        a=torch.randn(2,16,65,dtype=torch.float64,requires_grad=True);b=a.detach().clone().requires_grad_()
        first=TemporalPool()(a);second=nn.AdaptiveAvgPool1d(4)(b)
        torch.testing.assert_close(first,second,atol=1e-14,rtol=1e-14)
        g=torch.randn_like(first);first.backward(g);second.backward(g)
        torch.testing.assert_close(a.grad,b.grad,atol=1e-14,rtol=1e-14)

if __name__=='__main__':unittest.main()

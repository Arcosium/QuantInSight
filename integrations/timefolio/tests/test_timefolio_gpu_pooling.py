import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
from quant.timefolio_heatmap_gpu_worker_v2 import FixedHeatmapPool, reference
from quant.timefolio_heatmap_gpu_package import reference_source
from quant.timefolio_heatmap_walkforward import AlternativeNet

class PoolingTests(unittest.TestCase):
    def test_fixed_pool_matches_adaptive_forward_and_backward(self):
        torch.manual_seed(47)
        for dtype in [torch.float32,torch.float64]:
            x=torch.randn(4,32,8,16,dtype=dtype,requires_grad=True)
            z=x.detach().clone().requires_grad_()
            a=FixedHeatmapPool()(x);b=nn.AdaptiveAvgPool2d((2,4))(z)
            torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
            upstream=torch.randn_like(a);a.backward(upstream);b.backward(upstream)
            torch.testing.assert_close(x.grad,z.grad,rtol=0,atol=0)

    def test_model_state_and_predictions_match_original(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            Path(temp,'reference.py').write_text(reference_source());ref=reference(temp)
            torch.manual_seed(17);new=ref.AlternativeNet('cnn').eval()
            torch.manual_seed(17);old=AlternativeNet('cnn').eval()
            self.assertEqual(set(new.state_dict()),set(old.state_dict()))
            for k,v in new.state_dict().items():self.assertTrue(torch.equal(v,old.state_dict()[k]))
            x=torch.randn(12,1,32,65)
            torch.testing.assert_close(new(x),old(x),rtol=1e-6,atol=1e-7)

    def test_wrong_geometry_is_rejected(self):
        with self.assertRaises(ValueError):FixedHeatmapPool()(torch.zeros(2,32,8,17))

if __name__=='__main__':unittest.main()

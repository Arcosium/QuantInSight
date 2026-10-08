"""CUDA training checks. Run only on a rented research Pod, never locally."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from quant.timefolio_chart_models import ChartNet
from quant.timefolio_chart_worker import train, SearchStopped
from quant.timefolio_chart_lab import cases
from quant.timefolio_heatmap_gpu_worker import configure, predict


@unittest.skipUnless(torch.cuda.is_available(), 'Remote CUDA test requires a research GPU')
class RemoteChartTests(unittest.TestCase):
    def test_objectives_checkpoint_and_immediate_stop(self):
        configure('cuda')
        x = torch.randint(0, 256, (8, 1, 64, 192), dtype=torch.uint8, device='cuda')
        y = np.linspace(-1, 1, 8, dtype=np.float32); di = np.repeat([0, 1], 4)
        ref = SimpleNamespace(date_groups=lambda d,m:[np.flatnonzero(m & (d==v)) for v in np.unique(d[m])],
            pairwise_loss=lambda o,t:torch.nn.functional.softplus(-(o[:,None]-o[None,:])*torch.sign(t[:,None]-t[None,:])).mean())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for objective in ['mse','pairwise','bce']:
                cfg = dict(cases()[0], width=8, objective=objective, seed=17, epochs=1)
                model, report = train(ref,x,y,np.ones(8,bool),di,cfg,root)
                self.assertEqual(report['training_rows'],8); self.assertEqual(report['validation_rows'],0)
                before = predict(model,x,np.arange(8)); self.assertTrue(np.isfinite(before).all())
                saved = root / 'weights.pt'; torch.save(model.state_dict(),saved)
                other = ChartNet(cfg).cuda(); other.load_state_dict(torch.load(saved,weights_only=True))
                np.testing.assert_array_equal(before,predict(other,x,np.arange(8)))
            (root/'STOP.json').write_text('{}')
            with self.assertRaises(SearchStopped): train(ref,x,y,np.ones(8,bool),di,cfg,root)


if __name__ == '__main__': unittest.main()

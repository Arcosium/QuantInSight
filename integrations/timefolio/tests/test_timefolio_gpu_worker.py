import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from quant.timefolio_heatmap_gpu_package import reference_source
from quant.timefolio_heatmap_gpu_worker import reference, train, predict, verify, digest, write
from quant.timefolio_heatmap_walkforward import AlternativeNet, predict as old_predict
from quant.timefolio_heatmap_rank_training import train_daywise


class WorkerTests(unittest.TestCase):
    def test_portable_reference_and_cpu_optimizer_match_original(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(77)
        x = torch.from_numpy(rng.integers(0, 256, (48,1,32,65), dtype=np.uint8))
        y = rng.normal(size=48).astype(np.float32); di = np.repeat(np.arange(4),12)
        tr, va = di < 3, di == 3
        with tempfile.TemporaryDirectory() as temp:
            Path(temp,'reference.py').write_text(reference_source()); ref = reference(temp)
            for architecture in ['cnn','mlp']:
                torch.manual_seed(17); a = ref.AlternativeNet(architecture)
                torch.manual_seed(17); b = AlternativeNet(architecture)
                self.assertEqual(set(a.state_dict()), set(b.state_dict()))
                for k,v in a.state_dict().items(): self.assertTrue(torch.equal(v,b.state_dict()[k]))
                cfg = dict(architecture=architecture, objective='pairwise', seed=17, lr=.0007, epochs=2)
                actual, meta = train(ref,x,y,tr,va,cfg,di)
                expected, old = train_daywise(x,y,tr,va,cfg,di)
                self.assertEqual(meta,old)
                for k,v in actual.state_dict().items(): self.assertTrue(torch.equal(v,expected.state_dict()[k]), k)
                np.testing.assert_array_equal(predict(actual,x,np.arange(48)),old_predict(expected,x,np.arange(48)))

    def test_manifest_detects_tampering_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); data=root/'data';data.write_text('a')
            write(root/'manifest.json',{'data':digest(data)});verify(root)
            data.write_text('b')
            with self.assertRaises(AssertionError):verify(root)
            write(root/'manifest.json',{'../escape':'unused'})
            with self.assertRaises(AssertionError):verify(root)

    def test_nonfinite_training_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            Path(temp,'reference.py').write_text(reference_source());ref=reference(temp)
            x=torch.zeros((12,1,32,65),dtype=torch.uint8);y=np.zeros(12,dtype=np.float32);y[0]=np.nan
            cfg=dict(architecture='mlp',seed=17,lr=.0007,objective='pairwise',epochs=1)
            with self.assertRaises(ValueError): train(ref,x,y,np.ones(12,bool),np.zeros(12,bool),cfg,np.zeros(12,int))

if __name__=='__main__': unittest.main()

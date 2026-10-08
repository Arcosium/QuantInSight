import types
import unittest
import numpy as np
import torch
from quant.timefolio_heatmap_optimizer_lab import cases,train
from quant.timefolio_heatmap_gpu_worker import train as reference_train
from quant.timefolio_heatmap_lab_models import LabNet
from quant.timefolio_heatmap_rank_training import pairwise_loss,date_groups
from quant.timefolio_heatmap_walkforward import daily_ic,fold_masks


class OptimizerLabTests(unittest.TestCase):
    def inputs(self):
        torch.set_num_threads(1);rng=np.random.default_rng(57);n=60
        x=torch.from_numpy(rng.integers(0,256,(n,1,32,65),dtype=np.uint8))
        di=np.repeat(np.arange(5),12);y=rng.normal(0,.2,n).astype(np.float32)
        return x,y,di,di<3,di==4

    def test_baseline_matches_original_optimizer_exactly(self):
        x,y,di,tr,va=self.inputs()
        for kind in ['cnn','mlp']:
            cfg=dict(next(c for c in cases() if c['id']==kind+'_h10_rank_base'),seed=17,architecture=kind,epochs=2)
            ref=types.SimpleNamespace(AlternativeNet=lambda _:LabNet(cfg),date_groups=date_groups,daily_ic=daily_ic,pairwise_loss=pairwise_loss)
            a,ha=reference_train(ref,x,y,tr,va,cfg,di);b,hb=train(ref,x,y,tr,va,cfg,di)
            self.assertEqual(ha,hb)
            for name,value in a.state_dict().items():self.assertTrue(torch.equal(value,b.state_dict()[name]))

    def test_new_losses_and_regularizers_are_finite(self):
        x,y,di,tr,va=self.inputs()
        for objective in ['huber','mse']:
            cfg=dict(next(c for c in cases() if c['id']=='cnn_h10_'+objective),seed=29,architecture='cnn',epochs=2)
            ref=types.SimpleNamespace(AlternativeNet=lambda _:LabNet(cfg),date_groups=date_groups,daily_ic=daily_ic,pairwise_loss=pairwise_loss)
            model,history=train(ref,x,y,tr,va,cfg,di)
            self.assertTrue(all(torch.isfinite(v).all() for v in model.state_dict().values()))
            self.assertTrue(all(np.isfinite(row['loss']) for row in history['history']))

    def test_all_registered_masks_are_purged_and_forecasts_match(self):
        configs=cases();self.assertEqual(len(configs),28);self.assertEqual(len({c['id'] for c in configs}),28)
        di=np.repeat(np.arange(20,255),20);allowed=np.ones(len(di),bool)
        for c in configs:
            y=np.ones(len(di));y[di+c['horizon']>=255]=np.nan
            for start,end in [(76,97),(97,114),(238,255)]:
                tr,va,rf,pr,_=fold_masks(di,y,allowed,c['horizon'],start,end,c['train_window'],c['validation_sessions'])
                self.assertTrue(tr.any() and va.any() and rf.any())
                self.assertLess(max(di[tr]+c['horizon']),min(di[va]))
                self.assertLess(max(di[rf]+c['horizon']),start)
                self.assertFalse(np.any(tr&va));self.assertTrue(np.all(~tr|rf))
                np.testing.assert_array_equal(pr,(di>=start-1)&(di<end-1))
                if c['train_window']:self.assertGreaterEqual(min(di[tr]),start-c['train_window'])


if __name__=='__main__':unittest.main()

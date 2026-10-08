import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from quant.timefolio_heatmap_lab_pod import stop,validate_job
from quant.timefolio_heatmap_gpu_worker import digest,write
from quant.timefolio_heatmap_lab_models import cases


class LabPodTests(unittest.TestCase):
    def test_thirtieth_pod_allowed_and_next_allocation_blocked(self):
        from quant import timefolio_heatmap_lab_pod as pods
        self.assertEqual(pods.MAX_PODS,30)
        self.assertEqual(pods.MAX_HOURLY,.50)
        for count in [29,30,31]:
            with self.subTest(existing=count),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/'fresh'
                calls=[]
                def request(method,path,body=None):
                    calls.append((method,path))
                    if method=='GET':return [dict(id=str(i),name='existing'+str(i)) for i in range(count)]
                    raise RuntimeError('fixture allocation failure')
                def keygen(args,**kwargs):
                    self.assertEqual(args[0],'ssh-keygen')
                    Path(args[-1]).with_suffix('.pub').write_text('ssh-ed25519 fixture')
                with patch.object(pods,'api',return_value=request),patch.object(pods,'CANDIDATES',[('fixture','SECURE')]),patch.object(pods.subprocess,'run',side_effect=keygen):
                    expected='No eligible' if count==29 else 'limit reached'
                    with self.assertRaisesRegex(RuntimeError,expected):pods.rent(root)
                self.assertEqual(sum(method=='POST' for method,_ in calls),int(count==29))

    def test_cleanup_cannot_delete_other_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write(root/'pod.json',dict(id='owned',name='research',started_at=0,estimated_hourly=.3))
            with patch('quant.timefolio_heatmap_lab_pod.api') as factory:
                factory.return_value.return_value=[dict(id='owned',name='someone_else')]
                with self.assertRaisesRegex(RuntimeError,'ownership'):stop(root,'test')
                self.assertEqual(factory.return_value.call_count,1)
                self.assertFalse((root/'deletion.json').exists())

    def test_completion_needs_all_registered_artifacts_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=cases()[0];name=cfg['id']+'_seed17';sha='manifest'
            folds=[dict(month=f'2026{i:02}') for i in range(1,10)]
            plan=dict(cases=cases(),jobs=[dict(id=name,case=cfg['id'],seed=17)],folds=folds)
            hashes={}
            for fold in folds:
                for suffix in ['.pt','.json','.pred.npy']:
                    key=f'models/{name}/{fold["month"]}{suffix}';p=root/key;p.parent.mkdir(parents=True,exist_ok=True)
                    p.write_bytes(b'fixture');hashes[key]=digest(p)
            for key in ['untrained.pt','untrained.pred.npy']:
                (root/key).write_bytes(b'fixture');hashes[key]=digest(root/key)
            write(root/'artifact_hashes.json',hashes)
            write(root/'complete.json',dict(folds=9,model=name,package_sha256=sha))
            runtime=dict(device='cuda',gpu='fixture',config=dict(cfg,seed=17,id=name),package_sha256=sha)
            write(root/'runtime.json',runtime)
            self.assertEqual(validate_job(root,plan,sha,name)['artifacts'],29)
            runtime['config']['encoding']='reverse';write(root/'runtime.json',runtime)
            with self.assertRaises(AssertionError):validate_job(root,plan,sha,name)
            runtime['config']['encoding']=cfg['encoding'];write(root/'runtime.json',runtime)
            (root/'untrained.pt').write_bytes(b'changed')
            with self.assertRaises(AssertionError):validate_job(root,plan,sha,name)


if __name__=='__main__':unittest.main()

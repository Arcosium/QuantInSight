import io
import json
from pathlib import Path
import tarfile
import tempfile
import time
import unittest
from quant.timefolio_heatmap_gpu_pool import unpack_results, terminate_owned, verify_results


class PoolTests(unittest.TestCase):
    def test_termination_checks_exact_ownership_and_deletion(self):
        with tempfile.TemporaryDirectory() as temp:
            rows=[{'id':'owned','name':'study'}, {'id':'other','name':'other-service'}];calls=[]
            def rest(method,path):
                calls.append((method,path))
                if method=='GET':return list(rows)
                self.assertEqual(path,'/pods/owned');rows.pop(0);return {}
            record=dict(id='owned',name='study',started_at=time.time(),estimated_hourly=.35)
            terminate_owned(rest,record,Path(temp)/'receipt.json')
            self.assertTrue(record['deletion_verified']);self.assertEqual(rows,[{'id':'other','name':'other-service'}])
            self.assertEqual(sum(method=='DELETE' for method,path in calls),1)

    def test_ownership_mismatch_never_deletes(self):
        calls=[]
        def rest(method,path):calls.append(method);return [dict(id='same-id',name='unrelated')]
        with self.assertRaises(RuntimeError):terminate_owned(rest,dict(id='same-id',name='study'),Path('/unused'))
        self.assertEqual(calls,['GET'])

    def test_result_tar_rejects_links_and_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for i,name in enumerate(['../escape','out/../../escape','out/link']):
                archive=root/(str(i)+'.tar.gz')
                with tarfile.open(archive,'w:gz') as stream:
                    item=tarfile.TarInfo(name)
                    if name.endswith('link'):item.type=tarfile.SYMTYPE;item.linkname='/etc/passwd';stream.addfile(item)
                    else:item.size=1;stream.addfile(item,io.BytesIO(b'x'))
                with self.assertRaises(RuntimeError):unpack_results(archive,root/('dest'+str(i)))

    def test_result_verification_rejects_cpu_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'complete.json').write_text(json.dumps(dict(seed=17,folds=54,package_sha256='test')))
            (root/'runtime.json').write_text(json.dumps(dict(device='cpu',gpu=None,package_sha256='test')))
            with self.assertRaises(AssertionError):verify_results(root,'test',17)

if __name__=='__main__':unittest.main()

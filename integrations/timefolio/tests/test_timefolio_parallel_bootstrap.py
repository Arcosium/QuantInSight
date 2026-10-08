import hashlib
from pathlib import Path
import unittest
import numpy as np
from quant.timefolio_heatmap_parallel_bootstrap import family_bootstrap, PARENT_SHA256
from quant.timefolio_heatmap_streamed_bootstrap import family_bootstrap as reference


class ParallelBootstrapTests(unittest.TestCase):
    def test_registered_parent_and_all_columns_match_exactly(self):
        parent=Path(__file__).parents[1]/'quant/timefolio_heatmap_streamed_bootstrap.py'
        self.assertEqual(hashlib.sha256(parent.read_bytes()).hexdigest(), PARENT_SHA256)
        data=np.random.default_rng(710).normal(0,.01,(43,257))
        data[:,0]=0;data[:,1]=.001;data[:,2]=data[:,3]
        for block in [5,10]:
            kw=dict(block=block,draws=213,seed=57,column_batch=64,scratch_bytes=128*1024)
            old=reference(data,**kw)
            for workers in [1,4,8]:
                new=family_bootstrap(data,workers=workers,**kw)
                self.assertEqual(old.keys(),new.keys())
                for key in old:np.testing.assert_array_equal(new[key],old[key],err_msg=key)

    def test_one_column_and_invalid_worker_count(self):
        data=np.random.default_rng(711).normal(0,.01,(31,1))
        old=reference(data,draws=107)
        new=family_bootstrap(data,draws=107,workers=2)
        for key in old:np.testing.assert_array_equal(new[key],old[key])
        for workers in [0,9]:
            with self.assertRaises(ValueError):family_bootstrap(data,workers=workers)


if __name__=='__main__':unittest.main()

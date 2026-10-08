import unittest
import numpy as np
from quant.timefolio_heatmap_bounded_bootstrap import family_bootstrap
from quant.timefolio_heatmap_walkforward_eval import family_bootstrap as original


class BoundedBootstrapTests(unittest.TestCase):
    def test_draw_chunking_preserves_all_statistics_bitwise(self):
        rng=np.random.default_rng(817);a=rng.normal(0,.01,(179,19));a[:,0]=0;a[:,1]=.001;a[:,2]=a[:,3]
        for block in [5,10]:
            expected=original(a,block=block,draws=237,seed=57)
            for batch in [1,3,17,100]:
                actual=family_bootstrap(a,block=block,draws=237,seed=57,scratch_bytes=a.nbytes*batch)
                self.assertEqual(set(actual),set(expected))
                for key in actual:np.testing.assert_array_equal(actual[key],expected[key])

    def test_invalid_shape_and_insufficient_scratch_fail_before_allocation(self):
        with self.assertRaises(ValueError):family_bootstrap(np.empty((179,0)))
        with self.assertRaises(ValueError):family_bootstrap(np.ones((179,5)),scratch_bytes=1)


if __name__=='__main__':unittest.main()

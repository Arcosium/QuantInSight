"""Statistical arithmetic only; no model fitting or account replay."""
import unittest
import numpy as np
import pandas as pd
from quant.timefolio_chart_inference import count_bounds
from quant.timefolio_heatmap_parallel_bootstrap import family_bootstrap
from quant.timefolio_heatmap_fleet_audit_helpers import count_bootstrap


class CountBoundTests(unittest.TestCase):
    def test_bounded_count_matches_full_reference_and_retains_ties(self):
        x=np.random.default_rng(19).normal(0,.01,(31,9));x[:,0]=0.;x[:,1]=.001
        x[:,2]=x[:,3];family=[(str(i),'cash','synthetic') for i in range(9)];rows=[];actual={}
        for block in [5,10]:
            report=family_bootstrap(x,block=block,draws=137,seed=57,column_batch=4,workers=2)
            actual[block]=count_bounds(x,report,block=block,draws=137,columns=4)
            for i,(key,comp,origin) in enumerate(family):
                rows.append(dict(id=key,comparator=comp,origin=origin,block=block,
                    **{k:float(report[k][i]) for k in ['mean','standard_error','adjusted_p','marginal_p','simultaneous_lower95']}))
        _,reference=count_bootstrap(family,x.T,pd.DataFrame(rows),draws=137)
        for block in [5,10]:
            expected=reference[reference.block==block]
            for name in ['marginal_low','marginal_high','adjusted_low','adjusted_high','simultaneous_lower95']:
                np.testing.assert_allclose(actual[block][name],expected[name],rtol=0,atol=1e-13)
            self.assertLessEqual(actual[block]['adjusted_low'][0],actual[block]['adjusted_high'][0])


if __name__=='__main__':unittest.main()

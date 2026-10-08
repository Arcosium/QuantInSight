import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from quant.timefolio_heatmap_gpu_merge import merge_months

class MergeTests(unittest.TestCase):
    def test_portable_boundaries_and_predictions(self):
        dates=['20251230','20260102','20260105','20260202','20260203'];di=np.arange(5)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for month,start,end in [('202601',1,3),('202602',3,5)]:
                row=dict(fold=dict(month=month,start=start,end=end),boundaries=dict(first_execution=dates[start],last_execution=dates[end-1],
                    last_refit_label='20251229',last_inner_train_label='20251128',first_inner_validation_signal='20251201'))
                (root/(month+'.json')).write_text(json.dumps(row))
                values=np.full(5,np.nan,np.float32);values[start-1:end-1]=np.arange(start-1,end-1)
                np.save(root/(month+'.pred.npy'),values)
            np.testing.assert_array_equal(merge_months(root,di,dates),np.array([0,1,2,3,np.nan],np.float32))
            path=root/'202601.json';row=json.loads(path.read_text());row['boundaries']['last_refit_label']='20260102';path.write_text(json.dumps(row))
            with self.assertRaises(AssertionError):merge_months(root,di,dates)

    def test_missing_month_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(AssertionError):merge_months(temp,np.arange(2),['20251230','20260102'])

if __name__=='__main__':unittest.main()

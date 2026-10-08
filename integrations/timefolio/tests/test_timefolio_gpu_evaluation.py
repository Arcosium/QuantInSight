import unittest
import numpy as np
import pandas as pd
from quant.timefolio_heatmap_gpu_evaluation import (MODELS,TRAINED,UNTRAINED,CASES,BUFFERS,MODES,
    MEMBERS,account_id,comparison_ids,new_family,legacy_path,candidate_ids,model_info)

class EvaluationTests(unittest.TestCase):
    def test_complete_grid_and_matched_controls(self):
        self.assertEqual((len(MODELS),len(TRAINED),len(UNTRAINED)),(37,24,12))
        accounts={account_id(m,c['id'],b,s) for m in MODELS for c in CASES for b in BUFFERS for s in MODES}
        self.assertEqual(len(accounts),592);family=new_family();self.assertEqual(len(family),1248)
        for key,comp,other in family:
            self.assertIn(key,accounts)
            if other:self.assertIn(other,accounts)
            if comp!='same_cnn_research20' and other:
                self.assertEqual(key.split('__',1)[1],other.split('__',1)[1])
            if comp.startswith('other_'):
                self.assertEqual(key.split('__')[0].split('_')[-1],other.split('__')[0].split('_')[-1])
        legacy=[legacy_path(m,c['id'],b,s) for m in MODELS for c in CASES for b in BUFFERS for s in MODES]
        self.assertEqual(sum(p is not None for p in legacy),16)
        self.assertTrue(all(p.exists() for p in legacy if p is not None))

    def test_candidate_needs_all_seeds_and_all_comparators(self):
        case=CASES[0]['id'];buffer=BUFFERS[0];mode='formula';rows=[]
        for member in MEMBERS:
            model='cnn_dailygpu_history_h10_'+member
            rows.append(dict(id=account_id(model,case,buffer,mode),model=model,case=case,buffer=buffer,
                ceiling=mode,**model_info(model),**{'return':.1,'mdd':-.1,'positive_blocks':3,'four_week_turnover_stop':False}))
        frame=pd.DataFrame(rows);ensemble=frame[frame.member=='ensemble'].iloc[0]
        stats=pd.DataFrame([dict(id=ensemble.id,origin='daily_history',comparator=comp,block=block,
            adjusted_p=.01,simultaneous_lower95=.001) for comp,_ in comparison_ids(ensemble.model,case,buffer,mode) for block in [5,10]])
        self.assertEqual(candidate_ids(frame,stats),[ensemble.id])
        broken=frame.copy();broken.loc[0,'return']=-.01
        self.assertEqual(candidate_ids(broken,stats),[])
        failed=stats.copy();failed.loc[0,'adjusted_p']=.5
        self.assertEqual(candidate_ids(frame,failed),[])
        with self.assertRaises(AssertionError):candidate_ids(frame,stats.iloc[1:])

if __name__=='__main__':unittest.main()

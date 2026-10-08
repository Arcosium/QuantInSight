import unittest
import pandas as pd
from quant.timefolio_heatmap_lab_evaluation import (MODELS,TRAINED,UNTRAINED,CASES,BUFFERS,MODES,MEMBERS,
    account_id,comparison_ids,new_family,model_info,candidate_ids,legacy_path)


class LabEvaluationTests(unittest.TestCase):
    def test_all_comparisons_and_controls_registered(self):
        self.assertEqual((len(MODELS),len(TRAINED),len(UNTRAINED)),(161,80,80))
        accounts={account_id(m,c['id'],b,s) for m in MODELS for c in CASES for b in BUFFERS for s in MODES}
        self.assertEqual(len(accounts),2576);family=new_family();self.assertEqual(len(family),10560)
        for key,comp,reference in family:
            self.assertIn(key,accounts)
            if reference:
                self.assertIn(reference,accounts);self.assertNotEqual(key,reference)
                if comp!='same_model_research20':self.assertEqual(key.split('__',1)[1],reference.split('__',1)[1])
                if comp not in ['online','same_model_research20']:
                    self.assertEqual(key.split('__')[0].rsplit('_',1)[1],reference.split('__')[0].rsplit('_',1)[1])
        self.assertEqual(sum(legacy_path(m,c['id'],b,s) is not None for m in MODELS for c in CASES for b in BUFFERS for s in MODES),16)

    def test_gate_rejects_missing_seed_failed_comparison_and_weak_turnover(self):
        case=CASES[0]['id'];buffer=BUFFERS[0];ceiling='formula';rows=[]
        for member in MEMBERS:
            model='lab_history_width32_trained_'+member
            rows.append(dict(id=account_id(model,case,buffer,ceiling),model=model,case=case,buffer=buffer,ceiling=ceiling,
                **model_info(model),**{'return':.1,'mdd':-.1,'positive_blocks':3,'four_week_turnover_stop':False}))
        frame=pd.DataFrame(rows);row=rows[-1]
        stats=pd.DataFrame([dict(id=row['id'],origin='geometry_lab',comparator=c,block=b,adjusted_p=.01,simultaneous_lower95=.001)
            for c,_ in comparison_ids(row['model'],case,buffer,ceiling) for b in [5,10]])
        self.assertEqual(candidate_ids(frame,stats),[row['id']])
        frame.loc[0,'four_week_turnover_stop']=True;self.assertEqual(candidate_ids(frame,stats),[])
        frame.loc[0,'four_week_turnover_stop']=False;stats.loc[0,'adjusted_p']=.1
        self.assertEqual(candidate_ids(frame,stats),[])
        with self.assertRaises(AssertionError):candidate_ids(frame.iloc[1:],stats)
        with self.assertRaises(AssertionError):candidate_ids(frame,stats.iloc[1:])


if __name__=='__main__':unittest.main()

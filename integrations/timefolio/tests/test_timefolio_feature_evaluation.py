import unittest
import pandas as pd
from quant.timefolio_heatmap_feature_evaluation import (MODELS,TRAINED,UNTRAINED,CASES,BUFFERS,MODES,MEMBERS,
    account_id,comparison_ids,new_family,model_info,candidate_ids)


class FeatureEvaluationTests(unittest.TestCase):
    def test_registered_grid_and_matched_mask_control(self):
        self.assertEqual((len(MODELS),len(TRAINED),len(UNTRAINED)),(225,112,112))
        accounts={account_id(m,c['id'],b,s) for m in MODELS for c in CASES for b in BUFFERS for s in MODES}
        self.assertEqual(len(accounts),3600);family=new_family();self.assertEqual(len(family),9856)
        mask_comparisons=0
        for key,label,other in family:
            self.assertIn(key,accounts)
            if other:self.assertIn(other,accounts);self.assertNotEqual(key,other)
            if other and label!='same_model_research20':self.assertEqual(key.split('__',1)[1],other.split('__',1)[1])
            if label=='same_architecture_constant_mask':
                mask_comparisons+=1;a,b=[model_info(v.split('__')[0]) for v in [key,other]]
                self.assertEqual((a['architecture'],a['member']),(b['architecture'],b['member']))
                self.assertEqual((a['encoding'],b['encoding']),('mask_available','mask_constant'))
        self.assertEqual(mask_comparisons,128)

    def test_availability_needs_mask_control_and_all_seeds(self):
        case=CASES[0]['id'];buffer=BUFFERS[0];ceiling='formula';rows=[]
        for member in MEMBERS:
            model='fea_cnn_mask_available_trained_'+member
            rows.append(dict(id=account_id(model,case,buffer,ceiling),model=model,case=case,buffer=buffer,ceiling=ceiling,
                **model_info(model),**{'return':.1,'mdd':-.1,'positive_blocks':3,'four_week_turnover_stop':False}))
        frame=pd.DataFrame(rows);row=rows[-1]
        stats=pd.DataFrame([dict(id=row['id'],origin='feature_lab',comparator=c,block=b,adjusted_p=.01,simultaneous_lower95=.001)
            for c,_ in comparison_ids(row['model'],case,buffer,ceiling) for b in [5,10]])
        self.assertEqual(candidate_ids(frame,stats),[row['id']])
        changed=stats.copy();changed.loc[changed.comparator=='same_architecture_constant_mask','adjusted_p']=.1
        self.assertEqual(candidate_ids(frame,changed),[])
        with self.assertRaises(AssertionError):candidate_ids(frame,stats[stats.comparator!='same_architecture_constant_mask'])
        frame.loc[0,'return']=-.1;self.assertEqual(candidate_ids(frame,stats),[])


if __name__=='__main__':unittest.main()

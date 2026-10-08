import unittest
import pandas as pd
from quant.timefolio_heatmap_epoch_evaluation import (MODELS,TRAINED,UNTRAINED,CASES,BUFFERS,MODES,MEMBERS,
    account_id,comparison_ids,new_family,model_info,candidate_ids)


class EpochEvaluationTests(unittest.TestCase):
    def test_registered_grid_preserves_matched_conditions(self):
        self.assertEqual((len(MODELS),len(TRAINED),len(UNTRAINED)),(49,40,8))
        accounts={account_id(m,c['id'],b,s) for m in MODELS for c in CASES for b in BUFFERS for s in MODES}
        self.assertEqual(len(accounts),784);family=new_family();self.assertEqual(len(family),3392)
        for key,label,other in family:
            self.assertIn(key,accounts)
            if other:self.assertIn(other,accounts);self.assertNotEqual(key,other)
            if label=='matched_other_architecture':
                a,b=[model_info(v.split('__')[0]) for v in [key,other]]
                self.assertEqual((a['condition'],a['member']),(b['condition'],b['member']))
                self.assertNotEqual(a['architecture'],b['architecture'])
            if other and label!='same_model_research20':self.assertEqual(key.split('__',1)[1],other.split('__',1)[1])

    def test_all_seed_gate_and_all_comparators_required(self):
        c=CASES[0]['id'];b=BUFFERS[0];s='formula';rows=[]
        for member in MEMBERS:
            m='epo_cnn_fixed12_trained_'+member
            rows.append(dict(id=account_id(m,c,b,s),model=m,case=c,buffer=b,ceiling=s,**model_info(m),
                **{'return':.1,'mdd':-.1,'positive_blocks':3,'four_week_turnover_stop':False}))
        frame=pd.DataFrame(rows);row=rows[-1]
        stats=pd.DataFrame([dict(id=row['id'],origin='epoch_lab',comparator=label,block=block,adjusted_p=.01,simultaneous_lower95=.001)
            for label,_ in comparison_ids(row['model'],c,b,s) for block in [5,10]])
        self.assertEqual(candidate_ids(frame,stats),[row['id']])
        frame.loc[0,'return']=-.1;self.assertEqual(candidate_ids(frame,stats),[])
        with self.assertRaises(AssertionError):candidate_ids(frame,stats.iloc[1:])


if __name__=='__main__':unittest.main()

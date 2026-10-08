import unittest
from unittest.mock import patch
import pandas as pd
from quant import timefolio_heatmap_fleet_inference as study
from quant.timefolio_heatmap_fleet_lab import cases, MEMBERS
from quant.timefolio_heatmap_fleet_accounts import policy_grid, account_id


class FleetInferenceTests(unittest.TestCase):
    def test_full_family_controls_exist_and_no_model_compares_to_itself(self):
        ids = {account_id(f"flt_{c['id']}_{obj}_{m}", p) for c in cases()
               for obj in ['trained', 'untrained'] for m in MEMBERS for p in policy_grid()}
        ids.update(account_id('online_nonimage', p) for p in policy_grid())
        self.assertEqual(len(ids), 19984)
        family = study.new_family()
        self.assertEqual(len(family), 53376)
        self.assertTrue(all(k in ids and (r is None or r in ids and r != k) for k, _, r in family))

    def test_candidate_requires_every_comparator_block_and_all_seeds(self):
        cfg, policy = cases()[0], policy_grid()[0]
        rows = [dict(id=account_id(f"flt_{cfg['id']}_trained_{m}", policy),
                     **{'return': .2, 'mdd': -.1, 'positive_blocks': 3, 'four_week_turnover_stop': False})
                for m in MEMBERS]
        key = rows[-1]['id']
        tests = [dict(id=key, comparator=c, origin=study.ORIGIN, block=b,
                      adjusted_p=.001, simultaneous_lower95=.0001)
                 for c, _ in study.comparison_ids(cfg, 'ensemble', policy) for b in [5, 10]]
        frame, stats = pd.DataFrame(rows), pd.DataFrame(tests)
        with patch.object(study, 'cases', return_value=[cfg]), patch.object(study, 'policy_grid', return_value=[policy]):
            self.assertEqual(study.candidate_ids(frame, stats)[0], [key])
            stats.loc[0, 'adjusted_p'] = .025
            self.assertEqual(study.candidate_ids(frame, stats)[0], [])
            stats.loc[0, 'adjusted_p'] = .001
            frame.loc[0, 'return'] = -.01
            self.assertEqual(study.candidate_ids(frame, stats)[0], [])
            with self.assertRaises(AssertionError): study.candidate_ids(frame, stats.iloc[1:])


if __name__ == '__main__': unittest.main()

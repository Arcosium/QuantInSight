import unittest
import pandas as pd
from quant import timefolio_heatmap_locked_evaluation as study


class LockedEvaluationTests(unittest.TestCase):
    def test_only_new_diagnostic_is_excluded_from_ledger_equality(self):
        old = dict(daily=[dict(nav=100.)], trades=[], metrics={'return': .1})
        new = dict(old, metrics=dict(old['metrics'], locked_target_adjusted_days=0))
        self.assertTrue(study.same_ledger(new, old))
        self.assertIn('locked_target_adjusted_days', new['metrics'])
        self.assertFalse(study.same_ledger(dict(new, daily=[dict(nav=101.)]), old))
        self.assertFalse(study.same_ledger(dict(new, metrics=dict(new['metrics'], other=0)), old))

    def test_entire_cohort_and_same_account_pairing(self):
        accounts = {study.account_id(m, c['id'], b, s) for m, c, b, s in study.account_grid()}
        self.assertEqual(len(accounts), 2576)
        family = study.new_family(); self.assertEqual(len(family), 13136)
        paired = [row for row in family if row[2] == 'locked_rule_effect']
        self.assertEqual({row[0] for row in paired}, accounts)
        for key, comp, origin, root, ref in family:
            self.assertIn(key, accounts)
            if ref: self.assertIn(ref, accounts)
            if origin == 'locked_rule_effect':
                self.assertEqual(key, ref); self.assertEqual(root, study.SOURCE)
            else:
                self.assertEqual(root, study.DEST); self.assertNotEqual(key, ref)

    def test_rule_effect_cannot_qualify_or_disqualify_prediction(self):
        case = study.CASES[0]['id']; buffer = study.BUFFERS[0]; ceiling = 'formula'
        rows = []
        for member in study.MEMBERS:
            model = 'lab_history_width32_trained_' + member
            rows.append(dict(id=study.account_id(model, case, buffer, ceiling), model=model,
                             case=case, buffer=buffer, ceiling=ceiling, **study.model_info(model),
                             **{'return': .1, 'mdd': -.1, 'positive_blocks': 3, 'four_week_turnover_stop': False}))
        frame = pd.DataFrame(rows); row = rows[-1]
        stats = pd.DataFrame([dict(id=row['id'], origin='locked_geometry', comparator=c,
                                  block=b, adjusted_p=.01, simultaneous_lower95=.001)
                              for c, _ in study.comparison_ids(row['model'], case, buffer, ceiling) for b in [5, 10]])
        paired = pd.DataFrame([dict(id=row['id'], origin='locked_rule_effect', comparator='same_account_old_rule',
                                   block=b, adjusted_p=1., simultaneous_lower95=0.) for b in [5, 10]])
        self.assertEqual(study.candidate_ids(frame, pd.concat([stats, paired])), [row['id']])
        stats.loc[0, 'adjusted_p'] = .5; paired['adjusted_p'] = .0001; paired['simultaneous_lower95'] = .01
        self.assertEqual(study.candidate_ids(frame, pd.concat([stats, paired])), [])
        with self.assertRaises(AssertionError): study.candidate_ids(frame, stats.iloc[1:])
        with self.assertRaises(AssertionError): study.candidate_ids(frame.iloc[1:], stats)


if __name__ == '__main__': unittest.main()

import copy
import unittest
from quant.timefolio_heatmap_fleet_thresholds import MONTHS, classify, is_search_process


def metrics(pooled=3.01, minimum=1.51):
    return dict(pooled=dict(sharpe=pooled, sharpe_defined=True),
                folds=[dict(month=m, sharpe=minimum, sharpe_defined=True) for m in MONTHS],
                annual_observations=252, later_folds_use_previous_close=True,
                practical_metrics_passed=True)


class ThresholdTests(unittest.TestCase):
    def test_strict_stop_boundaries_and_record_threshold(self):
        self.assertTrue(classify(metrics())['stop_search_and_validate'])
        for pooled, minimum in [(3., 1.51), (3.01, 1.5), (2.01, 1.01)]:
            row = classify(metrics(pooled, minimum))
            self.assertFalse(row['stop_search_and_validate'])
            self.assertTrue(row['retain_candidate'])
        for pooled, minimum in [(2., 1.01), (2.01, 1.)]:
            self.assertFalse(classify(metrics(pooled, minimum))['retain_candidate'])

    def test_one_bad_missing_duplicate_or_undefined_fold_rejects(self):
        original = metrics()
        variants = []
        for field, value in [('sharpe', -1.), ('sharpe', float('nan')),
                             ('sharpe_defined', False), ('month', '202601')]:
            row = copy.deepcopy(original); row['folds'][-1][field] = value; variants.append(row)
        row = copy.deepcopy(original); row['folds'].pop(); variants.append(row)
        for row in variants:
            result = classify(row)
            self.assertFalse(result['stop_search_and_validate'])
            self.assertFalse(result['retain_candidate'])

    def test_stop_process_scope_excludes_accounts_and_other_projects(self):
        remote = '/workspace/arctrade_fleet10'
        self.assertTrue(is_search_process(['python3', 'worker.py', '--root', remote], remote, remote))
        self.assertTrue(is_search_process(['python3', '/workspace/arctrade_fleet_resume_queue.py'], remote, remote))
        for argv, cwd in [(['python3', 'worker.py', '--root', '/other'], remote),
                          (['python3', '-m', 'quant.timefolio_heatmap_fleet_accounts'], remote),
                          (['python3', '-m', 'quant.timefolio_heatmap_fleet_queue'], '/other')]:
            self.assertFalse(is_search_process(argv, cwd, remote))


if __name__ == '__main__': unittest.main()

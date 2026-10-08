import unittest
import numpy as np
from quant.timefolio_heatmap_fleet_metrics import monthly_target, EXPECTED_MONTHS


def account(patterns):
    nav = 1e9; daily = []
    for month, returns in zip(EXPECTED_MONTHS, patterns):
        for day, ret in enumerate(returns, 1):
            nav *= 1 + ret
            daily.append(dict(date=month + f'{day:02d}', nav=nav))
    return daily


class MonthlyTargetTests(unittest.TestCase):
    def test_positive_pooled_does_not_hide_one_bad_fold(self):
        good = np.array([.001, .003, .002, .004])
        patterns = [good] * 8 + [-good]
        result = monthly_target(account(patterns))
        self.assertGreater(result['pooled']['sharpe'], 3)
        self.assertFalse(result['all_nine_folds_above_one'])
        self.assertFalse(result['practical_metrics_passed'])

    def test_month_boundary_uses_prior_close_and_flat_fold_fails(self):
        returns = np.array([.001, .003, .002, .004])
        result = monthly_target(account([returns] * 9))
        expected = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
        self.assertTrue(result['practical_metrics_passed'])
        for fold in result['folds']:
            self.assertAlmostEqual(fold['sharpe'], expected, places=10)
        flat = monthly_target(account([returns] * 8 + [np.zeros(4)]))
        self.assertEqual(flat['undefined_folds'], ['202609'])
        self.assertFalse(flat['practical_metrics_passed'])

    def test_missing_month_is_rejected(self):
        with self.assertRaises(AssertionError):
            monthly_target(account([[.001, .002]] * 8))


if __name__ == '__main__': unittest.main()

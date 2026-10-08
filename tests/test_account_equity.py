import unittest
from autofolio.account_equity import curve

class EquityTests(unittest.TestCase):
    def test_deposit_is_not_profit(self):
        rows=[dict(ts='2026-10-01 15:00:00',total_eval=100),
              dict(ts='2026-10-02 15:00:00',total_eval=150,external_flow_cum=50)]
        self.assertEqual(curve(rows)[-1]['net_return'],0)

    def test_ledger_correction_is_not_loss(self):
        rows=[dict(ts='2026-10-01 15:00:00',ledger_eval=100),
              dict(ts='2026-10-02 15:00:00',ledger_eval=90,reconcile_adj=-10)]
        self.assertEqual(curve(rows)[-1]['net_return'],0)

    def test_ledger_preferred_and_missing_history_is_empty(self):
        rows=[dict(ts='2026-10-01 15:00:00',total_eval=1000),
              dict(ts='2026-10-02 15:00:00',total_eval=1100,ledger_eval=100),
              dict(ts='2026-10-03 15:00:00',total_eval=900,ledger_eval=110)]
        result=curve(rows)
        self.assertEqual(len(result),2)
        self.assertAlmostEqual(result[-1]['net_return'],.1)
        self.assertEqual(curve([]),[])

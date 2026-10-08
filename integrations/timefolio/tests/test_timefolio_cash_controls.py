import unittest
import numpy as np
from quant.timefolio_heatmap_cash_controls import schedules, price_schedules


def market():
    close = np.tile(100 * 1.001 ** np.arange(100), (16, 1))
    return dict(close=close, eligible=np.ones_like(close, bool), split=np.ones_like(close))


class CashControlTests(unittest.TestCase):
    def test_future_prices_eligibility_and_forecasts_cannot_change_past_exposure(self):
        p = market(); forecasts = np.full_like(p['close'], .1)
        before = schedules(p, forecasts)
        p['close'][:, 75:] *= .1; p['eligible'][:, 75:] = False
        p['split'][:, 75:] = 2; forecasts[:, 75:] = -100
        after = schedules(p, forecasts)
        for name in before:
            np.testing.assert_array_equal(before[name][:75], after[name][:75])
        self.assertEqual(after['forecast0'][75], .2)

    def test_split_does_not_create_a_false_market_crash(self):
        p = market(); baseline = price_schedules(p)
        p['close'][:, 65:] /= 2; p['split'][:, 65] = 2
        adjusted = price_schedules(p)
        for name in baseline: np.testing.assert_allclose(adjusted[name], baseline[name], atol=1e-14)

    def test_missing_forecast_and_negative_forecast_leave_cash(self):
        p = market(); x = np.full_like(p['close'], -.1); x[:, :10] = np.nan
        result = schedules(p, x)
        np.testing.assert_array_equal(result['forecast0'], .2)
        np.testing.assert_array_equal(result['forecast005'], .2)
        for exposure in result.values():
            self.assertTrue(np.isfinite(exposure).all())
            self.assertGreaterEqual(exposure.min(), 0)
            self.assertLessEqual(exposure.max(), .6)


if __name__ == '__main__': unittest.main()

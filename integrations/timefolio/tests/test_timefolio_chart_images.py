import unittest
import numpy as np
from quant.timefolio_chart_images import render, raster, history
from quant.timefolio_chart_lab import cases, assignments, VIEWS


class ChartImagesTests(unittest.TestCase):
    def setUp(self):
        p = np.arange(80, dtype=float) + 100
        self.daily = np.stack([p, p + 3, p - 2, p + 1, p * 10], axis=-1)[None]
        self.intra = np.repeat(self.daily[:, :, None, :], 13, axis=2)
        self.split = np.ones((1, 80))

    def test_future_mutation_does_not_change_any_view(self):
        for view in VIEWS:
            before = render(self.daily, self.intra, self.split, 0, 61, view)
            d = self.daily.copy(); i = self.intra.copy(); s = self.split.copy()
            d[:, 62:] *= 19; i[:, 62:] *= 3; s[:, 62:] = 2
            self.assertTrue(np.array_equal(before, render(d, i, s, 0, 61, view)), view)
            self.assertEqual(before.shape, (64, 192)); self.assertGreater((before < 255).sum(), 100)

    def test_missing_history_and_bars_are_not_filled(self):
        b = history(self.daily, self.intra, self.split, 0, 5, 'candle60')
        self.assertTrue(np.isnan(b[:54]).all()); self.assertTrue(np.isfinite(b[-6:]).all())
        self.assertTrue((raster(np.full((20, 5), np.nan)) == 255).all())

    def test_split_basis_adjustment_uses_only_past_ratios(self):
        self.split[0, 60] = 2
        b = history(self.daily, self.intra, self.split, 0, 61, 'candle20')
        np.testing.assert_allclose(b[-3, :4], self.daily[0, 59, :4] / 2)
        self.assertEqual(b[-3, 4], self.daily[0, 59, 4] * 2)
        np.testing.assert_array_equal(b[-1], self.daily[0, 61])

    def test_line_does_not_leak_high_low_through_scale(self):
        a = self.daily[0, :20].copy(); b = a.copy(); b[:, 1] *= 9; b[:, 2] /= 9
        np.testing.assert_array_equal(raster(a, 'line'), raster(b, 'line'))

    def test_full_preregistered_grid_and_assignment(self):
        c = cases(); self.assertEqual(len(c), 216)
        self.assertEqual(len({x['id'] for x in c}), 216)
        groups = assignments(30); flat = [x for g in groups for x in g]
        self.assertEqual(len(flat), len(set(flat))); self.assertTrue(all(groups))
        self.assertEqual(set(flat), {x['id'] for x in c})
        with self.assertRaises(ValueError): assignments(31)


if __name__ == '__main__': unittest.main()

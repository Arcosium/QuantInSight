import unittest
from quant.timefolio_heatmap_fleet_lab import cases, geometries, assignments, comparison_cases


class FleetRegistrationTests(unittest.TestCase):
    def test_complete_cross_and_matched_controls(self):
        items = cases(); by_id = {c['id']: c for c in items}
        self.assertEqual((len(geometries()), len(items)), (13, 156))
        for c in items:
            for _, other in comparison_cases(c):
                ref = by_id[other]
                self.assertNotEqual(c['id'], other)
                for key in ['horizon', 'objective', 'epochs', 'lr', 'train_window']:
                    self.assertEqual(c[key], ref[key])
        self.assertEqual(sum(16 * 4 * (3 + len(comparison_cases(c))) + 8 * 4 for c in items), 53376)

    def test_all_cases_assigned_once_and_cap_enforced(self):
        shards = assignments()
        flat = [c for shard in shards for c in shard]
        self.assertEqual(len(shards), 10)
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(set(flat), {c['id'] for c in cases()})
        self.assertEqual(assignments(), shards)
        full = assignments(30)
        self.assertEqual(len(full),30)
        all_cases = [c for shard in full for c in shard]
        self.assertEqual(len(all_cases),len(set(all_cases)))
        self.assertEqual(set(all_cases),{c['id'] for c in cases()})
        self.assertTrue(all(full))
        self.assertEqual(assignments(30),full)
        for bad in [0, 31]:
            with self.assertRaises(ValueError): assignments(bad)


if __name__ == '__main__': unittest.main()

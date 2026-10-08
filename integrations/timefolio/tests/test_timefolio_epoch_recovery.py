import ast
import inspect
import unittest

from quant import timefolio_heatmap_epoch_evaluation as original
from quant import timefolio_heatmap_epoch_recovery as recovery


class EpochRecoveryTests(unittest.TestCase):
    def test_only_untrained_namespace_changes(self):
        before = ast.parse(inspect.getsource(original.forecasts))
        after = ast.parse(inspect.getsource(recovery.forecasts))
        changed = 0
        for node in ast.walk(before):
            if isinstance(node, ast.Constant) and node.value == 'opt_':
                node.value = 'epo_'
                changed += 1
        self.assertEqual(changed, 1)
        self.assertEqual(ast.dump(before), ast.dump(after))

    def test_canonical_seed_keys_match_registered_controls(self):
        tree = ast.parse(inspect.getsource(recovery.forecasts))
        assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                       and isinstance(n.value, ast.Call)
                       and isinstance(n.value.func, ast.Name)
                       and n.value.func.id == 'score_matrix'
                       and isinstance(n.targets[0], ast.Subscript)]
        keys = set()
        for node in assignments:
            if 'untrained' not in ast.unparse(node.targets[0]):
                continue
            code = compile(ast.Expression(node.targets[0].slice), '<key>', 'eval')
            keys.update(eval(code, {}, {'cfg': {'kind': kind}, 'seed': seed})
                        for kind in ['cnn', 'mlp'] for seed in [17, 29, 43])
        expected = {m for m in original.UNTRAINED if not m.endswith('ensemble')}
        self.assertEqual(keys, expected)


if __name__ == '__main__':
    unittest.main()

import math
import unittest
from autofolio.metrics import ledger,statistics_for,pareto_front,evaluation_scope
from autofolio.genetics import DEFAULT,normalize,fingerprint,model_fingerprint,generate
from autofolio.catalogue import book_nodes


class ResearchTests(unittest.TestCase):
    def test_pareto_dominance_and_ties(self):
        points=[dict(id='a',net_return=.5,negative_months=2),dict(id='b',net_return=.6,negative_months=3),
                dict(id='c',net_return=.4,negative_months=3),dict(id='d',net_return=.5,negative_months=2),
                dict(id='bad',net_return=float('nan'),negative_months=0)]
        self.assertEqual([r['id'] for r in pareto_front(points)],['a','d','b'])

    def test_window_keeps_prior_nav_anchor(self):
        case=dict(initial_cash=100,daily=[dict(date='20240131',nav=110),dict(date='20240201',nav=99),dict(date='20240229',nav=108.9)])
        rows=ledger(case,'20240201','20240229');metrics=statistics_for(rows)
        self.assertAlmostEqual(metrics['net_return'],-.01)
        self.assertEqual(metrics['negative_months'],1)
        self.assertAlmostEqual(metrics['mdd'],-.1)
        self.assertEqual(rows[0]['previous_nav'],110)

    def test_invalid_nav_is_not_plotted(self):
        for nav in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):ledger(dict(daily=[dict(date='20240101',nav=nav)]))

    def test_evaluation_calendar_from_legacy_book(self):
        case=dict(evaluation=dict(monthly_folds={'202401':{},'202402':{}}))
        self.assertEqual(evaluation_scope(case)[:2],('20240101','20240229'))

    def test_typed_genome_and_model_reuse(self):
        with self.assertRaises(ValueError):normalize(dict(DEFAULT,top_n=True))
        with self.assertRaises(ValueError):normalize(dict(DEFAULT,short=True))
        altered=dict(DEFAULT,band=.02,top_n=8)
        self.assertNotEqual(fingerprint(DEFAULT),fingerprint(altered))
        self.assertEqual(model_fingerprint(DEFAULT,'v'),model_fingerprint(altered,'v'))
        self.assertNotEqual(model_fingerprint(DEFAULT,'v'),model_fingerprint(DEFAULT,'w'))
        a=dict(DEFAULT,family='nn_consensus',target='net5',window=255)
        b=dict(DEFAULT,family='nn_consensus')
        self.assertEqual(fingerprint(a),fingerprint(b))

    def test_evolution_determinism_and_no_repeat(self):
        parents=[dict(job_id='parent',genome=DEFAULT,net_return=.4,negative_months=12,rule_screen_pass=True)]
        seen={fingerprint(DEFAULT)}
        a=generate(parents,seen.copy(),100,7);b=generate(parents,seen.copy(),100,7)
        self.assertEqual(a,b);self.assertEqual(len(a),100)
        self.assertEqual(len({p['id'] for p in a}),100)
        self.assertNotIn(fingerprint(DEFAULT),{p['id'] for p in a})
        self.assertTrue(any(p['operator']=='refine' for p in a))
        self.assertTrue(any(p['operator']=='explore' for p in a))

    def test_nested_book_pointers_do_not_index_metrics(self):
        case=dict(daily=[dict(date='20240101',nav=1e9)],trades=[])
        nodes=list(book_nodes(dict(bundle={'alpha':case},monthly_folds={'x':case})))
        self.assertEqual(nodes[0][0],('bundle','alpha'))
        self.assertEqual(len(nodes),1)


if __name__=='__main__':unittest.main()

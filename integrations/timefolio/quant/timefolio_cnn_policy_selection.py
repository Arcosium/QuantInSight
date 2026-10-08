"""Joint epoch/retention selection on a strictly earlier account interval.

This wraps the frozen inner account panel. Candidate policies see no outer
prices or returns; all their ledgers pay the same costs and obey the same caps.
"""
import numpy as np

from quant.timefolio_cnn_account import evaluate_daily, turnover_windows
from quant.timefolio_cnn_account_selection import load_validator, selection_key
from quant.timefolio_heatmap_locked_weighted_replay import replay


class PolicyAccountValidator:
    """Choose among predeclared retention buffers by constrained net profit."""

    def __init__(self, base, *, rank_buffers=(5,20,50)):
        buffers=tuple(rank_buffers)
        if (base.rebalance!=5 or buffers!=(5,20,50)
                or any(type(value) is not int for value in buffers)):
            raise ValueError('Registered five-day, buffers5/20/50 policy grid required')
        self.base=base
        self.rank_buffers=buffers
        self.sample_ids=base.sample_ids
        self.last_mark_index=base.last_mark_index
        self.provenance=dict(getattr(base,'provenance',{}),
            rank_buffers=list(buffers),joint_epoch_and_policy_selection=True,
            criterion='Turnover screen first, then net terminal account profit; earlier epoch and smaller buffer on exact ties',
            outer_prices_accessible=False)
        self.provenance.pop('rank_buffer',None)

    def evaluate(self, scores):
        scores=np.asarray(scores)
        if scores.shape!=self.sample_ids.shape or not np.isfinite(scores).all():
            raise ValueError('Complete finite inner predictions required')
        base=self.base
        matrix=np.full(base.panel['close'].shape,np.nan)
        matrix[base.key,base.signal]=scores
        policies=[]
        for buffer in self.rank_buffers:
            result=replay(base.panel,base.index,matrix,base.index['dates'][1],base.index['dates'][-1],
                rebalance=5,rank_buffer=buffer,rebalance_band=.005,
                top_n=12,weight=.08,gross=.8,max_orders=10,slip=.0005,participation=.05)
            windows=turnover_windows(result['daily'])
            evaluation=evaluate_daily(result['daily'])['pooled']
            policies.append(dict(**evaluation,rank_buffer=buffer,
                turnover_screen_passed=not any(w['four_violation_screen_failed'] for w in windows),
                turnover_windows=windows,fees_krw=result['metrics']['fees_krw'],
                slippage_krw=result['metrics']['slippage_krw'],
                closing_weight_breach_days=result['metrics']['closing_weight_breach_days'],
                terminal_liquidation_reserve=result['metrics']['terminal_liquidation_reserve'],
                last_account_mark_index=self.last_mark_index,start=base.index['dates'][1],
                end=base.index['dates'][-1],prediction_rows=len(scores),contest_certified=False))
        # max returns the first exact tie. Candidate order is frozen ascending.
        winner=max(policies,key=selection_key)
        return dict(winner,selected_policy={'rebalance':5,'rank_buffer':winner['rank_buffer'],
            'rebalance_band':.005,'gross':.8,'weight':.08,'top_n':12,'max_orders':10},
            candidate_policies=policies)


def load_policy_validator(path,dataset_path,arrays,fold):
    base=load_validator(path,dataset_path,arrays,fold,rebalance=5,rank_buffer=5)
    return PolicyAccountValidator(base)

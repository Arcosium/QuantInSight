"""Match inner epoch selection to the frozen contest account execution policy."""
from __future__ import annotations

import copy
import numpy as np
from quant.timefolio_cnn_account import evaluate_daily, turnover_windows
from quant.timefolio_cnn_account_selection import load_validator
from quant.timefolio_cnn_core_retention_replay import replay


def contest_policy():
    """A development choice; it is never inferred from the outer fold outcome."""
    return dict(rebalance=5, rank_buffer=20, rebalance_band=.005, top_n=12,
                weight=.08, gross=.95, max_orders=10, slip=.0005, participation=.05,
                activity_policy=dict(target_turnover=.055, intervene_after_low_weeks=2, last_sessions=2),
                separate_activity_retention=True)


class PairedAccountValidator:
    """Both arms see identical preceding dates, quotes and signal-time members."""

    def __init__(self, legacy):
        self.legacy = legacy
        for name in ['sample_ids', 'signal', 'key', 'index', 'panel', 'last_mark_index']:
            setattr(self, name, getattr(legacy, name))
        self.provenance = dict(
            legacy=copy.deepcopy(getattr(legacy, 'provenance', {})),
            contest_policy=contest_policy(),
            paired_epoch_selection=True,
            primary_arm='contest',
            criterion='Turnover-screen survivors first, then preceding inner account net terminal NAV',
            both_arms_use_same_corrected_dynamic_sector_panel=True,
            terminal_liquidation_reserve_deducted=False,
            contest_certified=False)

    def evaluate(self, scores):
        scores = np.asarray(scores)
        if scores.shape != self.sample_ids.shape or not np.isfinite(scores).all():
            raise ValueError('Complete finite inner predictions required')
        matrix = np.full(self.panel['close'].shape, np.nan)
        matrix[self.key, self.signal] = scores
        policy = contest_policy()
        result = replay(self.panel, self.index, matrix,
                        self.index['dates'][1], self.index['dates'][-1],
                        gross_schedule=np.full(len(self.index['dates']), policy['gross']), **policy)
        windows = turnover_windows(result['daily'])
        value = evaluate_daily(result['daily'])['pooled']
        return dict(**value, turnover_screen_passed=not any(
            w['four_violation_screen_failed'] for w in windows), turnover_windows=windows,
            fees_krw=result['metrics']['fees_krw'], slippage_krw=result['metrics']['slippage_krw'],
            closing_weight_breach_days=result['metrics']['closing_weight_breach_days'],
            terminal_liquidation_reserve=result['metrics']['terminal_liquidation_reserve'],
            last_account_mark_index=self.last_mark_index, start=self.index['dates'][1],
            end=self.index['dates'][-1], prediction_rows=len(scores),
            contest_policy=policy, contest_certified=False,
            paired_legacy_account=self.legacy.evaluate(scores))


def load_paired_validator(path, dataset_path, arrays, fold):
    # Reuse the validated package hashes and strict pre-outer slicing.
    legacy = load_validator(path, dataset_path, arrays, fold, rebalance=5, rank_buffer=5)
    return PairedAccountValidator(legacy)

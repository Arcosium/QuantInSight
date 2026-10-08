"""Chronological epoch selection by constrained net account profit.

The validation ledger stops before the outer fold starts. Membership never uses
future return availability. Historical metadata remains an exploratory proxy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quant.timefolio_cnn_account import evaluate_daily, turnover_windows
from quant.timefolio_cnn_dataset import sha
from quant.timefolio_heatmap_locked_weighted_replay import replay


def selection_key(evaluation):
    """Prefer a turnover-screen survivor, then its net mark-to-market profit."""
    value = evaluation['net_return']
    if not np.isfinite(value) or type(evaluation['turnover_screen_passed']) is not bool:
        raise ValueError('Finite profit and an explicit constraint-screen result required')
    return evaluation['turnover_screen_passed'], float(value)


class AccountValidator:
    def __init__(self, panel, index, arrays, fold, *, rebalance=5, rank_buffer=0):
        if rebalance not in (5, 20) or rank_buffer not in (0, 5):
            raise ValueError('Unregistered account-selection configuration')
        start, stop = fold['validation_start'], fold['test_start']
        dates = np.asarray(index['dates'])
        if (not np.array_equal(dates, arrays['dates']) or not 0 <= start < stop-1 <= len(dates)-1
                or dates[-1] > '20260923'):
            raise ValueError('Account date axes or fold boundary mismatch')
        n = len(index['codes'])
        self.panel = {}
        for name, a in panel.items():
            if name == 'sector':
                if a.shape != (n,):
                    raise ValueError('Invalid sector axis')
                self.panel[name] = a.copy()
            else:
                if a.shape != (n, len(dates)):
                    raise ValueError('Invalid account axes: '+name)
                # Do not give the evaluator access to outer-fold price outcomes.
                self.panel[name] = a[:, start:stop].copy()
        signal, key = arrays['signal_index'], arrays['security_key']
        if np.any(key < 0) or np.any(key >= n):
            raise ValueError('Unknown security index')
        self.sample_ids = np.flatnonzero(arrays['eligible'] & (signal >= start) & (signal < stop-1))
        self.signal = signal[self.sample_ids]-start
        self.key = key[self.sample_ids]
        pairs = self.key*(stop-start)+self.signal
        if not len(pairs) or len(np.unique(pairs)) != len(pairs):
            raise ValueError('Empty or duplicate inner account membership')
        self.index = dict(codes=list(index['codes']), dates=dates[start:stop].tolist())
        self.rebalance, self.rank_buffer = rebalance, rank_buffer
        self.last_mark_index = stop-1

    def evaluate(self, scores):
        scores = np.asarray(scores)
        if scores.shape != self.sample_ids.shape or not np.isfinite(scores).all():
            raise ValueError('Complete finite inner predictions required')
        matrix = np.full(self.panel['close'].shape, np.nan)
        matrix[self.key, self.signal] = scores
        result = replay(self.panel, self.index, matrix, self.index['dates'][1], self.index['dates'][-1],
                        rebalance=self.rebalance, rank_buffer=self.rank_buffer, rebalance_band=.005,
                        top_n=12, weight=.08, gross=.8, max_orders=10, slip=.0005, participation=.05)
        windows = turnover_windows(result['daily'])
        evaluation = evaluate_daily(result['daily'])['pooled']
        return dict(**evaluation, turnover_screen_passed=not any(
            w['four_violation_screen_failed'] for w in windows), turnover_windows=windows,
            fees_krw=result['metrics']['fees_krw'], slippage_krw=result['metrics']['slippage_krw'],
            closing_weight_breach_days=result['metrics']['closing_weight_breach_days'],
            terminal_liquidation_reserve=result['metrics']['terminal_liquidation_reserve'],
            last_account_mark_index=self.last_mark_index, start=self.index['dates'][1],
            end=self.index['dates'][-1], prediction_rows=len(scores),
            contest_certified=False)


def load_validator(path, dataset_path, arrays, fold, *, rebalance=5, rank_buffer=0):
    path, dataset_path = Path(path).resolve(), Path(dataset_path).resolve()
    proof = json.loads((path/'receipt.json').read_text())
    if proof['dataset_manifest_sha256'] != sha(dataset_path):
        raise ValueError('Account panel belongs to another dataset')
    for filename, digest in proof['artifacts'].items():
        p = (path/filename).resolve()
        if not p.is_relative_to(path) or sha(p) != digest:
            raise ValueError('Account panel fingerprint changed')
    index = json.loads((path/'index.json').read_text())
    manifest = json.loads(dataset_path.read_text())
    spec = manifest['arrays']['codes']; codes_path = (dataset_path.parent/spec['path']).resolve()
    if (not codes_path.is_relative_to(dataset_path.parent) or sha(codes_path) != spec['sha256']
            or np.load(codes_path, allow_pickle=False).tolist() != index['codes']):
        raise ValueError('Account security order differs from the model dataset')
    with np.load(path/'panel.npz', allow_pickle=False) as z:
        panel = {name: z[name] for name in z.files}
    validator = AccountValidator(panel, index, arrays, fold, rebalance=rebalance, rank_buffer=rank_buffer)
    validator.provenance = dict(panel_receipt_sha256=sha(path/'receipt.json'),
        limitations=proof['limitations'], rebalance=rebalance, rank_buffer=rank_buffer,
        criterion='Turnover screen first, then net terminal mark-to-market NAV; no outer outcomes',
        terminal_liquidation_reserve_deducted=False, contest_certified=False)
    return validator

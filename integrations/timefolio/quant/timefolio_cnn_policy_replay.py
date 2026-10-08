"""Recover dated execution policies from audited inner-account selections.

Policies are indexed by the forecast signal date, never by the fill date.
No returns or outer prices are accepted by this adapter.
"""
import numpy as np

from quant.timefolio_cnn_account_selection import selection_key

FIXED_POLICY = dict(rebalance=5, rebalance_band=.005, gross=.8, weight=.08,
                    top_n=12, max_orders=10)


def policy_schedule(dates, folds, receipts):
    dates = np.asarray(dates)
    if (dates.ndim != 1 or len(dates) < 2 or len(np.unique(dates)) != len(dates)
            or np.any(dates[1:] <= dates[:-1]) or str(dates[-1]) > '20260923'
            or not folds or len(folds) != len(receipts)):
        raise ValueError('Complete registered chronological policy folds required')
    schedule = np.full(len(dates), 5, dtype=np.int32)
    covered = np.zeros(len(dates), bool); choices = []
    for fold, receipt in zip(folds, receipts):
        start, stop = fold['test_start'], fold['test_end']
        cfg = receipt['config']; selection = receipt['selection']
        provenance = cfg.get('account_selection', {})
        if (receipt['fold'] != fold or not 0 < start < stop <= len(dates)
                or covered[start:stop].any()
                or provenance.get('joint_epoch_and_policy_selection') is not True
                or provenance.get('rank_buffers') != [5, 20, 50]
                or selection['criterion'] != 'inner_account_net_profit_and_retention'
                or receipt['refit']['best_epoch'] != selection['best_epoch']):
            raise ValueError('Policy receipt differs from the registered joint fit')
        if choices and start != choices[-1]['signal_end_exclusive']:
            raise ValueError('Policy folds must form a contiguous chronological prefix')
        for key in ['last_training_label_index', 'last_selection_label_index',
                    'last_selection_account_mark_index']:
            if type(receipt.get(key)) is not int or not 0 <= receipt[key] < start:
                raise ValueError('Policy selection must precede its forecast signals')
        history = selection['history']
        if [h['epoch'] for h in history] != list(range(1, cfg['epochs']+1)):
            raise ValueError('Every registered selection epoch required')
        for row in history:
            inner = row['inner_account']; candidates = inner['candidate_policies']
            if [c['rank_buffer'] for c in candidates] != [5, 20, 50]:
                raise ValueError('Every registered retention candidate required')
            for candidate in candidates:
                if (type(candidate['rank_buffer']) is not int
                        or candidate['last_account_mark_index'] != start-1
                        or candidate['last_account_mark_index'] != receipt['last_selection_account_mark_index']):
                    raise ValueError('Candidate account must stop before the outer fold')
            winner = max(candidates, key=selection_key)
            expected = dict(FIXED_POLICY, rank_buffer=winner['rank_buffer'])
            if (inner['selected_policy'] != expected or inner['rank_buffer'] != winner['rank_buffer']
                    or selection_key(inner) != selection_key(winner)):
                raise ValueError('Policy was not selected by constrained account profit')
        best = max(history, key=lambda row: selection_key(row['inner_account']))
        policy = best['inner_account']['selected_policy']
        if (selection['best_epoch'] != best['epoch'] or selection['selected_policy'] != policy
                or cfg.get('selected_policy') != policy
                or selection['selected_policy_turnover_screen_passed']
                   != best['inner_account']['turnover_screen_passed']):
            raise ValueError('Best epoch policy differs from the saved execution policy')
        schedule[start:stop] = policy['rank_buffer']; covered[start:stop] = True
        choices.append(dict(fold=fold['id'], signal_start=start, signal_end_exclusive=stop,
            first_signal_date=str(dates[start]), last_signal_date=str(dates[stop-1]),
            selected_epoch=best['epoch'], selected_policy=policy,
            inner_turnover_screen_passed=best['inner_account']['turnover_screen_passed'],
            inner_net_return=best['inner_account']['net_return'],
            last_selection_account_mark_index=receipt['last_selection_account_mark_index']))
    return schedule, covered, choices

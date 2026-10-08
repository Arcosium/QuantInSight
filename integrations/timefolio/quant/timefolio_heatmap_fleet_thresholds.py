"""User-directed stopping/retention rules, independent of frozen research gates."""
import math

MONTHS = [f'2026{month:02d}' for month in range(1, 10)]


def classify(monthly):
    """Strict thresholds; no rounding, omitted folds, or undefined Sharpe."""
    pooled, folds = monthly['pooled'], monthly['folds']
    complete = ([r['month'] for r in folds] == MONTHS
                and monthly['annual_observations'] == 252
                and monthly['later_folds_use_previous_close'] is True)
    defined = (pooled['sharpe_defined'] and math.isfinite(pooled['sharpe'])
               and all(r['sharpe_defined'] and math.isfinite(r['sharpe']) for r in folds))
    valid = bool(complete and defined)
    stop = valid and pooled['sharpe'] > 3 and all(r['sharpe'] > 1.5 for r in folds)
    retain = valid and pooled['sharpe'] > 2 and all(r['sharpe'] > 1 for r in folds)
    return dict(valid_nine_fold_metrics=valid, stop_search_and_validate=bool(stop),
                retain_candidate=bool(retain), pooled_sharpe=pooled['sharpe'],
                minimum_fold_sharpe=min((r['sharpe'] for r in folds), default=None),
                final_validation_passed=False)


def is_search_process(argv, cwd, remote):
    """Match only this study's queue and GPU workers, never shared services."""
    if cwd != remote or len(argv) < 2:
        return False
    if argv[1:3] == ['-m', 'quant.timefolio_heatmap_fleet_queue']:
        return True
    if argv[1] == '/workspace/arctrade_fleet_resume_queue.py':
        return True
    return (argv[1] == 'worker.py' and '--root' in argv
            and argv[argv.index('--root') + 1:argv.index('--root') + 2] == [remote])

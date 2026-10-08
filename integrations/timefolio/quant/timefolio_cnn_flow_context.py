"""Lagged flow ranks with separate missingness and image-height controls.

This prepares inputs only. The current frozen trainer accepts 8/16 rows; its
22-row integration must be registered and tested after the active study ends.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from quant.timefolio_cnn_context import ranks
from quant.timefolio_cnn_dataset import sha

FEATURES = ['institution_net5_rank', 'foreign_net5_rank',
    'institution_net20_rank', 'foreign_net20_rank', 'flow5_available', 'flow20_available']
MODES = ('padded', 'missingness', 'flow')


def flow_rows(institution, foreign, membership, *, lag=1):
    institution, foreign = (np.asarray(v, dtype=np.float64) for v in (institution, foreign))
    membership = np.asarray(membership)
    if (institution.ndim != 2 or foreign.shape != institution.shape
            or membership.shape != institution.shape or membership.dtype != bool
            or isinstance(lag, bool) or not isinstance(lag, int) or lag < 1
            or np.isinf(institution).any() or np.isinf(foreign).any()):
        raise ValueError('Matching date/security arrays, NaN for missing and a positive integer lag required')
    observed = np.isfinite(institution) & np.isfinite(foreign)
    count = np.vstack([np.zeros((1,institution.shape[1]), np.int32), observed.cumsum(0,dtype=np.int32)])
    sums = [np.vstack([np.zeros((1,v.shape[1])), np.where(observed, v, 0).cumsum(0)])
            for v in (institution, foreign)]
    output = np.zeros((*institution.shape, len(FEATURES)), np.float32)
    available = np.zeros((*institution.shape, 2), bool)
    # Outside that day's investable cohort, all rows stay neutral. Missing
    # observations inside the cohort have explicit -1 masks, not observed 0.
    output[:, :, 4:] = np.where(membership[:, :, None], -1., 0.)
    for day in range(len(institution)):
        stop = day-lag+1  # Exclusive; lag1 ends on the preceding session.
        for column, window in enumerate((5, 20)):
            start = stop-window
            if start < 0:
                continue
            valid = membership[day] & ((count[stop]-count[start]) == window)
            ids = np.flatnonzero(valid)
            if len(ids):
                for investor, values in enumerate(sums):
                    output[day, ids, 2*column+investor] = ranks(values[stop, ids]-values[start, ids])
                output[day, ids, 4+column] = 1.
                available[day, ids, column] = True
    return output, available


def compose_images(images, signal, security, context, *, mode):
    """Append six equal-height rows; modes alter only which information enters."""
    images, signal, security, context = map(np.asarray, (images, signal, security, context))
    if (mode not in MODES or images.ndim != 4 or images.shape[1:3] != (1,16)
            or images.shape[-1] < 4 or context.ndim != 3 or context.shape[-1] != 6
            or signal.shape != (len(images),) or security.shape != signal.shape
            or not all(np.issubdtype(v.dtype, np.integer) for v in (signal, security))
            or np.any(signal < images.shape[-1]-1) or np.any(signal >= len(context))
            or np.any(security < 0) or np.any(security >= context.shape[1])):
        raise ValueError('Registered chart, context, identity axes and control mode required')
    days = signal[:, None]-np.arange(images.shape[-1]-1, -1, -1)[None]
    gathered = context[days, security[:, None]].transpose(0,2,1)
    if not np.isfinite(gathered).all() or np.any(np.abs(gathered) > 1):
        raise ValueError('Finite bounded image rows required')
    output = np.zeros((len(images),1,22,images.shape[-1]), np.float32)
    output[:, :, :16] = images
    if mode == 'flow':
        output[:, 0, 16:] = gathered
    elif mode == 'missingness':
        output[:, 0, 20:] = gathered[:, 4:]
    return output


def read_completed_reference(reference, dates, codes):
    reference = Path(reference).resolve()
    # Never consume an active or partial collection, even when some files exist.
    complete = json.loads((reference/'complete.json').read_text())
    plan = json.loads((reference/'plan.json').read_text())
    if (complete.get('research_pass_complete') is not True
            or complete['plan_sha256'] != sha(reference/'plan.json')
            or complete['archive_receipt_sha256'] != sha(reference/'archive_receipt.json')):
        raise ValueError('Completed flow reference identity mismatch')
    date_id = {str(day):i for i,day in enumerate(dates)}
    code_id = {str(code):i for i,code in enumerate(codes)}
    if (len(date_id) != len(dates) or len(code_id) != len(codes)
            or list(date_id) != sorted(date_id) or not date_id
            or min(date_id) < plan['start'] or max(date_id) > plan['end']
            or plan['end'] > '20260923'):
        raise ValueError('Unique dated identities within the reserved cutoff required')
    entries = complete['summaries']
    if sorted(x['code'] for x in entries) != sorted(plan['codes']):
        raise ValueError('Completed reference does not cover the registered code set')
    shares = np.full((len(dates),len(codes),2), np.nan, np.float64)
    conflicts = failures = outside_calendar = 0
    for entry in entries:
        code = entry['code']
        if not re.fullmatch(r'[0-9A-Z]{6}', code):
            raise ValueError('Invalid stock identifier')
        path = reference/'merged'/f'{code}.json'
        if sha(path) != entry['merged_sha256']:
            raise ValueError('Merged flow artifact changed')
        record = json.loads(path.read_text())
        if record['code'] != code or sha(reference/'archive'/f'{code}.json') != record['archive_sha256']:
            raise ValueError('Archived flow identity mismatch')
        rows = record['rows']
        row_dates = [x['date'] for x in rows]
        if row_dates != sorted(set(row_dates)) or len(rows) != entry['observations']:
            raise ValueError('Duplicate, unordered or missing flow dates')
        conflicts += bool(record['conflicting_overlap_dates'])
        failures += record['fresh_failure'] is not None
        for row in rows:
            if (set(row) != {'date','institution_net_shares','foreign_net_shares'}
                    or not plan['start'] <= row['date'] <= plan['end']
                    or not all(type(row[k]) is int for k in ('institution_net_shares','foreign_net_shares'))):
                raise ValueError('Unexpected, future or malformed flow field')
            if row['date'] not in date_id:
                outside_calendar += 1
            elif code in code_id:
                shares[date_id[row['date']],code_id[code]] = [row['institution_net_shares'],row['foreign_net_shares']]
    return shares, dict(reference_complete_sha256=sha(reference/'complete.json'),
        source_codes=len(entries), recent_refresh_failures=failures,
        recent_overlap_conflicts=conflicts, rows_outside_reference_calendar=outside_calendar)


def build_panel(dataset, reference, output, *, lag=1):
    dataset, reference, output = (Path(p).resolve() for p in (dataset, reference, output))
    if output.exists() or (dataset/'RETIRED.json').exists():
        raise ValueError('Fresh output and a nonretired parent required')
    manifest = json.loads((dataset/'manifest.json').read_text())
    if manifest.get('feature_rows') != 16:
        raise ValueError('The registered 16-row parent is required')
    arrays = {}
    for name in ('dates','codes','signal_index','security_key','eligible'):
        spec = manifest['arrays'][name]
        path = (dataset/spec['path']).resolve()
        if not path.is_relative_to(dataset) or sha(path) != spec['sha256']:
            raise ValueError('Parent input identity changed')
        arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
    shares, source = read_completed_reference(reference, arrays['dates'], arrays['codes'])
    membership = np.zeros(shares.shape[:2], bool)
    signal, security, eligible = (arrays[k] for k in ('signal_index','security_key','eligible'))
    membership[signal, security] = eligible
    context, available = flow_rows(shares[:,:,0], shares[:,:,1], membership, lag=lag)
    output.mkdir(parents=True)
    for name, value in [('net_shares',shares),('membership',membership),('rows',context),('available',available)]:
        np.save(output/(name+'.npy'), value, allow_pickle=False)
    for name in ('dates','codes'):
        os.link(dataset/manifest['arrays'][name]['path'], output/(name+'.npy'))
    yearly = {}
    for year in sorted({str(day)[:4] for day in arrays['dates']}):
        in_year = np.array([str(day).startswith(year) for day in arrays['dates']])[signal] & eligible
        yearly[year] = dict(eligible_samples=int(in_year.sum()),
            complete_samples=[int((available[signal,security,k] & in_year).sum()) for k in range(2)])
    result = dict(created_at=time.time(), parent_manifest_sha256=sha(dataset/'manifest.json'),
        source_module_sha256=sha(__file__), rank_module_sha256=sha(Path(__file__).with_name('timefolio_cnn_context.py')),
        source=source, lag_sessions=lag, cumulative_windows=[5,20], features=FEATURES,
        yearly_coverage=yearly, control_modes=list(MODES), composed_image_height=22,
        input_panel_only=True, trainer_integration_verified=False,
        artifacts={p.name:sha(p) for p in sorted(output.glob('*.npy'))},
        price_or_return_arrays_loaded=False, prediction_membership_preserved=True,
        limitations=['Net shares, not net money or an identified national-pension series.',
            'Ranks compare only observed, contemporaneously eligible stocks; missingness is explicit.',
            'One-session lag is a research convention; original publication and revision vintages are unverified.',
            'Partial current-source coverage is not a historical point-in-time universe.',
            'The current frozen worker must pass separate 22-row integration before any training.'])
    (output/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['lag_sessions','yearly_coverage','input_panel_only','trainer_integration_verified']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('dataset','reference','output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--lag',type=int,choices=[1,2],default=1)
    args = parser.parse_args()
    build_panel(args.dataset,args.reference,args.output,lag=args.lag)

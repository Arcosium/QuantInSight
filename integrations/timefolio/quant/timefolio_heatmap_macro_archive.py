"""Extract dated allocation signals locally, without exporting private reports.

This is a mixed news/index/disclosure/prior-allocation signal, not news sentiment
alone. No model/API calls, account access, collector changes, or trading occur.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from zoneinfo import ZoneInfo

from quant.timefolio_heatmap_data import atomic_json

KST = ZoneInfo('Asia/Seoul')
ARCHIVES = ['_migration_backup_20260526_055310/cycles.db',
            '_migration_backup_20260626_223024/cycles.db', 'cycles.db']
KR_SESSIONS = {'KR_TRADING','KR_PRE_MARKET','KR_AFTER_MARKET','KR_CLOSE_REVIEW'}


def parse_allocation(text):
    """Require the explicit recommendation; never infer from unrelated prose."""
    anchor = text.find('자산 배분 권고')
    if anchor < 0: return None
    section = text[anchor:anchor+200].split('직전')[0]
    # Ambiguous "old X% -> new Y%" text must not silently select the old weight.
    if re.search(r'→|->|기존|전회',section): return None
    values = {}
    for label,key in [('주식','stock_pct'),('현금','cash_pct')]:
        match = re.search(rf'{label}\s*\*{{0,2}}\s*(\d+(?:\.\d+)?)\s*%',section)
        values[key] = float(match.group(1))/100 if match else None
    if values['stock_pct'] is None: return None
    if any(v is not None and not 0 <= v <= 1 for v in values.values()): return None
    if values['cash_pct'] is not None and sum(values.values()) > 1.01: return None
    return values


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=KST) if parsed.tzinfo is None else parsed.astimezone(KST)


def first_available(records):
    """Repeated cached text remains dated at its first completed observation."""
    earliest = {}
    for record in records:
        digest = record['report_sha256']
        if digest not in earliest or timestamp(record['available_at']) < timestamp(earliest[digest]['available_at']):
            earliest[digest] = record
    return sorted(earliest.values(),key=lambda r:(timestamp(r['available_at']),r['report_sha256']))


def daily_signals(records, dates, *, max_age_hours=96):
    records = first_available(records); result=[]; j=0; latest=None
    for date in dates:
        cutoff=datetime.strptime(date,'%Y%m%d').replace(hour=15,minute=30,tzinfo=KST)
        while j<len(records) and timestamp(records[j]['available_at']) <= cutoff:
            latest=records[j];j+=1
        age=(cutoff-timestamp(latest['available_at'])).total_seconds()/3600 if latest else None
        valid=latest is not None and 0 <= age <= max_age_hours
        result.append({'date':date,'signal_cutoff':cutoff.isoformat(),'available':valid,
                       'report_available_at':latest['available_at'] if valid else None,
                       'age_hours':age if valid else None,
                       'stock_pct':latest['stock_pct'] if valid else None,
                       'cash_pct':latest['cash_pct'] if valid else None,
                       'report_sha256':latest['report_sha256'] if valid else None})
    return result


def extract(source_root, dest, panel_index):
    source_root,dest=Path(source_root),Path(dest)
    if dest.exists(): raise RuntimeError('Use a new archive extraction directory')
    dates=json.loads(Path(panel_index).read_text())['dates']
    last=datetime.strptime(dates[-1],'%Y%m%d').replace(hour=15,minute=30,tzinfo=KST)
    records=[]; audits=[]
    for relative in ARCHIVES:
        path=source_root/relative
        if not path.is_file():continue
        db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
        db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
        rows=db.execute('SELECT started_at,ended_at,session,macro_report FROM cycles ORDER BY started_at').fetchall()
        db.close(); counts=Counter(); digest=hashlib.sha256()
        for started,ended,session,report in rows:
            if session not in KR_SESSIONS:counts['non_kr']+=1;continue
            if not report or not ended:counts['missing_report_or_end']+=1;continue
            try:start,end=timestamp(started),timestamp(ended)
            except (ValueError,TypeError):counts['invalid_time']+=1;continue
            if end < start or end-start>timedelta(days=1):counts['invalid_time']+=1;continue
            if end>last:counts['after_study_cutoff']+=1;continue
            parsed=parse_allocation(report)
            if parsed is None:counts['unparseable_allocation']+=1;continue
            # Hash for provenance and cache deduplication; never export raw text.
            sha=hashlib.sha256(' '.join(report.split()).encode()).hexdigest()
            record={'available_at':end.isoformat(),'session':session,**parsed,'report_sha256':sha}
            records.append(record);counts['accepted']+=1
            digest.update(json.dumps(record,sort_keys=True).encode())
        audits.append({'archive':relative,'counts':dict(counts),'accepted_rows_sha256':digest.hexdigest()})
    unique=first_available(records);daily=daily_signals(unique,dates)
    monthly=Counter(row['date'][:6] for row in daily if row['available'])
    report={'status':'readiness_only; no portfolio performance evaluated',
            'source_prompt_checked':'current main_swarm macro inputs are news analysis, verified indices, disclosures, prior allocation; direct account holdings are not in this prompt',
            'historical_prompt_versions':'not fully reconstructed; current prompt is not proof all older versions matched',
            'time_policy':'first completed KR-session occurrence of exact whitespace-normalised report; KST; cutoff 15:30; maximum age 96 hours',
            'study_end':last.isoformat(),'records_before_deduplication':len(records),'unique_reports':len(unique),
            'first_available':unique[0]['available_at'] if unique else None,'covered_sessions':sum(monthly.values()),
            'covered_sessions_by_month':dict(monthly),'archives':audits,
            'privacy':'Only allocation percentages, report digest, and availability/session metadata exported; no raw report, user IDs, balances, holdings, or orders.',
            'limitations':'Mixed macro signal, several account streams, cross-account prior-allocation reuse possible; historical prompt/model changes; 8000-character storage truncation; not a standalone news-effect experiment.'}
    dest.mkdir(parents=True)
    atomic_json(dest/'records.json',unique);atomic_json(dest/'daily_signals.json',daily);atomic_json(dest/'readiness.json',report)
    print(json.dumps({k:report[k] for k in ['unique_reports','first_available','covered_sessions','covered_sessions_by_month']}),flush=True)
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--dest',type=Path,required=True);ap.add_argument('--panel-index',type=Path,required=True)
    a=ap.parse_args();extract(a.source_root,a.dest,a.panel_index)

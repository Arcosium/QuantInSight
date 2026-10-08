"""One isolated, restartable local experiment; never a broker executor."""
import argparse
import json
import os
import time
from .config import RUNS, STUDY
from .store import initialize, connect, event, set_setting
from .genetics import DEFAULT, normalize, fingerprint


def run(identity, baseline=False):
    from .adapter import evaluate, atomic_json, sha
    from .catalogue import ingest_file
    initialize()
    with connect() as db:
        row=db.execute('SELECT * FROM jobs WHERE id=?',(identity,)).fetchone()
    g=normalize(json.loads(row['genome'])) if row else dict(DEFAULT)
    if not row and not baseline:raise ValueError('Unknown job')
    destination=RUNS/('baseline' if baseline else 'experiments')/identity
    destination.mkdir(parents=True,exist_ok=True)
    def progress(kind,body):
        atomic_json(destination/'progress.json',dict(timestamp=time.time(),kind=kind,**body))
        event(kind,dict(job=identity,**body))
        print(json.dumps(dict(kind=kind,**body)),flush=True)
    try:
        path=destination/'review.json'
        if not path.exists():evaluate(g,destination,progress)
        report=json.loads(path.read_text())
        if baseline:
            from .period import accepts
            from .metrics import ledger,statistics_for
            assert len(report['cases'])==1
            metrics=statistics_for(ledger(report['cases'][0]))
            assert accepts(metrics), 'Recent 36-month coverage required'
            assert report['daily_rows_verified']==metrics['sessions']
            receipt=dict(status='passed',daily_rows=metrics['sessions'],months=36,
                         result_sha256=sha(path),baseline_genome=DEFAULT,
                         adapter_sha256=sha(__import__('autofolio.adapter',fromlist=['x']).__file__))
            atomic_json(destination/'parity.json',receipt)
            set_setting('baseline_verified',receipt)
            print(json.dumps(receipt),flush=True)
        status,count=ingest_file(path,path.stat())
        if status!='indexed' or not count:raise ValueError('Result could not be indexed')
        if row:
            with connect() as db:
                db.execute("UPDATE jobs SET status='done',finished=?,result=?,model_id=? WHERE id=?",
                           (time.time(),str(path.resolve()),report['model_id'],identity))
        if report.get('stop_for_validation'):
            set_setting('enabled',False)
            set_setting('validation_candidate',identity)
            event('validation_required',dict(job=identity,reason='Sharpe >3 and every monthly fold >1.5'))
    except Exception as exc:
        progress('failed',dict(error=type(exc).__name__+': '+str(exc)))
        if row:
            with connect() as db:
                db.execute("UPDATE jobs SET status='failed',finished=?,error=? WHERE id=?",
                           (time.time(),type(exc).__name__+': '+str(exc),identity))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--job')
    parser.add_argument('--baseline',action='store_true')
    args=parser.parse_args()
    run(args.job or fingerprint(DEFAULT),args.baseline)

"""Read-only historical book index. Runtime data stay in their original directories."""
import hashlib
import json
import math
import os
import statistics
import time
from functools import lru_cache
from pathlib import Path
from .config import REPORT_ROOTS
from .metrics import evaluation_scope, ledger, normalized_date, statistics_for
from .store import connect, event, setting, set_setting

INDEX_VERSION=12


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def book_nodes(value, path=()):
    if isinstance(value,dict):
        daily=value.get('daily')
        if isinstance(daily,list) and daily and isinstance(daily[0],dict) and 'nav' in daily[0] and 'date' in daily[0]:
            if isinstance(value.get('trades'),list):
                yield path,value
            return
        for key,item in value.items():
            if isinstance(item,(dict,list)) and key not in ['daily','trades','monthly_folds','training_proofs','model_proofs']:
                yield from book_nodes(item,path+(key,))
    elif isinstance(value,list):
        for i,item in enumerate(value):yield from book_nodes(item,path+(i,))


def report_files():
    seen=set()
    for root in REPORT_ROOTS:
        if not root.exists():continue
        # Only explicit roots may be links. Never traverse links into a collector,
        # another project, model archives or a recursive project copy.
        for folder,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=[d for d in dirs if d not in ['.git','node_modules','dataset','account_panel','__pycache__',
                                                 'venv','.venv','row_pool_source_v5','row_pool_source_v4']
                      and not (Path(folder)/d).is_symlink()]
            path=Path(folder)
            if root.name=='_workspace' and not any(k in str(path).lower() for k in ['timefolio','heatmap','noncnn','autofolio']):
                if path!=root:continue
            for name in files:
                if not name.endswith('.json'):continue
                if any(s in name.lower() for s in ['prespec','receipt','execution_protocol','lease','pod_return']):continue
                if any(s in str(path/name).lower() for s in ['capital_audit_failure','repair_response','design-check','design_check']):continue
                p=path/name
                try:
                    s=p.stat()
                    identity=(s.st_dev,s.st_ino)
                    if identity in seen or s.st_size>160*1024*1024:continue
                    seen.add(identity)
                    yield p,s
                except OSError:continue


def family_for(path, report):
    if isinstance(report,dict) and isinstance(report.get('genome'),dict):
        return {'quadratic62':'비선형 62계수','linear31':'선형 31계수','nn_consensus':'소형 신경망'}.get(report['genome'].get('family'),'계좌 전략')
    s=str(path).lower()
    if 'quadratic' in s:return '비선형 62계수'
    if 'linear31' in s:return '선형 31계수'
    if 'tiny_lambda' in s or 'consensus' in s or 'latent_gate' in s:return '소형 신경망'
    if 'ridge' in s:return 'Ridge'
    if any(k in s for k in ['tree','lgb','boost']):return '트리 모델'
    if any(k in s for k in ['cnn','resnet','heatmap','vision','vit','chart']):return '차트 이미지'
    return '계좌 전략'


def group_key(case):
    return str(next((case[k] for k in ['arm','kind','strategy','policy','label','variant','config','name']
                     if k in case and isinstance(case[k],(str,int,float))),''))


def title_for(path,key):
    name=path.parent.name if path.name in ['review.json','result.json','book.json'] else path.stem
    for suffix in ['_account_v1_20261007','_ranker_v1_20261008','_v1_20261006','_v1_20261007','_v1_20261008']:
        name=name.removesuffix(suffix)
    return name+(' · '+key if key else '')


def genome_title(g):
    family={'quadratic62':'비선형 62계수','linear31':'선형 31계수','nn_consensus':'신경망 합의'}[g['family']]
    window=f"{g['window']}일 학습" if g['window'] else '누적 학습'
    return f"{family} · {window} · {g['target']} · Top {g['top_n']} · {g['rebalance']}일 · {round(g['gross_low']*100)}/{round(g['gross_high']*100)}%"


def proof_status(path):
    if path.name=='review.json' and '_account_v1_' in path.parent.name:
        sibling=path.parent.with_name(path.parent.name.replace('_account_v1_','_account_review_v1_'))/'main_review.json'
        if sibling.exists():
            try:
                proof=json.loads(sibling.read_text())
                if 'passed' in proof.get('status','') and proof.get('result_sha256')==sha(path):
                    return '계좌 장부 검증'
            except (ValueError,OSError):pass
    return '연구 결과'


def ingest_file(path,stat):
    try:report=json.loads(path.read_text())
    except (ValueError,OSError):return 'incomplete',0
    nodes=list(book_nodes(report))
    if not nodes:return 'no_book',0
    from .period import window, using_window, require_complete, accepts
    from .evaluation import performance, segment_metrics, valid_window, PROTOCOL
    span=report.get("evaluation_window") or [report.get("recipe",{}).get("evaluation_start"),report.get("recipe",{}).get("evaluation_end")]
    if not all(span):span=window()
    if not valid_window(span):return "unmatched",0
    groups={}
    for pointer,case in nodes:
        codes=[str(t.get('code',t.get('ticker',''))) for t in case['trades'][:20]]
        if report.get('market','timefolio') in ('kr','timefolio') and codes and not any(c.isdigit() and len(c)==6 for c in codes):continue
        try:
            _,_,recorded=evaluation_scope(case)
            rows=ledger(case,*span)
            metrics=segment_metrics(rows,*span,36)
            if metrics is None:continue
            metrics['market']=report.get('market','timefolio')
            with using_window(*span):
                require_complete([r['date'] for r in rows],metrics['market'])
                if not accepts(metrics):continue
            split=performance(rows,*span)
            if recorded and isinstance(recorded.get('pooled'),dict):
                stored=recorded['pooled'].get('net_return')
                if isinstance(stored,(int,float)) and abs(stored-metrics['net_return'])>1e-6:
                    raise ValueError('Stored NAV performance mismatch')
            calendar=[r['date'] for r in rows]
            cohort=hashlib.sha256(json.dumps(calendar).encode()).hexdigest()[:12]
            key=group_key(case)
            turnover_pass=case.get('turnover_pass')
            if turnover_pass is None:
                account_metrics=case.get('account_metrics',case.get('metrics',{}))
                if isinstance(account_metrics,dict) and isinstance(account_metrics.get('four_week_turnover_stop'),bool):
                    turnover_pass=not account_metrics['four_week_turnover_stop']
            groups.setdefault((key,cohort),[]).append(dict(pointer=list(pointer),phase=case.get('phase',case.get('offset_sessions',len(groups))),
                metrics=metrics,performance=split,trade_count=sum(metrics['start']<=normalized_date(t.get('date',''))<=metrics['end'] for t in case['trades']),
                turnover_pass=turnover_pass,warm_start=normalized_date(case['daily'][0]['date'])))
        except (ValueError,TypeError,KeyError,ZeroDivisionError):continue
    count=0
    contest_validation=None
    if report.get('market','timefolio')=='timefolio':
        from .contest_validation import report_assessment
        contest_validation=report_assessment(dict(report,cases=[case for _,case in nodes]))
    source_digest=sha(path)
    verified=proof_status(path)
    with connect() as db:
        for (key,cohort),cases in groups.items():
            identity=hashlib.sha256((str(path.resolve())+'\0'+key+'\0'+cohort).encode()).hexdigest()[:20]
            means={k:statistics.mean(c['metrics'][k] for c in cases) for k in ['net_return','negative_months','mdd','mean_loss_month','worst_month']}
            sharpes=[c['metrics']['sharpe'] for c in cases if c['metrics']['sharpe'] is not None]
            first=cases[0]['metrics']
            splits={}
            for phase in ('is','os','ros'):
                parts=[c['performance'][phase] for c in cases]
                if any(p is None for p in parts):break
                splits[phase]=dict(parts[0])
                for field in ('net_return','negative_months','mdd','mean_loss_month','worst_month','sharpe'):
                    values=[p[field] for p in parts if p[field] is not None]
                    splits[phase][field]=statistics.mean(values) if values else None
            genome=(report.get('genome') or report.get('definition')) if isinstance(report,dict) else None
            payload=dict(id=identity,title=report.get('title') or (genome_title(genome) if genome else title_for(path,key)),family=report.get('family') or family_for(path,report),cohort=cohort,market=report.get('market','timefolio'),owner_id=report.get('owner_id'),
                **means,sharpe=statistics.mean(sharpes) if sharpes else None,months=first['months'],start=first['start'],end=first['end'],
                sessions=first['sessions'],phase_count=len(cases),return_min=min(c['metrics']['net_return'] for c in cases),
                return_max=max(c['metrics']['net_return'] for c in cases),verification=verified,
                rule_screen_pass=all(c['turnover_pass'] is True for c in cases),
                rule_screen_known=all(c['turnover_pass'] is not None for c in cases),
                independent_holdout=bool(report.get('independent_holdout',False)) if isinstance(report,dict) else False,
                contest_certified=bool(contest_validation and contest_validation['competition_compliance_verified']),
                competition_compliance_verified=bool(contest_validation and contest_validation['competition_compliance_verified']),
                contest_validation=contest_validation,
                evaluation_protocol=PROTOCOL,evaluation_window=list(span),performance=splits,
                selection_scope='os',training_summary=report.get('training_summary'),
                protocol_origin='native' if report.get('evaluation_protocol')==PROTOCOL else 'retrospective',
                cases=cases,source_digest=source_digest,source_name=path.parent.name+'/'+path.name,
                limitations=report.get('limitations') or ['과거 개발 표본 재사용','여러 시작일은 독립 폴드가 아닌 민감도 비교','종목·섹터·체결 자료의 근사치 및 현금 배당 미정산'],
                genome=(report.get('genome') or report.get('definition')) if isinstance(report,dict) else None)
            db.execute('INSERT INTO strategies VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET title=excluded.title,family=excluded.family,cohort=excluded.cohort,payload=excluded.payload,updated=excluded.updated',
                       (identity,payload['title'],payload['family'],cohort,json.dumps(payload,ensure_ascii=False,allow_nan=False),str(path.resolve()),time.time()))
            count+=1
        db.execute('INSERT INTO sources VALUES (?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime,size=excluded.size,digest=excluded.digest,status=excluded.status',
                   (str(path.resolve()),stat.st_mtime,stat.st_size,source_digest,'indexed' if count else 'unmatched'))
    return 'indexed' if count else 'unmatched',count


def refresh():
    with connect() as db:known={r['path']:(r['mtime'],r['size'],r['status']) for r in db.execute('SELECT * FROM sources')}
    force=setting('catalogue_version',0)!=INDEX_VERSION
    counts={'checked':0,'updated':0,'strategies':0,'unmatched':0,'incomplete':0}
    for path,stat in report_files():
        counts['checked']+=1
        if not force and known.get(str(path.resolve()),())[:2]==(stat.st_mtime,stat.st_size):continue
        status,n=ingest_file(path,stat)
        counts['updated']+=1
        counts['strategies']+=n
        if status in counts:counts[status]+=1
        if status=='no_book':
            with connect() as db:
                db.execute('INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?)',(str(path.resolve()),stat.st_mtime,stat.st_size,None,status))
    event('catalogue_refresh',counts)
    set_setting('catalogue_version',INDEX_VERSION)
    return counts


def rows(cohort=None):
    with connect() as db:
        query='SELECT payload FROM strategies'+(' WHERE cohort=?' if cohort else '')
        found=db.execute(query,(cohort,) if cohort else ()).fetchall()
    return [json.loads(r['payload']) for r in found
            if not any(s in json.loads(r['payload']).get('source_name','') for s in ['capital_audit_failure','repair_response'])]


@lru_cache(maxsize=4)
def read_report(path,mtime,size):
    return json.loads(Path(path).read_text())


def strategy_case(identity,phase=0):
    with connect() as db:r=db.execute('SELECT * FROM strategies WHERE id=?',(identity,)).fetchone()
    if not r:raise KeyError(identity)
    summary=json.loads(r['payload'])
    if not 0<=phase<len(summary['cases']):raise ValueError('Invalid phase')
    path=Path(r['source'])
    stat=path.stat()
    value=read_report(str(path),stat.st_mtime,stat.st_size)
    for key in summary['cases'][phase]['pointer']:value=value[key]
    return summary,value

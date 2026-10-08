"""Append the opening-score cohort to the unreduced cumulative return family."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from quant.timefolio_heatmap_opening_rank import VARIANTS, MEMBERS, hypotheses, identity
from quant.timefolio_heatmap_fleet_accounts import policy_grid
from quant.timefolio_heatmap_fleet_breadth_accounts import units
from quant.timefolio_heatmap_fleet_breadth_inference import nav_returns, should_resample
from quant.timefolio_heatmap_fleet_metrics import monthly_target
from quant.timefolio_heatmap_fleet_thresholds import classify
from quant.timefolio_heatmap_gpu_worker import digest, write
from quant.timefolio_heatmap_parallel_bootstrap import family_bootstrap
from quant.timefolio_heatmap_fleet_audit_helpers import count_bootstrap

ORIGIN='opening_rank_v1'
def read(p):return json.loads(p.read_text())


def candidates(cases,frame,stats,p_column='adjusted_p'):
    rows=frame.set_index('id').to_dict('index');tests={k:g for k,g in stats[stats.origin==ORIGIN].groupby('id',sort=False)}
    expected={(label,b) for label in ['cash','matching_untrained_opening','same_opening_nonimage','unchanged_original'] for b in [5,10]}
    selected,basic=[],[]
    for case in cases:
        for v in VARIANTS:
            for p in policy_grid():
                keys=[identity(f'flt_{case}_trained_{m}',v,p) for m in MEMBERS]
                stable=all(rows[k]['return']>0 and rows[k]['mdd']>-.2 and rows[k]['positive_blocks']>=2
                    and not rows[k]['four_week_turnover_stop'] for k in keys)
                key=keys[-1];group=tests[key]
                assert len(group)==8 and set(group[['comparator','block']].itertuples(index=False,name=None))==expected
                if stable:basic.append(key)
                if stable and (group[p_column]<.025).all() and (group.simultaneous_lower95>0).all():selected.append(key)
    return selected,basic


def run(root,source,accounts,prior,output):
    assert not output.exists();output.mkdir(parents=True)
    spec=read(root/'registration.json');cases=spec['source_cases'];done=read(accounts/'complete.json')
    assert len(done['completed'])==73 and not done['remaining'] and not done['stopped']
    assert not Path(spec['shared_stop']).exists()
    gate=read(root/'main_account_review.json')
    assert gate['accounts']==4672 and gate['units']==73 and gate['strict_thresholds_independently_recomputed']
    assert gate['opening_scores_and_causality_verified'] and gate['zero_overlay_account_parities_verified']
    assert gate['derived_scores_and_causality_verified'] and gate['maximum_independent_sharpe_error']<1e-10
    assert digest(prior/'artifact_hashes.json')==spec['prior_inference_manifest_sha256']
    for name,h in read(prior/'artifact_hashes.json').items():assert digest(prior/name)==h
    family=[tuple(r) for r in read(prior/'family.json')]
    old=np.load(prior/'joint_differences.npy',allow_pickle=False)
    assert old.shape==(179,208352) and len(family)==208352
    returns={p.stem:nav_returns(read(p)) for p in (source/'sources').glob('*/portfolios/*.json')}
    assert len(returns)==1152
    rows=[];records=[];stops=[];proofs={};max_error=0.
    for unit in units(cases):
        folder=accounts/'units'/unit['id'];receipt=read(folder/'complete.json')
        assert receipt['zero_overlay_accounts_reproduced']==16
        assert receipt['accounts']==64 and receipt['completed'] and receipt['original_accounts_reproduced']==16
        for name,key in [('summary.json','summary_sha256'),('source_score.json','source_score_sha256'),
                         ('baseline_parity.json','baseline_parity_sha256'),('zero_overlay_parity.json','zero_overlay_parity_sha256'),('derived_hashes.json','derived_hashes_sha256')]:
            assert digest(folder/name)==receipt[key]
        assert gate['unit_proofs'][unit['id']]==digest(folder/'complete.json')
        proofs[unit['id']]=digest(folder/'complete.json')
        for row in read(folder/'summary.json'):
            target=folder/row['id']
            for name,h in read(target/'complete.json')['hashes'].items():assert digest(target/name)==h
            result=read(target/'portfolio.json');monthly=monthly_target(result['daily'])
            assert monthly==read(target/'monthly.json');verdict=classify(monthly)
            assert all(row[k]==v for k,v in verdict.items())
            audit=read(target/'audit.json');assert audit['opening_decision_causality_passed']
            assert not any(audit[k] for k in ['additional_errors','post_buy_limit_violations','announced_action_errors'])
            assert not audit['dated_hynix_audit']['post_buy_limit_errors']
            max_error=max(max_error,audit['maximum_nav_reconstruction_error_krw'],audit['closing']['exposure']['maximum_nav_error'])
            assert row['id'] not in returns;returns[row['id']]=nav_returns(result);rows.append(row)
            if verdict['retain_candidate']:records.append(dict(id=row['id'],monthly=monthly,**verdict))
            if verdict['stop_search_and_validate']:stops.append(dict(id=row['id'],monthly=monthly,**verdict))
    additions=hypotheses(cases)
    assert len(additions)==11584 and read(root/'hypotheses.json')==[list(r) for r in additions]
    assert len(rows)==4672 and {r['id'] for r in rows}=={r[0] for r in additions} and max_error<.01
    matrix=np.empty((179,219936));matrix[:,:208352]=old
    for i,(key,label,ref) in enumerate(additions,208352):
        matrix[:,i]=returns[key]-(returns[ref] if ref else 0.);family.append((key,label,ORIGIN))
    assert len(set(family))==219936
    np.save(output/'joint_differences.npy',matrix);write(output/'family.json',family)
    frame=pd.DataFrame(rows);frame.to_csv(output/'portfolio_summary.csv',index=False)
    write(output/'retained_candidates.json',records);write(output/'stop_candidates.json',stops)
    write(output/'account_review.json',dict(accounts=4672,maximum_nav_error=max_error,source_unit_hashes=proofs))
    resampled=should_resample(records,stops);selected=None;extra={}
    if resampled:
        results=[]
        for block in [5,10]:
            tick=time.monotonic();report=family_bootstrap(matrix,block=block,draws=4000,seed=57,workers=8)
            for i,(key,label,origin) in enumerate(family):
                results.append(dict(id=key,comparator=label,origin=origin,block=block,
                    **{k:float(report[k][i]) for k in ['mean','standard_error','adjusted_p','marginal_p','simultaneous_lower95']}))
            write(output/f'block{block}_receipt.json',dict(seconds=time.monotonic()-tick,hypotheses=219936,draws=4000,critical_max_t=report['critical_max_t']))
        stats=pd.DataFrame(results);stats.to_csv(output/'joint_bootstrap.csv',index=False)
        reports,bounds=count_bootstrap(family,matrix.T,stats);bounds.to_csv(output/'bootstrap_roundoff_bounds.csv',index=False)
        selected,basic=candidates(cases,frame,stats);conservative,basic2=candidates(cases,frame,bounds,'adjusted_high')
        assert selected==conservative and basic==basic2
        extra=dict(all_seed_basic_cases=len(basic),independent_bootstrap=reports)
    write(output/'evaluation_summary.json',dict(at=time.time(),status='main_review_pending',accounts=4672,
        new_hypotheses=11584,joint_hypotheses=219936,record_accounts=len(records),stop_accounts=len(stops),
        resampling_performed=resampled,statistical_candidates=selected,**extra,
        statistical_validation='full_family_resampled' if resampled else 'deferred_no_record_or_stop_candidate',
        all_comparison_returns_preserved=True,independent_confirmation=False,full_contest_compliance_certified=False,
        reserved_outcomes_read=False,orders_submitted=False))
    write(output/'artifact_hashes.json',{p.name:digest(p) for p in output.iterdir() if p.is_file()})
    write(output/'complete.json',dict(status='main_review_pending',registration_sha256=digest(root/'registration.json'),
        source_sha256=digest(__file__),prior_family_hypotheses=208352,new_hypotheses=11584,joint_hypotheses=219936,
        resampling_performed=resampled,final_validation_passed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['root','source','accounts','prior','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.root,a.source,a.accounts,a.prior,a.output)

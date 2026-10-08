"""Full-family chart inference with an independent bounded-memory count audit."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from quant.timefolio_chart_lab import cases,MEMBERS
from quant.timefolio_heatmap_fleet_accounts import policy_grid,account_id
from quant.timefolio_heatmap_gpu_worker import digest,write
from quant.timefolio_heatmap_parallel_bootstrap import family_bootstrap


def count_bounds(matrix, report, *, block, draws=4000, seed=57, columns=256):
    """Independent day-count multiplication, retaining only one column block."""
    a=np.asarray(matrix,dtype=float)
    if a.ndim!=2 or len(a)<2 or not a.shape[1] or not np.isfinite(a).all():raise ValueError('Finite day-by-comparison matrix required')
    if columns<1 or draws<2 or block<1:raise ValueError('Positive dimensions required')
    mean=a.mean(axis=0);rng=np.random.default_rng(seed)
    starts=rng.integers(0,len(a),size=(draws,(len(a)+block-1)//block))
    indices=((starts[:,:,None]+np.arange(block))%len(a)).reshape(draws,-1)[:,:len(a)]
    counts=np.stack([np.bincount(row,minlength=len(a)) for row in indices]).astype(float)
    se=np.empty(a.shape[1]);maxima=np.full(draws,-np.inf)
    for start in range(0,a.shape[1],columns):
        end=min(start+columns,a.shape[1]);boot=counts@(a[:,start:end]-mean[start:end])/len(a)
        se[start:end]=np.maximum(boot.std(0,ddof=1),1e-12)
        maxima=np.maximum(maxima,(boot/se[start:end]).max(axis=1))
    observed=mean/se;lower=mean-np.quantile(maxima,.95)*se
    errors={k:float(np.max(np.abs(expected-report[k]))) for k,expected in [('mean',mean),('standard_error',se),('simultaneous_lower95',lower)]}
    assert max(errors.values())<1e-13
    values={k:np.empty(a.shape[1]) for k in ['marginal_low','marginal_high','adjusted_low','adjusted_high']}
    eps=np.finfo(float).eps
    for start in range(0,a.shape[1],columns):
        end=min(start+columns,a.shape[1]);boot=counts@(a[:,start:end]-mean[start:end])/len(a)
        tolerance=512*eps*np.maximum(np.max(np.abs(a[:,start:end]),axis=0),1e-300)
        t_tolerance=512*eps*(np.max(np.abs(maxima))+np.abs(observed[start:end])+1)
        values['marginal_low'][start:end]=(1+(boot>mean[start:end]+tolerance).sum(0))/(draws+1)
        values['marginal_high'][start:end]=(1+(boot>=mean[start:end]-tolerance).sum(0))/(draws+1)
        values['adjusted_low'][start:end]=(1+(maxima[:,None]>observed[start:end]+t_tolerance).sum(0))/(draws+1)
        values['adjusted_high'][start:end]=(1+(maxima[:,None]>=observed[start:end]-t_tolerance).sum(0))/(draws+1)
    for name in ['marginal','adjusted']:
        assert np.all(report[name+'_p']>=values[name+'_low']-1e-12)
        assert np.all(report[name+'_p']<=values[name+'_high']+1e-12)
    return dict(values,simultaneous_lower95=lower,maximum_point_errors=errors,
        maximum_bootstrap_columns=columns,recorded_p_values_within_roundoff_bounds=True)


def run(root,output):
    root,output=Path(root),Path(output)
    assert not output.exists();output.mkdir(parents=True)
    def read(path):return json.loads(path.read_text())
    for name,h in read(root/'input_hashes.json').items():
        path=root/name;assert path.resolve().is_relative_to(root.resolve()) and digest(path)==h
    spec=read(root/'registration.json');assert spec['cases']==216 and spec['accounts']==27664
    assert spec['prior_hypotheses']==219936 and spec['new_hypotheses']==106000 and spec['joint_hypotheses']==325936
    assert spec['all_cases_main_reviewed'] and not spec['stop_interrupted_partial_study']
    family=[tuple(v) for v in read(root/'prior_family.json')];assert len(family)==219936
    old=np.load(root/'prior_differences.npy',mmap_mode='r');assert old.shape==(179,219936)
    ids=read(root/'account_ids.json');metrics=read(root/'metrics.json');returns=np.load(root/'daily_returns.npy',mmap_mode='r')
    assert len(ids)==len(set(ids))==len(metrics)==27664 and returns.shape==(27664,179)
    assert [r['id'] for r in metrics]==ids and np.isfinite(returns).all()
    position={key:i for i,key in enumerate(ids)};additions=read(root/'hypotheses.json');assert len(additions)==106000
    matrix=np.lib.format.open_memmap(output/'joint_differences.npy',mode='w+',dtype='float64',shape=(179,325936))
    matrix[:,:219936]=old
    for col,(key,label,ref) in enumerate(additions,219936):
        matrix[:,col]=returns[position[key]]-(returns[position[ref]] if ref else 0.)
        family.append((key,label,'chart_expansion_v1'))
    assert len(set(family))==325936;matrix.flush()
    write(output/'family.json',family)
    records=[r for r in metrics if r['retain_candidate']];stops=[r for r in metrics if r['stop_search_and_validate']]
    write(output/'retained_candidates.json',records);write(output/'stop_candidates.json',stops)
    resample=bool(records or stops);selected=None;basic=[];reports=[]
    if resample:
        new_indices={}
        for i,(key,label,origin) in enumerate(family):
            if origin=='chart_expansion_v1':new_indices.setdefault(key,[]).append(i)
        passed={key:True for key in new_indices};rows={r['id']:r for r in metrics}
        with (output/'joint_bootstrap.csv').open('w') as stream,(output/'bootstrap_roundoff_bounds.csv').open('w') as audit:
            fields=['id','comparator','origin','block','mean','standard_error','adjusted_p','marginal_p','simultaneous_lower95']
            bound_fields=['id','comparator','origin','block','marginal_low','marginal_high','adjusted_low','adjusted_high','simultaneous_lower95']
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();bw=csv.DictWriter(audit,fieldnames=bound_fields);bw.writeheader()
            for block in [5,10]:
                tick=time.monotonic();report=family_bootstrap(matrix,block=block,draws=4000,seed=57,workers=8)
                bounds=count_bounds(matrix,report,block=block)
                for i,(key,label,origin) in enumerate(family):
                    identity=dict(id=key,comparator=label,origin=origin,block=block)
                    writer.writerow(dict(identity,**{k:float(report[k][i]) for k in fields[4:]}))
                    bw.writerow(dict(identity,**{k:float(bounds[k][i]) for k in bound_fields[4:]}))
                for key,indices in new_indices.items():
                    point=bool(np.all(report['adjusted_p'][indices]<.025) and np.all(report['simultaneous_lower95'][indices]>0))
                    conservative=bool(np.all(bounds['adjusted_high'][indices]<.025) and np.all(bounds['simultaneous_lower95'][indices]>0))
                    # A numerical boundary ambiguity cannot promote a strategy.
                    passed[key]&=point and conservative
                receipt=dict(block=block,draws=4000,hypotheses=325936,seconds=time.monotonic()-tick,
                    critical_max_t=report['critical_max_t'],maximum_point_errors=bounds['maximum_point_errors'],
                    recorded_p_values_within_roundoff_bounds=True,independent_count_audit_columns=256)
                reports.append(receipt);write(output/f'block{block}_receipt.json',receipt)
        selected=[]
        for cfg in cases():
            for policy in policy_grid():
                keys=[account_id(f"chart_{cfg['id']}_trained_{member}",policy) for member in MEMBERS]
                stable=all(rows[k]['return_value']>0 and rows[k]['mdd']>-.2 and rows[k]['positive_blocks']>=2
                    and not rows[k]['four_week_turnover_stop'] for k in keys)
                if stable:basic.append(keys[-1])
                if stable and passed[keys[-1]]:selected.append(keys[-1])
    summary=dict(at=time.time(),status='main_review_pending',accounts=27664,prior_hypotheses=219936,
        new_hypotheses=106000,joint_hypotheses=325936,record_accounts=len(records),stop_accounts=len(stops),
        resampling_performed=resample,statistical_candidates=selected,all_seed_basic_cases=len(basic) if resample else None,
        independent_bootstrap=reports,statistical_validation='full_family_resampled' if resample else 'deferred_no_record_or_stop_candidate',
        all_comparison_returns_preserved=True,last_predecessor_resampled_hypotheses=168816,
        independent_confirmation=False,full_contest_compliance_certified=False,reserved_outcomes_read=False,orders_submitted=False)
    write(output/'evaluation_summary.json',summary)
    write(output/'artifact_hashes.json',{p.name:digest(p) for p in output.iterdir() if p.is_file()})
    write(output/'complete.json',dict(summary,registration_sha256=digest(root/'registration.json'),
        source_sha256=digest(Path(__file__)),input_manifest_sha256=digest(root/'input_hashes.json'),final_validation_passed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)

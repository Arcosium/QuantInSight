"""Fixed refit durations versus inner-IC epoch selection under identical execution."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from quant.timefolio_heatmap_sector_ceiling import (DEST as PRIOR, MODES, sector_panel,
    prior_family as ceiling_prior_family, new_family as ceiling_new_family)
from quant.timefolio_heatmap_stock_relation_evaluation import (
    DEST as RAW_SOURCE, MEMBERS, CASES, BUFFERS, model_info as legacy_info)
from quant.timefolio_heatmap_stock_relation_training import DATA_ROOT, digest, check_hashes
from quant.timefolio_heatmap_initialisation_probe import DEST as INITIAL, state_digest
from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_gpu_merge import merge_months
from quant.timefolio_heatmap_consensus import rank_consensus
from quant.timefolio_heatmap_study import context, score_matrix
from quant.timefolio_heatmap_walkforward import AlternativeNet, predict
from quant.timefolio_heatmap_action_amendment import amend_actions, release_audit
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_stock_limits import historical_stock_caps, audit_pre_july_hynix
from quant.timefolio_heatmap_week_boundaries import assess_weeks
from quant.timefolio_heatmap_walkforward_eval import calendar_blocks
from quant.timefolio_heatmap_bounded_bootstrap import family_bootstrap
from quant.timefolio_heatmap_gpu_worker import verify as verify_package
from quant.timefolio_heatmap_lab_models import cases,LabNet
from quant.timefolio_heatmap_lab_pod import validate_job
from quant.timefolio_heatmap_gpu_worker import predict as device_predict
from quant.timefolio_heatmap_locked_replay import replay

def run():
    spec = freeze(); torch.set_num_threads(4)
    prior_review=json.loads((PRIOR/'validation_complete.json').read_text())
    assert prior_review['status']=='numerical_validation_complete'
    check_hashes(prior_review['main_review_evidence_hashes'])
    atomic_json(DEST/'prior_review_receipt.json',dict(path=str(PRIOR/'validation_complete.json'),sha256=digest(PRIOR/'validation_complete.json')))
    preflight = json.loads((DEST / 'preflight.json').read_text())
    if (preflight['prior_hypotheses'], preflight['joint_hypotheses']) != (60976, 64368): raise RuntimeError('Daily-history preflight required')
    p, ix, ci, di = context(DATA_ROOT); assert ix['dates'][-1] == '20260923'
    matrices = forecasts(p, ix, ci, di)
    p, release = amend_actions(p, ix, spec['training']['actions'])
    panels = {mode: sector_panel(p, mode) for mode in MODES}; caps = historical_stock_caps(ix['codes'], ix['dates'])
    dates = {date: day for day, date in enumerate(ix['dates'])}
    folder = DEST / 'portfolios'; folder.mkdir(exist_ok=True); scores = DEST / 'scores'; scores.mkdir(exist_ok=True)
    rows, audits, regressions = [], {}, []
    for model in MODELS:
        path = scores / (model + '.npy'); assert not path.exists(); np.save(path, matrices[model])
        if model in LEGACY: assert digest(path) == digest(RAW_SOURCE / 'scores' / path.name)
        held = {refresh: snapshot_scores(matrices[model], p['eligible'], ix['dates'], '20260101', '20260923', refresh) for refresh in [1, 5]}
        for case in CASES:
            alpha, origins = held[case['refresh']]
            for buffer in BUFFERS:
                for ceiling in MODES:
                    panel = panels[ceiling]; key = account_id(model, case['id'], buffer, ceiling)
                    path = folder / (key + '.json'); assert not path.exists()
                    result = replay(panel, ix, alpha, '20260101', '20260923', top_n=12, weight=.05,
                        max_orders=case['max_orders'], rank_buffer=buffer, rebalance=5, rebalance_band=.0005,
                        return_trades=True, planning_price='open', action_release_dates=release, stock_cap_schedule=caps, locked_repair=True)
                    for trade in result['trades']:
                        day = dates[trade['signal_date']]; origin = origins[day]; assert 0 <= origin <= day
                        trade['portfolio_score_date'] = ix['dates'][origin]
                    result['metrics'].update(assess_weeks(result)); atomic_json(path, result)
                    prior = legacy_path(model, case['id'], buffer, ceiling)
                    if prior is not None:
                        assert result == json.loads(prior.read_text()); regressions.append(dict(id=key, prior=str(prior), all_fields_exact=True))
                    audit = audit_fills(panel, ix, result); audit.update(additional_checks(panel, ix, result, None, max_orders=case['max_orders']))
                    audit['announced_action_errors'] = release_audit(panel, ix, result, release)
                    audit['dated_hynix_audit'] = audit_pre_july_hynix(panel, ix, result)
                    audit['snapshot_origin_errors'] = [trade['date'] for trade in result['trades']
                        if trade['portfolio_score_date'] != ix['dates'][origins[dates[trade['signal_date']]]]]
                    if any(audit[name] for name in ['post_buy_limit_violations', 'additional_errors', 'announced_action_errors', 'snapshot_origin_errors']) or audit['dated_hynix_audit']['post_buy_limit_errors'] or audit['maximum_nav_reconstruction_error_krw'] > .01:
                        atomic_json(DEST / 'failed_audit.json', dict(id=key, audit=audit)); raise AssertionError('Daily-history account audit failed')
                    audits[key] = audit; quarters = calendar_blocks(result['daily'])
                    rows.append(dict(id=key, model=model, **model_info(model), case=case['id'], buffer=buffer, band_bp=5,
                        ceiling=ceiling, **result['metrics'], **quarters, positive_blocks=sum(value > 0 for value in quarters.values())))
        print(json.dumps(dict(model=model, portfolios=len(rows))), flush=True)
    assert len(rows) == 784 and len(regressions) == 0
    atomic_json(DEST / 'score_hashes.json', {str(path): digest(path) for path in sorted(scores.glob('*.npy'))})
    atomic_json(DEST / 'independent_audit.json', audits); atomic_json(DEST / 'baseline_regression.json', regressions)
    frame = pd.DataFrame(rows); frame.to_csv(DEST / 'portfolio_summary.csv', index=False)
    family, differences, returns, error = prior_family()
    atomic_json(DEST / 'prior_family_reconstruction.json', dict(prior_hypotheses=len(family), maximum_mean_error=error))
    for key, comp, reference in new_family():
        family.append((key, comp, 'epoch_lab'))
        differences.append(returns(DEST, key) - (returns(DEST, reference) if reference is not None else 0.))
    matrix = np.column_stack(differences); assert matrix.shape == (179, 64368) and len(set(family)) == 64368
    records = []
    for block in [5, 10]:
        stats = family_bootstrap(matrix, block=block, draws=4000, seed=57)
        for i, (key, comp, origin) in enumerate(family):
            records.append(dict(id=key, comparator=comp, origin=origin, block=block,
                **{name: float(stats[name][i]) for name in ['mean', 'standard_error', 'adjusted_p', 'marginal_p', 'simultaneous_lower95']}))
    stats = pd.DataFrame(records); stats.to_csv(DEST / 'joint_bootstrap.csv', index=False)
    selected = candidate_ids(frame, stats); check_hashes(spec['hashes'])
    atomic_json(DEST / 'evaluation_summary.json', dict(status='development_only', portfolios=784, new_portfolios=784,
        legacy_regressions=0, joint_hypotheses=64368, new_hypotheses=3392, robust_candidate_gate_passed=selected,
        independent_confirmation=False, full_contest_compliance_certified=False))
    print(json.dumps(dict(complete=True, robust_candidates=selected)), flush=True)

from quant import timefolio_heatmap_capacity_evaluation as previous
from quant.timefolio_heatmap_epoch_package import ROOT as TRAIN
from quant.timefolio_heatmap_epoch_lab import cases

PRIOR=previous.DEST
DEST=TRAIN.with_name('20260929_epoch_evaluation_v1')
CONFIGS={c['id']:c for c in cases()}
TRAINED=[f'epo_{c}_trained_{m}' for c in CONFIGS for m in MEMBERS]
UNTRAINED=[f'epo_{kind}_untrained_{m}' for kind in ['cnn','mlp'] for m in MEMBERS]
LEGACY=[];MODELS=TRAINED+UNTRAINED+['online_nonimage']


def model_info(model):
    if model=='online_nonimage':return dict(legacy_info(model),encoding='none',geometry='none',condition='none')
    if model not in TRAINED+UNTRAINED:raise ValueError('Unregistered model')
    body,objective,member=model.rsplit('_',2);geometry=body[4:]
    if objective=='untrained':kind=geometry;condition='none'
    else:kind,condition=geometry.split('_',1)
    return dict(architecture=kind,encoding='history',geometry=geometry,condition=condition,member=member,
        objective='epoch_lab' if objective=='trained' else 'untrained',target='absolute' if objective=='trained' else 'none')


def account_id(model,case,buffer,ceiling):
    if model not in MODELS or case not in {c['id'] for c in CASES} or buffer not in BUFFERS or ceiling not in MODES:
        raise ValueError('Unregistered account')
    return f'{model}__{case}__buffer{buffer}__band5bp__sector_{ceiling}'


def legacy_path(model,case,buffer,ceiling):
    # The locked-share rule is new; every account is recomputed.
    return None


def comparison_ids(model,case,buffer,ceiling):
    if model not in TRAINED:raise ValueError('Trained model required')
    info=model_info(model);kind=info['architecture'];condition=info['condition'];member=info['member']
    other='mlp' if kind=='cnn' else 'cnn';key=lambda name:account_id(name,case,buffer,ceiling)
    result=[('cash',None),('own_architecture_untrained',key(f'epo_{kind}_untrained_{member}')),
        ('online',key('online_nonimage')),('matched_other_architecture',key(f'epo_{other}_{condition}_trained_{member}'))]
    if condition!='selected6':result.append(('same_architecture_selected6',key(f'epo_{kind}_selected6_trained_{member}')))
    if ceiling=='formula':result.append(('same_model_research20',account_id(model,case,buffer,'research20')))
    return result


def new_family():
    result=[(account_id(m,c['id'],b,s),label,reference) for m in TRAINED for c in CASES for b in BUFFERS for s in MODES
        for label,reference in comparison_ids(m,c['id'],b,s)]
    assert len(result)==len({(k,c) for k,c,_ in result})==3392
    return result


def prior_family():
    family,differences,returns,_=previous.prior_family()
    for key,comp,reference in previous.new_family():
        family.append((key,comp,'capacity_lab'))
        differences.append(returns(PRIOR,key)-(returns(PRIOR,reference) if reference else 0.))
    stored=pd.read_csv(PRIOR/'joint_bootstrap.csv');means=stored[stored.block==5].set_index(['id','comparator','origin'])['mean']
    assert set(family)==set(means.index) and len(family)==60976
    error=max(abs(float(np.mean(value))-means.loc[key]) for key,value in zip(family,differences))
    assert error<1e-14
    return family,differences,returns,error


def candidate_ids(frame,stats):
    trained=frame[frame.objective=='epoch_lab'];selected=[]
    for row in trained[trained.member=='ensemble'].to_dict('records'):
        group=trained[(trained.geometry==row['geometry'])&(trained.case==row['case'])&
            (trained.buffer==row['buffer'])&(trained.ceiling==row['ceiling'])]
        infer=stats[(stats.id==row['id'])&(stats.origin=='epoch_lab')]
        expected={(c,b) for c,_ in comparison_ids(row['model'],row['case'],row['buffer'],row['ceiling']) for b in [5,10]}
        assert len(group)==4 and set(group.member)==set(MEMBERS)
        assert len(infer)==len(expected) and set(infer[['comparator','block']].itertuples(index=False,name=None))==expected
        stable=((group['return']>0)&(group.mdd>-.2)&(group.positive_blocks>=2)&(~group.four_week_turnover_stop)).all()
        if stable and (infer.adjusted_p<.025).all() and (infer.simultaneous_lower95>0).all():selected.append(row['id'])
    return selected


def freeze():
    verify_package(TRAIN/'package');plan=json.loads((TRAIN/'package/plan.json').read_text())
    assert plan['cases']==cases() and plan['fits']==270 and len(plan['jobs'])==30
    parent=json.loads((PRIOR/'protocol.json').read_text());check_hashes(parent['hashes'])
    assert parent['plan']['joint_hypotheses']==60976
    geometry=TRAIN.with_name('20260929_geometry_lab_evaluation_v1')
    files=[Path(__file__),TRAIN/'package/manifest.json',TRAIN/'package/plan.json',TRAIN/'provenance.json',
        PRIOR/'protocol.json',geometry/'validation_complete.json']
    files += [Path(__file__).with_name('timefolio_heatmap_'+n+'.py') for n in
        ['epoch_lab','epoch_worker','epoch_package','lab_models','lab_queue','lab_pod','capacity_evaluation',
         'gpu_merge','gpu_worker','locked_replay','locked_targets','bounded_bootstrap','consensus',
         'retention_replay','snapshot','stock_limits','planned_audit','action_amendment','week_boundaries',
         'walkforward_eval','walkforward','study','data']]
    files += [Path(__file__).resolve().parents[1]/'tests'/n for n in ['test_timefolio_epoch_lab.py',
        'test_timefolio_epoch_evaluation.py','test_timefolio_bounded_bootstrap.py']]
    spec=dict(plan=plan['evaluation'],training=parent['training'],models=MODELS,cases=CASES,buffers=BUFFERS,
        execution=dict(locked_repair=True),bootstrap_scratch_ceiling_bytes=268435456,
        hashes={str(p.resolve()):digest(p) for p in files},
        prerequisite='Capacity cohort main review required before account evaluation; GPU training independently registered.',
        scope='Repeated development; every fixed duration, comparator and seed retained. Epoch selection diagnostics are not out-of-sample skill evidence.')
    DEST.mkdir(exist_ok=True);path=DEST/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==spec
    else:atomic_json(path,spec)
    assert len(new_family())==3392
    atomic_json(DEST/'preflight.json',dict(prior_hypotheses=60976,joint_hypotheses=64368,accounts=784,
        training_ready=True,prior_main_review_required_before_account_evaluation=True))
    return spec


def forecasts(panel,ix,ci,di):
    plan=json.loads((TRAIN/'package/plan.json').read_text());sha=digest(TRAIN/'package/manifest.json')
    matrices={};reviews=[];checkpoints=[];initial=[];canonical={};regressions=[];evidence={}
    original=TRAIN.with_name('20260929_geometry_lab_v2')
    source_plan=json.loads((original/'package/plan.json').read_text())
    source_sha=digest(original/'package/manifest.json')
    source_checked=set()
    x=torch.from_numpy(np.array(np.load(TRAIN/'package/images_history.npy',mmap_mode='r')[:,None],copy=True))
    for geometry,cfg in CONFIGS.items():
        masks=np.load(TRAIN/'package/labels_masks.npz')
        assert np.array_equal(di,masks['di']) and len(ix['dates'])==255
        for seed in [17,29,43]:
            name=f'{geometry}_seed{seed}';root=TRAIN/'recovered/jobs'/name
            reviews.append(validate_job(root,plan,sha,name));model_root=root/'models'/name
            for fold in plan['folds']:
                month=fold['month'];record=json.loads((model_root/(month+'.json')).read_text())
                assert record['fold']==fold and record['boundaries']==plan['boundaries'][month]
                assert record['config']['id']==name and record['config']['seed']==seed
                tr,va,rf=[masks[month+'_'+k] for k in ['tr','va','rf']]
                assert max(di[tr]+10)<min(di[va]) and max(di[rf]+10)<fold['start']
                assert record['refit_rows']==record['refit_fit']['training_rows']==int(rf.sum())
                assert record['selection_rule']==cfg['epoch_rule'] and record['refit_fit']['validation_rows']==0
                assert record['refit_epochs']==record['refit_fit']['best_epoch']==len(record['refit_fit']['history'])
                assert record['refit_fit']['training_dates']==len(np.unique(di[rf]))
                if cfg['epoch_rule']=='inner_ic':
                    assert record['inner_validation_used']
                    fit=record['fit'];assert (fit['training_rows'],fit['validation_rows'])==(int(tr.sum()),int(va.sum()))
                    assert len(fit['history'])==6 and fit['best_epoch']==1+int(np.argmax([v['inner_ic'] for v in fit['history']]))
                    assert record['refit_epochs']==fit['best_epoch']
                else:
                    assert record['fit'] is None and not record['inner_validation_used']
                    assert record['refit_epochs']==cfg['epochs']
                source_name=('history_base' if cfg['kind']=='cnn' else 'history_mlp')+f'_seed{seed}'
                source_job=original/'recovered/jobs'/source_name
                if source_name not in source_checked:
                    validate_job(source_job,source_plan,source_sha,source_name);source_checked.add(source_name)
                source_model=source_job/'models'/source_name
                source_record_path=source_model/(month+'.json');evidence[str(source_record_path)]=digest(source_record_path)
                source_record=json.loads(source_record_path.read_text())
                if cfg['epoch_rule']=='inner_ic':assert record['fit']==source_record['fit']
                match=cfg['epoch_rule']=='inner_ic' or cfg['epochs']==source_record['fit']['best_epoch']
                if match:
                    paths=[source_model/(month+suffix) for suffix in ['.pt','.pred.npy']]
                    for source_path in paths:evidence[str(source_path)]=digest(source_path)
                    before=torch.load(paths[0],map_location='cpu',weights_only=True)['state_dict']
                    after=torch.load(model_root/(month+'.pt'),map_location='cpu',weights_only=True)['state_dict']
                    assert set(before)==set(after) and all(torch.equal(v,after[k]) for k,v in before.items())
                    assert np.array_equal(np.load(paths[1]),np.load(model_root/(month+'.pred.npy')),equal_nan=True)
                regressions.append(dict(model=name,month=month,source_model=source_name,
                    selected_source_epoch=source_record['fit']['best_epoch'],refit_epochs=record['refit_epochs'],
                    expected_to_match=match,weights_and_predictions_exact=True if match else None))
            model=LabNet(cfg);saved=torch.load(model_root/'202609.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(saved['state_dict']);expected=np.load(model_root/'202609.pred.npy');ids=np.flatnonzero(np.isfinite(expected))
            actual=device_predict(model,x,ids);np.testing.assert_allclose(actual,expected[ids],atol=1e-5,rtol=1e-4)
            checkpoints.append(dict(model=name,rows=len(ids),maximum_error=float(np.max(np.abs(actual-expected[ids])))))
            matrices[f'epo_{geometry}_trained_seed{seed}']=score_matrix(merge_months(model_root,di,ix['dates']),ci,di,panel['close'].shape)
            torch.manual_seed(seed);model=LabNet(cfg);saved=torch.load(root/'untrained.pt',map_location='cpu',weights_only=True)
            assert set(model.state_dict())==set(saved)
            for k,v in model.state_dict().items():assert torch.equal(v,saved[k])
            values=np.load(root/'untrained.pred.npy');assert values.shape==(len(di),) and np.isfinite(values).all()
            key=cfg['kind'],seed
            if key not in canonical:
                ids=np.unique(np.linspace(0,len(di)-1,384,dtype=int));actual=device_predict(model,x,ids)
                np.testing.assert_allclose(actual,values[ids],atol=1e-5,rtol=1e-4)
                canonical[key]=values.copy()
                matrices[f"opt_{cfg['kind']}_untrained_seed{seed}"]=score_matrix(values,ci,di,panel['close'].shape)
            else:assert np.array_equal(values,canonical[key])
            initial.append(dict(model=name,rows=len(di),original_weights_exact=True,canonical_predictions_exact=True,optimizer_steps=0))
        prefix=f'epo_{geometry}_trained_';members={prefix+m:matrices[prefix+m] for m in MEMBERS[:3]}
        matrices[prefix+'ensemble']=rank_consensus(members,{m:cfg['kind'] for m in members},panel['eligible'],'mean')
    for kind in ['cnn','mlp']:
        prefix=f'epo_{kind}_untrained_';members={prefix+m:matrices[prefix+m] for m in MEMBERS[:3]}
        matrices[prefix+'ensemble']=rank_consensus(members,{m:kind for m in members},panel['eligible'],'mean')
    assert len({r['gpu'] for r in reviews})==1
    matrices['online_nonimage']=np.load(RAW_SOURCE/'scores/online_nonimage.npy')
    assert set(matrices)==set(MODELS) and len(checkpoints)==len(initial)==30
    atomic_json(DEST/'checkpoint_reproduction.json',checkpoints);atomic_json(DEST/'untrained_reproduction.json',initial)
    atomic_json(DEST/'gpu_runtime_review.json',reviews)
    assert len(regressions)==270 and len(source_checked)==6
    check_hashes(evidence)
    atomic_json(DEST/'epoch_regression.json',regressions)
    atomic_json(DEST/'epoch_source_hashes.json',evidence)
    return matrices


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run'])
    if parser.parse_args().action=='freeze':freeze()
    else:run()

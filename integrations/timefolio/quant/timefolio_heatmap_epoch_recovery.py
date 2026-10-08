"""Recover the registered epoch study's untrained-score namespace typo.

The original frozen evaluator remains intact. Only forecasts() is replaced;
all training, account execution, regression assertions and inference are reused.
"""
from quant.timefolio_heatmap_epoch_evaluation import *
from quant import timefolio_heatmap_epoch_evaluation as original

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
                matrices[f"epo_{cfg['kind']}_untrained_seed{seed}"]=score_matrix(values,ci,di,panel['close'].shape)
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

def run():
    amendment = DEST / 'namespace_recovery_amendment.json'
    record = json.loads(amendment.read_text())
    check_hashes(record['hashes'])
    assert record['status'] == 'registered_before_account_recovery'
    assert record['changed_expression'] == "opt_ -> epo_ for canonical untrained matrix keys only"
    original.forecasts = forecasts
    original.run()
    check_hashes(record['hashes'])


if __name__ == '__main__':
    run()

import pandas as pd
from quant import timefolio_heatmap_context_spacing_evaluation as study


def test_complete_grid_controls_and_joint_family():
    ids={study.account_id(m,c['id'],b,s) for m in study.MODELS for c in study.CASES
         for b in study.BUFFERS for s in study.MODES}
    assert len(study.MODELS)==97 and len(ids)==1552
    family=study.new_family()
    assert len(family)==len({(key,comp) for key,comp,_ in family})==4352
    assert all(key in ids and (ref is None or ref in ids) for key,_,ref in family)
    assert all(key!=ref for key,_,ref in family)
    legacy=sum(study.legacy_path(m,c['id'],b,s) is not None for m in study.MODELS
               for c in study.CASES for b in study.BUFFERS for s in study.MODES)
    assert legacy==528
    for model in study.TRAINED:
        cfg=study.CONFIGS[study.model_info(model)['geometry']]
        comp=dict(study.comparison_ids(model,study.CASES[0]['id'],12,'formula'))
        assert ('same_mask_stride1' in comp)==(cfg['stride']!=1)
        assert ('same_stride_constant_mask' in comp)==(cfg['availability_mask']=='available')


def test_all_seeds_all_comparators_and_both_blocks_required():
    rows=[]
    for model in study.TRAINED:
        for case in study.CASES:
            for buffer in study.BUFFERS:
                for ceiling in study.MODES:
                    rows.append(dict(id=study.account_id(model,case['id'],buffer,ceiling),model=model,
                        **study.model_info(model),case=case['id'],buffer=buffer,ceiling=ceiling,
                        **{'return':.1},mdd=-.05,positive_blocks=3,four_week_turnover_stop=False))
    frame=pd.DataFrame(rows)
    stats=pd.DataFrame([dict(id=key,comparator=comp,origin='context_spacing_lab',block=block,
        adjusted_p=.001,simultaneous_lower95=.0001) for key,comp,_ in study.new_family() for block in [5,10]])
    selected=study.candidate_ids(frame,stats);assert len(selected)==192
    target=selected[0];seed=target.replace('_ensemble__','_seed17__')
    frame.loc[frame.id==seed,'four_week_turnover_stop']=True
    assert target not in study.candidate_ids(frame,stats)
    frame.loc[frame.id==seed,'four_week_turnover_stop']=False
    mask=(stats.id==target)&(stats.block==10)&(stats.comparator=='cash')
    stats.loc[mask,'adjusted_p']=.025
    assert target not in study.candidate_ids(frame,stats)
    stats.loc[mask,'adjusted_p']=.001;stats.loc[mask,'simultaneous_lower95']=0
    assert target not in study.candidate_ids(frame,stats)

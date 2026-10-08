import pandas as pd
import pytest
from quant.timefolio_heatmap_fleet_accounts import policy_grid
from quant.timefolio_heatmap_fleet_selector import identity, MEMBERS, rules
from quant.timefolio_heatmap_fleet_selector_inference import candidates,ORIGIN


def test_selection_must_improve_on_fixed_blends_and_each_component_with_stable_seeds():
    pools={'three':list('abc'),'nine':list('abcdefghi')};rows=[];tests=[]
    for pool,components in pools.items():
        for rule in rules(components):
            for policy in policy_grid():
                for member in MEMBERS:
                    key=identity(pool,rule['id'],'trained',member,policy)
                    rows.append(dict(id=key,**{'return':.1},mdd=-.1,positive_blocks=2,four_week_turnover_stop=False))
                    if member!='ensemble':continue
                    for label in ['cash','matched_untrained_selector','online_nonimage','fixed_mean_blend']+['component_'+c for c in components]:
                        for block in [5,10]:tests.append(dict(id=key,comparator=label,origin=ORIGIN,block=block,adjusted_p=.01,simultaneous_lower95=.001))
    frame,stats=pd.DataFrame(rows),pd.DataFrame(tests)
    selected,basic=candidates(pools,frame,stats);assert len(selected)==len(basic)==96
    key=selected[0]
    changed=stats.copy();changed.loc[(changed.id==key)&(changed.comparator=='fixed_mean_blend'),'adjusted_p']=.025
    assert key not in candidates(pools,frame,changed)[0]
    changed=frame.copy();changed.loc[changed.id==key.replace('_ensemble__','_seed43__'),'four_week_turnover_stop']=True
    assert key not in candidates(pools,changed,stats)[0]
    with pytest.raises(AssertionError):candidates(pools,frame,stats[stats.comparator!='component_i'])

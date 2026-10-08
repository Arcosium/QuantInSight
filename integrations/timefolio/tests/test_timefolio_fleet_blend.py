import numpy as np
import pytest
from quant.timefolio_heatmap_fleet_blend import combine, source_names, hypotheses, units


def test_member_ensemble_uses_each_seed_directly_without_double_ranking_saved_ensembles():
    names=source_names(['a','b','c'],'trained','ensemble')
    assert len(names)==9 and len(set(names))==9 and not any(name.endswith('_ensemble') for name in names)
    assert source_names(['a','b'],'untrained','seed29')==['flt_a_untrained_seed29','flt_b_untrained_seed29']
    with pytest.raises(ValueError):source_names(['a','a'],'trained','ensemble')


@pytest.mark.parametrize('method',['mean','median'])
def test_consensus_is_causal_and_ignores_component_scale(method):
    arrays={'a':np.array([[1.,2.,3.],[2.,1.,4.],[3.,3.,1.],[4.,4.,2.]]),
            'b':np.array([[4.,1.,2.],[3.,2.,1.],[2.,4.,3.],[1.,3.,4.]])}
    eligible=np.ones((4,3),bool)
    before=combine(arrays,eligible,method)
    changed={k:v.copy() for k,v in arrays.items()};changed['a']=changed['a']*7+29
    np.testing.assert_array_equal(combine(changed,eligible,method),before)
    changed['a'][:,2]=[-9,100,50,7];eligible[:,2]=False
    np.testing.assert_array_equal(combine(changed,eligible,method)[:,:2],before[:,:2])
    changed['a'][0,0]=np.nan
    with pytest.raises(ValueError,match='coverage'):combine(changed,eligible,method)


def test_family_retains_every_component_and_untrained_control():
    pools={'small':['a','b','c'],'large':['a','b','c','d']}
    rows=hypotheses(pools);assert len(units(pools))==16
    assert len(rows)==1920 and len({(r[0],r[1]) for r in rows})==1920
    keys={r[0] for r in rows}
    assert len(keys)==512
    assert all(ref in keys for key,label,ref in rows if label=='matched_untrained_blend')
    labels={label for key,label,ref in rows if key.startswith('blend_large_mean_trained_seed17__')}
    assert labels=={'cash','matched_untrained_blend','online_nonimage','component_a','component_b','component_c','component_d'}

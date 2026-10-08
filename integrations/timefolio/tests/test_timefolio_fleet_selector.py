import numpy as np
import pytest
from quant.timefolio_heatmap_fleet_selector import monthly_scores, hypotheses, rules
from quant.timefolio_heatmap_fleet_blend import source_names, combine


def test_selection_excludes_origin_returns_and_carries_only_causal_signals():
    components=['a','b','c'];names=source_names(components,'trained','seed17')
    dates=np.array(['20260130','20260202','20260203','20260302','20260303'])
    scores={name:np.tile(np.roll(np.arange(4.),i)[:,None],(1,len(dates))) for i,name in enumerate(names)}
    eligible=np.ones((4,len(dates)),bool)
    shadow_dates=np.array([f'202601{i:02d}' for i in range(1,31)]+['20260202','20260203','20260302'])
    shadows={c:np.tile([-.001,.003],17)[:len(shadow_dates)]*(i+1)+(.004 if c=='a' else 0.) for i,c in enumerate(components)}
    before, choices=monthly_scores(scores,eligible,dates,components,'trained','seed17',shadows,shadow_dates,
                                  lookback=20,count=1,start='20260201')
    assert choices[0]['chosen']==['a'] and choices[0]['last_observation']=='20260130'
    changed={k:v.copy() for k,v in shadows.items()}
    for v in changed.values():v[shadow_dates>='20260202']=-999.
    changed_scores={k:v.copy() for k,v in scores.items()}
    for v in changed_scores.values():v[:,2:]=42.
    after,new_choices=monthly_scores(changed_scores,eligible,dates,components,'trained','seed17',changed,shadow_dates,
                                    lookback=20,count=1,start='20260201')
    np.testing.assert_array_equal(before[:,:2],after[:,:2])
    assert choices[0]==new_choices[0]
    assert np.isnan(before[:,-1]).all()


def test_warmup_matches_fixed_direct_seed_blend_and_duplicate_rules_are_absent():
    components=['a','b','c'];names=source_names(components,'trained','ensemble')
    dates=np.array(['20251230','20260102','20260105']);rng=np.random.default_rng(5)
    scores={name:rng.normal(size=(6,3)) for name in names};eligible=np.ones((6,3),bool)
    shadows={c:np.array([.01]) for c in components}
    actual,choices=monthly_scores(scores,eligible,dates,components,'trained','ensemble',shadows,['20260102'],lookback=20,count=1)
    expected=combine(scores,eligible,'mean')
    np.testing.assert_array_equal(actual[:,:2],expected[:,:2])
    assert choices[0]['observations']==0 and choices[0]['chosen']==components
    assert [r['count'] for r in rules(components)]==[1,1]
    scores[names[0]][0,0]=np.nan
    with pytest.raises(ValueError,match='coverage'):
        monthly_scores(scores,eligible,dates,components,'trained','ensemble',shadows,['20260102'],lookback=20,count=1)


def test_new_selectors_keep_every_fixed_component_and_previous_blend_comparison():
    pools={str(i):['a','b','c'] for i in range(4)};pools['broad']=list('abcdefghi')
    family=hypotheses(pools)
    assert len(family)==7680 and len({key for key,_,_ in family})==1536
    assert sum(label=='fixed_mean_blend' for _,label,_ in family)==768
    assert sum(label.startswith('component_') for _,label,_ in family)==3840

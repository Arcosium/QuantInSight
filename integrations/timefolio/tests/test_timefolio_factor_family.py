from quant.timefolio_heatmap_factor_residual import hypotheses,VARIANTS


def test_every_variant_seed_and_shared_control_is_retained():
    rows=hypotheses(['case'+str(i) for i in range(9)])
    assert len(rows)==11584 and len({k for k,_,_ in rows})==4672
    assert all(k.startswith('factor_') for k,_,_ in rows)
    assert sum(c=='same_factor_nonimage' for _,c,_ in rows)==2304
    assert sum(c=='matching_untrained_factor' for _,c,_ in rows)==2304
    assert sum(c=='unchanged_original' for _,c,_ in rows)==2304
    assert all(any(k.startswith('factor_'+v+'_') for k,_,_ in rows) for v in VARIANTS)

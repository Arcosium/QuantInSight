import json

import numpy as np
import pytest

from quant.timefolio_cnn_dataset import sha
from quant.timefolio_cnn_dynamic_sector import cohort_sector_caps,derive_panel


def test_sector_weights_follow_each_dates_cap_and_can_have_limit_above_one():
    cap=np.array([[90.,10.],[5.,40.],[5.,50.]])
    limits,coverage=cohort_sector_caps(cap,np.array([45,20,20]))
    np.testing.assert_allclose(limits,[[1.8,.2],[.2,1.8],[.2,1.8]])
    np.testing.assert_allclose(coverage['sector_weights'].sum(0),1.)
    np.testing.assert_array_equal(coverage['observed_cap_codes'],[3,3])


def test_unknown_sector_stays_in_denominator_and_missing_caps_are_not_backfilled():
    cap=np.array([[20.,np.nan],[80.,np.nan],[np.nan,np.nan]])
    limits,coverage=cohort_sector_caps(cap,np.array([45,-1,20]))
    np.testing.assert_allclose(limits,[[.4,.1],[.1,.1],[.1,.1]])
    np.testing.assert_allclose(coverage['unknown_sector_weight'],[.8,0.])
    assert coverage['no_observed_cap'].tolist()==[False,True]
    later=cap.copy();later[:,1]=[900.,10.,50.]
    np.testing.assert_equal(cohort_sector_caps(later,np.array([45,-1,20]))[0][:,0],limits[:,0])


def test_bad_classifications_axes_and_infinite_or_negative_caps_are_rejected():
    cap=np.ones((2,3),dtype=float)
    for caps,sectors in [(cap,np.array([45])),(cap,np.array([45,99])),
                         (cap,np.array([45.,20.])),(cap*-1,np.array([45,20])),
                         (cap*np.inf,np.array([45,20]))]:
        with pytest.raises(ValueError):
            cohort_sector_caps(caps,sectors)


def test_panel_derivation_changes_only_sector_caps_and_preserves_price_and_membership(tmp_path):
    parent=tmp_path/'parent';parent.mkdir();output=tmp_path/'derived'
    arrays=dict(market_cap=np.array([[90.,10.],[10.,90.]]),sector=np.array([45,20]),
        sector_cap=np.full((2,2),.1),close=np.array([[9.,1.],[1.,9.]]),
        eligible=np.array([[True,False],[False,True]]))
    np.savez(parent/'panel.npz',**arrays)
    (parent/'index.json').write_text(json.dumps(dict(codes=['000001','000002'],dates=['20240102','20240103'])))
    (parent/'provenance.json').write_text('[]')
    (parent/'receipt.json').write_text(json.dumps(dict(dataset_manifest_sha256='example',
        limitations=['Uniform10% sector caps','historical shares incomplete'],
        artifacts={p.name:sha(p) for p in parent.iterdir()})))
    proof=derive_panel(parent,output)
    with np.load(output/'panel.npz',allow_pickle=False) as z:
        for name,value in arrays.items():
            if name!='sector_cap':np.testing.assert_equal(z[name],value)
        np.testing.assert_allclose(z['sector_cap'],[[1.8,.2],[.2,1.8]])
    assert proof['contest_certified'] is False and proof['historical_sector_weights_certified'] is False
    assert proof['minimum_observed_cap_codes']==2
    assert proof['limitations'][0]=='historical shares incomplete'
    with pytest.raises(ValueError):derive_panel(parent,output)

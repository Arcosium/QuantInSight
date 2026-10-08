"""Date-varying sector-limit sensitivity from the covered cohort's cap proxy.

This is not official historical market weights. It preserves the parent panel's
share-count and classification limitations and reports the represented cohort.
"""
from pathlib import Path
import json
import shutil

import numpy as np

from quant.timefolio_cnn_dataset import sha

SECTORS=np.array([10,15,20,25,30,35,40,45,50,55,60])


def cohort_sector_caps(market_cap,sectors):
    """Use only each column's observed cap; never fill from a later column.

    The denominator includes positive caps with unknown sector, so missing
    classification cannot inflate the weights of known sectors. No liquidity
    or strategy-eligibility filter determines market composition.
    """
    cap,sector=map(np.asarray,[market_cap,sectors])
    if (cap.ndim!=2 or sector.shape!=(cap.shape[0],)
            or not np.issubdtype(cap.dtype,np.floating)
            or not np.issubdtype(sector.dtype,np.integer)
            or not np.isin(sector,np.r_[SECTORS,-1]).all()
            or np.isinf(cap).any() or np.any(cap<0)):
        raise ValueError('Matching positive-or-missing caps and known GICS sectors required')
    known=np.isfinite(cap)&(cap>0)
    observed=np.where(known,cap,0.)
    total=observed.sum(axis=0)
    if not np.isfinite(total).all():
        raise ValueError('Nonfinite aggregate capitalisation')
    weights=np.array([np.divide(observed[sector==code].sum(axis=0),total,
        out=np.zeros(cap.shape[1]),where=total>0) for code in SECTORS])
    limits=np.full(cap.shape,.1)
    for code,weight in zip(SECTORS,weights):
        limits[sector==code]=np.maximum(.1,2*weight)
    unknown=np.divide(observed[sector==-1].sum(axis=0),total,
        out=np.zeros(cap.shape[1]),where=total>0)
    return limits,dict(sectors=SECTORS.copy(),sector_weights=weights,
        observed_total_cap=total,observed_cap_codes=known.sum(axis=0),
        unknown_sector_weight=unknown,no_observed_cap=total==0)


def derive_panel(parent,output):
    parent,output=map(Path,[parent,output])
    if output.exists():
        raise ValueError('Use a fresh date-varying sector sensitivity panel')
    proof=json.loads((parent/'receipt.json').read_text())
    for name,digest in proof['artifacts'].items():
        path=(parent/name).resolve()
        if not path.is_relative_to(parent.resolve()) or sha(path)!=digest:
            raise ValueError('Parent panel fingerprint changed')
    index=json.loads((parent/'index.json').read_text())
    if index['dates']!=sorted(set(index['dates'])) or index['dates'][-1]>'20260923':
        raise ValueError('Ordered development dates within the reserved cutoff required')
    with np.load(parent/'panel.npz',allow_pickle=False) as z:
        panel={name:z[name] for name in z.files}
    if panel['market_cap'].shape!=(len(index['codes']),len(index['dates'])):
        raise ValueError('Capitalisation axes differ from account axes')
    panel['sector_cap'],coverage=cohort_sector_caps(panel['market_cap'],panel['sector'])
    output.mkdir(parents=True)
    np.savez(output/'panel.npz',**panel)
    np.savez(output/'sector_coverage.npz',**coverage)
    for name in ['index.json','provenance.json']:
        shutil.copy2(parent/name,output/name)
    proof['limitations']=[v for v in proof['limitations'] if not v.startswith('Uniform10% sector caps')]
    proof['limitations'].append('Date-varying weights use only this covered cohort and the parent anchored capitalisation proxy, including illiquid names. Missing issuers, later issuance and historical classification changes are not repaired. These are not official historical market weights.')
    proof.update(parent_receipt_sha256=sha(parent/'receipt.json'),
        sector_cap_method='max(10%,2*same-date observed cohort sector cap/observed cohort total cap)',
        sector_limit_signal_timing='Replay uses the previous close column.',
        market_universe_is_covered_cohort_only=True,historical_sector_weights_certified=False,
        minimum_observed_cap_codes=int(coverage['observed_cap_codes'].min()),
        maximum_observed_cap_codes=int(coverage['observed_cap_codes'].max()),
        no_observed_cap_dates=int(coverage['no_observed_cap'].sum()),
        source_sha256=sha(__file__),artifacts={p.name:sha(p) for p in output.iterdir()},
        contest_certified=False)
    (output/'receipt.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    return proof

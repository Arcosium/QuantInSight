"""Typed strategy genomes, Pareto survival, directed mutation and anti-repeat search.

Adapted conceptually from GenomicWQB's genome_models.py and alpha_search.py.
No WorldQuant API, cloud model, credentials or broker integration is imported.
"""
import hashlib
import json
import random
from .metrics import pareto_front

MODEL_GENES=('family','window','target','l2','features')
DOMAINS={
    'family':('linear31','quadratic62','nn_consensus'),
    'window':(0,255,510),
    'target':('risk20','net20','net5'),
    'l2':(.001,.003,.01),
    'features':('all','price'),
    'top_n':(8,12,16),
    'stock_weight':(.05,.075,.1,.125),
    'rebalance':(5,10),
    'band':(.003,.005,.01,.02),
    'gross_low':(.2,.4),
    'gross_high':(.6,.8),
}
DEFAULT=dict(family='quadratic62',window=0,target='risk20',l2=.001,features='all',
             top_n=12,stock_weight=.1,rebalance=5,band=.005,gross_low=.4,gross_high=.6)


def normalize(genome):
    if not isinstance(genome,dict) or set(genome)!=set(DOMAINS):
        raise ValueError('Unknown or missing strategy genes')
    g=dict(genome)
    for k,values in DOMAINS.items():
        if g[k] not in values or isinstance(g[k],bool):raise ValueError('Invalid gene: '+k)
    if g['family']=='nn_consensus':
        g.update(window=0,target='risk20',l2=.001,features='all')
    if g['gross_low']>g['gross_high']:raise ValueError('Invalid exposure range')
    return g


def canonical(g):
    return json.dumps(normalize(g),sort_keys=True,separators=(',',':'))


def fingerprint(g):
    return hashlib.sha256(canonical(g).encode()).hexdigest()[:20]


def model_fingerprint(g,version):
    g=normalize(g)
    return hashlib.sha256((json.dumps({k:g[k] for k in MODEL_GENES},sort_keys=True)+version).encode()).hexdigest()[:20]


def seeds():
    return [dict(DEFAULT),dict(DEFAULT,family='nn_consensus'),dict(DEFAULT,family='linear31'),
            dict(DEFAULT,window=255),dict(DEFAULT,family='linear31',window=255)]


def generate(results,seen,count,generation):
    rng=random.Random(2701+generation)
    eligible=[r for r in results if r.get('rule_screen_pass') and isinstance(r.get('genome'),dict)]
    front=pareto_front(eligible)
    survivors=front or sorted(eligible,key=lambda r:r['net_return'],reverse=True)[:8]
    parents=survivors[:24]
    generated=[]
    candidates=[(g,[], 'seed') for g in seeds()]
    for attempt in range(2000):
        if len(generated)>=count:break
        if attempt<len(candidates):
            child,ids,operator=candidates[attempt]
        elif not parents or attempt%4==0:
            child={k:rng.choice(v) for k,v in DOMAINS.items()}
            ids,operator=[],'explore'
        elif len(parents)>1 and attempt%4==1:
            a,b=rng.sample(parents,2)
            child={k:rng.choice([a['genome'][k],b['genome'][k]]) for k in DOMAINS}
            ids,operator=[a['job_id'],b['job_id']],'recombine'
        else:
            parent=rng.choice(parents)
            child=dict(parent['genome'])
            if parent.get('negative_months',0)>10 and attempt%3==0:
                axis=rng.choice(['gross_low','window','rebalance','band'])
                operator='refine'
            else:
                axis=rng.choice(list(DOMAINS))
                operator='mutate'
            child[axis]=rng.choice([v for v in DOMAINS[axis] if v!=child[axis]])
            ids=[parent['job_id']]
            if child['family']!='nn_consensus' and child['features']=='all' and attempt%7==0:
                child['features']='price'
                operator='simplify'
        child=normalize(child)
        identity=fingerprint(child)
        if identity in seen:continue
        seen.add(identity)
        generated.append(dict(id=identity,genome=child,parents=ids,operator=operator,generation=generation))
    return generated


"""Persistent four-market, locally proposed evolutionary research campaigns."""
import hashlib
import json
import os
import random
import time
from . import research, store

MARKETS=('kr','us','crypto','timefolio')
PROVIDER='local-evolution-v2'
THEMES={'kr':'한국주식 전체 보유 유니버스의 인공지능 학습 투자',
        'us':'미국주식 전체 보유 유니버스의 인공지능 학습 투자',
        'crypto':'크립토 전체 보유 유니버스의 인공지능 학습 투자',
        'timefolio':'타임폴리오 제약을 반영한 롱온리 인공지능 학습 투자'}


def enabled(market):
    return bool(store.setting('labs_v2_enabled',False) and store.setting('lab_enabled_'+market,False))


def state(market):
    return store.setting('lab_state_'+market,{})


def control(market,on):
    if market not in MARKETS:raise ValueError('Unknown laboratory')
    store.set_setting('lab_enabled_'+market,bool(on))
    store.event('lab_control',dict(market=market,message='연속 연구 시작' if on else '연구 중지 · 실행 중 실험은 안전하게 종료'))


def status(uid):
    research.initialize();markets={}
    with store.connect() as db:
        for market in MARKETS:
            counts=dict((r['status'],r['n']) for r in db.execute('SELECT status,count(*) n FROM alpha_candidates WHERE user_id=? AND market=? AND provider=? GROUP BY status',(uid,market,PROVIDER)))
            s=state(market)
            markets[market]=dict(enabled=enabled(market),generation=s.get('generation',0),
                pending=sum(counts.get(k,0) for k in ('queued','running','waiting_data')),
                running=counts.get('running',0),done=counts.get('done',0),failed=counts.get('failed',0),
                phase=s.get('phase','preparing') if enabled(market) else 'stopped',
                message=s.get('message',''),remaining=None,theme=THEMES[market])
    return dict(enabled=any(m['enabled'] for m in markets.values()),markets=markets,
                pending=sum(m['pending'] for m in markets.values()),done=sum(m['done'] for m in markets.values()),
                failed=sum(m['failed'] for m in markets.values()),remaining=None,evidence='development_only',continuous=True)


def parents(uid,market):
    # Aggregate metrics are safe inputs to the local planner; raw records stay in place.
    with store.connect() as db:
        rows=db.execute("SELECT a.id,a.definition,s.payload FROM alpha_candidates a JOIN strategies s ON s.source=a.result WHERE a.user_id=? AND a.market=? AND a.status='done' AND a.provider=? ORDER BY a.created DESC LIMIT 128",(uid,market,PROVIDER)).fetchall()
    found=[]
    for r in rows:
        p=json.loads(r['payload']);g=json.loads(r['definition'])
        found.append(dict(id=r['id'],genome=g,net_return=p['net_return'],negative_months=p['negative_months'],mdd=p['mdd']))
    # Loss-month/return/MDD trade-offs instead of selecting solely on returns.
    found.sort(key=lambda p:(p['negative_months'],-p['net_return'],abs(p['mdd'])))
    return found[:12]


def offspring(domain,pool,market,generation,count=4):
    seed=int(hashlib.sha256(f'{market}:{generation}'.encode()).hexdigest()[:8],16)
    rng=random.Random(seed);out=[]
    for i in range(count):
        a=rng.choice(pool) if pool else None;b=rng.choice(pool) if pool else None
        genome={k:(rng.choice([a['genome'][k],b['genome'][k]]) if a and b and k in a['genome'] and k in b['genome'] else rng.choice(values)) for k,values in domain.items()}
        for k in rng.sample(list(domain),min(2,len(domain))):genome[k]=rng.choice(domain[k])
        genome['seed']=(seed+i)%(2**31-2)+1
        out.append((genome,[p['id'] for p in (a,b) if p], 'crossover_mutation' if pool else 'exploration'))
    return out


def plan(uid,market):
    """Runs as a bounded child, so a slow local AI cannot block worker heartbeats."""
    from . import providers
    if not enabled(market):return
    domain=research.domains(market);pool=parents(uid,market);previous=state(market)
    generation=int(previous.get('generation',0))+1
    proposals=offspring(domain,pool,market,generation,2)
    system=('당신은 투자모델 연구 설계자입니다. '+THEMES[market]+'. 모든 후보는 실제 모델 학습이 필수입니다. '
            '최근 36개월은 순차 검증하며 미래 정보 금지. 사용 가능한 데이터는 이 시장의 시점 정렬된 OHLCV입니다. '
            'feature_set으로 가격/거래량/위험 필드를 선택합니다. 뉴스·공시는 아직 시점 정렬 미검증이므로 사용하지 마세요. '
            '모델·필드·학습기간·보유기간을 다양하게 조합하세요. JSON {"genomes":[객체,객체]}만 출력. 허용값: '+json.dumps(domain,ensure_ascii=False))
    message='로컬 AI 제안과 교차·변이';ai_ok=False
    try:
        result=providers.generate('local','',os.environ.get('AUTOFOLIO_LOCAL_MODEL','arc-local'),system,
                                  json.dumps(dict(generation=generation,parents=pool),ensure_ascii=False))
        raw=result.get('genomes',[])
        for i,g in enumerate(raw[:2]):
            g=research.normalize(g,market);proposals.append((g,[],'local_ai'));ai_ok=True
    except Exception:
        message='로컬 AI 응답 대기 · 유전 교차·변이로 연구 계속'
    saved=[]
    for g,lineage,operator in proposals:
        if not enabled(market):break
        try:ids=research.save_candidates(uid,market,[g],PROVIDER)
        except (ValueError,TypeError):continue
        for identity in ids:
            with store.connect() as db:
                db.execute('INSERT OR REPLACE INTO lab_lineage VALUES(?,?,?,?,?)',(identity,market,generation,json.dumps(lineage),operator))
        saved.extend(ids)
    previous.update(generation=generation,phase='researching' if saved else 'proposal_retry',message=message,
                    last_proposal=time.time(),retry_after=time.time()+(10 if saved else 60),local_ai=ai_ok)
    store.set_setting('lab_state_'+market,previous)
    store.event('campaign_proposals',dict(market=market,generation=generation,count=len(saved),message=message))


if __name__=='__main__':
    import sys
    plan(int(sys.argv[1]),sys.argv[2])

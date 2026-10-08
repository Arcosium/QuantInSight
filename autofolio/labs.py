"""Persistent four-market, locally proposed evolutionary research campaigns."""
import hashlib
import json
import os
import random
import time
from . import research, store

MARKETS=('kr','us','crypto','timefolio')
PROVIDER='local-evolution-v2'
NEURAL_MODELS=('neural_mlp','residual_mlp','lstm','gru','tcn','transformer')
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
    from .evaluation import selection_metrics
    with store.connect() as db:
        rows=db.execute("SELECT a.id,a.definition,s.payload FROM alpha_candidates a JOIN strategies s ON s.source=a.result WHERE a.user_id=? AND a.market=? AND a.status='done' AND a.provider=? ORDER BY a.created DESC LIMIT 128",(uid,market,PROVIDER)).fetchall()
    found=[]
    for r in rows:
        summary=json.loads(r['payload']);p=selection_metrics(summary);g=json.loads(r['definition'])
        if not p:continue
        found.append(dict(id=r['id'],genome=g,net_return=p['net_return'],negative_months=p['negative_months'],mdd=p['mdd']))
    # Loss-month/return/MDD trade-offs instead of selecting solely on returns.
    found.sort(key=lambda p:(p['negative_months'],-p['net_return'],abs(p['mdd'])))
    return found[:12]


def neural_population(market):
    """Saved attempts advance architecture rotation; quick classical trials do not."""
    with store.connect() as db:
        row=db.execute('''SELECT COALESCE(SUM(status IN ('queued','running','waiting_data')),0),
            COALESCE(SUM(provider=?),0) FROM alpha_candidates WHERE market=?
            AND json_extract(definition,'$.model') IN (?,?,?,?,?,?)''',(PROVIDER,market,*NEURAL_MODELS)).fetchone()
    return row[0],row[1]


def offspring(domain,pool,market,generation,count=4,neural_turn=None):
    seed=int(hashlib.sha256(f'{market}:{generation}'.encode()).hexdigest()[:8],16)
    rng=random.Random(seed);out=[]
    for i in range(count):
        a=rng.choice(pool) if pool else None;b=rng.choice(pool) if pool else None
        genome={k:(rng.choice([a['genome'][k],b['genome'][k]]) if a and b and k in a['genome'] and k in b['genome'] else rng.choice(values)) for k,values in domain.items()}
        for k in rng.sample(list(domain),min(2,len(domain))):genome[k]=rng.choice(domain[k])
        classical=[m for m in domain['model'] if m not in NEURAL_MODELS]
        neural=[m for m in NEURAL_MODELS if m in domain['model']]
        # Reserve alternating proposals for each family. Parent popularity and
        # fast completions must not squeeze slow architectures out of exploration.
        if store.setting('neural_research_enabled',False) and neural and i%2:
            turn=(max(0,generation-1) if neural_turn is None else neural_turn)+(i//2)
            genome['model']=neural[turn%len(neural)]
            if 'model_size' in domain:
                sizes=domain['model_size'];genome['model_size']=sizes[(turn//len(neural))%len(sizes)]
        elif classical:
            genome['model']=classical[(max(0,generation-1)+i//2)%len(classical)]
        genome['seed']=(seed+i)%(2**31-2)+1
        out.append((genome,[p['id'] for p in (a,b) if p], 'crossover_mutation' if pool else 'exploration'))
    return out


def plan(uid,market):
    """Runs as a bounded child, so a slow local AI cannot block worker heartbeats."""
    from .period import using_window,window
    with using_window(*window()):
        return _plan(uid,market)


def _plan(uid,market):
    from . import providers
    if not enabled(market):return
    domain=research.domains(market);pool=parents(uid,market);previous=state(market)
    if 'news_mode' in domain:
        from .feature_sources import news_status
        if not news_status(market).get('ready'):domain=dict(domain,news_mode=['off'])
    generation=int(previous.get('generation',0))+1
    neural_pending,neural_cursor=neural_population(market)
    proposal_domain=dict(domain)
    if neural_pending:proposal_domain['model']=[m for m in domain['model'] if m not in NEURAL_MODELS]
    proposals=offspring(proposal_domain,pool,market,generation,2,neural_turn=neural_cursor)
    system=('당신은 투자모델 연구 설계자입니다. '+THEMES[market]+'. 모든 후보는 실제 모델 학습이 필수입니다. '
            '최근 36개월을 IS 24개월·OS 9개월·ROS 3개월로 나누고 미래 정보 사용을 금지합니다. '
            '부모 선택과 성과 비교에는 OS 결과만 사용하며 ROS 결과는 제안·선발·튜닝에 사용하지 마세요. '
            '사용 가능한 데이터는 이 시장의 시점 정렬된 OHLCV입니다. '
            'feature_set으로 가격/거래량/위험 필드를 선택합니다. 뉴스·공시는 아직 시점 정렬 미검증이므로 사용하지 마세요. '
            '신경망이 허용되면 순환 신경망·합성곱·어텐션과 기존 회귀·트리 모델을 함께 탐색하고 compact/large 크기를 다양하게 선택하세요. '
            '학습기간·재학습주기·학습목표·모델·필드·매수/매도규칙·비중·손절/익절·노출·리밸런싱을 '
            '허용된 필드 안에서 함께 탐색하세요. 평가기간 자체는 최근 36개월로 고정하며 각 구간의 학습에는 이전 정보만 사용합니다. '
            'JSON {"genomes":[객체,객체]}만 출력. 허용값: '+json.dumps(domain,ensure_ascii=False))
    message='로컬 AI 제안과 교차·변이';ai_ok=False
    try:
        result=providers.generate('local','',os.environ.get('AUTOFOLIO_LOCAL_MODEL','arc-local'),system,
                                  json.dumps(dict(generation=generation,parents=pool),ensure_ascii=False))
        raw=result.get('genomes',[])
        for i,g in enumerate(raw[:2]):
            g=research.normalize(g,market)
            if g.get('model') in NEURAL_MODELS and not store.setting('neural_research_enabled',False):continue
            if research.is_neural(g):
                slot=next((j for j,(candidate,_,_) in enumerate(proposals) if research.is_neural(candidate)),None)
                if neural_pending or slot is None:continue
                # Let AI refine the reserved neural recipe while preserving the
                # architecture/size rotation and the one-outstanding-trial cap.
                planned=proposals[slot][0]
                g.update(model=planned['model'],model_size=planned['model_size'],seed=planned['seed'])
                proposals[slot]=(g,[],'local_ai');ai_ok=True
                continue
            proposals.append((g,[],'local_ai'));ai_ok=True
    except Exception:
        message='로컬 AI 응답 대기 · 유전 교차·변이로 연구 계속'
    saved=[]
    for g,lineage,operator in proposals:
        if not enabled(market):break
        if research.is_neural(g) and neural_population(market)[0]:continue
        try:ids=research.save_candidates(uid,market,[g],PROVIDER)
        except (ValueError,TypeError):continue
        for identity in ids:
            with store.connect() as db:
                db.execute('INSERT OR REPLACE INTO lab_lineage VALUES(?,?,?,?,?)',(identity,market,generation,json.dumps(lineage),operator))
        saved.extend(ids)
    previous.update(generation=generation,neural_cursor=neural_population(market)[1],phase='researching' if saved else 'proposal_retry',message=message,
                    last_proposal=time.time(),retry_after=time.time()+(10 if saved else 60),local_ai=ai_ok)
    store.set_setting('lab_state_'+market,previous)
    store.event('campaign_proposals',dict(market=market,generation=generation,count=len(saved),message=message))


if __name__=='__main__':
    import sys
    plan(int(sys.argv[1]),sys.argv[2])

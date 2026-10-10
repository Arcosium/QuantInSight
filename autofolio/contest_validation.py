"""Offline evidence checks; historical screening never authorizes broker orders."""
import math
from collections import defaultdict

VERSION = 2
MISSING = [
    '13회 공식 규정 원문과 적용 기간의 확인된 버전',
    '거래일별 보통주·상장일·시가총액·GICS·시장 업종 비중·매수금지 지정 이력',
    '기업행사·거래정지·호가 및 체결 한도 재현 자료',
]


def assess(case, *, rules=None, prices=None):
    """Recompute checks from the original ledger, ignoring claimed certificates."""
    import pandas as pd
    checks=[]
    def add(code, ok, detail):
        checks.append(dict(code=code,status='passed' if ok else 'failed',detail=detail))
    daily=case.get('daily') or []; trades=case.get('trades') or []
    if not daily:
        return dict(version=VERSION,status='missing',competition_compliance_verified=False,
                    checks=[],missing=['일별 계좌 원장',*MISSING],summary='검증 자료 부족: 일별 계좌 원장')
    try:
        d=pd.DataFrame(daily);d['date']=pd.to_datetime(d.date.astype(str))
        valid=not d.date.duplicated().any() and d.date.is_monotonic_increasing
        add('calendar',valid,'평가 날짜 순서·중복 검사')
        valid_nav=all(math.isfinite(float(x)) and float(x)>0 for x in d.nav)
        add('nav',valid_nav,'계좌 평가액 유한수·양수 검사')
        missing=list(MISSING)
        if rules:
            missing.remove(MISSING[0])
            add('official_rules',True,'보관된 13회 공식 규정 원본 해시 확인')
            if cash_initial:=case.get('initial_cash'):
                add('contest_capital',float(cash_initial)==rules['initial_cash'],'13회 초기 운용금액 10억 원')
        byday=defaultdict(float);hold=defaultdict(float)
        cash=float(case['initial_cash']) if 'initial_cash' in case else None
        if cash is None or 'cash' not in d:missing.append('초기 현금과 일별 현금 원장')
        if cash is not None:add('initial_cash',math.isfinite(cash) and cash>0,'초기 현금 유한수·양수 검사')
        fees_known=all('fee' in t for t in trades)
        if not fees_known:missing.append('체결별 실제 수수료')
        cash_errors=0; invalid=0; oversells=0; grouped=defaultdict(list)
        marks={};position_errors=0;position_checked=0;price_missing=0;fee_errors=0
        dates={x.strftime('%Y%m%d') for x in d.date}
        for t in trades:
            day=pd.Timestamp(str(t['date'])).strftime('%Y%m%d')
            qty=float(t['qty']);price=float(t['price']);fee=float(t.get('fee',0))
            if day not in dates or t.get('side') not in ('buy','sell') or not all(math.isfinite(x) for x in (qty,price,fee)) or qty<=0 or price<=0 or fee<0 or qty!=int(qty):
                invalid+=1;continue
            grouped[day].append((t,qty,price,fee));byday[day]+=qty*price
        for row in d.to_dict('records'):
            day=row['date'].strftime('%Y%m%d')
            for t,qty,price,fee in grouped[day]:
                code=str(t.get('code',t.get('ticker','')))
                if t['side']=='sell':
                    if hold[code]+1e-8<qty:oversells+=1
                    hold[code]-=qty
                    if cash is not None:cash+=qty*price-fee
                else:
                    hold[code]+=qty
                    if cash is not None:cash-=qty*price+fee
                if rules and 'fee' in t and fee+1e-6<qty*price*(.001 if t['side']=='buy' else .003):fee_errors+=1
                if rules and prices is not None and cash is not None and t['side']=='buy':
                    quote=prices.get(day,{})
                    values={s:q*(quote[s][0] if s in quote else marks.get(s,0)) for s,q in hold.items() if q>0}
                    if any(not math.isfinite(v) or v<=0 for v in values.values()):price_missing+=1
                    else:
                        nav=cash+sum(values.values());limit=rules['position_limits'].get(code,rules['position_limits']['default'])
                        position_checked+=1
                        if nav<=0 or hold[code]*price/nav>limit+1e-8:position_errors+=1
                if cash is not None and cash < -.01:cash_errors+=1
            if prices is not None:
                marks.update({s:px[1] for s,px in prices.get(day,{}).items() if s in hold})
            if cash is not None and 'cash' in row:
                value=float(row['cash'])
                if not math.isfinite(value) or abs(cash-value)>max(.01,abs(cash)*1e-9):cash_errors+=1
        add('fills',invalid==0,f'잘못된 체결 {invalid}건')
        if 'initial_cash' in case:add('holdings',oversells==0,f'보유 수량 초과 매도 {oversells}건')
        else:missing.append('초기 보유 수량')
        if cash is not None and 'cash' in d and fees_known:add('cash',cash_errors==0,f'현금·수수료 원장 불일치 {cash_errors}건')
        if rules:
            if fees_known:add('contest_costs',fee_errors==0,f'공식 매수 0.1%·매도 0.3%보다 적은 비용 {fee_errors}건; 초과분은 체결 비용')
            if prices is None:missing.append('거래 당시 종목 비중을 재현할 원본 시세')
            elif price_missing:missing.append(f'편입 한도 검사 시세 누락 {price_missing}건')
            if position_checked:add('position_limits',position_errors==0,f'매수 시 종목 편입 한도 {position_checked}건 검사 · 위반 {position_errors}건')
        d['traded']=[byday[x.strftime('%Y%m%d')] for x in d.date]
        weekly=d.groupby(d.date.dt.to_period('W-SUN')).agg(nav=('nav','mean'),traded=('traded','sum'),days=('date','size'))
        weekly['turnover']=.5*weekly.traded/weekly.nav
        # A 36-month count is not a contest's violation count. Keep observations
        # only until the actual edition's dates and boundary-week rules are known.
        weeks=[dict(week=str(i),turnover=float(r.turnover),days=int(r.days)) for i,r in weekly.iterrows() if math.isfinite(r.turnover)]
        turnover=None
        if rules:
            part=d[d.date.between(pd.Timestamp(rules['start']),pd.Timestamp(rules['end']))].copy()
            observations=[];low=0
            if not part.empty:
                from .period import expected_dates
                for week, group in part.groupby(part.date.dt.to_period('W-SUN')):
                    start=max(week.start_time,pd.Timestamp(rules['start']));end=min(week.end_time,pd.Timestamp(rules['end']))
                    required=set(expected_dates('timefolio',start.strftime('%Y%m%d'),end.strftime('%Y%m%d')))
                    complete=bool(required) and required.issubset({x.strftime('%Y%m%d') for x in group.date})
                    value=float(.5*group.traded.sum()/group.nav.mean())
                    low+=int(complete and value<rules['weekly_min'])
                    observations.append(dict(week=str(week),turnover=value,complete=complete))
                add('contest_weekly_turnover',low<=rules['max_violations'],f'13회 기간 완료 주 회전율 5% 미달 {low}회 · 최대 3회 허용')
            else:missing.append('13회 기간에 해당하는 평가 원장')
            turnover=dict(period=[rules['start'],rules['end']],low_turnover_weeks=low,weeks=observations,final=False)
            missing.append('13회 실계좌 시작 잔고·전체 기간 및 경계 주 공식 회전율 대조')
        else:missing.append('13회 대회 기간·경계 주 처리에 따른 주간 회전율 판정')
        failed=[x for x in checks if x['status']=='failed']
        certified=not failed and not missing
        status='failed' if failed else 'missing' if missing else 'passed'
        summary=('검증 실패: '+', '.join(x['detail'] for x in failed)) if failed else ('원장·13회 규칙 부분 검사 완료 · 과거 종목·체결 증빙 부족' if rules else '원장 검사 완료 · 대회 규정·과거 종목 자료 부족')
        return dict(version=VERSION,status=status,competition_compliance_verified=certified,checks=checks,missing=missing,weekly=weeks,contest_turnover=turnover,rule_profile=rules,summary=summary)
    except (KeyError,TypeError,ValueError,OverflowError):
        return dict(version=VERSION,status='failed',competition_compliance_verified=False,
                    checks=[dict(code='schema',status='failed',detail='원장 필드 또는 수치 형식 오류')],missing=list(MISSING),summary='원장 형식 오류')


def report_assessment(report):
    cases=report.get('cases') or []
    from .contest_rules_evidence import profile, price_book
    rules=profile();prices=price_book(report) if rules else None
    results=[assess(case,rules=rules,prices=prices) for case in cases]
    if not results:return assess({})
    first=next((r for r in results if r['status']=='failed'),next((r for r in results if r['status']=='missing'),results[0]))
    return dict(first,case_count=len(results),cases=results)

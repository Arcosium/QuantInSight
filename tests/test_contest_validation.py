from copy import deepcopy
from autofolio.contest_validation import assess, report_assessment


def ledger():
    return dict(initial_cash=1000,daily=[dict(date='20261005',cash=899,nav=999),dict(date='20261006',cash=1007,nav=1007)],
        trades=[dict(date='20261005',code='005930',side='buy',qty=1,price=100,fee=1),dict(date='20261006',code='005930',side='sell',qty=1,price=110,fee=2)])


def test_complete_cash_ledger_is_screened_but_missing_evidence_not_certified():
    result=assess(ledger())
    assert result['status']=='missing' and result['missing']
    assert all(x['status']=='passed' for x in result['checks'])
    assert not result['competition_compliance_verified']
    assert result['weekly'][0]['turnover']>0


def test_fee_or_cash_mismatch_is_failure():
    case=ledger();case['trades'][0]['fee']=0
    result=assess(case)
    assert result['status']=='failed'
    assert any(x['code']=='cash' and x['status']=='failed' for x in result['checks'])


def test_sell_without_position_fails():
    case=ledger();case['trades']=case['trades'][1:]
    assert any(x['code']=='holdings' and x['status']=='failed' for x in assess(case)['checks'])


def test_unknown_certificate_cannot_bypass_validation():
    result=report_assessment(dict(cases=[ledger()],competition_compliance_verified=True,contest_certified=True))
    assert not result['competition_compliance_verified']


def test_nonfinite_and_outside_dates_fail():
    for field,value in [('price',float('nan')),('date','20261007'),('qty',.5)]:
        case=ledger();case['trades'][0][field]=value
        assert assess(case)['status']=='failed'


def test_all_cases_checked():
    bad=deepcopy(ledger());bad['daily'][1]['cash']=0
    assert report_assessment(dict(cases=[ledger(),bad]))['status']=='failed'


def test_missing_ledger():
    assert assess({})['status']=='missing'


def test_missing_fees_are_evidence_gap_not_cash_failure():
    case=ledger()
    for trade in case['trades']:trade.pop('fee')
    result=assess(case)
    assert result['status']=='missing'
    assert '체결별 실제 수수료' in result['missing']
    assert not any(x['code']=='cash' for x in result['checks'])


def test_targets_and_apply_report_blocker_not_permission_error(monkeypatch):
    import pytest
    from fastapi import HTTPException
    from autofolio import market_routes as routes
    monkeypatch.setattr(routes,'user',lambda _:dict(id=1))
    monkeypatch.setattr(routes,'mutation_guard',lambda _:None)
    monkeypatch.setattr(routes.research,'visible',lambda *_:True)
    summary=dict(market='timefolio',contest_certified=True,contest_validation=dict(summary='과거 종목 자료 부족'))
    monkeypatch.setattr(routes,'strategy_case',lambda _: (summary,None))
    monkeypatch.setattr(routes,'get_connection',lambda *_:{'configured':True})
    target=routes.targets('example',None)['targets'][0]
    assert not target['available']
    assert '과거 종목 자료 부족' in target['reason']
    with pytest.raises(HTTPException) as error:routes.apply('example',routes.Apply(target='timefolio'),None)
    assert error.value.status_code==409 and error.value.detail==target['reason']


def contest_rules():
    return dict(initial_cash=1000,start='20261001',end='20261130',weekly_min=.05,max_violations=3,
                position_limits={'default':.15,'005930':.4,'000660':.3})


def test_official_position_limit_uses_replayed_nav_and_distinguishes_exceptions():
    case=ledger();case['trades'][0].update(qty=2,price=100,fee=1)
    case['trades'][1].update(qty=2,price=110,fee=2)
    case['daily'][0].update(cash=799,nav=999);case['daily'][1].update(cash=1017,nav=1017)
    prices={'20261005':{'005930':(100,100)},'20261006':{'005930':(110,110)}}
    assert next(x for x in assess(case,rules=contest_rules(),prices=prices)['checks'] if x['code']=='position_limits')['status']=='passed'
    for trade in case['trades']:trade['code']='123456'
    prices={d:{'123456':px['005930']} for d,px in prices.items()}
    assert next(x for x in assess(case,rules=contest_rules(),prices=prices)['checks'] if x['code']=='position_limits')['status']=='failed'


def test_four_completed_low_turnover_weeks_fail_within_contest(monkeypatch):
    import pandas as pd
    from autofolio import period
    monkeypatch.setattr(period,'expected_dates',lambda market,start,end:tuple(x.strftime('%Y%m%d') for x in pd.bdate_range(start,end)))
    case=dict(initial_cash=1000,trades=[],daily=[dict(date=d.strftime('%Y%m%d'),cash=1000,nav=1000) for d in pd.bdate_range('2026-10-01','2026-10-30')])
    result=assess(case,rules=contest_rules())
    assert result['contest_turnover']['low_turnover_weeks']==5
    assert next(x for x in result['checks'] if x['code']=='contest_weekly_turnover')['status']=='failed'


def test_missing_or_changed_rule_snapshot_is_not_accepted(tmp_path,monkeypatch):
    from autofolio import contest_rules_evidence as evidence
    monkeypatch.setattr(evidence,'RULE_ROOT',tmp_path)
    assert evidence.profile() is None
    for filename in evidence.HASHES:(tmp_path/filename).write_text('{}')
    assert evidence.profile() is None

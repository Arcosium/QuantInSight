from datetime import date,timedelta
import json
import pytest
from autofolio import evaluation as e, period, store, catalogue
from autofolio.metrics import ledger


def book(start='20231008',end='20261007'):
    days=period.expected_dates('crypto',start,end)
    nav=100.
    daily=[]
    for i,d in enumerate(days):
        nav*=1.001 if i%11 else .996
        daily.append(dict(date=d,nav=nav))
    return dict(initial_nav=100.,daily=daily,trades=[])


def test_daily_anchor_has_exactly_24_9_3_months_and_ros_cannot_change_os():
    case=book();rows=ledger(case)
    phases=e.performance(rows,'20231008','20261007')
    assert [phases[k]['months'] for k in ('is','os','ros')]==[24,9,3]
    assert phases['is']['end']=='20251007'
    assert phases['os']['start']=='20251008' and phases['os']['end']=='20260707'
    assert phases['ros']['start']=='20260708'
    for r in case['daily']:
        if r['date']>='20260708':r['nav']*=.01
    changed=e.performance(ledger(case),'20231008','20261007')
    assert phases['os']==changed['os']
    assert changed['ros']['net_return']!=phases['ros']['net_return']
    assert e.segment_metrics(rows,'20231008','20261007',36)['months']==36
    assert e.selection_metrics({'evaluation_protocol':e.PROTOCOL,'performance':changed})==phases['os']
    assert e.selection_metrics({'performance':changed}) is None


def test_leap_window_and_missing_month_rejected():
    assert e.valid_window(['20210228','20240228'])
    assert not e.valid_window(['20210301','20240228'])
    assert not e.valid_window(None)
    rows=[r for r in ledger(book()) if not '20241008'<=r['date']<'20241108']
    assert e.segment_metrics(rows,'20231008','20261007',36) is None


def test_ingest_pinned_trial_after_calendar_moves(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DB',tmp_path/'index.db');monkeypatch.setattr(store,'DATA',tmp_path)
    store.initialize()
    value=dict(market='crypto',evaluation_window=['20231008','20261007'],evaluation_protocol=e.PROTOCOL,
               family='학습 모델',title='pinned',genome={'engine':'learned_v1'},account=book())
    path=tmp_path/'review.json';path.write_text(json.dumps(value))
    with period.using_window('20231009','20261008'):
        assert catalogue.ingest_file(path,path.stat())==('indexed',1)
        result=catalogue.rows()[0]
        assert not period.accepts(result)
        assert e.valid_result(result)
    assert result['months']==36
    assert result['performance']['os']['months']==9
    assert result['protocol_origin']=='native'
    # An incomplete final trading day must never pass by presenting split metrics.
    value['account']['daily'].pop();path.write_text(json.dumps(value))
    assert catalogue.ingest_file(path,path.stat())==('unmatched',0)


def test_queued_trial_rolls_but_running_trial_keeps_its_window(tmp_path,monkeypatch):
    from autofolio import research,learning
    monkeypatch.setattr(store,'DB',tmp_path/'state.db');monkeypatch.setattr(store,'DATA',tmp_path)
    monkeypatch.setattr(research,'protocol',lambda m:dict(ready=True,message='ready'))
    store.initialize();research.initialize()
    genome={k:v[0] for k,v in learning.domains('crypto').items()}
    with period.using_window('20231008','20261007'):
        identity=research.save_candidates(1,'crypto',[genome],'test')[0]
    with store.connect() as db:old=dict(db.execute('SELECT * FROM alpha_candidates WHERE id=?',(identity,)).fetchone())
    with period.using_window('20231009','20261008'):
        assert research.refresh_queued_window(dict(old,status='running'))['id']==identity
        fresh=research.refresh_queued_window(old)
        assert fresh['id']!=identity
        assert research.candidate_window(fresh)==('20231009','20261008')
    with store.connect() as db:assert db.execute('SELECT status FROM alpha_candidates WHERE id=?',(identity,)).fetchone()[0]=='retired'

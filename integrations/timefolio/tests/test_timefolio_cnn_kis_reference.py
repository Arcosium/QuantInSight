import pytest

from quant.timefolio_cnn_kis_reference import collect_code, page_rows, next_cursor, ReadOnlyKIS


def bar(day):
    return dict(stck_bsop_date=day, stck_oprc='10', stck_hgpr='12', stck_lwpr='9',
        stck_clpr='11', acml_vol='100', acml_tr_pbmn='1050')


def test_pagination_is_exclusive_and_drops_current_snapshot():
    bars = page_rows({'rt_cd':'0', 'output1':{'hts_avls':'future'},
        'output2':[bar('20221202'), bar('20221201')]}, '20221123', '20221202')
    assert next_cursor(bars, '20221123') == '20221130'
    assert next_cursor([bar('20221123')], '20221123') is None
    assert next_cursor([], '20221123') is None
    assert all('hts_avls' not in b for b in bars)


def test_reserved_future_date_and_duplicate_days_are_rejected():
    for records in [[bar('20260928')], [bar('20260923'),bar('20260923')]]:
        with pytest.raises(ValueError):
            page_rows({'rt_cd':'0','output2':records}, '20221123', '20260923')


def test_completed_cache_never_requeries_or_overwrites_and_checks_hash(tmp_path):
    class Client:
        def __init__(self): self.calls=[]
        def fetch(self, code, start, cursor):
            self.calls.append(cursor)
            return [bar('20221124'),bar('20221123')]
    c=Client(); first=collect_code(c,tmp_path,'005930',end='20221124')
    assert collect_code(c,tmp_path,'005930',end='20221124')==first and c.calls==['20221124']
    p=tmp_path/'005930/20221124.json';p.write_text('{}')
    with pytest.raises(ValueError,match='fingerprint'):
        collect_code(c,tmp_path,'005930',end='20221124')


def test_aggressive_request_rate_rejected(tmp_path):
    with pytest.raises(ValueError):ReadOnlyKIS(tmp_path,interval=.1)


def test_alphanumeric_security_codes_are_collected_without_changing_identity(tmp_path):
    class Client:
        def fetch(self, code, start, cursor):
            assert code == '0009K0'
            return [bar('20221123')]
    r=collect_code(Client(),tmp_path,'0009K0',end='20221123')
    assert r['code']=='0009K0' and r['rows']==1
    with pytest.raises(ValueError):collect_code(Client(),tmp_path,'../bad')


def test_only_transient_server_errors_retry_without_refresh(monkeypatch,tmp_path):
    import quant.timefolio_cnn_kis_reference as module
    monkeypatch.setattr(module,'existing_credentials',lambda p:dict(token='test',key='test',secret='test'))
    sleeps=[];monkeypatch.setattr(module.time,'sleep',sleeps.append)
    class Response:
        def __init__(self,status):self.status_code=status
        def json(self):return dict(rt_cd='0',output2=[bar('20221123')])
    statuses=[503,200];calls=[]
    client=ReadOnlyKIS(tmp_path)
    def get(*args,**kwargs):calls.append(1);return Response(statuses.pop(0))
    monkeypatch.setattr(client.session,'get',get)
    assert len(client.fetch('0009K0','20221123','20221123'))==1 and len(calls)==2
    assert 2 in sleeps
    statuses[:]=[429,200];calls.clear()
    with pytest.raises(RuntimeError,match='429'):client.fetch('0009K0','20221123','20221123')
    assert len(calls)==1 and statuses==[200]


def test_disconnected_transport_is_retried_then_explicitly_deferred(monkeypatch,tmp_path):
    import requests
    import quant.timefolio_cnn_kis_reference as module
    monkeypatch.setattr(module,'existing_credentials',lambda p:dict(token='test',key='test',secret='test'))
    delays=[];monkeypatch.setattr(module.time,'sleep',delays.append)
    client=ReadOnlyKIS(tmp_path);calls=[]
    def get(*args,**kwargs):
        calls.append(1);raise requests.ConnectionError('fixture: peer disconnected')
    monkeypatch.setattr(client.session,'get',get)
    with pytest.raises(module.ReferenceTemporarilyUnavailable):client.fetch('005930','20221123','20221123')
    assert len(calls)==3 and delays[-2:]==[2,4]

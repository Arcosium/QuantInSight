from starlette.requests import Request
from autofolio import store,research
from autofolio.controls import logs


def test_logs_start_recent_and_keep_market_scope(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DATA',tmp_path)
    monkeypatch.setattr(store,'DB',tmp_path/'research.db')
    store.initialize();research.initialize()
    for i in range(305):store.event('training',dict(market='kr',quarter=str(i)))
    store.event('training',dict(market='us',quarter='OTHER_MARKET'))
    request=Request({'type':'http','headers':[]});request.state.user={'id':1,'role':'admin'}
    events=logs(request,after=0,market='kr')['events']
    assert len(events)==300 and events[0]['id']==6 and events[-1]['id']==305
    assert all('OTHER_MARKET' not in e['message'] for e in events)
    assert logs(request,after=305,market='kr')['events']==[]
    store.event('training',dict(market='kr',quarter='next'))
    assert len(logs(request,after=305,market='kr')['events'])==1

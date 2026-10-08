"""Isolate the existing KIS module namespace. Never starts an order loop."""
import asyncio
import json
import sys
from pathlib import Path

async def snapshot(root, mode):
    sys.path.insert(0,str(root))
    from infra import auth_store, user_paths
    from account_equity import curve
    from infra.kis_broker import KISBroker
    admins=[a for a in auth_store.list_accounts() if a['is_admin']]
    if len(admins)!=1:
        return dict(message='소유자 계좌를 확인해 주세요.')
    kind='kis_paper' if mode=='kis-paper' else 'kis_real'
    profiles=[p for p in auth_store.list_profiles(admins[0]['id']) if p['kind']==kind]
    if not profiles:return dict(message='등록된 계좌 없음')
    uid=profiles[0]['uid']
    credentials=auth_store.get_user_credentials(uid)
    broker=KISBroker(credentials,token_path=root/'data'/str(uid)/'kis_token.json')
    try:
        snap=await broker.portfolio_holdings()
    finally:await broker.close()
    power=snap.get('buying_power') or {}
    if power.get('ok') is False:return dict(message='계좌 조회 실패')
    holdings=[dict(code=h.get('ticker') or h.get('code'),name=h.get('name'),qty=h.get('qty',0),
                   value=h.get('krw_value'),pnl_ratio=(h['pnl_pct']/100 if h.get('pnl_pct') is not None else None))
              for h in snap.get('holdings',[])]
    def read(path):
        try:return json.loads(path.read_text())
        except (OSError,ValueError):return []
    equity=curve(read(user_paths.equity_path(uid)))
    trades=[dict(date=t.get('ts'),code=t.get('ticker') or t.get('code'),name=t.get('name'),
                 side=t.get('side'),qty=t.get('qty'),price=t.get('price') or t.get('fill_price'))
            for t in read(user_paths.trade_log_path(uid)) if t.get('type')=='trade_executed']
    return dict(equity=equity,connected=True,message='연결됨',total_eval=power.get('total_eval'),cash=power.get('cash'),
                pnl_ratio=equity[-1]['net_return'] if equity else None,holdings=holdings,trades=list(reversed(trades[-100:])))

if __name__=='__main__':
    try:result=asyncio.run(snapshot(Path(sys.argv[1]),sys.argv[2]))
    except Exception:result=dict(message='계좌 조회 실패')
    print(json.dumps(result,ensure_ascii=False))

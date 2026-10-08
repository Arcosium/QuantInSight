"""Read the selected RFM 13 book using a member's own login; never submit orders."""
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit


def snapshot(credentials):
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'integrations/timefolio'))
    from Auto_folio.autofolio.timefolio_browser import TimefolioBrowser,TimefolioCredentials
    captured={}
    def capture(response):
        path=urlsplit(response.url).path
        if path not in {'/api/Portfolio/PerfCharts','/api/Portfolio/Trades'}:return
        try:captured[path]=response.json()
        except Exception:pass
    with TimefolioBrowser(headless=True,live_enabled=False) as browser:
        browser.page.on('response',capture)
        status=browser.login(TimefolioCredentials(credentials['username'],credentials['password']))
        if not status.get('logged_in'):return dict(connected=False,message='타임폴리오 로그인 실패')
        selector=browser.page.locator('select').first
        options=selector.locator('option').evaluate_all('(els)=>els.map(o=>({text:o.text,value:o.value,selected:o.selected}))')
        found=next((o for o in options if re.search(r'(?<!\d)13\s*회',o['text']) and '종료' not in o['text']),None)
        if not found:return dict(connected=False,message='이 계정에서 13회 대회를 찾을 수 없습니다.')
        if not found['selected']:
            captured.clear();selector.select_option(found['value']);browser.page.wait_for_timeout(1800);browser._wait_nav(browser.page)
        summary=browser.scrape_summary()
        if not summary.get('total_eval'):return dict(connected=False,message='13회 계좌 평가액을 확인하지 못했습니다.')
        browser.page.get_by_text('진입/청산 내역',exact=True).click()
        browser.page.wait_for_timeout(1500)
        chart=captured.get('/api/Portfolio/PerfCharts',{}).get('daily',[])
        equity=[dict(date=str(p['d'])[:10].replace('-',''),net_return=float(p['nav'])/1000-1) for p in (chart[0] if chart else [])]
        holdings=[dict(code=p['ticker'],name=p['name'],qty=p['qty'],value=p['value_krw'],pnl_ratio=p['pnl_pct']/100) for p in summary.get('positions',[])]
        trades=[]
        for t in captured.get('/api/Portfolio/Trades',[]):
            for time_key,price_key,side in [('et','ep','buy'),('xt','xp','sell')]:
                if t.get(time_key) and t.get(price_key):trades.append(dict(date=(str(t[time_key])+'+09:00' if not re.search(r'(Z|[+-]\d{2}:?\d{2})$',str(t[time_key])) else t[time_key]),code=str(t['prodId']).removeprefix('A'),name=t.get('prodNm'),side=side,qty=t.get('qty'),price=t[price_key]))
        trades.sort(key=lambda t:str(t['date']),reverse=True)
        nav=summary['total_eval'];cash=nav-sum(p['value'] for p in holdings)
        return dict(connected=True,message='13회 대회 연결됨',contest='RFM 13회',contest_id=found['value'],total_eval=nav,cash=cash,pnl_ratio=nav/1e9-1,holdings=holdings,trades=trades,equity=equity)

if __name__=='__main__':
    try:result=snapshot(json.load(sys.stdin))
    except Exception:result=dict(connected=False,message='13회 대회 조회 실패')
    print(json.dumps(result,ensure_ascii=False))

"""Timefolio account adapter for the unified UI; no retired strategy loops."""
from fastapi import FastAPI
from web import autofolio as af
app=FastAPI(docs_url=None,redoc_url=None)
@app.get('/api/autofolio/summary')
def summary():return af.summary()
@app.get('/api/autofolio/trades')
def trades(limit:int=50):return {'trades':af.trades(limit)}

@app.get('/api/autofolio/equity')
def equity():
    raw=af.contest_store.get_account_raw(af.PAPER_UID) or {}
    return {'equity':af.contest_store.equity_series(raw,view='daily',limit=1500),'initial_cash':raw.get('initial_cash')}

from fastapi.responses import RedirectResponse
@app.get('/')
def index():return RedirectResponse('https://quantinsight.ai-ve.uk/?market=timefolio&view=trading',status_code=308)

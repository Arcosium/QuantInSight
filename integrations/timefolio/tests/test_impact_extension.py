import numpy as np
import pytest
from quant.impact_extension import price_quantity, refill_path, paired


def test_individual_price_never_becomes_next_best():
    asks=np.array([[101,200],[102,1000],[103,2000]],float)
    assert price_quantity(asks,1,101)==200
    moved=np.array([[102,1000],[103,2000],[104,3000]],float)
    assert price_quantity(moved,1,101)==0
    assert price_quantity(asks,1,104) is None
    bids=np.array([[99,1],[98,2],[97,3]],float)
    assert price_quantity(bids,0,96) is None


def test_refill_example_divides_additions_by_consumption():
    # 1000 -> trade 800 -> 200 -> add 1000 -> 1200: refill=1.25, not 1.20.
    result=refill_path([0,100,200],[1000,200,1200],[(100,800)],100)
    assert result['increase'][-1]/800==1.25
    assert result['adjusted'][-1]/800==1.25
    assert result['first_refill_ms']==200
    assert result['cycles']==1


def test_trade_adjusted_proxy_recovers_hidden_net_increase_not_cancels():
    # During one update: consume 800, add 500; displayed falls by 300.
    result=refill_path([0,100],[1000,700],[(100,800)],100)
    assert result['increase'][-1]==0
    assert result['adjusted'][-1]==500
    assert result['first_refill_ms'] is None
    # Pure cancellation cannot count as trade-associated depletion/refill.
    result=refill_path([0,100,200],[1000,200,500],[],0)
    assert result['cycles']==0


def test_repeated_depletion_refill_and_no_pretrade_supply_count():
    r=refill_path([0,50,100,200,300,400],[900,1000,200,1200,400,1000],[(100,800),(300,800)],100)
    assert r['cycles']==2
    assert r['increase'][-1]==1600


def test_regime_pairing_preserves_direction_size_and_no_reuse():
    def row(t,qty,side,regime,value):
        return {'ts':t,'quantity':qty,'side':side,'regime':regime,'impact':np.full(100,value)}
    rows=[row(1,10,'buy','Low',2),row(2,10,'buy','Low',3),
          row(3,10.5,'buy','High',1),row(4,50,'buy','High',8),row(5,10,'sell','High',9)]
    result=paired(rows,'regime','Low','High')
    assert result['pairs']==1
    assert result['difference_low_minus_high']['mean'][0]==pytest.approx(1)


def test_extended_results_are_json_serializable_and_keep_missing_pilot_status():
    import json
    import pandas as pd
    from quant.impact_research import analyse
    from quant.impact_extension import extend
    times=np.arange(0,100001,100)
    books=np.zeros((len(times),2,50,2))
    for i in range(50):
        books[:,0,i,0]=99.99-i*.01
        books[:,1,i,0]=100.01+i*.01
        books[:,:,i,1]=2
    records=[(t,1,'Buy',3.,100.01) for t in [15000,27000,39000,51000,63000,75000,87000]]
    data={'times':times,'books':books,'epochs':np.ones(len(times),dtype=int),
          'connections':np.ones(len(times),dtype=int),'quality':{},
          'trades':pd.DataFrame(records,columns=['ts','connection','side','quantity','price'])}
    base,events=analyse(data,{'symbol':'TESTUSDT'})
    ext,rows,same,market=extend(data,base,events)
    json.dumps(ext,allow_nan=False)
    assert ext['pilot']['actual_orders_observed']==0
    assert ext['resiliency']['positive_initial_events']==0
    assert ext['refill']['refill_observed']==0
    assert ext['regimes']['spread']['matched']['pairs']==0

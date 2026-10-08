"""Shared close-signal / next-open execution for evaluation and own paper books."""
import math


def policy(g):
    return dict(buy_rule=g.get('buy_rule','top_k'),sell_rule=g.get('sell_rule','rebalance'),
        position_sizing=g.get('position_sizing','equal'),stop_loss=g.get('stop_loss',.1),
        take_profit=g.get('take_profit',.2),max_weight=g.get('max_weight',.1),
        gross_exposure=g.get('gross_exposure',1.),
        rebalance_sessions=g.get('rebalance_sessions',g['holding_sessions']))


def _finite(value):
    return isinstance(value,(int,float)) and math.isfinite(value)


def choose_picks(records,g):
    p=policy(g);count=g['selection_count']
    ranked=sorted((dict(r) for r in records if _finite(r.get('score'))),key=lambda r:(-r['score'],r['symbol']))
    if p['buy_rule']=='positive_score':ranked=[r for r in ranked if r['score']>0]
    elif p['buy_rule']=='top_quantile':ranked=ranked[:max(1,math.ceil(len(ranked)*.2))]
    picks=ranked[:count]
    if p['position_sizing']=='inverse_volatility':
        raw=[1/max(float(r.get('volatility',.01)),1e-4) if _finite(r.get('volatility')) and r['volatility']>=0 else 0. for r in picks]
    elif p['position_sizing']=='score':
        # Shift negatives rather than betting an unbounded amount on raw scores.
        floor=min((r['score'] for r in picks),default=0.)
        raw=[r['score']-min(0.,floor)+1e-12 for r in picks]
    else:raw=[1. for _ in picks]
    total=sum(raw)
    result=[]
    for row,value in zip(picks,raw):
        # Equal sizing preserves the reserved slots when fewer names qualify.
        share=1/count if p['position_sizing']=='equal' else value/total if total else 0.
        weight=min(p['max_weight'],p['gross_exposure']*share)
        if weight>0:result.append(dict(symbol=str(row['symbol']),adv20=float(row.get('adv20',0.)),weight=weight))
    return result


def build_plan(signal_records,g,hold,marks,entries,rebalance):
    p=policy(g);picks=choose_picks(signal_records,g) if rebalance else []
    exits=[];stops=set()
    if p['sell_rule']=='stop_take':
        for symbol in hold:
            entry=entries.get(symbol);mark=marks.get(symbol)
            if entry and mark and (mark/entry-1<=-p['stop_loss'] or mark/entry-1>=p['take_profit']):
                exits.append(symbol);stops.add(symbol)
    elif rebalance:
        selected={r['symbol'] for r in picks}
        exits=list(hold) if p['sell_rule']=='rebalance' else [s for s in hold if s not in selected]
    return dict(buys=[r for r in picks if r['symbol'] not in stops],sells=exits,rebalance=bool(rebalance))


def fill_orders(cash,hold,marks,entries,quotes,pending,g,market,day,headroom=None):
    """Only open and tradability are read here; close prices cannot affect fills."""
    p=policy(g);buy_cost=.0015 if market=='timefolio' else .002 if market=='kr' else .001
    sell_cost=.0035 if market=='timefolio' else buy_cost
    if isinstance(pending,list):
        pending=dict(sells=list(hold),buys=[dict(r,weight=min(p['max_weight'],p['gross_exposure']/g['selection_count'])) for r in pending])
    pending=pending or {};trades=[];unfilled=[]
    def price(symbol):
        if symbol not in quotes.index:return None
        value=float(quotes.loc[symbol,'open'])
        return value if math.isfinite(value) and value>0 else None
    for symbol in pending.get('sells',[]):
        if symbol not in hold:continue
        px=price(symbol)
        if px is None or not bool(quotes.loc[symbol,'tradable_sell']):unfilled.append(symbol);continue
        qty=hold.pop(symbol);fee=px*qty*sell_cost;cash+=px*qty-fee;entries.pop(symbol,None)
        trades.append(dict(date=day,code=symbol,side='sell',qty=qty,price=px,fee=fee))
    values={s:q*(price(s) or marks.get(s,entries.get(s,0.))) for s,q in hold.items()}
    equity=cash+sum(values.values());invested=sum(values.values())
    for pick in pending.get('buys',[]):
        symbol=pick['symbol'];px=price(symbol)
        if symbol in hold or px is None or not bool(quotes.loc[symbol,'tradable_buy']):continue
        adv=pick.get('adv20',0.);weight=min(float(pick.get('weight',0.)),p['max_weight'])
        if not _finite(adv) or adv<=0 or not math.isfinite(weight) or weight<=0:continue
        exposure=max(0.,(p['gross_exposure']*equity-invested)/(1+p['gross_exposure']*buy_cost))
        budget=min(cash/(1+buy_cost),equity*weight,adv*.01,exposure)
        if headroom is not None:budget=min(budget,headroom(cash,hold,marks,quotes,symbol,buy_cost))
        qty=budget/px if market=='crypto' else math.floor(budget/px)
        if qty<=0:continue
        value=qty*px;fee=value*buy_cost;cash-=value+fee
        if cash < -1e-7:raise ValueError('Execution exceeded available cash')
        cash=max(0.,cash);hold[symbol]=qty;marks[symbol]=px;entries[symbol]=px
        invested+=value;equity-=fee
        trades.append(dict(date=day,code=symbol,side='buy',qty=qty,price=px,fee=fee))
    return cash,trades,unfilled


def account(prices,signals,g,market,start,end,headroom=None):
    from .period import expected_dates
    days=expected_dates(market,start,end)
    bydate={day:frame.set_index('symbol') for day,frame in prices.groupby(prices.date.dt.strftime('%Y%m%d'))}
    bysignal={day:frame.to_dict('records') for day,frame in signals.groupby(signals.date.dt.strftime('%Y%m%d'))}
    initial=1e9 if market=='timefolio' else 1e8 if market=='kr' else 1e5
    cash=float(initial);hold={};marks={};entries={};daily=[];trades=[];stale=0;pending={}
    interval=policy(g)['rebalance_sessions']
    for index,day in enumerate(days):
        quotes=bydate.get(day)
        if quotes is None:raise ValueError('평가일 시세 누락')
        cash,filled,unfilled=fill_orders(cash,hold,marks,entries,quotes,pending,g,market,day,headroom)
        trades.extend(filled)
        for symbol in hold:
            if symbol in quotes.index:
                close=float(quotes.loc[symbol,'close'])
                if math.isfinite(close) and close>0:marks[symbol]=close
                else:stale+=1
            else:stale+=1
        nav=cash+sum(q*marks[s] for s,q in hold.items())
        daily.append(dict(date=day,nav=nav,cash=cash,positions=len(hold)))
        pending=build_plan(bysignal.get(day,[]),g,hold,marks,entries,index%interval==0)
        pending['sells']=list(dict.fromkeys(unfilled+pending['sells']))
    return dict(initial_cash=initial,daily=daily,trades=trades),stale

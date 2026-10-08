"""Same-price refill, multiple regimes and post-flow response; public data only.

Positive displayed-depth changes are a lower-bound proxy for gross additions.
The trade-adjusted proxy assumes correctly aligned, fully observed executions.
Neither identifies hidden orders or cancels inside an aggregated book update.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant.impact_research import reconstruct, analyse, _curve_stats, HORIZONS, first_sustained


def price_quantity(levels, side, price):
    """Missing inside covered prices means zero; beyond depth means unknown."""
    p, q = levels[:, 0], levels[:, 1]
    if (side == 1 and price > p[-1]+1e-8) or (side == 0 and price < p[-1]-1e-8):
        return None
    return float(q[np.isclose(p, price, rtol=0, atol=1e-8)].sum())


def refill_path(times, quantities, executions, first_trade_ms):
    """Executions are (timestamp, quantity) at exactly one price/taker side."""
    q = np.asarray(quantities, dtype=float)
    delta = np.diff(q)
    trades = np.asarray(executions, dtype=float).reshape(-1, 2)
    filled = np.array([trades[(trades[:, 0] > a) & (trades[:, 0] <= b), 1].sum()
                       for a, b in zip(times[:-1], times[1:])])
    eligible = np.asarray(times[1:]) >= first_trade_ms
    increases = np.where(eligible, np.maximum(delta, 0), 0)
    adjusted = np.where(eligible, np.maximum(delta+filled, 0), 0)
    first = np.flatnonzero(increases > 1e-9)
    cycles, pending = 0, False
    for change, traded, allowed in zip(delta, filled, eligible):
        if not allowed:
            continue
        if change < -1e-9 and traded > 0:
            pending = True
        elif change > 1e-9 and pending:
            cycles += 1
            pending = False
    return {"increase": np.r_[0, np.cumsum(increases)],
            "adjusted": np.r_[0, np.cumsum(adjusted)],
            "first_refill_ms": int(times[first[0]+1]) if len(first) else None,
            "cycles": cycles}


def scalar(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def response_stats(rows, key="impact"):
    return _curve_stats([r[key] for r in rows], [r["minute"] for r in rows])


def paired(rows, field, low, high, tolerance=.1):
    a = [r for r in rows if r[field] == low]
    b = [r for r in rows if r[field] == high]
    used, pairs = set(), []
    for r in a:
        options = [(abs(np.log(r['quantity']/s['quantity'])), i, s)
                   for i, s in enumerate(b) if i not in used and s['side'] == r['side']
                   and max(r['quantity'], s['quantity'])/min(r['quantity'], s['quantity']) <= 1+tolerance
                   and abs(r['ts']-s['ts']) <= 600000]
        if options:
            _, i, s = min(options, key=lambda x: x[0])
            used.add(i); pairs.append((r, s))
    stats = _curve_stats([a['impact']-b['impact'] for a, b in pairs])
    return {"pairs": len(pairs), "quantity_ratio_limit": 1+tolerance,
            "difference_low_minus_high": stats,
            "interpretation": "동일 방향·수량 차이 10% 이내·10분 이내. 쌍 단위 탐색 구간, 인과효과 아님."}


def extend(data, base, events, tick_size=.01):
    times, books, epochs = data['times'], data['books'], data['epochs']
    mid = books[:, :, 0, 0].mean(axis=1)
    trades = data['trades'].sort_values('ts')
    enriched, same_price, exclusions = [], [], Counter()
    for event in events:
        r = {k: v for k, v in event.items() if k not in ['depth_fixed', 'depth_rolling']}
        t, pre = r['ts'], r['book_index']
        side, sign = (1, 1) if r['side'] == 'buy' else (0, -1)
        connection = data['connections'][pre]
        history_targets = np.arange(t-10000, t, 100)
        idx = np.searchsorted(times, history_targets, side='right')-1
        valid_history = idx.min() >= 0 and epochs[idx[0]] == epochs[pre] and np.max(history_targets-times[idx]) <= 1000
        r['volatility'] = float(np.std(np.diff(np.log(mid[idx])))*1e4) if valid_history else None
        past = trades[(trades.ts >= t-10000) & (trades.ts < t) & (trades.connection == connection)]
        r['activity'] = float(past.quantity.sum()) if valid_history else None
        r['spread_ticks'] = int(round((books[pre, 1, 0, 0]-books[pre, 0, 0, 0])/tick_size))
        r['spread_regime'] = '1 tick' if r['spread_ticks'] == 1 else '2 tick' if r['spread_ticks'] == 2 else '3+ tick'
        bid_depth, ask_depth = books[pre, :, :5, 1].sum(axis=1)
        r['obi'] = float((bid_depth-ask_depth)/(bid_depth+ask_depth))
        r['directional_obi'] = sign*r['obi']
        post = trades[(trades.ts >= t+100) & (trades.ts <= t+10000) & (trades.connection == connection)]
        own_side = 'Buy' if side == 1 else 'Sell'
        opposite = post[post.side != own_side]
        r['opposite_quantity'] = float(opposite.quantity.sum())
        r['opposite_share'] = float(opposite.quantity.sum()/post.quantity.sum()) if post.quantity.sum() else None
        r['opposite_first_seconds'] = float((opposite.ts.min()-t)/1000) if len(opposite) else None
        r['opposite_flow'] = '반대 흐름 있음' if len(opposite) else '반대 흐름 없음'
        r['half_initial_seconds'] = first_sustained(r['impact'], r['impact'][0]/2, above=False, after_index=1) if r['impact'][0] > 1e-9 else None
        r['returned_to_start_seconds'] = first_sustained(r['impact'], 0, above=False, after_index=1) if r['impact'][0] > 1e-9 else None
        initial = trades[(trades.ts >= t) & (trades.ts < t+100) & (trades.connection == connection) & (trades.side == own_side)]
        end_t = int(initial.ts.max())
        post_targets = end_t+(HORIZONS*1000).astype(int)
        post_idx = np.searchsorted(times, post_targets, side='right')-1
        good = (post_targets[-1] <= times[-1] and epochs[pre] == epochs[post_idx[-1]]
                and np.max(post_targets-times[post_idx]) <= 1000)
        r['post_end_impact'] = sign*(mid[post_idx]-mid[pre])/mid[pre]*1e4 if good else None
        r['event_end_ms'] = end_t
        enriched.append(r)

        price = float(books[pre, side, 0, 0])
        consumed = initial[np.isclose(initial.price, price, rtol=0, atol=1e-8)]
        c = float(consumed.quantity.sum())
        if c <= 0:
            exclusions['no_initial_trade_at_pre_best'] += 1
            continue
        last = int(np.searchsorted(times, t+10000, side='right')-1)
        ix = np.arange(pre, last+1)
        qty = [price_quantity(books[i, side], side, price) for i in ix]
        if any(v is None for v in qty) or np.diff(times[ix]).max(initial=0) > 1000:
            exclusions['same_price_coverage_or_gap'] += 1
            continue
        at_price = trades[(trades.ts > times[pre]) & (trades.ts <= times[last])
                          & (trades.side == own_side) & (trades.connection == connection)
                          & np.isclose(trades.price, price, rtol=0, atol=1e-8)]
        first_trade = int(consumed.ts.min())
        path = refill_path(times[ix], qty, at_price[['ts', 'quantity']].to_numpy(), first_trade)
        sample = np.searchsorted(times[ix], t+(HORIZONS*1000).astype(int), side='right')-1
        ratios = path['increase'][sample]/c
        adjusted = path['adjusted'][sample]/c
        one = int(np.searchsorted(times[ix], t+1000, side='right')-1)
        response_one = r['impact'][9]
        # The outcome starts after the one-second refill classification window.
        future_delta = np.asarray(r['impact'])-response_one
        held = books[ix[:one+1], side, 0, 0]
        broken = bool(np.any(held > price+1e-8)) if side == 1 else bool(np.any(held < price-1e-8))
        filled_one = float(at_price[at_price.ts <= t+1000].quantity.sum())
        absorption = bool(ratios[9] >= 1 and not broken and filled_one >= qty[0])
        s = {"ts": t, "minute": r['minute'], "side": r['side'], "price": price,
             "pre_quantity": qty[0], "consumed_initial": c, "quantity": r['quantity'],
             "ratio": ratios, "adjusted_ratio": adjusted,
             "refill_class": '높은 refill' if ratios[9] >= 1 else '낮은 refill',
             "first_refill_seconds": (path['first_refill_ms']-first_trade)/1000 if path['first_refill_ms'] is not None else None,
             "cycles": path['cycles'], "absorption_candidate": absorption,
             "filled_same_price_1s": filled_one, "impact": r['impact'], "after_1s": future_delta,
             "example": {"seconds": ((times[ix]-t)/1000).tolist(), "quantity": qty,
                         "executions": [[(int(a)-t)/1000, float(b)] for a, b in at_price[['ts', 'quantity']].to_numpy()]}}
        same_price.append(s)

    regimes = {}
    for field, bounds in [('depth', [.2, .8]), ('volatility', [1/3, 2/3]), ('activity', [1/3, 2/3])]:
        valid = [r[field] for r in enriched if r[field] is not None]
        lo, hi = np.quantile(valid, bounds) if valid else (None, None)
        for r in enriched:
            v = r[field]
            r[field+'_regime'] = '미분류' if v is None else 'Low' if v <= lo else 'High' if v >= hi else 'Normal'
        regimes[field] = {"thresholds": [lo, hi], "quantiles": bounds,
            "groups": {label: response_stats([r for r in enriched if r[field+'_regime'] == label]) for label in ['Low','Normal','High','미분류']},
            "matched": paired(enriched, field+'_regime', 'Low', 'High')}
    regimes['spread'] = {"groups": {label: response_stats([r for r in enriched if r['spread_regime'] == label]) for label in ['1 tick','2 tick','3+ tick']},
                          "matched": paired(enriched, 'spread_regime', '1 tick', '3+ tick')}
    # OBI bins are descriptive conditioning cells, not a predictive backtest.
    obi = []
    for lower, upper in [(-1, -.2), (-.2, .2), (.2, 1.000001)]:
        for label in ['Low', 'High']:
            pool = [r for r in enriched if lower <= r['directional_obi'] < upper and r['depth_regime'] == label]
            obi.append({'lower': lower, 'upper': min(upper, 1), 'depth': label, 'impact': response_stats(pool)})
    refill_groups = {}
    for side in ['buy', 'sell']:
        for label in ['높은 refill', '낮은 refill']:
            pool = [r for r in same_price if r['side'] == side and r['refill_class'] == label]
            refill_groups[side+'_'+label] = {'n':len(pool), 'ratio':response_stats(pool,'ratio'),
                'after_1s': response_stats(pool,'after_1s'), 'median_consumed':scalar([r['consumed_initial'] for r in pool]),
                'absorption_candidates':sum(r['absorption_candidate'] for r in pool)}
    fast = [r['first_refill_seconds'] for r in same_price if r['first_refill_seconds'] is not None]
    after = [r for r in enriched if r['post_end_impact'] is not None]
    examples = sorted(same_price, key=lambda r: (-int(r['absorption_candidate']), -r['cycles'], r['ts']))[:2]
    extension = {"schema_version":1, 'created_utc':datetime.now(timezone.utc).isoformat(),
        'events':len(enriched), 'tick_size':tick_size, 'horizons_seconds':HORIZONS.tolist(),
        'method': {'same_price':'사건 직전 상대편 최우선 개별 가격 고정',
                   'refill_ratio':'sum(max(delta displayed quantity,0))/initial 100ms same-price taker quantity',
                   'adjusted_proxy':'sum(max(delta displayed quantity + same-price executions,0))/initial consumption',
                   'refill_classification_seconds':1, 'high_refill_threshold':1,
                   'feature_lookback_seconds':10,'regime_depth_quantiles':[.2,.8],
                   'history_missing':sum(r['volatility'] is None for r in enriched),
                   'matching_quantity_ratio':1.1, 'sustain_seconds':.3},
        'refill': {'n':len(same_price), 'excluded':dict(exclusions),
            'ratio':response_stats(same_price,'ratio'), 'adjusted_ratio':response_stats(same_price,'adjusted_ratio'),
            'refill_observed':len(fast),'refill_not_observed':len(same_price)-len(fast),
            'first_refill_median_observed_only':scalar(fast),
            'repeated_cycles':sum(r['cycles'] >= 2 for r in same_price),
            'absorption_candidates':sum(r['absorption_candidate'] for r in same_price),
            'groups':refill_groups,
            'examples':[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items()} for r in examples]},
        'regimes':regimes, 'obi_cells':obi,
        'resiliency':{'after_end':response_stats(after,'post_end_impact'),
            'after_end_excluded':len(enriched)-len(after),
            'opposite_events':sum(r['opposite_quantity']>0 for r in enriched),
            'opposite_share_median':scalar([r['opposite_share'] for r in enriched]),
            'opposite_first_median_observed_only':scalar([r['opposite_first_seconds'] for r in enriched]),
            'flow_groups':{label:response_stats([r for r in enriched if r['opposite_flow']==label]) for label in ['반대 흐름 있음','반대 흐름 없음']},
            'early_impact_bps':base['groups']['all']['impact']['mean'][0],
            'late_impact_bps':base['groups']['all']['impact']['mean'][-1],
            'positive_initial_events':sum(bool(r['impact'][0] > 1e-9) for r in enriched),
            'half_initial_events':sum(r['half_initial_seconds'] is not None for r in enriched),
            'returned_to_start_events':sum(r['returned_to_start_seconds'] is not None for r in enriched),
            'half_initial_median_observed_only':scalar([r['half_initial_seconds'] for r in enriched]),
            'returned_to_start_median_observed_only':scalar([r['returned_to_start_seconds'] for r in enriched]),
            'positive_at_10s_events':sum(bool(r['impact'][-1] > 1e-9) for r in enriched),
            'interpretation':'10초 잔존 반응은 영구 충격이 아니다. 후속 주문 흐름과 시장 공통 변동을 포함한다.'},
        'pilot':{'status':'awaiting_user_order_records','actual_orders_observed':0,
                 'message':'실제 본인 주문 기록 없음. 주문 관측기 구현·합성자료 검증과 실제 파일럿 수행을 구분한다.'},
        'limitations':['표시 잔량의 양의 변화는 총 신규 공급의 하한 대용치다. 갱신 사이 취소·체결과 숨은 공급은 완전히 식별하지 못한다.',
            '체결 보정 대용치는 스트림 시각 정렬과 표시 체결 포착 가정에 민감하다.',
            'refill은 첫 1초로 분류하고 이후 가격 변화를 비교한다. 인과효과나 실시간 매매 수익 검증이 아니다.',
            '반복 소진·보충은 체결이 포함된 잔량 감소 뒤 증가를 세며 주문 단위 식별이 아니다.',
            '한 종목·30분의 사후 보완 분석이며 새로운 표본 검증이 아니다. 다중 비교를 확증 검정으로 해석하지 않는다.']}
    # Public market path supports browser-only matching to private user records.
    market = {'symbol':base['symbol'], 'venue':base['venue'], 'tick_size':tick_size,
              'columns':['ts','bid','ask','bid_depth5','ask_depth5','epoch'],
              'rows':[[int(t),float(b[0,0,0]),float(b[1,0,0]),float(b[0,:5,1].sum()),float(b[1,:5,1].sum()),int(e)] for t,b,e in zip(times,books,epochs)]}
    return extension, enriched, same_price, market


def chart(ext, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
    plt.rcParams.update({'font.family':font,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'font.size':10})
    t=ext['horizons_seconds']; colors=['#9b3c28','#252b2d','#81898b']
    fig,axes=plt.subplots(2,1,figsize=(8,5.8),layout='constrained')
    for k,(side,label) in enumerate([('buy','강한 매수'),('sell','강한 매도')]):
        for j,c in enumerate(['높은 refill','낮은 refill']):
            g=ext['refill']['groups'][side+'_'+c]
            if g['n']:
                axes[k].plot(t,g['after_1s']['mean'],label=f"{label} · {c} · {g['n']}건",color=colors[j])
        axes[k].axvspan(0,1,color='#eeeeee');axes[k].axhline(0,color='#aaaaaa',lw=.7)
        axes[k].set(xlim=(1,10),ylabel='1초 이후 추가 반응 (bp)');axes[k].legend(frameon=False,fontsize=9)
    axes[-1].set_xlabel('사건 시작 후 시간 (초)');fig.savefig(output/'04_same_price.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(8.4,6),layout='constrained')
    for ax,(name,title) in zip(axes.flat,[('depth','Depth · 하위/상위 20%'),('spread','스프레드'),('volatility','직전 10초 변동성'),('activity','직전 10초 체결량')]):
        for j,(label,g) in enumerate(ext['regimes'][name]['groups'].items()):
            if g['n'] and label!='미분류': ax.plot(t,g['mean'],label=f"{label} · {g['n']}건",color=colors[j%3])
        ax.set(title=title,xlabel='시간 (초)',ylabel='가격 반응 (bp)');ax.legend(frameon=False,fontsize=8)
    fig.savefig(output/'05_regimes_extended.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.4,3.5),layout='constrained')
    for j,(label,g) in enumerate(ext['resiliency']['flow_groups'].items()):
        if g['n']: ax.plot(t,g['mean'],label=f"{label} · {g['n']}건",color=colors[j])
    ax.set(xlabel='시간 (초)',ylabel='가격 반응 (bp)');ax.legend(frameon=False)
    fig.savefig(output/'06_opposite_flow.png',dpi=160);plt.close(fig)
    for i,r in enumerate(ext['refill']['examples']):
        fig,ax=plt.subplots(figsize=(8.4,3.6),layout='constrained')
        ax.step(r['example']['seconds'],r['example']['quantity'],where='post',color=colors[0],label='동일 가격 표시 잔량')
        for sec,qty in r['example']['executions']:
            ax.axvline(sec,color='#7b8387',alpha=.35,lw=.8)
        ax.set(xlim=(0,10),xlabel='시간 (초)',ylabel='잔량 (ETH)',title=f"{r['price']:.2f} USDT · {r['side']} · 반복 {r['cycles']}회")
        ax.legend(frameon=False);fig.savefig(output/f'04_episode_{i+1}.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tick-size',type=float,required=True)
    args=parser.parse_args()
    if args.tick_size<=0 or not np.isfinite(args.tick_size): parser.error('positive finite tick size required')
    if args.output.exists(): parser.error('output already exists')
    manifest=json.loads((args.input/'manifest.json').read_text())
    if not manifest.get('complete'): parser.error('completed capture required')
    data=reconstruct(args.input/'raw.jsonl.gz');base,events=analyse(data,manifest)
    ext,rows,same,market=extend(data,base,events,args.tick_size)
    base.update(extension=ext,pilot_market=market,source_manifest=manifest,
                raw_sha256=hashlib.file_digest((args.input/'raw.jsonl.gz').open('rb'),'sha256').hexdigest(),
                extension_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.mkdir(parents=True)
    (args.output/'study.json').write_text(json.dumps(base,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (args.output/'extension.json').write_text(json.dumps(ext,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for records,name in [(rows,'regime_events.csv'),(same,'same_price_events.csv')]:
        flat=[]
        for row in records:
            item={k:v for k,v in row.items() if not isinstance(v,(dict,list,np.ndarray))}
            for key in ['impact','post_end_impact','ratio','adjusted_ratio','after_1s']:
                if row.get(key) is not None:
                    for h in [.1,1,5,10]: item[f'{key}_{h:g}s']=float(row[key][round(h*10)-1])
            flat.append(item)
        pd.DataFrame(flat).to_csv(args.output/name,index=False)
    chart(ext,args.output)
    print(json.dumps({'events':ext['events'],'same_price_events':ext['refill']['n'],
        'refill_observed':ext['refill']['refill_observed'],'repeat':ext['refill']['repeated_cycles'],
        'absorption_candidates':ext['refill']['absorption_candidates'],'opposite_events':ext['resiliency']['opposite_events']},ensure_ascii=False))


if __name__=='__main__':
    main()

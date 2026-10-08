'use strict';
const cryptoPaper=(()=>{
 const c={book:null,summary:null,model:'cnn',symbol:'BTC',side:null,posPage:0,tradePage:0,seq:0,chartSeq:0,candles:null,busy:false};
 const active=()=>state.tab==='crypto'&&state.subtab==='paper'&&!$('crypto-panel').hidden;
 const num=v=>Number.isFinite(v)?new Intl.NumberFormat('ko-KR',{maximumFractionDigits:6}).format(v):'미확인';
 const usd=v=>Number.isFinite(v)?new Intl.NumberFormat('ko-KR',{maximumFractionDigits:2}).format(v):'미확인';
 const pct=v=>Number.isFinite(v)?`${v.toFixed(2)}%`:'미확인';
 const direction=v=>v==='long'?'롱':v==='short'?'숏':v||'미확인';
 const origin=v=>v==='retrospective'?'소급 계산':'운영 기록';
 const time=v=>v==null?'시각 미기록':tradeTime(v);
 function table(id,rows,columns,empty='기록 없음'){
  const body=$(id);body.replaceChildren();
  for(const row of rows){const tr=el('tr');for(const value of columns(row)){const td=el('td');if(value instanceof Node)td.append(value);else td.textContent=String(value??'미확인');tr.append(td);}body.append(tr);}
  if(!rows.length){const tr=el('tr'),td=el('td',empty);td.colSpan=8;tr.append(td);body.append(tr);}
 }
 function symbolButton(symbol){return el('span',symbol);}
 function fail(e){if(active()){$('worker-status').textContent='모의매매 · 연결 확인 필요';$('lamp').classList.remove('active');}$('crypto-error').hidden=false;$('crypto-error').textContent=e.message||'크립토 자료를 불러오지 못했습니다.';}
 function blank(id,message){const svg=$(id);svg.replaceChildren(node('text',{x:18,y:65},message));svg.setAttribute('viewBox','0 0 600 180');}
 function metrics(){const b=c.book;$('crypto-metrics').replaceChildren();for(const [label,value] of [['평가 순자산 (USD)',usd(b.mtm_usd)],['계좌 수익률',pct(b.upnl_pct_book)],['롱 / 숏',`${b.long??0} / ${b.short??0}`],['마감 순자산 (USD)',usd(b.equity_usd)]]){const item=el('div');item.append(el('span',label),el('strong',value));$('crypto-metrics').append(item);}
  $('worker-status').textContent=b.stale?'모의매매 · 갱신 지연':b.active?'모의매매 · 장부 연결됨':'모의매매 · 운용 기록 없음';$('lamp').classList.toggle('active',Boolean(b.active&&!b.stale));const tick=c.summary?.tick;$('crypto-status').textContent=[b.active?(b.stale?'갱신 지연':'장부 연결됨'):'운용 기록 없음',b.sub,b.last_signal?.ts?'최근 신호 '+time(b.last_signal.ts):'',tick?.next_ms?'다음 평가 '+time(tick.next_ms):''].filter(Boolean).join(' · ');
  $('crypto-basis').textContent=[b.valuation_note,`초기 자본 ${usd(b.capital)} USD`,Number.isFinite(b.gross)?`총 노출 ${usd(b.gross*b.capital)} USD`:'',`평가 ${b.upnl_n??0}/${b.n_open??0}종목`].filter(Boolean).join(' · ');
  $('crypto-history-note').textContent=[b.series_basis||'기록된 장부 순자산 기준',b.history_note,b.series_message].filter(Boolean).join(' · ');
 }
 function equity(){const b=c.book;if(!b)return;const rows=(b.series||[]).filter(r=>Number.isFinite(r.equity)&&r.ts!=null).map(r=>({...r,net_return:r.equity-1}));if(!rows.length){blank('crypto-equity','기록된 수익률 곡선 없음');$('crypto-period').textContent='';return;}
  const f=frame('crypto-equity',rows.map(r=>r.net_return),280);f.svg.append(node('path',{d:rows.map((r,i)=>`${i?'L':'M'}${f.x(i/Math.max(1,rows.length-1))},${f.y(r.net_return)}`).join(' '),stroke:'#d8ba7d','stroke-width':2,fill:'none'}));
  $('crypto-period').textContent=`${time(rows[0].ts)} ~ ${time(rows.at(-1).ts)}`;
 }
 function positions(){if(!c.book)return;const term=$('crypto-search').value.trim().toLowerCase(),sort=$('crypto-sort').value;
  const rows=[...(c.book.positions||[])].filter(p=>p.base.toLowerCase().includes(term));const score=p=>sort==='pnl'?p.upnl_pct:sort==='change'?p.chg24:Math.abs(p.score??0);rows.sort((a,b)=>(score(b)??-Infinity)-(score(a)??-Infinity));
  const pages=Math.max(1,Math.ceil(rows.length/10));c.posPage=Math.min(c.posPage,pages-1);
  table('crypto-positions',rows.slice(c.posPage*10,c.posPage*10+10),p=>[symbolButton(p.base,p.side),direction(p.side),num(p.entry),num(p.last),usd(p.notional_usd),pct(p.upnl_pct)],'보유 포지션 없음');
  $('crypto-pos-page').textContent=`${c.posPage+1} / ${pages} · ${rows.length}개`;$('crypto-pos-prev').disabled=!c.posPage;$('crypto-pos-next').disabled=c.posPage>=pages-1;
 }
 function history(){const b=c.book,closed=$('crypto-history-kind').value==='trades';if(!b)return;const rows=closed?(b.trades||[]):(b.fills||[]),total=closed?rows.length:(b.fills_total??rows.length),pages=Math.max(1,Math.ceil(total/10));c.tradePage=Math.min(c.tradePage,pages-1);
  const head=el('tr');for(const title of (closed?['청산 시각 (KST)','종목','방향','진입가','청산가','수익률','기록']:['진입 시각 (KST)','종목','방향','진입가','점수','기록']))head.append(el('th',title));$('crypto-trade-head').replaceChildren(head);
  table('crypto-trades',closed||b.fills_total==null?rows.slice(c.tradePage*10,c.tradePage*10+10):rows,p=>closed?[time(p.exit_ts),p.base,direction(p.side),num(p.entry),num(p.exit_px),Number.isFinite(p.ret)?pct(p.ret*100):'미확인',origin(p.origin)]:[time(p.ts),p.base,direction(p.side),num(p.entry),num(p.score),origin(p.origin)],closed?'청산 기록 없음':'진입 기록 없음');
  $('crypto-trade-note').textContent=closed?`원본 장부에서 제공하는 최근 ${rows.length}건 · 전체 청산 ${b.n_trades??0}건`:b.history_note||'운영 장부의 진입 기록';
  $('crypto-trade-page').textContent=`${c.tradePage+1} / ${pages} · ${total}건`;$('crypto-trade-prev').disabled=!c.tradePage;$('crypto-trade-next').disabled=c.tradePage>=pages-1;
 }
 async function load(){const seq=++c.seq;c.busy=true;const model=$('crypto-model').value,changed=model!==c.model;c.model=model;if(changed){c.book=null;c.posPage=0;c.tradePage=0;c.symbol='BTC';c.side=null;$('crypto-metrics').replaceChildren();for(const id of ['crypto-positions','crypto-trades'])$(id).replaceChildren();blank('crypto-equity','장부 조회 중');}
  $('crypto-error').hidden=true;$('crypto-status').textContent='장부 조회 중';
  try{const offset=$('crypto-history-kind').value==='fills'?c.tradePage*10:0;
   const result=await Promise.allSettled([get(`/crypto/api/book/${model}?history_offset=${offset}&history_limit=10`),get('/crypto/api/summary'),get('/crypto/api/log?limit=40')]);
   if(seq!==c.seq||!active())return;
   if(result[0].status!=='fulfilled')throw result[0].reason;
   c.book=result[0].value;c.summary=result[1].status==='fulfilled'?result[1].value:null;
   metrics();equity();positions();history();
   $('crypto-log').value=result[2].status==='fulfilled'?result[2].value.lines.join('\n'):'로그 조회 실패';
  }catch(e){if(seq===c.seq){$('crypto-status').textContent='조회 실패 · 새로고침해 주세요';fail(e);}}
  finally{if(seq===c.seq)c.busy=false;}
 }
 $('crypto-model').onchange=()=>load();$('crypto-refresh').onclick=()=>load();
 $('crypto-search').oninput=()=>{c.posPage=0;positions();};$('crypto-sort').onchange=()=>{c.posPage=0;positions();};
 $('crypto-pos-prev').onclick=()=>{c.posPage--;positions();};$('crypto-pos-next').onclick=()=>{c.posPage++;positions();};
 $('crypto-history-kind').onchange=()=>{c.tradePage=0;load();};
 for(const [id,delta] of [['crypto-trade-prev',-1],['crypto-trade-next',1]])$(id).onclick=()=>{c.tradePage+=delta;if($('crypto-history-kind').value==='trades')history();else load();};
 setInterval(()=>{if(active()&&!c.busy)load();},30000);
 return {load,resize:()=>{equity();}};
})();

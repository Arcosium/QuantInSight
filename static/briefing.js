'use strict';
const marketBriefing=(()=>{
 let data=null,sequence=0,category='all',busy=false;
 const categories={all:'전체',kr:'한국',us:'미국',crypto:'크립토',world:'경제·국제'};
 const value=v=>Number.isFinite(v)?new Intl.NumberFormat('ko-KR',{maximumFractionDigits:2}).format(v):'미확인';
 function external(a,url){try{const u=new URL(url);if(['http:','https:'].includes(u.protocol)){a.href=u.href;a.target='_blank';a.rel='noopener noreferrer';}}catch{}}
 function chart(host,item){
  const svg=node('svg',{role:'img','aria-label':item.label+' 최근 3개월 일봉'});host.append(svg);
  const rows=item.points||[],w=Math.max(280,host.clientWidth),h=175,L=6,R=6,T=18,B=29;
  svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
  if(rows.length<2){svg.append(node('text',{x:10,y:75},'시세 연결 대기'));return;}
  let low=Math.min(...rows.map(r=>r.close)),high=Math.max(...rows.map(r=>r.close));const pad=(high-low)*.12||1;low-=pad;high+=pad;
  const x=i=>L+i/(rows.length-1)*(w-L-R),y=v=>T+(high-v)/(high-low)*(h-T-B);
  for(let i=0;i<3;i++)svg.append(node('line',{x1:L,x2:w-R,y1:T+i*(h-T-B)/2,y2:T+i*(h-T-B)/2,class:'grid'}));
  svg.append(node('path',{d:rows.map((r,i)=>`${i?'L':'M'}${x(i)},${y(r.close)}`).join(' '),class:'market-line'}));
  const label=r=>new Date(r.time*1000).toLocaleDateString('ko-KR',{month:'2-digit',day:'2-digit',timeZone:'UTC'});
  svg.append(node('text',{x:L,y:h-4},label(rows[0])),node('text',{x:w-R,y:h-4,'text-anchor':'end'},label(rows.at(-1))));
  rows.forEach((r,i)=>{const dot=node('circle',{cx:x(i),cy:y(r.close),r:5,class:'chart-hit'});dot.append(node('title',{},`${new Date(r.time*1000).toISOString().slice(0,10)} · ${value(r.close)}`));svg.append(dot);});
 }
 function charts(){
  const root=$('market-charts');root.replaceChildren();
  for(const item of data.charts){const article=el('article',undefined,'market-instrument'),head=el('div',undefined,'instrument-heading'),title=el('h2',item.label),unit=el('span',item.id==='btc'?'USD':'지수','caption');head.append(title,unit);article.append(head);
   const line=el('div',undefined,'instrument-value');line.append(el('strong',value(item.latest)),el('span',Number.isFinite(item.change_pct)?`${item.change_pct>0?'+':''}${item.change_pct.toFixed(2)}%`:'시세 확인 필요',item.change_pct>=0?'positive-change':'negative-change'));article.append(line);
   const plot=el('div',undefined,'instrument-chart');article.append(plot);root.append(article);chart(plot,item);
   const meta=el('p',item.as_of?`${tradeTime(item.as_of)} KST · ${item.status==='stale'?'갱신 지연':'최근 관측'}`:item.message,'caption');article.append(meta);
   const source=el('a',item.source,'chart-source');external(source,item.source_url);article.append(source);
  }
 }
 function news(){
  const container=$('today-news');container.replaceChildren();const rows=data.news.items.filter(r=>category==='all'||r.category===category);
  $('news-status').textContent=data.news.message;
  if(!rows.length){container.append(el('p','이 분류에 새로 수집된 기사가 없습니다.','empty-note'));return;}
  for(const item of rows){const article=el('article',undefined,'news-item'),meta=el('p',`${categories[item.category]||'경제'} · ${item.source||'출처 확인'} · ${tradeTime(item.published_at||item.collected_at)}${item.published_at?'':' 수집'}`,'caption'),title=el('h3'),a=el('a',item.title);external(a,item.url);title.append(a);article.append(meta,title);if(item.summary)article.append(el('p',item.summary,'news-summary'));container.append(article);}
 }
 function discoveries(){
  $('discoveries-criteria').textContent=data.discoveries.criteria;
  const root=$('today-discoveries');root.replaceChildren();
  if(!data.discoveries.items.length)root.append(el('p',data.discoveries.status==='unavailable'?'연구 결과 연결을 확인하고 있습니다.':'아직 조건을 통과한 새 전략이 없습니다.','empty-note'));
  for(const p of data.discoveries.items){const a=el('a',undefined,'discovery-item');a.href=`/strategy/${p.id}?market=${p.market}`;a.append(el('span',`${MARKET_NAMES[p.market]||p.market} · ${tradeTime(p.discovered_at)}`,'caption'),el('strong',p.title),el('span',`${percent(p.net_return)} · 샤프 ${decimal(p.sharpe)} · OS 손실 ${p.negative_months}/9개월`,'discovery-metrics'));root.append(a);}
  root.append(el('p',data.discoveries.message,'caption'));
 }
 async function load(){
  const seq=++sequence;busy=true;$('today-refresh').disabled=true;$('today-status').textContent='시장 자료를 불러오는 중입니다.';
  try{const result=await get('/api/briefing');if(seq!==sequence||state.tab!=='today')return;data=result;
   $('today-date').textContent=new Date(data.as_of).toLocaleDateString('ko-KR',{year:'numeric',month:'long',day:'numeric',weekday:'long',timeZone:'Asia/Seoul'});
   $('today-status').textContent=`최근 3개월 일봉 · 변동률은 직전 관측 종가 대비 · ${tradeTime(data.as_of)} KST 조회`;
   charts();news();discoveries();
  }catch(e){if(seq===sequence&&state.tab==='today')$('today-status').textContent=e.message;}finally{if(seq===sequence){busy=false;$('today-refresh').disabled=false;}}
 }
 for(const [key,label] of Object.entries(categories)){const b=el('button',label);b.type='button';b.setAttribute('aria-pressed',String(key===category));b.onclick=()=>{category=key;for(const button of $('news-filters').children)button.setAttribute('aria-pressed',String(button===b));if(data)news();};$('news-filters').append(b);}
 $('today-refresh').onclick=()=>load();
 window.addEventListener('resize',()=>{if(state.tab==='today'&&data)charts();});
 setInterval(()=>{if(state.tab==='today'&&!busy)load();},60000);
 return {load};
})();

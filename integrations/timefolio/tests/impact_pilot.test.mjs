import test from 'node:test';
import assert from 'node:assert/strict';
import {observeOrders} from '../web/static/impact-pilot.mjs';
const T=1700000000000;
function market(){return {venue:'Bybit spot',symbol:'ETHUSDT',tick_size:.01,
  rows:Array.from({length:301},(_,i)=>[T+i*100,99.99,100.01,10,20,1])};}
function order(){return {venue:'Bybit spot',symbol:'ETHUSDT',environment:'live',side:'Buy',order_type:'Market',
  created_ms:T+1000,end_ms:T+1200,status:'Filled',quantity:1,
  fills:[{exec_id:'fixture-1',ts_ms:T+1100,quantity:1,price:100.01}]};}
function run(o=order(),m=market()){return observeOrders(m,{schema_version:1,orders:[o]});}
test('synthetic fixture: observed constant mid is zero, no invented self-impact',()=>{
  const r=run();assert.equal(r.accepted,1);assert.equal(r.results[0].ending.impact_bps,0);
  assert.equal(r.results[0].half_decay_after_end_seconds,null);
  assert.ok(Math.abs(r.results[0].execution_cost_bps-1)<1e-8);
  assert.ok(!JSON.stringify(r).includes('fixture-1'));
});
test('order before-start quote is strictly earlier; created timestamp cannot leak into baseline',()=>{
  const m=market();for(const r of m.rows)if(r[0]>=T+1000){r[1]+=.01;r[2]+=.01;}
  const r=run(order(),m);assert.equal(r.results[0].arrival_mid,100);assert.ok(r.results[0].ending.impact_bps>.99);
});
test('missing recording, demo orders, inconsistent fills and duplicate executions are rejected',()=>{
  for(const o of [ {...order(),created_ms:T-100,end_ms:T}, {...order(),environment:'demo'},
    {...order(),quantity:2},{...order(),fills:[...order().fills,...order().fills]},
    {...order(),end_ms:T+29000}, {...order(),status:'Cancelled'}])assert.equal(run(o).accepted,0);
});
test('snapshot reset and long gaps invalidate observation',()=>{
  const m=market();m.rows[40][5]=2;assert.equal(run(order(),m).accepted,0);
  const gap=market();gap.rows=gap.rows.filter(r=>r[0]<T+2000||r[0]>T+4000);assert.equal(run(order(),gap).accepted,0);
});
test('limit placement distinguished and unfilled cancelled orders remain observable',()=>{
  const o={...order(),order_type:'Limit',limit_price:99.99,status:'Cancelled',fills:[]};
  assert.equal(run(o).results[0].placement,'Best quote');
  assert.equal(run({...o,limit_price:100}).results[0].placement,'Inside spread');
  assert.equal(run(o).results[0].vwap,null);
});
test('no records never claims a real pilot was done',()=>{
  const r=observeOrders(market(),{schema_version:1,orders:[]});assert.equal(r.accepted,0);assert.equal(r.status,'no_observable_orders');
});

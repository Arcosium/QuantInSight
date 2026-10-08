// Offline: node quant/impact_pilot_cli.mjs study.json user-orders.json new-output.json
import {readFileSync,writeFileSync} from 'node:fs';
import {observeOrders} from '../web/static/impact-pilot.mjs';
const [marketPath,ordersPath,outputPath]=process.argv.slice(2);
if(!marketPath||!ordersPath||!outputPath)throw Error('Usage: node quant/impact_pilot_cli.mjs study.json user-orders.json new-output.json');
const source=JSON.parse(readFileSync(marketPath,'utf8'));
const result=observeOrders(source.pilot_market??source,JSON.parse(readFileSync(ordersPath,'utf8')));
writeFileSync(outputPath,JSON.stringify(result,null,2),{flag:'wx',mode:0o600});
console.log(JSON.stringify({accepted:result.accepted,rejected:result.rejected.length,status:result.status}));

"""Add only Autofolio to the existing remote tunnel, behind owner-only Access.

No service restarts, tunnel token changes, publication toggles or other ingress
edits. Cloudflare credentials and backups remain in vault. Default is a plan.
"""
import argparse
import copy
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ACCOUNT='2caf6d4a635bbf686c9fa9522279a28b'
ZONE='2cd0c9cdd4d442112156d0eb4641a114'
TUNNEL='d076ab40-34ee-43fa-b070-ae357076d155'
HOST='autofolio.ai-ve.uk'
HOME=Path.home()


def main(apply=False):
    token=next(line.strip() for line in (HOME/'vault/secrets_api_keys.txt').read_text().splitlines()
               if line.strip().startswith('cfut_'))
    def cf(path,method='GET',body=None):
        request=urllib.request.Request('https://api.cloudflare.com/client/v4'+path,
            method=method,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'},
            data=json.dumps(body).encode() if body is not None else None)
        with urllib.request.urlopen(request,timeout=25) as response:result=json.load(response)
        if not result.get('success'):raise RuntimeError('Cloudflare operation failed: '+method+' '+path)
        return result['result']
    access=f'/accounts/{ACCOUNT}/access/apps'
    apps=cf(access)
    template=next(a for a in apps if a.get('domain')=='quantinsight.ai-ve.uk')
    policies=cf(access+'/'+template['id']+'/policies')
    allow=[p for p in policies if p['decision']=='allow']
    if len(allow)!=1 or not allow[0].get('include'):raise RuntimeError('Owner access template is not unambiguous')
    config_path=f'/accounts/{ACCOUNT}/cfd_tunnel/{TUNNEL}/configurations'
    before=cf(config_path)
    dns_path=f'/zones/{ZONE}/dns_records?name='+urllib.parse.quote(HOST)
    dns=cf(dns_path)
    existing=next((a for a in apps if a.get('domain')==HOST),None)
    if existing:
        actual=cf(access+'/'+existing['id']+'/policies')
        if not any(p.get('decision')=='allow' and p.get('include')==allow[0]['include'] for p in actual):
            raise RuntimeError('Existing Autofolio Access differs; refusing overwrite')
    own=next((r for r in before['config']['ingress'] if r.get('hostname')==HOST),None)
    if own and own.get('service')!='http://127.0.0.1:8997':raise RuntimeError('Existing route differs; refusing overwrite')
    if dns and any(r.get('content')!=TUNNEL+'.cfargotunnel.com' for r in dns):raise RuntimeError('Existing DNS differs; refusing overwrite')
    print(json.dumps(dict(host=HOST,origin='http://127.0.0.1:8997',access='QuantInSight owner-only allow policy',
                          preserve_other_routes=len(before['config']['ingress']),apply=apply)))
    if not apply:return
    backup=HOME/'vault/Autofolio/wiring'/str(time.time_ns())
    backup.mkdir(parents=True,exist_ok=True)
    (backup/'tunnel_before.json').write_text(json.dumps(before,indent=2))
    (backup/'dns_before.json').write_text(json.dumps(dns,indent=2))
    if not existing:
        existing=cf(access,'POST',dict(name='Autofolio private',domain=HOST,type='self_hosted',
                                    session_duration='24h',app_launcher_visible=False))
        p=allow[0]
        policy={k:copy.deepcopy(p[k]) for k in ['name','decision','include','exclude','require','precedence'] if k in p}
        cf(access+'/'+existing['id']+'/policies','POST',policy)
    secured=cf(access+'/'+existing['id']+'/policies')
    if not any(p.get('decision')=='allow' and p.get('include')==allow[0]['include'] for p in secured):
        raise RuntimeError('Owner access could not be verified; routing aborted')
    latest=cf(config_path)
    if latest['version']!=before['version']:raise RuntimeError('Tunnel changed concurrently; rerun against current state')
    config=copy.deepcopy(latest['config'])
    if not own:
        config['ingress'].insert(len(config['ingress'])-1,dict(hostname=HOST,service='http://127.0.0.1:8997'))
        cf(config_path,'PUT',dict(config=config))
    after=cf(config_path)
    preserved=[r for r in after['config']['ingress'] if r.get('hostname')!=HOST]
    prior=[r for r in before['config']['ingress'] if r.get('hostname')!=HOST]
    if preserved!=prior:raise RuntimeError('Other ingress changed; inspect vault backup immediately')
    if not dns:cf(f'/zones/{ZONE}/dns_records','POST',dict(type='CNAME',name=HOST,
                  content=TUNNEL+'.cfargotunnel.com',proxied=True,ttl=1))
    receipt=dict(host=HOST,tunnel_version=after['version'],owner_access_verified=True,other_routes_preserved=True)
    (backup/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true')
    main(parser.parse_args().apply)

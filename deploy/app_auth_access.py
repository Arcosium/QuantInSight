"""Let this host reach its own login page; preserve every other Access application."""
import argparse,json,time,urllib.request
from pathlib import Path
ACCOUNT='2caf6d4a635bbf686c9fa9522279a28b'
HOST='quantinsight.ai-ve.uk'

def main(apply=False):
    vault=Path.home()/'vault'
    token=next(line.strip() for line in (vault/'secrets_api_keys.txt').read_text().splitlines() if line.strip().startswith('cfut_'))
    def cf(path,method='GET',body=None):
        req=urllib.request.Request('https://api.cloudflare.com/client/v4'+path,method=method,
            headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'},data=json.dumps(body).encode() if body is not None else None)
        with urllib.request.urlopen(req,timeout=30) as r:d=json.load(r)
        if not d.get('success'):raise RuntimeError('Cloudflare request failed')
        return d['result']
    base=f'/accounts/{ACCOUNT}/access/apps'
    apps=cf(base);matches=[a for a in apps if a.get('domain')==HOST]
    if len(matches)!=1:raise RuntimeError('Expected one exact-host Access application')
    app=matches[0];route=base+'/'+app['id']+'/policies';policies=cf(route)
    name='QuantInSight application login'
    if any(p.get('name')==name and p.get('decision')=='bypass' for p in policies):
        print('Application login routing already configured');return
    print(json.dumps(dict(host=HOST,action='route to application login',apply=apply)))
    if not apply:return
    # Verify deployed application authentication before opening its login page.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):return None
    try:urllib.request.urlopen('http://127.0.0.1:8500/api/auth/me',timeout=10)
    except urllib.error.HTTPError as e:
        if e.code!=401:raise RuntimeError('Authentication gate did not return 401')
    else:raise RuntimeError('Authentication is not enforced')
    backup=vault/'QuantInSight/backups'/('access-'+str(time.time_ns()));backup.mkdir(parents=True,mode=0o700)
    path=backup/'policies.json';path.write_text(json.dumps(policies));path.chmod(0o600)
    cf(route,'POST',dict(name=name,decision='bypass',include=[{'everyone':{}}],exclude=[],require=[],precedence=0))
    after=cf(route)
    if not any(p.get('name')==name and p.get('decision')=='bypass' for p in after):raise RuntimeError('Policy verification failed')
    print('Exact host now uses application login; previous policies backed up')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');main(p.parse_args().apply)

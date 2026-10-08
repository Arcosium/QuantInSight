"""One owned exploratory GPU lease, bounded queue, recovery, and release.

Packages only frozen research arrays and allowlisted credential-free workers.
The API helper is imported read-only; its shared state writer is never called.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time

import numpy as np

from quant.timefolio_cnn_dataset import sha


def observe_progress(raw, job_names, path):
    """Accept a complete queue observation, otherwise preserve the last one.

    Empty or malformed SSH output is not evidence of remote worker failure.
    A valid snapshot must partition the registered queue exactly; it cannot
    substitute unknown or duplicate jobs to trigger premature recovery/release.
    """
    try:
        progress = json.loads(raw)
        if not isinstance(progress, dict):return None
        done, active, pending = (progress[k] for k in ('completed', 'active', 'pending'))
        if not isinstance(done, list) or not isinstance(active, list) or type(pending) is not int or pending < 0:
            return None
        if any(not isinstance(j, dict) or not isinstance(j.get('name'), str)
               or type(j.get('exit_code')) is not int
               or not isinstance(j.get('seconds'), (int, float))
               or not math.isfinite(j['seconds']) or j['seconds'] < 0 for j in done):return None
        if any(not isinstance(name, str) for name in active):return None
        names = [j['name'] for j in done] + active
        if (len(names) != len(set(names)) or not set(names) <= set(job_names)
                or len(names) + pending != len(job_names)):return None
        payload = json.dumps(progress, indent=2, allow_nan=False)+'\n'
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    path = Path(path);temporary = path.with_suffix('.tmp')
    temporary.write_text(payload)
    temporary.replace(path)
    return progress


def upload_timeout(archive_bytes, bytes_per_second, *, minimum_speed=1024**2,
                   reserve_for_minimum=False):
    """Reject unusable routes; budget two observed transfer times plus startup."""
    if (not isinstance(archive_bytes, int) or archive_bytes <= 0
            or not math.isfinite(minimum_speed) or minimum_speed <= 0
            or not math.isfinite(bytes_per_second) or bytes_per_second < minimum_speed):
        raise ValueError(f'Upload speed {bytes_per_second}B/s is below required {minimum_speed}B/s')
    estimated = math.ceil(2*archive_bytes/bytes_per_second+60)
    # A short probe can catch a burst rate. A compressed archive also reserves
    # time for the admitted minimum rate, within the same30minute upper bound.
    if reserve_for_minimum:
        estimated = max(estimated, math.ceil(archive_bytes/minimum_speed+60))
    return min(1800, max(300, estimated))


def probe_upload(ssh_args, *, minimum_speed=1024**2):
    """Measure the actual encrypted path using harmless bytes, before large data."""
    payload = b'R'*(8*1024**2)
    code = 'import sys; print(len(sys.stdin.buffer.read()))'
    started = time.monotonic()
    result = subprocess.run(ssh_args+['python -c '+shlex.quote(code)], input=payload,
                            capture_output=True, check=True, timeout=30)
    elapsed = time.monotonic()-started
    if result.stdout.strip() != str(len(payload)).encode() or elapsed <= 0:
        raise ValueError('Upload probe acknowledgement did not match')
    speed = len(payload)/elapsed
    upload_timeout(len(payload), speed, minimum_speed=minimum_speed)
    return dict(bytes=len(payload), seconds=elapsed, bytes_per_second=speed)


def runtime_dependency_code(requirements):
    """Bootstrap the disposable GPU user site without changing OS packages."""
    if set(requirements) != {'pandas','python-dateutil','pytz','tzdata','six'}:
        raise ValueError('Only registered account-runtime dependencies are allowed')
    if any(not re.fullmatch(r'[0-9A-Za-z.+-]+', version) for version in requirements.values()):
        raise ValueError('Exact dependency versions required')
    return '''import importlib,importlib.metadata,json,site,subprocess,sys
expected=json.loads('''+repr(json.dumps(requirements))+''')
assert site.ENABLE_USER_SITE, 'A disposable GPU user site is required'
def version(name):
    try:return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:return None
if any(version(name)!=value for name,value in expected.items()):
    target=site.getusersitepackages()
    result=subprocess.run([sys.executable,'-m','pip','install','--no-cache-dir',
        '--disable-pip-version-check','--no-deps','--upgrade','--target',target,
        '--index-url','https://pypi.org/simple']+[name+'=='+value for name,value in expected.items()],
        capture_output=True,text=True)
    if result.returncode:
        print(result.stderr[-2000:],file=sys.stderr);sys.exit(result.returncode)
    if target in sys.path:sys.path.remove(target)
    sys.path.insert(0,target);importlib.invalidate_caches()
import pandas,torch,numpy
assert torch.cuda.is_available()
assert all(version(name)==value for name,value in expected.items())
print(json.dumps(dict(dependencies={name:version(name) for name in expected},
                     numpy=numpy.__version__,torch=torch.__version__,cuda=True)))
'''


def run(dataset, output, name, *, gpu_types=None, cloud='COMMUNITY', maximum_hourly_cost=.30,
        jobs_plan=None, account_panel=None, compress_input=False):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Lease directory already exists; inspect it instead of creating twice')
    if (dataset/'RETIRED.json').exists():
        raise ValueError('Refusing to lease a GPU for a retired dataset')
    manifest = json.loads((dataset/'manifest.json').read_text())
    if manifest.get('exploratory_ready') is not True or manifest.get('contest_certified') is not False:
        raise ValueError('An explicit frozen exploratory package is required')
    output.mkdir(parents=True)
    jobs = [dict(name=f'{fold}_{objective}_17', fold=fold, objective=objective, seed=17)
        for fold in ['202604', '202401', '202501', '202608'] for objective in ['listnet', 'mse']]
    parallel=2
    if jobs_plan:
        registered=json.loads(Path(jobs_plan).read_text())
        if registered['dataset_manifest_sha256'] != sha(dataset/'manifest.json'):
            raise ValueError('Registered batch belongs to another dataset')
        jobs,parallel=registered['jobs'],registered['parallel']
        if not isinstance(parallel,int) or not 1<=parallel<=8 or not jobs:
            raise ValueError('Bounded positive worker count and jobs required')
        if len({j['name'] for j in jobs})!=len(jobs):
            raise ValueError('Duplicate registered model jobs')
        for job in jobs:
            if (not re.fullmatch(r'[a-zA-Z0-9_]+',job['name']) or job['seed'] not in [17,29,43]
                    or job['objective'] not in ['mse','bce','listnet','topk_ce']):
                raise ValueError('Unregistered model configuration')
    if not {j['fold'] for j in jobs} <= {f['id'] for f in manifest['folds']}:
        raise ValueError('Requested folds unavailable')
    from quant.timefolio_cnn_batch import policy_mode
    for job in jobs:
        policy_mode(job, account_panel=account_panel)
    target_name = manifest.get('training_target', 'returns')
    if target_name not in ('returns','training_utility'):
        raise ValueError('Unregistered training target')
    spec = manifest['arrays'][target_name]; target_path = (dataset/spec['path']).resolve()
    if not target_path.is_relative_to(dataset) or sha(target_path) != spec['sha256']:
        raise ValueError('Training target changed before rental')
    s, end, eligible = [np.load(dataset/(k+'.npy'), mmap_mode='r') for k in
        ['signal_index','label_end_index','eligible']]
    y = np.load(target_path, mmap_mode='r')
    for fold in manifest['folds']:
        if fold['id'] not in {j['fold'] for j in jobs}:
            continue
        valid = eligible & np.isfinite(y)
        tr = valid & (end < fold['validation_start'])
        va = valid & (s >= fold['validation_start']) & (end < fold['test_start'])
        pred = eligible & (s >= fold['test_start']) & (s < fold['test_end'])
        if len(np.unique(s[tr])) < 126 or len(np.unique(s[va])) < 20 or not pred.any():
            raise ValueError('Insufficient fold coverage before rental: '+fold['id'])
    config = dict(dataset='dataset/manifest.json', output='results', parallel=parallel, jobs=jobs)
    panel_files = []
    if account_panel is not None:
        if jobs_plan is None:
            raise ValueError('Register the account selection before rental')
        account_panel = Path(account_panel).resolve()
        proof_path = account_panel/'receipt.json'
        proof = json.loads(proof_path.read_text())
        if (proof['dataset_manifest_sha256'] != sha(dataset/'manifest.json')
                or registered.get('account_panel_receipt_sha256') != sha(proof_path)):
            raise ValueError('Account selection package differs from registered inputs')
        for job in jobs:
            if job.get('rebalance') not in [5,20] or job.get('rank_buffer') not in [0,5]:
                raise ValueError('Unregistered account execution parameters')
        for filename, digest in proof['artifacts'].items():
            path = (account_panel/filename).resolve()
            if not path.is_relative_to(account_panel) or sha(path) != digest:
                raise ValueError('Account panel changed before rental')
            panel_files.append(path)
        panel_files.append(proof_path)
        config['account_panel'] = 'account_panel'
    elif jobs_plan and registered.get('account_panel_receipt_sha256'):
        raise ValueError('Registered profit selection cannot silently fall back to a price metric')
    dependencies = {name:importlib.metadata.version(name) for name in
                    ['pandas','python-dateutil','pytz','tzdata','six']} if panel_files else {}
    config['runtime_dependencies'] = dependencies
    (output/'jobs.json').write_text(json.dumps(config, indent=2)+'\n')
    archive = output/'input.tar'
    with tarfile.open(archive, 'w') as tar:
        provenance=[dataset/'manifest.json',dataset/'cohort.json']
        if (dataset/'price_reference_hashes.json').exists():provenance.append(dataset/'price_reference_hashes.json')
        for path in provenance:
            tar.add(path, arcname='dataset/'+path.name)
        for spec in manifest['arrays'].values():
            path = (dataset/spec['path']).resolve()
            if not path.is_relative_to(dataset) or sha(path) != spec['sha256']:
                raise ValueError('Frozen dataset hash mismatch')
            tar.add(path, arcname='dataset/'+path.name)
        workers = ['timefolio_cnn_core.py', 'timefolio_cnn_train.py', 'timefolio_cnn_batch.py']
        if panel_files:
            workers += ['timefolio_cnn_account_selection.py', 'timefolio_cnn_account.py',
                        'timefolio_cnn_dataset.py', 'timefolio_heatmap_locked_weighted_replay.py',
                        'timefolio_heatmap_replay.py', 'timefolio_heatmap_retention_replay.py',
                        'timefolio_heatmap_locked_targets.py']
            if any(job.get('policy_selection') == 'retention_grid' for job in jobs):
                workers.append('timefolio_cnn_policy_selection.py')
        for filename in workers:
            tar.add(Path(__file__).with_name(filename), arcname='quant/'+filename)
        for path in panel_files:
            tar.add(path, arcname='account_panel/'+path.name)
        entry = tarfile.TarInfo('quant/__init__.py'); entry.size = 0; tar.addfile(entry, io.BytesIO(b''))
        tar.add(output/'jobs.json', arcname='jobs.json')
    wire_archive = archive
    if compress_input:
        from quant.timefolio_cnn_transfer import compress_archive
        wire_archive = compress_archive(archive)
    key = output/'ssh_key'
    subprocess.run(['ssh-keygen', '-t', 'ed25519', '-f', str(key), '-N', '', '-q'], check=True)
    sys.path.insert(0, str(Path.home()/'projects/HYFE_QTPA'))
    from hyfe.runpod import rest, gql
    existing = rest('GET', '/pods')
    if any(p.get('name') == name for p in existing):
        raise ValueError('An identically named pod exists; reconcile before creating another')
    quote = gql('{ gpuTypes { id lowestPrice(input: {gpuCount: 1, secureCloud: '+str(cloud=='SECURE').lower()+'}) { uninterruptablePrice stockStatus } } }')
    gpu_types = gpu_types or ['NVIDIA RTX A4000', 'NVIDIA RTX A4500', 'NVIDIA GeForce RTX 3090']
    choices = []
    for p in quote.get('data', {}).get('gpuTypes', []):
        price = (p.get('lowestPrice') or {}).get('uninterruptablePrice')
        if p['id'] in gpu_types and price and price <= maximum_hourly_cost:
            choices.append((float(price), p['id']))
    if not choices:
        raise RuntimeError('No current affordable offer; package is ready for later rental')
    (output/'quote.json').write_text(json.dumps(choices)+'\n')
    lease = dict(name=name, created_at=time.time(), pod_id=None, state='creating',
        dataset_manifest_sha256=sha(dataset/'manifest.json'), input_sha256=sha(archive),
        existing_pod_ids=[p['id'] for p in existing], maximum_hourly_cost=maximum_hourly_cost, cloud=cloud)
    lease['transport'] = dict(compression='zstd-1' if compress_input else 'none',
        raw_bytes=archive.stat().st_size, wire_bytes=wire_archive.stat().st_size,
        wire_sha256=sha(wire_archive))
    def save():
        tmp = output/'lease.tmp'; tmp.write_text(json.dumps(lease, indent=2)+'\n'); tmp.replace(output/'lease.json')
    save(); pod = None; ssh_args = None
    def ssh(command, *, timeout=30, check=True):
        return subprocess.run(ssh_args+[command], capture_output=True, text=True, timeout=timeout, check=check)
    def recover():
        with (output/'results.tar').open('wb') as target:
            subprocess.run(ssh_args+['tar -cf - -C /workspace/cnn4y results batch.log jobs.json'], stdout=target, check=True, timeout=300)
        with tarfile.open(output/'results.tar') as tar:
            tar.extractall(output/'recovered', filter='data')
    try:
        for price, gpu in sorted(choices):
            try:
                pod = rest('POST', '/pods', dict(name=name, imageName='runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404',
                    cloudType=cloud, gpuCount=1, gpuTypeIds=[gpu], containerDiskInGb=30,
                    volumeInGb=0, ports=['22/tcp'], supportPublicIp=True,
                    minDownloadMbps=500, minUploadMbps=100,
                    minRAMPerGPU=16, minVCPUPerGPU=parallel, env={'PUBLIC_KEY':key.with_suffix('.pub').read_text().strip()}))
                break
            except Exception:
                # A failed POST may still have created a resource: GET before retry.
                matching = [p for p in rest('GET', '/pods') if p.get('name') == name]
                if len(matching) > 1:
                    raise RuntimeError('Ambiguous owned lease after creation failure')
                if matching:
                    pod = matching[0]; break
        if pod is None:
            raise RuntimeError('Quoted offers unavailable; no GPU leased')
        lease.update(pod_id=pod['id'], cost_per_hour=pod.get('costPerHr'), gpu=gpu, state='starting'); save()
        if pod.get('costPerHr') is None or float(pod['costPerHr']) > lease['maximum_hourly_cost']:
            raise RuntimeError('Actual lease cost outside registered budget')
        deadline = time.monotonic()+600
        while time.monotonic() < deadline:
            p = rest('GET', '/pods/'+pod['id']); ip = p.get('publicIp'); port = (p.get('portMappings') or {}).get('22')
            if ip and port:
                ssh_args = ['ssh', '-i', str(key), '-p', str(port), '-o', 'IdentitiesOnly=yes',
                    '-o', 'Compression=no',
                    '-o', 'StrictHostKeyChecking=accept-new', '-o', 'UserKnownHostsFile='+str(output/'known_hosts'),
                    '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', 'root@'+ip]
                try:
                    probe = ssh('python -c '+shlex.quote('import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'), timeout=20)
                    lease['cuda_probe'] = probe.stdout.strip(); break
                except (subprocess.SubprocessError, OSError):
                    pass
            time.sleep(10)
        else:
            raise TimeoutError('GPU startup deadline reached')
        lease.update(state='checking_upload_route'); save()
        minimum_speed = 256*1024 if compress_input else 1024**2
        lease['upload_probe'] = probe_upload(ssh_args, minimum_speed=minimum_speed)
        lease['upload_timeout_seconds'] = upload_timeout(
            wire_archive.stat().st_size, lease['upload_probe']['bytes_per_second'],
            minimum_speed=minimum_speed, reserve_for_minimum=compress_input)
        lease['transport']['minimum_speed_bytes_per_second'] = minimum_speed
        save()
        if dependencies:
            lease.update(state='preparing_account_runtime', runtime_dependencies=dependencies); save()
            prepared = ssh('python -c '+shlex.quote(runtime_dependency_code(dependencies)), timeout=180)
            lease['verified_runtime'] = json.loads(prepared.stdout); save()
        lease.update(state='uploading', ssh_host=ip, ssh_port=port); save()
        ssh('mkdir -p /workspace/cnn4y')
        upload_command = ('cat > /workspace/cnn4y.input.tar.zst' if compress_input
                          else 'tar -xf - -C /workspace/cnn4y')
        with wire_archive.open('rb') as data:
            subprocess.run(ssh_args+[upload_command], stdin=data, check=True,
                           timeout=lease['upload_timeout_seconds'])
        if compress_input:
            from quant.timefolio_cnn_transfer import decoder_code
            lease['state'] = 'verifying_lossless_input'; save()
            decoded = ssh('python -c '+shlex.quote(decoder_code(
                archive.stat().st_size, lease['input_sha256'], lease['transport']['wire_sha256'])),
                timeout=180)
            lease['transport']['remote_verification'] = json.loads(decoded.stdout); save()
        launcher = """import os,subprocess,sys
os.chdir('/workspace/cnn4y')
env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
with open('batch.log','w') as log:
    p=subprocess.Popen([sys.executable,'-u','-m','quant.timefolio_cnn_batch','--config','jobs.json'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
print(p.pid,flush=True)
"""
        try:
            launch = ssh('python -c '+shlex.quote(launcher), timeout=30)
            lease['remote_batch_pid'] = int(launch.stdout.strip())
        except subprocess.TimeoutExpired:
            # A lost acknowledgement does not mean remote execution failed.
            # Observe the fixed results path; never submit a duplicate batch.
            lease['launch_acknowledgement_timeout'] = True
        lease.update(state='training', training_started_at=time.time()); save()
        deadline = time.monotonic()+5400
        while time.monotonic() < deadline:
            try:
                check = ssh('cd /workspace/cnn4y && cat results/progress.json', check=False)
            except (subprocess.TimeoutExpired, OSError):
                time.sleep(10); continue
            if check.returncode == 0:
                progress = observe_progress(check.stdout, [j['name'] for j in jobs], output/'progress.json')
                if progress is None:
                    lease['progress_observation_retries'] = lease.get('progress_observation_retries', 0)+1
                    save()
                elif len(progress['completed']) == len(jobs):
                    lease['completed_jobs'] = progress['completed']; break
            time.sleep(20)
        else:
            raise TimeoutError('Registered GPU batch exceeded 90 minutes')
        lease['state'] = 'recovering'; save()
        recover()
        for job in jobs:
            receipt = output/'recovered/results'/job['name']/'receipt.json'
            if not receipt.exists():
                continue  # failed jobs retain their logs and exit statuses
            proof = json.loads(receipt.read_text())
            if proof['dataset_manifest_sha256'] != lease['dataset_manifest_sha256']:
                raise ValueError('Recovered model references a different dataset')
            for filename, digest in proof['artifacts'].items():
                if Path(filename).name != filename or sha(receipt.parent/filename) != digest:
                    raise ValueError('Recovered artifact hash mismatch')
        lease['artifacts_recovered'] = True; lease['results_sha256'] = sha(output/'results.tar'); save()
    except Exception as exc:
        lease['error'] = type(exc).__name__+': '+str(exc)[:300]; save()
        if ssh_args is not None and not lease.get('artifacts_recovered'):
            try:
                recover(); lease['partial_artifacts_recovered'] = True; save()
            except Exception as recovery_error:
                lease['recovery_error_type'] = type(recovery_error).__name__; save()
        raise
    finally:
        if pod is not None:
            existing = [p for p in rest('GET', '/pods') if p['id'] == pod['id']]
            if existing:
                p = existing[0]
                if p.get('name') != name or p['id'] in lease['existing_pod_ids']:
                    raise RuntimeError('Ownership mismatch; refusing pod deletion')
                rest('DELETE', '/pods/'+pod['id'])
            else:
                lease['already_released_externally'] = True
            if any(p['id'] == pod['id'] for p in rest('GET', '/pods')):
                raise RuntimeError('Pod release could not be verified')
            lease.update(state='released', released_at=time.time()); save()
        print(json.dumps(lease), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--gpu', action='append')
    ap.add_argument('--cloud', choices=['COMMUNITY','SECURE'], default='COMMUNITY')
    ap.add_argument('--maximum-hourly-cost',type=float,default=.30)
    ap.add_argument('--jobs-plan',type=Path)
    ap.add_argument('--account-panel',type=Path)
    ap.add_argument('--compress-input',action='store_true')
    args = ap.parse_args(); run(args.dataset, args.output, args.name,
        gpu_types=args.gpu, cloud=args.cloud, maximum_hourly_cost=args.maximum_hourly_cost,
        jobs_plan=args.jobs_plan,account_panel=args.account_panel,compress_input=args.compress_input)

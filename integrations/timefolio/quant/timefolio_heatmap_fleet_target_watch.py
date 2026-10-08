"""Record every user-threshold match; freeze only owned search on a stop match."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import tarfile
import time
from quant.timefolio_heatmap_fleet_thresholds import classify, is_search_process


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_name(path.name + '.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def scan(followup, output):
    seen_path, candidates_path = output / 'scanned_cases.json', output / 'candidates.json'
    seen = read(seen_path) if seen_path.exists() else {}
    candidates = read(candidates_path) if candidates_path.exists() else {}
    for case, receipt in read(followup / 'recovered_cases.json').items():
        if case in seen:
            assert seen[case]['audit_manifest_sha256'] == receipt['audit_manifest_sha256']
            continue
        source = followup / 'recovered' / case
        assert sha(source / 'artifact_hashes.json') == receipt['audit_manifest_sha256']
        hashes = read(source / 'artifact_hashes.json')
        for name in ['summary.json', 'paper_proximity.json']:
            assert sha(source / name) == hashes[name]
        complete = read(source / 'complete.json')
        assert complete['accounts'] == 128 and complete['maximum_nav_error'] < .01
        assert complete['source_account_manifest_sha256'] == receipt['account_manifest_sha256']
        summary = {r['id']: r for r in read(source / 'summary.json')}
        proximity = read(source / 'paper_proximity.json')
        assert len(proximity) == 128 and set(summary) == {r['id'] for r in proximity}
        for row in proximity:
            classified = classify(row['monthly_target'])
            if not classified['retain_candidate']: continue
            key = case + '/' + row['id']
            candidates[key] = dict(at=time.time(), case=case, id=row['id'], pod_index=receipt['pod_index'],
                **classified, monthly=row['monthly_target'], account=summary[row['id']],
                audit_manifest_sha256=receipt['audit_manifest_sha256'],
                account_manifest_sha256=receipt['account_manifest_sha256'],
                compliance_and_statistical_review_pending=True)
        # Candidate write precedes the cursor: interruption cannot lose a match.
        write(candidates_path, candidates)
        seen[case] = dict(at=time.time(), audit_manifest_sha256=receipt['audit_manifest_sha256'])
        write(seen_path, seen)
    if not candidates_path.exists(): write(candidates_path, candidates)
    return candidates, seen


def process_identity(pid):
    proc = Path('/proc') / str(pid)
    if not proc.exists(): return None
    argv = [v.decode() for v in (proc / 'cmdline').read_bytes().split(b'\0') if v]
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=fields[19], argv=argv)


def connection(root):
    pod = read(root / 'pod.json')
    assert pod['name'] == 'arctrade-lab-' + root.resolve().name
    common = ['-i', str(root / 'research_key'), '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new',
              '-o', 'UserKnownHostsFile=' + str(root / 'known_hosts'), '-o', 'ConnectTimeout=10']
    return (['ssh', *common, '-p', str(pod['port']), 'root@' + pod['ip']],
            ['scp', *common, '-P', str(pod['port'])], pod['ip'])


def command(args, timeout=120):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout).stdout


def freeze_pod(root, remote, output, candidates, stress_source, raw_study):
    ssh, scp, ip = connection(root)
    # Freeze the queue first, then its GPU workers. CPU account/audit work can finish.
    code = 'import os,json,signal,time\nfrom pathlib import Path\n' + inspect.getsource(is_search_process)
    code += '\nremote=' + repr(remote) + '\nstopped=set()\n'
    code += "for phase in ['queue','workers']:\n for p in Path('/proc').iterdir():\n  if not p.name.isdigit():continue\n  try:\n   argv=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\\0') if v]\n   cwd=str((p/'cwd').resolve())\n   if not is_search_process(argv,cwd,remote):continue\n   if phase=='queue' and argv[1]=='worker.py':continue\n   os.kill(int(p.name),signal.SIGSTOP);stopped.add(int(p.name))\n  except (FileNotFoundError,ProcessLookupError,PermissionError):pass\nprint(json.dumps(dict(at=time.time(),paused_pids=sorted(stopped),only_search_paused=True)))\n"
    receipt = json.loads(command([*ssh, 'python3 -c ' + shlex.quote(code)]))
    write(output / (root.name + '_paused.json'), receipt)
    if candidates:
        # A just-completed source Pod may already have been released. Move only
        # its hash-verified account artifacts to a remaining owned validation host.
        for case in sorted({row['case'] for row in candidates}):
            target = remote + '/accounts/' + case
            exists = command([*ssh, 'test -d ' + shlex.quote(target) + ' && echo yes || echo no']).strip()
            if exists == 'yes': continue
            source = raw_study / 'recovered/accounts' / case
            expected = next(row['account_manifest_sha256'] for row in candidates if row['case'] == case)
            assert sha(source / 'artifact_hashes.json') == expected
            archive = output / (root.name + '_' + case + '.tar')
            with tarfile.open(archive, 'w') as stream:
                stream.add(source, arcname='accounts/' + case)
            remote_archive = '/tmp/' + archive.name
            command([*scp, str(archive), 'root@' + ip + ':' + remote_archive], 300)
            assert command([*ssh, 'sha256sum ' + shlex.quote(remote_archive)]).split()[0] == sha(archive)
            command([*ssh, 'cd ' + shlex.quote(remote) + ' && tar -xf ' + shlex.quote(remote_archive)])
        request = dict(cases={})
        for row in candidates:
            request['cases'].setdefault(row['case'], []).append(row['id'])
        path = output / (root.name + '_validation_request.json'); write(path, request)
        remote_source = '/workspace/timefolio_candidate_stress_v2.py'
        remote_request = '/workspace/candidate_validation_request_v2.json'
        command([*scp, str(stress_source), 'root@' + ip + ':' + remote_source])
        command([*scp, str(path), 'root@' + ip + ':' + remote_request])
        actual = command([*ssh, 'sha256sum ' + remote_source]).split()[0]
        assert actual == sha(stress_source)
        overlay = '/workspace/arctrade_fleet_validation_v1'
        target = '/workspace/arctrade_candidate_checks_v2'
        shell = (f'cd {overlay} && PYTHONPATH={overlay}:{remote} OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '
                 f'nice -n15 python3 {remote_source} --root {remote} --request {remote_request} --output {target} '
                 '> /workspace/candidate_stress_v2.log 2>&1; echo $? > /workspace/candidate_stress_v2.exit')
        command([*ssh, '(nohup bash -c ' + shlex.quote(shell) + ' < /dev/null > /dev/null 2>&1 &)'])
        write(output / (root.name + '_validation_started.json'), dict(at=time.time(), **request,
            source_sha256=actual, source=str(stress_source), remote_output=target))
    return receipt


def trigger(registration, output, candidates):
    stop_rows = [r for r in candidates.values() if r['stop_search_and_validate']]
    assert stop_rows
    write(output / 'stop_requested.json', dict(at=time.time(), candidates=stop_rows,
        reason='pooled_sharpe_gt_3_and_every_month_gt_1_5', new_exploration_authorized=False,
        next_phase='frozen_candidate_additional_validation', final_validation_passed=False))
    expected = registration['controller_identity']
    actual = process_identity(expected['pid'])
    if actual is not None:
        assert actual == expected, 'Controller PID identity changed; refuse unrelated signal'
        os.kill(actual['pid'], signal.SIGSTOP)
    write(output / 'coordinator_paused.json', dict(at=time.time(), identity=actual,
        was_already_exited=actual is None))
    pool = Path(registration['pool']); failures = []
    live = {}
    for index in range(10, 20):
        root = pool / f'pod{index:02d}'
        if (root / 'pod.json').exists() and not (root / 'deletion.json').exists(): live[index] = root
    matching = {index: [] for index in live}
    for row in stop_rows:
        if not live:
            failures.append(dict(index=row['pod_index'], error='All source Pods released; provision validation host'))
            continue
        target_index = row['pod_index'] if row['pod_index'] in live else next(iter(live))
        matching[target_index].append(row)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(freeze_pod, root, registration['remote'], output, matching[index],
                    Path(registration['stress_source']), Path(registration['raw_study'])): index
                   for index, root in live.items()}
        for future, index in futures.items():
            try: future.result()
            except Exception as exc: failures.append(dict(index=index, error=type(exc).__name__))
    write(output / 'validation_phase.json', dict(at=time.time(), stopped_search=True,
        failures_requiring_review=failures, candidates=len(stop_rows), final_validation_passed=False,
        next_required=['recover stress outputs', 'audit timing and complete contest constraints',
                       'joint multiple-testing review including all searched hypotheses',
                       'heldout or forward confirmation', 'archive paused work and release surplus Pods']))
    if failures: raise RuntimeError('Some owned search processes need freeze/validation recovery')


def run(registration_path, once=False):
    registration = read(registration_path)
    for path, expected in registration['source_hashes'].items(): assert sha(Path(path)) == expected
    output = Path(registration['output']); output.mkdir(exist_ok=True)
    lock = (output / 'watch.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (output / 'stop_requested.json').exists(), 'Existing stop request needs main review'
    while True:
        candidates, seen = scan(Path(registration['followup']), output)
        stops = [r for r in candidates.values() if r['stop_search_and_validate']]
        write(output / 'status.json', dict(at=time.time(), cases=len(seen), accounts=len(seen)*128,
            recorded=len(candidates), stop_matches=len(stops), phase='validation' if stops else 'search',
            registration_sha256=sha(registration_path)))
        if stops:
            trigger(registration, output, candidates); return
        if once: return
        time.sleep(15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--once', action='store_true'); args = parser.parse_args()
    run(args.registration, args.once)

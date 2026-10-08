"""Run only this research child inside a bounded, low-priority user cgroup."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'): return int(line.split()[1]) * 1024
    raise RuntimeError('Cannot read available memory')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--receipt', type=Path, required=True)
    ap.add_argument('command', nargs=argparse.REMAINDER); args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command: raise ValueError('Missing research command')
    if args.receipt.exists(): raise RuntimeError('Use a fresh receipt; do not overwrite prior runs')
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    scope = 'arctrade-heatmap-' + time.strftime('%Y%m%d-%H%M%S') + '.scope'
    env = dict(os.environ, XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}',
               OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    gate = args.receipt.with_suffix('.ready')
    # Child imports no numerical libraries until the scope and limits are verified.
    child_code = '''import os,sys,time
from pathlib import Path
os.nice(15); os.sched_setaffinity(0,{16,17,18,19})
Path('/proc/self/oom_score_adj').write_text('900')
gate=Path(sys.argv[1])
for _ in range(600):
    if gate.exists(): break
    time.sleep(.1)
else: raise RuntimeError('Resource gate never opened')
os.execvpe(sys.argv[2],sys.argv[2:],os.environ)
'''
    while available() < 24 * 1024**3:
        print(json.dumps({'waiting_for_memory': True}), flush=True); time.sleep(10)
    child = subprocess.Popen([sys.executable, '-c', child_code, str(gate), *command], env=env)
    subprocess.run(['busctl', '--user', 'call', 'org.freedesktop.systemd1', '/org/freedesktop/systemd1',
                    'org.freedesktop.systemd1.Manager', 'StartTransientUnit', 'ssa(sv)a(sa(sv))',
                    scope, 'fail', '6', 'PIDs', 'au', '1', str(child.pid),
                    'MemoryHigh', 't', str(5*1024**3), 'MemoryMax', 't', str(8*1024**3),
                    'MemorySwapMax', 't', '0', 'CPUQuotaPerSecUSec', 't', '3000000',
                    'CPUWeight', 't', '10', '0'], env=env, check=True, stdout=subprocess.DEVNULL)
    for _ in range(100):
        cgroup = Path(f'/proc/{child.pid}/cgroup').read_text()
        if scope in cgroup: break
        time.sleep(.1)
    else: raise RuntimeError('Child never entered resource scope; gate stays closed')
    relative = cgroup.strip().split('::', 1)[1]
    base = Path('/sys/fs/cgroup') / relative.lstrip('/')
    limits = {k: (base/k).read_text().strip() for k in ['memory.high','memory.max','memory.swap.max','cpu.max']}
    assert limits['memory.max'] == str(8*1024**3) and limits['memory.swap.max'] == '0'
    quota, period = map(int, limits['cpu.max'].split()); assert quota / period <= 3
    subprocess.run(['ionice', '-c2', '-n7', '-p', str(child.pid)], check=True)
    receipt = {'pid': child.pid, 'scope': scope, 'command': command, 'limits': limits,
               'cpu_affinity': [16,17,18,19], 'nice': 15, 'oom_score_adj': 900,
               'memory_pause_below_gib': 16, 'memory_resume_above_gib': 24,
               'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'status': 'running'}
    args.receipt.write_text(json.dumps(receipt, indent=2)+'\n'); gate.touch()
    print(json.dumps({'research_started': receipt}), flush=True)
    frozen = False; peak = 0
    while child.poll() is None:
        free = available()
        action = 'freeze' if not frozen and free < 16*1024**3 else 'thaw' if frozen and free > 24*1024**3 else None
        if action:
            subprocess.run(['systemctl', '--user', action, scope], env=env, check=True)
            frozen = action == 'freeze'; print(json.dumps({'research_scope_action': action}), flush=True)
        if (base/'memory.current').exists(): peak = max(peak,int((base/'memory.current').read_text()))
        time.sleep(10)
    receipt.update(status='complete' if child.returncode == 0 else 'failed', exit_code=child.returncode,
                   observed_peak_memory_bytes=peak, finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    args.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
    gate.unlink(missing_ok=True)
    print(json.dumps({'research_finished': receipt}), flush=True)
    return child.returncode


if __name__ == '__main__': sys.exit(main())

"""Join one extra shard to an existing Pod and wait for both before recovery."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value)+'\n');temp.replace(path)


def run(root,seed):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);marker=root/'exit_code'
    # The original launcher writes its exit status here. A FIFO prevents the
    # controller's regular-file completion check until both jobs are done.
    os.mkfifo(marker)
    primary=[]
    def receive():
        with marker.open() as stream:primary.append(int(stream.read().strip()))
    reader=threading.Thread(target=receive,daemon=True);reader.start()
    started=time.time();child=None;secondary=1
    try:
        while not (root/'out/runtime.json').exists():
            if primary or time.time()-started>300:raise RuntimeError('Primary worker did not initialize')
            time.sleep(1)
        out=root/'out'/f'extra_seed{seed}';assert not out.exists()
        with (root/'out'/f'companion_seed{seed}.log').open('x') as log:
            child=subprocess.Popen([sys.executable,str(root/'worker.py'),'--root',str(root),
                '--out',str(out),'--seed',str(seed),'--device','cuda'],stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
            atomic(root/'out/companion_status.json',dict(state='parallel_training',seed=seed,pid=child.pid))
            secondary=child.wait(timeout=6300)
        reader.join(timeout=max(1,6500-(time.time()-started)))
        if not primary:raise TimeoutError('Primary exit status missing')
        combined=primary[0] if primary[0] else secondary
        atomic(root/'out/companion_status.json',dict(state='both_finished',seed=seed,
            primary_exit=primary[0],secondary_exit=secondary,combined_exit=combined))
    except Exception as exc:
        combined=1
        (root/'companion_error.json').write_text(json.dumps(dict(error=str(exc),secondary_exit=secondary)))
    finally:
        # Replace the FIFO atomically with the aggregate status. The existing
        # owner's unchanged deadline watchdog is still responsible for the Pod.
        completed=root/'combined_exit_code';completed.write_text(str(combined)+'\n');completed.replace(marker)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True);a=parser.parse_args();run(a.root,a.seed)

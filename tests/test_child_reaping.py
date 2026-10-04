"""fire-and-forget Popen 이 좀비(<defunct>)로 남지 않는지 — server/app.py 의 SIGCHLD 정책.

2026-08-02: 운용지원 워커(python3.11)가 uvicorn 밑에 1일 6시간 defunct 로 상주하던 건의 회귀 가드.
"""
import pathlib
import subprocess
import sys

APP = pathlib.Path(__file__).resolve().parent.parent / "server" / "app.py"

_PROBE = """
import os, signal, subprocess, sys, time
signal.signal(signal.SIGCHLD, signal.SIG_IGN)
subprocess.Popen([sys.executable, '-c', 'pass'], start_new_session=True)
time.sleep(1.5)
try:
    os.waitpid(-1, 0)
    sys.exit('좀비가 남았다')
except ChildProcessError:
    pass
"""


def test_server_installs_sigchld_ign():
    assert "signal.signal(signal.SIGCHLD, signal.SIG_IGN)" in APP.read_text(encoding="utf-8")


def test_sigchld_ign_actually_reaps():
    p = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr

"""이미지 기반 전략(heatf 소형 CNN)의 KRX 백테스트 — 결과 열람과 재평가.
학습(Runpod, hyfe/fullkrx.sh)은 HYFE_QTPA 쪽 CLI 가 하고, 여기서는 그 예측 파일(work/results/krx_*_pred.npz)로
코호트 백테스트를 다시 돌리고(비용·분위·롱온리 조정) 결과 json 을 읽어 표와 순자산 곡선으로 보여 준다.
판정 잣대는 HYFE_QTPA 와 같다: 겹침 코호트, 단순수익, 연율 Sharpe·Newey-West p·MDD.
"""
import glob
import json
import os
import re
import subprocess
import threading
from pathlib import Path

HYFE = Path(os.environ.get("HYFE_QTPA_DIR", "/home/arcosium/projects/HYFE_QTPA"))
R = HYFE / "work" / "results"
FOLDS = {"0": "2025-12~2026-03", "1": "2025-09~12", "2": "2025-06~09", "3": "2025-03~06"}
_lock = threading.Lock(); _job = {"status": "idle", "msg": "", "pct": 0}
_PAT = re.compile(r"^krx_(?P<model>[a-z0-9]+)_(?P<render>[a-z0-9]+)(?:_sd(?P<seed>\d+))?_1d_W(?P<W>\d+)_H(?P<H>\d+)_s(?P<fold>\d)$")


def _parse(stem):
    m = _PAT.match(stem)
    return {**m.groupdict(), "seed": int(m.group("seed") or 0)} if m else None


def preds():
    """학습된 예측 파일 목록(변형·시드·폴드)."""
    out = []
    for p in sorted(glob.glob(str(R / "krx_*_pred.npz"))):
        stem = Path(p).name[:-len("_pred.npz")]; info = _parse(stem)
        if info:
            out.append({"stem": stem, **info, "period": FOLDS.get(info["fold"], "")})
    return out


def _tag(cost, q):
    """기본값(cost 0.002·q 0.1)이 아니면 예측 파일의 심링크 사본 이름에 태그를 붙여 cohort.py 가 결과를 다른 이름으로 쓰게 한다."""
    return "" if (abs(cost - 0.002) < 1e-9 and abs(q - 0.1) < 1e-9) else f"_c{cost:g}_q{q:g}"


def _out(stem, H, cost, q, long_only, shuffle):
    return R / (stem + _tag(cost, q) + f"_cohort_H{H}" + ("_long" if long_only else "") + (f"_shuf{shuffle}" if shuffle else "") + ".json")


def results(cost=0.002, q=0.1, long_only=True):
    """저장된 코호트 결과를 표로. 같은 변형·시드의 폴드들을 합산(일별 수익 이어붙임)한다."""
    rows = []; groups = {}
    for pr in preds():
        f = _out(pr["stem"], pr["H"], cost, q, long_only, 0)
        if not f.exists():
            continue
        j = json.load(open(f)); s = j["summary"]
        sh = _out(pr["stem"], pr["H"], cost, q, long_only, 1); shuf = json.load(open(sh))["summary"]["sharpe"] if sh.exists() else None
        ls = _out(pr["stem"], pr["H"], cost, q, False, 0); ls_sh = json.load(open(ls))["summary"]["sharpe"] if (long_only and ls.exists()) else None
        run = f"{pr['render']} {pr['model']} 1d·{pr['W']}·{pr['H']} 시드 {pr['seed']}"
        rows.append({"run": run, "stem": pr["stem"], "fold": pr["fold"], "period": pr["period"], "H": int(pr["H"]), "seed": pr["seed"], "sharpe": s["sharpe"], "p": s["p"], "mdd": s["mdd"], "final": s["final"], "days": s["days"], "shuffle": shuf, "longshort": ls_sh})
        groups.setdefault(run, []).append((pr["fold"], j["daily"]))
    pooled = []
    for run, parts in groups.items():
        parts.sort(reverse=True)   # 폴드 번호가 클수록 이른 시기(3=2025-03~06 … 0=2025-12~03) → 시간순으로 이어 붙인다
        # 폴드별 일별 순자산을 수익률로 바꿔 이어 붙인다(각 폴드 시작을 1 로 정규화)
        import pandas as pd
        rs = []
        for _, d in parts:
            eq = pd.Series(d, dtype=float); eq.index = pd.to_datetime(eq.index); rs.append(eq.pct_change().dropna())
        if not rs:
            continue
        r = pd.concat(rs); eq = (1 + r).cumprod()
        try:
            import sys
            sys.path.insert(0, str(HYFE)); from hyfe.perf import stats
            st = stats(r, eq, 10)
        except Exception:
            st = {"sharpe": None, "p": None, "mdd": None, "final": float(eq.iloc[-1]), "days": len(r)}
        pooled.append({"run": run, "n_folds": len(parts), **st, "curve": [{"d": str(k.date()), "v": round(float(v), 5)} for k, v in eq.items()]})
    return {"rows": rows, "pooled": pooled, "params": {"cost": cost, "q": q, "long_only": long_only}, "n_preds": len(preds())}


def progress():
    return dict(_job)


def start(cost=0.002, q=0.1, long_only=True, shuffle=True):
    """예측 파일 전부에 대해 cohort 를 다시 돌린다(없는 결과만). 백그라운드 스레드, 진행률은 progress()."""
    if not _lock.acquire(blocking=False):
        return False
    def run():
        try:
            todo = []; tag = _tag(cost, q)
            for pr in preds():
                H = pr["H"]
                for lo, sh in ([(long_only, 0)] + ([(long_only, 1)] if shuffle else []) + ([(False, 0)] if long_only else [])):
                    out = _out(pr["stem"], H, cost, q, lo, sh)
                    if not out.exists():
                        todo.append((pr["stem"], H, lo, sh))
            _job.update(status="running", msg=f"{len(todo)}건 계산", pct=0)
            for i, (stem, H, lo, sh) in enumerate(todo):
                pred = R / (stem + tag + "_pred.npz")
                if tag and not pred.exists():   # 비기본 파라미터: 예측 파일 심링크 사본 → cohort.py 가 태그 붙은 이름으로 결과를 쓴다(기본 결과 보존)
                    os.symlink(stem + "_pred.npz", pred)
                cmd = ["python3", "-m", "hyfe.cohort", "--pred", str(pred), "--res", "1d", "--H", str(H), "--cost", str(cost), "--q", str(q)] + (["--long_only"] if lo else []) + (["--shuffle", str(sh)] if sh else [])
                subprocess.run(cmd, cwd=str(HYFE), env={**os.environ, "HYFE_BARS": "work/bars_krx"}, capture_output=True, timeout=900, check=True)
                _job.update(pct=int((i + 1) / max(1, len(todo)) * 100), msg=f"{i + 1}/{len(todo)} {stem}")
            _job.update(status="done", msg="완료", pct=100)
        except Exception as e:  # noqa: BLE001
            _job.update(status="error", msg=str(e)[:200])
        finally:
            _lock.release()
    threading.Thread(target=run, daemon=True).start()
    return True

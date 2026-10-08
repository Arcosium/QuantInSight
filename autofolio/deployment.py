"""User-scoped, durable retraining queue. No path in this module sends orders."""
import hashlib
import json
import time
from pathlib import Path
from .store import connect,event
from .config import RUNS


def initialize():
    with connect() as db:db.execute('''CREATE TABLE IF NOT EXISTS model_deployments(
        id TEXT PRIMARY KEY,user_id INTEGER NOT NULL,target TEXT NOT NULL,strategy_id TEXT NOT NULL,
        source TEXT NOT NULL,status TEXT NOT NULL,created REAL NOT NULL,updated REAL NOT NULL,
        artifact TEXT,message TEXT NOT NULL DEFAULT '')''')


def request_retrain(uid,strategy,target,auto_policy=None):
    if target not in ('kr-paper','us-paper','crypto-paper','timefolio'):raise ValueError('정지된 실매매에는 적용할 수 없습니다.')
    initialize()
    from .auto_apply import initialize as auto_initialize
    auto_initialize()
    with connect() as db:row=db.execute('SELECT source FROM strategies WHERE id=?',(strategy,)).fetchone()
    if not row:raise ValueError('전략 결과 없음')
    source=Path(row['source'])
    try:report=json.loads(source.read_text())
    except (OSError,ValueError):raise ValueError('원본 학습 결과를 읽을 수 없습니다.')
    if not report.get('recipe'):raise ValueError('재학습 명세가 없는 과거 전략입니다. 연구소에서 새로 학습해 주세요.')
    if target=='timefolio' and not report.get('competition_compliance_verified'):
        raise ValueError('타임폴리오 대회 규칙 검증 전입니다. 현재 전략은 연구 결과로만 사용할 수 있습니다.')
    identity=hashlib.sha256(f'{uid}:{target}:{strategy}:{report["recipe"]["recipe_id"]}'.encode()).hexdigest()[:24]
    now=time.time()
    with connect() as db:
        pending=db.execute("SELECT id,strategy_id FROM model_deployments WHERE user_id=? AND target=? AND status IN ('queued','training')",(uid,target)).fetchone()
        if pending:
            if pending['strategy_id']!=strategy:raise ValueError('이 운용 대상의 재학습이 이미 진행 중입니다.')
            return pending['id']
        db.execute("INSERT INTO model_deployments VALUES(?,?,?,?,?,'queued',?,?,NULL,'동일 조건 재학습 대기') ON CONFLICT(id) DO UPDATE SET status='queued',updated=excluded.updated,message=excluded.message",(identity,uid,target,strategy,str(source),now,now))
        if not auto_policy:
            db.execute("INSERT INTO strategy_assignments VALUES(?,?,?,?,'retraining') ON CONFLICT(user_id,target) DO UPDATE SET strategy_id=excluded.strategy_id,updated=excluded.updated,status='retraining'",(uid,target,strategy,now))
        if auto_policy:
            db.execute('INSERT OR REPLACE INTO auto_apply_jobs VALUES(?,?,?,?)',(identity,uid,*auto_policy))
        else:
            db.execute('DELETE FROM auto_apply_jobs WHERE deployment_id=?',(identity,))
    return identity


def run(identity):
    from .learning import retrain
    initialize()
    with connect() as db:r=db.execute('SELECT * FROM model_deployments WHERE id=?',(identity,)).fetchone()
    if not r:return
    dest=RUNS/'deployments'/str(r['user_id'])/identity
    activated=False
    try:
        artifact=retrain(r['source'],dest)
        message='재학습 완료 · 다음 시세부터 페이퍼 운용' if r['target'] in ('kr-paper','us-paper','crypto-paper') else '재학습 완료 · 새 모델 운용 연결 검증 대기'
        from .auto_apply import activation
        with activation(identity):
            with connect() as db:
                db.execute("UPDATE model_deployments SET status='ready',artifact=?,updated=?,message=? WHERE id=?",(str(artifact),time.time(),message,identity))
                db.execute("INSERT INTO strategy_assignments VALUES(?,?,?,?,'paper_ready') ON CONFLICT(user_id,target) DO UPDATE SET strategy_id=excluded.strategy_id,updated=excluded.updated,status='paper_ready'",(r['user_id'],r['target'],r['strategy_id'],time.time()))
            activated=True
        # Only the selected deployment retains weights; old recipes/results stay.
        with connect() as db:old=db.execute("SELECT id,artifact FROM model_deployments WHERE user_id=? AND target=? AND id!=? AND status='ready'",(r['user_id'],r['target'],identity)).fetchall()
        root=(RUNS/'deployments'/str(r['user_id'])).resolve()
        for previous in old:
            path=Path(previous['artifact']).resolve()
            if path.is_relative_to(root) and path.name=='model.joblib':path.unlink(missing_ok=True)
            with connect() as db:db.execute("UPDATE model_deployments SET status='retired',message='후속 모델 적용으로 가중치 정리' WHERE id=?",(previous['id'],))
        event('retrained',dict(message='적용 모델 재학습 완료',job=identity))
    except Exception as exc:
        if activated:
            event('deployment_cleanup',dict(message='이전 모델 정리 확인 필요',job=identity))
            return
        (dest/'model.joblib').unlink(missing_ok=True)
        with connect() as db:db.execute("UPDATE model_deployments SET status='failed',updated=?,message=? WHERE id=?",(time.time(),str(exc)[:250],identity))
        raise


if __name__=='__main__':
    import sys
    run(sys.argv[1])

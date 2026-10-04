"""
Arquant — 운용지원실장 자동 수정 이력 (사장 지시 2026-05-14).

ops_support_worker가 매 사이클(또는 멘션/주간 리뷰) 후 적용한 코드 변경을
data/ops_history.json에 영구 누적. 사장이 "@운용지원실장 여태까지 수정한 코드 보여줘"
같은 질의를 하면 main_swarm.py:_ops_support_execute가 이 모듈을 읽어 즉시 답한다.

스키마 (각 항목은 한 번의 worker run):
{
  ts:          'YYYY-MM-DD HH:MM:SS' (KST),
  trigger:     'cycle' | 'manual' | 'weekly',
  cycle_id:    int | None,
  summary:     str   (LLM이 제시한 변경 요약),
  rationale:   str   (LLM 근거),
  applied:     [str] (실제 적용된 변경 라인들),
  rejected:    [str] (가드에 막힌 변경),
  compile_errors: [str],
  restarted:   bool,
}
"""
from __future__ import annotations
import json, logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any

logger = logging.getLogger("OPS_HIST")
KST = timezone(timedelta(hours=9))
HISTORY_FILE = Path(__file__).parent.parent / "data" / "ops_history.json"
_HISTORY_CAP = 1000  # 디스크에 보관할 최대 항목 수


def _now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def load_history() -> List[Dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        d = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return d if isinstance(d, list) else []
    except Exception as e:
        logger.warning(f"load_history 실패: {e}")
        return []


def _save(h: List[Dict]):
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(h, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"_save 실패: {e}")


def append_run(meta: Dict[str, Any]):
    """Append one worker run record. Caller fills in the fields."""
    h = load_history()
    entry = {"ts": _now_kst(), **meta}
    h.append(entry)
    if len(h) > _HISTORY_CAP:
        h = h[-_HISTORY_CAP:]
    _save(h)


def summary_text(limit: int = 30, only_with_changes: bool = True) -> str:
    """대시보드에 보여줄 사람-친화 텍스트 요약 (최근 limit개)."""
    h = load_history()
    if only_with_changes:
        h = [r for r in h if (r.get("applied") or [])]
    h = h[-limit:]
    if not h:
        return "📋 운용지원실장이 자동으로 수정한 코드 변경 이력이 아직 없습니다.\n" \
               "(시스템이 안정적이라는 신호 — 사이클마다 LLM이 '변경 없음'으로 답했거나, 보호 패턴에 막혔습니다.)"
    lines = [f"📋 운용지원 라인 자동 수정 이력 — 최근 {len(h)}건 (실제 변경 적용된 것만):"]
    for i, r in enumerate(h, 1):
        ts = r.get("ts", "?")
        trig = r.get("trigger", "?")
        # 사장 지시 2026-05-14: 팀장 분할 후엔 어느 페르소나가 변경했는지도 표시
        role_disp = r.get("role_display") or {"investment":"투자관리팀장","operations":"경영관리팀장",
                                              "finance":"재무관리팀장"}.get(r.get("role"), "운용지원실장")
        summary = r.get("summary", "(요약 없음)")
        applied = r.get("applied") or []
        rationale = (r.get("rationale") or "").strip()
        lines.append(f"\n━━━ [{i}] {ts} ({trig} / {role_disp}) ━━━")
        lines.append(f"요약: {summary}")
        if rationale:
            lines.append(f"근거: {rationale[:300]}{'...' if len(rationale)>300 else ''}")
        lines.append(f"적용된 변경 ({len(applied)}건):")
        for a in applied:
            lines.append(f"  {a}")
        if r.get("rejected"):
            lines.append(f"거부됨 ({len(r['rejected'])}건):")
            for x in r["rejected"][:5]:
                lines.append(f"  {x}")
        if r.get("restarted"):
            lines.append("🔄 서버 재시작 트리거됨")
    lines.append(f"\n(총 누적: {len(load_history())}건의 실행 기록, 그 중 실제 코드 수정이 일어난 것만 표시)")
    return "\n".join(lines)


def stats() -> Dict:
    """Aggregate stats — 총 실행 수, 적용/거부 비율, 사장 지시 2026-05-14: role별 분포."""
    h = load_history()
    n = len(h)
    n_with_changes = sum(1 for r in h if r.get("applied"))
    n_restarted = sum(1 for r in h if r.get("restarted"))
    n_compile_err = sum(1 for r in h if r.get("compile_errors"))
    total_applied = sum(len(r.get("applied") or []) for r in h)
    total_rejected = sum(len(r.get("rejected") or []) for r in h)
    by_role: Dict[str, int] = {}
    for r in h:
        k = r.get("role") or "ops_support"
        by_role[k] = by_role.get(k, 0) + 1
    return {"runs": n, "runs_with_changes": n_with_changes,
            "runs_restarted": n_restarted, "runs_compile_errors": n_compile_err,
            "total_applied_changes": total_applied, "total_rejected_changes": total_rejected,
            "runs_by_role": by_role}

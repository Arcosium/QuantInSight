"""매매 복기 저장소 — per-uid data/<uid>/trade_reflections.json (사장 지시 2026-07-31).

TradingAgents(TauricResearch)의 결과 복기 루프 이식:
  Phase A (매수 체결 직후): 진입 근거 스냅샷(thesis·퀀트점수·위원회 스탠스)과
    벤치마크(KR=코스피 / US=S&P500) 진입 레벨을 pending 으로 기록.
  Phase B (전량 청산 감지): 원장(trade_ledger fills)의 **실제 매도 체결**로 수익률과
    벤치마크 대비 알파를 계산하고, LLM 이 2~4문장 복기를 작성해 resolved 확정.
    ⚠️ 원장에 매도 체결이 없으면 pending 그대로 둔다 — KIS 결제 과도기에 잔고가
    순간 빈값으로 읽혀 '전량 매도'로 오인돼도 복기가 오발동하지 않는 가드.
  주입: past_context() 가 같은 종목 복기(전문) + 교차 종목 교훈(복기만)을 포맷해
    위원회 매수 심의·매도 심의·PASS 2 프롬프트에 재주입된다.

동시성·저장 패턴은 position_thesis.py 와 동일(uid 당 단일 swarm 태스크, atomic write).
어떤 실패도 사이클을 막지 않는다 — 전 함수 fail-open.
"""
from __future__ import annotations
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from infra import user_paths

logger = logging.getLogger("REFLECTION")

_MAX_RESOLVED = 100        # resolved 초과분은 오래된 것부터 정리 (pending 은 항상 보존)
_LLM_MAX_TOKENS = 12000    # 로컬 추론모델은 max_tokens 가 작으면 content 가 빈다 — committee 와 동일
_LLM_TIMEOUT = 240

# 시장별 벤치마크 — KR/US 분리(KRW 수익률 vs 코스피, USD 수익률 vs S&P500. 통화 혼합 금지)
_BENCH = {"KR": ("KOSPI", "코스피"), "US": ("SPX", "S&P500")}


def _is_kr(code: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", str(code or "").strip()))


def _path(uid: int):
    return user_paths.user_dir(int(uid)) / "trade_reflections.json"


def _load(uid: int) -> List[Dict[str, Any]]:
    p = _path(uid)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception as e:
            logger.warning(f"[복기] uid={uid} 파일 로드 실패({e}) — 빈 목록으로 시작")
    return []


def _save(uid: int, data: List[Dict[str, Any]]) -> None:
    # resolved 캡 적용 — pending 은 항상 보존
    resolved = [e for e in data if e.get("status") == "resolved"]
    if len(resolved) > _MAX_RESOLVED:
        drop = set(id(e) for e in resolved[:len(resolved) - _MAX_RESOLVED])
        data = [e for e in data if id(e) not in drop]
    p = _path(uid)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)
    except Exception as e:
        logger.warning(f"[복기] uid={uid} 저장 실패: {e}")


def bench_snapshot(code: str) -> Dict[str, Any]:
    """이 종목의 벤치마크 지수 현재 레벨 — Phase A/B 공용. 실패 시 level=None(정직).
    네이버 크롤 1회(블로킹) — 호출부는 스레드에서 부를 것."""
    key, label = _BENCH["KR" if _is_kr(code) else "US"]
    try:
        from tools.market_data import _fetch_index
        d = _fetch_index(key) or {}
        v = d.get("value")
        return {"bench_name": label, "level": float(v) if v and v > 0 else None}
    except Exception as e:
        logger.warning(f"[복기] 벤치마크({key}) 조회 실패: {e}")
        return {"bench_name": label, "level": None}


def record_pending(uid: int, code: str, entry: Dict[str, Any]) -> None:
    """Phase A — 매수 체결 직후 진입 스냅샷 기록. 같은 종목의 기존 pending 은 교체
    (재매수 = 새 매매, thesis 덮어쓰기와 동일한 의미론)."""
    code = str(code).strip()
    data = [e for e in _load(int(uid))
            if not (e.get("code") == code and e.get("status") == "pending")]
    data.append({"code": code, "status": "pending",
                 "market": ("KR" if _is_kr(code) else "US"), **entry})
    _save(int(uid), data)


def pending_codes(uid: int) -> List[str]:
    return [e["code"] for e in _load(int(uid)) if e.get("status") == "pending"]


def _sell_fills_for(fills: List[dict], code: str, entry_ts: str) -> List[dict]:
    """원장 fills 에서 entry_ts 이후 이 종목의 매도 체결만 (ts 는 동일 KST 문자열 포맷 — 문자열 비교)."""
    return [f for f in (fills or [])
            if str(f.get("ticker", "")).strip() == code
            and str(f.get("side", "")).lower() == "sell"
            and (not entry_ts or str(f.get("ts", "")) >= entry_ts)]


def _hours_between(ts_a: str, ts_b: str) -> float:
    try:
        a = datetime.fromisoformat(str(ts_a).split("+")[0].strip())
        b = datetime.fromisoformat(str(ts_b).split("+")[0].strip())
        return (b - a).total_seconds() / 3600.0
    except Exception:
        return 0.0


def compute_outcome(entry: Dict[str, Any], fills: List[dict],
                    bench_now: Optional[float], now_ts: str) -> Optional[Dict[str, Any]]:
    """순수 계산 — 매도 체결들로 결과 확정. 매도 체결이 없으면 None(=아직 확정 불가)."""
    sells = _sell_fills_for(fills, str(entry.get("code", "")).strip(),
                            str(entry.get("entry_ts", "")))
    if not sells:
        return None
    qty_sum = sum(int(f.get("qty") or 0) for f in sells) or 1
    exit_price = sum(float(f.get("price") or 0.0) * int(f.get("qty") or 0) for f in sells) / qty_sum
    entry_price = float(entry.get("entry_price") or 0.0)
    if exit_price <= 0 or entry_price <= 0:
        return None
    raw = (exit_price / entry_price - 1.0) * 100.0
    bench_entry = entry.get("bench_entry")
    bench_ret = alpha = None
    if bench_now and bench_entry:
        bench_ret = (float(bench_now) / float(bench_entry) - 1.0) * 100.0
        alpha = raw - bench_ret
    exit_ts = str(sells[-1].get("ts") or now_ts)
    return {"exit_ts": exit_ts, "exit_price": round(exit_price, 4),
            "raw_ret_pct": round(raw, 2),
            "bench_ret_pct": round(bench_ret, 2) if bench_ret is not None else None,
            "alpha_pct": round(alpha, 2) if alpha is not None else None,
            "holding_days": round(_hours_between(entry.get("entry_ts", ""), exit_ts) / 24.0, 1)}


def _fallback_reflection(entry: Dict[str, Any], oc: Dict[str, Any]) -> str:
    a = (f", {entry.get('bench_name') or '벤치마크'} 대비 {oc['alpha_pct']:+.2f}%p"
         if oc.get("alpha_pct") is not None else "")
    return (f"수익률 {oc['raw_ret_pct']:+.2f}%{a}로 청산. "
            f"진입 사유({str(entry.get('entry_reason') or '')[:60]})의 사후 검증은 "
            f"LLM 복기 불가로 생략 — 결과 수치만 기록(결정론 폴백).")


async def _llm_reflection(entry: Dict[str, Any], oc: Dict[str, Any]) -> str:
    """복기문 생성 — 2~4문장. 실패 시 결정론 폴백(사이클 무영향)."""
    try:
        from agents.committee import _committee_model
        from infra.local_llm_client import chat_completion, response_text
        alpha_line = (f"{entry.get('bench_name') or '벤치마크'} 대비 알파 {oc['alpha_pct']:+.2f}%p"
                      if oc.get("alpha_pct") is not None else "벤치마크 비교 불가(지수 조회 실패)")
        stance = entry.get("committee_stance") or "?"
        prompt = (
            f"[진입] {entry.get('name') or entry.get('code')}({entry.get('code')}) "
            f"{entry.get('entry_ts')} @{float(entry.get('entry_price') or 0):,.2f} {entry.get('ccy') or ''}"
            f" — 퀀트점수 {entry.get('quant_score', '?')}/10 · 위원회 {stance}\n"
            f"진입 사유: {str(entry.get('entry_reason') or '(미기록)')[:200]}\n"
            f"계획: 목표가 {entry.get('target_price') or '?'} / 손절가 {entry.get('stop_price') or '?'}"
            f" / 계획 보유 {entry.get('planned_hold_hours') or '?'}h\n"
            f"[결과] {oc['exit_ts']} 청산 @{oc['exit_price']:,.2f} — 수익률 {oc['raw_ret_pct']:+.2f}%, "
            f"{alpha_line}, 보유 {oc['holding_days']:.1f}일")
        data = await chat_completion(
            api_key="", model=_committee_model(),
            messages=[
                {"role": "system", "content":
                    "당신은 ArQuant 운용위원회의 매매 복기 담당이다. 방금 청산이 확정된 매매를 복기한다.\n"
                    "정확히 2~4문장의 한국어 줄글로만 답하라(불릿·헤더·마크다운·JSON 금지). 순서대로:\n"
                    "① 방향 판단이 맞았는가(알파 수치 인용) ② 진입 논거의 어떤 부분이 유효했고 어떤 부분이 깨졌는가 "
                    "③ 다음 유사 매매에 적용할 구체적 교훈 하나.\n"
                    "이 복기는 이후 매수/매도 심의 프롬프트에 그대로 재주입된다 — 모든 문장이 값어치 있게."},
                {"role": "user", "content": prompt}],
            max_tokens=_LLM_MAX_TOKENS, temperature=0.3, timeout_sec=_LLM_TIMEOUT,
            thinking=False)
        text = " ".join((response_text(data) or "").split()).strip()
        return text[:600] if text else _fallback_reflection(entry, oc)
    except Exception as e:
        logger.warning(f"[복기] LLM 복기 실패(결정론 폴백): {e}")
        return _fallback_reflection(entry, oc)


async def resolve_position(uid: int, code: str, emit=None) -> bool:
    """Phase B — 청산된 종목 1건 복기 확정. 원장에 매도 체결이 없으면 False(다음 사이클 재시도)."""
    uid = int(uid)
    code = str(code).strip()
    try:
        entry = next((e for e in _load(uid)
                      if e.get("code") == code and e.get("status") == "pending"), None)
        if not entry:
            return False
        from infra import trade_ledger
        led = trade_ledger.load(uid) or {}
        import asyncio
        bench = await asyncio.to_thread(bench_snapshot, code)
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        oc = compute_outcome(entry, led.get("fills") or [], bench.get("level"), now_ts)
        if oc is None:
            logger.info(f"[복기] uid={uid} {code} 원장에 매도 체결 없음 — pending 유지(잔고 글리치 가드)")
            return False
        reflection = await _llm_reflection(entry, oc)
        data = _load(uid)
        for e in data:
            if e.get("code") == code and e.get("status") == "pending":
                e.update({"status": "resolved", "reflection": reflection, **oc})
                break
        _save(uid, data)
        if emit:
            try:
                a = (f" · {entry.get('bench_name') or '벤치'} 대비 {oc['alpha_pct']:+.2f}%p"
                     if oc.get("alpha_pct") is not None else "")
                await emit({"type": "agent_msg", "agent": "사후관리실장",
                            "message": (f"📓 [매매 복기] {entry.get('name') or code}({code}) — "
                                        f"수익률 {oc['raw_ret_pct']:+.2f}%{a} · 보유 {oc['holding_days']:.1f}일\n"
                                        f"{reflection}")})
            except Exception:
                pass
        return True
    except Exception as e:
        logger.warning(f"[복기] uid={uid} {code} 확정 실패(fail-open): {e}")
        return False


def tag_stats(uid: int, min_n: int = 2) -> Dict[str, Dict[str, Any]]:
    """확정 복기를 알파 계열(alpha_tag)별로 집계 — P2, 2026-08-02.

    "상관 낮은 알파를 섞는다"의 전제는 계열별 실적 데이터다. 표본이 min_n 미만인 계열은
    반환에서 뺀다(1건 승률 100% 같은 잡음이 위원회 프롬프트에 들어가지 않게).
    반환 {tag: {n, wins, win_rate, avg_ret_pct, avg_alpha_pct}} — 평균알파 내림차순."""
    try:
        rows: Dict[str, list] = {}
        for e in _load(int(uid)):
            if e.get("status") != "resolved" or e.get("raw_ret_pct") is None:
                continue
            rows.setdefault(str(e.get("alpha_tag") or "미태깅"), []).append(e)
        out = {}
        for tag, es in rows.items():
            if len(es) < max(1, min_n):
                continue
            rets = [float(e["raw_ret_pct"]) for e in es]
            alphas = [float(e["alpha_pct"]) for e in es if e.get("alpha_pct") is not None]
            wins = sum(1 for r in rets if r > 0)
            out[tag] = {"n": len(es), "wins": wins,
                        "win_rate": round(wins / len(es) * 100, 1),
                        "avg_ret_pct": round(sum(rets) / len(rets), 2),
                        "avg_alpha_pct": (round(sum(alphas) / len(alphas), 2) if alphas else None)}
        return dict(sorted(out.items(),
                           key=lambda kv: -(kv[1]["avg_alpha_pct"] if kv[1]["avg_alpha_pct"] is not None
                                            else kv[1]["avg_ret_pct"])))
    except Exception as e:
        logger.warning(f"[복기] tag_stats 실패(생략): {e}")
        return {}


def tag_stats_block(uid: int, min_n: int = 2) -> str:
    """tag_stats → 프롬프트/보고 한 블록. 표본 부족이면 빈 문자열(호출부가 블록 생략)."""
    st = tag_stats(uid, min_n=min_n)
    if not st:
        return ""
    lines = ["알파 계열별 실적 (청산 확정분):"]
    for tag, s in st.items():
        a = f" · 평균알파 {s['avg_alpha_pct']:+.2f}%p" if s["avg_alpha_pct"] is not None else ""
        lines.append(f"- {tag}: {s['n']}건 · 승률 {s['win_rate']:.0f}% · 평균 {s['avg_ret_pct']:+.2f}%{a}")
    return "\n".join(lines)


def _fmt_entry(e: Dict[str, Any], full: bool) -> str:
    a = f" | 알파 {e['alpha_pct']:+.2f}%p" if e.get("alpha_pct") is not None else ""
    tag = f" · 계열 {e['alpha_tag']}" if e.get("alpha_tag") else ""
    head = (f"[{str(e.get('entry_ts', ''))[:10]} 매수 → {str(e.get('exit_ts', ''))[:10]} 청산 | "
            f"수익률 {e.get('raw_ret_pct', 0):+.2f}%{a} | 보유 {e.get('holding_days', '?')}일 | "
            f"퀀트 {e.get('quant_score', '?')}/10 · 위원회 {e.get('committee_stance') or '?'}{tag}]")
    lines = [f"- {e.get('name') or e.get('code')}({e.get('code')}) {head}"]
    if full and e.get("entry_reason"):
        lines.append(f"    진입 사유: {str(e['entry_reason'])[:150]}")
    if e.get("reflection"):
        lines.append(f"    복기: {str(e['reflection'])[:400]}")
    return "\n".join(lines)


def past_context(uid: int, code: Optional[str] = None,
                 n_same: int = 3, n_cross: int = 3) -> str:
    """확정된 복기를 프롬프트 주입용 텍스트로 — 같은 종목(전문) + 교차 종목(복기만).
    없으면 빈 문자열(호출부가 블록 자체를 생략)."""
    try:
        resolved = [e for e in _load(int(uid)) if e.get("status") == "resolved"]
        if not resolved:
            return ""
        code = str(code).strip() if code else None
        same = [e for e in reversed(resolved) if code and e.get("code") == code][:max(0, n_same)]
        cross = [e for e in reversed(resolved) if not code or e.get("code") != code][:max(0, n_cross)]
        parts = []
        if same:
            parts.append("이 종목의 과거 매매 복기 (최신순):")
            parts.extend(_fmt_entry(e, full=True) for e in same)
        if cross:
            parts.append("최근 청산 매매 교훈 (타 종목):")
            parts.extend(_fmt_entry(e, full=False) for e in cross)
        # P2: 계열별 실적 — 개별 복기가 못 보여주는 '어떤 알파가 이 계정에서 밥값을 하는가'.
        tags = tag_stats_block(int(uid))
        if tags:
            parts.append(tags)
        return "\n".join(parts)
    except Exception as e:
        logger.warning(f"[복기] past_context 실패(생략): {e}")
        return ""

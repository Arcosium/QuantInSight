"""운용위원회 심의 계층 — QuantInSight 로직 이식 (사장 지시 2026-07-18).

QuantInSight(:8777)의 세 모듈을 ArQuant 사이클에 이식했다:
  · research.py  → build_report(): 출처 태그가 달린 사실(fact) + 신용 레드플래그 +
                   정직성 가드레일(출처 없는 '사실'은 코드가 [미확인]으로 강등).
  · meeting.py   → deliberate_target(): 팀장 3인 발언(실제 사이클 보고 발췌)
                   → 매수 심사역↔보유 심사역 찬반토론 N라운드
                   → 주식운용실장 최종 결정(+출처 없는 근거 자진신고 claims).
  · risk_gate.py → run_gate(): 결정론 게이트 — 모든 체크가 통과/차단 + 사유 + 근거를
                   남긴다(감사 가능성). LLM이 만장일치 '매수'여도 코드가 차단할 수 있다.

원칙:
  · 매수 심의는 PASS 2 이후 열린다. 매도 심의 뒤에는 포트폴리오기획팀장의 계획 기반
    1회 보류 정책이 적용되며, 손절·목표·thesis 훼손이나 다음 사이클 반복 매도는 통과한다.
  · 심의/LLM 실패는 사이클을 절대 막지 않는다 — 그 발언만 결정론 폴백, 스테이지 전체는
    fail-open(기록만 생략, 매매 파이프라인 불변).
  · **QuantInSight 파라미터는 고정 상수다** (사장 지시 2026-07-18: QIS 쪽 파라미터는
    웹·런타임 어디서도 조정 불가). 조정 가능한 것은 ArQuant 전략 파라미터뿐이며,
    게이트의 퀀트 하한(MIN_QUANT_SCORE)·비중 기준(MAX_CYCLE_BUDGET_RATIO)은 호출부가
    runtime 값을 읽어 인자로 넘긴다.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("COMMITTEE")

# ── QuantInSight 이식 상수 — 조정 불가(웹 설정 화면·runtime 키 없음) ──────────────
DEBATE_ROUNDS = 2          # 옹호↔반론 공방 라운드 (QIS 기본 3 → 사이클 LLM 부하 고려 2 고정)
UNVERIFIED_BLOCK = 0.50    # 리포트 미확인 비율 초과 시 매수 차단
UNVERIFIED_REDUCE = 0.30   # 초과 시 비중 50% 축소
SECTOR_CAP = 0.40          # 동일 섹터 합산 상한 (섹터 정보 있을 때만)
LLM_MAX_TOKENS = 12000     # 로컬 추론모델은 max_tokens 가 작으면 content 가 빈다 — 넉넉히
LLM_TIMEOUT = 240

FACT, ESTIMATE, UNVERIFIED = "사실", "추정", "미확인"
BUY, HOLD, AVOID = "매수", "보류", "회피"
PASS_V, REDUCE, BLOCK = "통과", "축소", "차단"
_STANCES = (BUY, HOLD, AVOID)
# 보유 매도 심의 스탠스 (사장 지시 2026-07-29) — 그대로 사후관리 매도지시(sell_directives)가 된다.
KEEP, HALF, ALL = "유지", "절반", "전량"
_SELL_STANCES = (KEEP, HALF, ALL)

# 공시·뉴스 텍스트에서 결정론으로 잡는 신용 레드플래그 키워드 (QIS research.py 사상)
_CRIT_KEYWORDS = ("관리종목", "거래정지", "상장폐지", "감사의견 거절", "의견거절",
                  "감사의견 한정", "자본잠식", "불성실공시")
_WARN_KEYWORDS = ("횡령", "배임", "소송", "감자", "유상증자", "전환사채", "신주인수권")


def risk_keywords(text: str) -> List[str]:
    """공시·뉴스 원문에서 신용 레드플래그 키워드를 찾는다(결정론). 정량 운용안 보유 종목의
    재량 매도는 이 근거가 있을 때만 허용한다(사장 지시 2026-09-25 회전 축소)."""
    t = str(text or "")
    return [k for k in _CRIT_KEYWORDS + _WARN_KEYWORDS if k in t]


# ── 데이터 구조 (QIS research.py / risk_gate.py 이식) ───────────────────────────
@dataclass
class Fact:
    text: str
    kind: str = FACT           # 사실 | 추정 | 미확인
    source: Optional[str] = None  # 사실이면 필수 — 없으면 가드레일이 미확인으로 강등


@dataclass
class Report:
    code: str
    name: str
    sector: str = ""
    facts: List[Fact] = field(default_factory=list)
    red_flags: List[dict] = field(default_factory=list)
    credit_view: str = ""
    unverified_ratio: float = 0.0
    guardrail_log: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    evidence: str = ""


@dataclass
class GateResult:
    verdict: str                    # 통과 | 축소 | 차단
    final_weight: float
    checks: List[Check] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ── 리서치 리포트 (research.py 이식 — 데이터원은 ArQuant 사이클 산출물) ─────────────
def build_report(code: str, name: str, *, sector: str = "",
                 quant_line: str = "", per_dart: str = "",
                 news_excerpt: str = "", fundamental: Optional[dict] = None,
                 extra_claims: Optional[List[str]] = None,
                 insight_excerpt: str = "") -> Report:
    """사이클이 이미 수집한 실데이터(결정론 지표·DART 공시·뉴스 발췌)로 출처 태그 리포트를 만든다.
    insight_excerpt: 기업리서치팀장이 종합한 기업 분석 리포트(사장 지시 2026-07-21) — 출처 있는 사실로 편입."""
    rep = Report(code=code, name=name, sector=sector or "")

    if insight_excerpt:
        rep.facts.append(Fact("기업리서치팀장 분석: " + " ".join(str(insight_excerpt).split())[:280],
                              FACT, "기업리서치팀장(기업 분석자료)"))

    if quant_line:
        rep.facts.append(Fact(quant_line[:240], FACT, "KIS 시세·결정론 지표(자체 계산)"))

    # DART 공시 — 사이클의 per_dart 텍스트에서 공시 제목 줄만 사실로 편입 (최대 6건)
    n_dart = 0
    for ln in (per_dart or "").splitlines():
        s = ln.strip().lstrip("-·• ").strip()
        if len(s) < 6 or s.startswith("["):
            continue
        rep.facts.append(Fact(s[:180], FACT, "DART 전자공시"))
        n_dart += 1
        if n_dart >= 6:
            break

    if news_excerpt:
        rep.facts.append(Fact("최근 뉴스: " + news_excerpt.replace("\n", " ")[:220],
                              FACT, "네이버 증권 속보"))

    if fundamental and fundamental.get("verdict"):
        rep.facts.append(Fact(
            f"장기 펀더멘털 판정 {fundamental['verdict']}"
            + (f" (품질 {fundamental.get('business_quality_score')})"
               if fundamental.get("business_quality_score") is not None else ""),
            ESTIMATE, "[추정 근거: ai-berkshire 리서치 스냅샷]"))

    # 출처 없는 주장 — 리서치 초안은 '사실'이라 주장하지만 가드레일이 코드로 강등한다
    for claim in (extra_claims or []):
        rep.facts.append(Fact(str(claim)[:160], FACT, None))

    _honesty_guardrail(rep)

    # 신용 레드플래그 (공시·뉴스 텍스트 결정론 스캔)
    scan_src = (per_dart or "") + "\n" + (news_excerpt or "")
    seen: set = set()
    for kw in _CRIT_KEYWORDS:
        if kw in scan_src and kw not in seen:
            seen.add(kw)
            rep.red_flags.append({"flag": kw, "severity": "critical",
                                  "evidence": _evidence_line(scan_src, kw)})
    for kw in _WARN_KEYWORDS:
        if kw in scan_src and kw not in seen:
            seen.add(kw)
            rep.red_flags.append({"flag": kw, "severity": "warn",
                                  "evidence": _evidence_line(scan_src, kw)})

    crit = [f for f in rep.red_flags if f["severity"] == "critical"]
    if crit:
        rep.credit_view = "고위험 — " + " · ".join(f["flag"] for f in crit)
    elif rep.red_flags:
        rep.credit_view = "주의 — " + " · ".join(f["flag"] for f in rep.red_flags)
    else:
        rep.credit_view = "건전 — 공시·뉴스상 중대한 신용 이슈 미발견"
    return rep


def _evidence_line(text: str, kw: str) -> str:
    for ln in text.splitlines():
        if kw in ln:
            return ln.strip()[:100]
    return "공시/뉴스 원문"


def _honesty_guardrail(rep: Report) -> None:
    """정직성 가드레일: 출처 없는 '사실'을 [미확인]으로 강등하고 비율을 계산한다.
    LLM이 무엇을 주장하든, 출처가 검증되지 않으면 사실 자격을 코드가 회수한다."""
    for f in rep.facts:
        if f.kind == FACT and not f.source:
            f.kind = UNVERIFIED
            rep.guardrail_log.append(f"출처 없는 주장 → [미확인] 강등: “{f.text[:40]}…”")
    substantive = [f for f in rep.facts if f.kind in (FACT, UNVERIFIED)]
    rep.unverified_ratio = (
        sum(1 for f in substantive if f.kind == UNVERIFIED) / len(substantive)
        if substantive else 0.0)


def ingest_claims(rep: Report, claims: List[dict]) -> None:
    """실장 결정의 fact_ref 없는(검증 불가) 근거 → [미확인] 편입 (정직성 가드레일 폐루프)."""
    n_facts = len(rep.facts)
    for c in claims:
        ref = c.get("fact_ref")
        if not (isinstance(ref, int) and 0 <= ref < n_facts):
            rep.facts.append(Fact(str(c.get("text", ""))[:160], UNVERIFIED, None))
            rep.guardrail_log.append(
                f"실장 결정의 출처 없는 근거 → [미확인] 편입: “{str(c.get('text', ''))[:40]}…”")
    substantive = [f for f in rep.facts if f.kind in (FACT, UNVERIFIED)]
    if substantive:
        rep.unverified_ratio = round(
            sum(1 for f in substantive if f.kind == UNVERIFIED) / len(substantive), 3)


# ── 회의 (meeting.py 이식) ─────────────────────────────────────────────────────
_PERSONAS = {
    "bull": ("매수 심사역", "주식운용실장 산하 서브에이전트. 역할상 무조건 매수를 옹호한다 — "
                          "회의록에서 매수 논거를 찾아 최대한 강하게 주장하고, 반론역의 직전 논거를 반박한다."),
    "bear": ("보유 심사역", "주식운용실장 산하 서브에이전트. 역할상 무조건 매수를 반대한다 — "
                            "회의록에서 리스크 논거를 찾아 최대한 강하게 주장하고, 옹호역의 직전 논거를 반박한다."),
    "chief": ("주식운용실장", "회의록 전체(팀장 의견 + 찬반 토론)를 종합해 최종 결정을 내린다. "
                            "옹호/반론 중 더 논증이 튼튼한 쪽을 채택하되, 회피 의견의 근거가 사실(출처)에 "
                            "기반하면 보수적으로 판단한다."),
    # 보유 매도 심의 (사장 지시 2026-07-29) — 매도 판단을 사후관리실장 단독이 아닌 찬반토론으로.
    "keep": ("유지 심사역", "사후관리실장 산하 서브에이전트. 역할상 무조건 '계속 보유'를 옹호한다 — "
                          "매수 논거가 아직 유효한 이유·매도의 기회비용·거래비용을 근거로 주장하고, "
                          "매도 심사역의 직전 논거를 반박한다."),
    "sell": ("매도 심사역", "사후관리실장 산하 서브에이전트. 역할상 무조건 '지금 매도'를 주장한다 — "
                          "손절·익절 규율, 훼손된 투자논거, 더 나은 대안을 근거로 주장하고, "
                          "유지 심사역의 직전 논거를 반박한다."),
}

_MEETING_RULES = """회의 규칙:
1. 제공된 자료 밖의 정보를 지어내지 마라.
2. 직전 발언을 재서술하지 말고 네 관점만 추가하라.
3. 발언은 2~3문장, 간결하고 단정적으로.
4. 반드시 JSON 한 개로만 답하라. 다른 텍스트 금지."""


def _committee_model() -> str:
    """토론 서브에이전트 모델 — 주식운용실장(chief_orchestrator) 배정을 따른다."""
    try:
        from infra import admin_config
        ov = admin_config.get_model_override("chief_orchestrator")
        if ov:
            return ov
    except Exception:
        pass
    from config import MODEL_ASSIGNMENTS
    return MODEL_ASSIGNMENTS.get("chief_orchestrator", "")


def _transcript(dialogue: List[dict]) -> str:
    if not dialogue:
        return "(아직 발언 없음)"
    return "\n".join(
        f"{u['speaker']}{'(' + u['stance'] + ')' if u.get('stance') else ''}: {u['text']}"
        for u in dialogue)


async def _turn(role: str, brief: str, dialogue: List[dict], ask: str,
                want_stance: bool, want_claims: bool = False,
                thinking: Optional[bool] = None,
                name_override: Optional[str] = None,
                persona_override: Optional[str] = None,
                stances: Optional[tuple] = None) -> Optional[dict]:
    """한 발언(턴) LLM 호출 → {"text", "stance"?, "confidence"?, "claims"?} | None(실패).
    name_override/persona_override: 슬리브 심의 등에서 chief 를 채권·원자재운용실장으로
    바꿔 부를 때 시스템 프롬프트의 직책·역할 설명을 치환한다(주식 심의 경로엔 미영향)."""
    name, persona = _PERSONAS[role]
    if name_override:
        name = name_override
    if persona_override:
        persona = persona_override
    _st = tuple(stances or _STANCES)
    if want_stance and want_claims:
        fmt = ('{"발언":"결정 근거 2~3문장", "스탠스":"' + "|".join(_st) + '", "확신":0~100, '
               '"claims":[{"text":"제공 사실 밖 주장(있을 때만)","fact_ref":null}]}')
    elif want_stance:
        fmt = '{"발언":"...", "스탠스":"' + "|".join(_st) + '", "확신":0~100}'
    else:
        fmt = '{"발언":"..."}'
    try:
        from infra.local_llm_client import chat_completion, response_text
        data = await chat_completion(
            api_key="", model=_committee_model(),
            messages=[
                {"role": "system",
                 "content": f"당신은 ArQuant 운용위원회의 {name}이다. {persona}\n{_MEETING_RULES}\n출력 형식: {fmt}"},
                {"role": "user",
                 "content": f"{brief}\n\n[지금까지의 회의록]\n{_transcript(dialogue)}\n\n[요청] {ask}"}],
            max_tokens=LLM_MAX_TOKENS, temperature=0.4, timeout_sec=LLM_TIMEOUT,
            thinking=thinking)
        raw = response_text(data)
        m = re.search(r"\{.*\}", raw or "", re.DOTALL)
        d = json.loads(m.group(0)) if m else {}
        text = str(d.get("발언", "")).strip()
        if not text:
            return None
        # 결정문 절단 완화(사장 지시 2026-07-20): 실장 최종 결정이 문장 중간에 잘리던 문제 —
        # 턴 텍스트 캡 400→600, 브로드캐스트 절단도 호출부에서 완화.
        out: Dict[str, Any] = {"text": text[:600]}
        if want_stance:
            if d.get("스탠스") not in _st:
                return None
            out["stance"] = d["스탠스"]
            try:
                out["confidence"] = max(0.0, min(1.0, float(d.get("확신", 60)) / 100))
            except (TypeError, ValueError):
                out["confidence"] = 0.6
        if want_claims:
            out["claims"] = [c for c in (d.get("claims") or [])
                             if isinstance(c, dict) and c.get("text")]
        return out
    except Exception as e:
        logger.warning("%s 턴 실패(결정론 폴백): %s", name, e)
        return None


def _fmt_brief(name: str, code: str, report: Report, quant_score: Optional[int],
               quant_excerpt: str, news_excerpt: str, macro_view: str,
               past_context: str = "") -> str:
    facts = "\n".join(f"  [{i}] ({f.kind}) {f.text} — 출처: {f.source or '없음'}"
                      for i, f in enumerate(report.facts))
    flags = "\n".join(f"  ⚑ {f['flag']} (근거: {f['evidence']})" for f in report.red_flags) or "  없음"
    # 과거 매매 복기(사장 지시 2026-07-31 — TradingAgents 복기 루프 이식): 같은 종목의
    # 청산 결과·교훈을 심의 브리핑에 재주입 — 같은 논거로 손절한 이력을 모른 채 토론 반복 금지.
    past = f"\n[과거 매매 복기 — 같은 실수 반복 금지]\n{past_context}" if past_context else ""
    return f"""[심의 대상] {name} ({code}){' · 섹터 ' + report.sector if report.sector else ''}
[퀀트점수] {quant_score if quant_score is not None else '미산정'}/10
[계량분석팀장 평가 발췌] {quant_excerpt or '(없음)'}
[마켓센티먼트 발췌] {news_excerpt or '(없음)'}
[리서치 리포트 — 번호가 fact_ref]
{facts or '  (사실 없음)'}
[신용 레드플래그]
{flags}
[신용진단] {report.credit_view}
[매크로] {macro_view or '(없음)'}{past}"""


async def deliberate_target(code: str, name: str, report: Report, *,
                            quant_score: Optional[int], quant_excerpt: str,
                            news_excerpt: str, macro_view: str,
                            select_rationale: str = "",
                            insight_excerpt: str = "",
                            past_context: str = "",
                            progress: Optional[Callable[[str, str], Awaitable[None]]] = None,
                            systematic: bool = False,
                            ) -> Tuple[List[dict], List[dict], dict, bool]:
    """매수 대상 1종목 위원회 심의 → (opinions, dialogue, chief 결정, llm_used).

    systematic=True(사장 지시 2026-09-25): 정량 운용안이 이미 고른 종목이다. 위원회는 매수 매력을
    다시 평가하지 않고 공시·뉴스·신용 위험으로 **거부(회피)할지만** 정한다 — 스탠스는 매수|회피.
    9/17~9/25 정량 후보 59건 중 30건이 위험 근거 없는 '보류'로 탈락했던 것을 막는다.

    팀장 3인 발언은 사이클의 실제 보고(발췌)를 회의록에 앉히고, 찬반토론과 실장 결정만
    LLM 턴을 돈다(QIS deliberate_live 의 ArQuant 적응 — 팀장 평가는 이미 사이클에서 수행됨)."""
    brief = _fmt_brief(name, code, report, quant_score, quant_excerpt, news_excerpt, macro_view,
                       past_context=past_context)
    dialogue: List[dict] = []
    opinions: List[dict] = []
    llm_used = False

    async def _emit(agent: str, msg: str):
        if progress:
            try:
                await progress(agent, msg)
            except Exception:
                pass

    # ①~③ 팀장 발언 — 사이클이 이미 생산한 실제 보고의 발췌를 회의록으로 편입
    gate_hint = quant_score if quant_score is not None else 0
    crit = [f for f in report.red_flags if f["severity"] == "critical"]
    seeded = [
        ("계량분석팀장", "quant",
         quant_excerpt or f"퀀트점수 {quant_score}/10 — 상세 평가는 사이클 리포트 참조.",
         BUY if gate_hint >= 6 else (HOLD if gate_hint >= 4 else AVOID), 0.6),
        ("마켓센티먼트팀장", "news",
         news_excerpt or "이 종목 직접 뉴스 없음 — 시장 전반 분위기만 참고.",
         HOLD if not news_excerpt else BUY, 0.55),
        ("기업리서치팀장", "insight",
         # 사장 지시 2026-07-21: 기업리서치팀장의 실제 기업 분석 리포트가 있으면 그것을 회의록에 앉힌다.
         (" ".join(str(insight_excerpt).split())[:400] if insight_excerpt else
          (f"출처검증 리포트 기준 신용진단 '{report.credit_view}'. 미확인 비율 "
           f"{report.unverified_ratio * 100:.0f}%.")),
         AVOID if crit else (HOLD if report.red_flags else BUY), 0.65),
    ]
    for speaker, role, text, stance, conf in seeded:
        utt = {"speaker": speaker, "role": role, "text": text[:400], "stance": stance}
        dialogue.append(utt)
        opinions.append({"agent": speaker, "role": role, "stance": stance,
                         "confidence": conf, "rationale": text[:400], "evidence": []})

    # ④ 찬반 토론 (옹호 ↔ 반론) — 주장 → 반박 → 재반박 (라운드 수는 QIS 이식 고정 상수)
    for rnd in range(1, DEBATE_ROUNDS + 1):
        ask_bull = ("매수 논거를 제시하라." if rnd == 1
                    else f"보유 심사역의 직전 논거를 반박하라. (라운드 {rnd})")
        ask_bear = ("매수를 반대하는 논거를 제시하라." if rnd == 1
                    else f"매수 심사역의 직전 논거를 반박하라. (라운드 {rnd})")
        if systematic and rnd == 1:
            ask_bull = "정량 운용안의 선정을 지지하고, 제기될 위험이 과장된 이유를 리포트 사실로 제시하라."
            ask_bear = "공시·뉴스·신용 측면에서 매수를 막아야 할 구체적 위험을 리포트 사실(fact_ref)로 제시하라."
        for role, ask, fb in (("bull", ask_bull,
                               f"모멘텀·수급이 유효하다. 퀀트점수 {quant_score}는 무시할 수 없다."),
                              ("bear", ask_bear,
                               "하락 리스크가 우선이다. 검증 안 된 재료로는 진입 근거가 약하다.")):
            # 토론 턴은 thinking OFF — 추론모드는 턴당 수십 초·빈 응답 위험(쇼츠 사후분석과 동일 처방).
            r = await _turn(role, brief, dialogue, ask, False, thinking=False)
            if r:
                llm_used = True
            else:
                r = {"text": fb}
            utt = {"speaker": _PERSONAS[role][0], "role": role, "text": r["text"], "round": rnd}
            dialogue.append(utt)
            await _emit(_PERSONAS[role][0], f"[{name} 심의 R{rnd}] {r['text'][:200]}")

    # ⑤ 주식운용실장 최종 결정 (+출처 없는 근거 자진신고 → 정직성 가드레일 폐루프)
    if systematic:
        r = await _turn("chief", brief, dialogue,
                        "이 종목은 정량 운용안(상대강도·변동성 규칙)이 이미 골랐다. 매수 매력은 다시 평가하지 말고, "
                        "리포트의 공시·뉴스·신용 사실 가운데 매수를 막아야 할 구체적 위험이 있는지만 판정하라. "
                        "막아야 하면 '회피'를 고르고 근거 fact_ref 를 밝혀라. 그런 위험이 없으면 '매수'다. "
                        "리포트 밖 근거를 썼다면 claims 로 자진 신고하라.", True, want_claims=True,
                        stances=(BUY, AVOID))
    else:
        r = await _turn("chief", brief, dialogue,
                        "최종 결정을 내려라. 근거가 위 리포트의 어느 사실(fact_ref)에 기반하는지 밝히고, "
                        "리포트 밖 근거를 썼다면 claims 로 자진 신고하라.", True, want_claims=True)
    if r is None:
        r = {"text": (select_rationale or "PASS 2 선정 유지 — LLM 심의 불가로 결정론 폴백.")[:300],
             "stance": BUY, "confidence": 0.6}
    else:
        llm_used = True
        ingest_claims(report, r.get("claims") or [])
    utt = {"speaker": "주식운용실장", "role": "chief", "text": r["text"], "stance": r["stance"]}
    dialogue.append(utt)
    opinions.append({"agent": "주식운용실장", "role": "chief", "stance": r["stance"],
                     "confidence": round(float(r.get("confidence", 0.6)), 2),
                     "rationale": r["text"], "evidence": [f"토론 {DEBATE_ROUNDS}R 종합"]})
    await _emit("주식운용실장", f"[{name} 심의 결정] {r['stance']} — {r['text'][:400]}")
    return opinions, dialogue, {"stance": r["stance"], "text": r["text"],
                                "confidence": float(r.get("confidence", 0.6))}, llm_used


# ── 슬리브(채권·원자재) 매수 심의 (사장 지시 2026-07-20) ─────────────────────────
def _fmt_sleeve_brief(code: str, name: str, sleeve_label: str, macro_view: str,
                      manager_rationale: str, weight_ctx: str, price: float) -> str:
    return f"""[심의 대상] {sleeve_label} ETF — {name} ({code})
[가격] 약 {price:,.0f} (원가격)
[자산배분 맥락] {weight_ctx or '(없음)'}
[매크로 요약] {macro_view or '(없음)'}
[{sleeve_label}운용실장 매수 논거]
{(manager_rationale or '(없음)')[:900]}
※ 이 ETF는 개별 종목 퀀트점수가 없는 자산배분 수단이다 — 매크로 정합성·목표 비중·과다노출 리스크로 판단하라."""


async def deliberate_sleeve_buy(code: str, name: str, *, sleeve_label: str,
                                chief_label: str, macro_view: str,
                                manager_rationale: str, weight_ctx: str = "",
                                price: float = 0.0,
                                progress: Optional[Callable[[str, str], Awaitable[None]]] = None,
                                ) -> Tuple[List[dict], dict, bool]:
    """채권·원자재 슬리브 매수 후보 1건 위원회 심의.

    매수 심사역↔보유 심사역 찬반토론 N라운드 → 슬리브운용실장(chief_label) 최종 매수/보류/회피.
    주식 deliberate_target 과 달리 퀀트점수·정직성 게이트가 없고 자산배분 정합성·과다노출만
    다툰다. 어떤 실패도 사이클을 막지 않는다(호출부 fail-open). 반환 (dialogue, decision, llm_used)."""
    brief = _fmt_sleeve_brief(code, name, sleeve_label, macro_view, manager_rationale, weight_ctx, price)
    _chief_persona = (f"채권·원자재 자산배분을 총괄하는 {chief_label}이다. 회의록의 찬반토론을 종합해 "
                      "이 ETF 매수 여부를 최종 결정한다 — 자산배분 정합성과 과다노출 리스크를 함께 본다.")
    dialogue: List[dict] = []
    llm_used = False

    async def _emit(agent: str, msg: str):
        if progress:
            try:
                await progress(agent, msg)
            except Exception:
                pass

    for rnd in range(1, DEBATE_ROUNDS + 1):
        ask_bull = (f"이 {sleeve_label} ETF 매수(자산배분)를 옹호하는 논거를 제시하라."
                    if rnd == 1 else f"보유 심사역의 직전 논거를 반박하라. (라운드 {rnd})")
        ask_bear = (f"이 {sleeve_label} ETF 매수를 반대하는 논거를 제시하라."
                    if rnd == 1 else f"매수 심사역의 직전 논거를 반박하라. (라운드 {rnd})")
        for role, ask, fb in (
                ("bull", ask_bull,
                 f"매크로가 {sleeve_label} 비중 확대를 권고한다 — 자산배분상 매수가 정합적이다."),
                ("bear", ask_bear,
                 "이미 목표 비중에 근접했거나 신호가 약하면 추가 매수는 과다 노출이다.")):
            r = await _turn(role, brief, dialogue, ask, False, thinking=False)
            if r:
                llm_used = True
            else:
                r = {"text": fb}
            dialogue.append({"speaker": _PERSONAS[role][0], "role": role,
                             "text": r["text"], "round": rnd})
            await _emit(_PERSONAS[role][0], f"[{name} {sleeve_label} 심의 R{rnd}] {r['text'][:200]}")

    r = await _turn("chief", brief, dialogue,
                    "찬반토론을 종합해 이 ETF 매수 여부를 최종 결정하라(매수|보류|회피).",
                    True, thinking=False,
                    name_override=chief_label, persona_override=_chief_persona)
    if r is None:
        r = {"text": f"{chief_label} 자산배분 판단 유지 — LLM 심의 불가로 매수 유지(결정론 폴백).",
             "stance": BUY, "confidence": 0.6}
    else:
        llm_used = True
    dialogue.append({"speaker": chief_label, "role": "chief", "text": r["text"], "stance": r["stance"]})
    await _emit(chief_label, f"[{name} {sleeve_label} 심의 결정] {r['stance']} — {r['text'][:400]}")
    return dialogue, {"stance": r["stance"], "text": r["text"],
                      "confidence": float(r.get("confidence", 0.6))}, llm_used


# ── 보유 매도 심의 (사장 지시 2026-07-29) ────────────────────────────────────────
def _fmt_sell_brief(code: str, name: str, *, qty: int, pnl_pct: float, hold_days,
                    manager_view: str, thesis: str, quant_excerpt: str,
                    news_excerpt: str, macro_view: str, chief_label: str,
                    planner_objection: str = "",
                    past_context: str = "") -> str:
    _hd = f"{float(hold_days):.1f}일" if hold_days is not None else "미상"
    past = f"\n[과거 매매 복기 — 같은 실수 반복 금지]\n{past_context}" if past_context else ""
    return f"""[심의 대상 — 보유 포지션] {name} ({code})
[보유] {int(qty or 0):,}주 · 평가손익 {float(pnl_pct or 0.0):+.2f}% · 보유기간 {_hd}
[{chief_label} 1차 판단·근거]
{(manager_view or '(없음)')[:1200]}
[매수 당시 계획(thesis)] {(thesis or '(없음)')[:400]}
[포트폴리오기획팀장 강한 매도 반론]
{(planner_objection or '(보류권 적용 요건 없음)')[:1200]}
[계량분석 발췌] {quant_excerpt or '(없음)'}
[뉴스 발췌] {news_excerpt or '(없음)'}
[매크로] {macro_view or '(없음)'}{past}
※ 결론은 반드시 '유지 | 절반 | 전량' 중 하나다. '절반'은 부분 익절/리스크 축소, '전량'은 청산이다."""


async def deliberate_position_sell(code: str, name: str, *, qty: int, pnl_pct: float,
                                   hold_days=None, manager_view: str = "",
                                   thesis: str = "", quant_excerpt: str = "",
                                   news_excerpt: str = "", macro_view: str = "",
                                   chief_label: str = "사후관리실장",
                                   fallback_directive: str = KEEP,
                                   planner_objection: str = "",
                                   past_context: str = "",
                                   progress: Optional[Callable[[str, str], Awaitable[None]]] = None,
                                   ) -> Tuple[List[dict], dict, bool]:
    """보유 1종목 매도 심의 — 유지 심사역 ↔ 매도 심사역 N라운드 → 사후관리실장 최종 결정.

    사장 지시 2026-07-29: 매도 판단이 사후관리실장 단독이던 것을 매수 심의와 동일한 찬반토론
    구조로 바꾼다. 결정 스탠스(유지|절반|전량)는 **그대로 매도지시가 되어 실주문에 반영**된다.
    LLM 실패는 사이클을 막지 않는다 — 그 턴만 결정론 폴백, 최종 실패면 `fallback_directive`
    (사후관리실장 1차 판단)를 그대로 유지한다(매도 누락 금지 원칙).
    반환 (dialogue, {stance, text, confidence}, llm_used)."""
    brief = _fmt_sell_brief(code, name, qty=qty, pnl_pct=pnl_pct, hold_days=hold_days,
                            manager_view=manager_view, thesis=thesis,
                            quant_excerpt=quant_excerpt, news_excerpt=news_excerpt,
                            macro_view=macro_view, chief_label=chief_label,
                            planner_objection=planner_objection,
                            past_context=past_context)
    _chief_persona = (f"보유 포지션 사후관리를 총괄하는 {chief_label}이다. 유지/매도 찬반토론을 종합해 "
                      "이 포지션을 '유지·절반·전량' 중 어떻게 할지 최종 결정한다 — "
                      "손절·익절 규율과 투자논거 훼손 여부를 함께 본다. 포트폴리오기획팀장의 강한 "
                      "매도 반론을 독립 근거로 다루고, 계획기간 중 매도를 택하면 새 반증을 명시한다.")
    dialogue: List[dict] = []
    llm_used = False

    async def _emit(agent: str, msg: str):
        if progress:
            try:
                await progress(agent, msg)
            except Exception:
                pass

    for rnd in range(1, DEBATE_ROUNDS + 1):
        ask_keep = ("포트폴리오기획팀장의 반론을 중심으로 이 포지션을 계속 보유해야 하는 논거를 제시하라."
                    if rnd == 1 else f"매도 심사역의 직전 논거를 반박하라. (라운드 {rnd})")
        ask_sell = ("이 포지션을 지금 매도해야 하는 논거를 제시하라."
                    if rnd == 1 else f"유지 심사역의 직전 논거를 반박하라. (라운드 {rnd})")
        for role, ask, fb in (
                ("keep", ask_keep,
                 f"평가손익 {float(pnl_pct or 0.0):+.1f}% — 매수 논거가 깨졌다는 근거가 없다면 회전비용만 늘어난다."),
                ("sell", ask_sell,
                 f"평가손익 {float(pnl_pct or 0.0):+.1f}% — 규율대로 이익은 확정하고 손실은 끊어야 한다.")):
            r = await _turn(role, brief, dialogue, ask, False, thinking=False)
            if r:
                llm_used = True
            else:
                r = {"text": fb}
            dialogue.append({"speaker": _PERSONAS[role][0], "role": role,
                             "text": r["text"], "round": rnd})
            await _emit(_PERSONAS[role][0], f"[{name} 매도심의 R{rnd}] {r['text'][:200]}")

    r = await _turn("chief", brief, dialogue,
                    "찬반토론을 종합해 이 포지션을 어떻게 할지 최종 결정하라(유지|절반|전량).",
                    True, thinking=False, stances=_SELL_STANCES,
                    name_override=chief_label, persona_override=_chief_persona)
    if r is None:
        _fb = str(fallback_directive or KEEP).strip()
        r = {"text": f"{chief_label} 1차 판단 유지 — LLM 심의 불가로 결정론 폴백('{_fb}').",
             "stance": (_fb if _fb in _SELL_STANCES else KEEP), "confidence": 0.6}
    else:
        llm_used = True
    dialogue.append({"speaker": chief_label, "role": "chief", "text": r["text"], "stance": r["stance"]})
    await _emit(chief_label, f"[{name} 매도심의 결정] {r['stance']} — {r['text'][:400]}")
    return dialogue, {"stance": r["stance"], "text": r["text"],
                      "confidence": float(r.get("confidence", 0.6))}, llm_used


# ── 결정론 리스크 게이트 (risk_gate.py 이식) ─────────────────────────────────────
def run_gate(report: Report, score: Optional[float], chief_confidence: float, *,
             min_quant_score: float, base_weight: float,
             sector_weights: Optional[Dict[str, float]] = None) -> GateResult:
    """결정론 최종 관문. min_quant_score·base_weight 는 ArQuant 전략 파라미터(runtime)를
    호출부가 읽어 넘긴다 — 정직성·섹터 문턱은 QIS 이식 고정 상수."""
    checks: List[Check] = []
    verdict = PASS_V
    sc = float(score) if score is not None else 0.0
    weight = max(0.0, float(base_weight)) * (0.6 + 0.4 * min(sc, 10) / 10) * \
        (0.7 + 0.3 * max(0.0, min(1.0, chief_confidence)))

    # 1) 신용 레드플래그 — 치명 플래그는 무조건 차단 (퀀트점수 무관)
    crit = [f for f in report.red_flags if f["severity"] == "critical"]
    if crit:
        checks.append(Check("신용 레드플래그", False,
                            " · ".join(f["flag"] for f in crit) + " → 매수 차단",
                            " / ".join(f["evidence"] for f in crit)))
        return GateResult(BLOCK, 0.0, checks)
    checks.append(Check("신용 레드플래그", True, "치명적 신용 이슈 없음", report.credit_view))

    # 2) 정직성 게이트 — 검증 안 된 리서치는 매매 근거 자격 상실 (QIS 고정 문턱)
    ur = report.unverified_ratio
    if ur > UNVERIFIED_BLOCK:
        checks.append(Check("리서치 정직성", False,
                            f"미확인 비율 {ur * 100:.0f}% > {UNVERIFIED_BLOCK * 100:.0f}% → 차단",
                            f"가드레일 로그 {len(report.guardrail_log)}건"))
        return GateResult(BLOCK, 0.0, checks)
    if ur > UNVERIFIED_REDUCE:
        weight *= 0.5
        verdict = REDUCE
        checks.append(Check("리서치 정직성", False,
                            f"미확인 비율 {ur * 100:.0f}% > {UNVERIFIED_REDUCE * 100:.0f}% → 비중 50% 축소",
                            "; ".join(report.guardrail_log[:2])))
    else:
        checks.append(Check("리서치 정직성", True,
                            f"미확인 비율 {ur * 100:.0f}% — 출처 검증 충족", ""))

    # 3) 퀀트점수 하한 — ArQuant 전략 파라미터(MIN_QUANT_SCORE)
    if min_quant_score > 0:
        if score is None or sc < float(min_quant_score):
            checks.append(Check("퀀트점수 하한", False,
                                f"{score if score is not None else '미산정'} < 하한 {min_quant_score}"
                                f" (ArQuant MIN_QUANT_SCORE) → 차단", ""))
            return GateResult(BLOCK, 0.0, checks)
        checks.append(Check("퀀트점수 하한", True,
                            f"{sc:g} ≥ 하한 {min_quant_score:g} (ArQuant MIN_QUANT_SCORE)", ""))
    else:
        checks.append(Check("퀀트점수 하한", True, "하한 미설정(0) — 통과", ""))

    # 4) 단일종목 비중 한도 — ArQuant 사이클 예산 비율 기준
    if weight > base_weight:
        checks.append(Check("단일종목 한도", True,
                            f"산출 비중 {weight * 100:.1f}% → 한도 {base_weight * 100:.0f}%로 절사"
                            f" (ArQuant MAX_CYCLE_BUDGET_RATIO 기준)", ""))
        weight = base_weight
    else:
        checks.append(Check("단일종목 한도", True,
                            f"{weight * 100:.1f}% ≤ 한도 {base_weight * 100:.0f}%"
                            f" (ArQuant MAX_CYCLE_BUDGET_RATIO 기준)", ""))

    # 5) 섹터 집중 한도 — 섹터 정보가 있을 때만 (QIS 고정 상한)
    sw = sector_weights or {}
    if report.sector:
        used = sw.get(report.sector, 0.0)
        if used + weight > SECTOR_CAP:
            allowed = max(0.0, SECTOR_CAP - used)
            checks.append(Check("섹터 집중 한도", allowed > 0,
                                f"{report.sector} 누적 {(used + weight) * 100:.0f}% > {SECTOR_CAP * 100:.0f}% "
                                f"→ {allowed * 100:.1f}%로 절사", ""))
            weight = allowed
            if weight <= 0:
                return GateResult(BLOCK, 0.0, checks)
        else:
            checks.append(Check("섹터 집중 한도", True,
                                f"{report.sector} 누적 {(used + weight) * 100:.0f}% ≤ {SECTOR_CAP * 100:.0f}%", ""))
    else:
        checks.append(Check("섹터 집중 한도", True, "섹터 정보 없음 — 통과(정보 부족)", ""))

    return GateResult(verdict, round(weight, 4), checks)

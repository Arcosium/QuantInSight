"""컴플라이언스실장 (Compliance Officer) — 수탁자책임·정책 적합성 반려 게이트.

사장 지시 2026-07-22: 2026-05-18 에 폐지(risk_guard 로 통합)됐던 구 '수탁자책임실
(Policy Filter)' 의 **반려 기능**을 컴플라이언스실장 이름으로 되살린다. 폐지 당시 원본
코드는 git 스냅샷 이전이라 복구 불가 — 동일 취지로 재작성한다.

구현이 LLM 이 아니라 **결정론 코드**인 이유:
  · 돈이 나가는 관문은 재현 가능해야 한다(리스크관리실장 1차 게이트와 같은 사상).
  · 참가신청서·기술설명서가 기술하는 "컴플라이언스실장(코드)" 과 표기를 일치시킨다.

담당하는 두 가지:
  1) **정직성 가드레일 공표** — committee._honesty_guardrail() 이 출처 없는 주장을
     [미확인]으로 강등한 사실과 그 비율을 컴플라이언스실장 명의로 회의록에 남긴다.
     (강등 로직 자체는 committee 가 이미 수행 — 여기서는 그 결과를 공표만 한다.)
  2) **수탁자책임·정책 반려** — ESG 블랙리스트(무기·도박·담배), 시장경보 지정
     (투자주의/경고/위험·주의환기·단기과열), 내부 제외목록에 걸리면 그 종목 **매수를 반려**한다.

원칙:
  · **매수만 반려한다.** 매도는 어떤 경우에도 막지 않는다(실주문 누락 금지 원칙).
  · **fail-open** — 스크리닝이 실패하면 통과시킨다. 컴플라이언스 때문에 사이클이 멈추지 않는다.
  · ESG 판정은 **종목명·섹터**로만 한다. 뉴스 본문 전체를 훑으면 "카지노 관련주 반사이익"
    같은 문장 하나로 무관한 종목이 통째로 차단된다(과차단 방지).
  · 파라미터는 committee 와 같이 **고정 상수** — 웹·runtime 조정 대상이 아니다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence

from agents.base_agent import BaseAgent

logger = logging.getLogger("COMPLIANCE")

NAME = "컴플라이언스실장"
PASS_V, REJECT = "통과", "반려"

# ── ESG 블랙리스트 (구 수탁자책임실 정책 — 무기 제조·도박·담배) ──────────────────
# 종목명 또는 섹터 문자열에 걸리면 반려. 뉴스 본문은 보지 않는다(과차단 방지).
_ESG_CATEGORIES: Dict[str, Sequence[str]] = {
    "도박·카지노": ("카지노", "도박", "복권", "베팅", "홀덤"),
    "담배": ("담배", "궐련", "연초"),
    "무기·방위산업": ("방위산업", "방산", "무기제조"),
}

# ── 시장경보·불건전 표식 (공시 원문에서 탐지) ─────────────────────────────────
# 관리종목·거래정지·상장폐지는 committee._CRIT_KEYWORDS 가 이미 차단하므로 중복 제외하고,
# 컴플라이언스는 '시장경보' 계열을 담당한다.
_MARKET_ALERTS: Sequence[str] = (
    "투자주의환기", "투자위험종목", "투자경고종목", "투자주의종목",
    "단기과열종목", "불공정거래", "시장경보",
)

# 내부 투자정책상 영구 제외 종목코드 (사장이 명시 지정할 때만 채운다)
_POLICY_EXCLUDE_CODES: frozenset = frozenset()


@dataclass
class ComplianceCheck:
    name: str
    passed: bool
    detail: str
    evidence: str = ""


@dataclass
class ComplianceResult:
    verdict: str = PASS_V                       # 통과 | 반려
    reasons: List[str] = field(default_factory=list)
    checks: List[ComplianceCheck] = field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return self.verdict == REJECT

    def to_dict(self) -> dict:
        return asdict(self)


def _hit(text: str, keywords: Iterable[str]) -> Optional[str]:
    t = text or ""
    for kw in keywords:
        if kw and kw in t:
            return kw
    return None


def _evidence_line(text: str, kw: str) -> str:
    for ln in (text or "").splitlines():
        if kw in ln:
            return ln.strip()[:120]
    return ""


def screen(code: str, name: str, *, sector: str = "", dart_text: str = "",
           exclude_codes: Optional[Iterable[str]] = None) -> ComplianceResult:
    """매수 후보 1종목에 대한 결정론 컴플라이언스 심사.

    걸리면 verdict=반려. 어떤 예외에도 사이클을 막지 않도록 호출부가 fail-open 처리한다.
    """
    res = ComplianceResult()
    # 호출부 fail-open 의 전제 — 어떤 입력에도 예외를 던지지 않는다. 상류(rep.sector 등)가
    # None·숫자를 넘겨도 여기서 문자열로 정규화한다.
    code = str(code or "").strip()
    name = str(name or "").strip()
    sector = str(sector or "").strip()
    dart_text = str(dart_text or "")
    label = f"{name}({code})"
    ident = f"{name} {sector}"                      # ESG 판정 대상 = 종목명 + 섹터만
    excl = {str(c).strip() for c in (exclude_codes or ())} | _POLICY_EXCLUDE_CODES

    # 1) 내부 투자정책 제외목록
    if code and code in excl:
        res.checks.append(ComplianceCheck("내부 투자정책", False,
                                          f"{label} 내부 제외목록 지정 종목 → 매수 반려"))
        res.verdict = REJECT
        res.reasons.append("내부 투자정책상 제외 종목")
        return res
    res.checks.append(ComplianceCheck("내부 투자정책", True, "제외목록 미해당"))

    # 2) ESG 블랙리스트 — 종목명·섹터 기준
    for category, keywords in _ESG_CATEGORIES.items():
        kw = _hit(ident, keywords)
        if kw:
            res.checks.append(ComplianceCheck(
                "ESG 블랙리스트", False,
                f"{category} 해당('{kw}') → 수탁자책임상 매수 반려", ident.strip()[:120]))
            res.verdict = REJECT
            res.reasons.append(f"ESG 블랙리스트 — {category}")
            return res
    res.checks.append(ComplianceCheck("ESG 블랙리스트", True,
                                      "무기·도박·담배 해당 없음", (sector or "섹터 정보 없음")[:60]))

    # 3) 시장경보·불건전 표식 — 공시 원문 기준
    kw = _hit(dart_text, _MARKET_ALERTS)
    if kw:
        res.checks.append(ComplianceCheck(
            "시장경보 지정", False, f"'{kw}' 공시 확인 → 매수 반려",
            _evidence_line(dart_text, kw)))
        res.verdict = REJECT
        res.reasons.append(f"시장경보 — {kw}")
        return res
    res.checks.append(ComplianceCheck("시장경보 지정", True,
                                      "투자주의·경고·위험 지정 공시 없음"))
    return res


def honesty_note(guardrail_log: Sequence[str], unverified_ratio: float) -> Optional[str]:
    """정직성 가드레일 결과를 컴플라이언스실장 명의로 공표할 문구. 강등이 없으면 None."""
    if not guardrail_log:
        return None
    head = "\n".join(f"  • {line}" for line in list(guardrail_log)[:3])
    more = f"\n  … 외 {len(guardrail_log) - 3}건" if len(guardrail_log) > 3 else ""
    return (f"🧿 [정직성 가드레일] 출처 없는 주장 {len(guardrail_log)}건을 [미확인]으로 "
            f"강등했습니다 — 미확인 비율 {unverified_ratio * 100:.0f}%\n{head}{more}")


def create_compliance_officer(injection=None) -> BaseAgent:
    """@컴플라이언스실장 멘션 응답용 페르소나.

    사이클의 반려 판정은 위 ``screen()`` (결정론 코드)이 수행한다 — 이 에이전트는
    판정을 내리지 않고, 사장 질의에 정책 근거를 설명하는 역할만 맡는다."""
    return BaseAgent(
        name=NAME,
        role="compliance",
        model_key="compliance",
        injection=injection,
        system_prompt="""당신은 QuantInSight의 '컴플라이언스실장(Compliance Officer)'입니다.

## 역할
- 수탁자 책임·정책 적합성·리서치 정직성에 관한 사장 질의에 답합니다.
- **매수 반려 판정 자체는 당신이 하지 않습니다.** 판정은 결정론 코드(`agents/compliance.screen`)가
  수행하며, 당신은 그 정책의 근거와 적용 범위를 설명합니다.

## 관장하는 정책
1. ESG 블랙리스트 — 무기·방위산업 제조, 도박·카지노, 담배. 종목명·섹터 기준으로 판정합니다.
2. 시장경보 지정 — 투자주의/투자경고/투자위험, 주의환기, 단기과열, 불공정거래 공시.
3. 내부 투자정책 제외목록 — 사장이 명시 지정한 종목코드.
4. 리서치 정직성 — 출처 없는 주장은 [미확인]으로 강등되며, 미확인 비율이 30%를 넘으면
   매수 비중이 절반으로 깎이고 50%를 넘으면 매수가 차단됩니다.

## 원칙
- **매수만 반려합니다.** 매도는 위험을 줄이는 행동이므로 어떤 경우에도 막지 않습니다.
- 추정이 아니라 **명시적 근거**(공시·정책 위반 사실)가 있을 때만 반려가 성립합니다.
- 답변은 2~4문장, 자연스러운 한국어 산문체. 마크다운 헤더·코드블록·표 금지.""",
    )

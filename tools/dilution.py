"""희석(dilution) 리스크 감지 — DART 공시 텍스트 파서 (2026-06-15 ROI#4).

매수 직전 결정론 안전장치. 전환사채(CB)·유상증자·신주인수권부사채(BW)·교환사채(EB)·제3자배정
등 '주식 수 증가 → 기존 주주 가치 희석' 이벤트를 공시 텍스트에서 감지한다. (자율 에이전트 X,
결정론 게이트 — 사장 거버넌스 방침에 부합.)
"""
from __future__ import annotations
from typing import Dict, List, Optional

# 심각도 high — 직접적 신주 발행/대규모 희석
_HIGH = ["전환사채", "유상증자", "신주인수권부사채", "교환사채", "제3자배정",
         "신주인수권", "CB발행", "BW발행", "전환청구", "주주배정"]
# 심각도 medium — 잠재적/소규모 희석
_MEDIUM = ["주식매수선택권", "스톡옵션", "주식분할", "무상증자", "메자닌"]
# 약어(영문) — 단어 경계 모호해 별도 처리
_HIGH_ABBR = ["(CB)", "(BW)", "(EB)"]


def detect_dilution(text: Optional[str]) -> Dict:
    """공시 텍스트 → {dilutive: bool, severity: 'high'|'medium'|'none', kinds: [매칭 키워드]}."""
    t = str(text or "")
    if not t.strip():
        return {"dilutive": False, "severity": "none", "kinds": []}
    kinds: List[str] = []
    sev = "none"
    for kw in _HIGH + _HIGH_ABBR:
        if kw in t:
            kinds.append(kw.strip("()")); sev = "high"
    if sev != "high":
        for kw in _MEDIUM:
            if kw in t:
                kinds.append(kw); sev = "medium"
    # 중복 제거(순서 유지)
    seen = set(); kinds = [k for k in kinds if not (k in seen or seen.add(k))]
    return {"dilutive": bool(kinds), "severity": sev, "kinds": kinds}

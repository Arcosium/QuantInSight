"""
ArQuant — 계정별 사용자 전략 (사장 지시 2026-07-03)

배경:
  하단 채팅창(사장 지시)에 사용자가 '본인의 매매 전략'을 서술하면 ceo_directive 의
  LLM 분류기가 STRATEGY 로 인식해 여기 저장한다. 저장된 전략은
  • 다음 사이클부터 계량분석팀장(quant) 종목 평가 프롬프트에 주입되고,
  • 운용지원실장(ops_support) 자동 파라미터 튜닝이 꺼진다
    (사용자 전략과 자동 튜닝이 서로 덮어쓰지 않도록; 전략 삭제 시 재활성화).

저장 위치 (런타임 데이터 — .gitignore 대상 data/ 하위):
  data/profiles/<uid>/user_strategy.json   {text, ts, source}

계정당 전략은 1건 — 새 전략 인식 시 교체된다.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("USER_STRATEGY")
KST = timezone(timedelta(hours=9))

_PROFILES_DIR = Path(__file__).parent.parent / "data" / "profiles"
_FILENAME = "user_strategy.json"
_MAX_LEN = 4000  # 저장 상한 (프롬프트 주입은 build_quant_block 에서 별도 절단)


def _path(uid: int) -> Path:
    d = _PROFILES_DIR / str(int(uid))
    d.mkdir(parents=True, exist_ok=True)
    return d / _FILENAME


def get_strategy(uid: Optional[int]) -> Optional[Dict[str, Any]]:
    if uid is None:
        return None
    p = _path(uid)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and (data.get("text") or "").strip() else None
    except Exception as e:
        logger.warning("사용자 전략 로드 실패(uid=%s): %s", uid, e)
        return None


def set_strategy(uid: int, text: str, *, source: str = "chat") -> Dict[str, Any]:
    text = (text or "").strip()[:_MAX_LEN]
    if not text:
        raise ValueError("전략 내용이 비어 있습니다.")
    data = {"text": text, "ts": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"), "source": source}
    _path(uid).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("사용자 전략 저장(uid=%s, %d자)", uid, len(text))
    return data


def clear_strategy(uid: int) -> bool:
    p = _path(uid)
    if not p.exists():
        return False
    try:
        p.unlink()
        logger.info("사용자 전략 삭제(uid=%s)", uid)
        return True
    except Exception as e:
        logger.warning("사용자 전략 삭제 실패(uid=%s): %s", uid, e)
        return False


def build_quant_block(uid: Optional[int], *, max_chars: int = 1500) -> str:
    """계량분석팀장 종목 평가 프롬프트 삽입용 블록. 전략이 없으면 빈 문자열."""
    strat = get_strategy(uid)
    if not strat:
        return ""
    text = (strat.get("text") or "").strip()[:max_chars]
    return (
        "## 사장님 지정 전략 (이 계정 한정 — 계량 평가에 반영)\n"
        f"{text}\n"
        "※ 위 전략의 종목 선정 기준·매매 규칙·지표 조건을 이번 종목 평가에 우선 반영하되, "
        "파이썬 리스크·guardrail 게이트가 항상 최종 우선합니다."
    )

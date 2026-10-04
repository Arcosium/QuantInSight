"""
ArQuant — 계정별 상시 지시사항 (사장 피드백 2026-05-19 ITEM6)

배경:
  /api/ceo → ceo_directive 는 일회성(one-shot) 처리라 사이클마다 지시가 휘발된다.
  사장님(hh09080)이 포트폴리오 운용 원칙을 **계정 한정**으로 영속 저장하고,
  주식운용실장이 매 사이클마다 그 지시를 최우선으로 준수하도록 하는 메커니즘.

저장 위치 (런타임 데이터 — .gitignore 대상 data/ 하위):
  data/profiles/<uid>/standing_directives.json   [{text, ts, id}, ...]

격리 보장:
  uid 별로 분리된 파일. 주식운용실장 프롬프트에는 활성 계정의 uid 지시만 삽입됨.
  다른 계정의 지시는 절대 섞이지 않는다.

삭제 영속 (tombstone) 모델 — 사장 지시 2026-05-20:
  • 사용자가 지시를 명시적으로 삭제(remove_directive/clear_directives)하면
    그 지시 id 를 **tombstone**(삭제 표식)으로 기록한다.
  • seed_admin_directive() 는 매 부팅 시 admin 지시를 멱등 보장(append)하되,
    해당 지시가 tombstone 되어 있으면 재시드하지 않는다.
      → "명시 삭제 = 영구 삭제", "삭제 안 함 + 재부팅 = 다시 채워짐".
  • tombstone 은 data/directive_tombstones.json (data/ 최상위)에 저장한다.
    프로필 디렉토리(data/profiles/<uid>)가 리셋되어도 삭제 의사가 보존되도록
    **프로필 폴더 바깥**에 둔다. (과거 sentinel 을 profiles/<uid> 안에 둬서
    프로필 리셋 시 함께 소실 → 재시드로 부활하던 버그를 근본 수정.)
"""
from __future__ import annotations

import json
import logging
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("STANDING_DIRECTIVES")
KST = timezone(timedelta(hours=9))

_DATA_DIR = Path(__file__).parent.parent / "data"
_PROFILES_DIR = _DATA_DIR / "profiles"
_DIRECTIVES_FILENAME = "standing_directives.json"
_TOMBSTONE_FILENAME = "directive_tombstones.json"  # data/ 최상위 — 명시 삭제 영속(프로필 리셋 생존)
_MAX_DIRECTIVES = 50  # 계정당 최대 저장 건수


def _now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def _now_dt_kst() -> datetime:
    return datetime.now(KST)


_DAY_ONLY_WORDS = ("오늘", "금일", "당일")
_CONTINUING_WORDS = ("오늘부터", "금일부터", "앞으로", "계속", "지속", "상시", "항상")


def _is_expired_day_directive(directive: Dict[str, Any], now: Optional[datetime] = None) -> bool:
    """'오늘/금일/당일' 한정 지시가 저장일을 넘겼는지 판정한다.

    기록은 감사 목적으로 그대로 보존하고 프롬프트 주입만 중단한다. '오늘부터 계속'처럼 지속
    의사가 함께 적힌 문장은 만료시키지 않는다.
    """
    text = str((directive or {}).get("text") or "")
    if not any(w in text for w in _DAY_ONLY_WORDS):
        return False
    if any(w in text for w in _CONTINUING_WORDS):
        return False
    try:
        created = datetime.strptime(str(directive.get("ts") or ""), "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
    except (TypeError, ValueError):
        return False  # 생성시각 불명은 임의 만료 금지
    current = now or _now_dt_kst()
    if current.tzinfo is None:
        current = current.replace(tzinfo=KST)
    return current.astimezone(KST).date() > created.date()


def _profile_dir(uid: int) -> Path:
    d = _PROFILES_DIR / str(int(uid))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _directives_path(uid: int) -> Path:
    return _profile_dir(uid) / _DIRECTIVES_FILENAME


def _directive_id(text: str) -> str:
    """지시 내용으로부터 결정론적 ID (SHA256 앞 12자). 멱등 체크용."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:12]


# ─── CRUD ──────────────────────────────────────────────────────────────────────

def load(uid: int) -> List[Dict[str, Any]]:
    """uid 계정의 상시 지시사항 목록 반환 (최신순). 없으면 []."""
    p = _directives_path(uid)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        return data
    except Exception as e:
        logger.warning("상시지시 로드 실패(uid=%s): %s", uid, e)
        return []


def _save(uid: int, directives: List[Dict[str, Any]]) -> None:
    try:
        _directives_path(uid).write_text(
            json.dumps(directives, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        logger.warning("상시지시 저장 실패(uid=%s): %s", uid, e)


def append_directive(uid: int, text: str) -> bool:
    """지시사항을 uid 계정에 추가. 동일 내용이 이미 있으면 추가하지 않음(멱등).
    반환: True=추가됨, False=중복으로 생략."""
    text = (text or "").strip()
    if not text:
        return False
    did = _directive_id(text)
    existing = load(uid)
    if any(d.get("id") == did for d in existing):
        logger.info("상시지시 중복 — 추가 생략(uid=%s id=%s)", uid, did)
        return False
    existing.append({"id": did, "text": text, "ts": _now_kst()})
    if len(existing) > _MAX_DIRECTIVES:
        existing = existing[-_MAX_DIRECTIVES:]
    _save(uid, existing)
    # 같은 지시를 다시 추가 = 삭제 의사 철회 → tombstone 해제.
    _remove_tombstone(uid, did)
    logger.info("상시지시 추가(uid=%s id=%s)", uid, did)
    return True


def clear_directives(uid: int) -> int:
    """uid 계정의 모든 상시 지시사항 삭제. 삭제된 건수 반환.
    명시 삭제이므로 각 지시 id 를 tombstone 처리 → 재부팅 부활 방지."""
    existing = load(uid)
    count = len(existing)
    for d in existing:
        if d.get("id"):
            _add_tombstone(uid, d["id"])
    try:
        p = _directives_path(uid)
        if p.exists():
            p.unlink()
    except Exception as e:
        logger.warning("상시지시 삭제 실패(uid=%s): %s", uid, e)
    return count


def remove_directive(uid: int, directive_id: str) -> bool:
    """특정 id의 지시 1건 삭제. 반환: True=삭제됨.
    명시 삭제이므로 tombstone 처리 → 재부팅 부활 방지."""
    existing = load(uid)
    new_list = [d for d in existing if d.get("id") != directive_id]
    if len(new_list) == len(existing):
        return False
    _save(uid, new_list)
    _add_tombstone(uid, directive_id)
    return True


# ─── 주식운용실장 프롬프트 주입 ────────────────────────────────────────────────

def build_orchestrator_directive_block(uid: Optional[int]) -> str:
    """uid 계정의 상시 지시사항을 주식운용실장 프롬프트 삽입용 텍스트로 조립.
    지시가 없으면 빈 문자열 반환.

    형식:
    ## 사장님 상시 지시사항 (당신 계정 한정 — 최우선 준수)
    [1] (지시 본문)
    [2] (지시 본문)
    ...
    ※ 위 지시사항은 이 계정 전용으로 매 사이클 자동 반영됩니다.
       신규 매수 종목 선정·자산 배분·리밸런싱 판단에서 최우선으로 준수하십시오.
    """
    if uid is None:
        return ""
    directives = [d for d in load(uid) if not _is_expired_day_directive(d)]
    if not directives:
        return ""
    lines = ["## 사장님 상시 지침 (당신 계정 한정 — 참고 지침: 다른 신호·리스크 게이트와 균형 있게 반영)"]
    for i, d in enumerate(directives, 1):
        lines.append(f"[{i}] {d['text']}")
    lines.append(
        "※ 위 지침은 이 계정 전용 참고사항입니다. 신규 매수·자산 배분·리밸런싱 시 다른 분석/리스크 신호와 균형 있게 고려하되,"
        " 파이썬 리스크·guardrail 게이트가 항상 최종 우선합니다."
    )
    return "\n".join(lines)


# ─── 삭제 tombstone 헬퍼 (data/ 최상위 — 프로필 리셋에도 생존) ──────────────────

def _tombstones_path() -> Path:
    return _DATA_DIR / _TOMBSTONE_FILENAME


def _load_tombstones() -> Dict[str, List[str]]:
    """{ "<uid>": ["<directive_id>", ...] } 형태의 삭제 표식 맵. 없으면 {}."""
    p = _tombstones_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("tombstone 로드 실패: %s", e)
        return {}


def _save_tombstones(data: Dict[str, List[str]]) -> None:
    try:
        _tombstones_path().parent.mkdir(parents=True, exist_ok=True)
        _tombstones_path().write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        logger.warning("tombstone 저장 실패: %s", e)


def _is_tombstoned(uid: int, directive_id: str) -> bool:
    """uid 계정에서 directive_id 가 명시 삭제(tombstone)된 적 있으면 True."""
    return directive_id in _load_tombstones().get(str(int(uid)), [])


def _add_tombstone(uid: int, directive_id: str) -> None:
    """directive_id 를 uid 의 삭제 표식에 추가 (멱등). 명시 삭제 시 호출."""
    data = _load_tombstones()
    key = str(int(uid))
    ids = data.get(key, [])
    if directive_id not in ids:
        ids.append(directive_id)
        data[key] = ids
        _save_tombstones(data)
        logger.info("tombstone 기록(uid=%s id=%s) — 명시 삭제 영구 보존", uid, directive_id)


def _remove_tombstone(uid: int, directive_id: str) -> None:
    """directive_id 의 삭제 표식 제거 (사용자가 같은 지시를 다시 추가 = 삭제 의사 철회)."""
    data = _load_tombstones()
    key = str(int(uid))
    ids = data.get(key, [])
    if directive_id in ids:
        ids = [i for i in ids if i != directive_id]
        if ids:
            data[key] = ids
        else:
            data.pop(key, None)
        _save_tombstones(data)
        logger.info("tombstone 해제(uid=%s id=%s)", uid, directive_id)


# ─── Admin 계정 시드 ───────────────────────────────────────────────────────────

# 사장 피드백 2026-05-19 ITEM6: 매크로 붕괴 시나리오 대응 상시 지시사항
MACRO_COLLAPSE_DIRECTIVE = (
    "한국의 재정 고갈과 기업 부채 전가에 따른 매크로 붕괴 시나리오를 상정하여, "
    "원화 자산 비중을 최소화하고 환차익과 이자 수익을 동시에 확보할 수 있는 달러 기반의 "
    "단기 국채 및 MMF를 포트폴리오의 핵심 축으로 설정하십시오. "
    "단순한 달러 보유를 넘어 부채 비율이 낮고 현금 흐름이 탄탄한 미국 내 '퀄리티 팩터' 우량주와 "
    "하락장에서 헤지가 가능한 인버스 상품을 전략적으로 배분하여 시스템 리스크 발생 시 자산 가치를 "
    "방어해야 합니다. 금과 비트코인처럼 변동성이나 실수요 왜곡이 심한 자산은 철저히 배제하되, "
    "시장 붕괴가 본격화되는 시점에 저평가된 국내 우량 자산을 다시 매입할 수 있도록 구체적인 "
    "리밸런싱 트리거와 현금화 계획을 수립하여 보고하기 바랍니다."
)


# 안전 주석: 이 상시 지시는 주식운용실장 LLM 프롬프트에만 주입된다.
# 결정론적 파이썬 리스크/guardrail 게이트는 영향받지 않으며(guardrails.py 미참조)
# 지시가 안전 게이트를 우회해 실매매를 강제할 수 없다.
def seed_admin_directive() -> None:
    """부팅 시 호출: admin(hh09080) 계정에 매크로 붕괴 지시사항을 멱등 보장.

    tombstone 모델:
      사용자가 명시 삭제(remove_directive/clear_directives)하면 지시 id 가
      tombstone 된다. tombstone 이 있으면 재시드하지 않는다(명시 삭제 = 영구 삭제).
      tombstone 이 없으면 매 부팅 append(멱등)로 지시를 보장한다
      (삭제 안 함 + 재부팅 → 다시 채워짐). tombstone 은 data/ 최상위라
      프로필 리셋(data/profiles/<uid> 삭제)에도 생존한다.

    admin 계정이 DB에 없으면 'data/profiles/admin_pending/' 에 대기 파일을 저장하고
    크래시 없이 종료 (안전한 no-op). 다음 부팅 시 계정이 생성되면 재시도된다.
    """
    try:
        from infra.auth_store import find_user_by_username, ADMIN_USERNAMES
    except Exception as e:
        logger.warning("seed_admin_directive: auth_store import 실패 — no-op: %s", e)
        return

    admin_username = next(iter(ADMIN_USERNAMES), None)
    if not admin_username:
        logger.warning("seed_admin_directive: ADMIN_USERNAMES 비어있음 — no-op")
        return

    try:
        user = find_user_by_username(admin_username)
    except Exception as e:
        logger.warning("seed_admin_directive: admin 계정 조회 실패 — no-op: %s", e)
        user = None

    if not user:
        logger.info(
            "seed_admin_directive: admin 계정(%s) DB 미존재 — "
            "pending 파일에 보관 (계정 생성 후 재시드 필요)", admin_username
        )
        # pending 저장 — 나중에 수동 또는 재시작 시 처리 가능
        pending_dir = _DATA_DIR / "profiles" / "admin_pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        pending_file = pending_dir / _DIRECTIVES_FILENAME
        try:
            pending_file.write_text(
                json.dumps([{"id": _directive_id(MACRO_COLLAPSE_DIRECTIVE),
                             "text": MACRO_COLLAPSE_DIRECTIVE,
                             "ts": _now_kst(),
                             "note": f"admin({admin_username}) 계정 생성 후 이동 필요"}],
                           ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            logger.info("seed_admin_directive: pending 저장 완료 → %s", pending_file)
        except Exception as pe:
            logger.warning("seed_admin_directive: pending 저장 실패: %s", pe)
        return

    uid = user["id"]
    macro_id = _directive_id(MACRO_COLLAPSE_DIRECTIVE)

    # tombstone 확인: 사용자가 명시 삭제한 지시면 재시드하지 않음 (영구 삭제 보존).
    # tombstone 은 data/ 최상위에 있어 프로필 리셋에도 생존한다 — 부활 버그 근본 차단.
    if _is_tombstoned(uid, macro_id):
        logger.info(
            "seed_admin_directive: tombstone 존재(uid=%s id=%s) — 재시드 생략 (명시 삭제 영구 보존)",
            uid, macro_id,
        )
        return

    # 삭제하지 않았으면 매 부팅 멱등 보장(append). 이미 있으면 생략.
    added = append_directive(uid, MACRO_COLLAPSE_DIRECTIVE)
    if added:
        logger.info("seed_admin_directive: 매크로 붕괴 지시 admin uid=%s 에 보장 삽입", uid)
    else:
        logger.info("seed_admin_directive: 매크로 붕괴 지시 이미 존재(uid=%s) — 생략", uid)

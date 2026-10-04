"""유휴 USD → KRW 자동 역환전 게이트 (사장 보고 2026-06-26).

문제(KR/US 비대칭): US 매수의 KRW→USD 환전은 KIS '통합증거금'이 결제 시 자동 처리하지만,
US 매도 후 남는 USD 예수금은 KRW 로 자동 환원되지 않는다 — KIS OpenAPI 에 공개 '환전' TR 이
없어 역방향 경로가 애초에 배선된 적이 없다(공식 open-trading-api·본 코드베이스 모두 0건).
그래서 사장이 매번 MTS 에서 수동 환전해 왔다(라이브 로그: 유휴 USD ≈₩1,424,204).

본 모듈은 유휴 USD 를 주기적으로 감지해
  (a) 자동 실행이 켜져 있고 실환전 경로가 있으면 broker.us_to_krw_exchange 로 실행,
  (b) 아니면(기본) 운영자에게 '환전 필요' 알림(중복억제)을 띄워 수동 환전을 자동 환기한다.
'조용히 누락 금지' 원칙: KRW 가 필요한데 USD 가 놀고 있으면 반드시 신호를 남긴다.

안전:
  - dry_run(=not LIVE_TRADING) 이면 실환전 절대 금지. is_mock 도 금지.
  - 멱등: 같은 액수 버킷은 알림 dedup_key 로 중복 억제(스팸 방지).
  - KRW 한도와 USD 평가액을 섞지 않는다 — 환전 의사결정은 USD 예수금(평가)으로만 한다.
    KRW 부족분(원)이 들어오면 환율로 USD 환산해 '비교'만 하고, 환전 단위는 항상 USD.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("ARQUANT")


async def maybe_reconvert_idle_usd(broker, *, dry_run: bool, uid: Optional[str] = None,
                                   notifier=None, krw_shortfall: float = 0.0) -> dict:
    """유휴 USD 를 감지해 KRW 로 역환전(또는 수동 환전 알림)한다.

    Parameters
    ----------
    broker        : KisBroker — idle_usd_deposit()/us_to_krw_exchange()/is_mock 제공.
    dry_run       : True 면 실환전 절대 금지(보통 not LIVE_TRADING).
    uid           : 프로필 uid (runtime override·dedup 스코프).
    notifier      : infra.notifier (alert()). None 이면 로깅으로 폴백.
    krw_shortfall : >0 이면 'KRW 부족분(원)'만큼만 환전(USD 환산), 0 이면 유휴 USD 전액 스윕.

    Returns dict {action: skip|alert|exchanged|manual_required|noop, ...} — 테스트·로깅용.
    """
    import runtime
    from config import (AUTO_USD_TO_KRW_RECONVERT as _AUTO_DEFAULT,
                        USD_RECONVERT_MIN_USD as _MIN_DEFAULT)

    enabled = bool(runtime.get("AUTO_USD_TO_KRW_RECONVERT", _AUTO_DEFAULT, uid=uid))
    min_usd = float(runtime.get("USD_RECONVERT_MIN_USD", _MIN_DEFAULT, uid=uid) or 0.0)

    # 모의계좌는 외화 데이터가 garbage(기준환율 비정상 등) — 아예 스킵.
    if getattr(broker, "is_mock", False):
        return {"action": "skip", "reason": "mock"}

    try:
        info = await broker.idle_usd_deposit()
    except Exception as e:
        logger.warning(f"[USD역환전] 유휴 USD 조회 실패(스킵): {e}")
        return {"action": "skip", "reason": "lookup_failed"}
    if not info.get("ok"):
        return {"action": "skip", "reason": "no_data"}

    idle_usd = float(info.get("usd") or 0.0)
    exrt = float(info.get("exrt") or 0.0)
    if idle_usd < min_usd:
        return {"action": "skip", "reason": "below_min", "idle_usd": idle_usd}

    # 환전할 USD 액수: KRW 부족분이 명시되면 그만큼만(USD 환산), 아니면 유휴 USD 전액 스윕.
    if krw_shortfall and krw_shortfall > 0 and exrt > 0:
        want_usd = min(idle_usd, krw_shortfall / exrt)
    else:
        want_usd = idle_usd
    want_usd = round(want_usd, 2)
    if want_usd <= 0:
        return {"action": "skip", "reason": "nothing_to_convert", "idle_usd": idle_usd}
    est_krw = round(want_usd * exrt) if exrt > 0 else 0

    # 멱등: 같은 100불 버킷은 중복 억제(알림 dedup_key).
    bucket = int(want_usd // 100)
    dedup = f"usd_reconvert:{uid or 'default'}:{bucket}"

    if not enabled:
        # 기본 모드: 자동 실행 OFF → 수동 환전 환기 알림만(조용히 누락 금지).
        msg = (f"유휴 USD ${idle_usd:,.2f}(≈₩{round(idle_usd * exrt):,}) — KRW 자동 역환전 OFF. "
               f"MTS 에서 ${want_usd:,.2f} 수동 환전 권장(환율 {exrt:,.1f}).")
        _emit(notifier, "WARN", "USD→KRW 환전 필요(유휴 USD)", msg, dedup, 6 * 3600)
        return {"action": "alert", "idle_usd": idle_usd, "want_usd": want_usd, "est_krw": est_krw}

    # 자동 실행 ON: 실환전 시도(단일 진입점). dry_run/TR미설정이면 내부에서 안전 no-op + manual_required.
    res = await broker.us_to_krw_exchange(want_usd, dry_run=dry_run,
                                          reason=f"유휴 USD 역환전 uid={uid}")
    if res.get("ok"):
        _emit(notifier, "INFO", "USD→KRW 자동 환전 실행",
              f"${want_usd:,.2f} → ≈₩{est_krw:,} (환율 {exrt:,.1f})", dedup, 3600)
        logger.info(f"[USD역환전] 실행 ${want_usd:,.2f} → ≈₩{est_krw:,}")
        return {"action": "exchanged", "want_usd": want_usd, "est_krw": est_krw}
    if res.get("manual_required"):
        msg = (f"유휴 USD ${idle_usd:,.2f} — 자동 환전 ON 이나 KIS 환전 경로 없음. "
               f"MTS 에서 ${want_usd:,.2f}(≈₩{est_krw:,}) 수동 환전 필요.")
        _emit(notifier, "WARN", "USD→KRW 수동 환전 필요", msg, dedup, 6 * 3600)
        return {"action": "manual_required", "idle_usd": idle_usd, "want_usd": want_usd, "est_krw": est_krw}
    # dry-run 또는 기타 — 조용히(로깅만). 잡음 방지.
    logger.info(f"[USD역환전] 미실행: {res.get('reason')} (유휴 ${idle_usd:,.2f}, 희망 ${want_usd:,.2f})")
    return {"action": "noop", "reason": res.get("reason"), "want_usd": want_usd}


def _emit(notifier, level: str, title: str, detail: str, dedup_key: str, window_sec: int) -> None:
    """알림 발송(있으면) — 없으면 로깅 폴백. 절대 예외를 던지지 않는다."""
    try:
        if notifier is not None:
            notifier.alert(level, title, detail, dedup_key=dedup_key, dedup_window_sec=window_sec)
        else:
            logger.warning(f"[USD역환전] {title} — {detail}")
    except Exception as e:
        logger.warning(f"[USD역환전] 알림 실패(무시): {e}")

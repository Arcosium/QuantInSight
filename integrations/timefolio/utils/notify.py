"""알림 — notify(msg) 는 [ALERT] 로 로깅하고, 텔레그램 토큰이 설정돼 있으면 전송한다.

호출부: core/strategy.py·execution.py·crawler.py 가 notify(msg, level=logging.INFO) 형태로 부른다.
전송 실패가 매매 흐름을 죽이면 안 되므로 모든 예외를 삼킨다(fail-soft).
"""
import logging

import config

log = logging.getLogger("lag.notify")


def notify(msg, level=logging.INFO, **_kwargs):
    log.log(level, "[ALERT] %s", msg)
    tok, chat = config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID
    if not (tok and chat):
        return
    try:
        import requests
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      json={"chat_id": chat, "text": str(msg)}, timeout=5)
    except Exception:                       # noqa: BLE001 — 알림 실패는 조용히
        log.debug("텔레그램 전송 실패", exc_info=True)


if __name__ == "__main__":       # ponytail: 최소 자체검증
    notify("selftest 진입: 005930 x1 @ 70000", level=logging.INFO)
    notify("토큰 없어도 예외 없이 로깅만")   # 토큰 미설정 경로
    print("notify OK")

"""로깅 설정 — setup(name) 이 루트 로거를 한 번만 구성하고 named 로거를 돌려준다.

여러 프로세스(data/engine/exec)와 웹이 각자 setup 을 부르지만, 루트 구성은 최초 1회만.
파일 핸들러는 두지 않는다 — 프로세스마다 같은 파일을 rotate 하면 경합한다(웹 app.py 주석 참조).
journald(systemd) 가 stdout/stderr 를 수집한다.
"""
import logging

_CONFIGURED = False


def setup(name: str = "lag") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        root = logging.getLogger()
        if not root.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter(
                "%(asctime)s [%(processName)s] %(levelname)s %(name)s: %(message)s",
                datefmt="%m-%d %H:%M:%S"))
            root.addHandler(h)
        root.setLevel(logging.INFO)
        for noisy in ("urllib3", "asyncio", "playwright", "httpx", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _CONFIGURED = True
    return logging.getLogger(f"lag.{name}")


if __name__ == "__main__":       # ponytail: 최소 자체검증
    log = setup("selftest")
    assert log.name == "lag.selftest"
    assert logging.getLogger().level == logging.INFO
    assert setup("again").name == "lag.again"   # 재호출 안전
    print("logging_setup OK")

"""주문 경로 공용 예외 — 무거운 의존성 없이 어디서든 import 할 수 있어야 한다.

`naver_cycle` 은 QuantInSight 스웜처럼 playwright 가 없는 호스트에서도 돌기 때문에
`timefolio_browser`(playwright 를 모듈 최상단에서 import)를 끌어오면 안 된다.
"""


class ContestClosedError(RuntimeError):
    """대회 종료로 주문 창이 닫힌 상태. 장애가 아니므로 호출측은 조용히 넘긴다."""

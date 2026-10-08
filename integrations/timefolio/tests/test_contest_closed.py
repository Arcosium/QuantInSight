"""대회 종료(주문 창 폐쇄)는 장애가 아니라 상태로 처리된다 — 2026-09-10 사장 지시.

타임폴리오 대회가 끝나면 '신규 주문' 버튼이 없어 주문이 불가능하다. 예전엔 이걸 종목마다
ERROR + 트레이스백으로 올려 하루 300건씩 쌓였다(9/1~9/10 총 1,663건). 이제는 사유를
계정에 기록하고 사이클을 조용히 끝낸다.

Run: python3 -m pytest tests/test_contest_closed.py -q
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFOLIO_STATE_PATH", str(tmp_path / "contest_state.json"))
    for mod in [m for m in list(sys.modules) if "autofolio" in m]:
        sys.modules.pop(mod, None)
    from Auto_folio.autofolio import contest_store
    contest_store.register(4242, contest_id="test")
    contest_store.reset_portfolio(4242, cash=10_000_000.0)
    return contest_store


def test_contest_closed_is_recorded_and_cycle_stays_ok(store, monkeypatch):
    from Auto_folio.autofolio import naver_cycle
    from Auto_folio.autofolio.timefolio_browser import ContestClosedError

    def _closed(*_a, **_kw):
        raise ContestClosedError("타임폴리오 대회 종료 — 신규 주문 창이 열리지 않습니다(주문 불가).")

    monkeypatch.setattr(naver_cycle, "_place_or_submit", _closed)
    res = naver_cycle.run_cycle(4242, targets=["005930"], sell_targets=[],
                                max_buys=1, executor=lambda o: {})

    assert res["ok"] is True and res["contest_closed"] is True   # 사이클 실패가 아니다
    assert any(e.get("contest_closed") for e in res["events"])
    rec = store.contest_closed(4242)
    assert rec and "대회 종료" in rec["reason"] and rec["since"]


def test_clear_on_resume(store):
    store.mark_contest_closed(4242, "대회 종료 — 주문 불가")
    assert store.contest_closed(4242) is not None
    store.clear_contest_closed(4242)          # 주문이 다시 접수되면 해제
    assert store.contest_closed(4242) is None

"""Running producer publishes; consumers reuse or compute after bounded waiting."""
import asyncio
from datetime import datetime, timezone, timedelta
import main_swarm
from infra.market_intel import MarketIntelligenceStore

KST = timezone(timedelta(hours=9))

class _StubRuntime:
    def __init__(self, params): self._p = params
    def get(self, k, default=None, uid=None): return self._p.get(k, default)

def _setup(monkeypatch, *, share=True, wait=0.05):
    from types import SimpleNamespace as N
    from infra.user_context import REGISTRY
    monkeypatch.setattr(REGISTRY, "all_contexts", lambda: {
        1: N(task=N(done=lambda: False)), 2: N(task=N(done=lambda: False))})
    monkeypatch.setattr(main_swarm, "_now_kst",
                        lambda: datetime(2026, 6, 8, 10, 5, 0, tzinfo=KST))
    monkeypatch.setattr(main_swarm, "runtime", _StubRuntime(
        {"SHARE_MARKET_INTELLIGENCE": share, "SHARE_PRODUCER_WAIT_SEC": wait}))
    store = MarketIntelligenceStore()
    monkeypatch.setattr(main_swarm, "get_intel_store", lambda: store)
    return store

def _orch(is_admin):
    o = main_swarm.ArquantOrchestrator.__new__(main_swarm.ArquantOrchestrator)
    o.uid = 1 if is_admin else 2
    o.is_admin = is_admin
    o._producer_absent_this_cycle = False
    return o

def _counter():
    calls = {"n": 0}
    async def compute():
        calls["n"] += 1
        return "COMPUTED"
    return calls, compute

def test_admin_computes_and_publishes(monkeypatch):
    store = _setup(monkeypatch)
    o = _orch(True); calls, compute = _counter()
    res = asyncio.run(o._shared_or_compute("macro_report", None, compute))
    assert res == "COMPUTED" and calls["n"] == 1
    assert store.peek("macro_report", "2026-06-08 10", None) == "COMPUTED"

def test_admin_empty_result_not_published(monkeypatch):
    store = _setup(monkeypatch)
    o = _orch(True)
    async def empty(): return ""
    res = asyncio.run(o._shared_or_compute("macro_report", None, empty))
    assert res == ""
    assert store.peek("macro_report", "2026-06-08 10", None) is None

def test_consumer_hit_does_not_compute(monkeypatch):
    store = _setup(monkeypatch)
    asyncio.run(store.publish("macro_report", "2026-06-08 10", "SHARED", None, uid=1, now=1.0))
    o = _orch(False); calls, compute = _counter()
    res = asyncio.run(o._shared_or_compute("macro_report", None, compute))
    assert res == "SHARED" and calls["n"] == 0

def test_consumer_miss_falls_back_to_compute(monkeypatch):
    """게시분이 전혀 없으면 대기 후 자체계산 — 단, 다음 단계까지 래치하지는 않는다."""
    _setup(monkeypatch, wait=0.05)
    o = _orch(False); calls, compute = _counter()
    res = asyncio.run(o._shared_or_compute("macro_report", None, compute))
    assert res == "COMPUTED" and calls["n"] == 1

def test_consumer_reuses_recent_publish_from_other_hour(monkeypatch):
    """2026-07-22: :59 게시분을 :00 사이클이 hour_key 불일치만으로 버리고 재계산하던 문제.
    SHARE_STALE_OK_SEC 안이면 시각이 달라도 재사용해야 한다."""
    import time
    store = _setup(monkeypatch, wait=0.05)
    asyncio.run(store.publish("macro_report", "2026-06-08 09", "SHARED", None,
                              uid=1, now=time.time()))
    o = _orch(False); calls, compute = _counter()
    res = asyncio.run(o._shared_or_compute("macro_report", None, compute))
    assert res == "SHARED" and calls["n"] == 0

def test_consumer_ignores_too_old_publish(monkeypatch):
    """TTL 을 넘긴 게시분은 재사용하지 않는다(신선도 보호)."""
    import time
    store = _setup(monkeypatch, wait=0.05)
    asyncio.run(store.publish("macro_report", "2026-06-08 09", "SHARED", None,
                              uid=1, now=time.time() - 3 * 3600))
    o = _orch(False); calls, compute = _counter()
    res = asyncio.run(o._shared_or_compute("macro_report", None, compute))
    assert res == "COMPUTED" and calls["n"] == 1

def test_share_disabled_always_computes(monkeypatch):
    store = _setup(monkeypatch, share=False)
    o = _orch(True); calls, compute = _counter()
    res = asyncio.run(o._shared_or_compute("macro_report", None, compute))
    assert res == "COMPUTED" and calls["n"] == 1
    assert store.peek("macro_report", "2026-06-08 10", None) is None

def test_absent_producer_latches_after_first_wait(monkeypatch):
    """게시 이력이 전혀 없으면 = 생산자 부재 → 래치해 남은 단계는 즉시 자체계산.
    (2026-07-22: 래치를 아예 없앴더니 부재 시 단계마다 420초씩 헛기다리는 회귀가 있었다.)"""
    _setup(monkeypatch, wait=0.05)
    o = _orch(False); calls, compute = _counter()
    assert asyncio.run(o._shared_or_compute("macro_report", None, compute)) == "COMPUTED"
    assert o._producer_absent_this_cycle is True
    assert asyncio.run(o._shared_or_compute("news_report", None, compute)) == "COMPUTED"
    assert calls["n"] == 2

def test_slow_producer_does_not_latch(monkeypatch):
    """게시 이력이 있으면(=생산자 살아있음, 이번 시각엔 아직 안 나옴) 래치하지 않는다 —
    기다리는 편이 이득이다(자체계산은 같은 GPU 를 두고 생산자와 경합)."""
    import time
    store = _setup(monkeypatch, wait=0.05)
    # TTL 밖(오래된) 게시 이력만 존재 → 재사용은 불가하지만 '생산자는 있다'는 신호
    asyncio.run(store.publish("macro_report", "2026-06-08 08", "OLD", None,
                              uid=1, now=time.time() - 10 * 3600))
    o = _orch(False); calls, compute = _counter()
    assert asyncio.run(o._shared_or_compute("macro_report", None, compute)) == "COMPUTED"
    assert o._producer_absent_this_cycle is False


def test_paper_profile_computes_immediately_when_real_is_stopped(monkeypatch):
    from types import SimpleNamespace as N
    from unittest.mock import AsyncMock
    from infra.user_context import REGISTRY
    store = _setup(monkeypatch)
    monkeypatch.setattr(REGISTRY, "all_contexts", lambda: {
        1: N(task=None), 2: N(task=N(done=lambda: False))})
    wait = AsyncMock(side_effect=AssertionError("must not wait for stopped real account"))
    monkeypatch.setattr(store, "wait_for", wait)
    o = _orch(False)
    calls, compute = _counter()
    assert asyncio.run(o._shared_or_compute("news_report", None, compute)) == "COMPUTED"
    assert calls["n"] == 1
    assert store.peek("news_report", "2026-06-08 10", None) == "COMPUTED"
    wait.assert_not_called()


def test_completed_task_cannot_remain_the_producer(monkeypatch):
    from types import SimpleNamespace as N
    from infra.user_context import REGISTRY
    _setup(monkeypatch)
    monkeypatch.setattr(REGISTRY, "all_contexts", lambda: {
        1: N(task=N(done=lambda: True)), 2: N(task=N(done=lambda: False))})
    assert _orch(False)._intelligence_producer_uid() == 2

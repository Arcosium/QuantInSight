"""유휴 USD → KRW 자동 역환전 게이트 (infra.fx_reconvert) — 사장 보고 2026-06-26.

핵심 불변식:
  - 기본(AUTO OFF) 이면 실환전 금지, 유휴 USD ≥ 임계 시 '환전 필요' 알림만 띄운다.
  - dry_run / is_mock 이면 실환전 절대 금지.
  - 임계 미만(잔돈)·무데이터는 조용히 스킵.
  - AUTO ON + KIS 환전 TR 없음 → 수동 환전 필요 신호(조용히 누락 금지).
  - KRW 부족분이 주어지면 그만큼만(USD 환산) 환전 — KRW 한도와 USD 평가를 분리.
"""
import asyncio

import pytest

from infra import fx_reconvert


class _Notifier:
    def __init__(self):
        self.calls = []

    def alert(self, level, title, detail, *, dedup_key=None, dedup_window_sec=0):
        self.calls.append({"level": level, "title": title, "detail": detail, "dedup_key": dedup_key})
        return True


class _Broker:
    def __init__(self, *, usd=0.0, exrt=1480.0, ok=True, is_mock=False):
        self.is_mock = is_mock
        self._idle = {"ok": ok, "usd": usd, "krw_value": usd * exrt, "exrt": exrt}
        self.exchange_calls = []

    async def idle_usd_deposit(self):
        return dict(self._idle)

    async def us_to_krw_exchange(self, usd_amount, *, dry_run=True, reason=""):
        self.exchange_calls.append({"usd": usd_amount, "dry_run": dry_run, "reason": reason})
        # 본 코드베이스 기본: KIS 환전 TR 없음 → manual_required (실주문 안 함).
        if dry_run:
            return {"ok": False, "reason": "dry-run", "manual_required": False, "dry_run": True}
        return {"ok": False, "reason": "KIS 공개 환전 TR 없음 — 수동 환전 필요",
                "manual_required": True, "usd": usd_amount}


def _run(coro):
    # asyncio.get_event_loop() 를 쓰면 **테스트 순서에 따라** 깨진다: 앞서 실행된 테스트가
    # asyncio.run() 을 한 번이라도 부르면(예: tests/test_app_per_uid_lifecycle.py,
    # tests/test_balance_failed_read_carryforward.py) Runner 가 종료 시 set_event_loop(None)
    # 을 호출해 policy 의 _set_called 가 True + 현재 루프 None 이 된다 → 이후 get_event_loop()
    # 는 루프를 새로 만들지 않고 RuntimeError('There is no current event loop') 를 던진다
    # (단독 실행 시엔 통과, 전체 실행 시엔 7건 전부 실패하던 원인).
    # 자기 루프를 직접 확보해 주변 상태에 의존하지 않는다(3.12 에서 get_event_loop 는 deprecated).
    return asyncio.run(coro)


def test_idle_usd_off_default_alerts_only(monkeypatch):
    monkeypatch.setattr("runtime.get", lambda k, d=None, uid=None: d)  # config 기본값 사용
    br = _Broker(usd=962.0, exrt=1480.0)
    nt = _Notifier()
    res = _run(fx_reconvert.maybe_reconvert_idle_usd(br, dry_run=True, uid="u1", notifier=nt))
    assert res["action"] == "alert"
    assert br.exchange_calls == [], "AUTO OFF 면 실환전 진입점을 호출하지 않는다"
    assert nt.calls and "환전" in nt.calls[0]["title"]


def test_below_min_skips_quietly(monkeypatch):
    monkeypatch.setattr("runtime.get", lambda k, d=None, uid=None: d)
    br = _Broker(usd=10.0, exrt=1480.0)  # USD_RECONVERT_MIN_USD(100) 미만
    nt = _Notifier()
    res = _run(fx_reconvert.maybe_reconvert_idle_usd(br, dry_run=True, uid="u1", notifier=nt))
    assert res["action"] == "skip" and res["reason"] == "below_min"
    assert nt.calls == []


def test_mock_account_skips(monkeypatch):
    monkeypatch.setattr("runtime.get", lambda k, d=None, uid=None: d)
    br = _Broker(usd=5000.0, is_mock=True)
    res = _run(fx_reconvert.maybe_reconvert_idle_usd(br, dry_run=True, uid="u1", notifier=_Notifier()))
    assert res["action"] == "skip" and res["reason"] == "mock"


def test_no_data_skips(monkeypatch):
    monkeypatch.setattr("runtime.get", lambda k, d=None, uid=None: d)
    br = _Broker(usd=0.0, ok=False)
    res = _run(fx_reconvert.maybe_reconvert_idle_usd(br, dry_run=True, uid="u1", notifier=_Notifier()))
    assert res["action"] == "skip" and res["reason"] == "no_data"


def test_auto_on_dry_run_never_fires_real(monkeypatch):
    # AUTO ON 이어도 dry_run 이면 실환전 금지 — 진입점은 dry_run=True 로 호출되고 noop 으로 끝난다.
    monkeypatch.setattr("runtime.get",
                        lambda k, d=None, uid=None: True if k == "AUTO_USD_TO_KRW_RECONVERT" else d)
    br = _Broker(usd=962.0, exrt=1480.0)
    res = _run(fx_reconvert.maybe_reconvert_idle_usd(br, dry_run=True, uid="u1", notifier=_Notifier()))
    assert br.exchange_calls and br.exchange_calls[0]["dry_run"] is True
    assert res["action"] == "noop"


def test_auto_on_live_no_tr_signals_manual(monkeypatch):
    # AUTO ON + LIVE(dry_run=False) 이지만 KIS 환전 TR 없음 → manual_required.
    monkeypatch.setattr("runtime.get",
                        lambda k, d=None, uid=None: True if k == "AUTO_USD_TO_KRW_RECONVERT" else d)
    br = _Broker(usd=962.0, exrt=1480.0)
    nt = _Notifier()
    res = _run(fx_reconvert.maybe_reconvert_idle_usd(br, dry_run=False, uid="u1", notifier=nt))
    assert res["action"] == "manual_required"
    assert br.exchange_calls and br.exchange_calls[0]["dry_run"] is False
    assert any("수동 환전" in c["title"] for c in nt.calls)


def test_krw_shortfall_caps_amount(monkeypatch):
    # KRW 부족분이 작으면 그만큼만(USD 환산) — 유휴 USD 전액이 아니라 부족분만 환전 대상.
    monkeypatch.setattr("runtime.get", lambda k, d=None, uid=None: d)
    br = _Broker(usd=962.0, exrt=1480.0)
    nt = _Notifier()
    res = _run(fx_reconvert.maybe_reconvert_idle_usd(
        br, dry_run=True, uid="u1", notifier=nt, krw_shortfall=148000.0))  # ≈ $100
    assert res["action"] == "alert"
    assert abs(res["want_usd"] - 100.0) < 1.0, "부족분 환산만큼만 환전 대상이어야 한다"

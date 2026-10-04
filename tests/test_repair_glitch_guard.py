"""원장 누락매수 자동보정(repair)의 KIS 글리치-高 방어 (버그 D, 2026-06-18).

KIS 잔고 글리치는 보유를 일시적으로 부풀려 읽을 수 있다(글리치-高). repair 가 그 한 번의
읽기를 믿고 누락매수로 baked 하면 phantom 이 생긴다(160980: 글리치 255 → 84주 주입 →
다음 사이클 KIS 171 과 괴리). prune_phantoms 처럼 '연속 N회' 확인된 괴리만 보정한다.
"""
import json
import infra.trade_ledger as tl


def _seed(tmp_path, uid, positions):
    d = tmp_path / str(uid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "ledger.json").write_text(json.dumps({
        "version": 1, "seeded_at": "2026-06-18 09:00:00", "seed_source": "test",
        "cash_krw": 0, "cash_usd": 0, "positions": positions, "fills": [], "degraded_fills": 0}),
        encoding="utf-8")


_PARTIAL_BUY_CYCLE = {"orders_executed": json.dumps([
    {"ticker": "160980", "side": "buy", "qty": 0, "order_qty": 84,
     "accepted": True, "filled": False, "fill_price": 10000, "avg_cost": 10000}])}


def test_repair_waits_for_confirmation(tmp_path, monkeypatch):
    # _DATA_DIR 만 tmp 로(≠ _DEFAULT_DATA_DIR) → _writes_allowed True
    monkeypatch.setattr(tl, "_DATA_DIR", tmp_path)
    monkeypatch.setattr("infra.cycle_store.list_cycles", lambda *a, **k: [_PARTIAL_BUY_CYCLE])
    _seed(tmp_path, 1, {"160980": {"qty": 171, "avg_cost": 10000, "ccy": "KRW", "last_price": 10000}})
    kis_glitch_high = [{"code": "160980", "qty": 255, "avg_price": 10000}]  # 글리치-高

    r1 = tl.repair_from_recent_partial_orders(1, kis_glitch_high)
    assert r1 == []                                           # 1회차: 확인 부족 → 보정 보류
    assert tl.load(1)["positions"]["160980"]["qty"] == 171    # 원장 불변

    r2 = tl.repair_from_recent_partial_orders(1, kis_glitch_high)
    assert r2 and "160980" in r2[0]                           # 2회차: 연속 확인 → 보정
    assert tl.load(1)["positions"]["160980"]["qty"] == 255


def test_transient_glitch_not_baked(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "_DATA_DIR", tmp_path)
    monkeypatch.setattr("infra.cycle_store.list_cycles", lambda *a, **k: [_PARTIAL_BUY_CYCLE])
    _seed(tmp_path, 1, {"160980": {"qty": 171, "avg_cost": 10000, "ccy": "KRW", "last_price": 10000}})

    # cycle1: 글리치-高 255 (스트릭 1)
    tl.repair_from_recent_partial_orders(1, [{"code": "160980", "qty": 255, "avg_price": 10000}])
    # cycle2: KIS 정상 171 로 복귀 → 괴리 사라짐 → 스트릭 리셋, 보정 없음
    r = tl.repair_from_recent_partial_orders(1, [{"code": "160980", "qty": 171, "avg_price": 10000}])
    assert r == []
    assert tl.load(1)["positions"]["160980"]["qty"] == 171     # phantom 안 생김

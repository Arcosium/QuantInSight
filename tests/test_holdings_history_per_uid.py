"""holdings_history per-uid 격리 — 두 계정이 같은 종목 보유 시 보유기간이 섞이지 않게 (2026-06-15).

버그: holdings_history 가 (code, first_seen) PK·uid 없음 → 462870 을 양 계정이 보유하면
한 계정의 upsert/매도가 다른 계정의 보유기간을 덮어씀. 수정: uid 를 스키마·전 쿼리에 추가.
"""
import infra.cycle_store as cs


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "DB_PATH", tmp_path / "cyc_test.db")
    monkeypatch.setattr(cs, "_conn", None)


def test_holding_period_isolated_per_uid(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    cs.upsert_holding_seen("462870", 8, 32000.0, uid=1)
    cs.upsert_holding_seen("462870", 314, 32050.0, uid=2)
    hp1 = cs.get_holding_period("462870", uid=1)
    hp2 = cs.get_holding_period("462870", uid=2)
    assert hp1 and hp1["qty"] == 8
    assert hp2 and hp2["qty"] == 314          # uid2가 uid1을 덮지 않음


def test_close_one_uid_does_not_affect_other(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    cs.upsert_holding_seen("462870", 8, 32000.0, uid=1)
    cs.upsert_holding_seen("462870", 314, 32050.0, uid=2)
    cs.mark_position_closed("462870", uid=1)
    assert cs.get_holding_period("462870", uid=1) is None
    assert cs.get_holding_period("462870", uid=2)["qty"] == 314   # uid2 보유기간 유지


def test_reconcile_holdings_scoped_to_uid(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    cs.upsert_holding_seen("462870", 8, 32000.0, uid=1)
    cs.upsert_holding_seen("462870", 314, 32050.0, uid=2)
    # uid1 사이클이 462870 을 더는 안 가짐 → uid1 것만 닫히고 uid2 는 영향 없어야
    cs.reconcile_holdings([], uid=1)
    assert cs.get_holding_period("462870", uid=1) is None
    assert cs.get_holding_period("462870", uid=2)["qty"] == 314

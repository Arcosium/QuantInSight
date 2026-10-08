"""자격증명 없는 페이퍼 등록 + auto_cycle 플래그 (사장 지시 2026-07-03)."""
import pytest

from Auto_folio.autofolio import contest_store


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(contest_store, "_STORE_PATH", tmp_path / "contest_state.json")
    monkeypatch.setattr(contest_store, "_DATA_DIR", tmp_path)


def test_register_paper_without_credentials(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    acc = contest_store.register(1, initial_cash=1_000_000)
    assert acc["contest_id"] == "paper"
    assert acc["has_site_credentials"] is False
    assert acc["portfolio"]["cash"] == 1_000_000


def test_register_with_password_requires_contest_id(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    try:
        contest_store.register(1, "", "pass123456!")
        assert False, "contest_id 없이 password 만 주면 거절해야 한다"
    except ValueError:
        pass


def test_relax_sector_allows_buy_without_sector(tmp_path, monkeypatch):
    """ArcTrade 러너 완화 모드: 섹터 데이터 부재는 경고로만 — 나머지 룰은 그대로 검증."""
    from Auto_folio.autofolio import contest_rules
    _isolate(tmp_path, monkeypatch)
    contest_store.register(1, initial_cash=100_000_000)
    meta = {"name": "삼성전자", "market": "KOSPI", "is_common_stock": True,
            "listed_business_days": 3000, "avg_5d_trading_value_krw": 1e12,
            "market_cap_krw": 4e14, "last_price": 70000}  # sector 없음
    monkeypatch.setattr(contest_rules, "_RELAX_SECTOR", False)
    strict = contest_store.check_order(1, "buy", "005930", 10, 70000, meta=meta)
    assert not strict["ok"] and any(v["code"] == "sector_missing" for v in strict["violations"])
    monkeypatch.setattr(contest_rules, "_RELAX_SECTOR", True)
    relaxed = contest_store.check_order(1, "buy", "005930", 10, 70000, meta=meta)
    assert relaxed["ok"] and relaxed["warnings"]


def test_auto_cycle_flag_roundtrip(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    contest_store.register(1)
    contest_store.register(2)
    assert contest_store.list_auto_cycle_uids() == []
    acc = contest_store.set_auto_cycle(1, True)
    assert acc["auto_cycle"] is True
    assert contest_store.list_auto_cycle_uids() == [1]
    contest_store.set_auto_cycle(1, False)
    assert contest_store.list_auto_cycle_uids() == []


def test_inferred_sell_uses_market_snapshot_not_cash_residual():
    """비동기 NAV/현금 잔차를 소량 수량으로 나눠 가짜 급등락 체결가를 만들지 않는다."""
    old = {"017960": {"qty": 1000, "avg_price": 25200, "last_price": 25150}}
    new = {"017960": {"qty": 985, "avg_price": 25200, "last_price": 25200}}
    fills = contest_store._infer_fills(
        old, new, old_cash=100_000_000, new_cash=100_400_000,
        ts="2026-08-18T00:50:00+00:00")
    sell = next(row for row in fills if row["side"] == "sell")
    assert sell["qty"] == 15
    assert sell["price"] == 25200
    assert sell["estimated"] is True
    assert sell["price_source"] == "market_snapshot"


def test_inferred_fill_session_rejects_after_hours():
    assert contest_store._in_site_fill_session("2026-08-18T00:00:00+00:00") is True   # 09:00 KST
    assert contest_store._in_site_fill_session("2026-08-18T06:40:00+00:00") is True   # 15:40 KST
    assert contest_store._in_site_fill_session("2026-08-18T09:05:00+00:00") is False  # 18:05 KST


def test_consolidate_inferred_partial_fills_preserves_episode_total():
    rows = [
        {"ts": "2026-08-18T00:50:00+00:00", "side": "sell", "ticker": "017960",
         "qty": 55, "price": 22371, "avg_price": 25217, "pnl": -156530,
         "accepted": True, "filled": True, "inferred": True},
        {"ts": "2026-08-18T00:50:20+00:00", "side": "sell", "ticker": "017960",
         "qty": 366, "price": 26858, "avg_price": 25217, "pnl": 600606,
         "accepted": True, "filled": True, "inferred": True},
        {"ts": "2026-08-18T00:51:00+00:00", "side": "sell", "ticker": "017960",
         "qty": 803, "price": 25006, "avg_price": 25217, "pnl": -169433,
         "accepted": True, "filled": True, "inferred": True},
    ]
    out = contest_store.consolidate_inferred_trades(rows)
    assert len(out) == 1
    assert out[0]["qty"] == 1224
    assert out[0]["fill_count"] == 3
    assert out[0]["price"] == pytest.approx(sum(r["qty"] * r["price"] for r in rows) / 1224)
    assert out[0]["pnl"] == sum(r["pnl"] for r in rows)


def test_browser_launch_failure_stops_playwright(monkeypatch):
    """Chromium 누락 시 Playwright 드라이버가 새어 다음 재시도를 asyncio 오류로 바꾸지 않는다."""
    from Auto_folio.autofolio import timefolio_browser

    class Chromium:
        def launch(self, **_kwargs):
            raise RuntimeError("browser executable missing")

    class Driver:
        chromium = Chromium()
        stopped = False

        def stop(self):
            self.stopped = True

    driver = Driver()

    class Starter:
        def start(self):
            return driver

    monkeypatch.setattr(timefolio_browser, "sync_playwright", lambda: Starter())
    browser = timefolio_browser.TimefolioBrowser(headless=True)
    with pytest.raises(RuntimeError, match="executable missing"):
        browser.open()
    assert driver.stopped is True
    assert browser._pw is None

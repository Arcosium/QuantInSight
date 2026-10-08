from datetime import timedelta

from web import autofolio


def test_new_site_trading_day_uses_kst_date():
    previous = autofolio.mt.now_kst() - timedelta(days=1)
    current = autofolio.mt.now_kst()
    assert autofolio._new_site_trading_day({"site_synced_at": previous.isoformat()}) is True
    assert autofolio._new_site_trading_day({"site_synced_at": current.isoformat()}) is False
    assert autofolio._new_site_trading_day({}) is False


def test_previous_day_working_order_is_force_canceled(monkeypatch):
    class Browser:
        def list_working_orders(self):
            return [{"ticker": "005930"}]

        def cancel_working_orders(self, ticker, *, min_age_min=0):
            assert ticker == "005930"
            assert min_age_min == 0
            return {"ok": True, "stopped": 1, "result": "정지"}

    monkeypatch.setattr(autofolio.contest_store, "track_fill_progress",
                        lambda *_args, **_kwargs: {"stalled_min": 0.0})
    monkeypatch.setattr(autofolio.contest_store, "clear_recent_sell", lambda *_args: None)
    account = {"portfolio": {"positions": [{"ticker": "005930", "qty": 10}]}}
    events = autofolio._reconcile_stale_orders(Browser(), account, force_cancel=True)
    assert events[0]["reason"] == "previous_day_order"

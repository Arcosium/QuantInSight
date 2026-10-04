"""Strategy parity and execution locks; no network or production state."""
import asyncio
from types import SimpleNamespace as N
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.mark.parametrize("mock", [True, False])
@pytest.mark.parametrize("pnl,expected_sells", [(20., 0), (-6., 1)])
def test_systematic_exits_identical_and_hard_stops_retained(monkeypatch, mock, pnl, expected_sells):
    import config
    import main_swarm as m
    params = dict(config.STRATEGY_DEFAULTS, TAKE_PROFIT_PCT=12., STOP_LOSS_PCT=5.,
                  ENABLE_LEADLAG_SELL=False, TRIM_OVER_RATIO=False, ENABLE_SELL_REBALANCE=True)
    monkeypatch.setattr(m.runtime, "get", lambda key, **kw: params.get(key, 0))
    monkeypatch.setattr(m, "is_market_session_now", lambda *a, **kw: False)
    monkeypatch.setattr(m, "get_current_session", lambda: "KR_TRADING")
    monkeypatch.setattr(m, "get_usdkrw", lambda *a, **kw: 1400.)
    broker = N(is_mock=mock, kr_account_snapshot=AsyncMock(return_value={
        "buying_power": {"cash": 1000000., "total_eval": 1000000.}}),
        kr_last_price=AsyncMock(return_value=100.), kr_psbl_sell_qty=AsyncMock(return_value=10))
    o = m.ArquantOrchestrator.__new__(m.ArquantOrchestrator)
    o.uid = 12345
    o.broker = broker
    o._emit = AsyncMock()
    held = [{"code": "005930", "qty": 10, "cur_price": 100., "pnl_pct": pnl,
             "avg_price": 100./(1+pnl/100)}]
    obj, _, _ = asyncio.run(o._build_orders([], [], "", "", held, systematic_policy={"status": "ready"}))
    sells = [x for x in obj["orders"] if x["side"] == "sell"]
    assert len(sells) == expected_sells
    if sells:
        assert sells[0]["ticker"] == "005930" and sells[0]["qty"] == 10


def test_real_start_cannot_create_task_or_resume_marker(monkeypatch, tmp_path):
    import config
    from infra import user_paths
    from server import app as a
    from fastapi import HTTPException
    marker = Mock(return_value=tmp_path / ".running")
    ctx = N(creds={"account_mode": "trading", "kis_app_key": "test", "kis_account_no": "test",
                   "kis_base_url": "https://openapi.koreainvestment.com:9443"}, task=None)
    monkeypatch.setattr(config, "PAPER_ONLY", True)
    monkeypatch.setattr(a.REGISTRY, "get_or_create", lambda uid: ctx)
    monkeypatch.setattr(user_paths, "running_marker", marker)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(a._start_uid(12345))
    assert exc.value.status_code == 409
    assert ctx.task is None
    marker.assert_not_called()


def test_removed_competition_routes_and_profile_kind_rejected():
    from pydantic import ValidationError
    from server import app as a
    from infra import auth_store
    assert not any("autofolio" in route.path for route in a.app.routes)
    with pytest.raises(ValidationError):
        a.ProfileUpsertReq(kind="timefolio")
    with pytest.raises(ValueError):
        auth_store.normalize_account_mode("timefolio")
    assert auth_store.profile_kind_of({"account_mode": "timefolio", "kis_app_key": "test",
                                       "kis_account_no": "test"}) is None

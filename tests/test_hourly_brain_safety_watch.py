import inspect
import asyncio
import time

import config
import main_swarm
from main_swarm import (ArquantOrchestrator, SAFETY_WATCH_INTERVAL_SEC,
                        _safety_sell_reason, _safety_watch_active)


def test_continuous_cycle_choice_is_removed_from_strategy_surface():
    removed = {"CONTINUOUS_CYCLES", "CONTINUOUS_MIN_GAP_SEC", "CONTINUOUS_MIN_CYCLE_SEC"}
    assert removed.isdisjoint(config.STRATEGY_TUNABLE_KEYS)
    assert removed.isdisjoint(config.STRATEGY_KEY_META)
    assert all(not hasattr(config, key) for key in removed)


def test_brain_is_hourly_and_guard_is_fixed_60_seconds():
    assert config.PERIODIC_CYCLE_SEC == 3600
    assert SAFETY_WATCH_INTERVAL_SEC == 60
    source = inspect.getsource(ArquantOrchestrator.start_continuous)
    assert "if periodic_due and is_trading_hours()" in source
    assert "market_open or periodic_due" not in source


def test_safety_watch_only_triggers_hard_stops():
    holding = {"cur_price": 98.0, "pnl_pct": -2.0}
    thesis = {"stop_price": 93.0, "target_price": 105.0}
    assert _safety_sell_reason(holding, thesis, 5.0) is None
    assert "계획 손절가" in _safety_sell_reason(
        {"cur_price": 92.0, "pnl_pct": -3.0}, thesis, 5.0)
    assert "하드 손절" in _safety_sell_reason(
        {"cur_price": 95.0, "pnl_pct": -5.1}, thesis, 5.0)


def test_safety_watch_does_not_take_profit_or_rebalance():
    assert _safety_sell_reason(
        {"cur_price": 120.0, "pnl_pct": 20.0}, {"stop_price": 90.0}, 5.0) is None




def test_kis_safety_watch_keeps_supported_sessions():
    assert _safety_watch_active("KR_TRADING") is True
    assert _safety_watch_active("US_TRADING") is True

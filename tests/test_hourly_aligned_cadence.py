"""사이클 트리거 앵커를 벽시계 시(hour)로 — 진입 즉시 발화 안 함, :00에 발화."""
import time
from datetime import datetime, timezone, timedelta
import main_swarm

KST = timezone(timedelta(hours=9))

def test_current_hour_key_floors_to_hour(monkeypatch):
    monkeypatch.setattr(main_swarm, "_now_kst",
                        lambda: datetime(2026, 6, 8, 10, 37, 12, tzinfo=KST))
    assert main_swarm._current_hour_key() == datetime(2026, 6, 8, 10, 0, 0, tzinfo=KST)
    assert main_swarm._current_hour_key_str() == "2026-06-08 10"

def _orch():
    o = main_swarm.ArquantOrchestrator.__new__(main_swarm.ArquantOrchestrator)
    return o

def test_no_periodic_within_same_hour(monkeypatch):
    monkeypatch.setattr(main_swarm, "_now_kst",
                        lambda: datetime(2026, 6, 8, 10, 37, 0, tzinfo=KST))
    o = _orch()
    o._last_cycle_hour_key = main_swarm._current_hour_key()
    monkeypatch.setattr(main_swarm, "_now_kst",
                        lambda: datetime(2026, 6, 8, 10, 59, 0, tzinfo=KST))
    assert o._should_run_periodic() is False

def test_periodic_fires_on_hour_rollover(monkeypatch):
    monkeypatch.setattr(main_swarm, "_now_kst",
                        lambda: datetime(2026, 6, 8, 10, 37, 0, tzinfo=KST))
    o = _orch()
    o._last_cycle_hour_key = main_swarm._current_hour_key()
    monkeypatch.setattr(main_swarm, "_now_kst",
                        lambda: datetime(2026, 6, 8, 11, 0, 1, tzinfo=KST))
    assert o._should_run_periodic() is True


def test_missed_hour_is_not_caught_up_at_market_open(monkeypatch):
    monkeypatch.setattr(main_swarm, "_now_kst",
                        lambda: datetime(2026, 6, 8, 19, 0, 1, tzinfo=KST))
    o = _orch()
    o._last_cycle_hour_key = main_swarm._current_hour_key()
    # 장외 뒤 미국장 개장(22:30)에 오래된 시간대를 따라잡아 실행하지 않는다.
    monkeypatch.setattr(main_swarm, "_now_kst",
                        lambda: datetime(2026, 6, 8, 22, 30, 0, tzinfo=KST))
    assert o._should_run_periodic() is False
    assert o._last_cycle_hour_key == datetime(2026, 6, 8, 22, 0, 0, tzinfo=KST)
    # 앵커가 넘어갔으므로 다음 정각은 정상 발화한다.
    monkeypatch.setattr(main_swarm, "_now_kst",
                        lambda: datetime(2026, 6, 8, 23, 0, 1, tzinfo=KST))
    assert o._should_run_periodic() is True


def test_hourly_trigger_window_boundary(monkeypatch):
    o = _orch()
    o._last_cycle_hour_key = datetime(2026, 6, 8, 9, 0, 0, tzinfo=KST)
    monkeypatch.setattr(main_swarm, "_now_kst",
                        lambda: datetime(2026, 6, 8, 10, 1, 59, tzinfo=KST))
    assert o._should_run_periodic() is True
    monkeypatch.setattr(main_swarm, "_now_kst",
                        lambda: datetime(2026, 6, 8, 10, 2, 0, tzinfo=KST))
    assert o._should_run_periodic() is False

def test_restart_invariant_first_fire_at_next_hour(monkeypatch):
    monkeypatch.setattr(main_swarm, "_now_kst",
                        lambda: datetime(2026, 6, 8, 8, 37, 0, tzinfo=KST))
    o = _orch()
    o._last_cycle_hour_key = main_swarm._current_hour_key()
    monkeypatch.setattr(main_swarm, "_now_kst",
                        lambda: datetime(2026, 6, 8, 8, 59, 59, tzinfo=KST))
    assert o._should_run_periodic() is False
    monkeypatch.setattr(main_swarm, "_now_kst",
                        lambda: datetime(2026, 6, 8, 9, 0, 0, tzinfo=KST))
    assert o._should_run_periodic() is True


# ── 부팅 앵커가 5분 중복가드를 오염시키지 않음 (2026-06-19) ─────────────────────
# 버그: 부팅 시 _last_cycle_at=time.time() 으로 찍어 → :00 5분 이내 재시작하면 그 정시
# 사이클이 '직전 사이클 N분 전 — 중복 스킵' 게이트에 걸려 통째로 누락(uid 10시 사이클 유실 관측).
def test_init_cycle_anchors_clears_last_cycle_at(monkeypatch):
    monkeypatch.setattr(main_swarm, "_now_kst",
                        lambda: datetime(2026, 6, 19, 9, 58, 0, tzinfo=KST))
    o = _orch()
    o._last_cycle_at = 999999999.0          # 잔존 가짜값(부팅 전 상태 모사)
    o._init_cycle_anchors(force=False)
    assert o._last_cycle_at == 0.0           # 5분 가드 비활성 → 다음 :00 정시 사이클 발화 보장
    assert o._last_cycle_hour_key == main_swarm._current_hour_key()  # 현재 시 앵커(같은 시 재발화 방지)


def test_init_cycle_anchors_force_clears_hour_key(monkeypatch):
    monkeypatch.setattr(main_swarm, "_now_kst",
                        lambda: datetime(2026, 6, 19, 9, 58, 0, tzinfo=KST))
    o = _orch()
    o._init_cycle_anchors(force=True)        # force_first_cycle 마커 경로
    assert o._last_cycle_hour_key is None    # periodic_due=True 유발 → 시작 즉시 1사이클
    assert o._last_cycle_at == 0.0


def test_recent_cycle_dedup_inactive_at_boot_active_within_5min():
    o = _orch()
    o._last_cycle_at = 0.0                    # 부팅 직후
    assert o._recent_cycle_dedup_active() is False
    o._last_cycle_at = time.time() - 60       # 1분 전 실제 사이클
    assert o._recent_cycle_dedup_active() is True
    o._last_cycle_at = time.time() - 400      # 6분여 전
    assert o._recent_cycle_dedup_active() is False

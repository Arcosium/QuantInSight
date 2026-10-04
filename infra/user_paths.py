"""Per-uid data paths. Every user's runtime state lives under data/<uid>/."""
from __future__ import annotations
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent / "data"


def user_dir(uid: int) -> Path:
    d = _DATA_DIR / str(int(uid))
    d.mkdir(parents=True, exist_ok=True)
    return d


def equity_path(uid: int) -> Path:
    return user_dir(uid) / "equity_curve.json"


def trade_log_path(uid: int) -> Path:
    return user_dir(uid) / "trade_log.json"


def token_path(uid: int) -> Path:
    return user_dir(uid) / "kis_token.json"


def running_marker(uid: int) -> Path:
    return user_dir(uid) / ".running"


def position_thesis_path(uid: int) -> Path:
    return user_dir(uid) / "position_thesis.json"


def planner_sell_deferral_path(uid: int) -> Path:
    """포트폴리오기획팀장 1회 매도 보류 상태 — per-uid, 재시작 내성."""
    return user_dir(uid) / "planner_sell_deferrals.json"


def trailing_peaks_path(uid: int) -> Path:
    """트레일링 익절용 종목별 고점(peak_price·peak_pnl) 영속 — 사이클 간 유지·재시작 내성."""
    return user_dir(uid) / "trailing_peaks.json"


def locked_sell_streak_path(uid: int) -> Path:
    """매도가능 0 잠김(펜딩없음) 연속 사이클 카운트 — 임계 도달 시 에스컬레이션(버그 C)."""
    return user_dir(uid) / "locked_sell_streak.json"


def ledger_drift_streak_path(uid: int) -> Path:
    """원장↔KIS 괴리 연속 사이클 카운트 — 전이성(자가치유) 괴리 알림 억제용(2026-06-22)."""
    return user_dir(uid) / "ledger_drift_streak.json"


def ops_param_log_path(uid: int) -> Path:
    """운용지원실장 직전 파라미터 조정(키→{from,to,ts}) — anti-oscillation 진동 차단용(버그 B)."""
    return user_dir(uid) / "ops_param_log.json"


def sleeve_thesis_path(uid: int, sleeve_key: str) -> Path:
    """자산슬리브 thesis 경로. sleeve_key='bond' → bond_thesis.json(기존 라이브 파일과 동일 — 무손실),
    'commodity' → commodity_thesis.json."""
    return user_dir(uid) / f"{sleeve_key}_thesis.json"

"""리드-랙 신호 피드(data/signals.jsonl) 해석 — 순수 함수만 둔다.

이 모듈은 import 부작용이 없다(env 설정·상태파일 오픈 없음). web/autofolio.py 는 import 시점에
AUTOFOLIO_STATE_PATH 를 확정하고 contest_store 를 붙이므로, 테스트가 그걸 끌어오지 않도록
신호 해석부를 분리했다.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                  # noqa: E402
from utils import market_time as mt            # noqa: E402

log = logging.getLogger("autofolio.signals")

SIGNAL_LOOKBACK_MIN = 10   # BUY/SELL 신호를 이 분(min) 내에서만 유효한 후보로 인정
_TAIL_BYTES = 262_144
_ts_cache: dict = {"size": -1, "ts": ""}
_gate_logged: set = set()  # 같은 신호를 사이클마다 반복 로깅하지 않기 위한 (follower, ts) 기억


def passes_edge_gate(ev: dict) -> bool:
    """최소 기대엣지 게이트 — 대회 계정 **진입**만 거른다(청산은 절대 게이트하지 않음).

    stat-arb 신호(exp_ret 보유, 2026-07-28~): 기대 반전폭 exp_ret 이 왕복 마찰비용
    (0.4% = 수수료 0.1%×2 + 거래세 0.2%)을 넘어야 주문한다. 잔차가 아무리 저평가여도
    잡을 수 있는 반전폭이 마찰을 못 넘으면 방향이 맞아도 순손실이기 때문.
    엔진(strategy)이 이미 같은 조건으로 후보를 걸러 보내므로 여기선 이중 확인 역할.

    레거시 리드-랙 신호(leader_ret): 선행 급등폭 문턱(AUTOFOLIO_MIN_LEADER_RET_PCT).
    두 경우 다 문턱 ≤ 0(또는 필드 없음)이면 게이트 해제.
    """
    if ev.get("source") == "ai_strategy":
        return True
    if ev.get("exp_ret") is not None:
        thr = float(config.AUTOFOLIO_ROUND_TRIP_COST_PCT or 0.0)
        if thr <= 0:
            return True
        try:
            return float(ev.get("exp_ret") or 0.0) * 100.0 >= thr
        except (TypeError, ValueError):
            return False
    thr = float(config.AUTOFOLIO_MIN_LEADER_RET_PCT or 0.0)
    if thr <= 0:
        return True
    try:
        return float(ev.get("leader_ret") or 0.0) * 100.0 >= thr
    except (TypeError, ValueError):
        return False


def gate_reason(ev: dict) -> str | None:
    """BUY 신호가 엣지 게이트에 걸려 진입이 막힌 이유. 통과했거나 BUY 가 아니면 None.

    대시보드가 "신호는 떴는데 대회 계정엔 주문이 없다"를 설명하지 못하던 문제(2026-07-09) 때문에
    추가했다. 게이트 스킵은 journald INFO 로만 남아 화면에서는 원인을 알 수 없었다.
    """
    if str(ev.get("kind") or "").upper() != "BUY":
        return None
    if passes_edge_gate(ev):
        return None
    cost = float(config.AUTOFOLIO_ROUND_TRIP_COST_PCT or 0.0)
    if ev.get("exp_ret") is not None:              # stat-arb 신호
        try:
            edge = float(ev.get("exp_ret") or 0.0) * 100.0
        except (TypeError, ValueError):
            return f"기대 반전폭 해석 실패 → 진입 차단 (왕복 마찰 {cost:.1f}%)"
        return f"엣지 게이트: 기대 반전 {edge:.2f}% < 왕복 마찰 {cost:.1f}% → 대회 계정 주문 안 냄"
    thr = float(config.AUTOFOLIO_MIN_LEADER_RET_PCT or 0.0)   # 레거시 리드-랙
    if thr <= 0:
        return None
    try:
        ret = float(ev.get("leader_ret") or 0.0) * 100.0
    except (TypeError, ValueError):
        return f"선행 수익률 해석 실패 → 진입 차단 (문턱 {thr:.1f}%)"
    return (f"엣지 게이트: 선행 {ret:+.2f}% < {thr:.1f}% (왕복 마찰 {cost:.1f}%) → 대회 계정 주문 안 냄")


def _log_gate_skip(ev: dict, follower: str) -> None:
    key = (follower, str(ev.get("ts") or ""))
    if key in _gate_logged:
        return
    if len(_gate_logged) > 500:
        _gate_logged.clear()
    _gate_logged.add(key)
    log.info("엣지 게이트 스킵: %s 선행 %s %+.2f%% < %.1f%% (왕복 마찰 %.1f%%)",
             follower, ev.get("leader"), float(ev.get("leader_ret") or 0) * 100,
             config.AUTOFOLIO_MIN_LEADER_RET_PCT, config.AUTOFOLIO_ROUND_TRIP_COST_PCT)


def _events():
    """signals.jsonl 최근분(tail)을 순회한다. 파일이 커도 앞부분은 읽지 않는다."""
    path = config.SIGNALS_JSONL
    try:
        if not Path(path).exists():
            return
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return
    for line in lines[-2000:]:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except ValueError:
            continue


def _parsed(ev: dict):
    follower = str(ev.get("follower") or "").zfill(6)
    if not follower.strip("0"):
        return None, None
    try:
        return follower, datetime.fromisoformat(str(ev.get("ts")))
    except (ValueError, TypeError):
        return None, None


def buy_targets(within_min: int = SIGNAL_LOOKBACK_MIN) -> list[str]:
    """최근 within_min 분 내 BUY 신호가 뜬 follower 중, 이후 SELL 로 닫히지 않은(=아직 열린 롱)
    종목만 타임폴리오 매수 후보로 돌려준다. 신호가 없으면 [] → 사이클이 매수를 하지 않는다.
    (과거 DEFAULT_TARGETS 블루칩 기계매수 폐지 — 사장 지시 2026-07-07.)
    마찰비용을 못 넘길 약한 신호는 passes_edge_gate 에서 걸러진다.
    """
    try:
        cutoff = mt.now_kst() - timedelta(minutes=max(0, int(within_min or 0)))
        open_buys: dict = {}
        for ev in _events():
            follower, ts = _parsed(ev)
            if follower is None:
                continue
            kind = str(ev.get("kind") or "").upper()
            if kind == "BUY":
                if ts >= cutoff:
                    if passes_edge_gate(ev):
                        open_buys[follower] = ts
                    else:
                        _log_gate_skip(ev, follower)
            elif kind == "SELL":               # 이후 SELL 이면 닫힌 것 → 후보에서 제거
                prev = open_buys.get(follower)
                if prev is not None and ts >= prev:
                    open_buys.pop(follower, None)
        return [f for f, _ in sorted(open_buys.items(), key=lambda kv: kv[1], reverse=True)]
    except Exception:  # noqa: BLE001 — 신호 해석 실패가 사이클을 죽이면 안 된다
        log.exception("buy_targets 해석 실패")
        return []


def sell_targets(within_min: int = SIGNAL_LOOKBACK_MIN) -> list[str]:
    """최근 within_min 분 내 SELL 이 뜬 follower 목록. 보유 중이면 이 사이클에 청산한다 —
    buy 뿐 아니라 sell 도 신호 로직을 따르게(사장 지시 2026-07-07). 게이트를 적용하지 않는다:
    진입은 걸러도 **청산은 절대 막지 않는다**.
    """
    try:
        cutoff = mt.now_kst() - timedelta(minutes=max(0, int(within_min or 0)))
        out: dict = {}
        for ev in _events():
            if str(ev.get("kind") or "").upper() != "SELL":
                continue
            follower, ts = _parsed(ev)
            if follower is None:
                continue
            if ts >= cutoff:
                out[follower] = ts
        return sorted(out, key=lambda k: out[k], reverse=True)
    except Exception:  # noqa: BLE001
        log.exception("sell_targets 해석 실패")
        return []


def latest_actionable_ts() -> str:
    """최신 BUY/SELL 시각(ISO). 신호 즉시 사이클을 띄우기 위한 트리거용.
    SHORT_SIGNAL(현물 미지원)은 무시한다. 파일 크기가 그대로면 파싱을 건너뛴다."""
    path = config.SIGNALS_JSONL
    try:
        size = os.path.getsize(path)
    except OSError:
        return _ts_cache["ts"]
    if size == _ts_cache["size"]:
        return _ts_cache["ts"]
    latest = ""
    try:
        with open(path, "rb") as f:
            if size > _TAIL_BYTES:
                f.seek(size - _TAIL_BYTES)
                f.readline()          # 잘린 첫 줄 버림
            for raw in f:
                try:
                    ev = json.loads(raw.decode("utf-8").strip() or "{}")
                except (ValueError, UnicodeDecodeError):
                    continue
                if str(ev.get("kind") or "").upper() in ("BUY", "SELL"):
                    ts = str(ev.get("ts") or "")
                    if ts > latest:
                        latest = ts
    except OSError:
        return _ts_cache["ts"]
    _ts_cache.update(size=size, ts=latest or _ts_cache["ts"])
    return _ts_cache["ts"]

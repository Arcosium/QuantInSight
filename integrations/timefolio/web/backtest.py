"""신호 로그 실현손익 + 선행→후행 페어의 역사적 적중률.

원래는 브라우저가 `/api/signals` 와 `/api/bars/*` 를 각각 불러 계산했다. 세 가지가 문제였다:

  1. **깜빡임** — 롱은 즉시, 숏은 `/api/bars` 왕복 뒤에 렌더돼 15초마다 롱숏 화면이
     "롱만" → "롱+숏" 으로 두 번 그려졌다. 여기서 한 번에 계산해 한 번만 렌더한다.
  2. **숏 원화손익 없음** — 숏은 실제 주문이 없어 수량이 없었다. 엔진이 롱에 쓰는 것과
     같은 사이징(`ORDER_NOTIONAL // 진입가`)으로 가상 수량을 잡아 원화 손익을 낸다.
  3. **지도 승률 없음** — 페어가 역사적으로 몇 번 맞고 틀렸는지는 bars.db 전량 스캔이라
     브라우저에서 할 수 없다.

적중 판정은 **엔진 진입 규칙을 그대로 복제**한다(`core/strategy.py::check_entries`):
선행주 1분 수익률이 ±LEADER_JUMP 를 넘고 그 순간 후행주가 아직 안 움직였으면(|f_ret| ≤
FOLLOWER_FLAT) 신호. lag 분 뒤 후행주가 선행주와 같은 방향으로 움직였으면 '맞춤'.
정의가 엔진과 어긋나면 승률은 아무 의미가 없으므로 상수는 config 에서 그대로 읽는다.

순수 로직(파일·DB 접근 없음) — 시세는 `closes: {code: {ts: close}}` 로 주입한다.
테스트: tests/test_backtest.py
"""
from __future__ import annotations

from datetime import datetime, timedelta

import config

# 숏 청산가를 찾을 때 lag 분 뒤부터 몇 분까지 뒤져볼지(분봉 결측 대비).
_EXIT_SEARCH_MIN = 4


def minute_key(dt: datetime) -> str:
    """bars.db 의 ts 포맷 — KST 'YYYYMMDDHHMM'."""
    return dt.strftime("%Y%m%d%H%M")


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def _short_qty(entry_price: float) -> int:
    """엔진이 롱 진입에 쓰는 사이징과 동일(`strategy._enter`). 숏은 가상 포지션."""
    if not entry_price or entry_price <= 0:
        return 0
    return max(1, int(config.ORDER_NOTIONAL // entry_price))


def _exit_close(bars: dict, entry_dt: datetime, lag: int) -> float | None:
    """신호 lag 분 뒤 후행주 종가. 그 분봉이 없으면 최대 _EXIT_SEARCH_MIN 분 더 본다."""
    for k in range(lag, lag + _EXIT_SEARCH_MIN + 1):
        c = bars.get(minute_key(entry_dt + timedelta(minutes=k)))
        if c is not None:
            return float(c)
    return None


def resolve_signals(signals: list[dict], closes: dict[str, dict]) -> dict:
    """signals(시간 오름차순)에 실현손익을 붙인다. 원본을 수정하지 않고 사본을 돌려준다.

    반환: {"signals": [...], "unresolved": int}
      · BUY  → ret/krw 없음(진입일 뿐이다).
      · SELL → 짝지어진 BUY 로 ret(로그의 pnl 우선)·krw 계산. type="L".
      · SHORT_SIGNAL → bars 로 청산가 시뮬 → ret/krw. 분봉이 없으면 resolved=False.
    """
    out: list[dict] = []
    open_buys: dict[str, list[dict]] = {}
    unresolved = 0

    for raw in signals:
        rec = dict(raw)
        kind = rec.get("kind")
        if kind == "BUY":
            open_buys.setdefault(rec["follower"], []).append(rec)
        elif kind == "SELL":
            q = open_buys.get(rec["follower"]) or []
            buy = q.pop(0) if q else None
            qty = rec.get("qty") or (buy or {}).get("qty") or 0
            entry = (buy or {}).get("price")
            pnl = rec.get("pnl")
            if pnl is None:
                pnl = (rec["price"] / entry - 1.0) if entry else 0.0
            rec.update(trade_type="L", resolved=True, entry_price=entry,
                       exit_price=rec.get("price"), ret=pnl,
                       krw=((rec["price"] - entry) * qty) if entry else 0.0)
        elif kind == "SHORT_SIGNAL":
            dt = _parse_ts(rec.get("ts", ""))
            bars = closes.get(rec.get("follower", "")) or {}
            entry = rec.get("price")
            exit_px = _exit_close(bars, dt, int(rec.get("lag") or 1)) if dt else None
            if exit_px is None or not entry:
                unresolved += 1
                rec.update(trade_type="S", resolved=False)
            else:
                qty = _short_qty(entry)
                rec.update(trade_type="S", resolved=True, entry_price=entry,
                           exit_price=exit_px, ret=(entry - exit_px) / entry,
                           krw=(entry - exit_px) * qty)   # 숏: 하락하면 이익
        out.append(rec)
    return {"signals": out, "unresolved": unresolved}


def _returns(bars: dict) -> dict[str, float]:
    """분봉 종가 → 직전 분 대비 1분 수익률. 분봉이 끊긴 구간은 건너뛴다."""
    if not bars:
        return {}
    keys = sorted(bars)
    rets: dict[str, float] = {}
    for i in range(1, len(keys)):
        prev_k, k = keys[i - 1], keys[i]
        # 분봉이 1분 간격일 때만 '1분 수익률' 이다(장 마감 경계에서 건너뛰기 금지).
        if (datetime.strptime(k, "%Y%m%d%H%M")
                - datetime.strptime(prev_k, "%Y%m%d%H%M")) != timedelta(minutes=1):
            continue
        p = bars[prev_k]
        if p:
            rets[k] = bars[k] / p - 1.0
    return rets


def pair_hit_rate(closes: dict[str, dict], leader: str, follower: str, lag: int) -> dict:
    """이 페어가 역사적으로 몇 번 맞고 몇 번 틀렸는지. 엔진 진입 규칙과 동일한 정의.

    반환 {"wins", "losses", "n", "win_rate"}. 표본이 없으면 n=0, win_rate=None.
    """
    l_bars, f_bars = closes.get(leader) or {}, closes.get(follower) or {}
    if not l_bars or not f_bars:
        return {"wins": 0, "losses": 0, "n": 0, "win_rate": None}

    l_rets, f_rets = _returns(l_bars), _returns(f_bars)
    jump, flat = config.LEADER_JUMP, config.FOLLOWER_FLAT
    wins = losses = 0

    for ts, l_ret in l_rets.items():
        if abs(l_ret) < jump:
            continue
        f_ret = f_rets.get(ts)
        if f_ret is None or abs(f_ret) > flat:
            continue                    # 후행주가 이미 움직였으면 엔진도 진입하지 않는다
        base = f_bars.get(ts)
        after = f_bars.get(minute_key(datetime.strptime(ts, "%Y%m%d%H%M")
                                      + timedelta(minutes=lag)))
        if not base or after is None:
            continue                    # lag 뒤 분봉이 없으면 판정 불가
        moved = after - base
        if moved == 0:
            losses += 1                 # 안 움직였으면 예측 실패로 본다(왕복 마찰 못 넘음)
        elif (moved > 0) == (l_ret > 0):
            wins += 1
        else:
            losses += 1

    n = wins + losses
    return {"wins": wins, "losses": losses, "n": n,
            "win_rate": (wins / n) if n else None}

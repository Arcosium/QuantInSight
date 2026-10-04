"""실거래 원장(trade ledger) — KIS 집계 TR 비의존 자산평가 (사장 지시 2026-06-11).

배경: KIS 통합총자산 TR 3종이 서로 불일치하고(자기모순), USD 결제 과도기·해외평가
결손(외화예수금 미포함, 모의 해외데이터 garbage)으로 자산곡선·수익률 KPI가 환각을
일으켰다 (uid1 누적 -43% 표시 사례). KIS 계정 '집계값' 대신 우리가 직접 체결 원장을
굴려 평가한다 — KIS는 ① 체결 사실(보유 변동) ② 종목 단위 평단/시세만 신뢰한다.

원리:
- 시드(1회): KIS 보유종목(qty/평단) + 예수금(KRW D+2, USD 외화예수금)으로 초기 원장 구성.
- 진화: 이후엔 '우리 체결'만으로 현금·포지션을 갱신한다 (US 수수료 매수·매도 각 0.3%).
- 평가(M2M): 자체 시세(스냅샷 종목시세·us_last_price·일봉CSV 폴백, 실패 시 직전가
  carry-forward) × 5분 크롤 환율로 원화 평가. 입출금·수동거래는 reconcile 이 qty 괴리로
  탐지해 경고하고, /api/ledger/reseed 로 재시드한다.

저장소: data/<uid>/ledger.json. '거래 내역 비우기'와 무관(별도 파일)하다.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
from collections import Counter
from contextlib import contextmanager
from threading import RLock
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("arquant.trade_ledger")

KST = ZoneInfo("Asia/Seoul")
_DATA_DIR = Path(__file__).parent.parent / "data"
_DEFAULT_DATA_DIR = _DATA_DIR
_PENDING_FILL_POLLS = Counter()
_PENDING_FILL_LOCK = RLock()


@contextmanager
def pending_fill_poll(uid, orders):
    """An active fill poll owns reconciliation until it completes or is cancelled."""
    keys = {(int(uid), str(o.get("ticker") or "").strip()) for o in orders}
    with _PENDING_FILL_LOCK:
        _PENDING_FILL_POLLS.update(keys)
    try:
        yield
    finally:
        with _PENDING_FILL_LOCK:
            for key in keys:
                _PENDING_FILL_POLLS[key] -= 1
                if _PENDING_FILL_POLLS[key] <= 0:
                    del _PENDING_FILL_POLLS[key]


def fill_poll_active(uid, ticker):
    with _PENDING_FILL_LOCK:
        return _PENDING_FILL_POLLS[(int(uid), str(ticker).strip())] > 0


def _writes_allowed() -> bool:
    """운영호스트 pytest 가 라이브 원장을 오염시키지 않게 하는 가드
    (전례: test_ceo_directive_routing 이 라이브 ops_history 오염 — 2026-06-09 memory).
    테스트가 _DATA_DIR 를 tmp 로 monkeypatch 하면 쓰기 허용."""
    return not (os.environ.get("PYTEST_CURRENT_TEST") and _DATA_DIR == _DEFAULT_DATA_DIR)

# Inclusive cost assumptions for economic P&L; not an exact broker fee quote.
# Applied prospectively. Existing account history is not rewritten.
US_TRADE_COST_RATE = 0.003
KR_BUY_COST_RATE = 0.0005
KR_SELL_COST_RATE = 0.003

_KR_CODE_RE = re.compile(r"^\d{6}$")
_FILLS_CAP = 300  # 감사용 체결 이력 보존 상한


def _now_str() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def _f(v, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if x == x else default  # NaN 가드
    except (TypeError, ValueError):
        return default


def _ledger_path(uid) -> Path:
    d = _DATA_DIR / str(int(uid))
    d.mkdir(parents=True, exist_ok=True)
    return d / "ledger.json"


def load(uid) -> Optional[dict]:
    p = _ledger_path(uid)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) and isinstance(d.get("positions"), dict) else None
    except Exception as e:
        logger.warning(f"[원장 uid={uid}] 읽기 실패: {e}")
        return None


def save(uid, led: dict) -> None:
    if not _writes_allowed():
        return
    try:
        _ledger_path(uid).write_text(json.dumps(led, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"[원장 uid={uid}] 저장 실패: {e}")


def reset(uid) -> bool:
    """원장 삭제(재시드 유도). /api/ledger/reseed 가 사용."""
    if not _writes_allowed():
        return False
    p = _ledger_path(uid)
    try:
        if p.exists():
            p.unlink()
            return True
    except Exception as e:
        logger.warning(f"[원장 uid={uid}] 삭제 실패: {e}")
    return False


def _is_kr(code: str) -> bool:
    return bool(_KR_CODE_RE.match(str(code or "").strip()))


def us_csv_close(ticker: str) -> float:
    """data/daily_US_<tk>.csv (us_daily_chart 가 적재) 마지막 종가 — 브로커 비의존 US 가격 폴백."""
    tk = str(ticker or "").strip().upper()
    p = _DATA_DIR / f"daily_US_{tk}.csv"
    if not tk or not p.exists():
        return 0.0
    try:
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        for row in reversed(rows):
            c = _f(row.get("close") or row.get("Close"))
            if c > 0:
                return c
    except Exception:
        pass
    return 0.0


_DEDUP_WINDOW_SEC = 600  # 동일 (ticker,side,qty,~price) 체결 중복 판정 윈도우(초)


def _is_duplicate_fill(fills, ticker, side, qty, price, now_str,
                       window_sec: int = _DEDUP_WINDOW_SEC, *, note: str = "") -> bool:
    """최근 window_sec 내 동일 (ticker, side, qty, ~price) 체결이 이미 있으면 중복(True).
    즉시체결(exec_immediate)이 같은 종목의 직전 사이클 미체결 주문 폴링(poll_confirm)에
    cross-cycle 로 재계상되어 원장에 이중 반영되던 버그(uid2 375500: 63주 두 번→보유 126
    전량 차감, 실제 매도 63) 방어. 폴링은 'baseline 대비 보유 감소=내 체결'로 가정하나 다른
    사이클의 매도가 그 감소를 흡수한다. 부분체결 누적은 _poll_increment 가 '증분(다른 qty)'만
    기록하므로 같은 qty 중복만 걸린다. 가격은 0.1%(또는 1원) 이내면 동일 체결로 본다.

    예외(2026-06-19): repair_from_recent_partial_orders 는 잔여 부분체결을 '추정가(fill_price)'로
    기록하는데, 살아있는 백그라운드 폴링이 같은 잔여분을 '실제 체결가'로 또 기록해 같은 qty 인데
    가격만 달라(86300 vs 86800) 게이트를 통과·이중계상됐다(uid2 161890: 매도 과다기록으로 원장이
    KIS 아래로 고착). repair 가 관여된(현재/이전 note='repair_partial_restart') 쌍은 가격 게이트를
    면제한다 — 같은 (ticker,side,qty) 면 추정가/실제가 차이 무시하고 중복으로 본다."""
    try:
        now = datetime.strptime(str(now_str), "%Y-%m-%d %H:%M:%S")
    except Exception:
        return False
    _s = str(getattr(side, "value", side) or "").strip().lower()
    try:
        _q = int(qty or 0)
    except (TypeError, ValueError):
        return False
    _p = _f(price)
    _cur_repair = "repair_partial_restart" in str(note or "")
    for f in reversed(fills or []):
        if str(f.get("ticker")) != str(ticker):
            continue
        if str(f.get("side")).strip().lower() != _s or int(f.get("qty") or 0) != _q:
            continue
        # repair(추정가)↔poll_confirm(실제가) 쌍은 가격 게이트 면제 — 같은 qty 면 중복.
        _repair_pair = _cur_repair or ("repair_partial_restart" in str(f.get("note") or ""))
        if not _repair_pair and abs(_f(f.get("price")) - _p) > max(1.0, _p * 0.001):
            continue
        try:
            t = datetime.strptime(str(f.get("ts")), "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
        if 0 <= (now - t).total_seconds() <= window_sec:
            return True
    return False


def apply_fill(uid, *, ticker: str, side: str, qty, price=None, ccy: str = None,
               avg_cost=None, note: str = "") -> bool:
    """체결 1건을 원장에 반영. 원장이 아직 시드 전이면 조용히 skip(False) —
    시드 스냅샷이 이미 그 체결을 반영한 상태로 찍히므로 이중계상을 피한다.

    가격 폴백: price 없으면 매수=avg_cost, 매도=직전가(last_price)→평단. 그래도 없으면
    skip + degraded 카운트(현금만 틀어지는 반쪽 반영을 하지 않는다)."""
    if not _writes_allowed():
        return False
    led = load(uid)
    if led is None:
        return False
    tk = str(ticker or "").strip()
    # OrderSide(str,Enum) 가드 — str(enum) 은 'OrderSide.SELL' 이 되어 조용히 거부됐다
    # (2026-06-11 라이브: uid2 132030 매도 72주가 원장에 미반영). value 를 우선 사용.
    side = str(getattr(side, "value", side) or "").strip().lower()
    try:
        qty = int(qty or 0)
    except (TypeError, ValueError):
        qty = 0
    if not tk or side not in ("buy", "sell") or qty <= 0:
        return False
    ccy = (str(ccy or "").upper() or ("KRW" if _is_kr(tk) else "USD"))
    positions: Dict[str, dict] = led.setdefault("positions", {})
    pos = positions.get(tk)
    px = _f(price)
    approx = False
    if px <= 0:
        if side == "buy":
            px = _f(avg_cost)
        else:
            px = _f((pos or {}).get("last_price")) or _f((pos or {}).get("avg_cost")) or _f(avg_cost)
        approx = True
    if px <= 0:
        led["degraded_fills"] = int(led.get("degraded_fills") or 0) + 1
        save(uid, led)
        logger.warning(f"[원장 uid={uid}] {tk} {side} x{qty} 가격 미상 — 원장 미반영(degraded)")
        return False

    # 사장 지시 2026-06-16: 멱등 — 즉시체결이 cross-cycle 폴링에 재계상되어 이중 반영되던
    # 버그(uid2 375500) 방어. 현금·포지션 변경 전에 중복을 차단한다.
    if _is_duplicate_fill(led.get("fills") or [], tk, side, qty, px, _now_str(), note=note):
        logger.warning(f"[원장 uid={uid}] {tk} {side} x{qty}@{px:,.0f} 중복 체결 추정 — 멱등 skip (note={note})")
        return False

    fee_rate = US_TRADE_COST_RATE if ccy == "USD" else (KR_BUY_COST_RATE if side == "buy" else KR_SELL_COST_RATE)
    fee = fee_rate * px * qty
    cash_key = "cash_usd" if ccy == "USD" else "cash_krw"
    _sell_realized = None   # 매도 권위 실현손익(native ccy) — fill 에 기록(버그 E)
    if side == "buy":
        led[cash_key] = _f(led.get(cash_key)) - (px * qty + fee)
        if pos:
            old_q = int(pos.get("qty") or 0)
            old_avg = _f(pos.get("avg_cost"))
            new_q = old_q + qty
            pos["avg_cost"] = ((old_avg * old_q + px * qty) / new_q) if new_q > 0 else px
            pos["qty"] = new_q
        else:
            pos = {"qty": qty, "avg_cost": px, "ccy": ccy}
            positions[tk] = pos
        pos["last_price"] = px
        pos["entry_fees"] = _f(pos.get("entry_fees")) + fee
        if approx:
            pos["approx_basis"] = True
    else:  # sell
        led[cash_key] = _f(led.get(cash_key)) + (px * qty - fee)
        if pos:
            # 권위 실현손익(버그 E, 2026-06-18): 평단(avg_cost) 기준·비용반영 = (체결가−평단)×수량 − 수수료.
            # 원장 fills 는 멱등이라 trade_log 부분체결 재방출 이중계상을 안 탄다 → realized_stats 권위 소스.
            _basis = _f(pos.get("avg_cost")) or _f(avg_cost)
            if _basis > 0:
                entry_fee = _f(pos.get("entry_fees"))*min(1., qty/max(1, int(pos.get("qty") or 0)))
                _sell_realized = (px - _basis) * qty - fee - entry_fee
                pos["entry_fees"] = max(0., _f(pos.get("entry_fees"))-entry_fee)
            pos["qty"] = int(pos.get("qty") or 0) - qty
            pos["last_price"] = px
            if pos["qty"] <= 0:
                positions.pop(tk, None)
        # 포지션 미등재 매도(시드 전 매수분 등) — 현금만 반영하면 자산이 부풀므로 경고만.
        else:
            led[cash_key] = _f(led.get(cash_key)) - (px * qty - fee)  # 롤백
            led["degraded_fills"] = int(led.get("degraded_fills") or 0) + 1
            save(uid, led)
            logger.warning(f"[원장 uid={uid}] {tk} 매도 체결인데 원장에 포지션 없음 — 미반영(reconcile 대상)")
            return False
    fills: List[dict] = led.setdefault("fills", [])
    _fill = {"ts": _now_str(), "ticker": tk, "side": side, "qty": qty,
             "price": px, "ccy": ccy, "fee": round(fee, 4),
             "approx_price": approx, "note": str(note or "")[:120],
             "cost_basis": "estimated_inclusive_v2"}
    if _sell_realized is not None:
        _fill["realized"] = round(_sell_realized, 4)
    fills.append(_fill)
    led["fills"] = fills[-_FILLS_CAP:]
    save(uid, led)
    return True


def recent_loss_streak(uid) -> int:
    """가장 최근 매도부터 연속된 손실 **거래** 횟수(회로차단용, 사장 지시 2026-07-21).
    승리 거래(또는 매도 없음)를 만나면 중단. 원장 fills 는 시간순 append 라 역순 순회.

    ⚠️ 부분체결 보정 (사장 지시 2026-07-22): 종전엔 **체결(fill) 단위**로 셌다. 매도 주문 하나가
    호가에 잘려 여러 조각으로 체결되면 조각마다 손실이 찍혀, 경제적으로는 한 번의 매도가
    '연속 손절 N회'가 됐다 — 실측(uid2): 148070 매도 1건이 13조각으로 나뉘어 streak 18 을
    만들었고(합계 -72,098원 = 1억 계좌의 -0.07%) 임계값 3 을 넘겨 **신규 매수가 전면 차단**됐다.
    이 가드의 취지는 '전략이 연속으로 깨지고 있다'이지 '주문이 잘게 잘렸다'가 아니다.

    그래서 **연속된 같은 종목 매도를 한 거래로 묶어** 실현손익을 합산한 뒤 센다.
    같은 종목을 되산 뒤 다시 판 것은 별개 거래이므로, 사이에 그 종목 **매수**가 있으면 묶음을 끊는다.
    """
    led = load(uid)
    if led is None:
        return 0
    streak = 0
    cur_ticker = None          # 현재 묶고 있는 매도 종목
    cur_realized = 0.0
    cur_open = False           # 묶음 진행 중 여부

    def _close(realized: float) -> bool:
        """묶음 1건 확정 → 손실이면 streak+1(계속), 아니면 중단 신호(False)."""
        nonlocal streak
        if realized < 0:
            streak += 1
            return True
        return False

    for f in reversed(led.get("fills") or []):
        side = str(f.get("side") or "").lower()
        tkr = str(f.get("ticker") or "")
        if side == "buy":
            # 같은 종목 재매수 = 그 앞쪽 매도는 별개 거래 → 진행 중인 묶음을 여기서 확정한다.
            if cur_open and tkr == cur_ticker:
                if not _close(cur_realized):
                    return streak
                cur_open, cur_ticker, cur_realized = False, None, 0.0
            continue
        if side != "sell" or f.get("realized") is None:
            continue
        if cur_open and tkr == cur_ticker:
            cur_realized += _f(f.get("realized"))       # 같은 매도의 다른 조각 — 합산
            continue
        if cur_open:                                    # 종목이 바뀜 = 앞 묶음 확정
            if not _close(cur_realized):
                return streak
        cur_ticker, cur_realized, cur_open = tkr, _f(f.get("realized")), True
    if cur_open:
        _close(cur_realized)
    return streak


def realized_stats(uid, fx: float = 0.0) -> dict:
    """원장 fills 의 권위 실현손익(매도 'realized' 필드)을 KR/US 분리 집계 — 운용지원실장 피드백용
    (버그 E·F3, 2026-06-18). trade_log 부분체결 재방출 이중계상 비의존(멱등 원장). USD 는 fx 로 원화환산.
    반환: 승률·평균이익/손실(원)·기대값(평균/거래, 원)·비용드래그(US 수수료 원화)·KR/US 세부."""
    led = load(uid)
    fxr = _f(fx) or 0.0
    out = {"sell_count": 0, "win_count": 0, "loss_count": 0, "win_rate": 0.0,
           "total_realized_krw": 0.0, "avg_win_krw": 0.0, "avg_loss_krw": 0.0,
           "expectancy_krw": 0.0, "cost_drag_krw": 0.0,
           "kr": {"sell_count": 0, "realized": 0.0}, "us": {"sell_count": 0, "realized_usd": 0.0}}
    if led is None:
        return out
    wins = losses = 0
    win_sum = loss_sum = 0.0
    for f in (led.get("fills") or []):
        is_usd = str(f.get("ccy") or "").upper() == "USD"
        # 비용드래그: US 매수·매도 양 leg 수수료(원화환산) 누적
        if is_usd:
            out["cost_drag_krw"] += _f(f.get("fee")) * (fxr if fxr > 0 else 1.0)
        if str(f.get("side") or "").lower() != "sell" or f.get("realized") is None:
            continue
        r_native = _f(f.get("realized"))
        r_krw = r_native * (fxr if fxr > 0 else 1.0) if is_usd else r_native
        out["sell_count"] += 1
        out["total_realized_krw"] += r_krw
        if is_usd:
            out["us"]["sell_count"] += 1; out["us"]["realized_usd"] += r_native
        else:
            out["kr"]["sell_count"] += 1; out["kr"]["realized"] += r_native
        if r_krw >= 0:
            wins += 1; win_sum += r_krw
        else:
            losses += 1; loss_sum += r_krw
    out["win_count"] = wins; out["loss_count"] = losses
    if out["sell_count"] > 0:
        out["win_rate"] = wins / out["sell_count"] * 100.0
        out["expectancy_krw"] = out["total_realized_krw"] / out["sell_count"]
    out["avg_win_krw"] = (win_sum / wins) if wins else 0.0
    out["avg_loss_krw"] = (loss_sum / losses) if losses else 0.0
    return out


def mark_to_market(uid, *, price_lookup: Dict[str, float], fx: float) -> Optional[dict]:
    """원장 평가. price_lookup 에 있는 종목은 last_price 갱신(영속), 없는 종목은
    직전가 carry-forward — 시세 결손으로 곡선이 튀지 않는다."""
    led = load(uid)
    if led is None:
        return None
    fx = _f(fx)
    total = _f(led.get("cash_krw")) + (_f(led.get("cash_usd")) * fx if fx > 0 else 0.0)
    stale: List[str] = []
    changed = False
    for tk, pos in (led.get("positions") or {}).items():
        live = _f((price_lookup or {}).get(tk))
        if live > 0 and live != _f(pos.get("last_price")):
            pos["last_price"] = live
            changed = True
        px = _f(pos.get("last_price")) or _f(pos.get("avg_cost"))
        if live <= 0:
            stale.append(tk)
        qty = int(pos.get("qty") or 0)
        if str(pos.get("ccy") or "").upper() == "USD":
            total += qty * px * (fx if fx > 0 else 0.0)
        else:
            total += qty * px
    if changed:
        save(uid, led)
    return {"value_krw": total, "stale": stale, "seeded_at": led.get("seeded_at")}


def value_from_snap(uid, snap: dict, fx: float) -> Optional[float]:
    """portfolio_holdings 스냅샷의 종목 시세만으로 즉시 평가 (서버 /api/balance 용, 시드 전이면 None).
    모의계정 US 시세는 garbage 일 수 있으나 last_price 는 폴러(ensure_value)가 실시세로
    유지하므로, 여기선 KR 시세만 반영하고 US 는 carry-forward 에 맡긴다."""
    led = load(uid)
    if led is None:
        return None
    lookup: Dict[str, float] = {}
    for h in (snap or {}).get("holdings") or []:
        code = str(h.get("code") or "").strip()
        cp = _f(h.get("cur_price"))
        if code and cp > 0 and _is_kr(code):
            lookup[code] = cp
    mtm = mark_to_market(uid, price_lookup=lookup, fx=fx)
    return mtm["value_krw"] if mtm else None


def reconcile(uid, holdings: List[dict]) -> List[str]:
    """KIS 보유 qty vs 원장 qty 괴리 목록 (수동거래/입출금/누락체결 탐지).
    KIS 보유 스냅샷이 일시 결손(빈 목록)일 수 있어, 빈 입력이면 비교하지 않는다.

    US 해외보유 글리치 가드(2026-06-15): KIS 해외보유 API 는 결제 과도기·장중에 US 종목을 통째
    빈값으로 주는 일시 글리치가 잦다(FCX 사례: 체결내역상 매수만·실보유 5주인데 스냅샷 0). 원장엔
    US 포지션이 있는데 KIS 스냅샷에 US 종목이 0이면 글리치로 보고 US 괴리는 보류한다(KR 은 신뢰).
    (단일 US 포지션의 진짜 매도도 같은 시그니처라 한 사이클 보류되지만, overseas_fills 가 권위 backstop.)"""
    led = load(uid)
    if led is None or not holdings:
        return []
    kis = {}
    for h in holdings:
        code = str(h.get("code") or "").strip()
        if code:
            kis[code] = kis.get(code, 0) + int(_f(h.get("qty")))
    led_pos = led.get("positions") or {}
    us_glitch = (any(not _is_kr(c) for c in led_pos)        # 원장에 US 포지션 있음
                 and not any(not _is_kr(c) for c in kis))   # KIS 스냅샷엔 US 0종목 → 글리치 의심
    diffs = []
    for tk in set(kis) | set(led_pos):
        if us_glitch and not _is_kr(tk):
            continue   # US 보유 일시결손 의심 — US 괴리 보고 보류
        kq = kis.get(tk, 0)
        lq = int((led_pos.get(tk) or {}).get("qty") or 0)
        if kq != lq:
            diffs.append(f"{tk}: KIS {kq}주 vs 원장 {lq}주")
    return diffs


def prune_phantoms(uid, holdings: List[dict], *, min_confirmations: int = 3) -> dict:
    """KIS 가 권위적으로 원장보다 적게 보유한 KR 포지션이 min_confirmations 회 '연속' 확인되면
    원장을 KIS 기준으로 하향 정정(허수 제거)한다. 정정으로 빠진 평가액(KRW, last_price 기준)을
    value_krw_removed 로 반환 — 호출측이 자산곡선에 'reconcile_adj'(매매손실 아닌 장부정정)로
    기록하게 한다 (2026-06-17: 047810 6일 허수 → 리시드 시 가짜 -31만원 곡선단차 재발 방지).

    안전장치(라이브 금융 원장 자동변경이라 보수적):
    - KR 만. US 는 결제 과도기 KIS 해외보유가 일시 0 빈번(글리치) → 자동정정 제외(reconcile 가드와 동일 철학).
    - 하향만. KIS > 원장(누락 매수)은 손대지 않는다 — repair_from_recent_partial_orders 영역.
    - 연속 확인(streak). 순간 잔고 글리치(빈보유·결제 과도기)는 1~2틱이라 임계 미달로 방어.
    - 빈 holdings(잔고 일시결손)면 아무것도 안 한다.
    스트릭은 ledger['_recon_streak'] 에 종목별 누적/리셋(재시작 내성)."""
    led = load(uid)
    if led is None or not holdings:
        return {"pruned": [], "value_krw_removed": 0.0}
    kis: Dict[str, int] = {}
    for h in holdings:
        code = str(h.get("code") or "").strip()
        if code:
            kis[code] = kis.get(code, 0) + int(_f(h.get("qty")))
    positions = led.get("positions") or {}
    streak: Dict[str, int] = led.setdefault("_recon_streak", {})
    threshold = max(1, int(min_confirmations))
    pruned: List[str] = []
    removed_krw = 0.0
    for code in list(positions.keys()):
        if fill_poll_active(uid, code):
            streak.pop(code, None)
            continue
        if not _is_kr(code):                 # KR 전용 (US 는 글리치 빈번 → 제외)
            streak.pop(code, None)
            continue
        lq = int((positions.get(code) or {}).get("qty") or 0)
        kq = int(kis.get(code, 0))
        if lq <= kq:                         # 일치 또는 KIS 가 더 많음(누락매수) → 정정 대상 아님 · 리셋
            streak.pop(code, None)
            continue
        streak[code] = int(streak.get(code, 0)) + 1   # 허수(원장>KIS) 연속 확인
        if streak[code] < threshold:
            continue
        pos = positions[code]                # 임계 도달 → KIS 기준 하향 정정
        last = _f(pos.get("last_price")) or _f(pos.get("avg_cost"))
        removed_qty = lq - kq
        removed_krw += removed_qty * last
        if kq <= 0:
            positions.pop(code, None)
        else:
            pos["qty"] = kq
        streak.pop(code, None)
        pruned.append(f"{code}: 원장 {lq}→KIS {kq}주 (허수 {removed_qty}주·{removed_qty*last:,.0f}원 정정)")
        logger.warning(f"[원장정정 uid={uid}] {code} 허수 {removed_qty}주 자동제거 "
                       f"(KIS {kq} vs 원장 {lq}, {threshold}회 연속 확인)")
    if pruned:
        led["reconcile_adj_cum_krw"] = _f(led.get("reconcile_adj_cum_krw")) - removed_krw  # 감사용 누적
        save(uid, led)
    elif streak:                             # 스트릭 갱신만 있어도 영속(연속 카운트 유지)
        save(uid, led)
    return {"pruned": pruned, "value_krw_removed": removed_krw}


def adopt_orphans(uid, holdings: List[dict], *, min_confirmations: int = 3) -> dict:
    """prune_phantoms 의 대칭(상향). KIS 가 권위적으로 원장보다 '많이' 보유한 KR 포지션이
    min_confirmations 회 '연속' 확인되면 원장을 KIS 기준으로 상향 채택한다(누락 포지션 복원).
    매도 이중계상 등으로 원장이 KIS 아래로 떨어져 고착되는 것을 막아 원장 qty 를 항상 KIS 로
    수렴시킨다(2026-06-19 defense-in-depth: 161890 'KIS 65 vs 원장 0' 고착 자동 해소). 채택분
    평가액(KRW, KIS 평단 기준)을 value_krw_added 로 반환 → 호출측이 자산곡선에 reconcile_adj(+,
    매매이익 아닌 장부정정)로 기록한다.

    안전장치(prune_phantoms 와 동일 철학 — 라이브 재무원장 자동변경이라 보수적):
    - KR 만. US 는 결제 과도기 KIS 해외보유 글리치 빈번 → 자동채택 제외.
    - 상향만. KIS < 원장(허수)은 손대지 않는다 — prune_phantoms 영역.
    - 연속 확인(streak). 순간 잔고 글리치(빈보유·결제 과도기)는 1~2틱이라 임계 미달로 방어.
      repair_from_recent_partial_orders(최근 주문 기반 보정)가 주문으로 설명되는 부분체결 갭을
      1~2 사이클 내 해소하므로, 채택은 '주문으로 설명 안 되는 지속 갭'만 잡는다.
    - 빈 holdings(잔고 일시결손)면 아무것도 안 한다.
    스트릭은 ledger['_adopt_streak'] 에 종목별 누적/리셋(재시작 내성)."""
    led = load(uid)
    if led is None or not holdings:
        return {"adopted": [], "value_krw_added": 0.0}
    kis: Dict[str, int] = {}
    kis_avg: Dict[str, float] = {}
    for h in holdings:
        code = str(h.get("code") or "").strip()
        if code:
            kis[code] = kis.get(code, 0) + int(_f(h.get("qty")))
            if _f(h.get("avg_price")) > 0:
                kis_avg[code] = _f(h.get("avg_price"))
    positions = led.setdefault("positions", {})
    streak: Dict[str, int] = led.setdefault("_adopt_streak", {})
    threshold = max(1, int(min_confirmations))
    adopted: List[str] = []
    added_krw = 0.0
    for code in list(kis.keys()):
        if fill_poll_active(uid, code):
            streak.pop(code, None)
            continue
        if not _is_kr(code):                 # KR 전용 (US 는 글리치 빈번 → 제외)
            streak.pop(code, None)
            continue
        kq = int(kis.get(code, 0))
        lq = int((positions.get(code) or {}).get("qty") or 0)
        if kq <= lq:                         # 일치 또는 원장이 더 많음(허수) → 채택 대상 아님 · 리셋
            streak.pop(code, None)
            continue
        streak[code] = int(streak.get(code, 0)) + 1   # 누락(KIS>원장) 연속 확인
        if streak[code] < threshold:
            continue
        pos = positions.get(code)            # 임계 도달 → KIS 기준 상향 채택
        add_qty = kq - lq
        px = _f(kis_avg.get(code)) or _f((pos or {}).get("last_price")) or _f((pos or {}).get("avg_cost"))
        approx = not (_f(kis_avg.get(code)) > 0)
        added_krw += add_qty * px
        if pos:
            old_q, old_avg = lq, _f(pos.get("avg_cost"))
            pos["avg_cost"] = ((old_avg * old_q + px * add_qty) / kq) if (kq > 0 and px > 0) else (old_avg or px)
            pos["qty"] = kq
            pos.setdefault("last_price", px)
            if approx:
                pos["approx_basis"] = True
        else:
            positions[code] = {"qty": kq, "avg_cost": px, "ccy": "KRW", "last_price": px}
            if approx:
                positions[code]["approx_basis"] = True
        streak.pop(code, None)
        adopted.append(f"{code}: 원장 {lq}→KIS {kq}주 (누락 {add_qty}주·{add_qty*px:,.0f}원 채택)")
        logger.warning(f"[원장채택 uid={uid}] {code} 누락 {add_qty}주 자동채택 "
                       f"(KIS {kq} vs 원장 {lq}, {threshold}회 연속 확인)")
    if adopted:
        led["reconcile_adj_cum_krw"] = _f(led.get("reconcile_adj_cum_krw")) + added_krw  # 감사용 누적(+)
        save(uid, led)
    elif streak:                             # 스트릭 갱신만 있어도 영속(연속 카운트 유지)
        save(uid, led)
    return {"adopted": adopted, "value_krw_added": added_krw}


def _repair_streak_gate(uid, tk: str, hit: bool, threshold: int) -> bool:
    """누락매수 상향보정 확인 스트릭 게이트(버그 D, 2026-06-18). hit=True(KIS>원장 괴리 관측)면
    tk 스트릭 +1, False 면 0 리셋. 임계 도달 시 True(보정 허용)·스트릭 리셋. 일시 글리치-高(1~2틱)는
    임계 미달로 방어. 스트릭은 ledger['_repair_streak'] 에 영속(재시작 내성)."""
    led = load(uid)
    if led is None:
        return False
    streak = led.setdefault("_repair_streak", {})
    if not hit:
        if streak.pop(tk, None) is not None:
            save(uid, led)
        return False
    streak[tk] = int(streak.get(tk, 0)) + 1
    if streak[tk] >= max(1, int(threshold)):
        streak.pop(tk, None)
        save(uid, led)
        return True
    save(uid, led)
    return False


def repair_from_recent_partial_orders(uid, holdings: List[dict], *, cycles_limit: int = 30) -> List[str]:
    """Best-effort 원장 보정 for restart-lost partial fill polling.

    즉시 확인에서 부분체결로 기록한 주문은 잔여분을 background polling 이 추적한다. 서버 재시작이
    그 polling task 를 날리면 KIS 잔고는 나중에 주문수량까지 늘었는데 원장은 첫 체결분만 남는다.
    최근 cycle 의 orders_executed 에 남은 부분체결 주문과 현재 KIS 보유수량이 정확히 맞을 때만
    누락 증분을 체결로 반영한다. 수동거래/입출금으로 보이는 큰 괴리는 reconcile 경고로 남긴다.
    """
    if not holdings:
        return []
    led = load(uid)
    if led is None:
        return []
    try:
        from config import LEDGER_REPAIR_CONFIRMATIONS as _repair_threshold
    except Exception:
        _repair_threshold = 2
    kis = {}
    kis_avg = {}
    for h in holdings or []:
        code = str(h.get("code") or "").strip()
        if not code:
            continue
        kis[code] = kis.get(code, 0) + int(_f(h.get("qty")))
        if _f(h.get("avg_price")) > 0:
            kis_avg[code] = _f(h.get("avg_price"))
    led_pos = led.get("positions") or {}
    try:
        from infra import cycle_store
        cycles = cycle_store.list_cycles(limit=cycles_limit, uid=int(uid))
    except Exception:
        cycles = []
    if not cycles:
        return []

    repaired: List[str] = []
    for c in cycles:
        try:
            rows = json.loads(c.get("orders_executed") or "[]")
        except Exception:
            rows = []
        for e in rows or []:
            tk = str(e.get("ticker") or "").strip()
            side = str(e.get("side") or "buy").strip().lower()
            if not tk or not _is_kr(tk) or side not in ("buy", "sell") or fill_poll_active(uid, tk):
                continue
            try:
                reported_qty = int(e.get("qty") or 0)
                order_qty = int(e.get("order_qty") or reported_qty)
            except (TypeError, ValueError):
                continue
            if not e.get("accepted") or order_qty <= 0:
                continue
            recorded_qty = reported_qty if e.get("filled") else 0
            if order_qty <= recorded_qty:
                continue
            lq = int((load(uid) or {}).get("positions", {}).get(tk, {}).get("qty") or 0)
            kq = int(kis.get(tk, 0) or 0)
            max_missing = order_qty - recorded_qty
            if side == "buy":
                missing = kq - lq
                if missing <= 0 or missing > max_missing:
                    _repair_streak_gate(uid, tk, False, _repair_threshold)   # 괴리 없음 → 스트릭 리셋
                    continue
                # KIS 글리치-高 방어: KIS>원장 상향괴리가 연속 확인돼야 보정(일시 글리치 baked 방지).
                if not _repair_streak_gate(uid, tk, True, _repair_threshold):
                    continue
                led_now = load(uid) or {}
                pos = (led_now.get("positions") or {}).get(tk) or {}
                lp = _f(pos.get("avg_cost"))
                kp = _f(kis_avg.get(tk)) or _f(e.get("fill_price")) or lp
                px = ((kp * kq - lp * lq) / missing) if missing > 0 and kp > 0 and lq >= 0 else kp
                if apply_fill(uid, ticker=tk, side="buy", qty=missing, price=px,
                              ccy="KRW", avg_cost=kp, note="repair_partial_restart"):
                    repaired.append(f"{tk}: 누락 매수 {missing}주 원장 보정")
            else:
                missing = lq - kq
                if missing <= 0 or missing > max_missing:
                    continue
                px = _f(e.get("fill_price")) or _f((led_pos.get(tk) or {}).get("last_price")) or _f(kis_avg.get(tk))
                if apply_fill(uid, ticker=tk, side="sell", qty=missing, price=px,
                              ccy="KRW", avg_cost=_f(e.get("avg_cost")), note="repair_partial_restart"):
                    repaired.append(f"{tk}: 누락 매도 {missing}주 원장 보정")
    return repaired


def _add_us_bdays(d: date, n: int) -> date:
    """주말만 건너뛰는 영업일 가산 (미 휴장일 미반영 — 오차는 reconcile/재시드가 흡수)."""
    cur = d
    while n > 0:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            n -= 1
    return cur


async def pending_usd_from_fills(broker, today_us: Optional[date] = None) -> float:
    """미결제(T+2 영업일) USD 매도대금 — 결제 과도기엔 KIS 통합총자산/외화예수금 TR 이 전부
    0 을 줘(2026-06-11 라이브 확인: uid1 매도대금 ~$1,682 가 6504·6010 에 미반영, 6548 만 포함)
    체결내역(TTTS3035R)이 유일한 권위 소스다.

    일별 net = 매도대금 − 매수대금. **양수 net 만** 미결제 USD 로 더한다 — 음수(순매수) 일은
    통합증거금으로 KRW(D+2 예수금)에서 차감되므로(라이브 확인: −$369.67×환율 = KRW −568,522)
    USD 쪽엔 반영하지 않는다."""
    today_us = today_us or datetime.now(ZoneInfo("America/New_York")).date()
    start = today_us - timedelta(days=8)
    try:
        fills = await broker.overseas_fills(start.strftime("%Y%m%d"), today_us.strftime("%Y%m%d"))
    except Exception as e:
        logger.warning(f"[원장] 해외체결내역 조회 실패(미결제 USD 0 으로 진행): {e}")
        return 0.0
    by_date: Dict[str, float] = {}
    for f in fills or []:
        if str(f.get("ccy") or "USD").upper() != "USD":
            continue
        amt = _f(f.get("amount")) or (_f(f.get("price")) * _f(f.get("qty")))
        if amt <= 0:
            continue
        sign = 1.0 if f.get("side") == "sell" else -1.0
        by_date[f["date"]] = by_date.get(f["date"], 0.0) + sign * amt
    pend = 0.0
    for ymd, net in by_date.items():
        try:
            td = datetime.strptime(ymd, "%Y%m%d").date()
        except ValueError:
            continue
        if _add_us_bdays(td, 2) > today_us and net > 0:
            pend += net
    return pend


async def seed(uid, broker, snap: dict) -> Optional[dict]:
    """KIS 스냅샷으로 원장 1회 시드. bp.ok 아니면 None (글리치 시점 시드 방지).
    - KR: 평단·현재가 신뢰. US 실전: 평단 신뢰 + 외화예수금(CTRP6504R) → cash_usd.
    - US 모의: 평단·시세 garbage → us_last_price/일봉CSV 실시세로 근사(approx), cash_usd=0."""
    if not _writes_allowed():
        return None
    bp = (snap or {}).get("buying_power") or {}
    if not bp.get("ok"):
        return None
    is_mock = bool(getattr(broker, "is_mock", False))
    positions: Dict[str, dict] = {}
    for h in (snap or {}).get("holdings") or []:
        code = str(h.get("code") or "").strip()
        qty = int(_f(h.get("qty")))
        if not code or qty <= 0:
            continue
        ccy = "USD" if (str(h.get("ccy") or "").upper() == "USD" or not _is_kr(code)) else "KRW"
        avg = _f(h.get("avg_price") or h.get("avg_cost"))
        cur = _f(h.get("cur_price"))
        approx = False
        if ccy == "USD" and is_mock:
            live = 0.0
            try:
                live = _f(await broker.us_last_price(code))
            except Exception:
                live = 0.0
            if live <= 0:
                live = us_csv_close(code)
            if live > 0:
                avg = cur = live
                approx = True
        if avg <= 0:
            avg = cur
            approx = True
        if avg <= 0 and cur <= 0:
            # 가격 완전 미상 — 0원 평가로 자산을 깎느니 제외하고 reconcile 경고에 맡긴다.
            logger.warning(f"[원장시드 uid={uid}] {code} 가격 미상 — 시드 제외")
            continue
        pos = {"qty": qty, "avg_cost": avg, "ccy": ccy, "last_price": cur or avg}
        if approx:
            pos["approx_basis"] = True
        positions[code] = pos
    cash_usd = 0.0
    pending = 0.0
    if not is_mock:
        try:
            pk = await broker._overseas_present_krw()
            exrt = _f(pk.get("exrt"))
            if pk.get("ok") and exrt > 500:
                cash_usd = max(0.0, _f(pk.get("deposit_krw")) / exrt)
        except Exception as e:
            logger.warning(f"[원장시드 uid={uid}] 외화예수금 조회 실패(0으로 시드): {e}")
        # 사장 지시 2026-06-11(라이브 디버깅): 미결제(T+2) USD 매도대금은 어떤 예수금 TR 에도
        # 안 잡힌다 — 체결내역 기반으로 가산해야 자산 과소평가(-2.5M 사례)가 없다.
        pending = await pending_usd_from_fills(broker)
        cash_usd += pending
    led = {
        "version": 1,
        "seeded_at": _now_str(),
        "seed_source": "kis_mock" if is_mock else "kis_live",
        "cash_krw": _f(bp.get("cash")),
        "cash_usd": cash_usd,
        "positions": positions,
        "fills": [],
        "degraded_fills": 0,
    }
    if pending:
        led["seed_pending_usd"] = round(pending, 4)   # 정보성 — 시드 당시 미결제 매도대금
    save(uid, led)
    logger.info(f"[원장시드 uid={uid}] 완료 — KRW {led['cash_krw']:,.0f} / USD {cash_usd:,.2f} "
                f"(미결제 {pending:,.2f}) / 종목 {len(positions)}개 ({led['seed_source']})")
    return led


async def ensure_value(uid, broker, snap: dict, fx: float) -> Optional[float]:
    """폴러용 원스톱: (필요시) 시드 → US 실시세 보강 → M2M 평가값(KRW) 반환.
    실패 시 None — 호출부는 ledger 포인트 없이 기존 KIS 곡선만 기록한다."""
    led = load(uid)
    if led is None:
        led = await seed(uid, broker, snap)
        if led is None:
            return None
    is_mock = bool(getattr(broker, "is_mock", False))
    lookup: Dict[str, float] = {}
    for h in (snap or {}).get("holdings") or []:
        code = str(h.get("code") or "").strip()
        cp = _f(h.get("cur_price"))
        if not code or cp <= 0:
            continue
        # 모의 해외 시세는 garbage(사장 확인 2026-06-10) — KR만 채택. 실전은 둘 다 채택.
        if _is_kr(code) or not is_mock:
            lookup[code] = cp
    for tk, pos in (led.get("positions") or {}).items():
        if str(pos.get("ccy") or "").upper() != "USD" or tk in lookup:
            continue
        live = 0.0
        try:
            live = _f(await broker.us_last_price(tk))
        except Exception:
            live = 0.0
        if live <= 0:
            live = us_csv_close(tk)
        if live > 0:
            lookup[tk] = live
    mtm = mark_to_market(uid, price_lookup=lookup, fx=fx)
    return mtm["value_krw"] if mtm else None

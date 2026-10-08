from __future__ import annotations

import json
import os
import threading
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

try:  # QuantInSight 안에서 돌 때는 본가 auth_store(argon2 + Fernet)
    from infra import auth_store
except ImportError:  # ArcTrade 독립 실행: 최소 대체 구현
    from . import auth_compat as auth_store

from .contest_rules import normalize_meta, validate_order

_ROOT = Path(__file__).resolve().parents[1]
# AUTOFOLIO_STATE_PATH 로 상태 파일을 분리할 수 있다(호스트 프로세스별 독립 장부).
_PRIVATE_ROOT = Path(os.environ.get("AUTOFOLIO_PRIVATE_DATA_DIR") or
                     (Path.home() / "vault" / "QuantInSight" / "Auto_folio"))
_STORE_PATH = Path(os.environ.get("AUTOFOLIO_STATE_PATH") or (_PRIVATE_ROOT / "contest_state.json"))
_DATA_DIR = _STORE_PATH.parent
_LOCK = threading.RLock()
DEFAULT_INITIAL_CASH = 1_000_000_000.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> dict[str, Any]:
    if not _STORE_PATH.exists():
        return {"accounts": {}, "securities": {}}
    try:
        data = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"accounts": {}, "securities": {}}
        data.setdefault("accounts", {})
        data.setdefault("securities", {})
        return data
    except Exception:
        return {"accounts": {}, "securities": {}}


def _save(data: dict[str, Any]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _STORE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_STORE_PATH)
    try:
        _STORE_PATH.chmod(0o600)
    except OSError:
        pass


def _public_account(account: dict[str, Any] | None) -> dict[str, Any] | None:
    if not account:
        return None
    out = {k: v for k, v in account.items() if k not in {"password_hash", "password_enc", "site_password_enc"}}
    out["has_site_credentials"] = bool(account.get("site_password_enc"))
    out["portfolio"] = portfolio_from_account(account)
    return out


def register(uid: int, contest_id: str = "", password: str | None = None, *,
             initial_cash: float = DEFAULT_INITIAL_CASH, reset: bool = True) -> dict[str, Any]:
    # password 없이 부르면 "순수 페이퍼" 계정: 대회 사이트 자격증명 없이 장부만 만든다.
    contest_id = (contest_id or "").strip()
    if password:
        if not contest_id:
            raise ValueError("타임폴리오 모의투자 아이디를 입력하세요.")
        if auth_store.password_policy_error(password):
            raise ValueError(auth_store.password_policy_error(password))
    contest_id = contest_id or "paper"
    initial_cash = float(initial_cash or DEFAULT_INITIAL_CASH)
    if initial_cash <= 0:
        raise ValueError("초기 운용금액은 0보다 커야 합니다.")
    with _LOCK:
        data = _load()
        existing = data.get("accounts", {}).get(str(int(uid)))
        if existing and not reset:
            existing["contest_id"] = contest_id
            if password:
                existing["password_hash"] = auth_store.hash_password(password)
                existing["site_password_enc"] = auth_store.encrypt(password)
            existing["updated_at"] = _now()
            _save(data)
            return _public_account(existing) or {}
        account = {
            "uid": int(uid),
            "contest_id": contest_id,
            "password_hash": auth_store.hash_password(password) if password else "",
            "site_password_enc": auth_store.encrypt(password) if password else "",
            "initial_cash": initial_cash,
            "cash": initial_cash,
            "positions": {},
            "trades": [],
            "created_at": _now(),
            "updated_at": _now(),
        }
        data["accounts"][str(int(uid))] = account
        _save(data)
        return _public_account(account) or {}


def set_auto_cycle(uid: int, enabled: bool) -> dict[str, Any]:
    """서버 내 자동 네이버 사이클 opt-in 플래그 (사장 지시 2026-07-03: 원클릭 모의투자)."""
    with _LOCK:
        data = _load()
        account = _account_raw(data, uid)
        account["auto_cycle"] = bool(enabled)
        account["updated_at"] = _now()
        _save(data)
        return _public_account(account) or {}


def list_auto_cycle_uids() -> list[int]:
    with _LOCK:
        accounts = _load().get("accounts", {})
    out = []
    for key, acc in accounts.items():
        if isinstance(acc, dict) and acc.get("auto_cycle"):
            try:
                out.append(int(key))
            except (TypeError, ValueError):
                continue
    return sorted(out)


def get_site_credentials(uid: int) -> dict[str, str]:
    with _LOCK:
        account = _load().get("accounts", {}).get(str(int(uid)))
    if not isinstance(account, dict):
        raise ValueError("타임폴리오 모의투자 계정이 없습니다. 먼저 가입하세요.")
    password_enc = account.get("site_password_enc") or account.get("password_enc") or ""
    password = auth_store.decrypt(password_enc) if password_enc else ""
    if not password:
        raise ValueError("타임폴리오 사이트 주문용 비밀번호가 저장되어 있지 않습니다. 프로필에서 대회 계정을 한 번 더 저장하세요.")
    return {"username": str(account.get("contest_id") or ""), "password": password}


def get_account(uid: int) -> dict[str, Any] | None:
    with _LOCK:
        return _public_account(_load().get("accounts", {}).get(str(int(uid))))


def get_account_raw(uid: int) -> dict[str, Any] | None:
    """트레이드 내역 포함 원본 계정(스냅샷 계산용). 비밀번호 필드가 포함되니 외부 노출 금지."""
    with _LOCK:
        account = _load().get("accounts", {}).get(str(int(uid)))
    return dict(account) if isinstance(account, dict) else None


def reset_portfolio(uid: int, *, cash: float = DEFAULT_INITIAL_CASH, clear_trades: bool = True) -> dict[str, Any]:
    cash = float(cash or DEFAULT_INITIAL_CASH)
    if cash <= 0:
        raise ValueError("초기 운용금액은 0보다 커야 합니다.")
    with _LOCK:
        data = _load()
        account = _account_raw(data, uid)
        account["initial_cash"] = cash
        account["cash"] = cash
        account["positions"] = {}
        if clear_trades:
            account["trades"] = []
        account["updated_at"] = _now()
        _save(data)
        return _public_account(account) or {}


_EQUITY_MIN_GAP_SEC = 60.0        # NAV 스냅샷 최소 간격 — 사이클마다 찍히니 과밀 방지
_EQUITY_KEEP_DAYS = 3             # 최근 N일만 원시 해상도, 그 이전은 일별 종가만 남긴다
_EQUITY_MAX_POINTS = 4000
_FILL_INFER_MAX_GAP_SEC = 900.0   # 직전 동기화가 이보다 오래됐으면 체결 복원 포기(여러 건이 섞여 가격 역산 불가)


def _record_equity(account: dict[str, Any], total_eval: float, *, ts: str | None = None) -> None:
    """사이트 NAV(시가평가 총액) 스냅샷을 계정에 누적한다.

    이 기록이 없으면 '오늘 수익률'의 기준선(어제 종가 NAV)이 아예 존재하지 않는다. 예전엔
    기준선을 못 찾으면 원금으로 폴백해서, 오늘·주간·월간 수익률이 전부 누적 수익률과 똑같은
    값으로 붕괴했다(2026-07-14 실측: 다섯 지표가 모두 -3.1179%).
    """
    ts = ts or _now()
    total_eval = float(total_eval or 0.0)
    if total_eval <= 0:
        return
    series = account.get("equity")
    if not isinstance(series, list):
        series = []
    if series:
        last = series[-1]
        if (_parse_dt(ts) - _parse_dt(last.get("ts"))).total_seconds() < _EQUITY_MIN_GAP_SEC:
            last["ts"] = ts                      # 같은 분 안에서는 마지막 값으로 갱신만 한다
            last["total_eval"] = total_eval
            account["equity"] = series
            return
    series.append({"ts": ts, "total_eval": total_eval})
    account["equity"] = _compact_equity(series)


def _compact_equity(series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """오래된 구간은 '그 날의 마지막 값'(=종가 NAV)만 남긴다 — 기간 기준선엔 그것만 있으면 된다."""
    if len(series) <= _EQUITY_MAX_POINTS:
        return series
    cutoff = datetime.now(timezone.utc) - timedelta(days=_EQUITY_KEEP_DAYS)
    daily: dict[str, dict[str, Any]] = {}
    recent: list[dict[str, Any]] = []
    for point in series:
        dt = _parse_dt(point.get("ts"))
        if dt >= cutoff:
            recent.append(point)
        else:
            daily[_to_kst(dt).strftime("%Y-%m-%d")] = point
    return sorted([*daily.values(), *recent], key=lambda x: str(x.get("ts")))


def _infer_fills(old_positions: dict[str, Any], new_positions: dict[str, Any],
                 old_cash: float, new_cash: float, *, ts: str) -> list[dict[str, Any]]:
    """사이트 보유·현금 변화로 체결을 복원한다.

    타임폴리오는 체결가를 '주문 내역' 탭 원장에만 두는데, 그 탭을 열면 이어지는 주문이 깨진다
    (timefolio_browser.ledger_orders 주석). 그래서 LIVE 모드에선 체결이 로컬 장부에 한 줄도
    쌓이지 않았고 실현손익·승률·거래내역이 영구히 0 이었다.
    매수가는 사이트 평단 변화로 역산한다. 매도가는 사이트가 노출하는 당시 현재가를 쓴다.

    예전에는 ``새 현금 - 이전 현금 + 매수대금``을 매도대금으로 간주했다. 하지만 NAV와 보유
    현재가는 서로 다른 순간에 렌더되므로 그 차이에는 다른 보유종목의 시가 변동과 반올림이
    섞인다. 이를 소량 분할 체결 수량으로 나누면 실제 하루 고가·저가를 벗어난 가짜 체결가가
    만들어진다. 타임폴리오 주문 원장에는 체결가가 없으므로 정확한 값인 척하지 않고, 사이트
    시장가 스냅샷을 ``estimated`` 체결가로 기록한다.
    """
    fills: list[dict[str, Any]] = []
    for ticker, new in new_positions.items():
        old = old_positions.get(ticker) or {}
        old_qty, new_qty = int(old.get("qty") or 0), int(new.get("qty") or 0)
        if new_qty <= old_qty:
            continue
        delta = new_qty - old_qty
        old_avg = float(old.get("avg_price") or 0.0)
        new_avg = float(new.get("avg_price") or 0.0)
        price = ((new_avg * new_qty) - (old_avg * old_qty)) / delta
        if price <= 0:
            price = float(new.get("last_price") or new_avg or 0.0)
        if price <= 0:
            continue
        fills.append({"ts": ts, "side": "buy", "ticker": ticker, "qty": delta, "price": price,
                      "amount": delta * price, "accepted": True, "filled": True, "inferred": True,
                      "estimated": True, "price_source": "position_average_delta"})

    sells: list[tuple[str, int, float, float]] = []
    for ticker, old in old_positions.items():
        old_qty = int(old.get("qty") or 0)
        new = new_positions.get(ticker) or {}
        new_qty = int(new.get("qty") or 0)
        if new_qty >= old_qty:
            continue
        # 부분 체결이면 이번 동기화의 현재가가 가장 가깝고, 전량 청산으로 보유행이 사라졌으면
        # 직전 동기화의 현재가가 최선이다. 둘 다 체결가 자체는 아니므로 estimated 로 표시한다.
        ref = float(new.get("last_price") or old.get("last_price") or old.get("avg_price") or 0.0)
        sells.append((ticker, old_qty - new_qty, ref, float(old.get("avg_price") or 0.0)))
    for ticker, qty, price, avg in sells:
        if price <= 0:
            continue
        pnl = (price - avg) * qty if avg > 0 else 0.0
        fills.append({"ts": ts, "side": "sell", "ticker": ticker, "qty": qty, "price": price,
                      "amount": qty * price, "accepted": True, "filled": True, "inferred": True,
                      "estimated": True, "price_source": "market_snapshot",
                      "avg_price": avg, "pnl": pnl,
                      "pnl_pct": ((price / avg - 1.0) * 100.0) if avg > 0 else 0.0})
    return fills


_INFERRED_FILL_GROUP_SEC = 10 * 60


def _in_site_fill_session(ts: Any) -> bool:
    """사이트 보유 변화로 체결을 복원해도 되는 한국 장중/마감 반영 시간."""
    dt = _to_kst(_parse_dt(ts))
    return dt.weekday() < 5 and dtime(8, 55) <= dt.time() <= dtime(15, 40)


def consolidate_inferred_trades(trades: list[dict[str, Any]], *,
                                window_sec: float = _INFERRED_FILL_GROUP_SEC) -> list[dict[str, Any]]:
    """같은 주문에서 쪼개진 추정 체결을 종목·방향별 한 줄로 합친다.

    동일 종목의 반대 방향 체결이 나오면 새 매매 에피소드로 보며, 다른 종목 체결이 사이에
    끼는 것은 허용한다. PAPER의 실제 로컬 체결과 거절 기록은 손대지 않는다.
    """
    out: list[dict[str, Any]] = []
    last_by_ticker: dict[str, int] = {}
    for original in trades or []:
        row = dict(original)
        ticker = str(row.get("ticker") or "").zfill(6)
        side = str(row.get("side") or "").lower()
        eligible = bool(row.get("inferred") and row.get("accepted") and row.get("filled")
                        and side in {"buy", "sell"} and int(row.get("qty") or 0) > 0)
        previous_idx = last_by_ticker.get(ticker)
        previous = out[previous_idx] if previous_idx is not None else None
        merge = False
        if eligible and previous and previous.get("inferred") and previous.get("side") == side:
            gap = (_parse_dt(row.get("ts")) - _parse_dt(previous.get("ts"))).total_seconds()
            merge = 0 <= gap <= float(window_sec)
        if not merge:
            if eligible:
                row.setdefault("estimated", True)
                row.setdefault("price_source", "cash_nav_episode_estimate")
                row.setdefault("first_ts", row.get("ts"))
                row.setdefault("fill_count", 1)
            out.append(row)
            if ticker.strip("0"):
                last_by_ticker[ticker] = len(out) - 1
            continue

        old_qty = int(previous.get("qty") or 0)
        add_qty = int(row.get("qty") or 0)
        total_qty = old_qty + add_qty
        if total_qty <= 0:
            continue
        old_price = float(previous.get("price") or 0.0)
        add_price = float(row.get("price") or 0.0)
        previous["qty"] = total_qty
        previous["price"] = ((old_price * old_qty) + (add_price * add_qty)) / total_qty
        previous["amount"] = previous["price"] * total_qty
        previous["ts"] = row.get("ts") or previous.get("ts")
        previous["fill_count"] = int(previous.get("fill_count") or 1) + int(row.get("fill_count") or 1)
        previous["estimated"] = True
        sources = {str(previous.get("price_source") or ""), str(row.get("price_source") or "")}
        previous["price_source"] = (sources.pop() if len(sources) == 1
                                    else "mixed_episode_estimate")
        if side == "sell":
            old_avg = float(previous.get("avg_price") or 0.0)
            add_avg = float(row.get("avg_price") or 0.0)
            if old_avg > 0 or add_avg > 0:
                previous["avg_price"] = ((old_avg * old_qty) + (add_avg * add_qty)) / total_qty
            if previous.get("pnl") is not None or row.get("pnl") is not None:
                previous["pnl"] = float(previous.get("pnl") or 0.0) + float(row.get("pnl") or 0.0)
            avg = float(previous.get("avg_price") or 0.0)
            previous["pnl_pct"] = ((previous["price"] / avg - 1.0) * 100.0) if avg > 0 else 0.0
    return out


def repair_inferred_trade_history(uid: int) -> dict[str, Any]:
    """기존 분할 추정 체결을 합쳐 장부와 승률의 조각 체결 중복을 제거한다."""
    with _LOCK:
        data = _load()
        account = _account_raw(data, uid)
        old = list(account.get("trades") or [])
        eligible = [row for row in old if not row.get("inferred") or _in_site_fill_session(row.get("ts"))]
        repaired = consolidate_inferred_trades(eligible)
        account["trades"] = repaired[-2000:]
        account["trade_history_repaired_at"] = _now()
        account["updated_at"] = _now()
        _save(data)
    return {"before": len(old), "after": len(repaired), "merged": len(eligible) - len(repaired),
            "removed_outside_session": len(old) - len(eligible)}


def set_site_orders(uid: int, rows: list[dict[str, Any]]) -> None:
    """사이트 '주문 내역' 원장 스냅샷(입력T·종목·매수도·비중·지정가·주문T·취소T).

    체결가·수량이 없는 표라 손익 계산엔 못 쓰지만, **실제로 사이트에 무엇이 접수/취소됐는지**의
    유일한 1차 기록이다(거절·취소까지 남는다). 표시·감사용으로만 보관한다.
    """
    with _LOCK:
        data = _load()
        account = _account_raw(data, uid)
        account["site_orders"] = list(rows or [])[-200:]
        account["site_orders_at"] = _now()
        _save(data)


def record_equity(uid: int) -> None:
    """현재 시가평가 총액을 NAV 스냅샷으로 남긴다 (PAPER 사이클 종료 시 호출)."""
    with _LOCK:
        data = _load()
        account = _account_raw(data, uid)
        total = float((portfolio_from_account(account) or {}).get("total_eval") or 0.0)
        if total <= 0:
            return
        _record_equity(account, total)
        _save(data)


def sync_site_portfolio(uid: int, *, positions: list[dict[str, Any]], total_eval: float,
                        weekly_turnover_pct_value: float | None = None) -> dict[str, Any]:
    total_eval = float(total_eval or DEFAULT_INITIAL_CASH)
    if total_eval <= 0:
        total_eval = DEFAULT_INITIAL_CASH
    new_positions: dict[str, dict[str, Any]] = {}
    position_value = 0.0
    for row in positions or []:
        ticker = str(row.get("ticker") or row.get("code") or "").replace("A", "").strip().zfill(6)
        qty = int(float(str(row.get("qty") or 0).replace(",", "")))
        last = float(str(row.get("last_price") or row.get("cur_price") or 0).replace(",", ""))
        avg = float(str(row.get("avg_price") or last or 0).replace(",", ""))
        if not ticker.strip("0") or qty <= 0 or last <= 0:
            continue
        new_positions[ticker] = {"qty": qty, "avg_price": avg or last, "last_price": last}
        position_value += qty * last
    with _LOCK:
        data = _load()
        account = _account_raw(data, uid)
        # 보유 시작 시각 보존/기록 — 고아 포지션(신호 만료 후 방치) 최대보유시간 청산의 근거.
        old_positions = account.get("positions") or {}
        for tkr, pos in new_positions.items():
            prev = old_positions.get(tkr) or {}
            pos["first_seen"] = prev.get("first_seen") or _now()
        # 포지션이 사라진 종목은 매도 미체결 재시도 카운터도 정리한다(청산 완료).
        attempts = account.get("sell_attempts") or {}
        for tkr in list(attempts.keys()):
            if tkr not in new_positions:
                attempts.pop(tkr, None)
        # NAV 스크레이프 글리치 가드: 로그인 직후 대시보드 미렌더 상태에서 NAV 정규식이 엉뚱한
        # 작은 수(관측: 8 → total_eval 8,000,000)를 잡으면 cash/수익률이 오염된다(-99% 로 튐).
        # 대회 시작가(대개 10억) 대비 10% 미만이면 글리치로 간주하고 이번 동기화를 무시한다
        # (직전 상태 보존 — 다음 정상 동기화가 갱신). 실제 -90% 손실은 이 신규 계정 맥락상 비현실적.
        #
        # 글리치 바닥은 '관측된 NAV 최고점'(site_peak_eval)으로 잡는다. 예전엔 이 자리에
        # initial_cash 를 최고점으로 끌어올려 썼는데(래칫), 그러면 원금이 수익을 따라 올라가
        # 누적 수익률이 구조적으로 절대 플러스가 될 수 없었다 — 이익은 즉시 원금에 흡수되고
        # 이후 하락분만 손실로 남는다(2026-07-14 실측: 원금이 10.0억 → 10.069억으로 부풀어 있었다).
        peak = max(float(account.get("site_peak_eval") or 0.0),
                   float(account.get("initial_cash") or 0.0))
        glitch_floor = max(DEFAULT_INITIAL_CASH, peak) * 0.1
        if total_eval < glitch_floor:
            return _public_account(account) or {}
        account["site_peak_eval"] = max(peak, total_eval)

        new_cash = max(0.0, total_eval - position_value)
        # 사이트 체결 복원 — 직전 동기화 이후의 보유·현금 변화가 곧 체결이다. 동기화 간격이
        # 벌어졌으면(재기동·장 마감 등) 여러 건이 섞여 가격 역산이 불가능하니 복원하지 않는다.
        last_sync = account.get("site_synced_at")
        gap_sec = ((datetime.now(timezone.utc) - _parse_dt(last_sync)).total_seconds()
                   if last_sync else None)
        fill_ts = _now()
        if (gap_sec is not None and gap_sec <= _FILL_INFER_MAX_GAP_SEC
                and _in_site_fill_session(fill_ts)):
            fills = _infer_fills(old_positions, new_positions,
                                 float(account.get("cash") or 0.0), new_cash, ts=fill_ts)
            if fills:
                account["trades"] = consolidate_inferred_trades(
                    [*(account.get("trades") or []), *fills])[-2000:]

        account["cash"] = new_cash
        account["positions"] = new_positions
        if weekly_turnover_pct_value is not None:
            account["site_weekly_turnover_pct"] = float(weekly_turnover_pct_value)
        _record_equity(account, total_eval)
        account["site_synced_at"] = _now()
        account["updated_at"] = _now()
        _save(data)
        return _public_account(account) or {}


def mark_contest_closed(uid: int, reason: str) -> None:
    """대회 종료(주문 창 폐쇄)를 계정에 기록한다 — 최초 시각·마지막 확인 시각·사유.

    2026-09-10 사장 지시: 대회가 끝나 매도가 안 되는 건 당연한 일이니 오류로 취급하지 말고
    사유를 남겨두라. UI·watcher 는 이 값을 보고 '장애 아님'으로 읽는다.
    """
    with _LOCK:
        data = _load()
        account = _account_raw(data, uid)
        cur = account.get("contest_closed")
        if not isinstance(cur, dict):
            cur = {"since": _now()}
        # 최초 사유(사이트에서 실제로 확인한 문구)를 남긴다 — 이후 사이클은 래치가 던지는
        # '재확인 대기' 문구라 그걸로 덮으면 원인 기록이 흐려진다.
        cur.setdefault("reason", str(reason or "대회 종료")[:200])
        cur["last_seen"] = _now()
        account["contest_closed"] = cur
        account["updated_at"] = _now()
        _save(data)


def contest_closed(uid: int) -> dict[str, Any] | None:
    """대회 종료 기록({since, reason, last_seen}) 또는 None."""
    account = get_account_raw(uid) or {}
    cur = account.get("contest_closed")
    return dict(cur) if isinstance(cur, dict) else None


def clear_contest_closed(uid: int) -> None:
    """주문이 다시 접수되면 해제 — 새 대회가 열리면 저절로 원상 복귀한다."""
    with _LOCK:
        data = _load()
        account = _account_raw(data, uid)
        if account.pop("contest_closed", None) is not None:
            account["updated_at"] = _now()
            _save(data)


def mark_recent_buy(uid: int, ticker: str) -> None:
    """방금 매수(제출)한 종목을 시각과 함께 기록 — 미체결 주문 재매수(처닝) 방지용.
    sync_site_portfolio 는 positions/cash 만 덮어쓰므로 recent_buys 는 보존된다."""
    _mark_recent(uid, "recent_buys", ticker)


def recently_bought(uid: int, *, within_min: int = 15) -> set[str]:
    """최근 within_min 분 내 매수 제출한 종목 집합 — 미체결분이 사이트 보유목록에 뜨기 전
    매 사이클 같은 종목을 반복 매수하는 처닝을 막는다."""
    return _recent(uid, "recent_buys", within_min)


def mark_recent_reject(uid: int, ticker: str, side: str, *, cooldown_min: int,
                       reason: str | None = None) -> None:
    """사이트 거절 종목을 잠시 쉬게 해 같은 실패를 매분 재제출하지 않는다."""
    ticker = str(ticker or "").zfill(6)
    if not ticker.strip("0"):
        return
    with _LOCK:
        data = _load()
        account = _account_raw(data, uid)
        rejects = account.setdefault("recent_rejects", {})
        rejects[f"{str(side).lower()}:{ticker}"] = {
            "ts": _now(), "cooldown_min": max(1, int(cooldown_min or 1)), "reason": str(reason or "")[:300]
        }
        cutoff = datetime.now(timezone.utc) - timedelta(hours=6)
        for key, value in list(rejects.items()):
            try:
                if _parse_dt((value or {}).get("ts")) < cutoff:
                    rejects.pop(key, None)
            except Exception:
                rejects.pop(key, None)
        account["updated_at"] = _now()
        _save(data)


def recently_rejected(uid: int, side: str) -> set[str]:
    """아직 종목별 거절 쿨다운이 끝나지 않은 종목 집합."""
    now = datetime.now(timezone.utc)
    prefix = f"{str(side).lower()}:"
    with _LOCK:
        account = _load().get("accounts", {}).get(str(int(uid))) or {}
    out: set[str] = set()
    for key, value in (account.get("recent_rejects") or {}).items():
        if not str(key).startswith(prefix) or not isinstance(value, dict):
            continue
        try:
            until = _parse_dt(value.get("ts")) + timedelta(minutes=max(1, int(value.get("cooldown_min") or 1)))
            if until > now:
                out.add(str(key).split(":", 1)[1].zfill(6))
        except Exception:
            continue
    return out


def mark_recent_sell(uid: int, ticker: str) -> None:
    """방금 매도(제출)한 종목 기록 — 체결이 사이트 보유목록에서 빠지기 전 다음 사이클이
    같은 종목을 다시 매도(중복 청산)하는 것을 막는다. recent_buys 와 대칭."""
    _mark_recent(uid, "recent_sells", ticker)


def recently_sold(uid: int, *, within_min: int = 2) -> set[str]:
    """최근 within_min 분 내 매도 제출한 종목 집합 — 중복 청산 방지."""
    return _recent(uid, "recent_sells", within_min)


def clear_recent_sell(uid: int, ticker: str) -> None:
    """중복청산 방지 마킹 해제 — 미체결 매도를 취소한 직후 같은/다음 사이클이 지체 없이
    재매도할 수 있게 한다(마킹이 남아 있으면 2분을 그냥 흘려보낸다)."""
    ticker = str(ticker or "").zfill(6)
    with _LOCK:
        data = _load()
        account = _account_raw(data, uid)
        if (account.get("recent_sells") or {}).pop(ticker, None) is not None:
            account["updated_at"] = _now()
            _save(data)


def bump_sell_attempt(uid: int, ticker: str, *, stale_min: int = 30) -> int:
    """매도 시도 카운터를 올리고 현재 값(n번째 시도)을 돌려준다.

    n 이 커질수록 상대호가 틱을 깊게 넣어 도망가는 가격을 쫓아간다(2026-07-13: 상대1호가
    손절 매도가 급락 구간에서 1시간 미체결). 마지막 시도가 stale_min 을 넘었으면 새 에피소드로
    보고 1부터 다시 센다. 카운터는 포지션이 사라지면 sync_site_portfolio 가 정리한다.
    """
    ticker = str(ticker or "").zfill(6)
    with _LOCK:
        data = _load()
        account = _account_raw(data, uid)
        attempts = account.setdefault("sell_attempts", {})
        entry = attempts.get(ticker) or {}
        n = int(entry.get("n") or 0)
        try:
            stale = _parse_dt(entry.get("ts")) < datetime.now(timezone.utc) - timedelta(minutes=stale_min)
        except Exception:
            stale = True
        n = 1 if stale else n + 1
        attempts[ticker] = {"n": n, "ts": _now()}
        account["updated_at"] = _now()
        _save(data)
        return n


def track_fill_progress(uid: int, ticker: str, qty: int) -> dict[str, Any]:
    """미체결 주문의 체결 진행을 추적한다 — {stalled_min: 수량이 그대로인 시간(분)}.

    사이트는 대량 매도를 '과주문 후 대기'로 **분할 체결**한다(2026-07-13 실측: 덕산네오룩스
    2907→2750→2102주). 진행 중인 주문을 취소·재주문하면 체결을 되레 방해하므로, 취소는
    수량이 일정 시간 **전혀 줄지 않은**(정체된) 주문에만 한다.
    """
    ticker = str(ticker or "").zfill(6)
    qty = int(qty or 0)
    with _LOCK:
        data = _load()
        account = _account_raw(data, uid)
        prog = account.setdefault("fill_progress", {})
        entry = prog.get(ticker) or {}
        prev_qty = int(entry.get("qty") or -1)
        if prev_qty != qty:                      # 수량 변화 = 체결 진행 → 시계를 리셋
            prog[ticker] = {"qty": qty, "ts": _now()}
            account["updated_at"] = _now()
            _save(data)
            return {"stalled_min": 0.0, "progressing": prev_qty > qty}
        try:
            stalled = (datetime.now(timezone.utc) - _parse_dt(entry.get("ts"))).total_seconds() / 60.0
        except Exception:
            stalled = 0.0
        return {"stalled_min": max(0.0, stalled), "progressing": False}


def sell_attempts(uid: int) -> dict[str, dict[str, Any]]:
    """종목별 매도 시도 카운터 {ticker: {n, ts}} — 대시보드 미체결 경고 배지용."""
    with _LOCK:
        account = _load().get("accounts", {}).get(str(int(uid))) or {}
    return dict(account.get("sell_attempts") or {})


def position_age_min(pos: dict[str, Any]) -> float | None:
    """포지션 보유 경과(분). first_seen 이 없으면 None."""
    try:
        first = _parse_dt(pos.get("first_seen"))
    except Exception:
        return None
    return max(0.0, (datetime.now(timezone.utc) - first).total_seconds() / 60.0)


def _mark_recent(uid: int, field: str, ticker: str) -> None:
    ticker = str(ticker or "").zfill(6)
    if not ticker.strip("0"):
        return
    with _LOCK:
        data = _load()
        account = _account_raw(data, uid)
        rb = account.setdefault(field, {})
        rb[ticker] = _now()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=2)   # 오래된 항목 정리(무한 성장 방지)
        for tkr in list(rb.keys()):
            try:
                if _parse_dt(rb[tkr]) < cutoff:
                    rb.pop(tkr, None)
            except Exception:
                rb.pop(tkr, None)
        account["updated_at"] = _now()
        _save(data)


def _recent(uid: int, field: str, within_min: int) -> set[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max(0, int(within_min or 0)))
    with _LOCK:
        account = _load().get("accounts", {}).get(str(int(uid))) or {}
    out: set[str] = set()
    for tkr, ts in (account.get(field) or {}).items():
        try:
            if _parse_dt(ts) >= cutoff:
                out.add(str(tkr).zfill(6))
        except Exception:
            continue
    return out


def update_position_price(uid: int, ticker: str, price: float, *, meta: dict[str, Any] | None = None) -> dict[str, Any] | None:
    ticker = str(ticker).strip().zfill(6)
    price = float(price or 0.0)
    if price <= 0:
        return get_account(uid)
    with _LOCK:
        data = _load()
        account = _account_raw(data, uid)
        pos = (account.get("positions") or {}).get(ticker)
        if pos:
            pos["last_price"] = price
            account["updated_at"] = _now()
        if meta:
            data.setdefault("securities", {})[ticker] = normalize_meta(ticker, {**meta, "last_price": price})
        _save(data)
        return _public_account(account)


def _account_raw(data: dict[str, Any], uid: int) -> dict[str, Any]:
    account = data.get("accounts", {}).get(str(int(uid)))
    if not isinstance(account, dict):
        raise ValueError("타임폴리오 모의투자 계정이 없습니다. 먼저 가입하세요.")
    account.setdefault("positions", {})
    account.setdefault("trades", [])
    return account


def verify_account_password(uid: int, password: str | None) -> bool:
    if not password:
        return True
    with _LOCK:
        account = _load().get("accounts", {}).get(str(int(uid)))
    return bool(account and auth_store.verify_pw_hash(account.get("password_hash", ""), password))


def upsert_security_meta(ticker: str, meta: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_meta(ticker, meta)
    normalized["updated_at"] = _now()
    with _LOCK:
        data = _load()
        data.setdefault("securities", {})[normalized["ticker"]] = normalized
        _save(data)
    return normalized


def get_security_meta(ticker: str) -> dict[str, Any] | None:
    ticker = str(ticker).zfill(6)
    with _LOCK:
        meta = _load().get("securities", {}).get(ticker)
    return dict(meta) if isinstance(meta, dict) else None


def list_security_meta() -> list[dict[str, Any]]:
    with _LOCK:
        rows = list((_load().get("securities") or {}).values())
    return sorted((dict(x) for x in rows if isinstance(x, dict)), key=lambda x: x.get("ticker", ""))


def check_order(uid: int, side: Literal["buy", "sell"], ticker: str, qty: int, price: float,
                *, meta: dict[str, Any] | None = None,
                relax_sector: bool | None = None) -> dict[str, Any]:
    ticker = str(ticker).strip().zfill(6)
    side = "buy" if side == "buy" else "sell"
    qty = int(qty or 0)
    price = float(price or 0.0)
    with _LOCK:
        data = _load()
        account = _account_raw(data, uid)
        metas = {k: dict(v) for k, v in (data.get("securities") or {}).items() if isinstance(v, dict)}
        if meta:
            metas[ticker] = normalize_meta(ticker, {**meta, "last_price": price or meta.get("last_price")})
        if ticker not in metas:
            metas[ticker] = normalize_meta(ticker, {"last_price": price})
        elif price > 0:
            metas[ticker] = normalize_meta(ticker, {**metas[ticker], "last_price": price})
        check = validate_order(account, side, ticker, qty, price, metas, relax_sector=relax_sector)
        return check.to_dict()


def place_order(uid: int, side: Literal["buy", "sell"], ticker: str, qty: int, price: float,
                *, password: str | None = None, meta: dict[str, Any] | None = None,
                relax_sector: bool | None = None) -> dict[str, Any]:
    ticker = str(ticker).strip().zfill(6)
    side = "buy" if side == "buy" else "sell"
    qty = int(qty or 0)
    price = float(price or 0.0)
    with _LOCK:
        data = _load()
        account = _account_raw(data, uid)
        if password and not auth_store.verify_pw_hash(account.get("password_hash", ""), password):
            raise ValueError("타임폴리오 모의투자 비밀번호가 일치하지 않습니다.")
        if meta:
            data.setdefault("securities", {})[ticker] = normalize_meta(ticker, {**meta, "last_price": price or meta.get("last_price")})
        metas = {k: dict(v) for k, v in (data.get("securities") or {}).items() if isinstance(v, dict)}
        if ticker not in metas:
            metas[ticker] = normalize_meta(ticker, {"last_price": price})
        elif price > 0:
            metas[ticker] = normalize_meta(ticker, {**metas[ticker], "last_price": price})
            data["securities"][ticker] = metas[ticker]

        check = validate_order(account, side, ticker, qty, price, metas, relax_sector=relax_sector)
        if not check.ok:
            rec = {"ts": _now(), "ticker": ticker, "side": side, "qty": qty, "price": price,
                   "accepted": False, "filled": False, "rule_check": check.to_dict()}
            account.setdefault("trades", []).append(rec)
            account["updated_at"] = _now()
            _save(data)
            return {"ok": False, "accepted": False, "filled": False, "rule_check": check.to_dict(), "order": rec,
                    "account": _public_account(account)}

        amount = qty * price
        positions = account.setdefault("positions", {})
        pos = positions.get(ticker) or {"qty": 0, "avg_price": price, "last_price": price}
        old_qty = int(pos.get("qty") or 0)
        if side == "buy":
            old_avg = float(pos.get("avg_price") or price)
            new_qty = old_qty + qty
            pos["qty"] = new_qty
            pos["avg_price"] = ((old_avg * old_qty) + amount) / max(1, new_qty)
            pos["last_price"] = price
            pos.setdefault("first_seen", _now())   # 최대보유시간(고아 청산) 판정 근거
            positions[ticker] = pos
            account["cash"] = float(account.get("cash") or 0.0) - amount
        else:
            new_qty = old_qty - qty
            if new_qty > 0:
                pos["qty"] = new_qty
                pos["last_price"] = price
                positions[ticker] = pos
            else:
                positions.pop(ticker, None)
            account["cash"] = float(account.get("cash") or 0.0) + amount
        rec = {"ts": _now(), "ticker": ticker, "side": side, "qty": qty, "price": price,
               "amount": amount, "accepted": True, "filled": True, "rule_check": check.to_dict()}
        account.setdefault("trades", []).append(rec)
        account["updated_at"] = _now()
        _save(data)
        return {"ok": True, "accepted": True, "filled": True, "rule_check": check.to_dict(), "order": rec,
                "account": _public_account(account)}


def portfolio_from_account(account: dict[str, Any]) -> dict[str, Any]:
    positions = account.get("positions") or {}
    pos_rows = []
    total_pos = 0.0
    for code, pos in positions.items():
        qty = int(pos.get("qty") or 0)
        avg = float(pos.get("avg_price") or 0.0)
        last = float(pos.get("last_price") or avg or 0.0)
        value = qty * last
        pnl_krw = (last - avg) * qty
        pnl_pct = ((last / avg - 1.0) * 100.0) if avg > 0 and last > 0 else 0.0
        total_pos += value
        pos_rows.append({"ticker": code, "qty": qty, "avg_price": avg, "last_price": last,
                         "value": value, "pnl_krw": pnl_krw, "pnl_pct": pnl_pct})
    cash = float(account.get("cash") or 0.0)
    total = cash + total_pos
    for row in pos_rows:
        row["weight_pct"] = (row["value"] / total * 100.0) if total > 0 else 0.0
    site_turnover = account.get("site_weekly_turnover_pct")
    weekly_turnover = float(site_turnover) if site_turnover is not None else weekly_turnover_pct(account)
    total_unrealized = sum(float(row.get("pnl_krw") or 0.0) for row in pos_rows)
    invested = sum(float(row.get("avg_price") or 0.0) * int(row.get("qty") or 0) for row in pos_rows)
    return {
        "cash": cash,
        "positions_value": total_pos,
        "total_eval": total,
        "positions": sorted(pos_rows, key=lambda x: x["ticker"]),
        "unrealized_pnl_krw": total_unrealized,
        "unrealized_pnl_pct": (total_unrealized / invested * 100.0) if invested > 0 else 0.0,
        "weekly_turnover_pct": weekly_turnover,
        "weekly_turnover_required_pct": 5.0,
        "weekly_turnover_ok": weekly_turnover >= 5.0,
    }


KST = timezone(timedelta(hours=9))


def _parse_dt(value: Any) -> datetime:
    try:
        dt = datetime.fromisoformat(str(value))
    except Exception:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _to_kst(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(KST)


def balance_snapshot(account: dict[str, Any]) -> dict[str, Any]:
    portfolio = portfolio_from_account(account)
    holdings = []
    for row in portfolio.get("positions") or []:
        meta = get_security_meta(row.get("ticker", "")) or {}
        holdings.append({
            "code": row.get("ticker"),
            "name": meta.get("name") or row.get("ticker"),
            "qty": row.get("qty") or 0,
            "avg_price": row.get("avg_price") or 0.0,
            "cur_price": row.get("last_price") or 0.0,
            "eval_amt": row.get("value") or 0.0,
            "pnl": row.get("pnl_krw") or 0.0,
            "pnl_pct": row.get("pnl_pct") or 0.0,
            "weight_pct": row.get("weight_pct") or 0.0,
            "category": "타임폴리오",
            "ccy": "KRW",
        })
    initial = float(account.get("initial_cash") or DEFAULT_INITIAL_CASH)
    total = float(portfolio.get("total_eval") or 0.0)
    pnl = total - initial
    return {
        "buying_power": {
            "cash": portfolio.get("cash") or 0.0,
            "total_eval": total,
            "positions_value": portfolio.get("positions_value") or 0.0,
            "pnl": pnl,
            "pnl_ratio": (pnl / initial) if initial > 0 else 0.0,
            "unrealized_pnl": portfolio.get("unrealized_pnl_krw") or 0.0,
            "unrealized_pnl_pct": portfolio.get("unrealized_pnl_pct") or 0.0,
            "weekly_turnover_pct": portfolio.get("weekly_turnover_pct") or 0.0,
            "weekly_turnover_ok": bool(portfolio.get("weekly_turnover_ok")),
            "ok": True,
        },
        "holdings": holdings,
        "holdings_stale": False,
        "is_mock": True,
        "is_timefolio": True,
    }


def _replay_points(account: dict[str, Any]) -> list[dict[str, Any]]:
    initial = float(account.get("initial_cash") or DEFAULT_INITIAL_CASH)
    cash = initial
    positions: dict[str, dict[str, float]] = {}
    created = _parse_dt(account.get("created_at") or _now())
    points = [{"ts": created.isoformat(), "total_eval": initial, "adj_total_eval": initial, "cum_pnl": 0.0}]

    # 실제 NAV 스냅샷이 있으면 그것이 유일한 진실이다. 아래 체결 재생(replay)은 보유를 '마지막
    # 체결가'로 마킹하기 때문에 시가평가가 아니라 취득원가에 가깝고, 기간 기준선으로 쓸 수 없다.
    snapshots = account.get("equity") or []
    if snapshots:
        for snap in snapshots:
            total = float(snap.get("total_eval") or 0.0)
            dt = _parse_dt(snap.get("ts"))
            if total <= 0 or dt <= created:
                continue
            points.append({"ts": dt.isoformat(), "total_eval": total,
                           "adj_total_eval": total, "cum_pnl": total - initial})
        portfolio = portfolio_from_account(account)
        total = float(portfolio.get("total_eval") or initial)
        updated = _parse_dt(account.get("updated_at") or _now())
        points.append({"ts": updated.isoformat(), "total_eval": total,
                       "adj_total_eval": total, "cum_pnl": total - initial})
        return sorted(points, key=lambda p: str(p.get("ts")))

    def mark_total() -> float:
        return cash + sum(float(p.get("qty") or 0) * float(p.get("last_price") or p.get("avg_price") or 0) for p in positions.values())

    rows = consolidate_inferred_trades(list(account.get("trades") or []))
    for trade in sorted(rows, key=lambda x: str(x.get("ts") or "")):
        if not trade.get("accepted") or not trade.get("filled"):
            continue
        ticker = str(trade.get("ticker") or "").zfill(6)
        side = str(trade.get("side") or "").lower()
        qty = int(trade.get("qty") or 0)
        price = float(trade.get("price") or 0.0)
        if not ticker or qty <= 0 or price <= 0:
            continue
        pos = positions.get(ticker) or {"qty": 0.0, "avg_price": price, "last_price": price}
        old_qty = float(pos.get("qty") or 0.0)
        old_avg = float(pos.get("avg_price") or price)
        amount = qty * price
        if side == "buy":
            new_qty = old_qty + qty
            pos["qty"] = new_qty
            pos["avg_price"] = ((old_avg * old_qty) + amount) / max(1.0, new_qty)
            pos["last_price"] = price
            positions[ticker] = pos
            cash -= amount
        elif side == "sell":
            new_qty = old_qty - qty
            cash += amount
            if new_qty > 0:
                pos["qty"] = new_qty
                pos["last_price"] = price
                positions[ticker] = pos
            else:
                positions.pop(ticker, None)
        total = mark_total()
        points.append({"ts": _parse_dt(trade.get("ts") or _now()).isoformat(), "total_eval": total,
                       "adj_total_eval": total, "cum_pnl": total - initial})

    portfolio = portfolio_from_account(account)
    total = float(portfolio.get("total_eval") or initial)
    updated = _parse_dt(account.get("updated_at") or _now())
    points.append({"ts": updated.isoformat(), "total_eval": total, "adj_total_eval": total, "cum_pnl": total - initial})
    return points


def equity_series(account: dict[str, Any], *, view: str = "realtime", limit: int = 500) -> list[dict[str, Any]]:
    view = view if view in {"realtime", "daily", "monthly"} else "realtime"
    points = _replay_points(account)
    bucket: dict[str, dict[str, Any]] = {}
    for point in points:
        dt = _to_kst(_parse_dt(point.get("ts")))
        if view == "daily":
            key = dt.strftime("%Y-%m-%d")
            label = key
        elif view == "monthly":
            key = dt.strftime("%Y-%m")
            label = key
        else:
            key = dt.strftime("%Y-%m-%d %H:%M")
            label = dt.strftime("%m-%d %H:%M")
        bucket[key] = {**point, "label": label, "ts_kst": dt.strftime("%Y-%m-%d %H:%M")}
    out = [bucket[k] for k in sorted(bucket)]
    return out[-max(1, int(limit or 500)):]


def _change_from(points: list[dict[str, Any]], start: datetime,
                 current: float) -> tuple[float | None, float | None]:
    """기간(오늘/주간/월간) 시작 직전의 마지막 실제 NAV 를 기준선으로 잡는다.

    기준선이 없으면 원금으로 폴백하지 **않고** None 을 돌려준다. 예전엔 폴백했기 때문에,
    NAV 기록이 없던 LIVE 모드에서 오늘·주간·월간 수익률이 전부 '원금 대비'가 되어 누적
    수익률과 완전히 같은 숫자로 붕괴했다(2026-07-14 실측: 다섯 지표 모두 -3.1179%).
    """
    if not points:
        return None, None
    first = _parse_dt(points[0].get("ts"))
    if first > start:
        # 계정 자체가 이 기간 안에서 시작 — 기간 수익률 = 개설 이후 수익률(원금 기준)이 맞다.
        base = float(points[0].get("total_eval") or 0.0)
    else:
        base = 0.0
        for point in points[1:]:                     # points[0] 은 개설 시점(원금) — 기준선이 아니다
            if _parse_dt(point.get("ts")) > start:
                break
            value = float(point.get("total_eval") or 0.0)
            if value > 0:
                base = value
    if base <= 0:
        return None, None                            # 기간 시작 전 NAV 기록 없음 → '계산 불가'
    chg = current - base
    return chg, (chg / base * 100.0)


def _realized_stats(account: dict[str, Any]) -> dict[str, Any]:
    positions: dict[str, dict[str, float]] = {}
    sell_count = 0
    win_count = 0
    realized = 0.0
    basis = 0.0
    rows = consolidate_inferred_trades(list(account.get("trades") or []))
    for trade in sorted(rows, key=lambda x: str(x.get("ts") or "")):
        if not trade.get("accepted") or not trade.get("filled"):
            continue
        ticker = str(trade.get("ticker") or "").zfill(6)
        side = str(trade.get("side") or "").lower()
        qty = int(trade.get("qty") or 0)
        price = float(trade.get("price") or 0.0)
        if qty <= 0 or price <= 0:
            continue
        pos = positions.get(ticker) or {"qty": 0.0, "avg_price": price}
        old_qty = float(pos.get("qty") or 0.0)
        old_avg = float(pos.get("avg_price") or price)
        if side == "buy":
            new_qty = old_qty + qty
            pos["qty"] = new_qty
            pos["avg_price"] = ((old_avg * old_qty) + qty * price) / max(1.0, new_qty)
            positions[ticker] = pos
        elif side == "sell":
            matched_qty = min(qty, int(old_qty)) if old_qty > 0 else qty
            if trade.get("pnl") is not None and float(trade.get("avg_price") or 0.0) > 0:
                # 사이트 체결 복원분은 그 자리에서 사이트 평단으로 실현손익을 확정해 둔다.
                # 여기서 장부를 되짚으면, 매수가 장부에 없는 매도(재기동·동기화 공백 중 체결)는
                # 평단을 체결가로 잡아 손익이 0 이 되고 승률이 통째로 0% 로 눌린다.
                pnl = float(trade["pnl"])
                cost = float(trade["avg_price"]) * matched_qty
            else:
                pnl = (price - old_avg) * matched_qty
                cost = old_avg * matched_qty
            realized += pnl
            basis += cost
            sell_count += 1
            if pnl > 0:
                win_count += 1
            new_qty = old_qty - qty
            if new_qty > 0:
                pos["qty"] = new_qty
                positions[ticker] = pos
            else:
                positions.pop(ticker, None)
    return {"realized_pnl": realized, "realized_basis": basis, "sell_count": sell_count,
            "win_count": win_count, "win_rate_pct": (win_count / sell_count * 100.0) if sell_count else 0.0}


def performance_snapshot(account: dict[str, Any]) -> dict[str, Any]:
    points = _replay_points(account)
    initial = float(account.get("initial_cash") or DEFAULT_INITIAL_CASH)
    portfolio = portfolio_from_account(account) or {}
    current = float(portfolio.get("total_eval") or initial)
    now = datetime.now(timezone.utc)
    today0 = now.astimezone(KST).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    week0 = (now.astimezone(KST).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=now.astimezone(KST).weekday())).astimezone(timezone.utc)
    month0 = now.astimezone(KST).replace(day=1, hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    eq_day, eq_day_pct = _change_from(points, today0, current)
    eq_week, eq_week_pct = _change_from(points, week0, current)
    eq_month, eq_month_pct = _change_from(points, month0, current)
    eq_all = current - initial
    eq_all_pct = (eq_all / initial * 100.0) if initial > 0 else 0.0
    peak = float(points[0].get("total_eval") or initial) if points else initial
    mdd = 0.0
    for point in points:
        val = float(point.get("total_eval") or 0.0)
        if val <= 0:
            continue
        peak = max(peak, val)
        if peak > 0:
            mdd = min(mdd, val / peak - 1.0)
    stats = _realized_stats(account)
    # 실현손익은 NAV 항등식으로도 구한다: (총손익) - (평가손익) = 실현손익(수수료·세 반영 후).
    # 체결 장부와 달리 이건 사이트 NAV 만으로 성립해서, 장부가 비어 있어도 항상 옳다.
    unrealized = float(portfolio.get("unrealized_pnl_krw") or 0.0)
    return {
        "has_equity": bool(account.get("equity")),
        "has_trades": bool(stats["sell_count"]),
        "unrealized_pnl": unrealized,
        "realized_net_pnl": eq_all - unrealized,
        "current": current,
        "start": initial,
        "points": len(points),
        "mdd_pct": mdd * 100.0,
        "eq_all_chg": eq_all,
        "eq_all_pct": eq_all_pct,
        "eq_today_chg": eq_day,
        "eq_today_pct": eq_day_pct,
        "eq_week_chg": eq_week,
        "eq_week_pct": eq_week_pct,
        "eq_month_chg": eq_month,
        "eq_month_pct": eq_month_pct,
        "cumulative_pnl": eq_all,
        "cumulative_pct": eq_all_pct,
        "realized_pnl": stats["realized_pnl"],
        "realized_pct": (stats["realized_pnl"] / stats["realized_basis"] * 100.0) if stats["realized_basis"] > 0 else 0.0,
        "sell_count": stats["sell_count"],
        "win_count": stats["win_count"],
        "win_rate_pct": stats["win_rate_pct"],
        "is_mock": True,
        "is_timefolio": True,
    }


def weekly_turnover_pct(account: dict[str, Any]) -> float:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=now.weekday(), hours=now.hour, minutes=now.minute, seconds=now.second, microseconds=now.microsecond)
    turnover = 0.0
    for trade in account.get("trades") or []:
        if not trade.get("accepted"):
            continue
        try:
            ts = datetime.fromisoformat(str(trade.get("ts")))
        except Exception:
            continue
        if ts >= start:
            turnover += float(trade.get("amount") or 0.0)
    avg_nav = float(account.get("initial_cash") or 0.0) or 1.0
    return turnover / avg_nav * 0.5 * 100.0

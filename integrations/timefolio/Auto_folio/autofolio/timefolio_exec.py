from __future__ import annotations

from typing import Any

from . import contest_store
from .timefolio_browser import TimefolioBrowser, TimefolioCredentials


def credentials(uid: int) -> TimefolioCredentials:
    creds = contest_store.get_site_credentials(uid)
    return TimefolioCredentials(creds["username"], creds["password"])


def _order_payload(uid: int, order: dict[str, Any]) -> dict[str, Any] | None:
    """주문 dict → 브라우저 티켓 payload. 유효하지 않으면 None."""
    ticker = str(order.get("ticker") or "").zfill(6)
    side = str(order.get("side") or "").lower()
    qty = int(order.get("qty") or 0)
    price = order.get("price") or order.get("limit_price")
    if not ticker.strip("0") or side not in {"buy", "sell"} or qty <= 0:
        return None

    account = contest_store.get_account(uid) or {}
    portfolio = account.get("portfolio") or {}
    total_eval = float(order.get("total_eval") or portfolio.get("total_eval") or contest_store.DEFAULT_INITIAL_CASH)
    amount = float(order.get("amount") or (float(price or 0.0) * qty))
    weight_pct = float(order.get("weight_pct") or 0.0)
    if weight_pct <= 0 and amount > 0 and total_eval > 0:
        weight_pct = amount / total_eval * 100.0
    if weight_pct <= 0:
        return None
    return {
        "ticker": ticker,
        "side": side,
        "qty": qty,
        "limit_price": price,
        "amount": amount,
        "total_eval": total_eval,
        # 주문 폼의 '비중%' = 이번에 매매할 비중. 매도는 보유 비중을 그대로 넣으면 전량 청산이다.
        "weight_pct": weight_pct,
        # 상대호가 틱(1~10, 기본 1). 미체결 재시도는 더 깊은 틱으로 가격을 쫓아간다(2026-07-13).
        "opp_tick": order.get("opp_tick") or 1,
        "submit": order.get("submit", True),
    }


def submit_with_browser(uid: int, browser: TimefolioBrowser, order: dict[str, Any]) -> dict[str, Any]:
    """이미 로그인된 세션으로 주문 제출 — 주문마다 브라우저를 새로 띄우지 않는다(지연 8~12초 절감)."""
    payload = _order_payload(uid, order)
    if payload is None:
        return {"accepted": False, "filled": False, "result": "invalid order"}
    if not browser.ensure_logged_in(credentials(uid)):
        return {"accepted": False, "filled": False, "result": "타임폴리오 로그인 실패"}
    return browser.place_order(payload)


def sync_with_browser(uid: int, browser: TimefolioBrowser, *, refresh: bool = True) -> dict[str, Any]:
    """이미 로그인된 세션에서 보유/NAV 를 읽어 로컬 장부에 동기화."""
    if not browser.ensure_logged_in(credentials(uid)):
        return {"ok": False, "result": "타임폴리오 로그인 실패"}
    if refresh:
        browser.refresh()
    summary = browser.scrape_summary()
    account = contest_store.sync_site_portfolio(
        uid,
        positions=summary.get("positions") or [],
        total_eval=float(summary.get("total_eval") or contest_store.DEFAULT_INITIAL_CASH),
        weekly_turnover_pct_value=summary.get("weekly_turnover_pct"),
    )
    return {"ok": True, "summary": summary, "account": account}


# ── 단발 호출용 래퍼 (QuantInSight TimefolioBroker 등 세션을 들고 있지 않은 호출자) ──────────
def submit_order(uid: int, order: dict[str, Any], *, headless: bool = True) -> dict[str, Any]:
    payload = _order_payload(uid, order)
    if payload is None:
        return {"accepted": False, "filled": False, "result": "invalid order"}
    with TimefolioBrowser(headless=headless, live_enabled=True) as browser:
        login_status = browser.login(credentials(uid))
        if not login_status.get("logged_in"):
            return {"accepted": False, "filled": False, "result": "타임폴리오 로그인 실패", "login_status": login_status}
        result = browser.place_order(payload)
        result["login_status"] = login_status
        return result


def fetch_site_summary(uid: int, *, headless: bool = True) -> dict[str, Any]:
    with TimefolioBrowser(headless=headless, live_enabled=False) as browser:
        login_status = browser.login(credentials(uid))
        if not login_status.get("logged_in"):
            return {"ok": False, "result": "타임폴리오 로그인 실패", "login_status": login_status}
        return {"ok": True, "login_status": login_status, "summary": browser.scrape_summary()}


def sync_site_account(uid: int, *, headless: bool = True) -> dict[str, Any]:
    res = fetch_site_summary(uid, headless=headless)
    if not res.get("ok"):
        return res
    summary = res.get("summary") or {}
    account = contest_store.sync_site_portfolio(
        uid,
        positions=summary.get("positions") or [],
        total_eval=float(summary.get("total_eval") or contest_store.DEFAULT_INITIAL_CASH),
        weekly_turnover_pct_value=summary.get("weekly_turnover_pct"),
    )
    return {"ok": True, "summary": summary, "account": account, "login_status": res.get("login_status")}

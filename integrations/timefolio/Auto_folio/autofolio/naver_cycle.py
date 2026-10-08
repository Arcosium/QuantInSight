from __future__ import annotations

import logging
from typing import Any

from . import contest_store
from . import order_limits
from .naver_data import fetch_security_meta
from .errors import ContestClosedError   # timefolio_browser 는 playwright 를 끌어온다

log = logging.getLogger("autofolio.cycle")

DEFAULT_TARGETS = ["005930", "000660", "035420", "051910", "068270", "005380"]


def run_cycle(uid: int, *, targets: list[str] | None = None, sell_targets: list[str] | None = None,
              max_buys: int = 1, force_sell: bool = False, take_profit_pct: float = 8.0,
              stop_loss_pct: float = 3.0, max_hold_min: float | None = None,
              min_order_weight_pct: float = 0.0, working_sells: set[str] | None = None,
              executor=None) -> dict[str, Any]:
    account = contest_store.get_account(uid)
    if not account:
        raise ValueError("타임폴리오 모의투자 계정이 없습니다. 먼저 가입하세요.")
    events: list[dict[str, Any]] = []
    sold = 0
    bought = 0
    just_sold: set[str] = set()   # 이번 사이클에 TP/SL 청산한 종목 — 같은 사이클 재매수 금지(churn 방지)
    # 리드-랙 SELL 신호 기반 청산 종목(arctrade 가 주입) — 보유 중이면 이 사이클에 매도한다.
    # (사장 지시 2026-07-07: buy 뿐 아니라 sell 도 신호 로직을 따르게.) None 이면 신호매도 없음.
    sell_set = {str(x).zfill(6) for x in (sell_targets or [])}
    # 열린 BUY 신호 집합 — 고아 판정(아래)에서 "신호가 아직 살아 있는 보유"를 보호한다.
    open_buy_set = {str(x).zfill(6) for x in (targets or [])}

    # 미체결 매도가 사이트 보유목록에서 빠지기 전 다음 사이클이 같은 종목을 또 파는 것을 막는다.
    recently_sold = contest_store.recently_sold(uid, within_min=2)
    rejected_sells = contest_store.recently_rejected(uid, "sell")
    # 사이트에 아직 살아 있는(분할 체결 중인) 매도 주문 — 같은 종목에 또 내면 사이트가
    # "전량 청산 주문 작동 시 추가 청산 불가"(TMS 오류)로 거부한다. 진행 중이니 그냥 기다린다.
    working_sells = {str(x).zfill(6) for x in (working_sells or set())}

    # 1) Reprice holdings from Naver and sell risk/forced/signal exits first.
    for code, pos in list((account.get("positions") or {}).items()):
        code = str(code or "").zfill(6)
        if not code.strip("0"):
            continue
        if code in recently_sold:
            continue
        if code in rejected_sells:
            events.append({"ok": False, "accepted": False, "filled": False, "pending": False,
                           "ticker": code, "side": "sell", "reason": "reject_cooldown",
                           "message": "최근 사이트 거절 후 쿨다운 중 — 재주문 생략"})
            continue
        if code in working_sells:   # 사이트에서 분할 체결 진행 중 — 재주문하면 TMS 거부만 난다
            events.append({"ok": False, "accepted": False, "filled": False, "pending": True,
                           "ticker": code, "side": "sell", "reason": "working_order",
                           "message": "미체결 매도 진행 중 — 재주문 생략"})
            continue
        meta = _fetch_and_store(code)
        price = float(meta.get("last_price") or pos.get("last_price") or pos.get("avg_price") or 0.0)
        avg = float(pos.get("avg_price") or price or 0.0)
        pnl_pct = ((price / avg - 1.0) * 100.0) if avg > 0 and price > 0 else 0.0
        signal_sell = code in sell_set
        # 고아 포지션 청산: 매도 신호를 (미체결/예외 등으로) 놓친 뒤 신호창(10분)이 지나면
        # TP/SL 전까지 팔 로직이 없어 1~5분 단타가 스윙으로 변질된다(2026-07-13 사건).
        # 열린 BUY 신호가 없는 보유가 max_hold_min 을 넘으면 강제 청산한다(arctrade 신호 모드 전용
        # — QuantInSight 스웜 호출은 max_hold_min=None 이라 불변).
        age_min = contest_store.position_age_min(pos)
        orphan = bool(max_hold_min is not None and age_min is not None
                      and age_min >= float(max_hold_min) and code not in open_buy_set)
        should_sell = bool(force_sell or signal_sell or orphan
                           or pnl_pct >= take_profit_pct or pnl_pct <= -abs(stop_loss_pct))
        if should_sell and price > 0:
            reason = ("force_sell" if force_sell else
                      "signal_sell" if signal_sell else
                      "take_profit" if pnl_pct >= take_profit_pct else
                      "stop_loss" if pnl_pct <= -abs(stop_loss_pct) else "max_hold")
            # 주문 1건의 예외가 사이클 전체(남은 청산·매수)를 죽이면 안 된다 — 종목 단위로 격리한다.
            # (2026-07-08 F&F: 신호매도 사이클이 통째로 실패해 10분 신호창을 놓치고 SL 까지 끌려감.)
            try:
                res = _place_or_submit(uid, "sell", code, int(pos.get("qty") or 0), price, meta, executor)
            except ContestClosedError as exc:
                # 대회가 끝나 주문 창이 없는 상태 — 장애가 아니다. 사유만 남기고 이번 사이클의
                # 남은 주문(청산·매수)은 시도하지 않는다(같은 결과를 종목 수만큼 반복할 뿐).
                contest_store.mark_contest_closed(uid, str(exc))
                log.info("대회 종료 상태 — 주문 생략 (%s)", exc)
                events.append({"ok": True, "accepted": False, "filled": False, "ticker": code,
                               "side": "sell", "reason": reason, "contest_closed": True,
                               "message": f"대회 종료 — 주문 불가: {exc}"})
                return {"ok": True, "sold": sold, "bought": 0, "events": events,
                        "contest_closed": True, "refreshed": [],
                        "account": contest_store.get_account(uid)}
            except Exception as exc:  # noqa: BLE001
                log.exception("[%s] 매도 주문 예외 uid=%s reason=%s", code, uid, reason)
                contest_store.mark_recent_reject(uid, code, "sell", cooldown_min=2, reason=str(exc))
                events.append({"ok": False, "accepted": False, "filled": False, "ticker": code,
                               "side": "sell", "qty": int(pos.get("qty") or 0), "price": price,
                               "reason": reason, "message": f"주문 예외: {exc}"})
                continue
            res["reason"] = reason
            events.append(res)
            if res.get("ok") or res.get("accepted"):
                sold += 1
                just_sold.add(code)
                contest_store.mark_recent_sell(uid, code)   # 중복 청산 방지 마킹
            else:
                site = res.get("site_execution") or {}
                reject_text = str(site.get("result") or res.get("message") or res.get("reason") or "")
                # 전일 주문/사이트 규정 거절은 같은 입력을 1분 뒤 다시 내도 결과가 같다.
                cooldown = 15 if site.get("rejected_reason") == "tms_error" else 5
                contest_store.mark_recent_reject(uid, code, "sell", cooldown_min=cooldown,
                                                 reason=reject_text)
            log.info("[%s] 매도 %s → accepted=%s filled=%s", code, reason,
                     res.get("accepted"), res.get("filled"))

    account = contest_store.get_account(uid) or account
    # targets 를 명시적으로 넘기면(빈 리스트 포함) 그것만 매수 후보로 쓴다 — arctrade 는 리드-랙 BUY
    # 신호에서 뽑은 follower 목록을 넘긴다(신호 없으면 []=매수 안함). targets=None 이면(QuantInSight 스웜 등)
    # 기존대로 store/DEFAULT_TARGETS 로 폴백. ('or' 대신 'is not None' — 빈 리스트가 폴백되지 않게.)
    target_list = [str(x).zfill(6) for x in (targets if targets is not None else _stored_or_default_targets())]
    held = {str(p.get("ticker") or "").zfill(6) for p in ((account.get("portfolio") or {}).get("positions") or [])}
    # 미체결(site_pending) 매수는 사이트 보유목록에 아직 안 떠 held 에 안 잡힌다 → 매 사이클
    # 같은 종목을 반복 매수(처닝)하는 것을 막기 위해 최근 매수분도 스킵 대상에 포함한다.
    recently = contest_store.recently_bought(uid, within_min=15)
    rejected_buys = contest_store.recently_rejected(uid, "buy")

    # 2) Buy up to max_buys names that pass contest rules.
    for code in target_list:
        if bought >= max(0, int(max_buys or 0)):
            break
        if code in held or code in just_sold or code in recently:  # 보유·방금청산·최근매수(미체결) 스킵
            continue
        if code in rejected_buys:
            events.append({"ok": False, "accepted": False, "filled": False, "ticker": code,
                           "side": "buy", "reason": "reject_cooldown",
                           "message": "최근 사이트 거절 후 쿨다운 중 — 재주문 생략"})
            continue
        meta = _fetch_and_store(code)
        price = float(meta.get("last_price") or 0.0)
        if price <= 0:
            events.append({"ok": False, "ticker": code, "side": "buy", "message": "네이버 현재가 조회 실패"})
            continue
        account_now = contest_store.get_account(uid) or account
        cash = float((account_now.get("portfolio") or {}).get("cash") or account_now.get("cash") or 0.0)
        total = float((account_now.get("portfolio") or {}).get("total_eval") or cash or 0.0)
        # 1주문 비중 상한: 일반 9%, 제외 대형주(삼전·SK하닉 등) 14% — 상승 시 섹터 한도(10%) 위반 방지.
        cap = order_limits.max_order_weight_pct(code) / 100.0
        budget = min(cash * 0.25, total * cap)
        qty = max(1, int(budget // price)) if budget >= price else 0
        if qty <= 0:
            events.append({"ok": False, "ticker": code, "side": "buy", "message": "예산 부족"})
            continue
        # 극소 주문 가드: 비중이 폼 입력 정밀도(0.0x%) 수준이면 못 파는 잔여물이 되기 쉽다
        # (2026-07-13: 두산밥캣 7주=0.045% 가 청산 불능 잔여물로 남음). 아예 사지 않는다.
        order_weight = (qty * price / total * 100.0) if total > 0 else 0.0
        if min_order_weight_pct > 0 and order_weight < float(min_order_weight_pct):
            events.append({"ok": False, "ticker": code, "side": "buy",
                           "message": f"주문 비중 {order_weight:.2f}% < 최소 {min_order_weight_pct:.2f}% — 극소 주문 스킵"})
            continue
        try:
            res = _place_or_submit(uid, "buy", code, qty, price, meta, executor)
        except ContestClosedError as exc:
            contest_store.mark_contest_closed(uid, str(exc))
            log.info("대회 종료 상태 — 매수 생략 (%s)", exc)
            events.append({"ok": True, "accepted": False, "filled": False, "ticker": code,
                           "side": "buy", "contest_closed": True,
                           "message": f"대회 종료 — 주문 불가: {exc}"})
            return {"ok": True, "sold": sold, "bought": bought, "events": events,
                    "contest_closed": True, "refreshed": [],
                    "account": contest_store.get_account(uid)}
        except Exception as exc:  # noqa: BLE001 — 종목 단위 격리(사이클 전체를 죽이지 않는다)
            log.exception("[%s] 매수 주문 예외 uid=%s", code, uid)
            contest_store.mark_recent_reject(uid, code, "buy", cooldown_min=2, reason=str(exc))
            events.append({"ok": False, "accepted": False, "filled": False, "ticker": code,
                           "side": "buy", "qty": qty, "price": price, "message": f"주문 예외: {exc}"})
            continue
        events.append(res)
        if res.get("ok") or res.get("accepted"):
            bought += 1
            held.add(code)
            contest_store.mark_recent_buy(uid, code)   # 미체결 재매수(처닝) 방지 마킹
        else:
            site = res.get("site_execution") or {}
            reject_text = str(site.get("result") or res.get("message") or res.get("reason") or "")
            contest_store.mark_recent_reject(uid, code, "buy", cooldown_min=15,
                                             reason=reject_text)
        log.info("[%s] 매수 → accepted=%s filled=%s", code, res.get("accepted"), res.get("filled"))

    refreshed = refresh_holdings(uid) if (contest_store.get_account(uid) or {}).get("positions") else {"refreshed": []}
    # 사이클마다 NAV 스냅샷 — 이 기록이 '오늘 수익률'의 기준선(어제 종가 NAV)이 된다.
    # (LIVE 는 sync_site_portfolio 에서도 찍히지만, PAPER 는 여기가 유일한 기록 지점이다.)
    contest_store.record_equity(uid)
    return {"ok": True, "sold": sold, "bought": bought, "events": events,
            "refreshed": refreshed.get("refreshed", []), "account": contest_store.get_account(uid)}


def _place_or_submit(uid: int, side: str, code: str, qty: int, price: float, meta: dict[str, Any], executor=None) -> dict[str, Any]:
    check = contest_store.check_order(uid, side, code, qty, price, meta=meta)
    if not check.get("ok"):
        return {"ok": False, "accepted": False, "filled": False, "ticker": code, "side": side,
                "qty": qty, "price": price, "rule_check": check}
    if executor is not None:
        account_now = contest_store.get_account(uid) or {}
        portfolio = account_now.get("portfolio") or {}
        total_eval = float(portfolio.get("total_eval") or account_now.get("initial_cash") or contest_store.DEFAULT_INITIAL_CASH)
        amount = float(qty) * float(price)
        weight_pct = (amount / total_eval * 100.0) if total_eval > 0 else 0.0
        order = {
            "ticker": code,
            "side": side,
            "qty": qty,
            "price": price,
            "limit_price": price,
            "amount": amount,
            "total_eval": total_eval,
            "weight_pct": weight_pct,
            "meta": meta,
        }
        if side == "sell":
            # 매도는 항상 전량 청산(사장 지시 2026-07-08). 주문 폼의 '비중%' 는 목표 비중이 아니라
            # **이번에 매매할 비중**이므로(사이트 주문내역 실측: 매도 비중 9 → 전량 T → 전량 청산),
            # 보유 전량을 수량으로, 현재 보유 비중을 주문 비중으로 넣으면 전량 청산이 된다.
            # (여기에 0 을 넣으면 아무 것도 팔리지 않는다 — 과거 target_weight_pct=0 코드는
            #  timefolio_exec 에서 탈락해 브라우저까지 간 적이 없었고, 갔다면 매도가 멈췄을 것이다.)
            held_qty = 0
            for p in (portfolio.get("positions") or []):
                if str(p.get("ticker") or "").zfill(6) == code:
                    held_qty = int(p.get("qty") or 0)
                    break
            if held_qty > 0:
                order["qty"] = held_qty
                amount = float(held_qty) * float(price)
                order["amount"] = amount
                order["weight_pct"] = (amount / total_eval * 100.0) if total_eval > 0 else 0.0
                # 극소 잔여물은 비중이 폼 정밀도 밑(0.0x%)이라 0으로 절사돼 영영 못 판다
                # (2026-07-13: 두산밥캣 7주=0.045%). 최소 0.1% 로 올리면 보유 초과분은
                # 사이트가 보유량으로 캡하고 전량 플래그를 세운다.
                order["weight_pct"] = max(float(order["weight_pct"]), 0.1)
            # 미체결 재시도 에스컬레이션: 같은 종목 n번째 매도일수록 상대호가 틱을 깊게(1→3→5→7→9→10)
            # 넣어 도망가는 가격을 쫓아간다 — 상대1호가 고정은 급락 손절에서 1시간 미체결을 낳았다.
            attempt = contest_store.bump_sell_attempt(uid, code)
            order["opp_tick"] = min(1 + 2 * (attempt - 1), 10)
        site = executor(order)
        if not site.get("accepted"):
            return {"ok": False, "accepted": False, "filled": False, "pending": False, "ticker": code, "side": side,
                    "qty": qty, "price": price, "rule_check": check, "site_execution": site}
        summary = site.get("summary") or {}
        if summary.get("positions") is not None:
            account = contest_store.sync_site_portfolio(
                uid,
                positions=summary.get("positions") or [],
                total_eval=float(summary.get("total_eval") or total_eval),
                weekly_turnover_pct_value=summary.get("weekly_turnover_pct"),
            )
            return {"ok": bool(site.get("filled")), "accepted": True, "filled": bool(site.get("filled")),
                    "pending": bool(site.get("pending") or not site.get("filled")), "ticker": code, "side": side,
                    "qty": qty, "price": price, "rule_check": check, "site_execution": site,
                    "account": account}
        if not site.get("filled"):
            return {"ok": False, "accepted": True, "filled": False, "pending": True, "ticker": code, "side": side,
                    "qty": qty, "price": price, "rule_check": check, "site_execution": site}
        res = contest_store.place_order(uid, side, code, qty, price, meta=meta,
                                        relax_sector=True)
        res["site_execution"] = site
        return res
    return contest_store.place_order(uid, side, code, qty, price, meta=meta,
                                     relax_sector=True)


def _stored_or_default_targets() -> list[str]:
    stored = [str(x.get("ticker") or "").zfill(6) for x in contest_store.list_security_meta() if x.get("ticker")]
    return stored or DEFAULT_TARGETS


def _fetch_and_store(code: str) -> dict[str, Any]:
    stored = contest_store.get_security_meta(code) or {}
    meta = fetch_security_meta(code, stored=stored)
    # Naver does not provide GICS reliably. Preserve user-entered sector data if present.
    contest_store.upsert_security_meta(code, meta)
    return contest_store.get_security_meta(code) or meta


def refresh_holdings(uid: int) -> dict[str, Any]:
    account = contest_store.get_account(uid)
    if not account:
        raise ValueError("타임폴리오 모의투자 계정이 없습니다. 먼저 가입하세요.")
    refreshed = []
    for code, pos in list((account.get("positions") or {}).items()):
        code = str(code or "").zfill(6)
        if not code.strip("0"):
            continue
        meta = _fetch_and_store(code)
        price = float(meta.get("last_price") or pos.get("last_price") or pos.get("avg_price") or 0.0)
        updated = contest_store.update_position_price(uid, code, price, meta=meta)
        refreshed.append({"ticker": code, "price": price, "meta": meta})
        account = updated or account
    return {"ok": True, "refreshed": refreshed, "account": contest_store.get_account(uid)}

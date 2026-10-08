"""Auto_folio(타임폴리오 규칙 거래) — ArcTrade 통합.

ArQuant 에서 이관된 Auto_folio 패키지를 ArcTrade 프로세스 안에서 돌린다.
- 장부는 data/autofolio_state.json (AUTOFOLIO_STATE_PATH) — ArQuant 쪽 상태와 완전 분리.
- 기본은 로컬 페이퍼 장부만 갱신한다.
- AUTOFOLIO_LIVE_ORDERS=1 이면 타임폴리오 대회 사이트에 주문을 제출하고,
  사이트 접수/체결에 실패하면 로컬 체결로 대체하지 않는다.
"""
import asyncio
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger("autofolio.runner")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                  # noqa: E402
from utils import market_time as mt            # noqa: E402
from web import signals_feed as sf             # noqa: E402  (신호 해석 — import 부작용 없음)

# contest_store 는 import 시점에 상태 파일 경로를 확정하므로 반드시 먼저 지정한다.
os.environ.setdefault("AUTOFOLIO_STATE_PATH", str(config.PRIVATE_DATA_DIR / "autofolio_state.json"))
from Auto_folio.autofolio import contest_store, naver_cycle    # noqa: E402

PAPER_UID = 0            # ArcTrade 로컬 페이퍼 계정 키 (contest_store 는 int uid 키)
_CYCLE_LOCK = asyncio.Lock()
SIGNAL_LOOKBACK_MIN = sf.SIGNAL_LOOKBACK_MIN

# Playwright sync 객체는 스레드 친화적이라, 로그인 세션을 재사용하려면 모든 사이클이
# **같은 스레드**에서 돌아야 한다 (asyncio.to_thread 는 매번 다른 워커를 줄 수 있다).
_CYCLE_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autofolio-cycle")
_SESSION = None          # 살아있는 TimefolioBrowser (로그인 유지) — _CYCLE_EXEC 스레드 전용
_SESSION_BORN = 0.0      # 세션 생성 시각(monotonic)


def _drop_session() -> None:
    global _SESSION
    if _SESSION is not None:
        try:
            _SESSION.close()
        except Exception:  # noqa: BLE001
            pass
        log.info("타임폴리오 세션 종료")
    _SESSION = None


def _site_session(*, allow_recycle: bool = True):
    """로그인된 브라우저 세션을 돌려준다(없거나 죽었으면 새로 만든다).
    주문/동기화마다 브라우저를 새로 띄우면 로그인에만 실측 7~14초가 든다 — 단타 신호엔 치명적.

    allow_recycle=False (신호 사이클) 면 오래된 세션이어도 교체하지 않는다 — 재로그인 14초가
    신호 반응을 늦추면 안 되기 때문. 장시간 SPA 페이지 누수는 한가한 주기 사이클에서 정리한다.
    """
    global _SESSION, _SESSION_BORN
    from Auto_folio.autofolio import timefolio_exec
    from Auto_folio.autofolio.timefolio_browser import TimefolioBrowser

    if _SESSION is not None and not _SESSION.alive():
        _drop_session()
    if (_SESSION is not None and allow_recycle
            and (time.monotonic() - _SESSION_BORN) > config.AUTOFOLIO_SESSION_MAX_MIN * 60):
        log.info("타임폴리오 세션 정기 교체(%d분 경과)", config.AUTOFOLIO_SESSION_MAX_MIN)
        _drop_session()
    if _SESSION is None:
        browser = TimefolioBrowser(headless=config.AUTOFOLIO_TIMEFOLIO_HEADLESS, live_enabled=True)
        try:
            browser.open()
        except Exception:
            browser.close()
            raise
        _SESSION = browser
        _SESSION_BORN = time.monotonic()
        log.info("타임폴리오 세션 신규 로그인")
    if not _SESSION.ensure_logged_in(timefolio_exec.credentials(PAPER_UID)):
        _drop_session()
        raise RuntimeError("타임폴리오 로그인 실패")
    return _SESSION


_last_cycle: dict = {}   # 마지막 사이클 결과 요약 (대시보드 표시용)
_env_credentials_seeded = False


def _seed_site_credentials_from_env() -> None:
    global _env_credentials_seeded
    if _env_credentials_seeded:
        return
    if not (config.AUTOFOLIO_SITE_USERNAME and config.AUTOFOLIO_SITE_PASSWORD):
        return
    contest_store.register(
        PAPER_UID,
        config.AUTOFOLIO_SITE_USERNAME,
        config.AUTOFOLIO_SITE_PASSWORD,
        initial_cash=config.AUTOFOLIO_INITIAL_CASH,
        reset=False,
    )
    _env_credentials_seeded = True


def ensure_account() -> dict:
    _seed_site_credentials_from_env()
    account = contest_store.get_account(PAPER_UID)
    if account is None:
        account = contest_store.register(
            PAPER_UID, "arctrade-paper", initial_cash=config.AUTOFOLIO_INITIAL_CASH)
    return account


def live_orders_ready() -> bool:
    account = ensure_account()
    return bool(config.AUTOFOLIO_LIVE_ORDERS and account.get("has_site_credentials"))


def _execution_mode(account: dict | None = None) -> str:
    account = account or ensure_account()
    if config.AUTOFOLIO_LIVE_ORDERS and account.get("has_site_credentials"):
        return "timefolio_live"
    if config.AUTOFOLIO_LIVE_ORDERS:
        return "live_unconfigured"
    return "paper"


def _save_reject_evidence(browser, order: dict, result: dict) -> None:
    """거부/미접수 주문의 사이트 응답 텍스트+스크린샷을 파일로 남긴다.

    예전엔 site_result 가 메모리의 _last_cycle 에만 있다가 다음 사이클에 소멸해, TMS 거부가
    하루 종일 반복돼도 원인을 볼 수 없었다(2026-07-13). 실패는 반드시 증거를 남긴다."""
    try:
        rej_dir = config.PRIVATE_DATA_DIR / "timefolio_rejects"
        rej_dir.mkdir(parents=True, exist_ok=True)
        stamp = mt.now_kst().strftime("%Y%m%d_%H%M%S")
        base = f"{stamp}_{order.get('side')}_{order.get('ticker')}"
        lines = [f"{k}: {order.get(k)}" for k in ("ticker", "side", "qty", "weight_pct", "opp_tick")]
        lines += ["", f"rejected_reason: {result.get('rejected_reason')}",
                  f"result: {result.get('result')}"]
        (rej_dir / f"{base}.txt").write_text("\n".join(str(x) for x in lines), encoding="utf-8")
        try:
            if browser.page is not None:
                browser.page.screenshot(path=str(rej_dir / f"{base}.png"))
        except Exception:  # noqa: BLE001 — 스크린샷 실패가 텍스트 증거를 막으면 안 된다
            pass
        old = sorted(rej_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)[80:]
        for p in old:
            p.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 — 증거 수집 실패가 주문 흐름을 죽이면 안 된다
        log.exception("거부 증거 저장 실패")


# 대회 종료(주문 창 폐쇄)를 확인한 시각 — 이 뒤 RECHECK_SEC 동안은 주문을 아예 시도하지
# 않는다. 대회가 끝난 뒤 매 사이클 로그인·클릭을 반복하면 CPU 만 태우고 로그만 더럽힌다.
# 한 시간마다 한 번은 다시 두드려 보므로 새 대회가 열리면 저절로 복귀한다.
_CONTEST_CLOSED_AT = 0.0
CONTEST_RECHECK_SEC = 3600


def _site_executor(browser, dirty: list):
    """주문 중 예외가 나면 페이지가 어정쩡한 상태로 남을 수 있다 — 세션을 '더럽다'고 표시해
    사이클 끝에 버린다(다음 사이클이 깨끗한 페이지로 재로그인). 예외는 그대로 올려
    naver_cycle 이 종목 단위로 격리하게 한다.

    대회 종료(ContestClosedError)만은 예외다 — 페이지가 망가진 게 아니라 주문 창이 없는
    정상 상태라, 세션도 증거도 건드리지 않고 그대로 올린다."""
    def _exec(order: dict) -> dict:
        from Auto_folio.autofolio import timefolio_exec
        from Auto_folio.autofolio.errors import ContestClosedError
        global _CONTEST_CLOSED_AT
        if time.time() - _CONTEST_CLOSED_AT < CONTEST_RECHECK_SEC:
            raise ContestClosedError("타임폴리오 대회 종료 — 주문 생략(재확인 대기)")
        try:
            res = timefolio_exec.submit_with_browser(PAPER_UID, browser, order)
        except ContestClosedError:
            _CONTEST_CLOSED_AT = time.time()
            raise
        except Exception:
            dirty.append(True)
            _save_reject_evidence(browser, order, {"result": "주문 중 예외 — 트레이스백은 journald"})
            raise
        _CONTEST_CLOSED_AT = 0.0        # 주문이 접수되면 대회가 다시 열린 것
        contest_store.clear_contest_closed(PAPER_UID)
        if not res.get("accepted"):
            log.warning("[%s] %s 거부/미접수 reason=%s result=%s", order.get("ticker"),
                        order.get("side"), res.get("rejected_reason"),
                        str(res.get("result") or "")[-160:])
            _save_reject_evidence(browser, order, res)
        return res
    return _exec


def _reconcile_stale_orders(browser, account: dict, *, force_cancel: bool = False) -> list[dict]:
    """사이트에 살아 있는 미체결 주문 중 **정체된 것만** 정지시킨다 — 사이클 서두에 호출.

    두 가지를 동시에 지켜야 한다:
    - 죽은 미체결 주문을 남겨두면 같은 종목 신규 청산이 전부 TMS 오류로 거부된다(2026-07-13 사건).
    - 반대로 사이트는 대량 매도를 '과주문 후 대기'로 **분할 체결**한다(2907→2750→2102주).
      진행 중인 주문을 취소하면 체결을 되레 방해한다.
    그래서 보유 수량이 STALE_ORDER_MIN 분 동안 **전혀 줄지 않은** 주문만 정지하고, 정지한
    종목은 recent_sells 마킹을 지워 이번 사이클이 곧장(더 깊은 틱으로) 재매도하게 한다.
    """
    events: list[dict] = []
    try:
        working = browser.list_working_orders()
    except Exception:  # noqa: BLE001
        log.exception("미체결 주문 조회 실패 — 이번 사이클은 정리 없이 진행")
        return events
    held = {str(p.get("ticker") or "").zfill(6): int(p.get("qty") or 0)
            for p in ((account.get("portfolio") or {}).get("positions") or [])}
    for w in working or []:
        ticker = str(w.get("ticker") or "").zfill(6)
        prog = contest_store.track_fill_progress(PAPER_UID, ticker, held.get(ticker, 0))
        stalled = float(prog.get("stalled_min") or 0.0)
        if not force_cancel and stalled < config.AUTOFOLIO_STALE_ORDER_MIN:
            log.info("[%s] 미체결 주문 진행 중(정체 %.1f분 < %.0f분) — 유지",
                     ticker, stalled, config.AUTOFOLIO_STALE_ORDER_MIN)
            continue
        try:
            res = browser.cancel_working_orders(ticker, min_age_min=0)
        except Exception:  # noqa: BLE001 — 종목 단위 격리
            log.exception("[%s] 미체결 주문 정지 실패", ticker)
            continue
        if res.get("stopped"):
            contest_store.clear_recent_sell(PAPER_UID, ticker)
            why = "전 거래일 잔존" if force_cancel else f"{stalled:.1f}분 무진행"
            log.warning("[%s] 미체결 주문 정지(%s) — 재주문", ticker, why)
            events.append({"ok": bool(res.get("ok")), "ticker": ticker, "side": "cancel",
                           "reason": "previous_day_order" if force_cancel else "stalled_order",
                           "message": res.get("result")})
        else:
            log.warning("[%s] 정체 주문 정지 실패: %s", ticker, res.get("result"))
    return events


def _working_sell_tickers(browser) -> set[str]:
    """정리 후에도 사이트에 살아 있는(분할 체결 중인) 주문 종목 — 이번 사이클 재매도 제외."""
    try:
        return {str(w.get("ticker") or "").zfill(6) for w in browser.list_working_orders()}
    except Exception:  # noqa: BLE001
        log.exception("잔여 미체결 주문 조회 실패")
        return set()


def _event_reason(e: dict) -> str | None:
    rc = e.get("rule_check") or {}
    violations = "; ".join(v.get("message", "") for v in (rc.get("violations") or []))
    return e.get("reason") or e.get("message") or violations or None


def _event_row(e: dict) -> dict:
    # 체결 성공 이벤트는 place_order 반환형이라 종목/수량이 order 하위에 있다.
    o = e.get("order") or {}
    site = e.get("site_execution") or {}
    return {
        "ticker": e.get("ticker") or o.get("ticker"),
        "side": e.get("side") or o.get("side"),
        "ok": e.get("ok"),
        "accepted": e.get("accepted") if e.get("accepted") is not None else o.get("accepted"),
        "filled": e.get("filled") if e.get("filled") is not None else o.get("filled"),
        "pending": e.get("pending"),
        "qty": e.get("qty") or o.get("qty"),
        "price": e.get("price") or o.get("price"),
        "reason": _event_reason(e),
        "message": e.get("message"),
        "site_accepted": site.get("accepted"),
        "site_filled": site.get("filled"),
        "site_pending": site.get("pending"),
        "site_rejected_reason": site.get("rejected_reason"),
        "site_ledger_confirmed": site.get("ledger_confirmed"),
        "site_opp_tick": site.get("opp_tick"),
        "site_result": site.get("result"),
    }


def _new_site_trading_day(account: dict) -> bool:
    """직전 정상 사이트 동기화가 오늘(KST) 이전이면 전일 미체결 정리가 필요하다."""
    value = account.get("site_synced_at")
    if not value:
        return False
    try:
        from datetime import datetime
        synced = datetime.fromisoformat(str(value))
        if synced.tzinfo is None:
            synced = synced.replace(tzinfo=mt.KST)
        return synced.astimezone(mt.KST).date() < mt.now_kst().date()
    except (TypeError, ValueError):
        return False


def _run_cycle_blocking(trigger: str = "auto") -> dict:
    """반드시 _CYCLE_EXEC 의 단일 스레드에서만 호출한다(Playwright 세션 스레드 친화성)."""
    account = ensure_account()
    new_site_day = _new_site_trading_day(account)
    executor = None
    dirty: list = []
    if config.AUTOFOLIO_LIVE_ORDERS:
        if not account.get("has_site_credentials"):
            raise ValueError(
                "AUTOFOLIO_LIVE_ORDERS=1 이지만 타임폴리오 사이트 자격증명이 없습니다. "
                "AUTOFOLIO_SITE_USERNAME/AUTOFOLIO_SITE_PASSWORD 또는 TIMEFOLIO_USERNAME/TIMEFOLIO_PASSWORD를 설정하세요.")
        from Auto_folio.autofolio import timefolio_exec
        # 세션 확보 + 동기화까지는 **읽기 전용**이라 실패 시 1회 재시도해도 주문 중복 위험이 없다.
        # Chromium 렌더러가 죽으면(Target crashed / TargetClosedError) page.is_closed() 는 여전히
        # False 라 alive() 가 속아 죽은 세션을 넘긴다 → 여기서 강제 폐기 후 새 브라우저로 1회 재시도.
        # (재시도를 주문 조립까지 확대하면 안 된다 — 부분 제출된 주문이 이중 접수될 수 있다.)
        synced = None
        for attempt in (1, 2):
            try:
                browser = _site_session(allow_recycle=(trigger != "signal"))
                synced = timefolio_exec.sync_with_browser(PAPER_UID, browser)
                if not synced.get("ok"):
                    raise RuntimeError(synced.get("result") or "타임폴리오 사이트 동기화 실패")
                break
            except Exception as exc:           # 브라우저/로그인 문제면 세션을 버리고 재생성
                _drop_session()
                if attempt == 2:
                    raise
                log.warning("세션/동기화 실패(%s) — 브라우저 강제 재생성 후 1회 재시도", exc)
        executor = _site_executor(browser, dirty)
        # 정체된 미체결 주문 정지 — 이게 살아 있으면 같은 종목 청산이 전부 TMS 거부된다.
        # (동기화 직후의 계정 상태로 체결 진행 여부를 본다.)
        stale_events = _reconcile_stale_orders(
            browser, synced.get("account") or ensure_account(), force_cancel=new_site_day)
        working_sells = _working_sell_tickers(browser)
    else:
        stale_events = []
        working_sells = set()
    try:
        res = naver_cycle.run_cycle(
            PAPER_UID,
            targets=sf.buy_targets(),        # BUY 신호 기반 매수 후보(신호 없거나 엣지 게이트 아웃이면 []=매수 안함)
            sell_targets=sf.sell_targets(),  # SELL 신호 기반 청산 종목(보유 중이면 매도, 게이트 없음)
            max_buys=config.AUTOFOLIO_MAX_BUYS,
            take_profit_pct=config.AUTOFOLIO_TAKE_PROFIT_PCT,
            stop_loss_pct=config.AUTOFOLIO_STOP_LOSS_PCT,
            # 신호창을 놓친 고아 보유는 최대보유시간에 강제 청산(0/음수면 비활성).
            max_hold_min=(config.AUTOFOLIO_MAX_HOLD_MIN if config.AUTOFOLIO_MAX_HOLD_MIN > 0 else None),
            min_order_weight_pct=config.AUTOFOLIO_MIN_ORDER_WEIGHT_PCT,
            working_sells=working_sells,   # 분할 체결 진행 중인 종목은 재주문하지 않는다
            executor=executor,
        )
        if stale_events:
            res["events"] = stale_events + list(res.get("events") or [])
        # 사이트 주문원장(실제 접수/취소 기록)을 사이클 **끝에서만** 읽는다. 이 탭 클릭은 화면을
        # 전환시켜 뒤따르는 주문의 '신규 주문' 버튼을 못 찾게 만들지만(2026-07-13 사건), 다음
        # 사이클은 sync_with_browser(refresh=True) 의 page.reload 로 시작하므로 여기선 안전하다.
        # 신호 사이클은 1초라도 아껴야 해서(진입 지연=가격 악화) 주기 사이클에서만 읽는다.
        if executor is not None and trigger != "signal":
            try:
                contest_store.set_site_orders(PAPER_UID, browser.ledger_orders())
            except Exception:  # noqa: BLE001 — 원장 조회 실패가 사이클 결과를 버리면 안 된다
                log.exception("사이트 주문원장 조회 실패")
        return res
    finally:
        if dirty:
            log.warning("주문 중 예외 발생 — 세션을 버리고 다음 사이클에 재로그인")
            _drop_session()


async def run_cycle(trigger: str = "auto") -> dict:
    global _last_cycle
    async with _CYCLE_LOCK:
        t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        try:
            res = await loop.run_in_executor(_CYCLE_EXEC, _run_cycle_blocking, trigger)
            took = time.monotonic() - t0
            closed = contest_store.contest_closed(PAPER_UID) if res.get("contest_closed") else None
            _last_cycle = {
                "ts": mt.now_kst().isoformat(timespec="seconds"),
                "trigger": trigger,
                "ok": bool(res.get("ok")),
                "took_sec": round(took, 1),
                "sold": res.get("sold", 0),
                "bought": res.get("bought", 0),
                "events": [_event_row(e) for e in (res.get("events") or [])],
                "execution_mode": _execution_mode(),
                # 대회 종료면 사유를 그대로 실어 화면이 '고장'이 아니라 '종료'로 읽히게 한다.
                "contest_closed": closed or None,
            }
            if closed:
                log.info("사이클 완료 trigger=%s %.1fs — 대회 종료(%s, %s부터)",
                         trigger, took, closed.get("reason"), closed.get("since"))
            else:
                log.info("사이클 완료 trigger=%s %.1fs 매도=%s 매수=%s 이벤트=%d",
                         trigger, took, res.get("sold", 0), res.get("bought", 0),
                         len(res.get("events") or []))
        except Exception as exc:  # 사이클 실패가 웹 프로세스를 죽이면 안 된다
            took = time.monotonic() - t0
            log.exception("사이클 실패 trigger=%s %.1fs", trigger, took)
            _last_cycle = {"ts": mt.now_kst().isoformat(timespec="seconds"),
                           "trigger": trigger, "ok": False, "took_sec": round(took, 1),
                           "error": str(exc), "execution_mode": _execution_mode()}
        return _last_cycle


async def auto_loop():
    """장중 자동 사이클. 리드-랙 신호(BUY/SELL)가 뜨면 **즉시** 사이클을 돌리고,
    신호가 없어도 AUTOFOLIO_CYCLE_MIN 분마다 한 번(보유 TP/SL 감시)은 돈다.

    예전엔 고정 60초 폴링이라 신호 후 최대 1분을 그냥 흘려보냈다. 신호 자체가 1~5분 보유
    단타(TIMEOUT k=1~5m)라 그 지연이 곧 진입/청산 가격 악화였다. arctrade.service 수명과 함께 돈다.
    """
    if not config.AUTOFOLIO_ENABLED:
        return
    await asyncio.sleep(10)  # 웹 기동 직후 워밍업 여유
    last_seen = sf.latest_actionable_ts()
    last_run = 0.0
    while True:
        try:
            # 주문창(~15:30)으로 게이트한다. 수집창(~15:35)을 쓰면 폐장 후 신호마다
            # 브라우저 왕복 후 TMS 가 전건 거부해 로그만 오염된다(2026-07-29 15:31~15:34 실측).
            if mt.in_order_session():
                latest = sf.latest_actionable_ts()
                new_signal = bool(latest) and latest > last_seen
                due = (time.monotonic() - last_run) >= max(15.0, config.AUTOFOLIO_CYCLE_MIN * 60)
                if new_signal or due:
                    if new_signal:
                        last_seen = latest
                        log.info("신호 감지 %s → 즉시 사이클", latest)
                    await run_cycle("signal" if new_signal else "auto")
                    last_run = time.monotonic()
            elif _SESSION is not None:
                # 장 마감 — 브라우저를 붙들고 있을 이유가 없다(세션 스레드에서 정리).
                await asyncio.get_running_loop().run_in_executor(_CYCLE_EXEC, _drop_session)
        except Exception:  # noqa: BLE001 — 루프는 절대 죽으면 안 된다
            log.exception("autofolio auto_loop 오류")
        await asyncio.sleep(config.AUTOFOLIO_POLL_SEC)


def _alerts(account: dict) -> list[dict]:
    """대시보드 경고 배지 — 매도 미체결 반복·손절선 돌파 보유는 사람이 알아야 한다.
    (2026-07-13: -8.4% 보유가 한 시간 넘게 조용히 재시도만 하고 있었다.)"""
    out: list[dict] = []
    positions = (account.get("portfolio") or {}).get("positions") or []
    # 매도 미체결 경고(stuck_sell)는 제거 — 미체결이면 틱 에스컬레이션으로 자동 재주문하므로
    # 정상 동작이고 경고가 노이즈였다(2026-07-29 사장 지시). 손실 안전망(sl_breach)만 남긴다.
    sl = float(config.AUTOFOLIO_STOP_LOSS_PCT or 0)
    for p in positions:
        tkr = str(p.get("ticker") or "").zfill(6)
        pnl = float(p.get("pnl_pct") or 0.0)
        if sl > 0 and pnl <= -sl:
            out.append({"kind": "sl_breach", "ticker": tkr, "pnl_pct": pnl,
                        "message": f"손절선(-{sl:.1f}%) 돌파 보유 중 ({pnl:+.1f}%)"})
    return out


def summary() -> dict:
    account = ensure_account()
    raw = contest_store.get_account_raw(PAPER_UID)
    perf = contest_store.performance_snapshot(raw) if raw else {}
    return {
        "enabled": config.AUTOFOLIO_ENABLED,
        "cycle_min": config.AUTOFOLIO_CYCLE_MIN,
        "market_open": mt.in_crawl_session(),
        "live_orders": bool(config.AUTOFOLIO_LIVE_ORDERS),
        "live_ready": live_orders_ready(),
        "execution_mode": _execution_mode(account),
        "account": account,
        "performance": perf,
        "alerts": _alerts(account),
        "last_cycle": _last_cycle,
    }


def _names() -> dict:
    nm = {}
    try:
        for row in contest_store.list_security_meta():
            nm[row.get("ticker")] = row.get("name") or row.get("ticker")
    except Exception:  # noqa: BLE001
        pass
    return nm


def trades(limit: int = 50) -> list[dict]:
    raw = contest_store.get_account_raw(PAPER_UID) or {}
    rows = contest_store.consolidate_inferred_trades(list(raw.get("trades") or []))
    rows = rows[-max(1, min(int(limit or 50), 300)):]
    nm = _names()
    for r in rows:
        r["name"] = nm.get(r.get("ticker"), r.get("ticker"))
    return list(reversed(rows))


def site_orders(limit: int = 50) -> dict:
    """사이트에 실제로 접수된 주문 원장(체결/취소 포함) — 로컬 추정 체결과 교차 검증용."""
    raw = contest_store.get_account_raw(PAPER_UID) or {}
    rows = list(raw.get("site_orders") or [])[-max(1, min(int(limit or 50), 200)):]
    nm = _names()
    for r in rows:
        r["name"] = r.get("name") or nm.get(r.get("ticker"), r.get("ticker"))
    return {"orders": list(reversed(rows)), "synced_at": raw.get("site_orders_at")}

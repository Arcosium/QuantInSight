import pytest

from Auto_folio.autofolio import contest_store, naver_cycle


def _meta(code="005930", price=70000):
    return {
        "ticker": code,
        "name": "삼성전자",
        "market": "KOSPI",
        "is_common_stock": True,
        "listed_business_days": 3000,
        "avg_5d_trading_value_krw": 1_000_000_000_000,
        "market_cap_krw": 400_000_000_000_000,
        "sector": "Information Technology",
        "market_sector_weight_pct": 30.0,
        "flags": [],
        "last_price": price,
    }


def test_naver_cycle_buys_and_force_sells(tmp_path, monkeypatch):
    monkeypatch.setattr(contest_store, "_STORE_PATH", tmp_path / "contest_state.json")
    monkeypatch.setattr(contest_store, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(naver_cycle, "fetch_security_meta", lambda code, stored=None: {**_meta(code), **(stored or {})})

    contest_store.register(99, "demo", "pass123456!")
    contest_store.upsert_security_meta("005930", _meta())

    buy = naver_cycle.run_cycle(99, targets=["005930"], max_buys=1)
    assert buy["bought"] == 1
    assert buy["account"]["positions"]["005930"]["qty"] > 0

    sell = naver_cycle.run_cycle(99, targets=["005930"], max_buys=0, force_sell=True)
    assert sell["sold"] == 1
    assert "005930" not in sell["account"]["positions"]



def test_naver_cycle_executor_reject_does_not_paper_fill(tmp_path, monkeypatch):
    monkeypatch.setattr(contest_store, "_STORE_PATH", tmp_path / "contest_state.json")
    monkeypatch.setattr(contest_store, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(naver_cycle, "fetch_security_meta", lambda code, stored=None: {**_meta(code), **(stored or {})})

    contest_store.register(77, "demo", "pass123456!")
    contest_store.upsert_security_meta("005930", _meta())

    def reject_executor(order):
        return {"accepted": False, "filled": False, "result": "site rejected"}

    res = naver_cycle.run_cycle(77, targets=["005930"], max_buys=1, executor=reject_executor)
    assert res["bought"] == 0
    assert res["events"][0]["accepted"] is False
    raw = contest_store.get_account_raw(77)
    assert raw["positions"] == {}
    assert raw["trades"] == []


def test_rejected_buy_is_not_resubmitted_every_cycle(tmp_path, monkeypatch):
    """동일한 사이트 규정 거절을 신호가 살아 있는 동안 매분 반복하지 않는다."""
    monkeypatch.setattr(contest_store, "_STORE_PATH", tmp_path / "contest_state.json")
    monkeypatch.setattr(contest_store, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(naver_cycle, "fetch_security_meta", lambda code, stored=None: {**_meta(code), **(stored or {})})
    contest_store.register(771, "demo", "pass123456!")
    contest_store.upsert_security_meta("005930", _meta())
    calls = []

    def reject_executor(order):
        calls.append(order["ticker"])
        return {"accepted": False, "filled": False, "rejected_reason": "sector_full",
                "result": "섹터 편입 여유 부족"}

    first = naver_cycle.run_cycle(771, targets=["005930"], max_buys=1, executor=reject_executor)
    second = naver_cycle.run_cycle(771, targets=["005930"], max_buys=1, executor=reject_executor)
    assert first["bought"] == second["bought"] == 0
    assert calls == ["005930"]
    assert any(e.get("reason") == "reject_cooldown" for e in second["events"])


def test_naver_cycle_executor_filled_syncs_site_positions_without_paper_trade(tmp_path, monkeypatch):
    monkeypatch.setattr(contest_store, "_STORE_PATH", tmp_path / "contest_state.json")
    monkeypatch.setattr(contest_store, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(naver_cycle, "fetch_security_meta", lambda code, stored=None: {**_meta(code), **(stored or {})})

    contest_store.register(78, "demo", "pass123456!")
    contest_store.upsert_security_meta("005930", _meta())

    def fill_executor(order):
        return {
            "accepted": True,
            "filled": True,
            "summary": {
                "total_eval": 100_000_000,
                "weekly_turnover_pct": 1.2,
                "positions": [{"ticker": "005930", "qty": 10, "last_price": 70000, "avg_price": 70000}],
            },
        }

    res = naver_cycle.run_cycle(78, targets=["005930"], max_buys=1, executor=fill_executor)
    assert res["bought"] == 1
    raw = contest_store.get_account_raw(78)
    assert raw["positions"]["005930"]["qty"] == 10
    assert raw["trades"] == []
    assert raw["site_weekly_turnover_pct"] == 1.2



def _seed_holding(uid, tmp_path, monkeypatch, qty=100, price=70000):
    monkeypatch.setattr(contest_store, "_STORE_PATH", tmp_path / "contest_state.json")
    monkeypatch.setattr(contest_store, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(naver_cycle, "fetch_security_meta", lambda code, stored=None: {**_meta(code, price), **(stored or {})})
    contest_store.register(uid, "demo", "pass123456!")
    contest_store.upsert_security_meta("005930", _meta(price=price))
    contest_store.sync_site_portfolio(
        uid, total_eval=100_000_000,
        positions=[{"ticker": "005930", "qty": qty, "last_price": price, "avg_price": price}])


def test_signal_sell_submits_holding_weight_not_zero(tmp_path, monkeypatch):
    """매도 주문 비중은 '팔 비중'(=보유 비중)이어야 한다. 0 이면 사이트에서 아무 것도 안 팔린다."""
    _seed_holding(80, tmp_path, monkeypatch)
    seen = []

    def capture(order):
        seen.append(order)
        return {"accepted": True, "filled": True,
                "summary": {"total_eval": 100_000_000, "positions": [], "weekly_turnover_pct": 1.0}}

    res = naver_cycle.run_cycle(80, targets=[], sell_targets=["005930"], max_buys=0, executor=capture)
    assert res["sold"] == 1
    assert res["events"][0]["reason"] == "signal_sell"
    order = seen[0]
    assert order["side"] == "sell"
    assert order["qty"] == 100                       # 보유 전량
    assert order["weight_pct"] > 0                   # 0 이면 no-op 매도
    assert "target_weight_pct" not in order          # 폼은 목표비중이 아니라 주문비중이다
    assert order["weight_pct"] == pytest.approx(100 * 70000 / 100_000_000 * 100)


def test_order_exception_does_not_abort_cycle(tmp_path, monkeypatch):
    """주문 1건의 예외가 사이클 전체를 죽이면 안 된다 (2026-07-08 F&F 신호매도 유실 재발 방지)."""
    _seed_holding(81, tmp_path, monkeypatch)

    def boom(order):
        raise RuntimeError("타임폴리오 종목 선택 실패")

    res = naver_cycle.run_cycle(81, targets=[], sell_targets=["005930"], max_buys=0, executor=boom)
    assert res["ok"] is True                         # 사이클은 살아남는다
    assert res["sold"] == 0
    assert "주문 예외" in res["events"][0]["message"]
    assert contest_store.get_account_raw(81)["positions"]["005930"]["qty"] == 100   # 장부 불변


def test_recently_sold_blocks_duplicate_liquidation(tmp_path, monkeypatch):
    """매도 제출 후 사이트 보유목록이 아직 갱신되지 않아도 다음 사이클이 또 팔지 않는다."""
    _seed_holding(82, tmp_path, monkeypatch)
    calls = []

    def pending_sell(order):
        calls.append(order["ticker"])
        # 미체결: 보유목록이 그대로 돌아온다
        return {"accepted": True, "filled": False, "pending": True,
                "summary": {"total_eval": 100_000_000, "weekly_turnover_pct": 1.0,
                            "positions": [{"ticker": "005930", "qty": 100, "last_price": 70000, "avg_price": 70000}]}}

    first = naver_cycle.run_cycle(82, targets=[], sell_targets=["005930"], max_buys=0, executor=pending_sell)
    second = naver_cycle.run_cycle(82, targets=[], sell_targets=["005930"], max_buys=0, executor=pending_sell)
    assert first["sold"] == 1
    assert second["sold"] == 0
    assert calls == ["005930"]                       # 두 번 팔지 않았다


def test_naver_cycle_pending_accept_counts_as_order(tmp_path, monkeypatch):
    monkeypatch.setattr(contest_store, "_STORE_PATH", tmp_path / "contest_state.json")
    monkeypatch.setattr(contest_store, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(naver_cycle, "fetch_security_meta", lambda code, stored=None: {**_meta(code), **(stored or {})})

    contest_store.register(79, "demo", "pass123456!")
    contest_store.upsert_security_meta("005930", _meta("005930"))
    contest_store.upsert_security_meta("000660", _meta("000660", 200000))
    calls = []

    def pending_executor(order):
        calls.append(order["ticker"])
        return {"accepted": True, "filled": False, "pending": True, "result": "pending", "summary": {"total_eval": 100_000_000, "positions": [], "weekly_turnover_pct": 0.0}}

    res = naver_cycle.run_cycle(79, targets=["005930", "000660"], max_buys=1, executor=pending_executor)
    assert res["bought"] == 1
    assert calls == ["005930"]


# ── 2026-07-13 매도 불능 사건 회귀 테스트 ─────────────────────────────


def _backdate_first_seen(uid, ticker, minutes):
    import json
    from datetime import datetime, timedelta, timezone
    path = contest_store._STORE_PATH
    data = json.loads(path.read_text())
    pos = data["accounts"][str(uid)]["positions"][ticker]
    pos["first_seen"] = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
    path.write_text(json.dumps(data))


def test_orphan_position_sold_after_max_hold(tmp_path, monkeypatch):
    """열린 BUY 신호가 없는 보유가 max_hold_min 을 넘으면 강제 청산된다 —
    신호창(10분)을 놓친 뒤 TP/SL 전까지 팔 로직이 없던 구멍의 회귀 방지."""
    _seed_holding(83, tmp_path, monkeypatch)
    _backdate_first_seen(83, "005930", minutes=60)
    seen = []

    def capture(order):
        seen.append(order)
        return {"accepted": True, "filled": True,
                "summary": {"total_eval": 100_000_000, "positions": [], "weekly_turnover_pct": 1.0}}

    res = naver_cycle.run_cycle(83, targets=[], sell_targets=[], max_buys=0,
                                max_hold_min=15, executor=capture)
    assert res["sold"] == 1
    assert res["events"][0]["reason"] == "max_hold"
    assert seen[0]["side"] == "sell"


def test_orphan_protected_by_open_buy_signal_or_no_max_hold(tmp_path, monkeypatch):
    """열린 BUY 신호가 있거나 max_hold_min=None(QuantInSight 스웜 경로)이면 팔지 않는다."""
    _seed_holding(84, tmp_path, monkeypatch)
    _backdate_first_seen(84, "005930", minutes=60)

    def boom(order):
        raise AssertionError("매도가 나가면 안 된다")

    protected = naver_cycle.run_cycle(84, targets=["005930"], sell_targets=[], max_buys=0,
                                      max_hold_min=15, executor=boom)
    assert protected["sold"] == 0
    legacy = naver_cycle.run_cycle(84, targets=[], sell_targets=[], max_buys=0,
                                   max_hold_min=None, executor=boom)
    assert legacy["sold"] == 0


def test_sell_retry_escalates_opp_tick(tmp_path, monkeypatch):
    """미체결 재시도는 상대호가 틱을 1→3→5 로 깊여 도망가는 가격을 쫓아간다."""
    _seed_holding(85, tmp_path, monkeypatch)
    ticks = []

    def pending_sell(order):
        ticks.append(order.get("opp_tick"))
        return {"accepted": True, "filled": False, "pending": True,
                "summary": {"total_eval": 100_000_000, "weekly_turnover_pct": 1.0,
                            "positions": [{"ticker": "005930", "qty": 100, "last_price": 70000,
                                           "avg_price": 70000}]}}

    for _ in range(3):
        # recently_sold(2분) 마킹을 지워 매 사이클 재시도 상황을 재현한다(미체결 취소 후 흐름).
        naver_cycle.run_cycle(85, targets=[], sell_targets=["005930"], max_buys=0,
                              executor=pending_sell)
        contest_store.clear_recent_sell(85, "005930")
    assert ticks == [1, 3, 5]


def test_sell_attempts_cleared_when_position_gone(tmp_path, monkeypatch):
    """청산이 끝나면(보유 소멸) 매도 시도 카운터도 정리된다."""
    _seed_holding(86, tmp_path, monkeypatch)
    contest_store.bump_sell_attempt(86, "005930")
    assert contest_store.sell_attempts(86)["005930"]["n"] == 1
    contest_store.sync_site_portfolio(86, total_eval=100_000_000, positions=[])
    assert contest_store.sell_attempts(86) == {}


def test_first_seen_preserved_across_syncs(tmp_path, monkeypatch):
    """사이트 동기화가 positions 를 덮어써도 보유 시작 시각은 보존된다(고아 판정 근거)."""
    _seed_holding(87, tmp_path, monkeypatch)
    first = contest_store.get_account_raw(87)["positions"]["005930"]["first_seen"]
    contest_store.sync_site_portfolio(
        87, total_eval=100_000_000,
        positions=[{"ticker": "005930", "qty": 100, "last_price": 71000, "avg_price": 70000}])
    assert contest_store.get_account_raw(87)["positions"]["005930"]["first_seen"] == first


def test_tiny_buy_skipped_by_min_order_weight(tmp_path, monkeypatch):
    """폼 정밀도 밑 극소 매수(못 파는 잔여물 후보)는 아예 사지 않는다."""
    monkeypatch.setattr(contest_store, "_STORE_PATH", tmp_path / "contest_state.json")
    monkeypatch.setattr(contest_store, "_DATA_DIR", tmp_path)
    # 현금이 아주 작아(총평가의 0.8%) 2주=0.14% 짜리 극소 매수만 가능한 상황을 만든다.
    prices = {"000660": 200000}
    monkeypatch.setattr(naver_cycle, "fetch_security_meta",
                        lambda code, stored=None: {**_meta(code, prices.get(code, 70000)), **(stored or {})})
    contest_store.register(88, "demo", "pass123456!", initial_cash=100_000_000)
    contest_store.upsert_security_meta("005930", _meta())
    contest_store.sync_site_portfolio(
        88, total_eval=100_000_000,
        positions=[{"ticker": "000660", "qty": 496, "last_price": 200000, "avg_price": 200000}])

    def boom(order):
        raise AssertionError("극소 매수가 제출되면 안 된다")

    res = naver_cycle.run_cycle(88, targets=["005930"], max_buys=1,
                                min_order_weight_pct=0.3, executor=boom)
    assert res["bought"] == 0
    assert any("극소 주문 스킵" in (e.get("message") or "") for e in res["events"])


def test_tms_error_regex_matches_observed_popup():
    """[TMS] 오류 팝업 텍스트(2026-07-13 실측)를 명시적으로 잡는다 — 성공 오판 재발 방지."""
    from Auto_folio.autofolio.timefolio_browser import TMS_ERROR_RE
    body = "보유 잔고\n체결\n...\n[TMS] 오류\n\n전량 청산 주문 작동 시 추가 청산 불가\n\n확인\n신규 주문"
    m = TMS_ERROR_RE.search(body)
    assert m and m.group(1).strip() == "전량 청산 주문 작동 시 추가 청산 불가"


def test_working_sell_is_not_reordered(tmp_path, monkeypatch):
    """사이트에서 분할 체결 중인 매도는 재주문하지 않는다 — 재주문하면 TMS 거부만 난다."""
    _seed_holding(89, tmp_path, monkeypatch)

    def boom(order):
        raise AssertionError("진행 중인 매도를 재주문하면 안 된다")

    res = naver_cycle.run_cycle(89, targets=[], sell_targets=["005930"], max_buys=0,
                                working_sells={"005930"}, executor=boom)
    assert res["sold"] == 0
    assert res["events"][0]["reason"] == "working_order"


def test_fill_progress_resets_clock_while_filling(tmp_path, monkeypatch):
    """수량이 줄어드는 동안(분할 체결 진행)에는 정체 시계가 0으로 리셋된다."""
    _seed_holding(90, tmp_path, monkeypatch)
    first = contest_store.track_fill_progress(90, "005930", 100)
    assert first["stalled_min"] == 0.0
    same = contest_store.track_fill_progress(90, "005930", 100)   # 변화 없음 → 시계 누적 시작
    assert same["stalled_min"] >= 0.0 and same["progressing"] is False
    filling = contest_store.track_fill_progress(90, "005930", 80)  # 체결 진행 → 리셋
    assert filling["stalled_min"] == 0.0 and filling["progressing"] is True

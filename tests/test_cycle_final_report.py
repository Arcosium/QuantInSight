from main_swarm import _build_cycle_final_report


def test_final_report_never_claims_unexecuted_order_filled():
    report = _build_cycle_final_report([], {"results": []}, [])
    assert "체결 확인:" not in report
    assert "실제 주문과 체결 없이" in report


def test_final_report_separates_filled_pending_and_failed():
    rows = [
        {"ticker": "005930", "side": "buy", "qty": 1, "accepted": True, "filled": True},
        {"ticker": "AAPL", "side": "buy", "qty": 2, "accepted": True, "filled": False},
        {"ticker": "153130", "side": "sell", "qty": 7, "accepted": False,
         "filled": False, "result": "NXT 종목정보 없음"},
    ]
    report = _build_cycle_final_report(rows, {"results": []}, [])
    assert "체결 확인: 005930 매수 1주" in report
    assert "체결 확인 중: AAPL 매수 2주" in report
    assert "주문 실패: 153130 매도 7주" in report


def test_final_report_shows_partial_fill_remainder():
    # 사장 지시 2026-06-16: 부분체결은 잔여 수량을 최종 보고에 명시한다(uid2 012510 1/84 → 잔여 83).
    rows = [{"ticker": "012510", "side": "sell", "qty": 1, "order_qty": 84, "accepted": True, "filled": True}]
    report = _build_cycle_final_report(rows, {"results": []}, [])
    assert "012510" in report
    assert ("1/84" in report) or ("잔여 83" in report)


def test_final_report_full_fill_no_partial_label():
    # 전량체결(qty==order_qty)은 부분 라벨을 붙이지 않는다
    rows = [{"ticker": "005930", "side": "buy", "qty": 5, "order_qty": 5, "accepted": True, "filled": True}]
    report = _build_cycle_final_report(rows, {"results": []}, [])
    assert "부분" not in report and "잔여" not in report


def test_final_report_surfaces_risk_rejections_and_notes():
    risk = {"results": [
        {"ticker": "005380", "status": "REJECTED"},
        {"ticker": "005930", "status": "APPROVED"},
    ]}
    report = _build_cycle_final_report([], risk, ["예산 초과로 제외"])
    assert "리스크 반려: 005380" in report
    assert "예산 초과로 제외" in report


def test_final_report_surfaces_dart_vetoed_buys():
    # 사장 지시 2026-06-16(투명성): DART 2차 공시 재심에서 반려된 매수는 1차 risk_result 엔
    # APPROVED 로 남아(rejected 에 안 잡힘) final_report 에서 완전히 누락됐다(cycle 379 035250).
    # 별도 'DART 공시 반려' 라인으로 미집행 매수를 명시해야 한다.
    risk = {"results": [
        {"ticker": "064350", "side": "sell", "qty": 1, "status": "APPROVED"},
        {"ticker": "035250", "side": "buy", "qty": 2, "status": "APPROVED"},
    ]}
    exec_results = [{"ticker": "064350", "side": "sell", "qty": 1, "accepted": True, "filled": True}]
    report = _build_cycle_final_report(exec_results, risk, [], dart_vetoed={"035250"})
    assert "DART" in report
    assert "035250" in report
    # 1차는 APPROVED 였으므로 '리스크 반려' 로는 표기하지 않는다(별도 라인)
    assert "리스크 반려: 035250" not in report


def test_final_report_no_dart_line_when_no_veto():
    # DART 반려 없으면 'DART' 라인을 만들지 않는다(노이즈 방지)
    risk = {"results": [{"ticker": "005930", "side": "buy", "qty": 1, "status": "APPROVED"}]}
    report = _build_cycle_final_report([], risk, [], dart_vetoed=set())
    assert "DART 공시 반려" not in report

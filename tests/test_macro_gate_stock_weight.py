from tools.account_weight import compute_stock_weight


def test_mock_usd_deposit_not_counted_as_stock():
    # 모의: total 485M(=KR현금100M + 해외USD예수금385M), 해외 주식분 0 → 주식비중 0
    w = compute_stock_weight(total_eval=485_000_000, kr_cash=100_000_000,
                             total_eval_kr=100_000_000, overseas_stock_krw=0)
    assert w == 0.0


def test_kr_stock_counted():
    w = compute_stock_weight(total_eval=100_000_000, kr_cash=50_000_000,
                             total_eval_kr=100_000_000, overseas_stock_krw=0)
    assert abs(w - 0.5) < 1e-9


def test_overseas_stock_counted():
    # total 100M, KR현금10M, KR총평가10M(해외분 90M 중 주식 60M·USD예수금 30M) → 주식 60M/100M
    w = compute_stock_weight(total_eval=100_000_000, kr_cash=10_000_000,
                             total_eval_kr=10_000_000, overseas_stock_krw=60_000_000)
    assert abs(w - 0.6) < 1e-9


def test_zero_total_fail_open():
    assert compute_stock_weight(total_eval=0, kr_cash=0, total_eval_kr=0, overseas_stock_krw=0) == 0.0
    assert compute_stock_weight(total_eval=None, kr_cash=None, total_eval_kr=None, overseas_stock_krw=None) == 0.0


def test_stock_weight_capped_by_cash_no_phantom_100pct():
    # 사장 지시 2026-06-16: 모의계정 해외평가 환율 오염(_overseas_stock_krw 과대)으로 주식비중이
    # 100% 로 캡돼 매수가 영구 차단되던 버그(uid2 cycle381: 예수금 6,121만/총평가 1억335만 → 현금
    # 59% 인데 주식 100% 오판). 주식가치는 (총평가 − KR예수금)을 넘을 수 없다(현금은 주식 아님) →
    # 비중 ≤ 40.8%. 가드가 환율 오염값에 무관하게 물리적 상한을 보장.
    w = compute_stock_weight(total_eval=103_359_638, kr_cash=61_217_578,
                             total_eval_kr=103_359_638, overseas_stock_krw=80_818_068)
    assert 0.0 < w <= 0.41   # (103.36M − 61.22M)/103.36M ≈ 0.408


def test_cash_cap_holds_whether_total_eval_kr_polluted_or_clean():
    # total_eval_kr 가 오염(=total, 해외포함)이든 정상(KR만)이든 동일한 상한(총평가−현금) 보장
    polluted = compute_stock_weight(103_359_638, 61_217_578, 103_359_638, 80_818_068)
    clean = compute_stock_weight(103_359_638, 61_217_578, 22_541_570, 80_818_068)
    assert polluted <= 0.41
    assert clean <= 0.41

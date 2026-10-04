"""원장 허수 보유분 자동 정정 — prune_phantoms (2026-06-17).

배경: 047810 2주가 06-11 시드 이후 KIS 실보유 0주인데 원장에 남아 6일간 ledger_eval 을
310,400원 부풀렸고(매 30분 'KIS 0주 vs 원장 2주' 경고만 반복), 수동 리시드 순간 자산곡선에
-310,749원 '손실'로 찍혔다. reconcile() 은 탐지·경고만 하고 정정하지 않던 게 근본원인.

prune_phantoms: KIS 가 권위적으로 원장보다 적게 보유한 KR 포지션이 N회 연속 확인되면
KIS 기준으로 하향 정정(허수 제거)하고, 빠진 평가액(KRW)을 반환한다.
안전장치 — KR 만, 하향만, 연속확인(잔고 글리치 방어), 빈 스냅샷 무시.
"""
import infra.trade_ledger as tl


def _ledger(positions):
    return {"positions": {c: dict(p) for c, p in positions.items()},
            "cash_krw": 0.0, "cash_usd": 0.0}


def test_kr_phantom_pruned_after_confirmations(monkeypatch):
    led = _ledger({"047810": {"qty": 2, "avg_cost": 154900.0, "last_price": 155200.0, "ccy": "KRW"}})
    monkeypatch.setattr(tl, "load", lambda uid: led)
    kis = [{"code": "005930", "qty": 1}]    # 047810 부재 = KIS 0주 (정상 스냅샷)
    # 임계(3) 미만에서는 정정 안 함
    r1 = tl.prune_phantoms(1, kis, min_confirmations=3)
    assert r1["value_krw_removed"] == 0.0
    assert led["positions"].get("047810")          # 아직 보유
    tl.prune_phantoms(1, kis, min_confirmations=3)  # 2회차
    r3 = tl.prune_phantoms(1, kis, min_confirmations=3)  # 3회차 → 정정
    assert "047810" not in led["positions"]         # 허수 제거
    assert abs(r3["value_krw_removed"] - 2 * 155200.0) < 1e-6


def test_transient_glitch_resets_streak(monkeypatch):
    led = _ledger({"047810": {"qty": 2, "last_price": 155200.0, "ccy": "KRW"}})
    monkeypatch.setattr(tl, "load", lambda uid: led)
    kis_gone = [{"code": "005930", "qty": 1}]            # 047810 사라짐
    kis_back = [{"code": "005930", "qty": 1}, {"code": "047810", "qty": 2}]  # 다시 일치
    tl.prune_phantoms(1, kis_gone, min_confirmations=3)  # streak 1
    tl.prune_phantoms(1, kis_back, min_confirmations=3)  # 일치 → streak 리셋
    tl.prune_phantoms(1, kis_gone, min_confirmations=3)  # streak 1 (재시작)
    tl.prune_phantoms(1, kis_gone, min_confirmations=3)  # streak 2
    assert led["positions"].get("047810")               # 아직 정정 안 됨(연속 3회 아님)


def test_partial_overhold_reduced_to_kis(monkeypatch):
    led = _ledger({"003070": {"qty": 8, "last_price": 9790.0, "ccy": "KRW"}})
    monkeypatch.setattr(tl, "load", lambda uid: led)
    kis = [{"code": "003070", "qty": 5}]    # KIS 5 vs 원장 8 → 3주 허수
    for _ in range(3):
        r = tl.prune_phantoms(1, kis, min_confirmations=3)
    assert led["positions"]["003070"]["qty"] == 5
    assert abs(r["value_krw_removed"] - 3 * 9790.0) < 1e-6


def test_never_invents_shares_when_kis_has_more(monkeypatch):
    # KIS > 원장(누락 매수)은 prune 영역이 아님 — 손대지 않는다 (repair_from_recent_partial_orders 담당)
    led = _ledger({"003070": {"qty": 2, "last_price": 9790.0, "ccy": "KRW"}})
    monkeypatch.setattr(tl, "load", lambda uid: led)
    kis = [{"code": "003070", "qty": 5}]
    for _ in range(5):
        r = tl.prune_phantoms(1, kis, min_confirmations=3)
    assert led["positions"]["003070"]["qty"] == 2     # 불변
    assert r["value_krw_removed"] == 0.0


def test_us_phantom_not_auto_pruned(monkeypatch):
    # US 는 결제 글리치로 일시 0 빈번 → 자동 정정 대상 아님(KR 만). 수동 리시드/overseas_fills 권위.
    led = _ledger({"NVDA": {"qty": 1, "last_price": 207.0, "ccy": "USD"},
                   "AAPL": {"qty": 2, "last_price": 200.0, "ccy": "USD"}})
    monkeypatch.setattr(tl, "load", lambda uid: led)
    kis = [{"code": "AAPL", "qty": 2}]   # US 스냅샷 정상(AAPL 존재)인데도 NVDA 자동정정 안 함
    for _ in range(5):
        r = tl.prune_phantoms(1, kis, min_confirmations=3)
    assert led["positions"].get("NVDA")               # US 는 보존
    assert r["value_krw_removed"] == 0.0


def test_empty_holdings_is_noop(monkeypatch):
    led = _ledger({"047810": {"qty": 2, "last_price": 155200.0, "ccy": "KRW"}})
    monkeypatch.setattr(tl, "load", lambda uid: led)
    for _ in range(5):
        r = tl.prune_phantoms(1, [], min_confirmations=3)   # 빈 스냅샷(잔고 일시결손) → 무시
    assert led["positions"].get("047810")
    assert r["value_krw_removed"] == 0.0

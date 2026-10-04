"""원장 누락 보유분 자동 채택 — adopt_orphans (2026-06-19, defense-in-depth).

prune_phantoms(하향)의 대칭(상향). 매도 이중계상 등으로 원장이 KIS 아래로 떨어져 고착되는 걸
막아 원장 qty 를 항상 KIS 로 수렴시킨다. KIS 가 권위적으로 원장보다 많이 보유한 KR 포지션이
N회 연속 확인되면 KIS 기준으로 상향 채택(누락분 KIS 평단 기준 복원)하고, 채택 평가액(KRW)을 반환.
안전장치 — KR 만, 상향만, 연속확인(글리치 방어), 빈 스냅샷 무시.
"""
import infra.trade_ledger as tl


def _ledger(positions):
    return {"positions": {c: dict(p) for c, p in positions.items()},
            "cash_krw": 0.0, "cash_usd": 0.0}


def test_kr_orphan_adopted_after_confirmations(monkeypatch):
    # 원장 부재 · KIS 65 → 3회 연속 확인 후 채택 (161890 고착 케이스)
    led = _ledger({})
    monkeypatch.setattr(tl, "load", lambda uid: led)
    kis = [{"code": "161890", "qty": 65, "avg_price": 86000.0}]
    r1 = tl.adopt_orphans(1, kis, min_confirmations=3)
    assert r1["value_krw_added"] == 0.0
    assert "161890" not in led["positions"]              # 아직 채택 안 함
    tl.adopt_orphans(1, kis, min_confirmations=3)         # 2회차
    r3 = tl.adopt_orphans(1, kis, min_confirmations=3)    # 3회차 → 채택
    assert led["positions"]["161890"]["qty"] == 65
    assert abs(r3["value_krw_added"] - 65 * 86000.0) < 1e-6
    assert abs(led["positions"]["161890"]["avg_cost"] - 86000.0) < 1e-6


def test_partial_underhold_raised_to_kis(monkeypatch):
    # 원장 36 · KIS 90 → 54주 상향 채택 (부분 매도 이중계상 잔재)
    led = _ledger({"161890": {"qty": 36, "avg_cost": 86700.0, "last_price": 86700.0, "ccy": "KRW"}})
    monkeypatch.setattr(tl, "load", lambda uid: led)
    kis = [{"code": "161890", "qty": 90, "avg_price": 86700.0}]
    for _ in range(3):
        r = tl.adopt_orphans(1, kis, min_confirmations=3)
    assert led["positions"]["161890"]["qty"] == 90
    assert abs(r["value_krw_added"] - 54 * 86700.0) < 1e-6


def test_transient_glitch_resets_streak(monkeypatch):
    led = _ledger({"161890": {"qty": 36, "last_price": 86700.0, "ccy": "KRW"}})
    monkeypatch.setattr(tl, "load", lambda uid: led)
    kis_gap = [{"code": "161890", "qty": 90, "avg_price": 86700.0}]
    kis_match = [{"code": "161890", "qty": 36, "avg_price": 86700.0}]
    tl.adopt_orphans(1, kis_gap, min_confirmations=3)    # streak 1
    tl.adopt_orphans(1, kis_match, min_confirmations=3)  # 일치(36=36) → streak 리셋
    tl.adopt_orphans(1, kis_gap, min_confirmations=3)    # streak 1 (재시작)
    tl.adopt_orphans(1, kis_gap, min_confirmations=3)    # streak 2
    assert led["positions"]["161890"]["qty"] == 36       # 아직 채택 안 됨(연속 3회 아님)


def test_never_removes_when_ledger_has_more(monkeypatch):
    # 원장 > KIS(허수)는 adopt 영역 아님 — 손대지 않는다 (prune_phantoms 담당)
    led = _ledger({"003070": {"qty": 8, "avg_cost": 9790.0, "last_price": 9790.0, "ccy": "KRW"}})
    monkeypatch.setattr(tl, "load", lambda uid: led)
    kis = [{"code": "003070", "qty": 5, "avg_price": 9790.0}]
    for _ in range(5):
        r = tl.adopt_orphans(1, kis, min_confirmations=3)
    assert led["positions"]["003070"]["qty"] == 8        # 불변
    assert r["value_krw_added"] == 0.0


def test_us_orphan_not_auto_adopted(monkeypatch):
    # US 는 결제 글리치로 일시 0 빈번 → 자동 채택 대상 아님(KR 만)
    led = _ledger({})
    monkeypatch.setattr(tl, "load", lambda uid: led)
    kis = [{"code": "NVDA", "qty": 3, "avg_price": 200.0}]
    for _ in range(5):
        r = tl.adopt_orphans(1, kis, min_confirmations=3)
    assert "NVDA" not in led["positions"]
    assert r["value_krw_added"] == 0.0


def test_empty_holdings_is_noop(monkeypatch):
    led = _ledger({"161890": {"qty": 36, "last_price": 86700.0, "ccy": "KRW"}})
    monkeypatch.setattr(tl, "load", lambda uid: led)
    for _ in range(5):
        r = tl.adopt_orphans(1, [], min_confirmations=3)   # 빈 스냅샷(잔고 일시결손) → 무시
    assert led["positions"]["161890"]["qty"] == 36
    assert r["value_krw_added"] == 0.0

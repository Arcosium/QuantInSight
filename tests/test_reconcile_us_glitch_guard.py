"""원장 reconcile — US 해외보유 일시 글리치 false-positive 가드 (2026-06-15).

FCX 사례: 원장 5주(KIS 체결내역상 매수만·매도 전무·실보유 5주 확인)인데, KIS 해외보유 API가
일시적으로 빈값을 줘 reconcile 이 'KIS 0 vs 원장 5' 괴리를 오보고 → 데이터품질 false alarm.
수정: 원장엔 US 포지션이 있는데 KIS 스냅샷에 US 종목이 0이면 글리치로 보고 US 괴리는 보류(KR은 신뢰).
"""
import infra.trade_ledger as tl


def test_us_holdings_glitch_suppresses_us_drift(monkeypatch):
    monkeypatch.setattr(tl, "load", lambda uid: {"positions": {
        "FCX": {"qty": 5}, "005930": {"qty": 10}}})
    kis = [{"code": "005930", "qty": 10}]    # US(FCX) 통째 누락 = 글리치 시그니처
    assert tl.reconcile(1, kis) == []        # FCX 괴리 보고 안 함


def test_real_kr_drift_still_reported(monkeypatch):
    monkeypatch.setattr(tl, "load", lambda uid: {"positions": {"005930": {"qty": 10}}})
    diffs = tl.reconcile(1, [{"code": "005930", "qty": 7}])
    assert any("005930" in d for d in diffs)


def test_real_us_drift_reported_when_us_snapshot_nonempty(monkeypatch):
    # 다른 US 보유가 있으면(글리치 아님) 개별 US 괴리는 실제 → 보고
    monkeypatch.setattr(tl, "load", lambda uid: {"positions": {"FCX": {"qty": 5}, "AAPL": {"qty": 2}}})
    kis = [{"code": "AAPL", "qty": 2}]       # AAPL 존재 = US 스냅샷 정상 → FCX 진짜 괴리
    assert any("FCX" in d for d in tl.reconcile(1, kis))


def test_no_drift_when_matched(monkeypatch):
    monkeypatch.setattr(tl, "load", lambda uid: {"positions": {"FCX": {"qty": 5}}})
    assert tl.reconcile(1, [{"code": "FCX", "qty": 5}]) == []

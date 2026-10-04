"""원장 권위 실현손익·기대값 (버그 E + F3 ops 피드백 토대, 2026-06-18).

trade_log 는 부분체결 재방출로 매도가 과다기록돼 실현손익이 부풀려진다(326030: 원장 228주 vs
trade_log 453주). 원장 fills 는 멱등(_is_duplicate_fill)이라 권위 소스다. apply_fill 이 매도 시
'권위 realized'(평단 기준·비용반영)를 fill 에 기록하고, realized_stats 가 KR/US 기대값을 집계해
운용지원실장이 고회전 수익성을 보고 튜닝한다.
"""
import json
import infra.trade_ledger as tl


def _seed(tmp_path, uid, positions):
    d = tmp_path / str(uid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "ledger.json").write_text(json.dumps({
        "version": 1, "seeded_at": "2026-06-18 09:00:00", "seed_source": "test",
        "cash_krw": 0, "cash_usd": 0, "positions": positions, "fills": [], "degraded_fills": 0}),
        encoding="utf-8")


def test_kr_sell_records_realized(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "_DATA_DIR", tmp_path)
    _seed(tmp_path, 1, {"005930": {"qty": 10, "avg_cost": 100.0, "ccy": "KRW", "last_price": 100.0}})
    tl.apply_fill(1, ticker="005930", side="sell", qty=4, price=110.0, ccy="KRW")
    sell = [f for f in tl.load(1)["fills"] if f["side"] == "sell"][-1]
    assert abs(sell["realized"] - (110 - 100) * 4) < 1e-6   # KR 수수료 0 → +40


def test_us_sell_realized_net_of_fee(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "_DATA_DIR", tmp_path)
    _seed(tmp_path, 1, {"AAA": {"qty": 5, "avg_cost": 100.0, "ccy": "USD", "last_price": 100.0}})
    tl.apply_fill(1, ticker="AAA", side="sell", qty=2, price=110.0, ccy="USD")
    sell = [f for f in tl.load(1)["fills"] if f["side"] == "sell"][-1]
    # fee = 0.003*110*2 = 0.66 ; realized = (110-100)*2 - 0.66 = 19.34
    assert abs(sell["realized"] - 19.34) < 1e-6


def test_realized_stats_aggregates_kr_us(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "_DATA_DIR", tmp_path)
    _seed(tmp_path, 1, {"005930": {"qty": 10, "avg_cost": 100.0, "ccy": "KRW", "last_price": 100.0},
                        "AAA": {"qty": 5, "avg_cost": 100.0, "ccy": "USD", "last_price": 100.0}})
    tl.apply_fill(1, ticker="005930", side="sell", qty=4, price=110.0, ccy="KRW")  # +40 KRW (win)
    tl.apply_fill(1, ticker="005930", side="sell", qty=2, price=90.0, ccy="KRW")   # -20 KRW (loss)
    tl.apply_fill(1, ticker="AAA", side="sell", qty=2, price=110.0, ccy="USD")     # +19.34 USD (win)
    st = tl.realized_stats(1, fx=1500.0)
    assert st["sell_count"] == 3
    assert st["win_count"] == 2
    assert abs(st["win_rate"] - (2 / 3 * 100.0)) < 0.1
    # 총 실현 KRW = 40 - 20 + 19.34*1500
    assert abs(st["total_realized_krw"] - (40 - 20 + 19.34 * 1500)) < 1.0
    # 기대값(평균/거래) = 총실현 / 매도수
    assert abs(st["expectancy_krw"] - st["total_realized_krw"] / 3) < 1.0


def test_realized_stats_empty_is_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "_DATA_DIR", tmp_path)
    _seed(tmp_path, 1, {})
    st = tl.realized_stats(1, fx=1500.0)
    assert st["sell_count"] == 0 and st["total_realized_krw"] == 0.0

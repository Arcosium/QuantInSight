"""모의계정은 ledger_eval(원장 평가)을 곡선/KPI에 쓰지 않고 KIS total_eval 을 쓴다 — 사장 지시 2026-06-16.

배경(근본원인): 모의서버는 해외(US) TR 을 구조적 미지원이라 환율을 garbage(exrt 223.85, 실환율 1,500대의 1/6.7)로
준다. 모의 US 매수는 이 garbage 시세/환율 세계에서 체결돼 avg_cost(예: CAT $933.93)가 그 스케일로 기록되는데,
원장 M2M(mark_to_market)은 동일 USD가에 '실환율(~1,500)' 을 곱한다 → US 53,524 USD 가 1,200만원(취득)이 아니라
8천만원으로 평가돼 약 6.7배 phantom 이득. ledger_eval 이 곡선 1순위라 모의 수익률이 +80%대로 날조됐다
(uid2 hh0908: total_eval 102.6M=+1.7% vs ledger_eval 183M=+80%). 모의는 ledger_eval 을 기록하지 않는다.

실계정(uid1)은 실환율로 취득·평가가 일치하므로 영향 없다(is_mock=False → 기존대로 ledger_eval 기록).
"""
import json

from main_swarm import _equity_points, record_equity


def test_record_equity_drops_ledger_eval_for_mock(tmp_path):
    """is_mock=True 면 ledger_eval 이 양수라도 포인트에 기록하지 않는다(KIS total_eval 만 사용)."""
    ep = tmp_path / "mock.json"
    record_equity(ep, {"ok": True, "total_eval": 102_595_008, "cash": 15_768_678, "pnl_ratio": 0.0},
                  "poll", ledger_eval=183_315_314.0, is_mock=True)
    pt = json.loads(ep.read_text(encoding="utf-8"))[-1]
    assert "ledger_eval" not in pt          # 모의: 부풀린 원장값 배제
    assert pt["total_eval"] == 102_595_008  # KIS 총평가는 그대로


def test_record_equity_keeps_ledger_eval_for_real(tmp_path):
    """실계정(is_mock=False, 기본값)은 기존대로 ledger_eval 을 기록한다(회귀 방지)."""
    ep = tmp_path / "real.json"
    record_equity(ep, {"ok": True, "total_eval": 4_000_000, "cash": 1_000_000, "pnl_ratio": 0.0},
                  "poll", ledger_eval=6_900_000.0)
    assert json.loads(ep.read_text(encoding="utf-8"))[-1]["ledger_eval"] == 6_900_000.0


def test_mock_curve_uses_total_eval_not_inflated_ledger():
    """모의 곡선: ledger_eval 미기록 → _equity_points 가 total_eval 시리즈를 쓴다(+1.7%, +80% 아님)."""
    raw = [
        {"ts": "2026-05-26 15:16:05", "total_eval": 100_780_211, "src": "poll"},
        {"ts": "2026-06-16 15:06:11", "total_eval": 102_491_858, "src": "poll"},
    ]
    pts = _equity_points(raw)
    vals = [v for _, v, _ in pts]
    assert vals == [100_780_211, 102_491_858]
    ret = (vals[-1] / vals[0] - 1) * 100
    assert 1.0 < ret < 2.5      # 정상 ~+1.7% (부풀린 +80% 아님)

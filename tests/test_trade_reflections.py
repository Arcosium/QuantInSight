"""매매 복기 저장소(infra/trade_reflections.py) — 사장 지시 2026-07-31 (TradingAgents 복기 루프 이식).

요구 동작:
  - record_pending: 진입 스냅샷을 pending 으로 기록, 같은 종목 재매수 시 기존 pending 교체.
  - compute_outcome: 원장 매도 체결(수량가중)로 수익률·벤치마크 알파·보유일 계산.
    **매도 체결이 없으면 None** — KIS 결제 과도기 잔고 글리치가 '전량 매도'로 오인돼도
    복기가 오발동하지 않는 핵심 가드.
  - past_context: 같은 종목(전문) + 교차 종목(복기만) 프롬프트 주입 텍스트. 없으면 빈 문자열.
  - resolved 캡: pending 은 항상 보존.
"""
import tempfile
from pathlib import Path
import pytest

import infra.trade_reflections as tr


@pytest.fixture
def tmp_data(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        from infra import user_paths
        monkeypatch.setattr(user_paths, "_DATA_DIR", Path(d))
        yield Path(d)


def _entry(**over):
    base = {"name": "한국공항", "entry_ts": "2026-07-25 10:00:00", "entry_price": 58000.0,
            "ccy": "KRW", "qty": 3, "quant_score": 7, "committee_stance": "매수",
            "committee_confidence": 0.7, "entry_reason": "골든크로스·외인 순매수",
            "target_price": 62000.0, "stop_price": 55000.0, "planned_hold_hours": 72,
            "bench_name": "코스피", "bench_entry": 3000.0}
    base.update(over)
    return base


def test_record_pending_and_rebuy_replaces(tmp_data):
    tr.record_pending(1, "005430", _entry())
    tr.record_pending(1, "005430", _entry(entry_price=59000.0))
    data = tr._load(1)
    assert len(data) == 1 and data[0]["entry_price"] == 59000.0
    assert data[0]["market"] == "KR" and data[0]["status"] == "pending"
    assert tr.pending_codes(1) == ["005430"]


def test_compute_outcome_no_sell_fills_returns_none(tmp_data):
    # 매도 체결 0건(잔고 글리치로 보유만 빈 경우) → 확정 불가 = None
    e = {"code": "005430", **_entry()}
    assert tr.compute_outcome(e, [], 3010.0, "2026-07-28 15:00:00") is None
    buy_only = [{"ticker": "005430", "side": "buy", "qty": 3, "price": 58000.0,
                 "ts": "2026-07-25 10:00:00"}]
    assert tr.compute_outcome(e, buy_only, 3010.0, "2026-07-28 15:00:00") is None


def test_compute_outcome_weighted_exit_and_alpha(tmp_data):
    e = {"code": "005430", **_entry()}
    fills = [
        {"ticker": "005430", "side": "sell", "qty": 2, "price": 60000.0, "ts": "2026-07-28 10:00:00"},
        {"ticker": "005430", "side": "sell", "qty": 1, "price": 57000.0, "ts": "2026-07-28 11:00:00"},
        {"ticker": "000660", "side": "sell", "qty": 5, "price": 999.0, "ts": "2026-07-28 12:00:00"},
        # entry_ts 이전 매도(직전 매매 잔재)는 제외
        {"ticker": "005430", "side": "sell", "qty": 9, "price": 1.0, "ts": "2026-07-01 09:00:00"},
    ]
    oc = tr.compute_outcome(e, fills, 3030.0, "2026-07-28 15:00:00")
    assert oc is not None
    exp_exit = (60000.0 * 2 + 57000.0 * 1) / 3
    assert abs(oc["exit_price"] - exp_exit) < 1e-6
    exp_raw = (exp_exit / 58000.0 - 1) * 100
    assert abs(oc["raw_ret_pct"] - round(exp_raw, 2)) < 1e-9
    assert abs(oc["bench_ret_pct"] - 1.0) < 1e-9          # 3000 → 3030 = +1.00%
    assert abs(oc["alpha_pct"] - round(exp_raw - 1.0, 2)) < 0.011
    assert oc["holding_days"] == 3.0                       # 07-25 10시 → 07-28 10~11시 ≈ 3.0일
    assert oc["exit_ts"] == "2026-07-28 11:00:00"


def test_compute_outcome_bench_missing_alpha_none(tmp_data):
    e = {"code": "005430", **_entry(bench_entry=None)}
    fills = [{"ticker": "005430", "side": "sell", "qty": 3, "price": 60000.0,
              "ts": "2026-07-28 10:00:00"}]
    oc = tr.compute_outcome(e, fills, None, "2026-07-28 15:00:00")
    assert oc is not None and oc["alpha_pct"] is None and oc["bench_ret_pct"] is None


def test_past_context_same_and_cross(tmp_data):
    tr.record_pending(1, "005430", _entry())
    tr.record_pending(1, "024110", _entry(name="기업은행", entry_reason="저평가 은행주"))
    # 수동 확정(LLM 없이) — resolve_position 의 저장 로직과 동일한 형태
    data = tr._load(1)
    for e in data:
        e.update({"status": "resolved", "raw_ret_pct": -3.9, "alpha_pct": -4.5,
                  "bench_ret_pct": 0.6, "holding_days": 3.1,
                  "exit_ts": "2026-07-28 15:00:00", "exit_price": 55700.0,
                  "reflection": "모멘텀 논거가 이틀 만에 깨졌다. 다음엔 수급 반전 확인 후 진입."})
    tr._save(1, data)

    ctx = tr.past_context(1, "005430")
    assert "이 종목의 과거 매매 복기" in ctx and "005430" in ctx
    assert "진입 사유" in ctx and "복기:" in ctx
    assert "타 종목" in ctx and "024110" in ctx
    # 교차 종목엔 진입 사유 없이 복기만
    cross_part = ctx.split("타 종목")[1]
    assert "진입 사유" not in cross_part

    # 교차 전용(PASS 2 주입) — 종목 미지정
    ctx2 = tr.past_context(1, None, n_same=0, n_cross=3)
    assert "005430" in ctx2 and "024110" in ctx2 and "이 종목의" not in ctx2

    # 복기 없음 → 빈 문자열
    assert tr.past_context(2, "005930") == ""


def test_resolved_cap_keeps_pending(tmp_data, monkeypatch):
    monkeypatch.setattr(tr, "_MAX_RESOLVED", 5)
    data = [{"code": f"{i:06d}", "status": "resolved", "entry_ts": f"2026-07-{i + 1:02d} 09:00:00"}
            for i in range(8)]
    data.append({"code": "005430", "status": "pending", "entry_ts": "2026-07-30 09:00:00"})
    tr._save(1, data)
    out = tr._load(1)
    assert sum(1 for e in out if e["status"] == "resolved") == 5
    assert [e["code"] for e in out if e["status"] == "resolved"] == \
        [f"{i:06d}" for i in range(3, 8)]                  # 오래된 것부터 삭제
    assert any(e["status"] == "pending" for e in out)


def test_fallback_reflection_mentions_numbers(tmp_data):
    e = _entry()
    oc = {"raw_ret_pct": -3.9, "alpha_pct": -4.5, "exit_ts": "x", "exit_price": 55700.0,
          "holding_days": 3.1, "bench_ret_pct": 0.6}
    txt = tr._fallback_reflection(e, oc)
    assert "-3.90%" in txt and "-4.50%p" in txt

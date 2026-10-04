"""사장 지시 2026-06-04 ④: 에이전트 예측 구조화 적재(전향적). uid 분리·결측 컬럼 허용."""
import importlib


def _fresh_store(tmp_path, monkeypatch):
    import infra.scorecard_store as ss
    importlib.reload(ss)
    monkeypatch.setattr(ss, "DB_PATH", tmp_path / "sc.db")
    ss._conn = None  # 강제 재연결
    return ss


def test_record_and_list_roundtrip(tmp_path, monkeypatch):
    ss = _fresh_store(tmp_path, monkeypatch)
    ss.record_signal({"uid": 1, "cycle_started_at": "2026-06-04 10:00:00", "ts": "2026-06-04 10:00:01",
                      "code": "005930", "name": "삼성전자", "news_sentiment": 0.8, "quant_score": 7,
                      "det_breakdown": {"S_quant": 6.2}})
    rows = ss.list_signals(uid=1)
    assert len(rows) == 1
    assert rows[0]["code"] == "005930" and rows[0]["quant_score"] == 7
    assert rows[0]["det_breakdown"]["S_quant"] == 6.2  # JSON 파싱돼 반환


def test_uid_isolation(tmp_path, monkeypatch):
    ss = _fresh_store(tmp_path, monkeypatch)
    ss.record_signal({"uid": 1, "cycle_started_at": "t", "ts": "t", "code": "A"})
    ss.record_signal({"uid": 2, "cycle_started_at": "t", "ts": "t", "code": "B"})
    assert {r["code"] for r in ss.list_signals(uid=1)} == {"A"}
    assert {r["code"] for r in ss.list_signals(uid=2)} == {"B"}


def test_missing_optional_columns_ok(tmp_path, monkeypatch):
    ss = _fresh_store(tmp_path, monkeypatch)
    rid = ss.record_signal({"uid": 1, "cycle_started_at": "t", "ts": "t", "code": "A"})
    assert rid is not None
    r = ss.list_signals(uid=1)[0]
    assert r["news_sentiment"] is None and r["quant_score"] is None


def test_fundamental_columns_roundtrip(tmp_path, monkeypatch):
    ss = _fresh_store(tmp_path, monkeypatch)
    ss.record_signal({
        "uid": 1, "cycle_started_at": "t", "ts": "t", "code": "005930",
        "fundamental_verdict": "QUALITY_VETO",
        "business_quality_score": 2.5,
        "moat_score": 4.0,
        "management_score": 3.0,
        "valuation_margin_score": 5.0,
        "thesis_invalidators": ["high:횡령"],
    })
    r = ss.list_signals(uid=1)[0]
    assert r["fundamental_verdict"] == "QUALITY_VETO"
    assert r["business_quality_score"] == 2.5
    assert r["thesis_invalidators"] == ["high:횡령"]

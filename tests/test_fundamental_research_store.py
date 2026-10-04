import importlib


def _fresh_store(tmp_path, monkeypatch):
    import infra.fundamental_research_store as fs
    importlib.reload(fs)
    monkeypatch.setattr(fs, "DB_PATH", tmp_path / "fundamental.db")
    fs._conn = None
    return fs


def test_record_latest_and_list(tmp_path, monkeypatch):
    fs = _fresh_store(tmp_path, monkeypatch)
    rid = fs.record_snapshot({
        "uid": 1, "cycle_started_at": "t", "ts": "t1", "code": "005930",
        "name": "삼성전자", "source": "ai_berkshire_advisory", "verdict": "WATCH",
        "business_quality_score": 6.5, "valuation_margin_score": 5.0,
        "moat_score": 6.0, "management_score": 6.0,
        "thesis_invalidators": ["medium:전환사채"],
        "financial_checks": [{"name": "market_cap", "state": "SKIPPED"}],
        "memo": "주의 신호",
    })
    assert rid is not None
    latest = fs.latest(1, "005930")
    assert latest["verdict"] == "WATCH"
    assert latest["thesis_invalidators"] == ["medium:전환사채"]
    assert len(fs.list_snapshots(uid=1)) == 1


def test_uid_isolation(tmp_path, monkeypatch):
    fs = _fresh_store(tmp_path, monkeypatch)
    fs.record_snapshot({"uid": 1, "ts": "t", "code": "A", "verdict": "WATCH"})
    fs.record_snapshot({"uid": 2, "ts": "t", "code": "A", "verdict": "QUALITY_VETO"})
    assert fs.latest(1, "A")["verdict"] == "WATCH"
    assert fs.latest(2, "A")["verdict"] == "QUALITY_VETO"

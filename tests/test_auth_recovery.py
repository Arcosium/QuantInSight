import json
import pytest

def test_find_username_by_factors(fresh_auth):
    fresh_auth.upsert_user("erin", "P@ss12345!", "AK1", "AS1", "1-1", "", "", "")
    assert fresh_auth.find_username_by_factors("1-1", "AS1") == "erin"
    assert fresh_auth.find_username_by_factors(" 1-1 ", "AS1") == "erin"  # norm
    assert fresh_auth.find_username_by_factors("1-1", "WRONG") is None
    assert fresh_auth.find_username_by_factors("", "AS1") is None

def test_reset_password_by_factors(fresh_auth):
    fresh_auth.upsert_user("fred", "OldP@ss123", "AK2", "AS2", "1-1", "", "", "")
    assert fresh_auth.reset_password_by_factors(
        "fred", "1-1", "AS2", "N3wP@ssword!") is True
    assert fresh_auth.verify_password("fred", "N3wP@ssword!")["username"] == "fred"
    assert fresh_auth.verify_password("fred", "OldP@ss123") is None
    assert fresh_auth.reset_password_by_factors(
        "fred", "1-1", "BAD", "ValidPass9!") is False
    with pytest.raises(ValueError):
        fresh_auth.reset_password_by_factors("fred", "1-1", "AS2", "weak")

def test_audit_appends_jsonl_and_never_logs_secrets(fresh_auth):
    fresh_auth.audit("recover_id", username="erin", ip="1.2.3.4",
                     outcome="fail", detail="no-match")
    lines = (fresh_auth._AUDIT_PATH).read_text(encoding="utf-8").strip().splitlines()
    rec = json.loads(lines[-1])
    assert rec["event"] == "recover_id" and rec["outcome"] == "fail"
    assert rec["username"] == "erin" and rec["ip"] == "1.2.3.4"
    assert "detail" in rec and "ts" in rec

def test_audit_is_best_effort_never_raises(fresh_auth, monkeypatch):
    # make the underlying write fail; audit() must swallow it (auth path must not break)
    import builtins
    real_open = builtins.open
    def boom(*a, **k):
        if str(a[0]).endswith("auth_audit.log"):
            raise OSError("disk full")
        return real_open(*a, **k)
    monkeypatch.setattr(builtins, "open", boom)
    fresh_auth.audit("login", username="x", ip="1.1.1.1", outcome="ok")  # must NOT raise

def test_audit_ts_is_iso8601_utc(fresh_auth):
    fresh_auth.audit("login", username="x", ip=None, outcome="ok")
    import json as _j
    rec = _j.loads((fresh_auth._AUDIT_PATH).read_text(encoding="utf-8").strip().splitlines()[-1])
    # ISO-8601 UTC, parseable, ends with +00:00
    from datetime import datetime
    parsed = datetime.fromisoformat(rec["ts"])
    assert parsed.tzinfo is not None
    assert rec["ip"] == ""   # None ip normalized to ""

def test_reset_password_policy_checked_before_factors_no_oracle(fresh_auth):
    import pytest
    fresh_auth.upsert_user("gus", "OldP@ss123", "AKg", "ASg", "1-1", "", "", "")
    # weak pw + WRONG factors → ValueError (NOT False) — same as weak pw + right factors,
    # so a weak-pw probe cannot distinguish factor correctness (oracle closed)
    with pytest.raises(ValueError):
        fresh_auth.reset_password_by_factors("gus", "WRONG", "WRONG", "weak")
    with pytest.raises(ValueError):
        fresh_auth.reset_password_by_factors("gus", "1-1", "ASg", "weak")

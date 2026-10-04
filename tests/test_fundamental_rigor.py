from tools.fundamental_rigor import (
    assess_fundamental_research,
    extract_risk_flags,
    format_research_for_prompt,
    validate_market_cap,
)


def test_market_cap_validation_ok_and_warn():
    ok = validate_market_cap("1000", "100", "100000")
    assert ok["state"] == "OK"
    warn = validate_market_cap("1000", "100", "130000", tolerance_pct=5)
    assert warn["state"] == "WARN"
    assert warn["diff_pct"] > 5


def test_extract_risk_flags_korean_disclosures():
    flags = extract_risk_flags("관리종목 지정 가능성 및 전환사채 발행")
    assert "관리종목" in flags["high"]
    assert "전환사채" in flags["medium"]


def test_assess_quality_veto_on_high_risk():
    r = assess_fundamental_research(
        "005930", "삼성전자",
        dart_text="횡령 및 상장폐지심사 관련 공시",
        financial_text="✅ 재무상태표 검증: 정상",
    )
    assert r["verdict"] == "QUALITY_VETO"
    assert r["business_quality_score"] < 7
    assert any(x.startswith("high:") for x in r["thesis_invalidators"])


def test_assess_advisory_only_and_prompt_format():
    r = assess_fundamental_research("AAPL", "Apple", dart_text="특이 공시 없음")
    assert r["verdict"] == "ADVISORY_ONLY"
    txt = format_research_for_prompt(r)
    assert "ADVISORY_ONLY" in txt
    assert "thesis invalidators" in txt

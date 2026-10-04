"""매수 전 희석(dilution) 게이트 — DART 공시에서 CB/유상증자 등 감지 (2026-06-15 ROI#4).

운용전략 직관 대신 결정론 안전장치: 매수 직전 해당 기업의 최근 전환사채·유상증자·신주인수권
공시를 감지해 희석 리스크가 높으면 매수 보류(또는 경고). MCP 자율성 없이 결정론 게이트로.
"""
from tools.dilution import detect_dilution


def test_convertible_bond_is_high():
    r = detect_dilution("[주요사항보고서] 전환사채(CB) 발행 결정 — 300억원")
    assert r["dilutive"] is True
    assert r["severity"] == "high"
    assert any("전환사채" in k for k in r["kinds"])


def test_rights_offering_is_high():
    r = detect_dilution("유상증자 결정 (제3자배정) 공시")
    assert r["dilutive"] is True
    assert r["severity"] == "high"


def test_stock_option_is_medium():
    r = detect_dilution("주식매수선택권 부여 결정")
    assert r["dilutive"] is True
    assert r["severity"] == "medium"


def test_clean_disclosure_not_dilutive():
    r = detect_dilution("분기보고서 제출 · 영업실적 잠정 공시")
    assert r["dilutive"] is False
    assert r["severity"] == "none"


def test_empty_text_not_dilutive():
    assert detect_dilution("")["dilutive"] is False
    assert detect_dilution(None)["dilutive"] is False

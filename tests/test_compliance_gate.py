"""컴플라이언스실장 반려 게이트 — 결정론 판정 회귀 테스트 (사장 지시 2026-07-22).

구 수탁자책임실(policy_filter) 폐지로 사라졌던 반려 기능의 부활분이다.
핵심 불변식:
  · 매수 후보만 반려한다 (매도 경로는 이 게이트를 타지 않는다 — 호출부 계약).
  · ESG 판정은 종목명·섹터로만 — 뉴스 본문 한 줄로 무관 종목이 차단되면 안 된다.
  · 어떤 입력에도 예외를 던지지 않는다 (호출부 fail-open 의 전제).
"""
import pytest

from agents import compliance


def _screen(**kw):
    return compliance.screen(kw.pop("code", "005930"), kw.pop("name", "삼성전자"), **kw)


# ── 통과 ─────────────────────────────────────────────────────────────────────
def test_clean_stock_passes():
    r = _screen(sector="반도체", dart_text="정기주주총회 소집공고")
    assert not r.rejected and r.verdict == compliance.PASS_V
    assert all(c.passed for c in r.checks)


def test_missing_context_passes():
    """섹터·공시가 비어 있어도 통과 — 정보 부족을 반려 사유로 쓰지 않는다."""
    assert not _screen(sector="", dart_text="").rejected


# ── ESG 블랙리스트 ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name,sector,expect", [
    ("강원랜드", "카지노·레저", "도박·카지노"),
    ("KT&G", "담배", "담배"),
    ("한국항공우주", "방위산업", "무기·방위산업"),
])
def test_esg_blacklist_rejects(name, sector, expect):
    r = _screen(code="036570", name=name, sector=sector)
    assert r.rejected
    assert any(expect in reason for reason in r.reasons)


def test_esg_does_not_scan_news_text():
    """'카지노 관련주 반사이익' 류 뉴스 한 줄로 무관 종목이 차단되면 안 된다.
    screen() 은 뉴스를 아예 인자로 받지 않으며, 공시 텍스트도 ESG 판정에 쓰지 않는다."""
    r = _screen(name="삼성전자", sector="반도체",
                dart_text="[투자참고] 카지노·담배 업종 반사이익 기대 — 단순 언급 공시")
    assert not r.rejected, r.reasons


# ── 시장경보 ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("kw", ["투자주의환기", "투자경고종목", "투자위험종목",
                                "단기과열종목", "불공정거래"])
def test_market_alert_rejects(kw):
    r = _screen(sector="기타", dart_text=f"거래소 공시 — {kw} 지정 안내")
    assert r.rejected and any(kw in x for x in r.reasons)


def test_market_alert_evidence_is_recorded():
    """반려는 반드시 근거 원문을 남긴다(감사 가능성)."""
    r = _screen(sector="기타", dart_text="공시1 정기보고서\n공시2 투자경고종목 지정")
    failed = [c for c in r.checks if not c.passed]
    assert failed and "투자경고종목" in failed[0].evidence


# ── 내부 제외목록 ─────────────────────────────────────────────────────────────
def test_policy_exclude_list_rejects():
    r = compliance.screen("123456", "제외종목", exclude_codes=["123456"])
    assert r.rejected and "내부 투자정책" in r.reasons[0]


def test_exclude_list_does_not_affect_others():
    assert not compliance.screen("005930", "삼성전자", exclude_codes=["123456"]).rejected


# ── 정직성 가드레일 공표 ───────────────────────────────────────────────────────
def test_honesty_note_none_when_clean():
    assert compliance.honesty_note([], 0.0) is None


def test_honesty_note_summarizes_and_truncates():
    note = compliance.honesty_note([f"강등 {i}" for i in range(5)], 0.42)
    assert "5건" in note and "42%" in note and "외 2건" in note


# ── 견고성 (fail-open 전제) ───────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "", 0, 12345])
def test_never_raises_on_odd_input(bad):
    r = compliance.screen(bad, bad, sector=bad or "", dart_text=bad or "")
    assert r.verdict in (compliance.PASS_V, compliance.REJECT)


def test_result_is_json_serializable():
    import json
    json.dumps(_screen(sector="반도체").to_dict(), ensure_ascii=False)

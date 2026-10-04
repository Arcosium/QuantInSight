"""후보 이름↔코드 검증 — LLM이 종목명에 틀린 6자리 코드를 붙이면 차단/재해석 (2026-06-15).

버그: 오케스트레이터가 '벡트(290650)'처럼 이름+코드를 함께 주면 _resolve_candidate_codes 가
코드만 신뢰 → 벡트(데이터센터) 의도였으나 290650(엘앤씨바이오)을 매수. 수정: 코드의 실제
종목명이 주어진 이름과 불일치하면 이름으로 재해석(코드 환각 무시), 못 찾으면 후보 제외.
"""
from main_swarm import _resolve_candidate_codes

# 코드→실제이름 / 이름→코드 (테스트용 미니 마스터)
_NAMES = {"290650": "엘앤씨바이오", "365900": "벡트", "005930": "삼성전자"}
def _name_check(code): return _NAMES.get(code)
def _resolver(name):
    for c, n in _NAMES.items():
        if n == name:
            return c
    return None

def _alloc(tokens): return f"후보종목: {tokens}"


def test_matching_name_code_uses_code():
    out = _resolve_candidate_codes(_alloc("엘앤씨바이오(290650)"),
                                   resolver=_resolver, name_check=_name_check)
    assert out == ["290650"]


def test_mismatch_reresolves_by_name():
    # 벡트(데이터센터)에 엘앤씨바이오 코드 → 이름으로 재해석해 365900(벡트)
    out = _resolve_candidate_codes(_alloc("벡트(290650)"),
                                   resolver=_resolver, name_check=_name_check)
    assert out == ["365900"]
    assert "290650" not in out


def test_mismatch_unresolvable_name_drops_candidate():
    # 이름으로도 못 찾으면 환각 코드를 신뢰하지 않고 제외
    out = _resolve_candidate_codes(_alloc("없는회사(290650)"),
                                   resolver=lambda n: None, name_check=_name_check)
    assert out == []


def test_code_only_still_trusted():
    out = _resolve_candidate_codes(_alloc("290650"),
                                   resolver=_resolver, name_check=_name_check)
    assert out == ["290650"]


def test_offline_no_namecheck_trusts_code():
    out = _resolve_candidate_codes(_alloc("벡트(290650)"), resolver=None, name_check=None)
    assert out == ["290650"]


# ── US 티커 대칭 검증 (2026-08-04) ────────────────────────────────────────
# 실사고: 뉴스의 'Rocket Lab(RVLV)'·'SpaceX(SPACE)' 를 그대로 후보로 채택. RVLV 는 실존
# 티커(Revolve Group)라 하류 일봉 게이트에도 안 걸려 엉뚱한 회사를 매수할 수 있었다.
_US = {"RVLV": "Revolve Group, Inc.", "AAPL": "Apple Inc."}
def _us_check(sym): return _US.get(sym.upper(), "")


def test_us_ticker_name_mismatch_dropped():
    out = _resolve_candidate_codes(_alloc("Rocket Lab(RVLV), Apple(AAPL)"),
                                   session="US_TRADING", name_check=_us_check)
    assert out == ["AAPL"]


def test_us_ticker_unknown_passes_fail_open():
    # 조회 실패(없는 티커/네트워크 장애)는 통과 — 하류 일봉 게이트가 거른다
    out = _resolve_candidate_codes(_alloc("SpaceX(SPACE)"),
                                   session="US_TRADING", name_check=_us_check)
    assert out == ["SPACE"]


def test_us_korean_name_skips_check():
    # 한글 이름은 영문 회사명과 대조 불가 → 검증 생략(오탐 방지)
    out = _resolve_candidate_codes(_alloc("애플(AAPL)"),
                                   session="US_TRADING", name_check=_us_check)
    assert out == ["AAPL"]

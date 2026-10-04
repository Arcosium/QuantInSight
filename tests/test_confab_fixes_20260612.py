"""2026-06-12 환각·정형사실 미주입 버그 수정 (사장 지시).

검증된 버그 5계열에 대한 회귀 테스트:
  A-#1  슬리브 매니저에 현재 보유·가격 미주입 → 137610 '미보유' 환각·$가격 날조
  A-#3/4/6  ops 가 자산타입·실제사유 없이 'market_open' 으로 인과 날조
  B-#2  thesis LLM 에 종목 무관 뉴스(센티 리포트 앞부분=삼성) 주입 → 사유 오염
  C-#5  코드↔이름 이중소스 → thesis 에 권위 종목명 미반영
  D-#7  모의계정 체결이 '실매매 체결확인' 으로 표기
"""
import pytest

from main_swarm import (
    _fill_badge,
    _extract_code_news,
    _build_sleeve_prompt,
)
from infra.asset_sleeves import format_sleeve_holdings_block, COMMODITY_SLEEVE
from infra import ops_support_worker as ops
from tools.market_data import canonical_name


# ── D-#7: 모의/실거래 체결 배지 ────────────────────────────────────────────────
def test_fill_badge_real_account():
    assert _fill_badge(True, is_mock=False) == "✅ 실매매 체결확인"


def test_fill_badge_mock_account():
    # 모의계정이면 '실매매' 가 아니라 '모의' 로 표기돼야 한다(운영자 오인 방지).
    assert _fill_badge(True, is_mock=True) == "✅ 모의 체결확인"


def test_fill_badge_failure_regardless_of_account():
    assert _fill_badge(False, is_mock=True) == "⚠ 주문 실패"
    assert _fill_badge(False, is_mock=False) == "⚠ 주문 실패"


# ── B-#2: thesis 용 종목별 뉴스 추출 ──────────────────────────────────────────
_SENTIMENT_REPORT = """[뉴스 감성 분석 리포트]
- 📰 삼성전자(005930)·SK하이닉스(000660) (KR, 반도체): 감성 +0.90
  - [긍정] 프리장서 9%대 급등, 애플 시리 AI 메모리 반도체 수요 확대 기대.
- 📰 대한항공(003490) (KR, 항공·화물): 감성 +0.75
  - [긍정] KB증권 화물운임 상승이 고유가 상쇄, 목표가 36,000원 유지.
- 📰 KB금융(105560) (KR, 금융): 감성 +0.85
  - [긍정] 하나증권 2분기 최대 실적 전망.
"""


def test_extract_code_news_returns_only_target_stock():
    seg = _extract_code_news(_SENTIMENT_REPORT, "003490", "대한항공")
    assert "대한항공" in seg and "화물운임" in seg
    # 오염 방지: 삼성전자 사유(프리장 급등·AI 반도체)가 섞이면 안 된다.
    assert "프리장" not in seg
    assert "반도체 수요" not in seg


def test_extract_code_news_empty_when_absent():
    assert _extract_code_news(_SENTIMENT_REPORT, "999999", "없는종목") == ""


# ── A-#1: 슬리브 보유 블록 (보유·가격·평가손익을 LLM 에 정형 주입) ──────────────
def test_format_sleeve_holdings_block_lists_holdings_in_krw():
    block = format_sleeve_holdings_block([
        {"code": "137610", "name": "TIGER 농산물선물Enhanced(H)", "qty": 650,
         "cur_price": 5050.0, "pnl_pct": -1.1},
        {"code": "261220", "name": "KODEX WTI원유선물(H)", "qty": 149,
         "cur_price": 24645.0, "pnl_pct": -6.1},
    ])
    # 보유 종목·수량이 명시돼야 '미보유' 환각이 차단된다.
    assert "137610" in block and "650" in block
    assert "261220" in block
    # 가격은 KRW 로 — '$' 표기(달러 환각) 금지.
    assert "$" not in block
    assert "5,050" in block or "5050" in block


def test_format_sleeve_holdings_block_empty_is_explicit():
    block = format_sleeve_holdings_block([])
    assert block  # 빈 문자열이 아니라 '보유 없음' 을 명시해야 LLM 이 추측하지 않는다.
    assert "없" in block


def test_build_sleeve_prompt_includes_holdings_block():
    prompt = _build_sleeve_prompt(
        COMMODITY_SLEEVE, "매크로", "뉴스", "  - 132030 금 (na·gold)",
        "권고 10% / 현재 19.9%", thesis_reminder="",
        holdings_txt="- KODEX WTI원유선물(H)(261220): 보유 149주",
    )
    assert "261220" in prompt
    assert "149주" in prompt


# ── A-#3: ops 자산 식별(코드→이름·자산군) — 132030(금) ≠ 261220(원유) ──────────
def test_asset_identity_lines_distinguishes_gold_from_oil():
    lines = ops._asset_identity_lines(["132030", "261220"])
    assert "132030" in lines and "261220" in lines
    # 132030 은 금(골드)이지 원유가 아니다 — ops 가 '같은 원유 ETF' 로 뭉뚱그리면 안 된다.
    assert "골드" in lines or "금" in lines


# ── A-#4/#8: 실행 미발생 사유 정형화 (risk_approved bool 의 모호성 해소) ────────
def test_execution_outcome_note_no_orders():
    note = ops._execution_outcome_note({"orders_planned": [], "risk_approved": False})
    assert "주문 없음" in note or "주문이 없" in note


def test_execution_outcome_note_risk_rejected():
    note = ops._execution_outcome_note({
        "orders_planned": [{"ticker": "033780"}], "risk_approved": False})
    # 반려는 risk_report 의 실제 사유를 인용하라고 유도; '비개장' 단정 금지.
    assert "리스크" in note
    assert "비개장" in note  # 금지 안내 문구로 등장


# ── A-#4/#6: ops 프롬프트에 반(反)환각 가드 + 자산식별 포함 ────────────────────
def _ops_ctx():
    return {
        "target_cycle": {
            "started_at": "2026-06-12 12:00:00", "ended_at": "2026-06-12 12:12:00",
            "session": "KR_TRADING", "market_open": False,
            "candidate_codes": ["033780", "005930"], "target_codes": ["033780"],
            "sell_directives": {}, "orders_planned": [{"ticker": "033780"}],
            "orders_executed": [], "risk_approved": False,
            "bp_cash": 3903669, "bp_total_eval": 6627446, "bp_pnl_ratio": 0.0,
            "risk_report": "[033780] KT&G — 담배 ESG 블랙리스트 위반 → 반려",
        },
        "recent_cycles": [], "recent_errors_skips": [],
    }


def test_ops_prompt_has_anticonfab_guard():
    p = ops.build_prompt(_ops_ctx())
    assert "추측" in p          # '사유를 추측하지 말라' 가드
    assert "실제 사유" in p


def test_ops_prompt_warns_not_to_invent_market_closed():
    p = ops.build_prompt(_ops_ctx())
    # KR_TRADING 인데 market_open=False 라고 '비개장' 으로 단정하지 말라는 안내가 있어야 한다.
    assert "비개장" in p


def test_ops_prompt_identifies_sleeve_codes_from_orders_not_just_candidates():
    # #3 의 문제 코드(132030 금·261220 원유)는 candidate 가 아니라 슬리브 *주문*에 등장한다.
    # candidate 가 주식뿐이어도 거래된 슬리브 코드의 자산식별이 떠야 '같은 원유 ETF' 환각이 막힌다.
    ctx = {
        "target_cycle": {
            "session": "KR_TRADING", "market_open": False,
            "candidate_codes": ["005930"], "target_codes": [],
            "sell_directives": {}, "orders_planned": [],
            "orders_executed": [
                {"ticker": "261220", "side": "sell", "qty": 24, "filled": True},
                {"ticker": "132030", "side": "buy", "qty": 16, "filled": True},
            ],
            "risk_approved": True,
        },
        "recent_cycles": [], "recent_errors_skips": [],
    }
    p = ops.build_prompt(ctx)
    # 가드의 예시 문구가 아니라 *실제 자산식별 블록*이 풀 정본명으로 해소해야 한다.
    assert "[자산 식별" in p
    assert "KODEX 골드선물" in p      # 132030 정본명(풀에서만 옴) — '같은 원유' 환각 차단
    assert "KODEX WTI원유선물" in p   # 261220 정본명


# ── C-#5: 권위 종목명 리졸브 (코드 기준 정본 이름) ────────────────────────────
def test_canonical_name_prefers_authoritative_resolver():
    # 뉴스 LLM 이 'PKC' 라 불러도, 코드 241520 의 정본 이름을 써야 한다.
    name = canonical_name("241520", resolver=lambda c: "DSC인베스트먼트")
    assert name == "DSC인베스트먼트"


def test_canonical_name_falls_back_when_resolver_blank():
    name = canonical_name("241520", resolver=lambda c: "", fallback="PKC")
    assert name == "PKC"

"""2026-07-29 사장 보고 4건에 대한 회귀 테스트.

1) 261220 매도 오류 — 주식 매도 트랙과 슬리브 매도 트랙이 같은 종목에 매도 주문을 각각
   만들어 보유의 2배를 팔았다(45주 → 22+22 → 1주 잔여 → 이후 매 사이클 '잔고내역이
   없습니다'). 합류점 불변식: 한 사이클 · 한 종목 · 매도 1건.
2) 모의계정 수익률 ±40% 튐 — 모의 해외 순평가는 USD 부채 때문에 **음수**이고, 그 부채는
   해외 보유목록이 비어도 남는다. 종전엔 주식분이 0 이면 산출을 포기해 부채가 사라졌다.
3) LLM 응답의 `**` 강조 마커.
"""
import json

from agents.committee import _SELL_STANCES, KEEP, HALF, ALL
from infra.kis_broker import KISBroker
from infra.local_llm_client import strip_markdown_emphasis
from main_swarm import dedupe_sell_orders


# ── 1) 매도 주문 중복 제거 ────────────────────────────────────────────────────
def test_dedupe_sell_keeps_one_order_per_ticker():
    orders = [
        # 주식 매도 트랙(_assemble_sell_orders)
        {"ticker": "261220", "side": "sell", "qty": 22, "market": "KR",
         "reason": "KODEX WTI원유선물(H) 사후관리실장 매도 판단 — 절반"},
        {"ticker": "148070", "side": "buy", "qty": 99, "reason": "채권운용실장 자산배분"},
        # 슬리브 매도 트랙(_build_sleeve_sell_orders) — 같은 지시로 만든 중복
        {"ticker": "261220", "side": "sell", "qty": 22, "reason": "원자재운용실장 자산배분"},
    ]
    out, dropped = dedupe_sell_orders(orders)
    sells = [o for o in out if o["side"] == "sell"]
    assert len(sells) == 1, "같은 종목 매도가 2건 나가면 보유 초과 매도 → 잔여 1주 고착"
    assert dropped == [("261220", 22)]
    assert len(out) == 2 and any(o["side"] == "buy" for o in out)   # 매수는 건드리지 않는다


def test_dedupe_sell_keeps_larger_qty():
    """전량 지시가 부분 지시에 잘리면 안 된다 — 큰 수량을 남긴다."""
    out, dropped = dedupe_sell_orders([
        {"ticker": "132030", "side": "sell", "qty": 5, "reason": "절반"},
        {"ticker": "132030", "side": "sell", "qty": 174, "reason": "전량"},
    ])
    assert [o["qty"] for o in out] == [174]
    assert dropped == [("132030", 5)]


def test_dedupe_sell_passthrough_when_unique():
    orders = [{"ticker": "005930", "side": "sell", "qty": 3},
              {"ticker": "000660", "side": "sell", "qty": 1}]
    out, dropped = dedupe_sell_orders(list(orders))
    assert out == orders and dropped == []


# ── 2) 모의 해외 순평가(USD 부채) ─────────────────────────────────────────────
def _broker_with_ledger(tmp_path, cash_usd):
    b = object.__new__(KISBroker)
    b._token_path = tmp_path / "kis_token.json"
    (tmp_path / "ledger.json").write_text(
        json.dumps({"cash_usd": cash_usd, "positions": {}}), encoding="utf-8")
    return b


def test_selfcalc_keeps_usd_debt_when_no_us_holdings(tmp_path, monkeypatch):
    """US 보유 조회가 비어도 원장 USD 부채는 순평가에 남아야 한다.

    이 부채(-24,911 USD ≈ -29M원)가 폴마다 들어갔다 빠지면서 uid2 총평가가
    71M ↔ 100M 로 튀었다(수익률 +40% 유령 급등)."""
    monkeypatch.setattr("infra.kis_broker._real_usdkrw", lambda: 1466.2)
    b = _broker_with_ledger(tmp_path, -24911.22)
    out = b._overseas_selfcalc_krw([])           # 해외 보유 없음(조회 실패/전량매도)
    assert out["ok"] is True
    assert out["stock_krw"] == 0.0
    assert out["krw"] < 0
    assert round(out["krw"]) == round(-24911.22 * 1466.2)


def test_selfcalc_nets_stock_against_debt(tmp_path, monkeypatch):
    monkeypatch.setattr("infra.kis_broker._real_usdkrw", lambda: 1000.0)
    b = _broker_with_ledger(tmp_path, -100.0)
    out = b._overseas_selfcalc_krw([{"ccy": "USD", "qty": 10, "cur_price": 20.0}])
    assert out["stock_krw"] == 200_000.0
    assert out["krw"] == 100_000.0               # 200,000 − 100,000 부채


def test_selfcalc_gives_up_without_fx(tmp_path, monkeypatch):
    monkeypatch.setattr("infra.kis_broker._real_usdkrw", lambda: 0.0)
    b = _broker_with_ledger(tmp_path, -100.0)
    assert b._overseas_selfcalc_krw([])["ok"] is False   # 환율 모르면 지어내지 않는다


# ── 3) `**` 강조 마커 제거 ────────────────────────────────────────────────────
def test_strip_markdown_emphasis():
    assert strip_markdown_emphasis("**농심(004370)**: +2.35% 수익") == "농심(004370): +2.35% 수익"
    assert strip_markdown_emphasis("굵직한 **40건** 선별") == "굵직한 40건 선별"
    assert strip_markdown_emphasis("잔여 ** 마커") == "잔여  마커"     # 짝 안 맞아도 제거
    assert strip_markdown_emphasis("정상 문장") == "정상 문장"          # 본문은 불변
    assert strip_markdown_emphasis(None) == ""


# ── 매도 심의 스탠스는 그대로 매도지시가 된다 ─────────────────────────────────
def test_sell_stances_are_valid_sell_directives():
    from main_swarm import _SELL_HOLD_WORDS, _SELL_HALF_WORDS, _SELL_ALL_WORDS
    assert _SELL_STANCES == (KEEP, HALF, ALL)
    assert KEEP in _SELL_HOLD_WORDS and HALF in _SELL_HALF_WORDS and ALL in _SELL_ALL_WORDS

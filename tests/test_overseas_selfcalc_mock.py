"""사장 지시 2026-07-22: 모의계정 US 평가를 제대로 잡는다.

배경 — 모의서버는 기준환율(frst_bltn_exrt)을 218.31 로, 그걸로 환산한 총평가(frcr_evlu_tota)를
357M 로 주는 garbage 지만, **보유수량과 현재가는 실제와 일치한다**(2026-07-22 uid2 실측:
IEF 97주 @ $93.31 — 우리 시세와 동일). 종전엔 _sanitize_overseas 가 통째로 0 처리해
모의계정의 US 평가·손익이 자산곡선에 영영 안 잡혔다.

핵심 불변식:
  - 자체산출 순액 = 해외주식 평가(실환율) + USD 예수금(실환율)
  - USD 예수금은 우리 체결 원장의 cash_usd — 모의 KIS 는 이걸 0 으로 오보한다.
    통합증거금으로 산 US 물량은 KRW 차감 없이 **USD 부채**로 남으므로(7/21 IEF 매수 전후
    D+2 예수금 불변), 주식분만 더하면 그만큼 가짜 이득이 된다.
  - 따라서 순액은 **음수일 수 있고**, 그래도 총평가에 반영돼야 한다.
  - 실계좌(정상 환율)는 이 경로를 타지 않는다.
"""
import json

from infra.kis_broker import KISBroker, _sanitize_overseas


def _broker(tmp_path, ledger: dict | None):
    b = object.__new__(KISBroker)
    b._token_path = tmp_path / "kis_token.json"
    if ledger is not None:
        (tmp_path / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
    return b


HOLDINGS = [
    {"code": "IEF", "qty": 97, "cur_price": 93.31, "ccy": "USD"},
    {"code": "005930", "qty": 10, "cur_price": 80000, "ccy": "KRW"},
]


def test_selfcalc_nets_usd_liability(tmp_path, monkeypatch):
    """주식 13.36M + USD부채 -9,086×1475.6 = 순 -51K. 주식분만 더하면 13.4M 가짜 이득."""
    monkeypatch.setattr("infra.kis_broker._real_usdkrw", lambda: 1475.6)
    b = _broker(tmp_path, {"cash_usd": -9086.00649, "positions": {}})
    r = b._overseas_selfcalc_krw(HOLDINGS)
    assert r["ok"] is True
    assert round(r["stock_krw"]) == round(97 * 93.31 * 1475.6)
    assert r["krw"] < 0                                  # 부채가 주식평가를 넘는다
    assert abs(r["krw"] - (r["stock_krw"] - 9086.00649 * 1475.6)) < 1.0
    assert abs(r["krw"]) < 100_000                       # 순액은 소액(가격변동+수수료)


def test_selfcalc_ignores_mock_garbage_exrt(tmp_path, monkeypatch):
    """모의 기준환율(218.31)이 아니라 실환율을 쓴다 — 자체산출의 존재 이유."""
    monkeypatch.setattr("infra.kis_broker._real_usdkrw", lambda: 1475.6)
    b = _broker(tmp_path, {"cash_usd": 0.0, "positions": {}})
    r = b._overseas_selfcalc_krw(HOLDINGS)
    assert r["fx"] == 1475.6
    assert r["stock_krw"] > 13_000_000                   # 218.31 로 환산하면 ~2M 에 그친다


def test_selfcalc_gives_up_without_ledger(tmp_path, monkeypatch):
    """원장이 없으면 USD 부채를 알 수 없다 → 산출 포기(종전 0 처리 유지). 지어내지 않는다."""
    monkeypatch.setattr("infra.kis_broker._real_usdkrw", lambda: 1475.6)
    b = _broker(tmp_path, None)
    assert b._overseas_selfcalc_krw(HOLDINGS)["ok"] is False


def test_selfcalc_gives_up_without_fx(tmp_path, monkeypatch):
    """실환율을 모르면 산출 포기 — 모의 환율로 대충 채우지 않는다."""
    monkeypatch.setattr("infra.kis_broker._real_usdkrw", lambda: 0.0)
    b = _broker(tmp_path, {"cash_usd": -9086.0, "positions": {}})
    assert b._overseas_selfcalc_krw(HOLDINGS)["ok"] is False


def test_selfcalc_keeps_debt_without_us_holdings(tmp_path, monkeypatch):
    """US 보유목록이 비어도 원장 USD 부채는 순평가에 남는다 (사장 보고 2026-07-29).

    종전엔 `usd_stock<=0` 이면 산출을 포기해, 해외 조회가 빈 폴마다 부채가 통째로 사라지고
    총평가가 71M↔100M(±40%) 로 튀었다. 주식분 0 + 부채만 = 음수 순평가가 정답이다."""
    monkeypatch.setattr("infra.kis_broker._real_usdkrw", lambda: 1475.6)
    b = _broker(tmp_path, {"cash_usd": -9086.0, "positions": {}})
    kr_only = [h for h in HOLDINGS if h["ccy"] == "KRW"]
    out = b._overseas_selfcalc_krw(kr_only)
    assert out["ok"] is True
    assert out["stock_krw"] == 0.0
    assert round(out["krw"]) == round(-9086.0 * 1475.6)


def test_selfcalc_carries_ledger_stock_with_debt_on_empty_live_holdings(tmp_path, monkeypatch):
    """일시 빈 잔고에서는 원장 주식과 USD 부채를 함께 이월해 매수원금 급락을 막는다."""
    monkeypatch.setattr("infra.kis_broker._real_usdkrw", lambda: 1000.0)
    b = _broker(tmp_path, {
        "cash_usd": -100.0,
        "positions": {
            "IEF": {"qty": 5, "last_price": 19.0, "avg_cost": 20.0, "ccy": "USD"},
            "005930": {"qty": 1, "last_price": 80000, "ccy": "KRW"},
        },
    })
    out = b._overseas_selfcalc_krw([])
    assert out["stock_krw"] == 95_000.0
    assert out["krw"] == -5_000.0                  # 부채만 잡으면 -100,000원으로 급락


def test_mock_flat_reconcile_requires_three_authoritative_empty_reads(tmp_path):
    """단발 빈 응답은 보존하고, 세 거래소 정상 빈 잔고 3회에만 허수 US 포지션을 청산한다."""
    b = _broker(tmp_path, {
        "cash_usd": -100.0,
        "positions": {"IEF": {"qty": 5, "last_price": 19.0, "ccy": "USD"}},
    })
    b.is_mock = True

    assert b._reconcile_mock_us_flat(True) is False
    assert b._reconcile_mock_us_flat(False) is False       # 조회 실패가 끼면 연속성 리셋
    assert b._reconcile_mock_us_flat(True) is False
    assert b._reconcile_mock_us_flat(True) is False
    assert b._reconcile_mock_us_flat(True) is True

    led = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert "IEF" not in led["positions"]
    assert abs(led["cash_usd"] - (-100.0 + 5 * 19.0 * 0.997)) < 1e-9
    assert led["reconciliations"][-1]["kind"] == "mock_us_authoritative_flat"


def test_sanitize_still_zeroes_garbage_first(tmp_path):
    """자체산출은 _sanitize_overseas 가 0 으로 만든 뒤에만 개입한다(실계좌 경로 불변)."""
    assert _sanitize_overseas(357_415_556, 13_355_758, 218.31) == (0.0, 0.0)
    assert _sanitize_overseas(609_275, 741_872, 1475.6) == (609_275, 741_872)

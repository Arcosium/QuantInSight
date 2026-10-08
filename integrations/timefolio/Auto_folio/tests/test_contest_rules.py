from Auto_folio.autofolio.contest_rules import validate_order


def _account(cash=100_000_000, positions=None):
    return {"cash": cash, "positions": positions or {}}


def _meta(ticker="123456", **overrides):
    base = {
        "ticker": ticker,
        "name": "테스트",
        "market": "KOSPI",
        "is_common_stock": True,
        "listed_business_days": 20,
        "avg_5d_trading_value_krw": 5_000_000_000,
        "market_cap_krw": 500_000_000_000,
        "sector": "Industrials",
        "market_sector_weight_pct": 10.0,
        "flags": [],
        "last_price": 10_000,
    }
    base.update(overrides)
    return base


def test_buy_rejects_missing_required_metadata():
    result = validate_order(_account(), "buy", "123456", 10, 10_000, {})
    codes = {v.code for v in result.violations}
    assert not result.ok
    assert "market" in codes
    assert "market_cap_missing" in codes
    assert "sector_missing" in codes


def test_buy_rejects_low_liquidity_new_listing_blocked_flag_and_low_cap():
    metas = {"123456": _meta(avg_5d_trading_value_krw=2_900_000_000,
                              listed_business_days=4,
                              market_cap_krw=90_000_000_000,
                              flags=["투자경고"])}
    result = validate_order(_account(), "buy", "123456", 10, 10_000, metas)
    codes = {v.code for v in result.violations}
    assert not result.ok
    assert {"avg5", "listed_days", "market_cap", "blocked_flags"} <= codes


def test_position_limit_15_percent_for_normal_stock():
    metas = {"123456": _meta()}
    result = validate_order(_account(), "buy", "123456", 2_000, 10_000, metas)
    assert not result.ok
    assert any(v.code == "position_limit" for v in result.violations)


def test_samsung_exception_allows_40_percent_position():
    metas = {"005930": _meta("005930", market_cap_krw=400_000_000_000_000, sector="IT", market_sector_weight_pct=30.0)}
    result = validate_order(_account(), "buy", "005930", 4_000, 10_000, metas)
    assert result.ok
    assert result.metrics["position_limit_pct"] == 40.0


def test_small_cap_total_limit_30_percent():
    metas = {
        "111111": _meta("111111", market_cap_krw=500_000_000_000, sector="A", market_sector_weight_pct=30.0, last_price=10_000),
        "222222": _meta("222222", market_cap_krw=500_000_000_000, sector="B", market_sector_weight_pct=30.0, last_price=10_000),
    }
    positions = {"111111": {"qty": 2_000, "avg_price": 10_000, "last_price": 10_000}}
    result = validate_order(_account(positions=positions), "buy", "222222", 2_000, 10_000, metas)
    assert not result.ok
    assert any(v.code == "small_cap_limit" for v in result.violations)

"""get_usdkrw/set_usdkrw — 라이브 USD/KRW 환율 (사장 지시 2026-05-22).

환율은 계속 변하므로 하드코딩하지 않고, 5분 지수 크롤(get_index_data)이 USDKRW 를
가져올 때마다 캐시를 갱신한다. 모든 USD↔KRW 환산(예산·리스크 사이징)이 이 값을 읽는다.
"""
import pytest

import tools.market_data as _md
from tools.market_data import get_usdkrw, set_usdkrw


@pytest.fixture(autouse=True)
def _isolate_fx_cache(tmp_path, monkeypatch):
    """실 data/usdkrw_fx.json 격리 (2026-07-22).

    set_usdkrw 는 하드코딩 경로(_FX_CACHE_PATH)에 환율을 영속한다 — 격리 없이 pytest 를
    돌리면 테스트용 가짜 환율(1512.0)이 라이브 캐시에 남고, 서버 재시작 직후 첫 사이클이
    그 값을 읽어 USD↔KRW 예산·리스크 사이징에 쓴다. 모듈 전역 _LAST_FX 도 함께 되돌려
    같은 세션의 다른 테스트로 새지 않게 한다(monkeypatch 가 원복)."""
    monkeypatch.setattr(_md, "_FX_CACHE_PATH", tmp_path / "usdkrw_fx.json")
    monkeypatch.setattr(_md, "_LAST_FX", 0.0)


def test_set_get_roundtrip():
    set_usdkrw(1515.9)
    assert get_usdkrw(1300.0) == 1515.9


def test_sanity_rejects_garbage():
    set_usdkrw(1512.0)
    set_usdkrw(0)        # 거부 (<=500)
    set_usdkrw(99999)    # 거부 (>=5000)
    set_usdkrw("x")      # 거부 (비숫자)
    assert get_usdkrw(1300.0) == 1512.0


def test_default_is_positive():
    assert get_usdkrw(1500.0) > 0

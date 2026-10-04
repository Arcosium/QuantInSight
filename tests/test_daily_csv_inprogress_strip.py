"""퀀트 일봉 로딩 — 진행 중(거래량 0) 당일 행 제거 (2026-06-15).

버그: 프리마켓 사이클에 daily CSV 에 당일(거래량 0) 행이 누적되고, 퀀트가 이를 읽어
"6/15일 거래량 0은 장 마감으로 추정"이라며 세션을 오인 + 지표가 하루 stale. 수정:
load_daily_csv 가 **말미의** 거래량 0 행(장 시작 전 placeholder)을 떼서 마지막 실거래일을
'현재'로 보게 한다. 중간 거래정지(0거래량) 행은 보존.
"""
import pandas as pd

from tools.market_data import _strip_trailing_zero_volume


def _df(volumes):
    n = len(volumes)
    return pd.DataFrame({
        "date": pd.date_range("2026-06-08", periods=n).astype(str),
        "close": [100 + i for i in range(n)],
        "volume": volumes,
    })


def test_strips_trailing_inprogress_zero_volume():
    out = _strip_trailing_zero_volume(_df([1000, 1100, 1200, 0]))
    assert len(out) == 3
    assert int(out["volume"].iloc[-1]) > 0


def test_keeps_interior_zero_volume_halt():
    out = _strip_trailing_zero_volume(_df([1000, 0, 1200, 1300]))
    assert len(out) == 4   # 중간 거래정지 행은 보존


def test_strips_multiple_trailing_zeros():
    out = _strip_trailing_zero_volume(_df([1000, 1100, 0, 0]))
    assert len(out) == 2


def test_never_returns_empty():
    out = _strip_trailing_zero_volume(_df([0, 0, 0]))
    assert len(out) >= 1   # 전부 0이어도 최소 1행 유지(fail-open)


def test_no_volume_column_is_noop():
    df = pd.DataFrame({"date": ["2026-06-08", "2026-06-09"], "close": [100, 101]})
    out = _strip_trailing_zero_volume(df)
    assert len(out) == 2

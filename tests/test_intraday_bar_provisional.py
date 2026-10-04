"""당일 일봉 고정 수정 — 미완성 당일봉 제외(읽기) + 동일날짜 upsert(쓰기) (2026-06-15).

버그: 당일 봉이 09시 첫 조회값에 고정(_append_csv 가 동일날짜 스킵) → ① 장중 부분 거래량을
전일 전체와 비교해 '98% 급감' 오독, ② 내일이면 틀린 과거봉으로 영구 고착. 수정:
  - 읽기: load_daily_csv 가 당일(미완성) 봉을 시계열에서 제외 → 완성봉만 분석(제외).
  - 쓰기: _append_csv 가 동일 날짜를 덮어써 당일 봉이 장중 갱신·종가확정(provisional).
"""
import pandas as pd

from tools.market_data import _strip_provisional_today, _append_csv


def _df(dates, vols):
    return pd.DataFrame({"date": pd.to_datetime(dates), "close": [100]*len(dates), "volume": vols})


def test_strips_today_provisional_bar():
    df = _df(["2026-06-12", "2026-06-15"], [171245, 3804])
    out = _strip_provisional_today(df, "2026-06-15")
    assert len(out) == 1
    assert str(out["date"].iloc[-1].date()) == "2026-06-12"   # 완성된 전일봉이 마지막


def test_keeps_all_when_no_today_bar():
    df = _df(["2026-06-11", "2026-06-12"], [100000, 171245])
    out = _strip_provisional_today(df, "2026-06-15")
    assert len(out) == 2


def test_never_empties_provisional():
    df = _df(["2026-06-15"], [3804])
    out = _strip_provisional_today(df, "2026-06-15")
    assert len(out) >= 1


def test_append_csv_overwrites_same_date(tmp_path):
    p = tmp_path / "daily_TEST.csv"
    cols = ["date", "close", "volume"]
    _append_csv(p, [{"date": "2026-06-12", "close": 100, "volume": 171245}], cols)
    # 같은 날짜(6/12) 갱신값 + 신규(6/15) 동시
    _append_csv(p, [{"date": "2026-06-12", "close": 102, "volume": 200000},
                    {"date": "2026-06-15", "close": 103, "volume": 3804}], cols)
    df = pd.read_csv(p)
    assert len(df) == 2                                   # 중복 없음
    row612 = df[df["date"] == "2026-06-12"].iloc[0]
    assert int(row612["volume"]) == 200000               # 덮어쓰기됨(스킵 아님)
    assert int(row612["close"]) == 102

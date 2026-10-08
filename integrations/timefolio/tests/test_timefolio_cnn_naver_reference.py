import pytest
from quant.timefolio_cnn_naver_reference import parse

HEADER=['날짜','시가','고가','저가','종가','거래량','외국인소진율']


def test_daily_reference_is_bounded_and_excludes_extra_current_fields():
    rows=parse(repr([HEADER,['20221123',100,120,90,110,1000,99]]))
    assert rows==[dict(date='20221123',open=100.,high=120.,low=90.,close=110.,volume=1000.)]
    with pytest.raises(ValueError,match='cutoff'):
        parse(repr([HEADER,['20260928',100,120,90,110,1000,99]]))


def test_duplicate_dates_and_executable_responses_are_refused():
    row=['20221123',100,120,90,110,1000,99]
    with pytest.raises(ValueError,match='Duplicate'):parse(repr([HEADER,row,row]))
    with pytest.raises((ValueError,SyntaxError)):parse("__import__('os').getcwd()")
    assert parse(repr([HEADER]))==[]

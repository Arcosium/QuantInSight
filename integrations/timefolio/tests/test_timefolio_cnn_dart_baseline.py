import pytest

from quant.timefolio_cnn_dart_baseline import normalize


def test_report_period_does_not_replace_filing_availability():
    rows=normalize([dict(rcept_no='20230307000542',se='보통주',istc_totqy='5,969,782,550',stlm_dt='2022-12-31')])
    assert rows[0]['receipt_date']=='20230307' and rows[0]['issued_shares']==5969782550
    assert not rows[0]['listed_shares_certified']


def test_missing_counts_and_post_cutoff_corrections_are_not_historical_baselines():
    result=normalize([dict(rcept_no='20261002000001',se='보통주',istc_totqy='-',stlm_dt='2021-12-31')])[0]
    assert result['issued_shares'] is None and not result['available_before_cutoff']
    with pytest.raises(ValueError):normalize([dict(rcept_no='missing')])

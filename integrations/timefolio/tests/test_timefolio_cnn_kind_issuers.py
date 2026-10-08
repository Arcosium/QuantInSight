import pytest

from quant.timefolio_cnn_kind_issuers import parse_issuer


def test_only_requested_security_fields_are_retained():
    html = '<table><tr><th>표준코드</th><td>KR7050540004</td><th>종목코드</th><td>050540</td></tr><tr><th>대표이사</th><td>unused</td><th>상장일</th><td>2005-12-12</td></tr></table>'
    result = parse_issuer(html, '05054')
    assert result['security_code'] == '050540' and result['listing_date'] == '2005-12-12'
    assert 'unused' not in str(result) and result['current_snapshot']
    assert not result['historical_identity_certified']


def test_no_ticker_is_inferred_from_popup_identifier():
    with pytest.raises(ValueError):parse_issuer('<p>missing</p>', '05054')


def test_conflicting_codes_are_rejected():
    with pytest.raises(ValueError):
        parse_issuer('<table><tr><th>종목코드</th><td>050540</td><th>종목코드</th><td>005930</td></tr></table>', '05054')

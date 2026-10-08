import pytest

from quant.timefolio_cnn_kind_history import parse_page, event_groups


def page(cells, onclick=''):
    return '<table><tbody><tr onclick="'+onclick+'">'+cells+'</tr></tbody></table>전체 1 건 : 1 /1'


def issuer():
    return '<td><img alt="코스닥"><a onclick="companysummary_open(\'10009\')">테스트회사</a><img alt="관리종목"></td>'


def test_transfer_is_not_assumed_terminal_and_popup_key_is_not_fabricated_stock_code():
    html = page('<td>1</td>'+issuer()+'<td>2023-04-19</td><td>유가증권시장 상장</td><td></td>')
    total, records = parse_page(html, 'delisting', '20220101', '20260923')
    row = records[0]
    assert total == 1 and row['market_transfer_candidate'] and not row['terminal_delisting_certified']
    assert row['issuer_popup_id'] == '10009' and row['security_code'] is None
    assert not row['current_status_icons_used']


def test_issuance_retains_negative_share_change_without_assuming_publication_date():
    html = page(issuer()+'<td>2022-12-07</td><td>변경상장</td><td>-4,149,262</td><td>5,000</td><td>주식소각</td>', "fnDetailView('20221201000123')")
    _, records = parse_page(html, 'issuance', '20220101', '20260923')
    assert records[0]['shares_signed'] == -4149262
    assert not records[0]['publication_time_verified'] and not records[0]['stock_class_verified']


def test_out_of_window_event_and_error_page_are_rejected():
    html = page('<td>1</td>'+issuer()+'<td>2026-10-01</td><td>폐지</td><td></td>')
    for source in [html, '<html>Please login</html>']:
        with pytest.raises(ValueError):parse_page(source,'delisting','20220101','20260923')


def test_identical_duplicates_are_grouped_but_conflicting_share_counts_have_no_canonical_event():
    row=dict(event_id='1',issuer_popup_id='12345',shares_signed=500,source_page=1,source_row=99)
    group=event_groups([row,{**row,'source_page':2,'source_row':1}],'issuance')[0]
    assert group['occurrences']==2 and group['canonical_record']['shares_signed']==500
    assert not group['ledger_application_certified']
    conflict=event_groups([row,{**row,'shares_signed':600}],'issuance')[0]
    assert conflict['conflicting'] and conflict['canonical_record'] is None

import pytest
from quant.timefolio_cnn_kind_details import parse_detail, match_share_chains


def page(**values):
    return '<table>'+''.join(f'<tr><th>{k}</th><td>{v}</td></tr>' for k, v in values.items())+'</table>'


def test_detail_preserves_signed_changes_without_inventing_share_class():
    event = dict(event_id='20260409000095', issuer_popup_id='00593', company_name='삼성전자',
                 date='20260414', listing_type='변경상장', shares_signed=-73359314,
                 detail_method='searchStockIssueDetail')
    html = page(회사명='삼성전자', 상장일='2026-04-14', 상장방식='변경상장',
                발행일='2026-04-02', 발행주식수='-73,359,314', 누적발행주식수='5,846,278,608')
    result = parse_detail(html, event)
    assert result['implied_previous_issued_shares'] == 5919637922
    assert result['security_code'] is None and result['share_class'] is None
    assert not result['publication_time_verified'] and not result['issued_equals_listed_verified']
    event['shares_signed'] = -1
    with pytest.raises(ValueError, match='disagree'):
        parse_detail(html, event)


def edge(identity, before, after, day='20220102'):
    return dict(event_id=identity, issuer_popup_id='sample', listing_date=day,
                implied_previous_issued_shares=before, cumulative_issued_shares=after,
                share_change=after-before)


def test_two_classes_and_out_of_order_same_day_changes_remain_separate():
    rows = [edge('second', 110, 130), edge('preferred', 50, 40), edge('first', 100, 110)]
    result = match_share_chains(rows, {'보통주': 100, '우선주': 50})
    assert result['final_counts'] == {'보통주': 130, '우선주': 40}
    assert result['complete_arithmetic_chain']
    assert not result['historical_listed_shares_certified']


def test_ambiguous_and_disconnected_transitions_are_not_assigned_to_common():
    result = match_share_chains([edge('x', 100, 120)], {'보통주': 100, '우선주': 100})
    assert not result['applied'] and len(result['unresolved']) == 1
    result = match_share_chains([edge('x', 90, 120)], {'보통주': 100})
    assert result['final_counts']['보통주'] == 100 and not result['complete_arithmetic_chain']


def test_duplicate_events_are_rejected_instead_of_applied_twice():
    event = edge('x', 100, 110)
    with pytest.raises(ValueError, match='Deduplicate'):
        match_share_chains([event, event], {'보통주': 100})

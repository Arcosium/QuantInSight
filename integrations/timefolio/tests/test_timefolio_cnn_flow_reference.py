import csv
import json

import pytest

from quant.timefolio_cnn_flow_reference import archive, merge, parse


def old(day='20260909', institution='-12', foreign='+34'):
    return dict(date=day, inst_net=institution, foreign_net=foreign)


def test_reserved_fields_are_discarded_before_numerical_parsing():
    result = parse([old(), old('20260924', 'private-future-value', None)])
    assert result['discarded_future_rows'] == 1
    assert result['rows'] == [dict(date='20260909', institution_net_shares=-12, foreign_net_shares=34)]
    mobile = dict(bizdate='20260909', itemCode='005930', organPureBuyQuant='-1,234',
        foreignerPureBuyQuant='+2,345', closePrice='must never be copied', individualPureBuyQuant='ignored')
    parsed = parse([mobile], mobile=True, code='005930')
    assert parsed['rows'][0]['institution_net_shares'] == -1234
    assert 'closePrice' not in json.dumps(parsed) and 'must never' not in json.dumps(parsed)


def test_missing_and_conflicting_flow_cannot_become_zero():
    for value in [None, '', 'NaN', '0.5', 'inf']:
        with pytest.raises(ValueError):
            parse([old(institution=value)])
    with pytest.raises(ValueError, match='Conflicting'):
        parse([old(), old(foreign='0')])
    assert parse([old(), old()])['identical_duplicate_rows'] == 1
    assert len(parse([old('20260901'), old('20260909')])['rows']) == 2


def test_wrong_security_or_invalid_date_cannot_enter_snapshot():
    with pytest.raises(ValueError, match='another security'):
        parse([dict(bizdate='20260909', itemCode='000660')], mobile=True, code='005930')
    with pytest.raises(ValueError):
        parse([old('20260230')])


def test_archive_is_read_only_and_drops_reserved_values(tmp_path):
    path = tmp_path/'investor_005930.csv'
    with path.open('w') as file:
        writer = csv.DictWriter(file, fieldnames=['date', 'inst_net', 'foreign_net'])
        writer.writeheader()
        writer.writerows([old('2026-09-09'), old('2026-09-24', 'future', 'future')])
    before, modified = path.read_bytes(), path.stat().st_mtime_ns
    result = archive(path)
    assert path.read_bytes() == before and path.stat().st_mtime_ns == modified
    assert len(result['rows']) == 1 and result['discarded_future_rows'] == 1
    assert 'future' not in json.dumps(result['rows'])


def test_changed_overlap_is_not_silently_substituted_or_filled():
    baseline = parse([old('20260901'), old('20260909')])['rows']
    good = parse([old('20260909'), old('20260910')])['rows']
    assert merge(baseline, good)['new_dates_added'] == 1
    bad = parse([old('20260909', foreign='35'), old('20260910')])['rows']
    result = merge(baseline, bad)
    assert result['rows'] == baseline and not result['fresh_rows_accepted']
    assert result['conflicting_overlap_dates'] == ['20260909']

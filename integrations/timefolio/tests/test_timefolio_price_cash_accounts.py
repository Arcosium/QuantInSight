import hashlib
import json
from pathlib import Path

from quant import timefolio_heatmap_price_cash_accounts as price
from test_timefolio_cash_accounts import monthly


def test_parent_is_preserved_and_price_family_has_no_forecast_thresholds():
    old = Path(price.__file__).with_name('timefolio_heatmap_cash_accounts.py')
    assert hashlib.sha256(old.read_bytes()).hexdigest() == price.PARENT_SHA256
    rows = price.hypotheses(['case_a', 'case_b', 'case_c'])
    assert len(rows) == 5184 and len({r[0] for r in rows}) == 2304
    assert not any('forecast' in r[0] for r in rows)
    keys = {r[0] for r in rows}
    assert all(reference in keys for key, label, reference in rows
               if label in ['matching_untrained_pipeline', 'same_exposure_nonimage'])


def test_either_study_stops_new_price_accounts_and_first_candidate_is_preserved(tmp_path, monkeypatch):
    original = tmp_path/'original'; original.mkdir()
    output = tmp_path/'price'; output.mkdir()
    shared = original/'STOP.json'
    monkeypatch.setattr(price, 'SHARED_STOP', shared)
    assert price.can_start(output)
    price.publish_threshold(output, 'price_candidate', monthly(3.2, 1.6))
    assert not price.can_start(output)
    assert json.loads(shared.read_text())['id'] == 'price_candidate'
    second = tmp_path/'other'; second.mkdir()
    assert not price.can_start(second)
    price.publish_threshold(second, 'second_candidate', monthly(3.3, 1.7))
    assert json.loads(shared.read_text())['id'] == 'price_candidate'


def test_existing_cash_stop_prevents_any_price_account(tmp_path, monkeypatch):
    shared = tmp_path/'original_stop.json'; shared.write_text('{}')
    output = tmp_path/'price'; output.mkdir()
    monkeypatch.setattr(price, 'SHARED_STOP', shared)
    assert not price.can_start(output)

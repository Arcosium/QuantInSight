import pytest
from infra.kis_broker import KISBroker


def broker(url):
    return KISBroker({"kis_app_key": "test", "kis_app_secret": "test",
                      "kis_account_no": "00000000-01", "kis_base_url": url})


@pytest.mark.parametrize("tr", ["TTTC0802U", "TTTC0801U", "TTTT1002U", "TTTT1006U", "TTTC0803U"])
def test_live_mutations_blocked(tr):
    with pytest.raises(PermissionError):
        broker("https://openapi.koreainvestment.com:9443")._h("test", tr)


def test_mock_orders_and_readonly_market_data_allowed():
    assert broker("https://openapivts.koreainvestment.com:29443")._h("test", "TTTC0802U")["tr_id"] == "VTTC0802U"
    assert broker("https://openapi.koreainvestment.com:9443")._h("test", "FHKST03010200")


def test_mock_substring_is_not_authorization():
    with pytest.raises(PermissionError):
        broker("https://openapivts.attacker.invalid:29443")._h("test", "TTTC0802U")

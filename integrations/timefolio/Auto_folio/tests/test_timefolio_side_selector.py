from Auto_folio.autofolio.timefolio_browser import TimefolioBrowser


class _Radio:
    def __init__(self, radio_id):
        self.radio_id = radio_id
        self.checked = False
        self.check_calls = 0

    @property
    def first(self):
        return self

    def count(self):
        return 1

    def get_attribute(self, name):
        return self.radio_id if name == "id" else None

    def check(self, **kwargs):
        self.check_calls += 1
        self.checked = True

    def evaluate(self, script):
        return self.checked


class _Label:
    def __init__(self, selected, opposite):
        self.selected = selected
        self.opposite = opposite
        self.click_calls = 0

    @property
    def first(self):
        return self

    def count(self):
        return 1

    def click(self, **kwargs):
        self.click_calls += 1
        self.selected.checked = True
        self.opposite.checked = False


class _Dialog:
    def __init__(self):
        self.buy = _Radio("매수도_true")
        self.sell = _Radio("매수도_false")
        self.labels = {
            "매수도_true": _Label(self.buy, self.sell),
            "매수도_false": _Label(self.sell, self.buy),
        }

    @property
    def last(self):
        return self

    def locator(self, selector):
        if 'value="true"' in selector:
            return self.buy
        if 'value="false"' in selector:
            return self.sell
        if selector.startswith('label[for="'):
            return self.labels[selector.split('"')[1]]
        raise AssertionError(selector)

    def inner_text(self, **kwargs):
        return "매수 매도"


class _Page:
    def __init__(self):
        self.dialog = _Dialog()

    def locator(self, selector):
        assert selector == '[role="dialog"]'
        return self.dialog

    def wait_for_timeout(self, ms):
        return None


def test_sell_side_clicks_visible_bootstrap_label_and_verifies_state():
    page = _Page()
    browser = TimefolioBrowser.__new__(TimefolioBrowser)
    browser._choose_side(page, "sell")
    assert page.dialog.sell.checked is True
    assert page.dialog.buy.checked is False
    assert page.dialog.labels["매수도_false"].click_calls == 1
    assert page.dialog.sell.check_calls == 0

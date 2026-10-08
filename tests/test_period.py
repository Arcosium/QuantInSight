from datetime import date,datetime,timezone

import pytest
from autofolio import period


def test_window_rolls_every_day_and_clamps_leap_day():
    assert period.window(date(2026,10,8))==('20231008','20261007')
    assert period.window(date(2026,10,9))==('20231009','20261008')
    assert period.window(date(2024,2,29))==('20210228','20240228')
    assert period.window(date(2024,3,1))==('20210301','20240229')


def test_pinned_window_is_nested_and_restored_even_after_failure():
    original=period.window()
    with period.using_window('20231008','20261007'):
        assert period.window()==('20231008','20261007')
        with pytest.raises(RuntimeError):
            with period.using_window('20221008','20251007'):
                assert period.window()==('20221008','20251007')
                raise RuntimeError('stop')
        assert period.window()==('20231008','20261007')
        assert period.window(date(2026,10,9))==('20231009','20261008')
    assert period.window()==original


def test_acceptance_and_completeness_use_frozen_trial_window():
    with period.using_window('20231008','20261007'):
        dates=period.expected_dates('crypto',*period.window())
        summary=dict(market='crypto',months=36,start=dates[0],end=dates[-1],sessions=len(dates))
        assert period.accepts(summary)
        assert period.require_complete(dates,'crypto')==dates
        with pytest.raises(ValueError):period.require_complete(dates[:-1],'crypto')
    with period.using_window('20231009','20261008'):
        assert not period.accepts(summary)


def test_kst_midnight_does_not_seal_open_us_or_crypto_day():
    now=datetime(2026,10,7,16,tzinfo=timezone.utc)  # 10/8 01:00 KST
    assert period.last_complete_date('crypto',now)=='20261006'
    assert period.last_complete_date('us',now)=='20261006'
    assert period.last_complete_date('kr',now)=='20261007'
    assert period.last_complete_date('us',datetime(2026,10,7,21,tzinfo=timezone.utc))=='20261007'

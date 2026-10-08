import pandas as pd

from quant.timefolio_heatmap_week_boundaries import assess_weeks


def result(ending):
    dates = pd.bdate_range('2026-08-28', ending).strftime('%Y%m%d').tolist()
    weeks = sorted({str(pd.Period(d, freq='W-SUN')) for d in dates})
    return {'daily': [{'date': d} for d in dates],
            'weekly': [{'week': w, 'turnover': .01 if i in [1, 2, 3, len(weeks)-1] else .1} for i, w in enumerate(weeks)],
            'metrics': {'low_turnover_weeks': 3, 'assessed_full_weeks': len(weeks)-2}}


def test_chuseok_three_session_week_is_complete_and_can_trigger_fourth_failure():
    r = result('20260923'); fixed = assess_weeks(r)
    assert r['weekly'][-1]['week'] in fixed['calendar_assessed_weeks']
    assert r['weekly'][0]['week'] not in fixed['calendar_assessed_weeks']
    assert fixed['low_turnover_weeks'] == 4
    assert fixed['four_week_turnover_stop']
    assert fixed['assessed_full_weeks'] == r['metrics']['assessed_full_weeks']+1


def test_unknown_closure_does_not_make_a_wednesday_complete():
    r = result('20260923'); fixed = assess_weeks(r, known_closed=[])
    assert r['weekly'][-1]['week'] not in fixed['calendar_assessed_weeks']
    assert fixed['low_turnover_weeks'] == 3


def test_regular_friday_is_complete_and_correction_is_idempotent():
    r = result('20260918'); fixed = assess_weeks(r)
    assert r['weekly'][-1]['week'] in fixed['calendar_assessed_weeks']
    r['metrics'].update(fixed)
    assert assess_weeks(r) == fixed

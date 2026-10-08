import math
import numpy as np
import pytest
from quant.timefolio_heatmap_paper_benchmark import nav_metrics, reported_reference, distance


def test_initial_day_cost_and_sample_standard_deviation_are_retained():
    result = nav_metrics([90., 99., 108.9], initial=100, annual_days=252)
    returns = [-.1, .1, .1]
    mean = sum(returns)/3
    sample_std = math.sqrt(sum((r-mean)**2 for r in returns)/2)
    assert result['sharpe'] == pytest.approx(mean/sample_std*math.sqrt(252))
    assert result['mdd'] == pytest.approx(-.1)
    assert result['annualized_growth'] == pytest.approx(1.089**84-1)
    crypto_clock = nav_metrics([90., 99., 108.9], initial=100, annual_days=365)
    assert crypto_clock['sharpe']/result['sharpe'] == pytest.approx(math.sqrt(365/252))


def test_rounding_and_drawdown_cannot_be_ignored_when_calling_an_exceedance():
    reference = reported_reference(dict(sharpe=3.87, mdd=-4.1, final=1.6959, days=408))
    metrics = dict(sharpe=3.871, sharpe_defined=True, annualized_growth=1., mdd=-.041)
    assert distance(metrics, reference)['nominal_joint_exceed']
    assert not distance(metrics, reference)['conservative_joint_exceed']
    metrics.update(sharpe=4., mdd=-.04)
    assert distance(metrics, reference)['conservative_joint_exceed']
    metrics['mdd']=-.10
    assert distance(metrics, reference)['sharpe_near']
    assert not distance(metrics, reference)['all_metrics_near']
    assert not distance(metrics, reference)['nominal_joint_exceed']
    assert reference['annualized_growth'] == pytest.approx(1.6959**(365/408)-1)


def test_empty_nonfinite_and_constant_accounts_do_not_become_high_sharpe_winners():
    for nav in [[], [100.], [100., np.nan], [100., 0.]]:
        with pytest.raises(ValueError):nav_metrics(nav, initial=100)
    result = nav_metrics([100.,100.,100.], initial=100)
    assert result['sharpe']==0 and not result['sharpe_defined']
    reference=reported_reference(dict(sharpe=3.87,mdd=-4.1,final=1.6959,days=408))
    assert not distance(result,reference)['sharpe_near']
    with pytest.raises(ValueError):distance(result,reference,near_fraction=1.)

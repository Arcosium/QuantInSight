"""Paired interventions: replay, isolation, restored traits and accounting."""
import copy

import pytest

from research.investor_society import Society
from research.society_shocks import (
    AddressedRandom, CoupledSociety, apply_conformity, choose_targets, contrast,
    digest, distribution, run_branch,
)


@pytest.fixture
def checkpoint():
    s = CoupledSociety.from_checkpoint(Society(60, 7).checkpoint(), 7)
    for _ in range(12):
        s.step()
    return s.checkpoint()


def draw(rng, context, extra=False):
    with rng.scope(*context):
        if extra:
            rng.random()
        return rng.random(), rng.gauss(0, 1)


def test_conditional_draw_does_not_shift_unrelated_sites_or_people(checkpoint):
    rng = AddressedRandom(7, Society.restore(checkpoint).rng.getstate())
    assert draw(rng, (20, 3), extra=True) == draw(rng, (20, 3))
    expected = draw(rng, (20, 4))
    for i in range(10):
        draw(rng, (20, 3), extra=bool(i % 2))
    assert draw(rng, (20, 4)) == expected
    assert draw(rng, (21, 4)) != expected


def test_zero_intervention_is_exact_replay_and_preserves_checkpoint(checkpoint):
    before = digest(checkpoint)
    ids = choose_targets(checkpoint, .2, 7)
    a = run_branch(checkpoint, 7, ids, 0, None, 15)
    b = run_branch(checkpoint, 7, ids, 0, 1, 15)
    assert a['path'] == b['path']
    assert digest(checkpoint) == before
    comparison = contrast(a, b)
    assert all(e['price_gap_pct'] == 0 for e in comparison['effects'])
    assert comparison['metrics']['integrated_abs_gap_pct_days'] == 0


@pytest.mark.parametrize('delta,duration', [(.2, None), (-.2, None), (.4, 1), (.2, 5)])
def test_shock_shares_exogenous_events_and_conserves_accounts(checkpoint, delta, duration):
    ids = choose_targets(checkpoint, .3, 7)
    a = run_branch(checkpoint, 7, ids, 0, None, 12)
    b = run_branch(checkpoint, 7, ids, delta, duration, 12)
    result = contrast(a, b)
    assert result['metrics']['economic_tapes_equal']
    assert b['initial_state_equal_except_trait']
    for p in b['path'][1:]:
        assert p['cash_error'] == 0 and p['aggregate_net_shares'] == [0, 0, 0]
        assert p['filled_shares_two_sided'] <= p['submitted_shares_two_sided']
    originals = [checkpoint['residents'][i]['conformity'] for i in ids]
    if duration:
        assert b['final_conformity'] == originals
        assert all(p['effective_delta'] == 0 for p in b['path'][duration + 1:])
    else:
        assert b['final_conformity'] == [max(0, min(1, x + delta)) for x in originals]


def test_trait_restoration_does_not_restore_memory_or_portfolio(checkpoint):
    s = CoupledSociety.from_checkpoint(checkpoint, 7)
    a = s.residents[0]
    original = {0: a.conformity}
    apply_conformity(s, [0], original, .2)
    marker = {'event': 'personal_note', 'day': s.day}
    a.memories.append(marker)
    a.cash -= 100
    s.government_cash += 100
    changed_cash = a.cash
    apply_conformity(s, [0], original, 0)
    assert a.conformity == original[0]
    assert a.cash == changed_cash and a.memories[-1] == marker
    s.validate()


def test_targets_are_reproducible_and_wealth_selection_is_actual(checkpoint):
    assert choose_targets(checkpoint, .2, 7) == choose_targets(checkpoint, .2, 7)
    s = Society.restore(checkpoint)
    ranked = sorted(s.residents, key=s.wealth, reverse=True)[:12]
    assert choose_targets(checkpoint, .2, 7, 'wealth') == sorted(a.id for a in ranked)
    with pytest.raises(ValueError):
        choose_targets(checkpoint, 0, 7)


def test_clipping_reports_actual_dose(checkpoint):
    ids = [0]
    checkpoint['residents'][0]['conformity'] = .95
    b = run_branch(checkpoint, 7, ids, .2, None, 5)
    assert b['effective_initial_delta'] == pytest.approx(.05)
    assert b['final_conformity'] == [1]


@pytest.mark.parametrize('corruption', ['events', 'cash', 'shares', 'fills', 'groups'])
def test_contrast_rejects_false_comparisons(checkpoint, corruption):
    ids = choose_targets(checkpoint, .2, 7)
    a = run_branch(checkpoint, 7, ids, 0, None, 5)
    b = copy.deepcopy(a)
    p = b['path'][1]
    if corruption == 'events':
        p['economic_tape_sha256'] = 'changed'
    elif corruption == 'cash':
        p['cash_error'] = 1
    elif corruption == 'shares':
        p['aggregate_net_shares'][0] = 1
    elif corruption == 'fills':
        p['filled_shares_two_sided'] = p['submitted_shares_two_sided'] + 1
    else:
        p['groups']['target']['shares'][0] += 1
    with pytest.raises(AssertionError):
        contrast(a, b)


def test_known_response_and_five_day_recovery(checkpoint):
    ids = choose_targets(checkpoint, .2, 7)
    a = run_branch(checkpoint, 7, ids, 0, None, 10)
    b = copy.deepcopy(a)
    for p, gap in zip(b['path'][1:], [1, 4, 2, .4, .3, .2, .1, 0, 0, 0]):
        p['index'] = a['path'][p['offset']]['index'] * (1 + gap / 100)
    m = contrast(a, b)['metrics']
    assert m['peak_price_gap_pct'] == pytest.approx(4)
    assert m['peak_offset'] == 2
    assert m['integrated_abs_gap_pct_days'] == pytest.approx(8)
    assert m['recovery_offset_inside_half_pct_for_5_days'] == 4
    assert m['recovery_censored'] is False
    for p in b['path'][1:]:
        p['index'] = a['path'][p['offset']]['index'] * 1.02
    assert contrast(a, b)['metrics']['recovery_censored'] is True


def test_quantiles_are_world_dispersion_and_missing_values_are_excluded():
    d = distribution([0, 10, 20, None])
    assert d == {'n': 3, 'mean': 10, 'median': 10, 'p10': 2, 'p90': 18,
                 'positive_fraction': 2 / 3}
    assert distribution([None])['n'] == 0

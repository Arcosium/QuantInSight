import pytest
from autofolio import campaign, research, store


@pytest.fixture
def database(tmp_path, monkeypatch):
    monkeypatch.setattr(store, 'DATA', tmp_path)
    monkeypatch.setattr(store, 'DB', tmp_path / 'research.sqlite')
    monkeypatch.setattr(research, 'protocol', lambda market: {'ready': True, 'message': 'ready'})
    monkeypatch.setattr(research, 'window', lambda: ('20231001', '20260930'))
    store.initialize()
    research.initialize()
    return tmp_path


def enable():
    store.set_setting('research_campaign_enabled', True)


def finish(status='done'):
    with store.connect() as db:
        db.execute('UPDATE alpha_candidates SET status=?', (status,))


def test_disabled_creates_no_jobs(database):
    assert campaign.replenish(7) == []
    assert campaign.status()['enabled'] is False
    assert campaign._rows(7) == []


def test_bounded_refill_and_restart_dedup(database):
    enable()
    first = campaign.replenish(7, 4)
    assert len(first) == len(set(first)) == 4
    assert campaign.replenish(7, 4) == []
    rows = campaign._rows(7)
    assert [r['market'] for r in rows].count('kr') == 2
    assert [r['market'] for r in rows].count('us') == 2
    finish()
    # Losing the scheduling cursor does not rerun definitions already recorded.
    store.set_setting(campaign.STATE_KEY, {})
    second = campaign.replenish(7, 4)
    assert len(second) == 4 and not set(first) & set(second)
    assert campaign.status(7)['pending'] == 4
    assert campaign.status(7)['done'] == 4


def test_single_slot_market_fairness(database):
    enable()
    markets = []
    for _ in range(6):
        identity, = campaign.replenish(7, 1)
        markets.append(next(r['market'] for r in campaign._rows(7) if r['id'] == identity))
        finish()
    assert markets == ['kr', 'us'] * 3


def test_manual_and_waiting_jobs_consume_capacity(database):
    enable()
    genome = next(campaign.proposals())[0]
    research.save_candidates(7, 'kr', [genome], 'manual')
    with store.connect() as db:
        db.execute("UPDATE alpha_candidates SET status='waiting_data'")
    assert len(campaign.replenish(7, 2)) == 1
    assert campaign.status(7)['pending'] == 2
    assert campaign.replenish(7, 2) == []


def test_exhaustion_failures_are_not_retried_and_month_rolls(database, monkeypatch):
    enable()
    tiny = dict(representation=['momentum'], selection_count=[20], holding_sessions=[14],
                ridge_alpha=[1., 10., 100.], mode=['rank_only'], window=[20])
    monkeypatch.setattr(research, 'STOCK_DOMAINS', tiny)
    # Baseline has no Ridge, so only one alpha setting is a distinct experiment.
    assert len(list(campaign.proposals())) == 1
    old = campaign.replenish(7, 4)
    assert len(old) == 2
    finish('failed')
    assert campaign.replenish(7, 4) == []
    state = campaign.status(7)
    assert state['phase'] == 'exhausted'
    assert state['remaining'] == 0 and state['failed'] == 2
    monkeypatch.setattr(research, 'window', lambda: ('20231101', '20261031'))
    new = campaign.replenish(7, 4)
    assert len(new) == 2 and not set(old) & set(new)
    assert campaign.status(7)['failed'] == 0


def test_plan_covers_domain_with_unique_meaningful_variants(database, monkeypatch):
    domain = dict(representation=['momentum', 'gasf'], selection_count=[10, 20],
                  holding_sessions=[5, 14], ridge_alpha=[1., 10.],
                  mode=['rank_only', 'regime_gate'], window=[20])
    monkeypatch.setattr(research, 'STOCK_DOMAINS', domain)
    plan = list(campaign.proposals())
    # Momentum: 2*2*1*2=8; GASF: 2*2*2*2=16.
    assert len(plan) == 24
    assert len({campaign._identity(7, 'kr', g, research.window()) for g, _ in plan}) == 24
    assert [g['representation'] for g, phase in plan[:2]] == ['momentum', 'gasf']
    assert plan[-1][1] == 'combined_checks'
    for genome, _ in plan:
        research.normalize(genome, 'kr')


def test_unready_market_cannot_fill_queue_and_starve_ready_market(database, monkeypatch):
    enable()
    monkeypatch.setattr(research, 'protocol', lambda market: {'ready': market == 'kr', 'message': 'coverage'})
    for _ in range(5):
        saved = campaign.replenish(7, 4)
        assert saved
        rows = campaign._rows(7)
        assert sum(r['market'] == 'us' and r['status'] == 'waiting_data' for r in rows) == 1
        assert sum(r['market'] == 'kr' and r['status'] == 'queued' for r in rows) == 3
        with store.connect() as db:
            db.execute("UPDATE alpha_candidates SET status='done' WHERE market='kr'")
    assert campaign.status(7)['done'] == 15


def test_single_slot_prioritizes_ready_work(database, monkeypatch):
    enable()
    monkeypatch.setattr(research, 'protocol', lambda market: {'ready': market == 'kr', 'message': 'coverage'})
    for _ in range(3):
        identity, = campaign.replenish(7, 1)
        assert next(r['market'] for r in campaign._rows(7) if r['id'] == identity) == 'kr'
        finish()


def test_rollover_retires_only_campaign_unstarted_jobs(database, monkeypatch):
    enable()
    original = campaign.replenish(7, 4)
    with store.connect() as db:
        db.execute("UPDATE alpha_candidates SET status='running' WHERE id=?", (original[0],))
        db.execute("UPDATE alpha_candidates SET provider='manual',status='waiting_data' WHERE id=?", (original[1],))
        db.execute("UPDATE alpha_candidates SET status='waiting_data' WHERE id=?", (original[2],))
    monkeypatch.setattr(research, 'window', lambda: ('20231101', '20261031'))
    new = campaign.replenish(7, 4)
    rows = {r['id']: r for r in campaign._rows(7)}
    assert len(new) == 2
    assert rows[original[0]]['status'] == 'running'
    assert rows[original[1]]['status'] == 'waiting_data'
    assert rows[original[2]]['status'] == rows[original[3]]['status'] == 'retired'
    assert campaign.status(7)['pending'] == 4

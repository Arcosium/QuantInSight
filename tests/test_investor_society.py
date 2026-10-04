"""Economic conservation, causal decision inputs and resumable cognition."""
import copy
import json
import math

import pytest

from research.investor_society import BIAS_NAMES, LocalBeliefs, Order, Society, prospect_value


def test_closed_economy_survives_long_run():
    s = Society(100, 42)
    for _ in range(180):
        row = s.step()
        assert row["cash_conservation_error"] == 0
        assert row["aggregate_resident_net_shares"] == [0, 0, 0]
    assert len(s.transactions) > 0
    assert any(a.consumption_total > 0 and a.income_total > 0 for a in s.residents)
    assert all(sum(a.shares[j] for a in s.residents) == f.supply for j, f in enumerate(s.firms))
    assert s.report()["verified_real_market_alpha"] is False


def test_all_residents_have_distinct_state_and_closed_peer_graph():
    s = Society(1000, 5)
    assert len({(a.risk, a.patience, a.cash, a.skill) for a in s.residents}) == 1000
    assert len({a.background for a in s.residents}) == 5
    assert all(a.id not in a.peers and all(0 <= p < 1000 for p in a.peers) for a in s.residents)
    assert all(a.memories[0]["event"] == "background" for a in s.residents)


def test_checkpoint_resume_matches_uninterrupted():
    full = Society(30, 22)
    resumed = Society(30, 22)
    for _ in range(15):
        full.step()
        resumed.step()
    resumed = Society.restore(json.loads(json.dumps(resumed.checkpoint())))
    for _ in range(25):
        full.step()
        resumed.step()
    assert full.checkpoint() == resumed.checkpoint()


def test_loan_and_interest_have_counterparty():
    s = Society(10)
    a = s.residents[0]
    cash_before, bank_before = a.cash, s.bank_cash
    assert s.borrow(a, 1000) == 1000
    assert a.cash == cash_before + 1000 and s.bank_cash == bank_before - 1000
    assert a.debt == 1000
    s.economy()
    s.validate()
    assert a.debt < 1000  # Household surplus repays bank principal.


def test_insolvent_firms_do_not_mint_wages_or_goods():
    s = Society(10, life_events=False)
    for f in s.firms:
        s.government_cash += f.cash
        f.cash = 0
    s.economy()
    s.validate()
    assert all(f.production == 0 and f.wage_bill == 0 for f in s.firms)
    assert all(a.last_income == 0 for a in s.residents)


def test_goods_shortages_cannot_create_inventory():
    s = Society(20, life_events=False)
    for f in s.firms:
        f.productivity = .01
    row = s.step()
    assert row["consumption_shortages"] > 0
    assert all(f.inventory >= -1e-8 for f in s.firms)
    assert any(a.hunger > 0 for a in s.residents)


def test_exact_fill_fees_cost_basis_and_share_conservation():
    s = Society(10, fee_bps=10)
    buyer, seller = s.residents[:2]
    buyer_old, seller_old = buyer.cash, seller.cash
    owned, old_basis = buyer.shares[0], buyer.cost_basis[0]
    buy = Order(0, 0, "buy", 2, 1200, 0, "test")
    sell = Order(1, 0, "sell", 2, 1100, 1, "test")
    assert s.settle(buy, sell, 2, 1150) == 2
    fee = s.fee(2300)
    assert buyer.cash == buyer_old - 2300 - fee
    assert seller.cash == seller_old + 2300 - fee
    assert buyer.cost_basis[0] == pytest.approx((owned * old_basis + 2300 + fee) / (owned + 2))
    s.validate()


def test_affordability_respects_rounded_fees():
    s = Society(10, fee_bps=10)
    a = s.residents[0]
    s.bank_cash += a.cash - 100
    a.cash = 100
    b = Order(0, 0, "buy", 3, 100, 0, "test")
    sell = Order(1, 0, "sell", 3, 100, 1, "test")
    assert s.settle(b, sell, 3, 100) == 0
    s.validate()


def test_limits_self_trade_and_no_short_selling():
    s = Society(10)
    buy = Order(0, 0, "buy", 10000, 100, 0, "test")
    sell = Order(1, 0, "sell", 10000, 90, 1, "test")
    with pytest.raises(ValueError):
        s.settle(buy, sell, 1, 101)
    self_sell = Order(0, 0, "sell", 2, 90, 1, "test")
    assert s.settle(buy, self_sell, 1, 95) == 0
    owned = s.residents[1].shares[0]
    assert s.settle(buy, sell, 10000, 95) <= owned
    s.validate()


def test_price_time_priority_and_resting_price():
    s = Society(10, fee_bps=0)
    orders = [Order(0, 0, "buy", 1, 120, 0, "test"),
              Order(1, 0, "buy", 1, 130, 1, "test"),
              Order(2, 0, "sell", 1, 100, 2, "test")]
    s.exchange(orders)
    assert len(s.transactions) == 1
    assert s.transactions[0]["buyer"] == 1
    assert s.transactions[0]["price_cents"] == 130
    s.validate()


def test_unmatched_orders_do_not_move_price():
    s = Society(10)
    initial = s.firms[0].price
    s.exchange([Order(0, 0, "buy", 1, 100, 0, "test"), Order(1, 0, "sell", 1, 200, 1, "test")])
    assert s.firms[0].price == initial
    assert s.transactions == []


def test_cash_need_changes_same_person_decision():
    s = Society(10, life_events=False)
    a = s.residents[0]
    a.information_lag = 0
    state = s.rng.getstate()
    rich, _ = s.signal(a, 0, s.observation(a))
    s.rng.setstate(state)
    s.bank_cash += a.cash
    a.cash = 0
    poor, reason = s.signal(a, 0, s.observation(a))
    assert poor < rich
    assert reason.startswith("cash_need")
    s.validate()


def test_loss_aversion_ablation_changes_decision():
    s = Society(10, life_events=False)
    a = s.residents[0]
    a.loss_aversion = 2.5
    a.risk = .1
    obs = s.observation(a)
    obs["market"][0]["value_gap"] = .8
    obs["market"][0]["volatility_10d"] = .10
    state = s.rng.getstate()
    on, reason = s.signal(a, 0, obs)
    s.rng.setstate(state)
    s.loss_aversion = False
    off, _ = s.signal(a, 0, obs)
    assert 0 <= on < off
    assert "loss_aversion" in reason


def test_private_peer_accounts_are_not_observable():
    s = Society(10)
    a = s.residents[0]
    before = s.observation(a)
    peer = s.residents[a.peers[0]]
    peer.cash += 10000
    peer.memories.append({"personal_note": "private household history"})
    assert s.observation(a) == before
    s.herding = False
    assert all(m["peer_signal"] == 0 for m in s.observation(a)["market"])


@pytest.mark.parametrize("reply", [
    {"resident_id": 0, "beliefs": [math.nan, 0, 0], "reason": "x"},
    {"resident_id": 0, "beliefs": [True, 0, 0], "reason": "x"},
    {"resident_id": 0, "beliefs": [2, 0, 0], "reason": "x"},
    {"resident_id": 1, "beliefs": [0, 0, 0], "reason": "x"},
])
def test_llm_rejects_invalid_and_wrong_person_reply(reply):
    with pytest.raises(ValueError):
        LocalBeliefs.validate_reply(reply, 0)


@pytest.mark.parametrize("url", ["https://example.com/v1", "http://example.com/v1", "http://secret@localhost/v1"])
def test_only_local_llm_endpoint(url):
    with pytest.raises(ValueError):
        LocalBeliefs(url, "model")


def test_llm_rotation_cohort_cutoff_and_no_direct_execution():
    s = Society(10)
    captured = []
    def transport(messages):
        payload = json.loads(messages[1]["content"])
        captured.append(copy.deepcopy(payload))
        i = payload["profile"]["id"]
        return {"resident_id": i, "beliefs": [.3, -.2, .1], "reason": "개인 생활비와 공개된 기업 실적을 고려함"}
    llm = LocalBeliefs("http://localhost:8080/v1", "model", agents=6, every=1, transport=transport)
    checkpoint = s.checkpoint()
    llm.update(s)
    llm.update(s)
    assert len({x["resident"] for x in s.llm_audit}) == 10
    assert all(x["observation"]["day"] == 0 for x in captured)
    assert [a.cash for a in s.residents] == [x["cash"] for x in checkpoint["residents"]]
    assert [a.shares for a in s.residents] == [x["shares"] for x in checkpoint["residents"]]
    assert s.transactions == []
    s.validate()


def test_llm_failure_audited_without_resetting_beliefs():
    s = Society(10)
    s.day = 10
    s.residents[0].beliefs = [.1, .2, .3]
    def fail(messages):
        raise RuntimeError("Do not copy exception text")
    llm = LocalBeliefs("http://localhost:8080/v1", "model", agents=1, transport=fail)
    llm.update(s)
    assert s.residents[0].beliefs == [.1, .2, .3]
    assert s.llm_audit[0]["status"] == "failed"
    assert s.llm_audit[0]["error_type"] == "RuntimeError"
    assert "Do not copy" not in json.dumps(s.llm_audit)


def test_routines_continue_without_thinking_or_llm():
    s = Society(20, life_events=False)
    baskets = [a.routine.consumption_basket[:] for a in s.residents]
    for _ in range(20):
        s.step()
    assert all(a.income_total > 0 and a.consumption_total > 0 for a in s.residents)
    assert [a.routine.consumption_basket for a in s.residents] == baskets
    assert all(a.thought.successful_reflections == 0 for a in s.residents)
    assert s.llm_audit == []


def test_important_event_precedes_periodic_review_and_changes_thought_only():
    s = Society(10)
    urgent = s.residents[9]
    urgent.emergency_remaining = 5000
    before_routine = copy.deepcopy(urgent.routine)
    old_cash, old_shares = urgent.cash, urgent.shares[:]
    def transport(messages):
        payload = json.loads(messages[1]["content"])
        assert payload["reflection_triggers"] == ["unexpected_expense"]
        return {"resident_id": payload["profile"]["id"], "beliefs": [-.7, -.5, -.4], "reason": "급한 지출이 생겨 투자 위험을 줄이고 싶음"}
    llm = LocalBeliefs("http://localhost:8080/v1", "model", agents=1, transport=transport)
    llm.update(s)
    assert s.llm_audit[0]["resident"] == 9
    assert urgent.thought.beliefs == [-.7, -.5, -.4]
    assert urgent.thought.trigger_reasons == ["unexpected_expense"]
    assert urgent.thought.thesis.startswith("급한 지출")
    assert urgent.routine == before_routine
    assert urgent.cash == old_cash and urgent.shares == old_shares
    llm.update(s)
    assert len(s.llm_audit) == 1  # Same-day retries/reflections are suppressed.


def test_failed_thought_attempt_respects_retry_cooldown():
    s = Society(10)
    s.day = 10
    def fail(messages):
        raise ValueError("bad reply")
    llm = LocalBeliefs("http://localhost:8080/v1", "model", agents=10, transport=fail)
    llm.update(s)
    assert len(s.llm_audit) == 10
    llm.update(s)
    s.day += 1
    llm.update(s)
    assert len(s.llm_audit) == 10
    s.day += 1
    llm.update(s)
    assert len(s.llm_audit) == 20


def test_successful_thought_is_preserved_by_checkpoint():
    s = Society(10)
    s.day = 10
    def transport(messages):
        i = json.loads(messages[1]["content"])["profile"]["id"]
        return {"resident_id": i, "beliefs": [.4, -.1, .2], "reason": "이전 손실을 기억해 선택함"}
    llm = LocalBeliefs("http://localhost:8080/v1", "model", agents=10, transport=transport)
    llm.update(s)
    restored = Society.restore(json.loads(json.dumps(s.checkpoint())))
    assert restored.residents[0].thought == s.residents[0].thought
    assert restored.residents[0].routine == s.residents[0].routine
    assert restored.llm_cursor == s.llm_cursor


def test_llm_total_budget_counts_failed_attempts_too():
    s = Society(10)
    s.day = 10
    def fail(messages):
        raise ValueError("bad reply")
    llm = LocalBeliefs("http://localhost:8080/v1", "model", agents=4, max_calls=5, transport=fail)
    llm.update(s)
    s.day += 2
    llm.update(s)
    s.day += 2
    llm.update(s)
    assert len(s.llm_audit) == llm.calls_made == 5
    assert all(x["status"] == "failed" for x in s.llm_audit)


def test_truncated_model_reply_is_audited_and_cannot_change_thought(monkeypatch):
    from io import BytesIO
    import research.investor_society as mod
    class Opener:
        def open(self, req, timeout):
            body = json.loads(req.data)
            assert body["reasoning"]["enabled"] is False
            return BytesIO(json.dumps({"choices": [{"finish_reason": "length", "message": {"content": '{"resident_id":0,"beliefs":[1,1,1],"reason":"x"}'}}]}).encode())
    monkeypatch.setattr(mod, "build_opener", lambda *args: Opener())
    s = Society(10)
    s.day = 10
    old = s.residents[0].beliefs[:]
    llm = LocalBeliefs("http://localhost:8080/v1", "model", agents=1)
    llm.update(s)
    assert s.llm_audit[0]["status"] == "failed"
    assert s.residents[0].beliefs == old


def test_prospect_value_penalizes_equal_losses_more_than_gains():
    assert prospect_value(.1, 2) == -prospect_value(-.1, 2) / 2
    assert prospect_value(.1, 1) == -prospect_value(-.1, 1)
    with pytest.raises(ValueError):
        prospect_value(.1, math.nan)


def test_disposition_delays_losses_without_forcing_purchase():
    s = Society(10)
    a = s.residents[0]
    a.biases.disposition = 1
    a.cost_basis[0] = 1500
    score = s.disposition_score(a, 0, -.4, 1000, False)
    assert -.4 < score < 0
    assert s.disposition_score(a, 0, .2, 1000, False) == .2
    assert s.disposition_score(a, 0, -.4, 1000, True) == -.4
    s.disabled_biases = ("disposition",)
    assert s.disposition_score(a, 0, -.4, 1000, False) == -.4


def test_disposition_encourages_realizing_owned_winners():
    s = Society(10)
    a = s.residents[0]
    a.biases.disposition = 1
    a.cost_basis[0] = 1000
    assert s.disposition_score(a, 0, .05, 1200, False) < 0
    a.shares[0] = 0
    assert s.disposition_score(a, 0, .05, 1200, False) == .05


def test_overconfidence_strength_depends_on_personal_outcomes():
    s = Society(10)
    a = s.residents[0]
    a.biases.overconfidence = 1
    a.confidence = 0
    initial = s.conviction_multiplier(a)
    a.confidence = .8
    assert s.conviction_multiplier(a) > initial
    s.disabled_biases = ("overconfidence",)
    assert s.conviction_multiplier(a) == 1


def test_attention_changes_candidate_distribution_not_scores():
    s = Society(10)
    a = s.residents[0]
    a.biases.attention = 1
    obs = s.observation(a)
    obs["market"][0]["momentum_5d"] = .005
    obs["market"][1]["momentum_5d"] = -.20
    weights = s.attention_weights(a, obs, [0, 1])
    draws = s.rng.choices([0, 1], weights=weights, k=1000)
    assert draws.count(1) > 700
    s.disabled_biases = ("attention",)
    assert s.attention_weights(a, obs, [0, 1]) == [1, 1]


def test_confirmation_resists_opposing_beliefs_but_allows_updates():
    s = Society(10)
    a = s.residents[0]
    a.biases.confirmation = 1
    a.beliefs = [.8, -.8, 0]
    incoming = [-.8, -.4, .5]
    accepted = s.accept_beliefs(a, incoming)
    assert -.8 < accepted[0] < .8
    assert accepted[1:] == [-.4, .5]
    s.disabled_biases = ("confirmation",)
    assert s.accept_beliefs(a, incoming) == incoming


@pytest.mark.parametrize("name,field,value", [("recency", "momentum_5d", .1), ("anchoring", "observed_price", 1000)])
def test_valuation_bias_toggle_changes_same_observation(name, field, value):
    s = Society(10, life_events=False, disabled_biases=tuple(b for b in BIAS_NAMES if b != name))
    a = s.residents[0]
    setattr(a.biases, name, 1)
    a.cost_basis[0] = 2000
    obs = s.observation(a)
    obs["market"][0][field] = value
    state = s.rng.getstate()
    on, reason = s.signal(a, 0, obs)
    s.rng.setstate(state)
    s.disabled_biases = BIAS_NAMES
    off, _ = s.signal(a, 0, obs)
    assert on > off
    assert name in reason


def test_biases_and_switches_persist_in_checkpoint():
    s = Society(10, disabled_biases=("disposition", "recency"))
    restored = Society.restore(json.loads(json.dumps(s.checkpoint())))
    assert restored.residents[0].biases == s.residents[0].biases
    assert restored.bias_enabled("disposition") is False
    assert restored.bias_enabled("recency") is False
    assert restored.report()["behavioral_biases"]["strengths_empirically_calibrated"] is False

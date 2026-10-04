"""A closed synthetic economy of residents, resident-owned firms and a bank.

Research prototype, not MiroFish/OASIS, the live QuantInSight strategy, or an
empirically calibrated human population. Integer cents and whole shares;
goods are divisible. Prices emerge from price/time-priority matched orders.
No external feeds, real accounts, broker imports, or money creation.

python3 -m research.investor_society --people 1000 --days 180 --seed 42
Optional local LLM belief updates: --llm-url http://localhost:PORT/v1
  --llm-model MODEL --llm-agents 30 --llm-every 10
All residents retain distinct persistent states; LLM updates rotate through
the population, and rules handle the other decisions. Output reports cannot
establish real-market alpha. Personality and economic priors are synthetic.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

VERSION = "closed-investor-society-v3-behavioral-biases"
ROOT = Path(__file__).resolve().parents[1]
SECTORS = ("food", "housing", "services")
STYLES = ("value", "momentum", "income", "social", "cautious")
BIAS_NAMES = ("loss_aversion", "disposition", "anchoring", "recency", "overconfidence", "attention", "confirmation", "herding")
BIAS_SOURCES = {
    "loss_aversion": "https://web.mit.edu/curhan/www/docs/Articles/15341_Readings/Behavioral_Decision_Theory/Kahneman_Tversky_1979_Prospect_theory.pdf",
    "disposition": "https://faculty.haas.berkeley.edu/odean/papers%20current%20versions/areinvestorsreluctant.pdf",
    "anchoring": "https://cebma.org/assets/Uploads/Tversky_Kahneman_1974.pdf",
    "recency": "https://dash.harvard.edu/entities/publication/73120378-bca1-6bd4-e053-0100007fdf3b",
    "overconfidence": "https://faculty.haas.berkeley.edu/odean/papers/learning/learning.html",
    "attention": "https://faculty.haas.berkeley.edu/odean/Papers%20current%20versions/AllThatGlitters_RFS_2008.pdf",
    "confirmation": "https://journals.sagepub.com/doi/abs/10.1037/1089-2680.2.2.175",
    "herding": "https://economics.mit.edu/sites/default/files/publications/banerjee92herd.pdf",
}
BIAS_MECHANISMS = {
    "loss_aversion": "Prospective purchase downside receives a larger reference-dependent value penalty than equal upside; does not mechanically cause averaging down.",
    "disposition": "Owned losers are harder to sell, owned winners attract earlier realization; essential liquidity needs override this preference.",
    "anchoring": "Purchase price, or the initial observed price for unowned shares, pulls perceived value away from the public fundamental estimate.",
    "recency": "Recent price moves receive an additional extrapolative weight beyond the resident's baseline factor preference.",
    "overconfidence": "Asymmetric attribution of realized wins/losses increases conviction, trading attention and purchase size; does not guarantee success.",
    "attention": "Extreme recent returns enlarge a stock's share of the buy candidate set, without assigning a positive return forecast or forcing a purchase.",
    "confirmation": "Contradictory proposed LLM beliefs are incorporated less fully than supporting beliefs; both raw and effective beliefs are audited.",
    "herding": "Public past opinions from a limited peer network affect a resident's current opinion; private accounts and future outcomes remain inaccessible.",
}


def prospect_value(change: float, loss_aversion: float, curvature: float = .88) -> float:
    """Reference-dependent value; priors are illustrative, never estimates."""
    if not math.isfinite(loss_aversion) or loss_aversion < 1 or not 0 < curvature <= 1 or not math.isfinite(change):
        raise ValueError("Invalid prospect-value inputs")
    return change ** curvature if change >= 0 else -loss_aversion * (-change) ** curvature


@dataclass
class Biases:
    disposition: float
    anchoring: float
    recency: float
    overconfidence: float
    attention: float
    confirmation: float


@dataclass
class Routine:
    """Repeated lifestyle choices, separate from investment reflections."""
    consumption_basket: list[float]
    absence_probability: float
    market_review_interval: int
    last_market_review: int = -1
    loan_repayment_fraction: float = .05


@dataclass
class Thought:
    beliefs: list[float] = field(default_factory=lambda: [0.0] * 3)
    thesis: str = "No personal reflection yet; follows existing habits."
    last_reflection_day: int = -1
    last_attempt_day: int = -1
    trigger_reasons: list[str] = field(default_factory=list)
    successful_reflections: int = 0


@dataclass
class Resident:
    id: int
    age: int
    background: str
    sector: int
    skill: float
    style: str
    risk: float
    patience: float
    loss_aversion: float
    conformity: float
    attention: float
    impulsivity: float
    reserve_days: int
    cash: int
    shares: list[int]
    cost_basis: list[float]
    peers: list[int]
    interest: int
    information_lag: int
    factor_weights: list[float]
    routine: Routine
    biases: Biases
    thought: Thought = field(default_factory=Thought)
    debt: int = 0
    interest_due: int = 0
    hunger: float = 0.0
    confidence: float = 0.0
    last_income: int = 0
    income_total: int = 0
    consumption_total: int = 0
    trade_cashflow: int = 0
    emergency_remaining: int = 0
    signals: list[float] = field(default_factory=lambda: [0.0] * 3)
    memories: list[dict] = field(default_factory=list)

    def remember(self, day: int, event: str, **details) -> None:
        self.memories.append({"day": day, "event": event, **details})
        self.memories = self.memories[-20:]

    @property
    def beliefs(self) -> list[float]:
        return self.thought.beliefs

    @beliefs.setter
    def beliefs(self, values: list[float]) -> None:
        self.thought.beliefs = values


@dataclass
class Firm:
    sector: str
    cash: int
    goods_price: int
    wage: int
    supply: int
    price: int
    inventory: float = 0.0
    profit_ema: float = 0.0
    production: float = 0.0
    sales: float = 0.0
    wage_bill: int = 0
    dividend_per_share: int = 0
    productivity: float = 3.0


@dataclass
class Order:
    resident: int
    sector: int
    side: str
    quantity: int
    limit: int
    sequence: int
    reason: str


class Society:
    def __init__(self, people: int = 1000, seed: int = 42, *,
                 herding: bool = True, loss_aversion: bool = True,
                 life_events: bool = True, fee_bps: int = 10,
                 disabled_biases: tuple[str, ...] = ()):
        if people < 10 or people > 10000:
            raise ValueError("people must be between 10 and 10000")
        if not 0 <= fee_bps <= 1000:
            raise ValueError("fee_bps must be between 0 and 1000")
        self.rng = random.Random(seed)
        self.seed, self.day = seed, 0
        self.herding, self.loss_aversion, self.life_events = herding, loss_aversion, life_events
        self.fee_bps = fee_bps
        if set(disabled_biases) - set(BIAS_NAMES):
            raise ValueError("Unknown behavioral bias")
        self.disabled_biases = tuple(disabled_biases)
        self.bank_cash = people * 2000
        self.government_cash = people * 500
        self.firms = [Firm(s, people * 20000, p, w, people * 20, price)
                      for s, p, w, price in zip(SECTORS, (55, 40, 30),
                                               (140, 110, 85), (1200, 1000, 900))]
        self.residents: list[Resident] = []
        for i in range(people):
            age = self.rng.randint(20, 75)
            background = ("first_job" if age < 28 else
                          self.rng.choice(("skilled_worker", "career_change", "experienced_worker"))
                          if age < 60 else "late_career")
            risk, patience = self.rng.betavariate(2, 2), self.rng.betavariate(2, 2)
            weights = [self.rng.expovariate(1) for _ in range(5)]
            total = sum(weights)
            # Factor preferences: value, momentum, dividend, social, noise.
            a = Resident(i, age, background,
                         self.rng.randrange(3), self.rng.uniform(.7, 1.4),
                         self.rng.choice(STYLES), risk, patience,
                         self.rng.uniform(1.0, 2.5), self.rng.betavariate(2, 3),
                         self.rng.uniform(.08, .8), self.rng.uniform(.005, .12),
                         self.rng.randint(10, 90),
                         max(10000, int(self.rng.lognormvariate(11.5, .85))),
                         [0, 0, 0], [0.0] * 3, [], self.rng.randrange(3),
                         self.rng.randint(0, 5), [v / total for v in weights],
                         Routine([self.rng.uniform(.8, 1.2) for _ in range(3)],
                                 self.rng.uniform(.005, .025),
                                 max(1, round(1 + patience * 6))),
                         Biases(*(self.rng.betavariate(1.5, 4) for _ in range(6))))
            # Explicit fictional life-course priors, not claims about real
            # demographic groups. Background changes constraints and learning.
            if background == "first_job":
                a.skill *= .8
                a.cash = max(10000, a.cash // 2)
                a.attention = min(.95, a.attention + .1)
                a.remember(0, "background", experience="first income; limited accumulated capital")
            elif background == "career_change":
                a.skill *= .85
                a.reserve_days += 20
                a.remember(0, "background", experience="recent career change; wants a larger cash buffer")
            elif background == "skilled_worker":
                a.skill *= 1.15
                a.factor_weights[0] += .2
                a.interest = a.sector
                a.remember(0, "background", experience="industry experience; specialised factor preference")
            elif background == "experienced_worker":
                a.loss_aversion *= 1.2
                a.information_lag = max(0, a.information_lag - 1)
                a.remember(0, "background", experience="remembers a previous fictional market loss")
            else:
                a.reserve_days += 30
                a.patience = min(1, a.patience + .15)
                a.remember(0, "background", experience="preparing for reduced earned income")
            self.residents.append(a)
        # All firms are owned by residents; no outside holder or cash faucet.
        for j, f in enumerate(self.firms):
            allocations = [max(1, int(a.cash / 10000 * self.rng.uniform(.5, 1.5)))
                           for a in self.residents]
            total = sum(allocations)
            counts = [int(f.supply * v / total) for v in allocations]
            for i in range(f.supply - sum(counts)):
                counts[i % people] += 1
            for a, q in zip(self.residents, counts):
                a.shares[j], a.cost_basis[j] = q, float(f.price)
        for a in self.residents:
            # Sector neighbours plus random weak ties, without omniscience.
            near = [x.id for x in self.residents if x.sector == a.sector and x.id != a.id]
            strong = self.rng.sample(near, min(4, len(near)))
            weak = self.rng.sample([i for i in range(people) if i != a.id], 4)
            a.peers = sorted(set(strong + weak))
        self.prices = [[f.price for f in self.firms]]
        self.rows: list[dict] = []
        self.transactions: list[dict] = []
        self.decisions: list[dict] = []
        self.llm_audit: list[dict] = []
        self.events: list[dict] = []
        self.initial_cash = self.total_cash()
        self.initial_wealth = [self.wealth(a) for a in self.residents]
        self.llm_cursor = 0
        self.validate()

    def total_cash(self) -> int:
        return sum(a.cash for a in self.residents) + sum(f.cash for f in self.firms) + self.bank_cash + self.government_cash

    def wealth(self, a: Resident) -> int:
        return a.cash + sum(q * f.price for q, f in zip(a.shares, self.firms)) - a.debt - a.interest_due

    def validate(self) -> None:
        if self.total_cash() != self.initial_cash:
            raise AssertionError("Closed economy cash conservation failed")
        for a in self.residents:
            if min(a.cash, a.debt, a.interest_due, *a.shares) < 0:
                raise AssertionError("Negative resident cash, debt or inventory")
        if min(self.bank_cash, self.government_cash, *(f.cash for f in self.firms)) < 0:
            raise AssertionError("Negative institutional cash")
        for j, f in enumerate(self.firms):
            if sum(a.shares[j] for a in self.residents) != f.supply:
                raise AssertionError("Share conservation failed")
            if f.inventory < -1e-8 or not math.isfinite(f.inventory) or f.price <= 0:
                raise AssertionError("Invalid goods inventory or share price")

    def fee(self, value: int) -> int:
        return (value * self.fee_bps + 9999) // 10000

    def borrow(self, a: Resident, requested: int) -> int:
        capacity = max(0, int(self.firms[a.sector].wage * a.skill * 30) - a.debt)
        amount = min(max(0, requested), capacity, self.bank_cash)
        self.bank_cash -= amount
        a.cash += amount
        a.debt += amount
        if amount:
            a.remember(self.day, "loan", cents=amount)
        return amount

    def daily_needs(self, a: Resident) -> int:
        return math.ceil(sum(f.goods_price * q for f, q in zip(self.firms, a.routine.consumption_basket)))

    def bias_enabled(self, name: str) -> bool:
        return name not in self.disabled_biases and (name != "herding" or self.herding) and (name != "loss_aversion" or self.loss_aversion)

    def conviction_multiplier(self, a: Resident) -> float:
        if not self.bias_enabled("overconfidence"):
            return 1.0
        return 1 + a.biases.overconfidence * max(0, a.confidence + .2)

    def disposition_score(self, a: Resident, j: int, score: float, price: int, urgent: bool) -> float:
        # Losing positions become harder to sell, never a forced new purchase.
        # Liquidity needs override the preference to avoid realizing a loss.
        if urgent or not self.bias_enabled("disposition") or not a.shares[j] or a.cost_basis[j] <= 0:
            return score
        gain = price / a.cost_basis[j] - 1
        if gain < 0 and score < 0:
            return score * (1 - .7 * a.biases.disposition)
        if gain > 0:
            return score - .20 * a.biases.disposition * min(1, gain / .10)
        return score

    def attention_weights(self, a: Resident, obs: dict, choices: list[int]) -> list[float]:
        if not self.bias_enabled("attention"):
            return [1.0] * len(choices)
        # Salience changes BUY candidate selection, not the price forecast.
        return [1 + a.biases.attention * min(8, abs(obs["market"][j]["momentum_5d"]) * 40) for j in choices]

    def accept_beliefs(self, a: Resident, incoming: list[float]) -> list[float]:
        if not self.bias_enabled("confirmation"):
            return incoming[:]
        result = []
        for old, new in zip(a.beliefs, incoming):
            opposing = old * new < 0
            rate = 1 - .8 * a.biases.confirmation if opposing else 1.0
            result.append(old + rate * (new - old))
        return result

    def economy(self) -> dict:
        order = list(self.residents)
        self.rng.shuffle(order)
        unemployed, shortages, consumption = 0, 0, 0
        for f in self.firms:
            f.production = f.sales = 0.0
            f.wage_bill = 0
        opening = [f.cash for f in self.firms]
        for a in order:
            f = self.firms[a.sector]
            if self.life_events and self.rng.random() < .002:
                a.emergency_remaining += self.rng.randint(500, 5000)
                a.remember(self.day, "unexpected_service_need", cents=a.emergency_remaining)
            absent = self.rng.random() < a.routine.absence_probability
            wage = max(1, int(f.wage * a.skill))
            if f.cash >= wage and not absent:
                f.cash -= wage
                tax = wage // 10
                self.government_cash += tax
                a.cash += wage - tax
                a.last_income = wage - tax
                a.income_total += a.last_income
                f.wage_bill += wage
                output = f.productivity * a.skill * max(.5, 1 - a.hunger * .2)
                f.inventory += output
                f.production += output
            else:
                a.last_income = 0
                unemployed += 1
                a.remember(self.day, "no_wage", reason="absence" if absent else "employer_cash_shortage")
                support = min(self.government_cash, sum(x.goods_price for x in self.firms) // 2)
                self.government_cash -= support
                a.cash += support
                a.income_total += support
            # Credit is funded by bank cash, with a matching principal claim.
            if a.cash < self.daily_needs(a) * 2:
                self.borrow(a, self.daily_needs(a) * 5)
            if a.debt:
                a.interest_due += max(1, a.debt // 1000)
                surplus = max(0, a.cash - self.daily_needs(a) * a.reserve_days)
                interest = min(surplus, a.interest_due)
                a.cash -= interest
                a.interest_due -= interest
                self.bank_cash += interest
                payment = min(a.debt, int(max(0, surplus - interest) * a.routine.loan_repayment_fraction))
                a.cash -= payment
                a.debt -= payment
                self.bank_cash += payment
        # Households buy goods actually produced by this society.
        self.rng.shuffle(order)
        for a in order:
            fulfilled = 0.0
            for j, f in enumerate(self.firms):
                desire = a.routine.consumption_basket[j]
                bought = min(desire, f.inventory, a.cash // f.goods_price)
                if bought > 0:
                    cost = int(bought * f.goods_price)
                    a.cash -= cost
                    f.cash += cost
                    f.inventory -= bought
                    f.sales += bought
                    a.consumption_total += cost
                    consumption += cost
                fulfilled += min(1.0, bought / desire)
            if fulfilled < 2.99:
                shortages += 1
            a.hunger = .8 * a.hunger + .2 * (1 - fulfilled / 3)
            # A bill is settled only against available service production.
            f = self.firms[2]
            settled = min(a.emergency_remaining, a.cash, int(f.inventory * f.goods_price))
            if settled:
                a.cash -= settled
                f.cash += settled
                a.emergency_remaining -= settled
                quantity = settled / f.goods_price
                f.inventory -= quantity
                f.sales += quantity
                a.consumption_total += settled
                consumption += settled
        for j, f in enumerate(self.firms):
            # Operating profit excludes shareholder payouts.
            f.profit_ema = .9 * f.profit_ema + .1 * (f.cash - opening[j])
            f.dividend_per_share = 0
            if self.day % 20 == 0:
                payout = max(0, f.cash - max(1, f.wage_bill) * 30) // 4
                per_share = payout // f.supply
                if per_share:
                    f.cash -= per_share * f.supply
                    for a in self.residents:
                        income = per_share * a.shares[j]
                        a.cash += income
                        a.income_total += income
                    f.dividend_per_share = per_share
            # Goods perish/depreciate; money and financial shares do not.
            f.inventory *= .98
        return {"no_wage_residents": unemployed, "consumption_shortages": shortages,
                "consumption_cents": consumption}

    def observation(self, a: Resident) -> dict:
        index = max(0, len(self.prices) - 1 - a.information_lag)
        observed = self.prices[index]
        old = self.prices[max(0, index - 5)]
        market = []
        for j, f in enumerate(self.firms):
            # Public accounts are available daily. Price observations can lag.
            fundamental = max(20, (f.cash + int(f.inventory * f.goods_price)
                                  + max(0, f.profit_ema) * 60) / f.supply)
            peers = statistics.fmean(self.residents[i].signals[j] for i in a.peers) if a.peers else 0.0
            recent = self.prices[max(0, index - 10):index + 1]
            returns = [b[j] / p[j] - 1 for p, b in zip(recent, recent[1:])]
            market.append({"sector": f.sector, "observed_price": observed[j],
                           "momentum_5d": observed[j] / old[j] - 1,
                           "value_gap": max(-1, min(1, fundamental / observed[j] - 1)),
                           "dividend_yield_today": f.dividend_per_share / observed[j],
                           "peer_signal": peers if self.bias_enabled("herding") else 0.0,
                           "volatility_10d": statistics.pstdev(returns) if returns else .005,
                           "owned_shares": a.shares[j], "purchase_price": a.cost_basis[j],
                           "firm_cash": f.cash, "operating_profit_ema": f.profit_ema})
        return {"day": self.day, "resident_id": a.id, "cash_cents": a.cash,
                "debt_cents": a.debt + a.interest_due,
                "income_cents": a.last_income, "unmet_needs": a.hunger,
                "unpaid_emergency_cents": a.emergency_remaining,
                "market": market, "memories": a.memories[-8:],
                "routine": asdict(a.routine), "previous_thought": asdict(a.thought)}

    def signal(self, a: Resident, j: int, obs: dict) -> tuple[float, str]:
        m = obs["market"][j]
        w = list(a.factor_weights)
        preferred = {"value": 0, "momentum": 1, "income": 2, "social": 3, "cautious": 0}[a.style]
        w[preferred] += .4
        value = m["value_gap"]
        momentum = max(-1, min(1, m["momentum_5d"] * 8))
        dividend = min(1, m["dividend_yield_today"] * 100)
        social = m["peer_signal"] * a.conformity
        noise = self.rng.gauss(0, .15)
        score = sum(x * y for x, y in zip(w, (value, momentum, dividend, social, noise))) / sum(w)
        score += .25 * a.beliefs[j]
        reason = a.style
        if self.bias_enabled("recency"):
            score += .15 * a.biases.recency * momentum
            if momentum:
                reason += "+recency"
        if self.bias_enabled("anchoring"):
            anchor = a.cost_basis[j] if a.shares[j] else self.prices[0][j]
            gap = max(-1, min(1, anchor / m["observed_price"] - 1))
            score += .10 * a.biases.anchoring * gap
            if gap:
                reason += "+anchoring"
        if self.bias_enabled("loss_aversion") and score > 0:
            # Compare a simple prospective purchase's +/- volatility outcomes
            # with its lambda=1 counterfactual. This is a simplified gamble,
            # not a full estimated prospect-theory trading policy.
            mu, sigma = score * .04, max(.005, m["volatility_10d"])
            with_loss = .5 * (prospect_value(mu + sigma, a.loss_aversion) + prospect_value(mu - sigma, a.loss_aversion))
            neutral = .5 * (prospect_value(mu + sigma, 1) + prospect_value(mu - sigma, 1))
            penalty = min(0, 3 * (1 - a.risk) * (with_loss - neutral))
            score = max(0, score + penalty)
            if penalty:
                reason += "+loss_aversion"
        multiplier = self.conviction_multiplier(a)
        score *= multiplier
        if multiplier > 1:
            reason += "+overconfidence"
        buffer = self.daily_needs(a) * a.reserve_days
        urgent = a.cash < buffer + a.emergency_remaining
        if urgent:
            score -= .35 + .35 * min(1, (buffer + a.emergency_remaining - a.cash) / max(1, buffer))
            reason = "cash_need"
        adjusted = self.disposition_score(a, j, score, m["observed_price"], urgent)
        if adjusted != score:
            reason += "+disposition"
        score = adjusted
        if self.life_events and self.rng.random() < a.impulsivity * (1 + abs(a.confidence)):
            if m["momentum_5d"] > .02:
                score += .45
                reason += "+chase_rally"
            elif m["momentum_5d"] < -.02:
                score -= .45
                reason += "+panic"
            else:
                score += self.rng.choice((-.25, .25))
                reason += "+impulse"
        return max(-1, min(1, score)), reason

    def orders(self) -> list[Order]:
        orders, new_signals = [], {}
        ids = list(range(len(self.residents)))
        self.rng.shuffle(ids)
        for i in ids:
            a = self.residents[i]
            obs = self.observation(a)
            signals, reasons = [], []
            for j in range(3):
                s, reason = self.signal(a, j, obs)
                signals.append(s)
                reasons.append(reason)
            new_signals[i] = signals
            urgent = a.cash < self.daily_needs(a) * a.reserve_days + a.emergency_remaining
            if not urgent and self.day - a.routine.last_market_review < a.routine.market_review_interval:
                continue
            activity = min(.95, a.attention * self.conviction_multiplier(a))
            if not urgent and (self.rng.random() >= activity or self.rng.random() < a.patience * .35):
                continue
            a.routine.last_market_review = self.day
            j = a.interest if self.rng.random() < .45 else self.rng.randrange(3)
            if signals[j] > 0 and self.bias_enabled("attention"):
                choices = [k for k in range(3) if signals[k] > 0]
                j = self.rng.choices(choices, weights=self.attention_weights(a, obs, choices))[0]
            s, reason = signals[j], reasons[j]
            if abs(s) < .08:
                continue
            side = "buy" if s > 0 else "sell"
            observed = obs["market"][j]["observed_price"]
            limit = max(20, int(observed * (1 + .06 * s)))
            if side == "buy":
                # A buy cannot use another asset's anticipated sale proceeds.
                buffer = self.daily_needs(a) * a.reserve_days
                investable = max(0, a.cash - buffer - a.emergency_remaining)
                budget = int(investable * (.02 + a.risk * .08) * abs(s) * self.conviction_multiplier(a))
                quantity = min(budget // (limit + self.fee(limit)), max(1, int(len(self.residents) * .02)))
            else:
                quantity = min(a.shares[j], max(1, int(a.shares[j] * (.03 + .17 * abs(s)))))
            if quantity:
                orders.append(Order(a.id, j, side, quantity, limit, len(orders), reason))
                self.decisions.append({"day": self.day, **asdict(orders[-1]), "signal": s})
        # All observe the previous signals, not earlier agents in this loop.
        for i, signals in new_signals.items():
            self.residents[i].signals = signals
        return orders

    def settle(self, buy: Order, sell: Order, quantity: int, price: int) -> int:
        if buy.resident == sell.resident or buy.sector != sell.sector or quantity <= 0:
            return 0
        if buy.side != "buy" or sell.side != "sell" or min(buy.quantity, sell.quantity) <= 0:
            raise ValueError("Invalid order sides or quantities")
        if price > buy.limit or price < sell.limit or price <= 0:
            raise ValueError("Fill violates limit price")
        buyer, seller = self.residents[buy.resident], self.residents[sell.resident]
        j = buy.sector
        q = min(quantity, buy.quantity, sell.quantity, seller.shares[j])
        # Adjust affordability with exact transaction-level rounding.
        q = min(q, buyer.cash // price)
        while q > 0 and q * price + self.fee(q * price) > buyer.cash:
            q -= 1
        if not q:
            return 0
        value, fee = q * price, self.fee(q * price)
        previous = buyer.shares[j]
        buyer.cost_basis[j] = (buyer.cost_basis[j] * previous + value + fee) / (previous + q)
        old_basis = seller.cost_basis[j]
        buyer.cash -= value + fee
        seller.cash += value - fee
        self.government_cash += 2 * fee
        buyer.trade_cashflow -= value + fee
        seller.trade_cashflow += value - fee
        buyer.shares[j] += q
        seller.shares[j] -= q
        if not seller.shares[j]:
            seller.cost_basis[j] = 0.0
        # Recent personal outcomes alter future conviction.
        pnl = (price - old_basis) / max(1, old_basis)
        # Self-attribution: successes increase conviction more than losses
        # reduce it. Confidence is distinct from genuine profitability.
        outcome = 1.0 if pnl > 0 else (-.35 if self.bias_enabled("overconfidence") else -1.0) if pnl < 0 else 0.0
        seller.confidence = max(-1, min(1, .85 * seller.confidence + .15 * outcome))
        seller.remember(self.day, "sale", sector=j, shares=q, pnl_fraction=pnl)
        buyer.remember(self.day, "purchase", sector=j, shares=q, price=price)
        buy.quantity -= q
        sell.quantity -= q
        self.firms[j].price = price
        self.transactions.append({"day": self.day, "sector": j, "buyer": buyer.id,
                                  "seller": seller.id, "shares": q, "price_cents": price,
                                  "fee_each_cents": fee})
        return q

    def exchange(self, orders: list[Order]) -> None:
        for j in range(3):
            buys = sorted((o for o in orders if o.sector == j and o.side == "buy"),
                          key=lambda o: (-o.limit, o.sequence))
            sells = sorted((o for o in orders if o.sector == j and o.side == "sell"),
                           key=lambda o: (o.limit, o.sequence))
            for buy in buys:
                for sell in sells:
                    if not buy.quantity:
                        break
                    if sell.limit > buy.limit:
                        break
                    if not sell.quantity or sell.resident == buy.resident:
                        continue
                    # The earlier order is the resting order's execution price.
                    price = buy.limit if buy.sequence < sell.sequence else sell.limit
                    self.settle(buy, sell, min(buy.quantity, sell.quantity), price)
        # Unfilled orders expire at the day's end. No invisible counterparties.

    def step(self, llm=None) -> dict:
        self.day += 1
        start = len(self.transactions)
        previous_totals = [sum(a.shares[j] for a in self.residents) for j in range(3)]
        econ = self.economy()
        if llm is not None:
            llm.update(self)
        self.exchange(self.orders())
        for a in self.residents:
            a.beliefs = [v * .98 for v in a.beliefs]
        self.prices.append([f.price for f in self.firms])
        self.validate()
        trades = self.transactions[start:]
        row = {"day": self.day, **econ, "trades": len(trades),
               "volume_shares": sum(t["shares"] for t in trades),
               "cash_conservation_error": self.total_cash() - self.initial_cash,
               "aggregate_resident_net_shares": [sum(a.shares[j] for a in self.residents) - previous_totals[j] for j in range(3)],
               "household_cash_cents": sum(a.cash for a in self.residents),
               "bank_cash_cents": self.bank_cash,
               "household_debt_cents": sum(a.debt + a.interest_due for a in self.residents),
               "government_cash_cents": self.government_cash,
               "prices_cents": [f.price for f in self.firms],
               "firms": [asdict(f) for f in self.firms]}
        self.rows.append(row)
        return row

    def checkpoint(self) -> dict:
        return {"version": VERSION, "seed": self.seed, "day": self.day,
                "settings": {"herding": self.herding, "loss_aversion": self.loss_aversion,
                             "life_events": self.life_events, "fee_bps": self.fee_bps,
                             "disabled_biases": list(self.disabled_biases)},
                "residents": [asdict(a) for a in self.residents],
                "firms": [asdict(f) for f in self.firms], "bank_cash": self.bank_cash,
                "government_cash": self.government_cash, "initial_cash": self.initial_cash,
                "initial_wealth": self.initial_wealth, "prices": self.prices,
                "rows": self.rows, "transactions": self.transactions,
                "decisions": self.decisions, "llm_audit": self.llm_audit,
                "events": self.events, "llm_cursor": self.llm_cursor,
                "random_state": self.rng.getstate()}

    @classmethod
    def restore(cls, state: dict) -> Society:
        if state["version"] != VERSION:
            raise ValueError("Unsupported society version")
        obj = cls.__new__(cls)
        obj.seed, obj.day = state["seed"], state["day"]
        for name, value in state["settings"].items():
            setattr(obj, name, value)
        obj.residents = []
        for values in state["residents"]:
            a = dict(values)
            a["routine"] = Routine(**a["routine"])
            a["thought"] = Thought(**a["thought"])
            a["biases"] = Biases(**a["biases"])
            obj.residents.append(Resident(**a))
        obj.firms = [Firm(**f) for f in state["firms"]]
        for name in ("bank_cash", "government_cash", "initial_cash", "initial_wealth",
                     "prices", "rows", "transactions", "decisions", "llm_audit", "events", "llm_cursor"):
            setattr(obj, name, state[name])
        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, list) else value
        obj.rng = random.Random()
        obj.rng.setstate(tuples(state["random_state"]))
        obj.validate()
        return obj

    def report(self) -> dict:
        returns = [[b[j] / a[j] - 1 for a, b in zip(self.prices, self.prices[1:])] for j in range(3)]
        marked = [self.wealth(a) / self.initial_wealth[a.id] - 1 for a in self.residents]
        styles = {}
        for style in STYLES:
            ids = [a.id for a in self.residents if a.style == style]
            styles[style] = {"people": len(ids), "median_marked_wealth_change": statistics.median(marked[i] for i in ids) if ids else None}
        return {"version": VERSION, "people": len(self.residents), "days": self.day,
                "seed": self.seed, "verified_real_market_alpha": False,
                "engine": "native rules with optional local LLM beliefs; not MiroFish/OASIS",
                "currency": "synthetic cents; 100 cents = 1 simulated currency unit",
                "initial_cash_cents": self.initial_cash, "final_cash_cents": self.total_cash(),
                "cash_conservation_error": self.total_cash() - self.initial_cash,
                "trade_count": len(self.transactions),
                "volume_shares": sum(t["shares"] for t in self.transactions),
                "prices_initial": self.prices[0], "prices_final": self.prices[-1],
                "daily_return_stdev": [statistics.pstdev(v) if v else None for v in returns],
                "median_marked_wealth_change": statistics.median(marked),
                "styles": styles, "decision_reasons": dict(Counter(x["reason"] for x in self.decisions)),
                "llm_successes": sum(x["status"] == "ok" for x in self.llm_audit),
                "llm_failures": sum(x["status"] != "ok" for x in self.llm_audit),
                "llm_unique_residents": len({x["resident"] for x in self.llm_audit if x["status"] == "ok"}),
                "cognitive_triggers": dict(Counter(reason for x in self.llm_audit for reason in x["input"]["reflection_triggers"])),
                "thoughts": "per-person persistent belief/thesis; routines continue between reflections",
                "behavioral_biases": {"enabled": [b for b in BIAS_NAMES if self.bias_enabled(b)],
                                      "references": BIAS_SOURCES,
                                      "mechanisms": BIAS_MECHANISMS,
                                      "strengths_empirically_calibrated": False,
                                      "implementation": "bounded mechanistic approximations; directional unit checks, not empirical replication"},
                "limitations": ["uncalibrated synthetic personality and economic priors",
                                "three representative firms; fixed goods prices and wage rates",
                                "closed resident-only stock market: aggregate personal net purchases are zero",
                                "marked wealth includes wages, consumption, dividends and debt: not trading alpha",
                                "no QuantInSight institutional policy or MiroFish integration yet",
                                "no empirical human validation or real-market prediction test",
                                "daily matching with expiring orders; no intraday exchange microstructure"]}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Local LLM endpoint redirects are disabled")


class LocalBeliefs:
    """Event-cadence cognition only, with isolated same-day observations.

    LLM text cannot move cash/shares or directly set exchange prices. Failed
    calls retain previous beliefs and are explicitly recorded in the audit.
    """
    def __init__(self, url: str, model: str, agents: int = 30, every: int = 10,
                 workers: int = 2, transport=None, max_calls: int | None = None):
        parsed = urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"} or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Only a credential-free local HTTP LLM endpoint is allowed")
        if not model or min(agents, every, workers) < 1 or workers > 8:
            raise ValueError("Invalid local LLM settings")
        if max_calls is not None and max_calls < 1:
            raise ValueError("max_calls must be positive")
        self.url, self.model = url.rstrip("/"), model
        self.agents, self.every, self.workers = agents, every, workers
        self.transport = transport or self.request
        self.max_calls, self.calls_made = max_calls, 0

    def request(self, messages: list[dict]) -> dict:
        body = {"model": self.model, "messages": messages, "temperature": .5,
                "max_tokens": 350, "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning": {"enabled": False}, "stream": False}
        req = Request(self.url + "/chat/completions", json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
        with build_opener(NoRedirect).open(req, timeout=90) as response:
            payload = json.loads(response.read())
        if payload["choices"][0].get("finish_reason") == "length":
            raise ValueError("Truncated model reply")
        content = payload["choices"][0]["message"]["content"].strip()
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        return json.loads(content)

    @staticmethod
    def validate_reply(reply: dict, resident: int) -> tuple[list[float], str]:
        if not isinstance(reply, dict) or type(reply.get("resident_id")) is not int or reply.get("resident_id") != resident:
            raise ValueError("Reply belongs to another resident")
        scores = reply.get("beliefs")
        if not isinstance(scores, list) or len(scores) != 3:
            raise ValueError("Expected three sector beliefs")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not -1 <= x <= 1 for x in scores):
            raise ValueError("Invalid belief score")
        reason = reply.get("reason")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 600:
            raise ValueError("Missing or oversized reason")
        return [float(x) for x in scores], reason

    def update(self, society: Society) -> None:
        if self.max_calls is not None and self.calls_made >= self.max_calls:
            return
        n = len(society.residents)
        candidates = []
        for a in society.residents:
            obs = society.observation(a)
            since = society.day - a.thought.last_reflection_day
            if a.thought.last_attempt_day >= 0 and society.day - a.thought.last_attempt_day < 2:
                continue
            triggers = []
            recent = [m for m in a.memories if m["day"] == society.day]
            if a.emergency_remaining or any(m["event"] == "unexpected_service_need" for m in recent):
                triggers.append("unexpected_expense")
            if any(m["event"] == "no_wage" and m.get("reason") == "employer_cash_shortage" for m in recent):
                triggers.append("employer_cannot_pay")
            if a.cash < society.daily_needs(a) * 3 and a.debt:
                triggers.append("cash_and_debt_pressure")
            if max(abs(m["momentum_5d"]) for m in obs["market"]) > .06 + .06 * a.patience:
                triggers.append("large_market_move")
            if society.bias_enabled("herding") and max(abs(m["peer_signal"] - a.beliefs[j]) for j, m in enumerate(obs["market"])) > .45:
                triggers.append("peer_disagreement")
            if triggers and since < max(2, self.every // 3):
                # Repeated market moves cannot trigger repeated thoughts every
                # tick; immediate expense/job events can bypass this cooldown.
                triggers = [t for t in triggers if t in {"unexpected_expense", "employer_cannot_pay"}]
            if not triggers and since >= self.every:
                triggers = ["scheduled_personal_review"]
            if triggers:
                priority = (0 if any(t in {"unexpected_expense", "employer_cannot_pay", "cash_and_debt_pressure"} for t in triggers) else
                            1 if triggers != ["scheduled_personal_review"] else 2)
                candidates.append((priority, -since, (a.id - society.llm_cursor) % n, a.id, obs, triggers))
        candidates.sort(key=lambda x: x[:3])
        remaining = self.max_calls - self.calls_made if self.max_calls is not None else self.agents
        chosen = candidates[:min(self.agents, n, remaining)]
        society.events.append({"day": society.day, "type": "cognition_budget",
                               "eligible": len(candidates), "selected": len(chosen),
                               "deferred": len(candidates) - len(chosen)})
        if not chosen:
            return
        society.llm_cursor = (chosen[-1][3] + 1) % n
        self.calls_made += len(chosen)
        inputs = []
        for _, _, _, i, obs, triggers in chosen:
            a = society.residents[i]
            # Background, personality, personal accounts and observable peers;
            # never another resident's cash, future prices or private memories.
            profile = {k: getattr(a, k) for k in ("id", "age", "background", "sector", "skill", "style", "risk", "patience", "loss_aversion", "conformity", "attention", "impulsivity", "reserve_days", "factor_weights", "confidence")}
            profile["behavioral_biases"] = asdict(a.biases)
            profile["enabled_biases"] = [b for b in BIAS_NAMES if society.bias_enabled(b)]
            inputs.append({"profile": profile, "observation": obs, "reflection_triggers": triggers})
        def call(payload):
            resident = payload["profile"]["id"]
            serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            audit = {"day": society.day, "resident": resident, "model": self.model,
                     "input_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
                     "input": payload}
            start = time.monotonic()
            try:
                messages = [{"role": "system", "content": "You are this fictional resident of a closed society. Reflect on sector prospects from your personality, experience and available information. Facts in observation are authoritative; do not invent future events. Distinguish your personal expense needs from a firm's value. Execution rules separately apply the listed bias mechanisms, cash needs and confirmation filtering: do not fabricate trades or claim numerical human probabilities. Reply JSON only: resident_id integer; beliefs [food,housing,services] each -1 to 1; reason short Korean explanation. These are proposed investment opinions, not executed orders."},
                            {"role": "user", "content": serialized}]
                raw = self.transport(messages)
                scores, reason = self.validate_reply(raw, resident)
                audit.update(status="ok", reply={"resident_id": resident, "beliefs": scores, "reason": reason})
            except Exception as exc:
                # Do not log exception text: endpoints can return sensitive data.
                audit.update(status="failed", error_type=type(exc).__name__)
            audit["duration_seconds"] = round(time.monotonic() - start, 3)
            return audit
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            results = list(pool.map(call, inputs))
        # Commit after the whole cohort finishes, preserving a common cutoff.
        for audit in results:
            society.llm_audit.append(audit)
            society.residents[audit["resident"]].thought.last_attempt_day = society.day
            if audit["status"] == "ok":
                a = society.residents[audit["resident"]]
                a.beliefs = society.accept_beliefs(a, audit["reply"]["beliefs"])
                audit["effective_beliefs"] = a.beliefs[:]
                a.thought.thesis = audit["reply"]["reason"]
                a.thought.last_reflection_day = society.day
                a.thought.trigger_reasons = audit["input"]["reflection_triggers"]
                a.thought.successful_reflections += 1
                a.remember(society.day, "reflection", reason=audit["reply"]["reason"])


def save_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people", type=int, default=1000)
    parser.add_argument("--days", type=int, default=180, help="Additional simulated days")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-herding", action="store_true")
    parser.add_argument("--no-loss-aversion", action="store_true")
    parser.add_argument("--no-life-events", action="store_true")
    parser.add_argument("--disable-bias", choices=BIAS_NAMES, action="append", default=[], help="Repeat to disable individual biases or build a neutral control")
    parser.add_argument("--fee-bps", type=int, default=10, help="Per side simulated trading fee")
    parser.add_argument("--resume", type=Path, help="Checkpoint to continue into a NEW run")
    parser.add_argument("--llm-url")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-agents", type=int, default=30, help="Maximum resident reflections per simulated day")
    parser.add_argument("--llm-every", type=int, default=10, help="Personal periodic review interval; important events can trigger earlier")
    parser.add_argument("--llm-workers", type=int, default=2)
    parser.add_argument("--llm-max-calls", type=int, help="Optional total reflection call budget for this run")
    args = parser.parse_args()
    if args.days < 1 or args.days > 10000:
        parser.error("days must be 1..10000")
    if bool(args.llm_url) != bool(args.llm_model):
        parser.error("--llm-url and --llm-model must be supplied together")
    society = Society.restore(json.loads(args.resume.read_text())) if args.resume else Society(
        args.people, args.seed, herding=not args.no_herding,
        loss_aversion=not args.no_loss_aversion, life_events=not args.no_life_events,
        fee_bps=args.fee_bps, disabled_biases=tuple(args.disable_bias))
    llm = LocalBeliefs(args.llm_url, args.llm_model, args.llm_agents,
                      args.llm_every, args.llm_workers, max_calls=args.llm_max_calls) if args.llm_url else None
    run = ROOT / "data" / "investor_society" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run.mkdir(parents=True, exist_ok=False)
    save_json(run / "initial_state.json", society.checkpoint())
    save_json(run / "bias_catalog.json", {b: {"enabled": society.bias_enabled(b),
              "mechanism": BIAS_MECHANISMS[b], "source": BIAS_SOURCES[b],
              "calibrated_strength": False} for b in BIAS_NAMES})
    save_json(run / "manifest.json", {"version": VERSION, "seed": society.seed,
              "people": len(society.residents), "additional_days": args.days,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cognition": "hybrid-local-LLM" if llm else "rules",
              "llm_model": args.llm_model, "llm_daily_budget": args.llm_agents if llm else 0,
              "llm_personal_review_interval": args.llm_every if llm else None,
              "llm_total_call_budget": args.llm_max_calls,
              "resume": str(args.resume) if args.resume else None,
              "purpose": "closed-economy mechanics; no claim of human fidelity or alpha"})
    start = time.monotonic()
    try:
        for _ in range(args.days):
            row = society.step(llm)
            if society.day % 20 == 0:
                print(json.dumps({"day": society.day, "prices_cents": row["prices_cents"],
                                  "trades": row["trades"], "cash_error": row["cash_conservation_error"]}), flush=True)
    finally:
        save_json(run / "checkpoint.json", society.checkpoint())
        save_json(run / "report.json", society.report())
        save_json(run / "residents.json", [asdict(a) for a in society.residents])
        for filename, rows in (("daily", society.rows), ("trades", society.transactions),
                               ("decisions", society.decisions), ("llm_audit", society.llm_audit)):
            with (run / (filename + ".jsonl")).open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"run": str(run), "elapsed_seconds": round(time.monotonic() - start, 2),
                      "report": society.report()}, ensure_ascii=False))


if __name__ == "__main__":
    main()

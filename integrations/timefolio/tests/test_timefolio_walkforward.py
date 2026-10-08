from copy import deepcopy

import numpy as np
import pytest
import torch

from test_timefolio_heatmap import synthetic_panel
from quant.timefolio_heatmap_replay import replay
from quant.timefolio_heatmap_walkforward import AlternativeNet, encode_images, fold_masks


def test_monthly_purges_and_rolling_window():
    di = np.repeat(np.arange(150), 3)
    masks = fold_masks(di, np.ones(len(di)), np.ones(len(di), bool), 5, 100, 122, 60)
    train, val, refit, pred, inner = masks
    assert np.all(di[train] >= 40) and np.max(di[train] + 5) < 80
    assert np.min(di[val]) == 80 and np.max(di[val] + 5) < 100
    assert np.all(di[refit] >= 40) and np.max(di[refit] + 5) < 100
    assert np.min(di[pred]) == 99 and np.max(di[pred]) == 120
    assert np.min(di[inner]) == 80 and np.max(di[inner]) == 99
    for mask in masks: assert np.all(mask.reshape(-1, 3) == mask.reshape(-1, 3)[:, :1])


def test_sector_encoding_does_not_pool_future_days_or_mutate_input():
    rng = np.random.default_rng(2)
    raw = rng.integers(0, 256, (16, 32, 65), dtype=np.uint8)
    raw[:, 14] = 255; raw[0, 14, 0] = 128
    original = raw.copy(); ci = np.tile(np.arange(8), 2); di = np.repeat([0, 1], 8)
    sec = np.repeat([10, 20], 4); vol = np.full((8, 2), .02)
    before = encode_images(raw, ci, di, sec, vol, "sector_center")
    np.testing.assert_array_equal(raw, original)
    raw[8:] = 250
    after = encode_images(raw, ci, di, sec, vol, "sector_center")
    np.testing.assert_array_equal(before[:8], after[:8])
    assert np.all(before[0, :14, 0] == 128)
    np.testing.assert_array_equal(before[:, 24:], original[:, 24:])


@pytest.mark.parametrize("arch", ["cnn", "split", "tcn", "mlp"])
def test_architectures_train_and_score_without_batch_interference(arch):
    torch.set_num_threads(2); torch.manual_seed(4)
    m = AlternativeNet(arch); x = torch.randn(3, 1, 32, 65)
    loss = m(x).square().mean(); loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    m.eval()
    with torch.no_grad():
        together = m(x); alone = torch.cat([m(row[None]) for row in x])
    torch.testing.assert_close(together, alone, rtol=1e-5, atol=1e-6)


def test_gross_schedule_is_lagged_and_risk_off_liquidates():
    p, ix = synthetic_panel(); scores = np.ones_like(p["close"])
    schedule = np.full(len(ix["dates"]), .8); schedule[2:] = 0
    r = replay(p, ix, scores, ix["dates"][1], ix["dates"][4],
               rebalance=99, gross_schedule=schedule, return_trades=True)
    assert r["daily"][1]["holdings"] > 0  # The second day's close signal acts tomorrow.
    assert r["daily"][2]["holdings"] == 0
    assert all(t["date"] == ix["dates"][3] for t in r["trades"] if t["side"] == "sell")


def test_order_budget_retries_on_non_rebalance_day_and_never_exceeds_cap():
    p, ix = synthetic_panel(); scores = np.broadcast_to(np.arange(12)[:, None], p["close"].shape)
    r = replay(p, ix, scores, ix["dates"][1], ix["dates"][7],
               rebalance=99, max_orders=3, return_trades=True)
    counts = {}
    for t in r["trades"]: counts[t["date"]] = counts.get(t["date"], 0) + 1
    assert max(counts.values()) <= 3
    assert ix["dates"][2] in counts and r["daily"][-1]["holdings"] > 3
    assert r["metrics"]["order_budget_deferred_requests"] > 0


def test_all_newly_designated_candidates_still_allow_existing_sales():
    p, ix = synthetic_panel(); p["trade_allowed"] = np.ones_like(p["eligible"])
    p["trade_allowed"][:, 2:] = False
    r = replay(p, ix, np.ones_like(p["close"]), ix["dates"][1], ix["dates"][3],
               rebalance=1, return_trades=True)
    assert r["daily"][0]["holdings"] > 0 and r["daily"][1]["holdings"] == 0
    assert all(t["side"] == "sell" for t in r["trades"] if t["date"] >= ix["dates"][2])


def test_regime_series_does_not_use_future_prices_or_caps():
    from quant.timefolio_heatmap_walkforward_eval import market_regimes
    p, _ = synthetic_panel(days=60)
    p['r1'][:] = np.random.default_rng(4).normal(0, .02, p['r1'].shape)
    before, returns = market_regimes(p)
    later = deepcopy(p); later['r1'][:, 31:] = -.2; later['market_cap'][:, 31:] *= 30
    after, returns_after = market_regimes(later)
    np.testing.assert_array_equal(returns[:31], returns_after[:31])
    for key in ['trend','volatility']:
        np.testing.assert_array_equal(before[key][:31], after[key][:31])
        assert np.all((before[key] >= .3) & (before[key] <= .8))


def test_family_correction_and_synthetic_signal_detection():
    from quant.timefolio_heatmap_walkforward_eval import family_bootstrap, block_indices
    rng = np.random.default_rng(23)
    idx = block_indices(53, 5, 20, rng)
    assert idx.shape == (20, 53) and idx.min() >= 0 and idx.max() < 53
    # Adjacent dates stay together inside each bootstrap block, including wraparound.
    assert np.all((idx[:, 1:5]-idx[:, :4]) % 53 == 1)
    noise = rng.normal(0, .01, (250, 8)); noise -= noise.mean(0)
    a = np.column_stack([noise, noise[:, 0] + .02])
    result = family_bootstrap(a, block=5, draws=499, seed=42)
    assert np.all(result['adjusted_p'] >= result['marginal_p'])
    assert np.all(result['adjusted_p'][:8] > .05)
    assert result['adjusted_p'][-1] < .01 and result['simultaneous_lower95'][-1] > 0


def test_independent_execution_audit_handles_float32_prices_and_catches_bad_fill():
    from quant.timefolio_heatmap_walkforward_audit import additional_checks
    p, ix = synthetic_panel()
    p['exec_price'] = np.full_like(p['exec_price'],12345.67,dtype=np.float32)
    schedule = np.full(len(ix['dates']),.6)
    r = replay(p,ix,np.ones_like(p['close']),ix['dates'][1],ix['dates'][4],
               gross_schedule=schedule,max_orders=4,return_trades=True)
    correct = additional_checks(p,ix,r,schedule,max_orders=4)
    assert correct['additional_errors'] == []
    corrupt = deepcopy(r); corrupt['trades'][0]['price'] += 10
    bad = additional_checks(p,ix,corrupt,schedule,max_orders=4)
    assert any(x['kind'] == 'fill_price' for x in bad['additional_errors'])

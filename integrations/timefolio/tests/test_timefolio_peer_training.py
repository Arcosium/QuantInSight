import numpy as np
import pytest
import torch

from quant.timefolio_heatmap_peer_training import ContextNet, train_context


@pytest.mark.parametrize('arch', ['cnn', 'mlp'])
def test_blank_has_identical_initial_parameters_but_cannot_read_context(arch):
    torch.manual_seed(17); blank = ContextNet(arch, 'blank').eval()
    torch.manual_seed(17); peer = ContextNet(arch, 'peer').eval()
    for key, value in blank.state_dict().items(): assert torch.equal(value, peer.state_dict()[key])
    x = torch.linspace(-1, 1, 2 * 2 * 32 * 8).reshape(2, 2, 32, 8)
    changed = x.clone(); changed[:, 1] = changed[:, 1] * -3 + 1
    with torch.no_grad():
        assert torch.equal(blank(x), blank(changed))
        assert not torch.equal(peer(x), peer(changed))
        empty = x.clone(); empty[:, 1] = 0
        torch.testing.assert_close(blank(x), peer(empty), rtol=0, atol=0)
    z = x.clone().requires_grad_(); blank(z).sum().backward()
    assert torch.count_nonzero(z.grad[:, 1]) == 0 and torch.count_nonzero(z.grad[:, 0]) > 0


@pytest.mark.parametrize('mode', ['blank', 'peer'])
def test_future_images_and_labels_do_not_change_fit_or_epoch_selection(mode):
    rng = np.random.default_rng(51)
    x = torch.from_numpy(rng.integers(0, 256, (40, 2, 32, 8), dtype=np.uint8))
    y = np.tile(np.linspace(-1, 1, 10), 4).astype(np.float32)
    di = np.repeat(np.arange(4), 10); train, val = di < 2, di == 2
    cfg = dict(architecture='mlp', context_mode=mode, objective='pairwise', seed=17, lr=.0007, epochs=2)
    first, first_fit = train_context(x, y, train, val, cfg, di)
    changed_x = x.clone(); changed_x[di == 3] = 255 - changed_x[di == 3]
    changed_y = y.copy(); changed_y[di == 3] = 100
    second, second_fit = train_context(changed_x, changed_y, train, val, cfg, di)
    assert first_fit == second_fit
    for key, value in first.state_dict().items(): assert torch.equal(value, second.state_dict()[key])

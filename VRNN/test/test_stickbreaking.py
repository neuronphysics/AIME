
import torch, pytest
from dpgmm_stickbreaking_prior_vrnn import AdaptiveStickBreaking

def test_stickbreaking_simple(device):
    model = AdaptiveStickBreaking(hidden_dim=32, max_K=8, device=device)
    h = torch.randn(4, 32, device=device, requires_grad=True)
    pi, params = model(h, n_samples=5, use_rand_perm=False)
    assert pi.shape == (4, 8), "pi shape must be [B, K]"
    assert torch.allclose(pi.sum(dim=-1), torch.ones(4, device=device), atol=1e-4)
    # gradient
    loss = (pi ** 2).sum()
    loss.backward()
    assert h.grad is not None

def test_effective_components(device):
    model = AdaptiveStickBreaking(hidden_dim=16, max_K=10, device=device)
    h = torch.zeros(2, 16, device=device)
    pi, _ = model(h, n_samples=1, use_rand_perm=False)
    # should still sum to 1 and have at least 1 effective component
    from dpgmm_stickbreaking_prior_vrnn import DPGMMVariationalRecurrentAutoencoder
    dummy = DPGMMVariationalRecurrentAutoencoder(
        image_size=64, input_channels=3, latent_dim=8, hidden_dim=32, max_K=10, action_dim=4, device=device
    )
    eff = dummy.prior.get_effective_components(pi)
    assert (eff >= 1).all()

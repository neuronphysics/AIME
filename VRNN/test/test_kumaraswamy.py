
import torch, math, pytest
from Kumaraswamy import KumaraswamyStable

def test_kumaraswamy_basic(device):
    a = torch.full((5,3), 2.0, device=device, dtype=torch.float32, requires_grad=True)
    b = torch.full((5,3), 3.0, device=device, dtype=torch.float32, requires_grad=True)
    dist = KumaraswamyStable(a, b)
    s = dist.rsample((10,))  # [10,5,3]
    assert s.min() >= 0 and s.max() <= 1, "Samples must lie in (0,1)"
    lp = dist.log_prob(s).mean()
    assert torch.isfinite(lp), "log_prob should be finite"
    # gradient
    lp.backward()
    assert a.grad is not None and b.grad is not None

def test_kumaraswamy_mean_close_to_expected(device):
    a = torch.tensor([2.0], device=device)
    b = torch.tensor([3.0], device=device)
    dist = KumaraswamyStable(a, b)
    samples = dist.rsample((20000,))  # Monte Carlo
    mc_mean = samples.mean().item()
    # analytic mean: b*Gamma(1+1/a)*Gamma(b)/Gamma(1+1/a+b)
    import torch
    analytic = b * torch.lgamma(1 + 1/a).exp() * torch.lgamma(b).exp() / torch.lgamma(1 + 1/a + b).exp()
    assert abs(mc_mean - analytic.item()) < 0.05

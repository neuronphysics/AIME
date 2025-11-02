"""
Test DPGMM Sampling

Verifies that the DPGMM prior generates valid mixture distributions and
produces sensible samples.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.generative_prior import DPGMMPrior


def test_dpgmm_basic():
    """Test basic DPGMM functionality"""
    print("=" * 60)
    print("Test 1: Basic DPGMM Sampling")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    hidden_dim = 128
    latent_dim = 16
    max_components = 10

    # Initialize DPGMM prior
    prior = DPGMMPrior(
        max_components=max_components,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        device=device
    )

    # Generate random hidden state
    h = torch.randn(batch_size, hidden_dim, device=device)

    print(f"\nInput hidden state: {h.shape}")

    # Forward pass
    mixture_dist, params = prior(h)

    print(f"\nOutput parameters:")
    print(f"  Mixing weights π: {params['pi'].shape}")
    print(f"  Component means μ: {params['means'].shape}")
    print(f"  Component log-vars log(σ²): {params['log_vars'].shape}")
    print(f"  Concentration α: {params['alpha'].shape}")

    # Check mixing weights sum to 1
    pi_sum = params['pi'].sum(dim=-1)
    print(f"\n✓ Mixing weights sum: {pi_sum[0].item():.6f} (should be ~1.0)")
    assert torch.allclose(pi_sum, torch.ones_like(pi_sum), atol=1e-5), \
        f"Mixing weights don't sum to 1: {pi_sum}"

    # Check all weights are non-negative
    assert (params['pi'] >= 0).all(), "Negative mixing weights detected"
    print(f"✓ All mixing weights non-negative")

    # Sample from mixture
    z_samples = mixture_dist.sample((100,))  # [100, batch_size, latent_dim]
    print(f"\n✓ Sampled from mixture: {z_samples.shape}")

    # Check sample statistics
    z_mean = z_samples.mean(dim=0)
    z_std = z_samples.std(dim=0)
    print(f"  Sample mean range: ({z_mean.min().item():.3f}, {z_mean.max().item():.3f})")
    print(f"  Sample std range: ({z_std.min().item():.3f}, {z_std.max().item():.3f})")

    # Count effective components
    effective_k = prior.get_effective_components(params['pi'])
    print(f"\n✓ Effective components: {effective_k.float().mean().item():.1f} / {max_components}")

    print("\n" + "=" * 60)
    print("✓ Test 1 PASSED: Basic DPGMM sampling works")
    print("=" * 60)


def test_kl_divergence():
    """Test KL divergence computation"""
    print("\n" + "=" * 60)
    print("Test 2: KL Divergence Computation")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    hidden_dim = 128
    latent_dim = 16
    max_components = 10

    prior = DPGMMPrior(
        max_components=max_components,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        device=device
    )

    # Generate hidden state and prior
    h = torch.randn(batch_size, hidden_dim, device=device)
    mixture_dist, params = prior(h)

    # Generate mock posterior (Gaussian)
    posterior_mean = torch.randn(batch_size, latent_dim, device=device)
    posterior_logvar = torch.randn(batch_size, latent_dim, device=device) * 0.5

    print(f"\nPosterior mean: {posterior_mean.shape}")
    print(f"Posterior log-var: {posterior_logvar.shape}")

    # Compute KL divergence (Monte Carlo)
    kl_mc = prior.compute_kl_divergence_mc(
        posterior_mean,
        posterior_logvar,
        params,
        n_samples=50
    )

    print(f"\n✓ KL divergence (MC): {kl_mc.item():.4f}")
    assert kl_mc.item() >= 0, "KL divergence should be non-negative"
    print(f"✓ KL divergence is non-negative")

    # Compute stick-breaking KL
    kl_stickbreaking = prior.compute_kl_loss(params, params['alpha'], h)
    print(f"✓ Stick-breaking KL: {kl_stickbreaking.item():.4f}")
    assert kl_stickbreaking.item() >= 0, "Stick-breaking KL should be non-negative"

    total_kl = kl_mc + kl_stickbreaking
    print(f"\nTotal KL: {total_kl.item():.4f}")

    print("\n" + "=" * 60)
    print("✓ Test 2 PASSED: KL divergence computation works")
    print("=" * 60)


def test_context_dependence():
    """Test that prior adapts to different hidden states"""
    print("\n" + "=" * 60)
    print("Test 3: Context Dependence")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    hidden_dim = 128
    latent_dim = 16
    max_components = 10

    prior = DPGMMPrior(
        max_components=max_components,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        device=device
    )

    # Generate two very different hidden states
    h1 = torch.randn(batch_size, hidden_dim, device=device)
    h2 = torch.randn(batch_size, hidden_dim, device=device) * 2.0 + 5.0

    # Get priors for each
    _, params1 = prior(h1)
    _, params2 = prior(h2)

    # Check that mixing weights are different
    pi_diff = (params1['pi'] - params2['pi']).abs().mean()
    print(f"\nMixing weight difference: {pi_diff.item():.4f}")
    assert pi_diff.item() > 0.01, "Prior should adapt to different hidden states"
    print(f"✓ Prior adapts to context (π differs by {pi_diff.item():.4f})")

    # Check that component means are different
    means_diff = (params1['means'] - params2['means']).abs().mean()
    print(f"Component means difference: {means_diff.item():.4f}")
    assert means_diff.item() > 0.01, "Component means should differ"
    print(f"✓ Component means adapt to context")

    # Check effective components for each
    eff_k1 = prior.get_effective_components(params1['pi']).float().mean()
    eff_k2 = prior.get_effective_components(params2['pi']).float().mean()
    print(f"\nEffective components h1: {eff_k1.item():.1f}")
    print(f"Effective components h2: {eff_k2.item():.1f}")

    print("\n" + "=" * 60)
    print("✓ Test 3 PASSED: Prior adapts to context")
    print("=" * 60)


def test_gradient_flow():
    """Test that gradients flow through the prior"""
    print("\n" + "=" * 60)
    print("Test 4: Gradient Flow")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    hidden_dim = 128
    latent_dim = 16

    prior = DPGMMPrior(
        max_components=10,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        device=device
    )

    # Generate hidden state with gradient tracking
    h = torch.randn(batch_size, hidden_dim, device=device, requires_grad=True)

    # Forward pass
    mixture_dist, params = prior(h)

    # Compute a loss (e.g., entropy of mixing weights)
    loss = -(params['pi'] * torch.log(params['pi'] + 1e-8)).sum()

    # Backward pass
    loss.backward()

    print(f"\nLoss: {loss.item():.4f}")
    print(f"✓ Backward pass completed")

    # Check that gradients exist
    assert h.grad is not None, "No gradients for hidden state"
    print(f"✓ Gradients computed for hidden state")

    grad_norm = h.grad.norm().item()
    print(f"  Gradient norm: {grad_norm:.4f}")
    assert grad_norm > 0, "Gradients should be non-zero"

    print("\n" + "=" * 60)
    print("✓ Test 4 PASSED: Gradients flow correctly")
    print("=" * 60)


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("DPGMM PRIOR TEST SUITE")
    print("=" * 60)

    try:
        test_dpgmm_basic()
        test_kl_divergence()
        test_context_dependence()
        test_gradient_flow()

        print("\n" + "=" * 60)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("=" * 60)
        print("\nDPGMM Prior is working correctly!")
        print("Ready for integration with VRNN model.")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()

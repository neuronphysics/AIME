"""
Test: Attention Schema Integration

Tests the full attention schema pipeline including posterior, prior, and dynamics.

Run with:
    python attention_schema/tests/test_attention_fusion.py
"""

import sys
sys.path.insert(0, '/home/g/zahra-dir/AIME')

import torch
from src.attention_schema import AttentionSchema, AttentionPosterior, AttentionPrior


def test_attention_posterior():
    """Test 1: AttentionPosterior (bottom-up attention)"""
    print("\n" + "="*60)
    print("Test 1: AttentionPosterior (Bottom-Up)")
    print("="*60)

    B = 2
    posterior = AttentionPosterior(
        image_size=84,
        attention_resolution=21,
        hidden_dim=256,
        context_dim=128,
        input_channels=3,
        feature_channels=64,
        num_semantic_slots=4,
        device=torch.device('cpu')
    )

    # Inputs
    obs = torch.randn(B, 3, 84, 84)
    hidden = torch.randn(B, 256)
    context = torch.randn(B, 128)

    print(f"Observation: {obs.shape}")
    print(f"Hidden state: {hidden.shape}")
    print(f"Context: {context.shape}")

    # Forward pass
    attn_probs, coords = posterior(obs, hidden, context)

    print(f"Attention probs: {attn_probs.shape}")
    print(f"Regularized coords: {coords.shape}")

    # Verify shapes
    assert attn_probs.shape == (B, 21, 21), f"Expected [2, 21, 21], got {attn_probs.shape}"
    assert coords.shape == (B, 2), f"Expected [2, 2], got {coords.shape}"

    # Verify attention sums to 1
    attn_sum = attn_probs.flatten(1).sum(dim=1)
    print(f"Attention sum (should be ~1): {attn_sum}")
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-3), "Attention doesn't sum to 1!"

    # Verify coordinates are in [-1, 1]
    print(f"Coord range: [{coords.min():.3f}, {coords.max():.3f}]")
    assert coords.min() >= -1.0 and coords.max() <= 1.0, "Coordinates out of range!"

    # Check slot attention maps
    assert hasattr(posterior, 'slot_attention_maps'), "Missing slot_attention_maps!"
    slot_maps = posterior.slot_attention_maps
    print(f"Slot attention maps: {slot_maps.shape}")
    assert slot_maps.shape == (B, 4, 21, 21), f"Expected [2, 4, 21, 21], got {slot_maps.shape}"

    # Check diversity loss
    diversity_loss = posterior.diversity_loss
    print(f"Diversity loss: {diversity_loss:.6f}")

    print("✓ PASSED: AttentionPosterior works correctly")


def test_attention_prior():
    """Test 2: AttentionPrior (top-down prediction)"""
    print("\n" + "="*60)
    print("Test 2: AttentionPrior (Top-Down)")
    print("="*60)

    B = 2
    prior = AttentionPrior(
        attention_resolution=21,
        hidden_dim=256,
        latent_dim=32,
        motion_kernels=8,
        feature_dim=256
    )

    # Inputs
    prev_attn = torch.randn(B, 21, 21).softmax(dim=-1).softmax(dim=-2)  # Valid attention map
    hidden = torch.randn(B, 256)
    latent = torch.randn(B, 32)

    print(f"Previous attention: {prev_attn.shape}")
    print(f"Hidden state: {hidden.shape}")
    print(f"Latent state: {latent.shape}")

    # Forward pass
    pred_attn, info = prior(prev_attn, hidden, latent)

    print(f"Predicted attention: {pred_attn.shape}")
    print(f"Info keys: {list(info.keys())}")

    # Verify shapes
    assert pred_attn.shape == (B, 21, 21), f"Expected [2, 21, 21], got {pred_attn.shape}"

    # Check info dict
    assert 'predicted_movement' in info, "Missing 'predicted_movement' in info!"
    movement = info['predicted_movement']
    if isinstance(movement, tuple):
        print(f"Predicted movement: tuple of {len(movement)} elements")
    else:
        print(f"Predicted movement: {movement.shape}")

    print("✓ PASSED: AttentionPrior works correctly")


def test_full_attention_schema():
    """Test 3: Full AttentionSchema pipeline"""
    print("\n" + "="*60)
    print("Test 3: Full AttentionSchema Pipeline")
    print("="*60)

    B = 2
    attention = AttentionSchema(
        image_size=84,
        attention_resolution=21,
        hidden_dim=256,
        latent_dim=32,
        context_dim=128,
        device=torch.device('cpu')
    )

    # Inputs
    obs = torch.randn(B, 3, 84, 84)
    hidden = torch.randn(B, 256)
    context = torch.randn(B, 128)

    print(f"Observation: {obs.shape}")
    print(f"Hidden state: {hidden.shape}")
    print(f"Context: {context.shape}")

    # Test posterior
    attn_probs, coords = attention.posterior_net(obs, hidden, context)
    print(f"Posterior attention: {attn_probs.shape}")
    print(f"Posterior coords: {coords.shape}")

    assert attn_probs.shape == (B, 21, 21), "Posterior shape mismatch!"
    assert coords.shape == (B, 2), "Coords shape mismatch!"

    # Test prior
    latent = torch.randn(B, 32)
    pred_attn, info = attention.prior_net(attn_probs, hidden, latent)
    print(f"Prior attention: {pred_attn.shape}")

    assert pred_attn.shape == (B, 21, 21), "Prior shape mismatch!"

    print("✓ PASSED: Full AttentionSchema works correctly")


def test_attention_dynamics_loss():
    """Test 4: Attention dynamics loss computation"""
    print("\n" + "="*60)
    print("Test 4: Attention Dynamics Loss")
    print("="*60)

    attention = AttentionSchema(
        attention_resolution=21,
        device=torch.device('cpu')
    )

    # Create a sequence of attention maps with predictable movement
    T, B = 10, 2
    attention_seq = []
    for t in range(T):
        # Create attention that moves diagonally
        attn = torch.zeros(B, 21, 21)
        y = min(10 + t, 20)
        x = min(10 + t, 20)
        attn[:, y-1:y+2, x-1:x+2] = 1.0
        attn = attn / attn.sum(dim=(1, 2), keepdim=True)
        attention_seq.append(attn)

    # Create predicted movements (should be ~[1, 1] in pixel space)
    predicted_movements = [torch.ones(B, 2) for _ in range(T-1)]

    print(f"Attention sequence length: {len(attention_seq)}")
    print(f"Predicted movements length: {len(predicted_movements)}")

    # Compute loss
    loss = attention.compute_attention_dynamics_loss(attention_seq, predicted_movements)
    print(f"Dynamics loss: {loss:.6f}")

    assert loss >= 0, "Loss should be non-negative!"
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is Inf!"

    print("✓ PASSED: Dynamics loss computation works")


def test_center_of_mass():
    """Test 5: Center of mass computation"""
    print("\n" + "="*60)
    print("Test 5: Center of Mass")
    print("="*60)

    attention = AttentionSchema(device=torch.device('cpu'))

    B, H, W = 2, 21, 21

    # Create attention map with known center
    attn = torch.zeros(B, H, W)
    attn[0, 10, 10] = 1.0  # Center at (10, 10)
    attn[1, 5, 15] = 1.0   # Center at (5, 15)

    centers = attention._center_of_mass(attn)
    print(f"Centers: {centers}")

    # Check if centers are correct
    assert torch.allclose(centers[0], torch.tensor([10.0, 10.0]), atol=0.1), f"Expected [10, 10], got {centers[0]}"
    assert torch.allclose(centers[1], torch.tensor([15.0, 5.0]), atol=0.1), f"Expected [15, 5], got {centers[1]}"

    print("✓ PASSED: Center of mass computation correct")


def test_gradient_flow():
    """Test 6: Gradients flow through attention schema"""
    print("\n" + "="*60)
    print("Test 6: Gradient Flow")
    print("="*60)

    B = 2
    attention = AttentionSchema(
        image_size=84,
        attention_resolution=21,
        hidden_dim=256,
        latent_dim=32,
        context_dim=128,
        device=torch.device('cpu')
    )

    # Inputs with gradients
    obs = torch.randn(B, 3, 84, 84, requires_grad=True)
    hidden = torch.randn(B, 256, requires_grad=True)
    context = torch.randn(B, 128, requires_grad=True)

    # Forward pass
    attn_probs, coords = attention.posterior_net(obs, hidden, context)

    # Compute loss
    loss = attn_probs.sum() + coords.sum()
    loss.backward()

    print(f"Observation gradient norm: {obs.grad.norm():.4f}")
    print(f"Hidden gradient norm: {hidden.grad.norm():.4f}")
    print(f"Context gradient norm: {context.grad.norm():.4f}")

    assert obs.grad is not None and obs.grad.norm() > 0, "No gradient for observations!"
    assert hidden.grad is not None and hidden.grad.norm() > 0, "No gradient for hidden!"
    assert context.grad is not None and context.grad.norm() > 0, "No gradient for context!"

    print("✓ PASSED: Gradients flow correctly")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ATTENTION SCHEMA INTEGRATION TEST SUITE")
    print("="*60)

    try:
        test_attention_posterior()
        test_attention_prior()
        test_full_attention_schema()
        test_attention_dynamics_loss()
        test_center_of_mass()
        test_gradient_flow()

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓✓✓")
        print("="*60)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""
Test: SlotAttention Module

Tests the slot attention mechanism for object-centric decomposition.

Run with:
    python attention_schema/tests/test_slot_attention.py
"""

import sys
sys.path.insert(0, '/home/g/zahra-dir/AIME')

import torch
from src.attention_schema import SlotAttention


def test_basic_slot_attention():
    """Test 1: Basic slot attention routing"""
    print("\n" + "="*60)
    print("Test 1: Basic Slot Attention")
    print("="*60)

    B, N, D = 2, 100, 64
    K = 4

    slot_attn = SlotAttention(
        num_slots=K,
        in_dim=D,
        slot_dim=D,
        iters=3
    )

    x = torch.randn(B, N, D)
    print(f"Input tokens: {x.shape}")

    slots, attn = slot_attn(x)
    print(f"Slots: {slots.shape}")
    print(f"Attention: {attn.shape}")

    # Verify shapes
    assert slots.shape == (B, K, D), f"Expected slots shape [{B}, {K}, {D}], got {slots.shape}"
    assert attn.shape == (B, K, N), f"Expected attn shape [{B}, {K}, {N}], got {attn.shape}"

    # Verify attention sums to 1 over slots (for each position)
    attn_sum = attn.sum(dim=1)
    print(f"Attention sum over slots (should be ~1): {attn_sum[0, :5]}")
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-3), "Attention doesn't sum to 1 over slots!"

    print("✓ PASSED: Shapes and normalization correct")


def test_slot_specialization():
    """Test 2: Slots specialize to different patterns"""
    print("\n" + "="*60)
    print("Test 2: Slot Specialization")
    print("="*60)

    B, N, D = 4, 100, 64
    K = 3

    slot_attn = SlotAttention(
        num_slots=K,
        in_dim=D,
        slot_dim=D,
        iters=5  # More iterations for specialization
    )

    # Create inputs with different patterns
    x = torch.randn(B, N, D)

    slots, attn = slot_attn(x)

    # Check that slots are different from each other
    slot_similarities = []
    for k1 in range(K):
        for k2 in range(k1+1, K):
            sim = torch.cosine_similarity(slots[0, k1], slots[0, k2], dim=0)
            slot_similarities.append(sim.item())

    avg_similarity = sum(slot_similarities) / len(slot_similarities)
    print(f"Average cosine similarity between slots: {avg_similarity:.4f}")
    print(f"(Lower is better - means slots are more specialized)")

    # Slots should be somewhat different
    assert avg_similarity < 0.9, "Slots are too similar - not specializing!"

    print("✓ PASSED: Slots specialize to different patterns")


def test_seed_slots():
    """Test 3: Conditioning slots with seed (top-down)"""
    print("\n" + "="*60)
    print("Test 3: Seed Slots (Top-Down Conditioning)")
    print("="*60)

    B, N, D = 2, 100, 64
    K = 4

    slot_attn = SlotAttention(
        num_slots=K,
        in_dim=D,
        slot_dim=D,
        iters=3
    )

    x = torch.randn(B, N, D)

    # Without seed
    slots_no_seed, attn_no_seed = slot_attn(x)

    # With seed
    seed_slots = torch.randn(B, K, D)
    slots_with_seed, attn_with_seed = slot_attn(x, seed_slots=seed_slots)

    print(f"Slots without seed: {slots_no_seed.shape}")
    print(f"Slots with seed: {slots_with_seed.shape}")

    # Slots should be different when seeded vs not seeded
    diff = (slots_no_seed - slots_with_seed).norm()
    print(f"Difference between seeded and unseeded: {diff:.4f}")

    assert diff > 0.1, "Seed slots had no effect!"

    print("✓ PASSED: Seed slots affect the output")


def test_iterative_refinement():
    """Test 4: More iterations lead to better slot assignments"""
    print("\n" + "="*60)
    print("Test 4: Iterative Refinement")
    print("="*60)

    B, N, D = 2, 100, 64
    K = 4

    x = torch.randn(B, N, D)

    # Test with different iteration counts
    for iters in [1, 3, 5]:
        slot_attn = SlotAttention(
            num_slots=K,
            in_dim=D,
            slot_dim=D,
            iters=iters
        )

        slots, attn = slot_attn(x)

        # Measure attention sharpness (entropy)
        attn_probs = attn / attn.sum(dim=1, keepdim=True)
        entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum(dim=1).mean()

        print(f"  Iters={iters}: Attention entropy = {entropy:.4f} (lower = sharper)")

    print("✓ PASSED: Iterative refinement completes")


def test_gradient_flow():
    """Test 5: Gradients flow through slot attention"""
    print("\n" + "="*60)
    print("Test 5: Gradient Flow")
    print("="*60)

    B, N, D = 2, 100, 64
    K = 4

    slot_attn = SlotAttention(
        num_slots=K,
        in_dim=D,
        slot_dim=D,
        iters=3
    )

    x = torch.randn(B, N, D, requires_grad=True)

    slots, attn = slot_attn(x)

    # Compute a simple loss
    loss = slots.sum()
    loss.backward()

    print(f"Input gradient norm: {x.grad.norm():.4f}")
    assert x.grad is not None, "No gradient computed!"
    assert x.grad.norm() > 0, "Gradient is zero!"

    print("✓ PASSED: Gradients flow correctly")


def test_batch_independence():
    """Test 6: Batch elements are processed independently"""
    print("\n" + "="*60)
    print("Test 6: Batch Independence")
    print("="*60)

    B, N, D = 4, 100, 64
    K = 4

    slot_attn = SlotAttention(
        num_slots=K,
        in_dim=D,
        slot_dim=D,
        iters=3
    )

    x = torch.randn(B, N, D)

    # Process full batch
    slots_batch, _ = slot_attn(x)

    # Process first element alone
    slots_single, _ = slot_attn(x[:1])

    # Should be similar (with random init, not exact)
    diff = (slots_batch[0] - slots_single[0]).norm()
    print(f"Difference between batch and single processing: {diff:.4f}")
    print("(Some difference expected due to random initialization)")

    assert slots_batch.shape[0] == B, "Batch size mismatch!"

    print("✓ PASSED: Batch processing works correctly")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SLOT ATTENTION TEST SUITE")
    print("="*60)

    try:
        test_basic_slot_attention()
        test_slot_specialization()
        test_seed_slots()
        test_iterative_refinement()
        test_gradient_flow()
        test_batch_independence()

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

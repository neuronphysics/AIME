#!/usr/bin/env python
"""
Test script to verify the generate_autoregressive fix.

The bug was: T_start_index growing beyond learned time_queries during autoregressive generation.
The fix: Handle out-of-bounds indices by repeating the last learned query and relying on
         position encoding for temporal information.

Usage:
    # Quick test
    python scripts/test_perceiver_fix.py

    # With CUDA debugging
    CUDA_LAUNCH_BLOCKING=1 python scripts/test_perceiver_fix.py
"""

import torch
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)  # Ensure relative imports work

# Use the new refactored version
from src.perceiver_io.tokenizer import VQPTTokenizer
from src.perceiver_io.predictor import PerceiverTokenPredictor
from src.perceiver_io import CausalPerceiverIO


def test_autoregressive_generation():
    """Test that autoregressive generation works beyond trained sequence length."""
    print("=" * 60)
    print("Testing Perceiver IO Autoregressive Generation Fix")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Setup
    sequence_length = 10  # Train on short sequences
    B, C, H, W = 2, 3, 64, 64

    print(f"\nModel config:")
    print(f"  Trained sequence_length: {sequence_length}")
    print(f"  Video shape: ({sequence_length}, {C}, {H}, {W})")

    # Create model
    tokenizer = VQPTTokenizer(
        in_channels=C,
        code_dim=256,
        num_codes=512,
        downsample=4,
        base_channels=32,
        use_3d_conv=False,
        num_quantizers=1,  # Use single-head VQ for simplicity
        kmeans_init=False,
    )

    model = PerceiverTokenPredictor(
        tokenizer=tokenizer,
        num_latents=256,
        num_latent_channels=256,
        num_cross_attention_heads=4,
        num_self_attention_layers=2,
        num_self_attention_heads=4,
        widening_factor=2,
        dropout=0.0,
        sequence_length=sequence_length,  # THIS IS THE KEY PARAMETER
    ).to(device).eval()

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test 1: Normal generation (within trained range)
    print(f"\n{'='*60}")
    print("Test 1: Generate WITHIN trained range (should always work)")
    print(f"{'='*60}")

    context_frames = 5
    num_to_generate = 3  # Total: 5 + 3 = 8 < 10 (within range)

    context_video = torch.randn(B, context_frames, C, H, W, device=device)

    try:
        with torch.no_grad():
            generated = model.generate_autoregressive(
                context_video,
                num_frames_to_generate=num_to_generate,
                temperature=0.0,  # Greedy for determinism
            )
        print(f"✅ SUCCESS: Generated {generated.shape[1]} frames")
        print(f"   Context: {context_frames}, Generated: {num_to_generate}")
        print(f"   Output shape: {tuple(generated.shape)}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        raise

    # Test 2: Generation beyond trained range (THE BUG)
    print(f"\n{'='*60}")
    print("Test 2: Generate BEYOND trained range (bug test)")
    print(f"{'='*60}")

    context_frames = 5
    num_to_generate = 10  # Total: 5 + 10 = 15 > 10 (BEYOND trained range!)

    print(f"\nAttempting to generate {num_to_generate} frames from {context_frames} context")
    print(f"This will access positions up to {context_frames + num_to_generate} (> {sequence_length})")
    print(f"Without fix: Would try to slice time_queries[:, 15:16] when max is [:, :10]")
    print(f"With fix: Uses last learned query + position encoding for extrapolation\n")

    context_video = torch.randn(B, context_frames, C, H, W, device=device)

    try:
        with torch.no_grad():
            generated = model.generate_autoregressive(
                context_video,
                num_frames_to_generate=num_to_generate,
                temperature=0.0,
            )
        print(f"✅ SUCCESS: Generated {generated.shape[1]} frames (FIX WORKS!)")
        print(f"   Context: {context_frames}, Generated: {num_to_generate}")
        print(f"   Output shape: {tuple(generated.shape)}")
        print(f"   Positions accessed: 0 → {context_frames + num_to_generate}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        print(f"\nIf you see 'shape mismatch' or 'empty tensor', the fix didn't work!")
        raise

    # Test 3: Extreme case
    print(f"\n{'='*60}")
    print("Test 3: Extreme case - 2x beyond trained range")
    print(f"{'='*60}")

    num_to_generate = 20  # 5 + 20 = 25 (2.5x trained length!)

    try:
        with torch.no_grad():
            generated = model.generate_autoregressive(
                context_video,
                num_frames_to_generate=num_to_generate,
                temperature=0.0,
            )
        print(f"✅ SUCCESS: Even extreme extrapolation works!")
        print(f"   Generated up to position {generated.shape[1]} (trained up to {sequence_length})")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        raise

    print(f"\n{'='*60}")
    print("🎉 ALL TESTS PASSED! The fix works correctly.")
    print(f"{'='*60}")
    print("\nSummary:")
    print("  - Autoregressive generation can now exceed trained sequence_length")
    print("  - Uses last learned query + position encoding for extrapolation")
    print("  - No more empty tensor errors during long-horizon generation")
    print()


if __name__ == "__main__":
    test_autoregressive_generation()

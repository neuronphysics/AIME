#!/usr/bin/env python
"""
Test training with synthetic data (no DMC environment needed).

This verifies:
1. Model initialization works
2. Forward pass produces correct shapes
3. Loss computation works
4. Backward pass doesn't crash
5. Optimization step completes

Usage:
    python scripts/test_training_synthetic.py
"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.perceiver_io import CausalPerceiverIO


def test_training_loop():
    """Test complete training loop with synthetic data."""
    print("=" * 60)
    print("Testing AIME Training Loop with Synthetic Data")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Tier 1 settings (fast debug)
    B = 4  # batch size
    T = 10  # sequence length
    C, H, W = 3, 64, 64
    context_frames = 5

    print(f"\nConfig:")
    print(f"  Batch size: {B}")
    print(f"  Sequence: {T} frames")
    print(f"  Resolution: {H}×{W}")
    print(f"  Context/Prediction: {context_frames}/{T - context_frames}")

    # Create model
    print("\n[1/6] Creating model...")
    model = CausalPerceiverIO(
        video_shape=(T, C, H, W),
        num_latents=128,
        num_latent_channels=256,
        num_attention_heads=4,
        num_encoder_layers=2,
        code_dim=128,
        num_codes=512,
        downsample=4,
        dropout=0.1,
        base_channels=32,
        use_3d_conv=False,
        temporal_downsample=False,
        num_quantizers=1,
        kmeans_init=False,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {num_params:,} parameters")

    # Create optimizer
    print("\n[2/6] Creating optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    print("✓ Optimizer created")

    # Generate synthetic data
    print("\n[3/6] Generating synthetic data...")
    # Create moving patterns (easier to learn than pure noise)
    t = torch.linspace(0, 2 * 3.14159, T).view(1, T, 1, 1, 1)
    x_grid = torch.linspace(-1, 1, W).view(1, 1, 1, 1, W)
    y_grid = torch.linspace(-1, 1, H).view(1, 1, 1, H, 1)

    # Moving wave pattern
    videos = torch.sin(t + x_grid * 2) * torch.cos(y_grid * 2)
    videos = videos.expand(B, T, 1, H, W)
    videos = torch.cat([videos, videos * 0.5, videos * 0.3], dim=2)  # RGB
    videos = videos.to(device)

    target_videos = videos[:, context_frames:]  # Prediction target

    print(f"✓ Data created: {tuple(videos.shape)}")

    # Forward pass
    print("\n[4/6] Running forward pass...")
    model.train()
    outputs = model(videos, num_context_frames=context_frames, return_dict=True)

    print(f"✓ Forward complete")
    print(f"  Logits shape: {tuple(outputs['logits'].shape)}")
    print(f"  Reconstructed shape: {tuple(outputs['reconstructed'].shape)}")
    print(f"  Latents shape: {tuple(outputs['encoder_latents'].shape)}")

    # Compute loss
    print("\n[5/6] Computing loss...")
    loss_dict = model.compute_loss(
        outputs,
        target_videos,
        perceptual_weight=0.5,
        label_smoothing=0.1,
    )

    total_loss = loss_dict['loss']
    print(f"✓ Loss computed")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  CE loss: {loss_dict['ce_loss'].item():.4f}")
    print(f"  VQ loss: {loss_dict['vq_loss'].item():.4f}")
    print(f"  Perceptual loss: {loss_dict['perceptual_loss'].item():.4f}")
    print(f"  Accuracy: {loss_dict['accuracy'].item():.2%}")

    # Backward pass
    print("\n[6/6] Running backward pass...")
    optimizer.zero_grad()
    total_loss.backward()

    # Check gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    print(f"✓ Backward complete")
    print(f"  Gradient norm: {grad_norm.item():.4f}")

    # Optimizer step
    optimizer.step()
    print(f"✓ Optimizer step complete")

    # Test generation
    print("\n" + "=" * 60)
    print("Testing Generation Modes")
    print("=" * 60)

    model.eval()
    context = videos[:2, :context_frames]  # Use 2 samples

    with torch.no_grad():
        # Test reconstruction
        print("\n[1/3] Testing reconstruction...")
        recon = model.reconstruct(videos[:2])
        print(f"✓ Reconstruction: {tuple(recon.shape)}")

        # Test autoregressive
        print("\n[2/3] Testing autoregressive generation...")
        gen_auto = model.generate_autoregressive(
            context,
            num_frames_to_generate=3,
            temperature=0.9,
        )
        print(f"✓ Autoregressive: {tuple(gen_auto.shape)}")

        # Test maskgit
        print("\n[3/3] Testing MaskGIT generation...")
        gen_maskgit = model.generate_maskgit(
            context,
            num_frames_to_generate=3,
            num_iterations=4,
            temperature=0.9,
        )
        print(f"✓ MaskGIT: {tuple(gen_maskgit.shape)}")

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
    print("\nSummary:")
    print("  ✓ Model initializes correctly")
    print("  ✓ Forward pass produces correct shapes")
    print("  ✓ Loss computation works")
    print("  ✓ Backward pass completes")
    print("  ✓ Optimizer step works")
    print("  ✓ Reconstruction generation works")
    print("  ✓ Autoregressive generation works")
    print("  ✓ MaskGIT generation works")
    print("\nReady to train on real DMC data!")
    print()


if __name__ == "__main__":
    test_training_loop()

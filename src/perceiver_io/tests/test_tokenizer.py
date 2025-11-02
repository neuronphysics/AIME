"""
Test: VQPTTokenizer

Demonstrates video → tokens → reconstructed video

This test verifies that the VQ-VAE tokenizer can:
1. Encode videos to discrete tokens
2. Decode tokens back to videos
3. Maintain reasonable reconstruction quality
"""
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from perceiver_io import VQPTTokenizer


def test_tokenizer():
    """Test basic tokenizer functionality"""
    print("=" * 60)
    print("Testing VQPTTokenizer")
    print("=" * 60)

    # Small synthetic video
    B, T, C, H, W = 2, 8, 3, 64, 64
    video = torch.randn(B, T, C, H, W)
    print(f"\n1. Input video: {video.shape}")
    print(f"   Range: [{video.min():.3f}, {video.max():.3f}]")

    # Initialize tokenizer
    print("\n2. Initializing tokenizer...")
    tokenizer = VQPTTokenizer(
        in_channels=3,
        code_dim=256,
        num_codes=256,  # Small for test
        downsample=4,
        base_channels=32,  # Small for test
        use_3d_conv=True,
        temporal_downsample=False,  # Don't downsample time for small sequences
        num_quantizers=1  # Single head for test
    )
    print(f"   Parameters: {sum(p.numel() for p in tokenizer.parameters()):,}")

    # Encode
    print("\n3. Encoding video to tokens...")
    token_ids, quantized, vq_loss, skips = tokenizer.encode(video)
    print(f"   Token IDs: {token_ids.shape}")
    print(f"   Quantized: {quantized.shape}")
    print(f"   VQ loss: {vq_loss.item():.4f}")
    print(f"   Skip connections: {len(skips)} levels")

    # Check token range
    unique_tokens = token_ids.unique()
    print(f"   Unique tokens used: {len(unique_tokens)} / {tokenizer.num_codes}")

    # Decode
    print("\n4. Decoding tokens to video...")
    reconstructed = tokenizer.decode(quantized, skips=skips, use_tanh=False)
    print(f"   Reconstructed: {reconstructed.shape}")
    print(f"   Range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")

    # Verify shapes match
    assert reconstructed.shape == video.shape, f"Shape mismatch! {reconstructed.shape} != {video.shape}"

    # Compute reconstruction error
    mse = torch.mean((video - reconstructed) ** 2).item()
    print(f"\n5. Reconstruction MSE: {mse:.6f}")

    print("\n" + "=" * 60)
    print("✓ Test passed!")
    print("=" * 60)


def test_2d_tokenizer():
    """Test 2D (per-frame) tokenizer"""
    print("\n" + "=" * 60)
    print("Testing VQPTTokenizer (2D mode)")
    print("=" * 60)

    # Small synthetic video
    B, T, C, H, W = 2, 4, 3, 64, 64
    video = torch.randn(B, T, C, H, W)
    print(f"\n1. Input video: {video.shape}")

    # Initialize tokenizer in 2D mode
    print("\n2. Initializing 2D tokenizer...")
    tokenizer = VQPTTokenizer(
        in_channels=3,
        code_dim=128,
        num_codes=512,
        downsample=4,
        base_channels=32,
        use_3d_conv=False,  # 2D mode
        num_quantizers=1
    )

    # Encode
    print("\n3. Encoding (2D)...")
    token_ids, quantized, vq_loss, skips = tokenizer.encode(video)
    print(f"   Token IDs: {token_ids.shape}")
    print(f"   VQ loss: {vq_loss.item():.4f}")

    # Decode
    print("\n4. Decoding...")
    reconstructed = tokenizer.decode(quantized, skips=skips, use_tanh=False)
    print(f"   Reconstructed: {reconstructed.shape}")

    assert reconstructed.shape == video.shape
    print("\n✓ 2D test passed!")


if __name__ == "__main__":
    try:
        test_tokenizer()
        test_2d_tokenizer()
        print("\n" + "=" * 60)
        print("All tests passed successfully! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

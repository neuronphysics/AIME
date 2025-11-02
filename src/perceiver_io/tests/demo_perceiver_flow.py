"""
Demo: Full Perceiver IO Pipeline

Shows complete flow: video → context → prediction

This demo demonstrates:
1. Context extraction for VRNN integration
2. Forward pass with video prediction
3. Full pipeline functionality
"""
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from perceiver_io import CausalPerceiverIO


def demo_context_extraction():
    """Demo: Extract context vectors for VRNN"""
    print("=" * 60)
    print("Demo 1: Context Extraction for VRNN")
    print("=" * 60)

    # Synthetic video
    B, T, C, H, W = 2, 8, 3, 64, 64
    video = torch.randn(B, T, C, H, W)
    print(f"\n1. Input video: {video.shape}")

    # Initialize Perceiver (small for demo)
    print("\n2. Initializing CausalPerceiverIO...")
    perceiver = CausalPerceiverIO(
        video_shape=(T, C, H, W),
        num_latents=64,  # Small for demo
        num_latent_channels=128,  # Small for demo
        num_codes=256,
        downsample=4
    )
    print(f"   Parameters: {sum(p.numel() for p in perceiver.parameters()):,}")

    # Extract context (main use case for VRNN)
    print("\n3. Extracting context vectors...")
    context = perceiver.extract_context(video)
    print(f"   Context shape: {context.shape}")
    print(f"   Expected: [B={B}, T_tokens, C=128]")

    # Verify shape
    assert context.shape[0] == B, f"Batch size mismatch: {context.shape[0]} != {B}"
    assert context.shape[2] == 128, f"Channel mismatch: {context.shape[2]} != 128"
    print(f"   Context range: [{context.min():.3f}, {context.max():.3f}]")

    print("\n✓ Context extraction successful!")
    return perceiver, video


def demo_forward_pass(perceiver, video):
    """Demo: Full forward pass with prediction"""
    print("\n" + "=" * 60)
    print("Demo 2: Forward Pass with Video Prediction")
    print("=" * 60)

    # Use first 4 frames as context, predict next 4
    num_context = 4
    print(f"\n1. Using first {num_context} frames as context")
    print(f"   Predicting remaining {video.shape[1] - num_context} frames")

    # Forward pass
    print("\n2. Running forward pass...")
    outputs = perceiver.forward(
        videos=video,
        num_context_frames=num_context,
        return_dict=True
    )

    # Display outputs
    print("\n3. Output tensors:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape}")
        else:
            print(f"   {key}: {value}")

    # Check reconstruction
    reconstructed = outputs['reconstructed']
    print(f"\n4. Reconstructed video: {reconstructed.shape}")
    print(f"   Reconstruction range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")

    print("\n✓ Forward pass successful!")


def demo_generation(perceiver):
    """Demo: Autoregressive generation"""
    print("\n" + "=" * 60)
    print("Demo 3: Autoregressive Generation")
    print("=" * 60)

    # Create context video
    B, C, H, W = 1, 3, 64, 64
    context_frames = 4
    context_video = torch.randn(B, context_frames, C, H, W)
    print(f"\n1. Context video: {context_video.shape}")

    # Generate future frames
    num_generate = 4
    print(f"\n2. Generating {num_generate} future frames...")
    print("   (This may take a moment...)")

    generated = perceiver.generate_autoregressive(
        context_videos=context_video,
        num_frames_to_generate=num_generate,
        temperature=1.0
    )

    print(f"\n3. Generated video: {generated.shape}")
    print(f"   Total frames: {generated.shape[1]} (context + generated)")
    print(f"   Generated range: [{generated.min():.3f}, {generated.max():.3f}]")

    # Verify shape
    expected_frames = context_frames + num_generate
    assert generated.shape[1] == expected_frames, \
        f"Expected {expected_frames} frames, got {generated.shape[1]}"

    print("\n✓ Generation successful!")


if __name__ == "__main__":
    try:
        # Run demos
        perceiver, video = demo_context_extraction()
        demo_forward_pass(perceiver, video)
        demo_generation(perceiver)

        print("\n" + "=" * 60)
        print("All demos completed successfully! ✓")
        print("=" * 60)
        print("\nKey takeaways:")
        print("• Context extraction: extract_context() → [B, T, C]")
        print("• Forward pass: forward() → full outputs dict")
        print("• Generation: generate_autoregressive() → new frames")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

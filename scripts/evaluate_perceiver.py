#!/usr/bin/env python
"""
Evaluate trained Perceiver IO model and generate video predictions.

Usage:
    python scripts/evaluate_perceiver.py \
        --checkpoint checkpoints/perceiver_simple/best_model.pt \
        --data data/tier1/cartpole_swingup.hdf5 \
        --num_samples 4 \
        --context_frames 5 \
        --generate_frames 10
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import h5py
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.perceiver_io import CausalPerceiverIO


def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model config from checkpoint if available
    # Otherwise use default Tier 1 config
    model = CausalPerceiverIO(
        video_shape=(10, 3, 64, 64),
        num_latents=128,
        num_latent_channels=256,
        num_attention_heads=4,
        num_encoder_layers=2,
        code_dim=128,
        num_codes=512,
        downsample=4,
        dropout=0.0,  # No dropout for eval
        base_channels=32,
        use_3d_conv=False,
        num_quantizers=1,
    ).to(device)

    # Load state dict with strict=False to handle any mismatches
    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if missing:
        print(f"Warning: Missing keys: {missing[:5]}...")  # Show first 5
    if unexpected:
        print(f"Warning: Unexpected keys: {unexpected[:5]}...")  # Show first 5

    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.4f}")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        print(f"  Accuracy: {metrics.get('accuracy', 0):.2%}")

    return model


def load_test_videos(hdf5_path, num_samples, sequence_length):
    """Load test videos from HDF5."""
    videos = []

    with h5py.File(hdf5_path, 'r') as f:
        episode_keys = sorted(list(f.keys()))

        # Use last episodes for testing
        test_keys = episode_keys[-num_samples:]

        for ep_key in test_keys:
            obs = f[ep_key]['observations'][:]

            if len(obs) >= sequence_length:
                # Take first sequence_length frames
                video = obs[:sequence_length]

                # Convert to tensor and normalize
                video = torch.from_numpy(video).float() / 255.0 * 2.0 - 1.0
                video = rearrange(video, 't h w c -> t c h w')

                videos.append(video)

    return torch.stack(videos) if videos else None


def denormalize_video(video):
    """Convert from [-1, 1] to [0, 255] uint8."""
    video = (video + 1.0) / 2.0 * 255.0
    video = video.clamp(0, 255).byte()
    return video


@torch.no_grad()
def evaluate_model(model, videos, context_frames, generate_frames, device, temperature=1.0):
    """Evaluate model on test videos."""
    videos = videos.to(device)
    B, T, C, H, W = videos.shape

    # Split into context and ground truth
    context = videos[:, :context_frames]
    ground_truth = videos[:, context_frames:]

    print(f"\nEvaluating on {B} videos:")
    print(f"  Context: {context_frames} frames")
    print(f"  Generate: {generate_frames} frames")
    print(f"  Temperature: {temperature}")

    # Autoregressive generation
    generated = model.generate_autoregressive(
        context,
        num_frames_to_generate=generate_frames,
        temperature=temperature,
    )

    # Compute reconstruction metrics
    outputs = model(videos, num_context_frames=context_frames, return_dict=True)
    target_videos = videos[:, context_frames:]
    loss_dict = model.compute_loss(outputs, target_videos)

    print(f"\nMetrics:")
    print(f"  Loss: {loss_dict['loss'].item():.4f}")
    print(f"  Accuracy: {loss_dict['accuracy'].item():.2%}")
    print(f"  CE Loss: {loss_dict['ce_loss'].item():.4f}")
    print(f"  VQ Loss: {loss_dict['vq_loss'].item():.4f}")
    print(f"  Perceptual: {loss_dict['perceptual_loss'].item():.4f}")

    # Get reconstructed video - check for possible key names
    if 'reconstructed_video' in outputs:
        recon_key = 'reconstructed_video'
    elif 'reconstructed' in outputs:
        recon_key = 'reconstructed'
    elif 'recon_video' in outputs:
        recon_key = 'recon_video'
    else:
        # Try to find the reconstruction key
        print(f"Available output keys: {list(outputs.keys())}")
        recon_key = [k for k in outputs.keys() if 'recon' in k.lower()][0]

    return {
        'context': context,
        'ground_truth': ground_truth,
        'generated': generated,
        'reconstructed': outputs[recon_key],
        'metrics': {k: v.item() for k, v in loss_dict.items()},
    }


def visualize_results(results, output_dir):
    """Create visualizations of results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context = denormalize_video(results['context'].cpu())
    ground_truth = denormalize_video(results['ground_truth'].cpu())
    generated = denormalize_video(results['generated'].cpu())
    reconstructed = denormalize_video(results['reconstructed'].cpu())

    B = context.shape[0]

    # Create comparison figure for each sample
    for b in range(B):
        max_frames = max(context.shape[1], generated.shape[1], reconstructed.shape[1])
        total_frames = context.shape[1] + max(generated.shape[1], reconstructed.shape[1])
        fig, axes = plt.subplots(4, total_frames, figsize=(20, 8))
        fig.suptitle(f'Sample {b+1} - Acc: {results["metrics"]["accuracy"]:.1%}', fontsize=14)

        # Row 1: Context
        for t in range(context.shape[1]):
            img = rearrange(context[b, t], 'c h w -> h w c').numpy()
            axes[0, t].imshow(img)
            axes[0, t].set_title(f'Context {t+1}')
            axes[0, t].axis('off')
        for t in range(context.shape[1], axes.shape[1]):
            axes[0, t].axis('off')

        # Row 2: Ground Truth
        for t in range(ground_truth.shape[1]):
            img = rearrange(ground_truth[b, t], 'c h w -> h w c').numpy()
            axes[1, context.shape[1] + t].imshow(img)
            axes[1, context.shape[1] + t].set_title(f'GT {t+1}')
            axes[1, context.shape[1] + t].axis('off')
        for t in range(context.shape[1]):
            axes[1, t].axis('off')
        for t in range(context.shape[1] + ground_truth.shape[1], axes.shape[1]):
            axes[1, t].axis('off')

        # Row 3: Generated (Autoregressive)
        for t in range(generated.shape[1]):
            img = rearrange(generated[b, t], 'c h w -> h w c').numpy()
            axes[2, context.shape[1] + t].imshow(img)
            axes[2, context.shape[1] + t].set_title(f'Gen {t+1}')
            axes[2, context.shape[1] + t].axis('off')
        for t in range(context.shape[1]):
            axes[2, t].axis('off')
        for t in range(context.shape[1] + generated.shape[1], axes.shape[1]):
            axes[2, t].axis('off')

        # Row 4: Reconstructed (Teacher Forcing)
        for t in range(reconstructed.shape[1]):
            img = rearrange(reconstructed[b, t], 'c h w -> h w c').numpy()
            axes[3, context.shape[1] + t].imshow(img)
            axes[3, context.shape[1] + t].set_title(f'Recon {t+1}')
            axes[3, context.shape[1] + t].axis('off')
        for t in range(context.shape[1]):
            axes[3, t].axis('off')
        for t in range(context.shape[1] + reconstructed.shape[1], axes.shape[1]):
            axes[3, t].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / f'sample_{b+1}.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization: {output_dir / f'sample_{b+1}.png'}")

    # Create animated GIFs
    for b in range(B):
        # Full sequence (context + generated)
        full_seq = torch.cat([context[b], generated[b]], dim=0)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axis('off')

        def update(frame):
            ax.clear()
            ax.axis('off')
            img = rearrange(full_seq[frame], 'c h w -> h w c').numpy()
            ax.imshow(img)
            if frame < context.shape[1]:
                ax.set_title(f'Context Frame {frame+1}', fontsize=12)
            else:
                ax.set_title(f'Generated Frame {frame - context.shape[1] + 1}', fontsize=12)
            return [ax]

        anim = FuncAnimation(fig, update, frames=len(full_seq), interval=200, blit=True)
        anim.save(output_dir / f'sample_{b+1}_generated.gif', writer=PillowWriter(fps=5))
        plt.close()

        print(f"Saved animation: {output_dir / f'sample_{b+1}_generated.gif'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to HDF5 test data')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of test samples')
    parser.add_argument('--context_frames', type=int, default=5, help='Context frames')
    parser.add_argument('--generate_frames', type=int, default=10, help='Frames to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("Perceiver IO Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data}")
    print(f"Device: {device}")
    print("=" * 60)

    # Load model
    model = load_checkpoint(args.checkpoint, device)

    # Load test videos
    print(f"\nLoading test videos...")
    videos = load_test_videos(
        args.data,
        args.num_samples,
        args.context_frames + args.generate_frames
    )

    if videos is None or len(videos) == 0:
        print("Error: No test videos found!")
        return

    print(f"Loaded {len(videos)} test videos")

    # Evaluate
    results = evaluate_model(
        model,
        videos,
        args.context_frames,
        args.generate_frames,
        device,
        args.temperature
    )

    # Visualize
    print(f"\nCreating visualizations...")
    visualize_results(results, args.output_dir)

    print("\n" + "=" * 60)
    print("✓ Evaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

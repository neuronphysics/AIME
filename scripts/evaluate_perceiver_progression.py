#!/usr/bin/env python
"""
Evaluate multiple checkpoints from a training run to visualize model progression.

Usage:
    python scripts/evaluate_perceiver_progression.py \
        --checkpoint_dir checkpoints/perceiver_tier1_final \
        --data data/tier1/cartpole_swingup.hdf5 \
        --num_samples 4 \
        --context_frames 5 \
        --generate_frames 20 \
        --output_dir evaluation_results/tier1_cartpole
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
import re

from src.perceiver_io import CausalPerceiverIO


def natural_sort_key(s):
    """Sort strings containing numbers naturally."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]


def find_checkpoints(checkpoint_dir, max_checkpoints=10):
    """Find checkpoint files and sample them evenly if there are too many."""
    checkpoint_dir = Path(checkpoint_dir)

    # Look for checkpoint files
    checkpoints = []
    for pattern in ['checkpoint_epoch_*.pt', 'model_epoch_*.pt']:
        checkpoints.extend(list(checkpoint_dir.glob(pattern)))

    # Sort by epoch number
    checkpoints = sorted(checkpoints, key=natural_sort_key)

    # Add best_model.pt if it exists and isn't already in the list
    best_model = checkpoint_dir / 'best_model.pt'
    if best_model.exists():
        checkpoints.append(best_model)

    # Remove duplicates while preserving order
    seen = set()
    unique_checkpoints = []
    for ckpt in checkpoints:
        if ckpt not in seen:
            seen.add(ckpt)
            unique_checkpoints.append(ckpt)
    checkpoints = unique_checkpoints

    # Sample evenly if we have too many checkpoints
    if len(checkpoints) > max_checkpoints:
        # Always include first and last
        indices = [0]
        # Add evenly spaced intermediate checkpoints
        step = (len(checkpoints) - 1) / (max_checkpoints - 1)
        for i in range(1, max_checkpoints - 1):
            idx = int(round(i * step))
            indices.append(idx)
        indices.append(len(checkpoints) - 1)

        # Remove duplicates and sort
        indices = sorted(set(indices))
        checkpoints = [checkpoints[i] for i in indices]

    return checkpoints


def load_checkpoint(checkpoint_path, device, dropout=0.0):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config if available
    if 'config' in checkpoint:
        config = checkpoint['config']
        model = CausalPerceiverIO(
            video_shape=(config.get('sequence_length', 10), 3, 64, 64),
            num_latents=config.get('num_latents', 128),
            num_latent_channels=config.get('num_latent_channels', 256),
            num_attention_heads=config.get('num_attention_heads', 4),
            num_encoder_layers=config.get('num_encoder_layers', 2),
            code_dim=config.get('code_dim', 128),
            num_codes=config.get('num_codes', 512),
            downsample=config.get('downsample', 4),
            dropout=dropout,
            base_channels=config.get('base_channels', 32),
            use_3d_conv=False,
            num_quantizers=1,
        ).to(device)
    else:
        # Use default Tier 1 config
        model = CausalPerceiverIO(
            video_shape=(10, 3, 64, 64),
            num_latents=128,
            num_latent_channels=256,
            num_attention_heads=4,
            num_encoder_layers=2,
            code_dim=128,
            num_codes=512,
            downsample=4,
            dropout=dropout,
            base_channels=32,
            use_3d_conv=False,
            num_quantizers=1,
        ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', checkpoint.get('val_loss', 0))

    return model, epoch, loss


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
def evaluate_checkpoint(model, videos, context_frames, generate_frames, device, temperature=1.0):
    """Evaluate single checkpoint."""
    videos = videos.to(device)
    B, T, C, H, W = videos.shape

    context = videos[:, :context_frames]
    ground_truth = videos[:, context_frames:context_frames+generate_frames]

    # Autoregressive generation
    generated = model.generate_autoregressive(
        context,
        num_frames_to_generate=generate_frames,
        temperature=temperature,
    )

    # Truncate generated to match desired length (model may generate more)
    generated = generated[:, :generate_frames]

    # Compute accuracy metric by comparing generated vs ground truth
    # Use simple MSE and pixel-level accuracy
    mse = torch.nn.functional.mse_loss(generated, ground_truth)

    # Pixel-level accuracy (within threshold)
    threshold = 0.1  # Within 0.1 in normalized [-1, 1] space
    pixel_accuracy = (torch.abs(generated - ground_truth) < threshold).float().mean()

    return {
        'context': context,
        'ground_truth': ground_truth,
        'generated': generated,
        'metrics': {
            'mse': mse.item(),
            'accuracy': pixel_accuracy.item(),
            'loss': mse.item(),
        },
    }


def visualize_progression(checkpoint_results, output_dir):
    """Create visualizations showing progression across checkpoints."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_samples = checkpoint_results[0]['results']['context'].shape[0]

    # For each sample, create a comprehensive comparison
    for sample_idx in range(num_samples):
        # Get data
        context = denormalize_video(checkpoint_results[0]['results']['context'][sample_idx:sample_idx+1].cpu())
        ground_truth = denormalize_video(checkpoint_results[0]['results']['ground_truth'][sample_idx:sample_idx+1].cpu())

        # Collect generated frames from all checkpoints
        all_generated = []
        all_epochs = []
        all_metrics = []

        for ckpt_data in checkpoint_results:
            generated = denormalize_video(ckpt_data['results']['generated'][sample_idx:sample_idx+1].cpu())
            all_generated.append(generated)
            all_epochs.append(ckpt_data['epoch'])
            all_metrics.append(ckpt_data['results']['metrics'])

        # Create large comparison figure
        num_checkpoints = len(checkpoint_results)
        context_frames = context.shape[1]
        gen_frames = all_generated[0].shape[1]

        # Calculate layout: Context + GT + one row per checkpoint
        num_rows = 2 + num_checkpoints  # Context, GT, then one per checkpoint
        num_cols = max(context_frames, gen_frames)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.5, num_rows * 1.5))
        fig.suptitle(f'Sample {sample_idx+1} - Model Progression Through Training', fontsize=16, fontweight='bold')

        # Row 0: Context frames
        for t in range(context_frames):
            img = rearrange(context[0, t], 'c h w -> h w c').numpy()
            axes[0, t].imshow(img)
            axes[0, t].set_title(f'Context {t+1}', fontsize=10)
            axes[0, t].axis('off')
        for t in range(context_frames, num_cols):
            axes[0, t].axis('off')
        axes[0, 0].set_ylabel('Context', fontsize=12, fontweight='bold', rotation=0, ha='right', va='center')

        # Row 1: Ground Truth
        for t in range(gen_frames):
            img = rearrange(ground_truth[0, t], 'c h w -> h w c').numpy()
            axes[1, t].imshow(img)
            axes[1, t].set_title(f'GT {t+1}', fontsize=10)
            axes[1, t].axis('off')
        for t in range(gen_frames, num_cols):
            axes[1, t].axis('off')
        axes[1, 0].set_ylabel('Ground Truth', fontsize=12, fontweight='bold', rotation=0, ha='right', va='center')

        # Rows 2+: Generated from each checkpoint
        for ckpt_idx, (generated, epoch, metrics) in enumerate(zip(all_generated, all_epochs, all_metrics)):
            row_idx = 2 + ckpt_idx

            for t in range(gen_frames):
                img = rearrange(generated[0, t], 'c h w -> h w c').numpy()
                axes[row_idx, t].imshow(img)
                if t == 0:
                    acc = metrics['accuracy']
                    axes[row_idx, t].set_title(f'Epoch {epoch}\nAcc: {acc:.1%}', fontsize=9)
                else:
                    axes[row_idx, t].set_title(f'Gen {t+1}', fontsize=9)
                axes[row_idx, t].axis('off')

            for t in range(gen_frames, num_cols):
                axes[row_idx, t].axis('off')

            axes[row_idx, 0].set_ylabel(f'Epoch {epoch}', fontsize=11, rotation=0, ha='right', va='center')

        plt.tight_layout()
        plt.savefig(output_dir / f'sample_{sample_idx+1}.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved progression visualization: {output_dir / f'sample_{sample_idx+1}.png'}")

        # Create animated GIF showing progression
        # Concatenate context + best generated
        best_ckpt_idx = np.argmax([m['accuracy'] for m in all_metrics])
        best_generated = all_generated[best_ckpt_idx]
        best_epoch = all_epochs[best_ckpt_idx]

        full_seq = torch.cat([context[0], best_generated[0]], dim=0)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.axis('off')

        def update(frame):
            ax.clear()
            ax.axis('off')
            img = rearrange(full_seq[frame], 'c h w -> h w c').numpy()
            ax.imshow(img)
            if frame < context_frames:
                ax.set_title(f'Context Frame {frame+1}', fontsize=14, fontweight='bold')
            else:
                ax.set_title(f'Generated Frame {frame - context_frames + 1}\n(Epoch {best_epoch})',
                           fontsize=14, fontweight='bold')
            return [ax]

        anim = FuncAnimation(fig, update, frames=len(full_seq), interval=300, blit=True, repeat=True)

        # Save with better settings
        writer = PillowWriter(fps=3)
        anim.save(output_dir / f'sample_{sample_idx+1}_generated.gif', writer=writer)
        plt.close()

        print(f"Saved animation: {output_dir / f'sample_{sample_idx+1}_generated.gif'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory with checkpoints')
    parser.add_argument('--data', type=str, required=True, help='Path to HDF5 test data')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of test samples')
    parser.add_argument('--context_frames', type=int, default=5, help='Context frames')
    parser.add_argument('--generate_frames', type=int, default=20, help='Frames to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output_dir', type=str, default='evaluation_results/tier1_cartpole',
                       help='Output directory')
    parser.add_argument('--max_checkpoints', type=int, default=10,
                       help='Maximum number of checkpoints to evaluate (evenly sampled)')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("Perceiver IO Progression Evaluation")
    print("=" * 80)
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Data: {args.data}")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)

    # Find all checkpoints (sampled if too many)
    all_checkpoints = list(Path(args.checkpoint_dir).glob('*.pt'))
    checkpoints = find_checkpoints(args.checkpoint_dir, args.max_checkpoints)

    print(f"\nFound {len(all_checkpoints)} total checkpoints")
    if len(checkpoints) < len(all_checkpoints):
        print(f"Sampling {len(checkpoints)} checkpoints evenly distributed across training:")
    else:
        print(f"Evaluating all {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        print(f"  - {ckpt.name}")

    if len(checkpoints) == 0:
        print("Error: No checkpoints found!")
        return

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

    # Evaluate each checkpoint
    checkpoint_results = []

    for ckpt_path in checkpoints:
        print(f"\nEvaluating {ckpt_path.name}...")

        model, epoch, loss = load_checkpoint(ckpt_path, device)
        results = evaluate_checkpoint(
            model,
            videos,
            args.context_frames,
            args.generate_frames,
            device,
            args.temperature
        )

        checkpoint_results.append({
            'path': ckpt_path,
            'epoch': epoch,
            'loss': loss,
            'results': results,
        })

        print(f"  Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {results['metrics']['accuracy']:.2%}")

    # Create visualizations
    print(f"\nCreating progression visualizations...")
    visualize_progression(checkpoint_results, args.output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("✓ Evaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print("\nCheckpoint Summary:")
    for ckpt_data in checkpoint_results:
        epoch = ckpt_data['epoch']
        acc = ckpt_data['results']['metrics']['accuracy']
        loss = ckpt_data['loss']
        print(f"  Epoch {epoch:3d}: Loss={loss:.4f}, Acc={acc:.2%}")
    print("=" * 80)


if __name__ == '__main__':
    main()

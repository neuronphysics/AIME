#!/usr/bin/env python
"""
Simple Perceiver IO training script using clean src/ code.

Usage:
    python scripts/train_perceiver_simple.py --data data/tier1/cartpole_swingup.hdf5
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from tqdm import tqdm
from einops import rearrange

from src.perceiver_io import CausalPerceiverIO


class HDF5VideoDataset(Dataset):
    """Simple HDF5 dataset for video sequences."""

    def __init__(self, hdf5_path, sequence_length=10, img_size=64):
        self.hdf5_path = hdf5_path
        self.sequence_length = sequence_length
        self.img_size = img_size

        # Load all episodes into memory (small dataset)
        with h5py.File(hdf5_path, 'r') as f:
            self.episodes = []
            for ep_key in sorted(f.keys()):
                obs = f[ep_key]['observations'][:]  # (T, H, W, 3)
                if len(obs) >= sequence_length:
                    self.episodes.append(obs)

        print(f"Loaded {len(self.episodes)} episodes from {hdf5_path}")

    def __len__(self):
        return len(self.episodes) * 10  # Multiple samples per episode

    def __getitem__(self, idx):
        ep_idx = idx % len(self.episodes)
        episode = self.episodes[ep_idx]

        # Random start position
        max_start = len(episode) - self.sequence_length
        start = np.random.randint(0, max_start + 1) if max_start > 0 else 0

        # Extract sequence
        video = episode[start:start + self.sequence_length]  # (T, H, W, 3)

        # Convert to tensor and normalize to [-1, 1]
        video = torch.from_numpy(video).float() / 255.0 * 2.0 - 1.0

        # Rearrange to (T, C, H, W)
        video = rearrange(video, 't h w c -> t c h w')

        return {'video': video}


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_ce = 0
    total_vq = 0
    total_lpips = 0
    total_acc = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        videos = batch['video'].to(device)  # (B, T, C, H, W)
        B, T = videos.shape[:2]
        context_frames = T // 2

        # Forward
        outputs = model(videos, num_context_frames=context_frames, return_dict=True)

        # Loss
        target_videos = videos[:, context_frames:]
        loss_dict = model.compute_loss(outputs, target_videos)

        total_loss += loss_dict['loss'].item()
        total_ce += loss_dict['ce_loss'].item()
        total_vq += loss_dict['vq_loss'].item()
        total_lpips += loss_dict['perceptual_loss'].item()
        total_acc += loss_dict['accuracy'].item()
        num_batches += 1

        # Backward
        optimizer.zero_grad()
        loss_dict['loss'].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        # Update progress
        pbar.set_postfix({
            'loss': f"{loss_dict['loss'].item():.3f}",
            'acc': f"{loss_dict['accuracy'].item():.2%}",
            'grad': f"{grad_norm:.2f}"
        })

    return {
        'loss': total_loss / num_batches,
        'ce_loss': total_ce / num_batches,
        'vq_loss': total_vq / num_batches,
        'perceptual_loss': total_lpips / num_batches,
        'accuracy': total_acc / num_batches,
    }


@torch.no_grad()
def visualize(model, dataloader, device, epoch, out_dir):
    """Generate sample videos."""
    model.eval()

    # Get one batch
    batch = next(iter(dataloader))
    videos = batch['video'][:4].to(device)  # Take 4 samples
    B, T = videos.shape[:2]
    context_frames = T // 2

    # Generate
    context = videos[:, :context_frames]

    # Autoregressive
    generated = model.generate_autoregressive(
        context,
        num_frames_to_generate=T - context_frames,
        temperature=0.9,
    )

    # Save (simple text summary for now)
    print(f"\n[Epoch {epoch}] Generated {generated.shape[1]} frames")
    print(f"  Context: {context_frames}, Generated: {T - context_frames}")
    print(f"  Output range: [{generated.min():.2f}, {generated.max():.2f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to HDF5 data file')
    parser.add_argument('--sequence_length', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_dir', type=str, default='checkpoints/perceiver_simple')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Perceiver IO Training (Simple)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data: {args.data}")
    print(f"Sequence: {args.sequence_length} frames")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Output: {out_dir}")
    print("=" * 60)

    # Dataset
    dataset = HDF5VideoDataset(
        args.data,
        sequence_length=args.sequence_length,
        img_size=args.img_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )

    # Model (Tier 1 size)
    print("\nCreating model...")
    model = CausalPerceiverIO(
        video_shape=(args.sequence_length, 3, args.img_size, args.img_size),
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
        num_quantizers=1,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params:,} parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Training loop
    print(f"\nTraining for {args.num_epochs} epochs...\n")

    best_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        metrics = train_epoch(model, dataloader, optimizer, device, epoch)

        print(f"\nEpoch {epoch}/{args.num_epochs}:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  CE: {metrics['ce_loss']:.4f} | VQ: {metrics['vq_loss']:.4f} | LPIPS: {metrics['perceptual_loss']:.4f}")

        # Visualize
        if epoch % 2 == 0:
            visualize(model, dataloader, device, epoch, out_dir)

        # Save checkpoint
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'metrics': metrics,
            }
            torch.save(checkpoint, out_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (loss: {best_loss:.4f})")

        # Save periodic checkpoint
        if epoch % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': metrics['loss'],
                'metrics': metrics,
            }
            torch.save(checkpoint, out_dir / f'checkpoint_epoch_{epoch}.pt')

    print("\n" + "=" * 60)
    print("✓ Training complete!")
    print("=" * 60)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints: {out_dir}")
    print()


if __name__ == '__main__':
    main()

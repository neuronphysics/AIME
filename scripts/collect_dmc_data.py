#!/usr/bin/env python
"""
Collect DMC trajectories and save to HDF5.

Creates small datasets for quick testing (Tier 1/2/3).

Usage:
    # Tier 1: Ultra-fast (100 episodes, 64x64)
    python scripts/collect_dmc_data.py --domain cartpole --task swingup \
        --num_episodes 100 --img_size 64 --output tier1_cartpole.hdf5

    # Tier 2: Medium (500 episodes, 84x84)
    python scripts/collect_dmc_data.py --domain reacher --task easy \
        --num_episodes 500 --img_size 84 --output tier2_reacher.hdf5

    # Tier 3: Full (2000 episodes, 84x84)
    python scripts/collect_dmc_data.py --domain humanoid --task walk \
        --num_episodes 2000 --img_size 84 --output tier3_humanoid.hdf5
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

# Enable headless rendering
os.environ['MUJOCO_GL'] = 'egl'  # or 'osmesa' if EGL not available

from dm_control import suite
from dm_control.suite.wrappers import pixels


def collect_random_policy_data(
    domain: str,
    task: str,
    num_episodes: int,
    max_steps: int = 1000,
    img_size: int = 64,
    camera_id: int = 0,
) -> dict:
    """Collect random policy trajectories from DMC."""

    # Load environment
    env = suite.load(domain, task)

    # Wrap for pixel observations
    env = pixels.Wrapper(
        env,
        pixels_only=False,
        render_kwargs={
            'height': img_size,
            'width': img_size,
            'camera_id': camera_id,
        }
    )

    action_spec = env.action_spec()

    # Storage
    episodes = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': [],
    }

    print(f"\nCollecting {num_episodes} episodes from {domain}-{task}...")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Action dim: {action_spec.shape[0]}")

    for ep in tqdm(range(num_episodes), desc="Episodes"):
        time_step = env.reset()

        ep_obs = []
        ep_actions = []
        ep_rewards = []
        ep_dones = []

        for step in range(max_steps):
            # Random action
            action = np.random.uniform(
                action_spec.minimum,
                action_spec.maximum,
                size=action_spec.shape
            )

            # Get pixel observation
            obs = time_step.observation['pixels']  # (H, W, 3)

            # Store
            ep_obs.append(obs)
            ep_actions.append(action)
            ep_rewards.append(time_step.reward if time_step.reward is not None else 0.0)
            ep_dones.append(time_step.last())

            # Step
            time_step = env.step(action)

            if time_step.last():
                break

        # Convert to arrays
        episodes['observations'].append(np.array(ep_obs))  # (T, H, W, 3)
        episodes['actions'].append(np.array(ep_actions))    # (T, A)
        episodes['rewards'].append(np.array(ep_rewards))
        episodes['dones'].append(np.array(ep_dones))

    return episodes


def save_to_hdf5(data: dict, output_path: Path):
    """Save episodes to HDF5 format."""

    print(f"\nSaving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        for ep_idx, (obs, actions, rewards, dones) in enumerate(zip(
            data['observations'],
            data['actions'],
            data['rewards'],
            data['dones']
        )):
            ep_group = f.create_group(f'episode_{ep_idx}')
            ep_group.create_dataset('observations', data=obs, compression='gzip')
            ep_group.create_dataset('actions', data=actions, compression='gzip')
            ep_group.create_dataset('rewards', data=rewards, compression='gzip')
            ep_group.create_dataset('dones', data=dones, compression='gzip')

    # Print stats
    total_frames = sum(len(ep) for ep in data['observations'])
    avg_length = total_frames / len(data['observations'])

    print(f"✓ Saved {len(data['observations'])} episodes")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Avg episode length: {avg_length:.1f}")
    print(f"  File size: {output_path.stat().st_size / 1024**2:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Collect DMC data for AIME")
    parser.add_argument('--domain', type=str, required=True,
                       help='DMC domain (e.g., cartpole, reacher, humanoid)')
    parser.add_argument('--task', type=str, required=True,
                       help='DMC task (e.g., swingup, easy, walk)')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of episodes to collect')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Max steps per episode')
    parser.add_argument('--img_size', type=int, default=64,
                       help='Image resolution (height=width)')
    parser.add_argument('--camera_id', type=int, default=0,
                       help='Camera ID for rendering')
    parser.add_argument('--output', type=str, required=True,
                       help='Output HDF5 file path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Collect data
    data = collect_random_policy_data(
        domain=args.domain,
        task=args.task,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        img_size=args.img_size,
        camera_id=args.camera_id,
    )

    # Save
    output_path = Path(args.output)
    save_to_hdf5(data, output_path)

    print(f"\n✓ Data collection complete!")
    print(f"\nNext steps:")
    print(f"  1. Use this data with: --base_path {output_path.parent}")
    print(f"  2. Or create a dataset class that loads from: {output_path}")


if __name__ == '__main__':
    main()

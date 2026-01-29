"""
Meta-World Data Collection Script for VRNN World Model Training

Collects transition data (observations, actions, rewards, done) from Meta-World
environments and saves them as H5 files for training the VRNN world model.

Usage:
    python -m VRNN.collect_metaworld_data \
        --task reach-v3 \
        --episodes 1000 \
        --output_dir ./transition_data/metaworld
"""

import argparse
import os
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any

import gymnasium as gym
import metaworld


# Meta-World task information
METAWORLD_INFO = {
    # MT1 single tasks (v3)
    'reach-v3': {'action_dim': 4, 'state_dim': 39},
    'push-v3': {'action_dim': 4, 'state_dim': 39},
    'pick-place-v3': {'action_dim': 4, 'state_dim': 39},
    'door-open-v3': {'action_dim': 4, 'state_dim': 39},
    'drawer-open-v3': {'action_dim': 4, 'state_dim': 39},
    'drawer-close-v3': {'action_dim': 4, 'state_dim': 39},
    'button-press-topdown-v3': {'action_dim': 4, 'state_dim': 39},
    'peg-insert-side-v3': {'action_dim': 4, 'state_dim': 39},
    'window-open-v3': {'action_dim': 4, 'state_dim': 39},
    'window-close-v3': {'action_dim': 4, 'state_dim': 39},
    # Add more tasks as needed
}


def get_available_tasks():
    """Return list of available Meta-World tasks."""
    return list(METAWORLD_INFO.keys())


def create_metaworld_env(task_name: str, seed: int = 42, render_mode: str = "rgb_array"):
    """
    Create a Meta-World environment for the given task.
    
    Args:
        task_name: Name of the Meta-World task (e.g., 'reach-v3')
        seed: Random seed for reproducibility
        render_mode: Render mode for the environment
        
    Returns:
        Gymnasium environment
    """
    env = gym.make(
        "Meta-World/MT1",
        env_name=task_name,
        seed=seed,
        render_mode=render_mode
    )
    return env


def collect_episode(
    env,
    policy: str = "random",
    max_steps: int = 500,
    img_size: tuple = (64, 64)
) -> Dict[str, np.ndarray]:
    """
    Collect a single episode from the environment.
    
    Args:
        env: Gymnasium environment
        policy: Policy type ('random' or 'scripted')
        max_steps: Maximum steps per episode
        img_size: Size of rendered images (height, width)
        
    Returns:
        Dictionary containing episode data
    """
    observations = []
    actions = []
    rewards = []
    dones = []
    infos_list = []
    
    obs, info = env.reset()
    
    for step in range(max_steps):
        # Render RGB observation
        rgb = env.render()
        
        # Resize if necessary
        if rgb.shape[:2] != img_size:
            import cv2
            rgb = cv2.resize(rgb, (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA)
        
        observations.append(rgb)
        
        # Select action based on policy
        if policy == "random":
            action = env.action_space.sample()
        else:
            # Could add scripted policies here
            action = env.action_space.sample()
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        actions.append(action)
        rewards.append(reward)
        dones.append(float(done))
        infos_list.append(info)
        
        if done:
            break
        
        obs = next_obs
    
    # Stack into arrays
    episode_data = {
        'observation_pixels': np.stack(observations, axis=0),  # [T, H, W, C]
        'action': np.stack(actions, axis=0),                    # [T, action_dim]
        'reward': np.array(rewards, dtype=np.float32),         # [T]
        'done': np.array(dones, dtype=np.float32),             # [T]
    }
    
    # Add success info if available
    if infos_list and 'success' in infos_list[-1]:
        episode_data['success'] = float(infos_list[-1]['success'])
    
    return episode_data


def save_episode_h5(episode_data: Dict[str, np.ndarray], save_path: Path):
    """Save episode data to HDF5 file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(save_path, 'w') as f:
        for key, value in episode_data.items():
            f.create_dataset(key, data=value, compression='gzip')


def collect_dataset(
    task_name: str,
    output_dir: str,
    num_episodes: int = 1000,
    policy: str = "random",
    max_steps: int = 500,
    img_size: tuple = (64, 64),
    seed: int = 42
):
    """
    Collect a dataset of episodes from a Meta-World task.
    
    Args:
        task_name: Name of the Meta-World task
        output_dir: Directory to save episodes
        num_episodes: Number of episodes to collect
        policy: Policy type ('random' or 'scripted')
        max_steps: Maximum steps per episode
        img_size: Size of rendered images
        seed: Random seed
    """
    output_path = Path(output_dir) / task_name / policy
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Collecting {num_episodes} episodes for task: {task_name}")
    print(f"Policy: {policy}, Max steps: {max_steps}, Image size: {img_size}")
    print(f"Output directory: {output_path}")
    
    # Create environment
    env = create_metaworld_env(task_name, seed=seed)
    
    total_steps = 0
    total_rewards = 0
    successes = 0
    
    for ep_idx in tqdm(range(num_episodes), desc="Collecting episodes"):
        # Collect episode
        episode_data = collect_episode(
            env,
            policy=policy,
            max_steps=max_steps,
            img_size=img_size
        )
        
        # Save episode
        ep_file = output_path / f"episode_{ep_idx:05d}.h5"
        save_episode_h5(episode_data, ep_file)
        
        # Track statistics
        total_steps += len(episode_data['reward'])
        total_rewards += episode_data['reward'].sum()
        if 'success' in episode_data:
            successes += episode_data['success']
    
    env.close()
    
    # Print summary
    print(f"\n=== Collection Summary ===")
    print(f"Task: {task_name}")
    print(f"Episodes: {num_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Average episode length: {total_steps / num_episodes:.1f}")
    print(f"Average reward: {total_rewards / num_episodes:.3f}")
    print(f"Success rate: {successes / num_episodes * 100:.1f}%")
    print(f"Data saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect Meta-World transition data for VRNN training"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default="reach-v3",
        choices=get_available_tasks(),
        help="Meta-World task name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./transition_data/metaworld",
        help="Output directory for collected data"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of episodes to collect"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="random",
        choices=["random"],
        help="Policy to use for data collection"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=64,
        help="Image size (height and width)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    collect_dataset(
        task_name=args.task,
        output_dir=args.output_dir,
        num_episodes=args.episodes,
        policy=args.policy,
        max_steps=args.max_steps,
        img_size=(args.img_size, args.img_size),
        seed=args.seed
    )


if __name__ == "__main__":
    main()

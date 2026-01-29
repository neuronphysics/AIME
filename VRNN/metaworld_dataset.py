"""
Meta-World Dataset for VRNN World Model Training

PyTorch Dataset class for loading pre-collected Meta-World transition data
for training the VRNN (DPGMM Variational Recurrent Autoencoder) world model.

This module provides:
- MetaWorldInfo: Task dimension information for Meta-World environments
- MetaWorldDataset: PyTorch Dataset for loading collected Meta-World episodes
"""

import torch
import numpy as np
import h5py
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset
from tqdm import tqdm


class MetaWorldInfo:
    """Meta-World task information (analogous to DMCVBInfo for DMC)"""
    
    METAWORLD_INFO = {
        # Single tasks (MT1) - Sawyer robot with 4D end-effector control
        'reach-v3': {
            'action_dim': 4,
            'state_dim': 39,
            'max_episode_length': 500,
        },
        'push-v3': {
            'action_dim': 4,
            'state_dim': 39,
            'max_episode_length': 500,
        },
        'pick-place-v3': {
            'action_dim': 4,
            'state_dim': 39,
            'max_episode_length': 500,
        },
        'door-open-v3': {
            'action_dim': 4,
            'state_dim': 39,
            'max_episode_length': 500,
        },
        'drawer-open-v3': {
            'action_dim': 4,
            'state_dim': 39,
            'max_episode_length': 500,
        },
        'drawer-close-v3': {
            'action_dim': 4,
            'state_dim': 39,
            'max_episode_length': 500,
        },
        'button-press-topdown-v3': {
            'action_dim': 4,
            'state_dim': 39,
            'max_episode_length': 500,
        },
        'peg-insert-side-v3': {
            'action_dim': 4,
            'state_dim': 39,
            'max_episode_length': 500,
        },
        'window-open-v3': {
            'action_dim': 4,
            'state_dim': 39,
            'max_episode_length': 500,
        },
        'window-close-v3': {
            'action_dim': 4,
            'state_dim': 39,
            'max_episode_length': 500,
        },
        'faucet-open-v3': {
            'action_dim': 4,
            'state_dim': 39,
            'max_episode_length': 500,
        },
        'hammer-v3': {
            'action_dim': 4,
            'state_dim': 39,
            'max_episode_length': 500,
        },
        'assembly-v3': {
            'action_dim': 4,
            'state_dim': 39,
            'max_episode_length': 500,
        },
    }
    
    @staticmethod
    def get_action_dim(task_name: str) -> int:
        if task_name in MetaWorldInfo.METAWORLD_INFO:
            return MetaWorldInfo.METAWORLD_INFO[task_name]['action_dim']
        # Default for all Meta-World tasks
        return 4
    
    @staticmethod
    def get_state_dim(task_name: str) -> int:
        if task_name in MetaWorldInfo.METAWORLD_INFO:
            return MetaWorldInfo.METAWORLD_INFO[task_name]['state_dim']
        return 39
    
    @staticmethod
    def get_max_episode_length(task_name: str) -> int:
        if task_name in MetaWorldInfo.METAWORLD_INFO:
            return MetaWorldInfo.METAWORLD_INFO[task_name]['max_episode_length']
        return 500
    
    @staticmethod
    def is_metaworld_task(task_name: str) -> bool:
        """Check if task name is a Meta-World task."""
        return task_name.endswith('-v3') or task_name in MetaWorldInfo.METAWORLD_INFO


class MetaWorldDataset(Dataset):
    """
    PyTorch Dataset for Meta-World transition data.
    
    Loads pre-collected episodes from H5 files and provides sequences
    for training the VRNN world model.
    """
    
    def __init__(
        self,
        data_dir: str,
        task_name: str = 'reach-v3',
        policy_level: str = 'random',
        split: str = 'train',
        sequence_length: int = 10,
        frame_stack: int = 1,
        img_height: int = 64,
        img_width: int = 64,
        normalize_images: bool = True,
        add_rewards: bool = True,
        transform: Optional[callable] = None,
        training_percent: float = 0.8,
    ):
        """
        Args:
            data_dir: Base directory containing metaworld data
            task_name: Meta-World task name (e.g., 'reach-v3')
            policy_level: Policy used for data collection ('random', etc.)
            split: 'train' or 'eval'
            sequence_length: Number of frames per sequence
            frame_stack: Number of frames to stack per observation
            img_height: Image height
            img_width: Image width
            normalize_images: Whether to normalize images to [-1, 1]
            add_rewards: Whether to include rewards in output
            transform: Optional data augmentation transform
            training_percent: Fraction of episodes for training
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.task_name = task_name
        self.policy_level = policy_level
        self.split = split
        self.sequence_length = sequence_length
        self.frame_stack = frame_stack
        self.img_height = img_height
        self.img_width = img_width
        self.normalize_images = normalize_images
        self.add_rewards = add_rewards
        self.transform = transform
        self.training_percent = training_percent
        
        # Get task info
        self.action_dim = MetaWorldInfo.get_action_dim(task_name)
        self.state_dim = MetaWorldInfo.get_state_dim(task_name)
        
        # Load episodes
        self.episodes = self._load_all_episodes()
        
        # Compute episode lengths
        self.episode_lengths = self._compute_episode_lengths()
        
        # Create sequence indices
        self.sequence_indices = self._create_sequence_indices()
        
        print(f"MetaWorldDataset loaded:")
        print(f"  Task: {task_name}")
        print(f"  Episodes: {len(self.episodes)}")
        print(f"  Total sequences: {len(self.sequence_indices)}")
        print(f"  Action dim: {self.action_dim}")
    
    def _load_episode_paths(self) -> List[Path]:
        """Load paths to all episode files."""
        # Check for metaworld subdirectory
        data_path = self.data_dir / "metaworld" / self.task_name / self.policy_level
        
        if not data_path.exists():
            # Try without 'metaworld' prefix
            data_path = self.data_dir / self.task_name / self.policy_level
        
        if not data_path.exists():
            raise ValueError(f"Data path not found: {data_path}")
        
        # Find all H5 files
        episode_files = sorted(data_path.glob("*.h5"))
        
        if len(episode_files) == 0:
            raise ValueError(f"No H5 files found in {data_path}")
        
        # Split into train/eval
        n_episodes = len(episode_files)
        n_train = int(self.training_percent * n_episodes)
        
        if self.split == 'train':
            return episode_files[:n_train]
        else:
            return episode_files[n_train:]
    
    def _load_all_episodes(self) -> List[Dict[str, np.ndarray]]:
        """Load all episodes into memory."""
        episode_paths = self._load_episode_paths()
        episodes = []
        
        print(f"Loading {len(episode_paths)} episodes...")
        for ep_path in tqdm(episode_paths):
            try:
                episode_data = self._load_h5_episode(ep_path)
                if episode_data is not None:
                    episodes.append(episode_data)
            except Exception as e:
                print(f"Error loading {ep_path}: {e}")
                continue
        
        return episodes
    
    def _load_h5_episode(self, h5_path: Path) -> Dict[str, np.ndarray]:
        """Load episode from HDF5 file."""
        data = {}
        
        with h5py.File(h5_path, 'r') as f:
            # Load observations
            if 'observation_pixels' in f:
                data['observation_pixels'] = f['observation_pixels'][:]
            elif 'pixels' in f:
                data['observation_pixels'] = f['pixels'][:]
            else:
                raise KeyError(f"No observation pixels found in {h5_path}")
            
            # Load actions
            if 'action' in f:
                data['action'] = f['action'][:]
            else:
                raise KeyError(f"No actions found in {h5_path}")
            
            # Load rewards
            if 'reward' in f:
                data['reward'] = f['reward'][:]
            
            # Load done flags
            if 'done' in f:
                data['done'] = f['done'][:].astype(np.float32)
            else:
                data['done'] = np.zeros(len(data['action']), dtype=np.float32)
                data['done'][-1] = 1.0
        
        return data
    
    def _compute_episode_lengths(self) -> List[int]:
        """Compute usable length for each episode."""
        lengths = []
        min_required = self.sequence_length + self.frame_stack - 1
        
        for ep in self.episodes:
            obs_len = len(ep['observation_pixels'])
            act_len = len(ep['action'])
            done_len = len(ep['done'])
            
            ep_len = min(obs_len, act_len, done_len)
            
            # Clip to first done
            done_idx = np.where(ep['done'][:ep_len] > 0)[0]
            if len(done_idx) > 0:
                ep_len = min(ep_len, done_idx[0] + 1)
            
            if ep_len >= min_required:
                lengths.append(ep_len)
        
        return lengths
    
    def _create_sequence_indices(self) -> List[Tuple[int, int]]:
        """Create (episode_idx, start_idx) pairs for all valid sequences."""
        indices = []
        min_required = self.sequence_length + self.frame_stack - 1
        
        for ep_idx, ep in enumerate(self.episodes):
            obs_len = len(ep['observation_pixels'])
            act_len = len(ep['action'])
            done_len = len(ep['done'])
            
            ep_len = min(obs_len, act_len, done_len)
            
            done_idx = np.where(ep['done'][:ep_len] > 0)[0]
            if len(done_idx) > 0:
                ep_len = min(ep_len, done_idx[0] + 1)
            
            if ep_len < min_required:
                continue
            
            max_start = ep_len - min_required + 1
            for start_idx in range(max_start):
                indices.append((ep_idx, start_idx))
        
        return indices
    
    def _process_observation(self, obs: np.ndarray) -> torch.Tensor:
        """Process observation: resize, normalize, convert to tensor."""
        # Resize if needed
        if obs.shape[0] != self.img_height or obs.shape[1] != self.img_width:
            obs = cv2.resize(obs, (self.img_width, self.img_height), 
                           interpolation=cv2.INTER_AREA)
        
        # Normalize to [-1, 1]
        if self.normalize_images:
            if obs.dtype == np.uint8:
                obs = obs.astype(np.float32) / 255.0
            elif obs.max() > 1.5:
                obs = obs.astype(np.float32) / 255.0
            else:
                obs = obs.astype(np.float32)
            
            obs = (obs * 2.0) - 1.0
            obs = np.clip(obs, -1.0, 1.0)
        
        # [H, W, C] -> [C, H, W]
        obs = np.transpose(obs, (2, 0, 1))
        return torch.from_numpy(obs).float()
    
    def __len__(self) -> int:
        return len(self.sequence_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence of transitions."""
        ep_idx, start_idx = self.sequence_indices[idx]
        episode = self.episodes[ep_idx]
        
        end_idx = start_idx + self.sequence_length + self.frame_stack - 1
        
        # Get observations
        obs_raw = episode['observation_pixels'][start_idx:end_idx]
        frames = [self._process_observation(obs_raw[i]) for i in range(len(obs_raw))]
        
        # Stack frames
        processed_obs = []
        for t in range(self.frame_stack, len(frames) + 1):
            stacked = torch.cat(frames[t - self.frame_stack:t], dim=0)
            processed_obs.append(stacked)
        
        observations = torch.stack(processed_obs, dim=0)  # [T, C*frame_stack, H, W]
        
        # Actions
        action_start = start_idx + self.frame_stack - 1
        action_end = action_start + len(processed_obs)
        actions = torch.from_numpy(
            episode['action'][action_start:action_end].astype(np.float32)
        )
        
        sample = {
            'observations': observations,
            'actions': actions,
        }
        
        # Add rewards
        if self.add_rewards and 'reward' in episode:
            rewards = episode['reward'][action_start:action_end]
            sample['rewards'] = torch.from_numpy(rewards.astype(np.float32))
        
        # Add done flags
        done_flags = episode['done'][action_start:action_end]
        sample['done'] = torch.from_numpy(done_flags.astype(np.float32))
        
        sample['episode_idx'] = torch.tensor(ep_idx, dtype=torch.long)
        sample['start_idx'] = torch.tensor(start_idx, dtype=torch.long)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def create_metaworld_dataloader(
    data_dir: str,
    task_name: str,
    batch_size: int = 32,
    sequence_length: int = 10,
    num_workers: int = 4,
    **kwargs
):
    """Helper function to create train and eval dataloaders."""
    from torch.utils.data import DataLoader
    
    train_dataset = MetaWorldDataset(
        data_dir=data_dir,
        task_name=task_name,
        split='train',
        sequence_length=sequence_length,
        **kwargs
    )
    
    eval_dataset = MetaWorldDataset(
        data_dir=data_dir,
        task_name=task_name,
        split='eval',
        sequence_length=sequence_length,
        **kwargs
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, eval_loader

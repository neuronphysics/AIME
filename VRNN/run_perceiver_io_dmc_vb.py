import os
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import json
from collections import defaultdict
import random, math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.hub import load_state_dict_from_url
from torchvision import models
from einops import rearrange
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import wandb
import h5py, zlib
import tensorflow as tf
from torch.utils.data._utils.collate import default_collate
import sys
from scipy import linalg
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
# Import from VRNN - adjust path if needed
sys.path.append('VRNN')
from VRNN.perceiver.video_prediction_perceiverIO import CausalPerceiverIO

# Plotting configuration
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.max_open_warning'] = 50


# ========================
# FID InceptionV3 Implementation
# ========================

FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


class FIDInceptionA(models.inception.InceptionA):
    """InceptionA block patched for FID computation"""
    def __init__(self, in_channels: int, pool_features: int) -> None:
        super().__init__(in_channels, pool_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(models.inception.InceptionC):
    """InceptionC block patched for FID computation"""
    def __init__(self, in_channels: int, channels_7x7: int) -> None:
        super().__init__(in_channels, channels_7x7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE1(models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""
    def __init__(self, in_channels: int) -> None:
        super().__init__(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE2(models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""
    def __init__(self, in_channels: int) -> None:
        super().__init__(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network for FID computation"""

    DEFAULT_BLOCK_INDEX = 3

    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling features
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks: List[int] = None,
                 resize_input: bool = True,
                 normalize_input: bool = True,
                 requires_grad: bool = False,
                 use_fid_inception: bool = True) -> None:
        """Build pretrained InceptionV3 for FID computation"""
        super().__init__()

        if output_blocks is None:
            output_blocks = [self.DEFAULT_BLOCK_INDEX]

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, 'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = self.fid_inception_v3()
        else:
            inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    @staticmethod
    def fid_inception_v3():
        """Build FID-specific inception v3 architecture"""
        inception = models.inception_v3(num_classes=1008, aux_logits=False, init_weights=False)
        inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
        inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
        inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
        inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
        inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
        inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
        inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
        inception.Mixed_7b = FIDInceptionE1(1280)
        inception.Mixed_7c = FIDInceptionE2(2048)

        state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
        inception.load_state_dict(state_dict)
        return inception

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get requested activation maps from input"""
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        if self.normalize_input:
            x = 2 * x - 1

        output = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                output.append(x)

            if idx == self.last_needed_block:
                break

        return output


class FIDCalculator:
    """Calculate Frechet Inception Distance"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)
        self.inception.eval()
    
    @torch.no_grad()
    def get_activations(self, images: torch.Tensor, batch_size: int = 50) -> np.ndarray:
        """Extract inception features from images"""
        n_images = len(images)
        n_batches = (n_images + batch_size - 1) // batch_size
        
        pred_arr = []
        
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_images)
            batch = images[start:end].to(self.device)
            
            pred = self.inception(batch)[0]
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr.append(pred)
        
        return np.concatenate(pred_arr, axis=0)
    
    @staticmethod
    def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, 
                                   mu2: np.ndarray, sigma2: np.ndarray, 
                                   eps: float = 1e-6) -> float:
        """Calculate Frechet distance between two Gaussians"""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        diff = mu1 - mu2
        
        # Product might be complex with imaginary component small enough to ignore
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical error might give complex numbers
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f'Imaginary component {m}')
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        return (diff.dot(diff) + np.trace(sigma1) + 
                np.trace(sigma2) - 2 * tr_covmean)
    
    def compute_statistics(self, images: torch.Tensor, batch_size: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and covariance of inception features"""
        act = self.get_activations(images, batch_size)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma
    
    def calculate_fid(self, real_images: torch.Tensor, fake_images: torch.Tensor, 
                     batch_size: int = 50) -> float:
        """Calculate FID between real and fake images"""
        m1, s1 = self.compute_statistics(real_images, batch_size)
        m2, s2 = self.compute_statistics(fake_images, batch_size)
        
        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)
        return float(fid_value)


# ========================
# Data Loading Utilities
# ========================

class TFRecordConverter:
    """Convert TFRecord data to PyTorch format"""
    
    @staticmethod
    def decode_zlib_observation(obs_bytes):
        """Decode zlib-compressed observation"""
        try:
            decompressed = zlib.decompress(obs_bytes)
            obs_array = np.frombuffer(decompressed, dtype=np.uint8)
            
            # Reshape based on expected dimensions
            if len(obs_array) == 84 * 84 * 3:
                obs = obs_array.reshape(84, 84, 3)
            elif len(obs_array) == 64 * 64 * 3:
                obs = obs_array.reshape(64, 64, 3)
            else:
                return None
            
            return obs
            
        except (zlib.error, ValueError) as e:
            return None
    
    @staticmethod
    def parse_tfrecord_episode(tfrecord_path, action_dim=21):
        """Parse a TFRecord file into episode data"""
        dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type='')
        
        timesteps = []
        
        for raw_record in dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            features = example.features.feature
            
            if 'episode_id' in features or 'episode_length' in features:
                continue
            
            timestep_data = {}
            
            # Robust observation decoding
            if 'steps/observation/pixels' in features:
                obs_bytes = features['steps/observation/pixels'].bytes_list.value[0]
                if obs_bytes[:2] == b'\x78\x9c':
                    obs = TFRecordConverter.decode_zlib_observation(obs_bytes)
                    if obs is None:
                        continue
                    timestep_data['observation'] = obs
                else:
                    try:
                        obs_tensor = tf.io.decode_image(obs_bytes, channels=3)
                        timestep_data['observation'] = obs_tensor.numpy()
                    except Exception as e:
                        print(f"Observation decoding failed: {e}")
                        continue
            
            # Parse actions
            if 'steps/action' in features:
                action_bytes = features['steps/action'].bytes_list.value[0]
                if action_bytes[:2] == b'\x78\x9c':
                    try:
                        decompressed = zlib.decompress(action_bytes)
                        action_values = np.frombuffer(decompressed, dtype=np.float64)
                    except zlib.error:
                        continue
                else:
                    action_values = np.frombuffer(action_bytes, dtype=np.float64)
                
                if len(action_values) == action_dim:
                    timestep_data['action'] = action_values
                else:
                    continue
            
            # Parse rewards
            if 'steps/reward' in features:
                reward_values = features['steps/reward'].float_list.value
                if reward_values:
                    timestep_data['reward'] = float(reward_values[0])
            
            # Parse termination flags
            for flag_name in ['is_first', 'is_last', 'is_terminal']:
                feature_key = f'steps/{flag_name}'
                if feature_key in features:
                    flag_values = features[feature_key].int64_list.value
                    if flag_values:
                        timestep_data[flag_name] = int(flag_values[0])
            
            if 'observation' in timestep_data and 'action' in timestep_data:
                timesteps.append(timestep_data)
        
        episode_data = {}
        
        if timesteps:
            episode_data['observations'] = np.stack([t['observation'] for t in timesteps])
            episode_data['action'] = np.stack([t['action'] for t in timesteps])
            episode_data['reward'] = np.array([t.get('reward', 0.0) for t in timesteps], dtype=np.float32)
            
            done_flags = []
            for i, t in enumerate(timesteps):
                is_done = t.get('is_last', 0) or t.get('is_terminal', 0)
                done_flags.append(float(is_done))
            
            episode_data['done'] = np.array(done_flags, dtype=np.float32)
            
            if not np.any(episode_data['done']):
                episode_data['done'][-1] = 1.0
            
            episode_data['episode_length'] = len(timesteps)
        
        return episode_data


class DMCVBDataset(Dataset):
    """PyTorch Dataset for DMC Vision Benchmark with dataset label"""
    
    def __init__(
        self,
        root: str,
        sequence_length: int = 16,
        stride: int = 1,
        img_height: int = 64,
        img_width: int = 64,
        channels: int = 3,
        split: str = 'train',
        transform: Optional[callable] = None,
        maintain_episode_order: bool = True,
        train_split_ratio: float = 0.7,
        dataset_label: str = "unknown",  # NEW: Label for this dataset (e.g., "expert", "medium", "mixed")
    ):
        super().__init__()
        
        self.root = Path(root) if isinstance(root, str) else root
        self.sequence_length = sequence_length
        self.stride = stride
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.transform = transform
        self.maintain_episode_order = maintain_episode_order
        self.dataset_label = dataset_label  # Store the dataset label
        
        self.episodes = []
        self.sequences = []
        
        print(f"\nLoading {split} dataset from {self.root} (label: {dataset_label})")
        
        # Try DMC Vision Benchmark naming pattern first (current directory)
        tfrecord_files = list(self.root.glob("distracting_control-*.tfrecord-*"))
        if not tfrecord_files:
            # Try general tfrecord patterns in current directory
            tfrecord_files = list(self.root.glob("*.tfrecord*"))
        
        # If no files found, search recursively in subdirectories
        if not tfrecord_files:
            print(f"No files in root, searching subdirectories recursively...")
            tfrecord_files = list(self.root.rglob("distracting_control-*.tfrecord-*"))
            if not tfrecord_files:
                tfrecord_files = list(self.root.rglob("*.tfrecord*"))
        
        hdf5_files = list(self.root.glob("*.hdf5"))
        if not hdf5_files:
            # Also check for .h5 extension
            hdf5_files = list(self.root.glob("*.h5"))
        
        # If no HDF5 files found in root, search recursively
        if not hdf5_files:
            hdf5_files = list(self.root.rglob("*.hdf5"))
            if not hdf5_files:
                hdf5_files = list(self.root.rglob("*.h5"))
        
        # Apply train/val split if needed
        all_files = tfrecord_files if tfrecord_files else hdf5_files
        if all_files:
            # Sort for reproducibility, then shuffle with fixed seed
            all_files = sorted(all_files)
            import random
            random.seed(42)
            random.shuffle(all_files)
            
            # Split into train/val
            n_files = len(all_files)
            n_train = int(train_split_ratio * n_files)
            
            if split == 'train':
                selected_files = all_files[:n_train]
                print(f"Using {len(selected_files)}/{n_files} files for training")
            else:  # val
                selected_files = all_files[n_train:]
                print(f"Using {len(selected_files)}/{n_files} files for validation")
            
            # Update file lists based on split
            if tfrecord_files:
                tfrecord_files = selected_files
                hdf5_files = []
            else:
                hdf5_files = selected_files
                tfrecord_files = []
        
        if tfrecord_files:
            print(f"Loading {len(tfrecord_files)} TFRecord files")
            for tfrecord_path in tqdm(tfrecord_files, desc=f"Loading {dataset_label} episodes"):
                episode_data = TFRecordConverter.parse_tfrecord_episode(tfrecord_path)
                if episode_data and len(episode_data['observations']) >= sequence_length:
                    self.episodes.append(episode_data)
        
        elif hdf5_files:
            print(f"Loading {len(hdf5_files)} HDF5 files")
            for hdf5_path in tqdm(hdf5_files, desc=f"Loading {dataset_label} episodes"):
                with h5py.File(hdf5_path, 'r') as f:
                    observations = f['observations'][:]
                    if len(observations) >= sequence_length:
                        episode_data = {
                            'observations': observations,
                            'episode_length': len(observations)
                        }
                        self.episodes.append(episode_data)
        else:
            raise ValueError(f"No TFRecord or HDF5 files found in {self.root}")
        
        self._create_sequences()
        
        print(f"Loaded {len(self.episodes)} episodes, created {len(self.sequences)} sequences from {dataset_label}")
    
    def _create_sequences(self):
        """Create sequences from episodes"""
        for episode_idx, episode in enumerate(self.episodes):
            observations = episode['observations']
            episode_length = len(observations)
            
            for start_idx in range(0, episode_length - self.sequence_length + 1, self.stride):
                end_idx = start_idx + self.sequence_length
                
                self.sequences.append({
                    'episode_idx': episode_idx,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'dataset_label': self.dataset_label  # Include label in sequence
                })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get a video sequence"""
        seq_info = self.sequences[idx]
        episode = self.episodes[seq_info['episode_idx']]
        
        # Extract observation sequence
        obs_sequence = episode['observations'][seq_info['start_idx']:seq_info['end_idx']]
        
        # Resize if needed
        if obs_sequence.shape[1:3] != (self.img_height, self.img_width):
            
            resized_sequence = []
            for frame in obs_sequence:
                resized_frame = cv2.resize(frame, (self.img_width, self.img_height))
                resized_sequence.append(resized_frame)
            obs_sequence = np.stack(resized_sequence)
        
        # Convert to tensor: (T, H, W, C) -> (T, C, H, W)
        obs_tensor = torch.from_numpy(obs_sequence).float().permute(0, 3, 1, 2) / 255.0
        obs_tensor = obs_tensor * 2.0 - 1.0  # Scale to [-1, 1]
        if self.transform:
            obs_tensor = self.transform(obs_tensor)
        
        # Return with dataset label
        return {
            'video': obs_tensor,
            'dataset_label': seq_info['dataset_label']
        }


def create_mixed_dataset(
    base_path: str,
    task: str,
    dataset_types: List[str],
    subdirs: List[str],
    split: str = 'train',
    sequence_length: int = 16,
    img_height: int = 64,
    img_width: int = 64,
    train_split_ratio: float = 0.7,
    dataset_weights: Optional[Dict[str, float]] = None,
) -> Dataset:
    """
    Create a mixed dataset from multiple dataset types and subdirectories.
    
    Args:
        base_path: Base path containing the dataset structure
        task: Task name (e.g., 'humanoid_walk')
        dataset_types: List of dataset types (e.g., ['expert', 'medium', 'mixed'])
        subdirs: List of subdirectories to include (e.g., ['dynamic_medium', 'none', 'static_medium'])
        split: 'train' or 'val'
        sequence_length: Length of video sequences
        img_height: Image height
        img_width: Image width
        train_split_ratio: Ratio for train/val split
        dataset_weights: Optional dict mapping dataset_type to sampling weight
        
    Returns:
        ConcatDataset containing all subdatasets
    """
    base_path = Path(base_path)
    datasets = []
    dataset_info = []
    
    print(f"\n{'='*60}")
    print(f"Creating Mixed Dataset for {task} ({split} split)")
    print(f"{'='*60}")
    
    for dataset_type in dataset_types:
        for subdir in subdirs:
            data_path = base_path / task / dataset_type / subdir
            
            if not data_path.exists():
                print(f"Warning: Path does not exist: {data_path}")
                continue
            
            label = f"{dataset_type}_{subdir}"
            
            try:
                dataset = DMCVBDataset(
                    root=str(data_path),
                    sequence_length=sequence_length,
                    img_height=img_height,
                    img_width=img_width,
                    split=split,
                    train_split_ratio=train_split_ratio,
                    dataset_label=label
                )
                
                if len(dataset) > 0:
                    datasets.append(dataset)
                    dataset_info.append({
                        'label': label,
                        'dataset_type': dataset_type,
                        'subdir': subdir,
                        'size': len(dataset),
                        'episodes': len(dataset.episodes)
                    })
                    print(f"✓ Added {label}: {len(dataset)} sequences, {len(dataset.episodes)} episodes")
                else:
                    print(f"✗ Skipped {label}: No valid sequences")
                    
            except Exception as e:
                print(f"✗ Error loading {label}: {e}")
    
    if not datasets:
        raise ValueError("No valid datasets found!")
    
    # Create concatenated dataset
    mixed_dataset = ConcatDataset(datasets)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Mixed Dataset Summary ({split})")
    print(f"{'='*60}")
    total_sequences = sum(info['size'] for info in dataset_info)
    total_episodes = sum(info['episodes'] for info in dataset_info)
    
    for info in dataset_info:
        pct = 100 * info['size'] / total_sequences
        print(f"  {info['label']:25s}: {info['size']:6d} seqs ({pct:5.1f}%), {info['episodes']:4d} episodes")
    
    print(f"  {'TOTAL':25s}: {total_sequences:6d} seqs (100.0%), {total_episodes:4d} episodes")
    print(f"{'='*60}\n")
    
    return mixed_dataset


def calculate_ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim_batch(
    real: torch.Tensor,
    pred: torch.Tensor,
    max_workers: Optional[int] = None,
) -> float:
    """
    Wrapper around `calculate_ssim` for a batch of images.

    real, pred: [N, C, H, W] torch tensors in [0,1].
    Returns mean SSIM over N images.

    Implementation:
      - detach + move to CPU once
      - convert to numpy in [0,255], CHW -> HWC
      - run your original `calculate_ssim(img1, img2)` in parallel.
    """
    assert real.shape == pred.shape, "SSIM: real and pred must have same shape"
    assert real.dim() == 4, "SSIM expects tensors of shape [N, C, H, W]"

    # 1) Detach, move to CPU, clamp, scale to [0,255]
    # shape: [N,C,H,W]
    real_np = real.detach().clamp(0, 1).cpu().numpy()
    pred_np = pred.detach().clamp(0, 1).cpu().numpy()

    real_np = (real_np * 255.0).astype(np.uint8)
    pred_np = (pred_np * 255.0).astype(np.uint8)

    # 2) Convert to HWC for your function: [N,C,H,W] -> [N,H,W,C]
    real_np = np.transpose(real_np, (0, 2, 3, 1))
    pred_np = np.transpose(pred_np, (0, 2, 3, 1))

    N = real_np.shape[0]

    # 3) Prepare pairs of images for SSIM
    pairs = [(real_np[i], pred_np[i]) for i in range(N)]

    # 4) Run calculate_ssim in parallel over the batch
    def _worker(pair):
        img1, img2 = pair
        return calculate_ssim(img1, img2)

    if max_workers is None or max_workers <= 1:
        # fallback: sequential (still using your original function)
        scores = [_worker(p) for p in pairs]
    else:
        # Thread-based parallelism; cv2 is C code and generally releases the GIL
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            scores = list(ex.map(_worker, pairs))

    return float(np.mean(scores))

# Visualization

class Visualizer:
    """Visualization utilities for video prediction"""
    
    @staticmethod
    def plot_video_sequence(context, ground_truth, predictions, save_path=None):
        """Plot context, ground truth, and predictions side by side"""
        T_context = context.shape[0]
        T_pred = predictions.shape[0]
        T_total = T_context + T_pred
        
        fig = plt.figure(figsize=(T_total * 2, 6))
        gs = GridSpec(3, T_total, hspace=0.3, wspace=0.1)
        
        # Plot context frames
        for t in range(T_context):
            ax = fig.add_subplot(gs[0, t])
            ax.imshow(context[t].permute(1, 2, 0).cpu().numpy())
            ax.set_title(f'Context {t}', fontsize=8)
            ax.axis('off')
        
        # Plot ground truth
        for t in range(T_pred):
            ax = fig.add_subplot(gs[1, T_context + t])
            ax.imshow(ground_truth[t].permute(1, 2, 0).cpu().numpy())
            ax.set_title(f'GT {t}', fontsize=8)
            ax.axis('off')
        
        # Plot predictions
        for t in range(T_pred):
            ax = fig.add_subplot(gs[2, T_context + t])
            ax.imshow(predictions[t].permute(1, 2, 0).cpu().numpy())
            ax.set_title(f'Pred {t}', fontsize=8)
            ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        return fig
    
    @staticmethod
    def plot_context_and_multistep_predictions(
        context,          # (T_ctx, C, H, W), [0,1]
        context_recon,    # (T_ctx, C, H, W), [0,1]
        gt_future,        # (T_f, C, H, W), [0,1], T_f <= 3
        pred_future_3step,# (T_f, C, H, W), [0,1], T_f <= 3 (AR rollout)
        save_path=None,
    ):
        """
        Plot:
          Row 0: Context (ground truth, first up to 3 frames)
          Row 1: Context reconstruction (same frames)
          Row 2: Future ground truth (first up to 3 frames)
          Row 3: 3-step autoregressive predictions (steps 1..T_f)
        """
        T_ctx = context.shape[0]
        T_ctx_show = min(T_ctx, 3)
        T_f = gt_future.shape[0]  # <= 3
        n_cols = max(T_ctx_show, T_f, 3)

        fig = plt.figure(figsize=(n_cols * 2.5, 8))
        gs = GridSpec(4, n_cols, hspace=0.3, wspace=0.1)

        # Row 0: context GT
        for t in range(T_ctx_show):
            ax = fig.add_subplot(gs[0, t])
            ax.imshow(context[t].permute(1, 2, 0).cpu().numpy())
            ax.set_title(f'Context GT {t}', fontsize=8)
            ax.axis('off')
        for t in range(T_ctx_show, n_cols):
            ax = fig.add_subplot(gs[0, t])
            ax.axis('off')
        fig.axes[0].set_ylabel("Context GT", fontsize=9)

        # Row 1: context recon
        for t in range(T_ctx_show):
            ax = fig.add_subplot(gs[1, t])
            ax.imshow(context_recon[t].permute(1, 2, 0).cpu().numpy())
            ax.set_title(f'Context Recon {t}', fontsize=8)
            ax.axis('off')
        for t in range(T_ctx_show, n_cols):
            ax = fig.add_subplot(gs[1, t])
            ax.axis('off')
        fig.axes[n_cols].set_ylabel("Context Recon", fontsize=9)

        # Row 2: future GT (steps 1..T_f)
        for t in range(T_f):
            ax = fig.add_subplot(gs[2, t])
            ax.imshow(gt_future[t].permute(1, 2, 0).cpu().numpy())
            ax.set_title(f'GT +{t+1}', fontsize=8)
            ax.axis('off')
        for t in range(T_f, n_cols):
            ax = fig.add_subplot(gs[2, t])
            ax.axis('off')
        fig.axes[2 * n_cols].set_ylabel("Future GT", fontsize=9)

        # Row 3: 3-step AR predictions (steps 1..T_f)
        for t in range(T_f):
            ax = fig.add_subplot(gs[3, t])
            ax.imshow(pred_future_3step[t].permute(1, 2, 0).cpu().numpy())
            ax.set_title(f'AR Pred +{t+1}', fontsize=8)
            ax.axis('off')
        for t in range(T_f, n_cols):
            ax = fig.add_subplot(gs[3, t])
            ax.axis('off')
        fig.axes[3 * n_cols].set_ylabel("AR (steps)", fontsize=9)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)

        return fig
    
    @staticmethod
    def plot_metrics(metrics_history, save_path=None):
        """Plot training metrics over epochs"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        epochs = range(len(metrics_history['train_loss']))
        
        # Plot training loss
        ax = axes[0, 0]
        ax.plot(epochs, metrics_history['train_loss'], label='Train Loss', marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot validation loss
        ax = axes[0, 1]
        ax.plot(epochs, metrics_history['val_loss'], label='Val Loss', marker='o', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot validation PSNR
        ax = axes[1, 0]
        ax.plot(epochs, metrics_history['val_psnr'], label='Val PSNR', marker='o', color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('Validation PSNR')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot validation FID
        ax = axes[1, 1]
        ax.plot(epochs, metrics_history['val_fid'], label='Val FID', marker='o', color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('FID')
        ax.set_title('Validation FID')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot validation SSIM
        ax = axes[0, 2]
        ax.plot(epochs, metrics_history['val_ssim'], label='Val SSIM', marker='o', color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('SSIM')
        ax.set_title('Validation SSIM')
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.delaxes(axes[1, 2])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        return fig

# Training
class Trainer:
    """Training pipeline for video prediction"""
    
    def __init__(self, model, train_dataset, val_dataset, config, use_wandb=False):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.use_wandb = use_wandb
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config.get("num_workers", 4),
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            pin_memory=True
        )
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.01)
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["num_epochs"],
            eta_min=1e-6
        )
        
        # FID calculator
        self.fid_calculator = FIDCalculator(device=self.device)
        
        # Metrics tracking
        self.metrics_history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_val_psnr = 0.0
        self.best_val_fid = float('inf')
        self.current_epoch = 0
        
        # Output directories
        self.out_dir = Path(config["out_dir"]) / config["run_name"]
        self.checkpoint_dir = self.out_dir / "checkpoints"
        self.plot_dir = self.out_dir / "plots"
        self.video_dir = self.out_dir / "videos"
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project="video-prediction-perceiver",
                name=config["run_name"],
                config=config
            )
            wandb.watch(model, log='all', log_freq=100)

    def _align_time_grids(self, decoded_seq, raw_videos, context_frames, T_future_raw):
        """
        decoded_seq: (B, T_dec, C, H, W)
        raw_videos:  (B, T_raw, C, H, W)
        context_frames: number of raw context frames used
        T_future_raw:   number of raw future frames to evaluate against
        """
        B, T_dec, C, H, W = decoded_seq.shape
        T_raw = raw_videos.shape[1]

        # 1) Map with inclusive endpoints to avoid off-by-one drift
        ratio = (T_raw - 1) / max(T_dec - 1, 1)

        # 2) Target slice in raw time
        start_raw = context_frames
        end_raw   = min(T_raw, start_raw + T_future_raw)
        target    = raw_videos[:, start_raw:end_raw]           # (B, T_tar, C, H, W)
        T_tar     = target.shape[1]

        # 3) Nearest-neighbor index mapping from raw->decoded time
        #    raw index r maps to decoded index d ≈ r/ratio
        start_dec = int(round(start_raw / ratio))
        t_rel     = torch.arange(T_tar, device=decoded_seq.device, dtype=torch.float32)
        dec_idx   = torch.round(start_dec + t_rel / ratio).to(torch.long)
        dec_idx   = dec_idx.clamp(0, T_dec - 1)                 # safety

        # 4) Gather predicted frames without temporal interpolation (better for PSNR/LPIPS)
        pred = decoded_seq.index_select(dim=1, index=dec_idx)   # (B, T_tar, C, H, W)

        return pred, target


    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            videos = batch['video'].to(self.device)  # (B, T, C, H, W)
            
            # Split into context and target
            context_frames = self.config["context_frames"]
            context = videos[:, :context_frames]
            target = videos[:, context_frames:]
            
            # Forward pass
            self.optimizer.zero_grad()
            full_video = torch.cat([context, target], dim=1)
            outputs = self.model(full_video, context_frames)
            
            # Compute loss
            loss_dict = self.model.compute_loss(outputs, 
                                                full_video, 
                                                perceptual_weight=self.config.get("perceptual_weight", 0.5), 
                                                label_smoothing=self.config.get("label_smoothing", 0.1),
                                                ar_cycle_consistency_weight=self.config.get("ar_cycle_weight", 0.2))
            loss = loss_dict['loss']
            # Backward pass
            loss.backward()
            if self.config.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"])
            self.optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(self.train_loader)
        self.metrics_history['train_loss'].append(avg_loss)
        
        return {'train_loss': avg_loss}
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        
        
        all_real_frames = []
        all_pred_frames = []
        first_batch_viz = None
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        
        for batch_idx, batch in enumerate(pbar):
            videos = batch['video'].to(self.device)
            
            # Split into context and target
            context_frames = self.config["context_frames"]
            context = videos[:, :context_frames]        # (B, Tc, C, H, W)
            target  = videos[:, context_frames:]        # (B, Tf, C, H, W)

            # Always compute TF loss (teacher-forced) so val_loss stays comparable
            full_video = torch.cat([context, target], dim=1)
            outputs = self.model(full_video, context_frames)  # dict with 'reconstructed' etc. 
            loss_dict = self.model.compute_loss(
                outputs, 
                full_video,
                perceptual_weight=self.config["perceptual_weight"],
                label_smoothing=self.config["label_smoothing"],
                ar_cycle_consistency_weight=self.config["ar_cycle_weight"]
            )
            val_loss += loss_dict["loss"].item()
            reconstructed_full = outputs["reconstructed"]
            # Choose which *predictions* to score/visualize
            gen_mode = self.config.get("gen_mode", "autoregressive")
            if gen_mode == "autoregressive":
                # how many frames to roll out
                T_future = target.shape[1] if self.config["num_gen_frames"] == 0 else self.config["num_gen_frames"]

                # Call the built-in AR generator on just the context
                ar_video = self.model.generate_autoregressive(
                    context_videos=context,
                    num_frames_to_generate=T_future,
                    temperature=self.config["temperature"],
                    top_k=self.config["top_k"],
                    top_p=self.config["top_p"],
                )
                # ar_video contains [context || generated]; slice out the generated part
                pred_seq, target_aligned = self._align_time_grids(
                    decoded_seq=ar_video,
                    raw_videos=videos,
                    context_frames=context_frames,
                    T_future_raw=T_future,
                )

            else:
                # Fall back to teacher-forced reconstruction
                reconstructed = outputs["reconstructed"]
                
                T_future = target.shape[1]
                pred_seq, target_aligned = self._align_time_grids(
                    decoded_seq=reconstructed,
                    raw_videos=videos,
                    context_frames=context_frames,
                    T_future_raw=T_future,
                )
            
            # scale to [0,1] like you already do
            pred_seq = (pred_seq + 1.0) / 2.0
            target   = (target_aligned + 1.0) / 2.0
            # PSNR (keep your current computation)
            mse  = torch.mean((pred_seq - target) ** 2)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            val_psnr += psnr.item()
            
            # Accumulate for FID 
            B, Tf = pred_seq.shape[:2]
            all_real_frames.append(target.reshape(B * Tf, *target.shape[2:]))
            all_pred_frames.append(pred_seq.reshape(B * Tf, *pred_seq.shape[2:]))

            if batch_idx == 0:
                with torch.no_grad():
                    recon_full = self.model.reconstruct(videos)    # [B,T,C,H,W]

                context_raw = videos[0, :context_frames]                 # (Tc, C, H, W)
                #The model produce reconstructed frames                  # (Tc+Tf, C, H, W)
                recon_context_raw = recon_full[0, :context_frames]      # (Tc, C, H, W)

                # We'll look at first up to 3 future frames
                T_multistep = min(3, target.shape[1])
                future_gt_raw = videos[0, context_frames:context_frames + T_multistep]  # (T_multistep, C, H, W)

                # Run a 3-step autoregressive rollout on the first sample's context (still in [-1,1])
                ar_video_3 = self.model.generate_autoregressive(
                    context_videos=context_raw.unsqueeze(0),  # (1, Tc, C, H, W)
                    num_frames_to_generate=T_multistep,
                    temperature=self.config["temperature"],
                    top_k=self.config["top_k"],
                    top_p=self.config["top_p"],
                )
                # Take only the generated future part
                future_pred3_raw = ar_video_3[0, -T_multistep:]  # (T_multistep, C, H, W)

                # Scale everything to [0,1] for visualization
                context_vis = (context_raw + 1.0) / 2.0
                recon_context_vis = (recon_context_raw + 1.0) / 2.0
                future_gt_vis = (future_gt_raw + 1.0) / 2.0
                future_pred3_vis = (future_pred3_raw + 1.0) / 2.0

                first_batch_viz = {
                    "context": context_vis,
                    "context_recon": recon_context_vis,
                    "gt_future": future_gt_vis,
                    "pred_future_3step": future_pred3_vis,
                }


        all_real_frames = torch.cat(all_real_frames, dim=0)
        all_pred_frames = torch.cat(all_pred_frames, dim=0)
        val_fid = self.fid_calculator.calculate_fid(all_real_frames, all_pred_frames)
        val_ssim = calculate_ssim_batch(all_real_frames, all_pred_frames)
        # ===== Visualization AFTER loop (happens once) =====
        if first_batch_viz is not None and self.current_epoch % self.config.get("vis_every", 5) == 0:
            
            fig = Visualizer.plot_context_and_multistep_predictions(
                context=first_batch_viz["context"],
                context_recon=first_batch_viz["context_recon"],
                gt_future=first_batch_viz["gt_future"],
                pred_future_3step=first_batch_viz["pred_future_3step"],
                save_path=self.video_dir / f"{gen_mode}_context_and_multistep_epoch{self.current_epoch}.png"
            )
            if self.use_wandb:
                wandb.log({f"{gen_mode}_context_and_multistep": wandb.Image(fig)})
            plt.close(fig)
        
        # Average metrics
        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_psnr = val_psnr / len(self.val_loader)
        
        self.metrics_history['val_loss'].append(avg_val_loss)
        self.metrics_history['val_psnr'].append(avg_val_psnr)
        self.metrics_history['val_fid'].append(val_fid)
        self.metrics_history['val_ssim'].append(val_ssim)

        return {
            'val_loss': avg_val_loss,
            'val_psnr': avg_val_psnr,
            'val_fid': val_fid,
            'val_ssim': val_ssim
        }
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_psnr': self.best_val_psnr,
            'best_val_fid': self.best_val_fid,
            'config': self.config,
            'metrics_history': dict(self.metrics_history)
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model: loss={self.best_val_loss:.4f}, psnr={self.best_val_psnr:.2f}dB, fid={self.best_val_fid:.2f}")
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.config['num_epochs']} epochs")
        print(f"Device: {self.device}")
        print(f"Train dataset: {len(self.train_dataset)} sequences")
        print(f"Val dataset: {len(self.val_dataset)} sequences")
        
        for epoch in range(self.config["num_epochs"]):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            all_metrics = {**train_metrics, **val_metrics}
            print(f"\nEpoch {epoch}: " + 
                  f"train_loss={train_metrics['train_loss']:.4f}, " +
                  f"val_loss={val_metrics['val_loss']:.4f}, " +
                  f"val_psnr={val_metrics['val_psnr']:.2f}dB, " +
                  f"val_fid={val_metrics['val_fid']:.2f}")
            
            if self.use_wandb:
                wandb.log({**all_metrics, 'epoch': epoch, 'lr': self.scheduler.get_last_lr()[0]})
            
            # Save checkpoint
            if epoch % self.config.get("save_every", 50) == 0:
                self.save_checkpoint()
            
            # Save best model
            is_best = False
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                is_best = True
            
            if val_metrics['val_psnr'] > self.best_val_psnr:
                self.best_val_psnr = val_metrics['val_psnr']
                is_best = True
            
            if val_metrics['val_fid'] < self.best_val_fid:
                self.best_val_fid = val_metrics['val_fid']
                is_best = True
            
            if is_best:
                self.save_checkpoint(is_best=True)
            
            # Plot metrics
            if epoch % 5 == 0:
                fig = Visualizer.plot_metrics(
                    self.metrics_history,
                    save_path=self.plot_dir / f"metrics_epoch{epoch}.png"
                )
                if self.use_wandb:
                    wandb.log({"metrics_plot": wandb.Image(fig)})
                plt.close(fig)
        
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation PSNR: {self.best_val_psnr:.2f}dB")
        print(f"Best validation FID: {self.best_val_fid:.2f}")
        
        # Save final metrics
        with open(self.out_dir / "metrics_history.json", "w") as f:
            metrics_dict = {k: v for k, v in self.metrics_history.items()}
            json.dump(metrics_dict, f, indent=2)


# Main Function

def main():
    parser = argparse.ArgumentParser(description="Train Video Prediction with Mixed Datasets")
    
    # Model arguments
    parser.add_argument("--num_latents", type=int, default=512)
    parser.add_argument("--num_latent_channels", type=int, default=1024)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--num_decoder_layers", type=int, default=2)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--code_dim", type=int, default=2048)
    parser.add_argument("--num_codes", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Data arguments 
    parser.add_argument("--base_path", type=str, 
                       default="./transition_data/dmc_vb",
                       help="Base path containing dataset structure")
    parser.add_argument("--task", type=str, 
                       default="humanoid_walk",
                       help="Task name (e.g., humanoid_walk)")
    parser.add_argument("--dataset_types", type=str, nargs='+',
                       default=["expert", "medium", "mixed"],
                       help="Dataset types to include (e.g., expert medium mixed)")
    parser.add_argument("--subdirs", type=str, nargs='+',
                       default=["dynamic_medium", "none", "static_medium"],
                       help="Subdirectories to include")
    parser.add_argument("--sequence_length", type=int, default=16)
    parser.add_argument("--context_frames", type=int, default=8)
    parser.add_argument("--img_height", type=int, default=64)
    parser.add_argument("--img_width", type=int, default=64)
    parser.add_argument("--train_split_ratio", type=float, default=0.9,
                       help="Ratio for train/val split")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Other arguments
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--vis_every", type=int, default=5)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perceptual_weight", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--gen_mode", choices=["reconstruct", "autoregressive"], default="autoregressive",
                    help="Which generation path to use during validation/visualization")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50, help="0 disables top-k")
    parser.add_argument("--top_p", type=float, default=0.0, help="0.0 disables nucleus sampling")
    parser.add_argument("--num_gen_frames", type=int, default=0,
                        help="How many future frames to generate for AR / MaskGIT; 0 = use target length")
    parser.add_argument("--kmeans_init", action="store_true",
                        help="Use K-means initialization for codebook")
    parser.add_argument("--ar_cycle_weight", type=float, default=0.5, help="Weight for AR latent cycle-consistency loss (0 disables it)")

    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create configuration
    config = {
        "num_latents": args.num_latents,
        "num_latent_channels": args.num_latent_channels,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "num_attention_heads": args.num_attention_heads,
        "code_dim": args.code_dim,
        "num_codes": args.num_codes,
        "downsample": args.downsample,
        "dropout": args.dropout,
        "context_frames": args.context_frames,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "num_workers": args.num_workers,
        "out_dir": args.out_dir,
        "save_every": args.save_every,
        "vis_every": args.vis_every,
        "use_wandb": args.use_wandb,
        "run_name": args.run_name or f"perceiver_mixed_{args.task}",
        "seed": args.seed,
        "dataset_types": args.dataset_types,
        "subdirs": args.subdirs,
        "perceptual_weight": args.perceptual_weight,
        "label_smoothing": args.label_smoothing,
        "gen_mode": args.gen_mode,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "num_gen_frames": args.num_gen_frames,
        "kmeans_init": args.kmeans_init,
        "ar_cycle_weight": args.ar_cycle_weight,
    }
    
    # Create model
    print("\nCreating model...")
    model = CausalPerceiverIO(
        video_shape=(args.sequence_length, 3, args.img_height, args.img_width),
        num_latents=args.num_latents,
        num_latent_channels=args.num_latent_channels,
        num_attention_heads=args.num_attention_heads,
        num_encoder_layers=args.num_encoder_layers,
        code_dim=args.code_dim,
        num_codes=args.num_codes,
        downsample=args.downsample,
        dropout=args.dropout,
        kmeans_init=args.kmeans_init,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create mixed datasets
    print("\nCreating mixed datasets...")
    train_dataset = create_mixed_dataset(
        base_path=args.base_path,
        task=args.task,
        dataset_types=args.dataset_types,
        subdirs=args.subdirs,
        split='train',
        sequence_length=args.sequence_length,
        img_height=args.img_height,
        img_width=args.img_width,
        train_split_ratio=args.train_split_ratio,
    )
    
    val_dataset = create_mixed_dataset(
        base_path=args.base_path,
        task=args.task,
        dataset_types=args.dataset_types,
        subdirs=args.subdirs,
        split='val',
        sequence_length=args.sequence_length,
        img_height=args.img_height,
        img_width=args.img_width,
        train_split_ratio=args.train_split_ratio,
    )
    
    # Create trainer and start training
    trainer = Trainer(model, train_dataset, val_dataset, config, use_wandb=args.use_wandb)
    trainer.train()
    
    print("\nTraining complete! Check the following directories:")
    print(f"  - Checkpoints: {trainer.checkpoint_dir}")
    print(f"  - Plots: {trainer.plot_dir}")
    print(f"  - Videos: {trainer.video_dir}")


if __name__ == "__main__":
    main()
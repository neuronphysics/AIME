import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import torch
import numpy as np
import argparse
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
from collections import defaultdict
import itertools
from VRNN.main import ModelState
from VRNN.vrnn_utilities import *
from VRNN.training_vrnn import run_train, run_test
import bisect
# Define constants from the MoCapAct dataset
CMU_HUMANOID_OBSERVABLES = (
    'walker/sensors_accelerometer',
    'walker/body_height',
    'walker/actuator_activation',
    'walker/appendages_pos',
    'walker/end_effectors_pos',
    'walker/joints_pos',
    'walker/joints_vel',
    'walker/sensors_gyro',
    'walker/sensors_torque',
    'walker/sensors_touch',
    'walker/sensors_velocimeter',
    'walker/world_zaxis'
)
def create_synthetic_metrics_file(output_path, obs_dim=236, act_dim=56):
    """
    Create a synthetic dataset_metrics.npz file with the correct structure
    but filled with reasonable default values.
    
    Args:
        output_path: Path to save the metrics file
        obs_dim: Dimension of observation space
        act_dim: Dimension of action space
    """
    # Create directional folders if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create empty/default metrics with correct shapes
    metrics = {
        'proprio_mean': np.zeros(obs_dim, dtype=np.float32),
        'proprio_var': np.ones(obs_dim, dtype=np.float32),
        'act_mean': np.zeros(act_dim, dtype=np.float32),
        'act_var': np.ones(act_dim, dtype=np.float32),
        'mean_act_mean': np.zeros(act_dim, dtype=np.float32),
        'mean_act_var': np.ones(act_dim, dtype=np.float32),
        'count': np.array([10000], dtype=np.int64),
        'snippet_returns': np.array({}),  # Empty dict
        'values': np.array({}),           # Empty dict
        'advantages': np.array({})        # Empty dict
    }
    
    # Save the metrics file
    np.savez(output_path, **metrics)
    print(f"Created synthetic metrics file at {output_path}")

##########################
def get_max_sequence_length(data_path, subset='small'):
    hdf5_files, _ = extract_hdf5_files(data_path, subset)
    max_length = 0

    for fname in hdf5_files:
        with h5py.File(fname, 'r') as dset:
            # Get all motion capture snippets
            snippets = [k for k in dset.keys() if k.startswith('CMU')]
            
            for snippet in snippets:
                # Get rollout counts
                n_start = dset['n_start_rollouts'][...].item()
                n_rsi = dset['n_rsi_rollouts'][...].item()
                
                # Get episode lengths from both start and RSI rollouts
                start_lengths = dset[f"{snippet}/start_metrics/episode_lengths"][:n_start]
                rsi_lengths = dset[f"{snippet}/rsi_metrics/episode_lengths"][:n_rsi]
                
                # Update maximum length
                current_max = max(start_lengths.max(), rsi_lengths.max())
                max_length = max(max_length, current_max)

    return max_length

def filter_zero_data_points(hdf5_files, metrics_path, observables=CMU_HUMANOID_OBSERVABLES):
    """Filter out data points where all values are zero."""
    
    # Store valid indices to keep
    valid_indices = []
    total_indices = 0
    removed_indices = 0
    
    # Create temporary dataset to analyze data
    temp_dataset = MoCapActDataset(
        hdf5_files=hdf5_files,
        metrics_path=metrics_path,
        observables=observables
    )
    
    print(f"Scanning dataset with {len(temp_dataset)} total samples")
    
    # Process each data point
    for idx in tqdm(range(len(temp_dataset))):
        inputs, targets = temp_dataset[idx]
        
        # Check if input and target contain non-zero values
        has_non_zero_input = (inputs != 0).any().item()
        has_non_zero_target = (targets != 0).any().item()
        
        # Keep only samples with non-zero values in both input and output
        if has_non_zero_input and has_non_zero_target:
            valid_indices.append(idx)
        else:
            removed_indices += 1
        
        total_indices += 1
        
        # Print progress periodically
        if idx % 1000 == 0:
            print(f"Processed {idx} samples, found {len(valid_indices)} valid, removed {removed_indices} zero samples")
    
    print(f"Filtering complete: {len(valid_indices)}/{total_indices} samples kept, {removed_indices} samples removed")
    
    return valid_indices

class FilteredMoCapActDataset(Dataset):
    """MoCapAct dataset filtered to remove zero-valued data points"""
    
    def __init__(self, original_dataset, valid_indices):
        self.original_dataset = original_dataset
        self.valid_indices = valid_indices
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        return self.original_dataset[original_idx]

class MoCapActDataset(Dataset):
    """
    Dataset class for loading and preprocessing MoCapAct data
    https://github.com/microsoft/MoCapAct
    """
    
    def __init__(self, 
                 hdf5_files, 
                 metrics_path, 
                 sequence_length=100,
                 normalize_obs=True,
                 normalize_act=True,
                 normalize_reward=True,
                 return_mean_act=False,
                 observables=CMU_HUMANOID_OBSERVABLES,
                 concat_observables=True,
                 keep_hdf5s_open=False,
                 relabel_type="none", 
                 subset=None,
                 **kwargs):
        """
        Args:
            hdf5_files: List of paths to HDF5 files to load
            metrics_path: Path to the dataset metrics file
            sequence_length: Maximum sequence length
            normalize_obs: Whether to normalize observations
            normalize_act: Whether to normalize actions
            normalize_reward: Whether to normalize rewards
            return_mean_act: Whether to use mean actions
            observables: What observables to use from the dataset
            concat_observables: Whether to concatenate observables
            keep_hdf5s_open: Whether to keep HDF5 files open
            relabel_type: Type of reward relabeling to apply
        """
        self._hdf5_fnames = hdf5_files
        self._observables = observables
        self._concat_observables = concat_observables
        self._max_seq_steps = sequence_length
        self.normalize_obs = normalize_obs
        self.normalize_act = normalize_act
        self.normalize_reward = normalize_reward
        self._return_mean_act = return_mean_act
        self._relabel_type = relabel_type
        self._metrics_path = metrics_path
        if subset:
           self._subset_hdf5_names =self._get_snset(subset)
        
        self._keep_hdf5s_open = keep_hdf5s_open
        if self._keep_hdf5s_open:
            self._dsets = [h5py.File(fname, 'r') for fname in self._hdf5_fnames]
        
        # Load clip snippets from HDF5 files
        self._clip_snippets = []
        for fname in self._hdf5_fnames:
            try:
                with h5py.File(fname, 'r') as dset:
                    self._clip_snippets.append(tuple([k for k in dset.keys() if k.startswith('CMU')]))
            except:
                print('Error in file: ', fname)
                
        # Flatten clip snippets and get unique clip IDs
        self._clip_snippets_flat = tuple(itertools.chain.from_iterable(self._clip_snippets))
        self._clip_ids = tuple({k.split('-')[0] for k in self._clip_snippets_flat})
        
        # Load reference steps and observable indices
        with h5py.File(self._hdf5_fnames[0], 'r') as dset:
            self._ref_steps = dset['ref_steps'][...]
            obs_ind_dset = dset['observable_indices/walker']
            self.observable_indices = {
                f"walker/{k}": obs_ind_dset[k][...] for k in obs_ind_dset
            }
            
        # Set up spaces and statistics
        self._set_spaces()
        self._set_stats()
        
        # Create offset indices for efficient data loading
        self._create_offsets()

    def _set_spaces(self):

        if self._concat_observables:
            dim = 0
            for key in self._observables:
                dim += len(self.observable_indices[key])
            obs_dim = dim
        else:
            obs_dim = sum(len(self.observable_indices[key]) for key in self._observables)

        with h5py.File(self._hdf5_fnames[0], 'r') as dset:
            act_dim = dset[f"{self._clip_snippets[0][0]}/0/actions"].shape[1]
        
        self._observation_dim = obs_dim
        self._action_dim = act_dim
        print(f"Observation dimension: {obs_dim}, Action dimension: {act_dim}")
    
    def _extract_observations(self, all_obs, observable_keys):
        return {k: all_obs[..., self.observable_indices[k]] for k in observable_keys}
    
    def _set_stats(self):
        """Load statistics from metrics file"""
        metrics = np.load(self._metrics_path, allow_pickle=True)
        
        self._count = metrics['count']
        self.proprio_mean = metrics['proprio_mean']
        self.proprio_var = metrics['proprio_var']
        self.act_mean = metrics['mean_act_mean'] if self._return_mean_act else metrics['act_mean']
        self.act_var = metrics['mean_act_var'] if self._return_mean_act else metrics['act_var']       

            
        self.snippet_returns = metrics['snippet_returns'].item()
        self.advantages = {k: v for k, v in metrics['advantages'].item().items() if k in self._clip_snippets_flat}
        self.values = {k: v for k, v in metrics['values'].item().items() if k in self._clip_snippets_flat}
        
        self.proprio_std = (np.sqrt(self.proprio_var) + 1e-4).astype(np.float32)
        self.act_std = (np.sqrt(self.act_var) + 1e-4).astype(np.float32)
        
        # Define reward and return statistics
        reward_mean = {
            "none": 0.6940667629241943,
            "speed": 0.5072286128997803,
            "rotate_y": 0.046567633748054504,
            "forward": 0.251115620136261,
            "backward": -0.25101155042648315,
            "shift_left": 0.002932528965175152,
            "jump": 0.058018457144498825,
            "rotate_x": -0.004368575755506754,
            "rotate_z": -0.003207581816241145
        }
        
        reward_std = {
            "none": 0.16416609287261963,
            "speed": 0.4620329737663269,
            "rotate_y": 0.8839800953865051,
            "forward": 0.5141152143478394,
            "backward": 0.5142410397529602,
            "shift_left": 0.22409744560718536,
            "jump": 0.08115938305854797,
            "rotate_x": 0.2102961540222168,
            "rotate_z": 0.23211251199245453
        }
        
        return_mean = {
            "none": 11.799545288085938,
            "speed": 8.622393608093262,
            "rotate_y": 0.791003942489624,
            "forward": 4.268054485321045,
            "backward": -4.270056247711182,
            "shift_left": 0.04919257014989853,
            "jump": 0.9864307641983032,
            "rotate_x": -0.0740567222237587,
            "rotate_z": -0.054631639271974564
        }
        
        return_std = {
            "none": 2.7906644344329834,
            "speed": 7.853872299194336,
            "rotate_y": 15.0303316116333,
            "forward": 8.739494323730469,
            "backward": 8.741517066955566,
            "shift_left": 3.811504364013672,
            "jump": 1.3811616897583008,
            "rotate_x": 3.5730364322662354,
            "rotate_z": 3.9460842609405518
        }
        
        self.reward_mean = reward_mean[self._relabel_type]
        self.reward_std = reward_std[self._relabel_type]
        self.return_mean = return_mean[self._relabel_type]
        self.return_std = return_std[self._relabel_type]


    def _create_offsets(self):
        """Create offset indices for efficient data access"""
        self._total_len = 0
        self._dset_indices = []
        self._logical_indices, self._dset_groups = [[] for _ in self._hdf5_fnames], [[] for _ in self._hdf5_fnames]
        self._snippet_len_weights = [[] for _ in self._hdf5_fnames]
        
        iterator = zip(
            self._hdf5_fnames,
            self._clip_snippets,
            self._logical_indices,
            self._dset_groups,
            self._snippet_len_weights
        )
        
        for fname, clip_snippets, logical_indices, dset_groups, snippet_len_weights in iterator:
            with h5py.File(fname, 'r') as dset:
                self._dset_indices.append(self._total_len)
                dset_start_rollouts = dset['n_start_rollouts'][...]
                dset_rsi_rollouts = dset['n_rsi_rollouts'][...]
                n_start_rollouts = dset_start_rollouts
                n_rsi_rollouts = dset_rsi_rollouts
                
                for snippet in clip_snippets:
                    _, start, end = snippet.split('-')
                    clip_len = int(end) - int(start)
                    snippet_weight = 1
                    
                    len_iterator = itertools.chain(
                        dset[f"{snippet}/start_metrics/episode_lengths"][:n_start_rollouts],
                        dset[f"{snippet}/rsi_metrics/episode_lengths"][:n_rsi_rollouts]
                    )
                    
                    for i, ep_len in enumerate(len_iterator):
                        logical_indices.append(self._total_len)
                        dset_groups.append(f"{snippet}/{i if i < n_start_rollouts else i-n_start_rollouts+dset_start_rollouts}")
                        snippet_len_weights.append(snippet_weight)
                        
                        # Skip sequences shorter than min_seq_steps (if we had one)
                        # if ep_len < self._min_seq_steps:
                        #     continue
                            
                        self._total_len += snippet_weight * ep_len
    
    def __len__(self):
        """Return total length of dataset"""
        return self._total_len
    
    def __getitem__(self, idx):
        """Get an item by index"""
        dset_idx = bisect.bisect_right(self._dset_indices, idx) - 1
        
        if self._keep_hdf5s_open:
            item = self._getitem(self._dsets[dset_idx], idx)
        else:
            with h5py.File(self._hdf5_fnames[dset_idx], 'r') as dset:
                item = self._getitem(dset, idx)
                
        return item
    
    def normalize_observations(self, states):
        """Normalize observations using dataset statistics"""
        states_std = np.squeeze(np.array(self.proprio_std))
        states_mean = np.squeeze(np.array(self.proprio_mean))
        
        obs_std = self._extract_observations(states_std, self._observables)
        obs_mean = self._extract_observations(states_mean, self._observables)
        
        if self._concat_observables:
            obs_std = np.concatenate(list(obs_std.values()), axis=-1)
            obs_mean = np.concatenate(list(obs_mean.values()), axis=-1)
        
        if torch.is_tensor(states):
            obs_std = torch.Tensor(obs_std).to(states.device)
            obs_mean = torch.Tensor(obs_mean).to(states.device)
        
        return (states - obs_mean) / obs_std
    
    def denormalize_observations(self, observations):
        """Denormalize observations using dataset statistics"""
        states_std = np.squeeze(np.array(self.proprio_std))
        states_mean = np.squeeze(np.array(self.proprio_mean))
        
        obs_std = self._extract_observations(states_std, self._observables)
        obs_mean = self._extract_observations(states_mean, self._observables)
        
        if self._concat_observables:
            obs_std = np.concatenate(list(obs_std.values()), axis=-1)
            obs_mean = np.concatenate(list(obs_mean.values()), axis=-1)
        
        if torch.is_tensor(observations):
            obs_std = torch.Tensor(obs_std).to(observations.device)
            obs_mean = torch.Tensor(obs_mean).to(observations.device)
            
        return observations * obs_std + obs_mean
    
    def normalize_actions(self, actions):
        """Normalize actions using dataset statistics"""
        if torch.is_tensor(actions):
            act_std = torch.Tensor(self.act_std).to(actions.device)
            act_mean = torch.Tensor(self.act_mean).to(actions.device)
        else:
            act_std = np.squeeze(np.array(self.act_std))
            act_mean = np.squeeze(np.array(self.act_mean))
            
        return (actions - act_mean) / act_std
    
    def denormalize_actions(self, actions):
        """Denormalize actions using dataset statistics"""
        if torch.is_tensor(actions):
            act_std = torch.Tensor(self.act_std).to(actions.device)
            act_mean = torch.Tensor(self.act_mean).to(actions.device)
        else:
            act_std = np.squeeze(np.array(self.act_std))
            act_mean = np.squeeze(np.array(self.act_mean))
            
        return actions * act_std + act_mean
    
    def normalize_rewards(self, rewards):
        """Normalize rewards using dataset statistics"""
        if not self.normalize_reward:
            return rewards
            
        if torch.is_tensor(rewards):
            return (rewards - self.reward_mean) / self.reward_std
        else:
            return (rewards - self.reward_mean) / self.reward_std
    
    def denormalize_rewards(self, rewards):
        """Denormalize rewards using dataset statistics"""
        if not self.normalize_reward:
            return rewards
            
        if torch.is_tensor(rewards):
            return rewards * self.reward_std + self.reward_mean
        else:
            return rewards * self.reward_std + self.reward_mean
    
    def _getitem(self, dset, idx):
        """Get an item from a dataset by index"""
        dset_idx = bisect.bisect_right(self._dset_indices, idx) - 1
        clip_idx = bisect.bisect_right(self._logical_indices[dset_idx], idx) - 1
        
        # Determine which action dataset to use
        act_name = "mean_actions" if self._return_mean_act else "actions"
        
        # Get dataset references
        proprio_dset = dset[f"{self._dset_groups[dset_idx][clip_idx]}/observations/proprioceptive"]
        act_dset = dset[f"{self._dset_groups[dset_idx][clip_idx]}/{act_name}"]
        reward_dset = dset[f"{self._dset_groups[dset_idx][clip_idx]}/rewards"]
        
        # Calculate start and end indices
        snippet_len_weight = self._snippet_len_weights[dset_idx][clip_idx]
        start_idx = int((idx - self._logical_indices[dset_idx][clip_idx]) / snippet_len_weight)
        end_idx = min(start_idx + self._max_seq_steps, act_dset.shape[0])
        
        # Load data
        all_obs = proprio_dset[start_idx:end_idx]
        act = act_dset[start_idx:end_idx]
        reward = reward_dset[start_idx:end_idx]
    

        # Extract observables
        obs = self._extract_observations(all_obs, self._observables)
        if self._concat_observables:
            obs = np.concatenate(list(obs.values()), axis=-1)


        # Create inputs (current obs + action) and targets (next obs + reward)

        # Create input: current observation and action
        current_obs = obs[:-1]  # All but last
        current_actions = act[:-1]  # All but last
        
        # Create target: next observation and reward
        next_obs = obs[1:]  # All but first
        next_rewards = reward[1:]  # All but first
        
        # Combine into input and target tensors
        if self.normalize_obs:
            current_obs = self.normalize_observations(current_obs)
            next_obs = self.normalize_observations(next_obs)
            
        if self.normalize_act:
            current_actions = self.normalize_actions(current_actions)
            
        if self.normalize_reward:
            next_rewards = self.normalize_rewards(next_rewards)
        
        # Convert to tensors
        input_tensor = np.concatenate([current_obs, current_actions], axis=-1)
        target_tensor = np.concatenate([next_obs, next_rewards.reshape(-1, 1)], axis=-1)
        
        # Convert to PyTorch tensors
        input_tensor = torch.FloatTensor(input_tensor)
        target_tensor = torch.FloatTensor(target_tensor)
        
        # Reshape to match VRNN expected format: [batch, features, seq_len]
        input_tensor = input_tensor.unsqueeze(0).permute(0, 2, 1)  # Add batch dim and permute
        target_tensor = target_tensor.unsqueeze(0).permute(0, 2, 1)

        
        return input_tensor, target_tensor

def collate_fixed_length_sequences(batch, max_length):
    """
    Custom collate function for batching variable length sequences
    
    Args:
        batch: List of (input, target) tuples
        
    Returns:
        Batched inputs and targets with padding
    """
    # Sort batch by sequence length (descending)
    batch.sort(key=lambda x: x[0].shape[-1], reverse=True)
    
    # Get feature dimensions
    batch_size = len(batch)
    input_features = batch[0][0].shape[1]
    output_features = batch[0][1].shape[1]
    
    # Create padded tensors
    padded_inputs = torch.zeros(batch_size, input_features, max_length)
    padded_targets = torch.zeros(batch_size, output_features, max_length)
    
    # Fill padded tensors
    for i, (input_seq, target_seq) in enumerate(batch):
        seq_len = input_seq.shape[-1]
        padded_inputs[i, :, :seq_len] = input_seq[0]
        padded_targets[i, :, :seq_len] = target_seq[0]
    seq_lens = [x[0].shape[-1] for x in batch]
    
    return padded_inputs, padded_targets  

def extract_hdf5_files(data_path, subset='small'):
    """
    Extract HDF5 file paths from the MoCapAct dataset
    
    Args:
        data_path: Path to the MoCapAct dataset
        subset: 'small' or 'large'
        
    Returns:
        List of HDF5 file paths, path to metrics file
    """
    # Determine the folder based on subset
    if subset == 'small':
        data_folder = os.path.join(data_path, 'all/small')
        # For testing, you can use sample: os.path.join(data_path, 'sample/small')
    elif subset == 'large':
        data_folder = os.path.join(data_path, 'all/large')
        # For testing, you can use sample: os.path.join(data_path, 'sample/large')
    else:
        raise ValueError(f"Unknown subset: {subset}")
    
    # Check if folder exists
    if not os.path.exists(data_folder):
        # Try using tarball path if direct folder doesn't exist
        if subset == 'small':
            tarball_files = [os.path.join(data_path, f'all/small/small_{i}.tar.gz') for i in range(1, 4)]
        else:
            tarball_files = [os.path.join(data_path, f'all/large/large_{i}.tar.gz') for i in range(1, 44)]
        
        # Check if tarballs exist
        if not all(os.path.exists(tf) for tf in tarball_files):
            raise FileNotFoundError(f"Could not find MoCapAct dataset at {data_path}")
        
        # Extract tarball files
        raise NotImplementedError("Tarball extraction not implemented yet - please extract the files manually")
    
    # Get list of HDF5 files
    hdf5_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.hdf5')]
    
    # Get metrics file path
    metrics_path = os.path.join(data_folder, 'dataset_metrics.npz')
    # If metrics file doesn't exist, create a synthetic one
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found at {metrics_path}")
        print(f"Creating synthetic metrics file...")
        
        # Try to determine observation/action dimensions from HDF5 files if available
        obs_dim = 236  # Default observation dimension
        act_dim = 56   # Default action dimension
        
        if hdf5_files:
            try:
                with h5py.File(hdf5_files[0], 'r') as f:
                    # Get a snippet name
                    snippet_keys = [k for k in f.keys() if k.startswith('CMU')]
                    if snippet_keys:
                        snippet_name = snippet_keys[0]
                        # Get action shape
                        if f"{snippet_name}/0/actions" in f:
                            act_dim = f[f"{snippet_name}/0/actions"].shape[1]
                        # Get observation shape
                        if f"{snippet_name}/0/observations/proprioceptive" in f:
                            obs_dim = f[f"{snippet_name}/0/observations/proprioceptive"].shape[1]
            except Exception as e:
                print(f"Error determining dimensions from HDF5: {e}")
                print(f"Using default dimensions: obs_dim={obs_dim}, act_dim={act_dim}")
        
        create_synthetic_metrics_file(metrics_path, obs_dim=obs_dim, act_dim=act_dim)

    
    return hdf5_files, metrics_path

def prepare_mocapact_dataloaders(data_path, 
                                batch_size=24, 
                                sequence_length=100, 
                                subset='small',
                                normalize_obs=True,
                                normalize_act=True,
                                normalize_reward=True,
                                return_mean_act=False,
                                relabel_type="none",
                                num_workers=4, 
                                validation_split=0.1,
                                test_split=0.1,
                                seed=1234):
    """
    Prepare DataLoaders for MoCapAct dataset
    
    Args:
        data_path: Path to the MoCapAct dataset
        batch_size: Batch size for training
        sequence_length: Maximum sequence length
        subset: 'small' or 'large'
        normalize_obs: Whether to normalize observations
        normalize_act: Whether to normalize actions
        normalize_reward: Whether to normalize rewards
        return_mean_act: Whether to use mean actions
        relabel_type: Type of reward relabeling
        num_workers: Number of workers for data loading
        validation_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader, test_loader, train_sampler
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Extract HDF5 files and metrics path
    hdf5_files, metrics_path = extract_hdf5_files(data_path, subset)
    
    # Create full dataset
    full_dataset = MoCapActDataset(
        hdf5_files=hdf5_files,
        metrics_path=metrics_path,
        sequence_length=sequence_length,
        normalize_obs=normalize_obs,
        normalize_act=normalize_act,
        normalize_reward=normalize_reward,
        return_mean_act=return_mean_act,
        relabel_type=relabel_type
    )
       # Get valid indices (non-zero data points)
    print("Filtering dataset to remove all-zero data points...")
    valid_indices = filter_zero_data_points(hdf5_files, metrics_path)
    
    # Create filtered dataset
    filtered_dataset = FilteredMoCapActDataset(full_dataset, valid_indices)
    print(f"Original dataset size: {len(full_dataset)}, Filtered dataset size: {len(filtered_dataset)}")
    
    # Now split the filtered dataset
    dataset_size = len(filtered_dataset)
    
    val_size = int(validation_split * dataset_size)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        filtered_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create samplers for distributed training if enabled
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_fixed_length_sequences(batch, sequence_length),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_fixed_length_sequences(batch, sequence_length),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_fixed_length_sequences(batch, sequence_length),
    )
    
    return train_loader, val_loader, test_loader, train_sampler, full_dataset
def compute_prediction_accuracy(model, data_loader, device, dataset, num_batches=None):
    """
    Compute prediction accuracy metrics for the VRNN model
    
    Args:
        model: The trained VRNN model
        data_loader: DataLoader with test data
        device: Device to run inference on
        dataset: The dataset object (for normalization)
        num_batches: Number of batches to evaluate (None for all)
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    total_obs_mse = 0.0
    total_obs_mae = 0.0
    total_reward_mse = 0.0
    total_reward_mae = 0.0
    total_samples = 0
    total_rewards = 0
    
    # Track progress
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    pbar.set_description("Computing prediction accuracy")
    
    with torch.no_grad():
        for i, (u, y) in pbar:
            if num_batches is not None and i >= num_batches:
                break
                
            # Move data to device
            u = u.to(device)
            y = y.to(device)
            
            # Generate predictions
            batch_size = u.shape[0]
            y_sample, y_sample_mu, y_sample_sigma, hidden = model.generate(u)
            
            # Split predictions and targets into observations and rewards
            # Last dimension of output is the reward
            pred_obs = y_sample_mu[:, :-1, :]
            pred_reward = y_sample_mu[:, -1:, :]
            true_obs = y[:, :-1, :]
            true_reward = y[:, -1:, :]
            
            # Compute observation accuracy
            obs_mse = torch.mean((pred_obs - true_obs) ** 2).item()
            obs_mae = torch.mean(torch.abs(pred_obs - true_obs)).item()
            
            # Compute reward accuracy
            reward_mse = torch.mean((pred_reward - true_reward) ** 2).item()
            reward_mae = torch.mean(torch.abs(pred_reward - true_reward)).item()
            
            # Accumulate statistics
            n_samples = batch_size * y.shape[2]  # batch_size * seq_len
            total_obs_mse += obs_mse * n_samples
            total_obs_mae += obs_mae * n_samples
            total_reward_mse += reward_mse * n_samples
            total_reward_mae += reward_mae * n_samples
            total_samples += n_samples
            total_rewards += batch_size * y.shape[2]  # batch_size * seq_len
            
            # Update progress bar
            pbar.set_postfix({
                'obs_mse': obs_mse,
                'reward_mse': reward_mse
            })
    
    # Compute final metrics
    metrics = {
        'observation_mse': total_obs_mse / total_samples,
        'observation_mae': total_obs_mae / total_samples,
        'reward_mse': total_reward_mse / total_rewards,
        'reward_mae': total_reward_mae / total_rewards,
    }
    
    # Add RMSE metrics
    metrics['observation_rmse'] = np.sqrt(metrics['observation_mse'])
    metrics['reward_rmse'] = np.sqrt(metrics['reward_mse'])
    
    return metrics

def generate_perceiver_input(observations, actions, sequence_length):
    """
    Generate correctly formatted input for the Perceiver model
    
    Args:
        observations: Tensor of observations [batch, obs_dim, seq_len]
        actions: Tensor of actions [batch, act_dim, seq_len]
        sequence_length: Maximum sequence length
        
    Returns:
        Properly formatted tensor for Perceiver input
    """
    batch_size = observations.shape[0]
    obs_dim = observations.shape[1]
    act_dim = actions.shape[1]
    
    # Stack observations and actions along the feature dimension
    combined = torch.cat([observations, actions], dim=1)  # [batch, obs_dim+act_dim, seq_len]
    
    # Reshape to match Perceiver's expected input format:
    # The Perceiver expects [batch*seq_len, feature_dim, 1]
    combined = combined.permute(0, 2, 1)  # [batch, seq_len, obs_dim+act_dim]
    combined = combined.reshape(batch_size * sequence_length, obs_dim + act_dim, 1)
    
    return combined

def train_vrnn_on_mocapact(args):
    """
    Train VRNN model on MoCapAct dataset to predict next observation and reward
    based on current observation and action
    
    Args:
        args: Command line arguments
    
    Returns:
        Trained model and evaluation metrics
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    max_seq_len = get_max_sequence_length(args.data_path, args.subset)
    # Prepare dataloaders
    train_loader, val_loader, test_loader, train_sampler, dataset = prepare_mocapact_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        sequence_length=max_seq_len,
        subset=args.subset,
        normalize_obs=args.normalize_obs,
        normalize_act=args.normalize_act,
        normalize_reward=args.normalize_reward,
        return_mean_act=args.return_mean_act,
        relabel_type=args.relabel_type,
        num_workers=args.num_workers,
        validation_split=args.validation_split,
        test_split=args.test_split,
        seed=args.seed
    )
    
    print(f"Dataset prepared. Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Get a sample batch to determine input/output dimensions
    sample_batch = next(iter(train_loader))
    
    # Input is current observation + action
    # Output is next observation + reward
    input_dim = sample_batch[0].shape[1]  # [batch, feature, seq_len]
    output_dim = sample_batch[1].shape[1]
    seq_len = sample_batch[0].shape[2]
    
    # Verify the input/output structure
    obs_dim = dataset._observation_dim
    act_dim = dataset._action_dim
    
    print(f"Input dim: {input_dim} (Obs+Act: {obs_dim}+{act_dim})")
    print(f"Output dim: {output_dim} (Next Obs+Reward: {obs_dim}+1)")
    print(f"Sequence length: {seq_len}")
    
    
    
    # Create model state
    modelstate = ModelState(
        seed=args.seed,
        nu=input_dim,
        ny=output_dim,
        sequence_length=max_seq_len,
        normalizer_input=None,
        normalizer_output=None,
        h_dim=args.h_dim,
        z_dim=args.z_dim,
        n_mixtures=args.n_mixtures
    )
    
    # Move model to device
    modelstate.model.to(device)
    
    # Test Perceiver input generation with sample batch
    sample_obs = sample_batch[0][:, :obs_dim, :]
    sample_act = sample_batch[0][:, obs_dim:, :]
    perceiver_input = generate_perceiver_input(sample_obs, sample_act, seq_len)
    print(f"Generated Perceiver input shape: {perceiver_input.shape}")
    
    # Set the correctly formatted input for the Perceiver model
    if hasattr(modelstate.model, 'generate_mock_input'):
        print("Updating Perceiver mock input format...")
        modelstate.model.generate_mock_input = lambda: {'latent_fea': perceiver_input[:1].to(device)}
    
    # Wrap model with DDP if using distributed training
    if dist.is_initialized():
        modelstate.model = torch.nn.parallel.DistributedDataParallel(
            modelstate.model,
            device_ids=[args.local_rank] if args.local_rank is not None else None,
            find_unused_parameters=True
        )
    
    # Set up paths for saving
    path_general = os.path.join(args.output_dir, 'log/')
    os.makedirs(path_general, exist_ok=True)
    file_name_general = f'MoCapAct_{args.subset}_{args.relabel_type}_VRNN_h{modelstate.h_dim}_z{modelstate.z_dim}_n{modelstate.n_layers}'
    
    # Run training
    df = {}
    df = run_train(
        modelstate=modelstate,
        loader_train=train_loader,
        loader_valid=val_loader,
        device=device,
        dataframe=df,
        path_general=path_general,
        file_name_general=file_name_general,
        lr=args.lr,
        max_epochs=args.max_epochs,
        train_rank=args.local_rank if args.local_rank is not None else 0,
        train_sampler=train_sampler,
        batch_size=args.batch_size
    )
    
    # Run testing
    df = run_test(
        seed=args.seed,
        nu=input_dim,
        ny=output_dim,
        seq_len=seq_len,
        loaders=test_loader,
        df=df,
        device=device,
        path_general=path_general,
        file_name_general=file_name_general,
        batch_size=args.batch_size,
        test_rank=args.local_rank if args.local_rank is not None else 0
    )
    
    # Compute accuracy metrics
    if args.local_rank is None or args.local_rank == 0:
        print("Computing detailed prediction accuracy metrics...")
        accuracy_metrics = compute_prediction_accuracy(
            model=modelstate.model,
            data_loader=test_loader,
            device=device,
            dataset=dataset,
            num_batches=min(20, len(test_loader))  # Limit evaluation to 20 batches for efficiency
        )
        
        # Add accuracy metrics to dataframe
        df.update(accuracy_metrics)
        print("\nPrediction Accuracy Metrics:")
        for metric, value in accuracy_metrics.items():
            print(f"  {metric}: {value:.5f}")

        # Collect predictions from entire test dataset
        print("Generating predictions for entire test dataset...")
        all_predictions = []
        all_ground_truth = []
        all_uncertainties = []
        
        with torch.no_grad():
            for u_test, y_test in tqdm(test_loader, desc="Processing test batches"):
                u_test = u_test.to(device)
                y_test = y_test.to(device)
                
                # Generate model predictions
                y_sample, y_sample_mu, y_sample_sigma, hidden = modelstate.model.generate(u_test)
                
                # Store batch results
                all_predictions.append(y_sample_mu.cpu().numpy())
                all_ground_truth.append(y_test.cpu().numpy())
                all_uncertainties.append(y_sample_sigma.cpu().numpy())
        
        # Concatenate all batches along the batch dimension
        predictions = np.concatenate(all_predictions, axis=0)
        ground_truth = np.concatenate(all_ground_truth, axis=0)
        uncertainties = np.concatenate(all_uncertainties, axis=0)
        
        print(f"Collected data from entire test set: {predictions.shape[0]} samples")
        
        # Compute dataset-wide statistics (optional)
        mean_prediction = np.mean(predictions, axis=0)  # Average across all samples
        std_prediction = np.std(predictions, axis=0)    # Standard deviation across samples
        mean_ground_truth = np.mean(ground_truth, axis=0)
        
        # Plot results for selected samples (can't plot all at once)
        num_samples_to_plot = min(5, predictions.shape[0])  # Plot up to 5 samples
        indices = np.linspace(0, predictions.shape[0]-1, num_samples_to_plot, dtype=int)
        
        for idx, sample_idx in enumerate(indices):
            # Create options dict for plotting
            options = {
                'h_dim': modelstate.h_dim,
                'z_dim': modelstate.z_dim,
                'n_layers': modelstate.n_layers,
                'wm_image_replay_buffer': 'MoCapAct',
                'showfig': True,
                'savefig': True
            }
            
            # Plot individual sample from the test set
            data_y_true = [ground_truth, np.ones_like(ground_truth) * 0.1]  # Assuming 0.1 as uncertainty
            data_y_sample = [predictions, uncertainties]
            label_y = ['ground truth', 'predicted']
            
            print(f"Generating plot for test sample {sample_idx}...")
            plot_time_sequence_uncertainty(
                data_y_true,
                data_y_sample,
                label_y,
                options,
                path_general=os.path.join(args.output_dir, 'plots/'),
                file_name_general=f'MoCapAct_{args.subset}_{args.relabel_type}_VRNN_sample{sample_idx}',
                batch_show=sample_idx,  # Show the chosen sample
                x_limit_show=[0, min(200, ground_truth.shape[2])]  # Show up to 200 time steps
            )
            
        # Also create an aggregate plot showing average performance (optional)


    # Save results
    os.makedirs(os.path.join(args.output_dir, 'data/'), exist_ok=True)
    file_name = os.path.join(args.output_dir, 'data/', f'{file_name_general}_results.csv')
    pd.DataFrame(df).to_csv(file_name)
    torch.manual_seed(42)
    
    # Get a random sample
    idx = np.random.randint(0, len(dataset))
    inputs, targets = dataset[idx]
    
    # Get current observation and action
    current_obs = inputs[0, :dataset._observation_dim, :1]
    current_action = inputs[0, dataset._observation_dim:, :1]
    
    # Predict next observation and reward
    next_obs, reward = predict_next_state_and_reward(
        modelstate.model, current_obs, current_action, normalize=True, dataset=dataset
    )
    
    # Print results
    print(f"Observation shape: {current_obs.shape}")
    print(f"Action shape: {current_action.shape}")
    print(f"Predicted next observation shape: {next_obs.shape}")
    print(f"Predicted reward: {reward.item()}")
    
    # Compare with ground truth
    true_next_obs = targets[0, :dataset._observation_dim, :1]
    true_reward = targets[0, dataset._observation_dim:, :1]
    
    print(f"True next observation MSE: {torch.mean((next_obs - true_next_obs) ** 2).item()}")
    print(f"True reward error: {abs(reward.item() - true_reward.item())}")
    
    # Visualize predictions
    visualize_predictions(modelstate.model, dataset)

    if dist.is_initialized():
        dist.destroy_process_group()

    return modelstate, df

def predict_next_state_and_reward(model, observation, action, normalize=True, dataset=None):
    """
    Predict the next observation and reward given current observation and action
    
    Args:
        model: Trained VRNN model
        observation: Current observation tensor [batch, obs_dim, 1]
        action: Current action tensor [batch, act_dim, 1]
        normalize: Whether to normalize inputs
        dataset: Dataset object for normalization
        
    Returns:
        Predicted next observation and reward
    """
    model.eval()
    
    # Combine observation and action
    if normalize and dataset is not None:
        observation = dataset.normalize_observations(observation)
        action = dataset.normalize_actions(action)
    
    # Combine into model input format [batch, obs_dim+act_dim, 1]
    model_input = torch.cat([observation, action], dim=1)
    
    # Generate prediction
    with torch.no_grad():
        # Use generate instead of forward to get predictions
        y_sample, y_mu, y_sigma, _ = model.generate(model_input)
    
    # Extract next observation and reward from y_mu
    next_obs = y_mu[:, :-1, :]  # All dimensions except the last
    reward = y_mu[:, -1:, :]    # Just the last dimension
    
    # Denormalize if needed
    if normalize and dataset is not None:
        next_obs = dataset.denormalize_observations(next_obs)
        reward = dataset.denormalize_rewards(reward)
    
    return next_obs, reward

def visualize_predictions(model, dataset, num_samples=3):
    """
    Visualize predictions of both next observation and reward against ground truth
    
    Args:
        model: Trained VRNN model
        dataset: MoCapAct dataset
        num_samples: Number of random samples to visualize
    """
    # Select random samples from the dataset
    indices = np.random.randint(0, len(dataset), num_samples)
    
    # Create subplots: 2 rows per sample (one for obs, one for reward)
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, 2)
    
    for i, idx in enumerate(indices):
        # Get the sample
        inputs, targets = dataset[idx]
        
        # For simplicity, just use a single timestep
        current_obs = inputs[0, :dataset._observation_dim, :1]
        current_action = inputs[0, dataset._observation_dim:, :1]
        true_next_obs = targets[0, :dataset._observation_dim, :1]
        true_reward = targets[0, dataset._observation_dim:, :1]
        
        # Predict next observation and reward
        pred_next_obs, pred_reward = predict_next_state_and_reward(
            model, current_obs, current_action, normalize=True, dataset=dataset
        )
        
        # 1. Plot observation comparison (first 5 dimensions for clarity)
        ax_obs = axes[i, 0]
        num_dims_to_plot = min(5, dataset._observation_dim)
        
        # Get observation values
        true_vals = true_next_obs[0, :num_dims_to_plot, 0].cpu().numpy()
        pred_vals = pred_next_obs[0, :num_dims_to_plot, 0].cpu().numpy()
        
        # Create bar positions
        x = np.arange(num_dims_to_plot)
        width = 0.35
        
        # Create bars
        ax_obs.bar(x - width/2, true_vals, width, label='True')
        ax_obs.bar(x + width/2, pred_vals, width, label='Predicted')
        
        # Add labels and legend
        ax_obs.set_title(f'Sample {i+1}: Next Observation Prediction')
        ax_obs.set_xlabel('Observation Dimension')
        ax_obs.set_ylabel('Value')
        ax_obs.set_xticks(x)
        ax_obs.set_xticklabels([f'Dim {j+1}' for j in range(num_dims_to_plot)])
        ax_obs.legend()
        ax_obs.grid(alpha=0.3)
        
        # 2. Plot reward prediction
        ax_reward = axes[i, 1]
        ax_reward.bar(['True', 'Predicted'], 
                     [true_reward.item(), pred_reward.item()], 
                     color=['blue', 'orange'])
        ax_reward.set_title(f'Sample {i+1}: Reward Prediction')
        ax_reward.set_ylabel('Reward Value')
        ax_reward.grid(alpha=0.3)
        
        # Print some statistics
        obs_mse = torch.mean((pred_next_obs - true_next_obs)**2).item()
        reward_error = abs(pred_reward.item() - true_reward.item())
        print(f"Sample {i+1} - Obs MSE: {obs_mse:.5f}, Reward Error: {reward_error:.5f}")
    
    plt.tight_layout()
    plt.savefig('predictions_vrnn.png')

def main():
    """Main function to run VRNN training on MoCapAct dataset"""
    parser = argparse.ArgumentParser(
        description='VRNN training for MoCapAct dataset'
    )
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the MoCapAct dataset')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for logs and results')
    
    parser.add_argument('--subset', type=str, default='small', choices=['small', 'large'],
                       help='Dataset subset to use (small or large)')
    parser.add_argument('--normalize_obs', action='store_true', default=True,
                       help='Normalize observations')
    parser.add_argument('--normalize_act', action='store_true', default=True,
                       help='Normalize actions')
    parser.add_argument('--normalize_reward', action='store_true', default=True,
                       help='Normalize rewards')
    parser.add_argument('--return_mean_act', action='store_true', default=False,
                       help='Use mean actions instead of sampled actions')
    parser.add_argument('--relabel_type', type=str, default='none',
                       choices=['none', 'speed', 'forward', 'backward', 'shift_left', 
                                'jump', 'rotate_x', 'rotate_y', 'rotate_z'],
                       help='Type of reward relabeling to apply')
    parser.add_argument('--validation_split', type=float, default=0.1,
                       help='Fraction of data to use for validation')
    parser.add_argument('--test_split', type=float, default=0.1,
                       help='Fraction of data to use for testing')
    
    # Training arguments
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=24,
                       help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=4,
                       help='Maximum number of epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed')
    
    # Distributed training arguments
    parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str,
                       help='URL for distributed training')
    parser.add_argument('--dist-backend', choices=['gloo', 'nccl'], default='gloo', type=str,
                       help='Distributed backend')
    parser.add_argument('--world_size', default=1, type=int,
                       help='Number of processes for distributed training')
    parser.add_argument('--distributed', action='store_true',
                       help='Enable distributed training')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='Local rank for distributed training')
    # Model architecture arguments
    parser.add_argument('--h_dim', type=int, default=32,
                    help='Hidden dimension size for VRNN')
    parser.add_argument('--z_dim', type=int, default=25,
                    help='Latent dimension size for VRNN')
    parser.add_argument('--n_mixtures', type=int, default=2,
                    help='Number of Gaussian mixture components')
    
    args = parser.parse_args()
    
    # Initialize distributed training if enabled
    if args.distributed:
        if args.local_rank is not None:
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(
                backend=args.dist_backend,
                init_method=args.init_method,
                world_size=args.world_size,
                rank=args.local_rank
                )
    
    # Run training
    train_vrnn_on_mocapact(args)
    
if __name__ == '__main__':
    main()

import collections
import numpy as np
import os
# import gin
import gym
# import mujoco_py
import tensor_specs
import time
import alf_gym_wrapper
from alf_environment import TimeLimit
import importlib
import torch
import torch.nn as nn
import sys
import shutil
import argparse


import dill
import nest
from torch.utils import data
from agac_torch.agac.memory import Memory, Transition
#####################################
# from train_eval_utils


MUJOCO_ENVS = [
    "Ant-v2",
    "HalfCheetah-v2",
    "Hopper-v2",
    "Humanoid-v2",
    "InvertedPendulum-v2",
    "InvertedDoublePendulum-v2",
    "Reacher-v2",
    "Swimmer-v2",
    "Walker2d-v2"
]
MUJOCO_ENVS_LENNGTH = {"Ant-v2": 1000,
                       "HalfCheetah-v2": 1000,
                       "Hopper-v2": 1000,
                       "Humanoid-v2": 1000,
                       "InvertedPendulum-v2": 1000,
                       "InvertedDoublePendulum-v2": 1000,
                       "Reacher-v2": 50,
                       "Swimmer-v2": 1000,
                       "Walker2d-v2": 1000,
                       "Pendulum-v0": 200
                       }




def gather(params, indices, axis=None):
    if axis is None:
        axis = 0
    if axis < 0:
        axis = len(params.shape) + axis
    if axis == 0:
        return params[indices]
    elif axis == 1:
        return params[:, indices]
    elif axis == 2:
        return params[:, :, indices]
    elif axis == 3:
        return params[:, :, :, indices]

class DatasetView(object):
    """Interface for reading from dataset."""

    def __init__(self, dataset, indices):
        self._dataset = dataset
        self._indices = indices
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_batch(self, indices):
        real_indices = self._indices[indices]
        return self._dataset.get_batch(real_indices)

    @property
    def size(self):
        return self._indices.shape[0]

class Dataset(data.Dataset):
    """Enhanced dataset based on Memory's Transition structure but with tensor storage."""

    def __init__(
            self,
            observation_spec,
            action_spec,
            size,
            circular=True,
    ):
        super(Dataset, self).__init__()
        self._size = size
        self._circular = circular
        
        # Get shapes from specs
        obs_shape = list(observation_spec.shape)
        obs_type = observation_spec.dtype
        action_shape = list(action_spec.shape)
        action_type = action_spec.dtype
        
        # Create tensor storage for all fields in the Transition structure
        self.observation = self._zeros([size] + obs_shape, obs_type)
        self.action = self._zeros([size] + action_shape, action_type)
        self.extrinsic_return = self._zeros([size], torch.float32)
        self.advantage = self._zeros([size], torch.float32)
        self.agac_advantage = self._zeros([size], torch.float32)
        self.value = self._zeros([size], torch.float32)
        self.log_pi = self._zeros([size], torch.float32)
        self.adv_log_pi = self._zeros([size], torch.float32)
        
        # For logits, we need to know the shape, but we'll assume a vector for now
        logits_shape = [size, action_shape[0] * 2]  # Placeholder - adjust as needed
        self.logits_pi = self._zeros(logits_shape, torch.float32)
        self.adv_logits_pi = self._zeros(logits_shape, torch.float32)
        
        self.done = self._zeros([size], torch.bool)
        
        # Store all tensors in a structured way for easy access
        self._data = Transition(
            observation=self.observation,
            action=self.action,
            extrinsic_return=self.extrinsic_return,
            advantage=self.advantage,
            agac_advantage=self.agac_advantage,
            value=self.value,
            log_pi=self.log_pi,
            adv_log_pi=self.adv_log_pi,
            logits_pi=self.logits_pi,
            adv_logits_pi=self.adv_logits_pi,
            done=self.done
        )
        
        # Track current size and position
        self.current_size = torch.tensor(0, requires_grad=False)
        self._current_idx = torch.tensor(0, requires_grad=False)
        self._capacity = torch.tensor(self._size)
        
        # Store configuration
        self._config = collections.OrderedDict(
            observation_spec=observation_spec,
            action_spec=action_spec,
            size=size,
            circular=circular
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a memory to help with batching and advantage normalization
        self._memory = Memory(max_size=size)

    def _zeros(self, shape, dtype):
        """Create a variable initialized with zeros."""
        return torch.zeros(shape, dtype=dtype)

    @property
    def config(self):
        return self._config

    @property
    def data(self):
        return self._data

    @property
    def capacity(self):
        return self._size

    @property
    def size(self):
        return self.current_size.item()

    def create_view(self, indices):
        return DatasetView(self, indices)

    def get_batch(self, indices):
        """Get a batch of transitions at the given indices."""
        def get_batch_(data_):
            return gather(data_, indices)

        transition_batch = nest.map_structure(get_batch_, self._data)
        return transition_batch
    
    def get_epoch_batches(self, batch_size):
        """Use Memory's get_epoch_batches to create normalized batches."""
        # First, fill the memory with our tensor data
        self._sync_memory()
        
        # Then use Memory's functionality to get normalized batches
        return self._memory.get_epoch_batches(batch_size)
    
    def _sync_memory(self):
        """Sync tensor data to Memory for normalization operations."""
        self._memory.reset()
        
        # Convert tensor data to Transition objects and add to memory
        for i in range(self.size):
            trans = Transition(
                observation=self.observation[i].cpu().numpy(),
                action=self.action[i].cpu().numpy(),
                extrinsic_return=self.extrinsic_return[i].cpu().numpy(),
                advantage=self.advantage[i].cpu().numpy(),
                agac_advantage=self.agac_advantage[i].cpu().numpy(),
                value=self.value[i].cpu().numpy(),
                log_pi=self.log_pi[i].cpu().numpy(),
                adv_log_pi=self.adv_log_pi[i].cpu().numpy(),
                logits_pi=self.logits_pi[i].cpu().numpy(),
                adv_logits_pi=self.adv_logits_pi[i].cpu().numpy(),
                done=self.done[i].cpu().numpy()
            )
            self._memory.add(trans)

    def add_transitions(self, transitions):
        """Add transitions to the dataset."""
        assert isinstance(transitions, Transition)
        
        # Process incoming transitions to ensure numpy arrays
        processed = {}
        for field in transitions._fields:
            data = getattr(transitions, field)
            
            # Convert tensors to numpy
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            
            # Handle lists
            if isinstance(data, list):
                # Convert list of tensors if needed
                if data and isinstance(data[0], torch.Tensor):
                    data = [t.cpu().numpy() for t in data]
                data = np.stack(data)
                
            processed[field] = data
        
        transitions = Transition(**processed)
        
        # Get batch size from first field shape
        batch_size = transitions.observation.shape[0]
        effective_batch_size = min(batch_size, self._size - self._current_idx)
        indices = self._current_idx + torch.arange(effective_batch_size)
        
        # Store each field in its corresponding tensor
        for key in transitions._fields:
            data = getattr(self._data, key)
            batch = getattr(transitions, key)
            
            # Ensure batch is the right shape for indexing
            if isinstance(batch, np.ndarray) and batch.ndim > 0:
                # If tensor has more dimensions than batch, add dimensions
                if data.ndim > batch.ndim:
                    for _ in range(data.ndim - batch.ndim):
                        batch = np.expand_dims(batch, axis=-1)
                # If batch has more dimensions than tensor, reduce dimensions
                elif batch.ndim > data.ndim:
                    while batch.ndim > data.ndim:
                        if batch.shape[-1] == 1:
                            batch = np.squeeze(batch, axis=-1)
                        else:
                            # Can't reduce safely, reshape needed
                            break
            
            # Convert to tensor and store
            data[indices] = torch.tensor(batch[:effective_batch_size])
        
        # Update size and index
        if self.current_size < self._size:
            self.current_size += effective_batch_size
        self._current_idx += effective_batch_size
        
        # Handle circular buffer
        if self._circular and self._current_idx >= self._size:
            self._current_idx = torch.tensor(0, requires_grad=False)
            
        # Update memory with the new transitions for normalization
        self._update_memory_with_transitions(transitions, effective_batch_size)
            
        return effective_batch_size
    
    def _update_memory_with_transitions(self, transitions, count):
        """Helper to update memory with new transitions."""
        # For a single transition
        if transitions.observation.ndim == 1 or (
            transitions.observation.ndim > 1 and transitions.observation.shape[0] == 1):
            self._memory.add(transitions)
        # For batched transitions
        else:
            for i in range(min(count, transitions.observation.shape[0])):
                # Extract individual transition
                trans = Transition(
                    observation=transitions.observation[i],
                    action=transitions.action[i],
                    extrinsic_return=transitions.extrinsic_return[i] 
                        if transitions.extrinsic_return.ndim > 0 else transitions.extrinsic_return,
                    advantage=transitions.advantage[i] 
                        if transitions.advantage.ndim > 0 else transitions.advantage,
                    agac_advantage=transitions.agac_advantage[i] 
                        if transitions.agac_advantage.ndim > 0 else transitions.agac_advantage,
                    value=transitions.value[i] 
                        if transitions.value.ndim > 0 else transitions.value,
                    log_pi=transitions.log_pi[i] 
                        if transitions.log_pi.ndim > 0 else transitions.log_pi,
                    adv_log_pi=transitions.adv_log_pi[i] 
                        if transitions.adv_log_pi.ndim > 0 else transitions.adv_log_pi,
                    logits_pi=transitions.logits_pi[i] 
                        if transitions.logits_pi.ndim > 0 else transitions.logits_pi,
                    adv_logits_pi=transitions.adv_logits_pi[i] 
                        if transitions.adv_logits_pi.ndim > 0 else transitions.adv_logits_pi,
                    done=transitions.done[i] 
                        if transitions.done.ndim > 0 else transitions.done
                )
                self._memory.add(trans)
        
    def add_transitions_batch(self, batch):
        """
        Add a batch of transitions using the Memory's Batch format.
        
        This method accepts a Batch namedtuple (containing pluralized fields like 
        'observations', 'actions', etc.) and converts it to the internal storage format.
        
        Args:
            batch: A Batch namedtuple containing batched transition data
            
        Returns:
            int: The number of transitions actually added
        """
        # First convert the Batch format to Transition format
        # Map plural batch fields to singular transition fields
        batch_size = len(batch.observations)
        
        # Create a list of individual transitions to add to memory
        # This preserves the Memory's normalization capabilities
        transitions = []
        for i in range(batch_size):
            trans = Transition(
                observation=batch.observations[i],
                action=batch.actions[i],
                extrinsic_return=batch.extrinsic_returns[i],
                advantage=batch.advantages[i],
                agac_advantage=batch.agac_advantages[i],
                value=batch.values[i],
                log_pi=batch.log_pis[i],
                adv_log_pi=batch.adv_log_pis[i],
                logits_pi=batch.logits_pi[i],
                adv_logits_pi=batch.adv_logits_pi[i],
                done=batch.dones[i]
            )
            transitions.append(trans)
        
        # Now use our existing add_transitions method to add each transition
        # This reuses all the shape handling and memory updating logic
        count = 0
        for trans in transitions:
            # Wrap in an additional dimension to indicate batch size of 1
            wrapped_trans = Transition(
                observation=np.expand_dims(trans.observation, 0),
                action=np.expand_dims(trans.action, 0),
                extrinsic_return=np.expand_dims(trans.extrinsic_return, 0) 
                    if np.isscalar(trans.extrinsic_return) else np.expand_dims(trans.extrinsic_return, 0),
                advantage=np.expand_dims(trans.advantage, 0) 
                    if np.isscalar(trans.advantage) else np.expand_dims(trans.advantage, 0),
                agac_advantage=np.expand_dims(trans.agac_advantage, 0) 
                    if np.isscalar(trans.agac_advantage) else np.expand_dims(trans.agac_advantage, 0),
                value=np.expand_dims(trans.value, 0) 
                    if np.isscalar(trans.value) else np.expand_dims(trans.value, 0),
                log_pi=np.expand_dims(trans.log_pi, 0) 
                    if np.isscalar(trans.log_pi) else np.expand_dims(trans.log_pi, 0),
                adv_log_pi=np.expand_dims(trans.adv_log_pi, 0) 
                    if np.isscalar(trans.adv_log_pi) else np.expand_dims(trans.adv_log_pi, 0),
                logits_pi=np.expand_dims(trans.logits_pi, 0) 
                    if np.isscalar(trans.logits_pi) else np.expand_dims(trans.logits_pi, 0),
                adv_logits_pi=np.expand_dims(trans.adv_logits_pi, 0) 
                    if np.isscalar(trans.adv_logits_pi) else np.expand_dims(trans.adv_logits_pi, 0),
                done=np.expand_dims(trans.done, 0) 
                    if np.isscalar(trans.done) else np.expand_dims(trans.done, 0)
            )
            
            result = self.add_transitions(wrapped_trans)
            count += result
            
            # If we've filled the buffer, stop adding
            if result == 0:
                break
        
        return count
        
    def get_advantages_stats(self):
        """Get advantage statistics using Memory's implementation."""
        self._sync_memory()
        return self._memory.get_advantages_stats()

def scatter_update(tensor, indices, updates):
    tensor = torch.tensor(tensor)
    indices = torch.tensor(indices, dtype=torch.long)
    updates = torch.tensor(updates)
    tensor[indices] = updates
    return tensor




def save_copy(data, ckpt_name):
    """Creates a copy of the current data and save as a checkpoint."""
    new_data = Dataset(
        observation_spec=data.config['observation_spec'],
        action_spec=data.config['action_spec'],
        size=data.size,
        circular=False)
    full_batch = data.get_batch(np.arange(data.size))
    new_data.add_transitions(full_batch)
    torch.save([new_data.size, new_data.state_dict()], ckpt_name + ".pt")
    with open(ckpt_name + '.pth', 'wb') as filehandler:
        dill.dump(new_data, filehandler)



#########################
# utils_VAE.py
def shuffle_indices_with_steps(n, steps=1, rand=None):
    """Randomly shuffling indices while keeping segments."""
    if steps == 0:
        return np.arange(n)
    if rand is None:
        rand = np.random
    n_segments = int(n // steps)
    n_effective = n_segments * steps
    batch_indices = rand.permutation(n_segments)
    batches = np.arange(n_effective).reshape([n_segments, steps])
    shuffled_batches = batches[batch_indices]
    shuffled_indices = np.arange(n)
    shuffled_indices[:n_effective] = shuffled_batches.reshape([-1])
    return shuffled_indices


#########################
# collect_data.py

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def get_sample_counts(n, distr):
    """Provides size of each sub-wm_image_replay_buffer based on desired distribution."""
    distr = torch.tensor(distr)
    distr = distr / torch.sum(distr)
    counts = []
    remainder = n
    for i in range(distr.shape[0] - 1):
        count = int(n * distr[i])
        remainder -= count
        counts.append(count)
    counts.append(remainder)
    return counts


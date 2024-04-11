
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import collections
import numpy as np
# import mujoco_py

from typing import Callable


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

Transition = collections.namedtuple(
    'Transition', 's1, s2, a1, a2, discount, reward')

def map_structure(func: Callable, structure):
    
    if not callable(func):
        raise TypeError("func must be callable, got: %s" % func)

    if isinstance(structure, list):
        return [map_structure(func, item) for item in structure]

    if isinstance(structure, dict):
        return {key: map_structure(func, structure[key]) for key in structure}

    if isinstance(structure, Transition):
        return Transition(
          s1 = map_structure(func, structure.s1), s2 = map_structure(func, structure.s2), a1 = map_structure(func, structure.a1), 
          a2 = map_structure(func, structure.a2), discount = map_structure(func, structure.discount), reward = map_structure(func, structure.reward))

    return func(structure)

  
#######################
def gather(params, indices, axis = None):
    to_ret = []
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
        return params[:,:,:, indices]

def scatter_update(tensor, indices, updates):
    tensor = torch.tensor(tensor, dtype=torch.float32)
    indices = torch.tensor(indices, dtype=torch.long)
    updates = torch.tensor(updates, dtype=torch.float32)
    tensor[indices] = updates
    return tensor
  
class DatasetView(object):
  """Interface for reading from wm_image_replay_buffer."""

  def __init__(self, dataset, indices):
    self._dataset = dataset
    self._indices = indices

  def get_batch(self, indices):
    real_indices = self._indices[indices]
    return self._dataset.get_batch(real_indices)

  @property
  def size(self):
    return self._indices.shape[0]

def save_copy(data, ckpt_name):
  """Creates a copy of the current data and save as a checkpoint."""
  new_data = Dataset(
      observation_spec=data.config['observation_spec'],
      action_spec=data.config['action_spec'],
      size=data.size,
      circular=False)
  full_batch = data.get_batch(np.arange(data.size))
  new_data.add_transitions(full_batch)
  torch.save(new_data, ckpt_name+".pt")
  
class Dataset(nn.Module):
  """Tensorflow module of wm_image_replay_buffer of transitions."""

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
    obs_shape = list(observation_spec.shape)
    obs_type = observation_spec.dtype
    action_shape = list(action_spec.shape)
    action_type = action_spec.dtype
    self._s1 = self._zeros([size] + obs_shape, obs_type)
    self._s2 = self._zeros([size] + obs_shape, obs_type)
    self._a1 = self._zeros([size] + action_shape, action_type)
    self._a2 = self._zeros([size] + action_shape, action_type)
    self._discount = self._zeros([size], torch.float32)
    self._reward = self._zeros([size], torch.float32)
    self._data = Transition(
        s1=self._s1, s2=self._s2, a1=self._a1, a2=self._a2,
        discount=self._discount, reward=self._reward)
    self._current_size = torch.autograd.Variable(torch.tensor(0))
    self._current_idx = torch.autograd.Variable(torch.tensor(0))
    self._capacity = torch.autograd.Variable(torch.tensor(self._size))
    self._config = collections.OrderedDict(
        observation_spec=observation_spec,
        action_spec=action_spec,
        size=size,
        circular=circular)
    print('Finished init')

  @property
  def config(self):
    return self._config

  def create_view(self, indices):
    return DatasetView(self, indices)

  def get_batch(self, indices):
    indices = torch.LongTensor(indices)
    def get_batch_(data_):
      return gather(data_, indices)
    transition_batch = map_structure(get_batch_, self._data)
    return transition_batch

  @property
  def data(self):
    return self._data

  @property
  def capacity(self):
    return self._size

  @property
  def size(self):
    return self._current_size

  def _zeros(self, shape, dtype):
    """Create a variable initialized with zeros."""
    ints = [int(i) for i in shape]
    return torch.autograd.Variable(torch.zeros(tuple(ints)))

  def add_transitions(self, transitions):
    assert isinstance(transitions, Transition)
    batch_size = transitions.s1.shape[0]
    effective_batch_size = torch.minimum(
        torch.tensor(batch_size), torch.tensor(self._size - self._current_idx))
    indices = self._current_idx + torch.arange(effective_batch_size)
    for key in transitions._asdict().keys():
      data = getattr(self._data, key)
      batch = getattr(transitions, key)
      print(batch)
      try:
        if len(batch) == 0:
          batch = [batch]
      except:
        batch = [batch]
      scatter_update(data, indices, batch[:effective_batch_size])
    # Update size and index.
    if torch.less(self._current_size, self._size):
      self._current_size+=effective_batch_size
    self._current_idx+=effective_batch_size
    if self._circular:
      if torch.greater_equal(self._current_idx, self._size):
        self._current_idx=0

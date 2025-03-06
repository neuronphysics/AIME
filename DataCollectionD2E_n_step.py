import collections
import numpy as np
import os
import gym
from absl import logging
import time
import alf_gym_wrapper
from alf_environment import TimeLimit
import importlib
import torch
import argparse
import pickle
from planner_D2E_regularizer_n_step import parse_policy_cfg, Transition, map_structure, maybe_makedirs, load_policy, \
    eval_policy_episodes
import dill
import nest
from torch.utils import data

from planner_D2E_regularizer_n_step import NStepTransitions

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


def get_transition(time_step, next_time_step, action, next_action):
    return Transition(
        s1=time_step.observation,
        s2=next_time_step.observation,
        a1=action,
        a2=next_action,
        reward=next_time_step.reward,
        discount=next_time_step.discount,
        done=next_time_step.done)


def strip_action(action):
    action = action.detach().cpu().numpy()
    if action.ndim == 0:
        action = action.expand_dims(axis=0)
    elif action.ndim > 2:
        action = action.squeeze()
    if action.ndim == 2 and action.shape[0] == 1:
        action = action.squeeze(0)
    return action

class DataCollector(object):
    def __init__(self, env, policy, data):
        self._env = env
        self._policy = policy
        self._data = data
        self._current_time_step = None
        self._saved_action = None

    def collect_transition(self, n_steps):
        if self._current_time_step is None or self._current_time_step.is_last():
            self._current_time_step = self._env.reset()
            self._saved_action = None

        if self._saved_action is None:
            self._saved_action = self._safe_get_action(self._current_time_step.observation)

        transition = NStepTransitions(
            start_time_step=self._current_time_step,
            start_action=self._saved_action
        )
        
        steps_collected = 0
        terminal_encountered = False

        for _ in range(n_steps):
            next_time_step = self._env.step(self._saved_action)
            steps_collected += 1

            if next_time_step.is_last():
                terminal_encountered = True
                next_action = None
                logging.info("Terminal step encountered at step %d", steps_collected)
            else:
                next_action = self._safe_get_action(next_time_step.observation)

            transition.add_step(next_time_step, next_action)
            if terminal_encountered:
                break
            else:
               self._current_time_step = next_time_step
               self._saved_action = next_action

        if steps_collected > 0:
            final_transition = transition.get_transition(group_size=n_steps)
            self._validate_transition(final_transition, terminal_encountered)
            self._data.add_transitions(final_transition)

        # Explicitly ensure terminal steps are marked
        if terminal_encountered:
            self._current_time_step = None
            self._saved_action = None
        return steps_collected

    def _safe_get_action(self, observation):
        obs_tensor = torch.from_numpy(observation).float()
        with torch.no_grad():
            return strip_action(self._policy(obs_tensor))

    def _validate_transition(self, transition, was_terminal):
        """Handle both tensor and numpy data"""
        def _to_numeric_array(x):
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
            if isinstance(x, np.ndarray) and x.dtype == bool:
                return x
            if isinstance(x, np.ndarray) and x.dtype == object:
               # Handle ragged arrays, converting each element to numpy array if needed
               converted = []
               for item in x:
                   if isinstance(item, torch.Tensor):
                      item = item.cpu().numpy()
                   converted.append(item.astype(np.float32))
               return np.array(converted, dtype=np.float32)

            return x.astype(np.float32) if isinstance(x, np.ndarray) else x
    
        a1_np = _to_numeric_array(transition.a1)
        s1_np = _to_numeric_array(transition.s1)

        # Check for NaNs
        if isinstance(a1_np, np.ndarray):
           assert not np.isnan(a1_np).any(), "NaN values in actions"
        if isinstance(s1_np, np.ndarray):
           assert not np.isnan(s1_np).any(), "NaN values in states"
        if was_terminal:
           done = _to_numeric_array(transition.done)
           done = done.astype(np.bool)
           discount = _to_numeric_array(transition.discount)
        
           # Check for at least one terminal step
           if not np.any(done):
               # Force mark the last step as terminal if missing
               done[-1] = True
               discount[-1] = 0.0
               logging.warning("Forced terminal flag on last transition step")

           # Ensure the last terminal step has discount 0
           terminal_indices = np.where(done)[0]
           for idx in terminal_indices:
               assert discount[idx] == 0, "Terminal discount must be 0"


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


def scatter_update(tensor, indices, updates):
    tensor = torch.tensor(tensor)
    indices = torch.tensor(indices, dtype=torch.long)
    updates = torch.tensor(updates)
    tensor[indices] = updates
    return tensor


class DatasetView(object):
    """Interface for reading from wm_image_replay_buffer."""

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


class Dataset(data.Dataset):
    """Tensorflow module of wm_image_replay_buffer of transitions."""

    def __init__(
            self,
            observation_spec,
            action_spec,
            size,
            group_size,
            circular=True,
    ):
        super(Dataset, self).__init__()
        self._size = size
        self._circular = circular
        obs_shape = list(observation_spec.shape)
        obs_type = observation_spec.dtype
        
        action_shape = list(action_spec.shape)
        action_type = action_spec.dtype
        self.s1 = self._zeros([size, group_size] + obs_shape, obs_type)
        self.s2 = self._zeros([size, group_size] + obs_shape, obs_type)
        self.a1 = self._zeros([size, group_size] + action_shape, action_type)
        self.a2 = self._zeros([size, group_size] + action_shape, action_type)
        self.discount = self._zeros([size, group_size], torch.float32)
        self.reward = self._zeros([size, group_size], torch.float32)
        self.done = self._zeros([size, group_size], torch.bool)
        self._data = Transition(
                                 s1=self.s1,
                                 s2=self.s2,
                                 a1=self.a1,
                                 a2=self.a2,
                                 discount=self.discount,
                                 reward=self.reward,
                                 done=self.done
                                )
        self.current_size = torch.autograd.Variable(torch.tensor(0), requires_grad=False)
        self._current_idx = torch.autograd.Variable(torch.tensor(0), requires_grad=False)
        self._capacity = torch.autograd.Variable(torch.tensor(self._size))
        self._config = collections.OrderedDict(
                                                observation_spec=observation_spec,
                                                action_spec=action_spec,
                                                size=size,
                                                circular=circular
                                             )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def config(self):
        return self._config

    def create_view(self, indices):
        return DatasetView(self, indices)

    def get_batch(self, indices):
        def get_batch_(data_):
            return gather(data_, indices)

        transition_batch = nest.map_structure(get_batch_, self._data)
        return transition_batch

    @property
    def data(self):
        return self._data

    @property
    def capacity(self):
        return self._size

    @property
    def size(self):
        return self.current_size.numpy()

    def _zeros(self, shape, dtype):
        """Create a variable initialized with zeros."""
        return torch.autograd.Variable(torch.zeros(shape, dtype=dtype))

    def add_transitions(self, transitions):
        assert isinstance(transitions, Transition)
        # Convert all elements to numpy arrays first
        processed = {}
        for field in transitions._fields:
            data = getattr(transitions, field)
            if isinstance(data, torch.Tensor):
               data = data.cpu().numpy()
            if isinstance(data, list):    
                data=np.stack(data)
            if field in ['a1', 'a2'] and isinstance(data[0], torch.Tensor):  # [T, action_dim]
                actions = [t.cpu().numpy() for t in data]
                data = np.stack(actions)  # Shape: [T, action_dim]
                data = np.expand_dims(data, axis=0)  # Shape: [1, T, action_dim]
            
            processed[field] = data
    
        transitions = Transition(**processed)
    
        batch_size = transitions.s1.shape[0]
        effective_batch_size = min(batch_size, self._size - self._current_idx)
        indices = self._current_idx + torch.arange(effective_batch_size)

        for key in transitions._asdict().keys():
            data = getattr(self._data, key)
            batch = getattr(transitions, key)
            # Ensure proper shape [batch_size, group_size, ...]
            if batch.ndim < data.ndim:
               for _ in range(data.ndim - batch.ndim):
                  batch = np.expand_dims(batch, axis=0)

            # Check if shapes match along group_size dimension
            if batch.shape[1] != data.shape[1]:
               # Handle potential mismatch in group_size dimension
               if batch.shape[1] < data.shape[1]:
                   # Pad batch to match expected group_size
                   pad_length = data.shape[1] - batch.shape[1]
                   pad_shape = list(batch.shape)
                   pad_shape[1] = pad_length
                
                   # Create padding based on data type
                   if key in ['done']:
                       # For boolean fields, pad with True (indicating terminal state)
                       padding = np.ones(pad_shape, dtype=batch.dtype)
                   elif key in ['discount']:
                       # For discount, pad with zeros
                       padding = np.zeros(pad_shape, dtype=batch.dtype)
                   else:
                       # For other fields, repeat the last values
                       # Extract the last values along dimension 1
                       last_values = batch[:, -1:].repeat(pad_length, axis=1)
                       padding = last_values
                
                   # Concatenate original data with padding along group_size dimension
                   batch = np.concatenate([batch, padding], axis=1)
               else:
                   # If batch has more steps than expected, truncate
                   batch = batch[:, :data.shape[1]]
        
            data[indices] = torch.from_numpy(batch[:effective_batch_size])
    
        if self.current_size < self._size:
           self.current_size += effective_batch_size
        self._current_idx += effective_batch_size
        if self._circular and self._current_idx >= self._size:
           self._current_idx = 0

    def add_transitions_batch(self, transitions):
        batch_size = transitions.s1.shape[0]
        # compute the capacity left
        effective_batch_size = torch.minimum(torch.tensor(batch_size), torch.tensor(self._size) - self._current_idx)
        indices = self._current_idx + torch.arange(effective_batch_size.item())
        # store the incoming data to dataset
        for key in transitions.__dict__:
            #TODO: check whether we get the correct shape of a1 and a2
            data = getattr(self._data, key)
            batch = getattr(transitions, key)
            data[indices] = batch[:effective_batch_size]
        # Update size and index.
        if torch.less(self.current_size, self._size):
            self.current_size += effective_batch_size
        self._current_idx += effective_batch_size
        if self._circular:
            if torch.greater_equal(self._current_idx, self._size):
                self._current_idx = 0


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



def collect_n_transitions(tf_env, policy, data, n, log_freq=1000, group_size=15):
    """Collects exactly n transitions with proper terminal handling"""
    print("Max steps from the gym spec:", tf_env.spec.max_episode_steps)

    collector = DataCollector(tf_env, policy, data)
    time_st = time.time()
    timed_at_step = 0
    steps_collected = 0
    
    while steps_collected < n:
        # Calculate remaining steps to avoid overshooting
        remaining = n - steps_collected
        actual_group_size = min(group_size, remaining)
        
        count = collector.collect_transition(actual_group_size)
        
        # Handle potential collection failures
        if count == 0:
            logging.warning("Failed to collect any transitions - check environment")
            time.sleep(0.1)  # Prevent tight loop on failure
            continue
            
        steps_collected += count
        
        # Logging with progress awareness
        if (steps_collected % log_freq == 0) or (steps_collected == n):
            elapsed = time.time() - time_st
            steps_per_sec = (steps_collected - timed_at_step) / elapsed
            logging.info(
                'Collected %d/%d (%.1f%%) @ %.2f steps/s', 
                steps_collected, n,
                100 * steps_collected / n,
                steps_per_sec
            )
            timed_at_step = steps_collected
            time_st = time.time()

    # Final validation
    if steps_collected != n:
        logging.error("Failed to collect requested transitions (%d/%d)", 
                     steps_collected, n)
    else:
        logging.info("Successfully collected %d transitions", n)
        
    
def collect_data(
        log_dir,
        data_config,
        n_samples=int(1e3),
        env_name='HalfCheetah-v2',
        log_freq=int(1e2),
        n_eval_episodes=20,
        group_size=15
):
    """
                 **** Main function ****
    Creates wm_image_replay_buffer of transitions based on desired config.
    """
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    dm_env = gym.spec(env_name).make()
    env = alf_gym_wrapper.AlfGymWrapper(dm_env, discount=0.99)
    env = TimeLimit(env, MUJOCO_ENVS_LENNGTH[env_name])
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()

    # Initialize wm_image_replay_buffer.
    sample_sizes = list([cfg[-1] for cfg in data_config])
    sample_sizes = get_sample_counts(n_samples, sample_sizes)
    logging.info(", ".join(["%s" % s for s in sample_sizes]))
    data = Dataset(
        observation_spec,
        action_spec,
        n_samples,
        group_size=group_size,
        circular=False)

    # Collect data for each policy in data_config.
    time_st = time.time()
    test_results = collections.OrderedDict()
    for (policy_name, policy_cfg, _), n_transitions in zip(
            data_config, sample_sizes):
        policy_cfg = parse_policy_cfg(policy_cfg)
        policy = load_policy(policy_cfg, action_spec, observation_spec)
        logging.info('Testing policy %s...', policy_name)
        eval_mean, eval_std = eval_policy_episodes(
            env, policy, n_eval_episodes)
        test_results[policy_name] = [eval_mean, eval_std]
        logging.info('Return mean %.4g, std %.4g.', eval_mean, eval_std)
        logging.info('Collecting data from policy %s...', policy_name)
        collect_n_transitions(env, policy, data, n_transitions, log_freq, group_size)
    # Save final wm_image_replay_buffer.
    assert data.size == data.capacity
    data_ckpt_name = os.path.join(log_dir, 'data_{}.pt'.format(env_name))
    torch.save({"wm_image_replay_buffer": data, "capacity": data.capacity}, data_ckpt_name)

    whole_data_ckpt_name = os.path.join(log_dir, 'data_{}.pth'.format(env_name))
    with open(whole_data_ckpt_name, 'wb') as filehandler:
        pickle.dump(data, filehandler)
    time_cost = time.time() - time_st
    logging.info('Finished: %d transitions collected, '
                 'saved at %s, '
                 'time cost %.4gs.', n_samples, data_ckpt_name, time_cost)


def main(args):
    logging.set_verbosity(logging.INFO)
    sub_dir = args.sub_offlinerl_dir
    log_dir = os.path.join(
        args.root_offlinerl_dir,
        args.env_name,
        args.data_name,
        sub_dir,
    )
    maybe_makedirs(log_dir)
    # print(args.config_dir)
    config_module = importlib.import_module(
        '{}.{}'.format(args.config_dir, args.config_file))
    collect_data(
        log_dir=log_dir,
        data_config=config_module.get_data_config(args.env_name,
                                                  args.policy_root_dir),
        n_samples=args.n_samples,
        env_name=args.env_name,
        n_eval_episodes=args.n_eval_episodes,
        group_size=args.group_size)


def collect_gym_data(args):
    args.sub_offlinerl_dir = '0'
    # args.env_name = 'HalfCheetah-v2'
    args.data_name = 'example'
    args.config_file = 'D2E_example'
    data_dir = 'testdata'
    args.test_srcdir = os.getcwd()
    args.policy_root_dir = os.path.join(args.test_srcdir,
                                        data_dir)
    args.n_samples = 100000  # Short collection.
    args.n_eval_episodes = 50
    main(args)


if __name__ == "__main__":
    repo_dir = "/mnt/e/pycharm_projects/AIME"

    parser = argparse.ArgumentParser(description='DreamToExplore')
    parser.add_argument('--root_offlinerl_dir', type=dir_path,
                        default=os.path.join(os.getenv('HOME', '/'), repo_dir, 'offlinerl'),
                        help='Root directory for saving data')
    parser.add_argument('--sub_offlinerl_dir', type=str, default=None, help='sub directory for saving data.')
    parser.add_argument('--test_srcdir', type=str, default=None, help='directory for saving test data.')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2', help='env name.')
    parser.add_argument('--data_name', type=str, default='random', help='data name.')
    parser.add_argument('--env_loader', type=str, default='mujoco', help='env loader, suite/gym.')
    parser.add_argument('--config_dir', type=str, default='configs', help='config file dir.')
    parser.add_argument('--config_file', type=str, default='d2e_pure', help='config file name.')
    parser.add_argument('--policy_root_dir', type=str, default=None,
                        help='Directory in which to find the behavior policy.')
    parser.add_argument('--n_samples', type=int, default=int(1e3), help='number of transitions to collect.')
    parser.add_argument('--n_eval_episodes', type=int, default=20, help='number episodes to eval each policy.')
    parser.add_argument("--gin_file", type=str, default=[], nargs='*', help='Paths to the gin-config files.')

    parser.add_argument('--gin_bindings', type=str, default=[], nargs='*', help='Gin binding parameters.')
    parser.add_argument('--group_size', type=int, default=15, help='number of steps required for general advantage'
                                                                   ' estimator')
    args = parser.parse_args()
    if not os.path.exists(os.path.join(args.root_offlinerl_dir, f"{args.env_name}/example/0")):
        os.makedirs(os.path.join(args.root_offlinerl_dir, f"{args.env_name}/example/0"))

    ##############################
    collect_gym_data(args)

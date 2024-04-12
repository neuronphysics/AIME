import collections
import numpy as np
import os
# import gin
import gym
# import mujoco_py
from absl import app
from absl import flags
from absl import logging
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
import pickle
from typing import Callable
from PIL import Image
from planner_D2E_regularizer import parse_policy_cfg, Transition, map_structure, maybe_makedirs, load_policy, \
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
    if action.ndim < 1:
        action = action.unsqueeze(0).detach().cpu()
    elif action.ndim > 1:
        action = action.detach().cpu()[0]
    else:
        action = action.detach().cpu()
    return action


class DataCollector(object):
    """Class for collecting sequence of environment experience."""

    def __init__(self, env, policy, data):
        self._env = env
        self._policy = policy
        self._data = data
        self._saved_action = None

    def collect_transition(self, t):
        time_step = self._env.current_time_step()
        if self._saved_action is None:
            self._saved_action = strip_action(self._policy(torch.from_numpy(time_step.observation)))
        action = self._saved_action

        n_step_tran = NStepTransitions(start_time_step=time_step, start_action=action)
        for i in range(t):
            next_time_step = self._env.step(action)
            next_action = strip_action(self._policy(torch.from_numpy(next_time_step.observation)))
            self._saved_action = next_action
            action = next_action

            if not time_step.is_last():
                n_step_tran.add_step(next_time_step, next_action)
            else:
                return 0
        self._data.add_transitions(n_step_tran.get_transition())
        return 1


#######################
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
        self.s1 = self._zeros([size] + [group_size] + obs_shape, obs_type)
        self.s2 = self._zeros([size] + [group_size] + obs_shape, obs_type)
        self.a1 = self._zeros([size] + [group_size] + action_shape, action_type)
        self.a2 = self._zeros([size] + [group_size] + action_shape, action_type)
        self.discount = self._zeros([size] + [group_size], torch.float32)
        self.reward = self._zeros([size] + [group_size], torch.float32)
        self.done = self._zeros([size] + [group_size], torch.bool)
        self._data = Transition(
            s1=self.s1,
            s2=self.s2,
            a1=self.a1,
            a2=self.a2,
            discount=self.discount,
            reward=self.reward,
            done=self.done)
        self.current_size = torch.autograd.Variable(torch.tensor(0), requires_grad=False)
        self._current_idx = torch.autograd.Variable(torch.tensor(0), requires_grad=False)
        self._capacity = torch.autograd.Variable(torch.tensor(self._size))
        self._config = collections.OrderedDict(
            observation_spec=observation_spec,
            action_spec=action_spec,
            size=size,
            circular=circular)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def config(self):
        return self._config

    def create_view(self, indices):
        return DatasetView(self, indices)

    def get_batch(self, indices):
        indices = torch.tensor(indices, dtype=torch.int64, requires_grad=False)

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
        for i in transitions._fields:
            attr = getattr(transitions, i)
            if torch.is_tensor(attr):
                attr = attr.detach().cpu()
            transitions = transitions._replace(**{i: np.expand_dims(attr, axis=0)})
        batch_size = transitions.s1.shape[0]
        effective_batch_size = torch.minimum(torch.tensor(batch_size), torch.tensor(self._size) - self._current_idx)
        indices = self._current_idx + torch.arange(effective_batch_size.item())
        for key in transitions._asdict().keys():
            data = getattr(self._data, key)
            batch = getattr(transitions, key)
            data[indices] = torch.tensor(batch[:effective_batch_size])
        # Update size and index.
        if torch.less(self.current_size, self._size):
            self.current_size += effective_batch_size
        self._current_idx += effective_batch_size
        if self._circular:
            if torch.greater_equal(self._current_idx, self._size):
                self._current_idx = 0

    def add_transitions_batch(self, transitions):
        batch_size = transitions.s1.shape[0]
        effective_batch_size = torch.minimum(torch.tensor(batch_size), torch.tensor(self._size) - self._current_idx)
        indices = self._current_idx + torch.arange(effective_batch_size.item())
        for key in transitions._asdict().keys():
            data = getattr(self._data, key)
            batch = getattr(transitions, key)
            data[indices] = torch.tensor(batch[:effective_batch_size])
        # Update size and index.
        if torch.less(self.current_size, self._size):
            self.current_size += effective_batch_size
        self._current_idx += effective_batch_size
        if self._circular:
            if torch.greater_equal(self._current_idx, self._size):
                self._current_idx = 0


#########################
# utils.py
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
    """Adds desired number of transitions to wm_image_replay_buffer."""
    collector = DataCollector(tf_env, policy, data)
    time_st = time.time()
    timed_at_step = 0
    steps_collected = 0
    while steps_collected < n:
        count = collector.collect_transition(group_size)
        steps_collected += count
        if (steps_collected % log_freq == 0
            or steps_collected == n) and count > 0:
            steps_per_sec = ((steps_collected - timed_at_step)
                             / (time.time() - time_st))
            timed_at_step = steps_collected
            time_st = time.time()
            logging.info('(%d/%d) steps collected at %.4g steps/s.', steps_collected,
                         n, steps_per_sec)


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

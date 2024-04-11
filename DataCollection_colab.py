import collections
import numpy as np
import os
import torch
# import gin
import gym
import torch.nn as nn
# import mujoco_py
# from absl import app
# from absl import flags
from absl import logging
import tensor_specs
import data_structures as ds
import time
import alf_gym_wrapper
import nest
import importlib
# import imageio
from alf_environment import TimeLimit
from planner_dataset import RandomSoftPolicy, GaussianRandomSoftPolicy, \
    ActorNetwork  # maybe_makedirs, load_policy, eval_policy_episodes


class ContinuousRandomPolicy(nn.Module):
    """Samples actions uniformly at random."""

    def __init__(self, action_spec):
        super(ContinuousRandomPolicy, self).__init__()
        self._action_spec = action_spec

    def __call__(self, observation):
        action = tensor_specs.sample_bounded_spec(
            self._action_spec,
            # outer_dims=[observation.shape[0]]
        )
        return action


PolicyConfig = collections.namedtuple(
    'PolicyConfig', 'ptype, ckpt, wrapper, model_params')

PTYPES = [
    'randwalk',
    'randinit',
    'load',
]

WRAPPER_TYPES = [
    'none',
    'gaussian',
]


# params: (wrapper_type, *wrapper_params)
# wrapper_type: none, eps, gaussian, gaussianeps
# !rm -r /content/gdrive/My\ Drive/testdata/Pendulum-v0/agent_partial_target

def wrap_policy(a_net, wrapper):
    """Wraps actor network with desired randomization."""
    if wrapper[0] == 'none':
        policy = RandomSoftPolicy(a_net)
    elif wrapper[0] == 'gaussian':
        policy = GaussianRandomSoftPolicy(
            a_net, std=wrapper[1])
    return policy


def load_policy(policy_cfg, action_spec, observation_spec):
    """Loads policy based on config."""
    if policy_cfg.ptype not in PTYPES:
        raise ValueError('Unknown policy type %s.' % policy_cfg.ptype)
    if policy_cfg.ptype == 'randwalk':
        policy = ContinuousRandomPolicy(action_spec)
    elif policy_cfg.ptype in ['randinit', 'load']:
        a_net = ActorNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=policy_cfg.model_params)
        if policy_cfg.ptype == 'load' and os.path.exists(policy_cfg.ckpt):
            logging.info('Loading policy from %s...', policy_cfg.ckpt)
            a_net = torch.load(policy_cfg.ckpt)
        policy = wrap_policy(a_net, policy_cfg.wrapper)
    return policy


def parse_policy_cfg(policy_cfg):
    return PolicyConfig(*policy_cfg)


def maybe_makedirs(log_dir):
    import os.path
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


#####################################
# from train_eval_utils
from typing import Callable

Transition = collections.namedtuple(
    'Transition', 's1, s2, a1, a2, discount, reward')

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image


# animation
def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,


def plot_animation(frames, repeat=False, interval=40):
    plt.close()  # or else nbagg sometimes plots in the previous cell
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    return animation.FuncAnimation(fig, update_scene,
                                   fargs=(frames, patch),
                                   frames=len(frames),
                                   repeat=repeat, interval=interval)


def eval_policy_episodes(env, policy, n_episodes):
    """Evaluates policy performance."""
    results = []

    discount = []
    # For video / GIF.
    for ep_idx in range(n_episodes):
        time_step = env.reset()
        total_rewards = 0.0
        # env.render()
        while not time_step.is_last():

            action = policy(torch.from_numpy(time_step.observation))[0]
            print(f"episode: {ep_idx}, action:{action}, observation : {time_step.observation}")
            if action.ndim < 1:
                time_step = env.step(action.unsqueeze(0))
            else:
                time_step = env.step(action)
            total_rewards += time_step.reward or 0.
            # img = env.render(mode = 'rgb_array')
            # img = Image.fromarray(img,'RGB')
            # img.save('/content/gdrive/My Drive/offlinerl/data/HalfCheetah_{}.jpg'.format(ep_idx))
            results.append(total_rewards)
            discount.append(time_step.discount)
        print("here...")
    results = torch.tensor(results)
    return torch.mean(results).to(dtype=torch.float32), torch.std(results).to(dtype=torch.float32), torch.tensor(
        discount).to(torch.float32)


def eval_policies(env, policies, n_episodes):
    results_episode_return = []
    infos = collections.OrderedDict()
    for name, policy in policies.items():
        mean, _, dsc = eval_policy_episodes(env, policy, n_episodes)
        results_episode_return.append(mean)
        infos[name] = collections.OrderedDict()
        infos[name]['episode_mean'] = mean
        infos[name]['discount'] = dsc
    results = results_episode_return
    return results, infos


def map_structure(func: Callable, structure):
    if not callable(func):
        raise TypeError("func must be callable, got: %s" % func)

    if isinstance(structure, list):
        return [map_structure(func, item) for item in structure]

    if isinstance(structure, dict):
        return {key: map_structure(func, structure[key]) for key in structure}

    return func(structure)


###############

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
                       "Pendulum-v1": 200
                       }


def get_transition(time_step, next_time_step, action, next_action):
    return Transition(
        s1=time_step.observation,
        s2=next_time_step.observation,
        a1=action,
        a2=next_action,
        reward=next_time_step.reward,
        discount=next_time_step.discount)


class DataCollector(object):
    """Class for collecting sequence of environment experience."""

    def __init__(self, env, policy, data):
        self._env = env
        self._policy = policy
        self._data = data
        self._saved_action = None

    def collect_transition(self):
        """Collect single transition from environment."""
        time_step = self._env.current_time_step()
        if self._saved_action is None:
            self._saved_action = self._policy(torch.from_numpy(time_step.observation))[0]
        action = self._saved_action
        if action.ndim < 1:
            next_time_step = self._env.step(action.unsqueeze(0))
        else:
            next_time_step = self._env.step(action)
        next_action = self._policy(torch.from_numpy(next_time_step.observation))[0]
        self._saved_action = next_action
        print("time_step", time_step)
        print("time_step.is_last()", time_step.is_last())
        if not time_step.is_last():
            transition = get_transition(time_step, next_time_step,
                                        action, next_action)
            print(f"transition {transition}")
            self._data.add_transitions(transition)
            return 1
        else:
            return 0


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
    torch.save(new_data, ckpt_name + ".pt")


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
        self._current_size = torch.tensor(0, dtype=torch.int32, requires_grad=False)
        self._current_idx = torch.tensor(0, dtype=torch.int32, requires_grad=False)
        self._capacity = torch.autograd.Variable(torch.tensor(self._size))
        self._config = collections.OrderedDict(
            observation_spec=observation_spec,
            action_spec=action_spec,
            size=size,
            circular=circular)

    @property
    def config(self):
        return self._config

    def create_view(self, indices):
        return DatasetView(self, indices)

    def get_batch(self, indices):
        indices = torch.LongTensor(indices)

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
        return self._current_size.numpy()

    def _zeros(self, shape, dtype):
        """Create a variable initialized with zeros."""
        return torch.autograd.Variable(torch.zeros(shape, dtype=dtype))

    def add_transitions(self, transitions):
        print("transitions", transitions)
        assert isinstance(transitions, Transition)
        batch_size = 1  # transitions.s1.shape[0]
        print(f" batch size: {batch_size}, {self._size - self._current_idx}, s1:{transitions.s1}")
        effective_batch_size = torch.minimum(torch.tensor(batch_size), torch.tensor(self._size - self._current_idx))
        indices = self._current_idx + torch.arange(effective_batch_size)
        for key in transitions._asdict().keys():
            data = getattr(self._data, key)
            batch = getattr(transitions, key)
            print(f"batch size {effective_batch_size}, keys: {key}, batch: {batch}")
            print(f"data:\n{data}, indexes:{indices}")
            if key != 'discount' and key != 'reward':
                print("effective_batch_size", effective_batch_size, "batch", batch, "indices", indices)
                # print("batch[:effective_batch_size]", batch[:effective_batch_size])
                print("data[indices]", data[indices])

                # print(f"...used batch: {batch[:effective_batch_size]} and indices are {indices}")
                # data[indices]= torch.tensor(batch[:effective_batch_size])
                data[indices] = torch.tensor(batch)
                print(f"data after:\n{data}")
            # scatter_update(data, indices, batch[:effective_batch_size])
        # Update size and index.
        if torch.less(self._current_size, self._size):
            self._current_size += effective_batch_size
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
# app.flags.DEFINE_string('f', '', 'kernel')
# FLAGS =app.flags.FLAGS
import sys


# def del_all_flags(FLAGS):
def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


import shutil
# if not os.path.exists("/content/configs"):
#    shutil.copytree('/content/gdrive/My Drive/configs', "/content/configs")
# #shutil.copy('/content/gdrive/My Drive/configs/D2E_example.py', "/content")

# if not os.path.exists('/content/gdrive/My Drive/offlinerl/data'):
#   os.makedirs('/content/gdrive/My Drive/offlinerl/data')
# else:
#   pass

# if not os.path.exists('/content/gdrive/My Drive/testdata/Pendulum-v0'):
#      os.makedirs('/content/gdrive/My Drive/testdata/Pendulum-v0/')
# else:
#   pass
# FLAGS = flags.FLAGS
# app.flags.DEFINE_string('f', '', 'kernel')
import argparse

parser = argparse.ArgumentParser(description='BRAC')
parser.add_argument('--root_offlinerl_dir', type=dir_path, default='./offlinerl/data',
                    help='Root directory for saving data')
parser.add_argument('--sub_offlinerl_dir', type=str, default=None, help='sub directory for saving data.')
parser.add_argument('--test_srcdir', type=str, default=None, help='directory for saving test data.')
parser.add_argument('--env_name', type=str, default='HalfCheetah-v2', help='env name.')
parser.add_argument('--data_name', type=str, default='random', help='data name.')
parser.add_argument('--env_loader', type=str, default='mujoco', help='env loader, suite/gym.')
parser.add_argument('--config_dir', type=str, default='configs', help='config file dir.')
parser.add_argument('--config_file', type=str, default='d2e_pure', help='config file name.')
parser.add_argument('--policy_root_dir', type=str, default=None, help='Directory in which to find the behavior policy.')
parser.add_argument('--n_samples', type=int, default=int(1e2), help='number of transitions to collect.')
parser.add_argument('--n_eval_episodes', type=int, default=20, help='number episodes to eval each policy.')
parser.add_argument("--gin_file", type=str, default=[], nargs='*', help='Paths to the gin-config files.')

parser.add_argument('--gin_bindings', type=str, default=[], nargs='*', help='Gin binding parameters.')
args = parser.parse_args()

maybe_makedirs(args.root_offlinerl_dir)


# data_dir = os.path.join(args.root_offlinerl_dir, args.env_name, "example/0")
# if not os.path.exists(data_dir):
#    os.makedirs(data_dir)
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


def collect_n_transitions(tf_env, policy, data, n, log_freq=1000):
    """Adds desired number of transitions to wm_image_replay_buffer."""
    collector = DataCollector(tf_env, policy, data)
    time_st = time.time()
    timed_at_step = 0
    steps_collected = 0
    while steps_collected < n:
        count = collector.collect_transition()
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
        circular=False)

    # Collect data for each policy in data_config.
    time_st = time.time()
    test_results = collections.OrderedDict()
    for (policy_name, policy_cfg, _), n_transitions in zip(
            data_config, sample_sizes):
        policy_cfg = parse_policy_cfg(policy_cfg)
        policy = load_policy(policy_cfg, action_spec, observation_spec)
        logging.info('Testing policy %s...', policy_name)
        eval_mean, eval_std, _ = eval_policy_episodes(
            env, policy, n_eval_episodes)
        test_results[policy_name] = [eval_mean, eval_std]
        logging.info('Return mean %.4g, std %.4g.', eval_mean, eval_std)
        logging.info('Collecting data from policy %s...', policy_name)
        collect_n_transitions(env, policy, data, n_transitions, log_freq)
    # Save final wm_image_replay_buffer.
    data_ckpt_name = os.path.join(log_dir, 'data_{}.pt'.format(env_name))
    torch.save(data, data_ckpt_name)
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
    # print(args.n_samples)
    config_module = importlib.import_module(
        '{}.{}'.format(args.config_dir, args.config_file))
    collect_data(
        log_dir=log_dir,
        data_config=config_module.get_data_config(args.env_name,
                                                  args.policy_root_dir),
        n_samples=args.n_samples,
        env_name=args.env_name,
        n_eval_episodes=args.n_eval_episodes)


def collect_gym_data(args):
    args.sub_offlinerl_dir = '0'
    args.env_name = 'Pendulum-v1'
    args.data_name = 'example'
    args.config_file = 'D2E_example'
    data_dir = 'testdata'
    args.test_srcdir = os.getcwd()
    args.policy_root_dir = os.path.join(args.test_srcdir,
                                        data_dir)
    args.n_samples = 100  # Short collection.
    args.n_eval_episodes = 1
    main(args)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    collect_gym_data(args)

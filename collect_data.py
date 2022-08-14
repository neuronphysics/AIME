
import collections
import numpy as np
import os
import gin
import gym
import alf_gym_wrapper
# import mujoco_py
from absl import app
from absl import flags
from absl import logging
import torch
from dataset import Dataset
import time
import utils_planner as utils
import argparse

from planner_regularizer_debug import parse_policy_cfg, load_policy, eval_policy_episodes
import importlib
from dataset import Transition
import pickle


USE_LISTS = True


# PLEASE USE GENERATE_TRANSITIONS_DATASET.PY TO CALL FUNCTIONS IN THIS MODULE

parser = argparse.ArgumentParser(description='AIME')
parser.add_argument('--root_offlinerl_dir', type=str, default='./offlinerl', help='Experiment ID')
parser.add_argument('--sub_offlinerl_dir', type=str, default='.', help='sub directory for saving data.')
parser.add_argument('--test_srcdir', type=str, default='0', help='directory for saving test data.')
parser.add_argument('--env_name', type=str, default='HalfCheetah-v2', help='env name.')
parser.add_argument('--data_name', type=str, default='random', help='data name.')
parser.add_argument('--env_loader', type=str, default='mujoco', help='env loader, suite/gym.')
parser.add_argument('--config_dir',
                        type=str, default=None, # 'behavior_regularized_offline_rl.brac.configs'
                        help='config file dir.')
parser.add_argument('--config_file', type=str, default='dcfg_pure', help='config file name.')
parser.add_argument('--policy_root_dir', type=str, default=None,
                        help='Directory in which to find the behavior policy.')
parser.add_argument('--n_samples', type=int, default=int(1e6), help='number of transitions to collect.')
parser.add_argument('--n_eval_episodes', type=int, default=20,
                        help='number episodes to eval each policy.')
parser.add_argument('--gin_file', type=str, default=None, help='Paths to the gin-config files.')
parser.add_argument('--gin_bindings', type=str, default=None, help='Gin binding parameters.')
parser.add_argument('--hiddenflags', type=str, default='0', help='Hidden flags parameter.') 

args = parser.parse_args()


# flags = flags.FLAGS

class PolicyTest():

  def __init__(self):
    args.sub_offlinerl_dir = '.'
    args.env_name = 'Pendulum-v0'
    args.data_name = 'example'
    args.config_file = 'dcfg_example'
    data_dir = 'testdata'
    args.test_srcdir= '.'
    # args.policy_root_dir = os.path.join(args.test_srcdir,
                                                # data_dir)
    args.n_samples = 10000  # Short collection.
    args.n_eval_episodes = 1
    self.main(None)

  # def setFlagsForCollect(self):
    

  def get_sample_counts(self, n, distr):
    """Provides size of each sub-dataset based on desired distribution."""
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


  def collect_n_transitions(self, tf_env, policy, data, n, log_freq=10000):
    """Adds desired number of transitions to dataset. Transitions come from policy"""
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


  def collect_data_with_policy(
      self, 
      log_dir,
      data_config,
      n_samples=int(1e6),
      env_name='Pendulum-v0',
      log_freq=int(1e4),
      n_eval_episodes=20,
      ):
    """Creates dataset of transitions based on desired policy config."""
    print('Entered Collect Data')
    seed=0
    torch.manual_seed(seed)
    np.random.seed(seed)
    dm_env = gym.make(env_name)
    env = alf_gym_wrapper.AlfGymWrapper(dm_env)
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    

    # Initialize dataset.
    # sample_sizes = list([cfg[-1] for cfg in data_config])
    sample_sizes = [2,2,2]
    sample_sizes = self.get_sample_counts(n_samples, sample_sizes)
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
      policy = load_policy(policy_cfg, action_spec)
      logging.info('Testing policy %s...', policy_name)
      eval_mean, eval_std = eval_policy_episodes(
          env, policy, n_eval_episodes)
      test_results[policy_name] = [eval_mean, eval_std]
      logging.info('Return mean %.4g, std %.4g.', eval_mean, eval_std)
      logging.info('Collecting data from policy %s...', policy_name)
      self.collect_n_transitions(env, policy, data, n_transitions, log_freq)

    # Save final dataset.
    data_ckpt_name = os.path.join(log_dir, 'data')
    torch.save(data,data_ckpt_name)
    time_cost = time.time() - time_st
    logging.info('Finished: %d transitions collected, '
                'saved at %s, '
                'time cost %.4gs.', n_samples, data_ckpt_name, time_cost)



  def main(self, argv):
    del argv
    logging.set_verbosity(logging.INFO)
    # gin.parse_config_files_and_bindings(flags.gin_file, flags.gin_bindings)
    sub_dir = args.sub_offlinerl_dir
    data_in_root_offline = utils.make_base_dir([args.root_offlinerl_dir, args.env_name, ])
    print('HELLO'+data_in_root_offline)

    log_dir = os.path.join(
        data_in_root_offline,
        args.data_name,
        # sub_dir,
        )
    utils.maybe_makedirs(log_dir)

    data_config = None
    if args.config_dir:
      config_module = importlib.import_module(
          '{}.{}'.format(args.config_dir, args.config_file))
      data_config=config_module.get_data_config(args.env_name,
                                                  args.policy_root_dir)
    # self.collect_data_with_policy(
    #     log_dir=log_dir,
    #     data_config=data_config,
    #     n_samples=args.n_samples,
    #     env_name=args.env_name,
    #     n_eval_episodes=args.n_eval_episodes)


def env_factory(env_name):
  py_env = gym.make(env_name)
  # py_env = gym.spec(env_name).make()
  # tf_env = tf_py_environment.TFPyEnvironment(py_env)
  return py_env
  

def get_transition(state, next_state, action, next_action, reward, discount):
  return Transition(
      s1=state,
      s2=next_state,
      a1=action,
      a2=next_action,
      reward=reward,
      discount= discount)
  

class DataCollector(object):
  """Class for collecting sequence of environment experience."""

  def __init__(self, env, policy, data, discount=0.99):
    self._env = env
    self._policy = policy
    self._data = data
    self._saved_action = None
    self.discount = discount
    self.states = []
    self.next_states = []
    self.actions = []
    self.next_actions = []
    self.rewards = []
    self.discounts = []

  
  def addTransitionData(self, state, next_state, action, next_action, reward, steps_so_far):
    self.states.append(state)
    self.next_states.append(next_state)
    self.actions.append(action)
    self.next_actions.append(next_action)
    self.rewards.append(reward)

    discount = self.discount**steps_so_far
    self.discounts.append(discount)

  
  def saveCollection(self, filename):
    suffix_map = {'states':self.states, 
                  'next_states':self.next_states,
                  'actions':self.actions,
                  'next_actions':self.next_actions,
                  'rewards': self.rewards,
                  'discounts':self.discounts}
    for k, v in suffix_map.items():
      torch.save(v, filename+'_'+k+'.pt')
      # with open(filename+'_'+k+'.pkl', 'wb') as f:
      #   pickle.dump(v, f)


  def collect_transition(self, state, steps_so_far):
    """Collect single transition from environment. Actions are from policy"""
    # time_step = self._env.current_time_step()
    if self._saved_action is None:
      self._saved_action = self._policy(state)[0]
      if self._saved_action == 'random':
        self._saved_action = self._env.action_space.sample()
    action = self._saved_action
    next_state, reward, done, _ = self._env.step(action)
    next_action = self._policy(next_state)[0]
    if next_action == 'random':
        next_action = self._env.action_space.sample()
    self._saved_action = next_action
    if not done:
      # Assuming standard discounted reward
      # CAN CHOOSE TO USE EITHER DATASET CLASS OR INDIVIDUAL LISTS
      if USE_LISTS:
        self.addTransitionData(state, next_state, action, next_action, reward, self.discount**steps_so_far)
      else:
        transition = get_transition(state, next_state,
                                    action, next_action, 
                                    reward, self.discount**steps_so_far)
        self._data.add_transitions(transition)
      return 1, next_state
    else:
      # if USE_LISTS:
      #   self.addTransitionData(state, next_state, action, None, reward, self.discount**steps_so_far)
      return 0, None



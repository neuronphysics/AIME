import utils_planner as utils
from math import inf
import torch
from torch import jit
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions import transforms as tT
from torch.distributions.transformed_distribution import TransformedDistribution
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase
import collections
import numpy as np
import os
import gin
import gym
# import mujoco_py
from absl import app
from absl import flags
from absl import logging
import argparse

import time
import alf_gym_wrapper
import importlib  

from dataset import Dataset
from collect_data import DataCollector
from planner_regularizer_debug import Flags, D2EAgent, eval_policies



def train_eval_offline(
    # Basic args.
    log_dir,
    data_file,
    agent_module,
    env_name='HalfCheetah-v2',
    n_train=int(1e6),
    shuffle_steps=0,
    seed=0,
    use_seed_for_data=False,
    # Train and eval args.
    total_train_steps=int(1e6),
    summary_freq=100,
    print_freq=1000,
    save_freq=int(2e4),
    eval_freq=5000,
    n_eval_episodes=20,
    # Agent args.
    model_params=(((200, 200),), 2),
    optimizers=(('adam', 0.001),),
    batch_size=256,
    weight_decays=(0.0,),
    update_freq=1,
    update_rate=0.005,
    discount=0.99,
    ):
  """Training a policy with a fixed dataset."""
  # Create tf_env to get specs.
  env = gym.spec(env_name).make()
  # env = alf_gym_wrapper.AlfGymWrapper(dm_env)
  
  observation_spec = env.reset()
  action = env.action_space.sample()
  action_spec = action
  initial_state = env.reset()
  

  # Prepare data.
  logging.info('Loading data from %s ...', data_file)
  data  = torch.load(data_file)
  data_size = data.size()      
  full_data = Dataset(observation_spec, action_spec, data_size)
  # Split data.
  n_train = min(n_train, full_data.size)
  logging.info('n_train %s.', n_train)
  if use_seed_for_data:
    rand = np.random.RandomState(seed)
  else:
    rand = np.random.RandomState(0)
  shuffled_indices = utils.shuffle_indices_with_steps(
      n=full_data.size, steps=shuffle_steps, rand=rand)
  train_indices = shuffled_indices[:n_train]
  train_data = full_data.create_view(train_indices)

  # Create agent.
  agent_flags = Flags(
      observation_spec=observation_spec,
      action_spec=action_spec,
      model_params=model_params,
      optimizers=optimizers,
      batch_size=batch_size,
      weight_decays=weight_decays,
      update_freq=update_freq,
      update_rate=update_rate,
      discount=discount,
      train_data=train_data)
  agent_args = agent_module.Config(agent_flags).agent_args
  agent = agent_module.Agent(**vars(agent_args)) #ATTENTION: Debugg ====> should it be D2EAgent here??
  agent_ckpt_name = os.path.join(log_dir, 'agent')

  # Restore agent from checkpoint if there exists one.
  if os.path.exists('{}.index'.format(agent_ckpt_name)):
    logging.info('Checkpoint found at %s.', agent_ckpt_name)
    agent = torch.load(agent_ckpt_name)

  # Train agent.
  train_summary_dir = os.path.join(log_dir, 'train')
  eval_summary_dir = os.path.join(log_dir, 'eval')
  # train_summary_writer = SummaryWriter(
  #     logdir=train_summary_dir)
  # eval_summary_writers = collections.OrderedDict()
  # for policy_key in agent.test_policies.keys():
  #   eval_summary_writer = SummaryWriter(
  #       logdir=os.path.join(eval_summary_dir, policy_key))
  #   eval_summary_writers[policy_key] = eval_summary_writer
  eval_results = []

  time_st_total = time.time()
  time_st = time.time()
  step = agent.global_step
  timed_at_step = step
  while step < total_train_steps:
    agent.train_step()
    step = agent.global_step
    # if step % summary_freq == 0 or step == total_train_steps:
    #   agent.write_train_summary(train_summary_writer)
    # if step % print_freq == 0 or step == total_train_steps:
    #   agent.print_train_info()
    if step % eval_freq == 0 or step == total_train_steps:
      time_ed = time.time()
      time_cost = time_ed - time_st
      logging.info(
          'Training at %.4g steps/s.', (step - timed_at_step) / time_cost)
      eval_result, eval_infos = eval_policies(
          env, agent.test_policies, n_eval_episodes)
      eval_results.append([step] + eval_result)
      logging.info('Testing at step %d:', step)
      for policy_key, policy_info in eval_infos.items():
        logging.info(utils.get_summary_str(
            step=None, info=policy_info, prefix=policy_key+': '))
        # utils.write_summary(eval_summary_writers[policy_key], step, policy_info)
      time_st = time.time()
      timed_at_step = step
    if step % save_freq == 0:
      torch.save(agent, agent_ckpt_name+'.pt')
      logging.info('Agent saved at %s.', agent_ckpt_name)

  torch.save(agent, agent_ckpt_name+'.pt')
  time_cost = time.time() - time_st_total
  logging.info('Training finished, time cost %.4gs.', time_cost)
  return torch.tensor(eval_results)

##############################
###train_offline.py
parser = argparse.ArgumentParser(description='AIME')

# parser.add_argument('--test_srcdir', type=str, default='0', help='directory for saving test data.')

parser.add_argument('--data_root_dir', type=str, default=os.getenv('HOME', '/'), help='Root directory for data.')
parser.add_argument('--data_subdir', type=str, default='auto', help='sub directory for saving data.')
parser.add_argument('--data_name', type=str, default='eps1', help='data name.')
parser.add_argument('--data_file_name', type=str, default='data', help='data checkpoint file name.')


parser.add_argument('--root_dir', type=str, default='./offlinerl', help='Root directory for writing logs/summaries/checkpoints.')
parser.add_argument('--sub_dir', type=str, default='auto', help='sub directory for saving results.')

parser.add_argument('--agent_name', type=str, default='D2E', help='agent name.')
parser.add_argument('--env_name', type=str, default='HalfCheetah-v2', help='env name.')
parser.add_argument('--env_loader', type=str, default='mujoco', help='env loader, suite/gym.')

# parser.add_argument('--config_dir',
#                         type=str, default=None, # 'behavior_regularized_offline_rl.brac.configs'
#                         help='config file dir.')
# parser.add_argument('--config_file', type=str, default='dcfg_pure', help='config file name.')
# parser.add_argument('--policy_root_dir', type=str, default=None,
#                         help='Directory in which to find the behavior policy.')
# parser.add_argument('--n_samples', type=int, default=int(1e6), help='number of transitions to collect.')
parser.add_argument('--seed', type=int, default=0, help='random seed, mainly for training samples.')
parser.add_argument('--total_train_steps', type=int, default=int(5e5))
parser.add_argument('--n_train', type=int, default=int(1e6))

parser.add_argument('--n_eval_episodes', type=int, default=20,
                        help='number episodes to eval each policy.')
parser.add_argument('--gin_file', type=str, default=None, help='Paths to the gin-config files.')
parser.add_argument('--gin_bindings', type=str, default=None, help='Gin binding parameters.')
# parser.add_argument('--hiddenflags', type=str, default='0', help='Hidden flags parameter.') 

args = parser.parse_args()


AGENT_MODULES_DICT = {
    'D2E': D2EAgent, ###Debug ===> is it correct to set it here??? 
}

def main(_):
    # logging.set_verbosity(logging.INFO)
    gin.parse_config_files_and_bindings(args.gin_file, args.gin_bindings)

    # Setup data file path.
    if args.data_subdir == 'auto':
        data_subdir = utils.get_datetime()
    data_dir = os.path.join(
        args.data_root_dir,
        args.env_name,
        args.data_name,
        data_subdir,
        )
    data_file = os.path.join(
        data_dir, args.data_file_name)

    # Setup log dir.
    if args.sub_dir == 'auto':
        sub_dir = utils.get_datetime()
    else:
        sub_dir = args.sub_dir

    base_dir = utils.make_base_dir([args.root_dir, args.env_name, args.data_name,'n'+str(args.n_train),args.agent_name,sub_dir])
    log_dir = os.path.join(base_dir, str(args.seed),)
    utils.maybe_makedirs(log_dir)
    train_eval_offline(
        log_dir=log_dir,
        data_file=data_file,
        agent_module=AGENT_MODULES_DICT[args.agent_name],
        env_name=args.env_name,
        n_train=args.n_train,
        total_train_steps=args.total_train_steps,
        n_eval_episodes=args.n_eval_episodes,
    )


# class TrainOfflineTest(TestCase):

#   def test_train_offline(self):
#     data_dir = 'testdata/data'
#     flags.FLAGS.data_root_dir = os.path.join(flags.FLAGS.test_srcdir, data_dir)
#     flags.FLAGS.sub_dir = '0'
#     flags.FLAGS.env_name = 'HalfCheetah-v2'
#     flags.FLAGS.data_name = 'example'
#     flags.FLAGS.agent_name = 'D2E'
#     flags.FLAGS.gin_bindings = [
#         'train_eval_offline.model_params=((200, 200),)',
#         'train_eval_offline.optimizers=(("adam", 5e-4),)']
#     flags.FLAGS.n_train = 100
#     flags.FLAGS.n_eval_episodes = 1
#     flags.FLAGS.total_train_steps = 100  # Short training.

#     main(None)  # Just test that it runs.

if __name__ == '__main__':
  main(None)
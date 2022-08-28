
"""Training and evaluation in the offline mode."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from planner_behavior_regularizer_actor_critic import eval_policies
import collections
import os
import time
import gym
import alf_gym_wrapper
from tensorboardX import SummaryWriter
from absl import logging
import gin
import re
import torch
import numpy as np
import datetime
import argparse
from alf_environment import TimeLimit
import utils_planner as utils
from DataCollection import *
from planner_behavior_regularizer_actor_critic import Flags, BRACAgent, Config
import sys
import pickle
import dill
gin.clear_config()
def get_datetime():
  now = datetime.datetime.now().isoformat()
  now = re.sub(r'\D', '', now)[:-6]
  return now

class CustomUnpickler(dill.Unpickler):

    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)



      
@gin.configurable
def train_eval_offline(
    # Basic args.
    log_dir,
    data_file,
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
    optimizers=(( 0.0001, 0.5, 0.99),),
    batch_size=256,
    weight_decays=(0.0,),
    update_freq=1,
    update_rate=0.005,
    discount=0.99,
    device=None
    ):
  ###Training a policy with a fixed dataset.###
  # Create tf_env to get specs.
  dm_env = gym.spec(env_name).make()
  env = alf_gym_wrapper.AlfGymWrapper(dm_env,discount=discount)
  env = TimeLimit(env, MUJOCO_ENVS_LENNGTH[env_name])
  observation_spec = env.observation_spec()
  action_spec = env.action_spec()

  # Prepare data.
  logging.info('Loading data from %s ...', data_file)
  
  
  data_ckpt_name = os.path.join(data_file, 'data_{}.pt'.format(env_name))
  whole_data_ckpt_name = os.path.join(data_file, 'data_{}.pth'.format(env_name))
  
  data_size, state = torch.load(data_ckpt_name, map_location=device)
  
  

  if os.path.getsize(whole_data_ckpt_name) > 0:       
    with open(whole_data_ckpt_name, "rb") as f:
        # if file is not empty scores will be equal
        # to the value unpickled
          full_data =CustomUnpickler(f).load()
 
  logging.info('Loading data from dataset with size %d , %d ...', data_size, full_data.size)  
  for k, v in full_data._config.items():
      if k =='observation_spec':
          full_data._config['observation_spec']=observation_spec
      elif k=='action_spec':
          full_data._config['action_spec']=action_spec
 
  # Split data.
  n_train = min(n_train, full_data.size)
  logging.info('n_train %s.', n_train)
  if use_seed_for_data:
    rand = np.random.RandomState(seed)
  else:
    rand = np.random.RandomState(0)
  shuffled_indices = shuffle_indices_with_steps(
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
  agent_args = Config(agent_flags).agent_args
  agent = BRACAgent(**vars(agent_args)) #ATTENTION: Debugg ====> should it be D2EAgent here??
  agent_ckpt_name = os.path.join(log_dir, 'agent')

  # Restore agent from checkpoint if there exists one.
  if os.path.exists('{}.index'.format(agent_ckpt_name)):
    logging.info('Checkpoint found at %s.', agent_ckpt_name)
    torch.load(agent, agent_ckpt_name)

  # Train agent.
  train_summary_dir = os.path.join(log_dir, 'train')
  eval_summary_dir = os.path.join(log_dir, 'eval')
  train_summary_writer = SummaryWriter(
      logdir=train_summary_dir)
  eval_summary_writers = collections.OrderedDict()
  for policy_key in agent.test_policies.keys():
    eval_summary_writer = SummaryWriter(
        logdir=os.path.join(eval_summary_dir, policy_key))
    eval_summary_writers[policy_key] = eval_summary_writer
  eval_results = []

  time_st_total = time.time()
  time_st = time.time()
  step = agent.global_step
  timed_at_step = step
  while step < total_train_steps:
    agent.train_step()
    step = agent.global_step
    if step % summary_freq == 0 or step == total_train_steps:
      agent.write_train_summary(train_summary_writer)
    if step % print_freq == 0 or step == total_train_steps:
      agent.print_train_info()
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
        utils.write_summary(eval_summary_writers[policy_key], policy_info, step)
      time_st = time.time()
      timed_at_step = step
    if step % save_freq == 0:
       agent.checkpoint_path=agent_ckpt_name
       agent._build_checkpointer()
       logging.info('Agent saved at %s.', agent_ckpt_name)

  agent._build_checkpointer()
  time_cost = time.time() - time_st_total
  logging.info('Training finished, time cost %.4gs.', time_cost)
  return torch.tensor(eval_results)

##############################
###train_offline.py
parser = argparse.ArgumentParser(description='BRAC')
parser.add_argument('--data_root_offlinerl_dir', type=dir_path, default='/home/memole/TEST/AIME/start-with-brac/offlinerl',
                     help='Root directory for data.')
parser.add_argument('--data_sub_offlinerl_dir',type=str, default=None, help= '')
parser.add_argument('--test_srcdir', type=str, default='/home/memole/TEST/AIME/start-with-brac/', help='directory for saving test data.')
parser.add_argument('--data_name', type=str, default='eps1',help= 'data name.')
parser.add_argument('--data_file_name', type=str, default='',help= 'data checkpoint file name.')

# Flags for offline training.
parser.add_argument('--root_dir',type=dir_path, default= os.path.join(os.getenv('HOME', '/'), 'TEST/AIME/start-with-brac/offlinerl/learn'),
                    help='Root directory for writing logs/summaries/checkpoints.')
parser.add_argument('--sub_dir', type=str, default='0', help='')

parser.add_argument('--agent_name',  type=str, default='BRAC', help='agent name.')
parser.add_argument('--env_name', type=str, default='HalfCheetah-v2',help = 'env name.')
parser.add_argument('--seed', type=int, default=0, help='random seed, mainly for training samples.')
parser.add_argument('--total_train_steps', type=int, default=int(5e5), help='')
parser.add_argument('--n_eval_episodes',type=int, default= 20,help= '')
parser.add_argument('--n_train', type=int, default=int(1e6),help= '')
parser.add_argument("--gin_file", type=str, default=[], nargs='*', help = 'Paths to the gin-config files.')

parser.add_argument('--gin_bindings', type=str, default=[], nargs='*', help = 'Gin binding parameters.')
args = parser.parse_args()



def main(args):
  logging.set_verbosity(logging.INFO)
  
  # Setup data file path.
  data_dir = os.path.join(
      args.data_root_offlinerl_dir,
      args.env_name,
      args.data_name,
      args.data_sub_offlinerl_dir,
      )
  data_file = os.path.join(
      data_dir, args.data_file_name)
  logging.info('Data directory %s.', args.data_root_offlinerl_dir)
  # Setup log dir.
  if args.sub_dir == 'auto':
    sub_dir = get_datetime()
  else:
    sub_dir = args.sub_dir
  log_dir = os.path.join(
      args.data_root_offlinerl_dir,
      args.env_name,
      args.data_name,
      'n'+str(args.n_train),
      args.agent_name,
      sub_dir,
      str(args.seed),
      )
  if not os.path.exists(log_dir):
     os.makedirs(log_dir)
  else:
    pass
  
  train_eval_offline(
      log_dir=log_dir,
      data_file=data_file,
      env_name=args.env_name,
      n_train=args.n_train,
      total_train_steps=args.total_train_steps,
      n_eval_episodes=args.n_eval_episodes,
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      )


def Train_offline_brac(args):
    data_dir = 'offlinerl'
    #args.test_srcdir = os.getcwd()
    args.data_root_offlinerl_dir = os.path.join(args.test_srcdir, data_dir)
    args.data_sub_offlinerl_dir = '0'
    args.env_name = 'Pendulum-v0'
    args.data_name = 'example'
    args.agent_name = 'BRAC'
    args.gin_bindings = [
        'train_eval_offline.model_params=((200, 200),)',
        'train_eval_offline.optimizers=((5e-4, 0.5, 0.99),)']
    args.n_train = 100
    args.n_eval_episodes = 20
    args.total_train_steps = 100  # Short training.

    main(args)  # Just test that it runs.

if __name__ == "__main__":
  args = parser.parse_args(sys.argv[1:])
  gin.parse_config_files_and_bindings([], args.gin_bindings,finalize_config=False, skip_unknown=True, print_includes_and_imports=True)
  Train_offline_brac(args)
  gin.clear_config()
  gin.config._REGISTRY.clear()
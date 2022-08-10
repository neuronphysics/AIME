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

from dataset import Dataset, save_copy
from collect_data import DataCollector, env_factory
from planner_regularizer_debug import ActorNetwork, eval_policies, ContinuousRandomPolicy

@gin.configurable
def generate_dataset(
    # Basic args.
    log_dir,
    agent_module,
    env_name='HalfCheetah-v2',
    # Train and eval args.
    total_train_steps=int(1e6),
    summary_freq=100,
    print_freq=1000,
    save_freq=int(1e8),
    eval_freq=5000,
    n_eval_episodes=20,
    # For saving a partially trained policy.
    eval_target=None,  # Target return value to stop training.
    eval_target_n=2,  # Stop after n consecutive evals above eval_target.
    # Agent train args.
    initial_explore_steps=10000,
    replay_buffer_size=int(1e6),
    model_params=(((200, 200),), 2),
    optimizers=(('adam', 0.001),),
    batch_size=256,
    weight_decays=(0.0,),
    update_freq=1,
    update_rate=0.005,
    discount=0.99,
    ):
    """Generating the dataset with ActorNetwork."""
    # Create tf_env to get specs.
    tf_env = env_factory(env_name)
    tf_env_test = env_factory(env_name)
    observation_spec = tf_env.observation_spec()
    action_spec = tf_env.action_spec()

    # Initialize dataset.
    train_data = Dataset(
        observation_spec,
        action_spec,
        replay_buffer_size,
        circular=True,
    )
    # checkpoint
    data_ckpt = torch.save(train_data, os.path.join(log_dir, utils.get_datetime()))
    data_ckpt_name = os.path.join(log_dir, 'replay')

    time_st_total = time.time()
    time_st = time.time()
    timed_at_step = 0

    # Collect data from random policy.
    explore_policy = ContinuousRandomPolicy(action_spec)
    steps_collected = 0
    log_freq = 5000
    logging.info('Collecting data ...')
    collector = DataCollector(tf_env, explore_policy, train_data)
    while steps_collected < initial_explore_steps:
        count = collector.collect_transition()
        steps_collected += count
        if (steps_collected % log_freq == 0
            or steps_collected == initial_explore_steps) and count > 0:
            steps_per_sec = ((steps_collected - timed_at_step)
                       / (time.time() - time_st))
            timed_at_step = steps_collected
            time_st = time.time()
            logging.info('(%d/%d) steps collected at %.4g steps/s.', steps_collected,
                   initial_explore_steps, steps_per_sec)

  # Construct agent.
    actor = ActorNetwork(action_spec)

    # Prepare savers for models and results.
    train_summary_dir = os.path.join(log_dir, 'train')
    eval_summary_dir = os.path.join(log_dir, 'eval')
    # train_summary_writer = tf.compat.v2.summary.create_file_writer(
    #   train_summary_dir)
    # eval_summary_writers = collections.OrderedDict()
    # for policy_key in agent.test_policies.keys():
    #     eval_summary_writer = tf.compat.v2.summary.create_file_writer(
    #     os.path.join(eval_summary_dir, policy_key))
    # eval_summary_writers[policy_key] = eval_summary_writer
    actor_ckpt_name = os.path.join(log_dir, 'actor')
    eval_results = []

    # Train actor.
    logging.info('Start training ....')
    time_st = time.time()
    timed_at_step = 0
    target_partial_policy_saved = False
    collector = DataCollector(
        tf_env, None, train_data)
    step = 0
    for _ in range(total_train_steps):
        # collector.collect_transition()
        a_tanh_mode, a_sample, log_pi_a = actor.sample()
        # agent.train_step()
        # step = agent.global_step
        step += 1
        # if step % summary_freq == 0 or step == total_train_steps:
        #     utils.write_train_summary(train_summary_writer, step)
        # if step % print_freq == 0 or step == total_train_steps:
        #     print_train_info()

        if step % eval_freq == 0 or step == total_train_steps:
            time_ed = time.time()
            time_cost = time_ed - time_st
            logging.info(
                'Training at %.4g steps/s.', (step - timed_at_step) / time_cost)
            eval_result, eval_infos = eval_policies(
                tf_env_test, agent.test_policies, n_eval_episodes)
            eval_results.append([step] + eval_result)
            # Cecide whether to save a partially trained policy based on current model
            # performance.
            if (eval_target is not None and len(eval_results) >= eval_target_n
                and not target_partial_policy_saved):
                evals_ = list([eval_results[-(i + 1)][1]
                        for i in range(eval_target_n)])
                evals_ = np.array(evals_)
                if np.min(evals_) >= eval_target:
                    torch.save(actor_ckpt_name + '_partial_target')
                    save_copy(train_data, data_ckpt_name + '_partial_target')
                    logging.info('A partially trained policy was saved at step %d,'
                        ' with episodic return %.4g.', step, evals_[-1])
                    target_partial_policy_saved = True
        logging.info('Testing at step %d:', step)
        for policy_key, policy_info in eval_infos.items():
            logging.info(utils.get_summary_str(
                step=None, info=policy_info, prefix=policy_key + ': '))
            # utils.write_summary(eval_summary_writers[policy_key], step, policy_info)
            time_st = time.time()
            timed_at_step = step
        if step % save_freq == 0:
            torch.save(actor_ckpt_name + '-' + str(step))

    # Final save after training.
    torch.save(actor_ckpt_name + '_final')
    data_ckpt.write(data_ckpt_name + '_final')
    time_cost = time.time() - time_st_total
    logging.info('Training finished, time cost %.4gs.', time_cost)
    return np.array(eval_results)



parser = argparse.ArgumentParser(description='AIME')

parser.add_argument('--root_dir', type=str, default= os.path.join(os.getenv('HOME', '/'),
                                 'tmp/offlinerl/policies'), 
                                 help='Root directory for writing logs/summaries/checkpoints.')
parser.add_argument('--sub_dir', type=str, default='auto', help='sub directory for saving results.')

parser.add_argument('--agent_name', type=str, default='sac', help='agent name.')
parser.add_argument('--env_name', type=str, default='HalfCheetah-v2', help='env name.')
parser.add_argument('--env_loader', type=str, default='mujoco', help='env loader, suite/gym.')
parser.add_argument('--eval_target', type=int, default=1000, help='threshold for a paritally trained policy')

parser.add_argument('--seed', type=int, default=0, help='random seed, mainly for training samples.')
parser.add_argument('--total_train_steps', type=int, default=int(5e5))
parser.add_argument('--n_train', type=int, default=int(1e6))

parser.add_argument('--n_eval_episodes', type=int, default=20,
                        help='number episodes to eval each policy.')
parser.add_argument('--gin_file', type=str, default=None, help='Paths to the gin-config files.')
parser.add_argument('--gin_bindings', type=str, default=None, help='Gin binding parameters.')
# parser.add_argument('--hiddenflags', type=str, default='0', help='Hidden flags parameter.') 

args = parser.parse_args()


def main(_):
  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(args.gin_file, args.gin_bindings)
  if args.sub_dir == 'auto':
    sub_dir = utils.get_datetime()
  else:
    sub_dir = args.sub_dir
  log_dir = os.path.join(
      args.root_dir,
      args.env_name,
      args.agent_name,
      sub_dir,
      )
  utils.maybe_makedirs(log_dir)
  generate_dataset(
      log_dir=log_dir,
      agent_module=ActorNetwork,
      env_name=args.env_name,
      total_train_steps=args.total_train_steps,
      n_eval_episodes=args.n_eval_episodes,
      eval_target=args.eval_target,
      )


if __name__ == '__main__':
    main(None)
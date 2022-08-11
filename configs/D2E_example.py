"""Config file for collecting policy data without noise."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


model_params = (200, 200)

default_policy_root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../trained_policies')


def get_data_config(env_name, policy_root_dir=None):
  if not policy_root_dir:
    policy_root_dir = default_policy_root_dir
  ckpt_file = os.path.join(
      policy_root_dir,
      env_name,
      'agent_partial_target',
      )
  randwalk = ['randwalk', '', ['none'], ()]
  p1_pure = ['load', ckpt_file, ['none',], model_params]
  p1_gaussian = ['load', ckpt_file, ['gaussian', 0.1], model_params]
  data_config = [
      ['randwalk', randwalk, 2],
      ['p1_pure', p1_pure, 1],
      ['p1_gaussian', p1_gaussian, 4],
  ]
  return data_config
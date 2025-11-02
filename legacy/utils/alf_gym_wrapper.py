# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wrapper providing an AlfEnvironment adapter for GYM environments.
Adapted from TF-Agents Environment API as seen in:
    https://github.com/tensorflow/agents/blob/master/tf_agents/environments/suite_gym.py
"""

import collections
import gym
import gym.spaces
import numbers

import numpy
import numpy as np
import torch
from tensor_utils import tensor_spec_from_gym_space, _as_array
import data_structures as ds
from alf_environment import AlfEnvironment
import nest
from tensor_specs import TensorSpec, BoundedTensorSpec, torch_dtype_to_str
from common import configurable
from collections import deque, OrderedDict
def _gym_space_to_nested_space(space):
    """Change gym Space to a nest which can be handled by alf.nest functions."""

    if isinstance(space, gym.spaces.Dict):
        return dict((k, _gym_space_to_nested_space(s))
                    for k, s in space.spaces.items())
    elif isinstance(space, gym.spaces.Tuple):
        return tuple(_gym_space_to_nested_space(s) for s in space.spaces)
    else:
        return space


def _nested_space_to_gym_space(space):
    """Change nested space to gym Space"""

    if isinstance(space, (dict, OrderedDict)):
        spaces = dict(
            (k, _nested_space_to_gym_space(s)) for k, s in space.items())
        return gym.spaces.Dict(spaces)
    elif isinstance(space, tuple):
        spaces = tuple(_nested_space_to_gym_space(s) for s in space)
        return gym.spaces.Tuple(spaces)
    else:
        return space

@configurable
class ContinuousActionMapping(gym.ActionWrapper):
    """Map continuous actions to a desired action space, while keeping discrete
    actions unchanged."""

    def __init__(self, env, low, high):
        """
        Args:
            env (gym.Env): Gym env to be wrapped
            low (float): the action lower bound to map to.
            high (float): the action higher bound to map to.
        """
        super(ContinuousActionMapping, self).__init__(env)

        def _space_bounds(space):
            if isinstance(space, gym.spaces.Box):
                assert np.all(np.isfinite(space.low))
                assert np.all(np.isfinite(space.high))
                return (space.low, space.high)

        nested_action_space = _gym_space_to_nested_space(self.action_space)
        self._bounds = nest.map_structure(_space_bounds,
                                              nested_action_space)
        self._nested_action_space = nest.map_structure(
            lambda space: (gym.spaces.Box(
                low=low, high=high, shape=space.shape, dtype=space.dtype)
                           if isinstance(space, gym.spaces.Box) else space),
            nested_action_space)
        self.action_space = _nested_space_to_gym_space(
            self._nested_action_space)

    def action(self, action):
        def _scale_back(a, b, space):
            if isinstance(space, gym.spaces.Box):
                # a and b should be mutually broadcastable
                b0, b1 = b
                a0, a1 = space.low, space.high
                return (a - a0) / (a1 - a0) * (b1 - b0) + b0
            return a

        # map action back to its original space
        action = nest.map_structure_up_to(action, _scale_back, action,
                                              self._bounds,
                                              self._nested_action_space)
        return action


class AlfGymWrapper(AlfEnvironment):
    """Base wrapper implementing AlfEnvironmentBaseWrapper interface for Gym envs.
    Action and observation specs are automatically generated from the action and
    observation spaces. See base class for ``AlfEnvironment`` details.
    """

    def __init__(self,
                 gym_env,
                 env_id=None,
                 discount=1.0,
                 auto_reset=True,
                 simplify_box_bounds=True):
        """
        Args:
            gym_env (gym.Env): An instance of OpenAI gym environment.
            env_id (int): (optional) ID of the environment.
            discount (float): Discount to use for the environment.
            auto_reset (bool): whether or not to reset the environment when done.
            simplify_box_bounds (bool): whether or not to simplify redundant
                arrays to values for spec bounds.
        """
        super(AlfGymWrapper, self).__init__()

        self._gym_env = gym_env
        self._discount = discount
        if env_id is None:
            env_id = 0
        self._env_id = np.int32(env_id)
        self._action_is_discrete = isinstance(self._gym_env.action_space,
                                              gym.spaces.Discrete)
        # TODO: Add test for auto_reset param.
        self._auto_reset = auto_reset
        self._observation_spec = tensor_spec_from_gym_space(
            self._gym_env.observation_space, simplify_box_bounds)
        self._action_spec = tensor_spec_from_gym_space(
            self._gym_env.action_space, simplify_box_bounds)
        if hasattr(self._gym_env, "reward_space"):
            self._reward_spec = tensor_spec_from_gym_space(
                self._gym_env.reward_space, simplify_box_bounds)
        else:
            self._reward_spec = TensorSpec(())
        self._done = True

        self._time_step_spec = ds.time_step_spec(
            self._observation_spec, self._action_spec, self._reward_spec, self._done)
        self._info = None

        self._zero_info = self._obtain_zero_info()

        self._env_info_spec = nest.map_structure(TensorSpec.from_array,
                                                 self._zero_info)

    @property
    def gym(self):
        """Return the gym environment. """
        return self._gym_env

    def _obtain_zero_info(self):
        """Get an env info of zeros only once when the env is created.
        This info will be filled in each ``FIRST`` time step as a placeholder.
        """
        self._gym_env.reset()
        action = nest.map_structure(lambda spec: spec.numpy_zeros(),
                                    self._action_spec)
        info = self._gym_env.step(action)[-1]
        self._gym_env.reset()
        info = _as_array(info)
        return nest.map_structure(lambda a: np.zeros_like(a), info)

    def __getattr__(self, name):
        """Forward all other calls to the base environment."""
        return getattr(self._gym_env, name)

    def get_info(self):
        """Returns the gym environment info returned on the last step."""
        return self._info

    def _reset(self):
        # TODO: Upcoming update on gym adds **kwargs on reset. Update this to
        # support that.
        res = self._gym_env.reset()
        info = None
        if isinstance(res, numpy.ndarray):
            observation = res
        elif isinstance(res, tuple):
            observation, info = res
        self._info = info
        self._done = False

        observation = self._to_spec_dtype_observation(observation)
        return ds.restart(
            observation=observation,
            action_spec=self._action_spec,
            done=self._done,
            reward_spec=self._reward_spec,
            env_id=self._env_id,
            env_info=self._zero_info)

    @property
    def done(self):
        return self._done

    def _step(self, action):
        # Automatically reset the environments on step if they need to be reset.
        if self._auto_reset and self._done:
            return self.reset()

        res = self._gym_env.step(action)
        if len(res) == 4:
            observation, reward, self._done, self._info = res
        else:
            observation, reward, self._done, truncated, self._info = res
        observation = self._to_spec_dtype_observation(observation)
        self._info = nest.map_structure(_as_array, self._info)

        if self._done:
            return ds.termination(
                observation,
                action,
                reward,
                self._done,
                self._reward_spec,
                self._env_id,
                env_info=self._info)
        else:
            return ds.transition(
                observation,
                action,
                reward,
                self._done,
                self._reward_spec,
                self._discount,
                self._env_id,
                env_info=self._info)

    def _to_spec_dtype_observation(self, observation):
        """Make sure observation from env is converted to the correct dtype.
        Args:
            observation (nested arrays or tensors): observations from env.
        Returns:
            A (nested) arrays of observation
        """

        def _as_spec_dtype(arr, spec):
            dtype = torch_dtype_to_str(spec.dtype)
            if str(arr.dtype) == dtype:
                return arr
            else:
                return arr.astype(dtype)

        return nest.map_structure(_as_spec_dtype, observation,
                                  self._observation_spec)

    def env_info_spec(self):
        return self._env_info_spec

    def time_step_spec(self):
        return self._time_step_spec

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def reward_spec(self):
        return self._reward_spec

    def done_spec(self):
        done_spec = TensorSpec((1,), dtype=torch.bool)
        if self._done:
            return done_spec.ones()
        else:
            return done_spec.zeros()

    def close(self):
        return self._gym_env.close()

    def seed(self, seed):
        return self._gym_env.seed(seed)

    def render(self, mode='rgb_array'):
        return self._gym_env.render(mode)

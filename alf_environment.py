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
"""ALF RL Environment API.
Adapted from TF-Agents Environment API as seen in:
    https://github.com/tensorflow/agents/blob/master/tf_agents/environments/py_environment.py
    https://github.com/tensorflow/agents/blob/master/tf_agents/environments/tf_environment.py
"""

import abc
import six
import torch
from data_structures import time_step_spec, StepType, _is_numpy_array, termination, transition, restart
from tensor_specs import TensorSpec, BoundedTensorSpec, torch_dtype_to_str
from collections import deque
import numpy as np
import nest
import gym
from typing import List
from collections import OrderedDict
from tensor_utils import _as_array, tensor_spec_from_gym_space
from alf_environment_base import AlfEnvironment

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
        self._time_step_spec = time_step_spec(
            self._observation_spec, self._action_spec, self._reward_spec)
        self._info = None
        self._done = True
        self._zero_info = self._obtain_zero_info()

        self._env_info_spec = nest.map_structure(TensorSpec.from_array,
                                                 self._zero_info)

    @property
    def gym(self):
        """Return the gym environment. """
        return self._gym_env

    @property
    def is_tensor_based(self):
        return False

    def _obtain_zero_info(self):
        """Get an env info of zeros only once when the env is created.
        This info will be filled in each ``FIRST`` time step as a placeholder.
        """
        self._gym_env.reset()
        action = nest.map_structure(lambda spec: spec.numpy_zeros(),
                                    self._action_spec)
        _, _, _, info = self._gym_env.step(action)
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
        observation = self._gym_env.reset()
        self._info = None
        self._done = False

        observation = self._to_spec_dtype_observation(observation)
        return restart(
            observation=observation,
            action_spec=self._action_spec,
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

        observation, reward, self._done, self._info = self._gym_env.step(
            action)
        # NOTE: In recent version of gym, the environment info may have
        # "TimeLimit.truncated" to indicate that the env has run beyond the time
        # limit. If so, it will removed to avoid having conflict with our env
        # info spec.
        self._info.pop("TimeLimit.truncated", None)
        observation = self._to_spec_dtype_observation(observation)
        self._info = _as_array(self._info)

        if self._done:
            return termination(
                observation,
                action,
                reward,
                self._reward_spec,
                self._env_id,
                env_info=self._info)
        else:
            return transition(
                observation,
                action,
                reward,
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

    def close(self):
        return self._gym_env.close()

    def seed(self, seed):
        return self._gym_env.seed(seed)

    def render(self, mode='rgb_array'):
        return self._gym_env.render(mode)

class AlfEnvironmentBaseWrapper(AlfEnvironment):
    """AlfEnvironment wrapper forwards calls to the given environment."""

    def __init__(self, env):
        """Create an ALF environment base wrapper.
        Args:
            env (AlfEnvironment): An AlfEnvironment instance to wrap.
        Returns:
            A wrapped AlfEnvironment
        """
        super(AlfEnvironmentBaseWrapper, self).__init__()
        self._env = env

    def __getattr__(self, name):
        """Forward all other calls to the base environment."""
        if name.startswith('_'):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name))
        return getattr(self._env, name)

    @property
    def batched(self):
        return self._env.batched

    @property
    def batch_size(self):
        return self._env.batch_size

    @property
    def num_tasks(self):
        return self._env.num_tasks

    @property
    def task_names(self):
        return self._env.task_names

    def _reset(self):
        return self._env.reset()

    def _step(self, action):
        return self._env.step(action)

    def get_info(self):
        return self._env.get_info()

    def env_info_spec(self):
        return self._env.env_info_spec()

    def time_step_spec(self):
        return self._env.time_step_spec()

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reward_spec(self):
        return self._env.reward_spec()

    def done_spec(self):
        return self._env.done_spec()

    def close(self):
        return self._env.close()

    def render(self, mode='rgb_array'):
        return self._env.render(mode)

    def seed(self, seed):
        return self._env.seed(seed)

    def wrapped_env(self):
        return self._env

class BatchedTensorWrapper(AlfEnvironmentBaseWrapper):
    """Wrapper that converts non-batched numpy-based I/O to batched tensors.
    """

    def __init__(self, env):
        assert not env.batched, (
            'BatchedTensorWrapper can only be used to wrap non-batched env')
        super().__init__(env)

    @property
    def is_tensor_based(self):
        return True

    @property
    def batched(self):
        return True

    @staticmethod
    def _to_batched_tensor(raw):
        """Convert the structured input into batched (batch_size = 1) tensors
        of the same structure.
        """
        return nest.map_structure(
            lambda x: (torch.as_tensor(x).unsqueeze(dim=0) if isinstance(
                x, (np.ndarray, np.number, float, int)) else x), raw)

    def _step(self, action):
        numpy_action = nest.map_structure(
            lambda x: x.squeeze(dim=0).cpu().numpy(), action)
        return BatchedTensorWrapper._to_batched_tensor(
            super()._step(numpy_action))

    def _reset(self):
        return BatchedTensorWrapper._to_batched_tensor(super()._reset())


class TensorWrapper(AlfEnvironmentBaseWrapper):
    """Wrapper that converts numpy-based I/O to tensors.
    """

    def __init__(self, env):
        assert env.batched, (
            'TensorWrapper can only be used to wrap batched env')
        super().__init__(env)

    @property
    def is_tensor_based(self):
        return True

    @staticmethod
    def _to_tensor(raw):
        """Convert the structured input into batched (batch_size = 1) tensors
        of the same structure.
        """
        return nest.map_structure(
            lambda x: (torch.as_tensor(x) if isinstance(
                x, (np.ndarray, np.number, float, int)) else x), raw)

    def _step(self, action):
        numpy_action = nest.map_structure(lambda x: x.cpu().numpy(), action)
        return TensorWrapper._to_tensor(super()._step(numpy_action))

    def _reset(self):
        return TensorWrapper._to_tensor(super()._reset())

class FrameStackWrapper(AlfEnvironmentBaseWrapper):
    #changed https://github.com/medric49/imitation-from-observation/blob/faae84a0bbc527eb428ac3c2be9aabb210e56367/dmc.py
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]

        self._obs_spec = BoundedTensorSpec(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()
    
    def done_spec(self):
        return self._env.done_spec()    

    def __getattr__(self, name):
        return getattr(self._env, name)

class ActionRepeatWrapper(AlfEnvironmentBaseWrapper):
    def __init__(self, env, num_repeats, discount = 1.0):
        self._env = env
        self._num_repeats = num_repeats
        self._discount = discount

    def step(self, action):
        reward = 0.0
        discount = self._discount
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()
    
    def done_spec(self):
        return self._env.done_spec()   

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)

class MultitaskWrapper(AlfEnvironment):
    """Multitask environment based on a list of environments.

    All the environments need to have same observation_spec, action_spec, reward_spec
    and info_spec. The action_spec of the new environment becomes:

    .. code-block:: python

        {
            'task_id': TensorSpec((), maximum=num_envs - 1, dtype='int64'),
            'action': original_action_spec
        }

    'task_id' is used to specify which task to run for the current step. Note
    that current implementation does not prevent switching task in the middle of
    one episode.
    """

    def __init__(self, envs, task_names, env_id=None):
        """
        Args:
            envs (list[AlfEnvironment]): a list of environments. Each one
                represents a different task.
            task_names (list[str]): the names of each task.
            env_id (int): (optional) ID of the environment.
        """
        assert len(envs) > 0, "`envs should not be empty"
        assert len(set(task_names)) == len(task_names), (
            "task_names should "
            "not contain duplicated names: %s" % str(task_names))
        self._envs = envs
        self._observation_spec = envs[0].observation_spec()
        self._action_spec = envs[0].action_spec()
        self._reward_spec = envs[0].reward_spec()
        self._env_info_spec = envs[0].env_info_spec()
        self._task_names = task_names
        if env_id is None:
            env_id = 0
        self._env_id = np.int32(env_id)

        def _nested_eq(a, b):
            return all(
                nest.flatten(
                    nest.map_structure(lambda x, y: x == y, a, b)))

        for env in envs:
            assert _nested_eq(
                env.observation_spec(), self._observation_spec), (
                    "All environment should have same observation spec. "
                    "Got %s vs %s" % (self._observation_spec,
                                      env.observation_spec()))
            assert _nested_eq(env.action_spec(), self._action_spec), (
                "All environment should have same action spec. "
                "Got %s vs %s" % (self._action_spec, env.action_spec()))
            assert _nested_eq(env.reward_spec(), self._reward_spec), (
                "All environment should have same reward spec. "
                "Got %s vs %s" % (self._reward_spec, env.reward_spec()))
            assert _nested_eq(env.env_info_spec(), self._env_info_spec), (
                "All environment should have same env_info spec. "
                "Got %s vs %s" % (self._env_info_spec, env.env_info_spec()))
            env.reset()

        self._current_env_id = np.int64(0)
        self._action_spec = OrderedDict(
            task_id=BoundedTensorSpec((),
                                          maximum=len(envs) - 1,
                                          dtype='int64'),
            action=self._action_spec)

    @staticmethod
    def load(load_fn, environment_name, env_id=None, **kwargs):
        """
        Args:
            load_fn (Callable): function used to construct the environment for
                each tasks. It will be called as ``load_fn(env_name, **kwargs)``
            environment_name (list[str]): list of environment names
            env_id (int): (optional) ID of the environment.
            kwargs (**): arguments passed to load_fn
        """
        # TODO: may need to add the option of using ProcessEnvironment to wrap
        # the underlying environment
        envs = []
        for name in environment_name:
            envs.append(load_fn(name, **kwargs))
        return MultitaskWrapper(envs, environment_name, env_id)

    @property
    def num_tasks(self):
        return len(self._envs)

    @property
    def task_names(self):
        return self._task_names

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def reward_spec(self):
        return self._reward_spec

    def env_info_spec(self):
        return self._env_info_spec

    def get_num_tasks(self):
        return len(self._envs)

    def _reset(self):
        time_step = self._envs[self._current_env_id].reset()
        return time_step._replace(
            env_id=self._env_id,
            prev_action=OrderedDict(
                task_id=self._current_env_id, action=time_step.prev_action))

    def _step(self, action):
        self._current_env_id = action['task_id']
        action = action['action']
        assert self._current_env_id < len(self._envs)
        time_step = self._envs[self._current_env_id].step(action)
        return time_step._replace(
            env_id=self._env_id,
            prev_action=OrderedDict(
                task_id=self._current_env_id, action=time_step.prev_action))

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name))
        return getattr(self._envs[self._current_env_id], name)

    def seed(self, seed):
        for env in self._envs:
            env.seed(seed)
class BatchEnvironmentWrapper(AlfEnvironment):
    """Wrapper to make a list of non-batched environment into a batched environment.

    Note the individual environments in ``envs`` are executed sequentially doring
    one ``step()`` of ``reset()``.

    Args:
        envs: a list of unbatched ``AlfEnvironment``.
    """

    def __init__(self, envs: List[AlfEnvironment]):
        self._envs = envs
        super().__init__()
        self._action_spec = self._envs[0].action_spec()
        self._observation_spec = self._envs[0].observation_spec()
        self._reward_spec = self._envs[0].reward_spec()
        self._time_step_spec = self._envs[0].time_step_spec()
        self._env_info_spec = self._envs[0].env_info_spec()
        self._num_tasks = self._envs[0].num_tasks
        self._task_names = self._envs[0].task_names
        if any(env.is_tensor_based for env in self._envs):
            raise ValueError('All environments must be array-based.')
        if any(env.action_spec() != self._action_spec for env in self._envs):
            raise ValueError(
                'All environments must have the same action spec.')
        if any(env.time_step_spec() != self._time_step_spec
               for env in self._envs):
            raise ValueError(
                'All environments must have the same time_step_spec.')
        if any(env.env_info_spec() != self._env_info_spec
               for env in self._envs):
            raise ValueError(
                'All environments must have the same env_info_spec.')
        if any(env.batched for env in self._envs):
            raise ValueError('All environments must be non-batched.')

    @property
    def metadata(self):
        return self._envs[0].metadata

    @property
    def is_tensor_based(self):
        return False

    @property
    def batched(self):
        return True

    @property
    def batch_size(self):
        return len(self._envs)

    @property
    def num_tasks(self):
        return self._num_tasks

    @property
    def task_names(self):
        return self._task_names

    def env_info_spec(self):
        return self._env_info_spec

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def reward_spec(self):
        return self._reward_spec

    def time_step_spec(self):
        return self._time_step_spec

    def close(self):
        for env in self._envs:
            env.close()

    def render(self, mode):
        return self._envs[0].render(mode)

    def seed(self, seed):
        for i, env in enumerate(self._envs):
            env.seed(seed * len(self._envs) + i)

    def _step(self, action):
        def _ith(i):
            return nest.map_structure(lambda x: x[i], action)

        time_steps = [env.step(_ith(i)) for i, env in enumerate(self._envs)]
        time_step = nest.map_structure(lambda *arrays: np.stack(arrays),
                                           *time_steps)
        return time_step

    def _reset(self):
        time_steps = [env.reset() for env in self._envs]
        time_step = nest.map_structure(lambda *arrays: np.stack(arrays),
                                           *time_steps)
        return time_step
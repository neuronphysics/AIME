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
from alf_gym_wrapper import _as_array, tensor_spec_from_gym_space
from common import configurable
###
@six.add_metaclass(abc.ABCMeta)
class AlfEnvironment(object):
    """Abstract base class for ALF RL environments.
    Observations and valid actions are described with ``TensorSpec``, defined in
    the ``specs`` module.
    The ``current_time_step()`` method returns current ``time_step``, resetting the
    environment if necessary.
    The ``step(action)`` method applies the action and returns the new ``time_step``.
    This method will also reset the environment if needed and ignore the action in
    that case.
    The ``reset()`` method returns ``time_step`` that results from an environment
    reset and is guaranteed to have ``step_type=ts.FIRST``.
    The ``reset()`` method is only needed for explicit resets. In general, the
    environment will reset automatically when needed, for example, when no
    episode was started or when it reaches a step after the end of the episode
    (i.e. ``step_type=ts.LAST``).
    If the environment can run multiple steps at the same time and take a batched
    set of actions and return a batched set of observations, it should overwrite
    the property batched to True.
    Example for collecting an episode:
    .. code-block:: python
        env = AlfEnvironment()
        # reset() creates the initial time_step and resets the environment.
        time_step = env.reset()
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
    """

    def __init__(self):
        self._current_time_step = None

    @property
    def num_tasks(self):
        """Number of tasks supported by this environment."""
        return 1

    @property
    def task_names(self):
        """The name of each tasks."""
        return [str(i) for i in range(self.num_tasks)]

    @property
    def batched(self):
        """Whether the environment is batched or not.
        If the environment supports batched observations and actions, then overwrite
        this property to True.
        A batched environment takes in a batched set of actions and returns a
        batched set of observations. This means for all numpy arrays in the input
        and output nested structures, the first dimension is the batch size.
        When batched, the left-most dimension is not part of the action_spec
        or the observation_spec and corresponds to the batch dimension.
        Returns:
            A boolean indicating whether the environment is batched or not.
        """
        return False

    @property
    def batch_size(self):
        """The batch size of the environment.
        Returns:
            The batch size of the environment, or 1 if the environment is not
            batched.
        Raises:
          RuntimeError: If a subclass overrode batched to return True but did not
              override the ``batch_size`` property.
        """
        if self.batched:
            raise RuntimeError(
                'Environment %s marked itself as batched but did not override the '
                'batch_size property' % type(self))
        return 1

    @abc.abstractmethod
    def env_info_spec(self):
        """Defines the env_info provided by the environment."""

    @abc.abstractmethod
    def observation_spec(self):
        """Defines the observations provided by the environment.
        May use a subclass of ``TensorSpec`` that specifies additional properties such
        as min and max bounds on the values.
        Returns:
            nested TensorSpec
        """

    @abc.abstractmethod
    def action_spec(self):
        """Defines the actions that should be provided to ``step()``.
        May use a subclass of ``TensorSpec`` that specifies additional properties such
        as min and max bounds on the values.
        Returns:
            nested TensorSpec
        """

    def reward_spec(self):
        """Defines the reward provided by the environment.
        The reward of the most environments is a scalar. So we provide a default
        implementation which returns a scalar spec.
        Returns:
            alf.TensorSpec
        """
        return TensorSpec(())

    def done_spec(self):
        """Defines the done provided by the environment.
        The done of the most environments is a boolean scalar. So we provide a default
        implementation which returns a scalar spec.
        Returns:
            alf.TensorSpec
        """
        return TensorSpec(())

    def time_step_spec(self):
        """Describes the ``TimeStep`` fields returned by ``step()``.
        Override this method to define an environment that uses non-standard values
        for any of the items returned by ``step()``. For example, an environment with
        tensor-valued rewards.
        Returns:
            A ``TimeStep`` namedtuple containing (possibly nested) ``TensorSpec`` defining
            the step_type, reward, discount, observation, prev_action, and end_id.
        """
        return time_step_spec(self.observation_spec(), self.action_spec(),
                              self.reward_spec(), self.done_spec())

    def current_time_step(self):
        """Returns the current timestep."""
        return self._current_time_step

    def reset(self):
        """Starts a new sequence and returns the first ``TimeStep`` of this sequence.
        Note: Subclasses cannot override this directly. Subclasses implement
        ``_reset()`` which will be called by this method. The output of ``_reset()``
        will be cached and made available through ``current_time_step()``.
        Returns:
            TimeStep:
        """
        self._current_time_step = self._reset()
        return self._current_time_step

    def step(self, action):
        """Updates the environment according to the action and returns a ``TimeStep``.
        If the environment returned a ``TimeStep`` with ``StepType.LAST`` at the
        previous step the implementation of ``_step`` in the environment should call
        ``reset`` to start a new sequence and ignore ``action``.
        This method will start a new sequence if called after the environment
        has been constructed and ``reset`` has not been called. In this case
        ``action`` will be ignored.
        Note: Subclasses cannot override this directly. Subclasses implement
        ``_step()`` which will be called by this method. The output of ``_step()``
        will be cached and made available through ``current_time_step()``.
        Args:
            action (nested Tensor): input actions.
        Returns:
            TimeStep:
        """
        if self._current_time_step is None:
            return self.reset()

        self._current_time_step = self._step(action)
        return self._current_time_step

    def close(self):
        """Frees any resources used by the environment.
        Implement this method for an environment backed by an external process.
        This method can be used directly:
        .. code-block:: python
            env = Env(...)
            # Use env.
            env.close()
        or via a context manager:
        .. code-block:: python
            with Env(...) as env:
            # Use env.
        """
        pass

    def __enter__(self):
        """Allows the environment to be used in a with-statement context."""
        return self

    def __exit__(self, unused_exception_type, unused_exc_value,
                 unused_traceback):
        """Allows the environment to be used in a with-statement context."""
        self.close()

    def render(self, mode='rgb_array'):
        """Renders the environment.
        Args:
            mode: One of ['rgb_array', 'human']. Renders to an numpy array, or brings
                up a window where the environment can be visualized.
        Returns:
            An ndarray of shape ``[width, height, 3]`` denoting an RGB image if mode is
            ``rgb_array``. Otherwise return nothing and render directly to a display
            window.
        Raises:
            NotImplementedError: If the environment does not support rendering.
        """
        del mode  # unused
        raise NotImplementedError('No rendering support.')

    def seed(self, seed):
        """Seeds the environment.
        Args:
            seed (int): Value to use as seed for the environment.
        """
        del seed  # unused
        raise NotImplementedError('No seed support for this environment.')

    def get_info(self):
        """Returns the environment info returned on the last step.
        Returns:
            Info returned by last call to ``step()``. None by default.
        Raises:
            NotImplementedError: If the environment does not use info.
        """
        raise NotImplementedError(
            'No support of get_info for this environment.')

    #  These methods are to be implemented by subclasses:

    @abc.abstractmethod
    def _step(self, action):
        """Updates the environment according to action and returns a ``TimeStep``.
        See ``step(self, action)`` docstring for more details.
        Args:
            action: A tensor, or a nested dict, list or tuple of tensors
                corresponding to ``action_spec()``.
        """

    @abc.abstractmethod
    def _reset(self):
        """Starts a new sequence, returns the first ``TimeStep`` of this sequence.
        See ``reset(self)`` docstring for more details
        """


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
# Used in ALF
@configurable
class TimeLimit(AlfEnvironmentBaseWrapper):
    """End episodes after specified number of steps."""

    def __init__(self, env, duration):
        """Create a TimeLimit ALF environment.
        Args:
            env (AlfEnvironment): An AlfEnvironment instance to wrap.
            duration (int): time limit, usually set to be the max_eposode_steps
                of the environment.
        """
        super(TimeLimit, self).__init__(env)
        self._duration = duration
        self._num_steps = None
        assert self.batch_size is None or self.batch_size == 1, (
            "does not support batched environment with batch size larger than one"
        )

    def _reset(self):
        self._num_steps = 0
        return self._env.reset()

    def _step(self, action):
        if self._num_steps is None:
            return self.reset()

        time_step = self._env.step(action)

        self._num_steps += 1
        if self._num_steps >= self._duration:
            if _is_numpy_array(time_step.step_type):
                time_step = time_step._replace(step_type=StepType.LAST)
            else:
                time_step = time_step._replace(
                    step_type=torch.full_like(time_step.step_type, StepType.
                                              LAST))

        if time_step.is_last():
            self._num_steps = None

        return time_step

    @property
    def duration(self):
        return self._duration

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
from __future__ import annotations
from common import configurable
from data_structures import StepType, _is_numpy_array
from alf_environment import AlfEnvironmentBaseWrapper
import torch
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

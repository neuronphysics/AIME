"""Meta-World (Farama) wrapper exposing the same interface as envs/dmc.py.

Design notes
------------
* The repo's rollout loop (``tools.simulate``) speaks the *old* gym 4-tuple API
  (``obs, reward, done, info``).  Meta-World is gymnasium-native and returns a
  5-tuple, so the conversion happens here.
* Meta-World has **no terminal state** in the MDP sense: an episode ends by time
  limit, and reaching ``success`` does not end it.  ``is_terminal`` is therefore
  always ``False`` (exactly like DMC), so the continuation head is not taught a
  spurious absorbing state.  Set ``terminate_on_success=True`` only if you
  deliberately want the easier early-termination variant -- it changes the
  benchmark and makes numbers non-comparable to published baselines.
* Success is emitted as ``log_success``.  ``tools.simulate`` strips every key
  containing ``log_`` before the agent sees it and *sums* it at episode end, so
  the flag is raised exactly once (the first success step).  The episode metric
  is then "did this episode ever succeed", which is the standard Meta-World
  success rate.
* ``_freeze_rand_vec`` defaults to ``True`` in the goal-observable classes,
  which silently pins the goal to a single position for the whole run.  That is
  the *easy* single-goal variant.  We default to randomised goals per episode
  (the harder variant used by TD-MPC / TD-MPC2); flip ``randomize_goal=False``
  if you are reproducing a paper that used the fixed-goal setting.
"""

import gym
import numpy as np

# 'corner2' is the camera used by MWM / TD-MPC2 for pixel Meta-World.
DEFAULT_CAMERA = "corner2"
# TD-MPC2's camera nudge so the whole workspace fits in a 64x64 frame.
CAMERA_POS_OVERRIDE = {"corner2": [0.75, 0.075, 0.7]}


def _normalize_task(name):
    """'pick_place' | 'pick-place' | 'pick-place-v3' -> 'pick-place-v3'."""
    name = name.replace("_", "-").strip().lower()
    head, _, tail = name.rpartition("-")
    if head and tail.startswith("v") and tail[1:].isdigit():
        return name
    return name + "-v3"


def _lookup(task):
    """Return (env_cls, resolved_name). Prefers v3, falls back to v2."""
    import metaworld

    for version, attr in (("v3", "ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE"),
                          ("v2", "ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE")):
        registry = getattr(metaworld, attr, None)
        if registry is None:
            # Older layouts kept the dicts under metaworld.envs.
            registry = getattr(getattr(metaworld, "envs", None), attr, None)
        if registry is None:
            continue
        candidate = task if task.endswith(f"-{version}") else \
            task.rsplit("-", 1)[0] + f"-{version}"
        key = f"{candidate}-goal-observable"
        if key in registry:
            return registry[key], candidate
    raise ValueError(
        f"Unknown Meta-World task '{task}'. Run "
        f"`python -c \"import metaworld; print(sorted(metaworld."
        f"ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE))\"` for the 50 valid names."
    )


class MetaWorld:
    metadata = {}

    def __init__(
        self,
        name,
        action_repeat=2,
        size=(64, 64),
        seed=0,
        camera=DEFAULT_CAMERA,
        render=True,
        randomize_goal=True,
        terminate_on_success=False,
    ):
        task = _normalize_task(name)
        env_cls, self._task_name = _lookup(task)

        self._env = env_cls(seed=seed, render_mode="rgb_array" if render else None)
        # Per-episode goal randomisation (see module docstring).
        self._env._freeze_rand_vec = not randomize_goal
        if hasattr(self._env, "seeded_rand_vec"):
            self._env.seeded_rand_vec = True

        self._action_repeat = action_repeat
        self._size = tuple(size)
        self._render = render
        self._camera = camera
        self._terminate_on_success = terminate_on_success
        self._succeeded = False
        self.reward_range = [-np.inf, np.inf]

        if render:
            self._setup_renderer()

    # ------------------------------------------------------------------ setup

    def _setup_renderer(self):
        """Rebuild the renderer at our resolution and camera.

        gymnasium's MujocoRenderer fixes camera and size at construction time and
        ``render()`` takes no camera argument, so the only reliable way to
        override both is to swap the renderer object.
        """
        import mujoco
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        model = self._env.model
        if self._camera in CAMERA_POS_OVERRIDE:
            cam_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, self._camera
            )
            if cam_id >= 0:
                model.cam_pos[cam_id] = CAMERA_POS_OVERRIDE[self._camera]

        try:
            self._env.mujoco_renderer.close()
        except Exception:
            pass
        self._env.mujoco_renderer = MujocoRenderer(
            model,
            self._env.data,
            width=self._size[0],
            height=self._size[1],
            camera_name=self._camera,
        )

    # ----------------------------------------------------------------- spaces

    @property
    def observation_space(self):
        low = np.full(self._env.observation_space.shape, -np.inf, np.float32)
        spaces = {
            "state": gym.spaces.Box(low, -low, dtype=np.float32),
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            # log_ keys are stripped before the encoder; declared for clarity.
            "log_success": gym.spaces.Box(0, 1, (1,), dtype=np.float32),
        }
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        space = self._env.action_space
        return gym.spaces.Box(
            space.low.astype(np.float32), space.high.astype(np.float32),
            dtype=np.float32,
        )

    # -------------------------------------------------------------- rollout

    def _obs(self, state, is_first, success):
        return {
            "state": np.asarray(state, np.float32),
            "image": self.render(),
            "is_first": is_first,
            "is_terminal": False,  # Meta-World has no absorbing state.
            "log_success": np.float32(success),
        }

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0.0
        success_now = False
        for _ in range(self._action_repeat):
            state, r, terminated, truncated, info = self._env.step(action)
            reward += float(r)
            if float(info.get("success", 0.0)) > 0.5:
                success_now = True
            if terminated or truncated:
                break

        # Raise the flag exactly once so the episode sum is a 0/1 indicator.
        first_success = success_now and not self._succeeded
        self._succeeded = self._succeeded or success_now

        done = bool(terminated or truncated)
        if self._terminate_on_success and self._succeeded:
            done = True

        obs = self._obs(state, is_first=False, success=first_success)
        return obs, reward, done, {"discount": np.float32(1.0)}

    def reset(self):
        self._succeeded = False
        state, _ = self._env.reset()
        return self._obs(state, is_first=True, success=False)

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        if not self._render:
            # Proprio-only runs: models.preprocess unconditionally divides
            # obs['image'] by 255, so a 1x1 placeholder keeps the pipeline happy
            # without paying render cost or replay storage.
            return np.zeros((1, 1, 3), np.uint8)
        return self._env.mujoco_renderer.render("rgb_array")

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

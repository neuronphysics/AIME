"""CARL contextual dm_control -> dreamerv3-torch env contract (pixels).
Tasks: dmc_walker | dmc_quadruped | dmc_finger | dmc_fish.
Obs keys: image, state, context, is_first, is_terminal."""
import itertools

import gym
import numpy as np


class CARL:
    metadata = {}

    _CLASSES = {
        "dmc_walker": "CARLDmcWalkerEnv",
        "dmc_quadruped": "CARLDmcQuadrupedEnv",
        "dmc_finger": "CARLDmcFingerEnv",
        "dmc_fish": "CARLDmcFishEnv",
    }
    _SELECTORS = {"round_robin": "RoundRobinSelector", "random": "RandomSelector", "static": "StaticSelector"}

    def __init__(
        self, name, action_repeat=1, size=(64, 64), camera=None, seed=0, contexts=None, selector="round_robin", start=0
    ):
        import carl.envs as carl_envs
        from carl.context import selection

        if name not in self._CLASSES:
            raise NotImplementedError(f"CARL task {name!r}; known: {sorted(self._CLASSES)}")
        cls = getattr(carl_envs, self._CLASSES[name])
        self._contexts = self.build_contexts(cls, contexts)
        self._rng = np.random.RandomState(seed)
        sel = getattr(selection, self._SELECTORS[selector])(self._contexts)
        if selector == "round_robin":  # stagger parallel workers so each batch covers all contexts
            sel.context_id = (start - 1) % len(self._contexts)
        if selector == "random":  # CARL's RandomSelector uses the global np.random
            sel._select = lambda: (lambda cid: (sel.contexts[sel.contexts_keys[cid]], cid))(
                int(self._rng.choice(sel.context_ids))
            )
        self._env = self._seeded(cls)(contexts=self._contexts, context_selector=sel, obs_context_as_dict=False)
        self._env._draw_seed = lambda: int(self._rng.randint(2**31 - 1))
        self._action_repeat = action_repeat
        self._size = tuple(size)
        domain = name.split("_", 1)[1]
        self._camera = dict(quadruped=2).get(domain, 0) if camera is None else camera
        self.reward_range = [-np.inf, np.inf]

    @staticmethod
    def _seeded(cls):
        # CARL rebuilds the dm_control env on every context switch and never
        # passes a seed, so initial states are unseeded. Pass one per rebuild.
        from carl.envs.dmc.loader import load_dmc_env
        from carl.envs.dmc.wrappers import MujocoToGymWrapper

        class Seeded(cls):
            def _update_context(self):
                env = load_dmc_env(
                    domain_name=self.domain,
                    task_name=self.task,
                    context=self.context,
                    task_kwargs={"random": self._draw_seed()},
                    environment_kwargs={"flat_observation": True},
                )
                self.env = MujocoToGymWrapper(env)

        Seeded.__name__ = cls.__name__
        return Seeded

    @staticmethod
    def build_contexts(cls, spec):
        # {feature: [values]} -> {id: full context}; CARL needs every feature present
        default = cls.get_default_context()
        if not spec:
            return {0: dict(default)}
        keys = list(spec.keys())
        unknown = set(keys) - set(default)
        if unknown:
            raise ValueError(f"unknown context features {unknown}; available: {sorted(default)}")
        return {
            i: {**default, **dict(zip(keys, values))}
            for i, values in enumerate(itertools.product(*(spec[k] for k in keys)))
        }

    @property
    def observation_space(self):
        base = self._env.observation_space.spaces
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
                "state": gym.spaces.Box(-np.inf, np.inf, base["obs"].shape, dtype=np.float32),
                "context": gym.spaces.Box(-np.inf, np.inf, base["context"].shape, dtype=np.float32),
            }
        )

    @property
    def action_space(self):
        a = self._env.action_space
        return gym.spaces.Box(a.low.astype(np.float32), a.high.astype(np.float32), dtype=np.float32)

    def _obs(self, raw, is_first, is_terminal):
        return {
            "image": self.render(),
            "state": np.asarray(raw["obs"], np.float32),
            "context": np.asarray(raw["context"], np.float32),
            "is_first": is_first,
            "is_terminal": is_terminal,
        }

    def reset(self):
        raw, _ = self._env.reset()
        return self._obs(raw, is_first=True, is_terminal=False)

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0.0
        terminated = truncated = False
        for _ in range(self._action_repeat):
            raw, r, terminated, truncated, _ = self._env.step(action)
            reward += float(r)
            if terminated or truncated:
                break
        obs = self._obs(raw, is_first=False, is_terminal=terminated)
        info = {"discount": np.array(0.0 if terminated else 1.0, np.float32), "context_id": self._env.context_id}
        return obs, reward, terminated or truncated, info

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        h, w = self._size  # self._env.env is rebuilt on every context switch; resolve each call
        return self._env.env.render(mode="rgb_array", camera_id=self._camera, height=h, width=w)

    @property
    def context_id(self):
        return self._env.context_id

    @property
    def context(self):
        return dict(self._env.context)
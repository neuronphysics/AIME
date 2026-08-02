"""ProcGen wrapper exposing the same interface as envs/dmc.py.

ProcGen is natively 64x64 RGB with a Discrete(15) action space, so no resizing
or action remapping is needed -- it maps onto the Dreamer pipeline almost
directly.

Protocol caveat: "beating DreamerV3 on ProcGen" is only meaningful if the level
distribution matches.  The two axes that matter are ``distribution_mode``
(easy/hard) and ``num_levels`` (0 = unlimited procedural levels, i.e. the
train-on-everything setting; a finite value creates a train/test generalisation
split).  Check the DreamerV3 appendix and set these to match before claiming a
comparison -- a mismatch here swamps any modelling difference.
"""

import gym
import numpy as np


GAMES = (
    "bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun",
    "dodgeball", "fruitbot", "heist", "jumper", "leaper", "maze", "miner",
    "ninja", "plunder", "starpilot",
)


class ProcGen:
    metadata = {}

    def __init__(
        self,
        name,
        action_repeat=1,
        size=(64, 64),
        seed=0,
        distribution_mode="easy",
        num_levels=0,
        start_level=0,
    ):
        game = name.replace("_", "-").strip().lower()
        if game not in GAMES:
            raise ValueError(f"Unknown ProcGen game '{game}'. Valid: {GAMES}")
        import procgen  # noqa: F401  (registers the gym ids)

        self._env = gym.make(
            f"procgen:procgen-{game}-v0",
            distribution_mode=distribution_mode,
            num_levels=num_levels,
            start_level=start_level,
            rand_seed=seed,
        )
        self._action_repeat = action_repeat
        self._size = tuple(size)
        if self._size != (64, 64):
            raise ValueError(
                "ProcGen renders natively at 64x64; set size: [64, 64]."
            )
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
        })

    @property
    def action_space(self):
        # Wrapped by wrappers.OneHotAction in dreamer.make_env.
        return self._env.action_space

    def _obs(self, image, is_first, is_terminal):
        return {
            "image": np.asarray(image, np.uint8),
            "is_first": is_first,
            "is_terminal": is_terminal,
        }

    def step(self, action):
        reward = 0.0
        for _ in range(self._action_repeat):
            image, r, done, info = self._env.step(action)
            reward += float(r)
            if done:
                break
        return (
            self._obs(image, is_first=False, is_terminal=bool(done)),
            reward,
            bool(done),
            {"discount": np.float32(1.0 - float(done))},
        )

    def reset(self):
        image = self._env.reset()
        return self._obs(image, is_first=True, is_terminal=False)

    def render(self, *args, **kwargs):
        raise NotImplementedError("ProcGen observations are already images.")

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

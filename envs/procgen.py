import gym
import numpy as np


class ProcGen:
    metadata = {}

    def __init__(self, task, size=(64, 64), seed=0, num_levels=0,
                 distribution_mode="easy"):
        import procgen  # registers procgen-* with legacy gym
        self._env = gym.make(
            f"procgen:procgen-{task}-v0",
            start_level=seed, num_levels=num_levels,
            distribution_mode=distribution_mode,
            render_mode="rgb_array",
        )
        self._size = tuple(size)
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, (*self._size, 3), dtype=np.uint8),
            "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            "log_reward": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
        })

    @property
    def action_space(self):
        space = self._env.action_space          # Discrete(15)
        space.discrete = True                   # OneHotAction relies on this
        return space

    def _image(self, obs):
        img = np.asarray(obs, dtype=np.uint8)
        if img.shape[:2] != self._size:
            from PIL import Image
            img = np.array(Image.fromarray(img).resize(
                (self._size[1], self._size[0]), Image.BILINEAR))
        return img

    def step(self, action):
        image, reward, done, info = self._env.step(action)
        reward = np.float32(reward)
        obs = {
            "image": self._image(image),
            "is_first": False,
            "is_last": done,
            # procgen episodes end by termination, never by time limit
            "is_terminal": done,
            "log_reward": reward,
        }
        return obs, reward, done, info

    def reset(self):
        image = self._env.reset()
        return {
            "image": self._image(image),
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "log_reward": np.float32(0.0),
        }

    def render(self):
        return self._env.render(mode="rgb_array")

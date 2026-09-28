from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

SHAPE_TYPES = ("circle", "square", "triangle", "pentagon")
_SHAPE_COLORS = np.array([
    [231, 76, 60],
    [46, 204, 113],
    [52, 152, 219],
    [241, 196, 15],
], dtype=np.float32)


def _regular_polygon(cx, cy, r, n, rot):
    ang = rot + np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in ang]


class _Shape:
    __slots__ = ("kind", "color", "x", "y", "vx", "vy", "r", "rot", "vrot",
                 "alpha", "state", "timer", "fade_len", "persist_len", "absent_len")

    def __init__(self, kind, color, rng, H, W, ss):
        self.kind = kind
        self.color = color
        self.r = ss * rng.uniform(0.10, 0.16) * min(H, W)
        self.x = rng.uniform(self.r, ss * W - self.r)
        self.y = rng.uniform(self.r, ss * H - self.r)
        sp = ss * rng.uniform(0.6, 1.8)
        ang = rng.uniform(0, 2 * np.pi)
        self.vx = sp * np.cos(ang)
        self.vy = sp * np.sin(ang)
        self.rot = rng.uniform(0, 2 * np.pi)
        self.vrot = rng.uniform(-0.05, 0.05)
        self.fade_len = int(rng.integers(14, 28))
        self.persist_len = int(rng.integers(25, 60))
        self.absent_len = int(rng.integers(12, 40))
        self.state = rng.choice(["absent", "in", "present", "out"],
                                p=[0.25, 0.15, 0.45, 0.15])
        self.alpha = {"absent": 0.0, "in": rng.uniform(0, 1),
                      "present": 1.0, "out": rng.uniform(0, 1)}[self.state]
        self.timer = 0

    def _advance_lifecycle(self):
        self.timer += 1
        if self.state == "absent":
            self.alpha = 0.0
            if self.timer >= self.absent_len:
                self.state, self.timer = "in", 0
        elif self.state == "in":
            self.alpha = min(1.0, self.timer / self.fade_len)
            if self.timer >= self.fade_len:
                self.state, self.timer, self.alpha = "present", 0, 1.0
        elif self.state == "present":
            self.alpha = 1.0
            if self.timer >= self.persist_len:
                self.state, self.timer = "out", 0
        elif self.state == "out":
            self.alpha = max(0.0, 1.0 - self.timer / self.fade_len)
            if self.timer >= self.fade_len:
                self.state, self.timer, self.alpha = "absent", 0, 0.0

    def step(self, H, W, ss, drift=(0.0, 0.0)):
        self.x += self.vx + ss * drift[0]
        self.y += self.vy + ss * drift[1]
        if self.x < self.r:
            self.x = self.r; self.vx = abs(self.vx)
        if self.x > ss * W - self.r:
            self.x = ss * W - self.r; self.vx = -abs(self.vx)
        if self.y < self.r:
            self.y = self.r; self.vy = abs(self.vy)
        if self.y > ss * H - self.r:
            self.y = ss * H - self.r; self.vy = -abs(self.vy)
        self.rot += self.vrot
        self._advance_lifecycle()

    def draw(self, draw_ctx):
        if self.alpha <= 1e-3:
            return
        a = int(round(255 * self.alpha))
        col = (int(self.color[0]), int(self.color[1]), int(self.color[2]), a)
        x, y, r = self.x, self.y, self.r
        if self.kind == "circle":
            draw_ctx.ellipse([x - r, y - r, x + r, y + r], fill=col)
        elif self.kind == "square":
            pts = _regular_polygon(x, y, r * 1.15, 4, self.rot + np.pi / 4)
            draw_ctx.polygon(pts, fill=col)
        elif self.kind == "triangle":
            pts = _regular_polygon(x, y, r * 1.2, 3, self.rot - np.pi / 2)
            draw_ctx.polygon(pts, fill=col)
        elif self.kind == "pentagon":
            pts = _regular_polygon(x, y, r * 1.15, 5, self.rot - np.pi / 2)
            draw_ctx.polygon(pts, fill=col)


class MovingShapesSim:
    def __init__(self, n_shapes=4, size=(64, 64), supersample=2, bg=12, seed=0):
        assert 1 <= n_shapes <= len(SHAPE_TYPES)
        self.n = n_shapes
        self.H, self.W = size
        self.ss = supersample
        self.bg = bg
        self.rng = np.random.default_rng(seed)
        self._build()

    def _build(self):
        self.shapes = [
            _Shape(SHAPE_TYPES[i], _SHAPE_COLORS[i], self.rng, self.H, self.W, self.ss)
            for i in range(self.n)
        ]

    def reset(self):
        self._build()
        return self.render()

    def step(self, drift=(0.0, 0.0)):
        for s in self.shapes:
            s.step(self.H, self.W, self.ss, drift=drift)
        return self.render()

    def render(self):
        Hs, Ws = self.H * self.ss, self.W * self.ss
        base = Image.new("RGBA", (Ws, Hs), (self.bg, self.bg, self.bg, 255))
        for s in self.shapes:
            layer = Image.new("RGBA", (Ws, Hs), (0, 0, 0, 0))
            s.draw(ImageDraw.Draw(layer))
            base = Image.alpha_composite(base, layer)
        img = base.convert("RGB").resize((self.W, self.H), Image.LANCZOS)
        return np.asarray(img, dtype=np.uint8)

    def factors(self):
        alpha = np.array([s.alpha for s in self.shapes], dtype=np.float32)
        present = (alpha > 0.5).astype(np.float32)
        x = np.array([s.x / (self.ss * self.W) for s in self.shapes], dtype=np.float32)
        y = np.array([s.y / (self.ss * self.H) for s in self.shapes], dtype=np.float32)
        return dict(alpha=alpha, present=present, x=x, y=y,
                    n_present=np.float32(present.sum()))


def generate_dataset(n_seq=8, T=200, n_shapes=4, size=(64, 64), seed=0, drift_std=0.0):
    H, W = size
    frames = np.zeros((n_seq, T, H, W, 3), dtype=np.uint8)
    fac = {k: [] for k in ("alpha", "present", "x", "y", "n_present")}
    rng = np.random.default_rng(seed)
    for s in range(n_seq):
        sim = MovingShapesSim(n_shapes=n_shapes, size=size, seed=int(rng.integers(1 << 30)))
        seq_fac = {k: [] for k in fac}
        frames[s, 0] = sim.reset()
        f0 = sim.factors()
        for k in fac:
            seq_fac[k].append(f0[k])
        for t in range(1, T):
            drift = (drift_std * rng.standard_normal(), drift_std * rng.standard_normal()) \
                if drift_std > 0 else (0.0, 0.0)
            frames[s, t] = sim.step(drift=drift)
            ft = sim.factors()
            for k in fac:
                seq_fac[k].append(ft[k])
        for k in fac:
            fac[k].append(np.stack(seq_fac[k], 0))
    factors = {k: np.stack(v, 0) for k, v in fac.items()}
    return frames, factors


class ShapesEnv:
    metadata = {}

    def __init__(self, task="four", size=(64, 64), seed=0, time_limit=200, drift=0.6):
        import gym
        self._gym = gym
        n = {"three": 3, "four": 4}.get(task, 4)
        self._sim = MovingShapesSim(n_shapes=n, size=size, seed=seed)
        self._n = n
        self._size = size
        self._drift = drift
        self._time_limit = time_limit
        self._t = 0
        self.reward_range = [0.0, float(n)]

    @property
    def observation_space(self):
        gym = self._gym
        H, W = self._size
        spaces = {
            "image": gym.spaces.Box(0, 255, (H, W, 3), dtype=np.uint8),
            "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            "factor_n_present": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
        }
        for i in range(self._n):
            for key in ("alpha", "present", "x", "y"):
                spaces[f"factor_shape{i}_{key}"] = gym.spaces.Box(
                    -np.inf, np.inf, (1,), dtype=np.float32)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        space = self._gym.spaces.Discrete(5)
        space.discrete = True
        return space

    def _drift_for(self, action):
        return {0: (0.0, 0.0), 1: (0.0, -self._drift), 2: (0.0, self._drift),
                3: (-self._drift, 0.0), 4: (self._drift, 0.0)}.get(int(action), (0.0, 0.0))

    def _obs(self, image, is_first, is_last):
        f = self._sim.factors()
        obs = {
            "image": image,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": False,
            "factor_n_present": np.float32(f["n_present"]),
        }
        for i in range(self._n):
            obs[f"factor_shape{i}_alpha"] = np.float32(f["alpha"][i])
            obs[f"factor_shape{i}_present"] = np.float32(f["present"][i])
            obs[f"factor_shape{i}_x"] = np.float32(f["x"][i])
            obs[f"factor_shape{i}_y"] = np.float32(f["y"][i])
        return obs

    def reset(self):
        self._t = 0
        image = self._sim.reset()
        return self._obs(image, is_first=True, is_last=False)

    def step(self, action):
        image = self._sim.step(drift=self._drift_for(action))
        self._t += 1
        f = self._sim.factors()
        reward = float((f["alpha"] >= 0.99).sum())
        done = self._t >= self._time_limit
        info = {"discount": 1.0, "n_present": float(f["n_present"])}
        return self._obs(image, is_first=False, is_last=done), reward, done, info

    def render(self):
        return self._sim.render()

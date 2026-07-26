"""Synthetic 'moving shapes' video environment for SHS-RSSM disentanglement studies.

Several geometric shapes (circle / square / triangle / pentagon) drift around a frame,
each with its own slow appear -> persist -> disappear lifecycle (smooth alpha fade). The
SET of currently-visible shapes therefore changes over time, and each such change is a
switch in the scene's dynamics -- exactly the kind of structure the sticky-HDP switching
RSSM is meant to discover. Every frame carries ground-truth factors (per-shape alpha,
presence, position; number of shapes present) so disentanglement can be measured rather
than eyeballed.

Two entry points:
  * `MovingShapesSim` / `generate_dataset(...)` -- pure numpy + PIL, no gym, no RL stack;
    used by the offline diagnostics demo.
  * `ShapesEnv` -- a gym-style wrapper matching the dreamerv3-torch env contract (image +
    is_first/is_last/is_terminal + `factor_*` factor channels), so the same world can be
    trained inside DreamerV3.

The factors are exposed to DreamerV3 as `factor_*` observation keys. Unlike the `log_*`
convention (which the rollout/replay pipeline strips after logging to TensorBoard, so it
would never reach training), `factor_*` keys survive into the replay and the training
batch, while the encoder and decoder skip them by name, so they are available at
diagnostics time without ever leaking into the model's inputs or reconstruction targets.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

SHAPE_TYPES = ("circle", "square", "triangle", "pentagon")
# distinct, easily separable colors (RGB, 0-255)
_SHAPE_COLORS = np.array([
    [231, 76, 60],     # red
    [46, 204, 113],    # green
    [52, 152, 219],    # blue
    [241, 196, 15],    # yellow
], dtype=np.float32)


def _regular_polygon(cx, cy, r, n, rot):
    ang = rot + np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in ang]


class _Shape:
    """One shape with position, velocity, type/color and a fade lifecycle."""
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
        # lifecycle timing (in frames) -- slow fades so appearance is gradual
        self.fade_len = int(rng.integers(14, 28))
        self.persist_len = int(rng.integers(25, 60))
        self.absent_len = int(rng.integers(12, 40))
        # randomize the initial phase so shapes are desynchronized
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
        # bounce off the walls
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
    """Pure-numpy/PIL simulator of moving, fading shapes with ground-truth factors.

    Args:
        n_shapes: number of distinct shapes in the scene (3 or 4 recommended).
        size: (H, W) output resolution.
        supersample: render at this multiple then downsample (anti-aliasing).
        bg: background grey level 0-255.
        seed: RNG seed.
    """
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
        # draw shapes back-to-front, each on its own alpha layer so fades composite
        for s in self.shapes:
            layer = Image.new("RGBA", (Ws, Hs), (0, 0, 0, 0))
            s.draw(ImageDraw.Draw(layer))
            base = Image.alpha_composite(base, layer)
        img = base.convert("RGB").resize((self.W, self.H), Image.LANCZOS)
        return np.asarray(img, dtype=np.uint8)

    def factors(self):
        """Ground-truth factors for the CURRENT frame.

        Returns a dict with per-shape alpha (n,), presence (n,), normalized x/y (n,),
        and the scalar count of shapes currently present (alpha > 0.5).
        """
        alpha = np.array([s.alpha for s in self.shapes], dtype=np.float32)
        present = (alpha > 0.5).astype(np.float32)
        x = np.array([s.x / (self.ss * self.W) for s in self.shapes], dtype=np.float32)
        y = np.array([s.y / (self.ss * self.H) for s in self.shapes], dtype=np.float32)
        return dict(alpha=alpha, present=present, x=x, y=y,
                    n_present=np.float32(present.sum()))


def generate_dataset(n_seq=8, T=200, n_shapes=4, size=(64, 64), seed=0, drift_std=0.0):
    """Generate a batch of shape-video sequences with ground-truth factors.

    Returns:
        frames: uint8 array (n_seq, T, H, W, 3)
        factors: dict of arrays, each leading dims (n_seq, T): 'alpha' (.,n), 'present'
            (.,n), 'x' (.,n), 'y' (.,n), 'n_present' (.,).
    """
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


# --------------------------------------------------------------- DreamerV3 gym wrapper
class ShapesEnv:
    """gym-style wrapper matching the dreamerv3-torch env contract.

    Discrete actions apply a small global drift to all shapes (0 = no-op; 1-4 = nudge
    N/S/E/W), giving the actor something to do while the scene dynamics stay largely
    autonomous. Reward is the number of fully-visible shapes (a smooth, well-defined
    signal). Ground-truth factors are exposed as `factor_*` observation channels.
    """
    metadata = {}

    def __init__(self, task="four", size=(64, 64), seed=0, time_limit=200, drift=0.6):
        import gym  # lazy: only needed inside the training stack
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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None


@dataclass(frozen=True)
class StreamSchedule:
    change_start: int = 300
    drift_steps: int = 150
    final_angle_deg: float = 90.0

    def angle_deg(self, update: int) -> float:
        if update < self.change_start:
            return 0.0
        if self.drift_steps <= 1:
            return float(self.final_angle_deg)
        f = (update - self.change_start) / float(self.drift_steps - 1)
        return float(np.clip(f, 0.0, 1.0) * self.final_angle_deg)


class ControlledSegmentationMovingMNIST:
    """One-digit Moving-MNIST trajectories plus deterministic rendering."""

    def __init__(
        self,
        mnist_root: str | Path = "./data",
        split: str = "train",
        frame_size: tuple[int, int] = (64, 64),
        num_frames: int = 20,
        digit_size: int = 28,
        min_speed: float = 2.0,
        max_speed: float = 6.0,
        download: bool = False,
    ):
        self.mnist_root = Path(mnist_root)
        self.split = split
        self.height, self.width = map(int, frame_size)
        self.num_frames = int(num_frames)
        self.digit_size = int(digit_size)
        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        self._mnist = None
        self._download = bool(download)
        if self.height < self.digit_size or self.width < self.digit_size:
            raise ValueError("frame_size must be >= digit_size")

    def _load_mnist(self):
        if self._mnist is not None:
            return self._mnist
        try:
            from torchvision.datasets import MNIST
        except Exception as e:
            raise RuntimeError("torchvision is required for the visual experiment") from e
        train = self.split.lower() != "test"
        self._mnist = MNIST(root=str(self.mnist_root), train=train, download=self._download)
        return self._mnist

    def sample_state_batch(self, rng: np.random.Generator, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Return exact [centre_x, centre_y, vx, vy] trajectories and random MNIST indices."""
        B, T = int(batch_size), self.num_frames
        lim_x = self.width - self.digit_size
        lim_y = self.height - self.digit_size
        centre_offset = 0.5 * (self.digit_size - 1)
        direction = np.pi * (2.0 * rng.random(B) - 1.0)
        speed = rng.uniform(self.min_speed, self.max_speed, size=B)
        vel = np.stack([speed * np.cos(direction), speed * np.sin(direction)], axis=-1)
        pos = np.stack([rng.random(B) * lim_x, rng.random(B) * lim_y], axis=-1)
        n = 60000 if self.split.lower() != "test" else 10000
        digit_idx = rng.integers(0, n, size=B, dtype=np.int64)

        state = np.empty((B, T, 4), dtype=np.float64)
        for t in range(T):
            state[:, t, :2] = pos + centre_offset
            state[:, t, 2:] = vel
            nxt = pos + vel
            hit_x = (nxt[:, 0] < 0.0) | (nxt[:, 0] > lim_x)
            hit_y = (nxt[:, 1] < 0.0) | (nxt[:, 1] > lim_y)
            vel[hit_x, 0] *= -1.0
            vel[hit_y, 1] *= -1.0
            pos = pos + vel
            pos[:, 0] = np.clip(pos[:, 0], 0.0, lim_x)
            pos[:, 1] = np.clip(pos[:, 1], 0.0, lim_y)
        return state, digit_idx

    def render_batch(self, state: np.ndarray, digit_indices: np.ndarray) -> np.ndarray:
        """Render clean grayscale videos, uint8 [B,T,1,H,W]."""
        mnist = self._load_mnist()
        s = np.asarray(state)
        B, T = s.shape[:2]
        out = np.zeros((B, T, 1, self.height, self.width), dtype=np.uint8)
        centre_offset = 0.5 * (self.digit_size - 1)
        for b in range(B):
            img, _ = mnist[int(digit_indices[b]) % len(mnist)]
            digit = np.asarray(img, dtype=np.uint8)
            for t in range(T):
                x0 = int(round(s[b, t, 0] - centre_offset))
                y0 = int(round(s[b, t, 1] - centre_offset))
                x1, y1 = x0 + self.digit_size, y0 + self.digit_size
                xa, xb = max(0, x0), min(self.width, x1)
                ya, yb = max(0, y0), min(self.height, y1)
                if xa >= xb or ya >= yb:
                    continue
                sx0, sy0 = xa-x0, ya-y0
                sx1, sy1 = sx0+(xb-xa), sy0+(yb-ya)
                patch = out[b, t, 0, ya:yb, xa:xb]
                np.maximum(patch, digit[sy0:sy1, sx0:sx1], out=patch)
        return out

    @staticmethod
    def downsample(frames: np.ndarray, side: int = 16) -> np.ndarray:
        """Area-average [B,T,1,H,W] frames to [B,T,side,side] in [0,1]."""
        x = np.asarray(frames, dtype=np.float32) / 255.0
        B, T, C, H, W = x.shape
        if C != 1 or H % side or W % side:
            raise ValueError("frame size must be divisible by downsample side")
        fh, fw = H//side, W//side
        return x[:, :, 0].reshape(B, T, side, fh, side, fw).mean(axis=(3, 5))

    @staticmethod
    def rotate_small(frames_small: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate already-downsampled frames; faster and deterministic for the quantitative loop."""
        from scipy.ndimage import rotate
        x = np.asarray(frames_small, dtype=np.float32)
        if abs(float(angle_deg)) < 1e-12:
            return x.copy()
        flat = x.reshape(-1, x.shape[-2], x.shape[-1])
        y = np.empty_like(flat)
        for i, frame in enumerate(flat):
            y[i] = rotate(frame, float(angle_deg), reshape=False, order=1, mode="constant", cval=0.0,
                          prefilter=False)
        return np.clip(y.reshape(x.shape), 0.0, 1.0)


class FixedPCAEncoder:

    def __init__(self, latent_dim: int = 8, eps: float = 1e-6):
        self.latent_dim = int(latent_dim)
        self.eps = float(eps)
        self.mean = None
        self.components = None
        self.scale = None

    def fit(self, frames_small: np.ndarray):
        X = np.asarray(frames_small, dtype=np.float64).reshape(-1, np.prod(frames_small.shape[-2:]))
        self.mean = X.mean(0)
        Xc = X - self.mean
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        d = min(self.latent_dim, vt.shape[0])
        self.components = vt[:d].copy()
        var = (s[:d]**2) / max(len(X)-1, 1)
        self.scale = np.sqrt(np.maximum(var, self.eps))
        self.latent_dim = d
        return self

    def encode(self, frames_small: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("fit the PCA encoder first")
        X = np.asarray(frames_small, dtype=np.float64).reshape(-1, len(self.mean))
        Z = ((X-self.mean) @ self.components.T) / self.scale
        return Z.reshape(*frames_small.shape[:-2], self.latent_dim)

    def decode(self, z: np.ndarray, side: int) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        X = (z.reshape(-1, self.latent_dim) * self.scale) @ self.components + self.mean
        return np.clip(X.reshape(*z.shape[:-1], side, side), 0.0, 1.0)

    def save(self, path: str | Path):
        np.savez_compressed(path, mean=self.mean, components=self.components, scale=self.scale,
                            latent_dim=np.array(self.latent_dim))

    @classmethod
    def load(cls, path: str | Path):
        d = np.load(path)
        obj = cls(int(d["latent_dim"]))
        obj.mean = d["mean"]
        obj.components = d["components"]
        obj.scale = d["scale"]
        return obj


def prepare_latent_for_model(z: np.ndarray, pixels_small: np.ndarray | None = None) -> dict[str, object]:
    """Create AIME regression/gate arrays from visual latents [B,T,D]."""
    import torch
    x = np.asarray(z, dtype=np.float64)
    prev, y = x[:, :-1], x[:, 1:]
    g = np.concatenate([prev, np.ones((*prev.shape[:2], 1), dtype=np.float64)], axis=-1)
    phi = np.concatenate([prev, np.ones((*prev.shape[:2], 1), dtype=np.float64)], axis=-1)
    out = {
        "y": torch.from_numpy(y), "g": torch.from_numpy(g), "phi": torch.from_numpy(phi),
        "y_np": y, "g_np": g, "phi_np": phi,
    }
    if pixels_small is not None:
        out["target_pixels"] = np.asarray(pixels_small[:, 1:], dtype=np.float64)
    return out

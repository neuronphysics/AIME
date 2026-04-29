ffrom __future__ import annotations
import os
import math
import json
import time
import random
import argparse
from tqdm.auto import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable

from torch.utils.data import ConcatDataset
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import PIL.Image
import h5py
from torch.utils.data import ConcatDataset
import wandb  
from VRNN.dpgmm_stickbreaking_prior_vrnn import DPGMMVariationalRecurrentAutoencoder
from VRNN.warp import image_warp
from VRNN.grad_diagnostics import GradDiagnosticsAggregator
from VRNN.visualize_latent_clusters import visualize_dpgmm_clustering
from torchvision.utils import make_grid


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    gauss = torch.tensor(
        [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)],
        device=device, dtype=dtype
    )
    gauss = gauss / gauss.sum()
    window_2d = gauss[:, None] @ gauss[None, :]
    return window_2d

def ssim_(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5, eps: float = 1e-6) -> torch.Tensor:
    """
    x,y: [B,3,H,W] in [0,1]
    Returns scalar SSIM averaged over batch.
    """
    assert x.ndim == 4 and y.ndim == 4, "SSIM expects BCHW"
    device, dtype = x.device, x.dtype
    window = _gaussian_window(window_size, sigma, device, dtype).unsqueeze(0).unsqueeze(0)  # [1,1,ws,ws]
    window = window.expand(3, 1, window_size, window_size)  # [3,1,ws,ws]

    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=3)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=3)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=3) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=3) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=3) - mu_xy

    # standard SSIM constants for data range 1
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + eps
    ssim_map = num / den
    return ssim_map.mean()

def psnr_(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x,y: [B,3,H,W] in [0,1]
    Returns scalar PSNR averaged over batch.
    """
    mse = (x - y).pow(2).mean(dim=(1,2,3)).clamp_min(eps)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return psnr.mean()


# -------------------------
# Utility: flow visualization (HSV)
# -------------------------

def flow_to_hsv_rgb(flow: torch.Tensor, max_flow: float = 20.0) -> torch.Tensor:
    """
    flow: [B,2,H,W] (pixels)
    returns rgb: [B,3,H,W] in [0,1]
    """
    assert flow.ndim == 4 and flow.shape[1] == 2
    u = flow[:, 0]
    v = flow[:, 1]
    mag = torch.sqrt(u*u + v*v + 1e-8)
    ang = torch.atan2(v, u)  # [-pi,pi]
    # hue [0,1]
    hue = (ang + math.pi) / (2 * math.pi)
    sat = torch.ones_like(hue)
    val = (mag / (max_flow + 1e-6)).clamp(0, 1)

    # HSV -> RGB
    h6 = hue * 6.0
    i = torch.floor(h6).long() % 6
    f = h6 - torch.floor(h6)
    p = val * (1.0 - sat)
    q = val * (1.0 - sat * f)
    t = val * (1.0 - sat * (1.0 - f))

    r = torch.zeros_like(val)
    g = torch.zeros_like(val)
    b = torch.zeros_like(val)

    # vectorized switch
    mask = (i == 0); r[mask] = val[mask]; g[mask] = t[mask];   b[mask] = p[mask]
    mask = (i == 1); r[mask] = q[mask];   g[mask] = val[mask]; b[mask] = p[mask]
    mask = (i == 2); r[mask] = p[mask];   g[mask] = val[mask]; b[mask] = t[mask]
    mask = (i == 3); r[mask] = p[mask];   g[mask] = q[mask];   b[mask] = val[mask]
    mask = (i == 4); r[mask] = t[mask];   g[mask] = p[mask];   b[mask] = val[mask]
    mask = (i == 5); r[mask] = val[mask]; g[mask] = p[mask];   b[mask] = q[mask]

    rgb = torch.stack([r,g,b], dim=1)
    return rgb.clamp(0, 1)


# -------------------------
# HDF5 inspection / autodetect
# -------------------------

def _is_image_dataset(dset: Any) -> bool:
    if not hasattr(dset, "shape") or not hasattr(dset, "dtype"):
        return False
    shape = tuple(dset.shape)
    if len(shape) != 4:
        return False
    # (T,H,W,C) or (T,C,H,W) possible; robomimic usually (T,H,W,C)
    if shape[-1] == 3 and str(dset.dtype).startswith("uint"):
        return True
    if shape[1] == 3 and str(dset.dtype).startswith("uint"):
        return True
    return False

def inspect_robomimic_hdf5(
    hdf5_path: str,
    max_keys: int = 50,
    *,
    verbose: bool = True,
    max_demos_scan: int = 25,
) -> Dict[str, Any]:
    """
    Returns summary dict and (optionally) prints a short layout.

    Notes:
      - Some RoboMimic variants store observations under /obs; others may use /observations.
      - Some files (e.g., low_dim or state-only variants) have no image observations at all.
      - We scan up to `max_demos_scan` demos to find the first demo that contains an image obs dataset.
    """
    info: Dict[str, Any] = {"path": hdf5_path}
    with h5py.File(hdf5_path, "r") as f:
        root_keys = list(f.keys())
        info["root_keys"] = root_keys
        if "data" not in f:
            raise KeyError(f"Expected top-level group 'data' in {hdf5_path}; found keys: {root_keys}")

        demos = sorted(list(f["data"].keys()))
        info["num_demos"] = len(demos)
        info["demo_keys"] = demos[:max_keys]

        if len(demos) == 0:
            raise ValueError("No demos found under /data")

        # Find a demo that actually contains image observations.
        chosen_demo = demos[0]
        chosen_group_name: Optional[str] = None
        all_obs_keys: List[str] = []
        img_keys: List[str] = []
        demo0_subkeys: List[str] = []

        for di, d in enumerate(demos[:max_demos_scan]):
            g = f["data"][d]
            subkeys = list(g.keys())
            if di == 0:
                demo0_subkeys = subkeys

            for group_name in ("obs", "observations"):
                if group_name in g:
                    obs_group = g[group_name]
                    keys = sorted(list(obs_group.keys()))
                    img = []
                    for k in keys:
                        try:
                            if _is_image_dataset(obs_group[k]):
                                img.append(k)
                        except Exception:
                            pass

                    # Prefer demos with at least one image key
                    if len(img) > 0:
                        chosen_demo = d
                        chosen_group_name = group_name
                        all_obs_keys = keys
                        img_keys = img
                        break

                    # Otherwise remember non-image obs keys 
                    if chosen_group_name is None:
                        chosen_demo = d
                        chosen_group_name = group_name
                        all_obs_keys = keys
                        img_keys = []

            if len(img_keys) > 0:
                break

        info["demo0_subkeys"] = demo0_subkeys
        info["chosen_demo_for_obs_scan"] = chosen_demo
        info["obs_group_name"] = chosen_group_name
        info["all_obs_keys"] = all_obs_keys
        info["image_obs_keys"] = img_keys

        # action shape
        g0 = f["data"][chosen_demo]
        if "actions" in g0:
            info["action_shape"] = tuple(g0["actions"].shape)
            info["action_dtype"] = str(g0["actions"].dtype)

        # length hints
        lengths = {}
        for k in ["actions", "rewards", "dones", "terminated", "terminals"]:
            if k in g0:
                lengths[k] = int(g0[k].shape[0])
        if chosen_group_name is not None and chosen_group_name in g0 and len(all_obs_keys) > 0:
            try:
                lengths[f"{chosen_group_name}/{all_obs_keys[0]}"] = int(g0[chosen_group_name][all_obs_keys[0]].shape[0])
            except Exception:
                pass
        info["lengths_demo0"] = lengths

    if verbose:
        print("=" * 80)
        print("[RoboMimic HDF5] ", hdf5_path)
        print("Root keys:", info["root_keys"])
        print("Num demos:", info["num_demos"])
        print("Demo[0] subkeys:", info.get("demo0_subkeys", []))
        print("Obs group:", info.get("obs_group_name", None), " | chosen demo:", info.get("chosen_demo_for_obs_scan", None))
        print("Obs keys (chosen demo):", (info["all_obs_keys"][:max_keys] if info["all_obs_keys"] else []))
        print("Image obs keys:", info["image_obs_keys"])
        print("Action shape:", info.get("action_shape", None), "dtype:", info.get("action_dtype", None))
        print("Length hints (chosen demo):", info.get("lengths_demo0", {}))
        print("=" * 80)

    return info

def auto_select_image_obs_keys(info: Dict[str, Any], prefer: Tuple[str,...] = ("agentview_image", "agentview", "rgb")) -> List[str]:
    """
    Choose image obs keys automatically. Preference order:
    - exact agentview_image if exists
    - any key containing prefer tokens
    - else first image key
    """
    img_keys = info.get("image_obs_keys", [])
    if not img_keys:
        return []
    if "agentview_image" in img_keys:
        return ["agentview_image"]
    lowered = [(k, k.lower()) for k in img_keys]
    for token in prefer:
        for k, kl in lowered:
            if token in kl:
                return [k]
    return [img_keys[0]]

# HDF5 discovery / multi-file support
def _expand_hdf5_inputs(
    hdf5: str,
    *,
    recursive: bool = True,
    prefer_demo_files: bool = True,
    include_low_dim: bool = False,
    max_files: Optional[int] = None,
) -> List[str]:
    """
    hdf5: may be
      - a single .hdf5 path
      - a directory containing .hdf5 files (optionally recursive)
      - a comma-separated list of files/dirs

    By default we prefer files that contain "demo" in the filename (to avoid low_dim-only datasets).
    """
    inputs = [p.strip() for p in str(hdf5).split(",") if p.strip()]
    out: List[str] = []
    for p in inputs:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                for fn in sorted(files):
                    if not (fn.endswith(".hdf5") or fn.endswith(".h5")):
                        continue
                    if prefer_demo_files and (not include_low_dim):
                        # keep demo files by default
                        if "demo" not in fn.lower():
                            continue
                    out.append(os.path.join(root, fn))
                    if max_files is not None and len(out) >= max_files:
                        break
                if max_files is not None and len(out) >= max_files:
                    break
                if not recursive:
                    break
        else:
            out.append(p)

        if max_files is not None and len(out) >= max_files:
            break

    # de-dup, stable sort
    out = sorted(list(dict.fromkeys(out)))
    if max_files is not None:
        out = out[:max_files]
    return out

def _filter_hdf5_with_images(
    paths: List[str],
    *,
    verbose: bool = True,
    max_print: int = 40,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Returns (usable_paths, infos) where usable_paths have at least one image obs key.
    """
    usable: List[str] = []
    infos: List[Dict[str, Any]] = []
    for p in paths:
        try:
            info = inspect_robomimic_hdf5(p, verbose=False)
            if info.get("image_obs_keys"):
                usable.append(p)
                infos.append(info)
        except Exception as e:
            if verbose:
                print(f"[scan] skip (unreadable): {p} | {e}")

    if verbose:
        print(f"[scan] candidates: {len(paths)}  usable_with_images: {len(usable)}")
        if len(usable) == 0:
            print("[scan] No image-observation HDF5 files found under the provided path(s).")
            print("       This usually means you pointed at a state-only or low-dim dataset (e.g. mg/low_dim_*).")
        else:
            for p in usable[:max_print]:
                print("  [use] ", p)
            if len(usable) > max_print:
                print(f"  ... ({len(usable) - max_print} more)")
    return usable, infos

def _choose_single_camera_key(
    infos: List[Dict[str, Any]],
    *,
    prefer: Tuple[str, ...] = ("agentview_image", "robot0_eye_in_hand_image", "frontview_image", "agentview", "rgb"),
) -> Optional[str]:
    """
    Pick ONE obs key to use across multiple files.
    Strategy:
      1) Choose a preferred key from the first file
      2) Keep only files that contain that key (handled by caller)
    """
    if not infos:
        return None
    # Start with per-file selector (already prefers agentview_image)
    k0_list = auto_select_image_obs_keys(infos[0], prefer=prefer)
    if not k0_list:
        return None
    return k0_list[0]


# Dataset

def to_float_image_tchw(x: np.ndarray) -> torch.Tensor:
    """
    x: uint8 image array in (T,H,W,C) or (H,W,C)
    returns float32 tensor in (T,C,H,W) or (C,H,W) scaled to [-1,1]
    """
    if x.ndim == 3:
        x = x[None, ...]
    assert x.ndim == 4
    # RoboMimic image observations are usually channel-last uint8: [T,H,W,3].
    # This also supports multiple concatenated RGB cameras: [T,H,W,3*num_cams].
    if x.shape[-1] >= 1 and (x.shape[-1] == 1 or x.shape[-1] % 3 == 0):
        x = np.transpose(x, (0, 3, 1, 2))  # T,C,H,W
    elif x.shape[1] >= 1 and (x.shape[1] == 1 or x.shape[1] % 3 == 0):
        # already T,C,H,W
        pass
    else:
        raise ValueError(
            f"Unexpected image shape {x.shape}; expected channel-last or channel-first RGB-like images."
        )

    x = x.astype(np.float32)
    # uint8 / [0,255] -> [0,1]. If already [0,1] or [-1,1], preserve the scale.
    if x.max() > 1.5:
        x = x / 255.0
    if x.min() >= 0.0:
        x = x * 2.0 - 1.0
    return torch.from_numpy(x)

def resize_tchw(x: torch.Tensor, size: int) -> torch.Tensor:
    """
    x: (T,C,H,W) float
    """
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)

def stack_frames(frames: torch.Tensor, frame_stack: int) -> torch.Tensor:
    """
    frames: (T_raw,C,H,W)
    returns: (T_out,C*frame_stack,H,W) where T_out = T_raw - frame_stack + 1
    """
    T, C, H, W = frames.shape
    if frame_stack <= 1:
        return frames
    out = []
    for t in range(frame_stack, T + 1):
        out.append(torch.cat([frames[t-frame_stack+i] for i in range(frame_stack)], dim=0))
    return torch.stack(out, dim=0)

class VideoRandomResizedCrop:
    """Random zoom-in crop, applied consistently across the whole sequence.
    x: [T, C, H, W] -> returns same shape, resized back to (H, W).
    Uses torch ops (works on CPU or GPU).
    """

    def __init__(self, zoom_prob: float = 0.6, area_range: Tuple[float, float] = (0.6, 0.85)):

        self.zoom_prob = float(zoom_prob)
        self.area_range = (float(area_range[0]), float(area_range[1]))


    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        if self.zoom_prob <= 0.0:
            return x

        if float(torch.rand((), device=x.device).item()) > self.zoom_prob:
            return x

        if x.ndim != 4:
            return x

        T, C, H, W = x.shape
        a0, a1 = self.area_range
        a0 = max(0.05, min(1.0, a0))
        a1 = max(a0, min(1.0, a1))
        area_frac = a0 + (a1 - a0) * float(torch.rand((), device=x.device).item())
        side_scale = math.sqrt(area_frac)
        crop_h = max(2, int(round(H * side_scale)))
        crop_w = max(2, int(round(W * side_scale)))
        if crop_h >= H or crop_w >= W:
            return x

        y0 = int(torch.randint(0, H - crop_h + 1, (1,), device=x.device).item())
        x0 = int(torch.randint(0, W - crop_w + 1, (1,), device=x.device).item())
        cropped = x[:, :, y0:y0 + crop_h, x0:x0 + crop_w]
        return F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)


def _safe_get_demo_dones(gdemo: h5py.Group) -> Optional[np.ndarray]:
    for k in ["dones", "terminated", "terminals"]:
        if k in gdemo:
            return np.asarray(gdemo[k]).astype(np.bool_)
    # robomimic often encodes dones in "rewards" only; absent => only last step done
    return None

class RoboMimicSequenceDataset(Dataset):
    """
    Yields sequences of (stacked) RGB frames + aligned actions/dones.

    Output dict:
      - observations: [T, C*frame_stack, H, W] float in [-1,1]
      - actions:      [T, A] float
      - done:         [T] bool (optional / always provided, synthesized if missing)
      - meta:         dict with demo_id + start index
    """
    def __init__(
        self,
        hdf5_path: str,
        demo_keys: List[str],
        obs_keys: List[str],
        sequence_length: int,
        frame_stack: int = 1,
        img_size: int = 64,
        obs_group_name: str = "obs",
        max_sequences_per_demo: Optional[int] = None,
        seed: int = 0,
        file_id: int = 0,
        augmenter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        self.hdf5_path = str(hdf5_path)
        self.demo_keys = list(demo_keys)
        self.obs_keys = list(obs_keys)
        self.obs_group_name = str(obs_group_name)
        self.sequence_length = int(sequence_length)
        self.frame_stack = int(frame_stack)
        self.img_size = int(img_size)
        self.max_sequences_per_demo = max_sequences_per_demo
        self.seed = int(seed)
        self.file_id = int(file_id)
        self.demo_to_idx = {d: i for i, d in enumerate(self.demo_keys)}
        self.augmenter = augmenter

        self._h5: Optional[h5py.File] = None  # opened lazily per worker
        self.indices: List[Tuple[str, int]] = []
        self.action_dim: Optional[int] = None
        self._build_index()

    def _open(self):
        if self._h5 is None:
            # Safe open for dataloader workers. Some HDF5 builds may not support SWMR on read-only files.
            try:
                self._h5 = h5py.File(self.hdf5_path, "r", libver="latest", swmr=True)
            except Exception:
                self._h5 = h5py.File(self.hdf5_path, "r")
    def _build_index(self):
        rng = random.Random(self.seed)
        with h5py.File(self.hdf5_path, "r") as f:
            data = f["data"]
            # Determine action_dim from first demo
            d0 = self.demo_keys[0]
            if "actions" not in data[d0]:
                raise KeyError(f"/data/{d0}/actions not found in {self.hdf5_path}")
            self.action_dim = int(data[d0]["actions"].shape[-1])

            for demo in self.demo_keys:
                gd = data[demo]

                if self.obs_group_name not in gd:
                    continue
                
                # Use first obs key length as obs length
                if len(self.obs_keys) == 0:
                    continue
                obs_len = int(gd[self.obs_group_name][self.obs_keys[0]].shape[0])

                act_len = int(gd["actions"].shape[0])
                # The stacked frames are built from raw obs frames (not next_obs),
                # we need exactly (seq_len + frame_stack - 1) raw frames.
                raw_needed = self.sequence_length + self.frame_stack - 1

                # We align actions with the LAST frame of each stack (same as dmc_vb trainer).
                # action_start = start + frame_stack - 1
                # We need action slice length = sequence_length -> require action_start + seq_len <= act_len.
                # So start <= act_len - seq_len - (frame_stack - 1).
                max_start_from_actions = act_len - self.sequence_length - (self.frame_stack - 1)
                max_start_from_obs = obs_len - raw_needed
                max_start = min(max_start_from_actions, max_start_from_obs)
                if max_start < 0:
                    continue

                starts = list(range(0, max_start + 1))
                if self.max_sequences_per_demo is not None and len(starts) > self.max_sequences_per_demo:
                    rng.shuffle(starts)
                    starts = sorted(starts[: self.max_sequences_per_demo])

                for s in starts:
                    self.indices.append((demo, s))

        if len(self.indices) == 0:
            raise RuntimeError("No valid sequences found; check sequence_length/frame_stack vs demo lengths.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self._open()
        demo, start = self.indices[idx]
        gd = self._h5["data"][demo]

        raw_needed = self.sequence_length + self.frame_stack - 1
        end = start + raw_needed  # exclusive

        # Load and concatenate image modalities along channel dimension
        imgs = []
        for k in self.obs_keys:
            arr = np.asarray(gd[self.obs_group_name][k][start:end])  # (T,H,W,C) typically
            imgs.append(arr)
        if len(imgs) == 1:
            img = imgs[0]
        else:
            # concat channel-last
            img = np.concatenate(imgs, axis=-1)

        # Convert to float tensor [-1,1] in TCHW
        x = to_float_image_tchw(img)  # (T_raw, C, H, W), C=3*num_cams
        x = resize_tchw(x, self.img_size)
        if self.augmenter is not None:
            x = self.augmenter(x)

        obs = stack_frames(x, self.frame_stack)  # (T, C*frame_stack, H, W)
        assert obs.shape[0] == self.sequence_length

        # actions aligned with last frame in stack
        action_start = start + self.frame_stack - 1
        actions = np.asarray(gd["actions"][action_start:action_start + self.sequence_length]).astype(np.float32)
        actions = torch.from_numpy(actions)

        dones_arr = _safe_get_demo_dones(gd)
        if dones_arr is None:
            done = torch.zeros((self.sequence_length,), dtype=torch.bool)
            # done true at last step of the full demo; align similarly
            # global_done_idx corresponds to action index
            if action_start + self.sequence_length - 1 == int(gd["actions"].shape[0]) - 1:
                done[-1] = True
        else:
            done = torch.from_numpy(dones_arr[action_start:action_start + self.sequence_length])

        local_demo_idx = int(self.demo_to_idx.get(demo, 0))
        # Unique numeric id for the demo across files, used by the DPGMM top-prior replay buffer.
        episode_idx = self.file_id * 1_000_000 + local_demo_idx

        return {
            "observations": obs,
            "actions": actions,
            "done": done,
            "episode_idx": torch.tensor(episode_idx, dtype=torch.long),
            "start_idx": torch.tensor(int(start), dtype=torch.long),
            "meta": {"demo": demo, "start": int(start), "file_id": int(self.file_id)},
        }


# Gradient monitor 

class GradientMonitor:
    """Component-level gradient norms (encoder/decoder/prior/vrnn/discriminators)."""
    def __init__(self, model: torch.nn.Module, writer: Optional[SummaryWriter]):
        self.model = model
        self.writer = writer
        self.component_groups = self._define_component_hierarchy()
        self.global_step = 0

    def _define_component_hierarchy(self):
        return {
            "encoder": ["vdvae.encoder."],
            "decoder": ["vdvae.decoder."],
            "prior_dynamics": ["prior.stick_breaking", "prior.component_nn"],
            "vrnn_core": ["_rnn", "rnn_layer_norm", ".rnn."],
            "flow_warp": ["flow", "warp", "flow_ctx", "flow_predict", "gru"],
            "discriminators": ["image_discriminator", "temporal_discriminator"],
        }

    def compute_component_gradients(self, update_step: bool = True, normalize: str = "rms_elem") -> Dict[str, float]:
        if update_step:
            self.global_step += 1

        out: Dict[str, float] = {}
        for group, pats in self.component_groups.items():
            sqsum = 0.0
            n_elems = 0
            n_tensors = 0
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                if not any(pat in name for pat in pats):
                    continue
                g = p.grad.detach()
                sqsum += g.pow(2).sum().item()
                n_elems += g.numel()
                n_tensors += 1

            if n_tensors == 0:
                out[group] = 0.0
                continue

            if normalize in ("rms_elem", "l2_per_sqrt_elem"):
                out[group] = math.sqrt(sqsum / max(n_elems, 1))
            elif normalize == "mean_tensor":
                out[group] = math.sqrt(sqsum) / max(n_tensors, 1)
            else:  # "none"
                out[group] = math.sqrt(sqsum)
        return out


# Logging helper: TensorBoard / wandb / both

class MultiLogger:
    """
    Drop-in-ish wrapper that supports .add_scalar and .add_image.

    - TensorBoard: uses SummaryWriter
    - wandb: uses wandb.log with step=global_step

    This keeps the rest of the trainer code simple (it can just call self.writer.add_*).
    """
    def __init__(
        self,
        tb_writer: Optional[SummaryWriter],
        *,
        use_wandb: bool = False,
        wandb_project: str = "D2E",
        wandb_entity: str = "",
        wandb_run_name: str = "run",
        wandb_dir: str = ".",
        wandb_config: Optional[Dict[str, Any]] = None,
        wandb_mode: str = "online",
    ):
        self.tb = tb_writer
        self.wandb_run = None
        self._wandb = None

        if use_wandb:
            try:
                
                self._wandb = wandb
                mode = (wandb_mode or "online").lower()
                if mode == "disabled":
                    mode = "disabled"
                elif mode == "offline":
                    mode = "offline"
                else:
                    mode = "online"

                self.wandb_run = wandb.init(
                    project=wandb_project,
                    entity=(wandb_entity or None),
                    name=wandb_run_name,
                    dir=wandb_dir,
                    config=(wandb_config or {}),
                    mode=mode,
                )
            except Exception as e:
                print(f"[warn] wandb init failed; continuing without wandb. Error: {e}")
                self.wandb_run = None
                self._wandb = None

    def add_scalar(self, tag: str, scalar_value: float, global_step: Optional[int] = None):
        if self.tb is not None:
            try:
                self.tb.add_scalar(tag, scalar_value, global_step)
            except Exception:
                pass
        if self.wandb_run is not None and self._wandb is not None:
            try:
                self.wandb_run.log({tag: float(scalar_value)}, step=global_step)
            except Exception:
                pass

    def add_image(self, tag: str, img_tensor: torch.Tensor, global_step: Optional[int] = None):
        """
        img_tensor: CHW in [0,1] (preferred) or [-1,1] (will be normalized)
        """
        if isinstance(img_tensor, torch.Tensor):
            img = img_tensor.detach()
        else:
            return

        # Ensure CHW float in [0,1]
        if img.dtype != torch.float32 and img.dtype != torch.float16 and img.dtype != torch.bfloat16:
            img = img.float()
        # If looks like [-1,1], map to [0,1]
        if img.min().item() < -0.1:
            img = (img + 1.0) * 0.5
        img = img.clamp(0, 1)

        if self.tb is not None:
            try:
                self.tb.add_image(tag, img, global_step)
            except Exception:
                pass

        if self.wandb_run is not None and self._wandb is not None:
            try:
                # wandb.Image expects HWC uint8 or float
                chw = img.cpu()
                if chw.ndim == 3 and chw.shape[0] in (1,3):
                    hwc = chw.permute(1,2,0).numpy()
                else:
                    hwc = chw.numpy()
                self.wandb_run.log({tag: self._wandb.Image(hwc)}, step=global_step)
            except Exception:
                pass

    def flush(self):
        if self.tb is not None:
            try:
                self.tb.flush()
            except Exception:
                pass

    def close(self):
        if self.tb is not None:
            try:
                self.tb.close()
            except Exception:
                pass
        if self.wandb_run is not None and self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass

# Trainer Class

@dataclass
class TrainConfig:
    # data (file/dir, or comma-separated list of files/dirs)
    hdf5: str
    obs_keys: Optional[List[str]]

    # multi-file scanning (when hdf5 is a directory or comma-separated inputs)
    scan_recursive: bool = True
    prefer_demo_files: bool = True
    include_low_dim: bool = False
    max_scan_files: int = 0  # 0 = no limit

    sequence_length: int = 16
    frame_stack: int = 1
    img_size: int = 64
    train_fraction: float = 0.95
    max_sequences_per_demo: Optional[int] = None
    seed: int = 0

    # data augmentation
    use_zoom_aug: bool = False
    zoom_prob: float = 0.6
    zoom_area_min: float = 0.6
    zoom_area_max: float = 0.85

    # DPGMM top prior refresh
    dpgmm_outer_batch_size: int = 256
    dpgmm_outer_n_laps: int = 4
    refresh_top_prior_every: int = 1

    # model update helpers
    use_ema: bool = True
    use_schedulers: bool = True

    # train
    batch_size: int = 30
    num_workers: int = 4
    n_epochs: int = 200

    # logging
    logger: str = "tensorboard"  # tensorboard | wandb | both
    log_dir: str = "./runs"
    run_name: str = "robomimic_run"
    wandb_project: str = "D2E"
    wandb_entity: str = ""
    wandb_mode: str = "online"  # online | offline | disabled

    save_every: int = 10
    eval_every: int = 1
    viz_every: int = 4
    log_images_every_steps: int = 500
    grad_diag_every_steps: int = 500
    eval_max_batches: int = 25

    # model
    max_components: int = 18
    latent_dim: int = 56
    hidden_dim: int = 48
    learning_rate: float = 2e-4
    grad_clip: float = 1.0
    prior_alpha: float = 16.0
    prior_beta: float = 2.0
    dropout: float = 0.1
    img_disc_layers: int = 3
    disc_num_heads: int = 8
    use_dynamic_weight_average: bool = False

    # loss weights
    beta: float = 1.0
    beta_min: float = 0.5
    beta_ramp_epochs: int = 30
    lambda_recon: float = 1.0
    lambda_img: float = 1.0
    n_critic: int = 1

    # rollout viz (future prediction)
    rollout_context_frames: int = 4
    rollout_horizon: int = 8

    # resume
    resume: Optional[str] = None

class RoboMimicTrainer:
    def __init__(self, cfg: TrainConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

                # --- Discover HDF5 files (single file, directory, or comma-separated list) ---
        max_files = None if int(getattr(cfg, "max_scan_files", 0) or 0) <= 0 else int(cfg.max_scan_files)
        candidate_paths = _expand_hdf5_inputs(
            cfg.hdf5,
            recursive=bool(getattr(cfg, "scan_recursive", True)),
            prefer_demo_files=bool(getattr(cfg, "prefer_demo_files", True)),
            include_low_dim=bool(getattr(cfg, "include_low_dim", False)),
            max_files=max_files,
        )
        if len(candidate_paths) == 0:
            raise RuntimeError(f"No .hdf5 files found from input: {cfg.hdf5}")

        usable_paths, infos = _filter_hdf5_with_images(candidate_paths, verbose=True)

        if len(usable_paths) == 0:
            raise RuntimeError(
                "No image-observation RoboMimic HDF5 files were found under the provided --hdf5 path(s). "
                "Try pointing to a 'ph' or 'mh' demo file (e.g., .../ph/demo_v15.hdf5) or to the task root "
                "(e.g., .../robomimic_v1.5/v1.5) so the script can scan."
            )

        # --- Decide obs key(s) ---
        # User asked for single-camera; by default we pick ONE image key and filter files to those that contain it.
        if cfg.obs_keys is None or len(cfg.obs_keys) == 0:
            chosen_key = _choose_single_camera_key(infos)
            if chosen_key is None:
                raise RuntimeError(
                    "Could not auto-detect image obs keys from the provided dataset(s). "
                    "Pass --obs_keys explicitly (comma-separated)."
                )
            obs_keys = [chosen_key]
            print("[Auto obs_keys] ", obs_keys)
        else:
            obs_keys = list(cfg.obs_keys)
            if len(obs_keys) > 1:
                print("[warn] You provided multiple --obs_keys; you said 'single camera', so consider passing exactly one key.")

        # Filter to files that actually contain the selected key
        filtered_paths: List[str] = []
        filtered_infos: List[Dict[str, Any]] = []
        for p, info in zip(usable_paths, infos):
            if len(obs_keys) == 0:
                continue
            if obs_keys[0] in (info.get("image_obs_keys") or []):
                filtered_paths.append(p)
                filtered_infos.append(info)

        if len(filtered_paths) == 0:
            # Helpful error: show what keys exist in the scanned files
            msg_lines = [
                "No scanned HDF5 file contained the requested obs key(s): " + ",".join(obs_keys),
                "Available image keys per file (first few):",
            ]
            for info in (infos[:10] if infos else []):
                msg_lines.append(f"  - {info.get('path','?')}: {info.get('image_obs_keys', [])}")
            raise RuntimeError("\n".join(msg_lines))

        self.hdf5_files = filtered_paths
        print(f"[data] Using {len(self.hdf5_files)} HDF5 file(s) with obs key {obs_keys[0]}")

        # --- Split demos train/eval across files (then ConcatDataset) ---

        train_augmenter = None

        if bool(getattr(cfg, "use_zoom_aug", False)):
            area_min = float(getattr(cfg, "zoom_area_min", 0.6))
            area_max = float(getattr(cfg, "zoom_area_max", 0.85))
            train_augmenter = VideoRandomResizedCrop(
                zoom_prob=float(getattr(cfg, "zoom_prob", 0.6)),
                area_range=(area_min, area_max),
            )        

        train_datasets: List[Dataset] = []
        eval_datasets: List[Dataset] = []
        total_train_demos = 0
        total_eval_demos = 0
        action_dims: List[int] = []

        for fi, (path, info) in enumerate(zip(self.hdf5_files, filtered_infos)):
            with h5py.File(path, "r") as f:
                demos = sorted(list(f["data"].keys()))
            rng = random.Random(cfg.seed + fi)
            rng.shuffle(demos)
            n_train = max(1, int(len(demos) * cfg.train_fraction))
            train_demos = sorted(demos[:n_train])
            eval_demos = sorted(demos[n_train:]) if n_train < len(demos) else sorted(demos[-max(1, len(demos)//20):])

            total_train_demos += len(train_demos)
            total_eval_demos += len(eval_demos)

            obs_group_name = info.get("obs_group_name", None) or "obs"

            train_ds_i = RoboMimicSequenceDataset(
                path, train_demos, obs_keys,
                sequence_length=cfg.sequence_length,
                frame_stack=cfg.frame_stack,
                img_size=cfg.img_size,
                max_sequences_per_demo=cfg.max_sequences_per_demo,
                seed=cfg.seed + fi,
                file_id=fi,
                obs_group_name=obs_group_name,
                augmenter=train_augmenter,
            )
            eval_ds_i = RoboMimicSequenceDataset(
                path, eval_demos, obs_keys,
                sequence_length=cfg.sequence_length,
                frame_stack=cfg.frame_stack,
                img_size=cfg.img_size,
                max_sequences_per_demo=min(cfg.max_sequences_per_demo or 10**9, 2000) if cfg.max_sequences_per_demo else 2000,
                seed=cfg.seed + 1000 + fi,
                file_id=fi,
                obs_group_name=obs_group_name,
                augmenter=None,
            )

            if getattr(train_ds_i, "action_dim", None) is not None:
                action_dims.append(int(train_ds_i.action_dim))

            train_datasets.append(train_ds_i)
            eval_datasets.append(eval_ds_i)

        print(f"[Split] demos total_train={total_train_demos} total_eval={total_eval_demos} across {len(self.hdf5_files)} file(s)")

        self.train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        self.eval_ds  = ConcatDataset(eval_datasets)  if len(eval_datasets)  > 1 else eval_datasets[0]

        # Determine action_dim (must be consistent across all files)
        action_dim = int(action_dims[0] if len(action_dims) > 0 else 0)
        if any(int(a) != action_dim for a in action_dims):
            print(f"[warn] action_dim differs across files: {action_dims}. Using first: {action_dim}")
        self.action_dim = action_dim


        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.eval_loader = DataLoader(
            self.eval_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=max(0, cfg.num_workers // 2),
            pin_memory=True,
            drop_last=False,
        )

        # --- Model ---
        input_channels = 3 * len(obs_keys) * cfg.frame_stack
        action_dim = int(getattr(self, 'action_dim', 0) or 0)
        self.model = DPGMMVariationalRecurrentAutoencoder(
            max_components=cfg.max_components,
            input_dim=cfg.img_size,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            action_dim=action_dim,
            sequence_length=cfg.sequence_length,
            img_disc_layers=cfg.img_disc_layers,
            disc_num_heads=cfg.disc_num_heads,
            device=device,
            input_channels=input_channels,
            learning_rate=cfg.learning_rate,
            grad_clip=cfg.grad_clip,
            prior_alpha=cfg.prior_alpha,
            prior_beta=cfg.prior_beta,
            dropout=cfg.dropout,
            use_dwa=cfg.use_dynamic_weight_average,
            rollout_context_frames=cfg.rollout_context_frames,
            rollout_horizon=cfg.rollout_horizon,
        ).to(device)

        if bool(getattr(cfg, 'use_ema', True)) and hasattr(self.model, 'ema_vdvae'):
            try:
                self.model.ema_vdvae.register()
            except Exception as e:
                print(f"[warn] EMA register failed: {e}")

        # --- Logging ---
        run_dir = Path(cfg.log_dir) / cfg.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)

        self.run_dir = run_dir
        tb_writer = SummaryWriter(str(run_dir)) if (cfg.logger in ('tensorboard','both')) else None
        self.writer = MultiLogger(
            tb_writer,
            use_wandb=(cfg.logger in ('wandb','both')),
            wandb_project=cfg.wandb_project,
            wandb_entity=cfg.wandb_entity,
            wandb_run_name=cfg.run_name,
            wandb_dir=str(run_dir),
            wandb_config=cfg.__dict__,
            wandb_mode=cfg.wandb_mode,
        )
        self.grad_monitor = GradientMonitor(self.model, self.writer)

        # Grad cosines between tasks
        self._agg = GradDiagnosticsAggregator(task_names=["elbo", "img_adv"])
        self.global_step = 0

        # stable eval batch for viz
        self._tb_viz_batch = None

        # resume
        if cfg.resume:
            self._load_checkpoint(cfg.resume)

        # Save config for reproducibility
        with open(run_dir / "config.json", "w") as f:
            json.dump(cfg.__dict__, f, indent=2)

    def _beta_schedule(self, epoch: int) -> float:
        # simple ramp
        beta_min = float(self.cfg.beta_min)
        beta_max = float(self.cfg.beta)
        ramp = float(self.cfg.beta_ramp_epochs)
        if ramp <= 0:
            return beta_max
        p = max(0.0, min(1.0, epoch / ramp))
        return beta_min + (beta_max - beta_min) * p

    def _save_checkpoint(self, epoch: int):
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
        }
        # include optimizers if present
        if hasattr(self.model, "gen_optimizer"):
            ckpt["gen_optimizer"] = self.model.gen_optimizer.state_dict()
        if hasattr(self.model, "img_disc_optimizer"):
            ckpt["img_disc_optimizer"] = self.model.img_disc_optimizer.state_dict()
        if hasattr(self.model, "temp_disc_optimizer"):
            ckpt["temp_disc_optimizer"] = self.model.temp_disc_optimizer.state_dict()
        path = self.run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
        torch.save(ckpt, path)
        print(f"[ckpt] saved {path}")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"], strict=False)
        if "gen_optimizer" in ckpt and hasattr(self.model, "gen_optimizer"):
            self.model.gen_optimizer.load_state_dict(ckpt["gen_optimizer"])
        if "img_disc_optimizer" in ckpt and hasattr(self.model, "img_disc_optimizer"):
            self.model.img_disc_optimizer.load_state_dict(ckpt["img_disc_optimizer"])
        if "temp_disc_optimizer" in ckpt and hasattr(self.model, "temp_disc_optimizer"):
            self.model.temp_disc_optimizer.load_state_dict(ckpt["temp_disc_optimizer"])
        self.global_step = int(ckpt.get("global_step", 0))
        print(f"[ckpt] loaded {path} (epoch={ckpt.get('epoch', '?')}, step={self.global_step})")

    def _to01(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert images to [0,1] for display/metrics.
        Uses model.denormalize_generated_images when available.
        """
        if x is None:
            return x
        if x.min().item() >= -1e-3 and x.max().item() <= 1.0 + 1e-3:
            # already looks like [0,1]
            return x.clamp(0, 1)
        if hasattr(self.model, "denormalize_generated_images"):
            return self.model.denormalize_generated_images(x).clamp(0, 1)
        return ((x + 1.0) * 0.5).clamp(0, 1)

    @torch.no_grad()
    def tb_log_reconstruction(self, epoch: int, tag: str = "viz/reconstruction", max_frames: int = 8):
        if self.writer is None:
            return
        self.model.eval()
        if self._tb_viz_batch is None:
            self._tb_viz_batch = next(iter(self.eval_loader))
        batch = self._tb_viz_batch
        obs = batch["observations"].to(self.device)  # [B,T,C,H,W]
        act = batch["actions"].to(self.device)
        done = batch.get("done", None)
        if done is not None:
            done = done.to(self.device)

        vae_losses, out = self.model.compute_total_loss(obs, act, done, beta=self.cfg.beta, lambda_recon=self.cfg.lambda_recon)
        recon = out.get("reconstructions", None)  # [B,T,3,H,W] (likely)
        if recon is None:
            return

        # pick B=0
        gt01 = self._to01(obs[0, :max_frames, -3:])
        re01 = self._to01(recon[0, :max_frames, -3:])
        err = (gt01 - re01).abs().mean(dim=1, keepdim=True).repeat(1,3,1,1)

        grid = make_grid(torch.cat([gt01, re01, err], dim=0), nrow=max_frames, padding=2)
        self.writer.add_image(tag, grid, global_step=epoch)

        # metrics
        psnr = psnr_(gt01, re01).item()
        ssim = ssim_(gt01, re01).item()
        self.writer.add_scalar("metrics/recon_psnr", psnr, epoch)
        self.writer.add_scalar("metrics/recon_ssim", ssim, epoch)

    @torch.no_grad()
    def tb_log_future_prediction(self, epoch: int, tag: str = "viz/future_prediction"):
        if self.writer is None:
            return
        self.model.eval()
        if self._tb_viz_batch is None:
            self._tb_viz_batch = next(iter(self.eval_loader))
        batch = self._tb_viz_batch
        obs = batch["observations"].to(self.device)  # [B,T,C,H,W]
        act = batch["actions"].to(self.device)
        done = batch.get("done", None)
        if done is not None:
            done = done.to(self.device)

        T_ctx = min(self.cfg.rollout_context_frames, obs.shape[1]-1)
        horizon = min(self.cfg.rollout_horizon, obs.shape[1]-T_ctx)
        if horizon <= 0:
            return

        dbg = self.model.generate_future_sequence(
            initial_obs=obs[:, :T_ctx],
            actions=act,
            horizon=horizon,
            top_temperature=getattr(self.model, "rollout_top_temperature", 0.2),
            decoder_temperature=getattr(self.model, "rollout_decoder_temperature", 0.4),
            dones=done,
            decode_mode=getattr(self.model, "rollout_decode_mode", "mean"),
            grad=False,
        )

        pred = dbg.get("vae_future", None)  # [B,H,C,H,W]
        if pred is None:
            return

        gt_future = obs[:, T_ctx:T_ctx+horizon, -3:]
        pred01 = self._to01(pred[:, :, -3:])
        gt01 = self._to01(gt_future)

        # metrics on first sample
        psnr = psnr_(gt01[0].reshape(-1,3,self.cfg.img_size,self.cfg.img_size), pred01[0].reshape(-1,3,self.cfg.img_size,self.cfg.img_size)).item()
        ssim = ssim_(gt01[0].reshape(-1,3,self.cfg.img_size,self.cfg.img_size), pred01[0].reshape(-1,3,self.cfg.img_size,self.cfg.img_size)).item()
        self.writer.add_scalar("metrics/pred_psnr", psnr, epoch)
        self.writer.add_scalar("metrics/pred_ssim", ssim, epoch)

        # visualize: [context | gt_future | pred_future] as one big grid (rows)
        ctx01 = self._to01(obs[0, :T_ctx, -3:])
        gt_row = gt01[0]
        pr_row = pred01[0]
        # pad to same length with zeros if needed
        # build panel: 3 rows (ctx, gt, pred)
        nrow = max(T_ctx, horizon)
        # create tensors with equal count by padding
        def _pad(seq: torch.Tensor, L: int) -> torch.Tensor:
            if seq.shape[0] == L:
                return seq
            pad = torch.zeros((L-seq.shape[0],) + tuple(seq.shape[1:]), device=seq.device, dtype=seq.dtype)
            return torch.cat([seq, pad], dim=0)
        ctxp = _pad(ctx01, nrow)
        gtp = _pad(gt_row, nrow)
        prp = _pad(pr_row, nrow)

        grid = make_grid(torch.cat([ctxp, gtp, prp], dim=0), nrow=nrow, padding=2)
        self.writer.add_image(tag, grid, global_step=epoch)

    @torch.no_grad()
    def tb_log_warp_panel(self, epoch: int, b: int = 0, t: int = 1, tag: str = "viz/warp_panel", max_flow_frac: float = 0.25):
        """
        Ported from dmc_vb trainer. Uses model.forward_sequence capture_flow_ctx + image_warp for robust flow visualization.
        """
        if self.writer is None:
            return
        self.model.eval()

        if self._tb_viz_batch is None:
            self._tb_viz_batch = next(iter(self.eval_loader))

        batch = self._tb_viz_batch
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        dones = batch.get("done", None)
        if dones is not None:
            dones = dones.to(self.device)

        B, T, C, H, W = observations.shape
        b = int(max(0, min(b, B - 1)))
        t = int(max(1, min(t, T - 1)))

        obs_short  = observations[:, :t+1]
        act_short  = actions[:, :t+1]
        done_short = dones[:, :t+1] if dones is not None else None

        out = self.model.forward_sequence(obs_short, act_short, done_short, capture_flow_ctx=(b, t))

        x_prev = observations[b, t-1:t]
        x_tgt  = observations[b, t:t+1]
        x_prev01 = self._to01(x_prev[:, -3:]).float()
        x_tgt01  = self._to01(x_tgt[:, -3:]).float()

        e_prev = self.model.canny(x_prev01).clamp(0,1).float() if hasattr(self.model, "canny") else torch.zeros((1,1,H,W), device=self.device)
        e_tgt  = self.model.canny(x_tgt01 ).clamp(0,1).float() if hasattr(self.model, "canny") else torch.zeros((1,1,H,W), device=self.device)

        ctx = out.get("captured_flow_ctx", None)
        if ctx is None:
            flow_ctx_dim = getattr(self.model, "flow_ctx_dim", 64)
            ctx = torch.zeros((1, flow_ctx_dim, H, W), device=self.device, dtype=torch.float32)
        else:
            ctx = ctx.to(device=self.device, dtype=torch.float32)
            if (ctx.shape[-2], ctx.shape[-1]) != (H, W):
                ctx = F.interpolate(ctx, size=(H, W), mode="bilinear", align_corners=False)

        max_flow_full = max(H, W) * float(max_flow_frac)

        flow_bw = out.get("captured_flow_bw", None)
        if flow_bw is None:
            # fallback
            flow_state0 = torch.zeros((1, 128, H, W), device=self.device, dtype=torch.float32)
            a_t = actions[b:b+1, t - 1]
            if hasattr(self.model, "_predict_flow_one_step"):
                flow_bw, _ = self.model._predict_flow_one_step(
                    x01=x_tgt01, e=e_tgt, ctx=ctx, a_t=a_t,
                    state=flow_state0, first=(t == 1), direction="bw"
                )
            else:
                flow_bw = torch.zeros((1,2,H,W), device=self.device, dtype=torch.float32)
        else:
            flow_bw = flow_bw.to(device=self.device, dtype=torch.float32)

        # warp prev -> current
        x_prev_to_cur = image_warp(x_prev01, flow_bw).clamp(0,1)
        rgb_err = (x_prev_to_cur - x_tgt01).abs().mean(dim=1, keepdim=True).repeat(1,3,1,1)

        flow_rgb = flow_to_hsv_rgb(flow_bw, max_flow=max_flow_full)
        # panel: prev | tgt | warped | flow | err
        panel = torch.cat([x_prev01, x_tgt01, x_prev_to_cur, flow_rgb, rgb_err], dim=0)
        grid = make_grid(panel, nrow=5, padding=2)
        self.writer.add_image(tag, grid, global_step=epoch)

    @torch.no_grad()
    def tb_log_latent_tsne(self, epoch: int, tag: str = "viz/latent_tsne"):
        """
        Saves to a temporary PNG then logs to TensorBoard.
        """
        if self.writer is None:
            return
        self.model.eval()
        tmp_path = self.run_dir / f"latent_tsne_epoch_{epoch:04d}.png"
        try:
            visualize_dpgmm_clustering(
                model=self.model,
                dataloader=self.eval_loader,
                device=self.device,
                max_batches=min(15, self.cfg.eval_max_batches),
                max_samples=5000,
                perplexity=30.0,
                save_path=str(tmp_path),
                image_level=True,
                t_select=4,
                use_rnn_context=True,
                tsne_dims=3,
            )
            # load image and log
            
            im = PIL.Image.open(tmp_path).convert("RGB")
            arr = torch.from_numpy(np.array(im)).permute(2,0,1).float() / 255.0
            self.writer.add_image(tag, arr, global_step=epoch)
        except Exception as e:
            print(f"[warn] latent TSNE visualization failed: {e}")

    def _log_scalars(self, prefix: str, metrics: Dict[str, Any], step: int):
        for k, v in metrics.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = float(v.detach().cpu().item())
            elif isinstance(v, (np.floating, np.integer)):
                v = float(v)
            elif isinstance(v, bool):
                v = float(v)
            elif isinstance(v, (int, float)):
                v = float(v)
            else:
                continue
            self.writer.add_scalar(f"{prefix}/{k}", v, step)

    def train(self):
        print(f"[run] logs: {self.run_dir}")
        for epoch in tqdm(range(1, self.cfg.n_epochs + 1), desc='Epoch', dynamic_ncols=True):
            self.model.train()
            if hasattr(self.model, "set_epoch"):
                try:
                    self.model.set_epoch(epoch)
                except Exception:
                    pass

            beta_t = self._beta_schedule(epoch)

            # epoch stats
            epoch_metrics: Dict[str, List[float]] = {}
            t0 = time.time()

            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Train e{epoch}', leave=False, dynamic_ncols=True)

            for batch_idx, batch in train_pbar:
                obs = batch["observations"].to(self.device, non_blocking=True)
                act = batch["actions"].to(self.device, non_blocking=True)
                done = batch.get("done", None)
                if done is not None:
                    done = done.to(self.device, non_blocking=True)

                # Stable sequence IDs let the DPGMM top-prior replay buffer recognize
                # posterior samples from the same RoboMimic demo / start time.
                seq_ids = None
                if "episode_idx" in batch and "start_idx" in batch:
                    episode_idx = batch["episode_idx"].to(self.device, dtype=torch.long, non_blocking=True)
                    start_idx = batch["start_idx"].to(self.device, dtype=torch.long, non_blocking=True)
                    abs_t0 = start_idx + int(self.cfg.frame_stack) - 1
                    seq_ids = torch.bitwise_left_shift(episode_idx.to(torch.int64), 32) + abs_t0.to(torch.int64)

                # Training step happens inside the model (optimizers inside model).
                # collect_top_buffer=True is important: otherwise the ConditionalTopDPGMM
                # never receives posterior top-latent samples for epoch-end VI updates.
                losses = self.model.training_step_sequence(
                    observations=obs,
                    actions=act,
                    dones=done,
                    beta=beta_t,
                    n_critic=self.cfg.n_critic,
                    lambda_img=self.cfg.lambda_img,
                    lambda_recon=self.cfg.lambda_recon,
                    batch_idx=self.global_step,
                    collect_top_buffer=True,
                    seq_ids=seq_ids,
                )

                # tqdm postfix (cheap live feedback)
                try:
                    if isinstance(losses, dict) and "total_gen_loss" in losses:
                        _l = losses["total_gen_loss"]
                        if isinstance(_l, torch.Tensor):
                            _l = float(_l.detach().cpu().item())
                        train_pbar.set_postfix({"loss": f"{_l:.4f}", "beta": f"{beta_t:.3f}"})
                except Exception:
                    pass

                # Track losses
                for k, v in losses.items():
                    if isinstance(v, torch.Tensor):
                        v = float(v.detach().cpu().item())
                    elif isinstance(v, (int, float, np.floating, np.integer)):
                        v = float(v)
                    else:
                        continue
                    epoch_metrics.setdefault(k, []).append(v)
                    # also log per-step for main keys
                    if k in ("total_gen_loss", "total_vae_loss", "recon_loss", "kl_z", "hierarchical_kl", "warp_sanity_total", "img_disc_loss", "temp_disc_loss"):
                        self.writer.add_scalar(f"train_step/{k}", v, self.global_step)

                # component gradient norms (after backward inside training_step_sequence)
                try:
                    comp = self.grad_monitor.compute_component_gradients(update_step=False, normalize="rms_elem")
                    for ck, cv in comp.items():
                        self.writer.add_scalar(f"grads/component_rms/{ck}", cv, self.global_step)
                except Exception:
                    pass

                # optional gradient cosine diagnostics every N steps
                if self.cfg.grad_diag_every_steps > 0 and (self.global_step % self.cfg.grad_diag_every_steps == 0):
                    try:
                        # Recompute a graph for diagnostics
                        # Recompute a graph for diagnostics (separate from the model's internal optim step)
                        vae_losses, outputs = self.model.compute_total_loss(obs, act, done, beta=self.cfg.beta, lambda_recon=self.cfg.lambda_recon)
                        elbo_loss = (
                            self.cfg.lambda_recon * vae_losses["recon_loss"]
                            + beta_t * vae_losses["kl_z"]
                            + beta_t * vae_losses["hierarchical_kl"]
                        )
                        adv_proxy = torch.zeros((), device=self.device)
                        if hasattr(self.model, "image_discriminator") and self.cfg.lambda_img > 0.0:
                            recon = outputs.get("reconstructions", None)
                            latents = outputs.get("latents", None)
                            h_seq = outputs.get("hidden_states", None)
                            if recon is not None and latents is not None and h_seq is not None:
                                # Build conditional discriminator input (matches dmc_vb trainer pattern)
                                z_seq = torch.cat([latents, h_seq], dim=-1)
                                # Freeze discriminator params so gradients represent generator pressure
                                disc_params = list(self.model.image_discriminator.parameters())
                                for p in disc_params:
                                    p.requires_grad_(False)
                                try:
                                    out_disc = self.model.image_discriminator(recon, z=z_seq, return_features=True)
                                    if isinstance(out_disc, (tuple, list)):
                                        score = out_disc[0]
                                    else:
                                        score = out_disc
                                    adv_proxy = -score.mean()
                                except Exception:
                                    try:
                                        score = self.model.image_discriminator(recon)
                                        adv_proxy = -score.mean()
                                    except Exception:
                                        adv_proxy = torch.zeros((), device=self.device)
                                finally:
                                    for p in disc_params:
                                        p.requires_grad_(True)
                        task_losses = [elbo_loss, self.cfg.lambda_img * adv_proxy]
                        named_params = list(self.model.named_parameters())
                        self._agg.update(
                            self.model,
                            shared_repr=None,
                            task_losses=task_losses,
                            named_params=named_params,
                            param_patterns=None,
                        )
                        cos = self._agg.get_cosine_similarity()
                        norms = self._agg.get_gradient_norms()
                        if cos is not None:
                            self.writer.add_scalar("grad_diag/cos_elbo_vs_imgadv", float(cos), self.global_step)
                        if norms is not None and len(norms) >= 2:
                            self.writer.add_scalar("grad_diag/norm_elbo", float(norms[0]), self.global_step)
                            self.writer.add_scalar("grad_diag/norm_imgadv", float(norms[1]), self.global_step)
                    except Exception as e:
                        print(f"[warn] grad diagnostics failed: {e}")

                # periodic image logs
                if self.cfg.log_images_every_steps > 0 and (self.global_step % self.cfg.log_images_every_steps == 0):
                    try:
                        self.tb_log_reconstruction(epoch, tag="viz/reconstruction_step", max_frames=min(8, self.cfg.sequence_length))
                        self.tb_log_future_prediction(epoch, tag="viz/future_step")
                        self.tb_log_warp_panel(epoch, tag="viz/warp_step", t=min(2, self.cfg.sequence_length-1))
                    except Exception as e:
                        print(f"[warn] image logging failed: {e}")

                self.global_step += 1

            # Epoch-end DPGMM update. This mirrors the DMC-VB trainer:
            # first collect posterior top latents during teacher-forced training,
            # then fit/refresh the frozen conditional top prior before eval/rollout.
            try:
                if hasattr(self.model, "top_replay_buffer") and hasattr(self.model, "refresh_top_prior_from_buffer"):
                    top_buffer_size = len(self.model.top_replay_buffer)
                    epoch_metrics.setdefault("top_buffer_size", []).append(float(top_buffer_size))
                    do_refresh = (
                        top_buffer_size > 0
                        and int(getattr(self.cfg, "refresh_top_prior_every", 1)) > 0
                        and (epoch % int(getattr(self.cfg, "refresh_top_prior_every", 1)) == 0)
                    )
                    if do_refresh:
                        self.model.refresh_top_prior_from_buffer(
                            batch_size=int(getattr(self.cfg, "dpgmm_outer_batch_size", 256)),
                            n_laps=int(getattr(self.cfg, "dpgmm_outer_n_laps", 4)),
                        )
                    if hasattr(self.model, "top_prior_model"):
                        epoch_metrics.setdefault("top_prior_K", []).append(float(self.model.top_prior_model.K))
            except Exception as e:
                print(f"[warn] top-prior refresh failed: {e}")

            # epoch average logs
            avg = {k: float(np.mean(v)) for k, v in epoch_metrics.items() if len(v) > 0}
            self._log_scalars("train_epoch", avg, epoch)

            dt = time.time() - t0
            self.writer.add_scalar("time/epoch_seconds", dt, epoch)
            print(f"[epoch {epoch:04d}] train: total_gen_loss={avg.get('total_gen_loss', float('nan')):.4f} beta={beta_t:.3f} time={dt:.1f}s")
            if bool(getattr(self.cfg, "use_schedulers", True)):
                try:
                    if hasattr(self.model, "img_disc_scheduler") and getattr(self.model, "img_disc_scheduler") is not None:
                        if "img_disc_loss" in avg:
                            self.model.img_disc_scheduler.step(avg["img_disc_loss"])

                    if hasattr(self.model, "gen_scheduler") and getattr(self.model, "gen_scheduler") is not None:
                        if "total_gen_loss" in avg:
                            self.model.gen_scheduler.step(avg["total_gen_loss"])
                except Exception as e:
                    print(f"[warn] scheduler step failed: {e}")

            # Eval + viz
            if self.cfg.eval_every > 0 and (epoch % self.cfg.eval_every == 0):
                self.evaluate(epoch)

            if self.cfg.viz_every > 0 and (epoch % self.cfg.viz_every == 0):
                try:
                    self.tb_log_reconstruction(epoch)
                    self.tb_log_future_prediction(epoch)
                    self.tb_log_warp_panel(epoch, tag="viz/warp_panel", t=min(3, self.cfg.sequence_length-1))
                    self.tb_log_latent_tsne(epoch)
                except Exception as e:
                    print(f"[warn] epoch viz failed: {e}")

            if self.cfg.save_every > 0 and (epoch % self.cfg.save_every == 0):
                self._save_checkpoint(epoch)

        print("[done] training finished")
        self.writer.flush()
        self.writer.close()

    @torch.no_grad()
    def evaluate(self, epoch: int):
        self.model.eval()
        metrics: Dict[str, List[float]] = {}
        # also compute psnr/ssim for recon and predictions on a few batches
        recon_psnrs, recon_ssims = [], []
        pred_psnrs, pred_ssims = [], []

        eval_pbar = tqdm(enumerate(self.eval_loader), total=min(len(self.eval_loader), self.cfg.eval_max_batches), desc=f'Eval e{epoch}', leave=False, dynamic_ncols=True)

        for bi, batch in eval_pbar:
            if bi >= self.cfg.eval_max_batches:
                break
            obs = batch["observations"].to(self.device, non_blocking=True)
            act = batch["actions"].to(self.device, non_blocking=True)
            done = batch.get("done", None)
            if done is not None:
                done = done.to(self.device, non_blocking=True)

            vae_losses, out = self.model.compute_total_loss(obs, act, done, beta=self.cfg.beta, lambda_recon=self.cfg.lambda_recon)
            for k, v in vae_losses.items():
                if isinstance(v, torch.Tensor):
                    v = float(v.detach().cpu().item())
                metrics.setdefault(k, []).append(v)

            recon = out.get("reconstructions", None)
            if recon is not None:
                gt01 = self._to01(obs[:, :, -3:]).reshape(-1,3,self.cfg.img_size,self.cfg.img_size)
                re01 = self._to01(recon[:, :, -3:]).reshape(-1,3,self.cfg.img_size,self.cfg.img_size)
                recon_psnrs.append(float(psnr_(gt01, re01).cpu().item()))
                recon_ssims.append(float(ssim_(gt01, re01).cpu().item()))

            # future pred metrics (cheap): first sample in batch
            T_ctx = min(self.cfg.rollout_context_frames, obs.shape[1]-1)
            horizon = min(self.cfg.rollout_horizon, obs.shape[1]-T_ctx)
            if horizon > 0:
                dbg = self.model.generate_future_sequence(
                    initial_obs=obs[:, :T_ctx],
                    actions=act,
                    horizon=horizon,
                    top_temperature=getattr(self.model, "rollout_top_temperature", 0.2),
                    decoder_temperature=getattr(self.model, "rollout_decoder_temperature", 0.4),
                    dones=done,
                    decode_mode=getattr(self.model, "rollout_decode_mode", "mean"),
                    grad=False,
                )
                pred = dbg.get("vae_future", None)
                if pred is not None:
                    gt_future = obs[:, T_ctx:T_ctx+horizon, -3:]
                    gt01 = self._to01(gt_future).reshape(-1,3,self.cfg.img_size,self.cfg.img_size)
                    pr01 = self._to01(pred[:, :, -3:]).reshape(-1,3,self.cfg.img_size,self.cfg.img_size)
                    pred_psnrs.append(float(psnr_(gt01, pr01).cpu().item()))
                    pred_ssims.append(float(ssim_(gt01, pr01).cpu().item()))

        avg = {k: float(np.mean(v)) for k, v in metrics.items() if len(v) > 0}
        if recon_psnrs:
            avg["recon_psnr"] = float(np.mean(recon_psnrs))
            avg["recon_ssim"] = float(np.mean(recon_ssims))
        if pred_psnrs:
            avg["pred_psnr"] = float(np.mean(pred_psnrs))
            avg["pred_ssim"] = float(np.mean(pred_ssims))

        self._log_scalars("eval", avg, epoch)
        print(f"[epoch {epoch:04d}] eval: recon_psnr={avg.get('recon_psnr', float('nan')):.2f} pred_psnr={avg.get('pred_psnr', float('nan')):.2f}")


# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", type=str, required=True, help="Path to RoboMimic HDF5 OR a directory containing HDF5s (can be comma-separated). Example: .../robomimic_v1.5/v1.5 or .../ph/demo_v15.hdf5")
    p.add_argument("--inspect_only", action="store_true", help="Only print HDF5 structure and exit")

    # If --hdf5 is a directory (or comma-separated list of dirs), the trainer can scan and use multiple HDF5 files.
    p.add_argument("--no_recursive", action="store_true", help="If --hdf5 is a directory, do NOT scan recursively.")
    p.add_argument("--include_low_dim", action="store_true", help="Also include non-demo files (e.g., low_dim_*.hdf5) when scanning directories.")
    p.add_argument("--max_scan_files", type=int, default=0, help="Limit number of scanned HDF5 files (0 = no limit).")

    p.add_argument("--use_zoom_aug", action="store_true", help="Enable random zoom-crop augmentation (consistent across frames).")
    p.add_argument("--zoom_prob", type=float, default=0.6)
    p.add_argument("--zoom_area_min", type=float, default=0.6)
    p.add_argument("--zoom_area_max", type=float, default=0.85)
    p.add_argument("--no_ema", action="store_true", help="Disable EMA for VDVAE params (EMA is recommended).")
    p.add_argument("--no_schedulers", action="store_true", help="Disable LR schedulers inside the model, if present.")

    p.add_argument("--obs_keys", type=str, default="", help="Comma-separated obs image keys under /data/<demo>/obs. If empty, auto-detect.")
    p.add_argument("--sequence_length", type=int, default=16)
    p.add_argument("--frame_stack", type=int, default=1)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--train_fraction", type=float, default=0.95)
    p.add_argument("--max_sequences_per_demo", type=int, default=0, help="0 means no cap; else cap per demo.")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_epochs", type=int, default=200)

    p.add_argument("--log_dir", type=str, default="./runs")

    p.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb", "both"], help="Where to log metrics/images.")
    p.add_argument("--wandb_project", type=str, default="D2E")
    p.add_argument("--wandb_entity", type=str, default="", help="Optional wandb entity / team.")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    p.add_argument("--run_name", type=str, default="robomimic_run")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--viz_every", type=int, default=1)
    p.add_argument("--log_images_every_steps", type=int, default=500)
    p.add_argument("--grad_diag_every_steps", type=int, default=500)
    p.add_argument("--eval_max_batches", type=int, default=25)

    # model hyperparams
    p.add_argument("--max_components", type=int, default=40)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--prior_alpha", type=float, default=1.0)
    p.add_argument("--prior_beta", type=float, default=10.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--img_disc_layers", type=int, default=3)
    p.add_argument("--disc_num_heads", type=int, default=4)
    p.add_argument("--use_dwa", action="store_true")

    # loss weights
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--beta_min", type=float, default=0.0)
    p.add_argument("--beta_ramp_epochs", type=int, default=50)
    p.add_argument("--lambda_recon", type=float, default=1.0)
    p.add_argument("--lambda_img", type=float, default=0.25)
    p.add_argument("--n_critic", type=int, default=3)

    # rollout
    p.add_argument("--rollout_context_frames", type=int, default=4)
    p.add_argument("--rollout_horizon", type=int, default=8)

    # DPGMM top-prior refresh. Keep these small for a smoke test.
    p.add_argument("--dpgmm_outer_batch_size", type=int, default=256)
    p.add_argument("--dpgmm_outer_n_laps", type=int, default=4)
    p.add_argument("--refresh_top_prior_every", type=int, default=1)

    p.add_argument("--resume", type=str, default="")
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()

def main():
    args = parse_args()
    if args.inspect_only:
        paths = _expand_hdf5_inputs(
            args.hdf5,
            recursive=(not args.no_recursive),
            prefer_demo_files=(not args.include_low_dim),
            include_low_dim=bool(args.include_low_dim),
            max_files=(None if int(args.max_scan_files or 0) <= 0 else int(args.max_scan_files)),
        )
        if len(paths) == 0:
            print(f"No HDF5 files found from: {args.hdf5}")
            return
        print(f"[inspect] Found {len(paths)} HDF5 file(s). Showing up to first 10.")
        for pth in paths[:10]:
            try:
                inspect_robomimic_hdf5(pth, verbose=True)
            except Exception as e:
                print(f"[inspect] failed on {pth}: {e}")
        return

    obs_keys = [k.strip() for k in args.obs_keys.split(",") if k.strip()] if args.obs_keys else None

    cfg = TrainConfig(
        hdf5=args.hdf5,
        obs_keys=obs_keys,
        scan_recursive=(not args.no_recursive),
        prefer_demo_files=(not args.include_low_dim),
        include_low_dim=bool(args.include_low_dim),
        max_scan_files=int(args.max_scan_files),
        sequence_length=args.sequence_length,
        frame_stack=args.frame_stack,
        img_size=args.img_size,
        train_fraction=args.train_fraction,
        max_sequences_per_demo=(None if args.max_sequences_per_demo == 0 else int(args.max_sequences_per_demo)),
        seed=args.seed,

        use_zoom_aug=bool(args.use_zoom_aug),
        zoom_prob=float(args.zoom_prob),
        zoom_area_min=float(args.zoom_area_min),
        zoom_area_max=float(args.zoom_area_max),
        dpgmm_outer_batch_size=int(args.dpgmm_outer_batch_size),
        dpgmm_outer_n_laps=int(args.dpgmm_outer_n_laps),
        refresh_top_prior_every=int(args.refresh_top_prior_every),
        use_ema=(not bool(args.no_ema)),
        use_schedulers=(not bool(args.no_schedulers)),

        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_epochs=args.n_epochs,
        logger=args.logger,
        log_dir=args.log_dir,
        run_name=args.run_name,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        save_every=args.save_every,
        eval_every=args.eval_every,
        viz_every=args.viz_every,
        log_images_every_steps=args.log_images_every_steps,
        grad_diag_every_steps=args.grad_diag_every_steps,
        eval_max_batches=args.eval_max_batches,

        max_components=args.max_components,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        prior_alpha=args.prior_alpha,
        prior_beta=args.prior_beta,
        dropout=args.dropout,
        img_disc_layers=args.img_disc_layers,
        disc_num_heads=args.disc_num_heads,
        use_dynamic_weight_average=args.use_dwa,

        beta=args.beta,
        beta_min=args.beta_min,
        beta_ramp_epochs=args.beta_ramp_epochs,
        lambda_recon=args.lambda_recon,
        lambda_img=args.lambda_img,
        n_critic=args.n_critic,

        rollout_context_frames=args.rollout_context_frames,
        rollout_horizon=args.rollout_horizon,

        resume=(args.resume if args.resume else None),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = RoboMimicTrainer(cfg, device)
    trainer.train()

if __name__ == "__main__":
    main()
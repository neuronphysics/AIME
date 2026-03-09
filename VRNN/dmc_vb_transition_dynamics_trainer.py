import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random, math, cv2, gc, os, h5py, zlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import wandb
from tqdm import tqdm
from collections import defaultdict
import time, contextlib
import tensorflow as tf
import argparse
import wandb

try:
    # completely disable TF GPUs (safest, simplest)
    tf.config.set_visible_devices([], 'GPU')
    print("[TF] Using CPU only (GPUs hidden from TensorFlow).")
except Exception as e:
    print("[TF] Could not hide GPUs, trying memory_growth:", e)
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[TF] Enabled memory growth on GPUs.")
    except Exception as e2:
        print("[TF] Could not set memory growth:", e2)

from torch.utils.tensorboard import SummaryWriter
from VRNN.dpgmm_stickbreaking_prior_vrnn import DPGMMVariationalRecurrentAutoencoder
"""Download data :gsutil -m cp -r gs://dmc_vision_benchmark/dmc_vision_benchmark/locomotion/humanoid_walk/medium ./transition_data/dmc_vb/humanoid_walk/"""
from torch.utils.data._utils.collate import default_collate
from VRNN.grad_diagnostics import GradDiagnosticsAggregator  
from contextlib import contextmanager
import matplotlib
from pathlib import Path
from distutils.util import strtobool
from VRNN.visualize_latent_clusters import visualize_dpgmm_clustering
from VRNN.warp import image_warp
def str2bool(v):
    return bool(strtobool(v))
    
os.environ["MPLBACKEND"] = "Agg" 
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent

print("Script dir:", SCRIPT_DIR)
print("Parent dir:", PARENT_DIR)
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
plt.ioff()

def print_vdvae_edge_report(model):
    H = model.vdvae.H
    min_res = int(getattr(H, "edge_condition_min_res", 32))
    print(f"\n[VDVAE edge report] use_edge_conditioning={getattr(H,'use_edge_conditioning',False)} "
          f"edge_condition_min_res={min_res}\n")

    for i, blk in enumerate(model.vdvae.decoder.dec_blocks):
        res = int(getattr(blk, "base", -1))
        is_top = bool(getattr(blk, "is_top", False))
        edge_on = bool(getattr(blk, "use_edge_conditioning", False))
        edge_ch = int(getattr(blk, "edge_channels", -1))

        # width at this resolution (DecBlock stores widths dict)
        width = blk.widths[res] if hasattr(blk, "widths") and res in blk.widths else None

        enc_in  = blk.enc.c1.in_channels
        prior_in = blk.prior.c1.in_channels


        print(f"[{i:02d}] res={res:>3} is_top={is_top} edge={edge_on} edge_ch={edge_ch} enc_in={enc_in} prior_in={prior_in}")

        # Strong sanity checks: NO edges when res < 32
        if res < min_res:
            if width is not None and enc_in != 2 * width:
                print(f"   !!! ERROR: enc_in should be {2*width} (width*2) when edge is OFF")
            # prior_in may include temporal width if one uses temporal priors at this res
            # so only check the ">= width" and "not including edges" aspect:
            if width is not None and prior_in < width:
                print(f"   !!! ERROR: prior_in looks too small (expected at least width={width})")

    print("")


@contextmanager
def disable_all_checkpoint_modules(model: torch.nn.Module):
    """
    Temporarily sets .use_checkpoint = False for all submodules of `model`
    that define this attribute, then restores the original flags.
    """
    modules = []
    flags = []

    for m in model.modules():
        if hasattr(m, "use_checkpoint"):
            modules.append(m)
            flags.append(m.use_checkpoint)
            m.use_checkpoint = False

    try:
        yield
    finally:
        for m, flag in zip(modules, flags):
            m.use_checkpoint = flag

def safe_collate(batch):
    all_keys = set().union(*(b.keys() for b in batch))
    out = {}
    for k in all_keys:
        vals = [b[k] for b in batch if k in b]
        if len(vals) == len(batch):
            out[k] = default_collate(vals)
        else:
            # keep as a list aligned with the batch (None where missing)
            aligned = [(b[k] if k in b else None) for b in batch]
            out[k] = aligned
    return out



class HumanoidAwareZoomTransform:
    """
    Zoom-crop augmentation for sequences shaped [T, C, H, W].


    """

    def __init__(
        self,
        zoom_prob: float = 0.5,
        area_range: Tuple[float, float] = (0.6, 0.85),
        center_key: Optional[str] = None,            # e.g., 'attn_center' (normalized [-1,1] xy)
        interpolation: str = "cubic",
        # Corner placement inside the crop (how close the humanoid should be to a corner)
        corner_min: float = 0.70,                    # 0.70 -> 70% toward corner inside the crop
        corner_max: float = 0.85,                    # 0.85 -> 85% toward corner inside the crop
    ):
        assert 0.0 <= zoom_prob <= 1.0
        assert 0.0 < area_range[0] <= area_range[1] <= 1.0
        assert 0.5 <= corner_min < corner_max <= 0.95, "Corner range should be inside (0.5, 1.0)"

        self.zoom_prob = float(zoom_prob)
        self.area_range = area_range
        self.center_key = center_key
        self.corner_min = float(corner_min)
        self.corner_max = float(corner_max)

        self._interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear":  cv2.INTER_LINEAR,
            "cubic":   cv2.INTER_CUBIC,
            "area":    cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        self._cv2_interp = self._interp_map.get(interpolation, cv2.INTER_CUBIC)


    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() > self.zoom_prob:
            return sample

        imgs = sample["observations"]  # [T, C, H, W]
        assert imgs.dim() == 4, f"`observations` must be [T,C,H,W], got {tuple(imgs.shape)}"
        T, C, H, W = imgs.shape
        device, orig_dtype = imgs.device, imgs.dtype

        # 1) Choose crop size via area fraction
        area_frac = random.uniform(*self.area_range)             # keep this fraction of the area
        side_scale = math.sqrt(area_frac)                        # square crop
        crop_h = max(1, int(round(H * side_scale)))
        crop_w = max(1, int(round(W * side_scale)))
        sx = crop_w / float(W)                                   # width kept (normalized)
        sy = crop_h / float(H)                                   # height kept (normalized)

        # 2) Get humanoid center (normalized xy in [-1,1]); default to (0,0) if missing
        h_xy = self._extract_humanoid_center(sample)

        # 3) Pick a crop center so the humanoid appears near a *corner* inside the crop
        center_norm = self._choose_center_for_corner(h_xy, sx, sy)

        # 4) Convert normalized center to pixel box; clamp to image bounds
        x0, y0, x1, y1 = self._crop_box_from_center(center_norm, W, H, crop_w, crop_h)

        # 5) Crop+resize the SAME box for all frames/channels (OpenCV path on CPU)
        zoomed = self._crop_and_resize_sequence(imgs, T, C, H, W, x0, y0, x1, y1)

        # 6) Back to torch with original dtype/device
        sample["observations"] = torch.from_numpy(zoomed).to(device=device, dtype=orig_dtype)

        # 7) Optional debug metadata
        sample["zoom_meta"] = {
            "area_frac": float(area_frac),
            "humanoid_norm_xy": h_xy.clone(),
            "center_norm_xy": center_norm.clone(),
            "box_xyxy": torch.tensor([x0, y0, x1, y1]),
        }
        return sample


    def _extract_humanoid_center(self, sample: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Returns tensor [2] in normalized coords [-1,1], best estimate of humanoid center.
        """
        if self.center_key and (self.center_key in sample):
            ctr = sample[self.center_key]
            # Accept [2], [T,2], or [...,2]; reduce over time/first dims if needed.
            if ctr.ndim >= 2:
                ctr = ctr.view(-1, 2).mean(0)
            ctr = ctr.detach().to("cpu").float()
            ctr = torch.clamp(ctr, -1.0, 1.0)
        else:
            ctr = torch.tensor([0.0, 0.0], dtype=torch.float32)   # assume near center if unknown
        return ctr

    def _choose_center_for_corner(
        self,
        humanoid_xy: torch.Tensor,  # [2], normalized
        sx: float,
        sy: float,
    ) -> torch.Tensor:
        """
        Choose the CROP CENTER (normalized) so that the humanoid appears near a corner INSIDE the crop.

        """
        # Pick a random corner and a target proximity within [corner_min, corner_max].
        ux = random.uniform(self.corner_min, self.corner_max)
        uy = random.uniform(self.corner_min, self.corner_max)
        sign_x = random.choice([-1.0, 1.0])  # left/right
        sign_y = random.choice([-1.0, 1.0])  # top/bottom

        # Edge of crop along x corresponds to ±sx in normalized image coordinates.
        # (Same for y with sy.) Map desired in-crop corner position to normalized offset.
        dx = sign_x * (2.0 * ux - 1.0) * sx
        dy = sign_y * (2.0 * uy - 1.0) * sy

        # To place the humanoid near that corner inside the crop, shift the CROP center oppositely.
        cx = float(humanoid_xy[0] - dx)
        cy = float(humanoid_xy[1] - dy)

        # Valid crop centers are within ±(1 - s) so the crop stays in bounds.
        cx = max(-(1.0 - sx), min(1.0 - sx, cx))
        cy = max(-(1.0 - sy), min(1.0 - sy, cy))

        return torch.tensor([cx, cy], dtype=torch.float32)

    @staticmethod
    def _crop_box_from_center(
        center_norm: torch.Tensor,
        W: int,
        H: int,
        crop_w: int,
        crop_h: int,
    ) -> Tuple[int, int, int, int]:
        """
        Convert normalized center [-1,1] to pixel crop box, clamped to image bounds.
        """
        cx = int((center_norm[0].item() + 1.0) * 0.5 * (W - 1))
        cy = int((center_norm[1].item() + 1.0) * 0.5 * (H - 1))

        x0 = max(0, min(W - crop_w, cx - crop_w // 2))
        y0 = max(0, min(H - crop_h, cy - crop_h // 2))
        x1, y1 = x0 + crop_w, y0 + crop_h
        return x0, y0, x1, y1

    def _crop_and_resize_sequence(
        self,
        imgs: torch.Tensor,  # [T, C, H, W]
        T: int,
        C: int,
        H: int,
        W: int,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> np.ndarray:
        """
        Crop same box for all frames and resize back to (W, H). Uses cv2 on CPU.
        """
        imgs_np = imgs.detach().to("cpu")
        # Convert to float32 for OpenCV; remember original dtype to restore later.
        if imgs_np.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            imgs_np = imgs_np.float()
        imgs_np = imgs_np.numpy().astype(np.float32, copy=False)  # [T, C, H, W]

        out = np.empty((T, C, H, W), dtype=np.float32)

        if C in (1, 3):  # fast path for grayscale/RGB
            for t in range(T):
                hwc = np.ascontiguousarray(imgs_np[t].transpose(1, 2, 0))  # [H, W, C]
                patch = hwc[y0:y1, x0:x1, :]
                resized = cv2.resize(patch, (W, H), interpolation=self._cv2_interp)
                out[t] = resized.transpose(2, 0, 1)
        else:  # generic per-channel
            for t in range(T):
                for c in range(C):
                    patch = np.ascontiguousarray(imgs_np[t, c, y0:y1, x0:x1])
                    out[t, c] = cv2.resize(patch, (W, H), interpolation=self._cv2_interp)
        out = np.clip(out, -1.0, 1.0)
        return out

class DMCVBInfo:
    """PyTorch version of DMC Vision Benchmark info"""
    
    DMC_INFO = {
        'humanoid': {
            'task_name': 'walk',
            'action_dim': 21,
            'state_dim': 67,
            'state_dim_no_velocity': 37,
            'cameras': ('pixels',),
        }
    }
    
    @staticmethod
    def get_action_dim(domain_name: str) -> int:
        return DMCVBInfo.DMC_INFO[domain_name]['action_dim']
    
    @staticmethod
    def get_state_dim(domain_name: str) -> int:
        return DMCVBInfo.DMC_INFO[domain_name]['state_dim']
    
    @staticmethod
    def get_camera_fields(domain_name: str, target_hidden: bool = False) -> tuple:
        return DMCVBInfo.DMC_INFO[domain_name]['cameras']

class TFRecordConverter:
    """Convert TFRecord files to PyTorch-compatible format"""
    @staticmethod
    def decode_zlib_observation(obs_bytes: bytes, target_shape=(64, 64, 3)) -> Optional[np.ndarray]:
        """
        
        This implementation embodies principles of information recovery through
        reversible compression transforms, revealing the underlying visual manifold.
        """
        
        try:
            # Phase 1: Decompress using zlib
            decompressed = zlib.decompress(obs_bytes)
            
            # Phase 2: Convert to numpy array
            obs_array = np.frombuffer(decompressed, dtype=np.uint8)
            
            # Phase 3: Reshape to canonical dimensions
            height, width, channels = target_shape
    
            return obs_array.reshape(height, width, channels)        
                
        except zlib.error as e:
            print(f"Zlib decompression failed: {e}")
            return None       
         
    @staticmethod
    def parse_tfrecord_episode(tfrecord_path: Path, action_dim:int) -> Dict[str, np.ndarray]:
        """
        Robust TFRecord parser acknowledging compressed observation encoding
        """
        
        dataset = tf.data.TFRecordDataset(str(tfrecord_path))
        
        # Accumulate all timesteps for complete episode reconstruction
        timesteps = []
        
        for idx, raw_record in enumerate(dataset):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            
            features = example.features.feature
            
            # Check if this is an episode boundary marker
            if 'episode_id' in features or 'episode_length' in features:
                continue  # Skip metadata records
            
            timestep_data = {}
            
            # Robust observation decoding with compression awareness
            if 'steps/observation/pixels' in features:
                obs_bytes = features['steps/observation/pixels'].bytes_list.value[0]
                if obs_bytes[:2] == b'\x78\x9c':  # zlib header
                    obs = TFRecordConverter.decode_zlib_observation(obs_bytes)
                    if obs is None:
                        print(f"Failed to decode observation at timestep {idx}, skipping...")
                        continue
                    else:
                        timestep_data['observation'] = obs
                else:
                    try:
                        # Utilize TensorFlow's format-agnostic decoder
                        # This leverages internal heuristics for JPEG/PNG/BMP/GIF detection
                        obs_tensor = tf.io.decode_image(obs_bytes, channels=3)
                        obs = obs_tensor.numpy()
                    
                        timestep_data['observation'] = obs
                    
                    except Exception as e:
                        print(f"Observation decoding failed for timestep: {e}")
                        # Critical: Continue to next timestep rather than attempting fallback
                        continue
            
            # Parse actions with robustness
            if 'steps/action' in features:
                action_bytes= features['steps/action'].bytes_list.value[0]
                if action_bytes[:2] == b'\x78\x9c':  # zlib header
                    try:
                        decompressed = zlib.decompress(action_bytes)
                        action_values = np.frombuffer(decompressed, dtype=np.float64)
                    except zlib.error as e:
                        print(f"Action decompression failed: {e}")
                        continue
                else:
                    action_values = np.frombuffer(action_bytes, dtype=np.float64)
                
                # Validate action dimension
                if len(action_values) == action_dim:
                    timestep_data['action'] = action_values
                else:
                    print(f"Unexpected action dimension: {len(action_values)}")
                    continue
            
            # Parse rewards
            if 'steps/reward' in features:
                reward_values = features['steps/reward'].float_list.value
                if reward_values:
                    timestep_data['reward'] = float(reward_values[0])
            
            # Parse termination flags
            for flag_name in ['is_first', 'is_last', 'is_terminal']:
                feature_key = f'steps/{flag_name}'
                if feature_key in features:
                    flag_values = features[feature_key].int64_list.value
                    if flag_values:
                        timestep_data[flag_name] = int(flag_values[0])
            
            # Only add complete timesteps
            if 'observation' in timestep_data and 'action' in timestep_data:
                timesteps.append(timestep_data)
        
        # Reconstruct episode from timesteps
        episode_data = {}
        
        if timesteps:
            # Stack observations
            episode_data['observation_pixels'] = np.stack([t['observation'] for t in timesteps])
            
            # Stack actions
            episode_data['action'] = np.stack([t['action'] for t in timesteps])
            
            # Stack rewards
            episode_data['reward'] = np.array([t.get('reward', 0.0) for t in timesteps], dtype=np.float32)
            
            # Construct done flags from termination indicators
            done_flags = []
            for i, t in enumerate(timesteps):
                is_done = t.get('is_last', 0) or t.get('is_terminal', 0)
                done_flags.append(float(is_done))
            
            episode_data['done'] = np.array(done_flags, dtype=np.float32)
            
            # Ensure at least one done flag
            if not np.any(episode_data['done']):
                episode_data['done'][-1] = 1.0
            
            episode_data['episode_length'] = len(timesteps)
            
            print(f"Parsed episode: {episode_data['episode_length']} timesteps, "
                f"observations shape: {episode_data['observation_pixels'].shape}")
        
        return episode_data
    

class DMCVBDataset(Dataset):
    """PyTorch Dataset for DMC Vision Benchmark data"""

    def __init__(
        self,
        data_dir: str,
        domain_name: str = 'humanoid',
        task_name: str = 'walk',
        policy_level: str = 'expert',
        split: str = 'train',
        sequence_length: int = 10,
        frame_stack: int = 3,
        img_height: int = 84,
        img_width: int = 84,
        normalize_images: bool = True,
        add_state: bool = False,
        add_rewards: bool = True,
        transform: Optional[callable] = None,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.domain_name = domain_name
        self.task_name = task_name
        self.policy_level = policy_level
        self.split = split
        self.sequence_length = sequence_length
        self.frame_stack = frame_stack
        self.img_height = img_height
        self.img_width = img_width
        self.normalize_images = normalize_images
        self.add_state = add_state
        self.add_rewards = add_rewards
        self.transform = transform

        # Get dataset info
        self.action_dim = DMCVBInfo.get_action_dim(domain_name)
        self.state_dim = DMCVBInfo.get_state_dim(domain_name)
        self.cameras = DMCVBInfo.get_camera_fields(domain_name)

        # Load episodes
        self.episodes = self._load_all_episodes()

        # Compute usable per-episode lengths (safe alignment)
        self.episode_lengths = self._compute_episode_lengths()
        self.min_episode_length = int(min(self.episode_lengths)) if len(self.episode_lengths) > 0 else 0

        # Create safe sequence indices per episode
        self.sequence_indices = self._create_sequence_indices()

        print(f"Loaded {len(self.episodes)} episodes")
        print(f"Minimum usable episode length (clipped to first done & array lengths): {self.min_episode_length}")
        print(f"Total sequences available: {len(self.sequence_indices)}")

    def _load_episode_paths(self, training_percent: float = 0.7) -> List[Path]:
        """Load all episode file paths"""
        all_episode_files = []

        # Respect user-specified policy_level if it's one of the known ones; else load all
        if self.policy_level in ['expert', 'medium', 'mixed']:
            policy_levels = [self.policy_level]
        else:
            policy_levels = ['expert', 'medium', 'mixed']

        #subfolders = ['none', 'dynamic_medium', 'static_medium']
        subfolders = ['none', 'static_medium']
        base_dir = self.data_dir / "dmc_vb" / f"{self.domain_name}_{self.task_name}"

        for policy_level in policy_levels:
            for subfolder in subfolders:
                episode_dir = base_dir / policy_level / subfolder
                if not episode_dir.exists():
                    continue

                episode_files = sorted(episode_dir.glob("distracting_control-*.tfrecord-*"))
                if len(episode_files) == 0:
                    episode_files = sorted(episode_dir.glob("*.tfrecord*"))

                if len(episode_files) > 0:
                    all_episode_files.extend(episode_files)

        if len(all_episode_files) == 0:
            raise ValueError(f"No episode files found under {base_dir}")

        # IMPORTANT: make split deterministic so eval is stable across runs
        rng = random.Random(0)
        rng.shuffle(all_episode_files)

        n_episodes = len(all_episode_files)
        n_train = int(training_percent * n_episodes)

        if self.split == 'train':
            return all_episode_files[:n_train]
        else:
            return all_episode_files[n_train:]

    def _load_all_episodes(self) -> List[Dict[str, np.ndarray]]:
        """Load all episodes into memory"""
        episode_paths = self._load_episode_paths()
        episodes = []

        print(f"Loading {len(episode_paths)} episodes...")
        for ep_path in tqdm(episode_paths):
            try:
                episode_data = self._load_episode_data(ep_path)
                if episode_data is not None:
                    episodes.append(episode_data)
            except Exception as e:
                print(f"Error loading {ep_path}: {e}")
                continue

        return episodes

    def _load_episode_data(self, episode_path: Path) -> Optional[Dict[str, np.ndarray]]:
        """Load episode data from file"""
        if episode_path.suffix == '.h5':
            return self._load_h5_episode(episode_path)
        elif '.tfrecord' in episode_path.suffix:
            return TFRecordConverter.parse_tfrecord_episode(episode_path, self.action_dim)
        elif episode_path.suffix == '.npz':
            return self._load_npz_episode(episode_path)
        else:
            raise ValueError(f"Unsupported file format: {episode_path.suffix}")

    def _load_h5_episode(self, h5_path: Path) -> Dict[str, np.ndarray]:
        """Load episode from HDF5 file"""
        data = {}

        with h5py.File(h5_path, 'r') as f:
            if 'observation_pixels' in f:
                data['observation_pixels'] = f['observation_pixels'][:]
            elif 'pixels' in f:
                data['observation_pixels'] = f['pixels'][:]
            else:
                raise KeyError("No observation pixels found in H5 episode.")

            if 'action' in f:
                data['action'] = f['action'][:]
            else:
                raise KeyError("No actions found in H5 episode.")

            if 'reward' in f and self.add_rewards:
                data['reward'] = f['reward'][:]

            if 'done' in f:
                data['done'] = f['done'][:]
            elif 'is_last' in f:
                data['done'] = f['is_last'][:].astype(np.float32)
            else:
                # done length aligned with actions
                data['done'] = np.zeros(len(data['action']), dtype=np.float32)
                data['done'][-1] = 1.0

        return data

    def _load_npz_episode(self, npz_path: Path) -> Dict[str, np.ndarray]:
        """Basic NPZ loader (kept for completeness)."""
        d = dict(np.load(npz_path))
        # Try common keys
        if 'observation_pixels' not in d and 'pixels' in d:
            d['observation_pixels'] = d['pixels']
        if 'done' not in d and 'is_last' in d:
            d['done'] = d['is_last'].astype(np.float32)
        if 'done' not in d:
            d['done'] = np.zeros(len(d['action']), dtype=np.float32)
            d['done'][-1] = 1.0
        return d

    def _compute_episode_lengths(self) -> List[int]:
        """
        Compute per-episode usable length:
        - clipped by min(len(obs), len(action), len(done))
        - clipped by first done (inclusive)
        This guarantees any sampled sequence stays within one episode and consecutive frames exist.
        """
        lengths = []
        for ep in self.episodes:
            if 'observation_pixels' not in ep or 'action' not in ep or 'done' not in ep:
                continue

            obs_len = len(ep['observation_pixels'])
            act_len = len(ep['action'])
            done_len = len(ep['done'])

            # Align to the shortest available array.
            # If obs_len is act_len+1, this safely clips to act_len (still fine).
            ep_len = int(min(obs_len, act_len, done_len))

            if ep_len <= 0:
                continue

            # Clip to first done (inclusive)
            done_clip = ep['done'][:ep_len]
            done_idx = np.where(done_clip > 0)[0]
            if len(done_idx) > 0:
                ep_len = int(min(ep_len, done_idx[0] + 1))

            # Must be long enough to build one sample
            min_required = self.sequence_length + self.frame_stack - 1
            if ep_len < min_required:
                continue

            lengths.append(ep_len)

        if len(lengths) == 0:
            raise RuntimeError("No valid episodes long enough for (sequence_length + frame_stack - 1).")

        return lengths

    def _create_sequence_indices(self) -> List[Tuple[int, int]]:
        """Create safe indices for all valid sequences per episode."""
        indices: List[Tuple[int, int]] = []

        # We recompute usable length per episode in the same way to stay consistent
        for ep_idx, ep in enumerate(self.episodes):
            if 'observation_pixels' not in ep or 'action' not in ep or 'done' not in ep:
                continue

            obs_len = len(ep['observation_pixels'])
            act_len = len(ep['action'])
            done_len = len(ep['done'])
            ep_len = int(min(obs_len, act_len, done_len))
            if ep_len <= 0:
                continue

            done_clip = ep['done'][:ep_len]
            done_idx = np.where(done_clip > 0)[0]
            if len(done_idx) > 0:
                ep_len = int(min(ep_len, done_idx[0] + 1))

            min_required = self.sequence_length + self.frame_stack - 1
            if ep_len < min_required:
                continue

            # last frame index used is start + (sequence_length + frame_stack - 2)
            max_start = ep_len - min_required + 1  # number of valid start positions
            for start_idx in range(max_start):
                indices.append((ep_idx, start_idx))

        return indices

    def _process_observation(self, obs: np.ndarray) -> torch.Tensor:
        """Process observation: normalize and convert to tensor [C,H,W]"""
        # Crop if needed
        if obs.shape[0] < self.img_height or obs.shape[1] < self.img_width:
            # If smaller, just crop what exists (avoid crashing)
            obs = obs[:min(obs.shape[0], self.img_height), :min(obs.shape[1], self.img_width)]
        else:
            obs = obs[:self.img_height, :self.img_width]

        # Normalize to [-1, 1]
        if self.normalize_images:
            if obs.dtype == np.uint8:
                obs = obs.astype(np.float32) / 255.0
            else:
                # Robust: if float but looks like 0..255, scale down
                if obs.max() > 1.5:
                    obs = obs.astype(np.float32) / 255.0
                else:
                    obs = obs.astype(np.float32)

            obs = (obs * 2.0) - 1.0
            obs = np.clip(obs, -1.0, 1.0)

        # [H,W,C] -> [C,H,W]
        obs = np.transpose(obs, (2, 0, 1))
        return torch.from_numpy(obs).float()

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence of data (consecutive frames from the same episode)."""
        ep_idx, start_idx = self.sequence_indices[idx]
        episode_data = self.episodes[ep_idx]

        # We need exactly (sequence_length + frame_stack - 1) raw frames to build sequence_length stacked frames
        end_idx = start_idx + self.sequence_length + self.frame_stack - 1  # exclusive

        # Slice raw pixels from ONE episode (consecutive in time)
        observations = episode_data['observation_pixels'][start_idx:end_idx]

        # Pre-process all frames once
        frames = [self._process_observation(observations[i]) for i in range(len(observations))]

        # Build stacked observations:
        # each stacked obs is frames[t-frame_stack : t] concatenated over channel dim
        processed_obs = []
        for t in range(self.frame_stack, len(frames) + 1):
            stacked_frames = frames[t - self.frame_stack : t]  # list of [C,H,W]
            stacked = torch.cat(stacked_frames, dim=0)         # [C*frame_stack, H, W]
            processed_obs.append(stacked)

        observations_tensor = torch.stack(processed_obs, dim=0)  # [T, C*frame_stack, H, W]

        # Actions aligned with the LAST frame of each stack
        action_start = start_idx + self.frame_stack - 1
        action_end = action_start + len(processed_obs)

        actions = episode_data['action'][action_start:action_end]
        actions_tensor = torch.from_numpy(actions).float()

        sample = {
            'observations': observations_tensor,
            'actions': actions_tensor,
        }

        # Add rewards if available
        if self.add_rewards and 'reward' in episode_data:
            rewards = episode_data['reward'][action_start:action_end]
            sample['rewards'] = torch.from_numpy(rewards).float()

        # Add done flags
        done_flags = episode_data['done'][action_start:action_end]
        sample['done'] = torch.from_numpy(done_flags).float()

        sample['episode_idx'] = torch.tensor(ep_idx, dtype=torch.long)
        sample['start_idx'] = torch.tensor(start_idx, dtype=torch.long)

        if self.transform:
            sample = self.transform(sample)

        return sample

def list_frozen_params(model):
    print("\n=== FROZEN PARAMETERS (requires_grad=False) ===")
    total = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            print(f"{name:80s}  shape={tuple(p.shape)}")
            total += p.numel()
    print(f"Total frozen params: {total}")
    print("==============================================\n")

def count_parameters(model, print_details=True):
    """
    Count the number of parameters in a model
    
    """
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Count by module
    module_params = defaultdict(int)
    module_trainable = defaultdict(int)
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            module_params[module.__class__.__name__] += sum(
                p.numel() for p in module.parameters()
            )
            module_trainable[module.__class__.__name__] += sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
    
    component_params = {}
    component_map = {
        'encoder': 'vdvae.encoder',
        'decoder': 'vdvae.decoder',
        'prior': 'prior',
        'rnn': '_rnn',
        'discriminators': 'image_discriminator'
    }
    
    for comp_name, attr_names in component_map.items():
        if isinstance(attr_names, str):
            attr_names = [attr_names]
        
        comp_total = 0
        for attr_name in attr_names:
            if hasattr(model, attr_name):
                component = getattr(model, attr_name)
                comp_total += sum(p.numel() for p in component.parameters())
        
        if comp_total > 0:
            component_params[comp_name] = comp_total
    
    if print_details:
        print("=" * 60)
        print(f"{'MODEL PARAMETER COUNT':^60}")
        print("=" * 60)
        
        # Total summary
        print(f"\nTOTAL PARAMETERS OF MODEL")
        print(f"  Total:         {total_params:,}")
        print(f"  Trainable:     {trainable_params:,}")
        print(f"  Non-trainable: {non_trainable_params:,}")
        print(f"  Memory (MB):   {(total_params * 4) / (1024**2):.2f} (assuming float32)")
        
        # Component breakdown
        if component_params:
            print(f"\nCOMPONENT BREAKDOWN")
            sorted_components = sorted(component_params.items(), key=lambda x: x[1], reverse=True)
            for name, count in sorted_components:
                percentage = (count / total_params) * 100
                print(f"  {name:15s}: {count:12,} ({percentage:5.1f}%)")
        
        # Module type breakdown
        print(f"\nMODULE TYPE BREAKDOWN")
        sorted_modules = sorted(module_params.items(), key=lambda x: x[1], reverse=True)[:10]
        for module_type, count in sorted_modules:
            percentage = (count / total_params) * 100
            trainable = module_trainable[module_type]
            print(f"  {module_type:20s}: {count:12,} ({percentage:5.1f}%) [{trainable:,} trainable]")
        
        print("=" * 60)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params,
        'by_component': component_params,
        'by_module': dict(module_params)
    }

class GradientMonitor:
    """Comprehensive gradient flow diagnostic system for hierarchical architectures"""
    
    def __init__(self, model, writer):
        self.model = model
        self.writer = writer
        self.component_groups = self._define_component_hierarchy()
        self.gradient_cache = {}
        self.step_counter = 0
        self.global_step = 0
        
    def _define_component_hierarchy(self):
        """Establish semantic groupings for architectural components"""
        return {
            'encoder': ['vdvae.encoder.'],
            'decoder': ['vdvae.decoder.'],
            'prior_dynamics': ['prior.stick_breaking.kumar_net', 'prior.component_nn'],
            'vrnn_core': ['_rnn', 'rnn_layer_norm'],
            'discriminators': ['image_discriminator']
        }
    
    def compute_component_gradients(
        self,
        update_global_step: bool = True,
        normalize: str = "rms_elem",  # "none" | "rms_elem" | "l2_per_sqrt_elem" | "mean_tensor"
    ) -> Dict[str, float]:
        """
        Returns a scalar per component group.

        normalize:
        - "none":             sqrt(sum g^2)  (total L2; best for dominance)
        - "rms_elem":         sqrt(mean g^2) (scale per parameter element; best for comparing modules)
        - "l2_per_sqrt_elem": sqrt(sum g^2) / sqrt(numel) (same as rms_elem)
        - "mean_tensor":      sqrt(sum g^2) / n_tensors   (NOT recommended for proportions)
        """
        if update_global_step:
            self.global_step += 1

        component_grads: Dict[str, float] = {}

        for group_name, module_patterns in self.component_groups.items():
            sqsum = 0.0
            n_tensors = 0
            n_elems = 0

            for name, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                if not any(pat in name for pat in module_patterns):
                    continue

                g = p.grad.detach()
                sqsum += g.pow(2).sum().item()
                n_tensors += 1
                n_elems += g.numel()

                # Optional: per-layer tracking
                if "kumar" in name and self.writer is not None:
                    self.writer.add_scalar(f"gradients/layers/{name}", g.norm(2).item(), self.global_step)

            if n_elems == 0:
                continue

            if normalize == "none":
                val = math.sqrt(sqsum)
            elif normalize in ("rms_elem", "l2_per_sqrt_elem"):
                val = math.sqrt(sqsum / n_elems)
            elif normalize == "mean_tensor":
                val = math.sqrt(sqsum) / max(1, n_tensors)
            else:
                raise ValueError(f"Unknown normalize={normalize}")

            component_grads[group_name] = val

        return component_grads

    def visualize_gradient_flow(self, component_grads, epoch):
        """Generate comprehensive gradient flow visualization"""
        # Create stacked bar chart for component contributions
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        components = list(component_grads.keys())
        values = list(component_grads.values())
        colors = plt.cm.viridis(np.linspace(0, 1, len(components)))
        
        # Gradient magnitude distribution
        ax1.bar(components, values, color=colors)
        ax1.set_ylabel('Gradient Norm (L2)')
        ax1.set_title(f'Component-wise Gradient Distribution - Epoch {epoch}')
        ax1.set_yscale('log')  # Log scale for better visibility
        ax1.tick_params(axis='x', rotation=45)
        # Relative contribution analysis
        total_grad = sum(values)
        percentages = [v/total_grad * 100 for v in values]
        ax2.pie(percentages, labels=components, autopct='%1.1f%%', colors=colors)
        ax2.set_title('Relative Gradient Contribution', y=-0.2)
        
        self.writer.add_figure('gradient_analysis/component_distribution', fig, epoch)
        plt.close()


def pick_bt(tensor_bt, i, t):
    """
    Utility to pick batch i and (optionally) time t from a variety of shapes.

    - 5D: [B, T, ...]  -> [1, ...] at time t
    - 4D: either
        * [B, T, K, 2] (slot_centers with time) -> [1, K, 2] at time t
        * [B, C, H, W] or [B, H, W, C] (no time) -> [1, ...]
    - 3D: [B, ...] -> [1, ...]
    """
    if tensor_bt is None:
        return None

    if tensor_bt.ndim == 5:
        # e.g. [B, T, K, H, W] or [B, T, H, W, K]
        return tensor_bt[i:i+1, t]

    elif tensor_bt.ndim == 4:
        # Two cases:
        #   (a) [B, T, K, 2]  (slot_centers over time)
        #   (b) [B, C, H, W]  (images/features without time)
        if tensor_bt.size(-1) == 2:
            # Interpret as [B, T, K, 2]
            return tensor_bt[i:i+1, t]   # -> [1, K, 2]
        else:
            # No explicit time dimension, just index batch
            return tensor_bt[i:i+1]

    elif tensor_bt.ndim == 3:
        # [B, ..., ...] (e.g., [B, K, 2] or [B, H, W])
        return tensor_bt[i:i+1]

    else:
        raise ValueError(f"Unexpected ndim={tensor_bt.ndim} in pick_bt")


def _to_01(x: torch.Tensor, name: str, tol: float = 1e-3, verbose: bool = False):
    """
    Convert an image tensor to [0,1] if it appears to be in [-1,1] or [0,255].
    Leaves it unchanged if it already appears in [0,1].
    """
    x = x.float()
    mn = x.amin().item()
    mx = x.amax().item()

    mode = "as_is_[0,1]"
    x01 = x

    # uint8 or clearly 0..255-ish
    if x.dtype == torch.uint8 or mx > 1.5:
        # common case: 0..255
        scale = 255.0 if mx <= 255.0 + tol else mx
        x01 = x / scale
        mode = f"scaled_[0,{int(scale)}]->[0,1]"

    # [-1,1]-ish
    elif mn < -0.1:
        x01 = (x + 1.0) * 0.5
        mode = "scaled_[-1,1]->[0,1]"

    # else: assume [0,1]
    x01 = x01.clamp(0.0, 1.0)

    if verbose:
        print(f"[PSNR] {name}: min={mn:.4f}, max={mx:.4f}, mode={mode}")

    return x01


def flow_to_hsv_rgb(flow_bchw: torch.Tensor, mag_percentile: float = 99.0, eps: float = 1e-6):
    """
    flow_bchw: [B, 2, H, W] with channels (u, v) in pixels
    returns:   [B, 3, H, W] RGB in [0,1]
    """
    assert flow_bchw.ndim == 4 and flow_bchw.shape[1] == 2, f"Expected [B,2,H,W], got {flow_bchw.shape}"

    u = flow_bchw[:, 0]
    v = flow_bchw[:, 1]

    # angle in [-pi, pi] -> hue in [0,1]
    ang = torch.atan2(v, u)
    h = (ang + torch.pi) / (2 * torch.pi)

    # magnitude -> value in [0,1] (robust normalization)
    mag = torch.sqrt(u * u + v * v)

    # robust per-batch max using percentile
    B = mag.shape[0]
    mag_flat = mag.reshape(B, -1)
    N = mag_flat.shape[1]
    k = int(round((mag_percentile / 100.0) * (N - 1))) + 1  # 1..N
    k = max(1, min(N, k))
    mag_max = torch.kthvalue(mag_flat, k, dim=1).values  # [B]
    mag_max = mag_max.view(B, 1, 1).clamp_min(eps)

    v_val = (mag / mag_max).clamp(0.0, 1.0)
    s = torch.ones_like(v_val)

    # HSV -> RGB (vectorized)
    # Based on standard conversion with h in [0,1]
    h6 = h * 6.0
    i = torch.floor(h6).to(torch.int64) % 6
    f = h6 - torch.floor(h6)

    p = v_val * (1.0 - s)
    q = v_val * (1.0 - s * f)
    t = v_val * (1.0 - s * (1.0 - f))

    r = torch.zeros_like(v_val)
    g = torch.zeros_like(v_val)
    b = torch.zeros_like(v_val)

    mask0 = (i == 0)
    mask1 = (i == 1)
    mask2 = (i == 2)
    mask3 = (i == 3)
    mask4 = (i == 4)
    mask5 = (i == 5)

    r[mask0], g[mask0], b[mask0] = v_val[mask0], t[mask0], p[mask0]
    r[mask1], g[mask1], b[mask1] = q[mask1], v_val[mask1], p[mask1]
    r[mask2], g[mask2], b[mask2] = p[mask2], v_val[mask2], t[mask2]
    r[mask3], g[mask3], b[mask3] = p[mask3], q[mask3], v_val[mask3]
    r[mask4], g[mask4], b[mask4] = t[mask4], p[mask4], v_val[mask4]
    r[mask5], g[mask5], b[mask5] = v_val[mask5], p[mask5], q[mask5]

    rgb = torch.stack([r, g, b], dim=1)  # [B,3,H,W]
    return rgb

class DMCVBTrainer:
    """Trainer for DPGMM-VRNN on DMC Vision Benchmark"""
    
    def __init__(
        self,
        model: nn.Module,
        data_dir: str,
        config: Dict,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.data_dir = data_dir
        default_ckpt_dir = PARENT_DIR / "results" / "dpgmm_vrnn_dmc_vb" / "checkpoints"
        self.ckpt_dir = Path(config.get("ckpt_dir", default_ckpt_dir))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        common_transform =HumanoidAwareZoomTransform(
                    zoom_prob=0.6,
                    area_range=(0.6, 0.85),
                    center_key="attn_center",  # optional; must be normalized [-1,1] xy
                    interpolation="lanczos",
                    corner_min=0.70,
                    corner_max=0.85,
                )
        # Setup datasets
        self.train_dataset = DMCVBDataset(
            data_dir=data_dir,
            domain_name=config['domain_name'],
            split='train',
            policy_level =config['policy_level'],
            sequence_length=config['sequence_length'],
            frame_stack=config['frame_stack'],
            img_height=config['img_height'],
            img_width=config['img_width'],
            add_state= False,
            transform = common_transform
        )
        
        self.eval_dataset = DMCVBDataset(
            data_dir=data_dir,
            domain_name=config['domain_name'],
            split='eval',
            sequence_length=config['sequence_length'],
            frame_stack=config['frame_stack'],
            img_height=config['img_height'],
            img_width=config['img_width'],
            add_state= False,
            transform = common_transform
        )
        
        # Setup dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            collate_fn=safe_collate,
            pin_memory=True
        )
        
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            collate_fn=safe_collate,
            pin_memory=True
        )
        self.viz_loader = DataLoader(
            self.eval_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0,              # critical
            collate_fn=safe_collate,
            pin_memory=False            # critical
        )
        # Initialize metrics tracking
        self.metrics_history = defaultdict(list)
        self.best_eval_loss = float('inf')
        self.episode_length = self.train_dataset.min_episode_length
        # Setup wandb if configured
        
        self.use_wandb = config.get('use_wandb', False) and self._try_init_wandb(config)
        if not self.use_wandb:
            log_dir = PARENT_DIR / "results" / "dpgmm_vrnn_dmc_vb" / "runs" / f"{config.get('experiment_name', 'dpgmm_vrnn')}_{time.strftime('%Y%m%d_%H%M%S')}"
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard logging to: {log_dir}")
        else:
            self.writer = None
        self.grad_monitor = GradientMonitor(model, self.writer)  # Initialize without writer first
        task_names = ["ELBO", "adversarial"]  
        self._agg = GradDiagnosticsAggregator(task_names, 
                                              component_groups=self.grad_monitor.component_groups,
                                              average_component_norms=True)

    def _try_init_wandb(self, config):
        """Attempt to initialize W&B, return success status"""
        try:
            import wandb
            wandb.init(
                project=config['wandb_project'],
                config=config,
                name=config['experiment_name']
            )
            return True
        except:
            print("W&B initialization failed, falling back to TensorBoard")
            return False

    def anneal_beta(self, epoch: int) -> float:
        import math
        beta_min = self.config.get('beta_min', 0.0)
        beta_max = self.config.get('beta_max', 1.0)
        T = max(1, int(self.config.get('beta_warmup_epochs', 40)))
        # progress in [0,1]
        p = min(1.0, max(0.0, (epoch + 1) / T))
        # cosine ramp-up: 0 -> 1
        ramp = 0.5 * (1.0 - math.cos(math.pi * p))
        return beta_min + (beta_max - beta_min) * ramp

    @torch.no_grad()
    def tb_log_warp_panel(
        self,
        epoch: int,
        b: int = 0,
        t: int = 1,
        tag: str = "warp/teacher_forced_panel",
        max_flow_frac: float = 0.25,
    ):
        """
        Logs:
        Main 2x3:
            [x_prev, x_tgt, x_warp_blend;
            e_prev, e_tgt, e_warp_blend]

        Extra diag panel (2x4):
            [x_warp_top1, x_warp_blend, rgb_err, w_top1;
            flow_mag, gate_entropy, top1_expert, edge_err]
        """
        if self.writer is None:
            return

        self.model.eval()

        # stable batch across epochs
        if not hasattr(self, "_tb_viz_batch") or self._tb_viz_batch is None:
            self._tb_viz_batch = next(iter(self.eval_loader))

        batch = self._tb_viz_batch
        observations = batch["observations"].to(self.device)  # [B,T,C,H,W]
        actions      = batch["actions"].to(self.device)       # [B,T,A]
        dones        = batch.get("done", None)
        if dones is not None:
            dones = dones.to(self.device)

        B, T, C, H, W = observations.shape
        b = int(max(0, min(b, B - 1)))
        t = int(max(1, min(t, T - 1)))   # need t>=1 for prev frame

        # run only prefix up to t so temporal_state is correct AND it's cheaper than full sequence
        obs_short  = observations[:, :t+1]
        act_short  = actions[:, :t+1]
        done_short = dones[:, :t+1] if dones is not None else None

        # ask model to capture the *real* warp ctx_map used in training (from temporal_state["64"])
        out = self.model.forward_sequence(
            obs_short,
            act_short,
            done_short,
            capture_flow_ctx=(b, t),
        )

        # frames
        x_prev = observations[b, t-1:t]  # [1,C,H,W] in [-1,1]
        x_tgt  = observations[b, t:t+1]  # [1,C,H,W] in [-1,1]

        # to [0,1] + force float32 for logging robustness
        x_prev01 = self.model.denormalize_generated_images(x_prev).clamp(0, 1).float()
        x_tgt01  = self.model.denormalize_generated_images(x_tgt ).clamp(0, 1).float()

        # edges in [0,1] (float32)
        e_prev = self.model.canny(x_prev01).clamp(0, 1).float()  # [1,1,H,W]
        e_tgt  = self.model.canny(x_tgt01 ).clamp(0, 1).float()  # [1,1,H,W]

        # ---- flow conditioning: use captured ctx_map (NOT hidden_states) ----
        ctx = out.get("captured_flow_ctx", None)  # expected [1, flow_ctx_dim, H, W] on CPU

        if ctx is None:
            # fallback: keep logging robust if capture didn't happen (e.g., warp disabled)
            flow_ctx_dim = getattr(self.model, "flow_ctx_dim", self.model.flow_ctx_proj[0].out_channels)
            ctx = torch.zeros((1, flow_ctx_dim, H, W), device=self.device, dtype=torch.float32)
        else:
            ctx = ctx.to(device=self.device, dtype=torch.float32)

            # safety: if spatial size differs, resize to match current H,W
            if (ctx.shape[-2], ctx.shape[-1]) != (H, W):
                ctx = F.interpolate(ctx, size=(H, W), mode="bilinear", align_corners=False)

        flow_in = torch.cat([x_prev01, e_prev, ctx], dim=1)

        # ---------- GRU flow + warp.py ----------
        max_flow_full = max(H, W) * float(max_flow_frac)

        def to3(x1):
            return x1 if x1.shape[1] == 3 else x1.repeat(1, 3, 1, 1)

        # Prefer flows captured from forward_sequence (best: matches GRU state used in training)
        flow_fw = out.get("captured_flow_fw", None)  # expected CPU [1,2,H,W]
        flow_bw = out.get("captured_flow_bw", None)  # optional, for cycle diag

        # Fallback if not captured: predict with zero GRU state (viz robustness only)
        if flow_fw is None:
            flow_state0 = torch.zeros((1, 128, H, W), device=self.device, dtype=torch.float32)
            a_t = actions[b:b+1, t - 1]  # predicts between t-1 and t
            # For forward flow, condition on prev frame
            flow_fw, _ = self.model._predict_flow_one_step(
                x01=x_prev01, e=e_prev, ctx=ctx, a_t=a_t,
                state=flow_state0, first=(t == 1), direction="fw"
            )
        else:
            flow_fw = flow_fw.to(device=self.device, dtype=torch.float32)

        # --- MATCH TRAINING EDGE GUIDE PATH 1:1 ---
        # forward splat prev -> current using flow_fw (prev->cur)
        x_prev_to_cur, denom = self.model.forward_splat_bilinear(x_prev01, flow_fw)
        x_prev_to_cur = x_prev_to_cur.clamp(0, 1)

        valid = (denom > 0.5).to(x_prev_to_cur.dtype)
        x_prev_to_cur = x_prev_to_cur * valid

        # edges are computed FROM warped RGB 
        e_prev_to_cur = self.model.canny(x_prev_to_cur.float()).clamp(0, 1).float()

        rgb_err  = (x_prev_to_cur - x_tgt01).abs().mean(dim=1, keepdim=True).clamp(0, 1)
        edge_err = (e_prev_to_cur - e_tgt).abs().clamp(0, 1)

        flow_mag = torch.sqrt((flow_fw ** 2).sum(dim=1, keepdim=True) + 1e-8)
        flow_mag = (flow_mag / (max_flow_full + 1e-6)).clamp(0, 1)

        # Optional: cycle-consistency diagnostic if backward flow exists
        cycle_mag = torch.zeros_like(flow_mag)
        if flow_bw is not None:
            flow_bw = flow_bw.to(device=self.device, dtype=torch.float32)
            # cycle: f_fw + warp(f_bw, f_fw)
            bw_warped = image_warp(flow_bw, flow_fw)
            cyc = flow_fw + bw_warped
            cycle_mag = torch.sqrt((cyc ** 2).sum(dim=1, keepdim=True) + 1e-8)
            cycle_mag = (cycle_mag / (max_flow_full + 1e-6)).clamp(0, 1)

        # --- flow HSV panel ---
        flow_fw_rgb = flow_to_hsv_rgb(flow_fw).clamp(0, 1)  # [1,3,H,W]
        flow_bw_rgb = torch.zeros_like(flow_fw_rgb)
        if flow_bw is not None:
            flow_bw_rgb = flow_to_hsv_rgb(flow_bw).clamp(0, 1)

        flow_grid = torchvision.utils.make_grid(
            torch.cat([flow_fw_rgb, flow_bw_rgb], dim=0),  # [2,3,H,W]
            nrow=2,
            padding=2
        )
        self.writer.add_image(tag + "_flow_hsv_fw_bw", flow_grid, epoch)

        # --- main 2x3 panel ---
        main_imgs = torch.cat([
            x_prev01, x_tgt01, x_prev_to_cur,
            to3(e_prev), to3(e_tgt), to3(e_prev_to_cur),
        ], dim=0)
        main_grid = torchvision.utils.make_grid(main_imgs, nrow=3, padding=2)
        self.writer.add_image(tag, main_grid, epoch)

        # --- diag 2x4 panel ---
        diag_imgs = torch.cat([
            x_prev_to_cur, x_tgt01, to3(rgb_err), to3(flow_mag),
            to3(e_prev_to_cur), to3(e_tgt), to3(edge_err), to3(cycle_mag),
        ], dim=0)
        diag_grid = torchvision.utils.make_grid(diag_imgs, nrow=4, padding=2)
        self.writer.add_image(tag + "_diag", diag_grid, epoch)

        # Scalars (note: now aligned to forward-splat)
        self.writer.add_images("warp/flow_hsv", flow_to_hsv_rgb(flow_fw.detach()), epoch)
        self.writer.add_scalar(tag + "/rgb_l1", (x_prev_to_cur - x_tgt01).abs().mean().item(), epoch)
        self.writer.add_scalar(tag + "/edge_l1", (e_prev_to_cur - e_tgt).abs().mean().item(), epoch)
        self.writer.add_scalar(tag + "/cycle_mag_mean", cycle_mag.mean().item(), epoch)


    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        # Calculate alpha for gradient normalization this epoch
        
        epoch_metrics = defaultdict(list)
        self.epoch_disc_losses = {'image': []}
        total_loss =[]
        epoch_component_grads = defaultdict(list)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        beta_t = self.anneal_beta(epoch)
        for batch_idx, batch in enumerate(pbar):
            
            # Move batch to device
            observations = batch['observations'].to(self.device)
            actions = batch['actions'].to(self.device)
            dones = batch['done'].to(self.device)
            # Training step
            losses = self.model.training_step_sequence(
                observations=observations,
                actions=actions,
                dones= dones,
                beta=beta_t,
                n_critic=self.config['n_critic'],
                lambda_img=self.config['lambda_img'],
                lambda_recon=self.config['lambda_recon'],
                batch_idx=batch_idx,
            )
            component_grads = self.grad_monitor.compute_component_gradients()
            for component, grad_norm in component_grads.items():
                epoch_component_grads[component].append(grad_norm)
            if 'img_disc_loss' in losses:
                self.epoch_disc_losses['image'].append(
                losses['img_disc_loss'].item() if torch.is_tensor(losses['img_disc_loss']) 
                else losses['img_disc_loss']
                )
            
            # Track metrics
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                epoch_metrics[key].append(value)
            total_loss.append(losses['total_gen_loss'].item() if torch.is_tensor(losses['total_gen_loss']) else float(losses['total_gen_loss']))
            # Update progress bar
            pbar.set_postfix({
                'total_loss': losses['total_gen_loss'],
                'recon_loss': losses['recon_loss'].item() if torch.is_tensor(losses['recon_loss']) else losses['recon_loss'],
                'kl_z': losses['kl_z'].item() if torch.is_tensor(losses['kl_z']) else losses['kl_z'],
                'beta': f'{beta_t:.3f}',
            })
        
        # Compute epoch averages
        avg_metrics = {f'train/{k}': np.mean(v) for k, v in epoch_metrics.items()}
        avg_img_disc_loss = np.mean(self.epoch_disc_losses['image']) 
        self.model.img_disc_scheduler.step(avg_img_disc_loss)
        avg_metrics['train/img_disc_lr'] = self.model.img_disc_optimizer.param_groups[0]['lr']

        avg_total_gen_loss = np.mean(total_loss)
        self.model.gen_scheduler.step(avg_total_gen_loss)
        avg_metrics['train/gen_lr'] = self.model.gen_optimizer.param_groups[0]['lr']

        avg_component_grads = {
                 component: np.mean(grads) 
                 for component, grads in epoch_component_grads.items()
       }
        self.grad_monitor.visualize_gradient_flow(avg_component_grads, epoch)

        # Log gradient norms to tensorboard
        for component, grad_norm in avg_component_grads.items():
            self.writer.add_scalar(f'gradients/components/{component}', grad_norm, epoch)

        return avg_metrics
    
    def evaluate(self, epoch: int) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()
        eval_metrics = defaultdict(list)
        print(f"Examine code to see if I am getting consecutive frames within each sample.")
        
        
        with torch.no_grad():
            pbar = tqdm(self.eval_loader, desc=f'Epoch {epoch} [Eval]')
            
            for batch in pbar:
                observations = batch['observations'].to(self.device)
                actions = batch['actions'].to(self.device)
                dones = batch['done'].to(self.device)
                
                # Forward pass and compute losses
                vae_losses, outputs = self.model.compute_total_loss(
                    observations=observations,
                    actions=actions,
                    dones=dones,
                    beta=self.config.get('beta_eval', self.config.get('beta_max', 1.0)),
                    lambda_recon =self.config['lambda_recon'],
                )
                
                # Track metrics
                for key, value in vae_losses.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    eval_metrics[key].append(value)
                
                # 1. Reconstruction quality (PSNR)
                recon = outputs['reconstructions']
                psnr = self.compute_psnr(observations, recon, dones)
                eval_metrics['psnr'].append(psnr.item())       
                
        if epoch % 10 == 0:
            with torch.no_grad():
                 num_samples = 16
                 samples =self.model.sample(num_samples)
                 if self.use_wandb:
                     wandb.log({
                         'eval/samples': wandb.Image(
                        torchvision.utils.make_grid(self.denormalize_image(samples), nrow=4
                        )
                    )
                     })
            save_path = str(self.ckpt_dir / f"dpgmm_prior_tsne_epoch_{epoch:04d}.png")
            
            try:
                fig = visualize_dpgmm_clustering(
                    model=self.model,
                    dataloader=self.viz_loader,
                    device=self.device,
                    max_batches=15,       # keep it cheap
                    max_samples=5000,
                    perplexity=30.0,
                    tsne_dims=3,
                    save_path=save_path,
                    image_level=False,     # start with image_level
                    t_select=5,            # choose which frame from the sequence
                    use_rnn_context=True,
                )
                plt.close(fig)
            except Exception as e:
                # Never crash training because a diagnostic plot failed.
                print(f"[eval] Warning: visualize_dpgmm_clustering failed at epoch {epoch}: {type(e).__name__}: {e}")
        # Compute averages
        avg_metrics = {f'eval/{k}': np.mean(v) for k, v in eval_metrics.items()}
        pred_metrics = self.evaluate_two_step_prediction(epoch, num_batches=5, T_ctx=8) 
        if self.config.get("use_wandb", False):
            wandb.log({f"{k}": v for k, v in pred_metrics.items()}, step=epoch)

        avg_metrics.update(pred_metrics)     
        # Save best model
        eval_loss = avg_metrics.get('eval/total_vae_loss', float('inf'))
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.save_checkpoint(epoch, is_best=True)
        
        return avg_metrics

    @torch.no_grad()
    def visualize_two_step_prediction(self, epoch: int, T_ctx: int = 8, num_examples: int = 4):
        """
        Log a simple grid comparing context, GT t+1/t+2 vs predicted t+1/t+2.
        Now also shows VAE-decoded predictions.
        """
        self.model.eval()
        horizon = 2  # predict two steps ahead
        batch = next(iter(self.eval_loader))
        observations = batch["observations"].to(self.device)  # [B, T, C, H, W]
        actions      = batch["actions"].to(self.device)       # [B, T, A]
        if batch.get("done") is not None:
            dones = batch["done"].to(self.device)             # [B, T]
        B, T, C, H, W = observations.shape

        assert T >= T_ctx + 2, f"Need at least T_ctx+2 frames, got T={T}"

        futures = self.model.generate_future_sequence(
            initial_obs=observations[:, :T_ctx],
            actions=actions[:,:T_ctx + horizon],
            horizon=horizon,
            dones=dones[:, :T_ctx + horizon] if dones is not None else None,
            grad=False,
            decode_mode="sample",
        )

        def denorm(x):
            # images are in [-1,1]
            return ((x + 1.0) * 0.5).clamp(0.0, 1.0)

        num_rows = min(num_examples, B)
        num_cols = 5  # ctx_last, GT t+1, GT t+2, VAE t+1, VAE t+2

         # Create figure
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, 3 * num_rows))

        if num_rows == 1:
            axes = axes[None, :]  # make it 2D

        for i in range(num_rows):
            # Last context frame
            ctx_last = denorm(observations[i, T_ctx - 1, -3:])
            gt_t1    = denorm(observations[i, T_ctx,     -3:])
            gt_t2    = denorm(observations[i, T_ctx + 1, -3:])

            # VAE-decoder predictions

            vae_t1   = denorm(futures["vae_future"][i, 0, -3:])
            vae_t2   = denorm(futures["vae_future"][i, 1, -3:])
            
            def show(ax, img, title):
                ax.imshow(img.permute(1, 2, 0).detach().cpu().numpy())
                ax.set_title(title)
                ax.axis("off")

            show(axes[i, 0], ctx_last, "Context last (t_ctx)")
            show(axes[i, 1], gt_t1,    "GT t+1")
            show(axes[i, 2], gt_t2,    "GT t+2")
            show(axes[i, 3], vae_t1,   "VAE t+1")
            show(axes[i, 4], vae_t2,   "VAE t+2")

        fig.tight_layout()

        if self.writer is not None:
            self.writer.add_figure("qualitative/two_step_future", fig, epoch)

        plt.close(fig)
        del futures
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()


    def run_grad_diag(
        self,
        max_B: int = 1,
        max_T: int = 4,
        include_adv_in_diag: bool = True,
        use_amp: bool = True,
    ):
        """
        One-shot gradient diagnostics on a tiny (B,T) slice to avoid OOM.

        IMPORTANT: This version just measures gradients w.r.t. whatever is currently
        trainable under normal training configuration.
        """

        # Remember original train/eval mode and then use train-mode for diag
        was_training = self.model.training
        self.model.train()

        try:
            # One tiny batch from eval_loader
            batch = next(iter(self.eval_loader))
            obs = batch["observations"][:max_B, :max_T].contiguous().to(
                self.device, non_blocking=True
            )
            act = batch["actions"][:max_B, :max_T].contiguous().to(
                self.device, non_blocking=True
            )
            done = batch["done"][:max_B, :max_T].contiguous().to(
                self.device, non_blocking=True
            )

            # Clear any stale grads
            self.model.zero_grad(set_to_none=True)

            amp_ctx = (
                torch.amp.autocast("cuda", dtype=torch.bfloat16)
                if (use_amp and self.device.type == "cuda")
                else contextlib.nullcontext()
            )

            with disable_all_checkpoint_modules(self.model), torch.set_grad_enabled(True), amp_ctx:
                vae_losses, outputs = self.model.compute_total_loss(
                    obs,
                    act,
                    done,
                    beta=self.config["beta"],
                    lambda_recon=self.config["lambda_recon"],
                )

                # Same ELBO combination 
                elbo_loss = (
                    self.config["lambda_recon"] * vae_losses["recon_loss"]
                    + self.config["beta"] * vae_losses["kl_z"]
                    + self.config["beta"] * vae_losses["hierarchical_kl"]
                )


                # Optional adversarial term – we DO NOT want grads into D's params
                if include_adv_in_diag and hasattr(self.model, "image_discriminator"):
                    D = self.model.image_discriminator

                    # Snapshot original flags and temporarily freeze D params
                    disc_flags = [p.requires_grad for p in D.parameters()]
                    for p in D.parameters():
                        p.requires_grad_(False)

                    recon = outputs["reconstructions"]
                    
                    z_seq = torch.cat([outputs['latents'], outputs['hidden_states']], dim=-1)

                    fake = D(recon, z=z_seq, return_features=True)
                    adv_loss = -fake["final_score"].mean()

                    # Restore original flags
                    for p, flag in zip(D.parameters(), disc_flags):
                        p.requires_grad_(flag)
                else:
                    adv_loss = torch.zeros((), device=obs.device)
                warmup_factor = self.model.get_warmup_factor()
                lambda_img_eff = (
                    self.config["lambda_img"] * warmup_factor
                    if warmup_factor > 0.0 else 0.0
                )                
                
                task_losses = [elbo_loss, lambda_img_eff * adv_loss]

                # Let GradDiagnosticsAggregator handle backward() etc.
                named_params = list(self.model.named_parameters())
                self._agg.update(
                    self.model,
                    shared_repr=None,
                    task_losses=task_losses,
                    named_params=named_params,
                    param_patterns=None,  # track all trainable params
                )

            # Log cosines, norms, etc.
            if self.writer is not None:
                self._agg.tensorboard_log(
                    self.writer,
                    tag_prefix="diag/grads",
                    global_step=getattr(self.grad_monitor, "global_step", 0),
                )

        finally:
            # detach outputs 
            for k, v in list(outputs.items()):
                if isinstance(v, torch.Tensor):
                    outputs[k] = v.detach().cpu()
                elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                    outputs[k] = [t.detach().cpu() for t in v]
            # Hard reset grads + restore train/eval mode
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)

            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


    def compute_psnr(self, obs, recon, dones=None, eps=1e-10, verbose=False):
        # Allow [B,C,H,W] by auto-adding T=1
        if obs.ndim == 4:
            obs = obs.unsqueeze(1)
        if recon.ndim == 4:
            recon = recon.unsqueeze(1)

        assert obs.ndim == 5 and recon.ndim == 5, f"Expected 5D after fix, got {obs.shape} and {recon.shape}"
        assert obs.shape == recon.shape, f"Shape mismatch: obs {obs.shape}, recon {recon.shape}"

        obs01 = _to_01(obs, "obs", verbose=verbose)
        rec01 = _to_01(recon, "recon", verbose=verbose)

        # Per-frame MSE: [B,T]
        mse_bt = (obs01 - rec01).pow(2).mean(dim=(-3, -2, -1))

        if dones is not None:
            if dones.ndim == 1:
                dones = dones.unsqueeze(1)
            valid = torch.ones_like(dones, dtype=mse_bt.dtype, device=mse_bt.device)
            valid[:, 1:] = torch.cumprod(1.0 - dones[:, :-1].float(), dim=1)
            mse = (mse_bt * valid).sum() / valid.sum().clamp_min(1.0)
        else:
            mse = mse_bt.mean()

        return 10.0 * torch.log10(1.0 / mse.clamp_min(eps))


    @torch.no_grad()
    def evaluate_two_step_prediction(
        self,
        epoch,
        num_batches: int = 10,
        T_ctx: int = 8,
    ) -> Dict[str, float]:
        """
        Evaluate 1-step and 2-step pixel prediction PSNR using the
        world model's generate_future_sequence method.
        """
        self.model.eval()
        psnr_t1, psnr_t2 = [], []

        for i, batch in enumerate(self.eval_loader):
            if i >= num_batches:
                break

            observations = batch["observations"].to(self.device)  # [B, T, C, H, W]
            actions      = batch["actions"].to(self.device)        # [B, T, A]
            dones        = batch["done"].to(self.device)           # [B, T]
            B, T, C, H, W = observations.shape

            if T < T_ctx + 2:
                continue  # skip too-short sequences

            # Rollout 2 future steps
            futures = self.model.generate_future_sequence(
                initial_obs=observations[:, :T_ctx],  # [B, T_ctx, C, H, W]
                actions=actions[:, :T_ctx + 2],
                horizon=2,
                dones=dones[:, :T_ctx + 2],
                grad=False,
                decode_mode="sample",
            )

            # PSNR per step (reuse compute_psnr which handles [-1,1])
            psnr_1 = self.compute_psnr(observations[:, T_ctx:T_ctx + 2][:, 0], futures["vae_future"][:, 0])  # t+1
            psnr_2 = self.compute_psnr(observations[:, T_ctx:T_ctx + 2][:, 1], futures["vae_future"][:, 1])  # t+2

            psnr_t1.append(psnr_1.item())
            psnr_t2.append(psnr_2.item())

        metrics = {
            "eval/pred_psnr_t+1": float(np.mean(psnr_t1)) if psnr_t1 else 0.0,
            "eval/pred_psnr_t+2": float(np.mean(psnr_t2)) if psnr_t2 else 0.0,
        }

        # Optional: log to TensorBoard / wandb
        if self.writer is not None:
            self.writer.add_scalar("eval/pred_psnr_t+1", metrics["eval/pred_psnr_t+1"], epoch)
            self.writer.add_scalar("eval/pred_psnr_t+2", metrics["eval/pred_psnr_t+2"], epoch)

        del observations, actions, dones, futures
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        return metrics


    def visualize_results(self, epoch: int):
        """
        
        Cols = [Original, Recon]
        """
        self.model.eval()

        # ---- fetch small train/eval minibatches ----
        train_batch = next(iter(self.train_loader))
        train_obs    = train_batch['observations'].to(self.device)
        train_actions= train_batch['actions'].to(self.device)
        train_dones = train_batch['done'].to(self.device)

        # random eval minibatch
        batch_idx = random.randint(0, len(self.eval_loader) - 1)
        for i, eval_batch in enumerate(self.eval_loader):
            if i == batch_idx: 
                break
        eval_obs     = eval_batch['observations'].to(self.device)
        eval_actions = eval_batch['actions'].to(self.device)
        eval_dones = eval_batch['done'].to(self.device)

        batch_size = min(train_obs.shape[0], eval_obs.shape[0], 4)  # cap at 4 rows per split
        n_cols = 3
        fig, axes = plt.subplots(2 * batch_size, n_cols, figsize=(3.2 * n_cols, 3.2 * (2 * batch_size)))
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)


        with torch.no_grad():
            for split_idx, (observations, actions, dones, split_name) in enumerate([
                (train_obs[:batch_size], train_actions[:batch_size], train_dones[:batch_size], "Train"),
                (eval_obs[:batch_size],  eval_actions[:batch_size], eval_dones[:batch_size], "Eval")
            ]):
                observations = observations.clamp(-1.0, 1.0)

                # run model once to populate outputs and the posterior's slot maps/assignments
                _, outputs = self.model.compute_total_loss(observations=observations, actions=actions, dones=dones)
                

                for i in range(batch_size):
                    row = split_idx * batch_size + i
                    t = 0  # visualize first timestep

                    # ---------- 1) Original ----------
                    orig_img = observations[i, t, :3]  # [3,H,W]
                    axes[row, 0].imshow(self.denormalize_image(orig_img))
                    axes[row, 0].set_title(f'{split_name} Original'); axes[row, 0].axis('off')

                    # ---------- 2) VAE Reconstruction ----------
                    recon_img = outputs['reconstruction_samples'][i, t, :3]  # [3,H,W]
                    axes[row, 1].imshow(recon_img.detach().clamp(0, 255).to(torch.uint8).cpu().permute(1,2,0).numpy())
                    axes[row, 1].set_title('VAE Recon samples'); axes[row, 1].axis('off')

                    recon_img = outputs['reconstructions'][i, t, :3]  # [3,H,W]
                    axes[row, 2].imshow(self.denormalize_image(recon_img))
                    axes[row, 2].set_title('VAE Recon'); axes[row, 2].axis('off')


        plt.tight_layout()
        if self.use_wandb:
            
            # Convert figure to image for wandb
            wandb.log({
                f'visualizations/train_eval_epoch_{epoch}': wandb.Image(fig),
                'epoch': epoch
            })
        elif self.writer:
            # Add figure to TensorBoard
            self.writer.add_figure(f'visualizations/train_eval_epoch_{epoch}', fig, epoch)
        else:
            if not os.path.exists('visualizations'):
                os.makedirs('visualizations')
            plt.savefig(f'visualizations/train_eval_epoch_{epoch}.png', dpi=150, bbox_inches='tight')    

        plt.close(fig)


    def denormalize_image(self, img: torch.Tensor) -> np.ndarray:
        """Convert from [-1, 1] to [0, 1] for visualization
        Args:
        image: Tensor of shape [C, H, W] or [B, C, H, W]
        to_uint8: If True, convert to uint8 (0-255), else keep as float (0-1)
    
        Returns:
        numpy array of shape [H, W, C] or [B, H, W, C]

        """
        if img.dim() == 3:  # [C, H, W]
            img = img.unsqueeze(0)  # Add batch dimension
            squeeze_batch = True
        else:
            squeeze_batch = False

        
        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)
        img = img.cpu().permute(0,2,3,1).numpy()
        img = (img * 255).astype(np.uint8)
        if squeeze_batch:
            img = img[0]
        return img
    
    def denormalize_for_grid(self, img):
        """For torchvision.make_grid: [-1,1] → [0,1] float"""
        img = (img + 1) / 2
        return torch.clamp(img, 0, 1)


    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'metrics_history': dict(self.metrics_history),
            'best_eval_loss': self.best_eval_loss
        }
        
        # Save regular checkpoint
        if epoch % 10 == 0:
           checkpoint_path = self.ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt"
           torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.ckpt_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def train(self, n_epochs: int):
        """Main training loop"""
        print(f"Starting training for {n_epochs} epochs...")
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Eval dataset size: {len(self.eval_dataset)}")
        # Register EMA model
        self.model.ema_vdvae.register()

        for epoch in range(n_epochs):
            
            self.model.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            eval_metrics = self.evaluate(epoch)
            self.run_grad_diag(max_B=1, max_T=self.episode_length, use_amp=False)            
            # Combine metrics
            all_metrics = {**train_metrics, **eval_metrics}
            
            # Log metrics
            if self.use_wandb:
                wandb.log(all_metrics, step=epoch)
            elif self.writer:
                for key, value in all_metrics.items():
                    self.writer.add_scalar(key, value, epoch)
            
            # Store metrics
            for key, value in all_metrics.items():
                self.metrics_history[key].append(value)
            
            # Print summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics.get('train/total_gen_loss', 0):.4f}")
            print(f"  Eval Loss: {eval_metrics.get('eval/total_vae_loss', 0):.4f}")
            print(f"  PSNR: {eval_metrics.get('eval/psnr', 0):.2f}")
            print(f"  Active Components: {train_metrics.get('train/effective_components', 0):.2f}")
            print(f"  Top 6 components: {train_metrics.get('train/Top 6 coverage', 0):.3f}")

            # Visualize periodically
            if epoch % self.config['visualize_every'] == 0:
                self.visualize_results(epoch)
                self.visualize_two_step_prediction(epoch, T_ctx=8)
                self.tb_log_warp_panel(epoch=epoch, b=0, t=1, tag="warp/teacher_forced_panel")
                

            
            # Save checkpoint
            if epoch % self.config['checkpoint_every'] == 0:
                self.save_checkpoint(epoch)

            del train_metrics, eval_metrics, all_metrics
            
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
        # Cleanup
        if self.writer:
            self.writer.close()
        
        print("Training completed!")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DPGMM-VRNN transition dynamics on DMC VB data"
    )

    # --- data / task ---
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override default transition_data dir")
    parser.add_argument("--domain_name", type=str, default=None,
                        help="DMC domain name (e.g. humanoid)")
    parser.add_argument("--task_name", type=str, default=None,
                        help="DMC task name (e.g. walk)")
    parser.add_argument("--policy_level", type=str, default=None,
                        help="dataset policy level (expert / medium / random/ all)")

    # --- model ---
    parser.add_argument("--max_components", type=int, default=None)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)

    # --- training ---
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--sequence_length", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)

    parser.add_argument("--use_dynamic_weight_average", type=str2bool, default=False )

    parser.add_argument("--beta_min", type=float, default=None)
    parser.add_argument("--beta_max", type=float, default=None)
    parser.add_argument("--beta_warmup_epochs", type=int, default=None)
    parser.add_argument("--beta_eval", type=float, default=None)

    parser.add_argument("--lambda_img", type=float, default=None)
    parser.add_argument("--lambda_recon", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--n_critic", type=int, default=None)

    # --- logging / wandb ---
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument(
        "--use_wandb", action="store_true",
        help="Enable Weights & Biases logging (overrides config to True)",
    )
    parser.add_argument(
        "--no_wandb", action="store_true",
        help="Disable Weights & Biases logging (overrides config to False)",
    )

    return parser.parse_args()
def override_config_from_args(config: dict, args: argparse.Namespace) -> dict:
    """
    Take the default config dict and override fields if the corresponding
    CLI arg is not None.
    """
    # keys in config that we allow overriding directly from args
    overridable_keys = [
        # data / task
        "data_dir", "domain_name", "task_name", "policy_level",
        # model
        "max_components", "latent_dim", "hidden_dim", 
        # training
        "batch_size", "sequence_length", "learning_rate", "n_epochs",
        "num_workers", "beta_min", "beta_max", "beta_warmup_epochs",
        "beta_eval", "lambda_img", "lambda_recon", 
        "grad_clip", "n_critic",
        # logging
        "experiment_name",
    ]

    for key in overridable_keys:
        if hasattr(args, key):
            val = getattr(args, key)
            if val is not None:
                # handle Path-like fields nicely
                if key == "data_dir":
                    config[key] = Path(val)
                else:
                    config[key] = val

    # W&B override logic
    if getattr(args, "use_wandb", False):
        config["use_wandb"] = True
    if getattr(args, "no_wandb", False):
        config["use_wandb"] = False

    return config

def main():
    """Main training script"""
    torch.autograd.set_detect_anomaly(True) #check issues with gradients
    
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')  # Enable TensorFloat32 cores
    torch.cuda.empty_cache()  
    # Additional foundational configurations

    # Configuration
    config = {
        # Data settings
        'data_dir': PARENT_DIR / "transition_data",
        'domain_name': 'humanoid',
        'task_name': 'walk',
        'policy_level': 'all',
        
        # Model settings
        'max_components': 12,
        'latent_dim': 56,
        'hidden_dim': 48, #must be divisible by 8
        'input_channels': 3*1,  # 3 stacked frames
        'prior_alpha': 16.0,  # Hyperparameters for prior
        'prior_beta': 2.0,
        'dropout': 0.1,

        # Training settings
        'batch_size': 20,
        'sequence_length': 10,
        'disc_num_heads': 8,
        'img_disc_layers': 2,
        'frame_stack': 1,
        'img_height': 64,
        'img_width': 64,
        'learning_rate': 0.0007,
        'n_epochs': 200,
        'num_workers': 4,

        'beta_min': 0.5,
        'beta_max': 1.0,
        'beta_warmup_epochs': 20,  # 20–50 is common
        'beta_eval': 1.0,          # force eval to use full KL 
        
        # Loss weights
        'beta': 1.0,
        'lambda_img': 1.0,
        'lambda_recon': 1.0,
        'grad_clip': 1.0,
        'n_critic': 1,       
        "use_dynamic_weight_average": False,
        # Logging
        'use_wandb': False,
        'wandb_project': 'dpgmm-vrnn-dmc',
        'wandb_entity': 'zahrasheikh',
        'experiment_name': 'humanoid_walk_expert',
        'visualize_every': 4,
        'checkpoint_every': 10
    }
    config = override_config_from_args(config, args)

    print("=== Final config ===")
    for k, v in config.items():
        print(f"{k}: {v}")
    print("====================")    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get action dimension from dataset
    action_dim = DMCVBInfo.get_action_dim(config['domain_name'])
    
    # Initialize model
    model = DPGMMVariationalRecurrentAutoencoder(
        max_components=config['max_components'],
        input_dim=config['img_height'],
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        action_dim=action_dim,
        sequence_length=config['sequence_length'],
        img_disc_layers=config["img_disc_layers"],
        disc_num_heads = config["disc_num_heads"] if "disc_num_heads" in config else 4,
        device=device,
        input_channels=config['input_channels'],
        learning_rate=config['learning_rate'],
        grad_clip= config['grad_clip'],
        prior_alpha =config['prior_alpha'],
        prior_beta = config['prior_beta'],
        dropout=config['dropout'],
        use_dwa = config["use_dynamic_weight_average"],
    )


    outputs = count_parameters(model, print_details=True)
    list_frozen_params(model)
    print_vdvae_edge_report(model)
    # Initialize trainer
    trainer = DMCVBTrainer(
        model=model,
        data_dir=config['data_dir'],
        config=config,
        device=device
    )
    
    # Train
    trainer.train(n_epochs=config['n_epochs'])


if __name__ == '__main__':
    main()

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import random
import os
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import tensorflow as tf
import zlib
from torch.utils.tensorboard import SummaryWriter
from VRNN.dpgmm_stickbreaking_prior_vrnn import DPGMMVariationalRecurrentAutoencoder
"""Download data :gsutil -m cp -r gs://dmc_vision_benchmark/dmc_vision_benchmark/locomotion/humanoid_walk/medium ./transition_data/dmc_vb/humanoid_walk/"""

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
        Decompress zlib-encoded raw pixel observations
        
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
        
        This implementation synthesizes insights from:
        - Information-theoretic compression principles
        - Empirical data morphology analysis
        - Canonical DMC-VB preprocessing pipelines
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
        
        # Load episode paths
        self.episodes = self._load_all_episodes()
        self.min_episode_length = self._compute_min_episode_length()
        
        # Create sequence indices
        self.sequence_indices = self._create_sequence_indices()
        print(f"Loaded {len(self.episodes)} episodes")
        print(f"Minimum episode length (at first done): {self.min_episode_length}")
        print(f"Total sequences available: {len(self.sequence_indices)}")
        

    def _load_episode_paths(self) -> List[Path]:
        """Load all episode file paths"""
        all_episode_files = []
        policy_levels = ['expert', 'medium']
        subfolders =['none', 'dynamic_medium', 'static_medium']
        base_dir = self.data_dir / "dmc_vb" / f"{self.domain_name}_{self.task_name}"
        # Collect episodes from all combinations
        for policy_level in policy_levels:
            for subfolder in subfolders:
                episode_dir = base_dir / policy_level / subfolder
                
                if not episode_dir.exists():
                    print(f"Skipping non-existent directory: {episode_dir}")
                    continue
                
                # Get all tfrecord files from this directory
                episode_files = sorted(episode_dir.glob("distracting_control-*.tfrecord-*"))
                
                if len(episode_files) == 0:
                    # Try general tfrecord pattern as fallback
                    episode_files = sorted(episode_dir.glob("*.tfrecord*"))
                
                if len(episode_files) > 0:
                    print(f"Found {len(episode_files)} episodes in {policy_level}/{subfolder}")
                    all_episode_files.extend(episode_files)
        
        if len(all_episode_files) == 0:
            raise ValueError(f"No episode files found in any directory under {base_dir}")
        
        # Shuffle all episodes to mix expert and medium data
        random.shuffle(all_episode_files)
        
        # Split into train/eval
        n_episodes = len(all_episode_files)
        n_train = int(0.95 * n_episodes)
        
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
            # Load observations
            if 'observation_pixels' in f:
                data['observation_pixels'] = f['observation_pixels'][:]
            elif 'pixels' in f:
                data['observation_pixels'] = f['pixels'][:]
            
            # Load actions
            if 'action' in f:
                data['action'] = f['action'][:]
            
            # Load rewards
            if 'reward' in f and self.add_rewards:
                data['reward'] = f['reward'][:]
            
            # Load or create done flags
            if 'done' in f:
                data['done'] = f['done'][:]
            elif 'is_last' in f:
                data['done'] = f['is_last'][:].astype(np.float32)
            else:
                # Create done flag - mark last timestep as done
                data['done'] = np.zeros(len(data['action']), dtype=np.float32)
                data['done'][-1] = 1.0
        
        return data
    
    def _compute_min_episode_length(self) -> int:
        """Compute minimum episode length across all episodes (up to first done)"""
        min_length = float('inf')
        valid_episodes = 0

        for episode in self.episodes:
            if 'done' not in episode or len(episode['done']) == 0:
                continue
                
            # Find first done flag
            done_indices = np.where(episode['done'] > 0)[0]
            if len(done_indices) > 0:
                episode_length = done_indices[0] + 1
            else:
                episode_length = len(episode['done'])
            
            if episode_length > 0:  # Only consider non-empty episodes
                valid_episodes += 1
                min_length = min(min_length, episode_length)

        if valid_episodes == 0:
            raise RuntimeError(f"No valid episodes found out of {len(self.episodes)} total")

        min_required = self.sequence_length + self.frame_stack
        if min_length < min_required:
            raise ValueError(f"Min episode length ({min_length}) < required ({min_required})")

        return int(min_length)        
    def _create_sequence_indices(self) -> List[Tuple[int, int]]:
        """Create indices for all valid sequences based on minimum episode length"""
        indices = []
        
        for ep_idx in range(len(self.episodes)):
            # Use minimum episode length to ensure consistency
            max_start = self.min_episode_length - self.sequence_length - self.frame_stack + 1
            
            for start_idx in range(max_start):
                indices.append((ep_idx, start_idx))
        
        return indices
    
    def _process_observation(self, obs: np.ndarray) -> torch.Tensor:
        """Process observation: normalize and convert to tensor"""
        # Ensure correct shape [H, W, C]
        if obs.shape != (self.img_height, self.img_width, 3):
            # Simple resize if needed
            obs = obs[:self.img_height, :self.img_width]  # Crop
        
        # Normalize to [-1, 1] if needed
        if self.normalize_images:
            if obs.dtype == np.uint8:
                obs = obs.astype(np.float32) / 255.0
            obs = (obs * 2.0) - 1.0
        
        # Convert to channels first [C, H, W]
        obs = np.transpose(obs, (2, 0, 1))
        
        return torch.from_numpy(obs).float()
 
    
    def __len__(self) -> int:
        return len(self.sequence_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence of data"""
        ep_idx, start_idx = self.sequence_indices[idx]
        episode_data = self.episodes[ep_idx]
        
        # Extract sequence (clipped to min_episode_length)
        end_idx = min(start_idx + self.sequence_length + self.frame_stack, self.min_episode_length)
        
        # Stack observations from pixels
        observations = episode_data['observation_pixels'][start_idx:end_idx]
        
        # Process observations with frame stacking
        processed_obs = []
        for t in range(self.frame_stack, end_idx - start_idx):
            # Stack frames
            stacked_frames = []
            for f in range(self.frame_stack):
                frame = self._process_observation(observations[t - self.frame_stack + f + 1])
                stacked_frames.append(frame)
            stacked = torch.cat(stacked_frames, dim=0)  # Stack along channel dimension
            processed_obs.append(stacked)
        
        observations_tensor = torch.stack(processed_obs)
        
        # Get actions
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
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

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
            'encoder': ['encoder.net', 'encoder.proj'],
            'decoder': ['decoder.net'],
            'perceiver': ['perceiver_model', 'perceiver_projection'],
            'prior_dynamics': ['prior.stick_breaking.kumar_net', 'prior.component_nn'],
            'attention_schema': ['attention_prior_posterior'],
            'vrnn_core': ['_rnn', 'rnn_layer_norm'],
            'self_model': ['self_model'],
            'discriminators': ['image_discriminator', 'latent_discriminator']
        }
    
    def compute_component_gradients(self, update_global_step: bool = True) -> Dict[str, float]:
        """Extract gradient magnitudes per architectural component"""
        if update_global_step:
            self.global_step += 1  
        
        component_grads = {}
        for group_name, module_patterns in self.component_groups.items():
            group_grad_norm = 0.0
            param_count = 0
            
            for name, param in self.model.named_parameters():
                if param.grad is not None and any(pattern in name for pattern in module_patterns):
                    grad_norm = param.grad.data.norm(2).item()
                    group_grad_norm += grad_norm ** 2
                    param_count += 1
                    
                    # Track individual layer gradients for critical components
                    if 'kumar' in name or 'attention' in name:
                        if self.writer is not None:  
                            self.writer.add_scalar(f'gradients/layers/{name}', grad_norm, self.global_step)
            
            if param_count > 0:
                component_grads[group_name] = (group_grad_norm ** 0.5) / param_count
        
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
        ax2.set_title('Relative Gradient Contribution')
        
        self.writer.add_figure('gradient_analysis/component_distribution', fig, epoch)
        plt.close()

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
        # Setup datasets
        self.train_dataset = DMCVBDataset(
            data_dir=data_dir,
            domain_name=config['domain_name'],
            split='train',
            sequence_length=config['sequence_length'],
            frame_stack=config['frame_stack'],
            img_height=config['img_height'],
            img_width=config['img_width'],
            add_state= False
        )
        
        self.eval_dataset = DMCVBDataset(
            data_dir=data_dir,
            domain_name=config['domain_name'],
            split='eval',
            sequence_length=config['sequence_length'],
            frame_stack=config['frame_stack'],
            img_height=config['img_height'],
            img_width=config['img_width'],
            add_state= False
        )
        
        # Setup dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        # Initialize metrics tracking
        self.metrics_history = defaultdict(list)
        self.best_eval_loss = float('inf')
        
        # Setup wandb if configured
        
        self.use_wandb = config.get('use_wandb', False) and self._try_init_wandb(config)
        if not self.use_wandb:
            log_dir = f"runs/{config.get('experiment_name', 'dpgmm_vrnn')}_{time.strftime('%Y%m%d_%H%M%S')}"
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard logging to: {log_dir}")
        else:
            self.writer = None
        self.grad_monitor = GradientMonitor(model, self.writer)  # Initialize without writer first

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
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = defaultdict(list)
        self.epoch_disc_losses = {'image': [], 'latent': []}

        epoch_component_grads = defaultdict(list)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            observations = batch['observations'].to(self.device)
            actions = batch['actions'].to(self.device)
            
            # Training step
            losses = self.model.training_step_sequence(
                observations=observations,
                actions=actions,
                beta=self.config['beta'],
                n_critic=self.config['n_critic'],
                lambda_img=self.config['lambda_img'],
                lambda_pred=self.config['lambda_pred'],
                lambda_att=self.config['lambda_att'],
                entropy_weight=self.config['entropy_weight']
            )
            component_grads = self.grad_monitor.compute_component_gradients()
            for component, grad_norm in component_grads.items():
                epoch_component_grads[component].append(grad_norm)
            if 'img_disc_loss' in losses:
                self.epoch_disc_losses['image'].append(
                losses['img_disc_loss'].item() if torch.is_tensor(losses['img_disc_loss']) 
                else losses['img_disc_loss']
                )
            if 'latent_disc_loss' in losses:
                self.epoch_disc_losses['latent'].append(
                    losses['latent_disc_loss'].item() if torch.is_tensor(losses['latent_disc_loss']) 
                    else losses['latent_disc_loss']
                )
            # Track metrics
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                epoch_metrics[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({
                'total_loss': losses['total_gen_loss'],
                'recon_loss': losses['recon_loss'],
                'kl_z': losses['kl_z']
            })
        
        # Compute epoch averages
        avg_metrics = {f'train/{k}': np.mean(v) for k, v in epoch_metrics.items()}
        avg_img_disc_loss = np.mean(self.epoch_disc_losses['image']) 
        self.model.img_disc_scheduler.step(avg_img_disc_loss)
        avg_metrics['train/img_disc_lr'] = self.model.img_disc_optimizer.param_groups[0]['lr']
        avg_latent_disc_loss = np.mean(self.epoch_disc_losses['latent']) 
        self.model.latent_disc_scheduler.step(avg_latent_disc_loss)
        avg_metrics['train/latent_disc_lr'] = self.model.latent_disc_optimizer.param_groups[0]['lr']
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
        
        with torch.no_grad():
            pbar = tqdm(self.eval_loader, desc=f'Epoch {epoch} [Eval]')
            
            for batch in pbar:
                observations = batch['observations'].to(self.device)
                actions = batch['actions'].to(self.device)
                
                # Forward pass and compute losses
                vae_losses, outputs = self.model.compute_total_loss(
                    observations=observations,
                    actions=actions,
                    beta=self.config['beta'],
                    entropy_weight=self.config['entropy_weight']
                )
                
                # Track metrics
                for key, value in vae_losses.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    eval_metrics[key].append(value)
                
                # Additional evaluation metrics
                
                # 1. Reconstruction quality (PSNR)
                recon = outputs['reconstructions']
                psnr = self.compute_psnr(observations, recon)
                eval_metrics['psnr'].append(psnr.item())
                
                # 2. Attention consistency
                if 'attention_maps' in outputs and len(outputs['attention_maps']) > 1:
                    att_consistency = self.compute_attention_consistency(outputs['attention_maps'])
                    eval_metrics['attention_consistency'].append(att_consistency.item())
                
                # 3. Predictive accuracy
                if 'self_predictions' in outputs:
                    pred_acc = self.compute_predictive_accuracy(outputs)
                    eval_metrics['predictive_accuracy'].append(pred_acc)
        if epoch %10 == 0:
             with torch.no_grad():
                 num_samples = 16
                 samples =self.model.sample(num_samples)
                 if self.use_wandb:
                     wandb.log({
                         'eval/samples': wandb.Image(
                        torchvision.utils.make_grid(
                            self.denormalize_image(samples), nrow=4, normalize=True, value_range=(0, 1)
                        )
                    )
                     })
        # Compute averages
        avg_metrics = {f'eval/{k}': np.mean(v) for k, v in eval_metrics.items()}
        
        # Save best model
        eval_loss = avg_metrics.get('eval/total_vae_loss', float('inf'))
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.save_checkpoint(epoch, is_best=True)
        
        return avg_metrics
    
    def compute_psnr(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Compute Peak Signal-to-Noise Ratio"""
        mse = F.mse_loss(reconstructed, original)
        psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))  # Assuming images in [-1, 1]
        return psnr
    
    def compute_attention_consistency(self, attention_maps: List[torch.Tensor]) -> torch.Tensor:
        """Compute temporal consistency of attention maps"""
        consistencies = []
        
        for t in range(1, len(attention_maps)):
            prev_att = attention_maps[t-1]
            curr_att = attention_maps[t]
            
            # Compute correlation between consecutive attention maps
            prev_flat = prev_att.view(prev_att.size(0), -1)
            curr_flat = curr_att.view(curr_att.size(0), -1)
            
            # Normalize
            prev_norm = F.normalize(prev_flat, p=2, dim=1)
            curr_norm = F.normalize(curr_flat, p=2, dim=1)
            
            # Compute cosine similarity
            similarity = (prev_norm * curr_norm).sum(dim=1)
            consistencies.append(similarity)
        
        return torch.stack(consistencies).mean()
    
    def compute_predictive_accuracy(self, outputs: Dict) -> float:

        """Compute accuracy of self-model predictions"""
        if not any(outputs.get(f'one_step_{x}_prediction_loss') for x in ['h', 'z', 'att']):
            return 0.0
        
        accuracies = []
        for modality in ['h', 'z', 'att']:
            loss_key = f'one_step_{modality}_prediction_loss'
            if loss_key in outputs and outputs[loss_key]:
                losses = outputs[loss_key]
                # Handle both list of tensors and single tensor cases
                if isinstance(losses, list):
                    loss_tensor = torch.stack([l.detach() for l in losses])
                else:
                    loss_tensor = losses.detach()
                
                # Compute negative mean as accuracy proxy
                accuracy = -loss_tensor.mean().cpu().item()
                accuracies.append(accuracy)
        
        return np.mean(accuracies) if accuracies else 0.0

    
    def visualize_results(self, epoch: int):
        self.model.eval()
        
        # Get training batch
        train_batch = next(iter(self.train_loader))
        train_obs = train_batch['observations'].to(self.device)
        train_actions = train_batch['actions'].to(self.device)
        
        # Get eval batch (with randomization)
        batch_idx = random.randint(0, len(self.eval_loader) - 1)
        for i, eval_batch in enumerate(self.eval_loader):
            if i == batch_idx:
                break
        eval_obs = eval_batch['observations'].to(self.device)
        eval_actions = eval_batch['actions'].to(self.device)
        
        # Use actual batch size from the data
        batch_size = min(train_obs.shape[0], eval_obs.shape[0], 4)  # Cap at 4 for visualization
        subplot_size = 4
        fig_width = 5 * subplot_size  # 5 columns for each type of visualization
        fig_height = 2 * batch_size * 4

        fig, axes = plt.subplots(2 * batch_size, 5, figsize=(fig_width, fig_height))
        
        with torch.no_grad():
            # Process both batches
            out=[]
            for dataset_idx, (observations, actions, dataset_name) in enumerate([
                (train_obs[:batch_size], train_actions[:batch_size], "Train"),
                (eval_obs[:batch_size], eval_actions[:batch_size], "Eval")
            ]):
                _, outputs = self.model.compute_total_loss(
                    observations=observations,
                    actions=actions
                )
                out.append(outputs)
                perceiver_reconst= outputs['perceiver_recon']
                for i in range(batch_size):
                    row_idx = dataset_idx * batch_size + i
                    
                    # Extract first RGB frame from stacked frames
                    # observations shape: [batch, seq, channels, H, W]
                    # For 3 stacked frames with 3 channels each: channels = 9
                    # Get first frame's RGB channels (indices 0, 1, 2)
                    orig_img = observations[i, 0, :3]  # [3, H, W]
                    
                    # Original
                    axes[row_idx, 0].imshow(self.denormalize_image(orig_img))
                    axes[row_idx, 0].set_title(f'{dataset_name} Original')
                    axes[row_idx, 0].set_aspect('equal')
                    axes[row_idx, 0].axis('off')
                    
                    # Reconstruction - also extract first RGB frame
                    recon_img = outputs['reconstructions'][i, 0, :3]  # [3, H, W]
                    axes[row_idx, 1].imshow(self.denormalize_image(recon_img))
                    axes[row_idx, 1].set_title(f'{dataset_name} Reconstruction')
                    axes[row_idx, 1].set_aspect('equal')
                    axes[row_idx, 1].axis('off')
                    seq_len = observations.shape[1]
                    perceiver_idx = i * seq_len + 0  # First timestep
                    
                    # Extract and reshape perceiver reconstruction
                    perceiver_img = perceiver_reconst[perceiver_idx]  # [H*W, C]
                    H, W = self.model.image_size, self.model.image_size
                    perceiver_img = perceiver_img.reshape(H, W, -1)  # [H, W, C]
                    perceiver_img = perceiver_img[:,:, :3]  # Keep only RGB channels
                    
                    # Convert to [C, H, W] for denormalization
                    perceiver_img = torch.from_numpy(perceiver_img.cpu().numpy()).permute(2, 0, 1)
                    
                    axes[row_idx, 2].imshow(self.denormalize_image(perceiver_img))
                    axes[row_idx, 2].set_title(f'{dataset_name} Perceiver Recon')
                    axes[row_idx, 2].set_aspect('equal')
                    axes[row_idx, 2].axis('off')
                                 
                    # Attention map
                    if 'attention_maps' in outputs and outputs['attention_maps'] is not None:
                        att_map = outputs['attention_maps'][i, 0].cpu().numpy()
                        axes[row_idx, 3].imshow(att_map, cmap='hot')
                        axes[row_idx, 3].set_title(f'{dataset_name} Attention')
                        axes[row_idx, 3].set_aspect('equal')
                        axes[row_idx, 3].axis('off')

                    # Cluster weights
                    if 'prior_params' in outputs and len(outputs['prior_params']) > 0:
                        pi = outputs['prior_params'][0]['pi'][i].cpu().numpy()
                        axes[row_idx, 4].bar(range(len(pi)), pi)
                        axes[row_idx, 4].set_title(f'{dataset_name} Clusters')
                        axes[row_idx, 4].set_xlabel('Component')
                        axes[row_idx, 4].set_ylabel('Weight')
                        axes[row_idx, 4].set_box_aspect(1)  # Make it square
                    
                        # Set reasonable limits
                        axes[row_idx, 4].set_xlim(-0.5, len(pi)-0.5)
                        axes[row_idx, 4].set_ylim(0, 1.0)  # Assuming weights are normalized
                    
                        # Adjust tick labels for readability
                        axes[row_idx, 4].tick_params(axis='both', which='major', labelsize=8)

        # Ensure tight layout with consistent spacing
        plt.tight_layout(pad=1.0, h_pad=1.5, w_pad=1.5)
        plt.rcParams["figure.dpi"] = 250

        if self.use_wandb:
            wandb.log({f'train_eval_viz/epoch_{epoch}': wandb.Image(fig)})
        elif self.writer:
            self.writer.add_figure(f'train_eval_viz/epoch_{epoch}', fig, epoch)
        else:
            plt.savefig(f'train_eval_viz_epoch_{epoch}.png')
        
        plt.close()

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
        checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch}.pt'
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            torch.save(checkpoint, 'checkpoints/best_model.pt')
    
    def train(self, n_epochs: int):
        """Main training loop"""
        print(f"Starting training for {n_epochs} epochs...")
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Eval dataset size: {len(self.eval_dataset)}")
        # Register EMA model
        #self.model.ema_encoder.register()
        #self.model.ema_decoder.register()

        for epoch in range(n_epochs):
            # Update temperature
            self.model.update_temperature(epoch)
            self.model.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            
            # Evaluate
            eval_metrics = self.evaluate(epoch)
            
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
            
            # Visualize periodically
            if epoch % self.config['visualize_every'] == 0:
                self.visualize_results(epoch)
            
            # Save checkpoint
            if epoch % self.config['checkpoint_every'] == 0:
                self.save_checkpoint(epoch)
        # Cleanup
        if self.writer:
            self.writer.close()
        
        print("Training completed!")


def main():
    """Main training script"""
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')  # Enable TensorFloat32 cores
    torch.cuda.empty_cache()  
    # Additional foundational configurations

    # Configuration
    config = {
        # Data settings
        'data_dir': '/media/zsheikhb/29cd0dc6-0ccb-4a96-8e75-5aa530301a7e/home/zahra/Work/progress/transition_data',
        'domain_name': 'humanoid',
        'task_name': 'walk',
        'policy_level': 'expert',
        
        # Model settings
        'max_components': 10,
        'latent_dim': 30,
        'hidden_dim': 32, #must be divisible by 8
        'context_dim': 36,
        'attention_dim': 34,
        'attention_resolution': 16,
        'input_channels': 3*1,  # 3 stacked frames
        'HiP_type': 'Mini',
        
        # Training settings
        'batch_size': 5,
        'sequence_length': 10,
        'frame_stack': 1,
        'img_height': 64,
        'img_width': 64,
        'learning_rate': 0.00008,
        'n_epochs': 400,
        'num_workers': 4,
        
        # Loss weights
        'beta': 1.0,
        'entropy_weight': 1.0,
        'lambda_img': 0.5,
        'lambda_latent': 0.5,
        'lambda_pred': 0.1,
        'lambda_att': 0.1,
        'grad_clip': 1.0,
        'n_critic': 2,
        
        # Logging
        'use_wandb': False,
        'wandb_project': 'dpgmm-vrnn-dmc',
        'wandb_entity': 'zahrasheikh',
        'experiment_name': 'humanoid_walk_expert',
        'visualize_every': 5,
        'checkpoint_every': 10
    }
    
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
        context_dim=config['context_dim'],
        attention_dim=config['attention_dim'],
        action_dim=action_dim,
        sequence_length=config['sequence_length'],
        img_disc_channels=64,
        img_disc_layers=3,
        latent_disc_layers=3,
        device=device,
        input_channels=config['input_channels'],
        learning_rate=config['learning_rate'],
        HiP_type=config['HiP_type'],
        attention_resolution=config['attention_resolution'],
        grad_clip= config['grad_clip'],
    )
    
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

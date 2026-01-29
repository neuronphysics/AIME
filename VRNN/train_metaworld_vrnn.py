"""
Train DPGMM-VRNN World Model on Meta-World Environments

This script trains the VRNN world model on pre-collected Meta-World data.
It reuses the core DPGMM-VRNN architecture from the DMC training code but
uses the MetaWorldDataset for data loading.

Usage:
    python -m VRNN.train_metaworld_vrnn --task_name reach-v3 --n_epochs 100
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict
from distutils.util import strtobool

from torch.utils.data import DataLoader

# Import VRNN model and trainer components
from VRNN.dpgmm_stickbreaking_prior_vrnn import DPGMMVariationalRecurrentAutoencoder
from VRNN.metaworld_dataset import MetaWorldDataset, MetaWorldInfo
from VRNN.dmc_vb_transition_dynamics_trainer import (
    DMCVBTrainer,
    count_parameters,
    list_frozen_params,
)


def str2bool(v):
    return bool(strtobool(v))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DPGMM-VRNN on Meta-World data"
    )
    
    # Data settings
    parser.add_argument("--data_dir", type=str, default="./transition_data",
                        help="Base directory for transition data")
    parser.add_argument("--task_name", type=str, default="reach-v3",
                        help="Meta-World task name (e.g., reach-v3)")
    parser.add_argument("--policy_level", type=str, default="random",
                        help="Policy level for data (random, etc.)")
    
    # Model settings
    parser.add_argument("--max_components", type=int, default=16)
    parser.add_argument("--latent_dim", type=int, default=56)
    parser.add_argument("--hidden_dim", type=int, default=48)
    
    # Training settings
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.0007)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # KL annealing
    parser.add_argument("--beta_min", type=float, default=0.5)
    parser.add_argument("--beta_max", type=float, default=1.0)
    parser.add_argument("--beta_warmup_epochs", type=int, default=20)
    
    # Loss weights
    parser.add_argument("--lambda_img", type=float, default=1.0)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # Logging
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    
    return parser.parse_args()


class MetaWorldTrainer(DMCVBTrainer):
    """
    Trainer for DPGMM-VRNN on Meta-World data.
    
    Extends DMCVBTrainer to use MetaWorldDataset instead of DMCVBDataset.
    """
    
    def __init__(
        self,
        model: nn.Module,
        data_dir: str,
        config: Dict,
        device: torch.device
    ):
        # Store config before calling parent __init__
        self.config = config
        self.device = device
        self.model = model.to(device)
        
        # Create Meta-World datasets
        self.train_dataset = MetaWorldDataset(
            data_dir=data_dir,
            task_name=config['task_name'],
            policy_level=config['policy_level'],
            split='train',
            sequence_length=config['sequence_length'],
            frame_stack=config.get('frame_stack', 1),
            img_height=config.get('img_height', 64),
            img_width=config.get('img_width', 64),
        )
        
        self.eval_dataset = MetaWorldDataset(
            data_dir=data_dir,
            task_name=config['task_name'],
            policy_level=config['policy_level'],
            split='eval',
            sequence_length=config['sequence_length'],
            frame_stack=config.get('frame_stack', 1),
            img_height=config.get('img_height', 64),
            img_width=config.get('img_width', 64),
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True,
            drop_last=True,
        )
        
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True,
            drop_last=False,
        )
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup experiment logging (tensorboard, wandb)."""
        from torch.utils.tensorboard import SummaryWriter
        
        experiment_name = self.config.get('experiment_name', 
                                          f"metaworld_{self.config['task_name']}")
        
        log_dir = Path("runs") / experiment_name
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.checkpoint_dir = log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        print(f"Logging to: {log_dir}")
        
        # Wandb setup
        if self.config.get('use_wandb', False):
            import wandb
            wandb.init(
                project=self.config.get('wandb_project', 'vrnn-metaworld'),
                name=experiment_name,
                config=self.config,
            )
    
    def train(self, n_epochs: int):
        """Training loop."""
        print(f"\nStarting training for {n_epochs} epochs")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Eval batches: {len(self.eval_loader)}")
        
        best_eval_loss = float('inf')
        
        for epoch in range(n_epochs):
            # Compute beta for KL annealing
            if epoch < self.config['beta_warmup_epochs']:
                beta = self.config['beta_min'] + (
                    self.config['beta_max'] - self.config['beta_min']
                ) * (epoch / self.config['beta_warmup_epochs'])
            else:
                beta = self.config['beta_max']
            
            # Train epoch
            train_metrics = self._train_epoch(epoch, beta)
            
            # Eval epoch
            eval_metrics = self._eval_epoch(epoch)
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, eval_metrics, beta)
            
            # Save checkpoint
            if eval_metrics.get('total_loss', float('inf')) < best_eval_loss:
                best_eval_loss = eval_metrics['total_loss']
                self._save_checkpoint(epoch, is_best=True)
            
            if (epoch + 1) % self.config.get('checkpoint_every', 10) == 0:
                self._save_checkpoint(epoch)
        
        print("\nTraining complete!")
        self.writer.close()
    
    def _train_epoch(self, epoch: int, beta: float) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_metrics = {}
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            observations = batch['observations'].to(self.device)
            actions = batch['actions'].to(self.device)
            
            # Forward pass
            self.model.optimizer.zero_grad()
            
            outputs = self.model(
                observations,
                actions,
                beta=beta,
            )
            
            # Backward pass
            loss = outputs['total_loss']
            loss.backward()
            
            # Gradient clipping
            if self.config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.model.optimizer.step()
            
            # Accumulate metrics
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    if key not in epoch_metrics:
                        epoch_metrics[key] = []
                    epoch_metrics[key].append(value.item())
        
        # Average metrics
        return {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
    
    def _eval_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one evaluation epoch."""
        self.model.eval()
        epoch_metrics = {}
        
        with torch.no_grad():
            for batch in self.eval_loader:
                observations = batch['observations'].to(self.device)
                actions = batch['actions'].to(self.device)
                
                outputs = self.model(
                    observations,
                    actions,
                    beta=self.config.get('beta_eval', 1.0),
                )
                
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor) and value.ndim == 0:
                        if key not in epoch_metrics:
                            epoch_metrics[key] = []
                        epoch_metrics[key].append(value.item())
        
        return {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
    
    def _log_metrics(self, epoch: int, train_metrics: Dict, eval_metrics: Dict, beta: float):
        """Log metrics to tensorboard and console."""
        # Console output
        print(f"\nEpoch {epoch + 1}")
        print(f"  Beta: {beta:.3f}")
        print(f"  Train Loss: {train_metrics.get('total_loss', 0):.4f}")
        print(f"  Eval Loss: {eval_metrics.get('total_loss', 0):.4f}")
        
        # Tensorboard
        for key, value in train_metrics.items():
            self.writer.add_scalar(f"train/{key}", value, epoch)
        
        for key, value in eval_metrics.items():
            self.writer.add_scalar(f"eval/{key}", value, epoch)
        
        self.writer.add_scalar("train/beta", beta, epoch)
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model.optimizer.state_dict(),
            'config': self.config,
        }
        
        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")


def main():
    args = parse_args()
    
    # Setup
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration
    config = {
        # Data
        'data_dir': args.data_dir,
        'task_name': args.task_name,
        'policy_level': args.policy_level,
        
        # Model
        'max_components': args.max_components,
        'latent_dim': args.latent_dim,
        'hidden_dim': args.hidden_dim,
        'input_channels': 3,
        'prior_alpha': 16.0,
        'prior_beta': 2.0,
        'dropout': 0.1,
        
        # Training
        'batch_size': args.batch_size,
        'sequence_length': args.sequence_length,
        'frame_stack': 1,
        'img_height': 64,
        'img_width': 64,
        'learning_rate': args.learning_rate,
        'n_epochs': args.n_epochs,
        'num_workers': args.num_workers,
        
        # KL annealing
        'beta_min': args.beta_min,
        'beta_max': args.beta_max,
        'beta_warmup_epochs': args.beta_warmup_epochs,
        'beta_eval': 1.0,
        
        # Loss
        'lambda_img': args.lambda_img,
        'lambda_recon': args.lambda_recon,
        'grad_clip': args.grad_clip,
        
        # Logging
        'experiment_name': args.experiment_name or f"metaworld_{args.task_name}",
        'use_wandb': args.use_wandb and not args.no_wandb,
        'checkpoint_every': 10,
    }
    
    print("\n=== Configuration ===")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Get action dimension for the task
    action_dim = MetaWorldInfo.get_action_dim(config['task_name'])
    print(f"\nAction dimension: {action_dim}")
    
    # Initialize model
    model = DPGMMVariationalRecurrentAutoencoder(
        max_components=config['max_components'],
        input_dim=config['img_height'],
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        action_dim=action_dim,
        sequence_length=config['sequence_length'],
        device=device,
        input_channels=config['input_channels'],
        learning_rate=config['learning_rate'],
        grad_clip=config['grad_clip'],
        prior_alpha=config['prior_alpha'],
        prior_beta=config['prior_beta'],
        dropout=config['dropout'],
    )
    
    # Print model info
    count_parameters(model, print_details=True)
    
    # Initialize trainer
    trainer = MetaWorldTrainer(
        model=model,
        data_dir=config['data_dir'],
        config=config,
        device=device,
    )
    
    # Train
    trainer.train(n_epochs=config['n_epochs'])


if __name__ == '__main__':
    main()

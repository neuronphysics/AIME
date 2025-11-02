"""
Training Module

Training infrastructure for AIME world models.

This module provides organized access to trainers and datasets
for training the DPGMM-VRNN world model.
"""

# Import from legacy location
from legacy.VRNN.dmc_vb_transition_dynamics_trainer import DMCVBTrainer, DMCVBDataset

__all__ = [
    'DMCVBTrainer',
    'DMCVBDataset',
]

"""
World Model Module

High-level wrapper for the complete AIME world model.

This module provides organized access to the main DPGMM-VRNN model
that integrates all five pillars of AIME.
"""

# Import main model from legacy location
from legacy.VRNN.dpgmm_stickbreaking_prior_vrnn import DPGMMVariationalRecurrentAutoencoder

__all__ = [
    'DPGMMVariationalRecurrentAutoencoder',
]

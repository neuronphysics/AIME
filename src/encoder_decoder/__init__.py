"""
Encoder-Decoder Module

VAE Encoder and Decoder components for AIME.

This module provides hierarchical VAE components for encoding observations
to latent representations and decoding them back to observations.
"""

# Import from src.nvae_architecture
from src.nvae_architecture import VAEEncoder, VAEDecoder, GramLoss
from src.nvae_architecture import Swish, SqueezeExcitation, NVAEResidualBlock
from src.nvae_architecture import EncoderResidualBlock, DecoderResidualBlock, MDLHead

__all__ = [
    'VAEEncoder',
    'VAEDecoder',
    'GramLoss',
    'Swish',
    'SqueezeExcitation',
    'NVAEResidualBlock',
    'EncoderResidualBlock',
    'DecoderResidualBlock',
    'MDLHead',
]

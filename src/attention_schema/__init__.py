"""
Attention Schema Module

PILLAR 4: ATTENTION - Precision-Weighted Inference

This module implements the complete attention schema mechanism for AIME,
combining bottom-up (stimulus-driven) and top-down (predictive) attention
processing.

Components:
-----------
- SlotAttention: Object-centric attention routing (Locatello et al. 2020)
- AttentionPosterior: Bottom-up multi-object attention from visual input
- AttentionPrior: Top-down prediction of attention dynamics
- AttentionSchema: Complete attention mechanism integrating posterior and prior
- ConvGRUCell: Spatial temporal modeling utility

Theory:
-------
The attention schema implements "Attention Schema Theory" where attention
itself is modeled as an internal state that can be predicted and controlled.
This enables:
1. Stimulus-driven attention (bottom-up from observations)
2. Predictive attention (top-down from beliefs)
3. Attention dynamics tracking (movement prediction)

Usage:
------
    from src.attention_schema import AttentionSchema

    # Initialize attention module
    attention = AttentionSchema(
        image_size=84,
        attention_resolution=21,
        hidden_dim=256,
        latent_dim=32,
        context_dim=128
    )

    # Compute bottom-up attention
    obs = torch.randn(2, 3, 84, 84)
    h = torch.randn(2, 256)
    c = torch.randn(2, 128)
    attn_probs, coords = attention.posterior_net(obs, h, c)

    # Compute top-down attention prediction
    prev_attn = torch.randn(2, 21, 21)
    z = torch.randn(2, 32)
    pred_attn, info = attention.prior_net(prev_attn, h, z)

Integration with AIME:
----------------------
The attention schema provides precision weighting for the VRNN model:
- Attention map guides VAE decoder (where to reconstruct)
- Slot features provide object-centric representations
- Diversity loss encourages multi-object decomposition
- Dynamics loss ensures temporal consistency
"""

from .slot_attention import SlotAttention
from .attention_posterior import AttentionPosterior
from .attention_prior import AttentionPrior
from .attention_schema import AttentionSchema
from .spatial_utils import ConvGRUCell

__all__ = [
    'SlotAttention',
    'AttentionPosterior',
    'AttentionPrior',
    'AttentionSchema',
    'ConvGRUCell',
]

__version__ = '1.0.0'

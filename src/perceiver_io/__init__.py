"""
Perceiver IO - PILLAR 1: PERCEPTION

Sensory compression via VQ tokenization and context extraction.

This module implements the first pillar of AIME's cognitive architecture:
converting raw video observations into discrete tokens and abstract context vectors.

Key Components:
    - VQPTTokenizer: VQ-VAE for video tokenization
    - PerceiverTokenPredictor: Token-level prediction with Perceiver
    - CausalPerceiverIO: Full pipeline (tokenize + predict + extract context)

Typical Usage:
    >>> from perceiver_io import CausalPerceiverIO
    >>>
    >>> perceiver = CausalPerceiverIO(
    ...     video_shape=(16, 3, 64, 64),  # T, C, H, W
    ...     num_latents=128,
    ...     num_latent_channels=256,
    ...     num_codes=1024,
    ...     downsample=4
    ... )
    >>>
    >>> # Extract context for VRNN
    >>> context = perceiver.extract_context(videos)  # [B, T, 256]

See README.md for theory and detailed documentation.
"""

from .causal_perceiver import CausalPerceiverIO
from .tokenizer import VQPTTokenizer
from .predictor import PerceiverTokenPredictor

__all__ = [
    'CausalPerceiverIO',
    'VQPTTokenizer',
    'PerceiverTokenPredictor',
]

__version__ = '0.1.0'

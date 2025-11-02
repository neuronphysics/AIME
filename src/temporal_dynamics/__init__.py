"""
Temporal Dynamics Module

PILLAR 3: Dynamics - Temporal Prediction and Belief Propagation

This module provides components for modeling temporal dynamics in AIME,
including LSTM-based recurrent processing.
"""

from .lstm import LSTMLayer

__all__ = [
    'LSTMLayer',
]

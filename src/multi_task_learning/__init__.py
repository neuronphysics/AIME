"""
Multi-Task Learning Module

PILLAR 5: Optimization - Gradient Harmony Across Multiple Objectives

This module provides multi-task learning strategies for balancing
multiple loss objectives in AIME.
"""

from .rgb_optimizer import RGB, AbsWeighting
from .loss_aggregator import LossAggregator

__all__ = [
    'RGB',
    'AbsWeighting',
    'LossAggregator',
]

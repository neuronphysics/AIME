"""
Distribution Utilities for Generative Prior

Provides distribution classes used in the DPGMM prior:
- GammaPosterior: Variational posterior for Gamma distributions
- KumaraswamyStable: Numerically stable Kumaraswamy distribution
"""

from .gamma_posterior import GammaPosterior, AddEpsilon
from .Kumaraswamy import KumaraswamyStable

__all__ = [
    'GammaPosterior',
    'AddEpsilon',
    'KumaraswamyStable',
]

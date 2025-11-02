"""
Perceiver IO Core Modules

This package contains the fundamental building blocks for Perceiver IO:
- Attention mechanisms (cross-attention, self-attention)
- Vector quantization (VQ-VAE components)
- Position encodings (RoPE, Fourier)
- Input/output adapters
- Utility functions
"""

# Core attention modules
from .modules import (
    MultiHeadAttention,
    CrossAttention,
    SelfAttention,
    CrossAttentionLayer,
    SelfAttentionLayer,
    SelfAttentionBlock,
    MLP,
    PerceiverEncoder,
    PerceiverDecoder,
    PerceiverIO,
    PerceiverAR,
    CausalSequenceModel,
)

# Vector quantization
from .vector_quantize import (
    VectorQuantize,
    EuclideanCodebook,
    CosineSimCodebook,
)

# Position encodings
from .position import (
    RotaryPositionEmbedding,
    FrequencyPositionEncoding,
    FourierPositionEncoding,
)

# Input/output adapters
from .adapter import (
    InputAdapter,
    RotarySupport,
    OutputAdapter,
    ClassificationOutputAdapter,
    QueryProvider,
    TrainableQueryProvider,
    TokenInputAdapter,
    TokenInputAdapterWithRotarySupport,
    TiedTokenOutputAdapter,
)

__all__ = [
    # Attention
    "MultiHeadAttention",
    "CrossAttention",
    "SelfAttention",
    "CrossAttentionLayer",
    "SelfAttentionLayer",
    "SelfAttentionBlock",
    "MLP",
    "PerceiverEncoder",
    "PerceiverDecoder",
    "PerceiverIO",
    "PerceiverAR",
    "CausalSequenceModel",
    # Vector quantization
    "VectorQuantize",
    "EuclideanCodebook",
    "CosineSimCodebook",
    # Position encodings
    "RotaryPositionEmbedding",
    "FrequencyPositionEncoding",
    "FourierPositionEncoding",
    # Adapters
    "InputAdapter",
    "RotarySupport",
    "OutputAdapter",
    "ClassificationOutputAdapter",
    "QueryProvider",
    "TrainableQueryProvider",
    "TokenInputAdapter",
    "TokenInputAdapterWithRotarySupport",
    "TiedTokenOutputAdapter",
]

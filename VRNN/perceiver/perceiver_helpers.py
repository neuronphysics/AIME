"""Helpers for Perceiver IO and HiP construction."""
import enum
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import List, Sequence, Tuple, Optional
import torch.nn.functional as func
from torch import einsum


# Optional xFormers import
try:
    import xformers.ops as xops
    _XFORMERS_AVAILABLE = True
except Exception:
    xops = None
    _XFORMERS_AVAILABLE = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@enum.unique
class ModelOutputKeys(str, enum.Enum):
    INPUT_RECONSTRUCTION = 'input_reconstruction'
    OUTPUT = 'output'
    LATENTS = 'latents'


def padding_to_make_divisible(index_dim: int, num_groups: int) -> int:
    return num_groups * math.ceil(index_dim / num_groups) - index_dim


def conv_1d(input_channels: int, output_channels: int, init_scale: float = 1.0, with_bias: bool = True) -> nn.Linear:
    """A 1D convolution (fully connected layer) in PyTorch."""
    layer = nn.Linear(in_features=input_channels, out_features=output_channels, bias=with_bias)
    # Initialize weights and biases
    fan_in, fan_out = input_channels, output_channels
    fan_avg = 0.5 * (fan_in + fan_out)
    std = (init_scale / math.sqrt(fan_avg))
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    if with_bias:
        nn.init.zeros_(layer.bias)
    return layer


def f32_softmax(x: torch.Tensor) -> torch.Tensor:
    if x.dtype in [torch.bfloat16, torch.float16]:
        return func.softmax(x.float(), dim=-1).to(x.dtype)
    else:
        return func.softmax(x, dim=-1)




def get_activation(activation_name: str):
    if activation_name == 'sq_relu':
        return lambda x: func.relu(x).pow(2)
    else:
        # Mapping JAX activation functions to PyTorch equivalents.
        activations = {
            'relu': func.relu,
            # Add other mappings as needed
        }
        return activations.get(activation_name, None)


def attend(q, k, v, dropout_prob=0.0, attention_mask=None):
    """Computes multi-head attention using a query, key and value.


    Args:
      q: Query with shape [..., q_indices, num_heads, head_dim].
      k: Key with shape [..., kv_indices, num_heads, head_dim].
      v: Value with shape [..., kv_indices, num_heads, head_dim].
        dropout_prob: dropout probability on the attention weights.
      attention_mask: Array of shape [..., q_indices, kv_indices] indicating
        which keys/vals each query can attend to.
    Returns:
       Output of the attention with shape [..., q_indices, hiddens]
    """
    num_head_channels = q.size(-1)
    attention = torch.einsum('...nhc,...mhc->...hnm', q, k)
    attention *= 1. / math.sqrt(num_head_channels)

    if attention_mask is not None:
        attention = attention.masked_fill(
            attention_mask.unsqueeze(-3) == 0,  # 0 means mask
            float('-inf')
        )

    normalized = func.softmax(attention, dim=-1)
    if dropout_prob > 0:
        dropout_layer = torch.nn.Dropout(dropout_prob)
        normalized = dropout_layer(normalized)

    summed = torch.einsum('...hnm,...mhd->...nhd', normalized, v)

    # Concatenate heads
    summed = summed.flatten(start_dim=-2)

    if attention_mask is not None:
        wipe_attn = torch.all(attention_mask == 0, dim=-1, keepdim=True)
        summed = torch.where(wipe_attn, torch.zeros_like(summed), summed)
    return summed


def assign_groups_to_modalities(
        num_groups: int, index_dim_per_modality: Sequence[int]
) -> Tuple[List[int], int]:
    """Computes the number of groups assigned to each modality in PyTorch."""
    num_modalities = len(index_dim_per_modality)
    if num_modalities > num_groups:
        raise ValueError(
            f'{num_modalities} > {num_groups}. '
            'Can\'t yet deal with groups that have '
            'multiple modalities.')

    extra_groups = num_groups - num_modalities
    num_groups_per_modality = [1] * num_modalities
    index_dim_per_group = torch.tensor(index_dim_per_modality, dtype=torch.float32)

    for _ in range(extra_groups):
        modality = torch.argmax(index_dim_per_group).item()
        num_groups_per_modality[modality] = num_groups_per_modality[modality] + 1
        index_dim_per_group[modality] = (
                index_dim_per_modality[modality] / num_groups_per_modality[modality])

    index_dim_per_group = math.ceil(torch.max(index_dim_per_group).item())
    return num_groups_per_modality, index_dim_per_group


class TrainablePositionEncoding(nn.Module):

    def __init__(self, index_dim: int, num_channels: int = 128, init_scale: float = 1.0):
        super().__init__()
        self._index_dim = int(index_dim)
        self._num_channels = int(num_channels)
        self._init_scale = float(init_scale)

        # Learnable delta table; start small so sinusoidal base dominates early.
        # (zeros is a safe default; small noise also works)
        self.pos_embs = nn.Parameter(
            torch.zeros(self._index_dim, self._num_channels)
        )

        # A single learnable gate to modulate the delta strength; helps stability.
        self.gamma = nn.Parameter(torch.tensor(1.0))

        # Cache for base encodings to avoid recompute when the requested length repeats.
        self.register_buffer("_cached_base", torch.empty(0, 0), persistent=False)
        self._cached_len = 0
        self._cached_dim = 0

        # Optional: small init to the delta (kept tiny)
        if self._init_scale > 0:
            with torch.no_grad():
                
                self.pos_embs = nn.Parameter(self.pos_embs + torch.randn_like(self.pos_embs) * self._init_scale)

    @torch.no_grad()
    def _grow_to(self, new_len: int):
        """Grow learnable delta to new_len; initialize extras near the sinusoidal base."""
        old_len, d = self.pos_embs.shape
        if new_len <= old_len:
            return

        device, dtype = self.pos_embs.device, self.pos_embs.dtype
        # Create new parameter and copy old part
    

        # Initialize the tail close to zero (so base dominates), with tiny noise
        tail = torch.zeros(new_len - old_len, d, device=device, dtype=dtype)
        if self._init_scale > 0:
            
            tail = tail + torch.randn_like(tail) * (self._init_scale * 1e-3)
        
        new_param = torch.cat([self.pos_embs, tail], dim=0)
    
        self.pos_embs = nn.Parameter(new_param, requires_grad=True)

        
        self._index_dim = new_len
        # Invalidate cache (base encodings length changed)
        self._cached_len = 0
        self._cached_dim = 0
        self._cached_base = torch.empty(0, 0, device=device, dtype=dtype)

    def _sinusoidal_base(self, length: int, dim: int, device, dtype):
        """Standard sin/cos positional encoding, cached per (length, dim, device, dtype)."""
        # Reuse cache if possible
        if (self._cached_len == length and self._cached_dim == dim
                and self._cached_base.device == device and self._cached_base.dtype == dtype):
            return self._cached_base

        # positions: [L, 1]
        positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)  # [L,1]
        # half-dim (pairs of sin/cos)
        half = dim // 2
        # inv_freq: [half]
        inv_freq = torch.exp(
            torch.arange(0, half, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(1, half))
        )  # like 1 / (10000^{2k/d})

        # angles: [L, half]
        angles = positions * inv_freq.unsqueeze(0)
        sin = torch.sin(angles)
        cos = torch.cos(angles)

        base = torch.cat([sin, cos], dim=1)  # [L, 2*half]
        if base.shape[1] < dim:
            # pad one column if odd dim
            pad = torch.zeros(length, dim - base.shape[1], device=device, dtype=dtype)
            base = torch.cat([base, pad], dim=1)

        # Cache
        self._cached_base = base
        self._cached_len = length
        self._cached_dim = dim
        return base

    def forward(self, batch_size: int | None = None, index_dim: int | None = None):
        # Resolve requested length
        L = int(index_dim) if index_dim is not None else self._index_dim
        if L > self.pos_embs.size(0):
            self._grow_to(L)

        # Build base on the fly (cached) and add learnable delta
        device, dtype = self.pos_embs.device, self.pos_embs.dtype
        base = self._sinusoidal_base(L, self._num_channels, device, dtype)  # [L, C]
        pos = base + self.gamma * self.pos_embs[:L]                         # [L, C]

        if batch_size is not None:
            pos = pos.unsqueeze(0).expand(int(batch_size), -1, -1)          # [B, L, C]
        return pos


class StochasticDepth(nn.Module):
    """Batch wise Dropout used in EfficientNet/NfNet, optionally sans rescaling."""

    def __init__(self,
                 drop_rate: float,
                 scale_by_keep: bool = False):
        super().__init__()
        self.drop_rate = drop_rate
        self.scale_by_keep = scale_by_keep

    def forward(self,
                x: torch.Tensor,
                is_training: bool):
        if not is_training:
            return x
        batch_size = x.shape[0]
        random_tensor = torch.rand([batch_size] + [1] * (x.ndim - 1), dtype=x.dtype, device=x.device)
        keep_prob = 1. - self.drop_rate
        binary_tensor = torch.floor(keep_prob + random_tensor)
        if self.scale_by_keep:
            x = x / keep_prob
        return x * binary_tensor


class Dense(nn.Module):
    """A Transformer-style dense module to follow attention."""

    def __init__(self,
                 input_channels: int = None,
                 widening_factor: int = 4,
                 dropout_prob: float = 0.0,
                 init_scale: float = 1.,
                 activation_name: str = 'sq_relu'):
        super().__init__()
        self.widening_factor = widening_factor
        self.dropout_prob = dropout_prob
        self.init_scale = init_scale
        self.activation = get_activation(activation_name)  # Assuming get_activation is defined as before
        self.D1 = conv_1d(input_channels=input_channels, output_channels=input_channels * self.widening_factor,
                          init_scale=self.init_scale)
        self.layer_norm_1 = nn.LayerNorm(input_channels * self.widening_factor)
        self.D2 = conv_1d(input_channels=input_channels * self.widening_factor, output_channels=input_channels,
                          init_scale=self.init_scale)
        self.layer_norm_2 = nn.LayerNorm(input_channels)

    def forward(self,
                x: torch.Tensor,
                is_training: bool
                ) -> torch.Tensor:
        x = self.D1(x)
        x = self.layer_norm_1(x)
        x = self.activation(x)
        x = self.D2(x)
        x = self.layer_norm_2(x)
        if is_training:
            x = func.dropout(x, p=self.dropout_prob, training=is_training)
        return x




class Attention(nn.Module):
    """
    Multi-headed cross/self-attention with optional xFormers memory-efficient path.

    """
    def __init__(self,
                 num_heads: int = 8,
                 init_scale: float = 1.0,
                 with_final_bias: bool = True,
                 dropout_prob: float = 0.1,
                 input_q_channel: int = 1,
                 input_k_channel: int = 1,
                 input_v_channel: int = 1,
                 qk_channels: int = 1,
                 v_channels: int = 1,
                 output_channels: int = 1,
                 use_xformers: bool = False,
                 return_attn_probs: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

        # xFormers toggle (and availability)
        self.use_xformers = bool(use_xformers) and _XFORMERS_AVAILABLE
        self.return_attn_probs = bool(return_attn_probs)

        # Store channel sizes
        assert qk_channels % num_heads == 0, "qk_channels must be divisible by num_heads"
        assert v_channels  % num_heads == 0, "v_channels must be divisible by num_heads"
        self.qk_channels = qk_channels
        self.v_channels  = v_channels
        self.qk_channels_per_head = qk_channels // num_heads
        self.v_channels_per_head  = v_channels  // num_heads

        # Projections
        self.query_linear  = conv_1d(input_q_channel, qk_channels, init_scale=init_scale)
        self.key_linear    = conv_1d(input_k_channel, qk_channels, init_scale=init_scale)
        self.value_linear  = conv_1d(input_v_channel, v_channels,  init_scale=init_scale)
        self.attention_output_linear = conv_1d(
            v_channels, output_channels, init_scale=init_scale, with_bias=with_final_bias
        )

        self.dropout = nn.Dropout(dropout_prob)


    def _prepare_mask_for_xformers(self, attention_mask, b, g, q_len, kv_len, dtype, device):
        """Convert {1,0} mask to additive bias of shape [B*G, 1, Q, K] for xFormers."""
        if attention_mask is None:
            return None

        if attention_mask.dim() == 3:      # [B, Q, K] -> [B, G, Q, K]
            attention_mask = attention_mask.unsqueeze(1).expand(b, g, q_len, kv_len)
        elif attention_mask.dim() != 4:    # must be [B, G, Q, K]
            raise ValueError(f"Unexpected attention_mask dimension: {attention_mask.dim()}")

        
        # Build bias in desired dtype/device
        bias = torch.zeros_like(attention_mask, dtype=dtype, device=device)
        bias = bias.masked_fill(attention_mask == 0, float('-inf'))
        # [B, G, Q, K] -> [B*G, 1, Q, K]
        return bias.reshape(b * g, 1, q_len, kv_len)

    def forward(self, inputs_q, inputs_kv, attention_mask=None):
        attn_probs = None

        # 1) Projections
        q = self.query_linear(inputs_q)   # [B, G, Q, C_qk]
        k = self.key_linear(inputs_kv)    # [B, G, K, C_qk]
        v = self.value_linear(inputs_kv)  # [B, G, K, C_v]

        b, g, q_len, _ = q.shape
        _, _, kv_len, _ = k.shape

        # 2) Reshape to heads: [B,G,L,C] -> [B,G,L,H,D]
        q = q.reshape(b, g, q_len, self.num_heads, self.qk_channels_per_head)
        k = k.reshape(b, g, kv_len, self.num_heads, self.qk_channels_per_head)
        v = v.reshape(b, g, kv_len, self.num_heads, self.v_channels_per_head)

        scale = 1.0 / math.sqrt(self.qk_channels_per_head)

        if self.use_xformers:
            # --- xFormers memory-efficient attention ---
            q_flat = q.reshape(b * g, q_len, self.num_heads, self.qk_channels_per_head).contiguous()
            k_flat = k.reshape(b * g, kv_len, self.num_heads, self.qk_channels_per_head).contiguous()
            v_flat = v.reshape(b * g, kv_len, self.num_heads, self.v_channels_per_head).contiguous()

            attn_bias = self._prepare_mask_for_xformers(
                attention_mask, b, g, q_len, kv_len, dtype=q_flat.dtype, device=q_flat.device
            )

            out = xops.memory_efficient_attention(
                q_flat, k_flat, v_flat,
                attn_bias=attn_bias,
                p=self.dropout_prob if self.training else 0.0,
                scale=scale,
            )  # [B*G, Q, H, Dv]

            result = out.reshape(b, g, q_len, self.num_heads * self.v_channels_per_head)

            if self.return_attn_probs:
                # Secondary pass to compute probabilities (optional)
                scores = einsum('bgqhd,bgkhd->bghqk', q, k) * scale
                if attention_mask is not None:
                    if attention_mask.dim() == 3:
                        mask = attention_mask.unsqueeze(1).expand(b, g, q_len, kv_len)
                    else:
                        mask = attention_mask
                    scores = scores.masked_fill(mask.unsqueeze(2) == 0, float('-inf'))
                attn_probs = func.softmax(scores, dim=-1)
                if self.training and self.dropout_prob > 0.0:
                    attn_probs = self.dropout(attn_probs)
        else:
           
            scores = einsum('bgqhd,bgkhd->bghqk', q, k) * scale
            if attention_mask is not None:
                # [B, G, Q, K] or [B, Q, K] -> broadcast to [B, G, 1, Q, K]
                if attention_mask.dim() == 3:
                    mask = attention_mask.unsqueeze(1).expand(b, g, q_len, kv_len)
                elif attention_mask.dim() == 4:
                    mask = attention_mask
                else:
                    raise ValueError("attention_mask must be [B, Q, K] or [B, G, Q, K]")
                scores = scores.masked_fill(mask.unsqueeze(2) == 0, float('-inf'))

            attn_probs = func.softmax(scores, dim=-1)
            attn_probs = self.dropout(attn_probs)

            # [B,G,H,Q,K] x [B,G,K,H,Dv] -> [B,G,Q,H,Dv] -> [B,G,Q,C_v]
            out = einsum('bghqk,bgkhd->bgqhd', attn_probs, v)
            result = out.reshape(b, g, q_len, self.num_heads * self.v_channels_per_head)

        # 4) Final projection
        final_output = self.attention_output_linear(result)
        return final_output, attn_probs

def sinusoidal_time_embedding(T: int, C: int, device):
    # C even is best; if odd, we’ll pad one dim
    half = C // 2
    t = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)     # [T,1]
    freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float32)
                      * (-math.log(10000.0) / max(1, half)))
    angles = t * freqs.unsqueeze(0)                                          # [T, half]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)          # [T, 2*half]
    if emb.shape[1] < C:
        emb = torch.cat([emb, torch.zeros(T, C - emb.shape[1], device=device)], dim=-1)
    return emb

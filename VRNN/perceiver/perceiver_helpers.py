"""Helpers for Perceiver IO and HiP construction."""
import enum
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import List, Sequence, Tuple, Optional
import torch.nn.functional as func
from torch import einsum
from xformers.ops.fmha import memory_efficient_attention
from xformers.ops.fmha.attn_bias import AttentionBias
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
    # init.normal_(layer.weight, mean=0.0, std=init_scale)
    nn.init.xavier_normal_(layer.weight)
    if with_bias:
        init.zeros_(layer.bias)
    return layer


def f32_softmax(x: torch.Tensor) -> torch.Tensor:
    if x.dtype in [torch.bfloat16, torch.float16]:
        return func.softmax(x.float(), dim=-1).to(x.dtype)
    else:
        return func.softmax(x, dim=-1)


# def layer_norm(x: torch.Tensor, name: Optional[str] = None) -> torch.Tensor:
# Note: In PyTorch, LayerNorm is usually used as a module, not a function.
# Hence, you might need to include this in a nn.Module class.
# layer_norm_module = nn.LayerNorm(x.size(-1), elementwise_affine=True).to(device)
# return layer_norm_module(x)


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
    """Computes multi-head attention using a query, key and value with linear time complexity.
    ... indicates multiple batch / group dimensions.
    
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
    # 1. Handle arbitrary batch dimensions
    orig_shape = q.shape
    batch_dims = orig_shape[:-3]
    q_indices, num_heads, head_dim = orig_shape[-3:]
    kv_indices = k.shape[-3]
    
    # Reshape to format expected by xFormers [batch, seq_len, heads, dim]
    batch_size = math.prod(batch_dims) if batch_dims else 1
    q_flat = q.reshape(batch_size, q_indices, num_heads, head_dim).contiguous()
    k_flat = k.reshape(batch_size, kv_indices, num_heads, head_dim).contiguous()
    v_flat = v.reshape(batch_size, kv_indices, num_heads, head_dim).contiguous()
    
    # 2. Handle attention mask
    if attention_mask is not None:
        # Reshape mask to match flattened batch dimensions
        mask_flat = attention_mask.reshape(batch_size, q_indices, kv_indices)
        
        # Create attention bias for xFormers (-inf for masked positions)
        # In the original, 0 means "don't attend here", so we need to invert the mask
        attn_bias = torch.zeros_like(mask_flat, dtype=q.dtype)
        attn_bias = attn_bias.masked_fill(mask_flat == 0, float('-inf'))
        
        # Save wipe_attn for later use (queries with all positions masked)
        wipe_attn = torch.all(mask_flat == 0, dim=-1, keepdim=True)
    else:
        attn_bias = None
        wipe_attn = None
    
    # 3. Call xFormers memory_efficient_attention
    # Scale is applied internally in memory_efficient_attention
    scale = 1.0 / math.sqrt(head_dim)
    output = memory_efficient_attention(
        q_flat, k_flat, v_flat,
        attn_bias=attn_bias,
        p=dropout_prob,
        scale=scale
    )
    
    # 4. Reshape back to original format
    output = output.reshape(*batch_dims, q_indices, num_heads * head_dim)
    
    # 5. Apply the wipe_attn logic (zero out queries with all positions masked)
    # This matches the behavior in the original implementation
    if attention_mask is not None:
        wipe_attn = wipe_attn.reshape(*batch_dims, q_indices, 1)
        output = torch.where(wipe_attn, torch.zeros_like(output), output)
    
    return output


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
        num_groups_per_modality[modality] += 1
        index_dim_per_group[modality] = (
                index_dim_per_modality[modality] / num_groups_per_modality[modality])

    index_dim_per_group = math.ceil(torch.max(index_dim_per_group).item())
    return num_groups_per_modality, index_dim_per_group


class TrainablePositionEncoding(nn.Module):
    """Trainable position encoding."""

    def __init__(self,
                 index_dim: int,
                 num_channels: int = 128,
                 init_scale: float = 1.0):
        super().__init__()
        self._index_dim = index_dim
        self._num_channels = num_channels
        self._init_scale = init_scale
        self.pos_embs = nn.Parameter(torch.randn(index_dim, num_channels).to(device) * init_scale, requires_grad=True)

    def forward(self, batch_size=None):
        pos_embs = self.pos_embs
        if batch_size is not None:
            pos_embs = pos_embs.unsqueeze(0).expand(batch_size, -1, -1)
        return pos_embs


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
    """Multi-headed {cross, self}-attention using xFormers for optimization."""

    def __init__(self,
                 num_heads: int = 8,
                 init_scale: float = 1.0,
                 with_final_bias: bool = True,
                 dropout_prob: float = 0.0,
                 input_q_channel: int = 1,
                 input_k_channel: int = 1,
                 input_v_channel: int = 1,
                 qk_channels: int = 1,
                 v_channels: int = 1,
                 output_channels: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self._init_scale = init_scale
        self.with_final_bias = with_final_bias
        self.dropout_prob = dropout_prob

        # Initialize channel sizes
        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.output_channels = output_channels
        self.input_q_channel = input_q_channel
        self.input_k_channel = input_k_channel
        self.input_v_channel = input_v_channel
        self.output_index_dim = None

        assert self.qk_channels % self.num_heads == 0, "qk_channels must be divisible by num_heads"
        assert self.v_channels % self.num_heads == 0, "v_channels must be divisible by num_heads"

        # Linear transformations for queries, keys, and values
        self.query_linear = conv_1d(input_q_channel, self.qk_channels, init_scale=self._init_scale)
        self.key_linear = conv_1d(input_k_channel, self.qk_channels, init_scale=self._init_scale)
        self.value_linear = conv_1d(input_v_channel, self.v_channels, init_scale=self._init_scale)
        self.attention_output_linear = conv_1d(self.v_channels, self.output_channels, init_scale=self._init_scale)

        # Channel sizes per head
        self.qk_channels_per_head = self.qk_channels // self.num_heads
        self.v_channels_per_head = self.v_channels // self.num_heads

    def reshape_for_heads(self, x, channels_per_head):
        b, g, i, c = x.size()
        return torch.reshape(x, (b, g, i, self.num_heads, channels_per_head))

    def forward(self, inputs_q, inputs_kv, attention_mask=None):
        # Linear projections
        q = self.query_linear(inputs_q)
        k = self.key_linear(inputs_kv)
        v = self.value_linear(inputs_kv)
        
        # Get dimensions
        b, g, q_len, _ = q.shape
        _, _, kv_len, _ = k.shape
        
        # Reshape for multi-head attention
        q = self.reshape_for_heads(q, self.qk_channels_per_head)
        k = self.reshape_for_heads(k, self.qk_channels_per_head)
        v = self.reshape_for_heads(v, self.v_channels_per_head)
        """
        # Prepare for xFormers: merge batch and group dims, move heads dim
        # [b, g, len, h, d] -> [b*g, len, h, d]
        q_flat = q.reshape(-1, q_len, self.num_heads, self.qk_channels_per_head).contiguous()
        k_flat = k.reshape(-1, kv_len, self.num_heads, self.qk_channels_per_head).contiguous()
        v_flat = v.reshape(-1, kv_len, self.num_heads, self.v_channels_per_head).contiguous()
        
        # Process attention mask if provided
        if attention_mask is not None:
            # Reshape mask: [b, g, q_len, kv_len] -> [b*g, q_len, kv_len]
            if len(attention_mask.shape) == 4:  # Mask is [b, g, q_len, kv_len]
                mask_flat = attention_mask.reshape(-1, q_len, kv_len)
            else:  # Need to broadcast if mask is [b, q_len, kv_len]
                mask_flat = attention_mask.unsqueeze(1).expand(b, g, q_len, kv_len)
                mask_flat = mask_flat.reshape(-1, q_len, kv_len)
            
            # Convert to attention bias format (-inf for masked positions)
            attn_bias = torch.zeros_like(mask_flat, dtype=q.dtype)
            attn_bias = attn_bias.masked_fill(mask_flat == 0, float('-inf'))
        else:
            attn_bias = None
        
        # Scale factor for attention
        scale = 1.0 / math.sqrt(self.qk_channels_per_head)
        
        # Call xFormers memory_efficient_attention
        output = memory_efficient_attention(
                                            q_flat, k_flat, v_flat,
                                            attn_bias=attn_bias,
                                            p=self.dropout_prob if self.training else 0.0,
                                            scale=scale,
                                            )
        
        # Reshape back to original format: [b*g, q_len, h*d] -> [b, g, q_len, h*d]
        result = output.reshape(b, g, q_len, self.num_heads * self.v_channels_per_head)
        """


        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.qk_channels_per_head)
        attn_scores = einsum('bgnhc,bgmhc->bghnm', q, k) * scale

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        attn_probs = func.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Combine heads
        result = einsum('bghnm,bgmhd->bgnhd', attn_probs, v)
        b, g, i, h, c = result.shape
        result = torch.reshape(result, (b, g, i, h * c))

        # Final linear projection
        return self.attention_output_linear(result), attn_probs
    

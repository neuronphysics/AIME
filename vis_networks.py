import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict, Union, List
from VRNN.perceiver.utilities import RopePositionEmbedding  # spatial RoPE
from VRNN.perceiver.modules import SeparateKVCrossAttention
import logging
import functools
from torch.utils.checkpoint import checkpoint as _ckpt
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch.autograd import Function
import collections.abc as abc
import timm
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
class AddEpsilon(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        return x + self.eps
 
@torch.jit.script
def check_tensor(tensor: torch.Tensor, name: str) -> None:
    """Validate tensor values for debugging"""
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")

def get_improved_scheduler(optimizer, warmup_steps=10000):
    """Create learning rate scheduler with warmup"""
    def lr_lambda(step):
        if step < warmup_steps:
            return min(1.0, step / warmup_steps)
        return max(0.1, 1.0 - (step - warmup_steps) / (100000 - warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
 
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count


class SlotAttention(nn.Module):
    """
    Locatello et al. 2020 Slot Attention (iterative routing).
    x: [B, N, D] tokens  ->  slots: [B, K, D], attn: [B, K, N] (sum_K=1 per position)
    """
    def __init__(self, num_slots, in_dim, slot_dim, iters=3, mlp_hidden=128, eps=1e-8):
        super().__init__()
        self.K = num_slots
        self.iters = iters
        self.eps = eps

        self.norm_x = nn.LayerNorm(in_dim)
        self.norm_s = nn.LayerNorm(slot_dim)

        # learned slot init (μ, σ)
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim) * 0.02)
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        # projections
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)

        # slot update
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, slot_dim),
        )

        self.scale = slot_dim ** -0.5

    def forward(self, x: torch.Tensor):
        """
        x: [B,N,D]
        seed_slots (optional): [B,K,D] to condition slots (e.g., top-down)
        """
        B, N, D = x.shape
        x = self.norm_x(x)
        k = self.to_k(x)                      # [B,N,D]
        v = self.to_v(x)                      # [B,N,D]

        mu = self.slots_mu.expand(B, self.K, -1)
        sigma = self.slots_logsigma.exp().expand(B, self.K, -1)
        slots = mu + sigma * torch.randn_like(mu)
        

        for _ in range(self.iters):
            s = self.norm_s(slots)
            q = self.to_q(s)                  # [B,K,D]
            # attn logits: [B,K,N]
            attn_logits = torch.einsum('bkd,bnd->bkn', q, k) * self.scale
            # responsibilities over slots (softmax over K for each position)
            attn = F.softmax(attn_logits, dim=1) + self.eps
            attn = attn / attn.sum(dim=1, keepdim=True)

            # updates per slot
            updates = torch.einsum('bkn,bnd->bkd', attn, v)  # [B,K,D]

            # GRU update (per slot)
            slots = self._gru_update(slots, updates)
            slots = slots + self.mlp(slots)

        return slots, attn  # [B,K,D], [B,K,N]

    def _gru_update(self, slots, updates):
        B, K, D = slots.shape
        s = slots.reshape(B*K, D)
        u = updates.reshape(B*K, D)
        out = self.gru(u, s)
        return out.view(B, K, D)

class Chomp2d(torch.nn.Module):
    """
    2D version of Chomp1d.
    Input:  [B, C, H, W]
    Output: [B, C, H - s, W - s]
    """
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size, :-self.chomp_size]


class EAAugmentedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding,
                 alpha, beta, k, v, Nh, att_downsample, with_att_conv,
                 relative=True):
        super(EAAugmentedConv2d, self).__init__()
        self.dk = int(out_channels * k)
        self.dv = int(out_channels * v)
        self.K = k
        self.Nh = Nh  # num_head
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.att_downsample = att_downsample
        self.with_att_conv = with_att_conv
        self.alpha = alpha
        self.beta = beta
        self.relative = relative

        assert self.dk // self.Nh or self.dk == 0

        # 2D conv branch
        self.conv = (
            nn.utils.weight_norm(
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels - self.dv,
                    self.kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            )
            if k < 1
            else None
        )

        self.pool_input = (
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
                bias=False,
            )
            if k > 0
            else None
        )

        self.chomp = Chomp2d(padding) if k < 1 else None

        self.pool_input_att = (
            nn.Conv2d(
                Nh,
                Nh,
                kernel_size,
                dilation=dilation,
                padding=padding,
                bias=False,
            )
            if k > 0
            else None
        )

        if att_downsample and k > 0:
            self.pool_x = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.upsample_x = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=False
            )
            self.pool_att = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.upsample_att = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=False
            )
        else:
            self.pool_x = None
            self.upsample_x = None
            self.pool_att = None
            self.upsample_att = None

        # qkv on 2D feature maps; we flatten H*W later
        self.qkv_conv = (
            nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=1, bias=True)
            if k > 0
            else None
        )

        if with_att_conv and k > 0:
            self.att_conv = nn.Sequential(
                nn.Conv2d(Nh, Nh, 3, padding=1, bias=True),
                nn.GroupNorm(1, Nh),
                nn.LeakyReLU(),
            )
            # currently unused, but fixed channels
            self.att_conv2 = nn.Sequential(
                nn.Conv2d(2 * Nh, Nh, 3, padding=1, bias=True),
                nn.GroupNorm(1, Nh),
                nn.LeakyReLU(),
            )
        else:
            self.att_conv = None
            self.att_conv2 = None

    def forward(self, x, prev_att):
        # x: [B, C, H, W]
        conv_out = None if self.conv is None else self.conv(x)

        # optional cropping like Chomp1d
        if self.K < 1 and self.padding != 0 and self.chomp is not None:
            conv_out = self.chomp(conv_out)

        if self.dk > 0:
            x_att = x
            if self.att_downsample and self.pool_x is not None:
                x_att = self.pool_x(x_att)
                if prev_att is not None and self.pool_att is not None:
                    prev_att = self.pool_att(prev_att)

            B, C, H_att, W_att = x_att.size()
            L = H_att * W_att

            # q, k, v on flattened grid
            q, k, v = self.compute_qkv(x_att, self.dk, self.dv, self.Nh)

            # logits: [B, Nh, L, L]
            logits = torch.matmul(q.transpose(2, 3), k)

            if self.relative:
                rel_logits = self.relative_logits(q)
                logits += rel_logits

            if self.with_att_conv and self.att_conv is not None:
                if prev_att is None:
                    prev_att = logits.detach()
                att_matrix = (1 - self.beta) * logits + self.beta * prev_att
                logits = self.att_conv(att_matrix)
                logits = self.alpha * logits + (1 - self.alpha) * att_matrix

            weights = F.softmax(logits, dim=-1)

            attn_out = torch.matmul(weights, v.transpose(2, 3))  # [B,Nh,L,dvh]
            attn_out = attn_out.reshape(B, self.Nh, self.dv // self.Nh, L)
            attn_out = self.combine_heads_2d(attn_out)           # [B,dv,L]
            attn_out = attn_out.view(B, self.dv, H_att, W_att)   # back to 2D [B,dv,H,W]

            if self.att_downsample and self.upsample_x is not None:
                attn_out = self.upsample_x(attn_out)
            if self.att_downsample and self.upsample_att is not None:
                logits = self.upsample_att(logits)

            # apply same chomp to attn_out so shapes match conv_out
            if self.K < 1 and self.padding != 0 and self.chomp is not None:
                attn_out = self.chomp(attn_out)

            if conv_out is not None:
                output = torch.cat((conv_out, attn_out), dim=1)
            else:
                output = attn_out

            return output, logits

        else:
            logits = None
            return conv_out, logits

    def compute_qkv(self, x, dk, dv, Nh):
        B, _, H, W = x.size()
        qkv = self.qkv_conv(x)              # [B,2*dk+dv,H,W]
        qkv = qkv.reshape(B, 2 * dk + dv, H * W)
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)

        q = self.split_heads(q, Nh)
        k = self.split_heads(k, Nh)
        v = self.split_heads(v, Nh)

        dkh = dk // Nh
        q = q * (dkh ** -0.5)

        return q, k, v

    def split_heads(self, x, Nh):
        batch, channels, seq_len = x.size()
        ret_shape = (batch, Nh, channels // Nh, seq_len)
        return torch.reshape(x, ret_shape)

    def combine_heads_2d(self, x):
        batch, Nh, dv, seq_len = x.size()
        ret_shape = (batch, Nh * dv, seq_len)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        # q: [B, Nh, dk/Nh, L]
        B, Nh, dk, seq_len = q.size()
        q = torch.transpose(q, 2, 3)  # [B, Nh, L, dk/Nh]

        # ephemeral relative embedding; for learnable, move to __init__
        rel_k = q.new_empty(2 * seq_len - 1, self.dk // Nh).normal_()

        rel_logits = self.relative_logits_2d(q, rel_k, seq_len, Nh)
        return rel_logits

    def relative_logits_2d(self, q, rel_k, seq_len, Nh):
        rel_logits = torch.einsum("bhld,md->bhlm", q, rel_k)
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, seq_len, seq_len))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1), device=x.device, dtype=x.dtype)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1), device=x.device, dtype=x.dtype)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]

        return final_x

class AttentionFusion(nn.Module):
    def __init__(self, slot_dim,dropout=0.1):
        super(AttentionFusion, self).__init__()
  
        self.query_fc = nn.Linear(slot_dim, slot_dim)
        self.key_fc = nn.Linear(slot_dim, slot_dim)
        self.value_fc = nn.Linear(slot_dim, slot_dim)
        self.slot_dim = slot_dim
        self.norm = nn.LayerNorm(slot_dim)
        self.fusion_fc = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.LayerNorm(slot_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, slot, fb_slot):
        B, N, D = slot.size()
        query = self.query_fc(slot) 
        key = self.key_fc(fb_slot)  
        value = self.value_fc(fb_slot)  
        
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(D)
        attention_weights = F.softmax(attention_scores, dim=-1)  

        attended_values = torch.matmul(attention_weights, value) 

        output_slot = slot + attended_values  # [batchsize, slot_num, slot_dim]
        fused = self.fusion_fc(output_slot)
        return output_slot


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            expanded_mask: torch.Tensor = mask.unsqueeze(dim=1).repeat(1, dots.size(dim=1), 1, 1)
            max_neg_value = -torch.finfo(dots.dtype).max
            dots.masked_fill_(~expanded_mask, max_neg_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        if return_attention:
            return self.to_out(out), attn

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0., drop_path_prob: float = .0):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

        self.drop_path = timm.layers.DropPath(
            drop_prob=drop_path_prob) if drop_path_prob > .0 else nn.Identity()

    def forward(self, x, mask: torch.Tensor = None, return_attention: bool = False
                ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if return_attention:
            attn, attn_map = self.attn(x, mask=mask, return_attention=return_attention)
        else:
            attn = self.attn(x, mask=mask)
        x = self.drop_path(attn) + x
        x = self.drop_path(self.ff(x)) + x

        if return_attention:
            return x, attn_map
        return x
class BottleneckFusionManyToOne(nn.Module):
    """Implementation of Bottleneck Fusion for two signals."""

    def __init__(self,
                 signals: int,
                 dim: int,
                 depth: int,
                 bottleneck_units: int,
                 heads: int,
                 dim_head: int,
                 mlp_dim,
                 dropout=0.):
        """
        :param signals: Number of signals to fuse.
        :param dim: Dimensionality of the input patches.
        :param depth: Number of fusion layers.
        :param bottleneck_units: Number of bottleneck units.
        :param heads: Number of attention heads in transformer layers.
        :param dim_head: Dimensionality of the input patch processed in each SA head.
        :param mlp_dim: Dimensionality of the MLP layer of the transformer.
        :param dropout: Dropout rate.
        """
        super().__init__()

        self.signals: int = signals
        self.depth: int = depth
        self.bottleneck_units: int = bottleneck_units

        self.layers: nn.ModuleList = nn.ModuleList([
            nn.ModuleList([
                Transformer(dim, heads, dim_head, mlp_dim, dropout) for _ in range(self.signals)
            ]) for _ in range(self.depth)
        ])

        self.bottleneck_token = nn.Parameter(torch.randn((bottleneck_units, dim)))
        nn.init.trunc_normal_(self.bottleneck_token, std=.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: A tensor of shape (B, S, H*W, C)
            where:
            - B: Batch size.
            - S: Number of 2D signals to fuse.
            - H: Height of each 2D signal.
            - W: Width of each 2D signal.
            - C: Dimensionality of each patch.

        :returns: A tensor of shape (B, bottleneck_units, C)
        """
        # Split input signals into (B, H*W, C) tensors.
        signals: list[torch.Tensor] = [torch.squeeze(x[:, i, :, :], dim=1)
                                       for i in range(self.signals)]

        # Repeat the bottleneck units for each sample in batch.
        bottleneck = torch.unsqueeze(self.bottleneck_token, 0).repeat((x.size(dim=0), 1, 1))

        for transformers in self.layers:
            # Append the bottleneck units to each signal.
            signals = [torch.cat([bottleneck, s], dim=1) for s in signals]

            # Pass each signal through a separate transformer.
            signals = [t(s) for t, s in zip(transformers, signals)]

            # Extract output bottleneck tokens.
            signals = [s.split([self.bottleneck_units, x.size(dim=2)], dim=1) for s in signals]
            partial_bottleneck_tokens: list[torch.Tensor] = [s[0] for s in signals]

            bottleneck = torch.mean(torch.stack(partial_bottleneck_tokens), dim=0)
            signals = [s[1] for s in signals]

        return bottleneck



# AttentionPosterior
class AttentionPosterior(nn.Module):

    def __init__(
        self,
        image_size: int = 84,
        attention_resolution: int = 21,
        hidden_dim: int = 256,
        context_dim: int = 128,
        input_channels: int = 3,
        feature_channels: int = 64,
        num_semantic_slots: int = 4,
        num_heads: int = 4,
        attention_fusion_mode: str = "max",  # kept for API, no more gated/weighted heads
        enforce_diversity: bool = True,
        device: Optional[torch.device] = None,
        expected_fused: bool = False,
        use_checkpoint: bool = True,
        ckpt_preserve_rng: bool = True,
        ckpt_use_reentrant: bool = False,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        assert feature_channels % num_heads == 0, "feature_channels must be divisible by num_heads"

        self.image_size           = image_size
        self.attention_resolution = attention_resolution
        self.hidden_dim           = hidden_dim
        self.context_dim          = context_dim
        self.in_channels          = input_channels
        self.feature_channels     = feature_channels
        self.num_semantic_slots   = num_semantic_slots
        self.num_heads            = num_heads
        self.d                    = feature_channels
        self.N                    = attention_resolution * attention_resolution
        self.attention_fusion_mode = attention_fusion_mode
        self.enforce_diversity    = enforce_diversity
        self.expected_fused       = expected_fused

        self.use_checkpoint       = use_checkpoint
        self.ckpt_preserve_rng    = ckpt_preserve_rng
        self.ckpt_use_reentrant   = ckpt_use_reentrant


        self.device = device if device is not None else torch.device("cpu")

        # ========= FPN backbone (image -> fused feature map) =========
        if not self.expected_fused:
            # 84 -> 42 -> 21
            self.pyramid_conv1 = nn.Sequential(
                nn.Conv2d(input_channels, 32, 5, stride=2, padding=2),
                nn.GroupNorm(8, 32),
                nn.SiLU(),
            )
            self.pyramid_conv2 = nn.Sequential(
                nn.Conv2d(32, 48, 5, stride=2, padding=2),
                nn.GroupNorm(12, 48),
                nn.SiLU(),
            )
            self.pyramid_conv3 = nn.Sequential(
                nn.Conv2d(48, feature_channels, 3, stride=1, padding=1),
                nn.GroupNorm(16, feature_channels),
                nn.SiLU(),
            )
            self.pyramid_fusion = nn.Conv2d(32 + 48 + feature_channels, feature_channels, 1)
        else:
            self.pyramid_conv1 = None
            self.pyramid_conv2 = None
            self.pyramid_conv3 = None
            self.pyramid_fusion = None

        # ========= Positional embeddings for H_att x W_att grid =========
        self.row_embed = nn.Parameter(
            torch.randn(attention_resolution, feature_channels // 2) * 0.02
        )
        self.col_embed = nn.Parameter(
            torch.randn(attention_resolution, feature_channels // 2) * 0.02
        )

        # ========= SlotAttention backbone =========
        self.slot_attn = SlotAttention(
            num_slots=num_semantic_slots,
            in_dim=feature_channels,
            slot_dim=feature_channels,
            iters=3,
            mlp_hidden=2 * feature_channels,
        )

        
        # Simple dropout on slots
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

        # ========= Cross-attention #1: inject (hidden + context) into slots =========
        # Q:   [B, 1, hidden_dim + context_dim]
        # K,V: [B, K, feature_channels]  (slots)
        self.slot_latent_fusion = SeparateKVCrossAttention(
            dim_q=hidden_dim + context_dim,
            dim_k=feature_channels,
            dim_v=feature_channels,
            dim_out=feature_channels,
            num_heads=num_heads,
            dropout=dropout_p,
        )

        # ========= Fusion attention (maps <-> enriched slots) =========
        self.attn_map_fusion = AttentionFusion(slot_dim=feature_channels, dropout=dropout_p)
        self.slot_gating_conv = nn.Conv2d(
            feature_channels,           # in_channels = D
            num_semantic_slots,         # out_channels = K
            kernel_size=1
        )

        # Diversity regularizers on slot maps
        self.diversity_loss = torch.tensor(0.0, device=self.device)
        self.ortho_loss     = torch.tensor(0.0, device=self.device)

        # caches
        self._last_saliency_features: Optional[torch.Tensor] = None
        self.slot_attention_maps: Optional[torch.Tensor]     = None
        self.slot_features: Optional[torch.Tensor]           = None
        self.slot_centers: Optional[torch.Tensor]            = None
        self.coords_raw: Optional[torch.Tensor]              = None
        self.group_assignments: Optional[torch.Tensor]       = None
        self.fusion_weights: Optional[torch.Tensor]          = None
        self.fused_features: Optional[torch.Tensor]          = None

        self.to(self.device)

    # ---------- helpers ----------

    def _positional_encoding_2d(self) -> torch.Tensor:
        """
        Returns [1, H*W, D] positional embeddings using row/column embeddings.
        """
        H = W = self.attention_resolution
        device = self.row_embed.device

        y = torch.arange(H, device=device)
        x = torch.arange(W, device=device)

        y_emb = self.row_embed[y]  # [H, D/2]
        x_emb = self.col_embed[x]  # [W, D/2]

        pos = (
            torch.cat(
                [
                    y_emb[:, None, :].expand(H, W, -1),
                    x_emb[None, :, :].expand(H, W, -1),
                ],
                dim=-1,
            )
            .view(1, H * W, self.d)
        )
        return pos

    @staticmethod
    def _compute_diversity_losses(slot_maps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        slot_maps: [B, K, H, W]
        Returns (diversity_loss, ortho_loss placeholder).
        """
        B, K, H, W = slot_maps.shape
        p = slot_maps.view(B, K, -1).softmax(dim=-1)
        p = F.normalize(p, p=2, dim=-1)

        corr = torch.bmm(p, p.transpose(1, 2))  # [B,K,K]
        I = torch.eye(K, device=corr.device).unsqueeze(0)
        div = ((corr - I) ** 2).sum() / (B * K * (K - 1) + 1e-8)
        ortho = torch.zeros((), device=slot_maps.device)
        return div, ortho

    def _maybe_ckpt(self, fn, *args):
        if not self.use_checkpoint:
            return fn(*args)
        return torch.utils.checkpoint.checkpoint(
            fn,
            *args,
            preserve_rng_state=self.ckpt_preserve_rng,
            use_reentrant=self.ckpt_use_reentrant,
        )

    # ---- FPN: image -> fused feature map [B,C,H_att,W_att] ----
    def _fpn_forward(self, observation: torch.Tensor, H_att: int, W_att: int) -> torch.Tensor:
        if self.expected_fused:
            raise RuntimeError("_fpn_forward should not be called when expected_fused=True")

        def _fn(x: torch.Tensor) -> torch.Tensor:
            l1 = self.pyramid_conv1(x)        # [B,32,42,42]
            l2 = self.pyramid_conv2(l1)       # [B,48,21,21]
            l3 = self.pyramid_conv3(l2)       # [B,C,21,21]

            l1_up = F.interpolate(l1, size=(H_att, W_att), mode="bilinear", align_corners=False)
            l2_up = F.interpolate(l2, size=(H_att, W_att), mode="bilinear", align_corners=False)
            l3_up = F.interpolate(l3, size=(H_att, W_att), mode="bilinear", align_corners=False)

            fused = self.pyramid_fusion(torch.cat([l1_up, l2_up, l3_up], dim=1))
            return fused

        return self._maybe_ckpt(_fn, observation)

    # ---- SlotAttention wrapper ----
    def _slot_forward(self, feat_seq: torch.Tensor,):
        def _fn(x: torch.Tensor):
            slots, attn = self.slot_attn(x)
            return slots, attn
        return self._maybe_ckpt(_fn, feat_seq)

    # ---- New fusion: maps + enriched slots -> fused map ----
    def _fusion_forward(
        self,
        slot_maps: torch.Tensor,      # [B, K, H, W]
        slots_enriched: torch.Tensor, # [B, K, D]
        fused_feat: torch.Tensor,     # [B, D, H, W]
    ):
        """
        1) Build per-slot features from the attention maps and conv features.
        2) Fuse these map-features with enriched slots via AttentionFusion.
        3) Turn fused per-slot features into weights over slots and fuse maps.
        """
        B, K, H, W = slot_maps.shape

        # Map -> features using conv features (no extra trainable layer)
        fused_flat    = fused_feat.view(B, self.d, H * W)               # [B, D, N]
        slot_maps_norm = slot_maps.view(B, K, H * W).softmax(dim=-1)    # [B, K, N]

        # map_feats[b,k,:] = sum_n p_k(n) * fused_feat[b,:,n]
        map_feats = torch.einsum("bkn,bdn->bkd", slot_maps_norm, fused_flat)  # [B, K, D]

        # Fuse these with enriched slots
        fused_slot_features = self.attn_map_fusion(map_feats, slots_enriched)  # [B, K, D]
        # 1×1 conv over fused feature map → per-pixel logits for each slot
        gating_base = self.slot_gating_conv(fused_feat)                 # [B, K, H, W]

        # Global slot bias from fused_slot_features
        slot_bias = fused_slot_features.mean(dim=-1, keepdim=True)      # [B, K, 1]
        slot_bias = slot_bias.view(B, K, 1, 1)                          # [B, K, 1, 1]

        gating_logits = gating_base + slot_bias                         # [B, K, H, W]

        # Per-pixel distribution over slots
        gating_map = torch.softmax(gating_logits, dim=1)                # [B, K, H, W]

        # Position-dependent fusion of slots
        fused_map = (slot_maps * gating_map).sum(dim=1)                 # [B, H, W]

        # Keep a global summary per slot (for logging / backward compat)
        fusion_weights = gating_map.mean(dim=(2, 3))                    # [B, K]

        # Optionally cache full gating map if you want it later
        self.fusion_weights_map = gating_map                            # [B, K, H, W]

        return fused_map, fusion_weights, fused_slot_features

    def _centers_forward(
        self,
        slot_maps: torch.Tensor,  # [B, K, H, W]
        fused_map: torch.Tensor,  # [B, H, W]
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute:
          - per-slot centers-of-mass (y,x) in normalized coordinates
          - fused map center-of-mass
          - per-slot probability distributions p over H*W
        """
        B, K, H, W = slot_maps.shape
        device = slot_maps.device

        # Flatten
        slot_maps_flat = slot_maps.view(B, K, H * W)           # [B,K,N]
        fused_flat      = fused_map.view(B, H * W)             # [B,N]

        # Normalize to get probabilities
        p_slots = slot_maps_flat / (slot_maps_flat.sum(dim=-1, keepdim=True) + eps)  # [B,K,N]
        p_fused = fused_flat / (fused_flat.sum(dim=-1, keepdim=True) + eps)          # [B,N]

        # Coordinate grid in [-1,1] x [-1,1]
        ys = torch.linspace(-1.0, 1.0, H, device=device)
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]
        coords = torch.stack([yy, xx], dim=-1).view(1, H * W, 2)  # [1,N,2]

        # Per-slot centers
        slot_centers = torch.einsum("bkn,bnd->bkd", p_slots, coords)  # [B,K,2]
        # Fused center
        fused_center = torch.einsum("bn,bnd->bd", p_fused, coords)    # [B,2]

        return slot_centers, fused_center, p_slots

    # ---------- main forward ----------

    def forward(
        self,
        observation: torch.Tensor,   # [B, C, H_img, W_img]
        hidden_state: torch.Tensor,  # [B, hidden_dim]
        context: torch.Tensor,       # [B, context_dim]
        fused_feat: Optional[torch.Tensor] = None,  # [B, D, H_att, W_att] if expected_fused=True
    ) -> torch.Tensor:
        """
        Returns:
            attention_probs_2d: [B, H_att, W_att], a single fused attention map per batch item.

        Side-effects / caches:
            - self.slot_attention_maps: [B, K, H_att, W_att]
            - self.slot_features:       [B, K, D]
            - self.fusion_weights:      [B, K]
            - self.slot_centers:        [B, K, 2]
            - self.coords_raw:          [B, 2]      (center of fused map)
        """
        B = observation.size(0)
        H_att = W_att = self.attention_resolution

        # ---- FPN features ----
        if self.expected_fused and (fused_feat is not None):
            fused = F.interpolate(
                fused_feat,
                size=(H_att, W_att),
                mode="bilinear",
                align_corners=False,
            )
        else:
            fused = self._fpn_forward(observation, H_att, W_att)
        self._last_saliency_features = fused  # [B,D,H_att,W_att]

        # ---- Flatten + positional encodings ----
        feat_seq = fused.flatten(2).transpose(1, 2)  # [B, N, D]
        pos = self._positional_encoding_2d().to(feat_seq.device)  # [1, N, D]
        feat_seq = feat_seq + pos  # [B,N,D]

        # ---- Seed slots from concat(hidden, context) ----
        top_down = torch.cat([hidden_state, context], dim=-1)  # [B, hidden+context]


        # ---- SlotAttention: slots + K maps ----
        slots, attn = self._slot_forward(feat_seq)  # slots: [B,K,D], attn: [B,K,N]
        slot_maps = attn.view(B, self.num_semantic_slots, H_att, W_att)  # [B,K,H,W]
        slot_features = self.dropout(slots)  # [B,K,D]

        self.slot_attention_maps = slot_maps
        self.slot_features       = slot_features
        self.group_assignments   = attn.transpose(1, 2).view(B, H_att, W_att, self.num_semantic_slots)

        # ---- Cross-attention #1: inject (hidden+context) into slots ----
        top_down_token = top_down.unsqueeze(1)  # [B,1,hidden+context]
        ctx_to_slot = self.slot_latent_fusion(
            q=top_down_token,
            k=slot_features,
            v=slot_features,
        )  # [B,1,D]

        ctx_to_slot = ctx_to_slot.expand(-1, self.num_semantic_slots, -1)  # [B,K,D]
        slots_enriched = slot_features + ctx_to_slot                        # [B,K,D]

        # ---- Fusion: maps + enriched slots -> single attention map ----
        fused_map, fusion_weights, fused_slot_features = self._fusion_forward(
            slot_maps, slots_enriched, fused
        )

        self.fusion_weights = fusion_weights
        self.fused_features = fused_slot_features
        self.slot_centers, self.coords_raw, _ = self._centers_forward(slot_maps, fused_map)

        # Normalize fused_map to a proper probability distribution over H*W
        attention_probs = fused_map.flatten(1).softmax(dim=-1)       # [B,N]
        attention_probs_2d = attention_probs.view(B, H_att, W_att)   # [B,H,W]

        # Optional diversity penalty on slot maps
        if self.enforce_diversity and self.training:
            div, ortho = self._compute_diversity_losses(slot_maps)
            self.diversity_loss = div
            self.ortho_loss     = ortho
        else:
            self.diversity_loss = torch.zeros((), device=slot_maps.device)
            self.ortho_loss     = torch.zeros((), device=slot_maps.device)

        return attention_probs_2d

    @torch.no_grad()
    def slot_colors_from_fused_features(
        fused_features: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Map fused slot features to RGB colors via PCA.

        Args:
            fused_features: [B, K, D] tensor of slot embeddings
                            (e.g. AttentionPosterior.fused_features).
        Returns:
            colors: [K, 3] tensor in [0, 1], one RGB color per slot.
        """
        assert fused_features.ndim == 3, "fused_features must be [B,K,D]"
        B, K, D = fused_features.shape

        # Aggregate over batch → [K, D]
        x = fused_features.mean(dim=0)

        # Center features
        x = x - x.mean(dim=0, keepdim=True)

        # Choose rank for PCA (must be ≤ min(K, D))
        q = min(3, K, D)
        if q == 0:
            # degenerate case, just return zeros
            return torch.zeros(K, 3, device=fused_features.device, dtype=fused_features.dtype)

        # Low-rank PCA
        # x: [K, D], V: [D, q]
        U, S, V = torch.pca_lowrank(x, q=q)
        coords = x @ V[:, :q]  # [K, q]

        # If q < 3, pad extra channels with zeros
        if q < 3:
            pad = (0, 3 - q)  # (pad_left, pad_right) over last dimension
            coords = torch.nn.functional.pad(coords, pad, value=0.0)  # [K,3]

        # Min-max normalize each channel to [0,1]
        c_min = coords.min(dim=0)[0]
        c_max = coords.max(dim=0)[0]
        denom = (c_max - c_min).clamp_min(eps)
        colors = (coords - c_min) / denom

        return colors.clamp(0.0, 1.0)  # [K,3]

    # Visualization 
    @torch.no_grad()
    def visualize_multi_object_attention(
        self,
        observation: torch.Tensor,                 # [B,3,H,W]
        slot_attention_maps: torch.Tensor,         # [B,K,H_att,W_att]
        slot_centers: torch.Tensor,                # [B,K,2] in [-1,1]
        group_assignments: Optional[torch.Tensor] = None,
        return_mode: str = "all",                  # 'all' | 'combined' | 'slots' | 'groups'
    ) -> Dict[str, torch.Tensor]:
        assert observation.ndim == 4, "observation must be [B,3,H,W]"
        assert slot_attention_maps.ndim == 4, "slot_attention_maps must be [B,K,H_att,W_att]"
        assert slot_centers.ndim == 3 and slot_centers.size(-1) == 2, "slot_centers must be [B,K,2]"

        B, C, H, W = observation.shape
        Bm, K, H_att, W_att = slot_attention_maps.shape
        assert Bm == B, "batch mismatch"

        device = observation.device
        dtype  = observation.dtype

        # Normalize to [0,1]
        obs = observation
        if obs.min() < 0:
            obs = (obs + 1) * 0.5
        obs = obs.clamp(0, 1)

        # color palette
        # color palette (default)
        base_colors = torch.tensor(
            [
                [0.9, 0.1, 0.1],
                [0.1, 0.9, 0.1],
                [0.1, 0.1, 0.9],
                [0.9, 0.9, 0.1],
                [0.9, 0.1, 0.9],
                [0.1, 0.9, 0.9],
                [0.8, 0.5, 0.2],
                [0.5, 0.2, 0.8],
            ],
            device=device,
            dtype=dtype,
        )

        # --- NEW: try to derive colors from fused slot features ---
        use_feature_colors = getattr(self, "fused_features", None) is not None
        if use_feature_colors:
            try:
                colors = self.slot_colors_from_fused_features(self.fused_features).to(device=device, dtype=dtype)
                # if shapes mismatch for some reason, fall back
                if colors.shape[0] != K or colors.shape[1] != 3:
                    raise RuntimeError("Unexpected color shape")
            except Exception:
                # fallback to base palette on any failure
                if K <= base_colors.size(0):
                    colors = base_colors[:K]
                else:
                    reps = math.ceil(K / base_colors.size(0))
                    colors = base_colors.repeat(reps, 1)[:K]
        else:
            # original behavior
            if K <= base_colors.size(0):
                colors = base_colors[:K]
            else:
                reps = math.ceil(K / base_colors.size(0))
                colors = base_colors.repeat(reps, 1)[:K]

        colors_bk3 = colors.view(1, K, 3, 1, 1)  # [1,K,3,1,1]

        # upsample slot maps to image size
        slot_maps_up = F.interpolate(
            slot_attention_maps.view(B * K, 1, H_att, W_att),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).view(B, K, H, W)

        m_min = slot_maps_up.amin(dim=(2, 3), keepdim=True)
        m_max = slot_maps_up.amax(dim=(2, 3), keepdim=True)
        slot_maps_norm = (slot_maps_up - m_min) / (m_max - m_min + 1e-8)

        out: Dict[str, torch.Tensor] = {}

        # 1) per-slot overlays
        colored_layers = slot_maps_norm.unsqueeze(2) * colors_bk3           # [B,K,3,H,W]
        overlays_bk = (0.6 * obs.unsqueeze(1) + 0.4 * colored_layers).clamp(0, 1)
        out["slot_overlays"] = overlays_bk.view(B * K, 3, H, W)

        # 2) semantic segmentation
        if group_assignments is not None:
            ga = group_assignments
            assert ga.dim() == 4 and ga.size(0) == B, "group_assignments must be [B,*,*,K] or [B,K,*,*]"
            if ga.shape[1] == K:        # [B,K,H_att,W_att]
                g_bkhw = ga
            else:                       # [B,H_att,W_att,K]
                assert ga.shape[-1] == K
                g_bkhw = ga.permute(0, 3, 1, 2).contiguous()
            g_up = F.interpolate(
                g_bkhw.view(B * K, 1, H_att, W_att),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).view(B, K, H, W)
            g_prob = g_up.softmax(dim=1)
        else:
            g_prob = slot_maps_up.softmax(dim=1)

        seg_idx = g_prob.argmax(dim=1)  # [B,H,W]
        one_hot_seg = F.one_hot(seg_idx, num_classes=K).permute(0, 3, 1, 2).float()  # [B,K,H,W]
        seg_rgb = (one_hot_seg.unsqueeze(2) * colors_bk3).sum(dim=1)                 # [B,3,H,W]
        out["semantic_segmentation"] = (0.7 * obs + 0.3 * seg_rgb).clamp(0, 1)

        # 3) combined_with_contours + centers
        combined = obs.clone()
        k_lap = torch.tensor(
            [[-1., -1., -1.],
             [-1.,  8., -1.],
             [-1., -1., -1.]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 3)

        thr = (slot_maps_up > 0.30).to(dtype)
        for s in range(K):
            edges = F.conv2d(thr[:, s].unsqueeze(1), k_lap, padding=1).squeeze(1)
            edges = (edges > 0.1)
            col = colors[s].view(1, 3, 1, 1)
            combined = torch.where(edges.unsqueeze(1), col, combined)

        if slot_centers is not None:
            xs = ((slot_centers[..., 0] + 1.0) * 0.5) * (W - 1)
            ys = ((slot_centers[..., 1] + 1.0) * 0.5) * (H - 1)
            for b in range(B):
                for s in range(K):
                    x = xs[b, s].round().long().clamp(0, W - 1)
                    y = ys[b, s].round().long().clamp(0, H - 1)
                    combined[b, :, y, x] = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
                    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                        yy = (y + dy).clamp(0, H - 1)
                        xx = (x + dx).clamp(0, W - 1)
                        combined[b, :, yy, xx] = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
                    for ry in range(-2, 3):
                        for rx in range(-2, 3):
                            if ry * ry + rx * rx in (4, 5):
                                yy = (y + ry).clamp(0, H - 1)
                                xx = (x + rx).clamp(0, W - 1)
                                combined[b, :, yy, xx] = colors[s]

        out["combined_with_contours"] = combined.clamp(0, 1)

        if return_mode == "all":
            return out
        return {return_mode: out[return_mode]}

 
class AttentionPrior(nn.Module):
    """
    Attention-based spatial dynamics prior with slot fusion.

    - If `attention` is [B, K, H, W], fuse K slot maps into a single map.
    - If `attention` is [B, H, W] or [B, 1, H, W], use it directly.
    - Flattens fused map, adds 2D RoPE, and uses SeparateKVCrossAttention
      to fuse (h_t, z_t) into the attention tokens.
    - Uses EAAugmentedConv2d on the fused spatial map (plus a previous
      attention matrix) to predict next attention logits, then normalizes
      with a spatial softmax.
    """

    def __init__(
        self,
        attention_resolution: int = 21,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        num_heads: int = 4,
        feature_dim: int = 64,
        num_slots: int = 4,            # K: number of slot maps to fuse
        use_checkpoint: bool = True,
        dropout: float = 0.1,
        rope_base: float = 100.0,
        bottleneck_mlp_dim: int = 32,
    ):
        super().__init__()

        self.attention_resolution = attention_resolution
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.num_slots = num_slots
        self.use_checkpoint = use_checkpoint

        H = W = attention_resolution
        self.spatial_dim = H * W

        # 1) RoPE on the 2D grid
        rope_embed_dim = feature_dim  # e.g. 64
        self.rope = RopePositionEmbedding(
            embed_dim=rope_embed_dim,
            num_heads=1,
            base=rope_base,
            normalize_coords="separate",
            shift_coords=None,
            jitter_coords=None,
            rescale_coords=None,
            dtype=torch.float32,
            device=None,
        )
        # sin + cos concatenated
        self.rope_feat_dim = 2 * rope_embed_dim

        # 2) Cross-attention: fuse (hidden_state, latent_state) into map
        # Q: scalar att + RoPE
        self.q_dim = 1 + self.rope_feat_dim
        self.ctx_dim = hidden_dim + latent_dim

        self.cross_attn = SeparateKVCrossAttention(
            dim_q=self.q_dim,
            dim_k=self.ctx_dim,
            dim_v=self.ctx_dim,
            dim_out=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 2b) Slot fusion: [B, K, H, W] -> [B, H, W] using bottleneck

        # BottleneckFusionManyToOne expects x: [B, S, L, C] and returns [B, bottleneck_units, C]
        # Here: S = num_slots, L = H*W, C = 1, bottleneck_units = H*W
        self.slot_fusion = BottleneckFusionManyToOne(
            signals=num_slots,
            dim=1,                          # scalar per position
            depth=1,                        # simplest: one fusion layer
            bottleneck_units=self.spatial_dim,
            heads=1,
            dim_head=8,
            mlp_dim=bottleneck_mlp_dim,                      # tiny MLP is fine
            dropout=dropout,
        )

        # 3) Spatial dynamics over fused map
        self.spatial_dynamics = EAAugmentedConv2d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=5,
            dilation=1,
            padding=2,
            alpha=0.5,
            beta=0.5,
            k=1.0,
            v=1.0,
            Nh=num_heads,
            att_downsample=False,
            with_att_conv=True,
            relative=True,
        )

        # Optional: motion predictor
        self.movement_predictor = nn.Conv2d(
            in_channels=1,
            out_channels=2,
            kernel_size=3,
            padding=1,
        )

    # Gradient checkpoint helper
    def _maybe_ckpt(self, fn, *tensors, reentrant: bool = False):
        """
        Checkpoint `fn(*tensors)` during training if enabled.
        Only Tensor args are allowed.
        """
        if self.training and self.use_checkpoint:
            return _ckpt(fn, *tensors, use_reentrant=reentrant, preserve_rng_state=False)
        else:
            return fn(*tensors)

    def gradient_checkpointing_disable(self):
        self.use_checkpoint = False

    def gradient_checkpointing_enable(self):
        self.use_checkpoint = True

    # Forward: a_t ~ p(a_t | attention, h_t, z_t, prev_attention)
    def forward(
        self,
        attention: torch.Tensor,               # [B,K,H,W] or [B,H,W] or [B,1,H,W]
        hidden_state: torch.Tensor,            # [B, hidden_dim]
        latent_state: torch.Tensor,            # [B, latent_dim]
        prev_attention: Optional[torch.Tensor] = None,  # [B,H,W] or [B,1,H,W]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            next_attention: [B, H, W]  (softmax-normalized spatial prior)
            aux: dict with intermediate tensors (for debugging/analysis).
        """

        # 0) Fuse K slot maps (if present) into a single [B,H,W] map
        # attention can be:
        #   - [B,K,H,W] (slot maps) -> fuse
        #   - [B,H,W] or [B,1,H,W] (already fused map)
        if attention.dim() == 4 and attention.shape[1] > 1:
            # [B,K,H,W]
            B, K, H, W = attention.shape
            assert H == self.attention_resolution and W == self.attention_resolution, \
                f"attention must be [B,K,{self.attention_resolution},{self.attention_resolution}]"
            assert K == self.num_slots, f"Expected {self.num_slots} slots, got {K}"

            L = H * W
            # reshape to [B, S, L, C] with C=1
            slot_tokens = attention.view(B, K, L, 1)      # [B,K,L,1]
            def _slot_fusion(x):
               return self.slot_fusion(x)
            # fuse K signals -> [B,L,1]
            fused_tokens = self._maybe_ckpt(_slot_fusion, slot_tokens)  # [B,L,1]
            fused_map = fused_tokens.squeeze(-1).view(B, H, W)  # [B,H,W]
        else:
            # [B,H,W] or [B,1,H,W]
            if attention.dim() == 4:
                attention = attention.squeeze(1)
            fused_map = attention
            B, H, W = fused_map.shape
            assert H == self.attention_resolution and W == self.attention_resolution, \
                f"attention must be [B,{self.attention_resolution},{self.attention_resolution}]"

        L = H * W
        prev_att_heads = None
        # prev_attention: if None, just use fused_map (detached) as temporal reference
        if prev_attention is not None:
            if prev_attention.dim() == 4:
                prev_attention = prev_attention.squeeze(1)
            assert prev_attention.shape == fused_map.shape, \
                f"prev_attention {prev_attention.shape} vs fused_map {fused_map.shape}"
            prev_for_motion = prev_attention
        else:
            # fallback: use current fused map as reference for motion
            prev_for_motion = fused_map 

        # 1) Flatten fused map + 2D RoPE → Q tokens
        # scalar per location
        att_flat = fused_map.view(B, L, 1)  # [B,L,1]

        # RoPE returns sin/cos: [L,D_pos]
        sin, cos = self.rope(H=H, W=W)
        pos_feat = torch.cat([sin, cos], dim=-1)           # [L,2*D_pos]
        pos_feat = pos_feat.unsqueeze(0).expand(B, -1, -1) # [B,L,2*D_pos]

        q_tokens = torch.cat([att_flat, pos_feat], dim=-1) # [B,L,q_dim]

        # 2) Context tokens from (h_t, z_t)
        ctx = torch.cat([hidden_state, latent_state], dim=-1)  # [B,D_ctx]
        ctx_tokens = ctx.unsqueeze(1)                          # [B,1,D_ctx]

        def _cross(q, k, v):
            return self.cross_attn(q, k, v)

        fused_tokens = self._maybe_ckpt(_cross, q_tokens, ctx_tokens, ctx_tokens)
        # [B,L,feature_dim]

        fused_feat = fused_tokens.view(B, H, W, self.feature_dim).permute(0, 3, 1, 2)
        # [B,feature_dim,H,W]

        # 3) Spatial dynamics over fused_feat with previous attention matrix
        if prev_attention is not None:
            prev_flat = prev_attention.view(B, L)                         # [B,L]
            prev_outer = torch.einsum("bi,bj->bij", prev_flat, prev_flat) # [B,L,L]
            prev_att_heads = prev_outer.unsqueeze(1).expand(
                -1, self.num_heads, -1, -1
            )  # [B,Nh,L,L]
        else:
            prev_att_heads = None
        def _ea(x, pa):
            return self.spatial_dynamics(x, pa)

        ea_out, ea_logits = self._maybe_ckpt(_ea, fused_feat, prev_att_heads)
        # ea_out:   [B,C_ea,H,W]
        # ea_logits:[B,Nh,L,L] or None

        # 4) Derive scalar attention logits + spatial softmax
        if ea_logits is not None:
            B_l, Nh, L1, L2 = ea_logits.shape
            assert L1 == L2 == L, "EAAugmentedConv2d logits shape mismatch"
            idx = torch.arange(L, device=ea_logits.device)
            self_att = ea_logits[:, :, idx, idx]        # [B,Nh,L]
            att_logits_flat = self_att.mean(dim=1)      # [B,L]
            attention_logits = att_logits_flat.view(B, H, W)
        else:
            # fallback: mean over channels
            attention_logits = ea_out.mean(dim=1)       # [B,H,W]

        att_probs_flat = F.softmax(attention_logits.view(B, -1), dim=-1)
        next_attention = att_probs_flat.view(B, H, W)   # [B,H,W]

        # 5) Motion prediction from prev_attention
        movement = self._maybe_ckpt(self.movement_predictor, prev_for_motion.unsqueeze(1))  # [B,2,H,W]
        dx = movement[:, 0].mean(dim=[1, 2])
        dy = movement[:, 1].mean(dim=[1, 2])

        aux = {
            "fused_slot_maps": fused_map,          # [B,H,W]
            "q_tokens": q_tokens,                  # [B,L,q_dim]
            "fused_tokens": fused_tokens,          # [B,L,feature_dim]
            "fused_feat": fused_feat,              # [B,feature_dim,H,W]
            "ea_out": ea_out,                      # [B,C_ea,H,W]
            "ea_logits": ea_logits,                # [B,Nh,L,L] or None
            "attention_logits": attention_logits,  # [B,H,W]
            "predicted_movement": (dx, dy),
        }

        return next_attention, aux
    
class LinearResidual(nn.Module):
    def __init__(
        self,
        input_feature: int,
        hidden_dim: int,
        *,
        nonlinearity=None,
        norm_type: str = "layer",   # 'layer' | 'batch' | None
        drop_p: float = 0.2         # dropout probability
    ):
        super().__init__()

        nl = nn.SiLU() if nonlinearity is None else nonlinearity
        self.residual_projection = (
            nn.Linear(input_feature, hidden_dim, bias=True)
            if input_feature != hidden_dim else nn.Identity()
        )

        layers = []
    
        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))

        if norm_type == "batch":
            layers.append(nn.BatchNorm1d(hidden_dim, affine=True))
        elif norm_type == "layer":
            layers.append(nn.LayerNorm(hidden_dim))

        layers.extend([nl, nn.Dropout(drop_p)])

        
        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))

        if norm_type == "batch":
            layers.append(nn.BatchNorm1d(hidden_dim, affine=True))
        elif norm_type == "layer":
            layers.append(nn.LayerNorm(hidden_dim))

        layers.extend([nl, nn.Dropout(drop_p)])

        self.fn = nn.Sequential(*layers)
        self.reset_parameters()     # custom init

    def reset_parameters(self):
        """
        Kaiming-uniform (fan_in) for all weight matrices (good for SiLU/ReLU).
        Last linear in the residual branch → weight = 0, bias = 0
          so the block begins as (roughly) the identity mapping.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        last_linear = None
        for m in reversed(self.fn):
            if isinstance(m, nn.Linear):
                last_linear = m
                break
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)

        # If a projection exists, re-init it the same way
        if isinstance(self.residual_projection, nn.Linear):
            nn.init.kaiming_uniform_(
                self.residual_projection.weight, a=math.sqrt(5)
            )
            if self.residual_projection.bias is not None:
                nn.init.zeros_(self.residual_projection.bias)

    def forward(self, x):
        residual = self.residual_projection(x)
        return self.fn(residual) + residual


class attentionBlock(nn.Module):
    def __init__(self, n_emb, n_heads=4):
        super().__init__()
        self.flatten = nn.Flatten(2)
        #self.n_input = n_input
        self.n_emb = n_emb
        self.norm = nn.GroupNorm(4, n_emb)
        self.attention = nn.MultiheadAttention(n_emb, n_heads, bias=True,  batch_first=True)

    def forward(self, x):
        batch_size, n_channels, h, w = x.size()
        residue = x
        x = self.norm(x)
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        x, _ = self.attention(x, x, x)
        x = x.permute(0, 2, 1).view(batch_size, n_channels, h, w)
        
        return x + residue

class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            self.loc = nn.Parameter(-mean)
            self.scale = nn.Parameter(1 / (std + 1e-6))
            

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


class ImageDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=16, n_layers=5, use_actnorm=False, device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):

        super(ImageDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        self.layers = nn.ModuleList()
        
        # Initial layer
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, inplace=False))
        )
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layer=nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=False)
            )
            if n==(n_layers-1):
               layer.append(attentionBlock(ndf* nf_mult))
            self.layers.append(layer)

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=False)
            )
        )
        self.layers.append(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))
        
        self.to(device)

    def get_features(self, x):
        """Extract features from intermediate layers"""
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features

    def forward(self, x, get_features=False):
        """Forward pass with option to return intermediate features"""
        if get_features:
            return self.get_features(x)
        
        # Regular forward pass
        for layer in self.layers:
            x = layer(x)
        return x


class LatentDiscriminator(nn.Module):
    # define the descriminator/critic
    def __init__(self, input_dims, num_layers=4, norm_type='layer', activation= nn.SiLU(), device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        super(LatentDiscriminator, self).__init__()
        self.norm_type = norm_type
        self.activation = activation
        layers = []
        layers.append(nn.Linear(input_dims, input_dims * 2, bias=False))

        if self.norm_type == 'batch':
            layers.append(nn.BatchNorm1d(input_dims * 2))
        elif self.norm_type == 'layer':
            layers.append(nn.LayerNorm(input_dims * 2))
        # Activation Function
        layers.append(self.activation)
        size = input_dims * 2
        # Fully Connected Block
        for i in range(num_layers - 2):
            # residual feedforward Layer
            layers.append(nn.Linear(size, size // 2))

            if self.norm_type == 'batch':
                layers.append(nn.BatchNorm1d(size // 2))
            elif self.norm_type == 'layer':
                layers.append(nn.LayerNorm(size // 2))

            layers.append(self.activation)
            if (i == (num_layers // 2 - 1)):
                # add a residual block
                layers.append(LinearResidual(size // 2, size // 2, nonlinearity=self.activation, norm_type=self.norm_type))
            size = size // 2
        layers.append(nn.Linear(size, size * 2, bias=False))

        if self.norm_type == 'batch':
            layers.append(nn.BatchNorm1d(size * 2))
        elif self.norm_type == 'layer':
            layers.append(nn.LayerNorm(size * 2))

        # Activation Function
        layers.append(self.activation)
        # add anther residual block
        layers.append(LinearResidual(size * 2, size * 2, nonlinearity=self.activation, norm_type=self.norm_type))
        layers.append(nn.Linear(size * 2, 1))
        self.model = nn.Sequential(*layers)
        self.device = device
        self.to(device=self.device)

    def forward(self, x):
        return self.model(x)


def make_temporal_padding_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    """
    lengths: (B,) int tensor of valid lengths
    return:  (B, T) bool tensor, True = valid, False = pad
    """
    device = lengths.device
    ar = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
    return ar < lengths.unsqueeze(1)                  # (B, T)



class CausalConv3d(nn.Module):
    """
    3D convolution that is causal along time: pads only on the "past" side in time,
    """
    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        kernel_size: Tuple[int, int, int] = (3, 4, 4),
        stride_t: int = 1,
        dilation_t: int = 1,
        pad_mode: str = "constant",
        bias: bool = True,
        spatial_padding: Optional[int] = None
    ):
        super().__init__()
        kt, kh, kw = kernel_size

        self.pad_mode = pad_mode
        time_pad = dilation_t * (kt - 1)       
        if spatial_padding is not None:
            h_pad = int(spatial_padding)
            w_pad = int(spatial_padding)
        elif (kh % 2 == 1) and (kw % 2 == 1):
            h_pad = kh // 2
            w_pad = kw // 2
        else:
            h_pad = 0
            w_pad = 0

        time_pad = dilation_t * (kt - 1)  # causal (left) time pad
        self._pad = (w_pad, w_pad, h_pad, h_pad, time_pad, 0)

        self.conv = SpectralNorm(nn.Conv3d(
            chan_in,
            chan_out,
            kernel_size=(kt, kh, kw),
            stride=(stride_t, kh, kw),  
            dilation=(dilation_t, 1, 1),
            bias=bias,
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = F.pad(x, self._pad, mode=self.pad_mode)
        return self.conv(x)

class RoPEMHA(nn.Module):
    """MHA that applies 2D RoPE (from RopePositionEmbedding) to Q/K before attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h = num_heads
        self.d = embed_dim // num_heads
        assert (self.d % 2) == 0, "RoPE requires even head dim (d_head % 2 == 0)"

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _rotate_half(x):
        d2 = x.size(-1) // 2
        x1, x2 = x[..., :d2], x[..., d2:]
        return torch.cat([-x2, x1], dim=-1)

    @staticmethod
    def _apply_rope(x, sin, cos):
        # x: [B, h, N, d], sin/cos: [N, d] -> broadcast to [1,1,N,d]
        sin = sin.unsqueeze(0).unsqueeze(0)
        cos = cos.unsqueeze(0).unsqueeze(0)
        return (x * cos) + (RoPEMHA._rotate_half(x) * sin)

    def forward(self, x, sin_spatial=None, cos_spatial=None):
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,h,N,d]

        if (sin_spatial is not None) and (cos_spatial is not None):
            q = self._apply_rope(q, sin_spatial, cos_spatial)
            k = self._apply_rope(k, sin_spatial, cos_spatial)

        att = (q @ k.transpose(-2, -1)) * (self.d ** -0.5)
        att = att.softmax(dim=-1)
        att = self.drop(att)

        y = (att @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(y)


class SpatialTemporalTokenizer(nn.Module):

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        patch_size: Tuple[int, int] = (8, 8),
        kernel_t: int = 3,
        stride_t: int = 1,
    ):
        super().__init__()
        ph, pw = patch_size
        self.patch_size = patch_size

        # causal 3D conv to form non-overlapping spatial patches, causal along time
        self.patch_embed = CausalConv3d(
            chan_in=input_channels,
            chan_out=hidden_dim,
            kernel_size=(kernel_t, ph, pw),
            stride_t=stride_t,
            dilation_t=1,
            pad_mode="constant",
            bias=True,
            spatial_padding=0
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # x: (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.transpose(1, 2)
        x = self.patch_embed(x)                   # (B, D, T', H', W')
        B, D, Tn, Hn, Wn = x.shape
        x = x.permute(0, 2, 3, 4, 1)              # (B, T', H', W', D)
        x = x.reshape(B, Tn, Hn * Wn, D)          # (B, T', N, D)
        x = self.norm(x)
        return x, (Hn, Wn)



class TemporalDiscriminator(nn.Module):
    """
    Temporal discriminator conditioned on latent z_t (per-timestep).

    Expects:
        x: [B, T, C, H, W]
        z: [B, T, z_dim] or [B, z_dim] or None
        sequence_lengths: [B] (optional)
    """

    def __init__(
        self,
        input_channels: int = 3,
        image_size: int = 64,
        hidden_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        max_sequence_length: int = 32,
        patch_size: int = 8,
        z_dim: int = 32,
        film_scale: float = 0.5,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        rope_base: float = 10000.0,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.image_size = image_size
        self.max_sequence_length = max_sequence_length
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.device = device
        self.z_dim = z_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.use_checkpoint = use_checkpoint
        self.ckpt_use_reentrant = False
        self.ckpt_preserve_rng = True

        # --- Spatial–temporal tokenizer (3D conv -> patches) ---
        self.frame_encoder = SpatialTemporalTokenizer(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            patch_size=(patch_size, patch_size),
            kernel_t=3,
            stride_t=1,
        )

        # Learnable temporal embeddings (temporal index)
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, max_sequence_length, 1, hidden_dim) * 0.02
        )

        # ---- RoPE for spatial positions (computed once per forward) ----
        rope_heads_spatial = max(1, n_heads // 2)
        self.spatial_rope = RopePositionEmbedding(
            embed_dim=hidden_dim,
            num_heads=rope_heads_spatial,
            base=rope_base,
            normalize_coords="separate",
            shift_coords=0.01,
            jitter_coords=1.02,
            dtype=torch.float32,
            device=device,
        )

        # Spatial attention per frame (RoPE-aware MHA)
        self.spatial_attention = nn.ModuleList(
            [
                RoPEMHA(
                    embed_dim=hidden_dim,
                    num_heads=rope_heads_spatial,
                    dropout=0.1,
                )
                for _ in range(max(1, n_layers // 2))
            ]
        )

        self.temporal_layers = nn.ModuleList(
            [
                Transformer(
                    dim=hidden_dim,
                    heads=n_heads,
                    dim_head=hidden_dim // n_heads,  # 512 / 8 = 64, etc.
                    mlp_dim=2 * hidden_dim,          # matches dim_feedforward
                    dropout=0.1,
                    drop_path_prob=dropout,              # keep 0.0 so we don't need timm.DropPath
                )
                for _ in range(n_layers)
            ]
        )

        # Norms
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, hidden_dim)
        )
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        # -------- Conditioning modules (cGAN-style) --------
        self.film = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, 2 * hidden_dim),
        )
        self.film_scale = nn.Parameter(
            torch.tensor(film_scale, dtype=torch.float32), requires_grad=True
        )

        # Small conditional residuals for heads
        self.head_temporal_cond = nn.Sequential(
            nn.LayerNorm(z_dim), nn.Linear(z_dim, hidden_dim)
        )
        self.head_spatial_cond = nn.Sequential(
            nn.LayerNorm(z_dim), nn.Linear(z_dim, hidden_dim)
        )
        self.head_frame_cond = nn.Sequential(
            nn.LayerNorm(z_dim), nn.Linear(z_dim, hidden_dim)
        )

        # ---- Heads ----
        self.temporal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.spatial_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.per_frame_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.to(device)

    # small helper to optionally checkpoint blocks
    def _maybe_ckpt(self, fn, *args):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                fn,
                *args,
                use_reentrant=self.ckpt_use_reentrant,
                preserve_rng_state=self.ckpt_preserve_rng,
            )
        return fn(*args)

    def forward(self, x, z=None, sequence_lengths=None, return_features=False):
        """
        x: [B, T, C, H, W]
        z: [B, T, z_dim] or [B, z_dim] or None
        sequence_lengths: [B] with valid timesteps (for padding mask)
        """
        B, T, C, H, W = x.shape

        # ---- sequence lengths & padding mask ----
        if sequence_lengths is None:
            sequence_lengths = torch.full(
                (B,),
                T,
                device=self.device,
                dtype=torch.long,
            )
        temporal_mask = make_temporal_padding_mask(
            sequence_lengths, T
        )  # (B, T) bool, True = valid

        # ---- z handling ----
        if z is None:
            z = torch.zeros(
                B,
                T,
                self.z_dim,
                device=self.device,
                dtype=x.dtype,
            )
        elif z.dim() == 2:
            # broadcast [B, z_dim] -> [B, T, z_dim]
            z = z.unsqueeze(1).expand(-1, T, -1)

        # 1) Tokenize video into [B, T, N_patches, D]
        def _frame_encoder(inp):
            return self.frame_encoder(inp)

        tokens, (Hp, Wp) = self._maybe_ckpt(_frame_encoder, x)
        B2, Tn, N, D = tokens.shape
        assert B2 == B and Tn == T, f"Tokenizer changed T: {T} -> {Tn}"

        # 2) Add learned  temporal pos embeddings
        tokens = tokens +  self.temporal_pos_embed[:, :T, :, :]

        # 3) FiLM conditioning in latent space
        film_params = self.film(z)  # [B, T, 2D]
        gamma, beta = film_params.chunk(2, dim=-1)  # [B, T, D] each
        gamma = torch.tanh(gamma) * self.film_scale
        gamma = gamma.unsqueeze(2)  # [B, T, 1, D]
        beta = beta.unsqueeze(2)
        tokens = tokens * (1.0 + gamma) + beta  # [B, T, N, D]

        # 4) Spatial attention per frame (RoPE)
        sin_sp, cos_sp = self.spatial_rope(H=Hp, W=Wp)  # [N, D]
        spatial_feats = []
        for t in range(T):
            s = tokens[:, t]  # [B, N, D]
            for attn in self.spatial_attention:
                s_norm = self.spatial_norm(s)
                s = s + attn(s_norm, sin_sp, cos_sp)
            spatial_feats.append(s)
        spatial_feats = torch.stack(spatial_feats, dim=1)  # [B, T, N, D]

        # 5) Temporal processing: flatten tokens across patches
        flat_tokens = spatial_feats.reshape(B, T * N, D)  # [B, S, D], S = T*N

        # Build mask for the patch tokens: [B, S]
        flat_mask = temporal_mask.unsqueeze(2).expand(-1, -1, N).reshape(B, T * N)

        # Add CLS token at position 0
        cls = self.cls_token.expand(B, 1, D)  # [B, 1, D]
        h = torch.cat([cls, flat_tokens], dim=1)  # [B, 1+S, D]

        # Extend mask: CLS is always valid
        cls_mask = torch.ones(B, 1, device=self.device, dtype=torch.bool)
        full_mask = torch.cat([cls_mask, flat_mask], dim=1)  # [B, 1+S]

        all_hidden = []

        for layer in self.temporal_layers:
            def _layer(h_in, mask_in):
                # mask_in: [B, S'], True = valid tokens
                if mask_in.dtype is not torch.bool:
                    mask_b = mask_in.bool()
                else:
                    mask_b = mask_in

                # Build attention mask: [B, S', S']
                attn_mask = mask_b.unsqueeze(1) & mask_b.unsqueeze(2)

                # Our custom Transformer takes `mask`
                return layer(h_in, mask=attn_mask)

            h = self._maybe_ckpt(_layer, h, full_mask)
            if return_features:
                all_hidden.append(h)

        # Final norm
        h = self.ln_f(h)  # [B, 1+S, D]

        # Separate CLS token and patch tokens
        cls_out = h[:, 0]          # [B, D]
        tokens_out = h[:, 1:]      # [B, S, D] = [B, T*N, D]

        # Reshape tokens back to [B, T, N, D]
        h_4d = tokens_out.reshape(B, T, N, D)  # [B, T, N, D]

        # 6) Heads

        # temporal summary per frame = patch-mean (still used for per-frame stuff)
        temporal_tokens = h_4d.mean(dim=2)  # [B, T, D]

        # a) sequence-level temporal score: use CLS + masked mean z
        idx = torch.arange(B, device=self.device)

        # masked mean of z over valid timesteps
        z_mask = temporal_mask.unsqueeze(-1).float()  # [B, T, 1]
        z_sum = (z * z_mask).sum(dim=1)               # [B, z_dim]
        z_count = z_mask.sum(dim=1).clamp(min=1.0)    # [B, 1]
        z_mean_masked = z_sum / z_count               # [B, z_dim]

        temporal_rep = cls_out + self.head_temporal_cond(z_mean_masked)  # [B, D]
        temporal_score = self._maybe_ckpt(self.temporal_head, temporal_rep)  # [B, 1]
        # b) global spatial score: average over time & patches
        spatial_rep = h_4d.mean(dim=1).mean(dim=1) + self.head_spatial_cond(
            z.mean(dim=1)
        )
        spatial_score = self._maybe_ckpt(self.spatial_head, spatial_rep)  # [B, 1]

        # c) per-frame scores (for consistency regularizers)
        per_frame_rep = temporal_tokens + self.head_frame_cond(z)  # [B, T, D]
        per_frame_scores = self._maybe_ckpt(self.per_frame_head, per_frame_rep)  # [B, T, 1]

        # mask out padded frames, then average
        per_frame_scores = per_frame_scores.masked_fill(
            ~temporal_mask.unsqueeze(-1), 0.0
        )  # zero padded
        valid = temporal_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        per_frame_avg = per_frame_scores.sum(dim=1) / valid  # [B, 1]

        out = {
            "temporal_score": temporal_score,
            "spatial_score": spatial_score,
            "per_frame_scores": per_frame_scores,
            "final_score": temporal_score + spatial_score + per_frame_avg,
            "sequence_lengths": sequence_lengths,
            "spatial_shape": (Hp, Wp),
        }

        if return_features:
            out["features"] = all_hidden   # list of [B, T*N, D]
            out["hidden_3d"] = h_4d        # [B, T, N, D]
            out["cls_token"] = cls_out     # [B, D]

        return out

    # handy for feature-matching pipelines
    def extract_features(self, x, z=None):
        B, T, C, H, W = x.shape
        if z is None:
            z = torch.zeros(B, T, self.z_dim, device=self.device, dtype=x.dtype)

        tokens, (Hp, Wp) = self.frame_encoder(x)  # [B, T, N, D]
        tokens = tokens + self.temporal_pos_embed[:, :T, :, :]

        film_params = self.film(z)
        gamma, beta = film_params.chunk(2, dim=-1)
        gamma = torch.tanh(gamma) * self.film_scale
        tokens = tokens * (1.0 + gamma.unsqueeze(2)) + beta.unsqueeze(2)

        return tokens, (Hp, Wp)

class UpFirDn2d(Function):
    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad
        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape
        
        input = input.reshape(-1, in_h, in_w, 1)
        if not input.is_contiguous():
            input = input.contiguous()
            
        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))
        
        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x
        ctx.out_size = (out_h, out_w)
        ctx.up = (up_x, up_y)
        ctx.down = (down_x, down_y)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)
        
        # Use native implementation for both CPU and GPU
        out = upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, 
                              pad_x0, pad_x1, pad_y0, pad_y1)
        out = out.view(-1, channel, out_h, out_w)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors
        grad_input = None
        
        if ctx.needs_input_grad[0]:
            grad_output_reshaped = grad_output.reshape(-1, ctx.out_size[0], ctx.out_size[1], 1)
            up_x, up_y = ctx.down  # Swap up and down for backward
            down_x, down_y = ctx.up
            pad_x0, pad_x1, pad_y0, pad_y1 = ctx.pad
            kernel_h, kernel_w = grad_kernel.shape
            
            # Calculate padding for backward pass
            g_pad_x0 = kernel_w - pad_x0 - 1
            g_pad_y0 = kernel_h - pad_y0 - 1
            g_pad_x1 = ctx.in_size[3] * ctx.up[0] - ctx.out_size[1] * ctx.down[0] + pad_x0 - ctx.up[0] + 1
            g_pad_y1 = ctx.in_size[2] * ctx.up[1] - ctx.out_size[0] * ctx.down[1] + pad_y0 - ctx.up[1] + 1
            
            grad_input = upfirdn2d_native(grad_output_reshaped, grad_kernel, 
                                         up_x, up_y, down_x, down_y,
                                         g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)
            grad_input = grad_input.view(ctx.in_size[0], ctx.in_size[1], 
                                        ctx.in_size[2], ctx.in_size[3])
        
        return grad_input, None, None, None, None

def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, 
                     pad_x0, pad_x1, pad_y0, pad_y1):
    """Native implementation of upfirdn2d"""
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape
    
    # Upsample by inserting zeros
    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    
    # Pad
    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), 
                     max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, 
              max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
              max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0), :]
    
    # Apply FIR filter (convolution)
    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, 
                      in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    
    # Reshape
    out = out.reshape(-1, minor,
                     in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
                     in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    out = out.permute(0, 2, 3, 1)
    
    # Downsample
    out = out[:, ::down_y, ::down_x, :]
    
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x
    
    return out.view(-1, minor, out_h, out_w).permute(0, 2, 3, 1).reshape(-1, out_h, out_w, minor)

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """Main interface for upfirdn2d operation"""
    if not isinstance(up, abc.Iterable):
        up = (up, up)
    if not isinstance(down, abc.Iterable):
        down = (down, down)
    if len(pad) == 2:
        pad = (pad[0], pad[1], pad[0], pad[1])
    
    # Use custom autograd Function
    out = UpFirDn2d.apply(input, kernel, up, down, pad)
    return out

def get_haar_wavelet(in_channels):
    """Generate Haar wavelet kernels"""
    haar_wav_l = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h[0, 0] = -1 * haar_wav_h[0, 0]
    
    haar_wav_ll = haar_wav_l.T * haar_wav_l
    haar_wav_lh = haar_wav_h.T * haar_wav_l
    haar_wav_hl = haar_wav_l.T * haar_wav_h
    haar_wav_hh = haar_wav_h.T * haar_wav_h
    
    return haar_wav_ll, haar_wav_lh, haar_wav_hl, haar_wav_hh

class HaarTransform(nn.Module):
    """Haar wavelet transform for downsampling"""
    def __init__(self, in_channels):
        super().__init__()
        
        ll, lh, hl, hh = get_haar_wavelet(in_channels)
        
        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)
        
    def forward(self, input):
        ll = upfirdn2d(input, self.ll, down=2)
        lh = upfirdn2d(input, self.lh, down=2)
        hl = upfirdn2d(input, self.hl, down=2)
        hh = upfirdn2d(input, self.hh, down=2)
        
        return torch.cat((ll, lh, hl, hh), 1)

class InverseHaarTransform(nn.Module):
    """Inverse Haar wavelet transform for upsampling"""
    def __init__(self, in_channels):
        super().__init__()
        
        ll, lh, hl, hh = get_haar_wavelet(in_channels)
        self.register_buffer('ll', ll)
        self.register_buffer('lh', -lh)
        self.register_buffer('hl', -hl)
        self.register_buffer('hh', hh)
        
    def forward(self, input):
        ll, lh, hl, hh = input.chunk(4, 1)
        ll = upfirdn2d(ll, self.ll, up=2, pad=(1, 0, 1, 0))
        lh = upfirdn2d(lh, self.lh, up=2, pad=(1, 0, 1, 0))
        hl = upfirdn2d(hl, self.hl, up=2, pad=(1, 0, 1, 0))
        hh = upfirdn2d(hh, self.hh, up=2, pad=(1, 0, 1, 0))
        
        return ll + lh + hl + hh    
    

class ResidualBlock(nn.Module):
    def __init__(
        self,
        n_channels,
        *,
        num_layers=2,
        kernel_size=3,
        dilation=1,
        groups=1,
        rezero=True,
    ):
        super().__init__()
        ch = n_channels
        assert kernel_size % 2 == 1
        pad = kernel_size // 2
        layers = []
        for i in range(num_layers):
            layers.extend(
                [
                    nn.Conv2d(
                        ch,
                        ch,
                        kernel_size=kernel_size,
                        padding=pad,
                        dilation=dilation,
                        groups=groups,
                    ),
                    nn.InstanceNorm2d(ch),  # <--- Insert norm here
                ]
            )
            layers.append(nn.PReLU())
        self.net = nn.Sequential(*layers)
        if rezero:
            self.gate = nn.Parameter(torch.tensor(0.9))
        else:
            self.gate = 1.0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.net(inputs) * self.gate


def log_residual_stack_structure(
    channel_size_per_layer: List[int],
    layers_per_block_per_layer: List[int],
    downsample: int,
    num_layers_per_resolution: List[int],
    encoder: bool = True,
) -> List[str]:
    logging.debug(f"Creating structure with {downsample} downsamples.")
    out = []

    assert len(channel_size_per_layer) == sum(num_layers_per_resolution)
    assert downsample <= len(num_layers_per_resolution)

    layer = 0

    for block_num, num_layers in enumerate(num_layers_per_resolution):
        for _ in range(num_layers):
            out.append(
                "Residual Block with "
                "{} channels and "
                "{} layers.".format(
                    channel_size_per_layer[layer], layers_per_block_per_layer[layer]
                )
            )

            layer = layer + 1
            # if it's not the last layer, check if the next one has more channels and connect them
            # using a conv layer
            if layer < len(channel_size_per_layer):
                if channel_size_per_layer[layer] != channel_size_per_layer[layer - 1]:
                    out.append(
                        "Con2d layer with "
                        "{} input channels and "
                        "{} output channels".format(
                            channel_size_per_layer[layer - 1],
                            channel_size_per_layer[layer],
                        )
                    )
                    # safe_channel_change(channel_size_per_layer, layer, encoder)

        # after the residual block, check if down-sampling (or up-sampling) is required
        if encoder:
            if downsample > 0:
                out.append("Avg Pooling layer.")
                downsample -= 1
        else:
            if block_num + downsample > (len(num_layers_per_resolution) - 1):
                out.append("Interpolation layer.")

    return out

    
def build_residual_stack(
    channel_size_per_layer: List[int],
    layers_per_block_per_layer: List[int],
    downsample: int,
    num_layers_per_resolution: List[int],
    encoder: bool = True,

) -> List[nn.Module]:
    logging.debug(
        "\n".join(
            log_residual_stack_structure(
                channel_size_per_layer=channel_size_per_layer,
                layers_per_block_per_layer=layers_per_block_per_layer,
                downsample=downsample,
                num_layers_per_resolution=num_layers_per_resolution,
                encoder=encoder,
            )
        )
    )
    layers = []

    assert len(channel_size_per_layer) == sum(num_layers_per_resolution)
    assert downsample <= len(num_layers_per_resolution)

    layer = 0

    for block_num, num_layers in enumerate(num_layers_per_resolution):
        for _ in range(num_layers):
            # add a residual block with the required number of channels and layers
            layers.append(
                ResidualBlock(
                    channel_size_per_layer[layer],
                    num_layers=layers_per_block_per_layer[layer],
                )
            )
            layers.append(nn.InstanceNorm2d(channel_size_per_layer[layer]))
            block_end_channels = channel_size_per_layer[layer]
            layer = layer + 1
            # if it's not the last layer, check if the next one has more channels and connect them
            # using a conv layer
            if layer < len(channel_size_per_layer):
                if channel_size_per_layer[layer] != channel_size_per_layer[layer - 1]:
                    # safe_channel_change(channel_size_per_layer, layer, encoder)

                    in_channels = channel_size_per_layer[layer - 1]
                    out_channels = channel_size_per_layer[layer]
                    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
                    block_end_channels = out_channels
        # after the residual blocks, check if down-sampling (or up-sampling) is required
        if encoder:
            if downsample > 0:
                current_channels = block_end_channels
                layers.append(HaarTransform(current_channels))
                next_channel = channel_size_per_layer[layer] if layer < len(channel_size_per_layer) else current_channels
                # After wavelet, we have 4x channels, project back
                layers.append(nn.Conv2d(current_channels * 4, next_channel, 1))
                
                #layers.append(nn.AvgPool2d(kernel_size=2, stride=2),)
                downsample -= 1
        else:
            if block_num + downsample > (len(num_layers_per_resolution) - 1):
                current_channels = block_end_channels
                # Prepare for inverse wavelet (need 4x channels)
            
                if layer < len(channel_size_per_layer):
                    next_channel = channel_size_per_layer[layer]
                else:
                    next_channel = current_channels
                layers.append(nn.Conv2d(current_channels, current_channels * 4, 1))
                layers.append(InverseHaarTransform(current_channels))
                if next_channel != current_channels:

                        layers.append(nn.Conv2d(current_channels, next_channel, kernel_size=1 ))
                #layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))

    return layers

class EMA:
    def __init__(self, model, decay=0.999, use_num_updates=True):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.num_updates = 0
        self.use_num_updates = use_num_updates
        self.register()
    
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        if self.use_num_updates:
            self.num_updates = self.num_updates + 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        else:
            decay = self.decay
            
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name].to(param.data.device, dtype=param.data.dtype)
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters for evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                #param.data = self.shadow[name]
                param.data.copy_(self.shadow[name].to(param.data.device, dtype=param.data.dtype))

    
    def restore(self):
        """Restore original parameters after evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

class VAEEncoder(torch.nn.Module):
    def __init__(
        self,
        channel_size_per_layer: List[int],
        layers_per_block_per_layer: List[int],
        latent_size: int,
        width: int,
        height: int,
        num_layers_per_resolution,
        mlp_hidden_size: int = 512,
        channel_size: int = 64,
        input_channels: int = 3,
        downsample: int = 4,
    ):
        super().__init__()
        self.latent_size = latent_size

        # compute final width and height of feature maps
        inner_width = width // (2**downsample)
        inner_height = height // (2**downsample)

        # conv layers
        layers = [
            nn.Conv2d(input_channels, channel_size, 5, padding=2, stride=2),
            nn.InstanceNorm2d(channel_size),
            nn.GELU(),
        ]
        
        layers.extend(
            build_residual_stack(
                channel_size_per_layer=channel_size_per_layer,
                layers_per_block_per_layer=layers_per_block_per_layer,
                downsample=downsample - 1,
                num_layers_per_resolution=num_layers_per_resolution,
                encoder=True,
            )
        )

        mlp_input_size = channel_size_per_layer[-1] * inner_width * inner_height

        # fully connected MLP with two hidden layers
        layers.extend(
            [
                nn.Flatten(),
                nn.GELU(),
                nn.Linear(mlp_input_size, mlp_hidden_size),
                nn.LayerNorm(mlp_hidden_size),
                nn.GELU(),
            ]
        )

        self.net = nn.Sequential(*layers)
        self.proj  = nn.Linear(mlp_hidden_size, 2*latent_size)
        
    def gradient_checkpointing_enable(self):
        for module in self.net:
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()

    def forward(self, x: torch.Tensor) -> dict:
        self.hlayer = self.net(x)
        mean, logvar = self.proj(self.hlayer).chunk(2, dim=-1)
        logvar = torch.clamp(logvar, -10.0, 2.0)  # numerical stability
        sigma = (logvar * 0.5).exp()
        eps= torch.randn_like(sigma)
        z = mean + eps * sigma
        return z, mean, logvar



class VAEDecoder(torch.nn.Module):
    def __init__(
        self,
        latent_size: int,
        width: int,
        height: int,
        channel_size_per_layer: List[int] = (256, 256, 256, 256, 128, 128, 64, 64),
        layers_per_block_per_layer: List[int] = (2, 2, 2, 2, 2, 2, 2, 2),
        num_layers_per_resolution: List[int] = (2, 2, 2, 2),
        input_channels: int = 3,
        downsample: Optional[int] = 4,
        mlp_hidden_size: Optional[int] = 512,
    ):
        super().__init__()
        # Add memory-efficient settings
        
        # compute final width and height of feature maps
        inner_width = width // (2**downsample)
        inner_height = height // (2**downsample)

        mlp_input_size = channel_size_per_layer[0] * inner_width * inner_height

        # fully connected MLP with two hidden layers
        layers = []
        layers.extend(
            [
                nn.Linear(latent_size, mlp_hidden_size),
                nn.LayerNorm(mlp_hidden_size),
                nn.GELU(),
                nn.Linear(mlp_hidden_size, mlp_input_size),
                nn.LayerNorm(mlp_input_size),
                nn.Unflatten(
                    1,
                    unflattened_size=(
                        channel_size_per_layer[0],
                        inner_height,
                        inner_width,
                    ),
                ),
                # B, 64*4, 4, 4
            ]
        )

        # conv layers
        layers.extend(
            build_residual_stack(
                channel_size_per_layer=channel_size_per_layer,
                layers_per_block_per_layer=layers_per_block_per_layer,
                downsample=downsample,
                num_layers_per_resolution=num_layers_per_resolution,
                encoder=False,
            )
        )
        layers.append(nn.InstanceNorm2d(channel_size_per_layer[-1]))
        layers.append(nn.GELU())
        final_conv = torch.nn.utils.spectral_norm(nn.Conv2d(channel_size_per_layer[-1], input_channels, 5, padding=2))
        
        layers.extend([
                        final_conv,
                        nn.Tanh()  # range of image pixel values between [-1,1]
                      ])
        
        self.net = nn.Sequential(*layers)
        
    def gradient_checkpointing_enable(self):
        for module in self.net:
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


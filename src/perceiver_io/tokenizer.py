"""
VQ-VAE Tokenizer for Video Prediction

Converts video sequences to discrete token sequences using 3D UNet + VQ.

Key Components:
- ResBlock3D, Down3D, Up3D: 3D convolutional building blocks
- UNetEncoder3D, UNetDecoder3D: Spatiotemporal encoders/decoders
- VQPTTokenizer: Complete VQ-VAE pipeline for videos

Supports both 2D (per-frame) and 3D (spatiotemporal) convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, List
from .modules.vector_quantize import VectorQuantize


# ==========================================
# Sampling Utilities
# ==========================================

def top_k_sampling(logits: torch.Tensor, k: int, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply top-k sampling to logits.

    """
    if k <= 0 or k >= logits.shape[-1]:
        # If k is invalid, fall back to standard sampling
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(
            rearrange(probs, '... v -> (...) v'),
            num_samples=1
        ).squeeze(-1)

    # Get top-k values and indices
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)

    # Create a mask for values outside top-k
    logits_filtered = torch.full_like(logits, float('-inf'))
    logits_filtered.scatter_(-1, top_k_indices, top_k_logits)

    # Sample from filtered distribution
    probs = F.softmax(logits_filtered / temperature, dim=-1)
    return torch.multinomial(
        rearrange(probs, '... v -> (...) v'),
        num_samples=1
    ).squeeze(-1)


def nucleus_sampling(logits: torch.Tensor, p: float, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply nucleus (top-p) sampling to logits.

    """
    if p <= 0.0 or p >= 1.0:
        # If p is invalid, fall back to standard sampling
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(
            rearrange(probs, '... v -> (...) v'),
            num_samples=1
        ).squeeze(-1)

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    # Compute cumulative probabilities
    sorted_probs = F.softmax(sorted_logits / temperature, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > p

    # Shift the indices to the right to keep also the first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Create filtered logits
    logits_filtered = logits.clone()
    for batch_idx in range(logits.shape[0] if logits.ndim > 1 else 1):
        if logits.ndim > 1:
            indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
            logits_filtered[batch_idx].scatter_(-1, indices_to_remove, float('-inf'))
        else:
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits_filtered.scatter_(-1, indices_to_remove, float('-inf'))

    # Sample from filtered distribution
    probs = F.softmax(logits_filtered / temperature, dim=-1)
    return torch.multinomial(
        rearrange(probs, '... v -> (...) v'),
        num_samples=1
    ).squeeze(-1)


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Unified token sampling function supporting temperature, top-k, and nucleus sampling.

    """
    if temperature <= 0.0:
        # Greedy sampling
        return logits.argmax(dim=-1)

    # Apply top-k filtering first if specified
    if top_k is not None and top_k > 0:
        return top_k_sampling(logits, top_k, temperature)

    # Apply nucleus sampling if specified
    if top_p is not None and 0.0 < top_p < 1.0:
        return nucleus_sampling(logits, top_p, temperature)

    # Standard temperature sampling
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(
        rearrange(probs, '... v -> (...) v'),
        num_samples=1
    ).squeeze(-1)


# ==========================================
# Helper Utilities
# ==========================================

def _gn_groups(channels: int, min_channels_per_group: int = 8) -> int:
    """Determine number of groups for GroupNorm"""
    divisors = [d for d in [32, 16, 8, 4, 2, 1] if channels % d == 0]
    for d in divisors:
        if channels // d >= min_channels_per_group:
            return d
    return 1


# ==========================================
# VQ Tokenizer Components
# ==========================================


class GEGLU(nn.Module):
    def forward(self, x):
        # Split along channels, not width
        x, gate = x.chunk(2, dim=1)
        return gate * F.gelu(x)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        groups = _gn_groups(out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 2, k, s, p),
            nn.GroupNorm(groups, out_ch * 2),
            GEGLU()
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvGNAct(in_ch, out_ch)
        self.conv2 = ConvGNAct(out_ch, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.skip(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.Conv2d(in_ch, in_ch, 3, 1, 1)
        self.down = nn.Conv2d(in_ch, out_ch, 4, 2, 1)
        self.res = ResBlock(out_ch, out_ch)

    def forward(self, x):
        return self.res(self.down(self.pool(x)))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.res = ResBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        return self.res(torch.cat([self.up(x), skip], dim=1))


class UNetEncoder(nn.Module):
    def __init__(self, in_ch, base_ch, code_dim, downsample):
        super().__init__()
        assert downsample in (2, 4, 8)
        depth = {2: 1, 4: 2, 8: 3}[downsample]
        chs = [base_ch * (2 ** i) for i in range(depth)]

        self.stem = ResBlock(in_ch, base_ch)
        downs = []
        skip_channels = [base_ch]
        curr = base_ch
        for i in range(depth):
            nxt = chs[i]
            downs.append(Down(curr, nxt))
            skip_channels.append(nxt)
            curr = nxt
        self.downs = nn.ModuleList(downs)
        self.to_code = nn.Conv2d(curr, code_dim, 3, 1, 1)
        self.skip_channels = skip_channels

    def forward(self, x):
        skips = [self.stem(x)]
        x = skips[0]
        for d in self.downs:
            x = d(x)
            skips.append(x)
        return self.to_code(x), skips


class UNetDecoder(nn.Module):
    def __init__(self, out_ch, base_ch, code_dim, skip_channels, downsample):
        super().__init__()
        depth = {2: 1, 4: 2, 8: 3}[downsample]
        ups = []
        curr = code_dim
        for i in range(depth, 0, -1):
            skip_ch = skip_channels[i-1] if i-1 >= 0 else base_ch
            out = max(base_ch, skip_ch)
            ups.append(Up(curr, skip_ch, out))
            curr = out
        self.ups = nn.ModuleList(ups)
        self.final = nn.Sequential(
            ResBlock(curr, base_ch),
            nn.Conv2d(base_ch, out_ch, 3, 1, 1)
        )

    def forward(self, z, skips):
        x = z
        for i, up in enumerate(self.ups):
            skip = skips[-(i+2)] if (i+2) <= len(skips) else skips[0]
            x = up(x, skip)
        return self.final(x)


class ConvGNAct3D(nn.Module):
    """3D version of ConvGNAct for video processing"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        groups = _gn_groups(out_ch)
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch * 2, k, s, p),
            nn.GroupNorm(groups, out_ch * 2),
            GEGLU()
        )

    def forward(self, x):
        return self.block(x)


class ResBlock3D(nn.Module):
    """3D version of ResBlock for video processing"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvGNAct3D(in_ch, out_ch)
        self.conv2 = ConvGNAct3D(out_ch, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.skip(x)


class Down3D(nn.Module):
    """3D version of Down for video processing"""
    def __init__(self, in_ch, out_ch, temporal_downsample=True):
        super().__init__()
        self.pool = nn.Conv3d(in_ch, in_ch, 3, 1, 1)
        # Downsample both spatial and temporal if specified
        stride = (2, 2, 2) if temporal_downsample else (1, 2, 2)
        padding = 1 if temporal_downsample else (0, 1, 1)
        self.down = nn.Conv3d(in_ch, out_ch, 4, stride, padding)
        self.res = ResBlock3D(out_ch, out_ch)

    def forward(self, x):
        return self.res(self.down(self.pool(x)))


class Up3D(nn.Module):
    """3D version of Up for video processing"""
    def __init__(self, in_ch, skip_ch, out_ch, temporal_upsample=True):
        super().__init__()
        stride = (2, 2, 2) if temporal_upsample else (1, 2, 2)
        padding = 1 if temporal_upsample else (0, 1, 1)
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 4, stride, padding)
        self.res = ResBlock3D(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x_up = self.up(x)
        if x_up.shape[-3:] != skip.shape[-3:]:
           # resize skip (T,H,W) to match x_up
           skip = F.interpolate(skip, size=x_up.shape[-3:], mode='trilinear', align_corners=False)

        return self.res(torch.cat([x_up, skip], dim=1))


class UNetEncoder3D(nn.Module):
    """3D UNet Encoder for video processing with temporal convolutions"""
    def __init__(self, in_ch, base_ch, code_dim, downsample, temporal_downsample=False):
        super().__init__()
        assert downsample in (2, 4, 8)
        depth = {2: 1, 4: 2, 8: 3}[downsample]
        chs = [base_ch * (2 ** i) for i in range(depth)]

        self.stem = ResBlock3D(in_ch, base_ch)
        downs = []
        skip_channels = [base_ch]
        curr = base_ch
        for i in range(depth):
            nxt = chs[i]
            # Only downsample temporal at first layer to preserve temporal resolution
            temporal_down = temporal_downsample and i == 0
            downs.append(Down3D(curr, nxt, temporal_downsample=temporal_down))
            skip_channels.append(nxt)
            curr = nxt
        self.downs = nn.ModuleList(downs)
        self.to_code = nn.Conv3d(curr, code_dim, 3, 1, 1)
        self.skip_channels = skip_channels

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        skips = [self.stem(x)]
        x = skips[0]
        for d in self.downs:
            x = d(x)
            skips.append(x)
        return self.to_code(x), skips


class UNetDecoder3D(nn.Module):
    """3D UNet Decoder for video processing with temporal convolutions"""
    def __init__(self, out_ch, base_ch, code_dim, skip_channels, downsample, temporal_downsample=False):
        super().__init__()
        depth = {2: 1, 4: 2, 8: 3}[downsample]
        ups = []
        curr = code_dim
        for i in range(depth, 0, -1):
            skip_ch = skip_channels[i-1] if i-1 >= 0 else base_ch
            out = max(base_ch, skip_ch)
            # Only upsample temporal at last layer to preserve temporal resolution
            temporal_up = temporal_downsample and i == depth
            ups.append(Up3D(curr, skip_ch, out, temporal_upsample=temporal_up))
            curr = out
        self.ups = nn.ModuleList(ups)
        self.final = nn.Sequential(
            ResBlock3D(curr, base_ch),
            nn.Conv3d(base_ch, out_ch, 3, 1, 1)
        )

    def forward(self, z, skips):
        x = z
        for i, up in enumerate(self.ups):
            skip = skips[-(i+2)] if (i+2) <= len(skips) else skips[0]
            x = up(x, skip)
        return self.final(x)


class VQPTTokenizer(nn.Module):
    """
    VQ-VAE tokenizer for video sequences.

    Encodes video sequences to discrete tokens via:
    1. UNet encoder (2D or 3D)
    2. Vector quantization
    3. UNet decoder for reconstruction

    Supports both per-frame (2D) and spatiotemporal (3D) processing.
    """
    def __init__(
        self,
        in_channels: int = 3,
        code_dim: int = 256,
        num_codes: int = 1024,
        downsample: int = 4,
        base_channels: int = 64,
        commitment_weight: float = 0.1,
        commitment_use_cross_entropy_loss: bool = True,
        use_cosine_sim: bool = False,
        kmeans_init: bool = True,
        gate_skips: bool = False,
        use_3d_conv: bool = False,
        temporal_downsample: bool = False,
        dropout: float = 0.1,
        num_quantizers: int = 4,
        codebook_diversity_loss_weight: float = 0.01,
        codebook_diversity_temperature: float = 1.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.code_dim = code_dim
        self.num_codes = num_codes
        self.downsample = downsample
        self.use_3d_conv = use_3d_conv
        self.temporal_downsample = temporal_downsample

        # Choose 2D or 3D architecture
        if use_3d_conv:
            self.encoder = UNetEncoder3D(in_channels, base_channels, code_dim,
                                        downsample, temporal_downsample)
            self.decoder = UNetDecoder3D(
                in_channels, base_channels, code_dim,
                self.encoder.skip_channels, downsample, temporal_downsample
            )
        else:
            self.encoder = UNetEncoder(in_channels, base_channels, code_dim, downsample)
            self.decoder = UNetDecoder(
                in_channels, base_channels, code_dim,
                self.encoder.skip_channels, downsample
            )

        self.vq = VectorQuantize(
            dim=code_dim,
            codebook_size=num_codes,
            heads=num_quantizers,
            codebook_dim=code_dim // num_quantizers,
            accept_image_fmap=True,
            channel_last=False,
            commitment_weight=commitment_weight,
            commitment_use_cross_entropy_loss=commitment_use_cross_entropy_loss,
            kmeans_init=kmeans_init,
            use_cosine_sim=use_cosine_sim,
            codebook_diversity_loss_weight=codebook_diversity_loss_weight,
            codebook_diversity_temperature=codebook_diversity_temperature
        )
        self.dropout = nn.Dropout3d(p=dropout) if use_3d_conv else nn.Dropout2d(p=dropout)

        # Optional skip gates (default: OFF, for ablation studies only)
        self.gate_skips = gate_skips
        if gate_skips:
            # Initialize to 1.0 (full skips), not 0.5
            self.skip_gates = nn.ParameterList([
                nn.Parameter(torch.tensor(1.0))
                for _ in self.encoder.skip_channels
            ])

    def encode(self, videos: torch.Tensor):
        """Encode videos to tokens and quantized features (returns skips)."""
        B, T = videos.shape[:2]

        if self.use_3d_conv:
            # 3D conv: process as (B, C, T, H, W)
            x = rearrange(videos, 'b t c h w -> b c t h w')
            z, skips = self.encoder(x)
            T_enc = z.shape[2]
            # Flatten temporal dimension for VQ
            z = rearrange(z, 'b d t h w -> (b t) d h w')
        else:
            # 2D conv: flatten batch and time
            x_flat = rearrange(videos, 'b t c h w -> (b t) c h w')
            z, skips = self.encoder(x_flat)
            T_enc = T  # No temporal downsampling

        quantized, ids, vq_loss = self.vq(z)

        token_ids = rearrange(ids, '(b t) h w -> b t h w', b=B, t=T_enc)
        quantized = rearrange(quantized, '(b t) d h w -> b t d h w', b=B, t=T_enc)

        return token_ids, quantized, vq_loss, skips

    def decode(self, quantized: torch.Tensor, skips: list | None = None, use_tanh: bool = True):
        """Decode quantized features to videos (uses real skips when provided)."""
        B, T_enc = quantized.shape[:2]

        if skips is None:
            # Fallback: dummy skips for generation
            if self.use_3d_conv:

                B, T, D, H, W = quantized.shape
                skips_to_use = []
                for i, ch in enumerate(self.encoder.skip_channels):
                    scale = self.downsample // (2 ** i)
                    #print(f"Scale for skip channel 3D {i}: {scale}")
                    t_scale = 2 if self.temporal_downsample and i == 0 else 1
                    skips_to_use.append(torch.zeros(
                        B, ch, T_enc * t_scale, H * scale, W * scale,
                        device=quantized.device, dtype=quantized.dtype
                    ))
            else:
                _, _, C, H, W = quantized.shape
                q_flat = rearrange(quantized, 'b t d h w -> (b t) d h w')
                skips_to_use = []
                for i, ch in enumerate(self.encoder.skip_channels):
                    scale = self.downsample // (2 ** i)
                    #print(f"Scale for skip channel in 2D {i}: {scale}")
                    skips_to_use.append(torch.zeros(
                        B*T_enc, ch, H*scale, W*scale,
                        device=quantized.device, dtype=q_flat.dtype
                    ))
        else:
            # Apply gates only if enabled (default: no gating)
            if self.gate_skips:
                skips_to_use = [
                    gate.sigmoid() * (self.dropout(s) if self.training else s)
                    for gate, s in zip(self.skip_gates, skips)
                ]
            else:
                # Apply dropout to skips during training
                skips_to_use = [
                    self.dropout(s) if self.training else s
                    for s in skips
                ]

        if self.use_3d_conv:
            # 3D conv: process as (B, C, T, H, W)
            q = rearrange(quantized, 'b t d h w -> b d t h w')
            recon = self.decoder(q, skips_to_use)
            recon = rearrange(recon, 'b c t h w -> b t c h w')
        else:
            # 2D conv: flatten batch and time
            q_flat = rearrange(quantized, 'b t d h w -> (b t) d h w')
            recon = self.decoder(q_flat, skips_to_use)
            recon = rearrange(recon, '(b t) c h w -> b t c h w', b=B, t=T_enc)

        if use_tanh:
            recon = torch.tanh(recon)

        return recon

    def _sanitize_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Ensure token indices are valid for codebook lookup"""
        token_ids = token_ids.long()
        # Codebook only has indices [0, num_codes-1]
        invalid = (token_ids < 0) | (token_ids >= self.num_codes)
        if invalid.any():
            # Replace invalid tokens with a valid default (0)
            token_ids = torch.where(invalid, torch.zeros_like(token_ids), token_ids)
        return token_ids

    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Get code vectors from token indices using public VQ API"""
        indices = self._sanitize_token_ids(indices)
        indices = indices.clamp(0, self.num_codes - 1)  # Extra safety

        return self.vq.get_codes_from_indices(indices)

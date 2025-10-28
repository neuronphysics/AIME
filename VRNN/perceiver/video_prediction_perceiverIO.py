
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from typing import Optional, Tuple, Dict, List
import lpips
import numpy as np
from VRNN.perceiver.adapter import TiedTokenOutputAdapter, TrainableQueryProvider
from VRNN.perceiver.modules import (
    CrossAttention, SelfAttentionBlock
)
from VRNN.perceiver.position import (
    FourierPositionEncoding, FrequencyPositionEncoding, 
    positions, RotaryPositionEmbedding
)
from VRNN.perceiver.utilities import ModuleOutput, _gn_groups, _finite_stats
from VRNN.perceiver.vector_quantize import VectorQuantize


# ==========================================
# Sampling Utilities (NEW)
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
        self.down = nn.Conv3d(in_ch, out_ch, 4, stride, 1)
        self.res = ResBlock3D(out_ch, out_ch)
    
    def forward(self, x):
        return self.res(self.down(self.pool(x)))


class Up3D(nn.Module):
    """3D version of Up for video processing"""
    def __init__(self, in_ch, skip_ch, out_ch, temporal_upsample=True):
        super().__init__()
        stride = (2, 2, 2) if temporal_upsample else (1, 2, 2)
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 4, stride, 1)
        self.res = ResBlock3D(out_ch + skip_ch, out_ch)
    
    def forward(self, x, skip):
        return self.res(torch.cat([self.up(x), skip], dim=1))


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
    def __init__(
        self,
        in_channels: int = 3,
        code_dim: int = 256,
        num_codes: int = 1024,
        downsample: int = 4,
        base_channels: int = 64,
        commitment_weight: float = 0.25,
        use_cosine_sim: bool = False,
        kmeans_init: bool = True,
        gate_skips: bool = False,
        use_3d_conv: bool = False,  
        temporal_downsample: bool = False,  
        dropout: float = 0.1,
        num_quantizers: int = 4,
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
            kmeans_init=kmeans_init,
            use_cosine_sim=use_cosine_sim,
        )
        self.dropout = nn.Dropout2d(p=dropout)
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
            # Flatten temporal dimension for VQ
            z = rearrange(z, 'b d t h w -> (b t) d h w')
        else:
            # 2D conv: flatten batch and time
            x_flat = rearrange(videos, 'b t c h w -> (b t) c h w')
            z, skips = self.encoder(x_flat)
        
        quantized, ids, vq_loss = self.vq(z)
        
        token_ids = rearrange(ids, '(b t) h w -> b t h w', b=B, t=T)
        quantized = rearrange(quantized, '(b t) d h w -> b t d h w', b=B, t=T)
        
        return token_ids, quantized, vq_loss, skips
    
    def decode(self, quantized: torch.Tensor, skips: list | None = None, use_tanh: bool = True):
        """Decode quantized features to videos (uses real skips when provided)."""
        B, T = quantized.shape[:2]
        
        if skips is None:
            # Fallback: dummy skips for generation
            if self.use_3d_conv:

                B, T, D, H, W = quantized.shape
                skips_to_use = []
                for i, ch in enumerate(self.encoder.skip_channels):
                    scale = 2 ** i
                    t_scale = 2 if self.temporal_downsample and i == 0 else 1
                    skips_to_use.append(torch.zeros(
                        B, ch, T * t_scale, H * scale, W * scale, 
                        device=quantized.device, dtype=quantized.dtype
                    ))
            else:
                _, _, C, H, W = quantized.shape
                q_flat = rearrange(quantized, 'b t d h w -> (b t) d h w')
                skips_to_use = []
                for i, ch in enumerate(self.encoder.skip_channels):
                    scale = 2 ** i
                    skips_to_use.append(torch.zeros(
                        B*T, ch, H*scale, W*scale, 
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
            recon = rearrange(recon, '(b t) c h w -> b t c h w', b=B, t=T)
        
        if use_tanh:
            recon = torch.tanh(recon)
        
        return recon
    
    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Get code vectors from token indices using public VQ API"""
        return self.vq.get_codes_from_indices(indices)


class PerceiverTokenPredictor(nn.Module):
    """
    Perceiver-based video prediction model with discrete token prediction.
    
    This model:
    1. Tokenizes videos using VQ-VAE
    2. Uses Perceiver encoder to process tokenized context frames
    3. Uses decoder to predict future frame tokens
    4. Reconstructs videos from predicted tokens
    """
    
    def __init__(
        self,
        # Tokenizer config
        tokenizer: VQPTTokenizer,
        # Perceiver config
        num_latents: int = 512,
        num_latent_channels: int = 512,
        num_cross_attention_heads: int = 8,
        num_self_attention_layers: int = 6,
        num_self_attention_heads: int = 8,
        widening_factor: int = 4,
        dropout: float = 0.0,
        sequence_length: int = None,
    ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.num_latents = num_latents
        self.num_latent_channels = num_latent_channels
        # Input adapter: Token embeddings -> latent space
        self.token_embedding = nn.Embedding(
            tokenizer.num_codes + 1,  # +1 for mask token
            num_latent_channels
        )
        self.mask_token_id = tokenizer.num_codes
        
        # Learnable latent queries
        self.latent_queries = nn.Parameter(
            torch.randn(num_latents, num_latent_channels) * 0.02
        )
        # Perceiver encoder: Cross-attention + Self-attention
        self.encoder_cross_attn = CrossAttention(
            num_heads=num_cross_attention_heads,
            num_q_input_channels=num_latent_channels,
            num_kv_input_channels=num_latent_channels,
            dropout=dropout,
        )
        
        self.encoder_self_attn = SelfAttentionBlock(
            num_layers=num_self_attention_layers,
            num_heads=num_self_attention_heads,
            num_channels=num_latent_channels,
            widening_factor=widening_factor,
            dropout=dropout,
            causal_attention=False,
        )
 
       #extract time
        self.time_queries = TrainableQueryProvider(
            num_queries=sequence_length,
            num_query_channels=self.num_latent_channels
        )

        d_head = num_latent_channels // num_self_attention_heads
        rotate_dim = d_head - (d_head % 2)
        
        self.time_pe_proj = nn.Linear(
            rotate_dim ,  
            num_latent_channels
        )
        
        # Separate cross-attention for temporal extraction
        self.temporal_cross_attn = CrossAttention(
            num_heads=num_cross_attention_heads,
            num_q_input_channels=num_latent_channels,
            num_kv_input_channels=num_latent_channels,
            dropout=dropout,
        )
        
        # Separate causal self-attention for temporal refinement
        self.temporal_self_attn = SelfAttentionBlock(
            num_layers=1,  # Can be different from encoder
            num_heads=num_self_attention_heads,
            num_channels=num_latent_channels,
            widening_factor=2,  # Can be different
            dropout=dropout,
            causal_attention=True,  
        )
        
        # Position encoding channels (will be created lazily)
        self._pos_shape = None
        self.pos_encoding = None
        self._pos_enc_channels = None
        self.input_proj = None
        self.decoder_query_proj = None
        
       
        self.num_self_attention_heads = num_self_attention_heads
        # Decoder: Predict tokens for future frames
        self.decoder_cross_attn = CrossAttention(
            num_heads=num_cross_attention_heads,
            num_q_input_channels=num_latent_channels,
            num_kv_input_channels=num_latent_channels,
            dropout=dropout,
        )
        
        # Output projection with weight tying
        self.output_proj = nn.Linear(num_latent_channels, tokenizer.num_codes, bias=False)
        # Tie weights with token embedding
        self.output_adapter = TiedTokenOutputAdapter(vocab_size=tokenizer.num_codes, emb_bias=False)
        self.lpips_loss = lpips.LPIPS(net='alex').eval().to(next(self.parameters()).device)
        for param in self.lpips_loss.parameters():
            param.requires_grad = False

    def _ensure_pos_encoding(self, T: int, Ht: int, Wt: int, device: torch.device):
        
        if self._pos_shape != (T, Ht, Wt):
            self._pos_shape = (T, Ht, Wt)
            # 3D Fourier PE over (T, H, W)
            self.pos_encoding = FourierPositionEncoding(
                input_shape=(T, Ht, Wt),
                num_frequency_bands=32
            ).to(device)
            
            # Initialize projection layers if not already done
            if self._pos_enc_channels is None:
                self._pos_enc_channels = self.pos_encoding.num_position_encoding_channels()
                self.input_proj = nn.Linear(
                    self.num_latent_channels + self._pos_enc_channels,
                    self.num_latent_channels
                ).to(device)
                self.decoder_query_proj = nn.Linear(
                    self.num_latent_channels + self._pos_enc_channels,
                    self.num_latent_channels
                ).to(device)
    
    def encoder(self, token_ids: torch.Tensor) -> ModuleOutput:
        """Encode tokenized context frames to latent representation."""
        B, T, Ht, Wt = token_ids.shape
        
        self._ensure_pos_encoding(T, Ht, Wt, token_ids.device)
        
        # Embed tokens
        token_emb = self.token_embedding(rearrange(token_ids, 'b t h w -> b (t h w)'))  # (B, T*H*W, C)
        
        # Add spatiotemporal position encoding
        pos_enc = self.pos_encoding(B)  # (B, T*H*W, pos_dim)
        
        # Concatenate embeddings and positions
        x = self.input_proj(torch.cat([token_emb, pos_enc], dim=-1))  # (B, T*H*W, C)
        
        # Latent queries
        latents = repeat(self.latent_queries, 'n c -> b n c', b=B)
        qk_total = self.num_latent_channels
        num_heads = self.num_self_attention_heads

        d_head = qk_total // num_heads
        # RoPE rotates pairs, so make sure it's even
        rotate_dim = d_head - (d_head % 2)
        # Cross-attention: latents attend to input tokens
        B, N_latent, _ = latents.shape
        abs_pos = positions(B, N_latent, device=latents.device)
        latents = self.encoder_cross_attn(latents, x).last_hidden_state
        freq = FrequencyPositionEncoding(dim=rotate_dim).to(x.device)(abs_pos)  # qk_dim per head*2
        rot = RotaryPositionEmbedding(freq)

        # Self-attention: latents process information
        latents = self.encoder_self_attn(latents, rot_pos_emb=rot).last_hidden_state
        
        return ModuleOutput(last_hidden_state=latents)

    def extract_temporal_bottleneck(
        self, 
        latents: torch.Tensor, 
        T_ctx: int,
    ) -> torch.Tensor:
        """
        Extract time-aligned context from latent bottleneck.
        
        Args:
            latents: [B, num_latents, C] - encoder output
            T_ctx: number of timesteps to extract
            
        Returns:
            temporal_context: [B, T_ctx, C] - time-aligned features
        """
        B, _, C = latents.shape
        device = latents.device
        
        # 1. Get base time queries (learned content)
        q = self.time_queries(B)[:, :T_ctx, :]  # [B, T_ctx, C]
        
        # 2. Add explicit time position encoding
        t_pos = positions(B, T_ctx, device=device)  # [B, T_ctx]
        d_head = C // self.num_self_attention_heads
        rotate_dim = d_head - (d_head % 2)
        freq = FrequencyPositionEncoding(dim=rotate_dim).to(device)(t_pos)
        q = q + self.time_pe_proj(freq)  # [B, T_ctx, C]
        
        # 3. Cross-attention: time queries extract from latents
        q = self.temporal_cross_attn(q, latents).last_hidden_state  # [B, T_ctx, C]
        
        # 4. Causal self-attention: temporal coherence
        rot = RotaryPositionEmbedding(freq)
        q = self.temporal_self_attn(q, rot_pos_emb=rot).last_hidden_state  # [B, T_ctx, C]
        
        return q  
      
    def decoder(
        self,
        latents: torch.Tensor,
        T_override: Optional[int],
        Ht: int,
        Wt: int,
    ) -> torch.Tensor:

        B = latents.shape[0]
        T = T_override if T_override is not None else 1
        
        self._ensure_pos_encoding(T, Ht, Wt, latents.device)
        
        # Position encoding for output queries
        pos_enc = self.pos_encoding(B)  # (B, T*H*W, pos_dim)
        
        temporal_context = self.extract_temporal_bottleneck(latents, T)
        temporal_expanded = repeat(
            temporal_context, 
            'b t d -> b (t h w) d', 
            h=Ht, w=Wt
        )
        # Combine content and position
        decoder_queries = self.decoder_query_proj(torch.cat([temporal_expanded, pos_enc], dim=-1))
        
        # Cross-attention: output queries attend to latents
        output = self.decoder_cross_attn(decoder_queries, latents).last_hidden_state
        
        # Project to logits using tied weights
        
        logits = self.output_adapter(output, self.token_embedding)
        # Reshape to spatial layout
        logits = rearrange(logits, 'b (t h w) v -> b t h w v', t=T, h=Ht, w=Wt)
        
        return logits, temporal_context
    
    def forward(
        self,
        videos: torch.Tensor,
        num_context_frames: int,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:

        B, T_total = videos.shape[:2]
        T_ctx = num_context_frames
        T_pred = T_total - T_ctx
        
        # Tokenize all frames
        token_ids, quantized, vq_loss, skips = self.tokenizer.encode(videos)
        
        assert token_ids.dim() == 4, f"Expected 4D token_ids, got {token_ids.dim()}D"
        
        # Split into context and target
        context_tokens = token_ids[:, :T_ctx]
        target_tokens = token_ids[:, T_ctx:]
        
        # Get actual token grid dimensions
        Ht, Wt = context_tokens.shape[-2:]
        
        # Encode context
        latents = self.encoder(context_tokens)
        
        logits_future, temporal_context = self.decoder(
            latents.last_hidden_state, 
            T_override=T_pred, 
            Ht=Ht, 
            Wt=Wt,
        )
        
        assert logits_future.shape[:4] == target_tokens.shape, \
            f"Logits shape {logits_future.shape[:4]} doesn't match targets {target_tokens.shape}"
        
        # Predictions are for future frames only
        predicted_tokens = logits_future.argmax(dim=-1)
        
        # Reconstruct video from predicted tokens
        if return_dict:
            # Get full sequence for reconstruction (GT context + predicted future)
            B_full, T_full, Ht, Wt = token_ids.shape
            gt_ctx = token_ids[:, :T_ctx]        # ground-truth context tokens
            pred_future = predicted_tokens       # all predictions are for future (no slicing needed)
            full_tokens = torch.cat([gt_ctx, pred_future], dim=1)     # (B, T_total, Ht, Wt)
            flat_tokens = rearrange(full_tokens, 'b t h w -> (b t h w)')
            codes = self.tokenizer.get_codes_from_indices(flat_tokens)
            codes = rearrange(codes, '(b t h w) d -> b t d h w',
                             b=B_full, t=T_full, h=Ht, w=Wt)
            
            # Decode to video using original skips for quality
            reconstructed = self.tokenizer.decode(codes, skips=skips)
        
        outputs = {
            'logits': logits_future,
            'context_tokens': context_tokens,
            'token_ids': target_tokens,
            'predicted_tokens': predicted_tokens,
            'encoder_latents': temporal_context,
            'reconstructed': reconstructed,
            'vq_loss': vq_loss,
            'quantized': quantized,
            'Ht': Ht,
            'Wt': Wt,
        }
        
        return outputs if return_dict else reconstructed
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target_videos: torch.Tensor,
        perceptual_weight: float = 0.5,
        label_smoothing: float = 0.1,
        ce_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses"""
        logits = outputs['logits']
        token_ids = outputs['token_ids']
        vq_loss = outputs['vq_loss']
        reconstructed = outputs['reconstructed']

        assert logits.shape[:4] == token_ids.shape, \
            f"Logits shape {logits.shape[:4]} doesn't match token_ids {token_ids.shape}"
        
        # Flatten for cross entropy
        logits_flat = rearrange(logits, 'b t h w v -> (b t h w) v')
        token_ids_flat = rearrange(token_ids, 'b t h w -> (b t h w)')
        #Diagnostic nan check
        _finite_stats("ce.logits", logits_flat)

        # Cross entropy loss
        ce_loss = F.cross_entropy(
            logits_flat,
            token_ids_flat,
            label_smoothing=label_smoothing,
        )
                
        #perceptual loss (LPIPS)
        B, T_target, C, H, W = target_videos.shape
        context_frames = reconstructed.shape[1]-T_target
        reconstructed_pred = reconstructed[:, context_frames:]
        target_videos_norm = target_videos*2.0 -1.0  # Rescale to [-1, 1] for LPIPS
        recon_flat =rearrange(reconstructed_pred, 'b t c h w -> (b t) c h w')
        target_flat = rearrange(target_videos_norm, 'b t c h w -> (b t) c h w')
        lpips_loss = self.lpips_loss(recon_flat, target_flat).mean()
        # Total loss
        total_loss = ce_weight*ce_loss + vq_loss + perceptual_weight * lpips_loss
        
        # Accuracy
        with torch.no_grad():
            predicted = logits_flat.argmax(dim=-1)
            accuracy = (predicted == token_ids_flat).float().mean()
        
        return {
            'loss': total_loss,
            'ce_loss': ce_loss,
            'vq_loss': vq_loss,
            'accuracy': accuracy,
            'perceptual_loss': lpips_loss,
        }
    
    @torch.no_grad()
    def generate_maskgit(
        self,
        context_videos: torch.Tensor,
        num_frames_to_generate: int,
        num_iterations: int = 12,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """MaskGIT-style iterative generation with enhanced sampling"""
        B = context_videos.shape[0]
        T_ctx = context_videos.shape[1]
        T_gen = num_frames_to_generate
        
        # Encode context 
        context_tokens, _, _, _ = self.tokenizer.encode(context_videos)
        Ht, Wt = context_tokens.shape[-2:]
        
        # Initialize future tokens randomly
        future_tokens = torch.randint(
            0, self.tokenizer.num_codes,
            (B, T_gen, Ht, Wt),
            device=context_videos.device
        )
        
        # Cosine schedule for masking (low→high: start keeping few, increase later)
        def cosine_schedule(step, total_steps):
            # Returns ratio of tokens to keep: starts low (0), ends high (~1)
            return 1.0 - np.cos(step / total_steps * np.pi / 2)
        
        # Track frozen tokens across iterations
        frozen_mask = torch.zeros(B, T_gen, Ht, Wt, dtype=torch.bool, device=context_videos.device)
        
        # Iterative refinement
        for iter_idx in range(num_iterations):
            # Combine context and current predictions
            current_tokens = torch.cat([context_tokens, future_tokens], dim=1)
            
            # Encode
            latents = self.encoder(current_tokens)
            
            logits_future, temporal_context = self.decoder(
                latents.last_hidden_state, 
                T_override=T_gen, 
                Ht=Ht, 
                Wt=Wt,
            )
            
            # Sample with enhanced sampling (temperature, top-k, top-p)
            if temperature > 0:
                # Reshape for sampling
                logits_flat = rearrange(logits_future, 'b t h w v -> (b t h w) v')
                
                # Use unified sampling function
                sampled_flat = sample_tokens(
                    logits_flat,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                
                sampled_tokens = rearrange(
                    sampled_flat, '(b t h w) -> b t h w',
                    b=B, t=T_gen, h=Ht, w=Wt
                )
            else:
                sampled_tokens = logits_future.argmax(dim=-1)
            
            # Get confidence for masking decisions
            probs = F.softmax(logits_future / max(temperature, 0.01), dim=-1)
            confidence = probs.max(dim=-1).values
            
            # Determine which tokens to keep vs. resample
            if iter_idx < num_iterations - 1:
                keep_ratio = cosine_schedule(iter_idx + 1, num_iterations)
                num_to_keep = max(1, int(keep_ratio * T_gen * Ht * Wt))
                
                # Scale noise by confidence std to avoid dominating [0,1] range
                confidence_std = confidence.std() + 1e-8
                confidence_with_noise = confidence + torch.randn_like(confidence) * confidence_std * 0.5
                confidence_flat = rearrange(confidence_with_noise, 'b t h w -> b (t h w)')
                _, keep_indices = torch.topk(confidence_flat, num_to_keep, dim=-1)
                
                keep_mask = torch.zeros_like(confidence_flat, dtype=torch.bool)
                keep_mask.scatter_(1, keep_indices, True)
                keep_mask = rearrange(keep_mask, 'b (t h w) -> b t h w', 
                                     t=T_gen, h=Ht, w=Wt)
                
                # Accumulate frozen mask: once frozen, stays frozen
                frozen_mask = frozen_mask | keep_mask
                
                # Update tokens: keep frozen, resample non-frozen
                future_tokens = torch.where(frozen_mask, future_tokens, sampled_tokens)
            else:
                # Final iteration: use all sampled tokens
                future_tokens = sampled_tokens
        
        # Decode final tokens
        full_tokens = torch.cat([context_tokens, future_tokens], dim=1)
        flat_tokens = rearrange(full_tokens, 'b t h w -> (b t h w)')
        codes = self.tokenizer.get_codes_from_indices(flat_tokens)
        codes = rearrange(codes, '(b t h w) d -> b t d h w',
                         b=B, t=T_ctx + T_gen, h=Ht, w=Wt)
        generated_videos = self.tokenizer.decode(codes)
        
        return generated_videos
    
    @torch.no_grad()
    def generate_autoregressive(
        self,
        context_videos: torch.Tensor,
        num_frames_to_generate: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Standard autoregressive generation with enhanced sampling"""
        B = context_videos.shape[0]
        
        # Encode context 
        context_tokens, _, _, _ = self.tokenizer.encode(context_videos)
        current_tokens = context_tokens
        Ht, Wt = context_tokens.shape[-2:]
        
        # Generate frame by frame
        for _ in range(num_frames_to_generate):
            # Encode current sequence
            latents = self.encoder(current_tokens)
            
            # Decode - predict only next frame with warm-start
            logits_next, temporal_context = self.decoder(
                latents.last_hidden_state, 
                T_override=1, 
                Ht=Ht, 
                Wt=Wt,
            )
            
            # Sample with enhanced sampling
            if temperature > 0:
                # Reshape for sampling
                logits_flat = rearrange(logits_next, 'b 1 h w v -> (b h w) v')
                
                # Use unified sampling function
                next_tokens_flat = sample_tokens(
                    logits_flat,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                
                next_tokens = rearrange(next_tokens_flat, '(b h w) -> b 1 h w',
                                       b=B, h=Ht, w=Wt)
            else:
                next_tokens = logits_next.argmax(dim=-1)
            
            current_tokens = torch.cat([current_tokens, next_tokens], dim=1)
        
        # Decode tokens to video
        flat_tokens = rearrange(current_tokens, 'b t h w -> (b t h w)')
        codes = self.tokenizer.get_codes_from_indices(flat_tokens)
        codes = rearrange(codes, '(b t h w) d -> b t d h w',
                         b=B, t=current_tokens.shape[1], h=Ht, w=Wt)
        generated_videos = self.tokenizer.decode(codes)
        
        return generated_videos
    
    @torch.no_grad()
    def reconstruct(self, videos: torch.Tensor) -> torch.Tensor:
        """Reconstruct videos through tokenization (with skip connections for quality)"""
        token_ids, _, _, skips = self.tokenizer.encode(videos)
        flat_tokens = rearrange(token_ids, 'b t h w -> (b t h w)')
        codes = self.tokenizer.get_codes_from_indices(flat_tokens)
        B, T, Ht, Wt = token_ids.shape
        codes = rearrange(codes, '(b t h w) d -> b t d h w',
                         b=B, t=T, h=Ht, w=Wt)
        # Use skips for better reconstruction quality
        return self.tokenizer.decode(codes, skips=skips)


class CausalPerceiverIO(nn.Module):
    
    def __init__(
        self,
        video_shape: Tuple[int, int, int, int],  # (T, C, H, W)
        num_latents: int = 512,
        num_latent_channels: int = 512,
        num_attention_heads: int = 8,
        num_encoder_layers: int = 6,
        code_dim: int = 256,
        num_codes: int = 1024,
        downsample: int = 4,
        dropout: float = 0.0,
        base_channels: int = 64,
        use_3d_conv: bool = False,
        temporal_downsample: bool = False,
        num_quantizers: int = 1,  # Multi-head VQ
    ):
        super().__init__()
        
        T, C, H, W = video_shape
        
        # Create tokenizer with multi-head VQ support
        self.tokenizer = VQPTTokenizer(
            in_channels=C,
            code_dim=code_dim,
            num_codes=num_codes,
            downsample=downsample,
            base_channels=base_channels,
            commitment_weight=0.25,
            use_cosine_sim=False,
            kmeans_init=False,
            gate_skips=False,
            use_3d_conv=use_3d_conv,
            temporal_downsample=temporal_downsample,
            dropout=dropout,
            num_quantizers=num_quantizers,
        )
        
        # Create main prediction model
        self.model = PerceiverTokenPredictor(
            tokenizer=self.tokenizer,
            num_latents=num_latents,
            num_latent_channels=num_latent_channels,
            num_cross_attention_heads=num_attention_heads,
            num_self_attention_layers=num_encoder_layers,
            num_self_attention_heads=num_attention_heads,
            widening_factor=4,
            dropout=dropout,
            sequence_length=T,
        )
    
    def forward(
        self,
        videos: torch.Tensor,
        num_context_frames: int,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass matching training script API"""
        return self.model.forward(videos, num_context_frames, return_dict)
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target_videos: torch.Tensor,
        perceptual_weight: float = 0.5,
        label_smoothing: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss matching training script API"""
        return self.model.compute_loss(outputs, target_videos, perceptual_weight, label_smoothing)

    @torch.no_grad()
    def generate_maskgit(
        self,
        context_videos: torch.Tensor,
        num_frames_to_generate: int,
        num_iterations: int = 12,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """MaskGIT generation matching training script API"""
        return self.model.generate_maskgit(
            context_videos,
            num_frames_to_generate,
            num_iterations,
            temperature,
            top_k,
            top_p
        )
    
    @torch.no_grad()
    def generate_autoregressive(
        self,
        context_videos: torch.Tensor,
        num_frames_to_generate: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Autoregressive generation matching training script API"""
        return self.model.generate_autoregressive(
            context_videos,
            num_frames_to_generate,
            temperature,
            top_k,
            top_p
        )
    
    @torch.no_grad()
    def reconstruct(self, videos: torch.Tensor) -> torch.Tensor:
        """Reconstruction matching training script API"""
        return self.model.reconstruct(videos)

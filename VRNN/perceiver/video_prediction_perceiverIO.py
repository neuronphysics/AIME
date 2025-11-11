
import torch
import math
from einops.layers.torch import Rearrange
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from typing import Optional, Tuple, Dict, List
import lpips
import numpy as np
from functools import partial
from VRNN.perceiver.adapter import TiedTokenOutputAdapter, TrainableQueryProvider
from VRNN.perceiver.modules import (
    CrossAttention, SelfAttentionBlock, KVCache
)
from VRNN.perceiver.position import (
    FourierPositionEncoding, FrequencyPositionEncoding, 
    positions, RotaryPositionEmbedding
)
from VRNN.perceiver.utilities import ModuleOutput, _gn_groups, Residual
from VRNN.perceiver.vector_quantize import VectorQuantize

from VRNN.perceiver.videovae import VideoEncoder, VideoDecoder, base_group_norm, Normalize, nonlinearity, SamePadConv3d
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


# VQ Tokenizer Components

# ---------- dVAE ----------
class DVAE(nn.Module):
    """
    A dVAE for videos.
    Expects x: [B, C, T, H, W]
    """
    def __init__(
        self,
        vocab_size: int,
        img_channels: int = 3,
        hidden: int = 128,
        kernel_size: int = 3,
        s_down: int = 4,   # spatial down factor (1,2,4,8,...)
        t_down: int = 1,   # kept for API compatibility, but must be 1 with current encoder/decoder
        ratio_vocab_to_latent_channels: int = 32,
        # explicit architecture knobs
        ch_mult: Optional[Tuple[int, ...]] = None,   # e.g. (1, 2, 4)
        num_res_blocks: int = 1,                    # passed to VideoEncoder/Decoder
    ):
        super().__init__()
        # make GroupNorm operate in spatial mode
        base_group_norm.spatial = True

        # We currently only support t_down = 1 with VideoEncoder/VideoDecoder
        assert t_down == 1, "Current DVAE only support t_down = 1"
     
        self.vocab_size = vocab_size
        self.img_channels = img_channels
        self.hidden = hidden
        self.kernel_size = kernel_size

        # 1) Spatial hierarchy (ch_mult)
        if ch_mult is None:
            assert s_down >= 1, "s_down must be >= 1"

            if s_down == 1:
                # no spatial downsampling, single resolution
                ch_mult = (1,)
            else:
                # two scales: top level uses ch, bottom uses ch * s_down
                levels = int(round(math.log2(max(1, s_down))))
                ch_mult = tuple(2 ** i for i in range(levels + 1))
        else:
            ch_mult = tuple(ch_mult)

        self.ch_mult = ch_mult
        self.num_resolutions = len(ch_mult)

        # 2) Latent channels vs vocab
        
        z_channels = vocab_size // ratio_vocab_to_latent_channels
        assert (
            z_channels * ratio_vocab_to_latent_channels == vocab_size
        ), "vocab_size must be divisible by ratio_vocab_to_latent_channels"
        self.z_channels = z_channels

        # ---------- Encoder: [B,C,T,H,W] -> [B,T,z_channels,H',W'] ----------
        self.video_encoder = VideoEncoder(
            ch=hidden,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            in_channels=img_channels,
            z_channels=z_channels,
            double_z=False,
            resamp_with_conv=True,
            kernel_size=kernel_size,
        )

        self.norm_pre_quant  = Normalize(z_channels)
        self.norm_post_quant = Normalize(z_channels)
                                           
        # Quantization: z_channels -> vocab_size
        self.quantize   = SamePadConv3d(z_channels, vocab_size, 1)
        # Dequantization: vocab_size -> z_channels
        self.post_quant = SamePadConv3d(vocab_size, z_channels, 1)

        # ---------- Decoder: [B,T,z_channels,H',W'] -> [B,C,T,H,W] ----------
        self.video_decoder = VideoDecoder(
            ch=hidden,
            z_channels=z_channels,
            out_channels=img_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            resamp_with_conv=True,
        )

    def _encode_logits(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T,H,W]
        assert x.ndim == 5, f"DVAE expects [B,C,T,H,W], got {x.shape}"
        B, C, T, H, W = x.shape

        # Encoder: [B,C,T,H,W] -> [B,T,z_channels,H',W']
        h = self.video_encoder(x, is_init=True)   # [B,T,z_channels,H',W']
        # [B,T,C,H',W'] -> [B,C,T,H',W']
        h = rearrange(h, "b t c h w -> b c t h w")
        h = self.norm_pre_quant(h)
        h = nonlinearity(h)
        # Project to vocab logits: [B,z_channels,T,H',W'] -> [B,vocab_size,T,H',W']
        logits = self.quantize(h)
        return logits

    def _decode_from_onehot(self, z_onehot: torch.Tensor) -> torch.Tensor:
        # z_onehot: [B,vocab_size,T,H',W']
        assert z_onehot.ndim == 5, (
            f"DVAE expects [B,V,T,H,W] codes, got {z_onehot.shape}"
        )

        # Map one-hot / soft codes back to latent space:
        # [B,V,T,H',W'] -> [B,z_channels,T,H',W']
        z_q = self.post_quant(z_onehot)

        z_q = self.norm_post_quant(z_q)
        z_q = nonlinearity(z_q)
        # [B,C,T,H',W'] -> [B,T,C,H',W']
        z = rearrange(z_q, "b c t h w -> b t c h w")
        
        # Decoder: [B,T,z_channels,H',W'] -> [B,C,T,H,W]
        h = self.video_decoder(z, is_init=True)
        return h

    @torch.no_grad()
    def get_z(self, x: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
        """Hard one-hot codes (no grad)."""
        logits = self._encode_logits(x)                            # [B,V,T,H',W']
        z = F.gumbel_softmax(logits, tau=tau, hard=True, dim=1)    # one-hot over vocab
        return z

    def forward(self, x: torch.Tensor, tau: float = 0.5, return_z: bool = False):
        """
        Returns: mse, recon, (optional) hard one-hot z
        """
        logits = self._encode_logits(x)
        z_soft = F.gumbel_softmax(logits, tau=tau, hard=False, dim=1)  # [B,V,T,H',W']
        recon = self._decode_from_onehot(z_soft)

        # MSE averaged over batch and all dims
        mse = ((x - recon) ** 2).mean(dim=(1, 2, 3, 4)).mean()

        if return_z:
            z_hard = F.gumbel_softmax(logits, tau=tau, hard=True, dim=1)
            return mse, recon.clamp(0, 1), z_hard.detach()

        return mse, recon.clamp(0, 1)

class TokenARDecoder(nn.Module):
    """
    AR decoder over token embeddings,
    with causal self-attention + cross-attention to temporal context.
    Supports KV caching for efficient incremental generation.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        temporal_ctx_channels: int,
        max_seq_len: int,
        dropout: float = 0.0,
        widening_factor: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Rotary: how many channels per head to rotate (must be even)
        d_head = d_model // num_heads
        rotate_dim = d_head - (d_head % 2)      # make it even

        self.rotary_dim_per_head = rotate_dim
        self.time_pos_encoder = FrequencyPositionEncoding(rotate_dim)

        # Precompute frequency encodings for the *maximum* sequence length
        # abs_pos: [1, max_seq_len]
        abs_pos = positions(1, max_seq_len, device=torch.device("cpu"))
        frq_pos_enc = self.time_pos_encoder(abs_pos)       # [1, max_seq_len, rotate_dim]

        # Buffer so it gets moved with the module
        self.register_buffer(
            "rotary_frq_pos_enc",
            frq_pos_enc,
            persistent=False,
        )

        # Self-attention stack; now we WILL use rotary (num_rotary_layers > 0)
        self.self_attn = SelfAttentionBlock(
            num_layers=num_layers,
            num_heads=num_heads,
            num_channels=d_model,
            num_qk_channels=None,
            num_v_channels=None,
            num_rotary_layers=-1,          # rotate all layers
            max_heads_parallel=None,
            causal_attention=True,
            widening_factor=widening_factor,
            dropout=dropout,
            residual_dropout=dropout,
            activation_checkpointing=False,
            activation_offloading=False,
        )

        # Cross-attention: tokens → temporal context with residual
        self.cross_attn = Residual(
           CrossAttention(
            num_heads=num_heads,
            num_q_input_channels=d_model,
            num_kv_input_channels=temporal_ctx_channels,
            num_qk_channels=None,
            num_v_channels=None,
            max_heads_parallel=None,
            causal_attention=False,
            dropout=dropout,
        ), dropout=dropout)

        # Output adapter tied to token embedding
        self.output_adapter = TiedTokenOutputAdapter(
            vocab_size=vocab_size,
            emb_bias=False,
        )


    def forward(
        self,
        token_ids: torch.Tensor,         # [B, L_chunk]
        temporal_ctx: torch.Tensor,      # [B, T_ctx_enc, C]
        kv_cache: Optional[List[KVCache]] = None,
    ):
        """
        AR decoding step: causal self-attn over token_ids, then cross-attn to temporal_ctx.
        """
        
        b, L = token_ids.shape
        device = token_ids.device

        # 1) Token embeddings
        x = self.token_embedding(token_ids)    # [B, L, D]

        # 2) Rotary PE over full (cached + current) sequence
        frq_pos_enc = self.rotary_frq_pos_enc.to(device)    # [1, max_seq_len, rotary_dim]

        # Build a RotaryPositionEmbedding object. We keep right_align=True so that when
        # kv_cache grows, Q/K stay aligned from the right.
        rot_emb = RotaryPositionEmbedding(
            frq_pos_enc=frq_pos_enc,
            right_align=True,
        )

        # 3) Causal self-attention with KV cache + rotary
        sa_out = self.self_attn(
            x,
            pad_mask=None,
            rot_pos_emb=rot_emb,
            kv_cache=kv_cache,
        )
        x = sa_out.last_hidden_state      # [B, L, D]
        kv_next = sa_out.kv_cache         # list[KVCache] or None

        # 4) Cross-attention to temporal context
        ca_out = self.cross_attn(
            x,
            x_kv=temporal_ctx,
            pad_mask=None,
            rot_pos_emb_q=None,
            rot_pos_emb_k=None,
            kv_cache=None,
        )
        x = ca_out.last_hidden_state      # [B, L, D]

        # 5) Project to logits
        logits = self.output_adapter(x, self.token_embedding)  # [B, L, vocab_size]

        return ModuleOutput(
            last_hidden_state=logits,
            kv_cache=kv_next,
        )



class VQPTTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        code_dim: int = 256,
        num_codes: int = 1024,
        downsample: int = 4,
        base_channels: int = 64,
        commitment_weight: float = 0.25,
        commitment_use_cross_entropy_loss: bool = False,
        use_cosine_sim: bool = False,
        kmeans_init: bool = True,
        dropout: float = 0.1,
        num_quantizers: int = 4,
        orthogonal_reg_weight: float = 0.0,
        orthogonal_reg_active_codes_only: bool = True,
        orthogonal_reg_max_codes: int = None,
        codebook_diversity_loss_weight: float = 0.0,
        codebook_diversity_temperature: float = 1.0,
        threshold_ema_dead_code: int = 2,

    ):
        super().__init__()
        self.in_channels = in_channels
        self.code_dim = code_dim
        self.num_codes = num_codes
        self.downsample = downsample

        self.dvae = DVAE(
                vocab_size=code_dim,
                img_channels=in_channels,
                hidden=base_channels,
                kernel_size=3,
                s_down=downsample,   # still kept for documentation / safety
                t_down=1,
                num_res_blocks=1,
            )

        
        self.vq = VectorQuantize(
            dim=code_dim,
            codebook_size=num_codes,
            heads=num_quantizers,
            codebook_dim=code_dim // num_quantizers,
            accept_image_fmap=True,
            channel_last=False,
            threshold_ema_dead_code=threshold_ema_dead_code,
            commitment_weight=commitment_weight,
            commitment_use_cross_entropy_loss=commitment_use_cross_entropy_loss,
            kmeans_init=kmeans_init,
            use_cosine_sim=use_cosine_sim,
            orthogonal_reg_weight=orthogonal_reg_weight,
            orthogonal_reg_active_codes_only=orthogonal_reg_active_codes_only,
            orthogonal_reg_max_codes=orthogonal_reg_max_codes,
            codebook_diversity_loss_weight=codebook_diversity_loss_weight,
            codebook_diversity_temperature=codebook_diversity_temperature,
            straight_through=True, 
            rotation_trick=False
        )
        self.dropout = nn.Dropout3d(p=dropout) 


    def _get_codebook_outputs(self) -> torch.Tensor:
        """
        Return a (num_codes, code_dim) matrix of decoder-side code vectors.

        Each row i is the decoder-side embedding for code index i.
        """
        device = next(self.parameters()).device

        # Indices of all codes in the codebook: 0 .. num_codes-1
        all_indices = torch.arange(self.num_codes, device=device)

        # a (num_codes, code_dim) tensor when given a 1D index tensor.
        codebook_outputs = self.vq.get_output_from_indices(all_indices)

        if codebook_outputs.shape[0] != self.num_codes:
            raise RuntimeError(
                f"Codebook output first dim {codebook_outputs.shape[0]} "
                f"!= num_codes {self.num_codes}"
            )
        if codebook_outputs.shape[1] != self.code_dim:
            raise RuntimeError(
                f"Codebook output second dim {codebook_outputs.shape[1]} "
                f"!= code_dim {self.code_dim}"
            )

        return codebook_outputs
    @torch.no_grad()
    def token_self_consistency(
        self,
        videos: torch.Tensor,
        use_tanh: bool = True,
    ) -> torch.Tensor:
        """
        Compute fraction of matching tokens between input videos and
        their reconstructions: encode(x) vs encode(decode(quantized(x))).

        Returns a scalar tensor in [0, 1].
        """
        # 1) Encode original
        tok1, quantized, _ = self.encode(videos)   # tok1: [B,T,Ht,Wt(,Q)]

        # 2) Decode from quantized latents
        recon = self.decode(quantized, use_tanh=use_tanh)  # [B,T,C,H,W]

        # 3) Re-encode reconstruction
        tok2, _, _ = self.encode(recon)

        # 4) Drop Q dim if present (multi-head)
        if tok1.dim() == 5:
            t1 = tok1[..., 0]
        else:
            t1 = tok1

        if tok2.dim() == 5:
            t2 = tok2[..., 0]
        else:
            t2 = tok2

        # 5) Flatten and compute match rate
        match = (t1 == t2).float().mean()
        return match

    
    def get_codebook_perplexity(self) -> torch.Tensor:
        # cluster_size: [num_heads, K]
        cluster_size = self.vq._codebook.cluster_size  # buffer
        probs = cluster_size / (cluster_size.sum(dim=-1, keepdim=True) + torch.finfo(cluster_size.dtype).eps)
        # entropy per head
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
        return entropy.exp()  # [num_heads]

    def encode(self, videos: torch.Tensor):
        """Encode videos to tokens and quantized features ."""
        B, T = videos.shape[:2]

        # 1) Channels-first for encoder
        x = rearrange(videos, 'b t c h w -> b c t h w')  # [B,C,T,H,W]

        # 2) Encoder -> latent z_channels in time-major layout
        #    [B,C,T,H,W] -> [B,T,z_channels,H',W'] -> [B,z_channels,T,H',W']
        latent_tm = self.dvae.video_encoder(x, is_init=True)               # [B,T,zc,H',W']
        latent_cf = rearrange(latent_tm, 'b t c h w -> b c t h w')         # [B,zc,T,H',W']

        # 3) 1x1x1 conv to code_dim (continuous logits)
        logits = self.dvae.quantize(latent_cf)                             # [B,code_dim,T,H',W']
        T_enc = logits.shape[2]

        # 4) VQ over (B*T, code_dim, H', W')
        z2d = rearrange(logits, 'b d t h w -> (b t) d h w').contiguous()
        quant2d, ids, vq_loss = self.vq(z2d)                               # quant2d: [B*T,code_dim,Ht,Wt]

        # 5) Back to time-major
        quantized = rearrange(quant2d, '(b t) d h w -> b t d h w', b=B, t=T_enc)

        if ids.ndim == 3:
            token_ids = rearrange(ids, '(b t) h w -> b t h w', b=B, t=T_enc)
        else:
            token_ids = rearrange(ids, '(b t) h w q -> b t h w q', b=B, t=T_enc)

        return token_ids.long(), quantized, vq_loss
    
    def decode(self, quantized: torch.Tensor,  use_tanh: bool = True):
        """Decode quantized features to videos ."""
        B, T_enc = quantized.shape[:2]
        

        q = rearrange(quantized, 'b t d h w -> b d t h w')
        recon = self.dvae._decode_from_onehot(q)
        recon = rearrange(recon, 'b c t h w -> b t c h w')

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
        idx = self._sanitize_token_ids(indices)
        # Build decoder-side code matrix: (num_codes, code_dim)
        codebook_outputs = self._get_codebook_outputs().to(idx.device)

        # If multi-head indices are provided (B,N,Q), use the first head by default
        if idx.ndim == 3:
            idx = idx[..., 0]

        # F.embedding -> [B, N, code_dim]
        codes_bn = F.embedding(idx, codebook_outputs)
        # -> [B, code_dim, N]
        return rearrange(codes_bn, 'b n d -> b d n')

    @torch.no_grad()
    def decode_code(
        self,
        token_ids: torch.Tensor,   # [B, T_enc, Ht, Wt] (or [B,T_enc,Ht,Wt,Q])
        T_enc: int,
        H_t: int,
        W_t: int,
        use_tanh: bool = True,
    ):
        """
          1) indices -> decoder-side vectors via F.embedding
          2) reshape to [B, code_dim, T_enc, H_t, W_t]
          3) post_quant + decoder
        """
        B = token_ids.shape[0]
        if token_ids.ndim == 5:
            token_ids = token_ids[..., 0]  # first head by default

        # Flatten spatial/time -> [B, N]
        flat_ids = token_ids.view(B, -1)

        # (num_codes, code_dim)
        codebook_outputs = self._get_codebook_outputs().to(flat_ids.device)
        # [B, N, code_dim]
        quant_seq = F.embedding(flat_ids, codebook_outputs)
        # [B, code_dim, T_enc, H_t, W_t]
        q_cf = rearrange(quant_seq, 'b (t h w) d -> b d t h w', t=T_enc, h=H_t, w=W_t)

        # Decode to pixels
        recon_cf_time = self.dvae._decode_from_onehot(q_cf)          # [B,C,T,H,W]
        recon = rearrange(recon_cf_time, 'b c t h w -> b t c h w')
        if use_tanh:
            recon = recon.tanh()
        return recon

class PerceiverTokenPredictor(nn.Module):
    """
    Perceiver-based video prediction model with discrete token prediction.

    The model works as follows:
      1. VQ tokenizer (VQPTTokenizer) turns videos -> discrete tokens + quantized latents.
      2. Perceiver encoder processes context tokens into latent features.
      3. Temporal bottleneck produces a per-future-timestep context.
      4. TokenARDecoder autoregressively predicts the flattened future token sequence.
      5. We reconstruct videos from (context tokens + predicted future tokens).
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
        # AR decoder config
        num_token_ar_decoder_layers: int = 4,
        num_token_ar_decoder_heads: int = 8,
        max_seq_len: int = 4096,
        ARdecoder_widening_factor: int = 2,
    ):
        super().__init__()

        # Tokenizer / vocab
        self.tokenizer = tokenizer
        self.code_vocab_size = tokenizer.num_codes     # codebook vocab (0..num_codes-1)
        # AR vocab is **codebook + 1 extra BOS/MASK slot**
        self.ar_vocab_size = self.code_vocab_size + 1
        self.bos_token_id = self.code_vocab_size       # used only in AR decoder
        self.mask_token_id = self.code_vocab_size      # used only for MaskGIT head (never sent to VQ) :contentReference[oaicite:0]{index=0}

        self.num_latents = num_latents
        self.num_latent_channels = num_latent_channels

        # Token embedding for Perceiver encoder
        self.token_embedding = nn.Embedding(
            self.code_vocab_size,     # only real VQ tokens
            num_latent_channels,
        )

        # Perceiver encoder core
        # Learnable latent queries
        self.latent_queries = nn.Parameter(
            torch.randn(num_latents, num_latent_channels) * 0.02
        )

        # Cross-attn: latents <- token embeddings with residual
        self.encoder_cross_attn = Residual( 
            CrossAttention(
            num_heads=num_cross_attention_heads,
            num_q_input_channels=num_latent_channels,
            num_kv_input_channels=num_latent_channels,
            dropout=dropout,
        ), dropout=dropout)

        # Self-attn over latents (non-causal, bidirectional)
        self.encoder_self_attn = SelfAttentionBlock(
            num_layers=num_self_attention_layers,
            num_heads=num_self_attention_heads,
            num_channels=num_latent_channels,
            widening_factor=widening_factor,
            dropout=dropout,
            causal_attention=False,
        )

        # Temporal bottleneck
        assert sequence_length is not None, "sequence_length must be set for temporal bottleneck"

        # Queries over time (T_max, C)
        self.time_queries = TrainableQueryProvider(
            num_queries=sequence_length,
            num_query_channels=self.num_latent_channels,
        )

        d_head = num_latent_channels // num_self_attention_heads
        rotate_dim = d_head - (d_head % 2)

        self.time_pe_proj = nn.Linear(
            rotate_dim,
            num_latent_channels,
        )

        # Cross-attn: time queries <- latents
        self.temporal_cross_attn = Residual(
            CrossAttention(
                num_heads=num_cross_attention_heads,
                num_q_input_channels=num_latent_channels,
                num_kv_input_channels=num_latent_channels,
                dropout=dropout,
            ), dropout=dropout
        )

        # Causal self-attn along time axis (temporal AR bottleneck)
        self.temporal_self_attn = SelfAttentionBlock(
            num_layers=1,
            num_heads=num_self_attention_heads,
            num_channels=num_latent_channels,
            widening_factor=2,
            dropout=dropout,
            causal_attention=True,
        )

        # Position encodings
        self.pos_freq_bands = 64
        self._pos_shape = None
        self.pos_encoding = None
        self._pos_enc_channels = 3 * (2 * self.pos_freq_bands + 1)

        # Linear projections from positional encodings -> latent channels
        self.input_proj = nn.Linear(self._pos_enc_channels, num_latent_channels)
        self.decoder_query_proj = nn.Linear(self._pos_enc_channels, num_latent_channels)
        self.num_self_attention_heads = num_self_attention_heads

        # 1D time positional encoder for the temporal bottleneck (over timesteps)
        self.time_tb_pos_encoder = FrequencyPositionEncoding(self.time_pe_proj.in_features)


        self.output_adapter = TiedTokenOutputAdapter(
            vocab_size=self.code_vocab_size,
            emb_bias=False,
        )

        # Perceptual loss
        self.lpips_loss = lpips.LPIPS(net='alex')
        self.lpips_loss.eval()
        for param in self.lpips_loss.parameters():
            param.requires_grad = False

        # AR decoder
        self.ARdecoder = TokenARDecoder(
            vocab_size=self.ar_vocab_size,            # <-- BOS+real tokens
            d_model=num_latent_channels,
            num_layers=num_token_ar_decoder_layers,
            num_heads=num_token_ar_decoder_heads,
            temporal_ctx_channels=num_latent_channels,
            max_seq_len=max_seq_len,
            dropout=dropout,
            widening_factor=ARdecoder_widening_factor,
        )

        # --- VQ freezing config ---
        self.vq_freeze_threshold = 0.9993       # freeze when >= this
        self.vq_unfreeze_threshold = 0.995     # unfreeze when <= this
        self.vq_freeze_patience = 5
        self.vq_unfreeze_patience = 5
        self._vq_frozen = False
        self._vq_freeze_counter = 0
        self._vq_unfreeze_counter = 0

        # DVAE: only freeze once very stable
        self.dvae_freeze_threshold = 0.9999     # your "99%" condition
        self.dvae_freeze_patience = 5
        self._dvae_frozen = False
        self._dvae_freeze_counter = 0

    # Token flatten / unflatten helpers (support Q>1 token heads)
    def _flatten_tokens(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        token_ids: [B, T, Ht, Wt] or [B, T, Ht, Wt, Q]
        returns:
            seq: [B, L]
            meta: shapes (for unflatten)
        """
        B, T = token_ids.shape[:2]

        if token_ids.dim() == 4:
            Ht, Wt = token_ids.shape[2:]
            Q = 1
            seq = rearrange(token_ids, "b t h w -> b (t h w)")
        elif token_ids.dim() == 5:
            Ht, Wt, Q = token_ids.shape[2:]
            seq = rearrange(token_ids, "b t h w q -> b (t h w q)")
        else:
            raise ValueError(f"Unexpected token_ids shape: {token_ids.shape}")

        tokens_per_frame = Ht * Wt * Q
        meta = dict(
            T=T,
            Ht=Ht,
            Wt=Wt,
            Q=Q,
            tokens_per_frame=tokens_per_frame,
        )
        return seq.long(), meta

    def _unflatten_tokens(self, seq: torch.Tensor, meta: Dict) -> torch.Tensor:
        """
        seq: [B, L]
        returns token_ids: [B, T, Ht, Wt] or [B, T, Ht, Wt, Q]
        """
        B, L = seq.shape
        tpf = meta["tokens_per_frame"]
        Ht, Wt, Q = meta["Ht"], meta["Wt"], meta["Q"]
        T = L // tpf

        if Q == 1:
            token_ids = rearrange(
                seq[:, : T * tpf],
                "b (t h w) -> b t h w",
                t=T, h=Ht, w=Wt,
            )
        else:
            token_ids = rearrange(
                seq[:, : T * tpf],
                "b (t h w q) -> b t h w q",
                t=T, h=Ht, w=Wt, q=Q,
            )
        return token_ids

    # Context/target split in encoded time
    def _split_context_target_tokens(
        self,
        token_ids: torch.Tensor,
        num_context_frames: int,
        T_raw: int,
    ):
        """
        Map raw-frame context length -> encoded timesteps and split tokens.

        token_ids: [B, T_enc, Ht, Wt] (or [B, T_enc, Ht, Wt, Q])
        num_context_frames: context frames in *raw* video space
        T_raw: total raw frames in the input video

        Returns:
            context_tokens: [B, T_ctx_enc, ...]
            target_tokens:  [B, T_pred_enc, ...]
            T_ctx_enc: int
            T_pred_enc: int
        """
        B, T_enc = token_ids.shape[:2]
        if T_enc <= 1:
            raise ValueError(
                f"T_enc={T_enc}. Need at least 2 encoded timesteps for "
                f"context+prediction. Check sequence_length/data."
            )

        # If no temporal downsampling in VQ encoder
        if T_enc == T_raw:
            T_ctx_enc = num_context_frames
        else:
            ratio = T_enc / float(T_raw)
            T_ctx_enc = int(round(num_context_frames * ratio))

        # Clamp to a valid split
        T_ctx_enc = max(1, min(T_ctx_enc, T_enc - 1))
        T_pred_enc = T_enc - T_ctx_enc
        assert T_pred_enc > 0, f"Unexpected T_pred_enc={T_pred_enc} (T_enc={T_enc}, T_ctx_enc={T_ctx_enc})"
        context_tokens = token_ids[:, :T_ctx_enc]
        target_tokens = token_ids[:, T_ctx_enc:]

        return context_tokens, target_tokens, T_ctx_enc, T_pred_enc

    # VQ tokenizer freezing (used by compute_loss / cycle loss)
    def freeze_VQPT_tokenizer(self):
        if hasattr(self.tokenizer, 'vq'):
            for p in self.tokenizer.vq.parameters():
                p.requires_grad = False
            self.tokenizer.vq.freeze_codebook = True
        self._vq_frozen = True

    def unfreeze_VQPT_tokenizer(self):
        if hasattr(self.tokenizer, 'vq'):
            for p in self.tokenizer.vq.parameters():
                p.requires_grad = True
            self.tokenizer.vq.freeze_codebook = False
        self._vq_frozen = False

    # Position encoding helpers
    def _ensure_pos_encoding(
        self,
        T: int,
        Ht: int,
        Wt: int,
        device: torch.device,
    ):
        """
        Lazily build a Fourier-based spatio-temporal positional encoding
        with shape [1, T*Ht*Wt, C_pos], then keep it on the correct device.
        """
        shape = (T, Ht, Wt)
        if self._pos_shape == shape and self.pos_encoding is not None:
            # Already computed for this (T, Ht, Wt) configuration
            self.pos_encoding = self.pos_encoding.to(device)
            return

        self._pos_shape = shape

        # Use the shared FourierPositionEncoding helper over a 3D grid (t, h, w).
        fpe = FourierPositionEncoding((T, Ht, Wt), self.pos_freq_bands)
        pe = fpe(b=1)                     # [1, T*Ht*Wt, C_pos]
        pe = pe.to(device)

        self.pos_encoding = pe
        self._pos_enc_channels = pe.shape[-1]


    def _get_spatiotemporal_encoding(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape[:2]
        Ht, Wt = token_ids.shape[2:4]
        device = token_ids.device

        self._ensure_pos_encoding(T, Ht, Wt, device)
        pe = self.pos_encoding.to(device)       # [1, T*Ht*Wt, C_pos]
        pe = self.input_proj(pe)                # [1, T*Ht*Wt, C]
        pe = pe.expand(B, -1, -1)               # [B, T*Ht*Wt, C]
        return pe

    # Perceiver encoder + temporal bottleneck
    def encoder(self, token_ids: torch.Tensor) -> ModuleOutput:
        """
        token_ids: [B, T_enc, Ht, Wt] or [B, T_enc, Ht, Wt, Q==1]
        Returns latents: [B, num_latents, C]
        """
        B, T_enc = token_ids.shape[:2]
        Ht, Wt = token_ids.shape[2:4]
        device = token_ids.device

        # drop Q dim if present (we only embed one index per site).
        if token_ids.dim() == 5:
            token_ids = token_ids[..., 0]

        tok = self.token_embedding(token_ids.long())          # [B,T,Ht,Wt,C]
        tok = rearrange(tok, "b t h w c -> b (t h w) c")      # [B, N, C]

        pe = self._get_spatiotemporal_encoding(token_ids)     # [B, N, C]
        enc_in = tok + pe                                     # [B, N, C]

        # Latent queries
        latents = self.latent_queries.expand(B, -1, -1)       # [B, L, C]

        # Latents attend to tokens
        ca_out = self.encoder_cross_attn(
            latents,
            x_kv=enc_in,
            pad_mask=None,
            rot_pos_emb_q=None,
            rot_pos_emb_k=None,
            kv_cache=None,
        )
        latents = ca_out.last_hidden_state                    # [B,L,C]

        sa_out = self.encoder_self_attn(
            latents,
            pad_mask=None,
            rot_pos_emb=None,
            kv_cache=None,
        )
        latents = sa_out.last_hidden_state                    # [B,L,C]

        return ModuleOutput(last_hidden_state=latents)

    def extract_temporal_bottleneck(
        self,
        latents: torch.Tensor,       # [B, L, C]
        T_to_extract: int,
        T_start_index: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Build a temporal sequence of length T_to_extract from Perceiver latents.

        We index time with learnable queries + Fourier PE in time,
        then causal self-attn along that axis.
        Returns temporal_ctx: [B, T_to_extract, C].
        """
        B, L, C = latents.shape
        device = latents.device

        # Time positions for [T_start_index, T_start_index+T_to_extract)
        t_idx = torch.arange(
            T_start_index,
            T_start_index + T_to_extract,
            device=device,
            dtype=torch.long,
        )  # [T_to_extract]

        # 1D time positional encoding -> [T_to_extract, D_rot]
        abs_pos = t_idx.unsqueeze(0)                                # [1, T]
        t_fpe = self.time_tb_pos_encoder(abs_pos)                   # [1, T, D_rot]
        t_fpe = t_fpe.squeeze(0)                                    # [T, D_rot]

        # Project to latent dimension and add queries
        t_pe = self.time_pe_proj(t_fpe)                             # [T, C]
        time_queries = self.time_queries()  # [1,T,C] or [B,T,C]
        assert T_to_extract <= time_queries.shape[1], f"T_to_extract={T_to_extract} > max={time_queries.shape[1]}"
        # Slice to the required number of timesteps
        time_queries = time_queries[:, :T_to_extract, :]            # [1, T, C]

        # Expand across batch
        time_queries = time_queries.expand(B, -1, -1)               # [B, T, C]

        time_q = time_queries + t_pe.unsqueeze(0)                   # [B,T,C]

        # Cross-attn: time_q <- latents
        ca_out = self.temporal_cross_attn(
            time_q,
            x_kv=latents,
            pad_mask=None,
            rot_pos_emb_q=None,
            rot_pos_emb_k=None,
            kv_cache=None,
        )
        t_feats = ca_out.last_hidden_state                            # [B,T,C]

        # Causal self-attn along time axis
        sa_out = self.temporal_self_attn(
            t_feats,
            pad_mask=None,
            rot_pos_emb=None,
            kv_cache=None,
        )
        t_feats = sa_out.last_hidden_state                            # [B,T,C]

        meta = dict(T_to_extract=T_to_extract, T_start_index=T_start_index)
        return t_feats, meta


    # The AR training over future tokens
    def forward(
        self,
        videos: torch.Tensor,        # [B, T_raw, C, H, W]
        num_context_frames: int,
        return_dict: bool = True,
    ):
        B, T_raw, C, H, W = videos.shape
        device = videos.device

        # 1) VQ tokenize full clip
        token_ids, quantized, vq_loss = self.tokenizer.encode(videos)
        # token_ids: [B, T_enc, Ht, Wt] or [B,T_enc,Ht,Wt,Q]
        token_ids = token_ids.long()

        # 2) Split into context / target (in encoded time)
        context_tokens, target_tokens, T_ctx_enc, T_pred_enc = self._split_context_target_tokens(
            token_ids, num_context_frames, T_raw
        )

        # 3) Encode **context tokens** with Perceiver
        enc_out = self.encoder(context_tokens)
        latents = enc_out.last_hidden_state          # [B, L, C]

        # 4) Temporal bottleneck for the future interval
        temporal_ctx, _ = self.extract_temporal_bottleneck(
            latents,
            T_to_extract=T_pred_enc,
            T_start_index=T_ctx_enc,
        )                                            # [B,T_pred_enc,C]

        # 5) Prepare AR targets: flatten future tokens (L = T_pred_enc * tokens_per_frame)
        target_flat, meta_future = self._flatten_tokens(target_tokens)  # [B,L]
        L_future = target_flat.shape[1]
        Ht, Wt, Q = meta_future["Ht"], meta_future["Wt"], meta_future["Q"]

        # 6) AR training:
        #    input_ids = [BOS, y_0, ..., y_{L-2}], targets = [y_0, ..., y_{L-1}]
        bos = torch.full(
            (B, 1),
            self.bos_token_id,
            device=device,
            dtype=torch.long,
        )
        ar_input = torch.cat([bos, target_flat[:, :-1]], dim=1)   # [B,L]
        # temporal_ctx is per-encoded-timestep; we can simply tile / interpolate if needed.
        # Here we just repeat each per-time embedding across its spatial tokens.
        T_enc = T_pred_enc
        tpf = meta_future["tokens_per_frame"]
        assert L_future == T_enc * tpf, "Flattened future tokens must align with temporal_ctx * tokens_per_frame"

        temporal_ctx_expanded = repeat(
            temporal_ctx,          # [B, T_enc, C]
            "b t c -> b (t r) c",
            r=tpf,
        )                         # [B, L_future, C]

        ar_out = self.ARdecoder(
            token_ids=ar_input,
            temporal_ctx=temporal_ctx_expanded,
            kv_cache=None,
        )
        logits_ar = ar_out.last_hidden_state           # [B,L,ar_vocab_size]

        # Drop BOS channel if needed, but targets are in [0, code_vocab_size-1]
        logits_ar = logits_ar[..., : self.code_vocab_size]

        # 7) Reshape AR logits back to [B,T_pred_enc,Ht,Wt,vocab]
        logits_future = rearrange(
            logits_ar,
            "b (t h w) v -> b t h w v",
            t=T_pred_enc,
            h=Ht,
            w=Wt,
        )

        # 8) Greedy prediction for reconstruction
        pred_flat = logits_ar.argmax(dim=-1)      # [B,L]
        pred_tokens = self._unflatten_tokens(pred_flat, meta_future)  # [B,T_pred_enc,Ht,Wt(,Q)]

        # 9) Combine context + predicted future tokens and decode to pixels
        all_tokens = torch.cat([context_tokens, pred_tokens], dim=1)  # [B,T_enc,Ht,Wt(,Q)]
        all_flat, meta_all = self._flatten_tokens(all_tokens)         # [B, L_all]
        codes_flat = self.tokenizer.get_codes_from_indices(all_flat)  # [B, L_all, D] channel_last=False -> [B,D,...]

        # reshape codes back to [B,T_enc,D,Ht,Wt]
        T_enc_all = meta_all["T"]
        Ht_all, Wt_all = meta_all["Ht"], meta_all["Wt"]
        #codes_flat.shape = [B, 256, 2048]
        codes = rearrange(
            codes_flat,
            "b d (t h w) -> b t d h w",
            t=T_enc_all,
            h=Ht_all,
            w=Wt_all,
        )

        reconstructed = self.tokenizer.decode(codes)   # [B, T_dec, C, H, W]
        # 10) Token accuracy on the future region
        with torch.no_grad():
            # match shapes (pred_tokens vs target_tokens)
            if pred_tokens.shape != target_tokens.shape:
                # If Q mismatch, take first head for accuracy
                pt = pred_tokens[..., 0] if pred_tokens.dim() == 5 else pred_tokens
                tt = target_tokens[..., 0] if target_tokens.dim() == 5 else target_tokens
            else:
                pt, tt = pred_tokens, target_tokens

            token_accuracy = (pt == tt).float().mean()

        outputs = {
            "logits": logits_future,         # [B,T_pred_enc,Ht,Wt,vocab]
            "token_ids": target_tokens,      # ground-truth future tokens (for CE)
            "quantized": quantized,          # [B,T_enc,D,Ht,Wt]
            "vq_loss": vq_loss,
            "reconstructed": reconstructed,  # [B,T_dec,C,H,W]
            "token_accuracy": token_accuracy,
        }

        if return_dict:
            return outputs
        return logits_future, outputs

    # Loss ( uses AR CE + recon + perceptual + optional cycle)
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target_videos: torch.Tensor,
        perceptual_weight: float = 0.5,
        recon_weight: float = 1.0,
        label_smoothing: float = 0.1,
        ce_weight: float = 1.0,
        ar_cycle_consistency_weight: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Full loss with AR latent cycle-consistency.

        Loss components:
          - vq_loss (from tokenizer)
          - CE over future tokens (AR head)
          - reconstruction loss (L1 + LPIPS on future frames)
          - optional AR cycle consistency in latent space
        """
        logits = outputs["logits"]             # [B,T_pred,Ht,Wt,V]
        token_ids = outputs["token_ids"]       # [B,T_pred,Ht,Wt(,Q)]
        quantized = outputs["quantized"]       # [B,T_enc,D,Ht,Wt] (context + target)
        vq_loss = outputs["vq_loss"]
        
        B, T_target = target_videos.shape[:2]

        device = logits.device

        # --------- CE over future tokens ---------
        if token_ids.dim() == 5:
            # if multi-head, use first head as target
            tgt = token_ids[..., 0]
        else:
            tgt = token_ids
        Bp, T_pred, Ht, Wt = tgt.shape
        assert Bp == B, "Batch mismatch between logits/targets and videos"

        logits_flat  = rearrange(logits, "b t h w v -> (b t h w) v")
        targets_flat = rearrange(tgt,   "b t h w   -> (b t h w)")

        ce_loss = F.cross_entropy(
            logits_flat,
            targets_flat.long(),
            label_smoothing=label_smoothing,
        )

        ce = ce_loss * ce_weight

        Bq, T_enc, D, Hq, Wq = quantized.shape
        assert Bq == B and Hq == Ht and Wq == Wt, "Quantized shape mismatch"
        # (optional but more logically correct)
        # assert T_enc >= T_pred, "Encoded sequence shorter than future tokens"

        # --------- Reconstruction loss on future frames ---------
        # Decode the last *encoded* future tokens, not raw frames
        recon_future = self.tokenizer.decode(
            quantized,  
            use_tanh=False,
        )  # [B, T_dec, C, H, W]
        # Flatten time into batch for LPIPS
        target_flat = rearrange(target_videos, "b t c h w -> (b t) c h w")
        recon_flat  = rearrange(recon_future,  "b t c h w -> (b t) c h w")

        # LPIPS expects inputs in [0, 1]
        target_flat = (target_flat + 1.0) / 2.0
        recon_flat  = (recon_flat  + 1.0) / 2.0

        lpips_loss = self.lpips_loss(recon_flat, target_flat).mean()
        recon_loss = F.l1_loss(recon_flat, target_flat)
        # --------- AR cycle consistency in latent space (full version) ---------
        #   For a future timestep τ:
        #     z_GT      = quantized GT latent for frame τ
        #     z_AR      = expected latent from AR logits at τ
        #     z_cycle   = encode(decode(z_AR))  through frozen VQ
        #     L_cycle   = || z_GT - z_cycle ||^2

        cycle_consistence_loss = torch.tensor(0.0, device=device)
        if ar_cycle_consistency_weight > 0.0:
            self.freeze_VQPT_tokenizer()

            Bq, T_enc2, D, Ht_q, Wt_q = quantized.shape
            assert Bq == B and Ht_q == Ht and Wt_q == Wt, "Quantized shape mismatch"

            # context + target tokens in encoder space
            T_tokens = T_enc2
            T_context_tokens = T_tokens - T_pred
            assert T_context_tokens > 0, "No context tokens for AR cycle consistency"

            # choose a random future step index in [0, T_pred)
            t_idx = torch.randint(
                low=0, high=T_pred, size=(1,), device=device
            ).item()

            # ground-truth latent at that step: (B, D, Ht, Wt)
            z_gt = quantized[:, T_context_tokens + t_idx]
            z_gt = z_gt.detach()   # do not backprop into VQ encoder

            # get "soft" AR latent from logits via codebook expectation
            logits_step = logits[:, t_idx]          # (B, Ht, Wt, V)
            probs_step  = F.softmax(logits_step, dim=-1)
            probs_flat  = rearrange(probs_step, "b h w v -> (b h w) v")

            # codebook: (V, D)
            codebook = self.tokenizer.vq.codebook.to(device)

            # expected embedding per spatial location
            z_pred_flat = probs_flat @ codebook      # (B*H*W, D)
            # reshape to (B, D, Ht, Wt)
            z_pred = rearrange(
                z_pred_flat,
                "(b h w) d -> b d h w",
                b=B,
                h=Ht,
                w=Wt,
            )

            z_pred_batch = z_pred.unsqueeze(1)       # (B, 1, D, Ht, Wt)
            decoded_step = self.tokenizer.decode(
                z_pred_batch,
                use_tanh=True,
            )                                        # (B, 1, C, H, W)

            # re-encode through frozen VQ tokenizer
            _, quant_cycle, _ = self.tokenizer.encode(decoded_step)
            # quant_cycle: (B, 1, D, Ht, Wt)
            z_cycle = quant_cycle[:, 0]              # (B, D, Ht, Wt)

            # cosine cycle-consistency in latent space
            z_gt_flat    = rearrange(z_gt,    "b d h w -> (b h w) d")
            z_cycle_flat = rearrange(z_cycle, "b d h w -> (b h w) d")

            cycle_consistence_loss = 1.0 - F.cosine_similarity(
                z_cycle_flat,
                z_gt_flat,
                dim=-1,
            ).mean()

            self.unfreeze_VQPT_tokenizer()

        # --------- Total loss ---------
        total_loss = vq_loss + ce + perceptual_weight * lpips_loss + recon_weight * recon_loss
        if ar_cycle_consistency_weight > 0.0:
            total_loss = total_loss + ar_cycle_consistency_weight * cycle_consistence_loss

        # --------- VQ self-consistency + dynamic freeze/unfreeze ---------
        with torch.no_grad():
            token_self_consistency_val = self.tokenizer.token_self_consistency(
                target_videos
            ).detach()

            s = float(token_self_consistency_val)

            # --- VQ dynamic freeze/unfreeze ---
            if not self._vq_frozen:
                # consider freezing
                if s >= self.vq_freeze_threshold:
                    self._vq_freeze_counter += 1
                    if self._vq_freeze_counter >= self.vq_freeze_patience:
                        print(f"[VQ] Freezing VQ codebook (self-consistency={s:.3f})")
                        self.freeze_VQPT_tokenizer()
                        self._vq_unfreeze_counter = 0
                else:
                    self._vq_freeze_counter = 0
            else:
                # consider unfreezing
                if s <= self.vq_unfreeze_threshold:
                    self._vq_unfreeze_counter += 1
                    if self._vq_unfreeze_counter >= self.vq_unfreeze_patience:
                        print(f"[VQ] Unfreezing VQ codebook (self-consistency={s:.3f})")
                        self.unfreeze_VQPT_tokenizer()
                        self._vq_freeze_counter = 0
                else:
                    self._vq_unfreeze_counter = 0

            # --- DVAE freeze only when very stable ---
            if not self._dvae_frozen:
                if s >= self.dvae_freeze_threshold:
                    self._dvae_freeze_counter += 1
                    if self._dvae_freeze_counter >= self.dvae_freeze_patience:
                        if hasattr(self.tokenizer, "dvae"):
                            for p in self.tokenizer.dvae.parameters():
                                p.requires_grad = False
                        self._dvae_frozen = True
                        print(f"[VQ] Freezing DVAE (self-consistency={s:.3f})")
                else:
                    self._dvae_freeze_counter = 0

        return {
            "loss": total_loss,
            "vq_loss": vq_loss.detach(),
            "ce_loss": ce.detach(),
            "lpips_loss": lpips_loss.detach(),
            "ar_cycle_loss": cycle_consistence_loss.detach(),
            "token_accuracy": outputs["token_accuracy"],
            "token_self_consistency": token_self_consistency_val.detach(),
        }
        

    # AR generation from context
    @torch.no_grad()
    def generate_autoregressive(
        self,
        context_videos: torch.Tensor,   # [B,T_ctx_raw,C,H,W]
        num_frames_to_generate: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        LlamaGen-style AR generation from context frames only.

        Steps:
          1) Encode context_videos to tokens.
          2) Build temporal_ctx for the desired number of future encoded steps.
          3) AR-sample a flattened future token sequence of length
             L_future = T_pred_enc * tokens_per_frame.
          4) Unflatten tokens, concatenate with context tokens, decode to pixels.
        """
        self.eval()
        B, T_ctx_raw, C, H, W = context_videos.shape
        device = context_videos.device

        # 1) Encode context
        ctx_token_ids, _, _ = self.tokenizer.encode(context_videos)
        ctx_token_ids = ctx_token_ids.long()
        B2, T_ctx_enc, Ht, Wt = ctx_token_ids.shape[:4]
        assert B2 == B

        # Infer frames_per_token and corresponding encoded steps for generation
        frames_per_token = T_ctx_raw / float(T_ctx_enc)
        T_pred_enc = max(1, int(round(num_frames_to_generate / frames_per_token)))

        # 2) Perceiver encoder on context
        enc_out = self.encoder(ctx_token_ids)
        latents = enc_out.last_hidden_state          # [B,L,C]

        # Temporal context for the *future* interval
        temporal_ctx, _ = self.extract_temporal_bottleneck(
            latents,
            T_to_extract=T_pred_enc,
            T_start_index=T_ctx_enc,
        )                                            # [B,T_pred_enc,C]

        # Flatten semantics: each temporal step corresponds to tokens_per_frame positions.
        _, meta_ctx = self._flatten_tokens(ctx_token_ids)
        tokens_per_frame = meta_ctx["tokens_per_frame"]
        L_future = T_pred_enc * tokens_per_frame

        # Expand temporal_ctx over flat positions
        temporal_ctx_expanded = repeat(
            temporal_ctx,
            "b t c -> b (t r) c",
            r=tokens_per_frame,
        )                                            # [B,L_future,C]

        # 3) AR sampling over L_future tokens
        input_ids = torch.full(
            (B, 1),
            self.bos_token_id,
            device=device,
            dtype=torch.long,
        )                                            # [B,1]
        kv_cache: Optional[List[KVCache]] = []

        generated = []

        for step in range(L_future):
            # NOTE: for simplicity we do not use KV cache here; we just
            # feed the full prefix each time. It's O(L^2) but correct.
            out = self.ARdecoder(
                token_ids=input_ids,
                temporal_ctx=temporal_ctx_expanded,
                kv_cache=kv_cache,
            )
            kv_cache = out.kv_cache         # update cache (not used here anyway
            logits_all = out.last_hidden_state        # [B,cur_len,ar_vocab]
            logits_last = logits_all[:, -1, : self.code_vocab_size]  # drop BOS channel

            next_token = sample_tokens(
                logits_last,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )                                         # [B]

            generated.append(next_token)
            # Next step sees only this newly generated token
            input_ids = next_token.unsqueeze(1)      # [B, 1]

        generated_seq = torch.stack(generated, dim=1)    # [B,L_future]

        # 4) Unflatten into [B,T_pred_enc,Ht,Wt(,Q)]
        meta_future = dict(
            T=T_pred_enc,
            Ht=Ht,
            Wt=Wt,
            Q=meta_ctx["Q"],
            tokens_per_frame=tokens_per_frame,
        )
        future_tokens = self._unflatten_tokens(generated_seq, meta_future)

        # 5) Concatenate context + generated tokens and decode
        all_tokens = torch.cat([ctx_token_ids, future_tokens], dim=1)
        T_enc_all = all_tokens.shape[1]
        Ht_all = all_tokens.shape[2]
        Wt_all = all_tokens.shape[3]

        videos = self.tokenizer.decode_code(
            all_tokens,
            T_enc=T_enc_all,
            H_t=Ht_all,
            W_t=Wt_all,
        )
        return videos

    # Simple reconstruction through tokenizer (for debugging)
    @torch.no_grad()
    def reconstruct(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct videos through the VQ tokenizer 
        """
        token_ids, _, _ = self.tokenizer.encode(videos)
        _, T_enc, Ht, Wt = token_ids.shape[:4]

        return self.tokenizer.decode_code(
            token_ids,
            T_enc=T_enc,
            H_t=Ht,
            W_t=Wt,
        )


class CausalPerceiverIO(nn.Module):
    
    def __init__(
        self,
        video_shape: Tuple[int, int, int, int],  # (T, C, H, W)
        num_latents: int = 512,
        num_latent_channels: int = 512,
        num_attention_heads: int = 8,
        num_encoder_layers: int = 6,
        max_seq_len: int = 4096,
        code_dim: int = 256,
        num_codes: int = 1024,
        downsample: int = 4,
        dropout: float = 0.0,
        base_channels: int = 64,
        num_quantizers: int = 1,  # Multi-head VQ
        kmeans_init: bool = False,
        commitment_weight: float = 0.5,
        commitment_use_cross_entropy_loss: bool = False,
        orthogonal_reg_weight: float = 0.0,
        orthogonal_reg_active_codes_only: bool = True,
        orthogonal_reg_max_codes: int = 2048,
        threshold_ema_dead_code: int = 0,
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
            commitment_weight=commitment_weight,
            use_cosine_sim=False,
            kmeans_init=kmeans_init,
            dropout=dropout,
            num_quantizers=num_quantizers,
            commitment_use_cross_entropy_loss=commitment_use_cross_entropy_loss,
            orthogonal_reg_weight=orthogonal_reg_weight,
            orthogonal_reg_active_codes_only=orthogonal_reg_active_codes_only,
            orthogonal_reg_max_codes=orthogonal_reg_max_codes,
            codebook_diversity_loss_weight=0.00,
            codebook_diversity_temperature=1.0,
            threshold_ema_dead_code=threshold_ema_dead_code
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
            max_seq_len= max_seq_len,
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
        recon_weight: float = 1.0,
        label_smoothing: float = 0.1,
        ce_weight: float = 0.75,
        ar_cycle_consistency_weight: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss matching training script API"""
        return self.model.compute_loss(outputs, target_videos, perceptual_weight=perceptual_weight, recon_weight=recon_weight, label_smoothing=label_smoothing, ce_weight=ce_weight, ar_cycle_consistency_weight=ar_cycle_consistency_weight)

    
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
    
    @property
    def encoder(self):
        """Expose encoder from wrapped model"""
        return self.model.encoder
    
    
    @torch.no_grad()
    def reconstruct(self, videos: torch.Tensor) -> torch.Tensor:
        """Reconstruction matching training script API"""
        return self.model.reconstruct(videos)

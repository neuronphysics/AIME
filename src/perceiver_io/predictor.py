"""
Perceiver Token Predictor for Video Generation

Perceiver-based model that:
1. Encodes tokenized video sequences to latent representations
2. Decodes latents to predict future frame tokens
3. Supports both autoregressive and MaskGIT-style generation

Key Components:
- PerceiverTokenPredictor: Main model with encoder/decoder architecture
- Temporal bottleneck extraction for time-aligned context
- Multiple generation strategies (autoregressive, MaskGIT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, Optional
import lpips
import numpy as np

from .tokenizer import VQPTTokenizer, sample_tokens
from .modules import CrossAttention, SelfAttentionBlock
from .modules.adapter import TiedTokenOutputAdapter, TrainableQueryProvider
from .modules.position import (
    FourierPositionEncoding,
    FrequencyPositionEncoding,
    RotaryPositionEmbedding,
)
from .modules.utilities import ModuleOutput, _finite_stats


def positions(batch_size: int, length: int, device: torch.device) -> torch.Tensor:
    """Generate position indices for a sequence."""
    return torch.arange(length, device=device).unsqueeze(0).expand(batch_size, -1)


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
        T_to_extract: int,
        T_start_index: int,
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
        # FIX: Handle autoregressive generation where T_start_index may exceed trained positions
        all_queries = self.time_queries(B)  # [B, sequence_length, C]
        max_learned_pos = all_queries.shape[1]

        if T_start_index >= max_learned_pos:
            # Beyond trained range: use last learned query and rely on position encoding
            # Repeat the last query for all positions we need to extract
            q = all_queries[:, -1:, :].expand(B, T_to_extract, C)
        elif T_start_index + T_to_extract > max_learned_pos:
            # Partially out of bounds: use available queries then repeat last
            available = all_queries[:, T_start_index:, :]  # [B, remaining, C]
            remaining = T_to_extract - available.shape[1]
            repeated = all_queries[:, -1:, :].expand(B, remaining, C)
            q = torch.cat([available, repeated], dim=1)
        else:
            # Within trained range: use as before
            q = all_queries[:, T_start_index:T_start_index + T_to_extract, :]  # [B, T_ctx, C]

        # 2. Add explicit time position encoding (provides actual temporal information)
        t_pos = positions(B, T_to_extract, device=device)  + T_start_index  # [B, T_ctx]
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
        T_to_extract: int,
        T_start_index: int,
        Ht: int,
        Wt: int,
    ) -> torch.Tensor:

        B = latents.shape[0]


        self._ensure_pos_encoding(T_to_extract, Ht, Wt, latents.device)

        # Position encoding for output queries
        pos_enc = self.pos_encoding(B)  # (B, T*H*W, pos_dim)

        temporal_context = self.extract_temporal_bottleneck(latents, T_to_extract, T_start_index)
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
        logits = rearrange(logits, 'b (t h w) v -> b t h w v', t=T_to_extract, h=Ht, w=Wt)

        return logits, temporal_context

    def forward(
        self,
        videos: torch.Tensor,
        num_context_frames: int,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:

        B, T_total = videos.shape[:2]
        T_ctx = num_context_frames


        # Tokenize all frames
        token_ids, quantized, vq_loss, skips = self.tokenizer.encode(videos)
        T_tokens = token_ids.shape[1]

        assert token_ids.dim() == 4, f"Expected 4D token_ids, got {token_ids.dim()}D"
        downsample_factor = T_total / T_tokens  # e.g., 2.0 if 2x downsampling
        T_ctx_enc = int(T_ctx / downsample_factor)  # floor division
        T_ctx_enc = min(T_ctx_enc, T_tokens - 1)  # ensure at least 1 prediction token

        T_pred_enc = T_tokens - T_ctx_enc
        # Split into context and target
        context_tokens = token_ids[:, :T_ctx_enc]
        target_tokens = token_ids[:, T_ctx_enc:]

        # Get actual token grid dimensions
        Ht, Wt = context_tokens.shape[-2:]

        # Encode context
        latents = self.encoder(context_tokens)

        logits_future, temporal_context = self.decoder(
            latents.last_hidden_state,
            T_to_extract=T_pred_enc,
            T_start_index=T_ctx_enc,
            Ht=Ht,
            Wt=Wt,
        )

        assert logits_future.shape[:4] == target_tokens.shape, \
            f"Logits shape {logits_future.shape[:4]} doesn't match targets {target_tokens.shape}"

        # Predictions are for future frames only
        predicted_tokens = logits_future.argmax(dim=-1)

        # Reconstruct video from predicted tokens
        if return_dict:
            # Use the actual token dimensions for reconstruction
            B_full, T_tokens_actual, Ht, Wt = token_ids.shape

            # Create full token sequence
            full_tokens = torch.cat([
                token_ids[:, :T_ctx_enc],  # Context
                predicted_tokens           # Predicted future
            ], dim=1)

            # Ensure we don't exceed original temporal dimension
            if full_tokens.shape[1] > T_tokens_actual:
                full_tokens = full_tokens[:, :T_tokens_actual]
            elif full_tokens.shape[1] < T_tokens_actual:
                # Pad if needed (though this shouldn't happen)
                padding = T_tokens_actual - full_tokens.shape[1]
                full_tokens = torch.cat([
                    full_tokens,
                    torch.zeros(B_full, padding, Ht, Wt, device=full_tokens.device, dtype=full_tokens.dtype)
                ], dim=1)

            # Flatten and get codes
            flat_tokens = rearrange(full_tokens, 'b t h w -> (b t h w)')
            codes = self.tokenizer.get_codes_from_indices(flat_tokens)

            # Reshape back
            codes = rearrange(codes, '(b t h w) d -> b t d h w',
                            b=B_full, t=T_tokens_actual, h=Ht, w=Wt)

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
        T_total = reconstructed.shape[1]
        T_context = T_total - T_target
        reconstructed_pred = reconstructed[:, T_context:]

        # Check value ranges
        expected_min, expected_max = -1.0, 1.0
        actual_min_re, actual_max_re = reconstructed_pred.min().item(), reconstructed_pred.max().item()
        actual_min_tg, actual_max_tg = target_videos.min().item(), target_videos.max().item()
        if actual_min_re < expected_min or actual_max_re > expected_max:
            print(f"WARNING: reconstructed range [{actual_min_re:.3f}, {actual_max_re:.3f}] "
                  f"outside expected range [{expected_min}, {expected_max}]")
        if actual_min_tg < expected_min or actual_max_tg > expected_max:
            print(f"WARNING: target range [{actual_min_tg:.3f}, {actual_max_tg:.3f}] "
                  f"outside expected range [{expected_min}, {expected_max}]")
        # Clamp to valid range
        target_videos_norm = target_videos.clamp(-1, 1)
        recon_flat = rearrange(reconstructed_pred, 'b t c h w -> (b t) c h w')
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

        T_gen = num_frames_to_generate

        # Encode context
        context_tokens, _, _, _ = self.tokenizer.encode(context_videos)
        T_ctx = context_tokens.shape[1]
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

            logits_future, _ = self.decoder(
                latents.last_hidden_state,
                T_to_extract=T_gen,
                T_start_index=T_ctx,
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
        for i in range(num_frames_to_generate):
            # Encode current sequence
            latents = self.encoder(current_tokens)
            T_start_index = current_tokens.shape[1]
            # Decode - predict only next frame with warm-start
            logits_next, temporal_context = self.decoder(
                latents.last_hidden_state,
                T_to_extract=1,
                T_start_index=T_start_index,
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

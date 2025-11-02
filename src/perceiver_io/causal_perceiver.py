"""
Causal Perceiver IO Wrapper

High-level API wrapper around the Perceiver Token Predictor.

This module provides a simplified interface for:
- Video prediction with discrete tokens
- Context extraction for VRNN integration
- Multiple generation strategies (autoregressive, MaskGIT)
- Training and inference pipelines

Key Components:
- CausalPerceiverIO: Main wrapper class with clean API
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .tokenizer import VQPTTokenizer
from .predictor import PerceiverTokenPredictor


class CausalPerceiverIO(nn.Module):
    """
    Causal Perceiver IO for video prediction and context extraction.

    This wrapper provides a clean API for the Perceiver-based video prediction
    model, with methods for training, generation, and context extraction for
    integration with VRNN models.
    """

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
        kmeans_init: bool = False,
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
            commitment_weight=0.05,
            use_cosine_sim=False,
            kmeans_init=kmeans_init,
            gate_skips=False,
            use_3d_conv=use_3d_conv,
            temporal_downsample=temporal_downsample,
            dropout=dropout,
            num_quantizers=num_quantizers,
            codebook_diversity_loss_weight=0.005,
            codebook_diversity_temperature=1.0
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
    def extract_context(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Extract per-frame context vectors for VRNN.

        This method extracts latent context representations from video frames
        that can be used as conditional inputs to VRNN models.

        Args:
            videos: [B, T, C, H, W] - Input video sequences

        Returns:
            context: [B, T, context_dim] - Per-frame context vectors

        Example:
            >>> perceiver = CausalPerceiverIO(video_shape=(16, 3, 64, 64))
            >>> videos = torch.randn(2, 16, 3, 64, 64)
            >>> context = perceiver.extract_context(videos)
            >>> print(context.shape)  # [2, 16, 512]
        """
        B, T_total = videos.shape[:2]

        # Tokenize all frames
        token_ids, quantized, vq_loss, skips = self.tokenizer.encode(videos)
        T_tokens = token_ids.shape[1]

        # Encode all tokens
        latents = self.model.encoder(token_ids)

        # Extract temporal bottleneck for all frames
        # This creates time-aligned features for each frame
        temporal_context = self.model.extract_temporal_bottleneck(
            latents.last_hidden_state,
            T_to_extract=T_tokens,
            T_start_index=0,
        )

        return temporal_context  # [B, T_tokens, C]

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

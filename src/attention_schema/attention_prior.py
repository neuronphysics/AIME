"""
AttentionPrior: Top-down Attention Prediction Mechanism

This module implements a sophisticated attention-based spatial dynamics prediction system
that combines recurrent spatial modeling with efficient self-attention mechanisms.

Key Features:
- Spatial dynamics modeling using ConvGRU for temporal coherence
- Motion-aware feature extraction with learnable kernels
- Efficient multi-head self-attention with relative position biases
- Context-aware cross-attention for incorporating hidden and latent states
- Gradient checkpointing support for memory-efficient training

The AttentionPrior predicts the next attention map given:
- Previous attention distribution
- Current hidden state from the world model
- Current latent state from the posterior

This enables the model to learn spatiotemporal attention patterns that guide
the visual processing pipeline in a top-down manner.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt
from typing import Optional, Tuple, Dict

from .spatial_utils import ConvGRUCell


class AttentionPrior(nn.Module):
    """
    Attention-based spatial dynamics prediction using efficient self-attention
    """

    def __init__(
        self,
        attention_resolution: int = 21,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        motion_kernels: int = 8,
        num_heads: int = 4,  # Reduced for efficiency
        feature_dim: int = 64,  # Internal feature dimension
        use_relative_position_bias: bool = True,  # Whether to use relative position bias
        use_checkpoint: bool = True,  # Gradient checkpointing for memory efficiency
        dropout: float = 0.1,  # Dropout rate for attention and FFN
    ):
        super().__init__()
        self.attention_resolution = attention_resolution
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.spatial_dim = attention_resolution * attention_resolution
        self.use_relative_position_bias = use_relative_position_bias
        self.use_checkpoint = use_checkpoint
        # === ORIGINAL COMPONENTS (PRESERVED) ===

        # Spatial attention dynamics modeling (kept for feature extraction)
        self.spatial_dynamics = ConvGRUCell(input_size=1, hidden_size=32, kernel_size=5, cuda_flag=torch.cuda.is_available())
        # Motion prediction kernels (preserved)
        self.motion_kernels = nn.Parameter(
            torch.randn(motion_kernels, 1, 5, 5) * 0.01
        )

        # Context integration (adapted for new feature dimension)
        self.context_projection = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim)  # Project to feature dimension
        )

        # === ATTENTION COMPONENTS ===

        # Efficient feature projection for attention
        self.spatial_downsampler = nn.Conv2d(32, feature_dim, 3, stride=1, padding=1)

        # Motion feature projection
        self.motion_projection = nn.Conv2d(motion_kernels, feature_dim // 2, 1)

        # Learnable position embeddings (2D-aware)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.spatial_dim, feature_dim) * 0.02
        )

        # Relative position bias for global attention
        if use_relative_position_bias:
            # Create learnable relative position biases
            max_relative_position = 2 * attention_resolution - 1
            self.relative_position_bias_table = nn.Parameter(
                torch.randn(max_relative_position, max_relative_position, num_heads) * 0.02
            )

            # Create relative position index
            coords_h = torch.arange(attention_resolution)
            coords_w = torch.arange(attention_resolution)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, H, W
            coords_flatten = torch.flatten(coords, 1)  # 2, H*W

            # Compute relative coordinates
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, H*W, H*W
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # H*W, H*W, 2
            rc_0 = relative_coords[:, :, 0] + attention_resolution - 1
            rc_1 = relative_coords[:, :, 1] + attention_resolution - 1
            relative_coords = torch.stack([rc_0, rc_1], dim=-1)

            self.register_buffer("relative_position_index", relative_coords)

        # Efficient self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention with context
        self.context_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)

        # FFN for processing attention output
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )

        # Output projection to attention logits
        self.output_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Movement predictor (preserved from original)
        self.movement_predictor = nn.Conv2d(1, 2, 3, padding=1)
        self.use_checkpoint = False

    def compute_motion_features(self, prev_attention: torch.Tensor) -> torch.Tensor:
        """Extract motion-relevant features from previous attention (unchanged)"""
        batch_size = prev_attention.shape[0]
        prev_attention = prev_attention.unsqueeze(1)  # [B, 1, H, W]

        # Apply learned motion kernels
        motion_responses = []
        for kernel in self.motion_kernels:
            response = F.conv2d(
                prev_attention,
                kernel.unsqueeze(0),
                padding=2
            )
            motion_responses.append(response)

        motion_features = torch.cat(motion_responses, dim=1)  # [B, K, H, W]
        return motion_features

    def gradient_checkpointing_disable(self):
        """
        Disable gradient checkpointing.
        """
        self.use_checkpoint = False

    def gradient_checkpointing_enable(self):
        """
        Enable gradient checkpointing for memory-efficient training.
        This will trade compute for memory by recomputing activations during backward pass.
        """
        self.use_checkpoint = True

    def _maybe_ckpt(self, fn, *tensors, reentrant: bool = False):
        """
        Checkpoint `fn(*tensors)` during training if enabled.
        - Only Tensor args are allowed (put non-tensor args in the closure).
        - Returns whatever `fn` returns (must be Tensor or tuple of Tensors).
        """
        if self.training and self.use_checkpoint :
            return _ckpt(fn, *tensors, use_reentrant=reentrant, preserve_rng_state=False)
        else:
            return fn(*tensors)

    def forward(
        self,
        prev_attention: torch.Tensor,  # [B, H, W]
        hidden_state: torch.Tensor,    # [B, hidden_dim]
        latent_state: torch.Tensor     # [B, latent_dim]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute spatially-aware attention prior using Vaswani attention
        Maintains exact interface of original AttentionPrior
        """
        batch_size = prev_attention.shape[0]
        H, W = self.attention_resolution, self.attention_resolution

        # 1. Extract spatial dynamics features (original component)
        spatial_features = self.spatial_dynamics(
            prev_attention.unsqueeze(1)
        )  # [B, 32, H, W]

        # 2. Compute motion features (original component)
        motion_features = self.compute_motion_features(prev_attention)

        # 3. Project features to attention dimension
        if self.training and self.use_checkpoint:
            spatial_feat_proj = self._maybe_ckpt(
                self.spatial_downsampler,
                spatial_features
            )  # [B, feature_dim, H, W]
            motion_feat_proj = self._maybe_ckpt(
                self.motion_projection,
                motion_features
            )  # [B, feature_dim//2, H, W]
        else:
            spatial_feat_proj = self.spatial_downsampler(spatial_features)  # [B, feature_dim, H, W]
            motion_feat_proj = self.motion_projection(motion_features)  # [B, feature_dim//2, H, W]

        # Combine spatial and motion features
        motion_feat_proj = F.interpolate(
            motion_feat_proj,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        # 4. Convert to sequence format
        spatial_seq = spatial_feat_proj.permute(0, 2, 3, 1).reshape(batch_size, -1, self.feature_dim)
        motion_seq = motion_feat_proj.permute(0, 2, 3, 1).reshape(batch_size, -1, self.feature_dim // 2)

        # Pad motion features and combine
        motion_seq_padded = F.pad(motion_seq, (0, self.feature_dim // 2))
        combined_seq = spatial_seq + motion_seq_padded  # [B, H*W, feature_dim]

        # 5. Add positional embeddings
        combined_seq = combined_seq + self.pos_embed

        # 6. Get context features
        if self.training and self.use_checkpoint:
            context_features = self._maybe_ckpt(
                self.context_projection,
                torch.cat([hidden_state, latent_state], dim=-1)
            )  # [B, feature_dim]
        else:
            context_features = self.context_projection(torch.cat([hidden_state, latent_state], dim=-1))  # [B, feature_dim]
        context_seq = context_features.unsqueeze(1)  # [B, 1, feature_dim]

        # 7. Self-attention with relative position bias (GLOBAL attention)
        normed_seq = self.norm1(combined_seq)

        def _self_attn(q, k, v):
            return self.self_attention(q, k, v)[0]

        if self.training and self.use_checkpoint:
            self_attn_out = self._maybe_ckpt(
                _self_attn,
                normed_seq,
                normed_seq,
                normed_seq
            )
        else:
            self_attn_out, _ = self.self_attention(
                normed_seq, normed_seq, normed_seq
            )

        combined_seq = combined_seq + self_attn_out

        # 8. Cross-attention with context
        normed_seq = self.norm2(combined_seq)
        context_expanded = context_seq.expand(-1, self.spatial_dim, -1)
        def _context_attn(q, k, v):
            return self.context_attention(q, k, v)[0]

        if self.training and self.use_checkpoint:
            cross_attn_out = self._maybe_ckpt(
                _context_attn,
                normed_seq,
                context_expanded,
                context_expanded
            )
        else:
            cross_attn_out, _ = self.context_attention(
                query=normed_seq,
                key=context_expanded,
                value=context_expanded
            )
        combined_seq = combined_seq + cross_attn_out

        # 9. FFN
        normed_seq = self.norm3(combined_seq)
        combined_seq = combined_seq + self.ffn(normed_seq)

        # 10. Generate attention logits
        attention_logits = self.output_projection(combined_seq).squeeze(-1)  # [B, H*W]

        # 11. Apply softmax to get probabilities
        attention_probs = F.softmax(attention_logits, dim=-1)
        attention_probs_2d = attention_probs.view(batch_size, H, W)

        # 12. Predict attention movement
        movement = self.movement_predictor(prev_attention.unsqueeze(1))
        dx, dy = movement[:, 0], movement[:, 1]

        return attention_probs_2d, {
            'spatial_features': spatial_features,
            'motion_features': motion_features,
            'predicted_movement': (dx, dy),
            'attention_logits': attention_logits.view(batch_size, H, W)
        }

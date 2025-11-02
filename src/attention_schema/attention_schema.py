"""
Attention Schema Module

Implements the complete attention schema mechanism that combines bottom-up
(stimulus-driven) and top-down (predictive) attention processing.

The attention schema integrates:
- AttentionPosterior: Bottom-up multi-object attention from visual input
- AttentionPrior: Top-down prediction of attention dynamics

This implements the "Attention Schema Theory" (Graziano & Kastner, 2011)
where attention itself is modeled as an internal state that can be predicted
and controlled.

Theory:
    The attention schema maintains:
    1. Posterior: What is currently attended (bottom-up from stimuli)
    2. Prior: Where attention should move next (top-down prediction)
    3. Dynamics loss: Consistency between predicted and actual attention movement

Architecture:
    observation → [FPN] → features
                    ↓
    features + h_t + c_t → AttentionPosterior → attention_map_t
                    ↓
    prev_attention + h_t + z_t → AttentionPrior → predicted_attention_{t+1}

Tensor Flow:
    Input:  observation [B, 3, 84, 84]
            hidden_state [B, hidden_dim]
            context [B, context_dim]
    Output: attention_probs_2d [B, H_att, W_att]
            regularized_coords [B, 2]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from .attention_posterior import AttentionPosterior
from .attention_prior import AttentionPrior


class AttentionSchema(nn.Module):
    """
    Complete attention schema implementation with proper spatial modeling.

    Combines bottom-up (posterior) and top-down (prior) attention mechanisms
    to model both stimulus-driven and predictive attention dynamics.

    Args:
        image_size (int): Input image size (e.g., 84)
        attention_resolution (int): Resolution of attention maps (e.g., 21)
        hidden_dim (int): Hidden state dimension from VRNN
        latent_dim (int): Latent state dimension from VRNN
        context_dim (int): Context dimension from Perceiver
        attention_dim (int): Internal attention feature dimension
        input_channels (int): Number of input channels (3 for RGB)
        device (torch.device): Device to run on

    Example:
        >>> attention = AttentionSchema(
        ...     image_size=84,
        ...     attention_resolution=21,
        ...     hidden_dim=256,
        ...     latent_dim=32,
        ...     context_dim=128
        ... )
        >>> obs = torch.randn(2, 3, 84, 84)
        >>> h = torch.randn(2, 256)
        >>> c = torch.randn(2, 128)
        >>> attn_probs, coords = attention.posterior_net(obs, h, c)
        >>> print(attn_probs.shape)  # [2, 21, 21]
        >>> print(coords.shape)  # [2, 2]
    """

    def __init__(
        self,
        image_size: int = 84,
        attention_resolution: int = 21,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        context_dim: int = 128,
        attention_dim: int = 64,
        input_channels: int = 3,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super().__init__()

        # Posterior (bottom-up, stimulus-driven attention)
        self.posterior_net = AttentionPosterior(
            image_size=image_size,
            attention_resolution=attention_resolution,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            input_channels=input_channels,
            feature_channels=64,         # must equal d in fused_attention_features(...)
            num_semantic_slots=4,
            num_heads=4,
            attention_fusion_mode="weighted",
            enforce_diversity=True,
            device=device,
            expected_fused=False          # <--- important: skip internal pyramid
        )
        # Prior (top-down, predictive attention schema)
        self.prior_net = AttentionPrior(
            attention_resolution=attention_resolution,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            motion_kernels=8,
            feature_dim=hidden_dim
        )
        self.prior_net.gradient_checkpointing_enable()
        self.to(device)

    def compute_attention_dynamics_loss(
        self,
        attention_sequence: List[torch.Tensor],  # list of [B,H,W] soft attention maps
        predicted_movements: List[torch.Tensor]  # list of dicts or tensors for dx,dy
    ) -> torch.Tensor:
        """
        Compute loss between predicted and actual attention movement.

        This measures how well the attention prior predicts the actual
        movement of attention over time.

        Args:
            attention_sequence: List of [B, H, W] attention maps over time
            predicted_movements: List of predicted movement tensors

        Returns:
            loss: Scalar loss value (smooth L1 between predicted and actual)

        Example:
            >>> attn_seq = [torch.randn(2, 21, 21) for _ in range(10)]
            >>> pred_mov = [torch.randn(2, 2) for _ in range(9)]
            >>> loss = attention.compute_attention_dynamics_loss(attn_seq, pred_mov)
        """
        if len(attention_sequence) < 2:
            return torch.tensor(0.0, device=attention_sequence[0].device)

        # [T, B, H, W]
        att = torch.stack(attention_sequence, dim=0)
        T, B, H, W = att.shape

        # center of mass per frame
        y = torch.arange(H, device=att.device, dtype=att.dtype)
        x = torch.arange(W, device=att.device, dtype=att.dtype)
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
        y_grid = y_grid[None, None]  # [1,1,H,W]
        x_grid = x_grid[None, None]

        mass = att.sum(dim=(2, 3), keepdim=True).clamp_min(1e-8)
        y_com = (att * y_grid).sum(dim=(2, 3)) / mass.squeeze(-1).squeeze(-1)   # [T,B]
        x_com = (att * x_grid).sum(dim=(2, 3)) / mass.squeeze(-1).squeeze(-1)   # [T,B]
        centers = torch.stack([x_com, y_com], dim=-1)                            # [T,B,2]

        # actual deltas between consecutive frames
        actual = centers[1:] - centers[:-1]                                      # [T-1,B,2]
        actual_dx = actual[:, :, 0]
        actual_dy = actual[:, :, 1]

        # predicted movements: accept either vector [T-1,B,2] or fields [T-1,B,2,H,W]
        if isinstance(predicted_movements, (list, tuple)):
            pred = torch.stack(predicted_movements, dim=0)                        # try [T-1,B,2,(H,W)?]
        else:
            pred = predicted_movements

        if pred.dim() == 5:
            # [T-1,B,2,H,W] → average to [T-1,B,2]
            pred_dx = pred[:, :, 0].mean(dim=(2, 3))
            pred_dy = pred[:, :, 1].mean(dim=(2, 3))
        elif pred.dim() == 3:
            # [T-1,B,2]
            pred_dx = pred[:, :, 0]
            pred_dy = pred[:, :, 1]
        else:
            raise ValueError(f"Unexpected predicted_movements shape: {tuple(pred.shape)}")

        # scale-match if needed (optional): both are already in pixel units of the attention grid
        loss = F.smooth_l1_loss(pred_dx, actual_dx) + F.smooth_l1_loss(pred_dy, actual_dy)
        return loss

    def _center_of_mass(self, attention_map: torch.Tensor) -> torch.Tensor:
        """
        Compute attention center of mass for movement tracking.

        Args:
            attention_map: [B, H, W] - Attention probability map

        Returns:
            centers: [B, 2] - (x, y) coordinates of attention center
        """
        batch_size, H, W = attention_map.shape

        # Create coordinate grids
        y_coords = torch.arange(H, device=attention_map.device).float()
        x_coords = torch.arange(W, device=attention_map.device).float()
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Compute weighted average
        total_mass = attention_map.sum(dim=[1, 2], keepdim=True) + torch.finfo(torch.float32).eps  # Avoid division by zero
        y_com = (attention_map * y_grid).sum(dim=[1, 2]) / total_mass.squeeze()
        x_com = (attention_map * x_grid).sum(dim=[1, 2]) / total_mass.squeeze()

        return torch.stack([x_com, y_com], dim=-1)

    def _compute_posterior_attention(
        self,
        encoder,
        obs: torch.Tensor,
        h_t: torch.Tensor,
        c_t: torch.Tensor,
        *,
        detach: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior (bottom-up) attention from observations.

        This method allows for optional integration with an encoder that
        provides pre-fused attention features.

        Args:
            encoder: VAE encoder with fused_attention_features method (optional)
            obs: [B, 3, H, W] - Observations
            h_t: [B, hidden_dim] - Hidden state
            c_t: [B, context_dim] - Context
            detach: Whether to detach gradients from encoder

        Returns:
            attention_probs_2d: [B, H_att, W_att] - Attention probabilities
            regularized_coords: [B, 2] - Attention center coordinates
        """
        if self.posterior_net.expected_fused:
            # 1) Build pre-fused encoder features that match the posterior's expectations
            fused = encoder.fused_attention_features(
                x=obs,                                   # [B,3,84,84]
                out_hw=(self.posterior_net.attention_resolution,
                        self.posterior_net.attention_resolution),  # (21,21)
                source=None,                             # or ("C3","C4","C5")
                fuse='concat+1x1',                       # project to d
                d=self.posterior_net.feature_channels,   # 64
                detach=detach                            # True to stop grads into encoder
            )
        else:
            # 1) Use raw encoder features directly (posterior has internal pyramid)
            fused = None
        # 2) Call the posterior with the pre-fused map
        return self.posterior_net(
            observation=obs,
            hidden_state=h_t,
            context=c_t,
            fused_feat=fused
        )

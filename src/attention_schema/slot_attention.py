"""
Slot Attention Module

Implements the Slot Attention mechanism from Locatello et al. 2020.
Slot Attention performs iterative attention-based routing to decompose
inputs into object-centric representations (slots).

Reference:
    Locatello, F., Weissenborn, D., Unterthiner, T., Mahendran, A., Heigold, G.,
    Uszkoreit, J., ... & Kipf, T. (2020). Object-centric learning with slot attention.
    Advances in Neural Information Processing Systems, 33, 11525-11538.

Theory:
    Slot Attention implements a competitive attention mechanism where:
    - Slots compete for explaining input features
    - Attention weights sum to 1 over slots (not over inputs)
    - Slots are updated iteratively using GRU + MLP

Tensor Flow:
    Input:  [B, N, D_in] sequence of input tokens
    Output: [B, K, D_slot] slot representations
            [B, K, N] attention weights (responsibilities)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SlotAttention(nn.Module):
    """
    Locatello et al. 2020 Slot Attention (iterative routing).
    x: [B, N, D] tokens  ->  slots: [B, K, D], attn: [B, K, N] (sum_K=1 per position)

    Args:
        num_slots (int): Number of slots K
        in_dim (int): Input feature dimension D_in
        slot_dim (int): Slot feature dimension D_slot
        iters (int): Number of iterative refinement steps
        mlp_hidden (int): Hidden dimension for slot update MLP
        eps (float): Small constant for numerical stability

    Returns:
        slots: [B, K, D_slot] - Updated slot representations
        attn: [B, K, N] - Attention weights (sum to 1 over K for each position)
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

    def forward(self, x: torch.Tensor, seed_slots: torch.Tensor = None):
        """
        Forward pass of Slot Attention.

        Args:
            x: [B, N, D_in] - Input tokens (N = H*W for images)
            seed_slots: [B, K, D_slot] - Optional slot initialization (for top-down conditioning)

        Returns:
            slots: [B, K, D_slot] - Updated slot representations
            attn: [B, K, N] - Attention weights

        Example:
            >>> slot_attn = SlotAttention(num_slots=4, in_dim=64, slot_dim=64)
            >>> x = torch.randn(2, 100, 64)  # 2 batches, 100 positions, 64 features
            >>> slots, attn = slot_attn(x)
            >>> print(slots.shape)  # [2, 4, 64]
            >>> print(attn.shape)   # [2, 4, 100]
        """
        B, N, D = x.shape
        x = self.norm_x(x)
        k = self.to_k(x)                      # [B,N,D]
        v = self.to_v(x)                      # [B,N,D]

        if seed_slots is None:
            mu = self.slots_mu.expand(B, self.K, -1)
            sigma = self.slots_logsigma.exp().expand(B, self.K, -1)
            slots = mu + sigma * torch.randn_like(mu)
        else:
            slots = seed_slots

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
        """Update slots using GRU cell"""
        B, K, D = slots.shape
        s = slots.reshape(B*K, D)
        u = updates.reshape(B*K, D)
        out = self.gru(u, s)
        return out.view(B, K, D)

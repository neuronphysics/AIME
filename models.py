import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict, Union, List

import logging
import functools
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch.autograd import Function
import collections.abc as abc

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
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




class AttentionPosterior(nn.Module):
    """
    Multi-object attention:
    Efficient: linear in N * K. Suitable for video pipelines.

    Returns:
        attention_probs_2d: [B, H_att, W_att] fused attention map (probabilities)
        regularized_coords: [B, 2] combined (x,y) in [-1,1] after light regularization

    Side attributes you may use:
        slot_attention_maps: [B, K, H_att, W_att] per-slot attention maps (probabilities)
        slot_centers:       [B, K, 2] per-slot (x,y)
        fusion_weights:     [B, K] (when attention_fusion_mode='weighted')
        diversity_loss:     scalar (0.0 if disabled)
        ortho_loss:         scalar (0.0 if disabled)
        bottleneck_features:[B, hidden_dim//2]
    """

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
        attention_fusion_mode: str = "weighted",  # 'weighted' | 'max' | 'gated'
        enforce_diversity: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert feature_channels % num_heads == 0, "feature_channels must be divisible by num_heads"
        assert attention_fusion_mode in {"weighted", "max", "gated"}

        self.image_size = image_size
        self.attention_resolution = attention_resolution
        self.feature_channels = feature_channels
        self.hidden_dim = hidden_dim
        self.num_semantic_slots = num_semantic_slots
        self.num_heads = num_heads
        self.attention_fusion_mode = attention_fusion_mode
        self.enforce_diversity = enforce_diversity
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.N = attention_resolution * attention_resolution
        self.d = feature_channels
        self.h = num_heads
        self.d_head = self.d // self.h

        # ========= Feature Pyramid (84 -> 42 -> 21) =========
        self.pyramid_conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, stride=2, padding=2),  # 84->42
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.pyramid_conv2 = nn.Sequential(
            nn.Conv2d(32, 48, 5, stride=2, padding=2),  # 42->21
            nn.GroupNorm(12, 48),
            nn.SiLU(),
        )
        self.pyramid_conv3 = nn.Sequential(
            nn.Conv2d(48, feature_channels, 3, stride=1, padding=1),  # stay 21
            nn.GroupNorm(16, feature_channels),
            nn.SiLU(),
        )
        self.pyramid_fusion = nn.Conv2d(32 + 48 + feature_channels, feature_channels, 1)

        # ========= GroupViT-style grouping =========
        # Learnable group tokens that compete for features
        self.group_tokens = nn.Parameter(torch.randn(1, num_semantic_slots, self.d) * 0.02)

        # Slot prototypes (semantic specialization)
        self.slot_specialization = nn.Parameter(torch.randn(num_semantic_slots, self.d) * 0.02)

        # Temperature for grouping / diversity; guarded at use time
        if enforce_diversity:
            self.diversity_temp = nn.Parameter(torch.tensor(1.0))

        # ========= Top-down modulation (per slot) =========
        self.slot_modulator = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, self.d * num_semantic_slots),
            nn.LayerNorm(self.d * num_semantic_slots),
            nn.SiLU(),
            nn.Linear(self.d * num_semantic_slots, self.d * num_semantic_slots),
        )

        # ========= Attention projections (shared K,V; per-slot Q) =========
        self.q_proj = nn.Linear(self.d, self.d)
        self.k_proj = nn.Linear(self.d, self.d)
        self.v_proj = nn.Linear(self.d, self.d)

        # ========= 2D positional encodings =========
        self.row_embed = nn.Parameter(torch.randn(attention_resolution, self.d // 2) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(attention_resolution, self.d // 2) * 0.02)

        # ========= Fusion heads =========
        if self.attention_fusion_mode == "weighted":
            self.fusion_weights_head = nn.Sequential(
                nn.Linear(self.d * num_semantic_slots, num_semantic_slots),
                nn.LayerNorm(num_semantic_slots),
                nn.Softmax(dim=-1),
            )
        elif self.attention_fusion_mode == "gated":
            self.fusion_gate = nn.Sequential(
                nn.Linear(hidden_dim + context_dim, num_semantic_slots),
                nn.LayerNorm(num_semantic_slots),
                nn.Sigmoid(),
            )

        # ========= Per-slot temperatures for attention sharpening =========
        self.slot_temperatures = nn.Parameter(torch.ones(num_semantic_slots))

        # ========= Coordinate regularizer (combine K centers -> 2) =========
        self.spatial_regularizer = nn.Sequential(
            nn.Linear(2 * num_semantic_slots, 16),
            nn.LayerNorm(16),
            nn.SiLU(),
            nn.Linear(16, 2),
            nn.Tanh(),
        )

        # ========= Bottleneck from concatenated slot features =========
        self.bottleneck_projector = nn.Sequential(
            nn.Linear(self.d * num_semantic_slots, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Bookkeeping
        self.diversity_loss = torch.tensor(0.0)
        self.ortho_loss = torch.tensor(0.0)

        self.to(self.device)

    # ---------- helpers ----------
    def _positional_encoding_2d(self) -> torch.Tensor:
        R, C, d = self.attention_resolution, self.attention_resolution, self.d
        pos = torch.cat(
            [
                self.row_embed.unsqueeze(1).expand(-1, C, -1),  # [R, C, d/2]
                self.col_embed.unsqueeze(0).expand(R, -1, -1),  # [R, C, d/2]
            ],
            dim=-1,
        )  # [R, C, d]
        return pos.flatten(0, 1).unsqueeze(0)  # [1, N, d]

    @staticmethod
    def _reshape_heads(x: torch.Tensor, B: int, seq_len: int, h: int, d_head: int) -> torch.Tensor:
        # [B, seq_len, d] -> [B, h, seq_len, d_head]
        return x.view(B, seq_len, h, d_head).permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def _merge_heads(x: torch.Tensor) -> torch.Tensor:
        # [B, h, seq_len, d_head] -> [B, seq_len, h*d_head]
        B, h, L, d_head = x.shape
        return x.permute(0, 2, 1, 3).reshape(B, L, h * d_head)

    # ---------- diversity losses ----------
    def _compute_diversity_losses(self, slot_maps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        slot_maps: [B, K, H, W] (logits or weights). We convert to probabilities.
        Returns (diversity_loss, ortho_loss)
        """
        B, K, H, W = slot_maps.shape
        # Normalize to probabilities over spatial dims
        p = slot_maps.view(B, K, -1)
        p = p.softmax(dim=-1)  # [B, K, N]
        p = F.normalize(p, p=2, dim=-1)
        corr = torch.bmm(p, p.transpose(1, 2))  # [B, K, K]
        I = torch.eye(K, device=corr.device).unsqueeze(0)
        div = ((corr - I) ** 2).sum() / (B * K * (K - 1) + 1e-8)

        # Optional: orthogonality in slot specialization space
        S = F.normalize(self.slot_specialization, dim=-1)  # [K, d]
        ortho = ((S @ S.T - torch.eye(K, device=S.device)) ** 2).mean()

        return div, ortho

    # ---------- forward ----------
    def forward(
        self,
        observation: torch.Tensor,  # [B, 3, 84, 84]
        hidden_state: torch.Tensor, # [B, hidden_dim]
        context: torch.Tensor,      # [B, context_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B = observation.size(0)
        H_att = W_att = self.attention_resolution
        device = observation.device

        # === FPN ===
        l1 = self.pyramid_conv1(observation)                       # [B, 32, 42, 42]
        l2 = self.pyramid_conv2(l1)                                # [B, 48, 21, 21]
        l3 = self.pyramid_conv3(l2)                                # [B, 64, 21, 21]

        l1_up = F.interpolate(l1, size=(H_att, W_att), mode="bilinear", align_corners=False)
        l2_up = F.interpolate(l2, size=(H_att, W_att), mode="bilinear", align_corners=False)
        l3_up = F.interpolate(l3, size=(H_att, W_att), mode="bilinear", align_corners=False)
        
        fused = self.pyramid_fusion(torch.cat([l1_up, l2_up, l3_up], dim=1))
        self._last_saliency_features = fused  # Save for visualization

        # === sequence + positions ===
        feat_seq = fused.flatten(2).transpose(1, 2)                # [B, N, d]
        pos = self._positional_encoding_2d().to(device)            # [1, N, d]
        feat_seq = feat_seq + pos                                  # [B, N, d]

        # === grouping via dot-product similarity ===
        group_tokens = self.group_tokens.expand(B, -1, -1)         # [B, K, d]
        tau = self.diversity_temp if self.enforce_diversity else 1.0
        assign_logits = torch.einsum("bnd,bkd->bnk", feat_seq, group_tokens) / float(tau)
        group_assignments = assign_logits.softmax(dim=-1)          # [B, N, K]
        # Save for visualization as [B, H, W, K]
        self.group_assignments = group_assignments.view(B, H_att, W_att, self.num_semantic_slots)

        # === per-slot queries (modulated) ===
        top_down = torch.cat([hidden_state, context], dim=-1)      # [B, hidden+context]
        modulation = self.slot_modulator(top_down).view(B, self.num_semantic_slots, self.d)  # [B,K,d]

        # slots = specialization + learnable bias (optional)
        slots = self.slot_specialization.unsqueeze(0).expand(B, -1, -1)  # [B,K,d]
        modulated_slots = slots + modulation                                # [B,K,d]

        # === shared K,V ===
        K_lin = self.k_proj(feat_seq)                             # [B, N, d]
        V_lin = self.v_proj(feat_seq)                             # [B, N, d]
        Kh = self._reshape_heads(K_lin, B, self.N, self.h, self.d_head)    # [B,h,N,d_head]
        Vh = self._reshape_heads(V_lin, B, self.N, self.h, self.d_head)    # [B,h,N,d_head]

        slot_features = []
        slot_maps = []

        # Precompute grids for centers
        y_coords = torch.linspace(-1, 1, H_att, device=device)
        x_coords = torch.linspace(-1, 1, W_att, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")  # [H,W]

        eps = 1e-6
        scale = 1.0 / math.sqrt(self.d_head)

        for k in range(self.num_semantic_slots):
            # Q
            q_lin = self.q_proj(modulated_slots[:, k, :]).unsqueeze(1)     # [B, 1, d]
            Qh = self._reshape_heads(q_lin, B, 1, self.h, self.d_head)     # [B,h,1,d_head]

            # bias from assignments: log-prob preference per position
            # group_assignments: [B, N, K] -> take slot k -> [B, N]
            bias_k = (group_assignments[:, :, k].clamp_min(eps)).log()      # [B, N]
            bias_k = bias_k.view(B, 1, 1, self.N)                           # [B,1,1,N] broadcast over heads

            # attention logits: QK^T / sqrt(d) + bias
            logits = torch.matmul(Qh, Kh.transpose(-2, -1)) * scale         # [B,h,1,N]
            logits = logits + bias_k                                        # additive bias

            # per-slot temperature (sharpening)
            temp = self.slot_temperatures[k].clamp(min=0.1)
            logits = logits / float(temp)

            # weights & context
            attn = logits.softmax(dim=-1)                                   # [B,h,1,N]
            context = torch.matmul(attn, Vh)                                 # [B,h,1,d_head]
            context = context.squeeze(2)                                     # [B,h,d_head]
            context = context.transpose(1, 2).reshape(B, self.d)             # [B,d]

            # save per-slot attention map (average heads)
            attn_avg = attn.mean(dim=1).squeeze(1)                           # [B,N]
            slot_map_2d = attn_avg.view(B, H_att, W_att)                     # [B,H,W]

            slot_features.append(context)           # [B,d]
            slot_maps.append(slot_map_2d)           # [B,H,W]

        slot_features = torch.stack(slot_features, dim=1)   # [B,K,d]
        slot_maps = torch.stack(slot_maps, dim=1)           # [B,K,H,W]
        self.slot_attention_maps = slot_maps
        self.slot_features = slot_features
        # === fusion ===
        if self.attention_fusion_mode == "weighted":
            all_feat = slot_features.reshape(B, -1)                           # [B,K*d]
            fusion_weights = self.fusion_weights_head(all_feat)               # [B,K], sums to 1
            self.fusion_weights = fusion_weights
            w = fusion_weights.view(B, self.num_semantic_slots, 1, 1)         # [B,K,1,1]
            fused_map = (slot_maps * w).sum(dim=1)                            # [B,H,W]
        elif self.attention_fusion_mode == "max":
            fused_map, _ = slot_maps.max(dim=1)                               # [B,H,W]
            self.fusion_weights = None
        else:  # 'gated'
            gates = self.fusion_gate(top_down).view(B, self.num_semantic_slots, 1, 1)  # [B,K,1,1]
            fused_map = (slot_maps * gates).sum(dim=1)                         # [B,H,W]
            self.fusion_weights = gates.squeeze(-1).squeeze(-1)               # [B,K]

        # Normalize fused map to probabilities over spatial positions
        attention_probs = fused_map.flatten(1).softmax(dim=-1)                # [B,N]
        attention_probs_2d = attention_probs.view(B, H_att, W_att)            # [B,H,W]

        # === per-slot centers & combined coords ===
        slot_centers = []
        for k in range(self.num_semantic_slots):
            pk = slot_maps[:, k].flatten(1).softmax(dim=-1).view(B, H_att, W_att)  # [B,H,W]
            y_c = (pk * y_grid).sum(dim=[1, 2])                                    # [B]
            x_c = (pk * x_grid).sum(dim=[1, 2])                                    # [B]
            slot_centers.append(torch.stack([x_c, y_c], dim=-1))                   # [B,2]
        slot_centers = torch.stack(slot_centers, dim=1)                             # [B,K,2]
        self.slot_centers = slot_centers

        if self.attention_fusion_mode == "weighted":
            w = self.fusion_weights  # [B,K]
            coords = (slot_centers * w.unsqueeze(-1)).sum(dim=1)                   # [B,2]
        else:
            y_c = (attention_probs_2d * y_grid).sum(dim=[1, 2])
            x_c = (attention_probs_2d * x_grid).sum(dim=[1, 2])
            coords = torch.stack([x_c, y_c], dim=-1)                                # [B,2]

        # Light reg to keep inside [-1,1] and stabilize
        regularized_coords = self.spatial_regularizer(slot_centers.flatten(1)) * 0.995  # [B,2]

        # === diversity losses (optional) ===
        if self.enforce_diversity and self.training:
            div, ortho = self._compute_diversity_losses(slot_maps)
            self.diversity_loss = div
            self.ortho_loss = ortho
        else:
            self.diversity_loss = torch.tensor(0.0, device=device)
            self.ortho_loss = torch.tensor(0.0, device=device)

        # === bottleneck features ===
        self.bottleneck_features = self.bottleneck_projector(slot_features.reshape(B, -1))  # [B,hidden//2]

        return attention_probs_2d, regularized_coords


    @torch.no_grad()
    def visualize_multi_object_attention(
        self,
        observation: torch.Tensor,                 # [B, 3, H, W] (RGB), in [0,1] or [-1,1]
        slot_attention_maps: torch.Tensor,         # [B, K, H_att, W_att]
        slot_centers: torch.Tensor,                # [B, K, 2]  ([-1,1] coords)
        group_assignments: Optional[torch.Tensor] = None,  # [B, H_att, W_att, K] or [B, K, H_att, W_att]
        return_mode: str = "all",                  # 'all' | 'combined' | 'slots' | 'groups'
    ) -> Dict[str, torch.Tensor]:
        """
        Pure-PyTorch, stateless visualization.

        Returns (all tensors on observation.device, observation.dtype):
        - 'slot_overlays'         : [B*K, 3, H, W]  (per-slot colored overlays)
        - 'semantic_segmentation' : [B,   3, H, W]  (argmax over slots; if no groups -> original)
        - 'combined_with_contours': [B,   3, H, W]  (colored edges + centers if available)
        """
        assert observation.ndim == 4, "observation must be [B,3,H,W]"
        assert slot_attention_maps.ndim == 4, "slot_attention_maps must be [B,K,H_att,W_att]"
        assert slot_centers.ndim == 3 and slot_centers.size(-1) == 2, "slot_centers must be [B,K,2]"

        B, C, H, W = observation.shape
        Bm, K, H_att, W_att = slot_attention_maps.shape
        Bc, Kc, _ = slot_centers.shape
        assert Bm == B and Bc == B and Kc == K, "Batch/slot mismatch among inputs"

        device = observation.device
        dtype  = observation.dtype

        # --- normalize image to [0,1] ---
        obs = observation
        if obs.min() < 0:
            obs = (obs + 1) * 0.5
        obs = obs.clamp(0, 1)

        # --- colors for slots (RGB in [0,1]) ---
        base_colors = torch.tensor([
            [1.0, 0.0, 0.0],  # R
            [0.0, 1.0, 0.0],  # G
            [0.0, 0.0, 1.0],  # B
            [1.0, 1.0, 0.0],  # Y
            [1.0, 0.0, 1.0],  # M
            [0.0, 1.0, 1.0],  # C
        ], device=device, dtype=dtype)
        if K > base_colors.size(0):
            # tile colors if more slots than palette
            reps = (K + base_colors.size(0) - 1) // base_colors.size(0)
            colors = base_colors.repeat(reps, 1)[:K]
        else:
            colors = base_colors[:K]
        # shape helpers
        colors_bk3 = colors.view(1, K, 3, 1, 1)  # [1,K,3,1,1]

        # --- upsample slot maps to image size & normalize per (B,K) ---
        # [B,K,H_att,W_att] -> [B,K,1,H_att,W_att] -> upsample -> [B,K,H,W]
        slot_maps_up = F.interpolate(
            slot_attention_maps.view(B * K, 1, H_att, W_att),
            size=(H, W),
            mode="bilinear",
            align_corners=False
        ).view(B, K, H, W)

        m_min = slot_maps_up.amin(dim=(2, 3), keepdim=True)
        m_max = slot_maps_up.amax(dim=(2, 3), keepdim=True)
        slot_maps_norm = (slot_maps_up - m_min) / (m_max - m_min + 1e-8)  # [B,K,H,W]

        out: Dict[str, torch.Tensor] = {}

        # ---------- 1) Per-slot overlays ----------
        # Build [B,K,3,H,W]: colored layer = map * color; overlay with original
        colored_layers = slot_maps_norm.unsqueeze(2) * colors_bk3              # [B,K,3,H,W]
        overlays_bk = (0.6 * obs.unsqueeze(1) + 0.4 * colored_layers).clamp(0, 1)  # [B,K,3,H,W]
        out["slot_overlays"] = overlays_bk.reshape(B * K, 3, H, W)             # [B*K,3,H,W]

        # ---------- 2) Semantic segmentation (argmax over K) ----------
        if group_assignments is not None:
            ga = group_assignments
            assert ga.dim() == 4 and ga.size(0) == B, "group_assignments must be [B,*,*,K] or [B,K,*,*]"
            # convert to [B,K,H_att,W_att]
            if ga.shape[1] == K:  # [B,K,H_att,W_att]
                g_bkhw = ga
            else:                 # [B,H_att,W_att,K]
                assert ga.shape[-1] == K, "group_assignments last dim must be K"
                g_bkhw = ga.permute(0, 3, 1, 2).contiguous()
            # upsample to image and argmax
            g_up = F.interpolate(g_bkhw.float(), size=(H, W), mode="nearest")  # [B,K,H,W]
            dom = g_up.argmax(dim=1)                                           # [B,H,W]
            # colorize via one-hot
            one_hot = F.one_hot(dom, num_classes=K).permute(0, 3, 1, 2).to(dtype)  # [B,K,H,W]
            seg = (one_hot.unsqueeze(2) * colors_bk3).sum(dim=1)                   # [B,3,H,W]
            out["semantic_segmentation"] = (0.5 * obs + 0.5 * seg).clamp(0, 1)
        else:
            out["semantic_segmentation"] = obs.clone()

        # ---------- 3) Combined with colored contours (+ centers) ----------
        # Start from original; draw 1-px edges for each slot in its color
        combined = obs.clone()
        # Laplacian kernel
        k = torch.tensor([[-1., -1., -1.],
                        [-1.,  8., -1.],
                        [-1., -1., -1.]], device=device, dtype=dtype).view(1, 1, 3, 3)
        thr = (slot_maps_up > 0.30).to(dtype)                                   # [B,K,H,W]
        # iterate slots to apply different colors
        for s in range(K):
            edges = F.conv2d(thr[:, s].unsqueeze(1), k, padding=1).squeeze(1)   # [B,H,W]
            edges = (edges > 0.1)                                                # bool
            col = colors[s].view(1, 3, 1, 1)                                     # [1,3,1,1]
            combined = torch.where(edges.unsqueeze(1), col, combined)            # color edges

        # draw centers if provided (convert from [-1,1] to pixels)
        if slot_centers is not None:
            xs = ((slot_centers[..., 0] + 1.0) * 0.5) * (W - 1)   # [B,K]
            ys = ((slot_centers[..., 1] + 1.0) * 0.5) * (H - 1)   # [B,K]
            for b in range(B):
                for s in range(K):
                    x = xs[b, s].round().long().clamp(0, W - 1)
                    y = ys[b, s].round().long().clamp(0, H - 1)
                    # small crosshair (3x3) in white, plus colored ring
                    combined[b, :, y, x] = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        yy = (y + dy).clamp(0, H - 1)
                        xx = (x + dx).clamp(0, W - 1)
                        combined[b, :, yy, xx] = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
                    # crude colored ring (optional)
                    for ry in range(-2, 3):
                        for rx in range(-2, 3):
                            if ry*ry + rx*rx in (4, 5):  # approximate circle
                                yy = (y + ry).clamp(0, H - 1)
                                xx = (x + rx).clamp(0, W - 1)
                                combined[b, :, yy, xx] = colors[s]

        out["combined_with_contours"] = combined.clamp(0, 1)

        # ---------- return selection ----------
        if return_mode == "all":
            return out
        return {return_mode: out[return_mode]}



class PerpetualOrthogonalProjectionLoss(nn.Module):
    def __init__(self, 
                 num_classes=10, 
                 feat_dim=2048, 
                 no_norm=False, 
                 use_attention=True,  
                 orthogonality_weight=0.3, 
                 slot_ortho_weight=0.2, 
                 device=None):
        
        super(PerpetualOrthogonalProjectionLoss, self).__init__()

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.no_norm = no_norm
        self.use_attention = use_attention
        self.orthogonality_weight = orthogonality_weight
        self.slot_ortho_weight = slot_ortho_weight
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #learnable class centres
        self.class_centres = nn.Parameter(torch.randn(self.num_classes, self.feat_dim, device=self.device))
        # Apply orthogonal initialization (significantly better than random)
        with torch.no_grad():
            nn.init.orthogonal_(self.class_centres)
            
        
        self.to(self.device)
            
    def orthogonal_center_loss(self, norm_centers):
        """Enforce class centers remain orthogonal (key addition)"""
        
        # Compute how far they deviate from orthogonal
        return torch.norm(torch.mm(norm_centers, norm_centers.t()) - torch.eye(self.num_classes, device=self.device), p='fro')**2  # Frobenius norm squared

    def slot_loss(self, features):
        """ Pushes slots from the same batch away from each other, preventing slot collapse"""
        batch_size = features.size(0) // self.num_classes
        features = features.view(batch_size, self.num_classes, -1)
        slots = F.normalize(features, dim=-1)
        # Compute share similarity matrix
        sim_matrix = torch.einsum('bnd,bmd->bnm', slots, slots)
        # Strict ortho loss
        
        return torch.norm(sim_matrix - torch.eye(self.num_classes, device=slots.device).unsqueeze(0), p='fro') / batch_size


    def forward(self, features, labels=None):
        device = features.device
        total_loss=0

        # 1. loss between slots
        total_loss += self.slot_ortho_weight * self.slot_loss(features)

        # 2. Orthogonal center regularization 
        normalized_class_centres = F.normalize(self.class_centres, p=2, dim=1)
        total_loss += self.orthogonality_weight * self.orthogonal_center_loss(normalized_class_centres)

        # 3. Feature-to center alignment 
        if self.use_attention:
            features_weights = torch.matmul(features, features.T)
            features_weights = F.gumbel_softmax(features_weights, dim=1)
            features = torch.matmul(features_weights, features)

        #  features are normalized
        if not self.no_norm:
            features = F.normalize(features, p=2, dim=1)
        

        #create mask for class assignment
        
        labels = labels[:, None]  # extend dim
        class_range = torch.arange(self.num_classes, device=device).long()
        class_range = class_range[:, None]  # extend dim
        label_mask = torch.eq(labels, class_range.t()).float().to(device)
        #calculate the feature centre loss
        feature_centre_variance = torch.matmul(features, normalized_class_centres.t())
        same_class_loss = (label_mask * feature_centre_variance).sum() / (label_mask.sum() + 1e-6)
        diff_class_loss = torch.relu(0.2 + (1 - label_mask) * feature_centre_variance).mean()

        total_loss += 0.5 * (1.0 - same_class_loss) + diff_class_loss
        

        return total_loss
    
class ConvGRUCell(nn.Module):

    
    def __init__(self,input_size,hidden_size,kernel_size,cuda_flag):
        super(ConvGRUCell,self).__init__()
        self.input_size  = input_size
        self.cuda_flag   = cuda_flag
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.ConvGates   = nn.Conv2d(self.input_size + self.hidden_size,2 * self.hidden_size,kernel_size,padding=self.kernel_size//2)
        self.Conv_ct     = nn.Conv2d(self.input_size + self.hidden_size,self.hidden_size,kernel_size,padding=self.kernel_size//2) 
        dtype            = torch.FloatTensor
        self.norm_gates = nn.GroupNorm(2, 2 * hidden_size)
        self.norm_candidate = nn.GroupNorm(8, hidden_size)

        self.reset_parameters()
    
    def reset_parameters(self):
        # Xavier initialization for gates
        nn.init.xavier_uniform_(self.ConvGates.weight, gain=0.5)
        nn.init.xavier_uniform_(self.Conv_ct.weight, gain=0.5)

        # Initialize biases to favor forgetting initially (stability)
        nn.init.constant_(self.ConvGates.bias, 0.0)
        nn.init.constant_(self.Conv_ct.bias, 0.0)

    def forward(self,input: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hidden is None:
           size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
           if self.cuda_flag  == True:
              hidden    = torch.autograd.Variable(torch.zeros(size_h)).cuda() 
           else:
              hidden    = torch.autograd.Variable(torch.zeros(size_h))
        c1           = self.norm_gates(self.ConvGates(torch.cat((input,hidden),1)))
        (rt,ut)      = c1.chunk(2, 1)
        reset_gate   = F.sigmoid(rt)
        update_gate  = F.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate,hidden)
        p1           = self.norm_candidate(self.Conv_ct(torch.cat((input,gated_hidden),1)))
        ct           = F.tanh(p1)
        next_h       = torch.mul(update_gate,hidden) + (1-update_gate)*ct
        return next_h
 
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
        use_relative_position_bias: bool = True  # Whether to use relative position bias
    ):
        super().__init__()
        self.attention_resolution = attention_resolution
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.spatial_dim = attention_resolution * attention_resolution
        self.use_relative_position_bias = use_relative_position_bias
        
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
            relative_coords[:, :, 0] += attention_resolution - 1  # Shift to start from 0
            relative_coords[:, :, 1] += attention_resolution - 1
            
            self.register_buffer("relative_position_index", relative_coords)
        
        # Efficient self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Cross-attention with context
        self.context_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
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
            nn.Dropout(0.1),
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
        context = torch.cat([hidden_state, latent_state], dim=-1)
        context_features = self.context_projection(context)  # [B, feature_dim]
        context_seq = context_features.unsqueeze(1)  # [B, 1, feature_dim]
        
        # 7. Self-attention with relative position bias (GLOBAL attention)
        normed_seq = self.norm1(combined_seq)
        
        self_attn_out, _ = self.self_attention(
            normed_seq, normed_seq, normed_seq
        )
        
        combined_seq = combined_seq + self_attn_out
        
        # 8. Cross-attention with context
        normed_seq = self.norm2(combined_seq)
        context_expanded = context_seq.expand(-1, self.spatial_dim, -1)
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

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

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
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
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
                nn.LeakyReLU(0.2, True))
        )
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layer=nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
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
                nn.LeakyReLU(0.2, True)
            )
        )
        self.layers.append(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))

        # output 1 channel prediction map
        
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

###################################################################
############ Image Temporal Discriminator Modules #################
###################################################################

class CausalSelfAttention(nn.Module):
    """
    Causal self-attention with proper masking for temporal discrimination
    """
    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1, max_sequence_length=128):
        super().__init__()
        assert n_embd % n_head == 0
        
        # Key, query, value projections for all heads
        self.kqv = nn.Linear(n_embd, 3* n_embd)

        
        # Output projection
        self.proj = nn.Linear(n_embd, n_embd)
        
        # Regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        
        # Causal mask to ensure attention only attends to the left
        self.max_sequence_length = max_sequence_length
        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer('causal_mask', 
                                torch.tril(torch.ones(max_sequence_length, max_sequence_length))
                                .view(1, 1, max_sequence_length, max_sequence_length))
        
    def forward(self, x, padding_mask=None):
        B, T, C = x.size()  # batch, sequence length, embedding dim
        kqv = self.kqv(x)  # (B, T, 3 * C)
        k, q, v = kqv.split(C, dim=2)  # each is (B, T, C)
        # Calculate query, key, values for all heads in batch
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        causal_mask = self.causal_mask[:, :, :T, :T]  # [1, 1, T, T]
        att = att.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply padding mask if provided
        if padding_mask is not None:
            # padding_mask: [B, T] -> [B, 1, 1, T]
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            att = att.masked_fill(padding_mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        
        # Re-assemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        return self.resid_drop(self.proj(y))

class TransformerBlock(nn.Module):
    """Transformer block with causal attention"""
    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1,
                 sequence_length=128):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop,
                                        sequence_length)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = nn.GELU(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), padding_mask=mask)
        m = self.mlp
        x = x + m.dropout(m.c_proj(m.act(m.c_fc(self.ln_2(x)))))
        return x
    
class SpatialTemporalTokenizer(nn.Module):
    """
    Tokenizes video into spatial-temporal patches while preserving structure
    """
    def __init__(self, input_channels, hidden_dim, patch_size=(4, 4), 
                 temporal_stride=1):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_stride = temporal_stride
        
        # Patch embedding using Conv3D
        self.patch_embed = nn.Conv3d(
            input_channels,
            hidden_dim,
            kernel_size=(temporal_stride, *patch_size),
            stride=(temporal_stride, *patch_size)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W]
        Returns:
            tokens: [B, T', N, D] where N = (H/p)*(W/p) spatial patches
            spatial_shape: (H', W') for reshaping
        """
        B, T, C, H, W = x.shape
        
        # Reshape for Conv3D: [B, C, T, H, W]
        x = x.transpose(1, 2)
        
        # Extract patches
        x = self.patch_embed(x)  # [B, D, T', H', W']
        _, D, T_new, H_new, W_new = x.shape
        
        # Reshape to [B, T', H'*W', D]
        x = x.permute(0, 2, 3, 4, 1)  # [B, T', H', W', D]
        x = x.reshape(B, T_new, H_new * W_new, D)
        
        x = self.norm(x)
        
        return x, (H_new, W_new)


class TemporalDiscriminator(nn.Module):
    """
    Temporal Discriminator with:
    - Variable-length sequence support
    - Preserved spatial structure
    - Spatial-temporal token processing
    """
    def __init__(
        self,
        input_channels: int = 3,
        image_size: int = 64,
        hidden_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        max_sequence_length: int = 32,  # Maximum, not fixed
        patch_size: int = 8,  # Spatial patch size
        device: torch.device = torch.device('cuda')
    ):
        super().__init__()
        self.image_size = image_size
        self.max_sequence_length = max_sequence_length
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.device = device
        
        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2
        
        # Spatial-Temporal Tokenizer (replaces frame_encoder)
        self.frame_encoder = SpatialTemporalTokenizer(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            patch_size=(patch_size, patch_size),
            temporal_stride=1
        )
        
        # Learnable spatial position embeddings
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, 1, self.num_patches, hidden_dim) * 0.02
        )
        
        # Learnable temporal position embeddings (up to max_sequence_length)
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, max_sequence_length, 1, hidden_dim) * 0.02
        )
        
        # Spatial attention (preserves spatial structure)
        self.spatial_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_heads // 2,  # Fewer heads for spatial
                dropout=0.1,
                batch_first=True
            ) for _ in range(n_layers // 2)
        ])
        
        # Temporal transformer blocks (with causal masking)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim, n_heads,
                sequence_length=max_sequence_length * self.num_patches
            ) for _ in range(n_layers)
        ])
        
        # Layer norms
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.ln_f = nn.LayerNorm(hidden_dim)
        
        # Discrimination heads
        self.temporal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.spatial_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.per_frame_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        self.to(device)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def create_padding_mask(self, sequence_lengths, max_length):
        """
        Create padding mask for variable-length sequences
        
        Args:
            sequence_lengths: [B] tensor of actual lengths
            max_length: int, maximum sequence length in batch
        
        Returns:
            mask: [B, max_length] where True = valid, False = padding
        """
        batch_size = len(sequence_lengths)
        mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=self.device)
        
        for i, length in enumerate(sequence_lengths):
            mask[i, :length] = True
        
        return mask
    
    def forward(self, x, sequence_lengths=None, return_features=False):
        """
        Forward pass with variable-length support
        
        Args:
            x: [batch_size, seq_len, channels, height, width]
            sequence_lengths: [batch_size] tensor of actual sequence lengths
            return_features: If True, return intermediate features
        
        Returns:
            dict with discrimination scores
        """
        B, T, C, H, W = x.shape
        
        # Handle variable lengths
        if sequence_lengths is None:
            sequence_lengths = torch.full((B,), T, device=self.device)
        
        # Create padding mask for temporal dimension
        temporal_mask = self.create_padding_mask(sequence_lengths, T)
        
        # 1. Extract spatial-temporal tokens
        tokens, (H_patches, W_patches) = self.frame_encoder(x)
        # tokens: [B, T, N, D] where N = num_patches
        
        # 2. Add positional embeddings
        # Spatial position embedding (same for all timesteps)
        tokens = tokens + self.spatial_pos_embed
        
        # Temporal position embedding (different for each timestep)
        temporal_pos = self.temporal_pos_embed[:, :T, :, :]
        tokens = tokens + temporal_pos
        
        # 3. Process spatial relationships within each frame
        spatial_features = []
        for t in range(T):
            # Process spatial attention for each timestep
            spatial_tokens = tokens[:, t]  # [B, N, D]
            
            for spatial_attn in self.spatial_attention:
                spatial_tokens_norm = self.spatial_norm(spatial_tokens)
                attn_out, _ = spatial_attn(
                    spatial_tokens_norm, 
                    spatial_tokens_norm, 
                    spatial_tokens_norm
                )
                spatial_tokens = spatial_tokens + attn_out
            
            spatial_features.append(spatial_tokens)
        
        # Stack back to [B, T, N, D]
        spatial_features = torch.stack(spatial_features, dim=1)
        
        # 4. Reshape for temporal processing
        # Combine spatial and temporal dimensions
        B, T, N, D = spatial_features.shape
        features_flat = spatial_features.reshape(B, T * N, D)
        
        # Create combined mask for flattened sequence
        # Each spatial token inherits the mask of its timestep
        combined_mask = temporal_mask.unsqueeze(2).expand(-1, -1, N)
        combined_mask = combined_mask.reshape(B, T * N)
        
        # 5. Process through temporal transformer
        hidden = features_flat
        all_hidden_states = []
        
        for block in self.blocks:
            hidden = block(hidden, combined_mask)
            if return_features:
                all_hidden_states.append(hidden)
        
        hidden = self.ln_f(hidden)
        
        # 6. Reshape back to [B, T, N, D]
        hidden = hidden.reshape(B, T, N, D)
        
        # 7. Compute discrimination scores
        
        # Temporal score: aggregate across space, use last valid timestep
        temporal_features = hidden.mean(dim=2)  # [B, T, D]
        last_timesteps = sequence_lengths - 1
        temporal_rep = temporal_features[torch.arange(B), last_timesteps]
        temporal_score = self.temporal_head(temporal_rep)
        
        # Spatial score: aggregate across time for each spatial location
        spatial_features = hidden.mean(dim=1)  # [B, N, D]
        spatial_score = self.spatial_head(spatial_features.mean(dim=1))
        
        # Per-frame scores
        per_frame_features = hidden.mean(dim=2)  # [B, T, D]
        per_frame_scores = self.per_frame_head(per_frame_features)
        
        # Mask out invalid frames
        per_frame_scores = per_frame_scores.masked_fill(
            ~temporal_mask.unsqueeze(-1), 0.0
        )
        
        # Compute valid frame average
        valid_frames = temporal_mask.sum(dim=1, keepdim=True).clamp(min=1)
        per_frame_avg = per_frame_scores.sum(dim=1) / valid_frames
        
        outputs = {
            'temporal_score': temporal_score,  # [B, 1]
            'spatial_score': spatial_score,    # [B, 1]
            'per_frame_scores': per_frame_scores,  # [B, T, 1]
            'final_score': temporal_score + spatial_score + per_frame_avg,
            'sequence_lengths': sequence_lengths,
            'spatial_shape': (H_patches, W_patches)
        }
        
        if return_features:
            outputs['features'] = all_hidden_states
            outputs['spatial_features'] = spatial_features
            outputs['hidden_3d'] = hidden  # [B, T, N, D] for visualization
        
        return outputs
    
    def extract_features(self, x):
        """
        Backward compatibility - extract features with spatial structure preserved
        """
        B, T, C, H, W = x.shape
        tokens, spatial_shape = self.frame_encoder(x)
        
        # Add positional embeddings
        tokens = tokens + self.spatial_pos_embed
        temporal_pos = self.temporal_pos_embed[:, :T, :, :]
        tokens = tokens + temporal_pos
        
        return tokens, spatial_shape

##########################################################################################
# The following lines define the Variational Autoencoder (VAE) encoder and decoder for image data.
##########################################################################################

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
            
            layer += 1
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
            layer += 1
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
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        else:
            decay = self.decay
            
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters for evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
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

##### Self modeling Blocks #####

class SelfModelBlock(nn.Module):
    def __init__(self, z_dim, A_dim, a_dim, c_dim, h_dim, d=256, nhead=4, dropout=0.2):
        super().__init__()
        
        # Project inputs to common dimension
        self.input_projections = nn.ModuleDict({
            'z': nn.Linear(z_dim, d),
            'A': nn.Linear(A_dim, d),
            'a': nn.Linear(a_dim, d),
            'c': nn.Linear(c_dim, d),
            'h': nn.Linear(h_dim, d)
        })
        self.time_proj = nn.Sequential(
                                       nn.Linear(1, d//2), 
                                       nn.LayerNorm(d//2), 
                                       nn.SiLU(), 
                                       nn.Linear(d//2, d)
                                       )
        # Learnable query tokens for prediction
        self.query_h = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.query_A = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        
        # Type embeddings
        self.type_embed = nn.Embedding(6, d)
        
        # EXPLICIT Cross-attention: queries attend to context
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=nhead,
            batch_first=True,
            dropout=dropout
        )
        
        # Self-attention for context processing
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=nhead,
            batch_first=True,
            dropout=dropout
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        
        self.activation = nn.Softplus()
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.LayerNorm(d * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 4, d)
        )
        
        # Output heads
        self.head_h = nn.Sequential(
            nn.Linear(d, d//2),
            nn.LayerNorm(d//2),
            nn.SiLU(),
            nn.Linear(d//2, 2*h_dim)
        )
        self.head_A = nn.Sequential(
            nn.Linear(d, d//2),
            nn.LayerNorm(d//2),
            nn.SiLU(),
            nn.Linear(d//2, 2*A_dim)
        )
    
    def forward(self, z_t, A_t, a_t, c_t, h_t, current_time=None):
        batch_size = z_t.shape[0]
        if current_time is not None:
            time_token = self.time_proj(current_time.view(-1, 1))
        else:
            time_token = self.time_proj(torch.zeros(batch_size, 1, device=z_t.device))
        # 1. Project and create context tokens
        context_tokens = torch.stack([
            self.input_projections['z'](z_t),
            self.input_projections['A'](A_t),
            self.input_projections['a'](a_t),
            self.input_projections['c'](c_t),
            self.input_projections['h'](h_t),
            time_token
        ], dim=1)  # [B,  6, d]



        # Add type embeddings
        type_ids = torch.arange(6, device=z_t.device).unsqueeze(0).expand(batch_size, -1)  # [B, 6]
        context_tokens = context_tokens + self.type_embed(type_ids)
        
        # 2. Process context with self-attention
        context_normed = self.norm1(context_tokens)
        context_attended, _ = self.self_attention(
            context_normed, context_normed, context_normed
        )
        context_tokens = context_tokens + context_attended
        
        # 3. Create query tokens
        queries = torch.cat([
            self.query_h.expand(batch_size, -1, -1),
            self.query_A.expand(batch_size, -1, -1)
        ], dim=1)  # [B, 2, d]
        
        # 4. Cross-attention: Queries attend to processed context
        # Q: What we want to predict (queries)
        # K, V: Information available (context)
        queries_normed = self.norm2(queries)
        context_normed = self.norm2(context_tokens)
        
        attended_queries, attention_weights = self.cross_attention(
            query=queries_normed,    # Queries ask about future
            key=context_normed,      # Context provides information
            value=context_normed,    # Context provides values
            need_weights=True
        )
        queries = queries + attended_queries
        
        # 5. FFN on queries
        queries = queries + self.ffn(self.norm3(queries))
        
        # 6. Extract predictions
        pred_h = self.head_h(queries[:, 0])  # First query predicts h
        pred_A = self.head_A(queries[:, 1])  # Second query predicts A
        
        # Split into mean and log variance
        h_mean, h_logvar = pred_h.chunk(2, dim=-1)
        A_mean, A_logvar = pred_A.chunk(2, dim=-1)

        return h_mean, self.bounded_logvar(h_logvar), A_mean, self.bounded_logvar(A_logvar), attention_weights

    def bounded_logvar(self, logvar):
        """Range: [min_val, max_val]"""
        min_val = -8.0
        max_val = 2.0
        
        # Softplus for positivity, then scale and shift
        positive = self.activation(logvar)
        return positive.clamp(min=min_val, max=max_val)
"""
AttentionPosterior: Bottom-up Multi-Object Attention Mechanism

This module implements a sophisticated bottom-up attention mechanism that processes visual
observations to extract multi-object attention maps and spatial coordinates. The architecture
combines Feature Pyramid Networks (FPN), slot-based attention, and spatial reasoning to handle
multiple objects in a scene efficiently.

Key Components:
- Feature Pyramid Network: Multi-scale feature extraction (84 -> 42 -> 21)
- Slot Attention: GroupViT-style semantic grouping with iterative routing
- Top-down Modulation: Context-driven attention refinement
- Multi-Head Fusion: Weighted, max, or gated fusion of per-slot attention maps
- Spatial Regularization: Coordinate regularization and noise injection for robustness

The module is designed to be linear in complexity (N * K) making it suitable for video pipelines
and real-time applications.

Returns:
    attention_probs_2d: [B, H_att, W_att] fused attention map (probabilities)
    regularized_coords: [B, 2] combined (x,y) in [-1,1] after regularization

Side Attributes (accessible after forward pass):
    slot_attention_maps: [B, K, H_att, W_att] per-slot attention maps (probabilities)
    slot_centers: [B, K, 2] per-slot (x,y) coordinates
    fusion_weights: [B, K] fusion weights (when attention_fusion_mode='weighted')
    diversity_loss: scalar diversity loss (0.0 if disabled)
    ortho_loss: scalar orthogonality loss (0.0 if disabled)
    bottleneck_features: [B, hidden_dim//2] compressed features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .slot_attention import SlotAttention


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
        expected_fused: bool = False,  # whether to expect fused attention maps (for visualization)
        use_checkpoint: bool = True,
        ckpt_preserve_rng: bool = True,
        ckpt_use_reentrant: bool = False, # PyTorch 2.x recommended
        dropout_p: float = 0.2, # dropout on slot_features & bottleneck
        coord_noise_std: float = 0.0, # Gaussian noise added to coords before regularizer
    ):
        super().__init__()
        assert feature_channels % num_heads == 0, "feature_channels must be divisible by num_heads"
        assert attention_fusion_mode in {"weighted", "max", "gated"}

        self.image_size = image_size
        self.attention_resolution = attention_resolution
        self.feature_channels = feature_channels
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_semantic_slots = num_semantic_slots
        self.num_heads = num_heads
        self.attention_fusion_mode = attention_fusion_mode
        self.enforce_diversity = enforce_diversity
        self.expected_fused = expected_fused
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.use_checkpoint = use_checkpoint
        self.ckpt_preserve_rng = ckpt_preserve_rng
        self.ckpt_use_reentrant = ckpt_use_reentrant
        self.dropout_p = float(dropout_p)
        self.coord_noise_std = float(coord_noise_std)
        self.N = attention_resolution * attention_resolution
        self.d = feature_channels
        self.h = num_heads
        self.d_head = self.d // self.h

        # ========= Feature Pyramid (84 -> 42 -> 21) =========
        if not self.expected_fused:
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
        else:
            self.pyramid_conv1 = None
            self.pyramid_conv2 = None
            self.pyramid_conv3 = None
            self.pyramid_fusion = None

        # ========= GroupViT-style grouping =========

        # Slot prototypes (semantic specialization)
        self.slot_attn = SlotAttention(
            num_slots=num_semantic_slots,
            in_dim=self.d,
            slot_dim=self.d,
            iters=3,
            mlp_hidden=2 * self.d,
        )


        # ========= Top-down modulation (per slot) =========
        self.slot_modulator = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, self.d * num_semantic_slots),
            nn.LayerNorm(self.d * num_semantic_slots),
            nn.SiLU(),
            nn.Linear(self.d * num_semantic_slots, self.d * num_semantic_slots),
        )

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
        else:
            self.fusion_weights_head = None
            self.fusion_gate = None

        # ========= Per-slot temperatures for attention sharpening =========
        self.slot_temperatures = nn.Parameter(torch.ones(num_semantic_slots))

        # ========= Coordinate regularizer (combine K centers -> 2) =========
        self.spatial_regularizer = nn.Sequential(
            nn.Linear(2 * num_semantic_slots, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh(),
        )

        # ========= Bottleneck from concatenated slot features =========
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
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


    # ---------- diversity losses ----------
    @staticmethod
    def _compute_diversity_losses(slot_maps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        slot_maps: [B, K, H, W] (probabilities or logits). We convert to probabilities then L2-normalize.
        Returns (diversity_loss, ortho_loss). The second term is a placeholder (kept for API);
        feel free to attach your slot prototype orthogonality if you want to keep it.
        """
        B, K, H, W = slot_maps.shape
        p = slot_maps.view(B, K, -1).softmax(dim=-1)
        p = F.normalize(p, p=2, dim=-1)
        corr = torch.bmm(p, p.transpose(1, 2))  # [B, K, K]
        I = torch.eye(K, device=corr.device).unsqueeze(0)
        div = ((corr - I) ** 2).sum() / (B * K * (K - 1) + 1e-8)
        ortho = torch.zeros((), device=slot_maps.device)
        return div, ortho

    def _maybe_ckpt(self, fn, *args):
        if not self.use_checkpoint:
            return fn(*args)
        return torch.utils.checkpoint.checkpoint(fn, *args, preserve_rng_state=self.ckpt_preserve_rng, use_reentrant=self.ckpt_use_reentrant)

    # Basic FPN to fused feature map [B, C, H_att, W_att]
    def _fpn_forward(self, observation: torch.Tensor, H_att: int, W_att: int) -> torch.Tensor:
        if self.expected_fused:
            raise RuntimeError("_fpn_forward should not be called when expected_fused=True")


        def _fn(x: torch.Tensor) -> torch.Tensor:
            l1 = self.pyramid_conv1(x)
            l2 = self.pyramid_conv2(l1)
            l3 = self.pyramid_conv3(l2)
            l1_up = F.interpolate(l1, size=(H_att, W_att), mode="bilinear", align_corners=False)
            l2_up = F.interpolate(l2, size=(H_att, W_att), mode="bilinear", align_corners=False)
            l3_up = F.interpolate(l3, size=(H_att, W_att), mode="bilinear", align_corners=False)
            fused = self.pyramid_fusion(torch.cat([l1_up, l2_up, l3_up], dim=1))
            return fused


        return self._maybe_ckpt(_fn, observation)


    def _slot_forward(self, feat_seq: torch.Tensor, modulation: torch.Tensor):
        def _fn(x: torch.Tensor, seed: torch.Tensor):
            slots, attn = self.slot_attn(x, seed_slots=seed)
            return slots, attn
        return self._maybe_ckpt(_fn, feat_seq, modulation)


    def _fusion_forward(self, slot_maps: torch.Tensor, slot_features: torch.Tensor, top_down: torch.Tensor):
        B = slot_maps.size(0)
        K = slot_maps.size(1)

        def _fn(maps: torch.Tensor, feats: torch.Tensor, td: torch.Tensor):
            if self.attention_fusion_mode == "weighted":
                all_feat = feats.reshape(B, -1)
                fusion_weights = self.fusion_weights_head(all_feat) # [B,K]
                w = fusion_weights.view(B, K, 1, 1)
                fused_map = (maps * w).sum(dim=1)
                return fused_map, fusion_weights
            elif self.attention_fusion_mode == "max":
                fused_map, _ = maps.max(dim=1)
                return fused_map, torch.empty(B, K, device=maps.device)
            else:  # gated
                gates = self.fusion_gate(td).view(B, K, 1, 1)
                fused_map = (maps * gates).sum(dim=1)
                return fused_map, gates.squeeze(-1).squeeze(-1)


        return self._maybe_ckpt(_fn, slot_maps, slot_features, top_down)


    def _centers_forward(self, slot_maps: torch.Tensor, fused_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, H, W = slot_maps.shape
        device = slot_maps.device
        # grids
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij") # [H,W]
        x_vec = x_grid.reshape(1, 1, H * W)
        y_vec = y_grid.reshape(1, 1, H * W)


        def _fn(maps: torch.Tensor, fused: torch.Tensor):
            # Per-slot probabilities [B,K,N]
            p = maps.view(B, K, -1).softmax(dim=-1)
            # Centers per slot
            x_c = (p * x_vec).sum(dim=-1)
            y_c = (p * y_vec).sum(dim=-1)
            slot_centers = torch.stack([x_c, y_c], dim=-1) # [B,K,2]


            # Fused-map expectation for raw coords
            attn_p = fused.flatten(1).softmax(dim=-1) # [B,N]
            x_r = (attn_p * x_vec.view(1, H * W)).sum(dim=-1)
            y_r = (attn_p * y_vec.view(1, H * W)).sum(dim=-1)
            coords_raw = torch.stack([x_r, y_r], dim=-1) # [B,2]
            return slot_centers, coords_raw, p


        return self._maybe_ckpt(_fn, slot_maps, fused_map)
    def forward(
        self,
        observation: torch.Tensor,  # [B, 3, 84, 84]
        hidden_state: torch.Tensor, # [B, hidden_dim]
        context: torch.Tensor,      # [B, context_dim]
        fused_feat: Optional[torch.Tensor] = None, # [B, feature_channels, H_att, W_att] (if expected_fused)
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B = observation.size(0)
        H_att = W_att = self.attention_resolution
        device = observation.device

        # === FPN ===
        if self.expected_fused and (fused_feat is not None):
            fused = F.interpolate(fused_feat, size=(H_att, W_att), mode="bilinear", align_corners=False)
        else:
            fused = self._fpn_forward(observation, H_att, W_att)
        self._last_saliency_features = fused  # Save for visualization

        # === sequence + positions ===
        feat_seq = fused.flatten(2).transpose(1, 2)                # [B, N, d]
        pos = self._positional_encoding_2d().to(device)            # [1, N, d]
        feat_seq = feat_seq + pos                                  # [B, N, d]

        # === per-slot queries (modulated) ===
        top_down = torch.cat([hidden_state, context], dim=-1)      # [B, hidden+context]
        modulation = self.slot_modulator(top_down).view(B, self.num_semantic_slots, self.d)  # [B,K,d]

        # slots = specialization + learnable bias (optional)
        slots, attn = self._slot_forward(feat_seq, modulation)    # slots [B,K,d], attn [B,K,N]

        # Save per-slot attention maps (reshape responsibilities)
        slot_maps = attn.view(B, self.num_semantic_slots, H_att, W_att)       # [B,K,H,W]
        slot_features = self.dropout(slots)                                               # [B,K,d]
        self.slot_attention_maps = slot_maps
        self.slot_features = slot_features

        # For API/back-compat visualization: store group assignments as [B,H,W,K]
        self.group_assignments = attn.transpose(1, 2).view(B, H_att, W_att, self.num_semantic_slots)


        # === fusion ===
        fused_map, fusion_weights = self._fusion_forward(slot_maps, slot_features, top_down)
        self.fusion_weights = fusion_weights if self.attention_fusion_mode != "max" else None


        # Normalize fused map to probabilities
        attention_probs = fused_map.flatten(1).softmax(dim=-1) # [B,N]
        attention_probs_2d = attention_probs.view(B, H_att, W_att) # [B,H,W]


        # === per-slot centers & raw coords ===
        slot_centers, coords_raw, _p = self._centers_forward(slot_maps, fused_map)
        self.slot_centers = slot_centers
        self.coords_raw = coords_raw

        # Light reg to keep inside [-1,1] and stabilize; also allows fusing K centers
        regularized_coords = self.spatial_regularizer(slot_centers.flatten(1)) * 0.995 # [B,2]
        self.regularized_coords = regularized_coords

        # === diversity losses (optional) ===
        if self.enforce_diversity and self.training:
            div, ortho = self._compute_diversity_losses(slot_maps)
            self.diversity_loss = div
            self.ortho_loss = ortho
        else:
            self.diversity_loss = torch.tensor(0.0, device=device)
            self.ortho_loss = torch.tensor(0.0, device=device)


        # === bottleneck features ===
        self.bottleneck_features = self.bottleneck_projector(self.dropout(slot_features.reshape(B, -1)))


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

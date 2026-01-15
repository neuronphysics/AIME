from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

from VRNN.update import FlowHead, SepConvGRU


# helpers
def topk_gating(logits: torch.Tensor, top_k: int = 2, temperature: float = 1.0):
    """
    logits: [B,K,H,W]
    returns topk_idx [B,top_k,H,W], topk_w [B,top_k,H,W], probs [B,K,H,W]
    """
    K = logits.shape[1]
    top_k = min(top_k, K)
    eps= torch.finfo(torch.float32).eps
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=1)
    topk_w, topk_idx = torch.topk(probs, k=top_k, dim=1)
    topk_w = topk_w / (topk_w.sum(dim=1, keepdim=True) + eps)
    return topk_idx, topk_w, probs


def tv_loss_gate(p: torch.Tensor) -> torch.Tensor:
    """Total-variation loss on gate probabilities p: [B,K,H,W]."""
    dx = (p[:, :, :, 1:] - p[:, :, :, :-1]).abs().mean()
    dy = (p[:, :, 1:, :] - p[:, :, :-1, :]).abs().mean()
    return dx + dy


def _coords_grid(B: int, H: int, W: int, device, dtype) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    coords = torch.stack([x, y], dim=0).to(dtype)  # [2,H,W]
    return coords.unsqueeze(0).repeat(B, 1, 1, 1)  # [B,2,H,W]


def convex_upsample_flow(flow: torch.Tensor, mask: torch.Tensor, up: int) -> torch.Tensor:
    """
    RAFT convex upsampling.
    flow: [N,2,h,w] low-res pixels
    mask:[N, up^2*9, h, w]
    returns [N,2, up*h, up*w] full-res pixels
    """
    N, _, h, w = flow.shape
    mask = mask.view(N, 1, 9, up, up, h, w)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(up * flow, [3, 3], padding=1)  # [N,2*9,h*w]
    up_flow = up_flow.view(N, 2, 9, 1, 1, h, w)

    up_flow = torch.sum(mask * up_flow, dim=2)  # [N,2,up,up,h,w]
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3).contiguous()
    up_flow = up_flow.view(N, 2, up * h, up * w)
    return up_flow


class ConfHead(nn.Module):
    """Predicts per-pixel confidence (logit) for one expert."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, gn_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=min(gn_groups, hidden_dim), num_channels=hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gn1(self.conv1(x)), inplace=True)
        return self.conv2(x)

def GN(c, g=8):
    return nn.GroupNorm(num_groups=min(g, c), num_channels=c)

def sobel_dxdy(x: torch.Tensor):
    """
    x: (B, C, H, W) float
    returns dx, dy: (B, C, H, W)
    """
    B, C, H, W = x.shape
    wx = x.new_tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]]).view(1, 1, 3, 3)
    wy = x.new_tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]]).view(1, 1, 3, 3)

    # apply same kernel to each channel using groups=C
    wx = wx.expand(C, 1, 3, 3).contiguous()
    wy = wy.expand(C, 1, 3, 3).contiguous()

    x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
    dx = F.conv2d(x_pad, wx, groups=C)
    dy = F.conv2d(x_pad, wy, groups=C)
    return dx, dy
    
# Final Warp Class
class PyramidMoEWarp(nn.Module):
    """
    Coarse->fine MoE flow refinement + warp-then-blend.

    Defaults:
      - top_k=2 (sparse)
      - confidence enabled
      - reg weights: 1/1024 for TV and disagreement
      - returns a warp pyramid (6 levels) via e_warp_by_factor
    """
    def __init__(
        self,
        flow_in_dim: int,
        hidden_dim: int = 48,
        K: int = 4,
        code_dim: int = 8,
        up_factor: int = 4,

        # refinement pyramid: factors relative to full-res (H/f, W/f)
        pyramid_factors: Sequence[int] = (32, 16, 8, 4),
        iters_per_level: Sequence[int] = (1, 1, 2, 2),

        # output pyramid (6 levels like you asked)
        out_factors: Sequence[int] = (1, 2, 4, 8, 16, 32),

        # routing
        top_k: int = 2,
        gate_temperature: float = 0.3,

        # regularizers
        lb_weight: float = 2e-2,
        tv_gate_weight: float = 1.0 / 1024.0,
        disagree_weight: float = 1.0 / 1024.0,

        # confidence
        use_confidence: bool = True,
        conf_floor: float = 0.75,   # prevents "dead" experts from making weights explode
        conf_ceil: float = 0.99,
        use_checkpoint: bool = True,
        checkpoint_use_reentrant: bool = False,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.K = int(K)
        self.code_dim = int(code_dim)
        self.up_factor = int(up_factor)
        self.use_checkpoint = bool(use_checkpoint)
        self.checkpoint_use_reentrant = bool(checkpoint_use_reentrant)


        self.pyramid_factors = tuple(int(x) for x in pyramid_factors)
        self.iters_per_level = tuple(int(x) for x in iters_per_level)
        self.out_factors = tuple(int(x) for x in out_factors)

        assert len(self.pyramid_factors) == len(self.iters_per_level)
        assert self.pyramid_factors[-1] == self.up_factor, "pyramid_factors[-1] must equal up_factor"
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        self.device = device
        self.top_k = int(top_k)
        self.gate_temperature = float(gate_temperature)

        self.lb_weight = float(lb_weight)
        self.tv_gate_weight = float(tv_gate_weight)
        self.disagree_weight = float(disagree_weight)

        self.use_confidence = bool(use_confidence)
        self.conf_floor = float(conf_floor)
        self.conf_ceil = float(conf_ceil)

        # shared context encoder
        self.cnet = nn.Sequential(
            nn.Conv2d(flow_in_dim, 2 * hidden_dim, 3, padding=1),
            GN(2 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * hidden_dim, 2 * hidden_dim, 3, padding=1),
            GN(2 * hidden_dim),
        )

        self.router = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, 64, 3, padding=1),
            GN(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, K, 1),
        )

        self.flow_enc = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 7, padding=3),
            GN(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            GN(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # shared GRU
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=2 * hidden_dim)
        self._eps = torch.nn.Parameter(torch.tensor(torch.finfo(torch.float32).eps), requires_grad=False)

        # symmetry breakers
        self.expert_code = nn.Parameter(torch.randn(K, code_dim))
        self.level_code = nn.Parameter(torch.randn(len(self.pyramid_factors), code_dim))

        # expert heads
        head_in = 2 * hidden_dim + 2 * code_dim
        self.delta_heads = nn.ModuleList(
            [FlowHead(input_dim=head_in, hidden_dim=max(128, 2 * hidden_dim)) for _ in range(K)]
        )

        self.conf_heads = nn.ModuleList(
            [ConfHead(input_dim=head_in, hidden_dim=max(128, 2 * hidden_dim)) for _ in range(K)]
        )

        # convex upsample mask
        self.mask_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, (up_factor ** 2) * 9, 1),
        )

        # stable init
        for head in self.delta_heads:
            nn.init.zeros_(head.conv2.weight)
            nn.init.zeros_(head.conv2.bias)
        nn.init.zeros_(self.mask_head[-1].weight)
        nn.init.zeros_(self.mask_head[-1].bias)

        self._grid_cache: Dict[Tuple[torch.device, torch.dtype, int, int], torch.Tensor] = {}
        self.to(self.device)

    def _maybe_checkpoint(self, fn, *args):
        if (not self.use_checkpoint) or (not self.training) or (not torch.is_grad_enabled()):
            return fn(*args)
        # checkpoint needs at least one Tensor input
        if not any(isinstance(a, torch.Tensor) for a in args):
            return fn(*args)

        # reentrant checkpoint errors if no input requires grad
        if self.checkpoint_use_reentrant and not any(
            isinstance(a, torch.Tensor) and a.requires_grad for a in args
        ):
            return fn(*args)

        return _checkpoint(fn, *args, use_reentrant=self.checkpoint_use_reentrant)

    def _coords_grid_cached(self, B: int, H: int, W: int, device, dtype) -> torch.Tensor:
        key = (device, dtype, H, W)
        if key not in self._grid_cache:
            self._grid_cache[key] = _coords_grid(1, H, W, device, dtype)
        return self._grid_cache[key].repeat(B, 1, 1, 1)

    def _warp(self, x: torch.Tensor, flow_px: torch.Tensor, padding_mode: str = "border") -> torch.Tensor:

        """
        x: [B,C,H,W], flow_px: [B,2,H,W] in pixels at that resolution
        """
        B, C, H, W = x.shape
        base = self._coords_grid_cached(B, H, W, x.device, x.dtype)
        grid = base + flow_px
        xg = 2.0 * grid[:, 0] / max(W - 1, 1) - 1.0
        yg = 2.0 * grid[:, 1] / max(H - 1, 1) - 1.0
        grid_norm = torch.stack([xg, yg], dim=-1)
        return F.grid_sample(x, grid_norm, mode="bilinear", padding_mode=padding_mode, align_corners=True)

    def warp_blend_tensor(
        self,
        x_prev: torch.Tensor,               # [B,C,H,W]
        flows_sel: torch.Tensor,            # [B,ksel,2,H,W]
        topk_w: torch.Tensor,               # [B,ksel,H,W]
        factor: int = 1,
        warp_fn=None,
    ) -> torch.Tensor:
        """Warp a generic tensor with the SAME selected expert flows + weights used for edge warping."""
        warp = self._warp if warp_fn is None else warp_fn
        B, C, H, W = x_prev.shape
        B2, ksel, _, Hf, Wf = flows_sel.shape
        assert B2 == B and Hf == H and Wf == W

        h_r, w_r = H // factor, W // factor
        x_r = F.interpolate(x_prev, size=(h_r, w_r), mode="bilinear", align_corners=False)

        flows_r = F.interpolate(
            flows_sel.reshape(B * ksel, 2, H, W),
            size=(h_r, w_r),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, ksel, 2, h_r, w_r) / float(factor)

        w_r_eff = F.interpolate(topk_w, size=(h_r, w_r), mode="bilinear", align_corners=False)

        x_rep = x_r.unsqueeze(1).expand(B, ksel, C, h_r, w_r).reshape(B * ksel, C, h_r, w_r)
        f_rep = flows_r.reshape(B * ksel, 2, h_r, w_r)

        warped_rep = warp(x_rep, f_rep, padding_mode="zeros").reshape(B, ksel, C, h_r, w_r)
        x_warp = (w_r_eff.unsqueeze(2) * warped_rep).sum(dim=1)
        return x_warp

    def _lb_loss(self, gate_probs: torch.Tensor) -> torch.Tensor:
        """Simple importance+load balancing on gate probs [B,K,H,W]."""
        B, K, H, W = gate_probs.shape
        importance = gate_probs.mean(dim=(0, 2, 3))  # [K]
        top1 = gate_probs.argmax(dim=1)              # [B,H,W]
        load = torch.stack([(top1 == k).float().mean() for k in range(K)], dim=0)
        tgt = 1.0 / float(K)
        return ((importance - tgt) ** 2).mean() + ((load - tgt) ** 2).mean()

    def _encode_all_expert_flows(self, flows: torch.Tensor) -> torch.Tensor:
        """flows: [B,K,2,H,W] -> [B,K,Hid,H,W]"""
        B, K, _, H, W = flows.shape
        y = self.flow_enc(flows.reshape(B * K, 2, H, W))
        return y.reshape(B, K, self.hidden_dim, H, W)

    def forward(
        self,
        flow_in: torch.Tensor,                 # [B,Cin,H,W]
        e_prev: Optional[torch.Tensor] = None, # [B,Cx,H,W]
        warp_fn=None,
        max_flow_full: Optional[float] = None,
    ) -> Dict[str, torch.Tensor | Dict[int, torch.Tensor]]:
        B, _, H, W = flow_in.shape
        warp = self._warp if warp_fn is None else warp_fn

        flows = None
        net = None
        prev_factor = None

        gate_logits_pyr = []
        lb_losses = []

        # keep last-level tensors for confidence head
        last_lvl_code = None
        last_motion_all = None
        last_net = None

        # coarse features -> fine refinement pyramid 
        for li, factor in enumerate(self.pyramid_factors):
            h_l, w_l = H // factor, W // factor
            x_l = F.interpolate(flow_in, size=(h_l, w_l), mode="bilinear", align_corners=False)

            net_inp = self._maybe_checkpoint(self.cnet, x_l) # [B,2Hid,h_l,w_l]
            net0, inp0 = net_inp.split(self.hidden_dim, 1)  # each [B,Hid,h_l,w_l]
            inp = F.relu(inp0)

            if net is None:
                net = torch.tanh(net0)
                flows = x_l.new_zeros(B, self.K, 2, h_l, w_l)
            else:
                scale = float(prev_factor) / float(factor)  # flow scale when resizing
                net = F.interpolate(net, size=(h_l, w_l), mode="bilinear", align_corners=False) + torch.tanh(net0)
                flows = F.interpolate(
                    flows.reshape(B * self.K, 2, flows.shape[-2], flows.shape[-1]),
                    size=(h_l, w_l),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(B, self.K, 2, h_l, w_l) * scale

            prev_factor = factor

            gate_logits = self._maybe_checkpoint(self.router, net_inp)  # [B,K,h_l,w_l]
            gate_probs = torch.softmax(gate_logits / self.gate_temperature, dim=1)
            lb_losses.append(self._lb_loss(gate_probs))
            gate_logits_pyr.append(gate_logits)

            lvl_code = self.level_code[li].view(1, -1, 1, 1).expand(B, -1, h_l, w_l)
            max_flow_l = (max_flow_full / factor) if max_flow_full is not None else None

            for _ in range(self.iters_per_level[li]):
                # encode current per-expert flows -> motion features
                motion_all = self._encode_all_expert_flows(flows)     # [B,K,Hid,h_l,w_l]
                motion_mean = motion_all.mean(dim=1)                  # [B,Hid,h_l,w_l]

                # GRU update (checkpoint-safe)
                gru_inp = torch.cat([inp, motion_mean], dim=1)        # [B,2*Hid,h_l,w_l]
                net = self._maybe_checkpoint(self.gru, net, gru_inp)  # [B,Hid,h_l,w_l]

                # compute all expert deltas first (no in-place writes into flows)
                deltas = []
                for k, head in enumerate(self.delta_heads):
                    exp_code = self.expert_code[k].view(1, -1, 1, 1).expand(B, -1, h_l, w_l)   # [B,code_dim,h_l,w_l]
                    head_in = torch.cat([net, motion_all[:, k], exp_code, lvl_code], dim=1)    # [B,2*Hid+2*code_dim,h_l,w_l]
                    delta_k = self._maybe_checkpoint(head, head_in)                            # [B,2,h_l,w_l]
                    deltas.append(delta_k)

                deltas = torch.stack(deltas, dim=1)             # [B,K,2,h_l,w_l]
                flows = flows + deltas                          # out-of-place update

                if max_flow_l is not None:
                    flows = max_flow_l * torch.tanh(flows / (max_flow_l + self._eps))

            # keep last-level tensors for confidence heads
            last_lvl_code = lvl_code
            last_motion_all = self._encode_all_expert_flows(flows)
            last_net = net


        # ---- convex upsample to full-res ----
        h8, w8 = H // self.up_factor, W // self.up_factor
        up_mask = 0.25 * self._maybe_checkpoint(self.mask_head, net)  # [B,up^2*9,h8,w8]

        flows_up = convex_upsample_flow(
            flows.reshape(B * self.K, 2, h8, w8),
            up_mask.unsqueeze(1).expand(-1, self.K, -1, -1, -1).reshape(B * self.K, -1, h8, w8),
            self.up_factor,
        ).reshape(B, self.K, 2, H, W)

        # ---- full-res gating (no blur) ----
        gate_logits_low = gate_logits_pyr[-1]  # [B,K,h8,w8]
        gate_logits_full = F.interpolate(gate_logits_low, size=(H, W), mode="bilinear", align_corners=False)

        topk_idx, topk_w, gate_probs_full = topk_gating(
            gate_logits_full,
            top_k=self.top_k,
            temperature=self.gate_temperature,
        )

        ksel = topk_idx.shape[1]
        idx_exp = topk_idx.unsqueeze(2).expand(-1, -1, 2, -1, -1)     # [B,ksel,2,H,W]
        flows_sel = torch.gather(flows_up, dim=1, index=idx_exp)      # [B,ksel,2,H,W]

        #  confidence 
        if self.use_confidence:
            conf_logits_low_list = []
            for k, chead in enumerate(self.conf_heads):
                exp_code = self.expert_code[k].view(1, -1, 1, 1).expand(B, -1, h8, w8)
                conf_in = torch.cat([last_net, last_motion_all[:, k], exp_code, last_lvl_code], dim=1)
                conf_logits_low_list.append(self._maybe_checkpoint(chead, conf_in))           # [B,1,h8,w8]
            conf_logits_low = torch.cat(conf_logits_low_list, dim=1)   # [B,K,h8,w8]
            conf_logits_full = F.interpolate(conf_logits_low, size=(H, W), mode="bilinear", align_corners=False)
            conf_full = torch.sigmoid(conf_logits_full).clamp(self.conf_floor, self.conf_ceil)  # [B,K,H,W]
            conf_sel = torch.gather(conf_full, dim=1, index=topk_idx)  # [B,ksel,H,W]

            topk_w_eff = topk_w * conf_sel
            topk_w_eff = topk_w_eff / (topk_w_eff.sum(dim=1, keepdim=True) + self._eps)
        else:
            conf_full = None
            topk_w_eff = topk_w

        # debug blended flow (not used for final warping)
        flow_blended = (topk_w_eff.unsqueeze(2) * flows_sel).sum(dim=1)  # [B,2,H,W]

        # ---- regularizers ----
        lb_loss = torch.stack(lb_losses).mean() * self.lb_weight
        tv_gate = tv_loss_gate(gate_probs_full) * self.tv_gate_weight

        disagree = flows_sel.new_tensor(0.0)
        if self.disagree_weight > 0 and ksel >= 2:
            # pairwise |flow_i - flow_j| weighted by overlap w_i*w_j
            diff = (flows_sel[:, :, None] - flows_sel[:, None, :]).abs().sum(dim=3)  # [B,ksel,ksel,H,W]
            wprod = (topk_w_eff[:, :, None] * topk_w_eff[:, None, :])                # [B,ksel,ksel,H,W]
            mask = torch.triu(torch.ones(ksel, ksel, device=flows_sel.device, dtype=flows_sel.dtype), diagonal=1)
            disagree = (diff * wprod * mask[None, :, :, None, None]).sum(dim=(1, 2)).mean()
            disagree = disagree * self.disagree_weight

        # ---- warp pyramid (6 levels) ----
        e_warp = None
        e_warp_by_factor: Dict[int, torch.Tensor] = {}

        if e_prev is not None:
            Cx = e_prev.shape[1]
            for factor in self.out_factors:
                h_r, w_r = H // factor, W // factor
                e_r = F.interpolate(e_prev, size=(h_r, w_r), mode="bilinear", align_corners=False)

                # flow at this resolution must be in "pixels of that resolution"
                flows_r = F.interpolate(
                    flows_sel.reshape(B * ksel, 2, H, W),
                    size=(h_r, w_r),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(B, ksel, 2, h_r, w_r) / float(factor)

                w_r_eff = F.interpolate(topk_w_eff, size=(h_r, w_r), mode="bilinear", align_corners=False)

                e_rep = e_r.unsqueeze(1).expand(B, ksel, Cx, h_r, w_r).reshape(B * ksel, Cx, h_r, w_r)
                f_rep = flows_r.reshape(B * ksel, 2, h_r, w_r)

                warped_rep = warp(e_rep, f_rep, padding_mode="zeros").reshape(B, ksel, Cx, h_r, w_r)
                e_r_warp = (w_r_eff.unsqueeze(2) * warped_rep).sum(dim=1)

                e_warp_by_factor[factor] = e_r_warp
                if factor == 1:
                    e_warp = e_r_warp

        return {
            "e_warp": e_warp,
            "e_warp_by_factor": e_warp_by_factor,   # keys: 1,2,4,8,16,32
            "flows_up": flows_up,                   # [B,K,2,H,W]
            "flows_sel": flows_sel,                 # [B,ksel,2,H,W]
            "flow_blended": flow_blended,           # debug
            "gate_probs_full": gate_probs_full,     # [B,K,H,W]
            "conf_full": conf_full,                 # [B,K,H,W] or None
            "topk_idx": topk_idx,
            "topk_w": topk_w_eff,                   # effective weights (gate or gate×conf)
            "lb_loss": lb_loss,
            "tv_gate_loss": tv_gate,
            "disagree_loss": disagree,
        }

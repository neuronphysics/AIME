# ------------------------------------------------------------
# Robust heteroscedastic line-alignment loss for optical flow using soft edges.
#
# Corrected centered-search version:
#   - We center the 1D normal search at p = c + flow_fwd(c)
#   - Therefore the per-point residual is r_i = mu_i  (NOT mu_i - d_i)
#     because mu_i is the *correction* along the normal around the predicted match.
#
# Optional robustness:
#   - motion gating (suppresses static checkerboard/background edges)
#   - forward-backward consistency gating (occlusion / impossible matches)
#   - symmetric term (sample from target and align back to prev using backward flow)
#
# Expected inputs:
#   E_prev, E_tgt: [B,1,H,W] soft edge probabilities in [0,1] 
#   flow_fwd:      [B,2,H,W] flow from prev->tgt in PIXELS (dx,dy)
#   flow_bwd:      [B,2,H,W] flow from tgt->prev in PIXELS (dx,dy), optional but recommended
#   x_prev01, x_tgt01: [B,3,H,W] images in [0,1] for motion gating, optional
#
# ------------------------------------------------------------
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Config
# ----------------------------

@dataclass
class LineAlignCfg:
    # Search along normal offsets in pixels: [-R..R]
    search_radius: int = 6
    beta: float = 12.0                 # softmax sharpness over sampled edge evidence
    sigma0: float = 0.5                # base uncertainty floor (px)
    n_points: int = 2048               # sampled edge points per frame

    # (grid_sample with base + flow).
    # "tgt2prev": flow is defined on target grid and points to where to sample in prev.
    # "prev2tgt": flow is defined on prev grid and points to where to sample in tgt.
    flow_direction: str = "tgt2prev"   # "tgt2prev" or "prev2tgt"

    # Center search at predicted correspondence (recommended)
    center_on_flow: bool = True

    # Robust penalty on residual
    robust_delta: float = 1.0          # Huber delta (in px)

    # Optional: bias sampling toward moving regions
    motion_gate: bool = True
    motion_gate_thresh: float = 0.03   # on |x_t - x_prev| in [0,1]
    motion_gate_power: float = 1.0

    # Optional: photometric gating as a practical occlusion/outlier filter
    use_photometric_gating: bool = True
    photo_gate_thresh: float = 0.20    # mean(|warp(prev)-tgt|) in [0,1]

    # Numerical
    eps: float = 1e-6
    detach_edges: bool = True


# ----------------------------
# Helpers
# ----------------------------

def _coords_to_grid_xy(coords_xy: torch.Tensor, H: int, W: int, align_corners: bool = True) -> torch.Tensor:
    """
    coords_xy: [B, N, 2] in pixel coords (x,y)
    returns:   [B, N, 1, 2] normalized grid for grid_sample
    """
    x = coords_xy[..., 0]
    y = coords_xy[..., 1]
    if align_corners:
        gx = 2.0 * x / (W - 1) - 1.0
        gy = 2.0 * y / (H - 1) - 1.0
    else:
        gx = 2.0 * (x + 0.5) / W - 1.0
        gy = 2.0 * (y + 0.5) / H - 1.0
    return torch.stack([gx, gy], dim=-1).unsqueeze(2)


def _sample_2ch_at(flow_2ch: torch.Tensor, coords_xy: torch.Tensor) -> torch.Tensor:
    """
    flow_2ch:  [B, 2, H, W]
    coords_xy: [B, N, 2]
    returns:   [B, N, 2]
    """
    B, _, H, W = flow_2ch.shape
    grid = _coords_to_grid_xy(coords_xy, H, W, align_corners=True)
    samp = F.grid_sample(flow_2ch, grid, mode="bilinear", padding_mode="border", align_corners=True)  # [B,2,N,1]
    return samp.permute(0, 2, 1, 3).squeeze(-1)  # [B,N,2]


def _sample_1ch_at(img_1ch: torch.Tensor, coords_xy: torch.Tensor) -> torch.Tensor:
    """
    img_1ch:   [B, 1, H, W]
    coords_xy: [B, N, 2]
    returns:   [B, N]
    """
    B, _, H, W = img_1ch.shape
    grid = _coords_to_grid_xy(coords_xy, H, W, align_corners=True)
    samp = F.grid_sample(img_1ch, grid, mode="bilinear", padding_mode="border", align_corners=True)  # [B,1,N,1]
    return samp.permute(0, 2, 1, 3).squeeze(-1).squeeze(-1)  # [B,N]


def _huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    ax = x.abs()
    q = torch.minimum(ax, torch.tensor(delta, device=x.device, dtype=x.dtype))
    lin = ax - q
    return 0.5 * q * q + delta * lin


def _sobel_grads(E: torch.Tensor) -> torch.Tensor:
    """E: [B,1,H,W] -> grads [B,2,H,W] in (dx, dy)"""
    device, dtype = E.device, E.dtype
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(E, kx, padding=1)
    gy = F.conv2d(E, ky, padding=1)
    return torch.cat([gx, gy], dim=1)


def _warp_with_flow_tgt2prev(x_prev: torch.Tensor, flow_tgt2prev: torch.Tensor) -> torch.Tensor:
    """
    x_prev:        [B,C,H,W]
    flow_tgt2prev: [B,2,H,W]  (defined on target grid; points to prev sampling coords)
    returns:       [B,C,H,W]  warped prev -> target grid
    """
    B, C, H, W = x_prev.shape
    # base grid in pixel coords
    ys, xs = torch.meshgrid(
        torch.arange(H, device=x_prev.device, dtype=x_prev.dtype),
        torch.arange(W, device=x_prev.device, dtype=x_prev.dtype),
        indexing="ij"
    )
    base = torch.stack([xs, ys], dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # [B,2,H,W]
    coords = base + flow_tgt2prev
    grid = _coords_to_grid_xy(coords.permute(0,2,3,1).reshape(B, H*W, 2), H, W).reshape(B, H, W, 2)
    return F.grid_sample(x_prev, grid, mode="bilinear", padding_mode="border", align_corners=True)


# ----------------------------
# Loss Module
# ----------------------------

class HeteroscedasticLineAlignmentLoss(nn.Module):
    """
    Edge line alignment with heteroscedastic (per-point) uncertainty.

    warp prev->tgt with grid_sample(base + flow)),
        cfg.flow_direction = "tgt2prev"
    i.e. flow is defined on the target grid and points to where to sample in prev.
    """
    def __init__(self, cfg: LineAlignCfg):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        E_prev: torch.Tensor,                 # [B,1,H,W]
        E_tgt: torch.Tensor,                  # [B,1,H,W]
        flow: torch.Tensor,                   # [B,2,H,W] direction depends on cfg.flow_direction
        x_prev01: Optional[torch.Tensor] = None,  # [B,3,H,W] in [0,1]
        x_tgt01: Optional[torch.Tensor] = None,   # [B,3,H,W] in [0,1]
        return_stats: bool = True,
        sample_mask: Optional[torch.Tensor] = None,
        lambda_ent: float = 0.01
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        cfg = self.cfg
        assert E_prev.ndim == 4 and E_prev.shape[1] == 1
        assert E_tgt.ndim == 4 and E_tgt.shape[1] == 1
        assert flow.ndim == 4 and flow.shape[1] == 2
        B, _, H, W = E_prev.shape
        assert E_tgt.shape[-2:] == (H, W)
        assert flow.shape[-2:] == (H, W)

        if cfg.detach_edges:
            E_prev = E_prev.detach()
            E_tgt = E_tgt.detach()

        # Choose which edge map we sample points from, and which map we search in
        if cfg.flow_direction == "tgt2prev":
            E_src = E_tgt
            E_search = E_prev
        elif cfg.flow_direction == "prev2tgt":
            E_src = E_prev
            E_search = E_tgt
        else:
            raise ValueError(f"Unknown flow_direction={cfg.flow_direction}")

        # Motion gate (optional)
        if cfg.motion_gate and (x_prev01 is not None) and (x_tgt01 is not None):
            with torch.no_grad():
                diff = (x_tgt01 - x_prev01).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
                gate = (diff > cfg.motion_gate_thresh).float()
                if cfg.motion_gate_power != 1.0:
                    gate = gate.pow(cfg.motion_gate_power)
        else:
            gate = None

        # Sampling distribution over pixels
        with torch.no_grad():
            w = E_src.clamp_min(0.0)
            if gate is not None:
                w = w * gate
            wsum = w.sum(dim=[2, 3], keepdim=True)
            # fallback to uniform if no edges/motion
            w = torch.where(wsum > cfg.eps, w / (wsum + cfg.eps), torch.full_like(w, 1.0 / (H * W)))
            w_flat = w.view(B, -1)  # [B,HW]
            idx = torch.multinomial(w_flat, num_samples=cfg.n_points, replacement=True)  # [B,N]
            y = (idx // W).float()
            x = (idx % W).float()
            c_xy = torch.stack([x, y], dim=-1)  # [B,N,2]

        # Normal from Sobel gradient of E_src at sampled points
        grads = _sobel_grads(E_src)  # [B,2,H,W]
        g_xy = _sample_2ch_at(grads, c_xy)  # [B,N,2]
        n_xy = g_xy / (g_xy.norm(dim=-1, keepdim=True) + 1e-6)

        # Flow sampled at points
        f_xy = _sample_2ch_at(flow, c_xy)  # [B,N,2]

        # Center for search in E_search
        if cfg.center_on_flow:
            center_xy = c_xy + f_xy
        else:
            center_xy = c_xy

        # Sample evidence along normal offsets
        offsets = torch.arange(-cfg.search_radius, cfg.search_radius + 1,
                               device=E_src.device, dtype=E_src.dtype)  # [R]
        R = offsets.numel()

        coords = center_xy.unsqueeze(2) + offsets.view(1, 1, R, 1) * n_xy.unsqueeze(2)  # [B,N,R,2]
        coords_flat = coords.reshape(B, cfg.n_points * R, 2)
        e = _sample_1ch_at(E_search, coords_flat).reshape(B, cfg.n_points, R)  # [B,N,R]

        # Soft argmax distribution over offsets
        q = torch.softmax(cfg.beta * e, dim=-1)  # [B,N,R]
        mu = (q * offsets.view(1, 1, R)).sum(dim=-1)  # [B,N]
        var = (q * (offsets.view(1, 1, R) - mu.unsqueeze(-1)).pow(2)).sum(dim=-1)  # [B,N]

        # Residual: if centered on flow, target is mu≈0
        if cfg.center_on_flow:
            resid = mu
        else:
            # If not centered, subtract the predicted normal component
            d = (n_xy * f_xy).sum(dim=-1)  # [B,N]
            resid = mu - d

        sigma2 = (var + cfg.sigma0**2).clamp(min=cfg.sigma0**2, max=(cfg.search_radius**2 + cfg.sigma0**2))
 
        per_point = 0.5 * _huber(resid, cfg.robust_delta) / (sigma2 + torch.finfo(torch.float32).eps) + 0.5 * torch.log(sigma2 + torch.finfo(torch.float32).eps)
        eps_q = torch.finfo(q.dtype).eps
        q_safe = q.clamp_min(eps_q)
        ent = -(q_safe * q_safe.log()).sum(dim=-1)   # [B,N]

        
        per_point = per_point - lambda_ent * ent

        # Photometric gating (optional, practical outlier/occlusion filter)
        if cfg.use_photometric_gating and (x_prev01 is not None) and (x_tgt01 is not None) and (cfg.flow_direction == "tgt2prev"):
            with torch.no_grad():
                xw = _warp_with_flow_tgt2prev(x_prev01, flow)  # [B,3,H,W]
                pe = (xw - x_tgt01).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
                pe_samp = _sample_1ch_at(pe, c_xy)  # [B,N]
                keep = (pe_samp < cfg.photo_gate_thresh).float()
            per_point = per_point * keep

        per_frame = per_point.mean(dim=1)  # [B]

        if sample_mask is None:
            loss = per_frame.mean()
        else:
            m = sample_mask.to(per_frame).view(-1).clamp(0.0, 1.0)  # [B]
            loss = (per_frame * m).sum() / (m.sum() + cfg.eps)

        if not return_stats:
            return loss, {}

        if sample_mask is None:
            stats = {
                "mu_abs_mean": float(mu.abs().mean().item()),
                "sigma_mean": float(torch.sqrt(sigma2).mean().item()),
                "resid_abs_mean": float(resid.abs().mean().item()),
                "edge_evidence_mean": float(e.mean().item()),
            }
        else:
            m = sample_mask.to(mu).view(B, 1).clamp(0.0, 1.0)      # [B,1]
            denom_pts = (m.sum() * cfg.n_points + cfg.eps)
            denom_e   = (m.sum() * cfg.n_points * e.shape[-1] + cfg.eps)

            stats = {
                "mu_abs_mean": float(((mu.abs() * m).sum() / denom_pts).item()),
                "sigma_mean": float(((torch.sqrt(sigma2) * m).sum() / denom_pts).item()),
                "resid_abs_mean": float(((resid.abs() * m).sum() / denom_pts).item()),
                "edge_evidence_mean": float(((e * m.unsqueeze(-1)).sum() / denom_e).item()),
            }

        return loss, stats

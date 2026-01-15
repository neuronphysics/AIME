import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from typing import Optional, Tuple, Dict


def _coords_to_grid(coords_xy: torch.Tensor, H: int, W: int) -> torch.Tensor:
    # coords_xy: [B,N,2] in pixel coords (x,y), align_corners=True
    x = coords_xy[..., 0]
    y = coords_xy[..., 1]
    gx = 2.0 * x / max(W - 1, 1) - 1.0
    gy = 2.0 * y / max(H - 1, 1) - 1.0
    return torch.stack([gx, gy], dim=-1).unsqueeze(2)  # [B,N,1,2]


def _sample_1ch(img: torch.Tensor, coords_xy: torch.Tensor) -> torch.Tensor:
    # img: [B,1,H,W], coords_xy: [B,N,2] -> [B,N]
    B, _, H, W = img.shape
    grid = _coords_to_grid(coords_xy, H, W)
    samp = F.grid_sample(
        img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )  # [B,1,N,1]
    return samp.squeeze(-1).squeeze(1)  # [B,N]


@torch.no_grad()
def edt_2d_squared(
    edge_bt: torch.Tensor,
    edge_thresh: float = 0.3,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    edge_bt: [B,T,1,H,W] OR [B,1,H,W]
    returns: same shape, squared EDT per frame (distance to nearest edge pixel)
    """
    if edge_bt.ndim == 5:
        B, T, C, H, W = edge_bt.shape
        assert C == 1
        flat = edge_bt.reshape(B * T, 1, H, W)   # [N,1,H,W]
        out_shape = (B, T, 1, H, W)
    elif edge_bt.ndim == 4:
        B, C, H, W = edge_bt.shape
        assert C == 1
        flat = edge_bt
        out_shape = (B, 1, H, W)
    else:
        raise ValueError(f"Expected 4D or 5D input, got {tuple(edge_bt.shape)}")

    # bool on CPU (single sync per call)
    edge_bin = (flat > edge_thresh).squeeze(1).detach()          # [N,H,W]
    edge_np = edge_bin.to(torch.bool).cpu().numpy()              # [N,H,W] bool

    N = edge_np.shape[0]
    out_np = np.empty((N, H, W), dtype=np.float32)

    # IMPORTANT: per-slice 2D EDT (NOT one 3D EDT)
    for i in range(N):
        d = distance_transform_edt(~edge_np[i]).astype(np.float32)  # [H,W]
        out_np[i] = d * d

    out = torch.from_numpy(out_np).to(device=edge_bt.device, dtype=out_dtype).unsqueeze(1)  # [N,1,H,W]
    return out.reshape(out_shape)


class WarpEdgeChamfer(nn.Module):
    """
    Chamfer-style edge alignment for warp training.

    Loss = w_p2g * DT_chamfer(pred -> gt)  +  w_g2p * local_soft_chamfer(gt -> pred)

    pred->gt uses GT EDT (can be precomputed & passed in)
    gt->pred uses differentiable local soft-nearest search (gives gradients)
    """
    def __init__(
        self,
        edge_thresh: float = 0.3,
        trunc_dist: float = 10.0,
        n_points: int = 256,   # << for 64x64 start small (256 or 512)
        radius: int = 3,       # << for 64x64 use 3 or 4 (NOT 6)
        beta: float = 12.0,
        dist_gamma: float = 0.2,
        w_p2g: float = 1.0,
        w_g2p: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.edge_thresh = edge_thresh
        self.trunc_dist = trunc_dist
        self.n_points = n_points
        self.radius = radius
        self.beta = beta
        self.dist_gamma = dist_gamma
        self.w_p2g = w_p2g
        self.w_g2p = w_g2p
        self.eps = eps
        # Precompute local offsets once to avoid per-forward allocations (helps fragmentation)
        r = int(self.radius)
        dy, dx = torch.meshgrid(
            torch.arange(-r, r + 1),
            torch.arange(-r, r + 1),
            indexing="ij",
        )
        off = torch.stack([dx, dy], dim=-1).reshape(1, 1, -1, 2).float()  # [1,1,K,2]
        dist2 = (dx * dx + dy * dy).reshape(1, 1, -1).float()            # [1,1,K]
        self.register_buffer("_off", off, persistent=False)
        self.register_buffer("_dist2", dist2, persistent=False)

    def forward(
        self,
        e_pred: torch.Tensor,
        e_gt: torch.Tensor,
        sample_mask: Optional[torch.Tensor] = None,
        d2_gt: Optional[torch.Tensor] = None,   # << pass precomputed EDT here
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        e_pred, e_gt: [B,1,H,W] in [0,1]
        d2_gt (optional): [B,1,H,W] squared EDT of e_gt
        """
        assert e_pred.shape == e_gt.shape
        B, _, H, W = e_pred.shape
        eps = self.eps

        # ------------------- pred -> gt (DT chamfer) -------------------
        if self.w_p2g != 0.0:
            with torch.no_grad():
                if d2_gt is None:
                    d2_gt = edt_2d_squared(e_gt, edge_thresh=self.edge_thresh, out_dtype=e_pred.dtype)  # [B,1,H,W]
                d_gt = torch.sqrt(d2_gt + eps).clamp_max(self.trunc_dist)  # [B,1,H,W]

            w_pred = e_pred.clamp_min(0.0)
            if spatial_mask is not None:
               w_pred = w_pred * spatial_mask.to(w_pred)

            p2g = (d_gt * w_pred).sum(dim=(1, 2, 3)) / (w_pred.sum(dim=(1, 2, 3)) + eps)  # [B]
        else:
            p2g = e_pred.new_zeros(B)

# ------------------- gt -> pred (local differentiable chamfer) -------------------
        if self.w_g2p != 0.0:
            with torch.no_grad():
                w = e_gt.clamp_min(0.0)
                #if spatial_mask is not None:
                #    w = w * spatial_mask.to(w)
                wsum = w.sum(dim=(2, 3), keepdim=True) # Don't mask GT
                if torch.any(wsum <= eps): # masking
                    w = e_gt.clamp_min(0.0)
                    wsum = w.sum(dim=(2,3), keepdim=True)
                w = torch.where(wsum > eps, w / (wsum + eps), torch.full_like(w, 1.0 / (H * W)))
                idx = torch.multinomial(w.view(B, -1), num_samples=self.n_points, replacement=True)  # [B,N]
                y = (idx // W).float()
                x = (idx % W).float()
                pts = torch.stack([x, y], dim=-1)  # [B,N,2]

            off = self._off.to(device=e_pred.device, dtype=e_pred.dtype)
            dist2 = self._dist2.to(device=e_pred.device, dtype=e_pred.dtype)

            coords = pts.unsqueeze(2) + off                  # [B,N,K,2]
            #supress messing background
            valid = (
                (coords[..., 0] >= 0.0) & (coords[..., 0] <= (W - 1)) &
                (coords[..., 1] >= 0.0) & (coords[..., 1] <= (H - 1))
            )  # [B,N,K]
            coords_flat = coords.reshape(B, -1, 2)           # [B,N*K,2]
            e_samp = _sample_1ch(e_pred, coords_flat).view(B, self.n_points, -1)  # [B,N,K]

            logits = self.beta * e_samp - self.dist_gamma * dist2  # [B,N,K]
            logits = logits.masked_fill(~valid.to(logits.device), -1e4) #supress messing background

            logits = torch.nan_to_num(logits, neginf=-1e4, posinf=1e4)

            log_den = torch.logsumexp(logits, dim=-1)                           # [B,N]
            log_num = torch.logsumexp(logits + torch.log(dist2 + eps), dim=-1)  # [B,N]
            exp_d2 = torch.exp(log_num - log_den).clamp_min(eps)                # [B,N]

            g2p = torch.sqrt(exp_d2).clamp_max(self.trunc_dist).mean(dim=1)     # [B]

        else:
            g2p = e_pred.new_zeros(B)

        per_frame = self.w_p2g * p2g + self.w_g2p * g2p

        if sample_mask is not None:
            m = sample_mask.to(per_frame).view(-1).clamp(0.0, 1.0)
            loss = (per_frame * m).sum() / (m.sum() + eps)
        else:
            loss = per_frame.mean()

        stats = {
            "ch_p2g": float(p2g.mean().item()),
            "ch_g2p": float(g2p.mean().item()),
            "ch_total": float(per_frame.mean().item()),
        }
        return loss, stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
from typing import Tuple, Optional
from typing import TypeVar
from typing import Union, List, Callable
import math
import numpy as np

T = TypeVar("T")
ItemOrList = Union[T, List[T]]

def downsample_flow(flow_src: torch.Tensor, dst_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Downsample flow without interpolation.
    Uses a box filter implemented as depthwise conv + striding.

    flow_src: [B,2,srcH,srcW] in src-grid pixels
    returns:  [B,2,dstH,dstW] in dst-grid pixels
    """
    assert flow_src.ndim == 4 and flow_src.size(1) == 2
    B, _, srcH, srcW = flow_src.shape
    dstH, dstW = dst_hw

    # require exact integer downsample (e.g., 64->8, 64->16, 32->8, ...)
    assert srcH % dstH == 0 and srcW % dstW == 0, \
        f"downsample_flow requires integer ratio: ({srcH},{srcW})->({dstH},{dstW})"

    sy = srcH // dstH
    sx = srcW // dstW

    # fixed box filter: average over each (sy,sx) block, applied separately to dx/dy
    w = flow_src.new_ones((2, 1, sy, sx)) / float(sy * sx)  # [2,1,sy,sx]
    flow = F.conv2d(flow_src, w, stride=(sy, sx), padding=0, groups=2)  # [B,2,dstH,dstW]

    # unit conversion: src-pixels -> dst-pixels
    flow[:, 0] *= (dstW / srcW)  # dx
    flow[:, 1] *= (dstH / srcH)  # dy
    return flow

def smooth_flow(flow: torch.Tensor, k: int = 3) -> torch.Tensor:
    assert flow.ndim == 4 and flow.size(1) == 2
    assert k % 2 == 1, "use odd k"
    w = flow.new_ones((2, 1, k, k)) / float(k * k)  # depthwise, per-channel
    pad = k // 2
    # optional: replicate padding to avoid zero-border bias
    flow = F.pad(flow, (pad, pad, pad, pad), mode="replicate")
    return F.conv2d(flow, w, padding=0, groups=2)

def upsample_flow(flow: torch.Tensor, dst_hw: tuple[int, int], k: int = 3) -> torch.Tensor:
    assert flow.ndim == 4 and flow.size(1) == 2
    B, _, Hs, Ws = flow.shape
    Hd, Wd = int(dst_hw[0]), int(dst_hw[1])
    assert Hd % Hs == 0 and Wd % Ws == 0, f"integer ratio required: ({Hs},{Ws})->({Hd},{Wd})"
    sy = Hd // Hs
    sx = Wd // Ws

    w = flow.new_ones((2, 1, sy, sx))  # [2,1,sy,sx]
    up = F.conv_transpose2d(flow, w, stride=(sy, sx), padding=0, groups=2)  # [B,2,Hd,Wd]

    # unit conversion: src-grid pixels -> dst-grid pixels
    up[:, 0] *= sx
    up[:, 1] *= sy

    return smooth_flow(up, k=k)

def upsample_flow_raft(flow: torch.Tensor, mask: torch.Tensor, scale: int) -> torch.Tensor:
    """
    RAFT-style learned convex upsampling.

    flow: [B,2,H,W] (low-res flow in low-res pixel units)
    mask: [B, 9*scale*scale, H, W] (learned weights)
    returns: [B,2,H*scale,W*scale] (high-res flow in high-res pixel units)
    """
    assert flow.ndim == 4 and flow.size(1) == 2
    B, _, H, W = flow.shape
    s = int(scale)

    assert mask.shape == (B, 9 * s * s, H, W), (mask.shape, (B, 9 * s * s, H, W))

    # Mask: [B, 9*s*s, H, W] -> [B,1,9,s,s,H,W], softmax over 9 neighbors
    mask = mask.view(B, 1, 9, s, s, H, W)
    mask = torch.softmax(mask, dim=2)
    # Unfold 3x3 neighborhoods: [B, 2*9, H, W] -> [B,2,9,H,W]
    flow_unfold = F.unfold(s*flow, kernel_size=3, padding=1)
    flow_unfold = flow_unfold.view(B, 2, 9, 1, 1, H, W)
    # Weighted sum over 9 neighbors -> [B,2,s,s,H,W]
    up = torch.sum(mask * flow_unfold, dim=2)

    # Rearrange to [B,2,H*s,W*s]
    up = up.permute(0, 1, 4, 2, 5, 3).contiguous()
    up = up.view(B, 2, H * s, W * s)

    return up


def _device_key(device: torch.device) -> str:
    # stable key for caching
    if device.type == "cuda":
        return f"cuda:{device.index}"
    return device.type

@lru_cache(maxsize=256)
def _base_grid_cached(device_key: str, dtype_str: str, h: int, w: int):
    """
    Returns base grid in normalized coords [-1,1] with align_corners=True:
      grid: [1, H, W, 2]
    Cached on (device,dtype,H,W). Uses torch ops only.
    """
    # This function is cached, so we rebuild tensors from scratch in the correct device/dtype.
    # dtype_str is like 'torch.float32'
    dtype = getattr(torch, dtype_str.split(".")[-1])

    device = torch.device(device_key)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    # indexing='ij' gives Y first, X second -> grid shape [H,W]
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # yy: [H,W], xx: [H,W]
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # [1,H,W,2]
    return grid


def _get_base_grid(image: torch.Tensor):
    """
    Returns cached [1,H,W,2] base grid on image.device/image.dtype.
    """
    _, _, h, w = image.shape
    device_key = _device_key(image.device)
    dtype_str = str(image.dtype)  # e.g. 'torch.float32'
    return _base_grid_cached(device_key, dtype_str, h, w)


# -----------------------------
# Warp
# -----------------------------

def image_warp(image, flow):
    '''
    image: torch.Size([B, C, H, W])
    flow: torch.Size([B, 2, H, W]) in *pixel units* (dx, dy)
    final_grid:  torch.Size([B, H, W, 2]) after normalization
    '''
    if image.dim() != 4:
        raise ValueError("image must be [B,C,H,W]")
    if flow.dim() != 4 or flow.size(1) != 2:
        raise ValueError("flow must be [B,2,H,W]")

    b, c, h, w = image.size()
    device = image.device
    dtype = image.dtype

    # Ensure flow is on same device/dtype (grid_sample requires float)
    flow = flow.to(device=device, dtype=dtype)

    # Normalize flow from pixels to normalized grid units for align_corners=True
    # x_norm = dx / ((W-1)/2), y_norm = dy / ((H-1)/2)
    flow_norm = torch.cat(
        [
            flow[:, 0:1, :, :] / ((w - 1.0) / 2.0),
            flow[:, 1:2, :, :] / ((h - 1.0) / 2.0),
        ],
        dim=1,
    )
    flow_norm = flow_norm.permute(0, 2, 3, 1)  # [B,H,W,2]

    # Cached base grid [1,H,W,2] -> expand to batch
    base_grid = _get_base_grid(image).to(device=device, dtype=dtype)  # [1,H,W,2]
    grid = base_grid + flow_norm  # [B,H,W,2] via broadcasting

    # IMPORTANT: align_corners=True matches the normalization above.
    output = F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return output


# -----------------------------
# Forward-backward consistency helpers
# -----------------------------
def flow_px_to_norm(flow_px: torch.Tensor, hw: Tuple[int, int] | None = None) -> torch.Tensor:
    """Convert pixel-unit flow to normalized align_corners=True units."""
    assert flow_px.ndim == 4 and flow_px.size(1) == 2
    if hw is None:
        _, _, H, W = flow_px.shape
    else:
        H, W = int(hw[0]), int(hw[1])

    sx = max(W - 1, 1) / 2.0
    sy = max(H - 1, 1) / 2.0

    out = flow_px.clone()
    out[:, 0] = out[:, 0] / sx
    out[:, 1] = out[:, 1] / sy
    return out


def abs_robust_loss(diff, eps=0.01, q=0.4):
  """The so-called robust loss used by DDFlow."""
  return (torch.abs(diff) + eps) ** q

def rgb2gray(img):
    # img: [..., 3, H, W]  (channels in dim -3)
    r = img[..., 0, :, :]
    g = img[..., 1, :, :]
    b = img[..., 2, :, :]
    img_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return img_gray.unsqueeze(-3)

class SSIM(object):
    def __init__(self):
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def __call__(self, x, y, mu_x=None, mu_x_sq=None):

        if mu_x == None:
            mu_x = nn.AvgPool2d(3, 1)(x)
            mu_x_sq = torch.pow(mu_x, 2)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x_sq + mu_y_sq + self.C1) * (sigma_x + sigma_y + self.C2)
        SSIM = SSIM_n / SSIM_d

        SSIM_img = torch.clamp((1 - SSIM) / 2, 0, 1)

        return F.pad(SSIM_img, pad=(1, 1, 1, 1), mode='constant', value=0)


class CensusTransform(nn.Module):
    def __init__(self):
        super(CensusTransform, self).__init__()
        self.patch_size = 7
        self.num_pix_per_patch = self.patch_size ** 2
        self.pad_size = self.patch_size // 2
        self.conv2d = nn.Unfold(self.patch_size, padding=self.pad_size)
        # self.conv2d = nn.Conv2d(1, self.patch_size**2, self.patch_size, padding=self.pad_size, padding_mode='zeros', bias=False)
        # kernel = torch.eye(self.patch_size ** 2).view(self.patch_size, self.patch_size, 1, self.patch_size * self.patch_size).permute(3, 2, 1, 0)
        # self.conv2d.weight = nn.Parameter(kernel, requires_grad=False)

    def get_neighbors(self, x):
        dims = x.size()
        return self.conv2d(x.view(-1, *dims[-3:])).view(*dims[:-3], -1, *dims[-2:])

    def census(self, img):
        intensities = rgb2gray(img)
        neighbors = self.get_neighbors(intensities)
        diff = neighbors - intensities
        diff_norm = diff / (.81 + diff**2)**0.5
        return diff_norm

    def forward(self, x):
        return self.census(x)


class CensusLoss(CensusTransform):
    def __init__(self, args, device):
        super(CensusLoss, self).__init__()
        self.args = args
        self.sequence_weights = args.sequence_weight ** torch.arange(args.iters, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(1)
        self.compute_loss = self.compute_loss_sequential if args.sequentially else self.compute_loss_parallel

    def soft_hamming(self, x, y, mask, thresh=.1):
        sq_dist = (x - y) ** 2
        soft_thresh_dist = sq_dist / (thresh + sq_dist)
        return soft_thresh_dist.sum(dim=-3, keepdims=True), self.zero_mask_border(mask)

    def zero_mask_border(self, mask):
        """Used to ignore border effects from census_transform."""
        mask = mask[..., self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
        return F.pad(mask, (self.pad_size, self.pad_size, self.pad_size, self.pad_size))

    def compute_loss_sequential(self, census_ims, ims_warp, masks):
        census_ims_warps = self.census(ims_warp)
        hamming, masks = self.soft_hamming(census_ims, census_ims_warps, masks)
        diff = (abs_robust_loss(hamming) * masks).sum((-3, -2, -1)) / (masks.sum(dim=(-3, -2, -1)).clamp_min(1e-5))
        return diff.mean()

    def compute_loss_parallel(self, ims, ims_warp, masks):
        census_ims = self.census(ims)
        census_ims_warps = self.census(ims_warp)
        hamming, masks = self.soft_hamming(census_ims.unsqueeze(-4), census_ims_warps, masks)
        diff = (abs_robust_loss(hamming) * masks).sum((-3, -2, -1)) / (masks.sum(dim=(-3, -2, -1)).clamp_min(1e-5))
        T = diff.shape[1]
        w = (self.args.sequence_weight ** torch.arange(T, device=diff.device, dtype=diff.dtype))  # [T]
        return (diff * w[None, :]).sum(dim=1).mean()

    def forward(self, ims, ims_warps, mask, covs=None):
        return self.compute_loss(ims, ims_warps, mask)


def length_sq(x):
    # x: [B,2,H,W] or [B,C,H,W] -> sum over channel dim 1
    return torch.sum(torch.square(x), dim=1, keepdim=True)


def fbConsistencyCheck(flow_fw, flow_bw, alpha1=0.01, alpha2=0.5):
    """
    Classic forward-backward check:
      flow_fw + warp(flow_bw, flow_fw) should be ~0 (non-occluded)
      flow_bw + warp(flow_fw, flow_bw) should be ~0
    """
    flow_bw_warped = image_warp(flow_bw, flow_fw)  # wb(wf(x))
    flow_fw_warped = image_warp(flow_fw, flow_bw)  # wf(wb(x))
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped

    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)
    mag_sq_bw = length_sq(flow_bw) + length_sq(flow_fw_warped)
    occ_thresh_fw = alpha1 * mag_sq_fw + alpha2
    occ_thresh_bw = alpha1 * mag_sq_bw + alpha2

    fb_occ_fw = (length_sq(flow_diff_fw) > occ_thresh_fw).float()
    fb_occ_bw = (length_sq(flow_diff_bw) > occ_thresh_bw).float()
    return fb_occ_fw, fb_occ_bw


# -----------------------------
# Loss utilities
# -----------------------------

def charbonnier_loss(x, mask=None, beta=255, epsilon=1e-3):
    """
    Charbonnier penalty. Compatible with your calls:
      charbonnier_loss(im_diff_fw, mask_fw, beta=beta)
      charbonnier_loss(occ_fw)
    """
    # scale like common optical-flow code (beta=255 for images in [0,255] or to match legacy)
    x = x / float(beta)
    loss = torch.sqrt(x * x + epsilon * epsilon)
    if mask is not None:
        loss = loss * mask
        denom = mask.sum().clamp_min(1.0)
        return loss.sum() / denom
    return loss.mean()


class FlowWarpingLoss(nn.Module):
    def __init__(self, metric):
        super(FlowWarpingLoss, self).__init__()
        self.metric = metric

    def warp(self, x, flow):
        """
        Args:
            x: torch tensor [b, c, h, w] (c can be 3 for rgb or 2 for flow)
            flow: torch tensor [b, 2, h, w] in pixel units

        Returns:
            warped x
        """
        return image_warp(x, flow)

    def __call__(self, x, y, flow, mask):
        """
        Keep your original API.
        """
        warped_x = self.warp(x, flow)
        loss = self.metric(warped_x * mask, y * mask)
        return loss


class TVLoss(nn.Module):
    # Total variation on x
    def __init__(self):
        super(TVLoss, self).__init__()

    def __call__(self, x):
        loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.mean(
            torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        )
        return loss


class WarpLoss(nn.Module):
    def __init__(self):
        super(WarpLoss, self).__init__()
        self.metric = nn.L1Loss()

    def forward(self, flow, mask, img1, img2):
        """
        flow indicates the motion from img1 to img2
        loss compares img1 vs warp(img2, flow) in valid mask region
        """
        img2_warped = image_warp(img2, flow)
        loss = self.metric(img2_warped * mask, img1 * mask)
        return loss


def create_outgoing_mask(flow):
    """
    Mask = 1 where (x+u, y+v) stays inside image bounds, else 0.
    Returns [B,1,H,W]
    """
    if flow.dim() != 4 or flow.size(1) != 2:
        raise ValueError("flow must be [B,2,H,W]")

    b, _, h, w = flow.shape
    device = flow.device
    dtype = flow.dtype

    # pixel grid (in pixel coordinates)
    grid_x = torch.arange(w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    grid_y = torch.arange(h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)

    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)  # [B,1,H,W]
    pos_x = grid_x + flow_u
    pos_y = grid_y + flow_v

    inside_x = (pos_x <= (w - 1)) & (pos_x >= 0)
    inside_y = (pos_y <= (h - 1)) & (pos_y >= 0)
    inside = inside_x & inside_y
    return inside  # bool [B,1,H,W]


def fbLoss(
    forward_flow,
    backward_flow,
    forward_gt_flow,
    backward_gt_flow,
    fb_loss_weight,
    image_warp_loss_weight=0,
    occ_weight=0,
    beta=255,
    first_image=None,
    second_image=None,
):
    """
    Forward-backward consistency loss + optional photometric warp loss.

    Keeps your original signature, but fixes a common issue:
      - cycle consistency should use predicted forward/backward flows
      - GT flows (if provided) are used for occlusion masking (more stable)
    """

    # outgoing masks (predicted flows)
    mask_fw = create_outgoing_mask(forward_flow).float()
    mask_bw = create_outgoing_mask(backward_flow).float()

    # --------
    # Cycle consistency on predicted flows (this is the main useful term)
    # --------
    backward_flow_warped = image_warp(backward_flow, forward_flow)  # f21(x + f12(x))
    forward_flow_warped  = image_warp(forward_flow, backward_flow)  # f12(x + f21(x))

    flow_diff_fw = forward_flow + backward_flow_warped
    flow_diff_bw = backward_flow + forward_flow_warped

    # --------
    # Occlusion masks from GT flows if available; otherwise fall back to predicted (no API change)
    # --------
    if forward_gt_flow is None:
        forward_gt_flow = forward_flow.detach()
    if backward_gt_flow is None:
        backward_gt_flow = backward_flow.detach()

    backward_flow_warped_gt = image_warp(backward_gt_flow, forward_gt_flow)
    forward_flow_warped_gt  = image_warp(forward_gt_flow, backward_gt_flow)

    flow_diff_fw_gt = forward_gt_flow + backward_flow_warped_gt
    flow_diff_bw_gt = backward_gt_flow + forward_flow_warped_gt

    # occlusion thresholds (classic)
    mag_sq_fw = length_sq(forward_gt_flow) + length_sq(backward_flow_warped_gt)
    mag_sq_bw = length_sq(backward_gt_flow) + length_sq(forward_flow_warped_gt)
    occ_thresh_fw = 0.01 * mag_sq_fw + 0.5
    occ_thresh_bw = 0.01 * mag_sq_bw + 0.5

    fb_occ_fw = (length_sq(flow_diff_fw_gt) > occ_thresh_fw).float()
    fb_occ_bw = (length_sq(flow_diff_bw_gt) > occ_thresh_bw).float()

    # remove occluded regions from masks
    mask_fw = mask_fw * (1.0 - fb_occ_fw)
    mask_bw = mask_bw * (1.0 - fb_occ_bw)

    occ_fw = 1.0 - mask_fw
    occ_bw = 1.0 - mask_bw

    # --------
    # Optional image warp photometric loss (only if enabled)
    # --------
    image_warp_loss = 0.0
    if image_warp_loss_weight != 0:
        if first_image is None or second_image is None:
            raise ValueError("first_image and second_image must be provided when image_warp_loss_weight != 0")

        second_image_warped = image_warp(second_image, forward_flow)   # frame2 -> frame1
        first_image_warped  = image_warp(first_image, backward_flow)   # frame1 -> frame2

        im_diff_fw = first_image - second_image_warped
        im_diff_bw = second_image - first_image_warped

        occ_loss = occ_weight * (charbonnier_loss(occ_fw, beta=1) + charbonnier_loss(occ_bw, beta=1))
        image_warp_loss = image_warp_loss_weight * (
            charbonnier_loss(im_diff_fw, mask_fw, beta=beta) +
            charbonnier_loss(im_diff_bw, mask_bw, beta=beta)
        ) + occ_loss

    # --------
    # Final FB consistency loss
    # --------
    fb_loss = fb_loss_weight * (
        charbonnier_loss(flow_diff_fw, mask_fw, beta=1) +
        charbonnier_loss(flow_diff_bw, mask_bw, beta=1)
    )

    return fb_loss + image_warp_loss

def edgeLoss(preds_edges, edges):
    """

    Args:
        preds_edges: with shape [b, c, h , w]
        edges: with shape [b, c, h, w]

    Returns: Edge losses

    """
    mask = (edges > 0.5).float()
    b, c, h, w = mask.shape
    num_pos = torch.sum(mask, dim=[1, 2, 3]).float()
    num_neg = c * h * w - num_pos
    neg_weights = (num_neg / (num_pos + num_neg)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    pos_weights = (num_pos / (num_pos + num_neg)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    weight = neg_weights * mask + pos_weights * (1 - mask)  # weight for debug
    losses = F.binary_cross_entropy_with_logits(preds_edges.float(), edges.float(), weight=weight, reduction='none')
    loss = torch.mean(losses)
    return loss

# Utility functions for converting between velocity and image formats.
def v2img_2d(velocity: torch.Tensor):
    ''' convert [B, H, W, chan] to [B, chan, H, W] '''
    return velocity.permute(0, 3, 1, 2)

def img2v_2d(image: torch.Tensor):
    ''' convert [B, chan, H, W] to [B, H, W, chan] '''
    return image.permute(0, 2, 3, 1)

@torch.jit.script
def _separable_filtering_conv(
    input_: torch.Tensor,
    kernels: List[torch.Tensor],
    pad_mode: str,
    spatial_dims: int,
    paddings: List[int],
    num_channels: int,
) -> torch.Tensor:

    # re-write from recursive to non-recursive for torch.jit to work
    # for d in range(spatial_dims-1, -1, -1):
    for d in range(spatial_dims):
        s = [1] * len(input_.shape)
        s[d + 2] = -1
        _kernel = kernels[d].reshape(s)
        # if filter kernel is unity, don't convolve
        if _kernel.numel() == 1 and _kernel[0] == 1:
            continue

        _kernel = _kernel.repeat([num_channels, 1] + [1] * spatial_dims)
        _padding = [0] * spatial_dims
        _padding[d] = paddings[d]
        _reversed_padding = _padding[::-1]

        # translate padding for input to torch.nn.functional.pad
        _reversed_padding_repeated_twice: list[list[int]] = [[p, p] for p in _reversed_padding]
        _sum_reversed_padding_repeated_twice: list[int] = []
        for p in _reversed_padding_repeated_twice:
            _sum_reversed_padding_repeated_twice.extend(p)
        # _sum_reversed_padding_repeated_twice: list[int] = sum(_reversed_padding_repeated_twice, [])

        padded_input = F.pad(input_, _sum_reversed_padding_repeated_twice, mode=pad_mode)
        # update input
        if spatial_dims == 1:
            input_ = F.conv1d(input=padded_input, weight=_kernel, groups=num_channels)
        elif spatial_dims == 2:
            input_ = F.conv2d(input=padded_input, weight=_kernel, groups=num_channels)
        elif spatial_dims == 3:
            input_ = F.conv3d(input=padded_input, weight=_kernel, groups=num_channels)
        else:
            raise NotImplementedError(f"Unsupported spatial_dims: {spatial_dims}.")
    return input_


def separable_filtering(x, kernels, mode='zeros') -> torch.Tensor:
    """
    Apply 1-D convolutions along each spatial dimension of `x`.
    Args:
        x: the input image. must have shape (batch, channels, H[, W, ...]).
        kernels: kernel along each spatial dimension.
            could be a single kernel (duplicated for all spatial dimensions), or
            a list of `spatial_dims` number of kernels.
        mode (string, optional): padding mode passed to convolution class. ``'zeros'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``. Default: ``'zeros'``. See ``torch.nn.Conv1d()`` for more information.
    Raises:
        TypeError: When ``x`` is not a ``torch.Tensor``.
    Examples:
    .. code-block:: python
        >>> import torch
        >>> img = torch.randn(2, 4, 32, 32)  # batch_size 2, channels 4, 32x32 2D images
        # applying a [-1, 0, 1] filter along each of the spatial dimensions.
        # the output shape is the same as the input shape.
        >>> out = separable_filtering(img, torch.tensor((-1., 0., 1.)))
        # applying `[-1, 0, 1]`, `[1, 0, -1]` filters along two spatial dimensions respectively.
        # the output shape is the same as the input shape.
        >>> out = separable_filtering(img, [torch.tensor((-1., 0., 1.)), torch.tensor((1., 0., -1.))])
    """

    # if not isinstance(x, torch.Tensor):
    #     raise TypeError(f"x must be a torch.Tensor but is {type(x).__name__}.")

    spatial_dims = len(x.shape) - 2
    if isinstance(kernels, torch.Tensor):
        kernels = [kernels] * spatial_dims
    _kernels = [s.to(x) for s in kernels]
    _paddings = [(k.shape[0] - 1) // 2 for k in _kernels]
    n_chs = x.shape[1]
    pad_mode = "constant" if mode == "zeros" else mode
    return _separable_filtering_conv(x, _kernels, pad_mode, spatial_dims, _paddings, n_chs)

class IFTLayer(torch.autograd.Function):
    ''' Implicit Function Theorem Layer '''
    @staticmethod
    def forward(ctx, warp, fixed_feature, moving_feature, grid, gaussian_grad, gaussian_warp):
        ctx.save_for_backward(warp, fixed_feature, moving_feature, grid, gaussian_grad, gaussian_warp)
        return warp
    
    @staticmethod
    def backward(ctx, grad_warp):
        ''' given dT/dw, we compute dT/dF '''
        warp, fixed_feature, moving_feature, grid, gaussian_grad, gaussian_warp = ctx.saved_tensors
        dims = warp.shape[-1]

        if gaussian_grad is not None:
            v2img = v2img_2d if dims == 2 else v2img_3d
            img2v = img2v_2d if dims == 2 else img2v_3d
            grad_warp = img2v(separable_filtering(v2img(grad_warp), gaussian_grad))
        
        # print_percentiles(torch.norm(grad_warp, p=2, dim=-1), 'grad_warp')

        # grad_warp = grad_result
        # return grad_warp, None, None, None, None, None
        with torch.enable_grad():
            # enable grad for warp if it does not exist
            if not warp.requires_grad:
                warp.requires_grad_(True)
            
            grid_sample_fn = grid_sample_2d if dims == 2 else grid_sample_3d

            moved_feature = grid_sample_fn(moving_feature, grid + warp, align_corners=align_corners)
            loss = F.mse_loss(moved_feature, fixed_feature)
            rho = torch.autograd.grad(loss, [warp], create_graph=True)[0]  # C(w, Ff, Fm)
            H = []
            for _ in range(dims):
                H.append(torch.autograd.grad(rho[..., _].sum(), [warp], create_graph=True)[0])
            H = torch.stack(H, dim=-1)
            # save 
            #eigvals = torch.linalg.eigvals(H).flatten().detach().cpu().numpy()
            #if not os.path.exists("./hessian_eig.npy"):
                #np.save("./hessian_eig.npy", eigvals)
            #input(".saved.")

            # print(H.shape, grad_warp.shape)
            if dims == 3:
                eye = 1e-4 * torch.eye(3, device=H.device)[None, None, None, None]
            else:
                eye = 1e-4 * torch.eye(2, device=H.device)[None, None, None]
            grad_warp_m = torch.linalg.solve(H + eye, grad_warp).detach()  # BHWDd
            # grad_warp_m = torch.linalg.lstsq(H, grad_warp).solution.detach()
            # print(grad_warp_m.shape)
            # print_percentiles(torch.norm(grad_warp_m, p=2, dim=-1), 'grad_warp_m')
            # multiply with d(rho)/dF 
            prod = torch.sum(grad_warp_m * rho)
            grad_fixed_feature = None
            grad_moving_feature = None
            if fixed_feature.requires_grad:
                grad_fixed_feature = torch.autograd.grad(-prod, [fixed_feature], create_graph=False)[0]  # BHWDc
                # print_percentiles(torch.norm(grad_fixed_feature, p=2, dim=1), 'grad_fixed_feature')
            if moving_feature.requires_grad:
                grad_moving_feature = torch.autograd.grad(-prod, [moving_feature], create_graph=False)[0]  # BHWDc
                # print_percentiles(torch.norm(grad_moving_feature, p=2, dim=1), 'grad_moving_feature')
                # grad_fixed_feature, grad_moving_feature = torch.autograd.grad(-prod, [fixed_feature, moving_feature], create_graph=False)[0]  # BHWDc
        # input("...")
        return grad_warp, grad_fixed_feature, grad_moving_feature, None, None, None

def multi_scale_warp_solver(
        fixed_features: List[torch.Tensor],
        moving_features: List[torch.Tensor],
        iterations: List[int],
        loss_function: Union[nn.Module, Callable],
        hessian_type: str = 'jfb',
        gaussian_warp: Optional[ItemOrList[torch.Tensor]] = None,
        gaussian_grad: Optional[ItemOrList[torch.Tensor]] = None,
        learning_rate: float = 3e-3,
        regularization: Optional[Callable] = None,
        debug: bool = False,
        beta1: float = 0.5,   # changing this from 0.9 to 0.5 to increase EMA decay for phantom steps to mimic actual optimization
        beta2: float = 0.99,
        eps: float = 1e-8,
        n_phantom_steps: int = 3,
        return_jacobian_norm: int = 1,  # how many estimators to compute
        phantom_step: str = 'sgd',   # choices = sgd, adam
        convergence_tol: int = 4,       # if loss increases for "C" iterations, abort
        convergence_eps: float = 1e-3,
        cfg: Optional[dict] = None,
        init_affine: Optional[torch.Tensor] = None,
        init_warp: Optional[torch.Tensor] = None,
        align_corners: bool = False,
):
    '''
    https://github.com/rohitrango/DIO/blob/4dddc40f16a35d6a63ff1e059d4a39b97a14f4ed/solver/diffeo.py
    Implements multi-scale SGD for warp fields with arbitrary feature images
    `fixed_features` contain  images of increasing resolutions of size [B, C_i, H_i, W_i, [D_i]] where i is the scale,
        and C_i is the number of channels at that scale
    '''
    hessian_type = hessian_type.lower()
    # collect statistics
    batch_size, shape = fixed_features[0].shape[0], fixed_features[0].shape[2:]
    n_dims = len(shape)
    # initialize flow
    v2img = v2img_2d if n_dims == 2 else v2img_3d
    img2v = img2v_2d if n_dims == 2 else img2v_3d

    if init_warp is None:
        warp = torch.zeros(
            (batch_size, *shape, n_dims),
            dtype=torch.float32,
            device=fixed_features[0].device,
        )
    else:
        # accept either [B,2,H,W] or [B,H,W,2]
        if init_warp.ndim == 4 and init_warp.shape[1] == n_dims:
            warp = init_warp.permute(0, 2, 3, 1).contiguous()
        else:
            warp = init_warp.contiguous()

        warp = warp.to(device=fixed_features[0].device, dtype=torch.float32)

        if warp.shape[1:-1] != shape:
            warp = img2v(
                F.interpolate(
                    v2img(warp),
                    size=shape,
                    mode='bilinear' if n_dims == 2 else 'trilinear',
                    align_corners=align_corners,
                )
            )


    exp_avg = torch.zeros_like(warp)
    exp_sq_avg = torch.zeros_like(warp)
    all_warps = []
    global_step = 1
    losses = []
    grid_sample_fn = (
        lambda x, g: F.grid_sample(
            x, g, mode='bilinear', padding_mode='border', align_corners=align_corners
        )
    )

    # iterate over scales
    # level is the level of iteration in the pyramid, i.e. max_levels = len(fixed_features) - 1

    for level, (iter_scale, (fixed_feature, moving_feature)) in enumerate(zip(iterations, zip(fixed_features, moving_features))):
        losses_lvl = []
        # initialize affine transform
        # this will typically have a gradient w.r.t. the affine parameters
        if init_affine is not None:
            pass
        else:
            init_affine = torch.eye(n_dims, n_dims+1, device=fixed_feature.device).unsqueeze(0).repeat(batch_size, 1, 1)
        # initialize grid
        grid = F.affine_grid(init_affine, fixed_feature.shape, align_corners=align_corners)
        # run optimization without grad
        warp.requires_grad_(True)
        exp_avg = exp_avg.detach()
        exp_sq_avg = exp_sq_avg.detach()
        # keep these variables to check for divergence and early-stop
        last_loss = math.inf
        iters_since_divergent = 0
        # run optimization
        with torch.no_grad():
            for step in range(1, iter_scale+1):
                # temporarily enable gradient here
                with torch.enable_grad():
                    moved_feature = F.grid_sample(moving_feature.detach(), grid + warp, align_corners=align_corners)
                    loss = loss_function(moved_feature, fixed_feature.detach())
                    if regularization is not None:
                        loss = loss + regularization(warp)
                    if debug:
                        losses_lvl.append(loss.item())
                    warp_grad = torch.autograd.grad(loss, warp)[0].detach()

                # divergence check
                lossitem = loss.item()
                # if lossitem > last_loss:
                rel_loss = lossitem/max(last_loss, 1e-8) - 1
                if rel_loss <= -convergence_eps:
                    ## (loss - loss_prev)/loss_prev should be negative, and should decrease by at least -eps each time
                    iters_since_divergent = 0
                else:
                    iters_since_divergent += 1
                    if iters_since_divergent >= convergence_tol:
                        break
                last_loss = lossitem
                # filtering
                if gaussian_grad is not None:
                    warp_grad = img2v(separable_filtering(v2img(warp_grad), gaussian_grad))
                # update SGD
                # now that we have warp grad, update exp_avg and exp_sq_avg
                if phantom_step == 'adam':
                    exp_avg.mul_(beta1).add_(warp_grad, alpha=1-beta1)
                    exp_sq_avg.mul_(beta2).addcmul_(warp_grad, warp_grad.conj(), value=1-beta2)
                    b1_correction = 1 - beta1 ** global_step
                    b2_correction = 1 - beta2 ** global_step
                    denom = (exp_sq_avg.sqrt() / math.sqrt(b2_correction)).add_(eps)
                    # # get updated gradient
                    warp_grad = exp_avg / b1_correction / denom
                    # # normalize
                    # gradmax = eps + warp_grad.norm(p=2, dim=-1, keepdim=True).flatten(1).max(1).values
                    # gradmax = gradmax.reshape(-1, *([1])*(n_dims+1))
                    # warp_grad = warp_grad / gradmax * half_res
                    # warp_grad.mul_(-learning_rate)
                    # update function
                    # warp_update = warp_grad + img2v(F.grid_sample(v2img(warp), grid + warp_grad, align_corners=align_corners))

                # If SGD, then we dont need all the postprocessing of the warp gradient
                warp_update = warp - learning_rate * warp_grad 
                # optionally smooth it
                if gaussian_warp is not None:
                    warp_update = img2v(separable_filtering(v2img(warp_update), gaussian_warp))
                warp.data.copy_(warp_update)
                global_step += 1

        # final step to capture gradient (here, warp = warp*)
        # warp.requires_grad_(False)
        jacobian_norm = torch.tensor(0).to(warp.device).float()
        if hessian_type == 'jfb':
            ### JFB: Jacobian-free backprop - essentially pretend to perform one-step optimization
            # simply perform another forward pass (with torch enabled grad)
            # moved_feature = F.grid_sample(moving_feature, grid + warp, align_corners=align_corners)
            for _ in range(n_phantom_steps):
                moved_feature = grid_sample_fn(moving_feature, grid + warp) 
                loss = loss_function(moved_feature, fixed_feature)
                if regularization is not None:
                    loss = loss + regularization(warp)
                if debug:
                    losses_lvl.append(loss.item())
                warp_grad = torch.autograd.grad(loss, warp, create_graph=True)[0]
                if gaussian_grad is not None:
                    warp_grad = img2v(separable_filtering(v2img(warp_grad), gaussian_grad))
                # we will NOT update exp_avg and exp_sq_avg here (we use the gradient directly as the update, and the sq_avg as the Hessian approximation)
                # now that we have warp grad, update exp_avg and exp_sq_avg

                ## Save this for jacobian norm
                warp_old = warp

                ### Algo 1: substitute exp_avg with warp_grad (doesnt work because the norm of warp_grad does not change)
                # warp_grad = no_backprop_mult(warp_grad, learning_rate / b1_correction / denom)

                ### Algo 2: find out the norm of updates of the warp, and rescale to that norm
                if phantom_step == 'sgd':
                    # oldnorm = (exp_avg / b1_correction / denom).norm() * learning_rate
                    # newnorm = warp_grad.norm()
                    # # add an extra term
                    # scale = (oldnorm / newnorm).item() # * min(1, (2**(max_levels - level)/4))
                    # # create new warp
                    # warp = warp - scale * warp_grad
                    warp = warp - learning_rate * warp_grad

                ### Algo 3: Perform the same update as in the iterations
                elif phantom_step == 'adam':
                    ### U2: I tried this as the update step
                    # exp_avg = beta1 * exp_avg + (1 - beta1) * warp_grad
                    # exp_sq_avg = beta2 * exp_sq_avg + (1 - beta2) * warp_grad * warp_grad
                    # denom = (exp_sq_avg / (1 - beta2 ** global_step)).sqrt() + eps
                    # warp_grad = exp_avg / (1 - beta1 ** global_step) / denom
                    # warp = warp - learning_rate * warp_grad

                    ### U1: Previously, I only matched the update equation without passing through the exp_avg and exp_sq_avg
                    ### U3: This works better!
                    warp_grad = warp_grad / b1_correction / denom * learning_rate
                    warp = warp - warp_grad
                else:
                    raise ValueError(f'Unknown phantom step {phantom_step}')

                if gaussian_warp is not None:
                    warp = img2v(separable_filtering(v2img(warp), gaussian_warp))
                # add to jac norm
                for _ in range(return_jacobian_norm):
                    v = torch.randn_like(warp)
                    vJ = torch.autograd.grad(warp, warp_old, v, create_graph=True, retain_graph=True)[0]
                    jacobian_norm = jacobian_norm + (vJ.norm()**2).mean() / v.numel() / return_jacobian_norm
            
        elif hessian_type == 'ift':
            # IFT layer here
            warp = IFTLayer.apply(warp, fixed_feature, moving_feature, grid, gaussian_grad, gaussian_warp)

        elif hessian_type == 'adam':
            raise NotImplementedError('Adam hessian not implemented yet')
        else:
            raise ValueError(f'Unknown hessian type {hessian_type}')

        # add this to all_warps 
        all_warps.append(warp)
        losses.append(losses_lvl)
        # interpolate for next stage
        if level != len(iterations) - 1:
            new_shape = fixed_features[level+1].shape[2:]
            warp = img2v(F.interpolate(v2img(warp.detach()), size=new_shape, mode='bilinear' if n_dims == 2 else 'trilinear', align_corners=align_corners))
            exp_avg = img2v(F.interpolate(v2img(exp_avg.detach()), size=new_shape, mode='bilinear' if n_dims == 2 else 'trilinear', align_corners=align_corners))
            exp_sq_avg = img2v(F.interpolate(v2img(exp_sq_avg.detach()), size=new_shape, mode='bilinear' if n_dims == 2 else 'trilinear', align_corners=align_corners))
    # return all_warps
    # print([len(x) for x in losses])
    if debug:
        return all_warps, losses, jacobian_norm
    else:
        return all_warps, jacobian_norm


def build_gauss_kernel(
    size: int = 5,
    sigma: float = 1.5,
    n_channels: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build a fixed depthwise Gaussian kernel of shape [C, 1, K, K].

    This is a constant filter, not a learnable parameter.
    """
    if size % 2 != 1:
        raise ValueError("kernel size must be odd")

    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()

    kernel = np.tile(kernel[None, None, :, :], (n_channels, 1, 1, 1))
    kernel = torch.tensor(kernel, device=device, dtype=dtype)
    return kernel


def conv_gauss(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    img:    [B, C, H, W]
    kernel: [C, 1, K, K]
    """
    _, _, kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    img = F.pad(img, (pad_w, pad_w, pad_h, pad_h), mode="replicate")
    return F.conv2d(img, kernel, groups=img.shape[1])


def laplacian_pyramid(img: torch.Tensor, kernel: torch.Tensor, max_levels: int = 3):
    current = img
    pyr = []

    for _ in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = F.avg_pool2d(filtered, kernel_size=2, stride=2)

    pyr.append(current)
    return pyr


class LapLoss(nn.Module):
    """
    Multi-scale Laplacian pyramid L1 loss.

    Supports:
      - [B, C, H, W]
      - [B, T, C, H, W]
    """

    def __init__(self, max_levels: int = 3, k_size: int = 5, sigma: float = 1.5):
        super().__init__()
        self.max_levels = int(max_levels)
        self.k_size = int(k_size)
        self.sigma = float(sigma)
        self.register_buffer("_gauss_kernel", torch.empty(0), persistent=False)

    def _get_kernel(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        need_new = (
            self._gauss_kernel.numel() == 0
            or self._gauss_kernel.shape[0] != c
            or self._gauss_kernel.device != x.device
            or self._gauss_kernel.dtype != x.dtype
        )
        if need_new:
            self._gauss_kernel = build_gauss_kernel(
                size=self.k_size,
                sigma=self.sigma,
                n_channels=c,
                device=x.device,
                dtype=x.dtype,
            )
        return self._gauss_kernel

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.ndim == 5:
            B, T, C, H, W = input.shape
            input = input.reshape(B * T, C, H, W)
            target = target.reshape(B * T, C, H, W)
        elif input.ndim != 4:
            raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(input.shape)}")

        if input.shape != target.shape:
            raise ValueError(f"input and target must match, got {input.shape} vs {target.shape}")

        kernel = self._get_kernel(input)
        pyr_input = laplacian_pyramid(input, kernel, self.max_levels)
        pyr_target = laplacian_pyramid(target, kernel, self.max_levels)

        loss = sum(F.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
        return loss
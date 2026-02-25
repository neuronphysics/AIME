import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache


def downsample_flow(flow_src: torch.Tensor, dst_hw: tuple[int, int]) -> torch.Tensor:
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

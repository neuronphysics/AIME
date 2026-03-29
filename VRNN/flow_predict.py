import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
# ----------------------------
# Helpers
# ----------------------------

def _is_first_step(flag) -> bool:
    """Robustly interpret 'flag' without device/dtype issues."""
    # CHANGED: original used torch.equal(flag, torch.ones(1)) which breaks on GPU/dtype mismatch
    if isinstance(flag, (bool, int)):
        return bool(flag)
    if torch.is_tensor(flag):
        if flag.numel() == 1:
            return bool(flag.detach().cpu().item())
    return False

def _broadcast_action(action, H, W, device, dtype):
    """
    action: [B, A] or [B, A, 1, 1] or [B, A, H, W]
    returns: [B, A, H, W]
    """
    if action is None:
        return None
    if not torch.is_tensor(action):
        action = torch.tensor(action, device=device, dtype=dtype)
    action = action.to(device=device, dtype=dtype)

    if action.dim() == 2:
        # [B, A] -> [B, A, 1, 1] -> [B, A, H, W]
        action = action[:, :, None, None]
    if action.dim() == 4 and action.shape[-2:] == (1, 1):
        action = action.expand(-1, -1, H, W)
    elif action.dim() == 4 and action.shape[-2:] == (H, W):
        pass
    else:
        raise ValueError(f"action must be [B,A] or [B,A,1,1] or [B,A,H,W], got {tuple(action.shape)}")
    return action

# Cache coordinate grids to avoid realloc every forward
_GRID_CACHE = {}

def coords_grid(batch, ht, wd, device, dtype):
    """coords in pixel units: [B,2,H,W] where channel0=x, channel1=y"""
    key = (batch, ht, wd, str(device), str(dtype))
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]
    # CHANGED: specify device/dtype and cache; original rebuilt each time and forced .to('cuda')
    y = torch.arange(ht, device=device, dtype=dtype)
    x = torch.arange(wd, device=device, dtype=dtype)
    # meshgrid indexing
    try:
        yy, xx = torch.meshgrid(y, x, indexing="ij")
    except TypeError:
        yy, xx = torch.meshgrid(y, x)
    coords = torch.stack([xx, yy], dim=0)  # [2,H,W]
    coords = coords[None].repeat(batch, 1, 1, 1)  # [B,2,H,W]
    _GRID_CACHE[key] = coords
    return coords

# ----------------------------
# Blocks 
# ----------------------------


class Hidden_state_conv(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1, bias=True)

    def forward(self, x):
        return torch.tanh(self.conv1(x))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=64, *, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def _cell(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return z, r, q

    def forward(self, h, x):
        if self.use_checkpoint and self.training and (h.requires_grad or x.requires_grad):
            z, r, q = checkpoint(self._cell, h, x, use_reentrant=False)
        else:
            z, r, q = self._cell(h, x)
        h = (1 - z) * h + z * q
        return h

class Encoder(nn.Module):
    """
    Your original "Encoder": updates a spatial hidden state from features.
    Now optionally action-conditioned by concatenation (broadcast).
    """
    def __init__(self, action_dim: int = 0, hidden_dim: int = 64):
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        self.hidden_state_conv = Hidden_state_conv(input_dim=64, hidden_dim=self.hidden_dim)
        self.E_gru = ConvGRU(hidden_dim=self.hidden_dim, input_dim=64 + self.action_dim, use_checkpoint= True)

    def forward(self, state, features, flag, action=None):
        """
        state:    [B,128,H,W] or dummy at first step
        features: [B, 64,H,W]
        flag:     scalar-like; first-step indicator
        action:   optional [B,A] or [B,A,1,1] or [B,A,H,W]
        """
        B, C, H, W = features.shape
        device, dtype = features.device, features.dtype

        if _is_first_step(flag):
            state = self.hidden_state_conv(features)

        # CHANGED: broadcast+concat action (no extra layer)
        if self.action_dim > 0 and action is not None:
            a_map = _broadcast_action(action, H, W, device, dtype)  # [B,A,H,W]
            x = torch.cat([features, a_map], dim=1)                 # [B,64+A,H,W]
        else:
            x = features

        state = self.E_gru(state, x)
        return state

def get_gru_encoder(action_dim: int = 0):
    # CHANGED: keep same function name, add optional action_dim with default (backwards compatible)
    return Encoder(action_dim=action_dim)

# ---- Decoder side ----

class StateFlowEncoder(nn.Module):
    """
    Encodes (inp/state-like feature) + current flow into a 62-d feature, then concatenates flow => 64 ch output.
    """
    def __init__(self, statedim=128, outdim=62):
        super().__init__()
        self.convc1 = nn.Conv2d(statedim, 64, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv   = nn.Conv2d(96, outdim, 3, padding=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, state, flow):
        state = self.relu(self.convc1(state))
        flo = self.relu(self.convf1(flow))
        flo = self.relu(self.convf2(flo))
        state_flo = torch.cat([state, flo], dim=1)
        out = self.relu(self.conv(state_flo))
        return torch.cat([out, flow], dim=1)  # 62 + 2 = 64

class FlowHead(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1, bias=True)
        self.gn = nn.GroupNorm(8, hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=False)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        return self.conv2(self.relu(self.gn(self.conv1(x))))

class Decoder(nn.Module):
    """
    Your original "Decoder": given feature [B,128,H,W] -> predicts future coordinate fields for fut steps.
    Now optionally conditions on action by concatenation (broadcast) WITHOUT adding new layers.
    """
    def __init__(self, future_len: int, action_dim: int = 0, hidden_dim: int = 64, use_checkpoint: bool =True):
        super().__init__()
        self.fut = int(future_len)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_checkpoint = bool(use_checkpoint)
        self.encode = StateFlowEncoder(statedim=64 + self.action_dim, outdim=62)

        # same channels as before
        self.D_gru  = ConvGRU(hidden_dim=64, input_dim=64, use_checkpoint= self.use_checkpoint)
        self.D_flow = FlowHead(input_dim=64, hidden_dim=self.hidden_dim)

    def _maybe_ckpt(self, fn, *args):
        if (
            self.use_checkpoint
            and self.training
            and any(torch.is_tensor(a) and a.requires_grad for a in args)
        ):
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, feature, action=None):
        """
        feature: [B,128,H,W], split into (state=64, inp=64)
        action:  optional [B,A] or [B,A,1,1] or [B,A,H,W]
        returns dict with 'f_flows': list of coords fields [B,2,H,W] length fut
        """
        B, C, H, W = feature.shape
        device, dtype = feature.device, feature.dtype

        # coords in pixel units
        coords0  = coords_grid(B, H, W, device=device, dtype=dtype).detach()
        coords1  = coords_grid(B, H, W, device=device, dtype=dtype).detach()

        # split
        state, inp = torch.split(feature, [64, 64], dim=1)
        state = torch.tanh(state)
        inp   = torch.relu(inp)

        # CHANGED: broadcast+concat action to 'inp' (no extra layer)
        if self.action_dim > 0 and action is not None:
            a_map = _broadcast_action(action, H, W, device, dtype)  # [B,A,H,W]
            inp_aug = torch.cat([inp, a_map], dim=1)                # [B,64+A,H,W]
        else:
            inp_aug = inp

        f_coords = []
        for _ in range(self.fut):
            # current flow in pixels
            flow = coords1 - coords0  # [B,2,H,W]
            # encode process: outputs 64 channels (62+flow)
            cat_encoded = self._maybe_ckpt(self.encode, inp_aug, flow)

            # recurrent update
            state = self.D_gru(state, cat_encoded)

            # delta coords update
            delta = self._maybe_ckpt(self.D_flow, state)    # [B,2,H,W]
            coords1 = coords1 + delta

            f_coords.append(coords1)

        return {"f_flows": f_coords}  # original

def get_gru_decoder(future_len: int, action_dim: int = 0):
    # CHANGED: keep same function name, add optional action_dim with default (backwards compatible)
    return Decoder(future_len=future_len, action_dim=action_dim)

# ============================================================
# Latent motion conditioning (attention -> latent flow -> warp)
# ============================================================

class ConvGNAct(nn.Module):
    """Conv -> GroupNorm -> SiLU """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=8, inplace_act=False):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.gn = nn.GroupNorm(g, out_ch)
        self.act = nn.SiLU(inplace=inplace_act)  # <= safer off when checkpointing

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class EdgeMapToTop(nn.Module):
    """
    Map an already-computed 1-channel edge map [B,1,H,W] to top grid
    [B,out_ch,topH,topW] using:

      1) average pooling to target grid
      2) normalize
      3) gamma boost
      4) normalize again
      5) trainable 1x1 projection
    """
    def __init__(
        self,
        out_ch: int = 16,
        target_hw=(8, 8),
        gamma: float = 0.8,
        use_checkpoint: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.target_hw = (int(target_hw[0]), int(target_hw[1]))
        self.gamma = float(gamma)
        self.use_checkpoint = bool(use_checkpoint)
        self.eps = float(eps)

        groups = min(8, int(out_ch))
        while groups > 1 and (int(out_ch) % groups != 0):
            groups -= 1

        self.proj = nn.Sequential(
            nn.Conv2d(1, int(out_ch), kernel_size=1, bias=True),
            nn.GroupNorm(groups, int(out_ch)),
            nn.SiLU(inplace=False),
        )

    def _maybe_ckpt(self, fn, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint(fn, x, use_reentrant=False)
        return fn(x)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.amax(dim=(-2, -1), keepdim=True) + self.eps)

    def _coerce_edge_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"edge map must be 4D, got {tuple(x.shape)}")

        # Accept [B,1,H,W] or [B,H,W,1]
        if x.shape[1] == 1:
            return x
        if x.shape[-1] == 1:
            return x.permute(0, 3, 1, 2).contiguous()

        raise ValueError(f"Expected [B,1,H,W] or [B,H,W,1], got {tuple(x.shape)}")

    def _pool_to_target(self, x: torch.Tensor) -> torch.Tensor:
        th, tw = self.target_hw
        h, w = x.shape[-2:]

        if (h, w) == (th, tw):
            return x

        if (h % th == 0) and (w % tw == 0):
            sy, sx = h // th, w // tw
            return F.avg_pool2d(x, kernel_size=(sy, sx), stride=(sy, sx))

        return F.adaptive_avg_pool2d(x, self.target_hw)

    def forward(self, edge_map: torch.Tensor) -> torch.Tensor:
        edge_map = self._coerce_edge_shape(edge_map)

        # 1) map to top grid with avg pooling
        e_top = self._pool_to_target(edge_map)

        # 2) normalize -> gamma boost -> normalize again
        e_top = self._normalize(e_top).clamp_min(self.eps).pow(self.gamma)
        e_top = self._normalize(e_top)

        # 3) always use trainable projection
        return self._maybe_ckpt(self.proj, e_top)


class PixelShuffleUpBlock(nn.Module):
    def __init__(self, channels: int, *, use_fir_blur: bool = True):
        super().__init__()
        self.channels = int(channels)
        self.use_fir_blur = bool(use_fir_blur)

        # sub-pixel conv
        self.conv = nn.Conv2d(self.channels, 4 * self.channels, kernel_size=3, padding=1, bias=True)
        self.icnr_init_(self.conv, scale=2)  # <- key

        self.ps = nn.PixelShuffle(2)

        if self.use_fir_blur:
            # StyleGAN-ish separable FIR filter: [1,3,3,1]
            # 1D normalized by 8; outer product normalized by 64
            f = torch.tensor([1.0, 3.0, 3.0, 1.0])
            k2d = torch.outer(f, f) / 64.0  # 4x4
            # depthwise conv kernel: [C,1,4,4]
            self.register_buffer("fir_kernel", k2d[None, None, :, :].repeat(self.channels, 1, 1, 1))

        # post-upsample stabilization
        g = min(32, self.channels)
        if self.channels % g != 0:
            # safe fallback (rare unless channels is weird)
            g = 1
        self.norm = nn.GroupNorm(num_groups=g, num_channels=self.channels)
        self.act = nn.SiLU(inplace=True)

    @staticmethod
    def icnr_init_(conv: nn.Conv2d, scale: int = 2, nonlinearity: str = "relu") -> None:
        with torch.no_grad():
            out_c, in_c, kH, kW = conv.weight.shape
            assert out_c % (scale * scale) == 0, "out_channels must be divisible by scale^2"
            inner_c = out_c // (scale * scale)

            sub = torch.empty((inner_c, in_c, kH, kW), device=conv.weight.device, dtype=conv.weight.dtype)
            nn.init.kaiming_normal_(sub, nonlinearity=nonlinearity)

            conv.weight.copy_(sub.repeat(scale * scale, 1, 1, 1))
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ps(self.conv(x))

        if self.use_fir_blur:
            # replicate pad by 1 on all sides for 4x4 kernel
            x = F.pad(x, (1, 2, 1, 2), mode="replicate")  # (left,right,top,bottom)
            k = self.fir_kernel.to(device=x.device, dtype=x.dtype)
            x = F.conv2d(x, k, groups=self.channels)

        x = self.act(self.norm(x))
        return x

class HFTopUpsampler(nn.Module):
    """
    Learned upsampler: [B,C,topH,topW] -> [B,C,target,target] using repeated 2x PixelShuffle blocks.

    Works when:
      - target % topH == 0 and target % topW == 0
      - and scale factor is power-of-2 (e.g., 4->64, 8->64, 16->64, 32->64).

    If top size varies (4 or 8 or 16...), create enough blocks for the smallest top you expect
    and we will run only the required number at runtime.
    """
    def __init__(self, channels: int, target: int = 64, min_top: int = 4, use_checkpoint: bool = False):
        super().__init__()
        assert target % min_top == 0, "target must be divisible by min_top"
        max_scale = target // min_top
        assert max_scale > 0 and (max_scale & (max_scale - 1)) == 0, "target/min_top must be power of two for repeated x2 blocks"

        self.channels = int(channels)
        self.target = int(target)
        self.use_checkpoint = bool(use_checkpoint)

        self.max_steps = int(math.log2(max_scale))
        self.blocks = nn.ModuleList([PixelShuffleUpBlock(self.channels) for _ in range(self.max_steps)])

        g = min(32, self.channels)
        while g > 1 and (self.channels % g != 0):
            g -= 1
        # optional final “polish” conv to reduce any remaining checkerboard / subpixel texture
        self.final = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=g, num_channels=self.channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H == self.target and W == self.target:
            return x
        if C != self.channels:
            raise ValueError(f"Expected C={self.channels}, got C={C}")

        if (self.target % H) != 0 or (self.target % W) != 0:
            raise ValueError(f"target={self.target} must be divisible by input spatial {(H,W)}")

        rH = self.target // H
        rW = self.target // W
        if rH != rW:
            raise ValueError(f"Need equal scale for H and W; got rH={rH}, rW={rW}")
        if not (rH > 0 and (rH & (rH - 1)) == 0):
            raise ValueError(f"Scale factor must be power-of-two, got {rH}")

        steps = int(math.log2(rH))
        if steps > self.max_steps:
            raise ValueError(
                f"Need {steps} upsample steps but module was built for max_steps={self.max_steps}. "
                f"Set min_top smaller (or increase target/min_top)."
            )

        for i in range(steps):
            blk = self.blocks[i]
            # checkpoint only makes sense during training and when grads are enabled
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        x = self.final(x)
        return x
        

class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """

    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out



def sinusoidal_2d_pos_embed(H: int, W: int, D: int, device=None, dtype=None) -> torch.Tensor:
    """
    Deterministic 2D sin/cos positional embedding.
    Returns: [1, H*W, D]
    """
    assert D % 4 == 0, "token_dim must be divisible by 4 for 2D sinusoidal embedding."
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32

    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")  # [H,W]
    yy = yy.reshape(-1)  # [H*W]
    xx = xx.reshape(-1)

    half = D // 2
    freq = torch.exp(
        torch.arange(0, half, 2, device=device, dtype=dtype) * (-math.log(10000.0) / half)
    )  # [half/2]

    emb_x = torch.cat([torch.sin(xx[:, None] * freq[None, :]),
                       torch.cos(xx[:, None] * freq[None, :])], dim=1)
    emb_y = torch.cat([torch.sin(yy[:, None] * freq[None, :]),
                       torch.cos(yy[:, None] * freq[None, :])], dim=1)
    emb = torch.cat([emb_x, emb_y], dim=1)  # [H*W, D]
    return emb[None, :, :]


class SEBlock2d(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        hidden = max(1, c // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=True),
            nn.SiLU(inplace=False),
            nn.Conv2d(hidden, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Conv2d(query_dim, inner_dim, 1, bias=False)
        self.to_k = nn.Conv2d(context_dim, inner_dim, 1, bias=False)
        self.to_v = nn.Conv2d(context_dim, inner_dim, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, query_dim, 1, bias=True)

    def forward(self, x, context):
        """
        x:       [B, Cq, H, W]   query map
        context: [B, Cc, H, W]   context map
        returns: [B, Cq, H, W]
        """
        B, _, H, W = x.shape
        h = self.heads
        d = self.dim_head

        q = self.to_q(x)        # [B, h*d, H, W]
        k = self.to_k(context)  # [B, h*d, H, W]
        v = self.to_v(context)  # [B, h*d, H, W]

        # [B, h, HW, d]
        q = q.view(B, h, d, H * W).permute(0, 1, 3, 2)
        k = k.view(B, h, d, H * W).permute(0, 1, 3, 2)
        v = v.view(B, h, d, H * W).permute(0, 1, 3, 2)

        # attention scores: [B, h, HW, HW]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # [B, h, HW, d]
        out = torch.matmul(attn, v)

        # [B, h*d, H, W]
        out = out.permute(0, 1, 3, 2).contiguous().view(B, h * d, H, W)

        return self.to_out(out)


class LatentFlowHead(nn.Module):
    """Predict coarse canonical flow on the top grid in normalized coordinates."""

    def __init__(
        self,
        mot_ch: int,
        ctx_ch: int,
        act_dim: int,
        edge_ch: int,
        top_hw: Tuple[int, int],
        hidden: int = 128,
        max_flow_top_px: float = 1.0,
        use_checkpoint: bool = False,
        attn_heads: int = 8,
        attn_dim_head: int = 32,
    ):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        topH, topW = int(top_hw[0]), int(top_hw[1])

        max_dx_norm = 2.0 * float(max_flow_top_px) / max(topW - 1, 1)
        max_dy_norm = 2.0 * float(max_flow_top_px) / max(topH - 1, 1)
        self.register_buffer(
            'max_flow_norm',
            torch.tensor([max_dx_norm, max_dy_norm], dtype=torch.float32).view(1, 2, 1, 1),
            persistent=False,
        )

        self.act_embed = nn.Sequential(
            nn.Conv2d(act_dim, 32, 1, 1, 0),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=False),
        )

        self.motion_hctx_attn = CrossAttention(
            query_dim=mot_ch,
            context_dim=ctx_ch,
            heads=attn_heads,
            dim_head=attn_dim_head,
        )

        g_mot = min(8, mot_ch)
        while mot_ch % g_mot != 0 and g_mot > 1:
            g_mot -= 1
        self.mot_fuse_norm = nn.GroupNorm(g_mot, mot_ch)

        in_ch = mot_ch + 32 + edge_ch
        self.net = nn.Sequential(
            ConvGNAct(in_ch, hidden, 3, 1, inplace_act=False),
            ConvGNAct(hidden, hidden, 3, 1, inplace_act=False),
            ConvGNAct(hidden, hidden, 3, 1, inplace_act=False),
            nn.Conv2d(hidden, 2, 3, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _maybe_ckpt(self, fn, *args):
        if self.use_checkpoint and self.training and any(torch.is_tensor(a) and a.requires_grad for a in args):
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _cross_attend_motion(self, mot_ctx, h_context_map):
        return self.motion_hctx_attn(mot_ctx, h_context_map)

    def forward(self, mot_ctx, h_context_map, a_map, edge_top):
        a_emb = self._maybe_ckpt(self.act_embed, a_map)
        mot_ctx_attn = self._maybe_ckpt(self._cross_attend_motion, mot_ctx, h_context_map)
        mot_ctx_fused = self.mot_fuse_norm(mot_ctx + mot_ctx_attn)
        x = torch.cat([mot_ctx_fused, a_emb, edge_top], dim=1)
        if self.use_checkpoint and self.training and x.requires_grad:
            raw = checkpoint_sequential(self.net, segments=3, input=x, use_reentrant=False)
        else:
            raw = self.net(x)
        return self.max_flow_norm.to(device=raw.device, dtype=raw.dtype) * torch.tanh(raw)

def resize_flow_norm(
    flow_norm: torch.Tensor,
    dst_hw: Tuple[int, int],
    mode: str = 'bicubic',
) -> torch.Tensor:
    """Resample a canonical flow field in normalized coordinates."""
    assert flow_norm.ndim == 4 and flow_norm.size(1) == 2
    Hd, Wd = int(dst_hw[0]), int(dst_hw[1])
    if flow_norm.shape[-2:] == (Hd, Wd):
        return flow_norm
    if mode not in {'bilinear', 'bicubic'}:
        raise ValueError(f'Unsupported mode={mode}')
    return F.interpolate(flow_norm, size=(Hd, Wd), mode=mode, align_corners=True)



class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features[0], 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features[i], hidden_features[i + 1], 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features[-1], out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features[-1]) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features[-1]) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features[-1], out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output     
    
    
class FineResidualFlowHead(nn.Module):
    """
    Predict a fine residual in canonical normalized coordinates using:
      - local fine feature context
      - bicubic coarse base flow at target resolution
      - 2 sub-cell coordinate channels (relative position inside each coarse cell)

    Then project the residual to the high-pass complement of the coarse bicubic
    function class, preserving the original cross-scale design.
    """

    def __init__(
        self,
        in_ch: int,
        fine_hw: Tuple[int, int],
        coarse_hw: Tuple[int, int],
        hidden: int = 64,
        max_fine_flow_norm: float = 0.05,
        use_checkpoint: bool = False,
        siren_omega_0: float = 30.0,
    ):
        super().__init__()
        self.fine_H, self.fine_W = int(fine_hw[0]), int(fine_hw[1])
        self.coarse_H, self.coarse_W = int(coarse_hw[0]), int(coarse_hw[1])
        self.use_checkpoint = bool(use_checkpoint)

        self.register_buffer(
            "max_norm",
            torch.tensor(float(max_fine_flow_norm), dtype=torch.float32),
            persistent=False,
        )

        def _groups(c: int) -> int:
            g = min(8, c)
            while c % g != 0 and g > 1:
                g -= 1
            return g

        # Local spatial context extractor.
        # It sees both the fine feature and the bicubic coarse base flow.
        stem_in = in_ch + 2
        g = _groups(hidden)
        self.local_ctx = nn.Sequential(
            nn.Conv2d(stem_in, hidden, 3, 1, 1, bias=True),
            nn.GroupNorm(g, hidden),
            nn.SiLU(inplace=False),

            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=True),
            nn.GroupNorm(g, hidden),
            nn.SiLU(inplace=False),

            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=True),
            nn.GroupNorm(g, hidden),
            nn.SiLU(inplace=False),
        )

        # Per-pixel implicit decoder.
        # Inputs = local context + 2 sub-cell coords + base flow (dx,dy).
        siren_hidden = [hidden, hidden, hidden * 2]
        self.imnet = Siren(
            in_features=hidden + 2 + 2,
            hidden_features=siren_hidden,
            hidden_layers=len(siren_hidden) - 1,
            out_features=2,
            outermost_linear=True,
            first_omega_0=siren_omega_0,
            hidden_omega_0=siren_omega_0,
        )

        # Make the new branch start near zero so it does not immediately fight
        # the existing coarse path.
        if hasattr(self.imnet, "net") and len(self.imnet.net) > 0 and isinstance(self.imnet.net[-1], nn.Linear):
            nn.init.zeros_(self.imnet.net[-1].weight)
            nn.init.zeros_(self.imnet.net[-1].bias)

    def _project_highpass(self, residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual_low = resize_flow_norm(residual, (self.coarse_H, self.coarse_W), mode="bicubic")
        residual_recon = resize_flow_norm(residual_low, (self.fine_H, self.fine_W), mode="bicubic")
        residual_high = residual - residual_recon
        leakage = resize_flow_norm(residual_high, (self.coarse_H, self.coarse_W), mode="bicubic")
        return residual_high, leakage

    def _subcell_coords(self, batch: int, device, dtype) -> torch.Tensor:
        """
        Returns [B,2,H,W] with coordinates in [-1,1] measuring each fine pixel's
        relative position *inside its coarse top-grid cell*.

        This is more appropriate than absolute x/y for coarse->fine lifting.
        """
        y = torch.arange(self.fine_H, device=device, dtype=dtype) + 0.5
        x = torch.arange(self.fine_W, device=device, dtype=dtype) + 0.5

        # Map fine-pixel centers into coarse-grid coordinates.
        coarse_y = y * (float(self.coarse_H) / float(self.fine_H)) - 0.5
        coarse_x = x * (float(self.coarse_W) / float(self.fine_W)) - 0.5

        frac_y = coarse_y - torch.floor(coarse_y)
        frac_x = coarse_x - torch.floor(coarse_x)

        rel_y = 2.0 * frac_y - 1.0
        rel_x = 2.0 * frac_x - 1.0

        yy, xx = torch.meshgrid(rel_y, rel_x, indexing="ij")
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0)  # [1,2,H,W]
        return coords.expand(batch, -1, -1, -1)

    def _forward_impl(self, feat: torch.Tensor, base_target_norm: torch.Tensor) -> torch.Tensor:
        B = feat.shape[0]

        if feat.shape[-2:] != (self.fine_H, self.fine_W):
            feat = F.interpolate(
                feat, size=(self.fine_H, self.fine_W),
                mode="bilinear", align_corners=True
            )

        if base_target_norm.shape[-2:] != (self.fine_H, self.fine_W):
            base_target_norm = resize_flow_norm(
                base_target_norm, (self.fine_H, self.fine_W), mode="bicubic"
            )

        # Local conv context over [fine feature, bicubic coarse base flow]
        ctx = self.local_ctx(torch.cat([feat, base_target_norm], dim=1))  # [B,C,H,W]

        # Relative coordinates inside coarse cell
        subcell = self._subcell_coords(B, feat.device, feat.dtype)        # [B,2,H,W]

        # Per-pixel implicit decoding
        # Inputs: local ctx + subcell coords + base flow at that pixel
        q = torch.cat([ctx, subcell, base_target_norm], dim=1)            # [B,C+4,H,W]
        q = q.permute(0, 2, 3, 1).reshape(B * self.fine_H * self.fine_W, -1)

        raw = self.imnet(q)                                               # [B*H*W,2]
        raw = raw.view(B, self.fine_H, self.fine_W, 2).permute(0, 3, 1, 2).contiguous()
        return raw

    def forward(
        self,
        feat: torch.Tensor,
        base_target_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self.use_checkpoint
            and self.training
            and (feat.requires_grad or base_target_norm.requires_grad)
        ):
            raw = checkpoint(self._forward_impl, feat, base_target_norm, use_reentrant=False)
        else:
            raw = self._forward_impl(feat, base_target_norm)

        residual = self.max_norm.to(device=raw.device, dtype=raw.dtype) * torch.tanh(raw)
        residual_high, leakage = self._project_highpass(residual)
        return residual_high, leakage

def flow_norm_to_px(flow_norm: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """Convert normalized align_corners=True flow to pixel units."""
    assert flow_norm.ndim == 4 and flow_norm.size(1) == 2
    H, W = int(hw[0]), int(hw[1])

    sx = max(W - 1, 1) / 2.0
    sy = max(H - 1, 1) / 2.0

    out = flow_norm.clone()
    out[:, 0] = out[:, 0] * sx
    out[:, 1] = out[:, 1] * sy
    return out


class CanonicalFlowField(nn.Module):
    """
    Canonical cross-scale flow module.
    The top-grid flow is the only coarse motion predictor.
    The target-resolution flow is deterministic bicubic lift plus an optional
    strictly high-frequency residual.
    """

    def __init__(
        self,
        mot_ch: int,
        ctx_ch: int,
        act_dim: int,
        edge_ch: int,
        fine_feat_ch: int,
        top_hw: Tuple[int, int],
        target_hw: Tuple[int, int],
        coarse_hidden: int = 128,
        max_flow_top_px: float = 1.0,
        use_checkpoint: bool = False,
        fine_hw: Optional[Tuple[int, int]] = None,
        fine_hidden: int = 64,
        max_fine_flow_px: float = 0.5,
    ):
        super().__init__()
        self.top_hw = (int(top_hw[0]), int(top_hw[1]))
        self.target_hw = (int(target_hw[0]), int(target_hw[1]))

        self.coarse_head = LatentFlowHead(
            mot_ch=mot_ch,
            ctx_ch=ctx_ch,
            act_dim=act_dim,
            edge_ch=edge_ch,
            top_hw=self.top_hw,
            hidden=coarse_hidden,
            max_flow_top_px=max_flow_top_px,
            use_checkpoint=use_checkpoint,
        )

        fine_hw = self.target_hw if fine_hw is None else (int(fine_hw[0]), int(fine_hw[1]))
        self.fine_head = FineResidualFlowHead(
            in_ch=fine_feat_ch,
            fine_hw=fine_hw,
            coarse_hw=self.top_hw,
            hidden=fine_hidden,
            max_fine_flow_norm=max_fine_flow_px,
            use_checkpoint=use_checkpoint,
        )

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask_t: Optional[torch.Tensor]) -> torch.Tensor:
        if mask_t is None:
            return x.mean()
        vm = mask_t.view(mask_t.size(0), 1, 1, 1).to(device=x.device, dtype=x.dtype)
        denom = (vm.sum() * x.shape[1] * x.shape[2] * x.shape[3]).clamp(min=1.0)
        return (x * vm).sum() / denom

    def forward(
        self,
        mot_ctx: torch.Tensor,
        h_ctx: torch.Tensor,
        a_map: torch.Tensor,
        edge_top: torch.Tensor,
        fine_feat: torch.Tensor,
        mask_t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        flow_top_norm = self.coarse_head(mot_ctx, h_ctx, a_map, edge_top)
        base_target_norm = resize_flow_norm(flow_top_norm, self.target_hw, mode="bicubic")

        fine_res_norm, fine_leak = self.fine_head(fine_feat, base_target_norm)
        fine_res_target_norm = resize_flow_norm(fine_res_norm, self.target_hw, mode="bicubic")
        flow_target_norm = base_target_norm + fine_res_target_norm
        fine_null_loss = self._masked_mean(fine_leak.abs(), mask_t)

        flow_back_to_top = resize_flow_norm(flow_target_norm, self.top_hw, mode="bicubic")
        cross_scale_cycle_loss = self._masked_mean((flow_back_to_top - flow_top_norm).abs(), mask_t) + self._masked_mean((flow_target_norm - base_target_norm).abs(), mask_t)

        flow_top_px = flow_norm_to_px(flow_top_norm, self.top_hw)
        flow_target_px = flow_norm_to_px(flow_target_norm, self.target_hw)
        base_target_px = flow_norm_to_px(base_target_norm, self.target_hw)

        if mask_t is not None:
            vm = mask_t.view(mask_t.size(0), 1, 1, 1).to(device=flow_top_px.device, dtype=flow_top_px.dtype)
            flow_top_px = flow_top_px * vm
            flow_target_px = flow_target_px * vm
            base_target_px = base_target_px * vm
            flow_top_norm = flow_top_norm * vm
            flow_target_norm = flow_target_norm * vm
            base_target_norm = base_target_norm * vm
            fine_res_norm = fine_res_norm * vm
            fine_res_target_norm = fine_res_target_norm * vm

        return {
            "flow_top_norm": flow_top_norm,
            "flow_top_px": flow_top_px,
            "flow_target_norm": flow_target_norm,
            "flow_target_px": flow_target_px,
            "base_target_norm": base_target_norm,
            "base_target_px": base_target_px,
            "fine_res_norm": fine_res_norm,
            "fine_res_target_norm": fine_res_target_norm,
            "fine_null_loss": fine_null_loss,
            "cross_scale_cycle_loss": cross_scale_cycle_loss,
        }

class ConvPatchTokenizer(nn.Module):
    """
    Patchify a spatial map into tokens using strided conv (NO pooling, NO interpolation).
    Input:  x [B,C,H,W]
    Output: tokens [B,N,D], where N = Ht*Wt, Ht=H//patch, Wt=W//patch
    """
    def __init__(self, in_ch: int, d_tok: int, patch: int, use_checkpoint: bool = False, gn_groups: int = 8):
        super().__init__()
        self.patch = int(patch)
        self.d_tok = int(d_tok)
        self.use_checkpoint = bool(use_checkpoint)

        # No Lazy: in_ch must be known.
        self.conv = nn.Conv2d(int(in_ch), self.d_tok,
                              kernel_size=self.patch, stride=self.patch,
                              padding=0, bias=False)

        g = min(gn_groups, self.d_tok)
        while self.d_tok % g != 0 and g > 1:
            g -= 1
        self.gn = nn.GroupNorm(g, self.d_tok)
        self.act = nn.SiLU(inplace=False)
        self._xy_cache = {}

    def _maybe_ckpt(self, fn, *args):
        if self.use_checkpoint and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _get_xy_tok(self, Ht: int, Wt: int, device):
        key = (Ht, Wt, str(device))
        if key in self._xy_cache:
            return self._xy_cache[key]
        xs = torch.arange(Wt, device=device, dtype=torch.float32)
        ys = torch.arange(Ht, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).unsqueeze(0)  # [1,N,2]
        self._xy_cache[key] = xy
        return xy

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        assert H % self.patch == 0 and W % self.patch == 0, (H, W, self.patch)

        def _stem(inp):
            h = self.conv(inp)
            h = self.act(self.gn(h))
            return h

        h = self._maybe_ckpt(_stem, x)  # [B,D,Ht,Wt]
        Ht, Wt = h.shape[-2], h.shape[-1]
        tokens = h.flatten(2).transpose(1, 2).contiguous()  # [B,N,D]
        xy_tok = self._get_xy_tok(Ht, Wt, device=x.device)  # [1,N,2]
        return tokens, (Ht, Wt), xy_tok


class TokenTransporter(nn.Module):
    """
    Cross-attention transport:
      query = tokens_prev, key/value = tokens_cur
    Adds a displacement prior as an attention *bias*.
    Stores attention entropy in self.last_attn_entropy (scalar tensor) for logging.
    """
    def __init__(self, d_tok: int, n_heads: int = 4, sigma: float = 1.5, use_checkpoint: bool = False):
        super().__init__()
        assert d_tok % n_heads == 0, (d_tok, n_heads)
        self.d_tok = int(d_tok)
        self.n_heads = int(n_heads)
        self.d_head = self.d_tok // self.n_heads
        self.sigma = float(sigma)
        self.use_checkpoint = bool(use_checkpoint)

        self.q_proj = nn.Linear(self.d_tok, self.d_tok, bias=False)
        self.k_proj = nn.Linear(self.d_tok, self.d_tok, bias=False)
        self.v_proj = nn.Linear(self.d_tok, self.d_tok, bias=False)
        self.o_proj = nn.Linear(self.d_tok, self.d_tok, bias=False)

        self.ln_q = nn.LayerNorm(self.d_tok)
        self.ln_kv = nn.LayerNorm(self.d_tok)

        self._key_xy_cache = {}  # (Ht,Wt,device_str)->xy [Nk,2]

    def _maybe_ckpt(self, fn, *args):
        if self.use_checkpoint and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _get_key_xy(self, Ht: int, Wt: int, device):
        key = (int(Ht), int(Wt), str(device))
        if key in self._key_xy_cache:
            return self._key_xy_cache[key]
        xs = torch.arange(Wt, device=device, dtype=torch.float32)
        ys = torch.arange(Ht, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # [N,2]
        self._key_xy_cache[key] = xy
        return xy

    def forward(
        self,
        tokens_prev: torch.Tensor,  # [B,N,D]
        tokens_cur: torch.Tensor,   # [B,N,D]
        *,
        dest_xy_prior: torch.Tensor = None,  # [B,N,2] in token units (float)
        token_hw: tuple = None,             # (Ht,Wt)
    ) -> torch.Tensor:
        B, N, D = tokens_prev.shape
        assert tokens_cur.shape[:2] == (B, N), tokens_cur.shape
        assert tokens_cur.shape[2] == D, tokens_cur.shape

        def _attn(prev, cur):
            q_in = self.ln_q(prev)
            kv_in = self.ln_kv(cur)

            q = self.q_proj(q_in)
            k = self.k_proj(kv_in)
            v = self.v_proj(kv_in)

            q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,N,dh]
            k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
            v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,N,N]

            if dest_xy_prior is not None:
                assert token_hw is not None, "token_hw=(Ht,Wt) required when using dest_xy_prior"
                Ht, Wt = int(token_hw[0]), int(token_hw[1])
                key_xy = self._get_key_xy(Ht, Wt, device=scores.device)  # [N,2]
                dx = dest_xy_prior[..., 0].unsqueeze(-1) - key_xy[:, 0].view(1, 1, N)
                dy = dest_xy_prior[..., 1].unsqueeze(-1) - key_xy[:, 1].view(1, 1, N)
                dist2 = dx * dx + dy * dy
                bias = -torch.sqrt(dist2  + torch.finfo(torch.float32).eps) / self.sigma # [B,N,N]
                scores = scores + bias.unsqueeze(1)  # broadcast over heads

            attn = torch.softmax(scores, dim=-1)

            out = torch.matmul(attn, v)  # [B,H,N,dh]
            out = out.transpose(1, 2).contiguous().view(B, N, D)
            out = self.o_proj(out)
            return out

        return self._maybe_ckpt(_attn, tokens_prev, tokens_cur)


class TokensToTopMap(nn.Module):
    """
    Project transported tokens -> top-grid feature map [B,C_top,topH,topW]
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        token_hw: tuple[int, int],
        top_hw: tuple[int, int],
        use_checkpoint: bool = False,
        gn_groups: int = 8,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.use_checkpoint = bool(use_checkpoint)

        self.token_hw = (int(token_hw[0]), int(token_hw[1]))
        self.top_hw   = (int(top_hw[0]),   int(top_hw[1]))

        # [B,D,Ht,Wt] -> [B,out_ch,Ht,Wt]
        self.proj0 = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, stride=1, padding=0, bias=False)

        g = min(gn_groups, self.out_ch)
        while self.out_ch % g != 0 and g > 1:
            g -= 1
        self.gn0 = nn.GroupNorm(g, self.out_ch)
        self.act = nn.SiLU(inplace=False)

        # Prebuild downsamplers (stride-2 convs) so nothing is created in forward.
        self.ds_blocks = nn.ModuleList()
        Ht, Wt = self.token_hw
        topH, topW = self.top_hw

        # This implementation downsamples H and W together with stride=2,
        # so ratios must be powers of 2 and match for H and W.
        assert Ht % topH == 0 and Wt % topW == 0, (self.token_hw, self.top_hw)
        rH = Ht // topH
        rW = Wt // topW
        assert rH == rW, f"Need same downsample ratio for H/W, got rH={rH}, rW={rW}"
        assert (rH & (rH - 1)) == 0, f"Downsample ratio must be power of 2, got {rH}"

        curH, curW = Ht, Wt
        while (curH > topH) or (curW > topW):
            self.ds_blocks.append(
                nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, stride=2, padding=1, bias=False)
            )
            curH //= 2
            curW //= 2

    def _maybe_ckpt(self, fn, *args):
        if self.use_checkpoint and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N, D]
        B, N, D = tokens.shape
        Ht, Wt = self.token_hw
        topH, topW = self.top_hw
        assert D == self.in_ch, (D, self.in_ch)
        assert N == Ht * Wt, (N, Ht, Wt)

        x = tokens.transpose(1, 2).contiguous().view(B, D, Ht, Wt)  # [B,D,Ht,Wt]

        def _p0(inp):
            h = self.proj0(inp)
            h = self.act(self.gn0(h))
            return h

        x = self._maybe_ckpt(_p0, x)

        for conv in self.ds_blocks:
            x = self._maybe_ckpt(conv, x)

        assert x.shape[-2:] == (topH, topW), (x.shape, (topH, topW))
        return x

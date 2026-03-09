import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
class FlowUpsampleMask(nn.Module):
    def __init__(self, in_ch: int, scale: int, hidden: int = 128, use_checkpoint: bool = False):
        super().__init__()
        self.scale = int(scale)
        self.use_checkpoint = bool(use_checkpoint)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 9 * self.scale * self.scale, 1, padding=0),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat):
        if self.use_checkpoint and self.training and feat.requires_grad:
            return checkpoint(self.net, feat, use_reentrant=False)
        else:
            return self.net(feat)

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
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

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

        # optional final “polish” conv to reduce any remaining checkerboard / subpixel texture
        self.final = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(32, self.channels), num_channels=self.channels),
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


class HFToTop(nn.Module):
    """
    High-frequency (e.g. 64x64) -> top grid using ONLY strided convs + optional interpolate.
    Adds activation checkpointing (no pooling, no gating).
    """
    def __init__(
        self,
        in_ch,
        out_ch,
        mid_ch,
        target_hw,
        max_down_blocks=6,
        *,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.target_hw = tuple(int(x) for x in target_hw)
        self.use_checkpoint = bool(use_checkpoint)

        blocks = []
        ch = in_ch
        for _ in range(max_down_blocks):
            blocks.append(nn.Sequential(
                ConvGNAct(ch, mid_ch, k=3, s=2),
                ConvGNAct(mid_ch, mid_ch, k=3, s=1),
            ))
            ch = mid_ch
        self.down_blocks = nn.ModuleList(blocks)

        self.final = nn.Sequential(
            ConvGNAct(ch, out_ch, k=3, s=1),
            nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=True),
        )

    def _ckpt(self, fn, x):
        """
        Activation checkpointing: saves memory by re-running `fn(x)` on backward.
        Only enabled during training and when gradients are needed.
        """
        if self.use_checkpoint and self.training and x.requires_grad:
            # use_reentrant=False is recommended for modern PyTorch.
            return checkpoint(fn, x, use_reentrant=False)
        return fn(x)

    def forward(self, x):
        th, tw = self.target_hw
        y = x

        # Down blocks (checkpoint each block)
        for blk in self.down_blocks:
            if (y.shape[-2] <= 2 * th) and (y.shape[-1] <= 2 * tw):
                break
            y = self._ckpt(blk, y)

        # Resize (NOT pooling)
        if (y.shape[-2], y.shape[-1]) != (th, tw):
            y = F.interpolate(y, size=(th, tw), mode="bilinear", align_corners=False)

        return self._ckpt(self.final, y)

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

class EdgeToTop(nn.Module):
    """Project a 1-channel edge map at full resolution to a fixed (topH, topW) feature map.

    No pooling: strided convs + optional final interpolate.
    """
    def __init__(
        self,
        out_ch: int,
        *,
        mid_ch: int = 32,
        target_hw: tuple[int, int] = (8, 8),
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.net = HFToTop(
            in_ch=1,
            out_ch=int(out_ch),
            mid_ch=int(mid_ch),
            target_hw=(int(target_hw[0]), int(target_hw[1])),
            use_checkpoint=bool(use_checkpoint),
        )

    def forward(self, edge_fullres: torch.Tensor) -> torch.Tensor:
        return self.net(edge_fullres)

class ConvTokenizer(nn.Module):
    """Convert a feature map into a token grid (patchified conv) + sinusoidal 2D pos-enc."""

    def __init__(
        self,
        in_ch: int,
        token_dim: int,
        patch: int = 4,
        *,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.token_dim = int(token_dim)
        self.patch = int(patch)
        self.use_checkpoint = bool(use_checkpoint)

        self.proj = nn.Sequential(
            nn.Conv2d(self.in_ch, self.token_dim, kernel_size=self.patch, stride=self.patch),
            nn.GELU(),
            nn.Conv2d(self.token_dim, self.token_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        assert H % self.patch == 0 and W % self.patch == 0, "H,W must be divisible by patch."
        Hp, Wp = H // self.patch, W // self.patch

        if self.use_checkpoint and self.training and x.requires_grad:
            y = checkpoint(self.proj, x, use_reentrant=False)
        else:
            y = self.proj(x)

        tok = y.flatten(2).transpose(1, 2)  # [B, N, D]
        pos = sinusoidal_2d_pos_embed(Hp, Wp, tok.shape[-1], device=tok.device, dtype=tok.dtype)  # [1,N,D]
        tok = tok + pos
        return tok, (Hp, Wp)

class MotionTokenBank(nn.Module):
    """A lightweight recurrent memory of motion tokens.

    Maintains a per-batch token bank and updates it with attention using
    the current top-grid hidden map + action as context.
    """

    def __init__(
        self,
        n_tokens: int,
        token_dim: int,
        action_dim: int,
        top_hidden_dim: int,
        *,
        alpha: float = 0.9,
        dropout: float = 0.0,
        layer_norm: bool = True,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.token_dim = int(token_dim)
        self.action_dim = int(action_dim)
        self.top_hidden_dim = int(top_hidden_dim)
        self.alpha = float(alpha)

        self.base_tokens = nn.Parameter(init_std * torch.randn(1, self.n_tokens, self.token_dim))

        self.ctx_to_kv = nn.Conv2d(self.top_hidden_dim, 2 * self.token_dim, 1, 1, 0, bias=True)
        self.action_to_key = nn.Linear(self.action_dim, self.token_dim, bias=True)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=max(1, self.token_dim // 64),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ln = nn.LayerNorm(self.token_dim) if layer_norm else nn.Identity()

    def init_tokens(self, B: int, device=None, dtype=None) -> torch.Tensor:
        device = device or self.base_tokens.device
        dtype = dtype or self.base_tokens.dtype
        return self.base_tokens.to(device=device, dtype=dtype).expand(int(B), -1, -1).contiguous()

    def _maybe_ckpt(self, fn, *args):
        if self.use_checkpoint and self.training and any(t.requires_grad for t in args):
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    @staticmethod
    def _reset_with_mask(tokens: torch.Tensor, base_tokens: torch.Tensor, mask_t: torch.Tensor | None) -> torch.Tensor:
        if mask_t is None:
            return tokens
        if mask_t.dim() == 2 and mask_t.shape[1] == 1:
            mask = mask_t
        elif mask_t.dim() == 1:
            mask = mask_t[:, None]
        else:
            raise ValueError(f"mask_t must be [B] or [B,1], got {tuple(mask_t.shape)}")
        keep = mask.to(device=tokens.device, dtype=tokens.dtype).view(tokens.shape[0], 1, 1)
        return tokens * keep + base_tokens.expand(tokens.shape[0], -1, -1) * (1.0 - keep)

    def update(
        self,
        motion_tokens: torch.Tensor,
        action: torch.Tensor,
        h_context_map: torch.Tensor,
        *,
        mask_t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (new_tokens, mean_attention_entropy)."""
        motion_tokens = self._reset_with_mask(motion_tokens, self.base_tokens, mask_t)
        kv = self._maybe_ckpt(self.ctx_to_kv, h_context_map)

        k_map, v_map = kv.chunk(2, dim=1)
        k = k_map.flatten(2).transpose(1, 2)  # [B,M,D]
        v = v_map.flatten(2).transpose(1, 2)
        a_key = self._maybe_ckpt(self.action_to_key, action).unsqueeze(1)  # [B,1,D]
        k = k + a_key

        upd, attn_w = self.attn(
            query=motion_tokens,
            key=k,
            value=v,
            need_weights=True,
            average_attn_weights=False,
        )
        p = attn_w.clamp(min=1e-8)
        ent = -(p * p.log()).sum(dim=-1).mean()

        new_tokens = self.ln(self.alpha * motion_tokens + (1.0 - self.alpha) * upd)
        return new_tokens, ent

    def spatial_context_map(
        self,
        motion_tokens: torch.Tensor,
        token_hw: tuple[int, int],
        top_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert motion tokens to a top-grid map; return (map, entropy)."""
        Ht, Wt = token_hw
        topH, topW = top_hw
        B, N, D = motion_tokens.shape
        assert N == Ht * Wt, f"N must equal Ht*Wt, got N={N} vs {Ht}*{Wt}"
        assert D == self.token_dim

        map0 = motion_tokens.transpose(1, 2).reshape(B, D, Ht, Wt)
        if (Ht, Wt) != (topH, topW):
            map0 = F.interpolate(map0, size=(topH, topW), mode="bilinear", align_corners=False)

        # A simple diversity proxy entropy: softmax over per-token norms
        probs = F.softmax(motion_tokens.norm(dim=-1), dim=-1).clamp(min=1e-8)  # [B,N]
        ent = -(probs * probs.log()).sum(dim=-1).mean()

        return map0, ent

class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(num_groups=min(16, out_ch), num_channels=out_ch)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


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


class SEResDilatedBlock(nn.Module):
    def __init__(self, c: int, dilation: int = 1, groups: int = 16):
        super().__init__()
        g = min(groups, c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation, bias=False)
        self.gn1   = nn.GroupNorm(g, c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation, bias=False)
        self.gn2   = nn.GroupNorm(g, c)
        self.se    = SEBlock2d(c, r=16)
        self.act   = nn.SiLU(inplace=False)

    def forward(self, x):
        y = self.act(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))
        y = self.se(y)
        return self.act(x + y)

class MaxDeltaCalibrator:
    def __init__(self, q=0.995, frac=0.5, ema=0.98, min_md=2.0, max_md=16.0):
        self.q, self.frac, self.ema = q, frac, ema
        self.min_md, self.max_md = min_md, max_md
        self.p_ema = None

    @torch.no_grad()
    def update(self, flow64_base):
        p = torch.quantile(flow64_base.abs().flatten(), self.q).item()
        self.p_ema = p if self.p_ema is None else (self.ema * self.p_ema + (1 - self.ema) * p)
        target = self.frac * self.p_ema
        return float(max(self.min_md, min(self.max_md, target)))


class FlowRefiner(nn.Module):
    """
    Predict delta-flow at same resolution as input:
      input:  [B, in_ch, H, W]
      output: [B, 2,    H, W]  (delta flow in pixels)
    """
    def __init__(
        self,
        in_ch: int,
        base: int = 128,
        num_blocks: int = 8,
        max_delta: float = 8.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.max_delta = float(max_delta)
        self.use_checkpoint = bool(use_checkpoint)

        g = min(16, base)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1, bias=False),
            nn.GroupNorm(g, base),
            nn.SiLU(inplace=False),
        )

        # Dilations cycle to grow receptive field without down/up
        dilations = [1, 2, 4, 1, 2, 4, 1, 2]
        dilations = (dilations * ((num_blocks + len(dilations) - 1) // len(dilations)))[:num_blocks]
        self.blocks = nn.ModuleList([SEResDilatedBlock(base, d) for d in dilations])

        self.head = nn.Conv2d(base, 2, 3, padding=1, bias=True)

        # start as "no refinement"
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _maybe_ckpt(self, fn, x):
        do_ckpt = self.use_checkpoint and self.training and torch.is_grad_enabled() and x.requires_grad
        return checkpoint(fn, x, use_reentrant=False) if do_ckpt else fn(x)

    def forward(self, x):
        x = self._maybe_ckpt(self.stem, x)
        for blk in self.blocks:
            x = self._maybe_ckpt(blk, x)
        delta = self.head(x)
        return self.max_delta * torch.tanh(delta)


class LatentFlowHead(nn.Module):
    """
    Predict top-grid flow in pixel units: [B,2,topH,topW]
    """
    def __init__(self, hf_ch, mot_ch, act_dim, edge_ch, hidden=128, max_flow=8.0, use_checkpoint=False):
        super().__init__()
        self.max_flow = float(max_flow)
        self.use_checkpoint = bool(use_checkpoint)

        self.act_embed = nn.Sequential(
            nn.Conv2d(act_dim, 32, 1, 1, 0),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=False),
        )
        in_ch = hf_ch + mot_ch + 32 + edge_ch
        self.net = nn.Sequential(
            ConvGNAct(in_ch, hidden, 3, 1, inplace_act=False),
            ConvGNAct(hidden, hidden, 3, 1, inplace_act=False),
            ConvGNAct(hidden, hidden, 3, 1, inplace_act=False),
            nn.Conv2d(hidden, 2, 3, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hf_top, mot_ctx, a_map, edge_top):
        if self.use_checkpoint and self.training and a_map.requires_grad:
            a_emb = checkpoint(self.act_embed, a_map, use_reentrant=False)
        else:
            a_emb = self.act_embed(a_map)
        x = torch.cat([hf_top, mot_ctx, a_emb, edge_top], dim=1)

        if self.use_checkpoint and self.training and x.requires_grad:
            # split sequential into segments (tune 2-4 depending on your depth)
            flow = checkpoint_sequential(self.net, segments=3, input=x, use_reentrant=False)
        else:
            flow = self.net(x)

        return self.max_flow * torch.tanh(flow)



class _ResGNAct(nn.Module):
    """Simple residual block: (Conv-GN-SiLU) x2 with skip."""
    def __init__(self, ch: int, groups: int = 8):
        super().__init__()
        g = min(groups, ch)
        while ch % g != 0 and g > 1:
            g -= 1
        self.gn1 = nn.GroupNorm(g, ch)
        self.gn2 = nn.GroupNorm(g, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.act = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gn1(x))
        h = self.c1(h)
        h = self.act(self.gn2(h))
        h = self.c2(h)
        return x + h


class HFMapPredictor(nn.Module):
    """Predict a full-res (default 64x64) HF feature map from (z_top, h_context, action).

    Backward compatible with two constructors:

    """
    def __init__(
        self,
        in_ch: int = None,
        out_ch: int = None,
        action_dim: int = None,
        *,
        z_ch: int = None,
        h_ch: int = None,
        a_ch: int = None,
        out_hw: tuple[int, int] = (64, 64),
        base: int = 128,
        use_checkpoint: bool = False,
        groups: int = 8,
        n_res: int = 6,
    ):
        super().__init__()
        if (z_ch is not None) or (h_ch is not None) or (a_ch is not None):
            if (z_ch is None) or (h_ch is None) or (a_ch is None) or (out_ch is None):
                raise ValueError("HFMapPredictor: when using z_ch/h_ch/a_ch you must also pass out_ch")
            in_ch = int(z_ch) + int(h_ch)
            action_dim = int(a_ch)
            out_ch = int(out_ch)
        if in_ch is None or out_ch is None or action_dim is None:
            raise ValueError("HFMapPredictor: missing required channel dims")

        self.out_hw = (int(out_hw[0]), int(out_hw[1]))
        self.use_checkpoint = bool(use_checkpoint)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch + action_dim, int(base), 3, 1, 1, bias=False),
            nn.GroupNorm(max(1, min(int(groups), int(base))), int(base)),
            nn.SiLU(inplace=False),
        )

        # AR path: downsample hf_prev to top grid, project to base, add into stem output
        self.prev_proj = nn.Conv2d(int(out_ch), int(base), 1, 1, 0, bias=False)

        self.res_blocks = nn.ModuleList([_ResGNAct(int(base), groups=int(groups)) for _ in range(int(n_res))])
        self.head = nn.Conv2d(int(base), int(out_ch), 3, 1, 1, bias=False)

    def _maybe_ckpt(self, fn, *args):
        if self.use_checkpoint and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(
        self,
        *,
        z_map: torch.Tensor,
        h_context_map: torch.Tensor,
        a_t: torch.Tensor,
        hf_prev: torch.Tensor = None,
    ) -> torch.Tensor:
        B, _, topH, topW = z_map.shape
        a_map = a_t.view(B, -1, 1, 1).expand(B, -1, topH, topW)
        x = torch.cat([z_map, h_context_map, a_map], dim=1)

        x = self._maybe_ckpt(self.stem, x)

        if hf_prev is not None:
            hp = F.interpolate(hf_prev, size=(topH, topW), mode="bilinear", align_corners=False)
            x = x + self._maybe_ckpt(self.prev_proj, hp)

        # lift to out resolution
        x = F.interpolate(x, size=self.out_hw, mode="nearest")

        for blk in self.res_blocks:
            x = self._maybe_ckpt(blk, x)

        x = self._maybe_ckpt(self.head, x)
        return x

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
        self.last_attn_entropy = None  # scalar tensor set each forward()

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
                bias = -dist2 / (2.0 * (self.sigma ** 2) + 1e-6)  # [B,N,N]
                scores = scores + bias.unsqueeze(1)  # broadcast over heads

            attn = torch.softmax(scores, dim=-1)

            # attention entropy for logging (mean over B,heads,queries)
            with torch.no_grad():
                p = attn.clamp_min(1e-8)
                ent = -(p * p.log()).sum(dim=-1)  # [B,H,N]
                self.last_attn_entropy = ent.mean()

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

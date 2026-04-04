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


        
def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

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

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 is_first: bool = False, omega_0: float = 30):
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
    def __init__(self, in_features: int, hidden_features: list[int], hidden_layers: int, out_features: int, outermost_linear: bool = False, 
                 first_omega_0: float = 30, hidden_omega_0: float = 30.):
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


backwarp_tenGrid = {}

def warpgrid(tenInput, tenFlow, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenFlow.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenFlow.shape[2] - 1.0) / 2.0)], 1)
    
    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return g, torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _batched_coord(
    hw: Tuple[int, int],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Returns [B, H*W, 2] coordinates in the make_coord convention.
    """
    coord = make_coord(hw, flatten=True).to(device=device, dtype=dtype)
    coord = coord.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    return coord


def _coord_map(
    hw: Tuple[int, int],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Returns [B, 2, H, W] coordinate map in the make_coord convention.
    """
    coord = make_coord(hw, flatten=False).to(device=device, dtype=dtype)  # [H, W, 2]
    coord = coord.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
    return coord


def _sample_at_coord(
    feat_map: torch.Tensor,
    coord: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """
    feat_map: [B, C, H, W]
    coord:    [B, Q, 2] in make_coord convention
    returns:  [B, Q, C]
    """
    sampled = F.grid_sample(
        feat_map,
        coord.flip(-1).unsqueeze(1),  # grid_sample expects x,y order
        mode=mode,
        align_corners=False,
    )  # [B, C, 1, Q]
    return sampled[:, :, 0, :].permute(0, 2, 1).contiguous()


def _expand_time(
    dt,
    batch_size: int,
    num_queries: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Returns [B, Q, 1].
    """
    if not torch.is_tensor(dt):
        dt = torch.tensor(dt, device=device, dtype=dtype)

    dt = dt.to(device=device, dtype=dtype)

    if dt.dim() == 0:
        dt = dt.view(1, 1, 1).expand(batch_size, num_queries, 1)
    elif dt.dim() == 1:
        if dt.numel() == batch_size:
            dt = dt.view(batch_size, 1, 1).expand(batch_size, num_queries, 1)
        elif dt.numel() == 1:
            dt = dt.view(1, 1, 1).expand(batch_size, num_queries, 1)
        else:
            raise ValueError(f"dt shape {tuple(dt.shape)} not supported")
    elif dt.dim() == 2:
        if dt.shape == (batch_size, 1):
            dt = dt.unsqueeze(1).expand(batch_size, num_queries, 1)
        elif dt.shape == (batch_size, num_queries):
            dt = dt.unsqueeze(-1)
        else:
            raise ValueError(f"dt shape {tuple(dt.shape)} not supported")
    elif dt.dim() == 3:
        if dt.shape[-1] != 1:
            raise ValueError(f"dt last dim must be 1, got {tuple(dt.shape)}")
    else:
        raise ValueError(f"dt shape {tuple(dt.shape)} not supported")

    return dt


def _expand_action(
    action: Optional[torch.Tensor],
    batch_size: int,
    num_queries: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """
    Returns [B, Q, A] or None.
    """
    if action is None:
        return None

    action = action.to(device=device, dtype=dtype)

    if action.dim() == 1:
        action = action.view(1, -1).expand(batch_size, -1)
    elif action.dim() != 2:
        raise ValueError(f"action must be [B,A] or [A], got {tuple(action.shape)}")

    if action.shape[0] != batch_size:
        raise ValueError(
            f"action batch {action.shape[0]} does not match batch_size={batch_size}"
        )

    return action.unsqueeze(1).expand(batch_size, num_queries, action.shape[-1]).contiguous()


# ---------------------------------------------------------------------
# continuous latent field
# ---------------------------------------------------------------------

class SpatialLatentINR(nn.Module):
    """
      Fs(xs) = fs(z*, xs - v*)

    where z* is the nearest latent vector from the discrete top-latent map,
    and v* is the coordinate of that nearest latent vector.
    """

    def __init__(
        self,
        z_channels: int,
        hidden_features: list[int] = [64, 64, 256],
        hidden_layers: int = 2,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.z_channels = int(z_channels)
        self.out_channels = int(out_channels) if out_channels is not None else int(z_channels)

        self.feat_imnet = Siren(
            in_features=self.z_channels + 2,      # nearest latent + relative coord
            out_features=self.out_channels,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            outermost_linear=True,
        )

    def query(
        self,
        z_top_prev: torch.Tensor,
        coord: torch.Tensor,
    ) -> torch.Tensor:
        """
        z_top_prev: [B, Cz, Ht, Wt]
        coord:      [B, Q, 2] in make_coord convention
        returns:    [B, Q, Cz_out]
        """
        if z_top_prev.dim() != 4:
            raise ValueError(f"z_top_prev must be [B,C,H,W], got {tuple(z_top_prev.shape)}")
        if coord.dim() != 3 or coord.shape[-1] != 2:
            raise ValueError(f"coord must be [B,Q,2], got {tuple(coord.shape)}")

        B, Cz, Ht, Wt = z_top_prev.shape
        if Cz != self.z_channels:
            raise ValueError(
                f"z_top_prev channels {Cz} do not match z_channels={self.z_channels}"
            )

        q_feat = _sample_at_coord(z_top_prev, coord, mode="nearest")  # [B, Q, Cz]

        feat_coord = _coord_map((Ht, Wt), B, z_top_prev.device, z_top_prev.dtype)
        q_coord = _sample_at_coord(feat_coord, coord, mode="nearest")  # [B, Q, 2]

        rel_coord = coord - q_coord
        rel_coord[..., 0] *= Ht
        rel_coord[..., 1] *= Wt

        inp = torch.cat([q_feat, rel_coord], dim=-1)  # VideoINR-style concatenation
        out = self.feat_imnet(inp.view(B * coord.shape[1], -1)).view(B, coord.shape[1], self.out_channels)
        return out

    def dense_field(
        self,
        z_top_prev: torch.Tensor,
        out_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the dense continuous latent field once over the whole target grid.

        returns:
          field_map: [B, Cz_out, Hout, Wout]
          coord:     [B, Hout*Wout, 2]
        """
        B, _, Ht, Wt = z_top_prev.shape
        out_hw = (Ht, Wt) if out_hw is None else tuple(out_hw)

        coord = _batched_coord(out_hw, B, z_top_prev.device, z_top_prev.dtype)
        dense_feat = self.query(z_top_prev, coord)  # [B, Q, Cz_out]
        field_map = dense_feat.permute(0, 2, 1).contiguous().view(B, self.out_channels, out_hw[0], out_hw[1])
        return field_map, coord


# ---------------------------------------------------------------------
# continuous motion field
# ---------------------------------------------------------------------

class TemporalMotionINR(nn.Module):
    """
    Temporal implicit representation:
      M(xs, xt) = ft(xt, Fs(xs), context)
    """

    def __init__(
        self,
        z_channels: int,
        h_channels: int,
        action_dim: int,
        hidden_features: list[int] = [64, 64, 256],
        hidden_layers: int = 2,
        flow_scale: float = 1.0,
    ):
        super().__init__()
        self.z_channels = int(z_channels)
        self.h_channels = int(h_channels)
        self.action_dim = int(action_dim)
        self.flow_scale = float(flow_scale)

        # concatenation only:
        # queried latent feature + queried recurrent context + query coord + dt + action
        in_dim = self.z_channels + self.h_channels + 2 + 1 + self.action_dim

        self.flow_imnet = Siren(
            in_features=in_dim,
            out_features=2,  # dx, dy in top-grid pixel units
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            outermost_linear=True,
        )

    def forward(
        self,
        spatial_field_map: torch.Tensor,
        h_t: torch.Tensor,
        action_prev: Optional[torch.Tensor],
        dt,
        coord: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        spatial_field_map: [B, Cz, Ht, Wt]
        h_t:               [B, Ch, Ht, Wt]  (same top-grid semantic context)
        action_prev:       [B, A] or None
        dt:                scalar, [B], [B,1], or [B,Q,1]
        coord:             [B, Q, 2] or None

        returns:
          {
            "flow_top_px": [B, 2, Ht, Wt],
            "coord":       [B, Q, 2],
          }
        """
        if spatial_field_map.dim() != 4:
            raise ValueError(
                f"spatial_field_map must be [B,C,H,W], got {tuple(spatial_field_map.shape)}"
            )
        if h_t.dim() != 4:
            raise ValueError(f"h_t must be [B,C,H,W], got {tuple(h_t.shape)}")

        B, Cz, Ht, Wt = spatial_field_map.shape
        if Cz != self.z_channels:
            raise ValueError(
                f"spatial_field_map channels {Cz} do not match z_channels={self.z_channels}"
            )

        if h_t.shape[-2:] != (Ht, Wt):
            h_t = F.interpolate(
                h_t,
                size=(Ht, Wt),
                mode="bilinear",
                align_corners=False,
            )

        if h_t.shape[1] != self.h_channels:
            raise ValueError(
                f"h_t channels {h_t.shape[1]} do not match h_channels={self.h_channels}"
            )

        if coord is None:
            coord = _batched_coord((Ht, Wt), B, spatial_field_map.device, spatial_field_map.dtype)

        Q = coord.shape[1]

        q_spatial = _sample_at_coord(spatial_field_map, coord, mode="nearest")   # [B,Q,Cz]
        q_h = _sample_at_coord(h_t, coord, mode="bilinear")                      # [B,Q,Ch]

        dt_in = _expand_time(dt, B, Q, spatial_field_map.device, spatial_field_map.dtype)
        a_in = _expand_action(action_prev, B, Q, spatial_field_map.device, spatial_field_map.dtype)

        if a_in is None:
            motion_inp = torch.cat([q_spatial, q_h, coord, dt_in], dim=-1)
        else:
            motion_inp = torch.cat([q_spatial, q_h, coord, dt_in, a_in], dim=-1)

        flow = self.flow_imnet(motion_inp.view(B * Q, -1)).view(B, Q, 2)
        flow = self.flow_scale * flow
        flow_top_px = flow.permute(0, 2, 1).contiguous().view(B, 2, Ht, Wt)

        return {
            "flow_top_px": flow_top_px,
            "coord": coord,
        }


class LatentTransportINR(nn.Module):
    """

      z_prev_top --SpatialLatentINR--> dense continuous latent field
      (field, h_t, a_prev, dt) --TemporalMotionINR--> flow_top_px
      warp(field, flow_top_px) --> z_tilde
      cond_top = concat(h_t, z_tilde)

    """

    def __init__(
        self,
        z_channels: int,
        h_channels: int,
        action_dim: int,
        hidden_features: list[int] = [64, 64, 256],
        hidden_layers: int = 2,
        flow_scale: float = 1.0,
        init_spatial_gate: float = 1e-3,
        init_flow_gate: float = 1e-3,
        init_transport_gate: float = 1e-3,
    ):
        super().__init__()
        self.z_channels = int(z_channels)
        self.h_channels = int(h_channels)
        self.action_dim = int(action_dim)

        self.spatial_inr = SpatialLatentINR(
            z_channels=z_channels,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_channels=z_channels,
        )

        self.temporal_inr = TemporalMotionINR(
            z_channels=z_channels,
            h_channels=h_channels,
            action_dim=action_dim,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            flow_scale=flow_scale,
        )
        self.spatial_gate_logit = nn.Parameter(self._inv_sigmoid(init_spatial_gate))
        self.flow_gate_logit = nn.Parameter(self._inv_sigmoid(init_flow_gate))
        self.transport_gate_logit = nn.Parameter(self._inv_sigmoid(init_transport_gate))


    @staticmethod
    def _inv_sigmoid(p: float, eps: float = torch.finfo(torch.float32).eps) -> torch.Tensor:
        """
        Convert a probability-like value in (0,1) into a logit parameter.
        Using logits lets us keep gates in (0,1) with sigmoid while initializing
        them to tiny but nonzero values.
        """
        
        p = float(min(max(p, eps), 1.0 - eps))
        return torch.tensor(math.log(p / (1.0 - p)), dtype=torch.float32)

    def forward(
        self,
        z_top_prev: torch.Tensor,
        h_t: torch.Tensor,
        action_prev: Optional[torch.Tensor],
        dt=1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        z_top_prev: [B, Cz, Ht, Wt]
        h_t:        [B, Ch, Ht, Wt]
        action_prev:[B, A] or None
        dt:         scalar or batch tensor

        returns:
          {
            "spatial_field": [B, Cz, Ht, Wt],
            "flow_top_px":   [B, 2, Ht, Wt],
            "z_tilde":       [B, Cz, Ht, Wt],
            "cond_top":      [B, Ch + Cz, Ht, Wt],
            "h_decoder_top": [B, Ch, Ht, Wt],
          }
        """
        if z_top_prev.dim() != 4:
            raise ValueError(f"z_top_prev must be [B,C,H,W], got {tuple(z_top_prev.shape)}")
        if h_t.dim() != 4:
            raise ValueError(f"h_t must be [B,C,H,W], got {tuple(h_t.shape)}")

        B, Cz, Ht, Wt = z_top_prev.shape
        if Cz != self.z_channels:
            raise ValueError(
                f"z_top_prev channels {Cz} do not match z_channels={self.z_channels}"
            )

        assert h_t.shape[-2:] == (Ht, Wt), f"h_t spatial dimensions {h_t.shape[-2:]} do not match expected {(Ht, Wt)}"

        if h_t.shape[1] != self.h_channels:
            raise ValueError(
                f"h_t channels {h_t.shape[1]} do not match h_channels={self.h_channels}"
            )

        # 1) Build dense continuous latent field once, VideoINR-style
        spatial_field, coord = self.spatial_inr.dense_field(z_top_prev, out_hw=(Ht, Wt))
        alpha = torch.sigmoid(self.spatial_gate_logit)
        spatial_field = alpha * spatial_field + z_top_prev  # gated residual to stabilize training at the start

        # 2) Predict continuous motion field on the same grid
        motion = self.temporal_inr(
            spatial_field_map=spatial_field,
            h_t=h_t,
            action_prev=action_prev,
            dt=dt,
            coord=coord,
        )
        flow_top_px = motion["flow_top_px"]
        beta = torch.sigmoid(self.flow_gate_logit)
        flow_top_px = beta * flow_top_px  # gated scaling to stabilize training at the

        # 3) Warp the dense continuous latent field to get transported latent z_tilde
        _, z_tilde = warpgrid(
            tenInput=spatial_field,
            tenFlow=flow_top_px,
            device=spatial_field.device,
        )
        gamma = torch.sigmoid(self.transport_gate_logit)
        z_tilde = gamma * z_tilde + (1.0 - gamma) * spatial_field
        # 4) Top conditioning s
        cond_top = torch.cat([h_t, z_tilde], dim=1)
        h_decoder_top = h_t

        return {
            "spatial_field": spatial_field,
            "flow_top_px": flow_top_px,
            "z_tilde": z_tilde,
            "cond_top": cond_top,
            "h_decoder_top": h_decoder_top,
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

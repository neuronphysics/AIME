import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, hidden_dim=128, input_dim=64):
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h

class Encoder(nn.Module):
    """
    Your original "Encoder": updates a spatial hidden state from features.
    Now optionally action-conditioned by concatenation (broadcast).
    """
    def __init__(self, action_dim: int = 0):
        super().__init__()
        self.action_dim = int(action_dim)

        self.hidden_state_conv = Hidden_state_conv(input_dim=64, hidden_dim=128)

        # CHANGED: ConvGRU input_dim becomes 64 + action_dim when action is provided.
        self.E_gru = ConvGRU(hidden_dim=128, input_dim=64 + self.action_dim)

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
        self.relu = nn.ReLU(inplace=True)

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
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class Decoder(nn.Module):
    """
    Your original "Decoder": given feature [B,128,H,W] -> predicts future coordinate fields for fut steps.
    Now optionally conditions on action by concatenation (broadcast) WITHOUT adding new layers.
    """
    def __init__(self, future_len: int, action_dim: int = 0):
        super().__init__()
        self.fut = int(future_len)
        self.action_dim = int(action_dim)

        # CHANGED: StateFlowEncoder convc1 input dim becomes 64 + action_dim when action is provided.
        self.encode = StateFlowEncoder(statedim=64 + self.action_dim, outdim=62)

        # same channels as before
        self.D_gru  = ConvGRU(hidden_dim=64, input_dim=64)
        self.D_flow = FlowHead(input_dim=64, hidden_dim=128)

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
            cat_encoded = self.encode(inp_aug, flow)

            # recurrent update
            state = self.D_gru(state, cat_encoded)

            # delta coords update
            delta = self.D_flow(state)    # [B,2,H,W]
            coords1 = coords1 + delta

            f_coords.append(coords1)

        return {"f_flows": f_coords}  # original

def get_gru_decoder(future_len: int, action_dim: int = 0):
    # CHANGED: keep same function name, add optional action_dim with default (backwards compatible)
    return Decoder(future_len=future_len, action_dim=action_dim)

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

"""LSTM modules."""

class LSTMLayer(nn.Module):
    
    def __init__(self, input_size, hidden_size, n_lstm_layers, use_orthogonal):
        super(LSTMLayer, self).__init__()
        self.n_lstm_layers = n_lstm_layers
        self._use_orthogonal = use_orthogonal

        # lstm. note that batch_first=False (default)
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=n_lstm_layers)
        # layer norm
        self.norm = nn.LayerNorm(hidden_size)

        # initialisation
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        
    def forward(self, x, hxs, cxs, masks):
        """
        Process one timestep input with correct dimension handling
        Args:
            x: Input tensor (batch_size, input_size)
            hxs: Hidden state (n_layers, batch_size, hidden_size)
            cxs: Cell state (n_layers, batch_size, hidden_size)
            masks: Binary mask for active sequences (batch_size,)
        """
        batch_size = x.size(0)
    
        # Reshape mask for broadcasting (n_layers, batch_size, 1)
        mask_expanded = masks.view(1, -1, 1).expand(self.n_lstm_layers, batch_size, 1)
    
        # Apply mask to states (zeros out states for completed sequences)
        hxs_masked = hxs * mask_expanded
        cxs_masked = cxs * mask_expanded
    
        # Add sequence dimension for LSTM input
        x_sequence = x.unsqueeze(0)
    
        # LSTM forward pass
        output, (hxs_new, cxs_new) = self.lstm(x_sequence, (hxs_masked, cxs_masked))
    
        # Remove sequence dimension
        output = output.squeeze(0)
    
        # Apply layer normalization
        output = self.norm(output)
    
        return output, (hxs_new, cxs_new)


def _trunc_normal_(w: torch.Tensor, std: float = 0.02) -> None:
    if hasattr(nn.init, "trunc_normal_"):
        nn.init.trunc_normal_(w, mean=0.0, std=std, a=-2 * std, b=2 * std)
    else:
        nn.init.normal_(w, mean=0.0, std=std)


def _largest_divisor_leq(n: int, k: int) -> int:
    k = min(k, n)
    for g in range(k, 0, -1):
        if n % g == 0:
            return g
    return 1


class ConvLSTMCell(nn.Module):
    """
    TF BasicConv2DLSTMCell-like implementation but with YOUR API:
      forward(x, h_prev, c_prev) -> (h_next, c_next)

    Supports optional:
      - normalization: None | "instance" | "group" | "layer"
      - separate_norms: per-gate norms vs single norm pre-split
      - recurrent dropout on candidate g
      - skip_connection: concatenate input to output (changes h channel count!)
      - optional non-spatial conditioning via a dense add:
            x can be (x_spatial, x_non_spatial) if non_spatial_dim is provided
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: Union[int, Tuple[int, int]],
        bias: bool = True,
        # --- TF-like knobs (optional; defaults mimic TF baseline) ---
        forget_bias: float = 1.0,
        activation_fn: str = "tanh",             # "tanh" matches TF default
        normalizer: Optional[str] = None,        # None | "instance" | "group" | "layer"
        separate_norms: bool = True,
        norm_gain: float = 1.0,
        norm_shift: float = 0.0,
        dropout_keep_prob: float = 1.0,
        skip_connection: bool = False,
        # --- non-spatial conditioning (no Lazy): must specify dim to enable ---
        non_spatial_dim: Optional[int] = None,
        # --- group norm config ---
        num_groups: int = 8,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.forget_bias = float(forget_bias)
        self.normalizer = normalizer
        self.separate_norms = bool(separate_norms)
        self.dropout_keep_prob = float(dropout_keep_prob)
        self.skip_connection = bool(skip_connection)
        self.non_spatial_dim = non_spatial_dim
        self.num_groups = int(num_groups)
        self.eps = float(eps)

        if isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        else:
            kh, kw = kernel_size
        self.kernel_size = (kh, kw)
        # "same" padding for odd kernels
        self.padding = (kh // 2, kw // 2)

        if activation_fn == "tanh":
            self.act = torch.tanh
        elif activation_fn == "relu":
            self.act = F.relu
        elif activation_fn == "gelu":
            self.act = F.gelu
        else:
            raise ValueError(f"Unsupported activation_fn='{activation_fn}'")

        # In TF: if skip_connection, output channels = filters + input_channels
        self.output_dim = self.hidden_dim + self.input_dim if self.skip_connection else self.hidden_dim

        # IMPORTANT: conv sees concat([x, h_prev])
        # TF conv in-channels depend on output_dim if skip_connection is on.
        conv_in = self.input_dim + self.output_dim
        conv_out = 4 * self.hidden_dim  # i, j, f, o

        # TF: if using normalizer, they omit bias in conv
        conv_bias = bool(bias) if self.normalizer is None else False

        self.conv = nn.Conv2d(conv_in, conv_out, kernel_size=self.kernel_size, padding=self.padding, bias=conv_bias)
        _trunc_normal_(self.conv.weight, std=0.02)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        # Optional dense for non-spatial input (TF "tile_concat" feature)
        if self.non_spatial_dim is not None:
            self.dense = nn.Linear(self.non_spatial_dim, conv_out, bias=False)
            _trunc_normal_(self.dense.weight, std=0.02)
        else:
            self.dense = None

        # Norm layers
        self.norm_concat = None
        self.norm_i = self.norm_j = self.norm_f = self.norm_o = None
        self.norm_state = None

        if self.normalizer is not None:
            if not self.separate_norms:
                self.norm_concat = self._make_norm(conv_out, norm_gain, norm_shift)
            else:
                self.norm_i = self._make_norm(self.hidden_dim, norm_gain, norm_shift)
                self.norm_j = self._make_norm(self.hidden_dim, norm_gain, norm_shift)
                self.norm_f = self._make_norm(self.hidden_dim, norm_gain, norm_shift)
                self.norm_o = self._make_norm(self.hidden_dim, norm_gain, norm_shift)
            self.norm_state = self._make_norm(self.hidden_dim, norm_gain, norm_shift)

    def _make_norm(self, channels: int, gain: float, shift: float) -> nn.Module:
        if self.normalizer == "instance":
            norm = nn.InstanceNorm2d(channels, eps=self.eps, affine=True)
            nn.init.constant_(norm.weight, gain)
            nn.init.constant_(norm.bias, shift)
            return norm

        if self.normalizer == "group":
            g = _largest_divisor_leq(channels, self.num_groups)
            norm = nn.GroupNorm(g, channels, eps=self.eps, affine=True)
            nn.init.constant_(norm.weight, gain)
            nn.init.constant_(norm.bias, shift)
            return norm

        if self.normalizer == "layer":
            # Per-pixel layer norm across channels: use GroupNorm with 1 group
            # (works for any H,W, unlike nn.LayerNorm which would need H,W known).
            norm = nn.GroupNorm(1, channels, eps=self.eps, affine=True)
            nn.init.constant_(norm.weight, gain)
            nn.init.constant_(norm.bias, shift)
            return norm

        raise ValueError(f"Unknown normalizer='{self.normalizer}'")

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            - spatial tensor [B, C_in, H, W]
            - OR tuple (x_spatial [B,C_in,H,W], x_non_spatial [B,D]) if non_spatial_dim is set
          h_prev: [B, output_dim, H, W]  (output_dim = hidden_dim (+ input_dim if skip_connection))
          c_prev: [B, hidden_dim, H, W]

        Returns:
          h_next: [B, output_dim, H, W]
          c_next: [B, hidden_dim, H, W]
        """
        tile_concat = isinstance(x, (tuple, list))
        if tile_concat:
            x_spatial, x_non_spatial = x
            if self.dense is None:
                raise ValueError("Got (x_spatial, x_non_spatial) but non_spatial_dim=None in init.")
        else:
            x_spatial, x_non_spatial = x, None

        if h_prev.shape[1] != self.output_dim:
            raise ValueError(
                f"h_prev has {h_prev.shape[1]} channels but this cell expects {self.output_dim}. "
                f"(skip_connection={self.skip_connection})"
            )
        if c_prev.shape[1] != self.hidden_dim:
            raise ValueError(
                f"c_prev has {c_prev.shape[1]} channels but this cell expects {self.hidden_dim}."
            )

        args = torch.cat([x_spatial, h_prev], dim=1)
        concat = self.conv(args)

        if tile_concat:
            concat = concat + self.dense(x_non_spatial)[:, :, None, None]

        # norm before split (TF separate_norms=False)
        if self.norm_concat is not None:
            concat = self.norm_concat(concat)

        # TF order: i, j, f, o
        i, j, f, o = torch.split(concat, self.hidden_dim, dim=1)

        # per-gate norms (TF separate_norms=True)
        if self.norm_i is not None:
            i = self.norm_i(i)
            j = self.norm_j(j)
            f = self.norm_f(f)
            o = self.norm_o(o)

        g = self.act(j)
        if self.dropout_keep_prob < 1.0:
            g = F.dropout(g, p=(1.0 - self.dropout_keep_prob), training=self.training)

        c_next = c_prev * torch.sigmoid(f + self.forget_bias) + torch.sigmoid(i) * g

        if self.norm_state is not None:
            c_next = self.norm_state(c_next)

        h_core = self.act(c_next) * torch.sigmoid(o)

        if self.skip_connection:
            h_next = torch.cat([h_core, x_spatial], dim=1)
        else:
            h_next = h_core

        return h_next, c_next


def _largest_divisor_leq(n: int, k: int) -> int:
    k = min(k, n)
    for g in range(k, 0, -1):
        if n % g == 0:
            return g
    return 1


class ConvLSTM(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 kernel_size: Tuple[int, int],
                 num_layers: int = 1,
                 bias: bool = True,
                 batch_first: bool = True,
                 return_sequence: bool = True,
                 num_groups: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_sequence = return_sequence

        cells = []
        for i in range(num_layers):
            cell_in = input_dim if i == 0 else hidden_dim
            cells.append(ConvLSTMCell(cell_in, hidden_dim, kernel_size, bias=bias))
        self.layers = nn.ModuleList(cells)

        gn_groups = _largest_divisor_leq(hidden_dim, num_groups)
        self.output_norm = nn.GroupNorm(num_groups=gn_groups, num_channels=hidden_dim)

    def _init_state(self, B: int, H: int, W: int, device, dtype):
        h_list = [torch.zeros(B, self.hidden_dim, H, W, device=device, dtype=dtype)
                  for _ in range(self.num_layers)]
        c_list = [torch.zeros(B, self.hidden_dim, H, W, device=device, dtype=dtype)
                  for _ in range(self.num_layers)]
        return h_list, c_list

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,
        done: Optional[torch.Tensor] = None,
    ):
        """
        x:
          batch_first=True:  [B, T, C_in, H, W]
          batch_first=False: [T, B, C_in, H, W]
        done:
          batch_first=True:  [B, T] with 1 for done/reset, 0 for keep
          batch_first=False: [T, B]
        """
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4)  # -> [B,T,C,H,W]
            if done is not None:
                done = done.permute(1, 0)  # -> [B,T]

        B, T, _, H, W = x.shape

        if state is None:
            h_list, c_list = self._init_state(B, H, W, x.device, x.dtype)
        else:
            h_list, c_list = state
            assert len(h_list) == self.num_layers and len(c_list) == self.num_layers

        outputs = [] if self.return_sequence else None

        for t in range(T):
            x_t = x[:, t]  # [B,C_in,H,W]

            if done is not None:
                # done[:,t] == 1 means reset -> keep_mask=0
                keep = (1.0 - done[:, t].float()).view(B, 1, 1, 1)

                for li, cell in enumerate(self.layers):
                    h_list[li] = h_list[li] * keep
                    c_list[li] = c_list[li] * keep
                    h_list[li], c_list[li] = cell(x_t, h_list[li], c_list[li])
                    x_t = h_list[li]
            else:
                for li, cell in enumerate(self.layers):
                    h_list[li], c_list[li] = cell(x_t, h_list[li], c_list[li])
                    x_t = h_list[li]

            # top layer output
            top_h = h_list[-1]
            # normalize correctly on 4D
            top_h = self.output_norm(top_h)

            if self.return_sequence:
                outputs.append(top_h)

        if self.return_sequence:
            out = torch.stack(outputs, dim=1)  # [B,T,C,H,W]
            return out, (h_list, c_list)
        else:
            return h_list[-1], (h_list, c_list)


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 7, height, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, height, width])
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 3, height, width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)


    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new



def safe_groups(C, max_groups=8):
    for g in range(min(max_groups, C), 0, -1):
        if C % g == 0:
            return g
    return 1

class SpatioTemporalCore(nn.Module):
    def __init__(self, zdim, action_dim, hidden_dim, *, height=4, width=4,
                 kernel=5, layer_norm=True, init_std=0.0, use_checkpoint=False,
                 extra_channels=0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.height = int(height)
        self.width = int(width)
        self.use_checkpoint = bool(use_checkpoint)
        self.extra_channels = int(extra_channels)

        g = safe_groups(self.action_dim, 8)
        self.action_proj = nn.Sequential(
            nn.Conv2d(self.action_dim, self.action_dim, kernel_size=1, bias=False),
            nn.GroupNorm(g, self.action_dim),
        )

        self.cell = SpatioTemporalLSTMCell(
            in_channel=int(zdim) + self.action_dim + self.extra_channels,
            num_hidden=self.hidden_dim,
            height=self.height,
            width=self.width,
            filter_size=int(kernel),
            stride=1,
            layer_norm=bool(layer_norm),
        )

        self.out_norm = nn.GroupNorm(num_groups=safe_groups(self.hidden_dim, 8),
                                     num_channels=self.hidden_dim)

        self.h0 = nn.Parameter(torch.zeros(1, self.hidden_dim, self.height, self.width))
        self.c0 = nn.Parameter(torch.zeros(1, self.hidden_dim, self.height, self.width))
        self.m0 = nn.Parameter(torch.zeros(1, self.hidden_dim, self.height, self.width))
        if float(init_std) > 0.0:
            nn.init.normal_(self.h0, std=float(init_std))
            nn.init.normal_(self.c0, std=float(init_std))
            nn.init.normal_(self.m0, std=float(init_std))

    def init_state(self, B, device, dtype):
        h = self.h0.expand(B, -1, -1, -1).to(device=device, dtype=dtype)
        c = self.c0.expand(B, -1, -1, -1).to(device=device, dtype=dtype)
        m = self.m0.expand(B, -1, -1, -1).to(device=device, dtype=dtype)
        return h, c, m

    def forward(self, z_map, a_t, state=None, mask_t=None, extra_maps=None):
        B, _, H, W = z_map.shape
        assert H == self.height and W == self.width, "z_map H/W must match core height/width"

        if state is None:
            h, c, m = self.init_state(B, z_map.device, z_map.dtype)
        else:
            h, c, m = state

        if mask_t is not None:
            keep = mask_t.view(B, 1, 1, 1).to(z_map.dtype)
            h0, c0, m0 = self.init_state(B, z_map.device, z_map.dtype)
            h = h * keep + h0 * (1.0 - keep)
            c = c * keep + c0 * (1.0 - keep)
            m = m * keep + m0 * (1.0 - keep)

        a_1x1 = a_t.view(B, self.action_dim, 1, 1).to(device=z_map.device, dtype=z_map.dtype)
        a_emb = self.action_proj(a_1x1)
        a_map = a_emb.expand(B, self.action_dim, H, W)

        x_t = torch.cat([z_map, a_map], dim=1)

        extra_c = 0
        if extra_maps is not None:
            for emap in extra_maps:
                assert emap.shape[0] == B and emap.shape[2] == H and emap.shape[3] == W, "extra_map shape mismatch"
                x_t = torch.cat([x_t, emap.to(dtype=z_map.dtype, device=z_map.device)], dim=1)
                extra_c += int(emap.shape[1])

        assert extra_c == self.extra_channels, f"Expected extra_channels={self.extra_channels}, got {extra_c}"

        if self.use_checkpoint and self.training:
            def cell_fn(x, h_in, c_in, m_in):
                return self.cell(x, h_in, c_in, m_in)
            h_new, c_new, m_new = checkpoint(cell_fn, x_t, h, c, m, use_reentrant=False)
        else:
            h_new, c_new, m_new = self.cell(x_t, h, c, m)

        # Optional stabilization (keep if you need it)
        c_new = c_new.clamp(-10.0, 10.0)
        m_new = m_new.clamp(-10.0, 10.0)

        return self.out_norm(h_new), (h_new, c_new, m_new)

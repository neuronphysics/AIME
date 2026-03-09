import torch
from torch import nn
from torch.nn import functional as F
from vdvae.vae_helpers import HModule, get_1x1, get_3x3, DmolNet, draw_gaussian_diag_samples, gaussian_analytical_kl
from collections import defaultdict
import numpy as np
import math
import itertools
from torch.utils.checkpoint import checkpoint as ckpt
from typing import Dict, Optional, Tuple
from VRNN.lstm import ConvLSTMCell
from VRNN.perceiver.modules import SelfAttentionBlock
from VRNN.perceiver.position import FourierPositionEncoding
#This code is from this source:https://github.com/openai/vdvae

class Block(nn.Module):
    def __init__(self, in_width, middle_width, out_width, down_rate=None, residual=False, use_3x3=True, zero_last=False):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c3 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out


def parse_layer_string(s):
    layers = []
    for ss in s.split(','):
        if 'x' in ss:
            res, num = ss.split('x')
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif 'm' in ss:
            res, mixin = [int(a) for a in ss.split('m')]
            layers.append((res, mixin))
        elif 'd' in ss:
            res, down_rate = [int(a) for a in ss.split('d')]
            layers.append((res, down_rate))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def pad_channels(t, width):
    d1, d2, d3, d4 = t.shape
    empty = torch.zeros(d1, width, d3, d4, device=t.device)
    empty[:, :d2, :, :] = t
    return empty


def get_width_settings(width, s):
    mapping = defaultdict(lambda: width)
    if s:
        s = s.split(',')
        for ss in s:
            k, v = ss.split(':')
            mapping[int(k)] = int(v)
    return mapping

class SpatialSelfAttention(nn.Module):
    """
    (B,C,H,W) -> tokens (B,HW,C) -> SelfAttentionBlock -> (B,C,H,W)
    With gated residual + gated Fourier pos enc for maximum stability.
    """
    def __init__(self, channels: int, res: int, H):
        super().__init__()
        self.channels = int(channels)
        self.res = int(res)
        self.use_checkpoint = getattr(H, "use_checkpoint", False)
        # --- defaults ---
        num_layers = int(getattr(H, "attn_num_layers", 1))
        num_heads  = int(getattr(H, "attn_num_heads", 4))
        widening   = int(getattr(H, "attn_widening_factor", 1))
        dropout    = float(getattr(H, "attn_dropout", 0.0))
        res_drop   = float(getattr(H, "attn_residual_dropout", 0.0))

        # Heads must divide qk/v channels 
        num_heads = min(num_heads, self.channels)
        while num_heads > 1 and (self.channels % num_heads != 0):
            num_heads -= 1

        # pick qk/v = channels (simple + safe), ensure divisible
        num_qk = self.channels
        num_v  = self.channels

        # optional pre-norm on attention branch (does NOT affect identity path)
        gn_groups = int(getattr(H, "attn_gn_groups", 32))
        gn_groups = max(1, min(gn_groups, self.channels))
        while self.channels % gn_groups != 0 and gn_groups > 1:
            gn_groups -= 1
        self.pre_norm = nn.GroupNorm(gn_groups, self.channels, eps=1e-6, affine=True)

        self.attn = SelfAttentionBlock(
            num_layers=num_layers,
            num_heads=num_heads,
            num_channels=self.channels,
            num_qk_channels=num_qk,
            num_v_channels=num_v,
            widening_factor=widening,
            dropout=dropout,
            residual_dropout=res_drop,
            activation_checkpointing=bool(getattr(H, "attn_activation_checkpointing", False)),
            activation_offloading=bool(getattr(H, "attn_activation_offloading", False)),
        )

        # --- Fourier 2D positional encoding (gated) ---
        self.use_pos = bool(getattr(H, "attn_use_pos_enc", True))
        if self.use_pos:
            n_bands = int(getattr(H, "attn_pos_num_bands", 6))
            self.pos_enc = FourierPositionEncoding((self.res, self.res), num_frequency_bands=n_bands)
            pos_dim = self.pos_enc.num_position_encoding_channels(include_positions=True) 
            self.pos_proj = nn.Linear(pos_dim, self.channels, bias=True)
            nn.init.normal_(self.pos_proj.weight, std=1e-3)
            nn.init.zeros_(self.pos_proj.bias)
            self.pos_gate = nn.Parameter(torch.zeros(()))  # init 0 => no pos effect
        else:
            self.pos_enc = None
            self.pos_proj = None
            self.pos_gate = None

        # --- attention residual gate (init 0 => exact identity) ---
        self.attn_gate = nn.Parameter(torch.zeros(()))  
        self.attn_gate.data.fill_(1e-3)  # 


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.channels, f"channels mismatch: got {C}, expected {self.channels}"
        assert H == self.res and W == self.res, f"res mismatch: got {H}x{W}, expected {self.res}x{self.res}"
        def attn_branch(x_in: torch.Tensor) -> torch.Tensor:
            # attention branch only (returns [B,C,H,W])
            xn = self.pre_norm(x_in)
            tok = xn.flatten(2).transpose(1, 2)  # [B, HW, C]

            if self.use_pos:
                pos = self.pos_enc(B).to(device=tok.device, dtype=tok.dtype)  # [B,HW,pos_dim]
                pos = self.pos_proj(pos)
                tok = tok + self.pos_gate * pos

            tok_out = self.attn(tok).last_hidden_state  # MUST be Tensor for checkpoint
            out_img = tok_out.transpose(1, 2).reshape(B, C, H, W)
            return out_img
        # Only checkpoint when it can actually save memory (training + grads)
        if self.use_checkpoint and self.training and x.requires_grad:
            out = ckpt(
                attn_branch,
                x,
                use_reentrant=False,  
            )
        else:
            out = attn_branch(x)
        g = torch.tanh(self.attn_gate)
        return x + g * (out - x)

class Encoder(HModule):
    def build(self):

        H = self.H
        self.use_checkpoint = getattr(H, "use_checkpoint", False)
        self.in_conv = get_3x3(H.image_channels, H.width)
        self.widths = get_width_settings(H.width, H.custom_width_str)
        
        self.attn_resolutions = set(getattr(self.H, "attn_resolutions", [8, 16]))
        use_spatial_attn = bool(getattr(H, "use_spatial_attn", True))
        attn_where = getattr(H, "attn_where", "last")  # options:"first" or "last"
        blockstr = parse_layer_string(H.enc_blocks)
        enc_blocks = []
        out_res_list = []

        for i, (res, down_rate) in enumerate(blockstr):
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            enc_blocks.append(Block(self.widths[res], int(self.widths[res] * H.bottleneck_multiple), self.widths[res], down_rate=down_rate, residual=True, use_3x3=use_3x3))
            out_res = res // down_rate if down_rate is not None else res
            
            out_res_list.append(out_res)
        stage_idxs = defaultdict(list)
        for i, r in enumerate(out_res_list):
            stage_idxs[r].append(i)

        # default: no attention anywhere
        enc_attn = [nn.Identity() for _ in enc_blocks]

        # place exactly one attention per requested resolution
        if use_spatial_attn:
            for r in self.attn_resolutions:
                if r in stage_idxs:
                    idx = stage_idxs[r][0] if attn_where == "first" else stage_idxs[r][-1]
                    enc_attn[idx] = SpatialSelfAttention(self.widths[r], r, H)

        n_blocks = len(blockstr)
        for b in enc_blocks:
            b.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.enc_attn = nn.ModuleList(enc_attn)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.in_conv(x)
        activations = {}
        activations[x.shape[2]] = x
        for i, block in enumerate(self.enc_blocks):
            if self.training and self.use_checkpoint:
                x = ckpt(block, x, use_reentrant=False)
            else:
                x = block(x)
            res = x.shape[2]
            x = x if x.shape[1] == self.widths[res] else pad_channels(x, self.widths[res])
            x = self.enc_attn[i](x)
            activations[res] = x
        return activations


class DecBlock(nn.Module):
    def __init__(self, H, res, mixin, n_blocks, is_top=False, use_temporal_prior: bool = False):
        super().__init__()
        self.base = res
        self.mixin = mixin
        self.is_top = is_top #added for top block
        self.H = H
        self.use_checkpoint = getattr(H, "use_checkpoint", False)
        self.widths = get_width_settings(H.width, H.custom_width_str)
        width = self.widths[res]
        use_3x3 = res > 2
        cond_width = int(width * H.bottleneck_multiple)
        self.zdim = H.zdim
        global_edge = bool(getattr(H, "use_edge_conditioning", False))
        min_edge_res = int(getattr(H, "edge_condition_min_res", 32))
        self.use_edge_conditioning = global_edge and (res >= min_edge_res)
        # Set this to match e_warp channels (typically 1 for edge map, 2 for flow, etc.)
        self.edge_channels = int(getattr(H, "edge_channels", 1))
        enc_in_width = width * 2 + (self.edge_channels if self.use_edge_conditioning else 0)
        self.enc = Block(enc_in_width, cond_width, H.zdim * 2, residual=False, use_3x3=use_3x3)
        # --- Temporal prior switch (Option B: no FiLM, no projection) ---
        # If H.use_temporal_priors is True, every NON-top stochastic block at this resolution
        # uses a *time-conditioned Gaussian prior*:
        #   p(z_t^(r) | x_t^(r), h_{t-1}^(r)) = N(mu(x,h), sigma^2(x,h))
        # We implement this by concatenating a broadcasted hidden map to x before the prior convs.
        self.use_temporal_priors = bool(getattr(H, "use_temporal_priors", False))
        self.action_dim = int(getattr(H, "action_dim", 0))
        self.use_temporal_prior = (bool(use_temporal_prior) and self.use_temporal_priors and (not self.is_top) and self.action_dim > 0)
        prior_in_width = width + (width if self.use_temporal_prior else 0) + (self.edge_channels if self.use_edge_conditioning else 0)

        self.prior = Block(prior_in_width, cond_width, H.zdim * 2 + width, residual=False, use_3x3=use_3x3, zero_last=True)
        self.z_proj = get_1x1(H.zdim, width)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
        self.resnet = Block(width, cond_width, width, residual=True, use_3x3=use_3x3)
        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.z_fn = lambda x: self.z_proj(x)

    def _prep_edge(self, edge_guide: Optional[torch.Tensor], x: torch.Tensor) -> Optional[torch.Tensor]:
        """edge_guide: [B,Ce,H0,W0] or [B,H0,W0,Ce] -> edge_map [B,Ce,H,W] matching x spatial size."""
        if not self.use_edge_conditioning:
            return None

        B, _, H, W = x.shape  # x is [B,C,H,W]
        Ce = self.edge_channels

        if edge_guide is None:
            # t=0 / reset: keep channel dims consistent by using zeros
            return torch.zeros(B, Ce, H, W, device=x.device, dtype=x.dtype)

        if edge_guide.dim() != 4:
            raise ValueError(f"edge_guide must be 4D, got {tuple(edge_guide.shape)}")

        # Accept NCHW or NHWC
        if edge_guide.shape[1] == Ce:
            edge = edge_guide
        elif edge_guide.shape[-1] == Ce:
            edge = edge_guide.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(
                f"edge_guide channel mismatch: expected Ce={Ce}, got {tuple(edge_guide.shape)}"
            )

        edge = edge.to(device=x.device, dtype=x.dtype)

        src_h, src_w = edge.shape[2], edge.shape[3]
        tgt_h, tgt_w = H, W

        if (src_h, src_w) != (tgt_h, tgt_w):
            # define block resolution r (your blocks are typically square)
            r = min(tgt_h, tgt_w)

            # only treat as downsampling if both dims shrink (or stay)
            downsample = (tgt_h <= src_h) and (tgt_w <= src_w)

            if downsample and r < 32:
                # anti-aliased downsample (keeps thin edges from disappearing as often)
                edge = F.adaptive_max_pool2d(edge, output_size=(tgt_h, tgt_w))

                # Alternative if one wants "edge presence" (keeps any edge in each cell):
                # edge = F.adaptive_max_pool2d(edge, output_size=(tgt_h, tgt_w))
            else:
                # keep edges crisp for higher resolutions, or when upsampling
                edge = F.interpolate(edge, size=(tgt_h, tgt_w), mode="nearest")

        return edge

    def sample(self, x, acts, edge_guide: Optional[torch.Tensor] = None):

        """Posterior sample z ~ q(z|x,enc_act) and compute Gaussian KL against the block prior.
        - For the top block, we do not use the Gaussian KL (because top prior is DPGMM).
          We return kl=0 for the top block, and VDVAE computes the DPGMM KL separately.
        """
        enc_in = torch.cat([x, acts], dim=1)
        edge_map = self._prep_edge(edge_guide, x)
        if edge_map is not None:
            enc_in = torch.cat([enc_in, edge_map], dim=1)

        if self.training and self.use_checkpoint:
            qm, qv = ckpt(self.enc, enc_in, use_reentrant=False).chunk(2, dim=1)
            feats = self._prior_forward(x, None, edge_map =edge_map)
        else:
            qm, qv = self.enc(enc_in).chunk(2, dim=1)
            feats = self._prior_forward(x, None, edge_map =edge_map)

        pm = feats[:, :self.zdim, ...]
        pv = feats[:, self.zdim:self.zdim * 2, ...]
        xpp = feats[:, self.zdim * 2:, ...]
        x = x + xpp

        z = draw_gaussian_diag_samples(qm, qv)

        if getattr(self, "is_top", False):
            kl = torch.zeros_like(qm)
        else:
            kl = gaussian_analytical_kl(qm, pm, qv, pv)

        return z, x, kl, qm, qv


    def sample_uncond(self, x, t=None, lvs=None, edge_guide: Optional[torch.Tensor] = None):
        n, c, h, w = x.shape
        edge_map = self._prep_edge(edge_guide, x)
        feats = self._prior_forward(x, None, edge_map =edge_map)
        pm, pv, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]
        x = x + xpp
        if lvs is not None:
            z = lvs
        else:
            if t is not None:
                pv = pv + torch.ones_like(pv) * np.log(t)
            z = draw_gaussian_diag_samples(pm, pv)
        return z, x
    
    def _prior_forward(self, x: torch.Tensor, h_map: Optional[torch.Tensor] = None, edge_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the Gaussian prior conv tower.
        If self.use_temporal_prior is True, the prior is conditioned on:
          - x:     current decoder feature map            [B, width, H, W]
          - h_map: temporal hidden map (per-location)     [B, width, H, W]
        Some call-sites (e.g. unconditional sampling) pass h_map=None.
        In that case, we fall back to zeros, which degenerates to an unconditional prior.
        """
        if self.use_temporal_prior:
            if h_map is None:
                h_map = torch.zeros_like(x)
            else:
                # Accept token form [B*H*W, C] (from per-location LSTM) and reshape to map.
                if h_map.dim() == 2:
                    B, C, H, W = x.shape
                    if h_map.shape[0] == B * H * W and h_map.shape[1] == C:
                        h_map = h_map.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                    else:
                        raise ValueError(
                            f"h_map token shape {tuple(h_map.shape)} incompatible with x {tuple(x.shape)}"
                        )
                elif h_map.dim() != 4:
                    raise ValueError(f"h_map must be None, [B,C,H,W], or [B*H*W,C]; got {tuple(h_map.shape)}")

                # If spatial size mismatches, resize (should usually already match).
                if h_map.shape[2:] != x.shape[2:]:
                    h_map = F.interpolate(h_map, size=x.shape[2:], mode='nearest')

                if h_map.shape[0] != x.shape[0]:
                    raise ValueError(f"Batch mismatch: h_map {h_map.shape[0]} vs x {x.shape[0]}")
                if h_map.shape[1] != x.shape[1]:
                    raise ValueError(f"Channel mismatch: h_map {h_map.shape[1]} vs x {x.shape[1]}")
            prior_in = torch.cat([x, h_map], dim=1)   # [B, 2*width, H, W]
        else:
            prior_in = x
        if self.use_edge_conditioning:
            if edge_map is None:
                # if enabled but missing, use zeros so shapes stay consistent
                edge_map = torch.zeros(x.shape[0], self.edge_channels, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
            prior_in = torch.cat([prior_in, edge_map], dim=1)
            
        if self.training and self.use_checkpoint:
            return ckpt(self.prior, prior_in, use_reentrant=False)
        return self.prior(prior_in)
 
    def sample_temporal(self, x: torch.Tensor, enc_act: torch.Tensor, h_map: torch.Tensor, edge_guide: Optional[torch.Tensor] = None):
        """Posterior sample + KL against a *time-conditioned* Gaussian prior.

        Args:
          x:      current decoder feature map at this block/resolution  [B,width,H,W]
          enc_act:encoder activation for q(z|x)                         [B,width,H,W]
          h_map:  token-wise temporal hidden map h_{t-1}^(r)           [B,width,H,W]

        Returns:
          z:  sampled posterior latent map                              [B,zdim,H,W]
          x:  updated feature map after adding xpp                       [B,width,H,W]
          kl: per-location Gaussian KL                                   [B,zdim,H,W]
          qm/qv: posterior parameters                                    [B,zdim,H,W]
        """
        enc_in = torch.cat([x, enc_act], dim=1)
        edge_map = self._prep_edge(edge_guide, x)
        if edge_map is not None:
            enc_in = torch.cat([enc_in, edge_map], dim=1)

        if self.training and self.use_checkpoint:
            qm, qv = ckpt(self.enc, enc_in, use_reentrant=False).chunk(2, dim=1)
            feats = self._prior_forward(x, h_map, edge_map =edge_map)
        else:
            qm, qv = self.enc(enc_in).chunk(2, dim=1)
            feats = self._prior_forward(x, h_map, edge_map=edge_map)

        pm = feats[:, :self.zdim, ...]
        pv = feats[:, self.zdim:self.zdim * 2, ...]
        xpp = feats[:, self.zdim * 2:, ...]
        x = x + xpp

        # posterior sample
        z = draw_gaussian_diag_samples(qm, qv)

        # KL(q || p) with temporal prior
        kl = gaussian_analytical_kl(qm, pm, qv, pv)
        return z, x, kl, qm, qv


    def sample_uncond_temporal(self, x: torch.Tensor, h_map: torch.Tensor, t: Optional[float] = None, lvs=None, temperature: float = 1.0, edge_guide: Optional[torch.Tensor] = None):
        """Prior sample from a *time-conditioned* Gaussian prior.

        Args:
          x:     decoder feature map at this block         [B,width,H,W]
          h_map: temporal hidden map h_{t-1}^(r)           [B,width,H,W]
        """
        edge_map = self._prep_edge(edge_guide, x)
        feats = self._prior_forward(x, h_map, edge_map=edge_map)
        pm = feats[:, :self.zdim, ...]
        pv = feats[:, self.zdim:self.zdim * 2, ...]
        xpp = feats[:, self.zdim * 2:, ...]
        x = x + xpp

        if lvs is not None:
            z = lvs
        else:
            temp_mult = float(temperature)
            if t is not None:
                temp_mult *= float(t)
            if temp_mult == 0.0: # TODO: is this ok?
                z = pm
                return z, x            
            if temp_mult != 1.0:
                pv = pv + torch.ones_like(pv) * math.log(temp_mult)
            z = draw_gaussian_diag_samples(pm, pv)
        return z, x

    def forward_temporal(
        self,
        xs: Dict[int, torch.Tensor],
        activations: Dict[int, torch.Tensor],
        temporal_cell: ConvLSTMCell,
        temporal_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        a_t: torch.Tensor,
        mask_t: Optional[torch.Tensor] = None,
        edge_guide: Optional[torch.Tensor] = None,
        get_latents: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Teacher-forced temporal forward for the first stochastic block at a resolution."""
        x, acts = self.get_inputs(xs, activations)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)

        B, _, H, W = x.shape
        if temporal_state is None:
            h_prev = torch.zeros(B, temporal_cell.hidden_dim, H, W, device=x.device, dtype=x.dtype)  
            c_prev = torch.zeros(B, temporal_cell.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        else:
            h_prev, c_prev = temporal_state

        if mask_t is not None:
            keep = mask_t.view(B, 1, 1, 1).to(device=x.device, dtype=x.dtype)
            h_prev = h_prev * keep
            c_prev = c_prev * keep

        z, x, kl, qm, qv = self.sample_temporal(x, acts, h_map=h_prev, edge_guide =edge_guide)
        stats = {'kl': kl, 'posterior_mean': qm, 'posterior_logvar': qv}

        action_dim = int(a_t.shape[-1])
        a_map = a_t.view(B, action_dim, 1, 1).to(device=x.device, dtype=x.dtype).expand(B, action_dim, H, W)
        u_map = torch.cat([z, a_map], dim=1)
        h_next, c_next = temporal_cell(u_map, h_prev, c_prev)

        x = x + self.z_fn(z)
        if self.training and self.use_checkpoint:
            x = ckpt(self.resnet, x, use_reentrant=False)
        else:
            x = self.resnet(x)

        xs[self.base] = x
        if get_latents:
            stats['z'] = z
        return xs, stats, (h_next, c_next)
    def forward_uncond_temporal(
        self,
        xs: Dict[int, torch.Tensor],
        temporal_cell: ConvLSTMCell,
        temporal_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        a_t: torch.Tensor,
        mask_t: Optional[torch.Tensor] = None,
        edge_guide: Optional[torch.Tensor] = None,
        t: Optional[float] = None,
        lvs: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Tuple[Dict[int, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Unconditional temporal forward (generation) for the first stochastic block at a resolution."""
        try:
            x = xs[self.base]
        except KeyError:
            # Create a zero feature map if this resolution hasn't been created yet
            ref = xs[min(xs.keys())]
            x = torch.zeros(ref.shape[0], temporal_cell.hidden_dim, self.base, self.base, device=ref.device, dtype=ref.dtype)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        B, _, H, W = x.shape
        if temporal_state is None:
            h_prev = torch.zeros(B, temporal_cell.hidden_dim, H, W, device=x.device, dtype=x.dtype)
            c_prev = torch.zeros(B, temporal_cell.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        else:
            h_prev, c_prev = temporal_state

        if mask_t is not None:
            keep = mask_t.view(B, 1, 1, 1).to(device=x.device, dtype=x.dtype)
            h_prev = h_prev * keep
            c_prev = c_prev * keep
        # sample z from prior p(z|x, h_prev)
        z, x = self.sample_uncond_temporal(x, h_prev, t=t, lvs=lvs, temperature=temperature, edge_guide=edge_guide)

        action_dim = int(a_t.shape[-1])
        a_map = a_t.view(B, action_dim, 1, 1).to(device=x.device, dtype=x.dtype).expand(B, action_dim, H, W)
        u_map = torch.cat([z, a_map], dim=1)
        h_next, c_next = temporal_cell(u_map, h_prev, c_prev)

        x = x + self.z_fn(z)
        if self.training and self.use_checkpoint:
            x = ckpt(self.resnet, x, use_reentrant=False)
        else:
            x = self.resnet(x)

        xs[self.base] = x
        return xs, (h_next, c_next)
    def get_inputs(self, xs, activations):
        acts = activations[self.base]
        try:
            x = xs[self.base]
        except KeyError:
            x = torch.zeros_like(acts)
        if acts.shape[0] != x.shape[0]:
            x = x.repeat(acts.shape[0], 1, 1, 1)
        return x, acts

    def forward(self, xs, activations, edge_guide: Optional[torch.Tensor] = None, get_latents=False):
        x, acts = self.get_inputs(xs, activations)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x, kl, qm, qv = self.sample(x, acts, edge_guide=edge_guide)
        x = x + self.z_fn(z)
        if self.training and self.use_checkpoint:
            x = ckpt(self.resnet, x, use_reentrant=False)
        else:
            x = self.resnet(x)
        xs[self.base] = x
        stats = {'kl': kl, 'posterior_mean': qm, 'posterior_logvar': qv}
        if get_latents:
            stats['z'] = z.detach()
            return xs, stats
        return xs, stats

    def forward_uncond(self, xs, t=None, lvs=None, edge_guide: Optional[torch.Tensor] = None):
        try:
            x = xs[self.base]
        except KeyError:
            ref = xs[list(xs.keys())[0]]
            x = torch.zeros(dtype=ref.dtype, size=(ref.shape[0], self.widths[self.base], self.base, self.base), device=ref.device)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x = self.sample_uncond(x, t, lvs=lvs, edge_guide=edge_guide)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x
        return xs


class Decoder(HModule):

    def build(self):
        H = self.H
        resos = set()
        dec_blocks = []
        self.widths = get_width_settings(H.width, H.custom_width_str)
        blocks = parse_layer_string(H.dec_blocks)
        # --- Decoder self-attn placement (mirrors encoder idea) ---
        self.attn_resolutions = set(getattr(H, "attn_resolutions", [8, 16]))
        use_spatial_attn = bool(getattr(H, "use_spatial_attn", True))
        attn_where = getattr(H, "attn_where", "last")  # "first" or "last"       
        n_blocks = len(blocks)
        first_only = bool(getattr(H, "temporal_first_block_only", True))
        seen_temporal_res = set()
        stage_idxs = defaultdict(list)
        for idx, (res, mixin) in enumerate(blocks):
            stage_idxs[res].append(idx)
            is_top = (idx == 0)
            # Temporal priors should apply ONLY to the *first* stochastic block at each resolution (requested).
            # If temporal_first_block_only=False, we temporalize every non-top stochastic block.
            use_temporal_here = False
            if bool(getattr(H, "use_temporal_priors", False)) and (not is_top) and int(getattr(H, "action_dim", 0)) > 0:
                if (not first_only) or (res not in seen_temporal_res):
                    use_temporal_here = True
                    seen_temporal_res.add(res)
            dec_blocks.append(DecBlock(H, res, mixin, n_blocks=n_blocks, is_top=is_top, use_temporal_prior=use_temporal_here))
            resos.add(res)
        self.resolutions = sorted(resos)
        dec_attn = [nn.Identity() for _ in blocks]

        # choose exactly one block per requested resolution
        self.dec_attn_indices = {}  # optional: for debugging/logging
        if use_spatial_attn:
            for r in self.attn_resolutions:
                if r in stage_idxs:
                    idx = stage_idxs[r][0] if attn_where == "first" else stage_idxs[r][-1]
                    dec_attn[idx] = SpatialSelfAttention(self.widths[r], r, H)
                    self.dec_attn_indices[r] = idx

        self.dec_attn = nn.ModuleList(dec_attn)

        self.dec_blocks = nn.ModuleList(dec_blocks)
        # ------------------------------------------------------------
        # Temporal priors: one LSTM per *resolution* (excluding the top block)
        #
        # For each resolution r, we keep a state (h^(r), c^(r)) and condition
        # the Gaussian prior at that resolution on:
        #   - current decoder feature map x_t^(r)
        #   - previous temporal hidden h_{t-1}^(r)
        #
        # LSTM input:
        #   rnn_in = concat(flatten(z_t^(r)), a_t)
        # ------------------------------------------------------------
        self.use_temporal_priors = bool(getattr(H, "use_temporal_priors", False)) and int(getattr(H, "action_dim", 0)) > 0
        self.action_dim = int(getattr(H, "action_dim", 0))
        self.temporal_n_lstm_layers = int(getattr(H, "temporal_n_lstm_layers", 1))
        self.temporal_use_orthogonal = bool(getattr(H, "temporal_use_orthogonal", True))


        if self.use_temporal_priors:
            # One ConvLSTMCell per resolution. Only the first stochastic block at each resolution
            # uses the temporal prior (unless H.temporal_first_block_only is False).
            action_dim = int(getattr(self.H, "action_dim", 0))
            if action_dim <= 0:
                raise ValueError(
                    "Temporal priors require H.action_dim > 0."
                )

            self.temporal_cells = nn.ModuleDict()
            self.temporal_resolutions = sorted({b.base for b in self.dec_blocks if (not b.is_top) and b.use_temporal_prior})
            for res in self.temporal_resolutions:
                width_r = int(self.widths[res])
                input_dim = int(self.H.zdim + action_dim)
                self.temporal_cells[str(res)] = ConvLSTMCell(
                    input_dim=input_dim,
                    hidden_dim=width_r,
                    kernel_size=int(getattr(H, "temporal_kernel_size",3)),
                    bias=True,
                )
        else:
            self.temporal_cells = nn.ModuleDict()
            self.temporal_resolutions = []
        self.bias_xs = nn.ParameterList([nn.Parameter(torch.zeros(1, self.widths[res], res, res)) for res in self.resolutions if res <= H.no_bias_above])
        self.out_net = DmolNet(H)
        self.gain = nn.Parameter(torch.ones(1, H.width, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, H.width, 1, 1))
        self.final_fn = lambda x: x * self.gain + self.bias

    def forward(self, activations, edge_guide: Optional[torch.Tensor] = None, get_latents=False):
        stats = []
        xs = {a.shape[2]: a for a in self.bias_xs}
        for idx, block in enumerate(self.dec_blocks):
            xs, block_stats = block(xs, activations, edge_guide=edge_guide, get_latents=get_latents)
            xs[block.base] = self.dec_attn[idx](xs[block.base]) #attention
            stats.append(block_stats)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size], stats

    def forward_uncond(self, n, t=None, y=None):
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        for idx, block in enumerate(self.dec_blocks):
            try:
                temp = t[idx]
            except TypeError:
                temp = t
            xs = block.forward_uncond(xs, temp)
            xs[block.base] = self.dec_attn[idx](xs[block.base])
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size]

    def forward_manual_latents(self, n, latents, t=None):
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        for idx, (block, lvs) in enumerate(itertools.zip_longest(self.dec_blocks, latents)):
            xs = block.forward_uncond(xs, t, lvs=lvs)
            xs[block.base] = self.dec_attn[idx](xs[block.base])  #attention
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size]


    def init_temporal_state(self, B: int, device, dtype=torch.float32) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Initialize ConvLSTM states per resolution.

        Returns dict: res(str) -> (h_map, c_map) with shapes [B, width_r, res, res].
        """
        if not self.use_temporal_priors:
            return {}

        state: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for res in self.temporal_resolutions:
            r = int(res)
            width_r = int(self.widths[r])
            h = torch.zeros(B, width_r, r, r, device=device, dtype=dtype)
            c = torch.zeros(B, width_r, r, r, device=device, dtype=dtype)
            state[str(res)] = (h, c)
        return state

    def forward_temporal(
        self,
        activations: Dict[int, torch.Tensor],
        temporal_state: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        a_t: torch.Tensor,
        mask_t: torch.Tensor,
        edge_guide: Optional[torch.Tensor] = None,
        get_latents: bool = False,
    ):
        """Decode ONE frame (teacher-forced) while using temporal priors for lower layers."""
        xs = {b.shape[2]: b for b in self.bias_xs}
        stats = []
        for idx, block in enumerate(self.dec_blocks):
            if self.use_temporal_priors and (not getattr(block, "is_top", False)) and block.use_temporal_prior:
                key = str(block.base)
                rnn = self.temporal_cells[key]
                st = temporal_state[key]
                xs, st_dict, st_new = block.forward_temporal(xs, activations, rnn, st, a_t, mask_t, edge_guide=edge_guide, get_latents=get_latents)
                temporal_state[key] = st_new
                xs[block.base] = self.dec_attn[idx](xs[block.base])
                stats.append(st_dict)
            else:
                xs, st_dict = block(xs, activations, edge_guide=edge_guide, get_latents=get_latents)
                xs[block.base] = self.dec_attn[idx](xs[block.base])
                stats.append(st_dict)

        px_z = self.final_fn(xs[self.H.image_size])
        return px_z, stats, temporal_state

    def forward_manual_latents_temporal(
        self,
        n: int,
        latents,
        temporal_state: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        a_t: torch.Tensor,
        mask_t: torch.Tensor,
        edge_guide: Optional[torch.Tensor] = None,
        t: Optional[float] = None,
        temperature : float=1.0,
    ):
        """Decode ONE frame from a provided top latent + temporally conditioned lower priors."""
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)

        for idx,(block, lvs) in enumerate(itertools.zip_longest(self.dec_blocks, latents)):
            if getattr(block, "is_top", False):
                xs = block.forward_uncond(xs, t=t, lvs=lvs, edge_guide=edge_guide)
                xs[block.base] = self.dec_attn[idx](xs[block.base])
                continue

            if self.use_temporal_priors and block.use_temporal_prior:
                key = str(block.base)
                rnn = self.temporal_cells[key]
                st = temporal_state[key]
                xs, st_new = block.forward_uncond_temporal(xs, rnn, st, a_t, mask_t, edge_guide=edge_guide, t=t, lvs=lvs, temperature= temperature)
                temporal_state[key] = st_new
                xs[block.base] = self.dec_attn[idx](xs[block.base])
            else:
                xs = block.forward_uncond(xs, t=t, lvs=lvs, edge_guide=edge_guide)
                xs[block.base] = self.dec_attn[idx](xs[block.base])

        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size], temporal_state

class VDVAE(HModule):
    def __init__(
        self,
        H,
        prior=None,
        top_kl_weight: float = 1.0,
        prior_kl_mc_samples: int = 10,
    ):
        """
        prior:            DPGMMPrior or None
        top_kl_weight:    scalar multiplier for the DPGMM KL term
        prior_kl_mc_samples: number of MC samples used in DPGMM KL estimation
        """
        super().__init__(H)  # calls self.build()
        self.prior = prior
        self.top_kl_weight = top_kl_weight
        self.prior_kl_mc_samples = prior_kl_mc_samples
        # Fourier PE for conditioning DP-GMM prior at top resolution.
        base = getattr(self.decoder.dec_blocks[0], "base", 4)
        if isinstance(base, (tuple, list)):
            top_h, top_w = int(base[0]), int(base[1])
        else:
            top_h = top_w = int(base)

        if self.prior is not None:
            C = int(self.prior.hidden_dim)
            D = 2  # (top_h, top_w)
            num_bands = max(1, int((C - D) // (2 * D)))
        else:
            num_bands = 2

        self.top_pe = FourierPositionEncoding(input_shape=(top_h, top_w), num_frequency_bands=num_bands)

    def build(self):
        self.encoder = Encoder(self.H)
        self.decoder = Decoder(self.H)

    @staticmethod
    def add_PosEncode(x: torch.Tensor, pe: nn.Module, scale: float = 0.05) -> torch.Tensor:
        """
        x: [B, C, H, W]
        pe(B) must return [B, H*W, Cpos]
        returns: [B, C, H, W] (no dim change, no projection)
        """
        B, C, H, W = x.shape

        enc = pe(B)  # [B, H*W, Cpos]
        enc = enc.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # [B, Cpos, H, W]
        Cpos = enc.shape[1]

        # tile/truncate to match C channels
        if Cpos >= C:
            encC = enc[:, :C]
        else:
            encC = x.new_zeros(B, C, H, W)
            encC[:, :Cpos] = enc

        return x + (scale * encC)

    def _coerce_h_context_map(self, h_context: torch.Tensor, Ht: int, Wt: int) -> torch.Tensor:

        """Coerce h_context into a spatial map [B, Hc, Ht, Wt] WITHOUT pooling/summing.
        Supported inputs:
          - [B, Hc, Ht, Wt] : already a spatial map (preferred; matches SpatioTemporalCore output)
          - [B, Hc]         : broadcast to all spatial locations (no information is destroyed; just repeated)
          - [B, Hc*Ht*Wt]   : reshaped into a map (legacy flattened-map path)
        If spatial size mismatches Ht/Wt we use interpolation (not pooling) to match.

        """
        assert self.prior is not None, "h_context is only used when self.prior is not None"

        Hc = int(self.prior.hidden_dim)
        if h_context.dim() == 4:
            # [B, Hc, H?, W?]
            h_map = h_context
            if h_map.shape[1] != Hc:
                raise ValueError(f"h_context channels {h_map.shape[1]} != prior.hidden_dim {Hc}")
        elif h_context.dim() == 2:
            B, D = h_context.shape
            if D == Hc:
                # Broadcast vector -> map (no pooling).
                h_map = h_context[:, :, None, None].expand(B, Hc, Ht, Wt).contiguous()
            elif D == Hc * Ht * Wt:
                # Legacy flattened map.
                h_map = h_context.view(B, Hc, Ht, Wt).contiguous()
            else:
                raise ValueError(f"h_context dim {D} is incompatible with prior.hidden_dim={Hc} and (Ht,Wt)=({Ht},{Wt}).")
        else:
            raise ValueError(f"h_context must be [B,Hc,Ht,Wt] or [B,Hc] or [B,Hc*Ht*Wt], got {tuple(h_context.shape)}")

        if h_map.shape[2] != Ht or h_map.shape[3] != Wt:
            # Match spatial size without destroying channel information.
            h_map = F.interpolate(h_map, size=(Ht, Wt), mode="bilinear", align_corners=False)

        return h_map

    # ======================================================================
    # Temporal extensions
    #   - Top block: DPGMM prior conditioned on top-level VRNN hidden state h_context.
    #   - Lower blocks (each resolution): time-conditioned Gaussian priors, one LSTM per resolution.
    #       * Prior: p(z_t^(r) | x_t^(r), h_{t-1}^(r))
    #       * LSTM update: h_t^(r) = LSTM([flatten(z_t^(r)), a_t], h_{t-1}^(r))
    #       * Masking: if mask_t[b] == 0, the LSTM state for that sample resets.
    # We flatten the full z map into a vector.
    # ======================================================================

    def init_temporal_state(self, B: int, device: torch.device, dtype: Optional[torch.dtype] = None):
        """Initialize temporal states for all lower-resolution LSTMs (one per resolution)."""
        return self.decoder.init_temporal_state(B, device=device, dtype=dtype)

    def forward_temporal_step(
        self,
        x: torch.Tensor,           # [B,H,W,C] NHWC in [-1,1]
        x_target: torch.Tensor,    # [B,H,W,C] NHWC in [-1,1]
        h_context: torch.Tensor,   # [B, hidden_dim, H, W] top VRNN hidden
        a_t: torch.Tensor,         # [B, action_dim]
        mask_t: torch.Tensor,      # [B] float {0,1}
        temporal_state: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        edge_guide: Optional[torch.Tensor] = None,
        get_latents: bool = False,
    ):
        """Teacher-forced forward for one time step with an image-level DP-GMM top prior."""
        activations = self.encoder.forward(x)

        px_z, stats, temporal_state = self.decoder.forward_temporal(
            activations=activations,
            temporal_state=temporal_state,
            a_t=a_t,
            mask_t=mask_t,
            edge_guide=edge_guide,
            get_latents=get_latents,
        )

        distortion_per_pixel = self.decoder.out_net.nll(px_z, x_target)   # [B]
        ndims = float(np.prod(x_target.shape[1:]))                        # C * H * W

        rate_gauss = torch.zeros_like(distortion_per_pixel)               # [B]
        top_q_mean_map = None
        top_q_logsig_map = None

        for block, st in zip(self.decoder.dec_blocks, stats):
            kl_block = st["kl"].sum(dim=(1, 2, 3))                        # [B]
            if getattr(block, "is_top", False):
                top_q_mean_map = st.get("posterior_mean", None)
                top_q_logsig_map = st.get("posterior_logvar", None)       # log-sigma
            else:
                rate_gauss = rate_gauss + kl_block

        rate_gauss = rate_gauss / ndims

        dp_kl_img = None
        dp_rate = torch.zeros_like(distortion_per_pixel)
        prior_params = None

        if (
            self.prior is not None
            and top_q_mean_map is not None
            and top_q_logsig_map is not None
        ):
            B, C, Ht, Wt = top_q_mean_map.shape

            h_map = self._coerce_h_context_map(h_context, Ht=Ht, Wt=Wt)
            h_map = self.add_PosEncode(h_map, self.top_pe, scale=0.05)

            top_q_mean_img = top_q_mean_map.contiguous().view(B, C * Ht * Wt)
            top_q_logvar_img = (2.0 * top_q_logsig_map).contiguous().view(B, C * Ht * Wt)

            _, prior_params = self.prior(h_map)
            dp_kl_img = self.prior.compute_kl_divergence_mc(
                posterior_mean=top_q_mean_img,
                posterior_logvar=top_q_logvar_img,
                prior_params=prior_params,
                n_samples=self.prior_kl_mc_samples,
                reduction="image",
            )  # [B]

            dp_rate = self.top_kl_weight * dp_kl_img / ndims

        total_rate = rate_gauss + dp_rate                                 # [B]
        elbo_per_sample = distortion_per_pixel + total_rate               # [B]

        vm = mask_t.float()
        den = vm.sum().clamp(min=1.0)

        def masked_mean(x):
            return (x * vm).sum() / den

        out = {
            "elbo": masked_mean(elbo_per_sample),
            "distortion": masked_mean(distortion_per_pixel),
            "gauss_rate": masked_mean(rate_gauss),
            "rate": masked_mean(total_rate),
            "dp_rate": masked_mean(dp_rate),
            "stats": stats,
            "px_z": px_z,
            "top_q_mean_map": top_q_mean_map,
            "top_q_logvar_map": top_q_logsig_map,
            "prior_params": prior_params,
            "valid_count": den,
        }

        if dp_kl_img is not None:
            out["dp_kl"] = masked_mean(dp_kl_img)

        return out, temporal_state

    def decode_from_top_latent_temporal(
        self,
        z_top_map: torch.Tensor,   # [B,zdim,Ht,Wt]
        a_t: torch.Tensor,         # [B,action_dim]
        mask_t: torch.Tensor,      # [B]
        temporal_state: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        e_warp: Optional[torch.Tensor] = None,
        t: Optional[float] = None,
        temperature: float = 1.0,
    ):
        """Decode ONE frame from a top z map while sampling lower latents from temporal priors."""
        n = z_top_map.shape[0]
        n_blocks = len(self.decoder.dec_blocks)
        latents = [z_top_map] + [None] * (n_blocks - 1)

        px_z, temporal_state = self.decoder.forward_manual_latents_temporal(
            n=n,
            latents=latents,
            temporal_state=temporal_state,
            a_t=a_t,
            mask_t=mask_t,
            edge_guide=e_warp,
            t=t,
            temperature = temperature
        )
        return px_z, temporal_state

    def forward(self, x: torch.Tensor, x_target: torch.Tensor, h_context: Optional[torch.Tensor] = None):
        """
        Non-temporal forward pass with an image-level DP-GMM prior at the top latent.
        """
        activations = self.encoder.forward(x)
        px_z, stats = self.decoder.forward(activations, get_latents=True)

        distortion_per_pixel = self.decoder.out_net.nll(px_z, x_target)  # [B]
        ndims = float(np.prod(x.shape[1:]))

        rate_gauss = torch.zeros_like(distortion_per_pixel)
        top_q_mean_map = None
        top_q_logsig_map = None

        for block, statdict in zip(self.decoder.dec_blocks, stats):
            kl_block = statdict["kl"].sum(dim=(1, 2, 3))
            if getattr(block, "is_top", False):
                if "posterior_mean" in statdict and "posterior_logvar" in statdict:
                    top_q_mean_map = statdict["posterior_mean"]
                    top_q_logsig_map = statdict["posterior_logvar"]   # log-sigma
            else:
                rate_gauss = rate_gauss + kl_block

        rate_gauss = rate_gauss / ndims

        dp_kl_img = None
        prior_params = None
        dp_rate = torch.zeros_like(distortion_per_pixel)

        if (
            self.prior is not None
            and top_q_mean_map is not None
            and top_q_logsig_map is not None
        ):
            B, C, Ht, Wt = top_q_mean_map.shape

            if h_context is None:
                raise ValueError("h_context must be provided when using the DPGMMPrior.")
            if h_context.shape[0] != B:
                raise ValueError(
                    f"h_context batch size {h_context.shape[0]} does not match top latent batch size {B}."
                )
            if h_context.shape[1] != self.prior.hidden_dim:
                raise ValueError(
                    f"h_context dim {h_context.shape[1]} does not match prior.hidden_dim={self.prior.hidden_dim}."
                )

            h_map = self._coerce_h_context_map(h_context, Ht=Ht, Wt=Wt)
            h_map = self.add_PosEncode(h_map, self.top_pe, scale=0.05)

            top_q_mean_img = top_q_mean_map.contiguous().view(B, C * Ht * Wt)
            top_q_logvar_img = (2.0 * top_q_logsig_map).contiguous().view(B, C * Ht * Wt)

            _, prior_params = self.prior(h_map)
            dp_kl_img = self.prior.compute_kl_divergence_mc(
                posterior_mean=top_q_mean_img,
                posterior_logvar=top_q_logvar_img,
                prior_params=prior_params,
                n_samples=self.prior_kl_mc_samples,
                reduction="image",
            )  # [B]

            dp_rate = self.top_kl_weight * dp_kl_img / ndims

        total_rate_per_pixel = rate_gauss + dp_rate
        elbo_per_sample = distortion_per_pixel + total_rate_per_pixel

        out = dict(
            elbo=elbo_per_sample.mean(),
            distortion=distortion_per_pixel.mean(),
            gauss_rate=rate_gauss.mean(),
            rate=total_rate_per_pixel.mean(),
            dp_rate=dp_rate.mean(),
            prior_params=prior_params,
            px_z=px_z,
            top_q_mean_map=top_q_mean_map,
            top_q_logvar_map=top_q_logsig_map,
        )

        if dp_kl_img is not None:
            out["dp_kl"] = dp_kl_img.mean()

        return out

    def sample(self, n_batch: int, h_context: torch.Tensor):
        """
        Sample from the image-level DP-GMM top prior and decode the resulting top latent map.

        h_context: [B, hidden_dim, Ht, Wt], where B == n_batch
        """
        assert self.prior is not None, "VDVAE.sample requires a DPGMMPrior."
        assert h_context.shape[0] == n_batch, (
            f"h_context batch {h_context.shape[0]} != n_batch {n_batch}"
        )
        assert h_context.shape[1] == self.prior.hidden_dim, (
            f"h_context dim {h_context.shape[1]} != prior.hidden_dim {self.prior.hidden_dim}"
        )

        top_block = self.decoder.dec_blocks[0]
        C = top_block.zdim
        res = top_block.base
        B = n_batch

        h_map = self._coerce_h_context_map(h_context, Ht=res, Wt=res)
        h_map = self.add_PosEncode(h_map, self.top_pe, scale=0.05)

        prior_dist, _ = self.prior(h_map)
        z_img = prior_dist.sample()  # [B, C * res * res]

        expected_D = C * res * res
        if z_img.shape != (B, expected_D):
            raise ValueError(
                f"Sampled top latent has shape {tuple(z_img.shape)}, expected {(B, expected_D)}"
            )

        z_top_map = z_img.view(B, C, res, res).contiguous()

        latents = [z_top_map] + [None] * (len(self.decoder.dec_blocks) - 1)
        px_z = self.decoder.forward_manual_latents(n_batch, latents, t=None)
        return self.decoder.out_net.sample(px_z)

class VAE(HModule):
    def build(self):
        self.encoder = Encoder(self.H)
        self.decoder = Decoder(self.H)

    def forward(self, x, x_target):
        activations = self.encoder.forward(x)
        px_z, stats = self.decoder.forward(activations)
        distortion_per_pixel = self.decoder.out_net.nll(px_z, x_target)
        rate_per_pixel = torch.zeros_like(distortion_per_pixel)
        ndims = np.prod(x.shape[1:])
        for statdict in stats:
            rate_per_pixel += statdict['kl'].sum(dim=(1, 2, 3))
        rate_per_pixel /= ndims
        elbo = (distortion_per_pixel + rate_per_pixel).mean()
        return dict(elbo=elbo, distortion=distortion_per_pixel.mean(), rate=rate_per_pixel.mean())

    def forward_get_latents(self, x):
        activations = self.encoder.forward(x)
        _, stats = self.decoder.forward(activations, get_latents=True)
        return stats

    def forward_uncond_samples(self, n_batch, t=None):
        px_z = self.decoder.forward_uncond(n_batch, t=t)
        return self.decoder.out_net.sample(px_z)

    def forward_samples_set_latents(self, n_batch, latents, t=None):
        px_z = self.decoder.forward_manual_latents(n_batch, latents, t=t)
        return self.decoder.out_net.sample(px_z)
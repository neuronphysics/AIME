import itertools
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as ckpt
from vdvae.vae_helpers import HModule,DmolNet, draw_gaussian_diag_samples,gaussian_analytical_kl,get_1x1,get_3x3
from VRNN.perceiver.modules import SelfAttentionBlock
from VRNN.perceiver.position import FourierPositionEncoding

class Block(nn.Module):
    def __init__(
        self,
        in_width,
        middle_width,
        out_width,
        down_rate=None,
        residual=False,
        use_3x3=True,
        zero_last=False,
    ):
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
    b, c, h, w = t.shape
    empty = torch.zeros(b, width, h, w, device=t.device, dtype=t.dtype)
    empty[:, :c, :, :] = t
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
    Applied only in the encoder, and only at 64x64 by default.
    """
    def __init__(self, channels: int, res: int, H):
        super().__init__()
        self.channels = int(channels)
        self.res = int(res)

        num_layers = int(getattr(H, 'attn_num_layers', 1))
        num_heads = int(getattr(H, 'attn_num_heads', 4))
        widening = int(getattr(H, 'attn_widening_factor', 1))
        dropout = float(getattr(H, 'attn_dropout', 0.0))
        res_drop = float(getattr(H, 'attn_residual_dropout', 0.0))

        num_heads = min(num_heads, self.channels)
        while num_heads > 1 and (self.channels % num_heads != 0):
            num_heads -= 1

        gn_groups = int(getattr(H, 'attn_gn_groups', 32))
        gn_groups = max(1, min(gn_groups, self.channels))
        while gn_groups > 1 and (self.channels % gn_groups != 0):
            gn_groups -= 1

        self.pre_norm = nn.GroupNorm(gn_groups, self.channels, eps=1e-6, affine=True)
        self.attn = SelfAttentionBlock(
            num_layers=num_layers,
            num_heads=num_heads,
            num_channels=self.channels,
            num_qk_channels=self.channels,
            num_v_channels=self.channels,
            widening_factor=widening,
            dropout=dropout,
            residual_dropout=res_drop,
            activation_checkpointing=bool(getattr(H, 'attn_activation_checkpointing', False)),
            activation_offloading=bool(getattr(H, 'attn_activation_offloading', False)),
        )

        n_bands = int(getattr(H, 'attn_pos_num_bands', 6))
        self.pos_enc = FourierPositionEncoding((self.res, self.res), num_frequency_bands=n_bands)
        pos_dim = self.pos_enc.num_position_encoding_channels(include_positions=True)
        self.pos_proj = nn.Linear(pos_dim, self.channels, bias=True)
        nn.init.normal_(self.pos_proj.weight, std=1e-3)
        nn.init.zeros_(self.pos_proj.bias)
        self.pos_gate = nn.Parameter(torch.zeros(()))
        self.attn_gate = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.channels or h != self.res or w != self.res:
            raise ValueError(
                f"SpatialSelfAttention expected [B,{self.channels},{self.res},{self.res}], got {tuple(x.shape)}"
            )

        xn = self.pre_norm(x)
        tok = xn.flatten(2).transpose(1, 2)
        pos = self.pos_enc(b).to(device=tok.device, dtype=tok.dtype)
        pos = self.pos_proj(pos)
        tok = tok + self.pos_gate * pos
        tok_out = self.attn(tok).last_hidden_state
        out = tok_out.transpose(1, 2).reshape(b, c, h, w)

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

class Upsample(nn.Module):
    def __init__(self, zdim: int, src_res: int, dst_res: int):
        super().__init__()
        if src_res == dst_res:
            self.net = nn.Identity()
        else:
            ratio = dst_res // src_res
            layers = []
            cur = src_res
            while cur < dst_res:
                step = min(2, dst_res // cur)
                if step == 2:
                    layers += [
                        nn.Conv2d(zdim, 4 * zdim, kernel_size=1),
                        nn.GELU(),
                        nn.PixelShuffle(2),
                        nn.Conv2d(zdim, zdim, kernel_size=3, padding=1),
                        nn.GELU(),
                    ]
                    cur *= 2
                else:
                    raise ValueError(f"Unsupported ratio path {src_res}->{dst_res}")
            self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class DecBlock(nn.Module):
    def __init__(self, H, res, mixin, n_blocks, is_top=False):
        super().__init__()
        self.base = int(res)
        self.mixin = mixin
        self.H = H
        self.is_top = bool(is_top)

        self.widths = get_width_settings(H.width, H.custom_width_str)
        width = self.widths[self.base]
        use_3x3 = self.base > 2
        cond_width = int(width * H.bottleneck_multiple)
        self.zdim = H.zdim
        self.use_checkpoint = getattr(H, "use_checkpoint", False)

        self.top_h_dim = int(getattr(H, "top_h_context_dim", 0))
        self.use_top_h = self.is_top and (self.top_h_dim > 0)

        if self.use_top_h:
            # Raw-h concatenation only. No h projection.
            self.enc = Block(
                width * 2 + self.top_h_dim+H.zdim,  # concat h and acts and top_h, plus skip connection from pre-z features
                cond_width,
                H.zdim * 2,
                residual=False,
                use_3x3=use_3x3,
            )
            self.prior = Block(
                width + self.top_h_dim + H.zdim,
                cond_width,
                H.zdim * 2 + width,
                residual=False,
                use_3x3=use_3x3,
                zero_last=True,
            )

            self.resnet = Block(
                width + self.top_h_dim,
                cond_width,
                width,
                residual=False,
                use_3x3=use_3x3,
            )
        else:
            self.enc = Block(
                width * 2,
                cond_width,
                H.zdim * 2,
                residual=False,
                use_3x3=use_3x3,
            )
            #P(Z_t^{level} | Z_t^{level above}, Z_{t-1}^{same level}, x_t)
            self.prior = Block(
                width + 2 * H.zdim, #prior at each level coditioned on the previous latent at the same level and same time latent at one level above 
                cond_width,
                H.zdim * 2 + width,
                residual=False,
                use_3x3=use_3x3,
                zero_last=True,
            )

            self.resnet = Block(
                width,
                cond_width,
                width,
                residual=True,
                use_3x3=use_3x3,
            )

        self.z_proj = get_1x1(H.zdim, width)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)

        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.z_fn = lambda z: self.z_proj(z)


    def _maybe_ckpt(self, fn, *args):
        if self.training and self.use_checkpoint:
            return ckpt(fn, *args, use_reentrant=False)
        return fn(*args)


    def sample(self, x, acts, cond_top=None, latent_prev=None, up_latent_cur=None):

        if self.use_top_h:
            enc_in = torch.cat([x, acts, cond_top], dim=1)
            qm, qv = self._maybe_ckpt(self.enc, enc_in).chunk(2, dim=1)

            prior_in = torch.cat([x, cond_top], dim=1)
            feats = self._maybe_ckpt(self.prior, prior_in)
        else:
            qm, qv = self._maybe_ckpt(
                self.enc, torch.cat([x, acts], dim=1)
            ).chunk(2, dim=1)
            prior_in = torch.cat([latent_prev, up_latent_cur, x], dim=1) 
            feats = self._maybe_ckpt(self.prior, prior_in)

        pm = feats[:, :self.zdim, ...]
        pv = feats[:, self.zdim : 2 * self.zdim, ...]
        xpp = feats[:, 2 * self.zdim :, ...]
        x = x + xpp

        z = draw_gaussian_diag_samples(qm, qv)

        if self.is_top:
            kl = torch.zeros_like(qm)
        else:
            kl = gaussian_analytical_kl(qm, pm, qv, pv)

        return z, x, kl, qm, qv

    def sample_uncond(self, x, cond_top =None, latent_prev=None, up_latent_cur=None, t=None, lvs=None):
        

        if self.use_top_h:
            prior_in = torch.cat([x, cond_top], dim=1)
            feats = self._maybe_ckpt(self.prior, prior_in)
        else:
            prior_in = torch.cat([latent_prev, up_latent_cur, x], dim=1) 
            feats = self._maybe_ckpt(self.prior, prior_in)

        pm = feats[:, :self.zdim, ...]
        pv = feats[:, self.zdim : 2 * self.zdim, ...]
        xpp = feats[:, 2 * self.zdim :, ...]
        x = x + xpp

        if lvs is not None:
            z = lvs
        else:
            if t is None:
                z = draw_gaussian_diag_samples(pm, pv)
            else:
                t_val = float(t)
                if t_val <= 0.0:
                    z = pm
                else:
                    pv = pv + torch.ones_like(pv) * np.log(t_val)
                    z = draw_gaussian_diag_samples(pm, pv)

        return z, x

    def get_inputs(self, xs, activations):
        acts = activations[self.base]
        try:
            x = xs[self.base]
        except KeyError:
            x = torch.zeros_like(acts)
        if acts.shape[0] != x.shape[0]:
            x = x.repeat(acts.shape[0], 1, 1, 1)
        return x, acts

    def _apply_post_z(self, x, z, h_decoder_top=None):
        x_in = x                      # save pre-latent features
        x = x + self.z_fn(z)          # latent injection always uses x_in

        if self.use_top_h:
            x = self._maybe_ckpt(self.resnet, torch.cat([x, h_decoder_top], dim=1))
        else:
            x = self._maybe_ckpt(self.resnet, x)

        return x
        
    def forward(
        self,
        xs,
        activations,
        get_latents=False,
        cond_top=None,
        latent_prev=None,
        up_latent_cur=None,
        h_decoder_top=None,
    ):
        x, acts = self.get_inputs(xs, activations)

        if self.mixin is not None:
            x = x + F.interpolate(
                xs[self.mixin][:, : x.shape[1], ...],
                scale_factor=self.base // self.mixin,
            )

        z, x, kl, qm, qv = self.sample(x, acts, cond_top=cond_top, latent_prev=latent_prev, up_latent_cur=up_latent_cur)
        x = self._apply_post_z(x, z, h_decoder_top=h_decoder_top)
        xs[self.base] = x

        out = {
            "kl": kl,
            "posterior_mean": qm,
            "posterior_logvar": qv,
        }
        if get_latents:
            out["z"] = z.detach()
        return xs, out, z

    def forward_uncond(
        self,
        xs,
        t=None,
        lvs=None,
        cond_top=None,
        latent_prev=None,
        up_latent_cur=None,
        h_decoder_top=None,
    ):
        try:
            x = xs[self.base]
        except KeyError:
            ref = xs[list(xs.keys())[0]]
            x = torch.zeros(
                dtype=ref.dtype,
                size=(ref.shape[0], self.widths[self.base], self.base, self.base),
                device=ref.device,
            )

        if self.mixin is not None:
            x = x + F.interpolate(
                xs[self.mixin][:, : x.shape[1], ...],
                scale_factor=self.base // self.mixin,
            )

        z, x = self.sample_uncond(x, t=t, lvs=lvs, cond_top=cond_top,latent_prev=latent_prev, up_latent_cur=up_latent_cur)
        x = self._apply_post_z(x, z, h_decoder_top=h_decoder_top)
        xs[self.base] = x
        return xs, z


class Decoder(HModule):
    def build(self):
        H = self.H
        resos = set()
        dec_blocks = []
        self.widths = get_width_settings(H.width, H.custom_width_str)
        blocks = parse_layer_string(H.dec_blocks)
        n_blocks = len(blocks)

        # one-block-above latent routing
        self.block_up_keys = []
        self.upsamplers = nn.ModuleDict()

        prev_res = None
        for idx, (res, mixin) in enumerate(blocks):
            dec_blocks.append(
                DecBlock(
                    H,
                    res,
                    mixin,
                    n_blocks=n_blocks,
                    is_top=(idx == 0),
                )
            )
            resos.add(res)

            if idx == 0:
                self.block_up_keys.append(None)
            else:
                key = f"{prev_res}->{res}"
                self.block_up_keys.append(key)

                if key not in self.upsamplers:
                    self.upsamplers[key] = Upsample(H.zdim, prev_res, res)

            prev_res = res

        self.resolutions = sorted(resos)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.bias_xs = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, self.widths[res], res, res))
                for res in self.resolutions
                if res <= H.no_bias_above
            ]
        )
        self.out_net = DmolNet(H)

        out_width = self.widths[int(H.image_size)]
        self.gain = nn.Parameter(torch.ones(1, out_width, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, out_width, 1, 1))
        self.final_fn = lambda x: x * self.gain + self.bias

    def forward(
        self,
        activations,
        get_latents=False,
        cond_top=None,
        h_decoder_top=None,
        prev_latents=None,
    ):
        stats = []
        xs = {a.shape[2]: a for a in self.bias_xs}
        current_latents = []

        for idx, block in enumerate(self.dec_blocks):
            blk_cond = cond_top if block.is_top else None
            blk_h = h_decoder_top if block.is_top else None

            latent_prev = prev_latents[idx]
            if idx ==0:
                up_latent_cur = None
            else:
                z = current_latents[idx-1]
                up_key = self.block_up_keys[idx]
                up_latent_cur = self.upsamplers[up_key](z)

            xs, block_stats, z_cur = block(
                xs,
                activations,
                get_latents=get_latents,
                cond_top=blk_cond,
                latent_prev=latent_prev,
                up_latent_cur=up_latent_cur,
                h_decoder_top=blk_h,
            )
            stats.append(block_stats)
            current_latents.append(z_cur)

        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size], stats, current_latents

    def forward_uncond(
        self,
        n,
        t=None,
        y=None,
        cond_top=None,
        h_decoder_top=None,
        prev_latents=None,
    ):
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)

        current_latents = []

        for idx, block in enumerate(self.dec_blocks):
            try:
                temp = t[idx]
            except TypeError:
                temp = t
            
            blk_cond = cond_top if block.is_top else None
            blk_h = h_decoder_top if block.is_top else None
            latent_prev = prev_latents[idx] 

            if idx == 0:
                up_latent_cur = None
            else:
                z = current_latents[idx-1]
                up_key = self.block_up_keys[idx]
                up_latent_cur = self.upsamplers[up_key](z)
            xs, z_cur = block.forward_uncond(
                xs,
                t=temp,
                cond_top=blk_cond,
                latent_prev=latent_prev,
                up_latent_cur=up_latent_cur,
                h_decoder_top=blk_h,
            )
            current_latents.append(z_cur)

        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size], current_latents

    def forward_manual_latents(
        self,
        n,
        latents,
        t=None,
        cond_top=None,
        h_decoder_top=None,
        prev_latents=None,
    ):
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)

        current_latents = []
        for idx, (block, lvs) in enumerate(itertools.zip_longest(self.dec_blocks, latents)):
            blk_cond = cond_top if block.is_top else None
            blk_h = h_decoder_top if block.is_top else None
            latent_prev = prev_latents[idx] 

            if idx == 0:
                up_latent_cur = None
            else:
                z = current_latents[idx-1]
                up_key = self.block_up_keys[idx]
                up_latent_cur = self.upsamplers[up_key](z)
            xs, z_cur = block.forward_uncond(
                xs,
                t,
                lvs=lvs,
                cond_top=blk_cond,
                h_decoder_top=blk_h,
                latent_prev=latent_prev,
                up_latent_cur=up_latent_cur,
            )
            current_latents.append(z_cur)

        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size], current_latents
    
    def forward_from_top_latent(
        self,
        z_top_map,
        t=None,
        cond_top=None,
        h_decoder_top=None,
        prev_latents=None,
    ):
        n = z_top_map.shape[0]
        latents = [z_top_map] + [None] * (len(self.dec_blocks) - 1)
        return self.forward_manual_latents(
            n,
            latents,
            t=t,
            cond_top=cond_top,
            h_decoder_top=h_decoder_top,
            prev_latents=prev_latents,
        )


class VDVAE(HModule):
    def __init__(
        self,
        H,
        prior=None,
        top_kl_weight: float = 1.0,
        prior_kl_mc_samples: int = 10,
    ):
        self.prior = prior
        self.top_kl_weight = float(top_kl_weight)
        self.prior_kl_mc_samples = int(prior_kl_mc_samples)
        super().__init__(H)

    def build(self):
        self.encoder = Encoder(self.H)
        self.decoder = Decoder(self.H)

        # IMPORTANT: This is the channel count of h_t only, NOT concat(h_t, z_tilde_t)
        self.top_h_dim = int(getattr(self.H, "top_h_context_dim", 0))

        if self.prior is not None and self.top_h_dim > 0:
            expected_prior_dim = self.top_h_dim + int(self.H.zdim)
            if int(self.prior.hidden_dim) != expected_prior_dim:
                raise ValueError(
                    f"prior.hidden_dim must be h_t_dim + zdim = "
                    f"{self.top_h_dim} + {int(self.H.zdim)} = {expected_prior_dim}, "
                    f"but got {int(self.prior.hidden_dim)}"
                )

    def _check_top_inputs(self, cond_top, h_decoder_top):
        if self.top_h_dim <= 0:
            return None, None

        if cond_top is None:
            raise ValueError("cond_top must be provided for top prior/posterior")
        if h_decoder_top is None:
            raise ValueError("h_decoder_top must be provided for top decoder")

        top_res = int(self.decoder.dec_blocks[0].base)
        expected_cond_ch = self.top_h_dim + int(self.H.zdim)

        if cond_top.dim() != 4:
            raise ValueError(f"cond_top must be [B,C,H,W], got {tuple(cond_top.shape)}")
        if h_decoder_top.dim() != 4:
            raise ValueError(
                f"h_decoder_top must be [B,C,H,W], got {tuple(h_decoder_top.shape)}"
            )

        if cond_top.shape[1] != expected_cond_ch:
            raise ValueError(
                f"cond_top channels must be top_h_dim + zdim = {expected_cond_ch}, "
                f"got {cond_top.shape[1]}"
            )
        if h_decoder_top.shape[1] != self.top_h_dim:
            raise ValueError(
                f"h_decoder_top channels must be top_h_dim = {self.top_h_dim}, "
                f"got {h_decoder_top.shape[1]}"
            )

        if cond_top.shape[2:] != (top_res, top_res):
            raise ValueError(
                f"cond_top spatial size must be {(top_res, top_res)}, "
                f"got {tuple(cond_top.shape[2:])}"
            )
        if h_decoder_top.shape[2:] != (top_res, top_res):
            raise ValueError(
                f"h_decoder_top spatial size must be {(top_res, top_res)}, "
                f"got {tuple(h_decoder_top.shape[2:])}"
            )

        if cond_top.shape[0] != h_decoder_top.shape[0]:
            raise ValueError(
                f"Batch mismatch: cond_top batch {cond_top.shape[0]} vs "
                f"h_decoder_top batch {h_decoder_top.shape[0]}"
            )

        return cond_top.contiguous(), h_decoder_top.contiguous()

    def _compute_top_prior_terms(self, top_q_mean_map, top_q_logvar_map, cond_top):
        if self.prior is None or top_q_mean_map is None or top_q_logvar_map is None:
            return None, None
        if cond_top is None:
            raise ValueError("cond_top must be provided when using the DPGMMPrior")

        b, c, ht, wt = top_q_mean_map.shape
        top_q_mean_img = top_q_mean_map.contiguous().view(b, c * ht * wt)

        # keep your current semantics:
        # qv is treated downstream as log-sigma, so convert to log-variance
        top_q_logvar_img = 2 * top_q_logvar_map.contiguous().view(b, c * ht * wt)

        _, prior_params = self.prior(cond_top)
        dp_kl_img = self.prior.compute_kl_divergence_mc(
            posterior_mean=top_q_mean_img,
            posterior_logvar=top_q_logvar_img,
            prior_params=prior_params,
            n_samples=self.prior_kl_mc_samples,
            reduction="image",
        )
        return dp_kl_img, prior_params

    def forward(
        self,
        x,
        x_target,
        cond_top=None,
        h_decoder_top=None,
        mask_t=None,
        prev_latents=None,
        get_latents=False,
    ):
        activations = self.encoder.forward(x)
        cond_top, h_decoder_top = self._check_top_inputs(cond_top, h_decoder_top)

        px_z, stats, current_latents = self.decoder.forward(
            activations,
            get_latents=get_latents,
            cond_top=cond_top,
            h_decoder_top=h_decoder_top,
            prev_latents=prev_latents,
        )

        distortion_per_pixel = self.decoder.out_net.nll(px_z, x_target)
        ndims = float(np.prod(x_target.shape[1:]))

        rate_gauss = torch.zeros_like(distortion_per_pixel)
        top_q_mean_map = None
        top_q_logvar_map = None

        for block, st in zip(self.decoder.dec_blocks, stats):
            kl_block = st["kl"].sum(dim=(1, 2, 3))
            if block.is_top:
                top_q_mean_map = st.get("posterior_mean", None)
                top_q_logvar_map = st.get("posterior_logvar", None)
            else:
                rate_gauss = rate_gauss + kl_block

        rate_gauss = rate_gauss / ndims
        dp_rate = torch.zeros_like(distortion_per_pixel)
        dp_kl_img = None
        prior_params = None

        if self.prior is not None:
            dp_kl_img, prior_params = self._compute_top_prior_terms(
                top_q_mean_map=top_q_mean_map,
                top_q_logvar_map=top_q_logvar_map,
                cond_top=cond_top,
            )
            if dp_kl_img is not None:
                dp_rate = self.top_kl_weight * dp_kl_img / ndims

        total_rate = rate_gauss + dp_rate
        elbo_per_sample = distortion_per_pixel + total_rate

        if mask_t is None:
            reduce = torch.mean
            valid_count = torch.tensor(
                float(elbo_per_sample.shape[0]),
                device=elbo_per_sample.device,
            )
        else:
            vm = mask_t.float().view(-1).to(
                device=elbo_per_sample.device,
                dtype=elbo_per_sample.dtype,
            )
            den = vm.sum().clamp(min=1.0)
            reduce = lambda z: (z * vm).sum() / den
            valid_count = den

        out = {
            "elbo": reduce(elbo_per_sample),
            "distortion": reduce(distortion_per_pixel),
            "gauss_rate": reduce(rate_gauss),
            "rate": reduce(total_rate),
            "dp_rate": reduce(dp_rate),
            "px_z": px_z,
            "top_q_mean_map": top_q_mean_map,
            "top_q_logvar_map": top_q_logvar_map,
            "prior_params": prior_params,
            "valid_count": valid_count,
            "current_latents": current_latents,
        }
        if dp_kl_img is not None:
            out["dp_kl"] = reduce(dp_kl_img)
        if get_latents:
            out["stats"] = stats
        return out

    def decode_from_top_latent(
        self,
        z_top_map,
        cond_top=None,
        h_decoder_top=None,
        t=None,
        prev_latents=None,
    ):  
        """
        return pz and current latents
        """
        cond_top, h_decoder_top = self._check_top_inputs(cond_top, h_decoder_top)
        return self.decoder.forward_from_top_latent(
            z_top_map,
            t=t,
            cond_top=cond_top,
            h_decoder_top=h_decoder_top,
            prev_latents=prev_latents,
        )

    def sample(
        self,
        n_batch,
        cond_top,
        h_decoder_top,
        prev_latents=None,
    ):
        if self.prior is None:
            raise ValueError("VDVAE.sample requires a DPGMMPrior")

        cond_top, h_decoder_top = self._check_top_inputs(cond_top, h_decoder_top)

        if cond_top.shape[0] != n_batch:
            raise ValueError(
                f"cond_top batch {cond_top.shape[0]} does not match n_batch={n_batch}"
            )

        top_block = self.decoder.dec_blocks[0]
        c = top_block.zdim
        res = top_block.base

        prior_dist, _ = self.prior(cond_top)
        z_img = prior_dist.sample()

        expected_dim = c * res * res
        if z_img.shape != (n_batch, expected_dim):
            raise ValueError(
                f"Sampled top latent has shape {tuple(z_img.shape)}, "
                f"expected {(n_batch, expected_dim)}"
            )

        z_top_map = z_img.view(n_batch, c, res, res).contiguous()
        px_z, current_latents = self.decoder.forward_from_top_latent(
            z_top_map,
            cond_top=cond_top,
            h_decoder_top=h_decoder_top,
            prev_latents=prev_latents,
        )
        return self.decoder.out_net.sample(px_z), current_latents

    def forward_get_latents(
        self,
        x,
        cond_top=None,
        h_decoder_top=None,
        prev_latents=None,
    ):
        activations = self.encoder.forward(x)
        cond_top, h_decoder_top = self._check_top_inputs(cond_top, h_decoder_top)
        _, stats, current_latents = self.decoder.forward(
            activations,
            get_latents=True,
            cond_top=cond_top,
            h_decoder_top=h_decoder_top,
            prev_latents=prev_latents,
        )
        return stats, current_latents

    def forward_uncond_samples(
        self,
        n_batch,
        t=None,
        cond_top=None,
        h_decoder_top=None,
        prev_latents=None,
    ):
        cond_top, h_decoder_top = self._check_top_inputs(cond_top, h_decoder_top)
        px_z, current_latents = self.decoder.forward_uncond(
            n_batch,
            t=t,
            cond_top=cond_top,
            h_decoder_top=h_decoder_top,
            prev_latents=prev_latents,
        )
        return self.decoder.out_net.sample(px_z), current_latents

    def forward_samples_set_latents(
        self,
        n_batch,
        latents,
        t=None,
        cond_top=None,
        h_decoder_top=None,
        prev_latents=None,
    ):
        cond_top, h_decoder_top = self._check_top_inputs(cond_top, h_decoder_top)
        px_z, current_latents = self.decoder.forward_manual_latents(
            n_batch,
            latents,
            t=t,
            cond_top=cond_top,
            h_decoder_top=h_decoder_top,
            prev_latents=prev_latents,
        )
        return self.decoder.out_net.sample(px_z), current_latents
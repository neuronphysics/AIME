import itertools
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as ckpt
from vdvae.vae_helpers import HModule, DmolNet, draw_gaussian_diag_samples, gaussian_analytical_kl, get_1x1, get_3x3, Block, TopSlotPosterior, mean_from_discretized_mix_logistic
from VRNN.perceiver.modules import SelfAttentionBlock
from VRNN.perceiver.position import FourierPositionEncoding
from vdvae.top_dpgmm_prior import compute_top_kl_conditional_frozen, sample_top_conditional_frozen, ConditionalTopDPGMM, compute_slot_kl_conditional_frozen, sample_slots_conditional_frozen, sample_slot_vectors_conditional_frozen




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
    def __init__(self, zdim: int, src_res: int, dst_res: int, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
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
        if self.training and self.use_checkpoint and z.requires_grad:
            return ckpt(self.net, z, use_reentrant=False)
        return self.net(z)

class DecBlock(nn.Module):
    def __init__(self, H, res, mixin, n_blocks, is_top=False, is_last =False):
        super().__init__()
        self.base = int(res)
        self.mixin = mixin
        self.H = H
        self.is_top = bool(is_top)
        self.is_last = bool(is_last)

        self.widths = get_width_settings(H.width, H.custom_width_str)
        width = self.widths[self.base]
        out_width = width + 1 if (self.is_last and self.base == int(H.image_size)) else width
        use_3x3 = self.base > 2
        cond_width = int(width * H.bottleneck_multiple)
        self.zdim = H.zdim
        self.use_checkpoint = getattr(H, "use_checkpoint", False)

        self.top_h_dim = int(getattr(H, "top_h_context_dim", 0))
        self.use_top_h = self.is_top and (self.top_h_dim > 0)

        if self.use_top_h:

            self.top_cond_channels = self.top_h_dim

            self.top_slot_posterior = TopSlotPosterior(
                in_channels=width,          # encoder activation channels at top resolution
                z_channels=H.zdim,
                top_h=self.base,
                top_w=self.base,
                cond_channels=self.top_cond_channels,
                cond_width=cond_width,
                num_slots=int(getattr(H, "top_num_slots", 6)),
                slot_dim=int(getattr(H, "top_slot_dim", H.zdim * 4)),
                n_iters=int(getattr(H, "top_slot_iters", 3)),
            )

            self.resnet = Block(
                width + self.top_h_dim,
                cond_width,
                out_width,
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
                out_width,
                residual=(out_width == width),
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


    def sample(self, x, acts, h_top=None, latent_prev=None, up_latent_cur=None):
        extra = {}

        if self.is_top:

            slot_out = self._maybe_ckpt(self.top_slot_posterior, acts, h_top)


            qm = slot_out["top_q_mean_map"]
            qv = slot_out["top_q_logsigma_map"]   # logsigma, not logvar
            z = slot_out["z_top_map"]
            z_slot_maps = slot_out["z_slot_maps"]
            extra = {
                    "slot_mu": slot_out["slot_mu"],
                    "slot_logsigma": slot_out["slot_logsigma"],
                    "slot_attn": slot_out["slot_attn"],
                    "slot_comp_maps": slot_out["slot_comp_maps"],
                    "slot_comp_logsigma": slot_out["slot_comp_logsigma"],

                    # Keep both names clear:
                    "z_top_map": z,       # [B, C, H, W], aggregate
                    "z_slot_maps": z_slot_maps,   # [B, S, C, H, W], decoder input
            }
            kl = torch.zeros_like(qm)

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

            kl = gaussian_analytical_kl(qm, pm, qv, pv)

        return z, x, kl, qm, qv, extra

    def sample_uncond(self, x, latent_prev=None, up_latent_cur=None, t=None, lvs=None):
        if self.is_top:
            z = lvs
            return z, x

        # For non-top block case
        prior_in = torch.cat([latent_prev, up_latent_cur, x], dim=1)
        feats = self._maybe_ckpt(self.prior, prior_in)

        pm = feats[:, :self.zdim, ...]
        pv = feats[:, self.zdim:2 * self.zdim, ...]
        xpp = feats[:, 2 * self.zdim:, ...]
        x = x + xpp

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

        if self.base in xs:
            x = xs[self.base]
        else:
            # Use batch 1 here, then expand below to the correct target batch.
            x = torch.zeros(1, self.widths[self.base], self.base, self.base, device=acts.device, dtype=acts.dtype)

        # Determine the effective decoder batch.
        # Top block: target_b = B.
        # Lower slot-wise blocks: target_b = B*S, usually inherited from xs[self.mixin].
        target_b = max(x.shape[0], acts.shape[0])

        if self.mixin is not None and self.mixin in xs:
            target_b = max(target_b, xs[self.mixin].shape[0])

        if x.shape[0] != target_b:
            if x.shape[0] == 1:
                x = x.expand(target_b, *x.shape[1:]).contiguous()
            elif target_b % x.shape[0] == 0:
                rep = target_b // x.shape[0]
                x = x[:, None].expand(
                    x.shape[0], rep, *x.shape[1:]
                ).reshape(target_b, *x.shape[1:]).contiguous()
            else:
                raise ValueError(
                    f"Cannot expand decoder x batch {x.shape[0]} to target batch {target_b} "
                    f"at resolution {self.base}."
                )

        if acts.shape[0] != target_b:
            if acts.shape[0] == 1:
                acts = acts.expand(target_b, *acts.shape[1:]).contiguous()
            elif target_b % acts.shape[0] == 0:
                rep = target_b // acts.shape[0]
                acts = acts[:, None].expand(
                    acts.shape[0], rep, *acts.shape[1:]
                ).reshape(target_b, *acts.shape[1:]).contiguous()
            else:
                raise ValueError(
                    f"Cannot expand encoder acts batch {acts.shape[0]} to target batch {target_b} "
                    f"at resolution {self.base}."
                )

        return x, acts

    def _apply_post_z(self, x, z, h_decoder_top=None):
        x_in = x                      # save pre-latent features
        x = x + self.z_fn(z)          # latent injection always uses x_in

        if self.use_top_h:
            x = self._maybe_ckpt(self.resnet, torch.cat([x, h_decoder_top], dim=1))
        else:
            x = self._maybe_ckpt(self.resnet, x)

        return x

    def forward(self, xs, activations, get_latents=False, latent_prev=None, up_latent_cur=None, h_decoder_top=None):
        x, acts = self.get_inputs(xs, activations)

        if self.mixin is not None:
            mix = F.interpolate(
                xs[self.mixin][:, : x.shape[1], ...],
                scale_factor=self.base // self.mixin,
            )

            if mix.shape[0] != x.shape[0]:
                target_b = x.shape[0]
                if mix.shape[0] == 1:
                    mix = mix.expand(target_b, *mix.shape[1:]).contiguous()
                elif target_b % mix.shape[0] == 0:
                    rep = target_b // mix.shape[0]
                    mix = mix[:, None].expand(
                        mix.shape[0], rep, *mix.shape[1:]
                    ).reshape(target_b, *mix.shape[1:]).contiguous()
                else:
                    raise ValueError(
                        f"Cannot expand mixin batch {mix.shape[0]} to x batch {target_b} "
                        f"at resolution {self.base}."
                    )

            x = x + mix

        z, x, kl, qm, qv, extra = self.sample(
            x,
            acts,
            h_top=h_decoder_top,
            latent_prev=latent_prev,
            up_latent_cur=up_latent_cur,
        )

        if self.is_top and "z_slot_maps" in extra:
            # z_state is the aggregate top latent for RNN/state.
            z_state = extra["z_top_map"]          # [B, C, H, W]

            # z_slot_maps are the actual per-slot decoder inputs.
            z_slot_maps = extra["z_slot_maps"]    # [B, S, C, H, W]
            B, S, C, Ht, Wt = z_slot_maps.shape

            z_for_decoder = z_slot_maps.reshape(B * S, C, Ht, Wt)

            x = x[:, None].expand(
                B, S, *x.shape[1:]
            ).reshape(B * S, *x.shape[1:]).contiguous()

            if h_decoder_top is not None:
                h_decoder_top = h_decoder_top[:, None].expand(
                    B, S, *h_decoder_top.shape[1:]
                ).reshape(B * S, *h_decoder_top.shape[1:]).contiguous()

            x = self._apply_post_z(x, z_for_decoder, h_decoder_top=h_decoder_top)
            xs[self.base] = x

            # Lower decoder blocks need B*S latent.
            z_cur = z_for_decoder

        else:
            z_state = z
            x = self._apply_post_z(x, z, h_decoder_top=h_decoder_top)
            xs[self.base] = x
            z_cur = z

        out = {
            "kl": kl,
            "posterior_mean": qm,
            "posterior_logvar": qv,
            **extra,
        }

        # Do not expose decoder z as the state latent.
        if get_latents:
            out["z_decoder"] = z_cur.detach()

        if self.is_top:
            out["z_top_state"] = z_state

        return xs, out, z_cur

    def forward_uncond(self, xs, t=None, lvs=None, latent_prev=None, up_latent_cur=None, h_decoder_top=None):
        # The live decoder batch is set by what this block consumes.
        if self.mixin is not None and self.mixin in xs:
            b = xs[self.mixin].shape[0]
        elif up_latent_cur is not None:
            b = up_latent_cur.shape[0]
        elif lvs is not None:
            b = lvs.shape[0]  # top block: image batch B, even if lvs is [B,S,C,H,W]
        else:
            b = next(iter(xs.values())).shape[0]

        # Get x or create it directly at the live batch.
        if self.base in xs:
            x = xs[self.base]

            if x.shape[0] != b:
                if b % x.shape[0] != 0:
                    raise ValueError(
                        f"res {self.base}: cannot promote x batch {x.shape[0]} to {b}."
                    )
                x = x.repeat_interleave(b // x.shape[0], dim=0)

        else:
            if self.mixin is not None and self.mixin in xs:
                ref = xs[self.mixin]
            elif up_latent_cur is not None:
                ref = up_latent_cur
            elif lvs is not None:
                ref = lvs
            else:
                ref = next(iter(xs.values()))

            x = torch.zeros(b, self.widths[self.base], self.base, self.base, dtype=ref.dtype, device=ref.device)

        if self.mixin is not None:
            mix = F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)

            if mix.shape[0] != b:
                raise ValueError(
                    f"res {self.base}: mixin batch {mix.shape[0]} != expected batch {b}."
                )

            x = x + mix

        if not self.is_top:
            if up_latent_cur is None or latent_prev is None:
                raise ValueError(
                    f"res {self.base}: up_latent_cur/latent_prev required at non-top block."
                )

            if up_latent_cur.shape[0] != b:
                raise ValueError(
                    f"res {self.base}: up_latent_cur batch {up_latent_cur.shape[0]} != expected {b}."
                )

            if latent_prev.shape[0] != b:
                if b % latent_prev.shape[0] != 0:
                    raise ValueError(
                        f"res {self.base}: cannot promote latent_prev batch "
                        f"{latent_prev.shape[0]} to {b}."
                    )
                latent_prev = latent_prev.repeat_interleave(b // latent_prev.shape[0], dim=0)

            if x.shape[0] != b:
                raise ValueError(
                    f"res {self.base}: x batch {x.shape[0]} != expected {b}."
                )

        z, x = self.sample_uncond( x, t=t, lvs=lvs, latent_prev=latent_prev, up_latent_cur=up_latent_cur)

        if self.is_top and lvs is not None and lvs.dim() == 5:
            B, S, C, Ht, Wt = lvs.shape

            if x.shape[0] != B:
                raise ValueError(
                    f"top block must start at image batch B={B}, got {x.shape[0]}."
                )

            z = lvs.reshape(B * S, C, Ht, Wt)

            x = (
                x[:, None]
                .expand(B, S, *x.shape[1:])
                .reshape(B * S, *x.shape[1:])
                .contiguous()
            )

            if h_decoder_top is not None:
                h_decoder_top = (
                    h_decoder_top[:, None]
                    .expand(B, S, *h_decoder_top.shape[1:])
                    .reshape(B * S, *h_decoder_top.shape[1:])
                    .contiguous()
                )

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
                    is_last=(idx == len(blocks) - 1),
                )
            )
            resos.add(res)

            if idx == 0:
                self.block_up_keys.append(None)
            else:
                key = f"{prev_res}->{res}"
                self.block_up_keys.append(key)

                if key not in self.upsamplers:
                    self.upsamplers[key] = Upsample(H.zdim, prev_res, res, use_checkpoint=H.use_checkpoint)

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

        out_width = self.widths[int(H.image_size)] + 1
        self.gain = nn.Parameter(torch.ones(1, out_width, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, out_width, 1, 1))
        self.final_fn = lambda x: x * self.gain + self.bias

    def forward( self, activations, get_latents=False, h_decoder_top=None, prev_latents=None):
        stats = []
        xs = {a.shape[2]: a for a in self.bias_xs}
        current_latents = [] #BxS which is slot-wise
        decode_latents = [] #B which is used for RNN as image-level states

        if prev_latents is None:
            prev_latents = [None] * len(self.dec_blocks)

        for idx, block in enumerate(self.dec_blocks):
            blk_h = h_decoder_top if block.is_top else None

            latent_prev = prev_latents[idx]

            if idx == 0:
                up_latent_cur = None
            else:
                z = current_latents[idx - 1]
                up_key = self.block_up_keys[idx]
                up_latent_cur = self.upsamplers[up_key](z)

                if latent_prev is not None and latent_prev.shape[0] != up_latent_cur.shape[0]:
                    target_b = up_latent_cur.shape[0]

                    if latent_prev.shape[0] == 1:
                        latent_prev = latent_prev.expand(
                            target_b, *latent_prev.shape[1:]
                        ).contiguous()

                    elif target_b % latent_prev.shape[0] == 0:
                        rep = target_b // latent_prev.shape[0]
                        latent_prev = latent_prev[:, None].expand(
                            latent_prev.shape[0], rep, *latent_prev.shape[1:]
                        ).reshape(target_b, *latent_prev.shape[1:]).contiguous()

                    else:
                        raise ValueError(
                            f"Cannot expand latent_prev batch {latent_prev.shape[0]} "
                            f"to decoder batch {target_b} at block {idx}."
                        )

            xs, block_stats, z_cur = block(
                xs,
                activations,
                get_latents=get_latents,
                latent_prev=latent_prev,
                up_latent_cur=up_latent_cur,
                h_decoder_top=blk_h,
            )

            stats.append(block_stats)
            current_latents.append(z_cur)
            if block.is_top:
                if "z_top_state" not in block_stats:
                    raise KeyError("Top block did not return z_top_state.")
                decode_latents.append(block_stats["z_top_state"])
            else:
                decode_latents.append(z_cur)

        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size], stats, decode_latents

    def forward_uncond(
        self,
        n,
        t=None,
        h_decoder_top=None,
        prev_latents=None,
        z_top=None,
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

            blk_h = h_decoder_top if block.is_top else None
            latent_prev = prev_latents[idx]

            if idx == 0:
                up_latent_cur = None
                lvs = z_top
            else:
                z = current_latents[idx-1]
                up_key = self.block_up_keys[idx]
                up_latent_cur = self.upsamplers[up_key](z)
                lvs = None
            xs, z_cur = block.forward_uncond(
                xs,
                t=temp,
                lvs=lvs,
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
        h_decoder_top=None,
        prev_latents=None,
    ):
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)

        if len(latents) != len(self.dec_blocks):
            raise ValueError(
                f"Expected {len(self.dec_blocks)} latents, got {len(latents)}"
            )
        current_latents = []
        for idx, (block, lvs) in enumerate(itertools.zip_longest(self.dec_blocks, latents)):
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
        h_decoder_top=None,
        prev_latents=None,
    ):
        n = z_top_map.shape[0]
        latents = [z_top_map] + [None] * (len(self.dec_blocks) - 1)
        return self.forward_manual_latents(
            n,
            latents,
            t=t,
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
        self.top_prior = prior
        self.top_prior_snapshot = None
        self.top_prior_gate = None
        self.top_prior_ready = False
        self.top_kl_weight = float(top_kl_weight)
        self.prior_kl_mc_samples = int(prior_kl_mc_samples)
        super().__init__(H)

    def build(self):
        self.encoder = Encoder(self.H)
        self.decoder = Decoder(self.H)

        # IMPORTANT: This is the channel count of h_t only, NOT concat(h_t, z_tilde_t)
        self.top_h_dim = int(getattr(self.H, "top_h_context_dim", 0)) #h_t


    def forward(
        self,
        x,
        x_target,
        h_prior_top=None,
        mask_t=None,
        prev_latents=None,
        get_latents=False,
        slot_repulsion_tau =1.0,
    ):
        activations = self.encoder.forward(x)
        px_z, stats, current_latents = self.decoder.forward(
            activations,
            get_latents=get_latents,
            h_decoder_top=h_prior_top,
            prev_latents=prev_latents,
        )
        B = x_target.shape[0]
        BS = px_z.shape[0]
        if BS % B != 0:
            raise ValueError(
                f"Decoder output batch {BS} is not divisible by image batch {B}. "
                "This means slot flattening B*S was not propagated consistently."
            )

        distortion_per_pixel = self.decoder.out_net.nll(px_z, x_target)
        ndims = float(np.prod(x_target.shape[1:]))

        rate_gauss = torch.zeros_like(distortion_per_pixel)
        top_q_mean_map = None
        top_q_logvar_map = None
        top_slot_mu = None
        top_slot_logsigma = None
        top_slot_resp = None
        top_slot_attn = None
        top_slot_comp_maps = None
        top_slot_comp_logsigma = None
        z_top_state = None
        for block, st in zip(self.decoder.dec_blocks, stats):
            kl_block = st["kl"]
            if block.is_top:
                top_q_mean_map = st.get("posterior_mean", None)
                top_q_logvar_map = st.get("posterior_logvar", None)
                top_slot_mu = st.get("slot_mu", None)
                top_slot_logsigma = st.get("slot_logsigma", None)
                top_slot_attn = st.get("slot_attn", None)
                top_slot_comp_maps = st.get("slot_comp_maps", None)
                top_slot_comp_logsigma = st.get("slot_comp_logsigma", None)
                z_top_state = st.get("z_top_state", None)
            else:
                if kl_block.shape[0] == B:
                    kl_img = kl_block
                elif kl_block.shape[0] % B == 0:
                    S_kl = kl_block.shape[0] // B
                    kl_img = kl_block.view(B, S_kl).sum(dim=1)
                else:
                    raise ValueError(
                        f"KL batch {kl_block.shape[0]} is not compatible with image batch {B}."
                    )
                rate_gauss = rate_gauss + kl_img

        rate_gauss = rate_gauss / ndims
        dp_rate = torch.zeros_like(distortion_per_pixel)
        slot_div_loss = torch.zeros_like(distortion_per_pixel)
        top_bridge_rate = torch.zeros_like(distortion_per_pixel)

        slot_div_w = float(getattr(self.H, "slot_diversity_weight", 0.0))

        if slot_div_w > 0.0:
            d2 = torch.cdist(top_slot_mu, top_slot_mu, p=2) ** 2
            slots = F.normalize(top_slot_mu, dim=-1, eps=torch.finfo(top_slot_mu.dtype).eps)
            sim = torch.einsum("bsd,btd->bst", slots, slots)

            S = top_slot_mu.shape[1]
            eye = torch.eye(
                S,
                device=top_slot_mu.device,
                dtype=top_slot_mu.dtype,
            )[None]  # [1, S, S]

            off = 1.0 - eye  # remove diagonal/self-comparisons

            denom = S * (S - 1) + torch.finfo(top_slot_mu.dtype).eps

            slot_div_img = ((sim * off) ** 2).sum(dim=(1, 2))

            slot_div_loss = slot_div_w *( slot_div_img + (torch.exp(-d2/slot_repulsion_tau) * off).sum(dim=(1,2))) / denom
        dp_kl_img = None

        if self.top_prior is not None and bool(getattr(self, "top_prior_ready", False)) and self.top_prior_snapshot is not None and self.top_prior_gate is not None and h_prior_top is not None and top_slot_mu is not None and top_slot_logsigma is not None:

            dp_kl_img , top_slot_resp = compute_slot_kl_conditional_frozen(
                snapshot=self.top_prior_snapshot,
                frozen_gate=self.top_prior_gate,
                h_t=h_prior_top,
                slot_mu=top_slot_mu,
                slot_logsigma=top_slot_logsigma,
            )

            dp_rate = self.top_kl_weight * dp_kl_img  / ndims
        if (h_prior_top is not None and top_slot_mu is not None and top_slot_comp_maps is not None and top_slot_comp_logsigma is not None):
            _, _, _, prior_comp_mu, prior_comp_logsigma, _ = self.decoder.dec_blocks[0].top_slot_posterior.slot_to_top_prior(
                top_slot_mu.detach(),
                h_prior_top,
                temperature=0.0,
            )

            B0, S0, C0, H0, W0 = top_slot_comp_maps.shape

            kl_flat = gaussian_analytical_kl(
                top_slot_comp_maps.detach().reshape(B0 * S0, C0, H0, W0),
                prior_comp_mu.reshape(B0 * S0, C0, H0, W0),
                top_slot_comp_logsigma.detach().reshape(B0 * S0, C0, H0, W0),
                prior_comp_logsigma.reshape(B0 * S0, C0, H0, W0),
            )

            top_bridge_rate = kl_flat.view(B0, S0).sum(dim=1) / ndims
        total_rate = rate_gauss + dp_rate + slot_div_loss + top_bridge_rate
        elbo_per_sample = distortion_per_pixel + total_rate

        if mask_t is None:
            reduce = torch.mean
            valid_count = torch.tensor(float(elbo_per_sample.shape[0]), device=elbo_per_sample.device)
        else:
            vm = mask_t.float().view(-1).to(device=elbo_per_sample.device, dtype=elbo_per_sample.dtype)
            den = vm.sum().clamp(min=1.0)
            reduce = lambda z: (z * vm).sum() / den
            valid_count = den

        out = {
            "elbo": reduce(elbo_per_sample),
            "distortion": reduce(distortion_per_pixel),
            "gauss_rate": reduce(rate_gauss),
            "rate": reduce(total_rate),
            "dp_rate": reduce(dp_rate),
            "top_bridge_rate": reduce(top_bridge_rate),
            "px_z": px_z,
            "top_q_mean_map": top_q_mean_map,
            "top_q_logvar_map": top_q_logvar_map,
            "valid_count": valid_count,
            # Actual top slot posterior.
            "top_slot_mu": top_slot_mu,
            "top_slot_logsigma": top_slot_logsigma,
            "top_slot_resp": top_slot_resp,
            "top_slot_attn": top_slot_attn,
            "top_slot_comp_maps": top_slot_comp_maps,
            "top_slot_comp_logsigma": top_slot_comp_logsigma,
            "current_latents": current_latents,
            "z_top_state": z_top_state,
        }
        if dp_kl_img is not None:
            out["dp_kl"] = reduce(dp_kl_img)
        if get_latents:
            out["stats"] = stats
        return out

    def decode_from_top_latent(
        self,
        z_top_map,
        h_decoder_top=None,
        t=None,
        prev_latents=None,
    ):
        """
        return pz and current latents
        """
        return self.decoder.forward_from_top_latent(
            z_top_map,
            t=t,
            h_decoder_top=h_decoder_top,
            prev_latents=prev_latents,
        )

    def sample(
        self,
        n_batch,
        h_prior_top,
        t=None,
        prev_latents=None,
        temperature: float =1.0,
    ):
        slots, _, _ = sample_slot_vectors_conditional_frozen(snapshot=self.top_prior_snapshot, frozen_gate=self.top_prior_gate, h_t=h_prior_top, num_slots = self.decoder.dec_blocks[0].top_slot_posterior.num_slots, assignment_temperature=temperature, slot_temperature=temperature)
        
        _, _, _, _, _, z_slot_maps = self.decoder.dec_blocks[0].top_slot_posterior.slot_to_top_prior(slots.contiguous(), h_prior_top, temperature=temperature)
        px_z, current_latents = self.decoder.forward_from_top_latent(
            z_slot_maps,
            t=t,
            h_decoder_top=h_prior_top,
            prev_latents=prev_latents,
        )
        return self.decoder.out_net.sample(px_z), current_latents

    def forward_get_latents(
        self,
        x,
        h_decoder_top=None,
        prev_latents=None,
    ):
        activations = self.encoder.forward(x)
        _, stats, current_latents = self.decoder.forward(
            activations,
            get_latents=True,
            h_decoder_top=h_decoder_top,
            prev_latents=prev_latents,
        )
        return stats, current_latents

    def forward_uncond_samples(
        self,
        n_batch,
        t=None,
        h_decoder_top=None,
        prev_latents=None,
        z_top=None,
    ):
        if z_top is None:
            raise ValueError("z_top must be provided for unconditional sampling with forward_uncond_samples")
        px_z, current_latents = self.decoder.forward_uncond(
            n_batch,
            t=t,
            h_decoder_top=h_decoder_top,
            prev_latents=prev_latents,
            z_top=z_top,
        )

        return self.decoder.out_net.sample(px_z), current_latents

    def forward_samples_set_latents(
        self,
        n_batch,
        latents,
        t=None,
        h_decoder_top=None,
        prev_latents=None,
    ):
        px_z, current_latents = self.decoder.forward_manual_latents(
            n_batch,
            latents,
            t=t,
            h_decoder_top=h_decoder_top,
            prev_latents=prev_latents,
        )
        return self.decoder.out_net.sample(px_z), current_latents

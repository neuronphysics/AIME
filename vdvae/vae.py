import torch
from torch import nn
from torch.nn import functional as F
from vdvae.vae_helpers import HModule, get_1x1, get_3x3, DmolNet, draw_gaussian_diag_samples, gaussian_analytical_kl
from collections import defaultdict
import numpy as np
import itertools
from torch.utils.checkpoint import checkpoint as ckpt

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


class Encoder(HModule):
    def build(self):

        H = self.H
        self.use_checkpoint = getattr(H, "use_checkpoint", False)
        self.in_conv = get_3x3(H.image_channels, H.width)
        self.widths = get_width_settings(H.width, H.custom_width_str)
        enc_blocks = []
        blockstr = parse_layer_string(H.enc_blocks)
        for res, down_rate in blockstr:
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            enc_blocks.append(Block(self.widths[res], int(self.widths[res] * H.bottleneck_multiple), self.widths[res], down_rate=down_rate, residual=True, use_3x3=use_3x3))
        n_blocks = len(blockstr)
        for b in enc_blocks:
            b.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.enc_blocks = nn.ModuleList(enc_blocks)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.in_conv(x)
        activations = {}
        activations[x.shape[2]] = x
        for block in self.enc_blocks:
            if self.train and self.use_checkpoint:
                x = ckpt(block, x, use_reentrant=False)  
            else:
                x = block(x)
            res = x.shape[2]
            x = x if x.shape[1] == self.widths[res] else pad_channels(x, self.widths[res])
            activations[res] = x
        return activations


class DecBlock(nn.Module):
    def __init__(self, H, res, mixin, n_blocks, is_top=False):
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
        self.enc = Block(width * 2, cond_width, H.zdim * 2, residual=False, use_3x3=use_3x3)
        self.prior = Block(width, cond_width, H.zdim * 2 + width, residual=False, use_3x3=use_3x3, zero_last=True)
        self.z_proj = get_1x1(H.zdim, width)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
        self.resnet = Block(width, cond_width, width, residual=True, use_3x3=use_3x3)
        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.z_fn = lambda x: self.z_proj(x)

    def sample(self, x, acts):
        enc_in = torch.cat([x, acts], dim=1)
        if self.train and self.use_checkpoint:
            qm, qv = ckpt(self.enc, enc_in, use_reentrant=False).chunk(2, dim=1)
            feats = ckpt(self.prior, x, use_reentrant=False)
        else:
            qm, qv = self.enc(enc_in).chunk(2, dim=1)
            feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]
        x = x + xpp
        z = draw_gaussian_diag_samples(qm, qv)
        kl = gaussian_analytical_kl(qm, pm, qv, pv)
        return z, x, kl, qm, qv

    def sample_uncond(self, x, t=None, lvs=None):
        n, c, h, w = x.shape
        feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]
        x = x + xpp
        if lvs is not None:
            z = lvs
        else:
            if t is not None:
                pv = pv + torch.ones_like(pv) * np.log(t)
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

    def forward(self, xs, activations, get_latents=False):
        x, acts = self.get_inputs(xs, activations)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x, kl, qm, qv = self.sample(x, acts)
        x = x + self.z_fn(z)
        if self.train and self.use_checkpoint:
            x = ckpt(self.resnet, x, use_reentrant=False)
        else:
            x = self.resnet(x)
        xs[self.base] = x
        stats = {'kl': kl, 'posterior_mean': qm, 'posterior_logvar': qv}
        if get_latents:
            stats['z'] = z.detach()
            return xs, stats
        return xs, stats

    def forward_uncond(self, xs, t=None, lvs=None):
        try:
            x = xs[self.base]
        except KeyError:
            ref = xs[list(xs.keys())[0]]
            x = torch.zeros(dtype=ref.dtype, size=(ref.shape[0], self.widths[self.base], self.base, self.base), device=ref.device)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x = self.sample_uncond(x, t, lvs=lvs)
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
        n_blocks = len(blocks)
        for idx, (res, mixin) in enumerate(blocks):
            is_top = (idx == 0)
            dec_blocks.append(DecBlock(H, res, mixin, n_blocks=n_blocks, is_top=is_top))
            resos.add(res)
        self.resolutions = sorted(resos)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.bias_xs = nn.ParameterList([nn.Parameter(torch.zeros(1, self.widths[res], res, res)) for res in self.resolutions if res <= H.no_bias_above])
        self.out_net = DmolNet(H)
        self.gain = nn.Parameter(torch.ones(1, H.width, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, H.width, 1, 1))
        self.final_fn = lambda x: x * self.gain + self.bias

    def forward(self, activations, get_latents=False):
        stats = []
        xs = {a.shape[2]: a for a in self.bias_xs}
        for block in self.dec_blocks:
            xs, block_stats = block(xs, activations, get_latents=get_latents)
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
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size]

    def forward_manual_latents(self, n, latents, t=None):
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        for block, lvs in itertools.zip_longest(self.dec_blocks, latents):
            xs = block.forward_uncond(xs, t, lvs=lvs)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size]

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
        self.prior = prior
        self.top_kl_weight = top_kl_weight
        self.prior_kl_mc_samples = prior_kl_mc_samples
        super().__init__(H)  # calls self.build()

    def build(self):
        self.encoder = Encoder(self.H)
        self.decoder = Decoder(self.H)

    @staticmethod
    def add_coord_no_proj(z_map: torch.Tensor, scale: float = 0.05):
        # z_map: [B, C, H, W]
        B, C, H, W = z_map.shape
        ys = torch.linspace(-1.0, 1.0, H, device=z_map.device, dtype=z_map.dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=z_map.device, dtype=z_map.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")              # [H,W]
        pos2 = torch.stack([gy, gx], dim=0)                         # [2,H,W]
        posC = pos2.repeat((C + 1) // 2, 1, 1)[:C]                  # [C,H,W] (tile/truncate)
        return z_map + scale * posC.unsqueeze(0)                    # [B,C,H,W]

    def forward(self, x: torch.Tensor, x_target: torch.Tensor, h_context: torch.Tensor):
        """
        x:        [B, H, W, C]
        x_target: same as x

        h_context: optional [B, hidden_dim] conditioning vector for the DPGMM prior.

        IMPORTANT: for the DPGMM we treat the top-level spatial latents
        z_top[b, c, h, w] as a *bag of feature vectors* of shape [C],
        i.e. we reshape [B, C, Ht, Wt] -> [B * Ht * Wt, C].
        """

        # 1) Encode / decode
        activations = self.encoder.forward(x)                     # encoder expects NHWC
        px_z, stats = self.decoder.forward(activations, get_latents=True)

        # 2) Reconstruction loss
        distortion_per_pixel = self.decoder.out_net.nll(px_z, x_target)  # [B]
        ndims = np.prod(x.shape[1:])  # H * W * C

        # 3) Gaussian KL from all NON-top blocks only
        rate_gauss = torch.zeros_like(distortion_per_pixel)  # [B]

        top_q_mean_map = None   # [B, C, Ht, Wt]
        top_q_logvar_map = None # [B, C, Ht, Wt]

        for block, statdict in zip(self.decoder.dec_blocks, stats):
            kl_block = statdict["kl"].sum(dim=(1, 2, 3))  # [B]

            if getattr(block, "is_top", False):
                # Top-most stochastic block: we REPLACE its Gaussian prior
                # with the DPGMM, so we do NOT add its KL here.
                if "posterior_mean" in statdict and "posterior_logvar" in statdict:
                    top_q_mean_map = statdict["posterior_mean"]      # [B, C, Ht, Wt]
                    top_q_logvar_map = statdict["posterior_logvar"]  # [B, C, Ht, Wt]
            else:
                # Standard Gaussian KL for all lower blocks
                rate_gauss += kl_block

        rate_gauss = rate_gauss / ndims  # [B]

        # 4) DPGMM KL at the top level, using spatial tokens [B*Ht*Wt, C]
        dp_kl = None
        dp_rate = torch.zeros(
            1, device=distortion_per_pixel.device, dtype=distortion_per_pixel.dtype
        ).squeeze(0)

        if (
            self.prior is not None
            and top_q_mean_map is not None
            and top_q_logvar_map is not None
        ):
            B, C, Ht, Wt = top_q_mean_map.shape

            # --- (a) Reshape latents: [B, C, Ht, Wt] -> [B*Ht*Wt, C] ---
            # we treat each spatial location as an independent feature vector
            top_q_mean_tokens = (
                top_q_mean_map.permute(0, 2, 3, 1)   # [B, Ht, Wt, C]
                .contiguous()
                .view(B * Ht * Wt, C)                # [N, C], N=B*Ht*Wt
            )
            top_q_logvar_tokens = (
                top_q_logvar_map.permute(0, 2, 3, 1)
                .contiguous()
                .view(B * Ht * Wt, C)
            )

            # --- (b) Build / validate context and broadcast per token ---
            if h_context.shape[0] != B:
                raise ValueError(
                    f"h_context batch size {h_context.shape[0]} does not match "
                    f"top latent batch size {B}."
                )
            if h_context.shape[1] != self.prior.hidden_dim:
                raise ValueError(
                    f"h_context dim {h_context.shape[1]} does not match "
                    f"prior.hidden_dim={self.prior.hidden_dim}."
                )
            # [B, hidden_dim]

            B_h, Hc = h_context.shape
            assert B_h == B
            h_map = h_context.view(B,1,1,Hc).expand(B,Ht,Wt,Hc).permute(0,3,1,2)  # [B,Hc,Ht,Wt]
            h_map = self.add_coord_no_proj(h_map, scale=0.05)                           # [B,Hc,Ht,Wt]
            h_tokens = h_map.permute(0,2,3,1).reshape(B*Ht*Wt, Hc)                 # [N,Hc]

            # --- (c) DPGMM prior over tokens ---
            prior_dist, prior_params = self.prior(h_tokens)

            # Monte Carlo KL between q(z_token|x) and DP-GMM prior p(z_token | h_tokens)
            # posterior_mean, posterior_logvar: [N, C] where N=B*Ht*Wt
            dp_kl = self.prior.compute_kl_divergence_mc(
                posterior_mean=top_q_mean_tokens,
                posterior_logvar=top_q_logvar_tokens,
                prior_params=prior_params,
                n_samples=self.prior_kl_mc_samples,
            )  # scalar (average over N tokens and MC samples)

            # Convert to "per-pixel" rate to match VDVAE convention
            dp_rate = self.top_kl_weight * dp_kl / ndims  # scalar

        # 5) Total rate and ELBO
        total_rate_per_pixel = rate_gauss + dp_rate  # [B] + scalar
        elbo = (distortion_per_pixel + total_rate_per_pixel).mean()
        out = dict(
            elbo=elbo,
            distortion=distortion_per_pixel.mean(),
            gauss_rate=rate_gauss.mean(),
            rate=total_rate_per_pixel.mean(),
            dp_kl=dp_kl,
            dp_rate=dp_rate,
            prior_params=prior_params,
            px_z=px_z,
            top_q_mean_map=top_q_mean_map,
            top_q_logvar_map=top_q_logvar_map,
        )

        if dp_kl is not None:
            out["dp_kl"] = dp_kl
            out["dp_rate"] = dp_rate
        return out

    def sample(self, n_batch: int, h_context: torch.Tensor):
        """
        Sample from the model using the DPGMM at the top level.

        h_context: [B, hidden_dim], where B == n_batch
        """
        assert self.prior is not None, "VDVAE.sample requires a DPGMMPrior."
        assert h_context.shape[0] == n_batch, (
            f"h_context batch {h_context.shape[0]} != n_batch {n_batch}"
        )
        assert h_context.shape[1] == self.prior.hidden_dim, (
            f"h_context dim {h_context.shape[1]} != prior.hidden_dim {self.prior.hidden_dim}"
        )

        top_block = self.decoder.dec_blocks[0]
        C = top_block.zdim    # latent channel dimension
        res = top_block.base  # top resolution Ht = Wt

        B = n_batch
        Hc = h_context.shape[1]
        h_map = h_context.view(B,1,1,Hc).expand(B,res,res,Hc).permute(0,3,1,2)  # [B,Hc,Ht,Wt]
        h_map = self.add_coord_no_proj(h_map, scale=0.05)                           # [B,Hc,Ht,Wt]
        h_tokens = h_map.permute(0,2,3,1).reshape(B*res*res, Hc)                 # [N,Hc]


        # DPGMM prior over tokens
        prior_dist, prior_params = self.prior(h_tokens)
        z_tokens = prior_dist.sample()  # [B * res * res, C]
        N, D = z_tokens.shape

        assert D == C, f"DPGMM latent_dim {D} != top_block.zdim {C}"
        assert N == B * res * res, (
            f"Sampled {N} tokens, expected B*res*res = {B * res * res}."
        )

        # Reshape back to a spatial map [B, C, res, res]
        z_top_map = (
            z_tokens.view(B, res, res, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # Feed sampled latents into decoder
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

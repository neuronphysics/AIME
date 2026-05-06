import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Iterable
from torch import Tensor
from einops import reduce
def assert_shape(actual: Union[torch.Size, Tuple[int, ...]], expected: Tuple[int, ...], message: str = ""):
    assert actual == expected, f"Expected shape: {expected} but passed shape: {actual}. {message}"

def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing="ij")
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)

class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size: int, resolution: Tuple[int, int]):
        super().__init__()
        self.dense = nn.Linear(in_features=4, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs: Tensor):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        assert_shape(inputs.shape[1:], emb_proj.shape[1:])
        return inputs + emb_proj

DEFAULT_WEIGHT_INIT = "default"
def init_parameters(layers: Union[nn.Module, Iterable[nn.Module]], weight_init: str = "default"):
    assert weight_init in ("default", "he_uniform", "he_normal", "xavier_uniform", "xavier_normal")
    if isinstance(layers, nn.Module):
        layers = [layers]

    for idx, layer in enumerate(layers):
        if hasattr(layer, "bias") and layer.bias is not None:
            nn.init.zeros_(layer.bias)

        if hasattr(layer, "weight") and layer.weight is not None:
            gain = 1.0
            if isinstance(layers, nn.Sequential):
                if idx < len(layers) - 1:
                    next = layers[idx + 1]
                    if isinstance(next, nn.ReLU):
                        gain = 2**0.5

            if weight_init == "he_uniform":
                torch.nn.init.kaiming_uniform_(layer.weight, gain)
            elif weight_init == "he_normal":
                torch.nn.init.kaiming_normal_(layer.weight, gain)
            elif weight_init == "xavier_uniform":
                torch.nn.init.xavier_uniform_(layer.weight, gain)
            elif weight_init == "xavier_normal":
                torch.nn.init.xavier_normal_(layer.weight, gain)

def get_activation_fn(name_or_instance: Union[str, nn.Module]) -> nn.Module:
    if isinstance(name_or_instance, nn.Module):
        return name_or_instance
    elif isinstance(name_or_instance, str):
        if name_or_instance.lower() == "relu":
            return nn.ReLU(inplace=True)
        elif name_or_instance.lower() == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation function {name_or_instance}")
    else:
        raise ValueError(
            f"Unsupported type for activation function: {type(name_or_instance)}. "
            "Can be `str` or `torch.nn.Module`."
        )

class MLP(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        hidden_dims: List[int],
        initial_layer_norm: bool = False,
        activation: Union[str, nn.Module] = "relu",
        final_activation: Union[bool, str] = False,
        residual: bool = False,
        weight_init: str = DEFAULT_WEIGHT_INIT,
    ):
        super().__init__()
        self.residual = residual
        if residual:
            assert inp_dim == outp_dim

        layers = []
        if initial_layer_norm:
            layers.append(nn.LayerNorm(inp_dim))

        cur_dim = inp_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(cur_dim, dim))
            layers.append(get_activation_fn(activation))
            cur_dim = dim

        layers.append(nn.Linear(cur_dim, outp_dim))
        if final_activation:
            if isinstance(final_activation, bool):
                final_activation = "relu"
            layers.append(get_activation_fn(final_activation))

        self.layers = nn.Sequential(*layers)
        init_parameters(self.layers, weight_init)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        outp = self.layers(inp)

        if self.residual:
            return inp + outp
        else:
            return outp

class SlotAttention(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        slot_dim: int,
        kvq_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        n_iters: int = 3,
        eps: float = 1e-8,
        use_gru: bool = True,
        use_mlp: bool = True,
    ):
        super().__init__()
        assert n_iters >= 1

        if kvq_dim is None:
            kvq_dim = slot_dim
        self.to_k = nn.Linear(inp_dim, kvq_dim, bias=False)
        self.to_v = nn.Linear(inp_dim, kvq_dim, bias=False)
        self.to_q = nn.Linear(slot_dim, kvq_dim, bias=False)

        if use_gru:
            self.gru = nn.GRUCell(input_size=kvq_dim, hidden_size=slot_dim)
        else:
            assert kvq_dim == slot_dim
            self.gru = None

        if hidden_dim is None:
            hidden_dim = 4 * slot_dim

        if use_mlp:
            self.mlp = MLP(slot_dim, slot_dim, [hidden_dim], initial_layer_norm=True, residual=True)
        else:
            self.mlp = None

        self.norm_features = nn.LayerNorm(inp_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

        self.n_iters = n_iters
        self.eps = eps
        self.scale = kvq_dim**-0.5

    def step(
        self, slots: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one iteration of slot attention."""
        slots = self.norm_slots(slots)
        queries = self.to_q(slots)

        dots = torch.einsum("bsd, bfd -> bsf", queries, keys) * self.scale
        pre_norm_attn = torch.softmax(dots, dim=1)
        attn = pre_norm_attn + self.eps
        attn = attn / attn.sum(-1, keepdim=True)

        updates = torch.einsum("bsf, bfd -> bsd", attn, values)

        if self.gru:
            updated_slots = self.gru(updates.flatten(0, 1), slots.flatten(0, 1))
            slots = updated_slots.unflatten(0, slots.shape[:2])
        else:
            slots = slots + updates

        if self.mlp is not None:
            slots = self.mlp(slots)

        return slots, pre_norm_attn

    def forward(self, slots: torch.Tensor, features: torch.Tensor, n_iters: Optional[int] = None):
        features = self.norm_features(features)
        keys = self.to_k(features)
        values = self.to_v(features)

        if n_iters is None:
            n_iters = self.n_iters

        for _ in range(n_iters):
            slots, pre_norm_attn = self.step(slots, keys, values)

        return {"slots": slots, "masks": pre_norm_attn}

class SlotToTopMap(nn.Module):
    """
    Converts top-level slots back into a VDVAE top latent map.
    slot_z: [B, S, D]
    output z_top: [B, C, H, W]
    """
    def __init__(
        self,
        slot_dim: int,
        z_channels: int,
        top_h: int,
        top_w: int,
        hidden: Optional[int] = None,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.z_channels = z_channels
        self.top_h = top_h
        self.top_w = top_w

        h = hidden if hidden is not None else 4 * slot_dim

        self.net = nn.Sequential(
            nn.Conv2d(slot_dim + 4, h, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(h, h, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(h, z_channels + 1, kernel_size=1),
        )

    def _coord_grid(self, B: int, S: int, device, dtype):
        y = torch.linspace(-1.0, 1.0, self.top_h, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, self.top_w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([xx, yy, 1.0 - xx, 1.0 - yy], dim=0)
        grid = grid[None, None].expand(B, S, 4, self.top_h, self.top_w)
        return grid

    def forward(self, slot_z: torch.Tensor):
        B, S, D = slot_z.shape

        z = slot_z[:, :, :, None, None].expand(B, S, D, self.top_h, self.top_w)
        pos = self._coord_grid(B, S, slot_z.device, slot_z.dtype)

        inp = torch.cat([z, pos], dim=2)
        inp = inp.reshape(B * S, D + 4, self.top_h, self.top_w)

        out = self.net(inp)
        out = out.view(B, S, self.z_channels + 1, self.top_h, self.top_w)

        comp = out[:, :, :self.z_channels]
        mask_logits = out[:, :, self.z_channels:self.z_channels + 1]
        masks = torch.softmax(mask_logits, dim=1)

        z_top = torch.sum(comp * masks, dim=1)
        return z_top, comp, masks

class SlotBottleneck(nn.Module):
    """
    Inputs:
        top_q_mean_map: [B, C, H, W]

    Outputs:
        slot posterior parameters, slot samples, and reconstructed top map.
    """
    def __init__(
        self,
        z_channels: int,
        top_h: int,
        top_w: int,
        hidden_dim: int,
        num_slots: int,
        slot_dim: int,
        n_iters: int = 3,
        logsigma_clamp=(-7.0, 3.0),
    ):
        super().__init__()

        self.z_channels = z_channels
        self.top_h = top_h
        self.top_w = top_w
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.logsigma_clamp = logsigma_clamp

        feat_dim = z_channels

        self.pre_pos = SoftPositionEmbed(hidden_size=feat_dim,resolution=(top_h, top_w))
        self.pre_norm = nn.LayerNorm(feat_dim)

        # Shared slot seed, preserving slot exchangeability.
        self.slot_init_mu = nn.Parameter(torch.zeros(1, 1, slot_dim))
        self.slot_init_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slot_init_mu)
        nn.init.xavier_uniform_(self.slot_init_logsigma)

        self.slot_attention = SlotAttention(
            inp_dim=feat_dim,
            slot_dim=slot_dim,
            kvq_dim=slot_dim,
            hidden_dim=4 * slot_dim,
            n_iters=n_iters,
        )

        self.to_mu = nn.Linear(slot_dim, slot_dim)
        self.to_logsigma = nn.Linear(slot_dim, slot_dim)

        self.slot_to_map = SlotToTopMap(
            slot_dim=slot_dim,
            z_channels=z_channels,
            top_h=top_h,
            top_w=top_w,
            hidden=hidden_dim
        )

    def forward(self, top_q_mean_map: torch.Tensor):
        B, C, H, W = top_q_mean_map.shape

        assert H == self.top_h and W == self.top_w
        feat = self.pre_pos(top_q_mean_map)

        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.pre_norm(tokens)

        eps = torch.randn(
            B,
            self.num_slots,
            self.slot_dim,
            device=feat.device,
            dtype=feat.dtype,
        )

        init_mu = self.slot_init_mu.expand(B, self.num_slots, -1)
        init_std = self.slot_init_logsigma.exp().expand(B, self.num_slots, -1)
        init_slots = init_mu + init_std * eps

        attn_out = self.slot_attention(init_slots, tokens)
        slots = attn_out["slots"]
        slot_attn = attn_out["masks"]

        slot_mu = self.to_mu(slots)
        slot_logsigma = self.to_logsigma(slots).clamp(*self.logsigma_clamp)

        slot_z = slot_mu + slot_logsigma.exp() * torch.randn_like(slot_mu)

        z_top_map, comp_maps, comp_masks = self.slot_to_map(slot_z)
        # Calculate entropy regularization
        entropy = reduce(
                         torch.special.entr(slot_attn.clamp(min=1e-20, max=1)),
                         "b s f -> b",
                          "mean"
                        )

        return {
            "slot_mu": slot_mu,
            "slot_logsigma": slot_logsigma,
            "slot_z": slot_z,
            "slot_attn": slot_attn,
            "z_top_map": z_top_map,
            "slot_comp_maps": comp_maps,
            "slot_masks": comp_masks,
            "entropy_reg": entropy,
        }

@torch.jit.script
def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)


@torch.jit.script
def draw_gaussian_diag_samples(mu, logsigma):
    eps = torch.empty_like(mu).normal_(0., 1.)
    return torch.exp(logsigma) * eps + mu


def get_conv(in_dim, out_dim, kernel_size, stride, padding, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c


def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups, scaled=scaled)


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups, scaled=scaled)


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.shape) - 1
    m = x.max(dim=axis, keepdim=True)[0]
    return x - m - torch.log(torch.exp(x - m).sum(dim=axis, keepdim=True))


def const_max(t, constant):
    other = torch.ones_like(t) * constant
    return torch.max(t, other)


def const_min(t, constant):
    other = torch.ones_like(t) * constant
    return torch.min(t, other)


def discretized_mix_logistic_loss(x, l, low_bit=False):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Adapted from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
    xs = [s for s in x.shape]  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = [s for s in l.shape]  # predicted distribution, e.g. (B,32,32,100)
    nr_mix = int(ls[-1] / 10)  # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = const_max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    x = torch.reshape(x, xs + [1]) + torch.zeros(xs + [nr_mix]).to(x.device)  # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = torch.reshape(means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    m3 = torch.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    means = torch.cat([torch.reshape(means[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix]), m2, m3], dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    if low_bit:
        plus_in = inv_stdv * (centered_x + 1. / 31.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 31.)
    else:
        plus_in = inv_stdv * (centered_x + 1. / 255.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)  # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    if low_bit:
        log_probs = torch.where(x < -0.999,
                                log_cdf_plus,
                                torch.where(x > 0.999,
                                            log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(const_max(cdf_delta, 1e-12)),
                                                        log_pdf_mid - np.log(15.5))))
    else:
        log_probs = torch.where(x < -0.999,
                                log_cdf_plus,
                                torch.where(x > 0.999,
                                            log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(const_max(cdf_delta, 1e-12)),
                                                        log_pdf_mid - np.log(127.5))))
    log_probs = log_probs.sum(dim=3) + log_prob_from_logits(logit_probs)
    mixture_probs = torch.logsumexp(log_probs, -1)
    return -1. * mixture_probs.sum(dim=[1, 2]) / np.prod(xs[1:])


def sample_from_discretized_mix_logistic(l, nr_mix):
    ls = [s for s in l.shape]
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    eps = torch.empty(logit_probs.shape, device=l.device).uniform_(1e-5, 1. - 1e-5)
    amax = torch.argmax(logit_probs - torch.log(-torch.log(eps)), dim=3)
    sel = F.one_hot(amax, num_classes=nr_mix).float()
    sel = torch.reshape(sel, xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = (l[:, :, :, :, :nr_mix] * sel).sum(dim=4)
    log_scales = const_max((l[:, :, :, :, nr_mix:nr_mix * 2] * sel).sum(dim=4), -7.)
    coeffs = (torch.tanh(l[:, :, :, :, nr_mix * 2:nr_mix * 3]) * sel).sum(dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.empty(means.shape, device=means.device).uniform_(1e-5, 1. - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = const_min(const_max(x[:, :, :, 0], -1.), 1.)
    x1 = const_min(const_max(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.), 1.)
    x2 = const_min(const_max(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.), 1.)
    return torch.cat([torch.reshape(x0, xs[:-1] + [1]), torch.reshape(x1, xs[:-1] + [1]), torch.reshape(x2, xs[:-1] + [1])], dim=3)

def mean_from_discretized_mix_logistic(l: torch.Tensor, nr_mix: int) -> torch.Tensor:
    """
    Differentiable mean reconstruction from DMOL parameters.

    Args:
        l: [B, H, W, 10*nr_mix] - output of DmolNet.forward()
        nr_mix: number of mixture components

    Returns:
        [B, H, W, 3] in [-1, 1], differentiable w.r.t. l
    """
    B, H, W, _ = l.shape

    # Mixture weights
    logit_probs = l[:, :, :, :nr_mix]
    probs = F.softmax(logit_probs, dim=-1)  # [B,H,W,M]

    # Reshape remaining params: [B,H,W,9M] -> [B,H,W,3,3M]
    params = l[:, :, :, nr_mix:].reshape(B, H, W, 3, 3 * nr_mix)

    # Extract per-channel params
    means = params[:, :, :, :, :nr_mix]                          # [B,H,W,3,M]
    coeffs = torch.tanh(params[:, :, :, :, 2*nr_mix:3*nr_mix])   # [B,H,W,3,M]

    # Autoregressive conditional expectations per mixture
    # Note: coeffs[:,…,i,:] is the i-th AR coefficient, not channel i's coeffs
    mu_r = means[:, :, :, 0, :]  # [B,H,W,M]
    mu_g = means[:, :, :, 1, :]
    mu_b = means[:, :, :, 2, :]

    c_rg = coeffs[:, :, :, 0, :]  # R → G coefficient
    c_rb = coeffs[:, :, :, 1, :]  # R → B coefficient
    c_gb = coeffs[:, :, :, 2, :]  # G → B coefficient

    # E[x|k] for each mixture component
    e_r_k = mu_r
    e_g_k = mu_g + c_rg * e_r_k
    e_b_k = mu_b + c_rb * e_r_k + c_gb * e_g_k

    # Marginal expectation: E[x] = Σ_k π_k E[x|k]
    e_r = (probs * e_r_k).sum(dim=-1)
    e_g = (probs * e_g_k).sum(dim=-1)
    e_b = (probs * e_b_k).sum(dim=-1)

    out = torch.stack([e_r, e_g, e_b], dim=-1)
    return out.clamp(-1., 1.)

class HModule(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.build()


class DmolNet(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.width = H.width
        self.out_conv = get_conv(H.width, H.num_mixtures * 10, kernel_size=1, stride=1, padding=0)

    def nll(self, px_z, x):
        return discretized_mix_logistic_loss(x=x, l=self.forward(px_z), low_bit=self.H.dataset in ['ffhq_256'])

    def forward(self, px_z):
        xhat = self.out_conv(px_z)
        return xhat.permute(0, 2, 3, 1)

    def sample(self, px_z):
        im = sample_from_discretized_mix_logistic(self.forward(px_z), self.H.num_mixtures)
        xhat = (im + 1.0) * 127.5
        xhat = xhat.detach().cpu().numpy()
        xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
        return xhat

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

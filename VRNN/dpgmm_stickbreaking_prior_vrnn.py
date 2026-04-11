import os
from pathlib import Path
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
_torch_cache = _project_root / "results" / "pretrained_weights"
os.environ['TORCH_HOME'] = str(_torch_cache)
# Now safe to import torch
import torch
import torchvision
from logging import config
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Gamma, Categorical, Independent, MixtureSameFamily
from typing import Dict, Tuple, List, Optional, Any
import sys
import gc
from einops import rearrange
from contextlib import contextmanager
from itertools import chain
import numpy as np
import math, inspect
from dataclasses import dataclass
from types import SimpleNamespace
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint as ckpt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vis_networks import EMA, TemporalDiscriminator, AddEpsilon, check_tensor, ImageDiscriminator
from VRNN.RGB import DynamicWeightAverage
from VRNN.lstm import LSTMLayer, SpatioTemporalCore
from VRNN.perceiver.position import RoPE
from vdvae.vae import VDVAE
from vdvae.hps import Hyperparams
from vdvae.vae_helpers import mean_from_discretized_mix_logistic, sample_from_discretized_mix_logistic, draw_gaussian_diag_samples
from VRNN.warp import (
    TVLoss, SSIM, create_outgoing_mask,
    charbonnier_loss, CensusLoss, multi_scale_warp_solver
)
from VRNN.flow_predict import LatentTransportINR

from VRNN.Kumaraswamy import KumaraswamyStable

def beta_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    """Compute beta function in log space for numerical stability"""
    eps= torch.finfo(torch.float32).eps
    return torch.lgamma(a + eps) + torch.lgamma(b + eps) - torch.lgamma(a + b + eps)

def prune_small_components(pi: torch.Tensor, threshold: float = 1e-5) -> torch.Tensor:
    """Prune components with very small mixing proportions"""
    # Find components above threshold
    eps =torch.finfo(torch.float32).eps
    mask = pi > threshold

    # Zero out small components
    pi = pi * mask

    # Renormalize
    pi = pi / (pi.sum(dim=-1, keepdim=True) + eps)

    return pi

def torch_load_version_compat(path, map_location=None):
    """
    Backward-compatible torch.load
    """

    sig = inspect.signature(torch.load)
    kwargs = {}


    if "map_location" in sig.parameters:
        kwargs["map_location"] = map_location

    # Newer torch versions have weights_only; set it to False explicitly
    if "weights_only" in sig.parameters:
        kwargs["weights_only"] = False

    return torch.load(path, **kwargs)

class GammaPosterior(nn.Module):
    """
    Example code:https://github.com/threewisemonkeys-as/PyTorch-VAE/blob/4ed0fc7581d4792b435134aa9e06d5e35a5db118/models/gamma_vae.py
    Amortized variational posterior for Gamma distribution
    Maps encoder hidden states to Gamma parameters
    """
    def __init__(self, hidden_dim: int, device: torch.device, eps: float= torch.finfo(torch.float32).eps):

        super().__init__()
        # Network for generating Gamma parameters from hidden states
        # Give unique names to each layer
        self.param_net = nn.Sequential(OrderedDict([
            ('gamma_fc', nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim))),
            ('gamma_ln', nn.LayerNorm(hidden_dim)),
            ('gamma_relu', nn.GELU()),
            ('gamma_out', nn.utils.spectral_norm(nn.Linear(hidden_dim, 2))),
            #('gamma_ln_out', nn.LayerNorm(2, eps=1e-6)),
            ('gamma_softplus', nn.Softplus(beta=0.5)),
            ('gamma_eps', AddEpsilon(eps))  # Ensure positive parameters
        ]))
        self.eps = eps
        self.device = device
        self.to(device)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate Gamma parameters from hidden representation
        """
        if torch.isnan(h).any() or torch.isinf(h).any():
            print(f"Warning: NaN/Inf in GammaPosterior input, replacing with safe values")
            mean_h = torch.nanmean(h)
            h = torch.where(torch.isnan(h)| torch.isinf(h), mean_h, h)

        params = self.param_net(h)
        concentration, rate = params.split(1, dim=-1)

        concentration = torch.clamp(concentration.squeeze(-1), min=self.eps)
        rate = torch.clamp(rate.squeeze(-1), min=self.eps)
        return concentration, rate

    def sample(self, h: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Sample from the Gamma posterior
        Returns [batch_size, n_samples]
        """
        concentration, rate = self.forward(h)
        gamma_dist = Gamma(concentration.unsqueeze(-1), rate.unsqueeze(-1))
        return gamma_dist.rsample((n_samples,)).transpose(0, 1)

    def kl_divergence(self, h: torch.Tensor, prior_concentration: torch.Tensor, prior_rate: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between Gamma posterior and prior

        KL(Gamma(α₁,β₁)||Gamma(α₂,β₂)) = α₂log(β₁/β₂) - logΓ(α₁) + logΓ(α₂) + (α₁-α₂)ψ(α₁) - (β₁-β₂)(α₁/β₁)
        """

        alpha, beta = self.forward(h)  # posterior parameters
        a, b = prior_concentration, prior_rate  # prior parameters

        term1 = a * torch.log(beta/b)
        term2 = -torch.lgamma(alpha) + torch.lgamma(a)
        term3 = (alpha - a) * torch.digamma(alpha)
        term4 = -(beta - b) * (alpha/beta)

        kl = term1 + term2 + term3 + term4
        return kl.mean()


class KumaraswamyNetwork(nn.Module):
    """
    Neural network to generate Kumaraswamy parameters
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_components: int, device: torch.device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.K = num_components
        self.device = device
        self.eps = torch.tensor(torch.finfo(torch.float32).eps, device=self.device)
        kumar_a_fc = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_uniform_(kumar_a_fc.weight, gain=0.5)
        nn.init.constant_(kumar_a_fc.bias, 0.0)

        kumar_a_out = nn.Linear(hidden_dim, self.K - 1)
        nn.init.normal_(kumar_a_out.weight, mean=0, std=0.001)
        nn.init.constant_(kumar_a_out.bias, 0.0)

        kumar_b_fc = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_uniform_(kumar_b_fc.weight, gain=0.5)
        nn.init.constant_(kumar_b_fc.bias, 0.0)

        kumar_b_out = nn.Linear(hidden_dim, self.K - 1)
        nn.init.normal_(kumar_b_out.weight, mean=0, std=0.001)
        nn.init.constant_(kumar_b_out.bias, 0.0)

        # Then apply spectral norm and build sequential
        self.net_a = nn.Sequential(OrderedDict([
            ('kumar_a_fc', nn.utils.spectral_norm(kumar_a_fc)),
            ('kumar_a_ln', nn.LayerNorm(hidden_dim, eps=1e-6)),
            ('kumar_a_relu', nn.SiLU()),
            ('kumar_a_out', nn.utils.spectral_norm(kumar_a_out)),
        ]))


        self.net_b = nn.Sequential(OrderedDict([
            ('kumar_b_fc', nn.utils.spectral_norm(kumar_b_fc)),
            ('kumar_b_ln', nn.LayerNorm(hidden_dim, eps=1e-6)),
            ('kumar_b_relu', nn.SiLU()),
            ('kumar_b_out', nn.utils.spectral_norm(kumar_b_out)),
        ]))
        self.to(device)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate Kumaraswamy parameters from hidden representation

        """
        if torch.isnan(h).any() or torch.isinf(h).any():
            print(f"Warning: NaN/Inf in KumaraswamyNetwork input, replacing with safe values")
            mean_h = torch.nanmean(h)
            h = torch.where(torch.isnan(h)| torch.isinf(h), mean_h, h)
        min_val = -5.0
        max_val = 5.0
        log_a = torch.clamp(self.net_a(h) , min=min_val, max=max_val)
        log_b = torch.clamp(self.net_b(h) , min=min_val, max=max_val)
        log_a = torch.nan_to_num(log_a, nan=0.0, posinf=5.0, neginf=-5.0)
        log_b = torch.nan_to_num(log_b, nan=0.0, posinf=5.0, neginf=-5.0)
        if torch.isnan(log_a).any() or torch.isnan(log_b).any():
            print(f"Warning: NaN in Kumaraswamy parameters: h_range= ({h.min()}, {h.max()})a_range=({log_a.min()}, {log_a.max()}), b_range=({log_b.min()}, {log_b.max()})")
            mean_a = torch.nanmean(log_a)  # Compute mean with ignoring NaNs
            log_a = torch.where(torch.isnan(log_a)| torch.isinf(log_a), mean_a, log_a)
            mean_b = torch.nanmean(log_b)
            log_b = torch.where(torch.isnan(log_b)| torch.isinf(log_b), mean_b, log_b)
        return log_a, log_b

class AdaptiveStickBreaking(nn.Module):
    """
    Implements adaptive stick-breaking construction for Dirichlet Process
    """
    def __init__(self, max_components: int, hidden_dim: int, device: torch.device,  prior_alpha: float = 1.0, prior_beta: float = 1.0, dkl_taylor_order:int=10):
        super().__init__()
        self._max_K = max_components
        self.hidden_dim = hidden_dim
        self.device = device
        self.M = dkl_taylor_order  # Taylor series order

        # Variational posterior over concentration parameter
        self.alpha_posterior = GammaPosterior(
                                             hidden_dim,          # positive initial rate
                                             device=device
                                            )
        # Gamma hyperprior parameters (can be learned or fixed)
        # Named gamma hyperprior parameters
        self.gamma_a = nn.Parameter(torch.tensor(prior_alpha, device=self.device), requires_grad=True)
        self.gamma_b = nn.Parameter(torch.tensor(prior_beta, device=self.device), requires_grad=True)

        # Neural network for generating stick-breaking proportions
        # Kumaraswamy parameter network
        self.kumar_net = KumaraswamyNetwork(input_dim=hidden_dim, hidden_dim=hidden_dim, num_components=max_components, device=device)
        self.to(self.device)

    @property
    def max_K(self) -> int:
        """Ensure max_K is always returned as an integer."""
        return int(self._max_K)  # Force it to be an int


    @staticmethod
    def sample_kumaraswamy(
                           log_a: torch.Tensor,
                           log_b: torch.Tensor,
                           max_k: int,
                           use_rand_perm: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample from Kumaraswamy distribution
        U~Uniform(0,1)
        X=(1-(1-U)**(1/b))**(1/a)
        https://arxiv.org/pdf/2410.00660
        """
        log_a = log_a.to(log_a.device)
        log_b = log_b.to(log_b.device)

        check_tensor(log_a, "kumar_a")
        check_tensor(log_b, "kumar_b")

        # Optional random permutation
        if use_rand_perm:
            # Generate random permutation indices
            perm = torch.argsort(torch.rand_like(log_a, device= log_a.device), dim=-1)
            # Apply permutation
            perm = perm.view(-1, max_k-1)

            log_a = torch.gather(log_a, dim=1, index=perm) # [batch_size, K-1]
            log_b = torch.gather(log_b, dim=1, index=perm) # [batch_size, K-1]
            # Generate full permutation for output

        else:
            perm = None

        # Sample uniformly with numerical stability
        # Create stable Kumaraswamy distributions
        kumar_dist = KumaraswamyStable(log_a, log_b)

        # Sample using the stable implementation
        v = kumar_dist.rsample()

        return v, perm

    @staticmethod
    def compute_stick_breaking_proportions(v: torch.Tensor,
                                        perm: Optional[torch.Tensor] = None) -> torch.Tensor:

        B = v.size(0)
        K = v.size(-1) + 1

        eps = torch.finfo(v.dtype).eps
        v = v.clamp(min=eps, max=1 - eps)

        log_prefix = torch.cumsum(torch.log1p(-v), dim=-1)
        log_prefix = F.pad(log_prefix, (1, 0), value=0.)

        log_v_padded = torch.log(F.pad(v, (0, 1), value=1.0))
        pi = torch.exp(log_v_padded + log_prefix)

        if perm is not None:
            if perm.size(0) != B or perm.size(-1) != K - 1:
                raise ValueError("perm must be [B, K-1] and match v's batch/K.")
            last_idx = torch.full((B, 1), K - 1, device=perm.device, dtype=perm.dtype)
            full_perm = torch.cat([perm, last_idx], dim=1)
            inv_perm = torch.argsort(full_perm, dim=1)
            pi = torch.gather(pi, 1, inv_perm)

        pi = pi / (pi.sum(dim=-1, keepdim=True) + eps)
        return pi

    def forward(self, h: torch.Tensor, use_rand_perm: bool = True, truncation_threshold: float = 0.999) -> Tuple[torch.Tensor, Dict]:

        # Generate stick-breaking proportions
        # Get Kumaraswamy parameters
        log_kumar_a, log_kumar_b = self.kumar_net(h)

        # Sample v from Kumaraswamy for each alpha sample
        v, perm = self.sample_kumaraswamy(log_kumar_a, log_kumar_b, self.max_K, use_rand_perm)  # [n_samples, batch, K-1]

        # Initialize mixing proportions
        pi = self.compute_stick_breaking_proportions(v, perm)
        # Adaptive truncation: find where cumulative sum exceeds threshold
        pi_sorted, sort_idx = torch.sort(pi.float(), dim=-1, descending=True)
        pi_cumsum = torch.cumsum(pi_sorted, dim=-1)
        K = pi_sorted.size(-1)

        # Find truncation point for each sample
        truncation_mask = (pi_cumsum >= truncation_threshold)
        T = truncation_mask.float().argmax(dim=-1)
        T = torch.where(~truncation_mask.any(dim=-1), torch.full_like(T, K-1), T).long()  # [B]
        idx = torch.arange(K, device=pi.device)[None, :]
        truncation_mask = idx <= T[:, None]  # [B, K]

        # Zero out unused components
        pi_truncated = pi_sorted * truncation_mask.float()

        # Restore original order
        leftover =1.0 - pi_truncated.sum(dim=-1, keepdim=True)
        pi_truncated.scatter_add_(1, T[:, None], leftover)
        unsort_idx = torch.argsort(sort_idx, dim=-1)
        pi_final = torch.gather(pi_truncated, 1, unsort_idx)
        assert torch.isfinite(pi_final).all()
        assert ((pi_final.sum(dim=-1) - 1).abs() < 1e-6).all()
        assert (pi_final >= 0).all()

        return pi_final, {
            'kumar_a': torch.exp(log_kumar_a),
            'kumar_b': torch.exp(log_kumar_b),
            "pi_raw": pi,
            "pi_final": pi_final,
            'v': v,
            'perm':perm,
            'active_components': truncation_mask.sum(dim=-1).float().mean(),  # Count active components
        }

    @staticmethod
    def compute_kumar2beta_kl( a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, n_approx: int, eps: float=100*torch.finfo(torch.float32).eps) -> torch.Tensor:
        """
        Compute KL divergence between Kumaraswamy(a,b) and Beta(alpha,beta)
        https://arxiv.org/pdf/1905.12052
        """
        EULER_GAMMA = 0.5772156649
        assert a.shape == b.shape == alpha.shape == beta.shape

        a = torch.clamp(a, min=eps, max=20)  # Ensure a is positive
        b = torch.clamp(b, min=eps, max=20)  # Ensure b is positive
        alpha = torch.clamp(alpha, min=eps)
        beta = torch.clamp(beta, min=eps)

        ab = torch.mul(a, b)
        a_inv = torch.reciprocal(a + eps)
        b_inv = torch.reciprocal(b + eps)

        # Taylor expansion for E[log(1-v)]

        log_taylor = torch.logsumexp(torch.stack([beta_fn(m *a_inv, b) - torch.log(m + ab) for m in range(1, n_approx + 1)], dim=-1), dim=-1)
        kl = torch.mul(torch.mul(beta - 1, b), torch.exp(log_taylor))
        # Add remaining terms
        psi_b = torch.digamma(b + eps)
        term1 = torch.mul(torch.div(a - alpha, a + eps), -EULER_GAMMA - psi_b - b_inv)
        term2 = torch.log(ab + eps) + beta_fn(alpha, beta)
        term2 = term2 + torch.div(-(b - 1), b + eps)
        kl = kl + term1 + term2
        if torch.any(kl > 1000):
            print(f"WARNING: Large Kumar-Beta KL detected: max={kl.max().item()}, mean={kl.mean().item()}")
            print(f"  a range: ({a.min().item()}, {a.max().item()})")
            print(f"  b range: ({b.min().item()}, {b.max().item()})")
            print(f"  alpha range: ({alpha.min().item()}, {alpha.max().item()})")
            print(f"  beta range: ({beta.min().item()}, {beta.max().item()})")

        return torch.clamp(kl, min=0.0)

    def compute_gamma2gamma_kl(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior over alpha
        """
        prior_concentration = self.gamma_a
        prior_rate = self.gamma_b
        return self.alpha_posterior.kl_divergence(h, prior_concentration, prior_rate)

class ComponentNN(nn.Module):
    def __init__(self, hidden_dim, latent_dim, max_components):
        super().__init__()
        inner_dim = hidden_dim * 4

        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.ln1 = nn.LayerNorm(inner_dim)

        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.out = nn.Linear(hidden_dim, 2 * latent_dim * max_components)
        nn.init.zeros_(self.out.bias)
        nn.init.normal_(self.out.weight, std=1e-3)

    def forward(self, x):
        residual = x                          # [B, hidden_dim]
        h = self.fc1(x)
        h = self.ln1(h)
        h = F.silu(h)
        h = self.fc2(h)
        h = self.ln2(h)
        h = F.silu(h)
        h = h + residual                      # true residual
        return self.out(h)                    # [B, 2 * latent_dim * K]

class DPGMMPrior(nn.Module):
    """
    Image-level Dirichlet-Process Gaussian-mixture prior for the top VDVAE latent.

    The top latent is treated as one random variable per image with shape
    [B, C * Ht * Wt]. Spatial structure from the recurrent hidden map is encoded
    with a CLS-transformer over spatial tokens, but the mixture itself is defined
    over the full flattened top latent map.
    """
    def __init__(
        self,
        max_components: int,
        latent_dim: int,
        hidden_dim: int,
        device: torch.device,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        use_checkpoint: bool = False,
        ctx_num_heads: int = 4,
        ctx_num_layers: int = 2,
        ctx_ff_mult: int = 1,
        ctx_dropout: float = 0.0,
        ctx_hw: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.max_K = int(max_components)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.device = device
        self.use_checkpoint = bool(use_checkpoint)

        self.stick_breaking = AdaptiveStickBreaking(
            max_components,
            hidden_dim,
            device,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
        )

        self.component_nn = ComponentNN(hidden_dim, latent_dim, max_components)
        self.ctx_hw = tuple(ctx_hw) if ctx_hw is not None else None
        if self.ctx_hw is None:
            raise ValueError("ctx_hw must be provided for DPGMMPrior to encode spatial context  information.")
        if hidden_dim % 4 != 0:
            raise ValueError(
                "For 2D RoPE here, hidden_dim should be divisible by 4."
            )
        self.ctx_rope = RoPE((self.ctx_hw[0], self.ctx_hw[1], hidden_dim))
        heads = min(int(ctx_num_heads), hidden_dim)
        while heads > 1 and (hidden_dim % heads != 0):
            heads -= 1

        self.ctx_cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.ctx_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=int(ctx_ff_mult * hidden_dim),
                dropout=float(ctx_dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(int(ctx_num_layers))
        ])
        self.ctx_norm = nn.LayerNorm(hidden_dim)

        self.to(device)

    def _encode_context_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, T, Hc]
        returns: [B, Hc]
        """
        B, T, C = tokens.shape
        if C != self.hidden_dim:
            raise ValueError(
                f"context channel dim {C} does not match hidden_dim={self.hidden_dim}"
            )

        cls = self.ctx_cls.expand(B, 1, C)
        x = torch.cat([cls, tokens], dim=1)

        for layer in self.ctx_layers:
            if self.use_checkpoint and self.training and x.requires_grad:
                x = ckpt(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        return self.ctx_norm(x[:, 0])

    def encode_context_map(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 2:
            if h.shape[1] != self.hidden_dim:
                raise ValueError(
                    f"h dim {h.shape[1]} does not match hidden_dim={self.hidden_dim}"
                )
            return h

        if h.dim() != 4:
            raise ValueError(
                f"h must be [B,Hc] or [B,Hc,Ht,Wt], got shape={tuple(h.shape)}"
            )

        B, Hc, Ht, Wt = h.shape
        if Hc != self.hidden_dim:
            raise ValueError(
                f"h channel dim {Hc} does not match hidden_dim={self.hidden_dim}"
            )

        x = h.permute(0, 2, 3, 1).contiguous()   # [B, Ht, Wt, Hc]

        if self.ctx_rope is not None:
            if (Ht, Wt) != self.ctx_hw:
                raise ValueError(
                    f"Got context map {(Ht, Wt)} but ctx_rope was built for {self.ctx_hw}"
                )
            x = self.ctx_rope(x)

        tokens = x.view(B, Ht * Wt, Hc).contiguous()
        return self._encode_context_tokens(tokens)

    def _normalize_pi(self, pi: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(pi.dtype).eps
        pi = pi.clamp_min(eps)
        return pi / pi.sum(dim=-1, keepdim=True)

    def build_mixture_distribution(self, prior_params: Dict[str, torch.Tensor]) -> MixtureSameFamily:
        """
        Build p(z_top | h) where z_top is one full image-level top latent vector [D].
        """
        pi = self._normalize_pi(prior_params["pi"])
        std = torch.exp(0.5 * prior_params["log_vars"]).clamp_min(torch.finfo(pi.dtype).eps)

        mix = Categorical(probs=pi)
        comp = Independent(Normal(prior_params["means"], std), 1)
        return MixtureSameFamily(mix, comp)

    def sample_image_latent(
        self,
        prior_params: Dict[str, torch.Tensor],
        temperature: float = 1.0,
        grad: bool = False,
        return_stats: bool = False,
    ):
        """
        Sample one full top latent vector per image from the image-level mixture.

        When grad=True, uses a relaxed Gumbel-Softmax component selection and a
        differentiable convex combination of per-component Gaussian samples.
        """
        pi = self._normalize_pi(prior_params["pi"])
        means = prior_params["means"]
        log_vars = prior_params["log_vars"]
        std = torch.exp(0.5 * log_vars).clamp_min(torch.finfo(means.dtype).eps)

        logits = torch.log(pi)
        tau = max(float(temperature), 1e-4)

        if grad:
            y = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)   # [B,K]
            eps_z = torch.randn_like(means)
            comp_samples = means + eps_z * std                          # [B,K,D]
            z_img = (y.unsqueeze(-1) * comp_samples).sum(dim=1)         # [B,D]

            mu_img = (y.unsqueeze(-1) * means).sum(dim=1)
            second_moment = (y.unsqueeze(-1) * (std.pow(2) + means.pow(2))).sum(dim=1)
            var_img = (second_moment - mu_img.pow(2)).clamp_min(torch.finfo(means.dtype).eps)
            component_probs = y
        else:
            if temperature != 1.0:
                logits = logits / tau
            probs = F.softmax(logits, dim=-1)
            idx = Categorical(probs=probs).sample()

            batch_idx = torch.arange(means.shape[0], device=means.device)
            mu_img = means[batch_idx, idx]
            std_img = std[batch_idx, idx]
            z_img = mu_img + torch.randn_like(mu_img) * std_img
            var_img = std_img.pow(2)
            component_probs = probs

        if return_stats:
            return z_img, {
                "component_probs": component_probs,
                "mean": mu_img,
                "var": var_img,
            }
        return z_img

    def compute_kl_divergence_mc(
        self,
        posterior_mean: torch.Tensor,
        posterior_logvar: torch.Tensor,
        prior_params: Dict[str, torch.Tensor],
        n_samples: int = 10,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Monte-Carlo estimate of KL(q(z|x) || p(z|h)) for full image-level top latents.

        posterior_mean/logvar: [B, D]
        prior_params:
            pi       [B, K]
            means    [B, K, D]
            log_vars [B, K, D]
        """
        if posterior_mean.dim() != 2 or posterior_logvar.dim() != 2:
            raise ValueError(
                f"posterior tensors must be [B,D], got {tuple(posterior_mean.shape)} and {tuple(posterior_logvar.shape)}"
            )

        B, D = posterior_mean.shape
        if posterior_logvar.shape != (B, D):
            raise ValueError(
                f"posterior_logvar shape {tuple(posterior_logvar.shape)} does not match posterior_mean {tuple(posterior_mean.shape)}"
            )

        pi = self._normalize_pi(prior_params["pi"])
        means = prior_params["means"]
        log_vars = prior_params["log_vars"]

        if means.shape[:2] != (B, pi.shape[-1]) or means.shape[-1] != D:
            raise ValueError(
                f"prior means shape {tuple(means.shape)} incompatible with posterior shape {(B, D)}"
            )
        if log_vars.shape != means.shape:
            raise ValueError(
                f"log_vars shape {tuple(log_vars.shape)} must match means shape {tuple(means.shape)}"
            )

        eps = torch.finfo(posterior_mean.dtype).eps
        noise = torch.randn(
            n_samples, B, D,
            device=posterior_mean.device,
            dtype=posterior_mean.dtype,
        )
        z = posterior_mean.unsqueeze(0) + noise * torch.exp(0.5 * posterior_logvar).unsqueeze(0)

        log_q = -0.5 * (
            D * math.log(2.0 * math.pi)
            + posterior_logvar.unsqueeze(0).sum(dim=-1)
            + (((z - posterior_mean.unsqueeze(0)) ** 2) * torch.exp(-posterior_logvar.unsqueeze(0))).sum(dim=-1)
        )

        log_component = -0.5 * (
            D * math.log(2.0 * math.pi)
            + log_vars.sum(dim=-1).unsqueeze(0)
            + (((z.unsqueeze(2) - means.unsqueeze(0)) ** 2) * torch.exp(-log_vars).unsqueeze(0)).sum(dim=-1)
        )
        log_p = torch.logsumexp(log_component + torch.log(pi.clamp_min(eps)).unsqueeze(0), dim=-1)

        kl_img = (log_q - log_p).mean(dim=0)
        if reduction in ("none", "image"):
            return kl_img
        if reduction == "mean":
            return kl_img.mean()
        raise ValueError(f"Unknown reduction={reduction}")

    def compute_kl_loss(self, params: Dict, alpha: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        KL for the stick-breaking variational factors.

        This term lives on the mixture process, not on image pixels, so keep it as
        its own regularizer instead of folding it into the per-pixel top-latent KL.
        We average the Kumaraswamy contribution over sticks so its scale does not
        grow with truncation level.
        """
        kl_terms = []
        alpha_scalar = alpha.mean(dim=1, keepdim=True) if alpha.dim() > 1 else alpha.unsqueeze(-1)

        for k in range(self.max_K - 1):
            kumar_a_k = params["kumar_a"][:, k:k+1]
            kumar_b_k = params["kumar_b"][:, k:k+1]

            beta_alpha = torch.ones_like(alpha_scalar).squeeze(-1)
            beta_beta = alpha_scalar.squeeze(-1)

            kl_k = self.stick_breaking.compute_kumar2beta_kl(
                kumar_a_k,
                kumar_b_k,
                beta_alpha,
                beta_beta,
                self.stick_breaking.M,
            ).squeeze(-1)
            kl_terms.append(kl_k)

        if len(kl_terms) > 0:
            kumar_kl = torch.stack(kl_terms, dim=-1).mean(dim=-1).mean()
        else:
            kumar_kl = torch.zeros((), device=h.device, dtype=h.dtype)

        alpha_kl = self.stick_breaking.compute_gamma2gamma_kl(h)
        return kumar_kl + alpha_kl

    def forward(
        self,
        h: torch.Tensor,
        n_samples: int = 10,
    ) -> Tuple[torch.distributions.Distribution, Dict]:
        """
        Build an image-level DP-GMM prior over the full flattened top latent map.
        """
        h_img = self.encode_context_map(h)
        batch_size = h_img.shape[0]

        pi, kumar_params = self.stick_breaking(h_img, use_rand_perm=False)
        pi = self._normalize_pi(pi)
        alpha = self.stick_breaking.alpha_posterior.sample(h_img, n_samples)

        params = self.component_nn(h_img)
        means, log_vars = torch.split(params, self.latent_dim * self.max_K, dim=1)
        log_vars = torch.clamp(log_vars, min=-10.0, max=1.0)

        means = means.view(batch_size, self.max_K, self.latent_dim)
        log_vars = log_vars.view(batch_size, self.max_K, self.latent_dim)

        prior_params = {
            "pi": pi,
            "alpha": alpha,
            "means": means,
            "log_vars": log_vars,
            "h_img": h_img,
            **kumar_params,
        }
        mix = self.build_mixture_distribution(prior_params)
        return mix, prior_params

    def get_effective_components(self, pi: torch.Tensor, threshold: float = 1e-3) -> torch.Tensor:
        sorted_pi, _ = torch.sort(self._normalize_pi(pi), descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_pi, dim=-1)
        return torch.sum(cumsum < (1.0 - threshold), dim=-1) + 1

    def compute_responsibilities(
        self,
        z_img: torch.Tensor,                 # [B, D]
        prior_params: dict,                  # pi:[B,K], means/log_vars:[B,K,D]
        eps: float = 1e-8,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Image-level posterior responsibilities r_{bk} ∝ pi_{bk} N(z_b | mu_{bk}, diag(var_{bk})).
        Returns:
          resp: [B, K]
        """
        if z_img.dim() != 2:
            raise ValueError(f"z_img must be [B,D], got {tuple(z_img.shape)}")

        B, D = z_img.shape
        pi = self._normalize_pi(prior_params["pi"])
        means = prior_params["means"]
        log_vars = prior_params["log_vars"]

        if means.shape[:2] != (B, pi.shape[-1]) or means.shape[-1] != D:
            raise ValueError(
                f"prior means shape {tuple(means.shape)} incompatible with z_img shape {tuple(z_img.shape)}"
            )
        if log_vars.shape != means.shape:
            raise ValueError(
                f"log_vars shape {tuple(log_vars.shape)} must match means shape {tuple(means.shape)}"
            )

        inv_var = torch.exp(-log_vars).clamp_max(1.0 / eps)
        diff2 = (z_img[:, None, :] - means).pow(2)
        maha = (diff2 * inv_var).sum(dim=-1)
        log_det = log_vars.sum(dim=-1)
        log_gauss = -0.5 * (D * math.log(2.0 * math.pi) + log_det + maha)
        logits = (torch.log(pi.clamp_min(eps)) + log_gauss) / max(float(temperature), eps)
        return torch.softmax(logits, dim=-1)


@contextmanager
def apply_emas(*emas):
    for e in emas: e.apply_shadow()
    try:
        yield
    finally:
        for e in reversed(emas): e.restore()


################### Main DPGMMVRNN Class ###################

class DPGMMVariationalRecurrentAutoencoder(nn.Module):
    """
    Using Dirichlet Process GMM Prior with Stick-Breaking, Flow-based Warping, and adversarial training for video prediction.
    This architecture incorporates adaptive temporal dynamics, and attention mechanisms
    """
    def __init__(
        self,
        max_components: int,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        action_dim: int,
        sequence_length: int,
        img_disc_layers:int,
        disc_num_heads:int,
        device: torch.device= torch.device('cuda'),
        patch_size: int = 16,
        input_channels: int = 3,  # Number of input channels (e.g., RGB images)
        learning_rate: float = 1e-5,
        grad_clip:float = 10.0,
        prior_alpha: float = 1.0,  # Add these parameters
        prior_beta: float = 10.0,
        weight_decay: float = 0.00001,
        warmup_epochs: int = 25,
        dropout: float = 0.1,
        use_ctx_checkpoint: bool = True,
        use_dwa: bool = False,
        dwa_temperature: float = 2.0,
        rollout_adv_every: int = 1,            # do rollout adversarial every N steps (0 disables)
        rollout_context_frames: int = 3,        # T_ctx
        rollout_horizon: int = 4,               # rollout length
        lambda_rollout_adv: float = 1.0,       # strength of rollout adversarial losses
        rollout_top_temperature: float = 0.2,   # sampling temperature for top prior
        rollout_decoder_temperature: float = 0.2,
        patch_disc_layers: int = 4,
        patch_disc_ndf:int = 32,
        latent_transport_hidden_features: tuple[int, ...] = (128, 128, 256),
        latent_transport_hidden_layers: int = 2,
        latent_transport_flow_scale: float = 1.0,
        lecam_ema_decay: float = 0.99,    # EMA decay for LeCam regularization anchors
    ):
        super().__init__()
        # core dimensions
        self.input_channels = input_channels
        self.image_size = input_dim
        self.max_K = max_components
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.device = device
        self.dropout = dropout
        self.latent_transport_hidden_features = latent_transport_hidden_features
        self.latent_transport_hidden_layers = latent_transport_hidden_layers
        self.latent_transport_flow_scale = latent_transport_flow_scale
        
        # Hyperparameters
        self._lr = learning_rate
        self._grad_clip = grad_clip
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.use_dwa = use_dwa
        self.use_ctx_checkpoint = use_ctx_checkpoint
        self.patch_disc_layers=patch_disc_layers
        self.patch_disc_ndf = patch_disc_ndf
        self.eps = torch.finfo(torch.float32).eps
        # KL overshooting
        self.overshoot_mc_samples = 35
        self.overshoot_w_decay = 0.9
        self.lambda_overshoot = 1.0
        # rollout GAN attributes
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long), persistent=True)

        self.rollout_adv_every = int(rollout_adv_every)
        self.rollout_context_frames = int(rollout_context_frames)
        self.rollout_horizon = int(rollout_horizon)
        self.lambda_rollout_adv = float(lambda_rollout_adv)
        self.rollout_top_temperature = float(rollout_top_temperature)
        self.rollout_decoder_temperature = float(rollout_decoder_temperature)
  
        self.lecam_ema_decay = float(lecam_ema_decay)

        # scalar EMA anchors for LeCam regularization
        self.register_buffer("lecam_initialized", torch.tensor(False), persistent=True)

        self.register_buffer("lecam_temporal_real_ema", torch.zeros((), dtype=torch.float32), persistent=True)
        self.register_buffer("lecam_temporal_fake_ema", torch.zeros((), dtype=torch.float32), persistent=True)

        self.register_buffer("lecam_patch_real_ema", torch.zeros((), dtype=torch.float32), persistent=True)
        self.register_buffer("lecam_patch_fake_ema", torch.zeros((), dtype=torch.float32), persistent=True)

        # initialization different parts of the model
        self._init_encoder_decoder(max_components, prior_alpha, prior_beta)

        # default: allow up to one full top-grid width/height in pixels
        self._init_vrnn_dynamics( extra_channels=0)
        self._init_discriminators(img_disc_layers, patch_size, num_heads=disc_num_heads)


        if use_dwa:
            self._init_DynamicWeightAverage(dwa_temperature)

        self.to(device)

        # Setup optimizers
        self._setup_optimizers(learning_rate, weight_decay)

    @torch.no_grad()
    def _update_lecam_ema(
        self,
        temporal_real_mean: torch.Tensor,
        temporal_fake_mean: torch.Tensor,
        patch_real_mean: torch.Tensor,
        patch_fake_mean: torch.Tensor,
    ) -> None:
        tr = temporal_real_mean.detach().to(
            device=self.lecam_temporal_real_ema.device,
            dtype=self.lecam_temporal_real_ema.dtype,
        )
        tf = temporal_fake_mean.detach().to(
            device=self.lecam_temporal_fake_ema.device,
            dtype=self.lecam_temporal_fake_ema.dtype,
        )
        pr = patch_real_mean.detach().to(
            device=self.lecam_patch_real_ema.device,
            dtype=self.lecam_patch_real_ema.dtype,
        )
        pf = patch_fake_mean.detach().to(
            device=self.lecam_patch_fake_ema.device,
            dtype=self.lecam_patch_fake_ema.dtype,
        )

        if not bool(self.lecam_initialized):
            self.lecam_temporal_real_ema.copy_(tr)
            self.lecam_temporal_fake_ema.copy_(tf)
            self.lecam_patch_real_ema.copy_(pr)
            self.lecam_patch_fake_ema.copy_(pf)
            self.lecam_initialized.fill_(True)
            return

        d = self.lecam_ema_decay

        self.lecam_temporal_real_ema.mul_(d).add_(tr, alpha=1.0 - d)
        self.lecam_temporal_fake_ema.mul_(d).add_(tf, alpha=1.0 - d)
        self.lecam_patch_real_ema.mul_(d).add_(pr, alpha=1.0 - d)
        self.lecam_patch_fake_ema.mul_(d).add_(pf, alpha=1.0 - d)

    def _init_DynamicWeightAverage(self, temperature: float = 2.0):
        self.total_weighter = DynamicWeightAverage(
            loss_keys_to_consider=[
                "recon_loss",
                "kl_z",
                "hierarchical_kl",
                "component_margin",
                "img_adv_loss",
                "temporal_adv_loss",
                "feat_match_loss",
                "rollout_img_adv_loss",
                "rollout_temporal_adv_loss",
                "rollout_feat_match_loss",
                "overshoot_kl",
            ],
            temperature=temperature,
        )

    @staticmethod
    def to01(x: torch.Tensor) -> torch.Tensor:
        # Commonly observations are in [-1, 1]
        return (x * 0.5 + 0.5).clamp(0.0, 1.0)


    @staticmethod
    def _length_sq(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x * x, dim=1, keepdim=True)


    def _maybe_ckpt(self, fn, *args):
        if (
            self.use_ctx_checkpoint
            and self.training
            and any(torch.is_tensor(arg) and arg.requires_grad for arg in args)
        ):
            return ckpt(fn, *args, use_reentrant=False)
        return fn(*args)


    def _init_encoder_decoder(self, max_components: int, prior_alpha: float, prior_beta: float, prior_mc_samples: int = 80):
        """
        Initialize VDVAE + DPGMM prior.

        """
        # 1) Build VDVAE hyperparams 
        H = Hyperparams()
        # --- Temporal priors (ConvLSTM-conditioned) ---
        H.action_dim = self.action_dim      # required by the ConvLSTM temporal prior

        H.use_checkpoint = self.use_ctx_checkpoint
        H.image_channels = self.input_channels   # usually 3
        H.zdim = self.latent_dim                 # or set explicitly (e.g., 16)
        H.bottleneck_multiple = 0.25
        H.width = 96
        H.image_size = self.image_size          # e.g. 64
        H.dataset = 'imagenet64'
        H.num_mixtures = 10
        H.skip_threshold = 100.0
        H.enc_blocks = "64x2,64d2,32x3,32d2,16x3,16d2,8x3"   # +2 blocks at 8
        H.dec_blocks = "8x3,16m8,16x3,32m16,32x3,64m32,64x2"
        H.attn_resolutions = [32,16]
        H.use_spatial_attn = True
        H.attn_where = "last"
        H.top_h_context_dim = self.hidden_dim

        H.no_bias_above = 64
        H.custom_width_str = ""
        # --- Attention defaults ---
        H.attn_num_layers = 1
        H.attn_num_heads = 8
        H.attn_widening_factor = 1
        H.attn_dropout = 0.0
        H.attn_residual_dropout = 0.0
        H.attn_gn_groups = 32
        H.attn_pos_num_bands = 6
        H.overshoot_K = 8

        # ---- 2) Instantiate VDVAE (no prior yet) ----
        self.vdvae = VDVAE(
            H,
            prior=None,              # we'll set self.prior separately
            top_kl_weight=2.0,
            prior_kl_mc_samples=prior_mc_samples,
        ).to(self.device)
        # ---- 3) Extract top block latent dim & resolution ----
        top_block = self.vdvae.decoder.dec_blocks[0]        # is_top=True for first block
        C = top_block.zdim                                  # latent channels
        res = top_block.base      
        self.zdim = C                           # spatial resolution (e.g. 8)
        self.top_zdim = C * res * res
        
        self.top_H = res #TODO:Is this correct?
        self.top_W = res
        # ---- 4) Build DPGMM prior over flattened top-layer tokens ----
        self.prior = DPGMMPrior(
            max_components=max_components,
            latent_dim= self.top_zdim,
            hidden_dim=self.hidden_dim + self.zdim,      # VRNN hidden state dimension
            device=self.device,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            use_checkpoint=self.use_ctx_checkpoint,
            ctx_hw=(self.top_H, self.top_W),
        )
        # EMA for VDVAE
        self.ema_decay = 0.999
        self.ema_vdvae = EMA(self.vdvae, decay=self.ema_decay)

        # Attach prior to VDVAE so its forward() computes dp_kl / dp_rate
        self.vdvae.prior = self.prior
        self.overshoot_K = int(getattr(H, "overshoot_K", 8))
        self.latent_transport = LatentTransportINR(
            z_channels=self.zdim,
            h_channels=int(self.vdvae.H.top_h_context_dim),
            action_dim=self.action_dim,
            hidden_features=list(self.latent_transport_hidden_features),
            hidden_layers=int(self.latent_transport_hidden_layers),
            flow_scale=float(self.latent_transport_flow_scale),
        ).to(self.device)

    def _init_decoder_prev_latents(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Per decoder block index.
        Top block gets None because it does not use latent_prev.
        Non-top blocks get zero tensors [B, zdim, res, res].
        """
        prev_latents = []
        for idx, block in enumerate(self.vdvae.decoder.dec_blocks):
            if idx == 0:
                prev_latents.append(None)
            else:
                prev_latents.append(
                    torch.zeros(
                        batch_size,
                        block.zdim,
                        block.base,
                        block.base,
                        device=device,
                        dtype=dtype,
                    )
                )
        return prev_latents


    def _init_discriminators(self, img_disc_layers:int, patch_size: int, num_heads: int = 4):
        # Initialize discriminators
        self.image_discriminator = TemporalDiscriminator(
            input_channels=self.input_channels,
            cond_channels=self.zdim+self.hidden_dim,
            num_layers=int(img_disc_layers),
            conv_type="standard",
            attn_dim=self.hidden_dim,
            num_heads=num_heads,
            use_checkpoint=False,
            ckpt_use_reentrant=False,
        ).to(self.device)
        self.patch_discriminator = ImageDiscriminator(
                input_nc=self.input_channels,
                ndf=int(self.patch_disc_ndf),
                n_layers=int(self.patch_disc_layers),
                norm_type= "group",
                gn_groups= 32,
                use_checkpoint=False,
                checkpoint_use_reentrant=False,
                device=self.device,
            )


    def _init_vrnn_dynamics(self, extra_channels: int =0):
        """Initialize VRNN components with context conditioning"""
        # Feature extractors

        # VRNN recurrence: h_t = f(h_{t-1}, z_t, a_t)
        self._rnn = SpatioTemporalCore(
            zdim=self.zdim,                # C, not C*H*W
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,    # correct keyword
            height=self.top_H,
            width=self.top_W,
            kernel=5,
            use_checkpoint=self.use_ctx_checkpoint,
            extra_channels=extra_channels
        )
        
        
    @property
    def rnn(self):
        """Property to access the RNN layer."""
        return self._rnn

    def _setup_optimizers(self, learning_rate: float, weight_decay: float) -> None:
        def collect_unique_params(*items, exclude_params=None):
            """
            Collect trainable params exactly once from:
            - nn.Module
            - nn.Parameter
            - nested list/tuple/set
            """
            params = []
            seen_ids = set()
            exclude_ids = {id(p) for p in (exclude_params or []) if isinstance(p, nn.Parameter)}

            def visit(x):
                if x is None:
                    return

                if isinstance(x, nn.Parameter):
                    if x.requires_grad and id(x) not in exclude_ids and id(x) not in seen_ids:
                        params.append(x)
                        seen_ids.add(id(x))
                    return

                if isinstance(x, nn.Module):
                    for p in x.parameters():
                        if p.requires_grad and id(p) not in exclude_ids and id(p) not in seen_ids:
                            params.append(p)
                            seen_ids.add(id(p))
                    return

                if isinstance(x, (list, tuple, set)):
                    for y in x:
                        visit(y)
                    return

                raise TypeError(f"Unsupported param container type: {type(x)}")

            for item in items:
                visit(item)

            return params

        def split_by_weight_decay(params, wd):
            decay, no_decay = [], []
            for p in params:
                if not p.requires_grad:
                    continue
                if p.ndim >= 2:
                    decay.append(p)
                else:
                    no_decay.append(p)

            groups = []
            if decay:
                groups.append({"params": decay, "weight_decay": wd})
            if no_decay:
                groups.append({"params": no_decay, "weight_decay": 0.0})
            return groups

        def add_groups(dst, params, lr, wd):
            if not params:
                return
            for g in split_by_weight_decay(params, wd):
                g["lr"] = lr
                dst.append(g)

        # -------------------------
        # LR / WD policy
        # -------------------------
        base_lr = float(learning_rate)

        lr_core  = base_lr * 1.0
        lr_prior = base_lr * 1.0
        lr_inr   = base_lr * 2.0
        lr_gamma = base_lr * 5e-5   # small but not frozen
        lr_state = base_lr * 1.0

        wd_core  = float(weight_decay)
        wd_prior = float(weight_decay)
        wd_inr   = float(weight_decay) * 0.1

        # -------------------------
        # Special params first
        # -------------------------
        gamma_params = [
            getattr(self.prior.stick_breaking, "gamma_a", None),
            getattr(self.prior.stick_breaking, "gamma_b", None),
        ]
        gamma_params = [p for p in gamma_params if isinstance(p, nn.Parameter) and p.requires_grad]

        scalar_state_params = [
            getattr(self.rnn, "h0", None),
            getattr(self.rnn, "c0", None),
            getattr(self.rnn, "m0", None),
        ]
        scalar_state_params = [p for p in scalar_state_params if isinstance(p, nn.Parameter) and p.requires_grad]

        special_exclude = gamma_params + scalar_state_params

        # -------------------------
        # Main generator buckets
        # -------------------------
        prior_params = collect_unique_params(self.prior, exclude_params=special_exclude)

        # since self.vdvae.prior = self.prior, exclude ALL prior params from core vdvae group
        prior_and_special_exclude = prior_params + special_exclude

        core_params = collect_unique_params(
            self.vdvae,
            self.rnn,
            exclude_params=prior_and_special_exclude,
        )

        inr_params = collect_unique_params(
            getattr(self, "latent_transport", None),
            exclude_params=special_exclude,
        )

        gen_param_groups = []
        add_groups(gen_param_groups, core_params,  lr_core,  wd_core)
        add_groups(gen_param_groups, prior_params, lr_prior, wd_prior)
        add_groups(gen_param_groups, inr_params,   lr_inr,   wd_inr)

        if gamma_params:
            gen_param_groups.append({
                "params": gamma_params,
                "lr": lr_gamma,
                "weight_decay": 0.0,
            })

        if scalar_state_params:
            gen_param_groups.append({
                "params": scalar_state_params,
                "lr": lr_state,
                "weight_decay": 0.0,
            })

        self.gen_optimizer = torch.optim.AdamW(
            gen_param_groups,
            lr=lr_core,
            betas=(0.9, 0.999),
            eps=1e-5,
        )

        # -------------------------
        # Discriminator optimizer
        # -------------------------
        disc_param_groups = []

        if hasattr(self, "image_discriminator") and self.image_discriminator is not None:
            disc_param_groups.append({
                "params": [p for p in self.image_discriminator.parameters() if p.requires_grad],
                "lr": base_lr * 0.5,
            })

        if hasattr(self, "patch_discriminator") and self.patch_discriminator is not None:
            disc_param_groups.append({
                "params": [p for p in self.patch_discriminator.parameters() if p.requires_grad],
                "lr": base_lr * 0.07,
            })

        # flatten away any empty groups
        disc_param_groups = [g for g in disc_param_groups if len(g["params"]) > 0]

        self.img_disc_optimizer = None
        if disc_param_groups:
            self.img_disc_optimizer = torch.optim.Adamax(
                disc_param_groups,
                betas=(0.0, 0.9),
                weight_decay=5e-5,
            )

        gen_ids = [id(p) for g in self.gen_optimizer.param_groups for p in g["params"]]
        if len(gen_ids) != len(set(gen_ids)):
            raise RuntimeError("Duplicate parameter detected inside gen_optimizer.")

        disc_ids = []
        if self.img_disc_optimizer is not None:
            disc_ids = [id(p) for g in self.img_disc_optimizer.param_groups for p in g["params"]]
            if len(disc_ids) != len(set(disc_ids)):
                raise RuntimeError("Duplicate parameter detected inside img_disc_optimizer.")

        overlap = set(gen_ids).intersection(disc_ids)
        if overlap:
            raise RuntimeError("Some parameters are present in both generator and discriminator optimizers.")

        # expected generator params = all trainable params in self minus discriminator params
        all_trainable_ids = {id(p) for p in self.parameters() if p.requires_grad}
        expected_disc_ids = set(disc_ids)
        expected_gen_ids = all_trainable_ids - expected_disc_ids

        if set(gen_ids) != expected_gen_ids:
            missing = expected_gen_ids - set(gen_ids)
            extra = set(gen_ids) - expected_gen_ids
            raise RuntimeError(
                f"Generator optimizer coverage mismatch: missing={len(missing)}, extra={len(extra)}"
            )

        self._setup_schedulers()


    def _setup_schedulers(self):
        """Setup learning-rate schedulers for all optimizers (Option B)."""
        # 1) Trunk scheduler (PCGrad or not, this uses the base optimizer)
        self.gen_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.gen_optimizer,
            mode='min',
            factor=0.2,
            patience=10,
            threshold=0.0001,
        )
        # 3) Discriminator scheduler (unchanged)
        if hasattr(self, "img_disc_optimizer"):
            self.img_disc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.img_disc_optimizer,
                mode="min",
                factor=0.5,
                patience=10,
                min_lr=1e-7,
            )

    def _lengths_from_dones(self, dones: torch.Tensor, T: int, assume_padded_after_done: bool = True):
        """
        dones: [B, T] where dones[t]=1 means episode terminates AFTER frame t (i.e., t is valid, t+1 invalid).
        returns:
        alive_mask: [B, T] bool (True = valid frame)
        lengths:    [B] long (#valid frames)
        """
        B = dones.shape[0]
        dones = dones[:, :T].bool()

        alive = torch.ones(B, T, device=dones.device, dtype=torch.bool)
        if assume_padded_after_done:
            # once done happens at time t, frames t+1, t+2, ... are invalid
            if T > 1:
                done_prev = dones[:, :T-1]                        # termination on transition to next frame
                ended_before = (done_prev.cumsum(dim=1) > 0)      # [B, T-1]
                alive[:, 1:] = ~ended_before

        lengths = alive.long().sum(dim=1)
        return alive, lengths


    def forward_sequence(
        self,
        observations,
        actions=None,
        dones=None,
        capture_flow_ctx=None,
        store_reconstruction_samples: bool = False,
    ):
        """
        observations: [B, T, C, H, W] in [-1, 1]
        actions:      [B, T, action_dim]
        dones:        [B, T] optional; expects done[t-1] to reset state at step t

        Returns a dict of per-timestep outputs.
        """
        # Unpack basic shapes and default action tensor
        batch_size, seq_len, C, H, W = observations.shape
        device = observations.device
        dtype = observations.dtype
        B = batch_size

        if actions is None:
            actions = torch.zeros(
                batch_size, seq_len, self.action_dim, device=device, dtype=dtype
            )

        # Initialize recurrent state and top-grid context
        core_state = self.rnn.init_state(batch_size, device=device, dtype=dtype)

        z_prev_top = torch.zeros(B, self.zdim, self.top_H, self.top_W, device=device, dtype=dtype)
        prev_latents = self._init_decoder_prev_latents(B, device, dtype)

        # Prepare output containers
        outputs = {
            "reconstructions": [],
            "reconstruction_samples": [] if store_reconstruction_samples else None,
            "latents": [],
            "hidden_states": [],              # post-RNN-update normalized hidden maps, flattened
            "core_h_maps": [],
            "core_c_maps": [],
            "core_m_maps": [],
            "prior_pi": [],
            "reconstruction_losses": [],
            "gauss_rate": [],
            "dp_rate": [],
            "kl_latents": [],
            "elbo": [],
            "top_q_mean_map": [],
            "top_q_logvar_map": [],
            "kumaraswamy_kl_losses": [],
            "K_eff": [],
            "component_margin": [],
            "z_tilde_seq": [],
            "flow_top_seq": [],
            "cond_top_seq": [],    
            "z_seq_maps": [],
            # overshoot anchor for the new transport dynamics
            "overshoot_z_prev_top": [],        
        }
        # Initialize previous top-posterior mean for LatentWarp
        # NOTE: channel dim must match the top latent map, i.e. self.zdim

        for t in range(seq_len):
            # Read the current observation x_t
            x_t = observations[:, t]
            #if t == 0: print(f"observation range: [{x_t.min().item():.4f}, {x_t.max().item():.4f}]")

            # Use previous action a_{t-1} for the current recurrent step
            a_t = (
                torch.zeros(batch_size, self.action_dim, device=device, dtype=dtype)
                if t == 0
                else actions[:, t - 1]
            )

            # Build keep-mask for step t; resets state if done at t-1
            if dones is None or t == 0:
                mask_t = torch.ones(batch_size, device=device, dtype=torch.float32)
            else:
                mask_t = (1.0 - dones[:, t - 1].float()).to(torch.float32)

            h, c, m = core_state
            keep = mask_t.view(B, 1, 1, 1).to(device=device, dtype=dtype)

            h0, c0, m0 = self.rnn.init_state(B, device=device, dtype=dtype)
            core_state = (
                h * keep + h0 * (1.0 - keep),
                c * keep + c0 * (1.0 - keep),
                m * keep + m0 * (1.0 - keep),
            )
            z_prev_top_masked = z_prev_top * keep
            prev_latents = [pl * keep if pl is not None else None for pl in prev_latents]
            h_t = self.rnn.out_norm(core_state[0])  # [B, hidden_dim, topH, topW]
            tr = self.latent_transport(
                z_top_prev=z_prev_top_masked,
                h_t=h_t,
                action_prev=a_t,
                dt=1.0,
            )

            # Encode the current frame with VDVAE conditioned on current context
            x_t_nhwc = x_t.permute(0, 2, 3, 1).contiguous()
            vdvae_out = self.vdvae.forward(
                x_t_nhwc,
                x_t_nhwc,
                cond_top=tr["cond_top"],
                h_decoder_top=tr["h_decoder_top"],  
                mask_t=mask_t,
                prev_latents=prev_latents,
                get_latents=True,
            )

            prior_params = vdvae_out["prior_params"]
            pi_img = prior_params["pi"]

            B2, zdim, Ht, Wt = vdvae_out["top_q_mean_map"].shape

            # Sample the current posterior top latent z_t
            z_top_map = draw_gaussian_diag_samples(vdvae_out["top_q_mean_map"], vdvae_out["top_q_logvar_map"])

            # Compute mixture diagnostics and DP/Kumaraswamy regularizers
            z_img = z_top_map.contiguous().view(B2, zdim * Ht * Wt)
            eps_ = torch.finfo(z_img.dtype).eps

            resp = self.prior.compute_responsibilities(
                z_img=z_img,
                prior_params=prior_params,
                eps=eps_,
                temperature=1.0,
            )

            u = resp.mean(dim=0).clamp_min(eps_)
            H_marg = -(u * u.log()).sum()
            r = resp.clamp_min(eps_)
            H_cond = -(r * r.log()).sum(dim=-1).mean()
            outputs["component_margin"].append(H_marg - H_cond)

            kumar_beta_kl = self.prior.compute_kl_loss(
                prior_params,
                prior_params["alpha"],
                prior_params["h_img"],
            )
            outputs["kumaraswamy_kl_losses"].append(kumar_beta_kl)

            K_eff = self.prior.get_effective_components(pi_img)
            outputs["K_eff"].append(K_eff.float().mean())

            # Flatten latent for discriminator conditioning
            z_flat = z_top_map.permute(0, 2, 3, 1).contiguous().view(B2, Ht * Wt * zdim)

            a_cur = actions[:, t] 
            # Update recurrent state using current z_t, previous action a_{t-1}, and motion-dependent extra maps
            h_context_next, core_state = self.rnn(
                z_top_map,
                a_cur,
                state=core_state,
                mask_t=None,
                extra_maps=None,
            )

            z_prev_top = z_top_map.detach()
            prev_latents = [None if idx == 0 else pl.detach() for idx, pl in enumerate(vdvae_out["current_latents"])]

            # Store recurrent states and latent statistics for this timestep
            outputs["core_h_maps"].append(core_state[0].detach())
            outputs["core_c_maps"].append(core_state[1].detach())
            outputs["core_m_maps"].append(core_state[2].detach())
            outputs["prior_pi"].append(pi_img.detach())
            outputs["top_q_mean_map"].append(vdvae_out["top_q_mean_map"].detach())
            outputs["top_q_logvar_map"].append(vdvae_out["top_q_logvar_map"].detach())
            outputs["z_tilde_seq"].append(tr["z_tilde"].detach())
            outputs["flow_top_seq"].append(tr["flow_top_px"].detach())
            outputs["cond_top_seq"].append(tr["cond_top"].detach())
            outputs["overshoot_z_prev_top"].append(z_prev_top)
            

            # Optionally store stochastic reconstruction samples
            if store_reconstruction_samples:
                with torch.no_grad():
                    sample = self.vdvae.decoder.out_net.sample(vdvae_out["px_z"])
                    sample = torch.from_numpy(sample).to(device=device, dtype=torch.float32)
                    sample = sample.permute(0, 3, 1, 2).contiguous()
                outputs["reconstruction_samples"].append(sample)

            # Decode reconstruction mean and log scalar VAE losses
            dmol_out = self.vdvae.decoder.out_net.forward(vdvae_out["px_z"])
            recon_mean = mean_from_discretized_mix_logistic(
                dmol_out, self.vdvae.H.num_mixtures
            )
            recon_mean = recon_mean.permute(0, 3, 1, 2).contiguous()

            outputs["reconstructions"].append(recon_mean)
            outputs["latents"].append(z_flat.detach())
            outputs["hidden_states"].append(
                h_context_next.permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, Ht * Wt * self.hidden_dim)
                .detach()
            )
            outputs["z_seq_maps"].append(torch.cat([z_top_map.detach(), h_context_next.detach()], dim=1))
            outputs["reconstruction_losses"].append(vdvae_out["distortion"])
            outputs["gauss_rate"].append(vdvae_out["gauss_rate"])
            outputs["dp_rate"].append(vdvae_out["dp_rate"])
            outputs["kl_latents"].append(vdvae_out["rate"])
            outputs["elbo"].append(vdvae_out["elbo"])

        # Stack time-major tensors where downstream code expects [B, T, ...]
        for k in [
            "core_h_maps",
            "core_c_maps",
            "core_m_maps",
            "z_tilde_seq",
            "flow_top_seq",
            "cond_top_seq",
            "overshoot_z_prev_top",
            "z_seq_maps",
        ]:
            if len(outputs[k]) > 0:
                outputs[k] = torch.stack(outputs[k], dim=1)

        for key in ["reconstructions", "latents", "hidden_states"]:
            if len(outputs[key]) > 0 and isinstance(outputs[key][0], torch.Tensor):
                outputs[key] = torch.stack(outputs[key], dim=1)

        if store_reconstruction_samples and len(outputs["reconstruction_samples"]) > 0:
            outputs["reconstruction_samples"] = torch.stack(outputs["reconstruction_samples"], dim=1)

        outputs["top_q_mean_maps"] = torch.stack(outputs.pop("top_q_mean_map"), dim=1)
        outputs["top_q_logvar_maps"] = torch.stack(outputs.pop("top_q_logvar_map"), dim=1)

        if outputs["K_eff"]:
            outputs["K_eff"] = torch.stack(outputs["K_eff"]).mean()
        return outputs


    def compute_lecam_loss(
        self,
        logits_real: torch.Tensor,
        logits_fake: torch.Tensor,
        ema_logits_real: torch.Tensor,
        ema_logits_fake: torch.Tensor
    ) -> torch.Tensor:
        """Computes the LeCam loss for the given average real and fake logits.

        Returns:
            lecam_loss -> torch.Tensor: The LeCam loss.
        """
        lecam_loss = torch.pow(F.relu(logits_real - ema_logits_fake), 2).mean()
        lecam_loss += torch.pow(F.relu(ema_logits_real - logits_fake), 2).mean()
        return lecam_loss

    def compute_multistep_kl_overshoot(
        self,
        top_q_mean_maps,      # [B,T,z,Ht,Wt]
        top_q_logsig_maps,    # [B,T,z,Ht,Wt]
        actions,              # [B,T,A]
        dones=None,
        K=15,
        mc_samples=50,
        w_decay=0.9,
        core_state_maps=None,          # (h,c,m): [B,T,hidden,Ht,Wt]
        overshoot_anchor_state=None,   # dict
        rollout_temperature: float = 1.0,
    ):
        device = top_q_mean_maps.device
        dtype = top_q_mean_maps.dtype
        B, T, zdim, Ht, Wt = top_q_mean_maps.shape

        if T < 2 or K <= 0:
            return torch.zeros((), device=device, dtype=dtype)

        if dones is not None:
            alive, _ = self._lengths_from_dones(dones, T, assume_padded_after_done=True)
        else:
            alive = torch.ones(B, T, device=device, dtype=torch.bool)

        if core_state_maps is None or overshoot_anchor_state is None:
            raise ValueError("Pass core_state_maps and overshoot_anchor_state.")

        core_h_seq, core_c_seq, core_m_seq = core_state_maps
        z_prev_seq = overshoot_anchor_state["z_prev_top"]

        A = T - 1
        anchors = torch.arange(A, device=device) #0, ..., T-2
        BA = B * A

        anchor_alive = alive[:, anchors]  # [B,A]

        h = core_h_seq[:, anchors].reshape(BA, *core_h_seq.shape[2:]).contiguous()
        c = core_c_seq[:, anchors].reshape(BA, *core_c_seq.shape[2:]).contiguous()
        m = core_m_seq[:, anchors].reshape(BA, *core_m_seq.shape[2:]).contiguous()
        z_prev_top = z_prev_seq[:, anchors].reshape(BA, *z_prev_seq.shape[2:]).contiguous()


        topH, topW = self.top_H, self.top_W

        numer = torch.zeros((), device=device, dtype=dtype)
        denom = torch.zeros((), device=device, dtype=dtype)

        for k in range(1, K + 1):
            t_idx = anchors + k
            in_range = (t_idx < T)
            if not in_range.any():
                break

            t_idx_safe = t_idx.clamp(max=T - 1)

            future_alive = torch.zeros(B, A, device=device, dtype=torch.bool)
            future_alive[:, in_range] = alive[:, t_idx[in_range]]

            valid_ba = (anchor_alive & future_alive).reshape(BA)
            if not valid_ba.any():
                continue

            mask_roll = valid_ba.to(dtype)

            keep = mask_roll.view(BA, 1, 1, 1).to(device=device, dtype=dtype)
            h0, c0, m0 = self.rnn.init_state(BA, device=device, dtype=dtype)
            h = h * keep + h0 * (1.0 - keep)
            c = c * keep + c0 * (1.0 - keep)
            m = m * keep + m0 * (1.0 - keep) 
            z_prev_top = z_prev_top * keep
            if actions is None:
                a_prev = torch.zeros(BA, self.action_dim, device=device, dtype=dtype)
            else:
                a_prev = actions[:, t_idx_safe - 1].reshape(BA, -1).contiguous()
                a_prev = a_prev * mask_roll.unsqueeze(-1)
            h_t = self.rnn.out_norm(h)  # [BA, hidden_dim, topH, topW]
            tr = self.latent_transport(
                z_top_prev=z_prev_top,
                h_t=h_t,
                action_prev=a_prev,
                dt=1.0,
            )

            _, prior_params = self.prior(tr["cond_top"])                       
            # 3) prior at this imagined step

            q_mu_map = top_q_mean_maps[:, t_idx_safe].reshape(BA, zdim, Ht, Wt).contiguous()
            q_lv_map = (2.0 * top_q_logsig_maps[:, t_idx_safe]).reshape(BA, zdim, Ht, Wt).contiguous()

            # Flatten to image-level latent vectors, exactly like the main VDVAE KL path
            q_mu_img = q_mu_map.view(BA, zdim * Ht * Wt)
            q_lv_img = q_lv_map.view(BA, zdim * Ht * Wt)

            kl_ba = self.prior.compute_kl_divergence_mc(
                posterior_mean=q_mu_img,
                posterior_logvar=q_lv_img,
                prior_params=prior_params,
                n_samples=mc_samples,
                reduction="image",
            )

            wk = torch.as_tensor(w_decay ** (k - 1), device=device, dtype=dtype)
            numer = numer + wk * (kl_ba * mask_roll).sum()
            denom = denom + wk * mask_roll.sum()

            # 4) imagined top sample
            z_img, _ = self.vdvae.prior.sample_image_latent(
                prior_params,
                temperature=rollout_temperature,
                grad=True,
                return_stats=True,
            )
            z_samp = z_img.view(BA, zdim, Ht, Wt).contiguous()

            # 5) latent 64 update, no decoder
            if actions is None:
                a_cur = torch.zeros(BA, self.action_dim, device=device, dtype=dtype)
            else:
                a_cur = actions[:, t_idx_safe].reshape(BA, -1).contiguous()
                can_step_fwd = (
                    (t_idx_safe < (T - 1))[None, :]
                    .expand(B, A)
                    .reshape(BA)
                    .to(dtype)
                )
                a_cur = a_cur * (mask_roll * can_step_fwd).unsqueeze(-1) #TODO: ???Is this correct?

            _, (h, c, m) = self.rnn(
                z_samp,
                a_cur,
                state=(h, c, m),
                mask_t=None,
                extra_maps=None,
            )
            z_prev_top = z_samp
   
        if denom <= 0:
            return torch.zeros((), device=device, dtype=dtype)
        return numer / denom

    def compute_total_loss(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
        beta: float = 1.0,
        lambda_recon: float = 1.0,
        store_reconstruction_samples: bool = False,
    ):
        outputs = self.forward_sequence(
            observations,
            actions,
            dones,
            store_reconstruction_samples=store_reconstruction_samples,
        )
        device = observations.device
        z0 = torch.zeros((), device=device)

        if isinstance(outputs.get("reconstructions", None), list):
            outputs["reconstructions"] = torch.stack(outputs["reconstructions"], dim=1)
        if isinstance(outputs.get("reconstruction_samples", None), list):
            outputs["reconstruction_samples"] = torch.stack(outputs["reconstruction_samples"], dim=1)

        def _mean_seq(key: str):
            xs = outputs.get(key, [])
            if xs is None or len(xs) == 0:
                return z0
            if torch.is_tensor(xs):
                return xs.mean()
            return torch.stack(xs).mean()

        recon_loss = _mean_seq("reconstruction_losses")
        kl_z = _mean_seq("kl_latents")
        hierarchical_kl = _mean_seq("kumaraswamy_kl_losses")
        component_margin = _mean_seq("component_margin")

        total_vae_loss = (
            lambda_recon * recon_loss
            + beta * (kl_z + hierarchical_kl)
            - component_margin
        )


        overshoot_kl = z0
        if getattr(self, "lambda_overshoot", 0.0) > 0.0:
            T = outputs["top_q_mean_maps"].shape[1]
            K_effect = min(self.overshoot_K, T - 2)
            overshoot_kl = self.compute_multistep_kl_overshoot(
                outputs["top_q_mean_maps"],
                outputs["top_q_logvar_maps"],
                actions,
                dones=dones,
                K=K_effect,
                mc_samples=self.overshoot_mc_samples,
                w_decay=self.overshoot_w_decay,
                core_state_maps=(
                    outputs["core_h_maps"],
                    outputs["core_c_maps"],
                    outputs["core_m_maps"],
                ),
                overshoot_anchor_state={
                    "z_prev_top": outputs["overshoot_z_prev_top"],
                },
            )
            _, _, C, H, W = observations.shape
            overshoot_kl = overshoot_kl / float(self.zdim * self.top_H * self.top_W)

        total_vae_loss = total_vae_loss + self.lambda_overshoot * overshoot_kl

        vae_losses = {
            "recon_loss": recon_loss,
            "kl_z": kl_z,
            "hierarchical_kl": hierarchical_kl,
            "component_margin": component_margin,
            "overshoot_kl": overshoot_kl,
            "total_vae_loss": total_vae_loss,
        }

        return vae_losses, outputs


    def compute_gradient_penalty_patch(self, D2d, real_x, fake_x, device, mask_flat=None):
        N = real_x.size(0)
        alpha = torch.rand(N, 1, 1, 1, device=device)
        x_hat = (alpha * real_x + (1 - alpha) * fake_x).requires_grad_(True)
        d_hat = D2d(x_hat).mean(dim=(1, 2, 3))
        grads = torch.autograd.grad(
            outputs=d_hat,
            inputs=x_hat,
            grad_outputs=torch.ones_like(d_hat, device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0].view(N, -1)

        gp_per = (grads.norm(2, dim=1) - 1.0).pow(2)
        if mask_flat is None:
            return gp_per.mean()
        return (gp_per * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)

    def _make_temporal_mask(self, B: int, T: int, device, sequence_lengths):
        if sequence_lengths is None:
            return None
        t = torch.arange(T, device=device)[None, :]           # [1,T]
        return (t < sequence_lengths[:, None])                # [B,T] bool

    def _masked_mean(self, x, mask):
        # x: [B,T] or [N]
        if mask is None:
            return x.mean()
        mask_f = mask.float()
        return (x * mask_f).sum() / mask_f.sum().clamp(min=1.0)

    def discriminator_step(
        self,
        real_images: torch.Tensor, #[B, T, C, H, W]
        fake_images: torch.Tensor, #[B, T, C, H, W]
        latents: torch.Tensor,  #[B, T, Cz, Ht, Wt] 
        sequence_lengths: Optional[torch.Tensor] = None,
        WGAN_GP_Coeff: float = 5.0,
        lambda_consistency: float = 0.4,
        lambda_temporal_lecam: float = 0.1,
        lambda_patch_lecam: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Training step for both discriminators
        """

        B, T, C, H, W = real_images.shape
        temporal_mask = self._make_temporal_mask(B, T, real_images.device, sequence_lengths )         # [B,T] bool
        # For per-frame PatchGAN losses
        mask_flat = None
        if temporal_mask is not None:
            mask_flat = temporal_mask.reshape(B * T).float()  # [B*T]

        disc_losses: Dict[str, torch.Tensor] = {}

        # Temporal Image Discriminator
        real_img_outputs = self.image_discriminator(real_images, z=latents.detach(), mask=temporal_mask)
        fake_img_outputs = self.image_discriminator(fake_images.clamp(-1.0, 1.0).detach(), z=latents.detach(), mask=temporal_mask)

        #Hinge Discriminator Loss
        temporal_disc_loss = (F.relu(1.0 - real_img_outputs['final_score']) + F.relu(1.0 + fake_img_outputs['final_score'])).mean()

        # Temporal consistency losses
        img_consistency_loss = torch.zeros((), device=self.device)
        fake_per_t = fake_img_outputs["per_timestep_score"].squeeze(-1)   # [B,Tdisc]
        if fake_per_t.size(1) > 1:
            pair_mask = None
            if temporal_mask is not None:
                tmask = temporal_mask[:, :fake_per_t.size(1)]
                pair_mask = (tmask[:, 1:] & tmask[:, :-1]).float()
            diffs = (fake_per_t[:, 1:] - fake_per_t[:, :-1]).abs()
            img_consistency_loss = self._masked_mean(diffs, pair_mask)

        real_frames = real_images.reshape(B * T, C, H, W).contiguous()
        fake_frames = fake_images.detach().reshape(B * T, C, H, W).contiguous()

        real_logits = self.patch_discriminator(real_frames)                 # [B*T,1,h,w]
        fake_logits = self.patch_discriminator(fake_frames)                 # [B*T,1,h,w]
        real_frame_score = real_logits.mean(dim=(1,2,3))  # [B*T]
        fake_frame_score = fake_logits.mean(dim=(1,2,3))  # [B*T]

        patch_disc_loss = self._masked_mean(fake_frame_score, mask_flat) - self._masked_mean(real_frame_score, mask_flat)

        patch_gp = self.compute_gradient_penalty_patch(
            D2d=self.patch_discriminator,
            real_x=real_frames,
            fake_x=fake_frames,
            device= real_images.device,
            mask_flat=mask_flat,
        )

        # --- LeCam inputs ---
        real_temporal_for_lc = real_img_outputs["final_score"].reshape(-1)   # one score per sequence
        fake_temporal_for_lc = fake_img_outputs["final_score"].reshape(-1)

        if mask_flat is not None:
            valid = mask_flat > 0.5
            real_patch_for_lc = real_frame_score[valid]
            fake_patch_for_lc = fake_frame_score[valid]
        else:
            real_patch_for_lc = real_frame_score
            fake_patch_for_lc = fake_frame_score

        # current batch means for anchor updates
        temporal_real_mean = real_temporal_for_lc.mean()
        temporal_fake_mean = fake_temporal_for_lc.mean()
        patch_real_mean = real_patch_for_lc.mean()
        patch_fake_mean = fake_patch_for_lc.mean()

        # don't apply LeCam until anchors are initialized
        if bool(self.lecam_initialized):
            temporal_lecam_loss = self.compute_lecam_loss(
                logits_real=real_temporal_for_lc,
                logits_fake=fake_temporal_for_lc,
                ema_logits_real=self.lecam_temporal_real_ema.to(
                    device=real_temporal_for_lc.device, dtype=real_temporal_for_lc.dtype
                ),
                ema_logits_fake=self.lecam_temporal_fake_ema.to(
                    device=fake_temporal_for_lc.device, dtype=fake_temporal_for_lc.dtype
                ),
            )

            patch_lecam_loss = self.compute_lecam_loss(
                logits_real=real_patch_for_lc,
                logits_fake=fake_patch_for_lc,
                ema_logits_real=self.lecam_patch_real_ema.to(
                    device=real_patch_for_lc.device, dtype=real_patch_for_lc.dtype
                ),
                ema_logits_fake=self.lecam_patch_fake_ema.to(
                    device=fake_patch_for_lc.device, dtype=fake_patch_for_lc.dtype
                ),
            )
        else:
            temporal_lecam_loss = real_temporal_for_lc.new_zeros(())
            patch_lecam_loss = real_frame_score.new_zeros(())

        img_disc_loss = temporal_disc_loss + lambda_consistency * img_consistency_loss + patch_disc_loss + WGAN_GP_Coeff * patch_gp + lambda_temporal_lecam * temporal_lecam_loss + lambda_patch_lecam * patch_lecam_loss

        if hasattr(self, 'img_disc_optimizer'):
            # Update image discriminator
            self.img_disc_optimizer.zero_grad()
            img_disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.image_discriminator.parameters()) + list(self.patch_discriminator.parameters()),
                self._grad_clip
            )
            self.img_disc_optimizer.step()
            self._update_lecam_ema(
                temporal_real_mean=temporal_real_mean,
                temporal_fake_mean=temporal_fake_mean,
                patch_real_mean=patch_real_mean,
                patch_fake_mean=patch_fake_mean,
            )

        disc_losses.update({
            'img_disc_loss': img_disc_loss,
            # Temporal discriminator metrics
            'temporal_disc_loss': temporal_disc_loss.detach(),
            'temporal_disc_real': real_img_outputs["final_score"].mean().detach(),
            'temporal_disc_fake': fake_img_outputs["final_score"].mean().detach(),
            'temporal_consistency_loss': img_consistency_loss.detach(),  # Renamed
            # PatchGAN metrics
            'patch_disc_loss': patch_disc_loss.detach(),  # New key
            'patch_gp': patch_gp.detach(),
            'patch_disc_real': real_frame_score.mean().detach(),
            'patch_disc_fake': fake_frame_score.mean().detach(),
            'temporal_lecam_loss': temporal_lecam_loss.detach(),
            'patch_lecam_loss': patch_lecam_loss.detach(),
        })
        return  disc_losses


    def compute_feature_matching_loss(
        self,
        real_features: torch.Tensor,
        fake_features: torch.Tensor,
        temporal_mask: torch.Tensor | None = None,
    ):
        if temporal_mask is None:
            return F.l1_loss(fake_features, real_features.detach())

        # fake_features, real_features: [B, T, C, H, W]
        diff = torch.abs(fake_features - real_features.detach())
        m = temporal_mask.to(dtype=diff.dtype).view(diff.shape[0], diff.shape[1], 1, 1, 1)

        denom = (m.sum() * diff.shape[2] * diff.shape[3] * diff.shape[4]).clamp(min=1.0)
        return (diff * m).sum() / denom

    def compute_adversarial_losses(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        z_seq: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None,
        lambda_final: float = 0.8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        D = self.image_discriminator
        Dpatch = self.patch_discriminator
        B, T, C, H, W = reconstruction.shape

        temporal_mask = self._make_temporal_mask(B, T, reconstruction.device, sequence_lengths)
        mask_flat = None
        if temporal_mask is not None:
            mask_flat = temporal_mask.reshape(B * T).float()

        flags = [p.requires_grad for p in D.parameters()]
        for p in D.parameters():
            p.requires_grad_(False)

        fake_img_outputs = D(
            reconstruction.clamp(-1.0, 1.0),
            z=z_seq,
            mask=temporal_mask,
            return_features=True
        )
        real_img_outputs = D(
            x,
            z=z_seq.detach(),
            mask=temporal_mask,
            return_features=True
        )

        final_scores = fake_img_outputs["final_score"].squeeze(-1).squeeze(-1)
        final_adv_loss = -final_scores.mean()
        temporal_adv_loss = lambda_final * final_adv_loss

        mask_tdisc = None
        if temporal_mask is not None:
            mask_tdisc = temporal_mask[:, :fake_img_outputs["frame_features"].shape[1]]

        feature_match_loss = self.compute_feature_matching_loss(
            real_features=real_img_outputs["frame_features"],
            fake_features=fake_img_outputs["frame_features"],
            temporal_mask=mask_tdisc,
        )

        for p, f in zip(D.parameters(), flags):
            p.requires_grad_(f)

        patch_flags = [p.requires_grad for p in Dpatch.parameters()]
        for p in Dpatch.parameters():
            p.requires_grad_(False)

        fake_frames = reconstruction.reshape(B * T, C, H, W).contiguous()
        patch_logits = Dpatch(fake_frames)
        patch_scores = patch_logits.mean(dim=(1, 2, 3))
        img_adv_loss = -self._masked_mean(patch_scores, mask_flat)

        for p, f in zip(Dpatch.parameters(), patch_flags):
            p.requires_grad_(f)

        return img_adv_loss, temporal_adv_loss, feature_match_loss

    def denormalize_generated_images(self, images):
        """
        Convert generated images from [-1, 1] back to [0, 1] for visualization
        """
        return (images + 1) / 2


    def training_step_sequence(self,
                            observations: torch.Tensor,
                            actions: torch.Tensor = None,
                            dones: torch.Tensor = None,
                            beta: float = 1.0,
                            n_critic: int = 3,
                            lambda_img: float = 0.5,
                            lambda_recon: float = 1.0,
                            batch_idx: Optional[int] = None,
                            ) -> Dict[str, torch.Tensor]:
        """
        - Dynamic Weight Averaging (DWA) over either:
            * ELBO / adv  streams (when grad_balance_method != "none"), or
            * all individual loss terms together (when grad_balance_method == "none").
        """
        self.train()
        if getattr(self, "image_discriminator", None) is not None:
            self.image_discriminator.train()


        # 1) Prepare data and compute VAE loss
        warmup_factor = self.get_warmup_factor()

        vae_losses, outputs = self.compute_total_loss(
            observations,
            actions,
            dones,
            beta,
            lambda_recon
        )

        # z_seq for disc conditioning (teacher-forced)
        z_seq_tf = outputs["z_seq_maps"]  # [B, T, z+h, Ht, Wt]
        B, seq_len = observations.shape[:2]
        if dones is not None:
            _, lengths_full = self._lengths_from_dones(dones, T=seq_len, assume_padded_after_done=True)
        else:
            lengths_full = torch.full((B,), seq_len, device=observations.device, dtype=torch.long)

        # 2) Decide whether to do rollout GAN this step
        T_total = seq_len
        do_rollout = (
            self.lambda_rollout_adv > 0.0
            and self.rollout_adv_every > 0
            and (self.global_step % self.rollout_adv_every == 0)
            and (T_total >= 2)
        )

        rollout_horizon = 0
        T_ctx = 0
        real_future = None
        fake_future_D = None
        z_seq_roll_D = None
        seq_len_future = None
        keep = None
        actions_slice = None
        dones_slice = None

        if do_rollout:
            T_ctx = min(self.rollout_context_frames, T_total - 1)
            rollout_horizon = min(self.rollout_horizon, T_total - T_ctx)

            if rollout_horizon <= 0:
                do_rollout = False
            else:
                real_future = observations[:, T_ctx:T_ctx + rollout_horizon]
                actions_slice = actions[:, :T_ctx + rollout_horizon] if actions is not None else None
                dones_slice = dones[:, :T_ctx + rollout_horizon] if dones is not None else None

                if dones_slice is not None:
                    alive_slice, _ = self._lengths_from_dones(
                        dones_slice, T_ctx + rollout_horizon, assume_padded_after_done=True
                    )
                    future_alive = alive_slice[:, T_ctx:T_ctx + rollout_horizon]   # [B, H]
                    future_len = future_alive.long().sum(dim=1)                    # [B]
                    keep = (future_len > 0)
                else:
                    keep = torch.ones(B, device=observations.device, dtype=torch.bool)
                    future_len = torch.full((B,), rollout_horizon, device=observations.device, dtype=torch.long)

                if keep.sum().item() == 0:
                    do_rollout = False
                    real_future = None
                else:
                    # Apply keep consistently
                    real_future = real_future[keep]
                    seq_len_future = future_len[keep]

                    # Rollout once WITHOUT grad for discriminator updates
                    dbgD = self.generate_future_sequence(
                        initial_obs=observations[keep, :T_ctx],
                        actions=(actions_slice[keep] if actions_slice is not None else None),
                        horizon=rollout_horizon,
                        top_temperature=self.rollout_top_temperature,
                        decoder_temperature=self.rollout_decoder_temperature,
                        dones=(dones_slice[keep] if dones_slice is not None else None),
                        grad=False,
                    )
                    fake_future_D = dbgD["vae_future"]   # [B_keep, H, C, H, W]
                    z_seq_roll_D = dbgD["z_seq"]         # [B_keep, H, Z] (should already be detached inside)

        # 3) Discriminator updates (n_critic)
        disc_losses_list: List[Dict[str, torch.Tensor]] = []
        for _ in range(n_critic):
            # teacher-forced recon fakes
            disc_loss = self.discriminator_step(
                real_images=observations,
                fake_images=outputs["reconstructions"].detach(),
                latents=z_seq_tf,
                sequence_lengths=lengths_full,
            )

            # rollout fakes (optional)
            if do_rollout and (fake_future_D is not None) and (z_seq_roll_D is not None):
                disc_loss_roll = self.discriminator_step(
                    real_images=real_future,
                    fake_images=fake_future_D,
                    latents=z_seq_roll_D,
                    sequence_lengths=seq_len_future,
                )
                for k, v in disc_loss_roll.items():
                    disc_loss[f"rollout_{k}"] = v

            disc_losses_list.append(disc_loss)

        avg_disc_losses: Dict[str, torch.Tensor] = {}
        if disc_losses_list:
            avg_disc_losses = {
                k: sum(d[k] for d in disc_losses_list) / len(disc_losses_list)
                for k in disc_losses_list[0].keys()
            }

        # 4) Generator adversarial losses (teacher-forced)
        lambda_img_eff = (lambda_img * warmup_factor) if warmup_factor > 0.0 else 0.0

        img_adv_loss, temporal_adv_loss, feat_match_loss = self.compute_adversarial_losses(
            x=observations,
            reconstruction=outputs["reconstructions"],
            z_seq=z_seq_tf,
            sequence_lengths=lengths_full
        )

        # 5) Generator adversarial losses (rollout) — SAVP-style prior realism
        rollout_img_adv_loss = torch.zeros((), device=observations.device)
        rollout_temporal_adv_loss = torch.zeros((), device=observations.device)
        rollout_feat_match_loss = torch.zeros((), device=observations.device)

        if do_rollout and rollout_horizon > 0:
            # Rollout again WITH grad for generator update
            dbgG = self.generate_future_sequence(
                initial_obs=observations[keep, :T_ctx],
                actions=(actions_slice[keep] if actions_slice is not None else None),
                horizon=rollout_horizon,
                top_temperature=self.rollout_top_temperature,
                decoder_temperature=self.rollout_decoder_temperature,
                dones=(dones_slice[keep] if dones_slice is not None else None),
                grad=True,
            )
            fake_future_G = dbgG["vae_future"]      # [B_keep, H, C, H, W] (requires grad)
            z_seq_roll_G = dbgG["z_seq"].detach()   # keep conditioning stable + save memory
            # Flatten time for LPIPS
            fake_bt = rearrange(fake_future_G, "b t c h w -> (b t) c h w")
            real_bt = rearrange(real_future,   "b t c h w -> (b t) c h w")

            # --- Edge consistency loss (rollout) ---
            # Optional: work in [0,1] for more stable edge magnitudes
            fake01 = self.denormalize_generated_images(fake_future_G)  # [-1,1] -> [0,1]
            real01 = self.denormalize_generated_images(real_future)

            Bf, Tf, C, H, W = fake01.shape
            fake_flat = fake01.reshape(Bf * Tf, C, H, W)
            real_flat = real01.reshape(Bf * Tf, C, H, W)


            rollout_img_adv_loss, rollout_temporal_adv_loss, rollout_feat_match_loss = self.compute_adversarial_losses(
                x=real_future,
                reconstruction=fake_future_G,
                z_seq=z_seq_roll_G,
                sequence_lengths=seq_len_future,
            )

        # 6) Combine losses with DWA or fixed weights
        if self.use_dwa:
            total_components = {
                "recon_loss": vae_losses["recon_loss"].reshape([]),
                "kl_z": vae_losses["kl_z"].reshape([]),
                "hierarchical_kl": vae_losses["hierarchical_kl"].reshape([]),
                "component_margin": (-vae_losses["component_margin"]).reshape([]),

                "img_adv_loss": (warmup_factor * img_adv_loss).reshape([]),
                "temporal_adv_loss": (warmup_factor * temporal_adv_loss).reshape([]),
                "feat_match_loss": (warmup_factor * feat_match_loss).reshape([]),

                "rollout_img_adv_loss": (warmup_factor * rollout_img_adv_loss).reshape([]),
                "rollout_temporal_adv_loss": ( warmup_factor * rollout_temporal_adv_loss).reshape([]),
                "rollout_feat_match_loss": ( warmup_factor * rollout_feat_match_loss).reshape([]),
                

                "overshoot_kl": vae_losses["overshoot_kl"].reshape([]),
            }

            total_gen_loss = self.total_weighter.reduce_losses(total_components, batch_idx)
        else:
            adv_base = (
                warmup_factor * lambda_img_eff * img_adv_loss
                + warmup_factor * temporal_adv_loss
                + warmup_factor * lambda_img * feat_match_loss
            )
            adv_roll = self.lambda_rollout_adv * (
                warmup_factor * lambda_img_eff * rollout_img_adv_loss
                + warmup_factor * rollout_temporal_adv_loss
                + warmup_factor * lambda_img * rollout_feat_match_loss
            )
            total_gen_loss = (
                vae_losses["total_vae_loss"]
                + adv_base
                + adv_roll
            )

        # 7) Backprop generator
        self.gen_optimizer.zero_grad(set_to_none=True)
        total_gen_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for g in self.gen_optimizer.param_groups for p in g["params"]],
            self._grad_clip,
        )
        self.gen_optimizer.step()
        self.global_step += 1
        grad_norm_sq = 0.0
        for group in self.gen_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    g = p.grad.data
                    grad_norm_sq = grad_norm_sq + float((g.norm(2)).item() ** 2)
        grad_norm = (grad_norm_sq ** 0.5) if grad_norm_sq > 0 else 0.0

        # 8) EMA updates for encoder/decoder
        with torch.no_grad():
            self.ema_vdvae.update()

        pi_seq = torch.stack(outputs["prior_pi"], dim=1)

        eff_comp = pi_seq.max(dim=-1).values.mean(dim=0).mean().item()                 # [T]
        top6_cov = pi_seq.topk(min(6, pi_seq.size(-1)), dim=-1).values.sum(dim=-1).mean(dim=0).mean().item()  # [T]

        return {
            **vae_losses,
            **avg_disc_losses,

            "img_adv_loss": float(img_adv_loss.item()),
            "temporal_adv_loss": float(temporal_adv_loss.item()),
            "feat_match_loss": float(feat_match_loss.item()),
            "rollout_img_adv_loss": float(rollout_img_adv_loss.item()),
            "rollout_temporal_adv_loss": float(rollout_temporal_adv_loss.item()),
            "rollout_feat_match_loss": float(rollout_feat_match_loss.item()),
            "did_rollout_adv": float(1.0 if do_rollout else 0.0),
            "total_gen_loss": float(total_gen_loss.item()),
            "grad_norm": float(grad_norm),
            "effective_components": eff_comp,
            "Top 6 coverage": top6_cov,
        }


    def get_warmup_factor(self) -> float:
        """Calculate warmup factor for adversarial and flow-alignment losses."""
        if self.current_epoch < self.warmup_epochs:
            return self.current_epoch / self.warmup_epochs
        return 1.0


    @torch.no_grad()
    def sample(self, num_samples: int) -> torch.Tensor:
        device = self.device
        dtype = next(self.parameters()).dtype

        core_state = self.rnn.init_state(num_samples, device=device, dtype=dtype)
        h_t = self.rnn.out_norm(core_state[0])
        z_prev_top = torch.zeros(
            num_samples, self.zdim, self.top_H, self.top_W, device=device, dtype=dtype
        )

        tr = self.latent_transport(
            z_top_prev=z_prev_top,
            h_t=h_t,
            action_prev=torch.zeros(num_samples, self.action_dim, device=device, dtype=dtype),
            dt=1.0,
        )

        if hasattr(self, "ema_vdvae"):
            self.ema_vdvae.apply_shadow()
        prev_latents = self._init_decoder_prev_latents(num_samples, device, dtype)

        x_np, current_latents = self.vdvae.sample(
            num_samples,
            cond_top=tr["cond_top"],
            h_decoder_top=tr["h_decoder_top"],
            prev_latents=prev_latents,
        )

        if hasattr(self, "ema_vdvae"):
            self.ema_vdvae.restore()

        x = torch.from_numpy(x_np).permute(0, 3, 1, 2).contiguous().float() / 127.5 - 1.0
        return x

    def generate_future_sequence(
        self,
        initial_obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        horizon: int = 15,
        top_temperature: float = 1.0,
        decoder_temperature: float = 1.0,  # kept for API compatibility; currently unused
        dones: Optional[torch.Tensor] = None,
        grad: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Warm up on context frames with teacher forcing, then roll forward by sampling
        one full top latent map per image from the image-level DP-GMM prior.

        Returns:
            vae_future: [B, H_roll, C, H, W]
            z_seq:      [B, H_roll, zdim + hidden_dim, top_H, top_W]
            pi_seq:     [B, H_roll, K]
        """
        B, T_ctx, C, H, W = initial_obs.shape
        device = initial_obs.device
        dtype = initial_obs.dtype

        if actions is None:
            actions = torch.zeros(
                B, T_ctx + horizon, self.action_dim, device=device, dtype=dtype
            )
        elif actions.shape[1] < (T_ctx + horizon):
            pad = (T_ctx + horizon) - actions.shape[1]
            actions = torch.cat(
                [
                    actions,
                    torch.zeros(B, pad, actions.shape[2], device=device, dtype=actions.dtype),
                ],
                dim=1,
            )

        if dones is not None and dones.shape[1] < (T_ctx + horizon):
            pad = (T_ctx + horizon) - dones.shape[1]
            dones = torch.cat(
                [dones, torch.zeros(B, pad, device=device, dtype=dones.dtype)],
                dim=1,
            )

        def mask_at(t_abs: int) -> torch.Tensor:
            if dones is None or t_abs == 0:
                return torch.ones(B, device=device, dtype=torch.float32)
            return (1.0 - dones[:, t_abs - 1].float()).to(torch.float32)

        core_state = self.rnn.init_state(B, device=device, dtype=dtype)
        z_prev_top = torch.zeros(B, self.zdim, self.top_H, self.top_W, device=device, dtype=dtype)
        prev_latents = self._init_decoder_prev_latents(B, device, dtype)

        pred_imgs: List[torch.Tensor] = []
        z_seq: List[torch.Tensor] = []
        pi_seq: List[torch.Tensor] = []

        with torch.set_grad_enabled(grad):
            # ---------------------------------
            # teacher-forced warmup on context
            # ---------------------------------
            for t in range(T_ctx):
                x_t = initial_obs[:, t]
                mask_t = mask_at(t)

                keep = mask_t.view(B, 1, 1, 1).to(device=device, dtype=dtype)
                h, c, m_ = core_state
                h0, c0, m0 = self.rnn.init_state(B, device=device, dtype=dtype)
                core_state = (
                    h * keep + h0 * (1.0 - keep),
                    c * keep + c0 * (1.0 - keep),
                    m_ * keep + m0 * (1.0 - keep),
                )

                z_prev_top_masked = z_prev_top * keep
                prev_latents = [pl * keep if pl is not None else None for pl in prev_latents]
                h_t = self.rnn.out_norm(core_state[0])

                tr = self.latent_transport(
                    z_top_prev=z_prev_top_masked,
                    h_t=h_t,
                    action_prev=(
                        actions[:, t - 1] * mask_t.view(B, 1).to(device=device, dtype=dtype)
                        if t > 0
                        else torch.zeros(B, self.action_dim, device=device, dtype=dtype)
                    ),
                    dt=1.0,
                )

                x_t_nhwc = x_t.permute(0, 2, 3, 1).contiguous()
                vdvae_out = self.vdvae.forward(
                    x_t_nhwc,
                    x_t_nhwc,
                    cond_top=tr["cond_top"],
                    h_decoder_top=tr["h_decoder_top"],
                    mask_t=mask_t,
                    prev_latents=prev_latents,
                    get_latents=True,
                )

                z_top_map = draw_gaussian_diag_samples(
                    vdvae_out["top_q_mean_map"],
                    vdvae_out["top_q_logvar_map"],
                )

                a_cur = actions[:, t]
                _, core_state = self.rnn(
                    z_top_map,
                    a_cur,
                    state=core_state,
                    mask_t=None,
                    extra_maps=None,
                )

                z_prev_top = z_top_map.detach()
                prev_latents = [None if idx == 0 else pl.detach() for idx, pl in enumerate(vdvae_out["current_latents"])]

            # -------------------------
            # future rollout from prior
            # -------------------------
            for k in range(horizon):
                t_abs = T_ctx + k
                mask_t = mask_at(t_abs)

                keep = mask_t.view(B, 1, 1, 1).to(device=device, dtype=dtype)
                h, c, m_ = core_state
                h0, c0, m0 = self.rnn.init_state(B, device=device, dtype=dtype)
                core_state = (
                    h * keep + h0 * (1.0 - keep),
                    c * keep + c0 * (1.0 - keep),
                    m_ * keep + m0 * (1.0 - keep),
                )

                z_prev_top_masked = z_prev_top * keep
                prev_latents = [pl * keep if pl is not None else None for pl in prev_latents]
                h_t = self.rnn.out_norm(core_state[0])

                a_prev = (
                    actions[:, t_abs - 1] * mask_t.view(B, 1).to(device=device, dtype=dtype)
                    if t_abs > 0
                    else torch.zeros(B, self.action_dim, device=device, dtype=dtype)
                )

                tr = self.latent_transport(
                    z_top_prev=z_prev_top_masked,
                    h_t=h_t,
                    action_prev=a_prev,
                    dt=1.0,
                )

                _, prior_params = self.prior(tr["cond_top"])
                pi_seq.append(prior_params["pi"].detach())

                z_img, _ = self.prior.sample_image_latent(
                    prior_params,
                    temperature=top_temperature,
                    grad=grad,
                    return_stats=True,
                )

                z_top_map = z_img.view(B, self.zdim, self.top_H, self.top_W).contiguous()

                px_z, current_latents = self.vdvae.decode_from_top_latent(
                    z_top_map,
                    cond_top=tr["cond_top"],
                    h_decoder_top=tr["h_decoder_top"],
                    prev_latents=prev_latents,
                )

                dmol_out = self.vdvae.decoder.out_net.forward(px_z)
                if grad:
                    x_hat = mean_from_discretized_mix_logistic(
                        dmol_out, self.vdvae.H.num_mixtures
                    )
                else:
                    x_hat = sample_from_discretized_mix_logistic(
                        dmol_out, self.vdvae.H.num_mixtures
                    )

                x_hat = x_hat.permute(0, 3, 1, 2).contiguous()
                pred_imgs.append(x_hat)

                a_cur = (
                    actions[:, t_abs]
                    if t_abs < (T_ctx + horizon - 1)
                    else torch.zeros_like(a_prev)
                )

                h_next, core_state = self.rnn(
                    z_top_map,
                    a_cur,
                    state=core_state,
                    mask_t=None,
                    extra_maps=None,
                )

                # preserve spatial structure for discriminator / analysis
                z_seq.append(torch.cat([z_top_map, h_next], dim=1).detach())
                z_prev_top = z_top_map.detach()
                prev_latents = [None if idx == 0 else pl.detach() for idx, pl in enumerate(current_latents)]

        vae_future = (
            torch.stack(pred_imgs, dim=1)
            if pred_imgs
            else initial_obs.new_zeros((B, 0, C, H, W))
        )

        z_seq_out = (
            torch.stack(z_seq, dim=1)
            if z_seq
            else initial_obs.new_zeros(
                (B, 0, self.zdim + self.hidden_dim, self.top_H, self.top_W)
            )
        )

        pi_seq_out = (
            torch.stack(pi_seq, dim=1)
            if pi_seq
            else initial_obs.new_zeros((B, 0, self.max_K))
        )

        return {
            "vae_future": vae_future,
            "z_seq": z_seq_out,
            "pi_seq": pi_seq_out,
        }
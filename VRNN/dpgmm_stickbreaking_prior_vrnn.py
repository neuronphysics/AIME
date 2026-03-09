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
from vdvae.vae import VDVAE
from vdvae.hps import Hyperparams
from vdvae.vae_helpers import mean_from_discretized_mix_logistic, sample_from_discretized_mix_logistic, draw_gaussian_diag_samples
from VRNN.utils.canny_net import CannyFilter
from VRNN.warp import (
    image_warp, FlowWarpingLoss, TVLoss, WarpLoss, fbLoss,
    fbConsistencyCheck, create_outgoing_mask, edgeLoss,
    charbonnier_loss, CensusLoss, rgb2gray, downsample_flow, upsample_flow, upsample_flow_raft,
)
from VRNN.flow_predict import get_gru_encoder, get_gru_decoder, coords_grid, FlowUpsampleMask, HFToTop, ConvPatchTokenizer, LatentFlowHead, HFMapPredictor, EdgeToTop, TokenTransporter, TokensToTopMap, FlowRefiner, MaxDeltaCalibrator, AntiAliasInterpolation2d, HFTopUpsampler
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
    def __init__(self, hidden_dim: int, num_components: int, device: torch.device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.K = num_components
        self.device = device
        self.eps = torch.tensor(torch.finfo(torch.float32).eps, device=self.device)
        kumar_a_fc = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(kumar_a_fc.weight, gain=0.5)
        nn.init.constant_(kumar_a_fc.bias, 0.0)

        kumar_a_out = nn.Linear(hidden_dim, self.K - 1)
        nn.init.normal_(kumar_a_out.weight, 0, 0.01)
        nn.init.constant_(kumar_a_out.bias, 0.0)

        kumar_b_fc = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(kumar_b_fc.weight, gain=0.5)
        nn.init.constant_(kumar_b_fc.bias, 0.0)

        kumar_b_out = nn.Linear(hidden_dim, self.K - 1)
        nn.init.normal_(kumar_b_out.weight, 0, 0.01)
        nn.init.constant_(kumar_b_out.bias, 0.0)

        # Then apply spectral norm and build sequential
        self.net_a = nn.Sequential(OrderedDict([
            ('kumar_a_fc', nn.utils.spectral_norm(kumar_a_fc)),
            ('kumar_a_ln', nn.LayerNorm(hidden_dim, eps=1e-6)),
            ('kumar_a_relu', nn.GELU()),
            ('kumar_a_out', nn.utils.spectral_norm(kumar_a_out)),
        ]))


        self.net_b = nn.Sequential(OrderedDict([
            ('kumar_b_fc', nn.utils.spectral_norm(kumar_b_fc)),
            ('kumar_b_ln', nn.LayerNorm(hidden_dim, eps=1e-6)),
            ('kumar_b_relu', nn.GELU()),
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
        self.kumar_net = KumaraswamyNetwork(hidden_dim, max_components, device)
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

    def forward(self, x):
        residual = x                          # [B, hidden_dim]
        h = self.fc1(x)
        h = F.gelu(h)
        h = self.ln1(h)
        h = self.fc2(h)
        h = F.gelu(h)
        h = self.ln2(h)
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
        """
        Accept either:
          [B, Hc]         -> returned unchanged
          [B, Hc, Ht, Wt] -> pooled image-level summary [B, Hc]
        """
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

        tokens = h.flatten(2).transpose(1, 2).contiguous()
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

@dataclass
class FlowCfg:
    input_channels: int = 3
    flow_ctx_dim: int = 64
    flow_state_channels: int = 64
    flow_dec_feat_channels: int = 128

    # Edge weighting
    thr_e: float = 0.2
    edge_temp: float = 0.05
    eps: float = 1e-6

    # Warping
    warp_padding_mode: str = "border"   # key fix vs zeros artifacts
    rgb_warp_mode: str = "bilinear"
    edge_warp_mode: str = "bilinear"

    # Loss weights
    w_edge: float = 0.2
    w_photo: float = 1.0
    w_tv: float = 0.1

    # Visibility imbalance: occluded pixels are rare, upweight negatives (v_target=0)
    vis_neg_weight: float = 5.0

    # Residual should not "invent" stuff where warp is valid
    w_res_suppress: float = 0.05

    # Scheduled sampling: early use v_target, later switch to v_pred
    blend_use_target_steps: int = 2000

    # FB schedule
    fb_warmup_steps: int = 15
    w_fb: float = 0.1
    w_fb_photo: float = 0.3
    fb_alpha1: float = 0.01
    fb_alpha2: float = 0.5

    # Charbonnier epsilon
    charbonnier_eps: float = 1e-3

    # Optional extra losses
    w_census: float = 0.1
    w_grad: float = 0.1

    # visibility + residual inpainting
    w_vis: float = 0.2
    w_inpaint_photo: float = 0.5
    w_vis_area: float = 0.0

##############################
### Main DPGMMVRNN Class #####

class DPGMMVariationalRecurrentAutoencoder(nn.Module):
    """
    Using Dirichlet Process GMM Prior with Stick-Breaking, Self-Modeling
    This architecture incorporates adaptive temporal dynamics, self-modeling, and attention mechanisms
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
        use_orthogonal: bool = True,  # Use orthogonal initialization for LSTM,
        number_lstm_layer: int = 3,  # Number of LSTM layers
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
        rollout_decoder_temperature: float = 0.4,
        rollout_decode_mode: str = "mean",      # "mean" or "sample"
        flow_ctx_dim: int = 64,
        patch_disc_layers: int = 2,
        patch_disc_ndf:int = 32,
        hf_top_channels: int = 64,
        motion_ctx_channels: int = 64,
        motion_token_dim: int = 64,
        num_motion_tokens: int = 64,
        motion_token_heads: int = 4,
        hf_token_patch: int = 4,
        edge_top_channels: int = 16,
        latent_flow_hidden: int = 64,
        warp_core_state: bool =True,
        lambda_lat_feat: float = 0.5,
        lambda_lat_distill:float = 0.5,
        lambda_top_bottom_warp: float = 0.75,
        beta_flow_agree: float = 0.1,
        # --- LatentWarp on top z ---
        lambda_z_lw: float = 0.75,         # weight of single top-z temporal consistency loss
        lw_alpha: float = 5.0,             # residual penalty in keep score
        lw_threshold: float = 0.6,         # keep threshold in keep score
        lw_tau: float = 0.15,              # temperature for soft keep mask (smaller = sharper)
        lw_hard_mask: bool = False,        # True -> hard (0/1) keep mask; False -> soft mask
        lw_detach_flow: bool = True,       # True -> LatentWarp does NOT backprop into flow nets
        lambda_roll_prop: float = 0.25,    # rollout cosine alignment between prior mean and warped latent
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
        
        # Hyperparameters
        self._lr = learning_rate
        self._grad_clip = grad_clip
        self.number_lstm_layer = number_lstm_layer
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.use_dwa = use_dwa
        self.use_ctx_checkpoint = use_ctx_checkpoint
        self.patch_disc_layers=patch_disc_layers
        self.patch_disc_ndf = patch_disc_ndf
        self.eps = torch.finfo(torch.float32).eps
        # KL overshooting
        self.overshoot_mc_samples = 32
        self.overshoot_w_decay = 0.9
        self.lambda_overshoot = 0.75
        self.warp_edge_tf_weight = 1.0
        self.warp_rgb_tf_weight = 1.0
        self.flow_ctx_dim = int(flow_ctx_dim)
        self.warp_mode = "bilinear"
        # rollout GAN attributes
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long), persistent=True)

        self.rollout_adv_every = int(rollout_adv_every)
        self.rollout_context_frames = int(rollout_context_frames)
        self.rollout_horizon = int(rollout_horizon)
        self.lambda_rollout_adv = float(lambda_rollout_adv)
        self.rollout_top_temperature = float(rollout_top_temperature)
        self.rollout_decoder_temperature = float(rollout_decoder_temperature)
        self.rollout_decode_mode = str(rollout_decode_mode)
        self.hf_top_channels = hf_top_channels
        self.motion_ctx_channels = motion_ctx_channels
        self.motion_token_dim = motion_token_dim
        self.num_motion_tokens = num_motion_tokens
        self.motion_token_heads = motion_token_heads
        self.hf_token_patch = hf_token_patch
        self.edge_top_channels = edge_top_channels
        self.latent_flow_hidden = latent_flow_hidden
        self.beta_flow_agree = beta_flow_agree

        self.warp_core_state = warp_core_state

        self.extra_channels = (2 * hf_top_channels + motion_ctx_channels + edge_top_channels)

        self.lambda_lat_feat = lambda_lat_feat
        self.lambda_lat_distill = lambda_lat_distill
        self.lambda_top_bottom_warp = float(lambda_top_bottom_warp)

        # --- LatentWarp (top-z alignment) ---
        self.lambda_z_lw = float(lambda_z_lw)
        self.lw_alpha = float(lw_alpha)
        self.lw_threshold = float(lw_threshold)
        self.lw_tau = float(lw_tau)
        self.lw_hard_mask = bool(lw_hard_mask)
        self.lw_detach_flow = bool(lw_detach_flow)
        self.lambda_roll_prop = float(lambda_roll_prop)
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
        self._init_vrnn_dynamics(use_orthogonal=use_orthogonal,number_lstm_layer=number_lstm_layer, extra_channels=self.extra_channels)
        self._init_discriminators(img_disc_layers, patch_size, num_heads=disc_num_heads)
        self._init_motion_scaffold()
        self._init_latent_motion_conditioners() #should be after initialization of encoder, decoder
        #
        self.fb_w = 1.0
        self.tv_w = 1.0
        self.disagree_w = 1.0
        self.thr_e = 0.2
        self.dilate_k = 7

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
                "img_adv_loss",
                "temporal_adv_loss",
                "feat_match_loss",
                "rollout_img_adv_loss",
                "rollout_temporal_adv_loss",
                "rollout_feat_match_loss",
                "rollout_edge_loss",
                "rollout_warp_edge_loss",
                #"warp_sanity_total",
                "warp_lb_loss",
                "warp_tv_gate_loss",
                "warp_disagree_loss",
                "warp_edge_tf_loss",
                "warp_rgb_tf_loss",
                "lat_feat",
                "lat_distill",
                #"lat_distill64",
                "top_bottom_warp_consistency",
                "overshoot_kl",           #
            ],
            temperature=temperature,
        )

    def _logit_safe(self, p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        p = p.clamp(eps, 1.0 - eps)
        return torch.log(p) - torch.log1p(-p)

    @staticmethod
    def to01(x: torch.Tensor) -> torch.Tensor:
        # Commonly observations are in [-1, 1]
        return (x * 0.5 + 0.5).clamp(0.0, 1.0)

    @staticmethod
    def warp_with_mode(
        img: torch.Tensor,
        flow_pix: torch.Tensor,
        mode: str = "bilinear",
        padding_mode: str = "border",
        align_corners: bool = True,
    ) -> torch.Tensor:
        """
        Backward warp convention: sample img at base_coords + flow_pix (flow in pixel units).
        """
        if img.dim() != 4:
            raise ValueError("img must be [B,C,H,W]")
        if flow_pix.dim() != 4 or flow_pix.size(1) != 2:
            raise ValueError("flow_pix must be [B,2,H,W]")

        b, c, h, w = img.shape
        base = coords_grid(b, h, w, device=img.device, dtype=img.dtype)  # [B,2,H,W] pixel coords
        coords = base + flow_pix

        x = 2.0 * (coords[:, 0] / (w - 1.0)) - 1.0
        y = 2.0 * (coords[:, 1] / (h - 1.0)) - 1.0
        grid = torch.stack([x, y], dim=-1)  # [B,H,W,2]

        return F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    @staticmethod
    def _length_sq(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x * x, dim=1, keepdim=True)

    @staticmethod
    def _img_grads_gray(x_gray: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_gray: [B,1,H,W]
        dx = x_gray[..., :, 1:] - x_gray[..., :, :-1]
        dy = x_gray[..., 1:, :] - x_gray[..., :-1, :]
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        return dx, dy

    def _fb_occlusion_masks_from_pred(
        self,
        flow_fw: torch.Tensor,
        flow_bw: torch.Tensor,
        *,
        alpha1: float = 0.01,
        alpha2: float = 0.5,
        padding_mode: str = "border",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute occlusion masks via forward-backward check using predicted flows.
        Returns occ_fw, occ_bw in {0,1}, shape [B,1,H,W] (1=occluded).
        """
        flow_bw_warped = self.warp_with_mode(flow_bw, flow_fw, mode="bilinear", padding_mode=padding_mode)
        flow_fw_warped = self.warp_with_mode(flow_fw, flow_bw, mode="bilinear", padding_mode=padding_mode)

        flow_diff_fw = flow_fw + flow_bw_warped
        flow_diff_bw = flow_bw + flow_fw_warped

        mag_sq_fw = self._length_sq(flow_fw) + self._length_sq(flow_bw_warped)
        mag_sq_bw = self._length_sq(flow_bw) + self._length_sq(flow_fw_warped)

        occ_thresh_fw = alpha1 * mag_sq_fw + alpha2
        occ_thresh_bw = alpha1 * mag_sq_bw + alpha2

        occ_fw = (self._length_sq(flow_diff_fw) > occ_thresh_fw).float()
        occ_bw = (self._length_sq(flow_diff_bw) > occ_thresh_bw).float()
        return occ_fw, occ_bw

    # -----------------------------
    # LatentWarp (top-z) helpers
    # -----------------------------
    def _pool_to_top(self, x: torch.Tensor, top_hw: tuple[int, int], *, mode: str = "avg") -> torch.Tensor:
        """Pool a full-res [B,C,H,W] tensor down to [B,C,topH,topW] using integer ratio."""
        topH, topW = int(top_hw[0]), int(top_hw[1])
        H, W = int(x.shape[-2]), int(x.shape[-1])
        assert (H % topH) == 0 and (W % topW) == 0, f"pool requires integer ratio: ({H},{W})->({topH},{topW})"
        sy, sx = H // topH, W // topW
        if mode == "avg":
            return F.avg_pool2d(x, kernel_size=(sy, sx), stride=(sy, sx))
        elif mode == "max":
            return F.max_pool2d(x, kernel_size=(sy, sx), stride=(sy, sx))
        else:
            raise ValueError(f"Unknown pool mode: {mode}")

    def _latentwarp_build_keep_top(
        self,
        *,
        x_prev01: torch.Tensor,      # [B,3,H,W] in [0,1]
        x_cur01: torch.Tensor,       # [B,3,H,W] in [0,1]
        flow_fw: torch.Tensor,       # [B,2,H,W] prev->cur (pixel units)
        flow_bw: torch.Tensor,       # [B,2,H,W] cur->prev (pixel units)
        top_hw: tuple[int, int],     # (topH, topW)
        mask_t: torch.Tensor,        # [B]
        padding_mode: str = "border",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LatentWarp keep mask (paper-style): keep = sigmoid((vis - alpha*resid - thr)/tau).
        Returns:
          flow_bw_top: [B,2,topH,topW] in *top-grid* pixel units
          keep_top:    [B,1,topH,topW] in [0,1] (soft or hard depending on lw_hard_mask)
        """
        if self.lw_detach_flow:
            flow_fw = flow_fw.detach()
            flow_bw = flow_bw.detach()

        # forward-backward occlusion (1=occluded)
        occ_fw, occ_bw = self._fb_occlusion_masks_from_pred(
            flow_fw, flow_bw,
            alpha1=float(self.flow_cfg.fb_alpha1) if self.flow_cfg is not None else 0.01,
            alpha2=float(self.flow_cfg.fb_alpha2) if self.flow_cfg is not None else 0.5,
            padding_mode=padding_mode,
        )
        vis = (1.0 - occ_bw).clamp(0.0, 1.0)  # [B,1,H,W]

        # residual: warp prev frame into current coords using backward flow
        x_prev_in_cur = self.warp_with_mode(
            x_prev01, flow_bw,
            mode="bilinear",
            padding_mode=padding_mode,
        )
        resid = (x_prev_in_cur - x_cur01).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]

        # score -> pooled to top grid
        score = vis - (self.lw_alpha * resid)
        score_top = self._pool_to_top(score, top_hw, mode="avg")

        # keep mask (soft by default)
        keep_top = torch.sigmoid((score_top - self.lw_threshold) / max(self.lw_tau, 1e-6))
        if self.lw_hard_mask:
            keep_top = (keep_top > 0.5).to(dtype=keep_top.dtype)

        # episode boundary mask
        B = keep_top.shape[0]
        keep_top = keep_top * mask_t.view(B, 1, 1, 1).to(dtype=keep_top.dtype, device=keep_top.device)

        # downsample flow to top grid (with correct unit conversion)
        flow_bw_top = downsample_flow(flow_bw, (int(top_hw[0]), int(top_hw[1])))
        return flow_bw_top, keep_top

    def _masked_cosine_align(
        self,
        *,
        pred_map: torch.Tensor,
        target_map: torch.Tensor,
        weight_map: torch.Tensor,
        detach_target: bool = True,
    ) -> torch.Tensor:
        """
        Masked cosine-alignment loss between a predicted top-latent map and a warped target map.
        This aligns direction only and avoids the norm inflation problem of raw dot products.
        """
        tgt = target_map.detach() if detach_target else target_map

        pred = F.normalize(pred_map, dim=1, eps=self.eps)
        tgt = F.normalize(tgt, dim=1, eps=self.eps)

        sim = (pred * tgt).sum(dim=1, keepdim=True)  # [B,1,H,W]
        w = weight_map.detach().to(sim.dtype)

        denom = w.sum().clamp_min(1.0)
        return -(sim * w).sum() / denom


    def _coords_to_flow_pix(self, coords: torch.Tensor) -> torch.Tensor:
        b, _, h, w = coords.shape
        base = coords_grid(b, h, w, device=coords.device, dtype=coords.dtype)
        return coords - base

    def _latent_motion_step(
        self,
        *,
        temporal_state: dict,
        h_context_map: torch.Tensor,
        motion_tokens: torch.Tensor,
        y_warp_in: torch.Tensor,
        a_in: torch.Tensor,
        a_fwd: torch.Tensor = None,
        flow_teacher_fw_64: torch.Tensor = None,
        edge_fullres: torch.Tensor = None,
        mask_t: torch.Tensor = None,
        step: int = 0,
        h64_override: torch.Tensor = None,
    ):
        """Build the canonical 64->tokens->top bridge at one timestep.

        Returns:
          hf_top_map      : [B,Chf,topH,topW]
          motion_ctx_map  : [B,Cmot,topH,topW]
          flow_lat        : [B,2,topH,topW]   (top-grid pixels)
          y_warp_next     : [B,Chf,topH,topW] (forward-splat hf_top_map with flow_lat)
          motion_tokens   : [B,N,D]
          extra_maps      : list of feature maps (constant channels)
          stats           : dict of scalars for logging
        """
        if a_fwd is None:
            a_fwd = a_in

        B = h_context_map.size(0)
        # Reset motion tokens on episode boundaries (mask_t==0)
        if mask_t is not None:
            if mask_t.dim() == 2 and mask_t.shape[1] == 1:
                m = mask_t
            elif mask_t.dim() == 1:
                m = mask_t[:, None]
            else:
                raise ValueError(f"mask_t must be [B] or [B,1], got {tuple(mask_t.shape)}")
            
            keep = m.to(device=motion_tokens.device, dtype=motion_tokens.dtype).view(B, 1, 1)
            base = self.motion_tokens_base.expand(B, -1, -1)
            motion_tokens = motion_tokens * keep + base * (1.0 - keep)

        topH, topW = self.top_H, self.top_W

        # 1) Get 64 features from temporal state (or override for rollout/overshoot parity)
        if h64_override is not None:
            h64_t = h64_override
        else:
            h64_t = temporal_state["64"][0]  # [B,C64,64,64]
        assert h64_t.ndim == 4 and h64_t.shape[-2:] == (64, 64), f"expected h64 [B,C,64,64], got {tuple(h64_t.shape)}"

        # 2) Project HF -> top grid
        hf_top_map = self.hf_to_top(h64_t)
        # 3) Tokenize current HF (grid tokens)
        hf_tokens, token_hw, _xy_tok = self.hf_tokenizer(h64_t)

        Ht, Wt = token_hw
        assert (Ht, Wt) == (64 // self.hf_token_patch, 64 // self.hf_token_patch), \
            (token_hw, self.hf_token_patch)

        # 4) Displacement prior from teacher flow (prev->cur) in 64-grid pixels
        dest_xy_prior = None
        if flow_teacher_fw_64 is not None:
            if flow_teacher_fw_64.shape[-2:] != (64, 64):
                Htch, Wtch = flow_teacher_fw_64.shape[-2:]
                assert Htch >= 64 and Wtch >= 64 and (Htch % 64 == 0) and (Wtch % 64 == 0), \
                    f"teacher flow must be >=64 and divisible to downsample, got {(Htch, Wtch)}"
                flow_teacher_fw_64 = downsample_flow(flow_teacher_fw_64, (64, 64))

            dest_xy_prior, _, _ = self._token_dest_from_flow(flow_teacher_fw_64, token_hw, self.hf_token_patch)
        # 5) Transport tokens with prior bias 
        motion_tokens = self.token_transporter(motion_tokens, hf_tokens, dest_xy_prior=dest_xy_prior, token_hw=token_hw)
        # 6) Tokens -> top map
        motion_ctx_map = self.tokens_to_top(motion_tokens)
        assert motion_ctx_map.shape[-2:] == (topH, topW), (motion_ctx_map.shape, (topH, topW))

        # 7) Edge top map (or zeros)
        if edge_fullres is None:
            # t=0 / reset / edge disabled upstream: keep shapes consistent
            edge_top = h64_t.new_zeros((B, self.edge_top_channels, topH, topW))

        else:
            if edge_fullres.shape[1] == 1:
                edge_in = edge_fullres
            elif edge_fullres.shape[-1] == 1:
                edge_in = edge_fullres.permute(0, 3, 1, 2).contiguous()
            else:
                raise ValueError(f"edge_fullres channel mismatch, expected 1ch, got {tuple(edge_fullres.shape)}")

            # Optional but usually correct: zero out edges for reset items in the batch
            if mask_t is not None:
                if mask_t.dim() == 1:
                    vm = mask_t.view(B, 1, 1, 1).to(dtype=edge_in.dtype, device=edge_in.device)
                else:
                    # if mask_t is already spatial or broadcastable
                    vm = mask_t.to(dtype=edge_in.dtype, device=edge_in.device)
                edge_in = edge_in * vm

            edge_in = edge_in.to(device=h64_t.device, dtype=h64_t.dtype)
            edge_top = self.edge_to_top(edge_in) #(B, self.edge_top_channels, topH, topW)

        # 8) Predict latent top flow (current->next) and warp HF for next-step input
        a_fwd_map = a_fwd.view(B, -1, 1, 1).expand(B, -1, topH, topW)
        flow_lat = self.latent_flow_head(hf_top_map, motion_ctx_map, a_fwd_map, edge_top)  # [B,2,topH,topW]
        y_warp_next, _den = self.forward_splat_bilinear(hf_top_map, flow_lat)

        # 9) Build extra maps (constant list)
        extra_maps = [y_warp_in, hf_top_map, motion_ctx_map, edge_top]

        # stats
        with torch.no_grad():
            mag = flow_lat.square().sum(dim=1).sqrt().mean()
        stats = {"flow_mag": mag}

        return hf_top_map, motion_ctx_map, flow_lat, y_warp_next, motion_tokens, extra_maps, stats

    def _token_dest_from_flow(
        self,
        flow_teacher_fw: torch.Tensor,  # [B,2,64,64] pixel units
        token_hw: tuple,
        patch: int,
    ):
        """
        Per-token displacement prior from teacher flow using NEAREST pixel lookup.
        Returns:
        dest_xy_prior [B,N,2] in token units (float)
        dest_idx      [B,N]   long (rounded/clamped)
        valid         [B,N]   bool (in-bounds before clamp)
        """
        B, _, H, W = flow_teacher_fw.shape
        Ht, Wt = int(token_hw[0]), int(token_hw[1])
        N = Ht * Wt
        assert Ht * patch == H and Wt * patch == W, (H, W, Ht, Wt, patch)

        device = flow_teacher_fw.device
        # token grid coords
        xs = torch.arange(Wt, device=device, dtype=torch.float32)
        ys = torch.arange(Ht, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        x_tok = xx.reshape(-1)  # [N]
        y_tok = yy.reshape(-1)

        # pixel centers (nearest lookup)
        cx = (x_tok * patch + (patch // 2)).round().long().clamp(0, W - 1)  # [N]
        cy = (y_tok * patch + (patch // 2)).round().long().clamp(0, H - 1)

        # gather flow at centers: [B,N]
        fu = flow_teacher_fw[:, 0, cy, cx]  # advanced indexing, no interpolation
        fv = flow_teacher_fw[:, 1, cy, cx]

        # dest in token units (float)
        dest_x = x_tok.view(1, N) + fu / float(patch)
        dest_y = y_tok.view(1, N) + fv / float(patch)
        dest_xy = torch.stack([dest_x, dest_y], dim=-1)  # [B,N,2]

        # rounded index target
        dxr = torch.round(dest_x).long()
        dyr = torch.round(dest_y).long()

        valid = (dxr >= 0) & (dxr < Wt) & (dyr >= 0) & (dyr < Ht)
        dxr = dxr.clamp(0, Wt - 1)
        dyr = dyr.clamp(0, Ht - 1)
        dest_idx = dyr * Wt + dxr  # [B,N]
        return dest_xy, dest_idx, valid

    def _maybe_ckpt(self, fn, *args):
        if self.use_ctx_checkpoint and self.training:
            return ckpt(fn, *args, use_reentrant=False)
        return fn(*args)

    def _predict_flow_one_step(
        self,
        *,
        x01: torch.Tensor,
        e: torch.Tensor,
        ctx: torch.Tensor,
        a_t: torch.Tensor,
        state: torch.Tensor,
        first: bool,
        direction: str,
        x2: Optional[torch.Tensor] = None,
        e2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict flow in pixel units.

        Training (teacher-forced): pass x2/e2 explicitly (pairwise).
        Rollout: one can omit x2/e2 (defaults to x01/e).
        """
        if x2 is None:
            x2 = x01
        if e2 is None:
            e2 = e

        if direction == "fw":
            feat_net = self.flow_feat_net_fw
            enc = self.flow_enc_fw
            dec_in = self.flow_dec_in_fw
            dec = self.flow_dec_fw
        elif direction == "bw":
            feat_net = self.flow_feat_net_bw
            enc = self.flow_enc_bw
            dec_in = self.flow_dec_in_bw
            dec = self.flow_dec_bw
        else:
            raise ValueError(direction)

        def _feat(x01_, x2_, e1_, e2_, ctx_):
            return feat_net(torch.cat([x01_, x2_, e1_, e2_, ctx_], dim=1))
        def _dec_in(x):
            return dec_in(x)

        def _core(x01_, x2_, e1_, e2_, ctx_, a_t_, state_, first_):
            flow_feat = self._maybe_ckpt(_feat, x01_, x2_, e1_, e2_, ctx_)
            state_out = enc(state_, flow_feat, flag=first_, action=a_t_)
            dec_feat = self._maybe_ckpt(_dec_in, state_out)
            coords_list = dec(dec_feat, action=a_t_)["f_flows"]
            flow_pix = self._coords_to_flow_pix(coords_list[-1])
            return flow_pix, state_out

        # checkpoint knob
        flow_pix, state_out = _core(x01, x2, e, e2, ctx, a_t, state, first)

        return flow_pix, state_out


    def _predict_vis_and_residual(
        self,
        *,
        x_cur01: torch.Tensor,
        x_cur_to_prev: torch.Tensor,
        rgb_err: torch.Tensor,
        e_prev: torch.Tensor,
        e_cur_to_prev: torch.Tensor,
        flow_fw: torch.Tensor,
        out_fw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict:
        v_pred: [B,1,H,W] in [0,1] (warp validity)
        r_pred: [B,3,H,W] in [0,1] (inpaint RGB)
        """
        feat = torch.cat(
            [
                x_cur01.detach(),
                x_cur_to_prev.detach(),
                rgb_err.detach(),
                e_prev.detach(),
                e_cur_to_prev.detach(),
                flow_fw.detach(),
                out_fw.detach(),
            ],
            dim=1,
        )
        h = self._maybe_ckpt(self.inpaint_trunk, feat)

        v_pred = torch.sigmoid(self.vis_head(h))
        r_pred = torch.sigmoid(self.rgb_head(h))
        return v_pred, r_pred


    def _compute_flow_warp_losses_pair01(
        self,
        *,
        x_prev01: torch.Tensor,     # [B,3,H,W] in [0,1]
        x_cur01: torch.Tensor,      # [B,3,H,W] in [0,1]
        a_t: torch.Tensor,          # [B,action_dim]
        ctx_map: torch.Tensor,      # [B,ctx_dim,H,W]
        flow_state_fw: torch.Tensor,
        flow_state_bw: torch.Tensor,
        first: bool,
        step: int,
        mask_t: Optional[torch.Tensor] = None,  # [B] or [B,1,1,1] ok
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute flow warp losses for a pair of frames (x_prev01, x_cur01).

        Returns:
        total_loss,
        stats (tensors),
        extras (tensors),
        flow_fw, flow_bw,
        flow_state_fw, flow_state_bw
        """
        cfg = self.flow_cfg
        B, _, H, W = x_prev01.shape
        dtype = x_prev01.dtype
        device = x_prev01.device

        # vm mask
        if mask_t is None:
            vm = torch.ones((B, 1, H, W), device=device, dtype=dtype)
        else:
            vm = mask_t
            if vm.ndim == 1:
                vm = vm[:, None, None, None]
            if vm.shape[-2:] != (H, W):
                vm = vm.expand(B, 1, H, W)
            vm = vm.to(dtype=dtype)

        # edges (normalize per-sample)
        e_prev = self.canny(x_prev01.float()).float()
        e_cur  = self.canny(x_cur01.float()).float()
        e_prev = e_prev / (e_prev.amax(dim=(-2, -1), keepdim=True) + cfg.eps)
        e_cur  = e_cur  / (e_cur.amax(dim=(-2, -1), keepdim=True) + cfg.eps)
        e_prev = e_prev.to(dtype)
        e_cur  = e_cur.to(dtype)

        # flows
        flow_fw, flow_state_fw = self._predict_flow_one_step(
            x01=x_prev01, x2=x_cur01,
            e=e_prev, e2=e_cur,
            ctx=ctx_map, a_t=a_t,
            state=flow_state_fw, first=first, direction="fw"
        )
        flow_bw, flow_state_bw = self._predict_flow_one_step(
            x01=x_cur01, x2=x_prev01,
            e=e_cur, e2=e_prev,
            ctx=ctx_map, a_t=a_t,
            state=flow_state_bw, first=first, direction="bw"
        )

        flow_fw = flow_fw * vm
        flow_bw = flow_bw * vm

        # backward warp x_cur -> x_prev (using flow_fw prev->cur)
        x_cur_to_prev = self.warp_with_mode(
            x_cur01, flow_fw, mode=cfg.rgb_warp_mode, padding_mode=cfg.warp_padding_mode
        )
        e_cur_to_prev = self.warp_with_mode(
            e_cur, flow_fw, mode=cfg.edge_warp_mode, padding_mode=cfg.warp_padding_mode
        )

        # in-bounds mask
        out_fw = create_outgoing_mask(flow_fw) * vm
        m_valid = out_fw.float()

        # occlusion after warmup
        occ_fw = torch.zeros_like(m_valid)
        occ_bw = torch.zeros_like(m_valid)
        if step >= cfg.fb_warmup_steps:
            occ_fw, occ_bw = self._fb_occlusion_masks_from_pred(
                flow_fw, flow_bw,
                alpha1=cfg.fb_alpha1,
                alpha2=cfg.fb_alpha2,
                padding_mode=cfg.warp_padding_mode,
            )

        # pseudo-target visibility
        v_target = (m_valid * (1.0 - occ_fw)).clamp(0.0, 1.0).detach()
        m_sup = v_target

        # census loss (masked)
        ims = x_prev01.float().unsqueeze(1)                            # [B,1,3,H,W]
        ims_warp = x_cur_to_prev.float().unsqueeze(1).unsqueeze(2)     # [B,1,1,3,H,W]
        mask_c = m_sup.float().unsqueeze(1).unsqueeze(2)               # [B,1,1,1,H,W]
        census_loss = self.census_loss(ims, ims_warp, mask_c).to(dtype)

        # gradient loss (masked)
        g_prev = rgb2gray(x_prev01.float()).to(dtype)                  # [B,1,H,W]
        g_warp = rgb2gray(x_cur_to_prev.float()).to(dtype)
        dx1, dy1 = self._img_grads_gray(g_prev)
        dx2, dy2 = self._img_grads_gray(g_warp)
        grad_loss = 0.5 * (
            charbonnier_loss(dx2 - dx1, mask=m_sup, beta=1.0, epsilon=cfg.charbonnier_eps) +
            charbonnier_loss(dy2 - dy1, mask=m_sup, beta=1.0, epsilon=cfg.charbonnier_eps)
        )

        # edge/photo losses (masked)
        edge_w = torch.sigmoid((e_prev - cfg.thr_e) / max(cfg.edge_temp, 1e-6)).to(dtype)
        edge_w = edge_w * vm
        weights = (0.2 + 0.8 * edge_w).clamp(min=0.0)#TODO: tune these constants or make learnable? What is this even doing?
        weights_valid = weights * m_sup

        edge_loss = charbonnier_loss(
            e_cur_to_prev - e_prev,
            mask=weights_valid,
            beta=1.0,
            epsilon=cfg.charbonnier_eps,
        )
        photo_diff = (x_cur_to_prev - x_prev01)
        photo_loss = charbonnier_loss(
            photo_diff,
            mask=weights_valid.expand(-1, 3, -1, -1),
            beta=1.0,
            epsilon=cfg.charbonnier_eps,
        )

        # visibility + residual branch
        rgb_err = photo_diff.abs().mean(dim=1, keepdim=True)
        v_pred, r_pred = self._predict_vis_and_residual(
            x_cur01=x_cur01,
            x_cur_to_prev=x_cur_to_prev,
            rgb_err=rgb_err,
            e_prev=e_prev,
            e_cur_to_prev=e_cur_to_prev,
            flow_fw=flow_fw,
            out_fw=m_valid,
        )

        # visibility BCE with imbalance
        bce = F.binary_cross_entropy(v_pred, v_target, reduction="none")
        w_vis_pix = v_target * 1.0 + (1.0 - v_target) * cfg.vis_neg_weight
        vis_w = w_vis_pix * vm  # vm is [B,1,1,1] broadcastable
        vis_loss = (bce * vis_w).sum() / (vis_w.sum() + self.eps)

        # scheduled sampling blend
        use_target = (step < cfg.blend_use_target_steps)
        v_blend = v_target if use_target else v_pred

        warp_det = x_cur_to_prev.detach()
        x_hat_prev_infer = v_pred * warp_det + (1.0 - v_pred) * r_pred
        x_hat_prev_train = v_blend * warp_det + (1.0 - v_blend) * r_pred

        # inpaint only where invalid
        invalid = (1.0 - v_target) * vm
        inpaint_photo_loss = charbonnier_loss(
            r_pred - x_prev01,
            mask=invalid.expand(-1, 3, -1, -1),
            beta=1.0,
            epsilon=cfg.charbonnier_eps,
        )

        # suppress hallucination where valid
        valid = v_target * vm
        res_suppress_loss = charbonnier_loss(
            r_pred - warp_det,
            mask=valid.expand(-1, 3, -1, -1),
            beta=1.0,
            epsilon=cfg.charbonnier_eps,
        )

        vis_area_loss = (v_pred * vm).sum() / (vm.sum() + self.eps)

        # TV smoothness
        tv_loss = self.flow_tv_loss_fn(flow_fw) + self.flow_tv_loss_fn(flow_bw)

        # optional FB consistency after warmup
        fb_flow_loss = torch.zeros((), device=device, dtype=dtype)
        fb_photo_loss = torch.zeros((), device=device, dtype=dtype)
        fb_enabled = (step >= cfg.fb_warmup_steps) and (cfg.w_fb > 0.0 or cfg.w_fb_photo > 0.0)

        if fb_enabled:
            mask_fw = create_outgoing_mask(flow_fw) * (1.0 - occ_fw) * vm
            mask_bw = create_outgoing_mask(flow_bw) * (1.0 - occ_bw) * vm

            flow_bw_warped = self.warp_with_mode(flow_bw, flow_fw, mode="bilinear", padding_mode=cfg.warp_padding_mode)
            flow_fw_warped = self.warp_with_mode(flow_fw, flow_bw, mode="bilinear", padding_mode=cfg.warp_padding_mode)

            fb_flow_loss = 0.5 * (
                charbonnier_loss(flow_fw + flow_bw_warped, mask=mask_fw, beta=1.0, epsilon=cfg.charbonnier_eps) +
                charbonnier_loss(flow_bw + flow_fw_warped, mask=mask_bw, beta=1.0, epsilon=cfg.charbonnier_eps)
            )

            if cfg.w_fb_photo > 0.0:
                x2_to_1 = self.warp_with_mode(x_cur01, flow_fw, mode="bilinear", padding_mode=cfg.warp_padding_mode)
                x1_to_2 = self.warp_with_mode(x_prev01, flow_bw, mode="bilinear", padding_mode=cfg.warp_padding_mode)
                fb_photo_loss = 0.5 * (
                    charbonnier_loss(x_prev01 - x2_to_1, mask=mask_fw.expand(-1, 3, -1, -1), beta=1.0, epsilon=cfg.charbonnier_eps) +
                    charbonnier_loss(x_cur01 - x1_to_2,   mask=mask_bw.expand(-1, 3, -1, -1), beta=1.0, epsilon=cfg.charbonnier_eps)
                )

        total_loss = (
            cfg.w_edge * edge_loss +
            cfg.w_photo * photo_loss +
            cfg.w_vis * vis_loss +
            cfg.w_inpaint_photo * inpaint_photo_loss +
            cfg.w_res_suppress * res_suppress_loss +
            cfg.w_vis_area * vis_area_loss +
            cfg.w_tv * tv_loss +
            cfg.w_fb * fb_flow_loss +
            cfg.w_fb_photo * fb_photo_loss +
            cfg.w_census * census_loss +
            cfg.w_grad * grad_loss
        )

        stats = {
            "warp_photo": photo_loss.detach(),
            "warp_edge": edge_loss.detach(),
            "warp_tv": tv_loss.detach(),
            "warp_census": census_loss.detach(),
            "warp_grad": grad_loss.detach(),
            "warp_vis": vis_loss.detach(),
            "warp_inpaint_photo": inpaint_photo_loss.detach(),
            "warp_res_suppress": res_suppress_loss.detach(),
            "warp_fb_flow": fb_flow_loss.detach(),
            "warp_fb_photo": fb_photo_loss.detach(),
            "warp_v_target_mean": v_target.mean().detach(),
            "warp_v_pred_mean": v_pred.mean().detach(),
            "warp_use_target_blend": torch.tensor(int(use_target), device=device),
            "warp_fb_enabled": torch.tensor(int(fb_enabled), device=device),
        }

        extras = {
            "x_cur_to_prev": x_cur_to_prev.detach(),
            "x_hat_prev": x_hat_prev_infer.detach(),
            "x_hat_prev_train": x_hat_prev_train.detach(),
            "flow_fw": flow_fw.detach(),
            "flow_bw": flow_bw.detach(),
            "v_target": v_target.detach(),
            "v_pred": v_pred.detach(),
            "r_pred": r_pred.detach(),
            "out_fw": out_fw.detach(),
        }

        return total_loss, stats, extras, flow_fw, flow_bw, flow_state_fw, flow_state_bw

    def forward_splat_bilinear(
        self,
        x: torch.Tensor,
        flow: torch.Tensor,
        metric: torch.Tensor | None = None,
        *,
        radius: float = 1.0,
        normalize: bool = True,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Robust differentiable forward splat (optional soft-splat + optional tent radius).

        Args:
        x:      [B,C,H,W] source features
        flow:   [B,2,H,W] forward flow (dx,dy) in pixel units, source->dest
        metric: [B,1,H,W] optional log-importance; soft-splat uses exp(clamped(metric))
        radius: 1.0 -> standard bilinear (4 corners)
                >1.0 -> separable 2D tent kernel over (2*ceil(radius)+1)^2 neighborhood

        Returns:
        out:   [B,C,H,W] (normalized if normalize=True)
        denom: [B,1,H,W] accumulated mass / importance mass (float32)
        """
        assert x.ndim == 4 and flow.ndim == 4 and flow.size(1) == 2
        B, C, H, W = x.shape
        N = H * W
        device = x.device

        # Accumulate in fp32 for stability even under AMP/bf16.
        acc_dtype = torch.float32
        x_acc = x.to(acc_dtype)
        flow_acc = flow.to(acc_dtype)

        # --- base grid ---
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=acc_dtype),
            torch.arange(W, device=device, dtype=acc_dtype),
            indexing="ij",
        )
        base_x = xx.unsqueeze(0).expand(B, H, W)
        base_y = yy.unsqueeze(0).expand(B, H, W)

        dst_x = (base_x + flow_acc[:, 0]).reshape(B, N)
        dst_y = (base_y + flow_acc[:, 1]).reshape(B, N)

        finite = torch.isfinite(dst_x) & torch.isfinite(dst_y)

        # --- metric importance (soft-splat) ---
        if metric is not None:
            m = metric.to(acc_dtype).reshape(B, N)
            m = m.clamp(-20.0, 20.0)
            src_w = m.exp()  # [B,N]
        else:
            src_w = None

        src = x_acc.reshape(B, C, N)

        def _lin(xi: torch.Tensor, yi: torch.Tensor) -> torch.Tensor:
            xi = xi.clamp(0, W - 1).to(torch.long)
            yi = yi.clamp(0, H - 1).to(torch.long)
            return yi * W + xi  # [B,*]

        out = torch.zeros(B, C, N, device=device, dtype=acc_dtype)
        denom = torch.zeros(B, 1, N, device=device, dtype=acc_dtype)

        if radius <= 1.0 + 1e-6:
            # ---- Standard bilinear (4 corners) ----
            x0 = torch.floor(dst_x)
            y0 = torch.floor(dst_y)
            x1 = x0 + 1
            y1 = y0 + 1

            wa = (x1 - dst_x) * (y1 - dst_y)  # (x0,y0)
            wb = (x1 - dst_x) * (dst_y - y0)  # (x0,y1)
            wc = (dst_x - x0) * (y1 - dst_y)  # (x1,y0)
            wd = (dst_x - x0) * (dst_y - y0)  # (x1,y1)

            def _valid(xi, yi):
                return finite & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)

            va = _valid(x0, y0)
            vb = _valid(x0, y1)
            vc = _valid(x1, y0)
            vd = _valid(x1, y1)

            wa = wa * va.to(acc_dtype)
            wb = wb * vb.to(acc_dtype)
            wc = wc * vc.to(acc_dtype)
            wd = wd * vd.to(acc_dtype)

            ia = _lin(x0, y0)
            ib = _lin(x0, y1)
            ic = _lin(x1, y0)
            id_ = _lin(x1, y1)

            all_idx = torch.cat([ia, ib, ic, id_], dim=1)  # [B,4N]
            all_w = torch.cat([wa, wb, wc, wd], dim=1)     # [B,4N]

            if src_w is not None:
                all_w = all_w * src_w.repeat(1, 4)

            src4 = src.repeat(1, 1, 4)
            all_idx_C = all_idx.unsqueeze(1).expand(B, C, 4 * N)
            all_idx_1 = all_idx.unsqueeze(1)

            out.scatter_add_(2, all_idx_C, src4 * all_w.unsqueeze(1))
            denom.scatter_add_(2, all_idx_1, all_w.unsqueeze(1))

        else:
            # ---- Tent kernel over neighborhood ----
            r = float(radius)
            r_int = int(math.ceil(r))
            offs = torch.arange(-r_int, r_int + 1, device=device, dtype=acc_dtype)  # [K]
            K = offs.numel()
            KK = K * K

            # Center candidates on floor(dst) to avoid discontinuity at .5 boundaries.
            xbase = torch.floor(dst_x)
            ybase = torch.floor(dst_y)

            xi = xbase.unsqueeze(-1) + offs.view(1, 1, K)  # [B,N,K]
            yi = ybase.unsqueeze(-1) + offs.view(1, 1, K)  # [B,N,K]

            wx = (1.0 - (dst_x.unsqueeze(-1) - xi).abs() / r).clamp(min=0.0)  # [B,N,K]
            wy = (1.0 - (dst_y.unsqueeze(-1) - yi).abs() / r).clamp(min=0.0)  # [B,N,K]

            w2 = (wx.unsqueeze(-1) * wy.unsqueeze(-2)).reshape(B, N, KK)  # [B,N,KK]

            xi2 = xi.unsqueeze(-1).expand(B, N, K, K).reshape(B, N, KK)
            yi2 = yi.unsqueeze(-2).expand(B, N, K, K).reshape(B, N, KK)

            v = finite.unsqueeze(-1) & (xi2 >= 0) & (xi2 < W) & (yi2 >= 0) & (yi2 < H)
            w2 = w2 * v.to(acc_dtype)

            if src_w is not None:
                w2 = w2 * src_w.unsqueeze(-1)

            idx = _lin(xi2, yi2).reshape(B, N * KK)   # [B,KK*N]
            w = w2.reshape(B, N * KK)                 # [B,KK*N]

            src_rep = src.repeat(1, 1, KK)            # [B,C,KK*N]
            idx_C = idx.unsqueeze(1).expand(B, C, idx.size(1))
            idx_1 = idx.unsqueeze(1)

            out.scatter_add_(2, idx_C, src_rep * w.unsqueeze(1))
            denom.scatter_add_(2, idx_1, w.unsqueeze(1))

        if normalize:
            out = out / (denom + self.eps)

        out = out.view(B, C, H, W).to(x.dtype)
        denom = denom.view(B, 1, H, W)  # keep float32 for stable thresholding
        return out, denom

    def _init_motion_scaffold(self):
        """
        Motion scaffold matching :
        - fw/bw flow prediction with pairwise inputs (x1,x2,e1,e2,ctx)
        - occlusion masks via forward-backward check
        - visibility head v_pred and residual head r_pred for disocclusions
        - masked photo/edge/census/grad + TV + optional FB consistency losses
        """
        # config container
        self.flow_cfg = FlowCfg(input_channels=self.input_channels, flow_ctx_dim=self.flow_ctx_dim)

        cfg = self.flow_cfg

        # Canny (frozen)
        self.canny = CannyFilter(k_gaussian=3, sigma=1.5, use_cuda=self.device.type == "cuda").to(self.device)
        for p in self.canny.parameters():
            p.requires_grad_(False)
        hidden_flow_proj = int(self.vdvae.decoder.widths[64])
        
        # Context projection from the VDVAE 64-resolution ConvLSTM hidden state
        self.flow_ctx_proj = nn.Sequential(
            nn.Conv2d(hidden_flow_proj, cfg.flow_ctx_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=cfg.flow_ctx_dim, eps=cfg.eps),
            nn.SiLU(inplace=False),
        ).to(self.device)

        # Pairwise flow input: [x1,x2,e1,e2,ctx]
        flow_in_dim = 2 * cfg.input_channels + 2 + cfg.flow_ctx_dim

        def make_feat_net(inputs):
            return nn.Sequential(
                nn.Conv2d(inputs, 64, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=64, eps=cfg.eps),
                nn.SiLU(inplace=False),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=64, eps=cfg.eps),
                nn.SiLU(inplace=False),
            )

        self.flow_feat_net_fw = make_feat_net(inputs =flow_in_dim).to(self.device)
        self.flow_feat_net_bw = make_feat_net(inputs =flow_in_dim).to(self.device)

        self.flow_enc_fw = get_gru_encoder(action_dim=self.action_dim).to(self.device)
        self.flow_enc_bw = get_gru_encoder(action_dim=self.action_dim).to(self.device)

        self.flow_dec_in_fw = nn.Conv2d(cfg.flow_state_channels, cfg.flow_dec_feat_channels, kernel_size=1).to(self.device)
        self.flow_dec_in_bw = nn.Conv2d(cfg.flow_state_channels, cfg.flow_dec_feat_channels, kernel_size=1).to(self.device)

        self.flow_dec_fw = get_gru_decoder(future_len=1, action_dim=self.action_dim).to(self.device)
        self.flow_dec_bw = get_gru_decoder(future_len=1, action_dim=self.action_dim).to(self.device)

        # Keep these (existing fields used elsewhere)
        self.flow_loss_fn = FlowWarpingLoss(metric=torch.nn.L1Loss(reduction="mean"))
        self.flow_tv_loss_fn = TVLoss()
        self.fb_loss_fn = fbLoss

        # Visibility + residual/inpaint head (NEW)
        in_ch = 3 + 3 + 1 + 1 + 1 + 2 + 1  # x_cur, warp(x_cur), rgb_err, e_prev, warp(e_cur), flow_fw, out_fw
        self.inpaint_trunk = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64, eps=cfg.eps),
            nn.SiLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64, eps=cfg.eps),
            nn.SiLU(inplace=False),
        ).to(self.device)
        self.vis_head = nn.Conv2d(64, 1, kernel_size=3, padding=1).to(self.device)
        self.rgb_head = nn.Conv2d(64, 3, kernel_size=3, padding=1).to(self.device)
        self.flow_res = max(int(r) for r in self.vdvae.decoder.temporal_resolutions)  # e.g. 64
        scale = float(self.flow_res) / float(self.image_size)
        self.down_img_to64 = AntiAliasInterpolation2d(channels=3, scale=scale).to(self.device)  # 128->64

        self.hf_top_upsampler = HFTopUpsampler( channels=self.hf_top_channels, target=self.flow_res, min_top=self.top_H, use_checkpoint=self.use_ctx_checkpoint).to(self.device)
        self.mot_top_upsampler = HFTopUpsampler(channels=self.motion_ctx_channels, target=self.flow_res, min_top=self.top_H, use_checkpoint=self.use_ctx_checkpoint).to(self.device)
        # CensusLoss (NEW; used masked like in sanity script)
        cargs = SimpleNamespace(sequence_weight=1.0, iters=1, sequentially=False)
        self.census_loss = CensusLoss(cargs, device=self.device)

    def _init_encoder_decoder(self, max_components: int, prior_alpha: float, prior_beta: float, prior_mc_samples: int = 80):
        """
        Initialize VDVAE + DPGMM prior.

        """
        # ---- 1) Build VDVAE hyperparams ----
        H = Hyperparams()
        # --- Temporal priors (ConvLSTM-conditioned) ---
        H.use_temporal_priors = True
        H.temporal_first_block_only = True  # apply temporal prior only to the first block at each resolution
        H.action_dim = self.action_dim      # required by the ConvLSTM temporal prior

        H.use_checkpoint = self.use_ctx_checkpoint
        H.image_channels = self.input_channels   # usually 3
        H.zdim = self.latent_dim                 # or set explicitly (e.g., 16)
        H.bottleneck_multiple = 0.25
        H.width = 64
        H.image_size = self.image_size          # e.g. 64
        H.dataset = 'imagenet64'
        H.num_mixtures = 10
        H.skip_threshold = 100.0
        H.enc_blocks = "64x2,64d2,32x2,32d2,16x2,16d2,8x2"   # +2 blocks at 8
        H.dec_blocks = "8x2,16m8,16x2,32m16,32x2,64m32,64x2"
        H.attn_resolutions = [16]
        H.use_spatial_attn = True
        H.attn_where = "last"

        H.temporal_n_lstm_layers = 1
        H.temporal_use_orthogonal = True
        H.temporal_kernel_size = 3
        H.use_edge_conditioning = True
        H.edge_condition_min_res = 64

        H.edge_channels = 4
        H.no_bias_above = 64
        H.custom_width_str = ""
        # --- Attention defaults ---
        H.attn_num_layers = 1
        H.attn_num_heads = 4
        H.attn_widening_factor = 1
        H.attn_dropout = 0.0
        H.attn_residual_dropout = 0.0
        H.attn_gn_groups = 32
        H.attn_use_pos_enc = True
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
        res = top_block.base                                # spatial resolution (e.g. 8)
        self.top_zdim = C * res * res
        self.zdim = C #TODO:Is this correct?
        self.top_H = res #TODO:Is this correct?
        self.top_W = res
        # ---- 4) Build DPGMM prior over flattened top-layer tokens ----
        self.prior = DPGMMPrior(
            max_components=max_components,
            latent_dim= self.top_zdim,
            hidden_dim=self.hidden_dim,      # VRNN hidden state dimension
            device=self.device,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            use_checkpoint=self.use_ctx_checkpoint,
        )
        # EMA for VDVAE
        self.ema_decay = 0.999
        self.ema_vdvae = EMA(self.vdvae, decay=self.ema_decay)
        self.C64 = int(self.vdvae.decoder.widths[64])

        # Attach prior to VDVAE so its forward() computes dp_kl / dp_rate
        self.vdvae.prior = self.prior
        self.use_edge_conditioning = bool(getattr(H, "use_edge_conditioning", False))
        self.edge_condition_min_res = int(getattr(H, "edge_condition_min_res", 64))
        self.overshoot_K = int(getattr(H, "overshoot_K", 8))


    def _init_discriminators(self, img_disc_layers:int, patch_size: int, num_heads: int = 4):
        # Initialize discriminators
        self.image_discriminator = TemporalDiscriminator(
                input_channels=self.input_channels,
                image_size=self.image_size,
                hidden_dim=self.hidden_dim,
                n_layers=img_disc_layers,
                n_heads=num_heads,
                max_sequence_length=self.sequence_length,
                patch_size=patch_size,
                z_dim= self.top_zdim + (self.hidden_dim * self.top_H * self.top_W),
                device=self.device,
                use_checkpoint= False,
            )
        self.patch_discriminator = ImageDiscriminator(
                input_nc=self.input_channels,
                ndf=int(self.patch_disc_ndf),
                n_layers=int(self.patch_disc_layers),
                norm_type= "group",
                gn_groups= 32,
                use_checkpoint=self.use_ctx_checkpoint,
                checkpoint_use_reentrant=False,
                device=self.device,
            )


    def _init_vrnn_dynamics(self,use_orthogonal: bool = True, number_lstm_layer: int = 1, extra_channels: int =0):
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
            layer_norm=True,
            use_checkpoint=self.use_ctx_checkpoint,
            extra_channels=extra_channels
        )

    def _init_latent_motion_conditioners(self, base_dim: int = 64):
        # (A) HF feature projection to top grid
        self.hf_to_top = HFToTop(
            in_ch=self.C64,
            out_ch=self.hf_top_channels,
            mid_ch=64,
            target_hw=(self.top_H, self.top_W),
            use_checkpoint=self.use_ctx_checkpoint,
        )

        # (B) Edge projection 
        self.edge_to_top = EdgeToTop(
            out_ch=self.edge_top_channels,
            target_hw=(self.top_H, self.top_W),
            use_checkpoint=self.use_ctx_checkpoint,
        )

        # (C) Tokenization + transport + token->top mapping (canonical bridge)
        self.hf_tokenizer = ConvPatchTokenizer(
            in_ch=self.C64,
            d_tok=self.motion_token_dim,
            patch=self.hf_token_patch,
            use_checkpoint=self.use_ctx_checkpoint,
        )

        self.token_transporter = TokenTransporter(
            d_tok=self.motion_token_dim,
            n_heads=self.motion_token_heads,
            sigma=1.5,
            use_checkpoint=self.use_ctx_checkpoint,
        )

        Ht = 64 // self.hf_token_patch
        Wt = 64 // self.hf_token_patch

        self.tokens_to_top = TokensToTopMap(
            in_ch=self.motion_token_dim,
            out_ch=self.motion_ctx_channels,
            token_hw=(Ht, Wt),
            top_hw=(self.top_H, self.top_W),
            use_checkpoint=self.use_ctx_checkpoint,
        )

        # Base motion tokens used for resets (mask_t==0). Shape is fixed by (64//patch)^2.
        assert 64 % self.hf_token_patch == 0, f"hf_token_patch must divide 64, got {self.hf_token_patch}"
        
        N = Ht * Wt
        self.motion_tokens_base = nn.Parameter(torch.zeros(1, N, self.motion_token_dim))
        latent_flow_max = 1.0

        # (D) Top flow head: predicts flow in *top-grid pixels* (8x8 by default)
        self.latent_flow_head = LatentFlowHead(
            hf_ch=self.hf_top_channels,
            mot_ch=self.motion_ctx_channels,
            act_dim=self.action_dim,
            edge_ch=self.edge_top_channels,
            hidden=self.latent_flow_hidden,
            max_flow=latent_flow_max,
            use_checkpoint=self.use_ctx_checkpoint,
        )

        # (E) Predict next-step 64×64 feature map without future RGB (used for self-supervised flow training)
        hf64_ch = int(self.vdvae.decoder.widths[64])
        self.hf_pred_net = HFMapPredictor(
            z_ch=self.zdim,
            h_ch=self.hidden_dim,
            a_ch=self.action_dim,
            out_ch=hf64_ch,
            out_hw=(64, 64),
            base=base_dim,
            use_checkpoint=self.use_ctx_checkpoint,
        )

        # (F) Refine upsampled top-grid flow into a dense 64×64 flow field
        proj_ch = 32
        g = 8
        if proj_ch % g != 0:
            g = 4 if proj_ch % 4 == 0 else 2 if proj_ch % 2 == 0 else 1

        self.flow64_mot_proj = nn.Sequential(
            nn.Conv2d(self.motion_ctx_channels, proj_ch, 1),
            nn.GroupNorm(g, proj_ch, eps=self.flow_cfg.eps),
            nn.SiLU(inplace=False),
        ).to(self.device)

        self.flow64_feat_proj = nn.Sequential(
            nn.Conv2d(hf64_ch, proj_ch, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=proj_ch, eps=self.flow_cfg.eps),
            nn.SiLU(inplace=False),
        ).to(self.device)

        self.flow64_hf_proj = nn.Sequential(
            nn.Conv2d(self.hf_top_channels, proj_ch, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=proj_ch, eps=self.flow_cfg.eps),
            nn.SiLU(inplace=False),
        ).to(self.device)

        self.flow64_scale = 64 // self.top_H
        self.flow64_upmask = FlowUpsampleMask(
            in_ch=2 + self.hf_top_channels + self.motion_ctx_channels,
            scale=self.flow64_scale,
            use_checkpoint=self.use_ctx_checkpoint,
        )
        self.flow64_refiner = FlowRefiner(
            in_ch=(2 + 3 * proj_ch),
            base=base_dim,                 
            num_blocks=8,
            max_delta=8.0,            # clamp refinement magnitude
            use_checkpoint=self.use_ctx_checkpoint,
        )
        self.max_delta_calib = MaxDeltaCalibrator(q=0.995, frac=0.5, ema=0.98, min_md=2.0, max_md=16.0)


    @property
    def rnn(self):
        """Property to access the RNN layer."""
        return self._rnn

    def _setup_optimizers(self, learning_rate: float, weight_decay: float) -> None:
        def get_params(*modules, exclude_params=None):
            params = []
            exclude_ids = {id(p) for p in (exclude_params or [])}
            seen_ids = set()

            for m in modules:
                if m is None:
                    continue

                if isinstance(m, nn.Module):
                    for p in m.parameters():
                        if (p.requires_grad
                            and id(p) not in exclude_ids
                            and id(p) not in seen_ids):
                            params.append(p)
                            seen_ids.add(id(p))

                elif isinstance(m, nn.Parameter):
                    if (m.requires_grad
                        and id(m) not in exclude_ids
                        and id(m) not in seen_ids):
                        params.append(m)
                        seen_ids.add(id(m))

            return params

        def split_by_weight_decay(params, wd):
            decay, no_decay = [], []
            for p in params:
                if not p.requires_grad:
                    continue
                (decay if p.ndim >= 2 else no_decay).append(p)

            groups = []
            if decay:
                groups.append({"params": decay, "weight_decay": wd})
            if no_decay:
                groups.append({"params": no_decay, "weight_decay": 0.0})
            return groups

        # Choose LR multipliers
        base_lr = learning_rate
        lr_core = base_lr * 1.2
        lr_flow = base_lr * 15.0                  # stronger for motion scaffold (try 2-5x)
        lr_inpaint = base_lr * 15.0               # small head, can go higher (try 3-8x)

        wd_core = weight_decay
        wd_flow = 0.0                            # recommended for flow (often best)
        wd_inpaint = weight_decay * 0.5          # optional

        gen_param_groups = []

        # These should NOT be duplicated across groups.
        gamma_params = [self.prior.stick_breaking.gamma_a, self.prior.stick_breaking.gamma_b]
        scalar_params = [self.rnn.h0, self.rnn.c0, self.rnn.m0]
        exclude = gamma_params + scalar_params

        # -------------------------
        # 1) Core world model params
        # -------------------------
        core_modules = [
            self.vdvae,
            self.rnn,
        ]
        core_params = get_params(*core_modules, exclude_params=exclude)
        if core_params:
            for g in split_by_weight_decay(core_params, wd_core):
                g["lr"] = lr_core
                gen_param_groups.append(g)

        # -------------------------
        # 2) Motion scaffold (flow + ctx)
        # -------------------------
        flow_modules = [
            self.flow_feat_net_fw,
            self.flow_feat_net_bw,
            self.flow_enc_fw,
            self.flow_enc_bw,
            self.flow_dec_in_fw,
            self.flow_dec_in_bw,
            self.flow_dec_fw,
            self.flow_dec_bw,
            self.flow_ctx_proj,
            self.hf_to_top,
            self.edge_to_top,
            self.hf_tokenizer,
            self.token_transporter,
            self.tokens_to_top,
            self.latent_flow_head,
            self.hf_pred_net,
            self.flow64_feat_proj,
            self.flow64_hf_proj,
            self.flow64_mot_proj,
            self.flow64_refiner,
            self.motion_tokens_base,
        ]
        flow_params = get_params(*flow_modules, exclude_params=exclude)
        if flow_params:
            for g in split_by_weight_decay(flow_params, wd_flow):
                g["lr"] = lr_flow
                gen_param_groups.append(g)

        # -------------------------
        # 3) Visibility + inpaint branch
        # -------------------------
        inpaint_modules = [
            self.inpaint_trunk, self.vis_head, self.rgb_head,
        ]
        inpaint_params = get_params(*inpaint_modules, exclude_params=exclude)
        if inpaint_params:
            for g in split_by_weight_decay(inpaint_params, wd_inpaint):
                g["lr"] = lr_inpaint
                gen_param_groups.append(g)

        # 4) Special tiny LR params
        tiny_lr = base_lr * 5e-5
        gammas = [p for p in gamma_params if isinstance(p, nn.Parameter) and p.requires_grad]
        if gammas:
            gen_param_groups.append({"params": gammas, "lr": tiny_lr, "weight_decay": 0.0})

        scalars = [p for p in scalar_params if isinstance(p, nn.Parameter) and p.requires_grad]
        if scalars:
            gen_param_groups.append({"params": scalars, "lr": base_lr, "weight_decay": 1e-5})

        # Optimizer
        self.gen_optimizer = torch.optim.Adamax(
            gen_param_groups,
            lr=lr_core,            # default; groups override anyway
            betas=(0.9, 0.999),
            eps=1e-4,
        )

        # Discriminators unchanged
        if hasattr(self, "image_discriminator") and hasattr(self, "patch_discriminator"):
            disc_param_groups = [
                {"params": self.image_discriminator.parameters(), "lr": base_lr * 0.2},
                {"params": self.patch_discriminator.parameters(), "lr": base_lr * 0.8},
            ]
            self.img_disc_optimizer = torch.optim.Adamax(
                disc_param_groups,
                betas=(0.0, 0.9),
                weight_decay=5e-5,
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

    def _top_prior_from_h_context_map(self, h_context_map):
        # h_context_map: [B, hidden_dim, Ht, Wt]
        B, _, Ht, Wt = h_context_map.shape

        h_map = self.vdvae._coerce_h_context_map(h_context_map, Ht, Wt)
        h_map = self.vdvae.add_PosEncode(h_map, self.vdvae.top_pe, scale=0.05)

        mix, prior_params = self.vdvae.prior(h_map)  
        return mix, prior_params, (B, Ht, Wt)

    def _sample_top_from_prior(self, mix, BHW):
        B, Ht, Wt = BHW
        z_img = mix.sample()  # [B, D] with D = C * Ht * Wt
        if z_img.dim() != 2 or z_img.shape[0] != B:
            raise ValueError(f"Expected image-level prior sample [B,D], got {tuple(z_img.shape)}")

        D = z_img.shape[1]
        HW = Ht * Wt
        if D % HW != 0:
            raise ValueError(f"Sampled latent dim {D} is not divisible by Ht*Wt={HW}")

        C = D // HW
        z_map = z_img.view(B, C, Ht, Wt).contiguous()
        return z_map

    def _upsample_lat_flow_to_64(
        self,
        flow_lat: torch.Tensor,         # [B,2,topH,topW]
        hf_top_map: torch.Tensor,       # [B,Chf,topH,topW]
        motion_ctx_map: torch.Tensor,   # [B,Cmot,topH,topW]
        *,
        mask_t: torch.Tensor | None = None,   # [B] float 0/1
    ) -> torch.Tensor:
        topH, topW = flow_lat.shape[-2:]
        assert 64 % topH == 0 and 64 % topW == 0, f"need integer ratio to 64: {(topH,topW)}"
        sH = 64 // topH
        sW = 64 // topW
        assert sH == sW == self.flow64_scale, (sH, sW, self.flow64_scale)

        # (important) zero-out reset items in order to never warp across episode boundaries
        if mask_t is not None:
            vm = mask_t.view(flow_lat.size(0), 1, 1, 1).to(dtype=flow_lat.dtype, device=flow_lat.device)
            flow_lat = flow_lat * vm

        # mask conditioning at top-grid resolution
        # detaching hf/mot here is optional but often stabilizes training:
        mask_in = torch.cat([flow_lat, hf_top_map.detach(), motion_ctx_map.detach()], dim=1)
        up_mask = self.flow64_upmask(mask_in)  # [B,9*s*s,topH,topW]

        return upsample_flow_raft(flow_lat, up_mask, self.flow64_scale)  # [B,2,64,64]

    def forward_sequence(self, observations, actions=None, dones=None, capture_flow_ctx=None):
        """
        observations: [B, T, C, H, W] in [-1, 1]
        actions:      [B, T, action_dim]
        dones:        [B, T] optional (expects 'done at t-1')

        Returns a dict of per-timestep lists.
        """
        batch_size, seq_len, C, H, W = observations.shape
        device = observations.device
        dtype = observations.dtype
        B = batch_size
        capture_b = capture_t = None
        if capture_flow_ctx is not None:
            capture_b, capture_t = capture_flow_ctx

        if actions is None:
            actions = torch.zeros(batch_size, seq_len, self.action_dim, device=device, dtype=dtype)

        # --- init recurrent states ---
        core_state = self.rnn.init_state(batch_size, device=device, dtype=dtype)
        h_context_map = self.rnn.out_norm(core_state[0])  # [B, hidden, topH, topW]

        temporal_state = self.vdvae.init_temporal_state(batch_size, device=device, dtype=dtype)
        # --- latent motion buffers (persistent across time) ---
        topH, topW = self.top_H, self.top_W
        motion_tokens = self.motion_tokens_base.expand(B, -1, -1).to(device=device, dtype=dtype)  # [B,N,D]
        flow_lat_prev = torch.zeros(B, 2, topH, topW, device=device, dtype=dtype)
        flow64_base_prev = torch.zeros(B, 2, 64, 64, device=device, dtype=dtype)
        flow64_prev = torch.zeros(B, 2, 64, 64, device=device, dtype=dtype)
        y_warp_in = torch.zeros(B, self.hf_top_channels, topH, topW, device=device, dtype=dtype)
        outputs = {
            "reconstructions": [],
            "reconstruction_samples": [],
            "latents": [],
            "hidden_states": [],
            "core_h_maps": [],
            "core_c_maps": [],
            "core_m_maps": [],
            "prior_params": [],
            "reconstruction_losses": [],
            "gauss_rate": [],
            "dp_rate": [],
            "kl_latents": [],
            "elbo": [],
            "pi_token": [],
            "top_q_mean_map": [],
            "top_q_logvar_map": [],
            "kumaraswamy_kl_losses": [],
            "K_eff": [],
            "component_margin": [],

            # legacy warp slots (kept for logging compatibility)
            "warp_lb_loss": [],
            "warp_tv_gate_loss": [],
            "warp_disagree_loss": [],
            "warp_edge_tf_loss": [],
            "warp_rgb_tf_loss": [],

            # sanity total (NEW)
            "warp_sanity_total_loss": [],
            "lat_feat_loss": [],
            "lat_distill_loss": [],

            #"lat_distill64_loss": [],
            "lat_flow_mag": [],
            "motion_tok_ent": [],
            "motion_ctx_ent": [],
            "top_bottom_warp_consistency":[],
            "extra_maps_seq": [],
            # LatentWarp (top-z)
            "z_lw_loss": [],
            "lw_keep_frac": [],
        }

        # teacher-forced images in [0,1] for warp/flow stack
        obs01_all = self.denormalize_generated_images(observations).clamp(0.0, 1.0)
        cfg = self.flow_cfg
        flow_state_C = int(getattr(cfg, "flow_state_channels", 128)) if cfg is not None else 128
        # flow scaffold states
        flow_state_fw = torch.zeros(batch_size, flow_state_C, H, W, device=device, dtype=torch.float32)
        flow_state_bw = torch.zeros(batch_size, flow_state_C, H, W, device=device, dtype=torch.float32)

        step_global = int(self.global_step.item())


        # --- LatentWarp buffers for top z (kept across time, reset by mask_t) ---
        top_q_mean_prev = torch.zeros(B, self.latent_dim, topH, topW, device=device, dtype=dtype)
        have_prev_top = False

        for t in range(seq_len):
            x_t = observations[:, t]  # [-1,1], [B,C,H,W]

            a_t = (
                torch.zeros(batch_size, self.action_dim, device=device, dtype=dtype)
                if t == 0 else actions[:, t - 1]
            )

            if dones is None or t == 0:
                mask_t = torch.ones(batch_size, device=device, dtype=torch.float32)
            else:
                mask_t = (1.0 - dones[:, t - 1].float()).to(torch.float32)

            vm = mask_t.view(batch_size, 1, 1, 1).to(device=device, dtype=dtype)  # float32 mask

            # reset flow states per-episode
            flow_state_fw = flow_state_fw * vm
            flow_state_bw = flow_state_bw * vm
            flow_lat_prev = flow_lat_prev * vm
            flow64_base_prev = flow64_base_prev * vm
            flow64_prev   = flow64_prev   * vm
            y_warp_in     = y_warp_in     * vm
            edge_guide = None
            flow_fw = None
            e_warp = None

            # -------------------------
            # Flow / Warp sanity stack
            # -------------------------
            if self.use_edge_conditioning and (t > 0):
                cfg = self.flow_cfg

                x_prev01 = obs01_all[:, t - 1].float()  # [B,3,H,W] in [0,1]
                x_cur01  = obs01_all[:, t].float()

                # ctx_map from 64-res ConvLSTM hidden BEFORE consuming current frame
                h64_prev, _ = temporal_state["64"]
                ctx64 = self.flow_ctx_proj(h64_prev.float())
                if (ctx64.shape[-2], ctx64.shape[-1]) != (H, W):
                    ctx_map = F.interpolate(ctx64, size=(H, W), mode="bilinear", align_corners=False)
                else:
                    ctx_map = ctx64
                vm = mask_t  # [B] 0/1 float/bool
                vm = vm.view(batch_size, 1, 1, 1).to(flow_state_fw.dtype)  # broadcast mask

                # save previous states
                flow_state_fw_prev = flow_state_fw
                flow_state_bw_prev = flow_state_bw
                # exact sanity loss stack (teacher forced)
                warp_total, stats, extras, flow_fw, flow_bw, flow_state_fw, flow_state_bw = self._compute_flow_warp_losses_pair01(
                    x_prev01=x_prev01,
                    x_cur01=x_cur01,
                    a_t=a_t.float(),
                    ctx_map=ctx_map.float(),
                    flow_state_fw=flow_state_fw,
                    flow_state_bw=flow_state_bw,
                    first=(t == 1),
                    step=step_global,
                    mask_t=vm,  # ok: method expands it
                )
                flow_state_fw = flow_state_fw * vm + flow_state_fw_prev * (1.0 - vm)
                flow_state_bw = flow_state_bw * vm + flow_state_bw_prev * (1.0 - vm)
                # store sanity total and legacy-compatible metrics
                outputs["warp_sanity_total_loss"].append(warp_total)  # keep float32
                outputs["warp_edge_tf_loss"].append(stats["warp_edge"].to(torch.float32))
                outputs["warp_rgb_tf_loss"].append(stats["warp_photo"].to(torch.float32))
                outputs["warp_tv_gate_loss"].append(stats["warp_tv"].to(torch.float32))
                outputs["warp_lb_loss"].append(stats.get("warp_fb_flow", torch.zeros((), device=device)).to(torch.float32))
                outputs["warp_disagree_loss"].append(torch.zeros((), device=device, dtype=torch.float32))

                # build edge_guide (matches generate_future_sequence): forward splat prev -> cur
                x_warp01, denom = self.forward_splat_bilinear(x_prev01, flow_fw)  # already normalized inside
                x_warp01 = x_warp01.clamp(0.0, 1.0)
                valid = (denom > 0.5).to(x_warp01.dtype) * vm  # [B,1,H,W]
                x_warp01 = x_warp01 * valid + x_prev01 * (1 - valid)  # hole-fill with original where denom is bad

                with torch.amp.autocast(device_type="cuda", enabled=False):
                    e_warp = self.canny(x_warp01.float()).float()

                edge_guide = torch.cat([e_warp.to(dtype), x_warp01.to(dtype)], dim=1)
                edge_guide = (edge_guide * vm.to(edge_guide.dtype))

                # optional debug capture for trainer
                if (capture_b is not None) and (capture_t is not None) and (t == capture_t):
                    b = max(0, min(batch_size - 1, capture_b))
                    outputs["captured_flow_ctx"] = ctx_map[b:b + 1].detach()
                    outputs["captured_flow_fw"] = flow_fw[b:b + 1].detach()
                    outputs["captured_flow_bw"] = flow_bw[b:b + 1].detach()

            else:
                # keep list lengths aligned (zeros are tensors, not python floats)
                z = torch.zeros((), device=device, dtype=torch.float32)
                outputs["warp_sanity_total_loss"].append(z)
                outputs["warp_lb_loss"].append(z)
                outputs["warp_tv_gate_loss"].append(z)
                outputs["warp_disagree_loss"].append(z)
                outputs["warp_edge_tf_loss"].append(z)
                outputs["warp_rgb_tf_loss"].append(z)

            # -------------------------
            # Warp core state early (t-1 -> t) so VDVAE conditioning is aligned
            # -------------------------
            if self.warp_core_state and t > 0:
                h, c, m_ = core_state
                vm_map = mask_t.view(batch_size, 1, 1, 1).to(h.dtype)  # [B,1,1,1]

                # forward-splat warp (t-1 -> t)
                h_w, den_h = self.forward_splat_bilinear(h,  flow_lat_prev)
                c_w, den_c = self.forward_splat_bilinear(c,  flow_lat_prev)
                m_w, den_m = self.forward_splat_bilinear(m_, flow_lat_prev)

                # hole-fill: use warped where denom is good, else keep original
                vh = (den_h > 0.5).to(h.dtype) * vm_map
                vc = (den_c > 0.5).to(h.dtype) * vm_map
                vmh = (den_m > 0.5).to(h.dtype) * vm_map

                h  = h_w * vh  + h  * (1 - vh)
                c  = c_w * vc  + c  * (1 - vc)
                m_ = m_w * vmh + m_ * (1 - vmh)

                core_state = (h, c, m_)

            # update context map AFTER warp
            h_context_map = self.rnn.out_norm(core_state[0])
            # -------------------------
            # VDVAE temporal step
            # -------------------------
            x_t_nhwc = x_t.permute(0, 2, 3, 1).contiguous()
            vdvae_out, temporal_state = self.vdvae.forward_temporal_step(
                x_t_nhwc,
                x_t_nhwc,
                h_context=h_context_map,
                a_t=a_t,
                mask_t=mask_t,         # [B] float32
                edge_guide=edge_guide, # [B,4,H,W] or None
                temporal_state=temporal_state,
            )
            prior_params = vdvae_out["prior_params"]

            # Image-level top posterior statistics: q(z_top | x_t) over the full map.
            pi_img = prior_params["pi"]
            top_q_mean_map = vdvae_out["top_q_mean_map"]            # [B,C,Ht,Wt]
            top_q_logsig_map = vdvae_out["top_q_logvar_map"]        # [B,C,Ht,Wt] (log-sigma)

            B2, zdim, Ht, Wt = top_q_mean_map.shape

            # One Monte-Carlo posterior sample for auxiliary component-usage statistics.
            z_top_map = draw_gaussian_diag_samples(top_q_mean_map, top_q_logsig_map)

            # --- LatentWarp top-z consistency (training path; no latent correction) ---
            vm_map = mask_t.view(B, 1, 1, 1).to(dtype=top_q_mean_prev.dtype, device=top_q_mean_prev.device)
            top_q_mean_prev = top_q_mean_prev * vm_map

            z_rnn = z_top_map
            z_lw = z_top_map.new_zeros(())
            keep_frac = z_top_map.new_zeros(())

            do_lw = have_prev_top and (flow_fw is not None) and (flow_bw is not None)
            if do_lw:
                flow_bw_top, keep_top = self._latentwarp_build_keep_top(
                    x_prev01=obs01_all[:, t - 1].float(),
                    x_cur01=obs01_all[:, t].float(),
                    flow_fw=flow_fw,
                    flow_bw=flow_bw,
                    top_hw=(Ht, Wt),
                    mask_t=mask_t,
                    padding_mode=cfg.warp_padding_mode if cfg is not None else "border",
                )

                flow_bw_top_use = flow_bw_top.detach() if self.lw_detach_flow else flow_bw_top
                keep_top_use = keep_top.detach() if self.lw_detach_flow else keep_top
                keep_frac = keep_top_use.mean().detach()

                if self.lambda_z_lw > 0.0:
                    q_prev_w = self.warp_with_mode(
                        top_q_mean_prev.detach(),
                        flow_bw_top_use,
                        mode="bilinear",
                        padding_mode="border",
                    )
                    denom = (keep_top_use.sum().clamp_min(1.0) * float(zdim))
                    z_lw = ((top_q_mean_map - q_prev_w).pow(2) * keep_top_use).sum() / denom

            outputs["z_lw_loss"].append(z_lw)
            outputs["lw_keep_frac"].append(keep_frac)

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
            I = H_marg - H_cond
            outputs["component_margin"].append(I)

            # KL of the stick-breaking variational factors.
            kumar_beta_kl = self.prior.compute_kl_loss(
                prior_params,
                prior_params["alpha"],
                prior_params["h_img"],
            )
            outputs["kumaraswamy_kl_losses"].append(kumar_beta_kl)

            K_eff = self.prior.get_effective_components(pi_img)
            outputs["K_eff"].append(K_eff.float().mean())

            # store z_flat for disc conditioning
            z_flat = z_rnn.permute(0, 2, 3, 1).contiguous().view(B2, Ht * Wt * zdim)

            # update top-level RNN
            extra_maps = None
            flow_teacher_fw_64 = flow_fw if (flow_fw is not None) else None
            edge_fullres = e_warp if (e_warp is not None) else None
            a_fwd = actions[:, t] if t < (seq_len - 1) else torch.zeros_like(a_t)

            hf_top_map, motion_ctx_map, flow_lat_cur, y_warp_next, motion_tokens, extra_maps, lat_stats = self._latent_motion_step(
                temporal_state=temporal_state,
                h_context_map=h_context_map,
                motion_tokens=motion_tokens,
                y_warp_in=y_warp_in,
                a_in=a_t,
                a_fwd=a_fwd,
                flow_teacher_fw_64=flow_teacher_fw_64,
                edge_fullres=edge_fullres,
                mask_t=mask_t,
                step=t,
            )
            outputs["extra_maps_seq"].append([m.detach() for m in extra_maps])

            # motion token / context entropies (for logging)
            tok_ent = getattr(self.token_transporter, "last_attn_entropy", None)
            if tok_ent is None:
                tok_ent = hf_top_map.new_zeros(())
            outputs["motion_tok_ent"].append(tok_ent.detach() if tok_ent.requires_grad else tok_ent)

            with torch.no_grad():
                p_ctx = torch.softmax(motion_ctx_map.float(), dim=1).clamp_min(1e-8)
                ctx_ent = -(p_ctx * p_ctx.log()).sum(dim=1).mean()
            outputs["motion_ctx_ent"].append(ctx_ent.to(dtype=hf_top_map.dtype))

            # refine top-grid flow to dense 64×64 flow (pixel units) for feature transport + distillation
            flow64_base = self._upsample_lat_flow_to_64(
                flow_lat_cur, hf_top_map, motion_ctx_map, mask_t=mask_t
            )

            hf_top_up = self.hf_top_upsampler(hf_top_map)

            mot_up = self.mot_top_upsampler(motion_ctx_map)

            h64_t = temporal_state["64"][0]

            flow64_in = torch.cat([flow64_base, self.flow64_feat_proj(h64_t), self.flow64_hf_proj(hf_top_up), self.flow64_mot_proj(mot_up)], dim=1)
            md = self.max_delta_calib.update(flow64_base)
            self.flow64_refiner.max_delta = md
            flow64_cur = flow64_base + self.flow64_refiner(flow64_in)

            # Top/bottom warp consistency loss (prev->cur at 64×64):
            #   - base flow:   upsample_flow(flow_lat_prev)  (coarse 8×8 -> 64×64)
            #   - refined flow: flow64_prev                  (refined 64×64)
            # We warp x_{t-1} to predict x_t using BOTH flows, then:
            #   (1) enforce base and refined warps agree (agree_l1)
            #   (2) encourage refined to be no worse than base vs ground truth (hinge)
            # This is computed only for t>0 because we need (t-1, t) frames and prev flow.

            if (t > 0) and (float(getattr(self, "lambda_top_bottom_warp", 0.0)) > 0.0):
                # Previous and current frames in [0,1], shape [B,3,H,W]
                x_prev01 = obs01_all[:, t - 1].float()
                x_cur01  = obs01_all[:, t].float()

                # Ensure we compare everything on the same 64×64 grid (the grid where flow64 lives).
                if x_prev01.shape[-2:] == (64, 64):
                    x_prev64 = x_prev01
                    x_cur64  = x_cur01
                else:
                    # Resize images to 64×64 so they match the flow grid.
                    x_prev64 = self.down_img_to64(x_prev01)
                    x_cur64  = self.down_img_to64(x_cur01)

                # Build the "base" 64×64 flow from the previous-step latent flow (t-1 -> t).
                # upsample_flow also scales magnitudes appropriately for the grid ratio.
                #TODO: Should I use upsample_flow_raft here instead, for better quality? It is more expensive but might be worth it for a stronger consistency signal.

                # Warp x_{t-1} forward with:
                #   - base flow (coarse upsampled)
                #   - refined flow 
                # forward_splat_bilinear returns:
                #   x_warp: [B,3,64,64] and denom: [B,1,64,64]
                x_base, den_b = self.forward_splat_bilinear(x_prev64, flow64_base_prev)
                x_ref,  den_r = self.forward_splat_bilinear(x_prev64, flow64_prev)

                # Validity mask: require both splats to have "enough mass" at that pixel.
                # denom>0.5 is a practical threshold that rejects holes from forward splatting.
                valid = ((den_b > 0.5) & (den_r > 0.5)).to(x_base.dtype)  # [B,1,64,64]
                valid2 = valid.expand(-1, 2, -1, -1)                        # [B,2,64,64]
                valid2 = valid2 * mask_t.view(batch_size,1,1,1).to(valid2.dtype)

                flow_agree_l1 = ((flow64_base_prev - flow64_prev).abs() * valid2).sum() / (valid2.sum() + self.eps)

                # add it into loss_flow64 with a small coefficient
                # residual map: where the refined warp actually matches the current frame
                resid = (x_ref - x_cur64).abs().mean(dim=1, keepdim=True)  # [B,1,64,64]

                # confidence mask: "valid splat" AND "small residual"
                thr = 0.15  # in [0,1] space; tune 0.10–0.25
                conf = ((den_b > 0.5) & (den_r > 0.5) & (resid < thr)).to(x_base.dtype)  # [B,1,64,64]
                conf3 = conf.expand(-1, 3, -1, -1)

                vm = mask_t.view(batch_size,1,1,1).to(x_base.dtype)
                conf3 = conf3 * vm

                # then use conf3 instead of valid3 for agree/err terms:
                agree_l1 = (x_base.sub(x_ref).abs() * conf3).sum() / (conf3.sum() + self.eps)
                err_base = (x_base.sub(x_cur64).abs() * conf3).sum() / (conf3.sum() + self.eps)
                err_ref  = (x_ref .sub(x_cur64).abs() * conf3).sum() / (conf3.sum() + self.eps)

                # Penalize refined only when it is worse than base; otherwise no penalty.
                loss_flow64 = agree_l1 + F.relu(err_ref - err_base) + self.beta_flow_agree * flow_agree_l1

                # Store per-step loss (later averaged in compute_total_loss via _mean_or_zero).
                outputs["top_bottom_warp_consistency"].append(loss_flow64)
            else:
                # Keep list lengths consistent across timesteps.
                outputs["top_bottom_warp_consistency"].append(
                    torch.zeros((), device=device, dtype=torch.float32)
                )

            # feature transport supervision at 64 (rollout-safe)
            if t < (seq_len - 1):
                h64_t = temporal_state["64"][0]
                h64_w, den = self.forward_splat_bilinear(h64_t, flow64_cur)
                hf_pred_next = self.hf_pred_net(z_map=z_rnn, h_context_map=h_context_map, a_t=a_fwd, hf_prev=h64_t)
                v = (den > 0.5).to(h64_t.dtype)
                h64_warp_next = h64_w * v + h64_t * (1 - v)
                diff = (h64_warp_next - hf_pred_next).abs()                

                if dones is None:
                    outputs["lat_feat_loss"].append(diff.mean())
                else:
                    # mask transition t->t+1 if episode ends at t
                    mask_fwd = (1.0 - dones[:, t].float()).to(device=device, dtype=diff.dtype)  # [B]
                    m = mask_fwd.view(batch_size, 1, 1, 1)  # [B,1,1,1]
                    denom = m.sum() * diff.shape[1] * diff.shape[2] * diff.shape[3] + self.eps
                    outputs["lat_feat_loss"].append((diff * m).sum() / denom)
            else:
                outputs["lat_feat_loss"].append(hf_top_map.new_zeros(()))
            if t > 0 and flow_teacher_fw_64 is not None:
                flow_teacher_top = downsample_flow(flow_teacher_fw_64, (self.top_H, self.top_W))
                m_prev = mask_t.view(batch_size, 1, 1, 1).to(dtype=flow_lat_prev.dtype, device=device)
                den_top = m_prev.sum() * flow_lat_prev.shape[1] * flow_lat_prev.shape[2] * flow_lat_prev.shape[3] + self.eps
                #den_64  = m_prev.sum() * flow64_prev.shape[1]   * flow64_prev.shape[2]   * flow64_prev.shape[3]   + self.eps

                outputs["lat_distill_loss"].append(((flow_lat_prev - flow_teacher_top.detach()).abs() * m_prev).sum() / den_top)
                teacher_flow64 = flow_teacher_fw_64 if flow_teacher_fw_64.shape[-2:] == (64, 64) else downsample_flow(flow_teacher_fw_64, (64, 64))
                #outputs["lat_distill64_loss"].append(((flow64_prev - teacher_flow64.detach()).abs() * m_prev).sum() / den_64)
            else:
                outputs["lat_distill_loss"].append(hf_top_map.new_zeros(()))
                #outputs["lat_distill64_loss"].append(hf_top_map.new_zeros(()))
            outputs["lat_flow_mag"].append(lat_stats["flow_mag"])
            h_context_map, core_state = self.rnn(z_rnn, a_t, state=core_state, mask_t=mask_t, extra_maps=extra_maps)
            # update LatentWarp buffers for next step
            top_q_mean_prev = top_q_mean_map.detach()
            have_prev_top = True
            flow_lat_prev = flow_lat_cur
            flow64_prev = flow64_cur
            flow64_base_prev = flow64_base
            y_warp_in = y_warp_next

            # store states
            outputs["core_h_maps"].append(core_state[0].detach())
            outputs["core_c_maps"].append(core_state[1].detach())
            outputs["core_m_maps"].append(core_state[2].detach())

            outputs["prior_params"].append({"pi": pi_img.detach()})
            outputs["top_q_mean_map"].append(top_q_mean_map)
            outputs["top_q_logvar_map"].append(top_q_logsig_map)

            # reconstructions
            with torch.no_grad():
                sample = self.vdvae.decoder.out_net.sample(vdvae_out["px_z"])
                sample = torch.from_numpy(sample).to(device=device, dtype=torch.float32)
                sample = sample.permute(0, 3, 1, 2).contiguous()

            dmol_out = self.vdvae.decoder.out_net.forward(vdvae_out["px_z"])
            recon_mean = mean_from_discretized_mix_logistic(dmol_out, self.vdvae.H.num_mixtures)
            recon_mean = recon_mean.permute(0, 3, 1, 2).contiguous()

            outputs["reconstructions"].append(recon_mean)
            outputs["reconstruction_samples"].append(sample)
            outputs["latents"].append(z_flat)
            outputs["hidden_states"].append(
                h_context_map.permute(0, 2, 3, 1).contiguous().view(batch_size, Ht * Wt * self.hidden_dim)
            )
            outputs["reconstruction_losses"].append(vdvae_out["distortion"])
            outputs["gauss_rate"].append(vdvae_out["gauss_rate"])
            outputs["dp_rate"].append(vdvae_out["dp_rate"])
            outputs["kl_latents"].append(vdvae_out["rate"])
            outputs["elbo"].append(vdvae_out["elbo"])
            outputs["pi_token"].append(pi_img.detach())

        # stack some outputs
        for k in ["core_h_maps", "core_c_maps", "core_m_maps"]:
            if len(outputs[k]) > 0:
                outputs[k] = torch.stack(outputs[k], dim=1)  # [B,T,hidden,Ht,Wt]

        for key in ["reconstructions", "latents", "hidden_states", "reconstruction_samples"]:
            if len(outputs[key]) > 0 and isinstance(outputs[key][0], torch.Tensor):
                outputs[key] = torch.stack(outputs[key], dim=1)  # [B,T,...]

        outputs["top_q_mean_maps"] = torch.stack(outputs["top_q_mean_map"], dim=1)
        outputs["top_q_logvar_maps"] = torch.stack(outputs["top_q_logvar_map"], dim=1)

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
        top_q_logsig_maps,    # [B,T,z,Ht,Wt] (log-sigma)
        actions,              # [B,T,A]
        dones=None,
        K=15,
        mc_samples=50,
        w_decay=0.9,
        core_state_maps=None,  # (h,c,m): [B,T,hidden,Ht,Wt]
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

        if core_state_maps is None:
            raise ValueError(
                "Pass core_state_maps=(core_h,core_c,core_m) from forward_sequence to avoid redundant filtering."
            )

        core_h_seq, core_c_seq, core_m_seq = core_state_maps
        if core_h_seq.shape[:2] != (B, T) or core_c_seq.shape[:2] != (B, T) or core_m_seq.shape[:2] != (B, T):
            raise ValueError("core_state_maps must each have shape [B,T,hidden,Ht,Wt].")

        # Use all anchors that could have at least one future step.
        A = T - 1
        anchors = torch.arange(A, device=device)  # 0 .. T-2

        anchor_alive = alive[:, anchors]  # [B,A]
        BA = B * A

        h = core_h_seq[:, anchors].reshape(BA, -1, Ht, Wt).contiguous()
        c = core_c_seq[:, anchors].reshape(BA, -1, Ht, Wt).contiguous()
        m = core_m_seq[:, anchors].reshape(BA, -1, Ht, Wt).contiguous()

        topH, topW = self.top_H, self.top_W
        motion_tokens = self.motion_tokens_base.expand(BA, -1, -1).to(device=device, dtype=dtype)
        flow_lat_prev = torch.zeros(BA, 2, topH, topW, device=device, dtype=dtype)
        y_warp_in = torch.zeros(BA, self.hf_top_channels, topH, topW, device=device, dtype=dtype)
        hf_prev_64 = torch.zeros(BA, self.C64, 64, 64, device=device, dtype=dtype)

        numer = torch.zeros((), device=device, dtype=dtype)
        denom = torch.zeros((), device=device, dtype=dtype)

        for k in range(1, K + 1):
            t_idx = anchors + k                    # [A]
            in_range = (t_idx < T)                 # [A]
            if not in_range.any():
                break

            t_idx_safe = t_idx.clamp(max=T - 1)

            future_alive = torch.zeros(B, A, device=device, dtype=torch.bool)
            future_alive[:, in_range] = alive[:, t_idx[in_range]]

            valid_ba = (anchor_alive & future_alive).reshape(BA)   # [BA]
            if not valid_ba.any():
                continue

            valid_f = valid_ba.to(dtype)

            if actions is None:
                a_ba = torch.zeros(BA, self.action_dim, device=device, dtype=dtype)
            else:
                a_prev = actions[:, (t_idx_safe - 1)].reshape(BA, -1).contiguous()
                a_ba = a_prev * valid_f.unsqueeze(-1)

            h_context_map = self.rnn.out_norm(h)
            
            prior_dist, prior_params, _ = self._top_prior_from_h_context_map(h_context_map)

            q_mu = top_q_mean_maps[:, t_idx_safe].detach().reshape(BA, zdim, Ht, Wt).contiguous()
            q_logsig = top_q_logsig_maps[:, t_idx_safe].detach().reshape(BA, zdim, Ht, Wt).contiguous()
            q_logvar = 2.0 * q_logsig

            q_mu_img = q_mu.view(BA, zdim * Ht * Wt)
            q_lv_img = q_logvar.view(BA, zdim * Ht * Wt)

            kl_ba = self.prior.compute_kl_divergence_mc(
                posterior_mean=q_mu_img,
                posterior_logvar=q_lv_img,
                prior_params=prior_params,
                n_samples=mc_samples,
                reduction="image",
            )  # [BA]

            wk = torch.as_tensor(w_decay ** (k - 1), device=device, dtype=dtype)
            numer = numer + wk * (kl_ba * valid_f).sum()
            denom = denom + wk * valid_f.sum()

            # Differentiable image-level rollout sample for recursive prior prediction.
            z_img = self.vdvae.prior.sample_image_latent(
                prior_params,
                temperature=rollout_temperature,
                grad=True,
                return_stats=False,
            )  # [BA, zdim * Ht * Wt]
            z_samp = z_img.view(BA, zdim, Ht, Wt).contiguous()

            hf_64 = self.hf_pred_net(
                z_map=z_samp,
                h_context_map=h_context_map,
                a_t=a_ba,
                hf_prev=hf_prev_64,
            )
            temporal_fake = {"64": (hf_64, torch.zeros_like(hf_64))}

            if actions is None:
                a_fwd = torch.zeros(BA, self.action_dim, device=device, dtype=dtype)
            else:
                a_next = actions[:, t_idx_safe].reshape(BA, -1).contiguous()
                valid_fwd = future_alive.reshape(BA).to(dtype).unsqueeze(-1)
                a_fwd = a_next * valid_fwd

            _, _, flow_lat_cur, y_warp_next, motion_tokens, extra_maps, _ = self._latent_motion_step(
                temporal_state=temporal_fake,
                h_context_map=h_context_map,
                motion_tokens=motion_tokens,
                y_warp_in=y_warp_in,
                a_in=a_ba,
                a_fwd=a_fwd,
                flow_teacher_fw_64=None,
                edge_fullres=None,
                mask_t=valid_f,
                step=int(k),
                h64_override=hf_64,
            )

            if self.warp_core_state and k > 1:
                vm = valid_f.view(BA, 1, 1, 1)

                h_w, den_h = self.forward_splat_bilinear(h, flow_lat_prev)
                c_w, den_c = self.forward_splat_bilinear(c, flow_lat_prev)
                m_w, den_m = self.forward_splat_bilinear(m, flow_lat_prev)

                vh = (den_h > 0.5).to(dtype) * vm
                vc = (den_c > 0.5).to(dtype) * vm
                vm_ = (den_m > 0.5).to(dtype) * vm

                h = h_w * vh + h * (1.0 - vh)
                c = c_w * vc + c * (1.0 - vc)
                m = m_w * vm_ + m * (1.0 - vm_)

            flow_lat_prev = flow_lat_cur
            y_warp_in = y_warp_next
            hf_prev_64 = hf_64

            _, (h, c, m) = self.rnn(
                z_samp,
                a_ba,
                state=(h, c, m),
                mask_t=valid_f,
                extra_maps=extra_maps,
            )

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
    ):
        outputs = self.forward_sequence(observations, actions, dones)
        device = observations.device

        # --- Reconstruction term ---
        recon_loss = (
            torch.stack(outputs["reconstruction_losses"]).mean()
            if len(outputs["reconstruction_losses"]) > 0
            else torch.zeros((), device=device)
        )

        # ensure stacked tensors for downstream (forward_sequence already stacks, but keep safe)
        if isinstance(outputs.get("reconstructions", None), list):
            outputs["reconstructions"] = torch.stack(outputs["reconstructions"], dim=1)
        if isinstance(outputs.get("reconstruction_samples", None), list):
            outputs["reconstruction_samples"] = torch.stack(outputs["reconstruction_samples"], dim=1)

        # --- Latent KLs ---
        kl_z = (
            torch.stack(outputs["kl_latents"]).mean()
            if len(outputs["kl_latents"]) > 0
            else torch.zeros((), device=device)
        )
        hierarchical_kl = (
            torch.stack(outputs["kumaraswamy_kl_losses"]).mean()
            if len(outputs["kumaraswamy_kl_losses"]) > 0
            else torch.zeros((), device=device)
        )
        component_margin = (
            torch.stack(outputs["component_margin"]).mean()
            if len(outputs["component_margin"]) > 0
            else torch.zeros((), device=device)
        )

        total_vae_loss = (
            lambda_recon * recon_loss
            + beta * (kl_z + hierarchical_kl)
            - component_margin
        )

        # Warp / Flow losses
        z0 = torch.zeros((), device=device)

        def _mean_or_zero(key: str):
            xs = outputs.get(key, [])
            if xs is None or len(xs) == 0:
                return z0
            return torch.stack(xs).mean()

        # sanity total always present after the forward fix (but keep robust)
        use_sanity = ("warp_sanity_total_loss" in outputs) and (len(outputs["warp_sanity_total_loss"]) > 0)

        warp_sanity_total = _mean_or_zero("warp_sanity_total_loss") if use_sanity else z0

        # legacy-compatible scalars
        warp_edge_tf_loss = _mean_or_zero("warp_edge_tf_loss")
        warp_rgb_tf_loss  = _mean_or_zero("warp_rgb_tf_loss")
        warp_lb_loss      = _mean_or_zero("warp_lb_loss")
        warp_tv_gate_loss = _mean_or_zero("warp_tv_gate_loss")
        warp_disagree_loss = _mean_or_zero("warp_disagree_loss")

        if use_sanity:
            # for logging: define "tf" and "reg" in a reasonable way
            cfg = getattr(self, "flow_cfg", None)
            if cfg is not None:
                warp_tf_loss = cfg.w_edge * warp_edge_tf_loss + cfg.w_photo * warp_rgb_tf_loss
            else:
                warp_tf_loss = warp_edge_tf_loss + warp_rgb_tf_loss

            warp_reg_loss = (warp_sanity_total - warp_tf_loss).detach()  # diagnostic
            total_vae_loss = total_vae_loss + warp_sanity_total
        else:
            # fallback to old weighting scheme (keep tensors)
            warp_tf_loss = self.warp_edge_tf_weight * warp_edge_tf_loss + self.warp_rgb_tf_weight * warp_rgb_tf_loss
            warp_reg_loss = (
                self.fb_w * warp_lb_loss
                + self.tv_w * warp_tv_gate_loss
                + self.disagree_w * warp_disagree_loss
            )
            total_vae_loss = total_vae_loss + warp_tf_loss + warp_reg_loss

        # Latent-motion distillation / feature transport losses (always active when enabled)
        lat_feat = self.lambda_lat_feat * _mean_or_zero("lat_feat_loss")
        
        lat_distill = self.lambda_lat_distill * _mean_or_zero("lat_distill_loss")
        #lat_distill64 = self.lambda_lat_distill * _mean_or_zero("lat_distill64_loss")
        top_bottom_warp_consistency = self.lambda_top_bottom_warp * _mean_or_zero("top_bottom_warp_consistency")

        z_lw = self.lambda_z_lw * _mean_or_zero("z_lw_loss")
        total_vae_loss = total_vae_loss + lat_feat + lat_distill + top_bottom_warp_consistency + z_lw
        # -------------------------
        # Overshoot KL (unchanged)
        # -------------------------
        overshoot_kl = torch.zeros((), device=device)
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
                core_state_maps=(outputs["core_h_maps"], outputs["core_c_maps"], outputs["core_m_maps"]),
            )
            topH, topW = outputs["top_q_mean_maps"].shape[-2:]
            _, _, C, H, W = observations.shape
            overshoot_kl = overshoot_kl / float(C * H * W)

        total_vae_loss = total_vae_loss + self.lambda_overshoot * overshoot_kl

        vae_losses = {
            "recon_loss": recon_loss,
            "kl_z": kl_z,
            "hierarchical_kl": hierarchical_kl,
            "component_margin": component_margin,
            "lat_feat": lat_feat,
            "lat_distill": lat_distill,
            #"lat_distill64": lat_distill64,
            "top_bottom_warp_consistency": top_bottom_warp_consistency,
            "z_lw": z_lw,

            "warp_sanity_total": warp_sanity_total,
            "warp_tf_loss": warp_tf_loss,
            "warp_reg_loss": warp_reg_loss,

            "warp_edge_tf_loss": warp_edge_tf_loss,
            "warp_rgb_tf_loss": warp_rgb_tf_loss,
            "warp_lb_loss": warp_lb_loss,
            "warp_tv_gate_loss": warp_tv_gate_loss,
            "warp_disagree_loss": warp_disagree_loss,

            "overshoot_kl": overshoot_kl,
            "total_vae_loss": total_vae_loss,
        }

        return vae_losses, outputs

    def compute_gradient_penalty(self, discriminator, real_x, fake_x, z, device: torch.device, sequence_lengths: Optional[torch.Tensor] = None):
        """
        WGAN-GP with conditioning. Interpolates both x and z (optional).
        Shapes:
        real_x, fake_x: [B, T, C, H, W]
        z: [B, T, Z] (or [B, Z], broadcast inside the disc)
        """
        B = real_x.size(0)
        alpha = torch.rand(B, 1, 1, 1, 1, device=device)

        x_hat = (alpha * real_x + (1 - alpha) * fake_x).requires_grad_(True)  # [B, T, C, H, W]
        # Interpolate z as well (keeps conditioning aligned).
        z_hat = None if z is None else z.detach()

        d_hat = discriminator(x_hat, z=z_hat.detach(), sequence_lengths=sequence_lengths)["final_score"]  # [B, 1]

        grads = torch.autograd.grad(
            outputs=d_hat,
            inputs=x_hat,
            grad_outputs=torch.ones_like(d_hat, device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grads = grads.contiguous().view(B, -1)
        gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
        return gp

    def compute_gradient_penalty_patch(self, D2d, real_x, fake_x, device, mask_flat=None):
        alpha = torch.rand(real_x.size(0), 1, 1, 1, device=device)
        x_hat = (alpha * real_x + (1 - alpha) * fake_x).requires_grad_(True)
        d_hat = D2d(x_hat)                       # [N,1,h,w]
        d_hat = d_hat.mean(dim=(1,2,3))          # [N]
        grads = torch.autograd.grad(
            outputs=d_hat,
            inputs=x_hat,
            grad_outputs=torch.ones_like(d_hat, device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0].view(real_x.size(0), -1)

        gp_per = (grads.norm(2, dim=1) - 1.0).pow(2)  # [N]
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
        latents: torch.Tensor,  #[B, T, Z]
        sequence_lengths: Optional[torch.Tensor] = None,
        WGAN_GP_Coeff: float = 10.0,
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
        real_img_outputs = self.image_discriminator(real_images, z=latents.detach(), sequence_lengths=sequence_lengths)
        fake_img_outputs = self.image_discriminator(fake_images.detach(), z=latents.detach(), sequence_lengths=sequence_lengths)

        # Extract final scores
        real_img_score = real_img_outputs['final_score']
        fake_img_score = fake_img_outputs['final_score']
        # WGAN: maximize real - fake => minimize fake - real
        temporal_disc_loss = fake_img_score.mean() - real_img_score.mean()

        # Gradient penalty requires computing second-order gradients
        img_gp = self.compute_gradient_penalty(
            self.image_discriminator,
            real_images,
            fake_images,
            latents.detach(),
            device=real_images.device,
            sequence_lengths=sequence_lengths,
        )

        # Temporal consistency losses
        img_consistency_loss = torch.zeros((), device=self.device)
        if fake_img_outputs['per_frame_scores'] is not None and fake_img_outputs['per_frame_scores'].numel() > 1:
            diffs = (fake_img_outputs['per_frame_scores'][:,1:] - fake_img_outputs['per_frame_scores'][:,:-1]).abs()
            diffs = diffs.squeeze(-1)  # [B, T-1]
            md = (temporal_mask[:, 1:] & temporal_mask[:, :-1]).float()  # [B, T-1]
            img_consistency_loss = self._masked_mean(diffs, md)

        real_frames = real_images.reshape(B * T, C, H, W).contiguous()
        fake_frames = fake_images.reshape(B * T, C, H, W).contiguous()

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
        real_temporal_for_lc = real_img_score.reshape(-1)   # one score per sequence
        fake_temporal_for_lc = fake_img_score.reshape(-1)

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

        img_disc_loss = temporal_disc_loss + lambda_consistency * img_consistency_loss + WGAN_GP_Coeff *img_gp + patch_disc_loss + WGAN_GP_Coeff * patch_gp + lambda_temporal_lecam * temporal_lecam_loss   + lambda_patch_lecam * patch_lecam_loss

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
            'temporal_gp': img_gp.detach(),
            'temporal_disc_real': real_img_score.mean().detach(),
            'temporal_disc_fake': fake_img_score.mean().detach(),
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

    def compute_feature_matching_loss(self,
                                      real_features: torch.Tensor,
                                      fake_features: torch.Tensor,
                                      temporal_mask: torch.Tensor | None = None):

        if temporal_mask is None:
           return F.l1_loss(fake_features, real_features.detach())
        m = temporal_mask.to(dtype=fake_features.dtype).unsqueeze(-1).unsqueeze(-1) #[B, T, 1, 1]
        diff = torch.abs(fake_features - real_features.detach())  # [B,T,N,D]

        denom = (m.sum() * diff.shape[2] * diff.shape[3]).clamp(min=1.0)
        return (diff * m).sum() / denom

    def compute_adversarial_losses(
        self,
        x: torch.Tensor, #[B, T, C, H, W  ]
        reconstruction: torch.Tensor, #[B, T, C, H, W]
        z_seq: torch.Tensor, #[B, T, Z]
        sequence_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute adversarial losses for both image and latent space
        """
        D = self.image_discriminator
        B, T, C, H, W = reconstruction.shape

        temporal_mask = self._make_temporal_mask(B, T, reconstruction.device, sequence_lengths)
        mask_flat = None
        if sequence_lengths is not None:
            mask_flat = temporal_mask.reshape(B * T).float()      # [B*T]

        flags = [p.requires_grad for p in D.parameters()] #freeze Discriminator parameters
        for p in D.parameters():
            p.requires_grad_(False)
        fake_img_outputs = D(reconstruction, z=z_seq.detach(), sequence_lengths=sequence_lengths, return_features=True)

        real_img_outputs = D(x,             z=z_seq.detach(), sequence_lengths=sequence_lengths, return_features= True)

        temporal_adv_loss = -self._masked_mean(fake_img_outputs['final_score'], temporal_mask)

        # Feature Matching Loss:L1 loss between feature statistics
        feature_match_loss = self.compute_feature_matching_loss(
            real_features=real_img_outputs["hidden_3d"],
            fake_features=fake_img_outputs["hidden_3d"],
            temporal_mask=temporal_mask,
        )
        # Restore Discriminator parameter gradients
        for p, f in zip(D.parameters(), flags):
            p.requires_grad_(f)
        # ---- PatchGAN discriminator (frames) ----
        Dpatch = self.patch_discriminator
        patch_flags = [p.requires_grad for p in Dpatch.parameters()]
        for p in Dpatch.parameters():
            p.requires_grad_(False)

        fake_frames = reconstruction.reshape(B * T, C, H, W).contiguous()
        patch_logits = Dpatch(fake_frames)                 # [B*T,1,h,w]
        patch_scores = patch_logits.mean(dim=(1,2,3))       # [B*T]
        img_adv_loss = -self._masked_mean(patch_scores, mask_flat)


        for p, f in zip(Dpatch.parameters(), patch_flags):
            p.requires_grad_(f)

        return img_adv_loss , temporal_adv_loss, feature_match_loss


    def denormalize_generated_images(self, images):
        """
        Convert generated images from [-1, 1] back to [0, 1] for visualization
        """
        return (images + 1) / 2

    def denormalize(self, images):
        return self.denormalize_generated_images(images)

    def training_step_sequence(self,
                            observations: torch.Tensor,
                            actions: torch.Tensor = None,
                            dones: torch.Tensor = None,
                            beta: float = 1.0,
                            n_critic: int = 3,
                            lambda_img: float = 0.5,
                            lambda_recon: float = 1.0,
                            lambda_edge: float = 0.4,
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
        z_seq_tf = torch.cat([outputs["latents"], outputs["hidden_states"]], dim=-1)

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
                        decode_mode=self.rollout_decode_mode,
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
        rollout_edge_loss = torch.zeros((), device=observations.device)
        rollout_warp_edge_loss = torch.zeros((), device=observations.device)
        rollout_prop_cosine = torch.zeros((), device=observations.device)
        rollout_prop_reg = torch.zeros((), device=observations.device)

        if do_rollout and rollout_horizon > 0:
            # Rollout again WITH grad for generator update
            dbgG = self.generate_future_sequence(
                initial_obs=observations[keep, :T_ctx],
                actions=(actions_slice[keep] if actions_slice is not None else None),
                horizon=rollout_horizon,
                top_temperature=self.rollout_top_temperature,
                decoder_temperature=self.rollout_decoder_temperature,
                decode_mode="sample",  # keep rollout differentiable
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

            edge_fake = self.canny(fake_flat)   # [Bf*Tf,1,H,W]
            edge_real = self.canny(real_flat)

            # mask out padded future frames using seq_len_future
            t = torch.arange(Tf, device=observations.device)[None, :]          # [1,Tf]
            mask_t = (t < seq_len_future[:, None]).float()                     # [Bf,Tf]

            # edge_fake/edge_real: [Bf*Tf, 1, H, W]
            diff = (edge_fake - edge_real).abs().view(Bf, Tf, 1, H, W).mean(dim=(2,3,4))  # [Bf,Tf]
            rollout_edge_loss = (diff * mask_t).sum() / mask_t.sum().clamp(min=1.0)
            # Warp-edge supervision: teach flow to move STRUCTURE (edges) correctly
            # Compare warped edges (from previous frame + predicted flow) to true edges at t+1.
            edge_warp = dbgG.get("edge_warp", None)                 # [Bf,Tf,1,H,W]
            if (edge_warp is not None) and (edge_warp.ndim == 5) and (edge_warp.shape[1] == Tf):
                edge_real_seq = edge_real.view(Bf, Tf, 1, H, W)                    # [Bf,Tf,1,H,W]
                diff_warp = (edge_warp - edge_real_seq).abs().mean(dim=(2, 3, 4))  # [Bf,Tf]
                rollout_warp_edge_loss = (diff_warp * mask_t).sum() / mask_t.sum().clamp(min=1.0)

            rollout_img_adv_loss, rollout_temporal_adv_loss, rollout_feat_match_loss = self.compute_adversarial_losses(
                x=real_future,
                reconstruction=fake_future_G,
                z_seq=z_seq_roll_G,
                sequence_lengths=seq_len_future,
            )
            rollout_prop_cosine = dbgG.get("rollout_prop_cosine", torch.zeros((), device=observations.device))
            rollout_prop_reg = warmup_factor * self.lambda_roll_prop * rollout_prop_cosine

        # 6) Combine losses with DWA or fixed weights
        if self.use_dwa:
            total_components = {
                "recon_loss":      vae_losses["recon_loss"].reshape([]),
                "kl_z":            vae_losses["kl_z"].reshape([]),
                "hierarchical_kl": vae_losses["hierarchical_kl"].reshape([]),
                "overshoot_kl":    vae_losses["overshoot_kl"].reshape([]),
                "component_margin": (-vae_losses["component_margin"]).reshape([]),

                "img_adv_loss":      (warmup_factor * img_adv_loss).reshape([]),
                "temporal_adv_loss": (warmup_factor * temporal_adv_loss).reshape([]),
                "feat_match_loss":   feat_match_loss.reshape([]),

                "rollout_img_adv_loss":      (self.lambda_rollout_adv * warmup_factor * rollout_img_adv_loss).reshape([]),
                "rollout_temporal_adv_loss": (self.lambda_rollout_adv * warmup_factor * rollout_temporal_adv_loss).reshape([]),
                "rollout_feat_match_loss":   (self.lambda_rollout_adv * rollout_feat_match_loss).reshape([]),

                # single warp term
                #"warp_sanity_total": vae_losses["warp_sanity_total"].reshape([]),
                "warp_lb_loss":       vae_losses["warp_lb_loss"].reshape([]),
                "warp_tv_gate_loss":  vae_losses["warp_tv_gate_loss"].reshape([]),
                "warp_disagree_loss": vae_losses["warp_disagree_loss"].reshape([]),
                "warp_edge_tf_loss":  vae_losses["warp_edge_tf_loss"].reshape([]),
                "warp_rgb_tf_loss":   vae_losses["warp_rgb_tf_loss"].reshape([]),
                "lat_feat":      vae_losses["lat_feat"].reshape([]),
                "lat_distill":   vae_losses["lat_distill"].reshape([]),
                #"lat_distill64": vae_losses["lat_distill64"].reshape([]),
                "top_bottom_warp_consistency": vae_losses["top_bottom_warp_consistency"].reshape([]),

                "rollout_edge_loss":      (warmup_factor * lambda_edge * rollout_edge_loss).reshape([]),
                "rollout_warp_edge_loss": (warmup_factor * lambda_edge * rollout_warp_edge_loss).reshape([]),
            }

            total_gen_loss = self.total_weighter.reduce_losses(total_components, batch_idx)
            # LatentWarp z loss: keep fixed weight (not DWA-weighted)
            total_gen_loss = total_gen_loss + vae_losses.get("z_lw", total_gen_loss.new_zeros(()))
            total_gen_loss = total_gen_loss + rollout_prop_reg
        else:
            adv_base = (
                lambda_img_eff * img_adv_loss
                + warmup_factor * temporal_adv_loss
                + lambda_img * feat_match_loss
            )
            adv_roll = self.lambda_rollout_adv * (
                lambda_img_eff * rollout_img_adv_loss
                + warmup_factor * rollout_temporal_adv_loss
                + lambda_img * rollout_feat_match_loss
            )
            elbo_loss = vae_losses["total_vae_loss"]
            total_gen_loss = elbo_loss + adv_base + adv_roll
            total_gen_loss = total_gen_loss + warmup_factor * lambda_edge * rollout_edge_loss
            total_gen_loss = total_gen_loss + warmup_factor * lambda_edge * rollout_warp_edge_loss
            total_gen_loss = total_gen_loss + rollout_prop_reg

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

        eff_comp = outputs["prior_params"][0]["pi"].max(1)[0].mean().item()
        top6_cov = outputs["prior_params"][0]["pi"].topk(6, dim=-1)[0].sum(dim=-1).mean().item()

        del outputs, z_seq_tf, real_future, fake_future_D, z_seq_roll_D, actions_slice, dones_slice, keep

        if do_rollout:
          del dbgG, dbgD, fake_future_G
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
            "rollout_edge_loss": float(rollout_edge_loss.item()),
            "rollout_warp_edge_loss": float(rollout_warp_edge_loss.item()),
            "rollout_prop_cosine": float(rollout_prop_cosine.item()),
            "rollout_prop_reg": float(rollout_prop_reg.item()),
            "total_gen_loss": float(total_gen_loss.item()),
            "grad_norm": float(grad_norm),
            "effective_components": eff_comp,
            "Top 6 coverage": top6_cov,
        }

    def set_epoch(self, epoch: int):
        """Update current epoch for scheduling purposes"""
        self.current_epoch = epoch

    def get_warmup_factor(self) -> float:
        """Calculate warmup factor for adversarial and flow-alignment losses."""
        if self.current_epoch < self.warmup_epochs:
            return self.current_epoch / self.warmup_epochs
        return 1.0

    @torch.no_grad()
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Unconditional-ish samples: pick a context vector h_context, sample top z from DP-GMM prior,
        decode with VDVAE. Returns float tensor in [-1,1], NCHW.
        """
        device = self.device
        # (A) Choose an initial *spatial* context for the top prior / decoder.

        # With SpatioTemporalCore, h_context is a feature map at the top latent resolution.
        dtype = next(self.parameters()).dtype
        core_state = self.rnn.init_state(
            num_samples,
            device=device,
            dtype=dtype,
        )
        h_context = self.rnn.out_norm(core_state[0])  # [B, hidden_dim, 4, 4]

        # (B) apply EMA E
        if hasattr(self, "ema_vdvae"):
            self.ema_vdvae.apply_shadow()

        x_np = self.vdvae.sample(num_samples, h_context)

        if hasattr(self, "ema_vdvae"):
            self.ema_vdvae.restore()

        # uint8 NHWC -> float NCHW in [-1,1]
        x = torch.from_numpy(x_np).permute(0, 3, 1, 2).contiguous().float()/ 127.5 - 1.0
        return x

    def init_core_state(self, B: int, *, device: torch.device, dtype: torch.dtype):
        return self.rnn.init_state(B, device=device, dtype=dtype)

    def reset_core_state_with_mask(self, core_state, mask_t: torch.Tensor):
        if mask_t is None:
            return core_state
        if mask_t.ndim == 1:
            m = mask_t.view(-1, 1, 1, 1)
        else:
            m = mask_t.view(mask_t.size(0), 1, 1, 1)
        m = m.to(dtype=core_state[0].dtype, device=core_state[0].device)
        base = self.rnn.init_state(core_state[0].size(0), device=core_state[0].device, dtype=core_state[0].dtype)
        return tuple(s * m + b * (1.0 - m) for s, b in zip(core_state, base))

    def generate_future_sequence(
        self,
        initial_obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        horizon: int = 15,
        top_temperature: float = 1.0,
        decoder_temperature: float = 1.0,
        decode_mode: str = "sample",
        dones: Optional[torch.Tensor] = None,
        grad: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Warm up on context frames with teacher forcing, then roll forward by sampling
        one full top latent map per image from the image-level DP-GMM prior.

        This version is consistent with image-level prior semantics:
        pi       : [B, K]
        means    : [B, K, C*Ht*Wt]
        log_vars : [B, K, C*Ht*Wt]
        """
        B, T_ctx, C, H, W = initial_obs.shape
        device = initial_obs.device
        dtype = initial_obs.dtype

        if actions is None:
            actions = torch.zeros(B, T_ctx + horizon, self.action_dim, device=device, dtype=dtype)
        elif actions.shape[1] < (T_ctx + horizon):
            pad = (T_ctx + horizon) - actions.shape[1]
            actions = torch.cat(
                [actions, torch.zeros(B, pad, actions.shape[2], device=device, dtype=actions.dtype)],
                dim=1,
            )

        if dones is not None and dones.shape[1] < (T_ctx + horizon):
            pad = (T_ctx + horizon) - dones.shape[1]
            dones = torch.cat([dones, torch.zeros(B, pad, device=device, dtype=dones.dtype)], dim=1)

        def mask_at(t_abs: int) -> torch.Tensor:
            if dones is None or t_abs == 0:
                return torch.ones(B, device=device, dtype=torch.float32)
            return (1.0 - dones[:, t_abs - 1].float()).to(torch.float32)

        initial_obs01 = (initial_obs * 0.5 + 0.5).clamp(0.0, 1.0)

        cfg = self.flow_cfg
        flow_state_C = int(getattr(cfg, "flow_state_channels", 128)) if cfg is not None else 128
        flow_state_fw = torch.zeros(B, flow_state_C, H, W, device=device, dtype=torch.float32)
        flow_state_bw = torch.zeros(B, flow_state_C, H, W, device=device, dtype=torch.float32)

        core_state = self.rnn.init_state(B, device=device, dtype=dtype)
        temporal_state = self.vdvae.init_temporal_state(B, device=device, dtype=dtype)

        topH, topW = self.top_H, self.top_W
        motion_tokens = self.motion_tokens_base.expand(B, -1, -1).to(device=device, dtype=dtype)
        flow_lat_prev = torch.zeros(B, 2, topH, topW, device=device, dtype=dtype)
        flow64_prev = torch.zeros(B, 2, 64, 64, device=device, dtype=dtype)
        flow64_base_prev = torch.zeros(B, 2, 64, 64, device=device, dtype=dtype)
        y_warp_in = torch.zeros(B, self.hf_top_channels, topH, topW, device=device, dtype=dtype)

        mu_top_prev = torch.zeros(B, self.latent_dim, topH, topW, device=device, dtype=dtype)
        have_prev_top = False

        pred_imgs: List[torch.Tensor] = []
        z_seq: List[torch.Tensor] = []
        pi_seq: List[torch.Tensor] = []
        edge_warp_seq: List[torch.Tensor] = []
        rollout_prop_terms: List[torch.Tensor] = []

        prev_img01 = initial_obs01[:, -1]

        with torch.set_grad_enabled(grad):
            for t in range(T_ctx):
                x_t = initial_obs[:, t]
                mask_t = mask_at(t)
                a_t = torch.zeros(B, self.action_dim, device=device, dtype=dtype) if t == 0 else actions[:, t - 1]

                vm_tok = mask_t.view(B, 1, 1)
                vm_map = mask_t.view(B, 1, 1, 1)
                base_tok = self.motion_tokens_base.expand(B, -1, -1).to(device=device, dtype=dtype)

                motion_tokens = motion_tokens * vm_tok + base_tok * (1.0 - vm_tok)
                flow_lat_prev = flow_lat_prev * vm_map
                flow64_prev = flow64_prev * vm_map
                flow64_base_prev = flow64_base_prev * vm_map
                y_warp_in = y_warp_in * vm_map
                mu_top_prev = mu_top_prev * vm_map

                flow_fw = None
                flow_bw = None
                e_warp = None
                edge_guide = None

                if self.use_edge_conditioning and (t > 0):
                    x_prev01 = initial_obs01[:, t - 1].float()
                    x_cur01 = initial_obs01[:, t].float()

                    h64_prev, _ = temporal_state["64"]
                    ctx64 = self.flow_ctx_proj(h64_prev.float())
                    ctx_map = ctx64 if (ctx64.shape[-2], ctx64.shape[-1]) == (H, W) else F.interpolate(
                        ctx64, size=(H, W), mode="bilinear", align_corners=False
                    )

                    vm = vm_map.to(flow_state_fw.dtype)
                    flow_state_fw_prev = flow_state_fw
                    flow_state_bw_prev = flow_state_bw

                    with torch.no_grad():
                        e_prev = self.canny(x_prev01).float()
                        e_cur = self.canny(x_cur01).float()

                    e_prev = e_prev / (e_prev.amax(dim=(-2, -1), keepdim=True) + cfg.eps)
                    e_cur = e_cur / (e_cur.amax(dim=(-2, -1), keepdim=True) + cfg.eps)
                    e_prev = e_prev.to(x_prev01.dtype)
                    e_cur = e_cur.to(x_prev01.dtype)

                    flow_fw, flow_state_fw = self._predict_flow_one_step(
                        x01=x_prev01,
                        x2=x_cur01,
                        e=e_prev,
                        e2=e_cur,
                        ctx=ctx_map,
                        a_t=a_t,
                        state=flow_state_fw,
                        first=(t == 1),
                        direction="fw",
                    )
                    flow_bw, flow_state_bw = self._predict_flow_one_step(
                        x01=x_cur01,
                        x2=x_prev01,
                        e=e_cur,
                        e2=e_prev,
                        ctx=ctx_map,
                        a_t=a_t,
                        state=flow_state_bw,
                        first=(t == 1),
                        direction="bw",
                    )

                    flow_fw = flow_fw * vm
                    flow_bw = flow_bw * vm
                    flow_state_fw = flow_state_fw * vm + flow_state_fw_prev * (1.0 - vm)
                    flow_state_bw = flow_state_bw * vm + flow_state_bw_prev * (1.0 - vm)

                    x_warp01, den = self.forward_splat_bilinear(x_prev01, flow_fw)
                    x_warp01 = x_warp01.clamp(0.0, 1.0)
                    valid = (den > 0.5).to(x_warp01.dtype) * vm
                    x_warp01 = x_warp01 * valid + x_prev01 * (1.0 - valid)

                    with torch.amp.autocast(device_type="cuda", enabled=False):
                        e_warp = self.canny(x_warp01.float()).float()

                    edge_guide = torch.cat([e_warp.to(dtype), x_warp01.to(dtype)], dim=1)
                    edge_guide = edge_guide * vm.to(edge_guide.dtype)

                if self.warp_core_state and (t > 0):
                    h, c, m_ = core_state
                    h = h * vm_map
                    c = c * vm_map
                    m_ = m_ * vm_map

                    h_w, den_h = self.forward_splat_bilinear(h, flow_lat_prev)
                    c_w, den_c = self.forward_splat_bilinear(c, flow_lat_prev)
                    m_w, den_m = self.forward_splat_bilinear(m_, flow_lat_prev)

                    vh = (den_h > 0.5).to(h.dtype) * vm_map
                    vc = (den_c > 0.5).to(c.dtype) * vm_map
                    vmh = (den_m > 0.5).to(m_.dtype) * vm_map

                    core_state = (
                        h_w * vh + h * (1.0 - vh),
                        c_w * vc + c * (1.0 - vc),
                        m_w * vmh + m_ * (1.0 - vmh),
                    )

                h_context_map = self.rnn.out_norm(core_state[0])

                x_t_nhwc = x_t.permute(0, 2, 3, 1).contiguous()
                vdvae_out, temporal_state = self.vdvae.forward_temporal_step(
                    x_t_nhwc,
                    x_t_nhwc,
                    h_context=h_context_map,
                    a_t=a_t,
                    mask_t=mask_t,
                    edge_guide=edge_guide,
                    temporal_state=temporal_state,
                )

                z_used = draw_gaussian_diag_samples(
                    vdvae_out["top_q_mean_map"],
                    vdvae_out["top_q_logvar_map"],
                )

                mu_top_prev = vdvae_out["top_q_mean_map"].detach()
                have_prev_top = True

                a_fwd = actions[:, t] if t < (T_ctx + horizon - 1) else a_t
                hf_top_map, motion_ctx_map, flow_lat_cur, y_warp_next, motion_tokens, extra_maps, _ = self._latent_motion_step(
                    temporal_state=temporal_state,
                    h_context_map=h_context_map,
                    motion_tokens=motion_tokens,
                    y_warp_in=y_warp_in,
                    a_in=a_t,
                    a_fwd=a_fwd,
                    flow_teacher_fw_64=flow_fw if flow_fw is not None else None,
                    edge_fullres=e_warp if e_warp is not None else None,
                    mask_t=mask_t,
                    step=t,
                )

                flow64_base = self._upsample_lat_flow_to_64(
                    flow_lat_cur, hf_top_map, motion_ctx_map, mask_t=mask_t
                )
                hf_top_up = self.hf_top_upsampler(hf_top_map)
                mot_up = self.mot_top_upsampler(motion_ctx_map)
                h64_t = temporal_state["64"][0]

                flow64_in = torch.cat(
                    [
                        flow64_base,
                        self.flow64_feat_proj(h64_t),
                        self.flow64_hf_proj(hf_top_up),
                        self.flow64_mot_proj(mot_up),
                    ],
                    dim=1,
                )
                self.flow64_refiner.max_delta = self.max_delta_calib.update(flow64_base)
                flow64_cur = flow64_base + self.flow64_refiner(flow64_in)

                _, core_state = self.rnn(
                    z_used, a_t, state=core_state, mask_t=mask_t, extra_maps=extra_maps
                )

                flow_lat_prev = flow_lat_cur
                flow64_prev = flow64_cur
                flow64_base_prev = flow64_base
                y_warp_in = y_warp_next
                prev_img01 = initial_obs01[:, t]

            for k in range(horizon):
                t_abs = T_ctx + k
                mask_t = mask_at(t_abs)
                a_t = actions[:, t_abs - 1]

                vm_tok = mask_t.view(B, 1, 1)
                vm_map = mask_t.view(B, 1, 1, 1)
                base_tok = self.motion_tokens_base.expand(B, -1, -1).to(device=device, dtype=dtype)

                motion_tokens = motion_tokens * vm_tok + base_tok * (1.0 - vm_tok)
                flow_lat_prev = flow_lat_prev * vm_map
                flow64_prev = flow64_prev * vm_map
                y_warp_in = y_warp_in * vm_map
                mu_top_prev = mu_top_prev * vm_map

                if self.warp_core_state and (t_abs > 0):
                    h, c, m_ = core_state
                    h = h * vm_map
                    c = c * vm_map
                    m_ = m_ * vm_map

                    h_w, den_h = self.forward_splat_bilinear(h, flow_lat_prev)
                    c_w, den_c = self.forward_splat_bilinear(c, flow_lat_prev)
                    m_w, den_m = self.forward_splat_bilinear(m_, flow_lat_prev)

                    vh = (den_h > 0.5).to(h.dtype) * vm_map
                    vc = (den_c > 0.5).to(c.dtype) * vm_map
                    vmh = (den_m > 0.5).to(m_.dtype) * vm_map

                    core_state = (
                        h_w * vh + h * (1.0 - vh),
                        c_w * vc + c * (1.0 - vc),
                        m_w * vmh + m_ * (1.0 - vmh),
                    )

                h_context_map = self.rnn.out_norm(core_state[0])

                if t_abs > 0:
                    h64_prev, c64_prev = temporal_state["64"]

                    h64_w, den = self.forward_splat_bilinear(h64_prev, flow64_prev)
                    v = (den > 0.5).to(h64_prev.dtype) * vm_map.to(h64_prev.dtype)
                    h64_prev = h64_w * v + h64_prev * (1.0 - v)

                    c64_w, den = self.forward_splat_bilinear(c64_prev, flow64_prev)
                    v = (den > 0.5).to(c64_prev.dtype) * vm_map.to(c64_prev.dtype)
                    c64_prev = c64_w * v + c64_prev * (1.0 - v)

                    temporal_state["64"] = (h64_prev, c64_prev)

                edge_guide = None
                e_warp = None
                if self.use_edge_conditioning:
                    flow_img = flow64_prev
                    if (H, W) != (64, 64):
                        if (H % 64 == 0) and (W % 64 == 0):
                            flow_img = upsample_flow(flow64_prev, (H, W))
                        else:
                            flow_img = F.interpolate(flow64_prev, size=(H, W), mode="bilinear", align_corners=False)
                            flow_img[:, 0] *= float(W) / 64.0
                            flow_img[:, 1] *= float(H) / 64.0

                    x_warp01, den = self.forward_splat_bilinear(prev_img01, flow_img)
                    x_warp01 = x_warp01.clamp(0.0, 1.0)
                    valid = (den > 0.5).to(x_warp01.dtype) * vm_map.to(x_warp01.dtype)
                    x_warp01 = x_warp01 * valid + prev_img01 * (1.0 - valid)

                    with torch.amp.autocast(device_type="cuda", enabled=False):
                        e_warp = self.canny(x_warp01.float()).float()

                    edge_warp_seq.append(e_warp * vm_map.to(e_warp.dtype))
                    edge_guide = torch.cat([e_warp.to(dtype), x_warp01.to(dtype)], dim=1)
                    edge_guide = edge_guide * vm_map.to(edge_guide.dtype)

                _, prior_params, _ = self._top_prior_from_h_context_map(h_context_map)

                pi_seq.append(
                    prior_params["pi"][:, :, None, None].expand(B, -1, topH, topW).detach()
                )

                z_img, stats = self.vdvae.prior.sample_image_latent(
                    prior_params,
                    temperature=top_temperature,
                    grad=grad,
                    return_stats=True,
                )

                z_used = z_img.view(B, self.zdim, topH, topW).contiguous()
                mu_sel_map = stats["mean"].view(B, self.zdim, topH, topW).contiguous()

                roll_prop_step = z_used.new_zeros(())
                if have_prev_top and grad and (self.lambda_roll_prop > 0.0):
                    z_prop, den = self.forward_splat_bilinear(mu_top_prev, flow_lat_prev)
                    keep = (den > 0.5).to(z_used.dtype) * vm_map.to(z_used.dtype)
                    roll_prop_step = self._masked_cosine_align(
                        pred_map=mu_sel_map,
                        target_map=z_prop,
                        weight_map=keep,
                        detach_target=True,
                    )
                rollout_prop_terms.append(roll_prop_step)

                mu_top_prev = mu_sel_map.detach()
                have_prev_top = True

                px_z, temporal_state = self.vdvae.decode_from_top_latent_temporal(
                    z_top_map=z_used,
                    a_t=a_t,
                    mask_t=mask_t,
                    temporal_state=temporal_state,
                    e_warp=edge_guide,
                    temperature=(0.0 if decode_mode == "mean" else decoder_temperature),
                )

                dmol_out = self.vdvae.decoder.out_net.forward(px_z)
                x_hat = mean_from_discretized_mix_logistic(dmol_out, self.vdvae.H.num_mixtures)
                x_hat = x_hat.permute(0, 3, 1, 2).contiguous()

                pred_imgs.append(x_hat)
                z_seq.append(torch.cat([z_used.flatten(1), h_context_map.flatten(1)], dim=-1))

                a_fwd = actions[:, t_abs] if t_abs < (T_ctx + horizon - 1) else a_t
                hf_top_map, motion_ctx_map, flow_lat_cur, y_warp_next, motion_tokens, extra_maps, _ = self._latent_motion_step(
                    temporal_state=temporal_state,
                    h_context_map=h_context_map,
                    motion_tokens=motion_tokens,
                    y_warp_in=y_warp_in,
                    a_in=a_t,
                    a_fwd=a_fwd,
                    flow_teacher_fw_64=None,
                    edge_fullres=e_warp if e_warp is not None else None,
                    mask_t=mask_t,
                    step=t_abs,
                )

                flow64_base = self._upsample_lat_flow_to_64(
                    flow_lat_cur, hf_top_map, motion_ctx_map, mask_t=mask_t
                )
                hf_top_up = self.hf_top_upsampler(hf_top_map)
                mot_up = self.mot_top_upsampler(motion_ctx_map)
                h64_t = temporal_state["64"][0]

                flow64_in = torch.cat(
                    [
                        flow64_base,
                        self.flow64_feat_proj(h64_t),
                        self.flow64_hf_proj(hf_top_up),
                        self.flow64_mot_proj(mot_up),
                    ],
                    dim=1,
                )
                self.flow64_refiner.max_delta = self.max_delta_calib.update(flow64_base)
                flow64_cur = flow64_base + self.flow64_refiner(flow64_in)

                _, core_state = self.rnn(
                    z_used, a_t, state=core_state, mask_t=mask_t, extra_maps=extra_maps
                )

                flow_lat_prev = flow_lat_cur
                flow64_prev = flow64_cur
                flow64_base_prev = flow64_base
                y_warp_in = y_warp_next

                prev_img01 = (x_hat * 0.5 + 0.5).clamp(0.0, 1.0)
                if not grad:
                    prev_img01 = prev_img01.detach()

        vae_future = (
            torch.stack(pred_imgs, dim=1)
            if pred_imgs
            else initial_obs.new_zeros((B, 0, C, H, W))
        )
        z_seq_out = (
            torch.stack(z_seq, dim=1)
            if z_seq
            else initial_obs.new_zeros((B, 0, 1))
        )
        pi_seq_out = (
            torch.stack(pi_seq, dim=1)
            if pi_seq
            else initial_obs.new_zeros((B, 0, 1))
        )
        rollout_prop_cosine = (
            torch.stack(rollout_prop_terms).mean()
            if rollout_prop_terms
            else initial_obs.new_zeros(())
        )

        out = {
            "vae_future": vae_future,
            "z_seq": z_seq_out,
            "pi_seq": pi_seq_out,
            "rollout_prop_cosine": rollout_prop_cosine,
        }
        if edge_warp_seq:
            out["edge_warp"] = torch.stack(edge_warp_seq, dim=1)
        return out
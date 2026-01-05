from logging import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Gamma, Categorical, Independent, MixtureSameFamily
from typing import Dict, Tuple, List, Optional, Any
import sys
import os, gc
from einops import rearrange
from contextlib import contextmanager
from itertools import chain
import numpy as np
import math, inspect
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint as ckpt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vis_networks import EMA, TemporalDiscriminator, AddEpsilon, check_tensor, ImageDiscriminator

from VRNN.RGB import DynamicWeightAverage
from VRNN.lstm import LSTMLayer
from vdvae.vae import VDVAE
from vdvae.hps import Hyperparams
from vdvae.vae_helpers import mean_from_discretized_mix_logistic, sample_from_discretized_mix_logistic, draw_gaussian_diag_samples 
from VRNN.utils.canny_net import CannyFilter
from VRNN.update import FlowHead
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
            ('gamma_ln_out', nn.LayerNorm(2, eps=1e-6)),
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
            ('kumar_a_ln_out', nn.LayerNorm(self.K - 1)),
        ]))


        self.net_b = nn.Sequential(OrderedDict([
            ('kumar_b_fc', nn.utils.spectral_norm(kumar_b_fc)),
            ('kumar_b_ln', nn.LayerNorm(hidden_dim, eps=1e-6)),
            ('kumar_b_relu', nn.GELU()),
            ('kumar_b_out', nn.utils.spectral_norm(kumar_b_out)),
            ('kumar_b_ln_out', nn.LayerNorm(self.K - 1)),

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
        inner_dim = hidden_dim * 2

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
    Implements Dirichlet Process GMM prior for VAE
    """
    def __init__(
        self,
        max_components: int,
        latent_dim: int,
        hidden_dim: int,
        device: torch.device,
        prior_alpha: float = 1.0,  # Add these parameters
        prior_beta: float = 1.0,
    ):
        super().__init__()
        self.max_K = max_components
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Stick-breaking process
        self.stick_breaking = AdaptiveStickBreaking(max_components, hidden_dim, device, prior_alpha=prior_alpha, prior_beta=prior_beta)

        # Component parameters generators
        self.component_nn = ComponentNN(hidden_dim, latent_dim, max_components)

        self.to(device)

    def compute_kl_divergence_mc(
        self,
        posterior_mean: torch.Tensor,
        posterior_logvar: torch.Tensor,
        prior_params: Dict[str, torch.Tensor],
        n_samples: int = 10,
        reduction: str ="mean", #mean |token 
    ) -> torch.Tensor:
        """

        Uses Monte Carlo sampling with reparameterization.
        """
        B_img = prior_params["pi"].shape[0]      # true batch
        N = posterior_mean.shape[0]              # token count
        rep = N // B_img
        assert N == B_img * rep

        prior_weights = prior_params["pi"].repeat_interleave(rep, dim=0)      # [N, K]
        prior_means   = prior_params["means"].repeat_interleave(rep, dim=0)   # [N, K, C]
        prior_logvars = prior_params["log_vars"].repeat_interleave(rep, dim=0)# [N, K, C]

        B, D = posterior_mean.shape
        K = prior_weights.shape[1]
        eps = torch.finfo(torch.float32).eps

        # Sample from posterior (n_samples, B, D)
        noise = torch.randn(n_samples, B, D, device=posterior_mean.device)
        z = posterior_mean.unsqueeze(0) + noise * torch.exp(0.5 * posterior_logvar).unsqueeze(0)

        # Log-posterior: (n_samples, B)
        log_q = -0.5 * (
            D * math.log(2 * math.pi) +
            posterior_logvar.unsqueeze(0).sum(dim=-1) +
            ((z - posterior_mean.unsqueeze(0)) ** 2 * torch.exp(-posterior_logvar).unsqueeze(0)
        ).sum(dim=-1))

        # Log-prior: (n_samples, B, K) -> (n_samples, B)
        z_expanded = z.unsqueeze(2).expand(-1, -1, K, -1)  # (n_samples, B, K, D)
        log_component_densities = -0.5 * (
            D * math.log(2 * math.pi) +
            prior_logvars.sum(dim=-1).unsqueeze(0) +  # (1, B, K)
            ((z_expanded - prior_means.unsqueeze(0)) ** 2 *
            torch.exp(-prior_logvars).unsqueeze(0)).sum(dim=-1)
        )
        log_prior_components = log_component_densities + torch.log(prior_weights.unsqueeze(0).clamp(min=eps)+eps)
        log_p = torch.logsumexp(log_prior_components, dim=2)  # (n_samples, B)

        # KL divergence: average over samples and batch
        kl_samples = (log_q - log_p)  # (n_samples, B)
        kl_token = kl_samples.mean(dim=0)
        if reduction == "token":
            return kl_token
        elif reduction == "mean":
            return kl_token.mean()
        else:
            raise ValueError(f"Unknown reduction={reduction}")

    def compute_kl_loss(self, params: Dict, alpha: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between Kumaraswamy and Beta distributions and prior gamma distributions
        #alpha: [batch_size] or [batch_size, 1]
        #Kumaraswamy params: [batch_size, K-1]
        """
        total_kl = 0.0
        alpha_scalar = alpha.mean(dim=1, keepdim=True) if alpha.dim() > 1 else alpha
        # For each stick-breaking weight
        for k in range(self.max_K - 1):

            kumar_a_k = params['kumar_a'][:, k:k+1]  # [batch_size, 1]
            kumar_b_k = params['kumar_b'][:, k:k+1]  # [batch_size, 1]

            beta_alpha = torch.ones_like(alpha_scalar).squeeze(-1)  # Beta(1,α)
            beta_beta = alpha_scalar.squeeze(-1)  # Current stick's α


            assert kumar_a_k.shape == kumar_b_k.shape == beta_alpha.shape == beta_beta.shape, \
            f"Shape mismatch: kumar_a={kumar_a_k.shape}, kumar_b={kumar_b_k.shape}, " \
            f"beta_alpha={beta_alpha.shape}, beta_beta={beta_beta.shape} at component {k} of {self.max_K}"

            # Compute KL between Kumaraswamy and Beta
            kl = self.stick_breaking.compute_kumar2beta_kl(
                                                           kumar_a_k,
                                                           kumar_b_k,
                                                           beta_alpha,
                                                           beta_beta,
                                                           self.stick_breaking.M
                                                           )
            total_kl = total_kl + kl

        # Add KL for hierarchical prior over alpha (positive or negative?)
        alpha_kl = self.stick_breaking.compute_gamma2gamma_kl(h)
        total_kl = total_kl.mean() + alpha_kl
        return total_kl

    def forward(self, h: torch.Tensor, n_samples: int = 10) -> Tuple[torch.distributions.Distribution, Dict]:
        """
        Generate mixture distribution and return relevant parameters
        """
        batch_size = h.shape[0]
        # Get mixing proportions and Kumaraswamy parameters
        pi, kumar_params = self.stick_breaking(h, use_rand_perm=False)
        # Sample concentration parameter from posterior
        alpha = self.stick_breaking.alpha_posterior.sample(h, n_samples)  # [n_samples]
        # Generate component parameters
        params = self.component_nn(h)
        means, log_vars = torch.split(params, self.latent_dim * self.max_K, dim=1)
        log_vars = torch.clamp(log_vars, min=-10.0, max=2.0)
        # Reshape parameters
        means = means.view(batch_size, self.max_K, self.latent_dim)
        log_vars = log_vars.view(batch_size, self.max_K, self.latent_dim)

        # Create mixture distribution
        mix = Categorical(probs=pi)
        comp = Independent(Normal(means, torch.exp(0.5 * log_vars)), 1)
        mixture = MixtureSameFamily(mix, comp)
        return mixture, {
            'pi': pi,
            'alpha': alpha,
            'means': means,
            'log_vars': log_vars,
            **kumar_params
        }

    def get_effective_components(self, pi: torch.Tensor, threshold: float = 1e-3) -> torch.Tensor:
        """
        Determine effective number of components based on mixing proportions
        """
        sorted_pi, _ = torch.sort(pi, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_pi, dim=-1)
        return torch.sum(cumsum < (1.0 - threshold), dim=-1) + 1

    @staticmethod
    def compute_responsibilities(
        z_tokens: torch.Tensor,              # [N, D]
        prior_params: dict,                  # must contain pi, means, log_vars
        eps: float = 1e-8,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        r_{nk} ∝ pi_{nk} * N(z_n | mu_{nk}, diag(var_{nk})).
        Returns:
          resp: [N, K]
        """
        pi = prior_params["pi"]
        means = prior_params["means"]
        log_vars = prior_params["log_vars"]

        assert z_tokens.dim() == 2, f"z_tokens must be [N,D], got {z_tokens.shape}"
        N, D = z_tokens.shape
        K = pi.shape[-1]

        # If prior params are per-image, repeat to match N tokens
        if pi.shape[0] != N:
            B_img = pi.shape[0]
            assert N % B_img == 0, f"N={N} not divisible by B_img={B_img}"
            rep = N // B_img
            pi = pi.repeat_interleave(rep, dim=0)            # [N,K]
            means = means.repeat_interleave(rep, dim=0)      # [N,K,D]
            log_vars = log_vars.repeat_interleave(rep, dim=0)# [N,K,D]

        # log N(z|mu,var) for diagonal Gaussian:
        # -0.5 * [ D log(2π) + sum_d logvar + sum_d (z-mu)^2 * exp(-logvar) ]
        inv_var = torch.exp(-log_vars).clamp_max(1.0 / eps)  # [N,K,D]
        diff2 = (z_tokens[:, None, :] - means).pow(2)        # [N,K,D]
        maha = (diff2 * inv_var).sum(dim=-1)                 # [N,K]
        log_det = log_vars.sum(dim=-1)                       # [N,K]
        log_gauss = -0.5 * (D * math.log(2.0 * math.pi) + log_det + maha)  # [N,K]

        log_pi = torch.log(pi.clamp_min(eps))                # [N,K]
        logits = (log_pi + log_gauss) / max(temperature, eps)
        resp = torch.softmax(logits, dim=-1)                 # [N,K]
        return resp


@contextmanager
def apply_emas(*emas):
    for e in emas: e.apply_shadow()
    try:
        yield
    finally:
        for e in reversed(emas): e.restore()

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
        grad_clip:float =10.0,
        prior_alpha: float = 1.0,  # Add these parameters
        prior_beta: float = 10.0,
        weight_decay: float = 0.00001,
        use_orthogonal: bool = True,  # Use orthogonal initialization for LSTM,
        number_lstm_layer: int = 3,  # Number of LSTM layers
        warmup_epochs=25,
        dropout: float = 0.1,
        use_ctx_checkpoint: bool = True,
        use_dwa: bool = False,
        dwa_temperature: float = 2.0,
        rollout_adv_every: int = 1,            # do rollout adversarial every N steps (0 disables)
        rollout_context_frames: int = 3,        # T_ctx
        rollout_horizon: int = 4,               # rollout length
        lambda_rollout_adv: float = 0.5,       # strength of rollout adversarial losses
        rollout_top_temperature: float = 0.5,   # sampling temperature for top prior
        rollout_decoder_temperature: float = 1.0,
        rollout_decode_mode: str = "mean",      # "mean" or "sample"
        hidden_flow_proj=96,
        patch_disc_layers=4,
        patch_disc_ndf =32
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

        # rollout GAN attributes
        self.global_step = 0
        self.rollout_adv_every = int(rollout_adv_every)
        self.rollout_context_frames = int(rollout_context_frames)
        self.rollout_horizon = int(rollout_horizon)
        self.lambda_rollout_adv = float(lambda_rollout_adv)
        self.rollout_top_temperature = float(rollout_top_temperature)
        self.rollout_decoder_temperature = float(rollout_decoder_temperature)
        self.rollout_decode_mode = str(rollout_decode_mode)

        # initialization different parts of the model
        self._init_encoder_decoder(max_components, prior_alpha, prior_beta)
        self._init_vrnn_dynamics(use_orthogonal=use_orthogonal,number_lstm_layer=number_lstm_layer)
        self._init_discriminators(img_disc_layers, patch_size, num_heads=disc_num_heads)
        self._init_motion_scaffold(hidden_flow_proj)

        if use_dwa:
            self._init_DynamicWeightAverage(dwa_temperature)

        self.to(device)

        # Setup optimizers
        self._setup_optimizers(learning_rate, weight_decay)

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
            ],
            temperature=temperature,
        )
        
    def _init_motion_scaffold(self, flow_proj:int=64):
        # 1) differentiable canny (returns continuous thin edges)
        self.canny = CannyFilter(k_gaussian=3, sigma=1.5, use_cuda=(self.device.type == "cuda"))
        
        for p in self.canny.parameters():  # redundant but explicit
            p.requires_grad_(False)
        # 2) compress h_context so we can broadcast it spatially (keeps channels reasonable)
        self.flow_ctx_dim = flow_proj
        self.flow_ctx_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.flow_ctx_dim),
            nn.LayerNorm(self.flow_ctx_dim, eps=self.eps)
        )
        # 2.5) project warped edges into decoder feature space 
        dec_width = int(self.vdvae.decoder.out_net.width)
        self.edge_cond_proj = nn.Sequential(
            nn.Conv2d(1, dec_width, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dec_width, dec_width, kernel_size=1),
        )
        # Set to False for ablations
        self.use_edge_conditioning = True
        self.add_edge_to_pxz = False
        # 3) flow head: input = [x_prev (C), edge(1), ctx(64)] => C+1+64
        flow_in_dim = self.input_channels + 1 + self.flow_ctx_dim
        self.flow_head = FlowHead(input_dim=flow_in_dim, hidden_dim=256)  # FlowHead outputs 2-ch flow 

        nn.init.zeros_(self.flow_head.conv2.weight)
        nn.init.zeros_(self.flow_head.conv2.bias)


    def _init_encoder_decoder(self, max_components: int, prior_alpha: float, prior_beta: float, prior_mc_samples: int = 200):
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
        H.width = 192
        H.image_size = self.image_size          # e.g. 64
        H.dataset = 'imagenet64'
        H.num_mixtures = 10
        H.skip_threshold = 100.0
        H.dec_blocks = "4x2,8m4,8x4,16m8,16x4,32m16,32x4,64m32,64x2"
        H.enc_blocks = "64x2,64d2,32x4,32d2,16x4,16d2,8x4,8d2,4x2"
        H.attn_resolutions = [8, 16, 32]
        H.use_spatial_attn = True
        H.attn_where = "last"

        H.temporal_n_lstm_layers = 1
        H.temporal_use_orthogonal = True
        H.temporal_kernel_size = 3
        H.use_edge_conditioning = True
        H.edge_condition_min_res = 64

        H.edge_channels = 1 
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
            latent_dim=C,
            hidden_dim=self.hidden_dim,      # VRNN hidden state dimension
            device=self.device,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
        )
        # EMA for VDVAE
        self.ema_decay = 0.999
        self.ema_vdvae = EMA(self.vdvae, decay=self.ema_decay)

        # Attach prior to VDVAE so its forward() computes dp_kl / dp_rate
        self.vdvae.prior = self.prior


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
                z_dim=self.top_zdim + self.hidden_dim,
                device=self.device,
                use_checkpoint=False,
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

    def _init_vrnn_dynamics(self,use_orthogonal: bool = True, number_lstm_layer: int = 1):
        """Initialize VRNN components with context conditioning"""
        # Feature extractors

        # VRNN recurrence: h_t = f(h_{t-1}, z_t, a_t)
        self._rnn = LSTMLayer(
            input_size=self.top_zdim + self.action_dim,  # z_t + a_t
            hidden_size=self.hidden_dim,
            n_lstm_layers= number_lstm_layer,
            use_orthogonal=use_orthogonal
        )
        self.rnn_layer_norm = nn.LayerNorm(self.hidden_dim)
        # Initialize hidden states
        self.h0 = nn.Parameter(torch.zeros(self.number_lstm_layer, 1, self.hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(self.number_lstm_layer, 1, self.hidden_dim))

    @property
    def rnn(self):
        """Property to access the RNN layer."""
        return self._rnn

    def init_weights(self, module: nn.Module):
        """Initialize the weights using the typical initialization schemes."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.MultiheadAttention):
            # qkv packed in in_proj_weight; out proj separate
            if module.in_proj_weight is not None:
                nn.init.xavier_uniform_(module.in_proj_weight)
            if module.out_proj.weight is not None:
                nn.init.xavier_uniform_(module.out_proj.weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)

        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)

    def warp(self, x, flow):
        # x: [B,C,H,W], flow: [B,2,H,W] in pixels (dx,dy)
        B, C, H, W = x.shape
        # base grid in pixel coords
        yy, xx = torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=0).float()  # [2,H,W]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B,2,H,W]
        coords = grid + flow  # [B,2,H,W]

        # normalize to [-1,1]
        coords_x = 2.0 * (coords[:, 0] / (W - 1)) - 1.0
        coords_y = 2.0 * (coords[:, 1] / (H - 1)) - 1.0
        grid_norm = torch.stack([coords_x, coords_y], dim=-1)  # [B,H,W,2]

        return F.grid_sample(x, grid_norm, mode="bilinear", padding_mode="border", align_corners=True)

 
    def _setup_optimizers(self, learning_rate: float, weight_decay: float) -> None:
        def get_params(*modules, exclude_params=None):
            params = []
            exclude_ids = {id(p) for p in (exclude_params or [])}
            for m in modules:
                if m is None:
                    continue
                if isinstance(m, nn.Module):
                    params.extend(
                        [p for p in m.parameters()
                         if p.requires_grad and id(p) not in exclude_ids]
                    )
                elif isinstance(m, nn.Parameter):
                    if m.requires_grad and id(m) not in exclude_ids:
                        params.append(m)
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

        # 1) Generator param groups
        gen_param_groups = []

        # Trunk / world-model modules
        trunk_modules = [
            self.vdvae.encoder,
            self.vdvae.decoder,
            self.prior,
            self._rnn,
            self.rnn_layer_norm,
            self.flow_ctx_proj,
            self.flow_head,
            self.edge_cond_proj,
        ]
        
        tiny_lr = learning_rate * 5e-5  # e.g. 1000x smaller than base
        gamma_params = [self.prior.stick_breaking.gamma_a, self.prior.stick_breaking.gamma_b]  
        trunk_params = get_params(*trunk_modules, exclude_params=gamma_params)

        if trunk_params:
            gen_param_groups.extend(split_by_weight_decay(trunk_params, weight_decay))
        gammas = [p for p in gamma_params if isinstance(p, nn.Parameter) and p.requires_grad]
        if gammas:
            gen_param_groups.append({
                "params": gammas,
                "lr": tiny_lr,
                "weight_decay": 0.0,   
            })

        # Scalar initial states (h0, c0)
        scalar_params = [self.h0, self.c0]
        scalars = [
            p for p in scalar_params
            if isinstance(p, nn.Parameter) and p.requires_grad
        ]
        if scalars:
            gen_param_groups.append(
                {"params": scalars, "lr": learning_rate, "weight_decay": 1e-4}
            )
        #
        self.gen_optimizer = torch.optim.Adamax(
            gen_param_groups,
            learning_rate*1.2,
            betas=(0.9, 0.999),
            eps=1e-4,
        )
     
        #  4) Discriminator optimizer (unchanged)
        if hasattr(self, "image_discriminator"):
            disc_params =[
                {"params": self.image_discriminator.parameters(), "lr": learning_rate * 0.2},
               {"params": self.patch_discriminator.parameters(), "lr": learning_rate * 0.8},
                ]
            if disc_params:
                self.img_disc_optimizer = torch.optim.Adamax(
                    disc_params,
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

    def forward_sequence(self, observations, actions=None, dones=None):
        """        
        observations: [B, T, C, H, W] in [-1, 1]
        actions:      [B, T, action_dim]
        dones:        [B, T] optional (expects 'done at t-1')

        Returns a dict of per-timestep lists.
        """
        batch_size, seq_len = observations.shape[:2]

        if actions is None:
            actions = torch.zeros(batch_size, seq_len, self.action_dim, device=observations.device, dtype=observations.dtype)

        # Top-level (global) LSTM state (vector)
        h = self.h0.expand(self.number_lstm_layer, batch_size, -1).contiguous()
        c = self.c0.expand(self.number_lstm_layer, batch_size, -1).contiguous()

        # Spatial ConvLSTM temporal states inside the VDVAE decoder
        temporal_state = self.vdvae.init_temporal_state(batch_size, device=observations.device, dtype=observations.dtype)

        outputs = {
            "reconstructions": [],
            "reconstruction_samples": [],
            "latents": [],
            "hidden_states": [],
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
        }

        for t in range(seq_len):
            x_t = observations[:, t]  # [B, C, H, W]
            x_target = x_t
            a_t = actions[:, t]       # [B, action_dim]

            if dones is None:
                mask_t = torch.ones(batch_size, device=observations.device, dtype=observations.dtype)
            else:
                if t == 0:
                    mask_t = torch.ones(batch_size, device=observations.device, dtype=observations.dtype)
                else:
                    mask_t = 1.0 - dones[:, t-1].float()

            # h_context conditions the top DPGMM prior network
            h_context = h[-1]
            edge_guide = None
            if self.use_edge_conditioning and (t > 0):
                # Warp STRUCTURE (edges) from x_{t-1} -> x_t. No RGB copying.
                with torch.no_grad():
                    x_prev = observations[:, t - 1]  # [B,C,H,W] in [-1,1]
                    x_prev01 = self.denormalize_generated_images(x_prev).clamp(0.0, 1.0)
                    e_prev = self.canny(x_prev01)
                    B_, _, H_, W_ = x_prev.shape
                ctx = self.flow_ctx_proj(h_context).view(B_, -1, 1, 1).expand(B_, -1, H_, W_)
                flow_in = torch.cat([x_prev01.detach(), e_prev, ctx], dim=1)
                max_flow = 0.25 * float(max(H_, W_))
                flow = torch.tanh(self.flow_head(flow_in)) * max_flow
                edge_guide = self.warp(e_prev, flow) #[B,1,H,W]
                edge_guide = edge_guide * mask_t.view(batch_size, 1, 1, 1) 
            elif self.use_edge_conditioning and (t == 0):
                B_, _, H_, W_ = x_t.shape
                with torch.no_grad():
                    edge_guide = torch.zeros(B_, 1, H_, W_, device=x_t.device, dtype=x_t.dtype)
            x_t_nhwc = x_t.permute(0, 2, 3, 1).contiguous()
            x_target_nhwc = x_target.permute(0, 2, 3, 1).contiguous()

            vdvae_out, temporal_state = self.vdvae.forward_temporal_step(
                x_t_nhwc,
                x_target_nhwc,
                h_context=h_context,
                a_t=a_t,
                mask_t=mask_t,
                edge_guide=edge_guide,
                temporal_state=temporal_state,
            )
            prior_params = vdvae_out["prior_params"]

            # --- (A) Pull prior params safely ---
            pi_tok = prior_params["pi"]                  # [N,K] or [B,K]
            mu_tok = prior_params["means"]               # [N,K,C]
            logvar_tok = prior_params["log_vars"]        # [N,K,C]  (this is log-variance)
            var_tok = torch.exp(logvar_tok)

            # --- (B) top posterior maps ---
            top_q_mean_map = vdvae_out["top_q_mean_map"]       # [B,C,Ht,Wt]
            top_q_logsigma_map = vdvae_out["top_q_logvar_map"] # NOTE: this is log-sigma 

            B2, zdim, Ht, Wt = top_q_mean_map.shape
            tokens_per_img = Ht * Wt
            N = B2 * tokens_per_img

            # --- (C) token samples for MI term  ---
            top_q_mean_tokens = top_q_mean_map.permute(0, 2, 3, 1).contiguous().view(N, zdim)
            top_q_logsigma_tokens = top_q_logsigma_map.permute(0, 2, 3, 1).contiguous().view(N, zdim)

            z_tokens = top_q_mean_tokens + torch.randn_like(top_q_mean_tokens) * torch.exp(top_q_logsigma_tokens)

            eps_ = torch.finfo(z_tokens.dtype).eps
            resp = self.prior.compute_responsibilities(
                z_tokens=z_tokens,
                prior_params=prior_params,
                eps=eps_,
                temperature=1.0,
            )  # [N,K]

            u = resp.mean(dim=0).clamp_min(eps_)
            H_marg = -(u * u.log()).sum()
            r = resp.clamp_min(eps_)
            H_cond = - (r * r.log()).sum(dim=-1).mean()
            I = H_marg - H_cond
            outputs["component_margin"].append(I)

            # (D) Kumaraswamy KL: build token-wise h_tokens 
            Hc = h_context.shape[1]
            h_map = h_context.view(B2, -1, 1, 1).expand(B2, -1, Ht, Wt).contiguous()
            h_map = self.vdvae.add_coord_no_proj(h_map, scale=0.05)  
            h_tokens = h_map.permute(0, 2, 3, 1).contiguous().view(N, Hc)

            kumar_beta_kl = self.prior.compute_kl_loss(prior_params, prior_params["alpha"], h_tokens)
            outputs["kumaraswamy_kl_losses"].append(kumar_beta_kl)

            K_eff = self.prior.get_effective_components(pi_tok)  # token-level is OK; or average per-image first
            outputs["K_eff"].append(K_eff.float().mean())

            # --- (E) sample z_top_map correctly (logσ!) ---
            z_top_map = draw_gaussian_diag_samples(top_q_mean_map, top_q_logsigma_map)  

            z_flat = z_top_map.permute(0, 2, 3, 1).reshape(B2, Ht * Wt * zdim)
            rnn_in = torch.cat([z_flat, a_t], dim=-1)
            rnn_out, (h, c) = self.rnn(rnn_in, h, c, mask_t)
            
            outputs["prior_params"].append({
                "pi": pi_tok.detach(),   # [B, K]
            })
            outputs["top_q_mean_map"].append(top_q_mean_map.detach().cpu())
            outputs["top_q_logvar_map"].append(top_q_logsigma_map.detach().cpu())
            with torch.no_grad():
                sample = self.vdvae.decoder.out_net.sample(vdvae_out["px_z"])   # [B, H, W, C] uint8
                sample = torch.from_numpy(sample).to(device=self.device, dtype=torch.float32)
                sample = sample.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
            
            dmol_out = self.vdvae.decoder.out_net.forward(vdvae_out["px_z"])  # [B, H, W, 10*nr_mix]
            recon_mean = mean_from_discretized_mix_logistic(dmol_out, self.vdvae.H.num_mixtures)  # [B,H,W,3]
            #print(f"image reconstruction range: min {recon_mean.min().item()}, max {recon_mean.max().item()}")
            recon_mean = recon_mean.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
            outputs["reconstructions"].append(recon_mean)

            outputs["reconstruction_samples"].append(sample)
            outputs["latents"].append(z_flat)
            outputs["hidden_states"].append(h[-1])
            outputs["reconstruction_losses"].append(vdvae_out["distortion"])
            outputs["gauss_rate"].append(vdvae_out["gauss_rate"])
            outputs["dp_rate"].append(vdvae_out["dp_rate"])
            outputs["kl_latents"].append(vdvae_out["rate"])
            outputs["elbo"].append(vdvae_out["elbo"])
            outputs["pi_token"].append(pi_tok)

        for key in ["reconstructions", "latents", "hidden_states", "reconstruction_samples"]:
            if len(outputs[key]) > 0 and isinstance(outputs[key][0], torch.Tensor):
                outputs[key] = torch.stack(outputs[key], dim=1)  # [B, T, ...]
        if outputs["K_eff"]:
            outputs["K_eff"] = torch.stack(outputs["K_eff"]).mean()
        return outputs

    def compute_total_loss(
        self,
        observations:torch.Tensor,
        actions:Optional[torch.Tensor]=None,
        dones:Optional[torch.Tensor]=None,
        beta: float = 1.0,
        lambda_recon: float = 1.0,
    ):
        outputs = self.forward_sequence(observations, actions, dones)
        device = self.device

        # --- Reconstruction term ---
        if len(outputs["reconstruction_losses"]) > 0:
            recon_loss = torch.stack(outputs["reconstruction_losses"]).mean()
        else:
            recon_loss = torch.zeros((), device=device)

        # --- Latent KL (Gaussian + DP at top) ---
        if len(outputs["kl_latents"]) > 0:
            kl_z = torch.stack(outputs["kl_latents"]).mean()
        else:
            kl_z = torch.zeros((), device=device)

        # --- Stick-breaking / hierarchical KL (Kumaraswamy + alpha prior) ---
        if len(outputs["kumaraswamy_kl_losses"]) > 0:
            hierarchical_kl = torch.stack(outputs["kumaraswamy_kl_losses"]).mean()
        else:
            hierarchical_kl = torch.zeros((), device=device)
        if len(outputs["component_margin"]) > 0:
            component_margin = torch.stack(outputs["component_margin"]).mean()
        else:
            component_margin = torch.zeros((), device=device)
        total_vae_loss = (
            lambda_recon * recon_loss
            + beta * (kl_z + hierarchical_kl)
            - component_margin # encourage diverse component usage (maximize margin entropy)
        )

        vae_losses = {
            "recon_loss": recon_loss,
            "kl_z": kl_z,
            "hierarchical_kl": hierarchical_kl,
            "component_margin": component_margin,
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
    ) -> Dict[str, torch.Tensor]:
        """
        Training step for both discriminators
        """
        
        B, T, C, H, W = real_images.shape
        temporal_mask = self._make_temporal_mask(B, T, real_images.device, sequence_length )         # [B,T] bool
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
            if temporal_mask is not None:
                md = (temporal_mask[:, 1:] & temporal_mask[:, :-1]).float()
                img_consistency_loss = self._masked_mean(diffs, md)
            else:
                img_consistency_loss = diffs.mean()
        
        real_frames = real_images.reshape(B * T, C, H, W)
        fake_frames = fake_images.reshape(B * T, C, H, W)

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
        img_disc_loss = temporal_disc_loss + lambda_consistency * img_consistency_loss + WGAN_GP_Coeff *img_gp + patch_disc_loss + WGAN_GP_Coeff * patch_gp

        if hasattr(self, 'img_disc_optimizer'):
            # Update image discriminator
            self.img_disc_optimizer.zero_grad()
            img_disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.image_discriminator.parameters()) + list(self.patch_discriminator.parameters()),
                self._grad_clip
            )
            self.img_disc_optimizer.step()

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
        })
        return  disc_losses

    def compute_feature_matching_loss(self, 
                                      real_features: torch.Tensor, 
                                      fake_features: torch.Tensor, 
                                      temporal_mask: torch.Tensor | None = None):

        real_mean = real_features.mean(dim=2)  # [B, T, D]
        fake_mean = fake_features.mean(dim=2)
         
        if temporal_mask is None:
           return F.l1_loss(fake_mean, real_mean.detach())
        m = temporal_mask.float().unsqueeze(-1) #[B, T, 1]
        diff = torch.abs(fake_mean - real_mean.detach())  # [B,T,D]

        denom = (m.sum() * diff.shape[-1]).clamp(min=1.0)
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
        def masked_mean_1(x_bt, m_bt):
            if m_bt is None:
                return x_bt.mean()
            denom = m_bt.sum().clamp(min=1.0)
            return (x_bt * m_bt).sum() / denom

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

        fake_frames = reconstruction.reshape(B * T, C, H, W)
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

        self.global_step += 1

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

        if do_rollout and rollout_horizon > 0:
            # Rollout again WITH grad for generator update
            dbgG = self.generate_future_sequence(
                initial_obs=observations[keep, :T_ctx],
                actions=(actions_slice[keep] if actions_slice is not None else None),
                horizon=rollout_horizon,
                top_temperature=self.rollout_top_temperature,
                decoder_temperature=self.rollout_decoder_temperature,
                decode_mode="mean",  # keep rollout differentiable
                dones=(dones_slice[keep] if dones_slice is not None else None),
                grad=True,
            )
            fake_future_G = dbgG["vae_future"]      # [B_keep, H, C, H, W] (requires grad)
            z_seq_roll_G = dbgG["z_seq"].detach()   # keep conditioning stable + save memory
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
            edge_warp = dbgG.get("edge_warp", None)
            if isinstance(edge_warp, list):
                edge_warp = torch.stack(dbgG["edge_warp"], dim=1)                 # [Bf,Tf,1,H,W]
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

        # 6) Combine losses with DWA or fixed weights
        if self.use_dwa:
            total_components = {
                "recon_loss":      vae_losses["recon_loss"].reshape([]),
                "kl_z":            vae_losses["kl_z"].reshape([]),
                "hierarchical_kl": vae_losses["hierarchical_kl"].reshape([]),
                "img_adv_loss":      (warmup_factor * img_adv_loss).reshape([]),
                "temporal_adv_loss": (warmup_factor * temporal_adv_loss).reshape([]),
                "feat_match_loss":   feat_match_loss.reshape([]),

                "rollout_img_adv_loss":      (self.lambda_rollout_adv * warmup_factor * rollout_img_adv_loss).reshape([]),
                "rollout_temporal_adv_loss": (self.lambda_rollout_adv * warmup_factor * rollout_temporal_adv_loss).reshape([]),
                "rollout_feat_match_loss":   (self.lambda_rollout_adv * rollout_feat_match_loss).reshape([]),

                "component_margin": (-vae_losses["component_margin"]).reshape([]),
                "rollout_edge_loss": (warmup_factor * lambda_edge * rollout_edge_loss).reshape([]),
                "rollout_warp_edge_loss": (warmup_factor * lambda_edge * rollout_warp_edge_loss).reshape([]),
            }
            total_gen_loss = self.total_weighter.reduce_losses(total_components, batch_idx)
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
            elbo_loss = (
                lambda_recon * vae_losses["recon_loss"]
                + beta * vae_losses["kl_z"]
                + beta * vae_losses["hierarchical_kl"]
                - vae_losses["component_margin"]
            )
            total_gen_loss = elbo_loss + adv_base + adv_roll
            total_gen_loss = total_gen_loss + warmup_factor * lambda_edge * rollout_edge_loss
            total_gen_loss = total_gen_loss + warmup_factor * lambda_edge * rollout_warp_edge_loss


        # 7) Backprop generator
        self.gen_optimizer.zero_grad(set_to_none=True)
        total_gen_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for g in self.gen_optimizer.param_groups for p in g["params"]],
            self._grad_clip,
        )
        self.gen_optimizer.step()


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
            "total_gen_loss": float(total_gen_loss.item()),
            "grad_norm": float(grad_norm),
            "effective_components": eff_comp,
            "Top 6 coverage": top6_cov,
        }

    def set_epoch(self, epoch: int):
        """Update current epoch for scheduling purposes"""
        self.current_epoch = epoch

    def get_warmup_factor(self) -> float:
        """Calculate warmup factor for adversarial losses"""
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

        # (A) choose a context vector for the prior
        h_context = torch.randn(num_samples, self.hidden_dim, device=device)

        # (B) apply EMA E
        if hasattr(self, "ema_vdvae"):
            self.ema_vdvae.apply_shadow()

        x_np = self.vdvae.sample(num_samples, h_context)

        if hasattr(self, "ema_vdvae"):
            self.ema_vdvae.restore()

        # uint8 NHWC -> float NCHW in [-1,1]
        x = torch.from_numpy(x_np).permute(0, 3, 1, 2).contiguous().float()/ 127.5 - 1.0
        return x

    def generate_future_sequence(
        self,
        initial_obs: torch.Tensor,          # [B, T_ctx, C, H, W] in [-1, 1]
        actions: torch.Tensor | None,       # [B, T_total, action_dim] (needs >= T_ctx + horizon)
        horizon: int,
        top_temperature: float = 1.0,
        decoder_temperature: float = 1.0,
        dones: torch.Tensor | None = None,  # [B, T_ctx] or [B, T_total]
        decode_mode: str = "mean",          # "mean" | "sample"
        grad: bool = True,
    ):
        """Generate future frames using:
          1) teacher-forced scan over context frames to warm up both:
             - top-level LSTM state (vector)
             - VDVAE decoder ConvLSTM temporal states (spatial)
          2) for each future step:
             - sample top latent map from token-wise DPGMM conditioned on h_context
             - decode with temporal VDVAE (ConvLSTM-conditioned lower priors)
             - update both temporal_state and top-level LSTM
        """
        B, T_ctx = initial_obs.shape[:2]
        device = initial_obs.device
        dtype = initial_obs.dtype
        def compute_mask_t(t: int) -> torch.Tensor:
            if dones is None:
                return torch.ones(B, device=device, dtype=torch.float32)

            if t == 0:
                return torch.ones(B, device=device, dtype=torch.float32)

            # Safe for both dones shapes: [B, T_ctx] or [B, T_total]
            L = dones.shape[1]
            idx = min(t - 1, L - 1)   # clamp index so we never go out of range
            return 1.0 - dones[:, idx].float()

        with torch.set_grad_enabled(grad):
            if actions is None:
                actions = torch.zeros(B, T_ctx + horizon, self.action_dim, device=device, dtype=dtype)
            assert actions.shape[1] >= T_ctx + horizon, "actions must have length >= T_ctx + horizon"

            # --- init states ---
            h = self.h0.expand(self.number_lstm_layer, B, -1).contiguous()
            c = self.c0.expand(self.number_lstm_layer, B, -1).contiguous()
            temporal_state = self.vdvae.init_temporal_state(B, device=device, dtype=dtype)
            # --- context scan ---
            for t in range(T_ctx):
                x_t = initial_obs[:, t]
                a_t = actions[:, t]
                mask_t = compute_mask_t(t)

                h_context = h[-1]
                x_nhwc = x_t.permute(0, 2, 3, 1).contiguous()

                vdvae_out, temporal_state = self.vdvae.forward_temporal_step(
                    x_nhwc,
                    x_nhwc,
                    h_context=h_context,
                    a_t=a_t,
                    mask_t=mask_t,
                    temporal_state=temporal_state,
                )

                # Update top-level LSTM with posterior mean top-latent (teacher forced)
                top_q_mean_map = vdvae_out["top_q_mean_map"]
                _, zdim, topH, topW = top_q_mean_map.shape
                z_flat = top_q_mean_map.permute(0, 2, 3, 1).reshape(B, topH * topW * zdim)
                rnn_in = torch.cat([z_flat, a_t], dim=-1)
                _, (h, c) = self.rnn(rnn_in, h, c, mask_t)
            assert T_ctx >= 1, "Need at least 1 context frame (T_ctx>=1)"

            x_prev = initial_obs[:, T_ctx - 1]  # [B,C,H,W]
            # future rollout 
            pred_imgs = []
            hidden_state =[]
            latent_state =[]
            debug = {"z_top": [], "c_top": [], "flow": [], "edge_warp": []}

            for k in range(horizon):
                t = T_ctx + k
                a_t = actions[:, t]
                mask_t = compute_mask_t(t)

                h_context = h[-1]

                # Token-wise DPGMM sampling for top latent map
                topH, topW = self.top_H, self.top_W
                Hc = h_context.shape[1]
                h_map = h_context.view(B, Hc, 1, 1).expand(B, Hc, topH, topW).contiguous()  # [B,Hc,topH,topW]
                h_map = self.vdvae.add_coord_no_proj(h_map, scale=0.05)                     # [B,Hc(+coords),topH,topW]
                h_flat = h_map.permute(0, 2, 3, 1).reshape(B * topH * topW, -1)             # [N, Hc(+coords)]

                _, prior_params = self.prior(h_flat)
                K = prior_params['pi'].shape[-1]
                pi_tok = prior_params['pi'].view(B, topH * topW, K)
                mu_tok = prior_params['means'].view(B, topH * topW, K, self.zdim)
                var_tok = torch.exp(prior_params['log_vars']).view(B, topH * topW, K, self.zdim)

                # Sample component per token, then sample z from that Gaussian
                # (simple categorical + Gaussian sampling)
                cat = torch.distributions.Categorical(probs=pi_tok)
                c_tok = cat.sample()  # [B, Ttok]
                mu_sel = torch.gather(mu_tok, 2, c_tok[..., None, None].expand(B, topH * topW, 1, self.zdim)).squeeze(2)
                var_sel = torch.gather(var_tok, 2, c_tok[..., None, None].expand(B, topH * topW, 1, self.zdim)).squeeze(2)
                eps = torch.randn_like(mu_sel)
                z_tok = mu_sel + top_temperature * eps * torch.sqrt(var_sel.clamp_min(self.eps))  # [B, Ttok, zdim]

                z_top_map = z_tok.reshape(B, topH, topW, self.zdim).permute(0, 3, 1, 2).contiguous()

               
                #  2) motion scaffold: predict flow + mask from (x_prev, edges(x_prev), h_context) 
                # CannyFilter returns continuous thin edges [B,1,H,W]. 
                x_prev01 =self.denormalize_generated_images(x_prev).clamp(0.0, 1.0)
                e_prev = self.canny(x_prev01).detach()

                ctx = self.flow_ctx_proj(h_context)  # [B, flow_ctx_dim]
                ctx_map = ctx[:, :, None, None].expand(-1, -1, x_prev.shape[2], x_prev.shape[3])

                flow_in = torch.cat([x_prev01, e_prev, ctx_map], dim=1)  # [B, C+1+ctx, H, W]

                flow_raw = self.flow_head(flow_in)   # [B,2,H,W] 

                # bound the flow (CRITICAL for stability)
                Himg, Wimg = x_prev.shape[2], x_prev.shape[3]
                max_flow = 0.25 * float(max(Himg, Wimg))     # e.g. 16 for 64x64
                flow = torch.tanh(flow_raw) * max_flow

                e_warp =self.warp(e_prev, flow)
                debug["flow"].append(flow if grad else flow.detach())
                debug["edge_warp"].append(e_warp if grad else e_warp.detach())
                 # Decode through temporal VDVAE
                px_z, temporal_state = self.vdvae.decode_from_top_latent_temporal(
                    z_top_map=z_top_map,
                    a_t=a_t,
                    mask_t=mask_t,
                    temporal_state=temporal_state,
                    e_warp=e_warp,
                    t=None,
                    temperature=decoder_temperature,
                )

                if self.use_edge_conditioning and getattr(self, "add_edge_to_pxz", False):
                    px_z = px_z +self.edge_cond_proj(e_warp)
                dmol_out = self.vdvae.decoder.out_net.forward(px_z)

                # ---- 1) scratch prediction (what you already had) ----
                if decode_mode == "mean":
                    x_scratch = mean_from_discretized_mix_logistic(
                        dmol_out, self.vdvae.decoder.out_net.H.num_mixtures
                    )
                elif decode_mode == "sample":
                    x_scratch = sample_from_discretized_mix_logistic(
                        dmol_out, self.vdvae.decoder.out_net.H.num_mixtures
                    )
                else:
                    raise ValueError(f"decode_mode must be 'mean' or 'sample', got {decode_mode}")
                x_scratch = x_scratch.permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]
                # ---- 2) decode from latent as usual ----
                pred_imgs.append(x_scratch)

                #  4) update x_prev safely with dones-mask 
                vm = mask_t.view(B, 1, 1, 1).float()
                x_prev = vm * x_scratch + (1.0 - vm) * x_prev

                x_prev = x_prev if grad else x_prev.detach()

                # Update top-level LSTM with sampled top latent (flattened) + action
                z_flat = z_top_map.permute(0, 2, 3, 1).reshape(B, topH * topW * self.zdim)
                rnn_in = torch.cat([z_flat, a_t], dim=-1)
                _, (h, c) = self.rnn(rnn_in, h, c, mask_t)
                latent_state.append(z_flat.detach())
                hidden_state.append(h[-1].detach())

                debug["c_top"].append(c_tok.reshape(B, topH, topW).detach())
            debug["z_seq"] =torch.cat([torch.stack(latent_state, dim=1),torch.stack(hidden_state, dim=1)], dim=-1)
            pred_imgs = torch.stack(pred_imgs, dim=1)  # [B, horizon, C, H, W]
            debug["vae_future"] = pred_imgs
            if len(debug.get("flow", [])) > 0:
                debug["flow"] = torch.stack(debug["flow"], dim=1)
            if len(debug.get("edge_warp", [])) > 0:
                debug["edge_warp"] = torch.stack(debug["edge_warp"], dim=1)

            return debug

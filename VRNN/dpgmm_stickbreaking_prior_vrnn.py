from logging import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Gamma, Categorical, Independent, MixtureSameFamily
from typing import Dict, Tuple, List, Optional, Any
import sys
import os
from einops import rearrange
from contextlib import contextmanager
from itertools import chain
import numpy as np
import math, inspect
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint as ckpt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vis_networks import EMA, AddEpsilon, check_tensor
from VRNN.RGB import RGB, GradNorm, PCGrad, DynamicWeightAverage, MGDA
from VRNN.lstm import LSTMLayer
from vdvae.vae import VDVAE
from vdvae.hps import Hyperparams
from vdvae.vae_helpers import mean_from_discretized_mix_logistic

from VRNN.perceiver.video_prediction_perceiverIO import CausalPerceiverIO
from VRNN.Kumaraswamy import KumaraswamyStable
try:
    from torchjd.autogram import Engine as JD_Engine
    from torchjd.aggregation import UPGradWeighting
    HAS_TORCHJD = True
except ImportError:
    JD_Engine = None
    UPGradWeighting = None
    HAS_TORCHJD = False


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
    Backward-compatible torch.load:
    - On PyTorch >= 2.6, sets weights_only=False explicitly.
    - On older versions, just calls torch.load(path, map_location=...).
    Use ONLY with checkpoints you trust.
    """
    sig = inspect.signature(torch.load)
    kwargs = {}

    # keep your map_location behavior
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
        # Your improved version - already perfect!
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
    
    def forward(self, h: torch.Tensor, use_rand_perm: bool = True, truncation_threshold: float = 0.995) -> Tuple[torch.Tensor, Dict]:
       
        # Generate stick-breaking proportions
        # Get Kumaraswamy parameters
        log_kumar_a, log_kumar_b = self.kumar_net(h)
        
        # Sample v from Kumaraswamy for each alpha sample
        v, perm = self.sample_kumaraswamy(log_kumar_a, log_kumar_b, self.max_K, use_rand_perm)  # [n_samples, batch, K-1]


        # Initialize mixing proportions
        pi = self.compute_stick_breaking_proportions(v, perm)
        # Adaptive truncation: find where cumulative sum exceeds threshold
        pi_sorted, sort_idx = torch.sort(pi, dim=-1, descending=True)
        pi_cumsum = torch.cumsum(pi_sorted, dim=-1)
        
        # Find truncation point for each sample
        truncation_mask = pi_cumsum < truncation_threshold
        # Add one more component to exceed threshold
        truncation_mask[:, 1:] = truncation_mask[:, 1:] | truncation_mask[:, :-1]
        
        # Zero out unused components
        pi_truncated = pi_sorted * truncation_mask.float()
        
        # Restore original order
        _, unsort_idx = torch.sort(sort_idx, dim=-1)
        pi_final = torch.gather(pi_truncated, 1, unsort_idx)
        
        # Renormalize
        pi_final = pi_final / (pi_final.sum(dim=-1, keepdim=True) + torch.finfo(torch.float32).eps )
        
        return pi, {
            'kumar_a': torch.exp(log_kumar_a),
            'kumar_b': torch.exp(log_kumar_b),
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
        prior_beta: float = 1.0
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
        n_samples: int = 10  # Add n_samples argument
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
            posterior_logvar.sum(dim=-1) +
            ((z - posterior_mean.unsqueeze(0)) ** 2 * torch.exp(-posterior_logvar.unsqueeze(0))
        ).sum(dim=-1))

        # Log-prior: (n_samples, B, K) -> (n_samples, B)
        z_expanded = z.unsqueeze(2).expand(-1, -1, K, -1)  # (n_samples, B, K, D)
        log_component_densities = -0.5 * (
            D * math.log(2 * math.pi) +
            prior_logvars.sum(dim=-1).unsqueeze(0) +  # (1, B, K)
            ((z_expanded - prior_means.unsqueeze(0)) ** 2 * 
            torch.exp(-prior_logvars.unsqueeze(0))).sum(dim=-1)
        )
        log_prior_components = log_component_densities + torch.log(prior_weights.unsqueeze(0).clamp(min=eps)+eps)
        log_p = torch.logsumexp(log_prior_components, dim=2)  # (n_samples, B)

        # KL divergence: average over samples and batch
        kl_samples = log_q - log_p  # (n_samples, B)
        return kl_samples.mean()  # Scalar
    
    
    def compute_kl_loss(self, params: Dict, alpha: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between Kumaraswamy and Beta distributions and prior gamma distributions
        """
        total_kl = 0.0
        # Alpha is [batch_size, n_samples], we need [batch_size, 1]
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
        pi, kumar_params = self.stick_breaking(h, use_rand_perm=True)
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


@contextmanager
def apply_emas(*emas):
    for e in emas: e.apply_shadow()
    try:
        yield
    finally:
        for e in reversed(emas): e.restore()


class FrameAttentionPool(nn.Module):
    """
    Attention pool per frame over P slots:
      x: [B, T, P, C_in]  ->  [B, T, C_out]
    Uses a single learnable query per frame (content-agnostic), projects K/V, and MHA.
    """
    def __init__(self, c_in: int, c_out: int, n_heads: int = 4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, c_out) * 0.02)  # [1,1,C_out], shared
        self.k_proj = nn.Linear(c_in,  c_out)
        self.v_proj = nn.Linear(c_in,  c_out)
        self.attn = nn.MultiheadAttention(embed_dim=c_out, num_heads=n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, P, C_in]
        B, T, P, _ = x.shape
        x_bt = x.reshape(B * T, P, -1)               # [B*T, P, C_in]
        K = self.k_proj(x_bt)                         # [B*T, P, C_out]
        V = self.v_proj(x_bt)                         # [B*T, P, C_out]
        Q = self.q.expand(B * T, -1, -1)              # [B*T, 1, C_out]
        out, _ = self.attn(Q, K, V)                   # [B*T, 1, C_out]
        out = out.squeeze(1).view(B, T, -1)           # [B, T, C_out]
        return out


class CausalContextPredictor(nn.Module):
    """
    Simple causal Transformer over (context, action) sequences.
    Input:
      context_seq: [B, T, C_ctx]
      action_seq:  [B, T, A]
    Output:
      pred_context_seq: [B, T, C_ctx] where time t can only see <= t.
    """
    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        latent_dim: int,
        rnn_hidden_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or context_dim

        in_dim = context_dim + action_dim + latent_dim + rnn_hidden_dim

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_proj_ln = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward= hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_proj = nn.Linear(hidden_dim, context_dim)
        self.output_proj_ln = nn.LayerNorm(context_dim)

    def forward(
        self,
        context_seq: torch.Tensor,   # [B, T, C_ctx]
        action_seq: torch.Tensor,    # [B, T, A]
        latent_seq: torch.Tensor,    # [B, T, D_z]
        hidden_seq: torch.Tensor,    # [B, T, D_h]
    ) -> torch.Tensor:
        """
        Returns pred_context_seq with the same temporal length as the input.
        Training will shift targets so we do 1-step-ahead prediction.
        """
        squeeze_time = False
        if context_seq.dim() == 2:
            # Upgrade to [B, 1, *]
            context_seq = context_seq.unsqueeze(1)
            action_seq  = action_seq.unsqueeze(1)
            latent_seq  = latent_seq.unsqueeze(1)
            hidden_seq  = hidden_seq.unsqueeze(1)
            squeeze_time = True

        x = torch.cat(
            [context_seq, action_seq, latent_seq, hidden_seq],
            dim=-1
        ) 
        x = self.input_proj_ln(self.input_proj(x))                            # [B,T,H]

        B, T, H = x.shape
        device = x.device

        # causal mask: position t can attend to <= t
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool),
            diagonal=1
        )
        # nn.Transformer expects float mask with -inf on masked entries if using attn_mask,
        # but bool mask works in recent PyTorch; if not, convert to float with -inf.
        x = self.encoder(x, mask=causal_mask)

        pred_ctx = self.output_proj_ln(self.output_proj(x))  # [B,T,C_ctx]
        if squeeze_time:
            pred_ctx = pred_ctx.squeeze(1)  # [B,C_ctx]
        return pred_ctx

class DPGMMVariationalRecurrentAutoencoder(nn.Module):
    """
    Using Dirichlet Process GMM Prior with Stick-Breaking
    
    """
    def __init__(
        self,
        max_components: int,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        action_dim: int,
        sequence_length: int,
        img_perceiver_channels:int,
        device: torch.device= torch.device('cuda'),
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
        num_encoder_perceiver_layers: int = 4,
        num_perceiver_heads: int = 8,
        num_latent_perceiver: int = 128,
        num_latent_channels_perceiver: int = 128,
        num_codebook_perceiver: int = 1024,
        perceiver_code_dim: int = 256,
        downsample_perceiver: int = 4,
        perceiver_lr_multiplier: float = 0.1,
        use_ctx_checkpoint: bool = True,
        contrastive_weight: float = 0.01,
        grad_balance_method: str = 'rgb',
        gradnorm_alpha: float = 1.5, 
        use_dwa: bool = False,
        dwa_temperature: float = 2.0,
        mgda_gn: str = "l2",
        lambda_ctx_pred: float = 1.0,
        ):
        super().__init__()
        #core dimensions
        self.input_channels = input_channels
        self.image_size = input_dim
        self.max_K = max_components
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.context_dim = num_latent_channels_perceiver
        self.action_dim = action_dim ##
        self.sequence_length = sequence_length ##
        self.device = device
        self.dropout = dropout
        # Hyperparameters
        self._lr = learning_rate
        self._grad_clip = grad_clip
        self.number_lstm_layer = number_lstm_layer
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.num_encoder_perceiver_layers = num_encoder_perceiver_layers
        self.num_perceiver_heads = num_perceiver_heads
        self.num_latent_perceiver = num_latent_perceiver
        self.num_codebook_perceiver = num_codebook_perceiver
        self.perceiver_code_dim = perceiver_code_dim
        self.img_perceiver_channels = img_perceiver_channels
        self.downsample_perceiver = downsample_perceiver
        self.perceiver_lr_multiplier = perceiver_lr_multiplier  
        self.lambda_ctx_pred = lambda_ctx_pred
        self.contrastive_weight = contrastive_weight 
        self.grad_balance_method = grad_balance_method
        self.gradnorm_alpha = gradnorm_alpha
        self.use_dwa = use_dwa
        self.mgda_gn = mgda_gn
        #initialization different parts of the model
        self.use_ctx_checkpoint = use_ctx_checkpoint
        self._init_perceiver_context()
        self._init_encoder_decoder(max_components, prior_alpha, prior_beta)  # defines self.top_zdim
        self._init_vrnn_dynamics(use_orthogonal=use_orthogonal, number_lstm_layer=number_lstm_layer)

        if use_dwa:
            self._init_DynamicWeightAverage(dwa_temperature)

        self.to(device)
        
        # Setup optimizers
        self.has_optimizers = True
        self._setup_optimizers(learning_rate, weight_decay)

    def _init_DynamicWeightAverage(self, temperature: float = 2.0):
        if  self.grad_balance_method != 'none':
            # ELBO components: recon + two KLs + Gram
            self.elbo_weighter = DynamicWeightAverage(
                loss_keys_to_consider=[
                    "recon_loss",
                    "kl_z",
                    "hierarchical_kl",
                ],
                temperature=temperature,
            )


            # Perceiver components: you can start simple and refine later
            self.perceiver_weighter = DynamicWeightAverage(
                loss_keys_to_consider=[
                    "perceiver_vq_loss",
                    "perceiver_ce_loss",
                    "perceiver_lpips_loss",
                    "perceiver_recon_loss",
                    "context_cycle_consistency_loss",
                ],
                temperature=temperature,
            )
        else:
            self.total_weighter = DynamicWeightAverage(
                loss_keys_to_consider=["recon_loss",
                                        "kl_z",
                                        "hierarchical_kl",
                                        "perceiver_vq_loss",
                                        "perceiver_ce_loss",
                                        "perceiver_lpips_loss",
                                        "perceiver_recon_loss",
                                        "context_cycle_consistency_loss",
                                    ], temperature=temperature,
                                    )


    def _init_encoder_decoder(self, max_components: int, prior_alpha: float, prior_beta: float, prior_mc_samples: int =15):
        # DPGMM Prior
        # ---- 1) Build VDVAE hyperparams ----
        H = Hyperparams()
        H.use_checkpoint = self.use_ctx_checkpoint
        H.image_channels = self.input_channels   # usually 3
        H.zdim = self.latent_dim                 # or set explicitly (e.g., 16)
        H.bottleneck_multiple = 0.25 
        H.width = 200
        H.image_size = self.image_size          # e.g. 64
        H.dataset = 'imagenet64'
        H.num_mixtures = 10
        H.skip_threshold = 500.
        # ~9–10 blocks encoder/decoder
        H.dec_blocks = "1x1,4m1,4x2,8m4,8x4,16m8,16x6,32m16,32x7,64m32,64x2"
        H.enc_blocks = "64x2,64d2,32x7,32d2,16x6,16d2,8x4,8d2,4x2,4d4,1x3"

        H.no_bias_above = 64
        H.custom_width_str = ""

        # ---- 2) Instantiate VDVAE (no prior yet) ----
        self.vdvae = VDVAE(
            H,
            prior=None,              # we'll set self.prior separately
            top_kl_weight=1.0,
            prior_kl_mc_samples=prior_mc_samples,
        ).to(self.device)

        # ---- 3) Extract top block latent dim & resolution ----
        top_block = self.vdvae.decoder.dec_blocks[0]        # is_top=True for first block
        C = top_block.zdim                                  # latent channels
        res = top_block.base                                # spatial resolution (e.g. 8)
        self.top_zdim = C * res * res

        self.prior = DPGMMPrior(max_components, 
                                latent_dim=C,   # latent dim 
                                hidden_dim=self.hidden_dim+ self.context_dim, #This is because that the input for the prior is the hidden state of recurrent model plus context which is coming from perceiver
                                device=self.device, 
                                prior_alpha=prior_alpha,
                                prior_beta=prior_beta)
        # EMA for VDVAE
        self.ema_decay = 0.999
        self.ema_vdvae = EMA(self.vdvae, decay=self.ema_decay)  

        # Attach prior to VDVAE so its forward() computes dp_kl / dp_rate
        self.vdvae.prior = self.prior


    def load_pretrained_vq_tokenizer(
        self,
        ckpt_path: str,
        freeze_codebook: bool = True,
        freeze_entire_tokenizer: bool = False,
        freeze_dvae_backbone: bool = True,   
        strict: bool = True,
    ):
        """
        Load a VQPTTokenizer checkpoint trained with pretrain_vqvae.py into
        self.perceiver_model.tokenizer and optionally freeze it.
        """
        device = self.device if hasattr(self, "device") else next(self.parameters()).device

        ckpt = torch_load_version_compat(ckpt_path, map_location=device)

        vq_state = ckpt["model_state"]
        vq_config = ckpt.get("config", None)

        # (Optional but recommended) sanity checks on architecture
        if vq_config is not None:
            assert vq_config["code_dim"] == self.perceiver_code_dim, \
                f"code_dim mismatch: pretrain {vq_config['code_dim']} vs model {self.perceiver_code_dim}"
            assert vq_config["num_codes"] == self.num_codebook_perceiver, \
                f"num_codes mismatch: pretrain {vq_config['num_codes']} vs model {self.num_codebook_perceiver}"
            assert vq_config["downsample"] == self.downsample_perceiver, \
                f"downsample mismatch: pretrain {vq_config['downsample']} vs model {self.downsample_perceiver}"
            assert vq_config["in_channels"] == self.input_channels, \
                f"in_channels mismatch: pretrain {vq_config['in_channels']} vs model {self.input_channels}"

        # Load into the tokenizer living inside the Perceiver
        tokenizer = self.perceiver_model.tokenizer
        missing, unexpected = tokenizer.load_state_dict(vq_state, strict=strict)
        print(f"[DPGMM] Loaded VQPT tokenizer from {ckpt_path}")
        if missing:
            print("  missing keys:", missing)
        if unexpected:
            print("  unexpected keys:", unexpected)

        # freeze DVAE backbone (encoder + quant convs + norms) 
        if freeze_dvae_backbone and hasattr(tokenizer, "dvae"):
            dvae = tokenizer.dvae

            # Encoder path: video_encoder
            for p in dvae.video_encoder.parameters():
                p.requires_grad = False

            # Quantization projection & norms inside DVAE
            for p in dvae.quantize.parameters():
                p.requires_grad = False
            for p in dvae.post_quant.parameters():
                p.requires_grad = False
            for p in dvae.norm_pre_quant.parameters():
                p.requires_grad = False
            for p in dvae.norm_post_quant.parameters():
                p.requires_grad = False

            dvae.eval()
            print("[DPGMM] Frozen DVAE encoder + quantization backbone")

        # Optionally freeze only the codebook 
        if freeze_codebook and hasattr(tokenizer, "vq"):
            for p in tokenizer.vq.parameters():
                p.requires_grad = False
            tokenizer.vq.freeze_codebook = True
            print("[DPGMM] Frozen VQ codebook parameters")

        # Optionally freeze everything in the tokenizer (encoder + decoder + VQ)
        if freeze_entire_tokenizer:
            for p in tokenizer.parameters():
                p.requires_grad = False
            if hasattr(tokenizer, "vq"):
                tokenizer.vq.freeze_codebook = True
            print("[DPGMM] Frozen entire VQPT tokenizer")

    def _init_perceiver_context(self):
        """Initialize Perceiver with architecture-aware dimensions"""

        # 3) Create the Perceiver with Utils.generate_model
        self.perceiver_model = CausalPerceiverIO(
            video_shape=(self.sequence_length, self.input_channels, self.image_size, self.image_size),
            num_latents=self.num_latent_perceiver,
            num_latent_channels=self.context_dim,
            num_attention_heads=self.num_perceiver_heads,
            num_self_attention_layers=self.num_encoder_perceiver_layers,
            max_seq_len= 4096,
            code_dim=self.perceiver_code_dim,
            num_codes=self.num_codebook_perceiver,
            downsample=self.downsample_perceiver,
            dropout=self.dropout,
            base_channels=self.img_perceiver_channels,
            num_quantizers=1,
            kmeans_init=False,
            commitment_weight=0.5,
            commitment_use_cross_entropy_loss=False,
            orthogonal_reg_weight=0.0,
            orthogonal_reg_active_codes_only=True,
            orthogonal_reg_max_codes=2048,
            threshold_ema_dead_code=0,
            token_ar_checkpoint=False,
        )
        self.frame_slot_pool = FrameAttentionPool(
                c_in=self.context_dim,    # Perceiver bottleneck channel
                c_out=self.context_dim,   # keep same dim for the VRNN interface
                n_heads= self.num_perceiver_heads,
            )
        self.context_predictor = CausalContextPredictor(
            context_dim=self.context_dim,
            action_dim=self.action_dim,
            latent_dim=self.latent_dim,
            rnn_hidden_dim=self.hidden_dim,
            hidden_dim=self.context_dim,
            num_layers=1,
            num_heads=self.num_perceiver_heads,
            dropout=self.dropout,
        )                       


    def _init_vrnn_dynamics(self,use_orthogonal: bool = True, number_lstm_layer: int = 1):
        """Initialize VRNN components with context conditioning"""
        # Feature extractors        
        
        # VRNN recurrence: h_t = f(h_{t-1}, z_t, c_t, a_t)
        self._rnn = LSTMLayer(
            input_size=self.top_zdim + self.action_dim + self.context_dim ,  # z_t + c_t +  a_t
            hidden_size=self.hidden_dim,
            n_lstm_layers= number_lstm_layer,
            use_orthogonal=use_orthogonal
        )
        self.rnn_layer_norm = nn.LayerNorm(self.hidden_dim)
        # Initialize hidden states
        self.h0 = nn.Parameter(torch.zeros(self.number_lstm_layer, 1, self.hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(self.number_lstm_layer, 1, self.hidden_dim))
        

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

        # Perceiver / context branch
        perceiver_params = get_params(
            self.perceiver_model,
            self.context_predictor,
            self.frame_slot_pool,
        )
        if perceiver_params:
            perceiver_groups = split_by_weight_decay(perceiver_params, weight_decay)
            for g in perceiver_groups:
                g["lr"] = learning_rate * self.perceiver_lr_multiplier
                if g.get("weight_decay", 0.0) > 0:
                    g["weight_decay"] *= 0.5
            gen_param_groups.extend(perceiver_groups)

        # Trunk / world-model modules
        trunk_modules = [
            self.vdvae.encoder,
            self.vdvae.decoder,
            self.prior,
            self._rnn,
            self.rnn_layer_norm,
        ]
        trunk_params = get_params(*trunk_modules)
        if trunk_params:
            gen_param_groups.extend(split_by_weight_decay(trunk_params, weight_decay))

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

        # 2) Gradient balancer (RGB, GradNorm, or none)
        method = getattr(self, "grad_balance_method", "rgb")
        method = (method or "none").lower()

        self.grad_balancer = None
        task_names = ["ELBO", "perceiver"]   # must match training_step_sequence
        balancer_param_groups = []

        if method == "rgb":
            self.grad_balancer = RGB()
        elif method == "gradnorm":
            self.grad_balancer = GradNorm()
        elif method == "mgda":
            self.grad_balancer = MGDA()
        elif method == "upgrad" and HAS_TORCHJD:
            self._jd_engine = JD_Engine(self, batch_dim=None)
            self._jd_weighting = UPGradWeighting()

        if self.grad_balancer is not None:

            self.grad_balancer.task_name = task_names
            self.grad_balancer.task_num  = len(task_names)
            self.grad_balancer.device    = self.device
            self.grad_balancer.rep_grad  = False

            # Grad balancer should only see *generator* params, not its own params
            # After gen_param_groups is fully built, inside _setup_optimizers:

            _shared_params = [p for g in gen_param_groups for p in g["params"]]

            self.grad_balancer.get_share_params = lambda: iter(_shared_params)

            if method == "rgb":
                # RGB has no learnable parameters
                self.grad_balancer.alpha_steps     = 1
                self.grad_balancer.update_interval = 2
                self.grad_balancer.lr_inner        = 0.05
                self.grad_balancer.epoch = 0
                self.grad_balancer.train_loss_buffer = None

            elif method == "gradnorm":
                # GradNorm has a trainable loss_scale vector
                self.grad_balancer.init_param()  # creates self.loss_scale
                self.grad_balancer.epoch = 0
                self.grad_balancer.train_loss_buffer = None

                balancer_param_groups.append({
                    "params": [self.grad_balancer.loss_scale],
                    "lr": learning_rate,
                    "weight_decay": 0.0,
                })

        # 3) Base optimizer over generator (+ optional balancer params)
        param_groups = gen_param_groups + balancer_param_groups
        base_optimizer = torch.optim.Adamax(
            param_groups,
            lr=learning_rate*1.2,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # If requested, wrap the base optimizer with PCGrad
        if method == "pcgrad":
            # Two tasks: generator loss and perceiver loss (see training_step_sequence)
            self.gen_optimizer = PCGrad(num_tasks=len(task_names), optimizer=base_optimizer)
        else:
            self.gen_optimizer = base_optimizer


        self._setup_schedulers()


    def _setup_schedulers(self):
        """Setup learning-rate schedulers for all optimizers (Option B)."""

        # If we're using PCGrad, schedulers should see the wrapped optimizer
        opt_for_sched = getattr(self.gen_optimizer, "_optim", self.gen_optimizer)

        # Store initial lrs for trunk optimizer
        for group in opt_for_sched.param_groups:
            group["initial_lr"] = group["lr"]

        # 1) Trunk scheduler (PCGrad or not, this uses the base optimizer)
        self.gen_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_for_sched,
            mode='min',
            factor=0.2,
            patience=10,
            threshold=0.0001,
        )

    def maybe_checkpoint_ctx(self, fn, *x, use_reentrant: bool = False):
        if self.training and self.use_ctx_checkpoint:
            return ckpt(fn, *x, use_reentrant=use_reentrant)
        return fn(*x)

    def forward_sequence(self, observations, actions=None, dones=None):

        batch_size, seq_len = observations.shape[:2]
        
        # Default actions if not provided
        if actions is None:
            actions = torch.zeros(batch_size, seq_len, self.action_dim).to(self.device)
        perceiver_future_steps = getattr(self, "perceiver_future_steps", 2)
        num_context_frames = max(1, seq_len - perceiver_future_steps)
        # Extract global context using Perceiver
        context_sequence, context_info = self.compute_global_context(
            observations,
            num_context_frames=num_context_frames,
            generate_future=False
        )
        ctx_pred_loss = 0.0
        # Initialize LSTM hidden states
        h = self.h0.expand(self.number_lstm_layer, batch_size, -1).contiguous()
        c = self.c0.expand(self.number_lstm_layer, batch_size, -1).contiguous()
        c_prev = torch.zeros(batch_size, self.context_dim).to(self.device)
        # Storage for outputs
        outputs = {
            'reconstructions': [],
            "reconstruction_samples": [],
            'latents': [],
            'hidden_states': [],
            'context_states': [],
            'prior_params': [],
            "kumaraswamy_kl_losses": [],
            "kl_latents": [],              # per-step total KL (gauss + DP), scalar
            "reconstruction_losses": [],   # per-step distortion, scalar
            "vae_elbos": [],
            "gauss_rates": [],
            "gm_rates": [],
            'K_eff': [],
        }
        outputs['perceiver_reconstructed_img'] = context_info['reconstruct']
        # Process sequence step by step
        for t in range(seq_len):
            # Get current inputs
            o_t = observations[:, t]
            x_t = o_t.permute(0,2,3,1).contiguous()  # to NHWC for VDVAE
            a_t = actions[:, t]
            # Get context (global or local)
            c_t = context_sequence[:, t]
            
            outputs['context_states'].append(c_t)
            
            # === Prior Network p(z_t|o_<t, z_<t) ===
            h_context = torch.cat([h[-1], c_prev], dim=-1)
            vdvae_out = self.vdvae(x_t, x_t, h_context)
            distortion_t = vdvae_out["distortion"]   # recon per-pixel
            rate_t = vdvae_out["rate"]              # total KL per-pixel (gauss + DP)
            elbo_t = vdvae_out["elbo"]
            gauss_rate_t = vdvae_out["gauss_rate"]
            dp_rate_t = vdvae_out["dp_rate"]

            outputs["reconstruction_losses"].append(distortion_t)
            outputs["kl_latents"].append(rate_t)
            outputs["vae_elbos"].append(elbo_t)
            outputs["gauss_rates"].append(gauss_rate_t)
            outputs["gm_rates"].append(dp_rate_t)

            # ---- 2) Stick-breaking / hierarchical KL (Kumaraswamy vs Beta/Gamma priors) ----
            prior_params = vdvae_out["prior_params"]

            # 2. KL for stick-breaking (Kumaraswamy vs Beta and Gamma prior vs Gamma posterior)

            kumar_beta_kl = self.prior.compute_kl_loss(
                prior_params, 
                prior_params['alpha'], 
                h_context.detach(),
            )
            # Stick-breaking KL
            outputs['kumaraswamy_kl_losses'].append(kumar_beta_kl)
            outputs['prior_params'].append(prior_params)
            
            # 4. Compute effective number of components and penalty for unused components
            K_eff = self.prior.get_effective_components(prior_params['pi'])
            eps = torch.randn_like(vdvae_out["top_q_logvar_map"])
            z_map = vdvae_out["top_q_mean_map"] + eps * torch.exp(0.5 * vdvae_out["top_q_logvar_map"])

            z_t = z_map.contiguous().view(batch_size, -1)  # flatten

            outputs['K_eff'].append(K_eff.float().mean())
            outputs['latents'].append(z_t)

            #build mask based on done signals
            if dones is None:
                mask_t = torch.ones(batch_size, device=self.device)
            else:
                if t == 0:
                    first_t = torch.ones(batch_size, device=self.device)
                else:
                    first_t = dones[:, t-1].to(torch.float32).to(self.device)
                mask_t = 1.0 - first_t
            
            rnn_input = torch.cat([z_t, c_t, a_t], dim=-1)
            rnn_output, (h, c) = self._rnn(
                rnn_input, h, c, 
                mask_t
            )
            c_prev = c_t
            rnn_output = self.rnn_layer_norm(rnn_output)
            outputs['hidden_states'].append(h[-1])
            dmol_out = self.vdvae.decoder.out_net.forward(vdvae_out["px_z"])  # [B, H, W, 10*nr_mix]
            recon_mean = mean_from_discretized_mix_logistic(dmol_out, self.vdvae.H.num_mixtures)  # [B,H,W,3]
            #print(f"image reconstruction range: min {recon_mean.min().item()}, max {recon_mean.max().item()}")
            recon_mean = recon_mean.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
            outputs["reconstructions"].append(recon_mean)

            with torch.no_grad():
                sample = self.vdvae.decoder.out_net.sample(vdvae_out["px_z"])   # [B, H, W, C] uint8
                sample = torch.from_numpy(sample).to(device=self.device, dtype=torch.float32)
                sample = sample.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
                #numpy array in [0, 255]
            outputs["reconstruction_samples"].append(sample) #in range [0, 255]

        outputs['K_eff'] = torch.stack(outputs['K_eff']).mean() 
        # === Compute aggregated losses ===
        # Stack temporal sequences
        for key in ['reconstructions', 'latents', 'hidden_states', 'context_states', 'reconstruction_samples']:
            if outputs[key]:
                outputs[key] = torch.stack(outputs[key], dim=1)
        outputs['perceiver_total_loss'] =       context_info["total_loss"]


        latents_seq = outputs['latents']            # [B, T, D_z]
        hidden_seq  = outputs['hidden_states']      # [B, T, D_h]
        B, T, _ = context_sequence.shape

        # Sanity guard: if T < 2, skip
        if T > 1:
            # Inputs at times 0..T-2
            c_prev = context_sequence[:, :-1, :]        # [B, T-1, C_ctx]
            a_prev = actions[:, :-1, :]            # [B, T-1, A]
            z_prev = latents_seq[:, :-1, :]        # [B, T-1, D_z]
            h_prev = hidden_seq[:, :-1, :]         # [B, T-1, D_h]

            # Targets are next contexts 1..T-1
            c_target = context_sequence[:, 1:, :]       # [B, T-1, C_ctx]

            # Predictor is causal, so at index t it sees indices <= t
            # [B, T-1, C_ctx]
            def _ctx_predictor(context_seq, action_seq, latent_seq, hidden_seq):
                return self.context_predictor(
                    context_seq=context_seq,
                    action_seq=action_seq,
                    latent_seq=latent_seq,
                    hidden_seq=hidden_seq,
                )

            pred_ctx_seq = self.maybe_checkpoint_ctx(
                _ctx_predictor,
                c_prev,
                a_prev,
                z_prev,
                h_prev,
            ) 

            ctx_pred_loss = F.mse_loss(pred_ctx_seq, c_target.detach())

            
            # Add to perceiver task loss (so it goes into MGDA "perceiver" bucket)
            outputs['perceiver_total_loss'] = outputs['perceiver_total_loss'] + self.lambda_ctx_pred * ctx_pred_loss

            outputs['context_pred_loss'] = ctx_pred_loss
        # Add auxiliary outputs
        outputs['perceiver_reconstructed_img'] = context_info['reconstruct']
        outputs['perceiver_vq_loss']          = context_info['vq_loss']
        outputs['perceiver_ce_loss']          = context_info['ce_loss']
        outputs['perceiver_lpips_loss']       = context_info['lpips_loss']
        outputs["perceiver_recon_loss"] =       context_info["recon_loss"]
        outputs["perceiver_cycle_loss"] =       context_info["ar_cycle_loss"]
        return outputs
    
    def compute_total_loss(self, 
                           observations, 
                           actions=None, 
                           beta=1.0,  
                           lambda_recon=1.0, 
                           ):
        """
        Corrected loss computation recognizing that kumar_beta_kl already includes alpha prior KL
        """
        outputs = self.forward_sequence(observations, actions)
        if len(outputs["reconstruction_losses"]) > 0:
            recon_loss = torch.stack(outputs["reconstruction_losses"]).mean()
        else:
            recon_loss = torch.zeros((), device=self.device)

        # --- Latent KL (Gaussian + DP at top) ---
        if len(outputs["kl_latents"]) > 0:
            kl_z = torch.stack(outputs["kl_latents"]).mean()
        else:
            kl_z = torch.zeros((), device=self.device)

        # --- Stick-breaking / hierarchical KL (Kumaraswamy + alpha prior) ---
        if len(outputs["kumaraswamy_kl_losses"]) > 0:
            hierarchical_kl = torch.stack(outputs["kumaraswamy_kl_losses"]).mean()
        else:
            hierarchical_kl = torch.zeros((), device=self.device)

        total_vae_loss = (
            lambda_recon * recon_loss
            + beta * (kl_z + hierarchical_kl)
        )

        vae_losses = {
            "recon_loss": recon_loss,
            "kl_z": kl_z,
            "hierarchical_kl": hierarchical_kl,
            "total_vae_loss": total_vae_loss,
        }

        return vae_losses, outputs
       

    def compute_global_context(
        self,
        videos: torch.Tensor,           # [B, T_raw, C, H, W]
        num_context_frames: int,
        generate_future: bool = False,
        num_future_frames: Optional[int] = None,
    ):
        """
        Build per-frame global context for the VRNN by extracting the Perceiver's
        temporal self-attention bottleneck and attention-pooling it per frame.

        TRAIN:   videos = x_{1:T},           return context c_{1:T}
        GENERATE: videos = x_{1:T_ctx}, AR-generate T_future, return c_{1:T_ctx+T_future}

        Returns:
        temporal_context: [B, T_out, C_ctx]
        info: dict with losses, Ht/Wt, and recon / generated videos for logging.
        """
        
        B, T_raw, C_img, H_img, W_img = videos.shape
        assert 1 <= num_context_frames <= T_raw, "Invalid num_context_frames"
        T_ctx = num_context_frames

        #  Helper: tokens → per-frame temporal context 
        def _tokens_to_temporal_context(
            token_ids: torch.Tensor,       # [B, T_enc, Ht, Wt] or [B, T_enc, Ht, Wt, Q]
            encoder_latents: torch.Tensor, # [B, T_enc, C_enc, H_lat, W_lat]
            T_frames: int,                 # raw frames represented by token_ids
            T_ctx_frames: int,             # desired context length in raw frames
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:

            # If tokenizer returns multi-head codes, drop the codebook head axis
            if token_ids.dim() == 5:
                token_ids_flat = token_ids[..., 0]        # [B, T_enc, Ht, Wt]
            else:
                token_ids_flat = token_ids                 # [B, T_enc, Ht, Wt]

            B_tok, T_enc, Ht, Wt = token_ids_flat.shape

            # Split in ENCODED time (internally clamps so T_pred_enc ≥ 1)
            ctx_tokens, future_tokens, T_ctx_enc, T_pred_enc = \
                self.perceiver_model.model._split_context_target_tokens(
                    token_ids_flat, num_context_frames=T_ctx_frames, T_raw=T_frames
                )

            # Run temporal bottleneck (context + future queries)
            temporal_ctx_future, meta = self.perceiver_model.model.extract_temporal_bottleneck(
                ctx_token_ids=ctx_tokens,
                encoder_latents=encoder_latents,
                T_ctx_enc=T_ctx_enc,
                T_pred_enc=T_pred_enc,
            )
            # meta["whole_temporal_bottleneck"]: [B, T_enc_total, C_ctx]
            whole_tb = meta["whole_temporal_bottleneck"]
            T_enc_total = whole_tb.shape[1]

            # Map encoded timesteps back to raw frame timesteps
            assert T_enc_total % T_frames == 0, \
                f"T_enc_total={T_enc_total}, T_frames={T_frames} not divisible"
            tokens_per_frame = T_enc_total // T_frames

            # [B, T_enc_total, C] → [B, T_frames, P, C] → attention-pool P # [B, T, P, C]
            per_frame_slots = rearrange(
                whole_tb, "b (t p) c -> b t p c", t=T_frames, p=tokens_per_frame
            )   
            def _frame_slot_pool(x):
                return self.frame_slot_pool(x)         
                                                  
            per_frame_ctx = self.maybe_checkpoint_ctx(_frame_slot_pool, per_frame_slots)              # [B, T, C]

            meta_out = dict(
                T_frames=T_frames,
                T_ctx_frames=T_ctx_frames,
                T_enc_total=T_enc_total,
                T_ctx_enc=T_ctx_enc,
                T_pred_enc=T_pred_enc,
                tokens_per_frame=tokens_per_frame,
                Ht=Ht, Wt=Wt,                       # expose token grid size
            )
            return per_frame_ctx, meta_out
        # 

        if not generate_future:
            # Tokenize full training clip once
            token_ids, _, latent_tm, vq_loss_tok = self.perceiver_model.tokenizer.encode(videos)

            # Temporal bottleneck for all frames 1..T_raw
            temporal_context, meta_ctx = _tokens_to_temporal_context(
                token_ids=token_ids,
                encoder_latents=latent_tm,
                T_frames=T_raw,
                T_ctx_frames=T_ctx,   # pass raw-frame T_ctx; split will ensure T_pred_enc≥1
            )

            # Perceiver AR+recon losses for logging/optimization
            perc_out = self.perceiver_model(
                videos, num_context_frames=T_ctx, return_dict=True
            )
            perc_losses = self.perceiver_model.model.compute_loss(
                perc_out,
                target_videos=videos,
                perceptual_weight=getattr(self, "perceptual_weight", 1.0),
                label_smoothing=0.05,
                ce_weight=self.contrastive_weight,
                ar_cycle_consistency_weight=0.01,
            )

            info = {
                "vq_loss":          perc_losses["vq_loss"],
                "ce_loss":          perc_losses["ce_loss"],
                "lpips_loss":       perc_losses["lpips_loss"],
                "recon_loss":       perc_losses["recon_loss"],
                "ar_cycle_loss":    perc_losses["ar_cycle_loss"],
                "token_accuracy":   perc_out["token_accuracy"],
                "total_loss":       perc_losses["loss"],
                "Ht": meta_ctx["Ht"], "Wt": meta_ctx["Wt"],
                "reconstruct":      perc_out["reconstructed"],  # needed by forward_sequence
                "generated_videos": None,
            }
            # Backwards-compat alias if the training loop still references it:
            info["perceptual_loss"] = info["lpips_loss"]
            return temporal_context, info

        # Generation path (context + AR future) 
        assert T_ctx >= 1, "Need ≥1 context frame for generation"
        if num_future_frames is None:
            num_future_frames = max(T_raw - T_ctx, 1)

        # AR generate future frames from the context
        context_videos = videos[:, :T_ctx]   # [B, T_ctx, C, H, W]
        generated_videos = self.perceiver_model.generate_autoregressive(
            context_videos=context_videos,
            num_frames_to_generate=num_future_frames,
        )                                    # [B, T_ctx+T_future, C, H, W]

        # Tokenize the concatenated (ctx+future) trajectory
        tok_gen, _, lat_gen, vq_loss_gen = self.perceiver_model.tokenizer.encode(generated_videos)

        # Temporal bottleneck over the full generated timeline
        temporal_context_gen, meta_gen = _tokens_to_temporal_context(
            token_ids=tok_gen,
            encoder_latents=lat_gen,
            T_frames=generated_videos.shape[1],
            T_ctx_frames=generated_videos.shape[1],  # pool context over the whole timeline
        )

        info = {
            "vq_loss":          vq_loss_gen,
            "ce_loss":          None,
            "lpips_loss":       None,
            "ar_cycle_loss":    None,
            "token_accuracy":   None,
            "total_loss":       None,
            "Ht": meta_gen["Ht"], 
            "Wt": meta_gen["Wt"],
            "reconstruct":      None,
            "generated_videos": generated_videos,
        }
        return temporal_context_gen, info

    
    def prepare_images_for_training(self, images):
        """
        Ensure images are in [-1, 1] range for model processing.
        This is defensive - dataset should already provide [-1, 1] images.
        """
        with torch.no_grad():
            # Quick check if already normalized
            img_min, img_max = images.min().item(), images.max().item()
            
            # If already in [-1, 1] range (with small tolerance), return as-is
            if img_min >= -1.1 and img_max <= 1.1:
                return images
            
            # If uint8, convert to float [0, 1] first
            if images.dtype == torch.uint8:
                images = images.float() / 255.0
            # If in [0, 255] range
            elif img_max > 1.5:
                images = images / 255.0
            
            # Now images should be in [0, 1], convert to [-1, 1]
            # Check again to be sure
            img_min, img_max = images.min().item(), images.max().item()
            if img_min >= -0.01 and img_max <= 1.01:
                return images * 2.0 - 1.0
            
            # Something unexpected - clamp and warn
            print(f"Warning: Unexpected image range [{img_min:.3f}, {img_max:.3f}], clamping to [-1, 1]")
            return images.clamp(min=-1.0, max=1.0)

    def denormalize_generated_images(self, images):
        """
        Convert generated images from [-1, 1] back to [0, 1] for visualization
        """
        return (images + 1) / 2
    
    @staticmethod
    def _flatten_for_diag(x: torch.Tensor | list | tuple):
        if x is None:
            return None
        # If it is a list/tuple of tensors, stack along time to match compute_total_loss
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return None
            x = torch.stack(x, dim=1)  # [B, T, ...]
        return x.reshape(-1)  # flatten all dims

    def training_step_sequence(self, 
                            observations: torch.Tensor,
                            actions: torch.Tensor = None,
                            beta: float = 1.0,
                            lambda_recon: float = 1.0,
                            batch_idx: Optional[int] = None,
                            ) -> Dict[str, torch.Tensor]:
        """
        Supports:
        - Dynamic Weight Averaging (DWA) over either:
            * ELBO /  perceiver streams (when grad_balance_method != "none"), or
            * all individual loss terms together (when grad_balance_method == "none").
        - RGB / GradNorm / PCGrad /MGDAmulti-task gradient balancing when requested.
        """
        self.train()
        # 1) Prepare data and compute VAE & perceiver losses

        vae_losses, outputs = self.compute_total_loss(
            observations, actions, beta,
            lambda_recon, 
        )

        # 4) Combine losses with DWA
        if self.use_dwa and hasattr(self, "elbo_weighter"):
            # Per-stream DWA (grad_balance_method != "none")
            elbo_components = {
                "recon_loss":      vae_losses["recon_loss"].reshape([]),
                "kl_z":            vae_losses["kl_z"].reshape([]),
                "hierarchical_kl": vae_losses["hierarchical_kl"].reshape([]),
            }

            perceiver_components = {
                "perceiver_vq_loss":    outputs["perceiver_vq_loss"].reshape([]),
                "perceiver_ce_loss":    outputs["perceiver_ce_loss"].reshape([]),
                "perceiver_lpips_loss": outputs["perceiver_lpips_loss"].reshape([]),
                "perceiver_recon_loss": outputs["perceiver_recon_loss"].reshape([]),
                "context_cycle_consistency_loss": outputs["perceiver_cycle_loss"].reshape([]),
                "context_pred_loss": outputs["context_pred_loss"].reshape([]),
            }

            elbo_loss      = self.elbo_weighter.reduce_losses(elbo_components, batch_idx)
            perceiver_loss = self.perceiver_weighter.reduce_losses(perceiver_components, batch_idx)

            total_gen_loss = elbo_loss + perceiver_loss

        elif self.use_dwa and hasattr(self, "total_weighter"):
            # grad_balance_method == "none": single DWA over *all* loss terms
            total_components = {
                "recon_loss":      vae_losses["recon_loss"].reshape([]),
                "kl_z":            vae_losses["kl_z"].reshape([]),
                "hierarchical_kl": vae_losses["hierarchical_kl"].reshape([]),
                "perceiver_vq_loss":    outputs["perceiver_vq_loss"].reshape([]),
                "perceiver_ce_loss":    outputs["perceiver_ce_loss"].reshape([]),
                "perceiver_lpips_loss": outputs["perceiver_lpips_loss"].reshape([]),
                "perceiver_recon_loss": outputs["perceiver_recon_loss"].reshape([]),
                "context_cycle_consistency_loss": outputs["perceiver_cycle_loss"].reshape([]),
            }
            # This is the actual scalar we differentiate
            total_gen_loss = self.total_weighter.reduce_losses(total_components, batch_idx)

            # For logging / diagnostics only (no grads through these):
            with torch.no_grad():
                elbo_loss = (
                    lambda_recon * vae_losses['recon_loss']
                    + beta * vae_losses['kl_z']
                    + beta * vae_losses['hierarchical_kl']
                )
                perceiver_loss = outputs['perceiver_total_loss']

        else:
            # No DWA: fixed scalar weights

            elbo_loss = (
                lambda_recon * vae_losses['recon_loss']
                + beta * vae_losses['kl_z']
                + beta * vae_losses['hierarchical_kl']
            )
            perceiver_loss = outputs['perceiver_total_loss']

            total_gen_loss = elbo_loss + perceiver_loss

        # Losses used for grad balancing & logging
        loss_dict = {
            "ELBO":        elbo_loss.reshape([]),
            "perceiver":   perceiver_loss.reshape([]),
        }

        L_elbo = loss_dict["ELBO"]
        L_ctx  = loss_dict["perceiver"]

        task_losses = [L_elbo, L_ctx]
        method = getattr(self, "grad_balance_method", "rgb")
        method = (method or "none").lower()

        # 5) Gradient balancing + step
        if method == "pcgrad":
            # PCGrad wrapper around an AdamW optimizer
            base_opt = self.gen_optimizer._optim

            # Clear grads & internal accumulators
            self.gen_optimizer.zero_grad()

            # Accumulate per-task gradients
            self.gen_optimizer.backward(task_losses)

            # Project conflicting gradients and set them on base optimizer params
            grads, shapes, has_grads = self.gen_optimizer._pack_accum_grads()
            pc_grad = self.gen_optimizer._project_conflicting(grads, has_grads)
            pc_grad = self.gen_optimizer._unflatten_grad(pc_grad, shapes[0])
            self.gen_optimizer._set_grad(pc_grad)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for g in base_opt.param_groups for p in g["params"]],
                self._grad_clip,
            )

            # Step with optional AMP scaler
            if self.gen_optimizer.scaler is not None:
                self.gen_optimizer.scaler.step(base_opt)
                self.gen_optimizer.scaler.update()
            else:
                base_opt.step()

            # Grad norm for logging
            grad_norm_sq = 0.0
            for group in base_opt.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        g = p.grad.data
                        grad_norm_sq += float((g.norm(2)).item() ** 2)
            grad_norm = (grad_norm_sq ** 0.5) if grad_norm_sq > 0 else 0.0

            # Reset accumulators
            self.gen_optimizer.zero_grad()
        elif method == "upgrad":
            loss_vec = torch.stack(task_losses, dim=0)   # shape [3]
            # -------- UPGrad via TorchJD (low memory) --------
            if not (hasattr(self, "_jd_engine") and hasattr(self, "_jd_weighting")) and self._jd_engine is None or self._jd_weighting is None:
                raise RuntimeError(
                    "grad_balance_method='upgrad' but TorchJD engine/weighting "
                    "are not initialized. Did you call setup_optimizer?"
                )

            self.gen_optimizer.zero_grad(set_to_none=True)

            # Gramian of Jacobian of [L_elbo, L_ctx] wrt generator params
            gramian = self._jd_engine.compute_gramian(loss_vec)   # shape [2, 2]

            # UPGrad weights for the 2 objectives
            weights = self._jd_weighting(gramian)                 # shape [2]
            # Weighted sum of losses → standard backward
            weighted_loss = (weights * loss_vec).sum()
            weighted_loss.backward()

            # Gradient clipping & step exactly as before
            gen_params = [p for g in self.gen_optimizer.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(gen_params, self._grad_clip)
            self.gen_optimizer.step()

            # For logging: compute grad_norm after clipping / before step if you prefer
            grad_norm_sq = 0.0
            for p in gen_params:
                if p.grad is not None:
                    g = p.grad.detach()
                    grad_norm_sq += float(g.norm(2).item() ** 2)
            grad_norm = grad_norm_sq ** 0.5
        else:
            # RGB / GradNorm / "none"
            self.gen_optimizer.zero_grad(set_to_none=True)

            if self.grad_balancer is not None and method in ("rgb", "gradnorm", "mgda"):
                # Keep GradNorm's internal epoch in sync, if present
                if hasattr(self.grad_balancer, "epoch"):
                    self.grad_balancer.epoch = getattr(self, "current_epoch", 0)

                if method == "gradnorm":
                    # Initialize loss buffer on first step
                    if getattr(self.grad_balancer, "train_loss_buffer", None) is None:
                        with torch.no_grad():
                            base = torch.stack([loss.detach() for loss in task_losses])
                        # shape [T, 1] as expected in GradNorm
                        self.grad_balancer.train_loss_buffer = base.view(-1, 1).to(self.device)

                    self.grad_balancer.backward(
                        task_losses,
                        mode="backward",
                        alpha=self.gradnorm_alpha,
                        log_grads=True,
                    )
                elif method == "mgda":
                    # MGDA: use per-task gradients to find minimum-norm weights.
                    # mgda_gn controls gradient normalization: "none", "l2", "loss", "loss+"
                    
                    self.grad_balancer.backward(
                        task_losses,
                        mgda_gn=self.mgda_gn,
                    )

                else:  # "rgb"
                    self.grad_balancer.backward(task_losses, mode="backward")
            else:
                # No multi-task balancer: just backprop total_gen_loss
                total_gen_loss.backward(retain_graph=True)

            # Clip and step with standard optimizer
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.gen_optimizer.param_groups for p in g["params"]],
                self._grad_clip,
            )
            self.gen_optimizer.step()

            # Grad norm for logging
            grad_norm_sq = 0.0
            for group in self.gen_optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        g = p.grad.data
                        grad_norm_sq = grad_norm_sq + float((g.norm(2)).item() ** 2)
            grad_norm = (grad_norm_sq ** 0.5) if grad_norm_sq > 0 else 0.0

        # 6) EMA updates for encoder/decoder
        with torch.no_grad():
            if hasattr(self, "ema_encoder"):
                self.ema_encoder.update()
            if hasattr(self, "ema_decoder"):
                self.ema_decoder.update()


        eff_comp = outputs['prior_params'][0]['pi'].max(1)[0].mean().item()
        top6_cov = outputs['prior_params'][0]['pi'].topk(6, dim=-1)[0].sum(dim=-1).mean().item()

        return {
            # VAE & auxiliaries
            **vae_losses,
            # Training stats
            'total_gen_loss': float(total_gen_loss.item()),
            'grad_norm': float(grad_norm),
            'effective_components': eff_comp,
            'Top 6 coverage': top6_cov,
        }
    
    def set_epoch(self, epoch: int):
        """Update current epoch for scheduling purposes"""
        self.current_epoch = epoch
    

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Generate unconditional samples from the model.

        Decoder expects a latent of size:
            latent_size = latent_dim + context_dim + hidden_dim
        so we build [z, c, h] here to match.
        """
        # 1) Sample a joint (h, c) vector used by the prior
        #    h_ctx shape: [B, hidden_dim + context_dim]
        h_ctx = torch.randn(num_samples,
                            self.hidden_dim + self.context_dim,
                            device=self.device)

        # (B) apply EMA E
        if hasattr(self, "ema_vdvae"):
            self.ema_vdvae.apply_shadow()

        x_np = self.vdvae.sample(num_samples, h_ctx)

        if hasattr(self, "ema_vdvae"):
            self.ema_vdvae.restore()

        # uint8 NHWC -> float NCHW in [-1,1]
        x = torch.from_numpy(x_np).permute(0, 3, 1, 2).contiguous().float() / 127.5 - 1.0
        return x
    
    @torch.no_grad()
    def generate_future_sequence(self, initial_obs, actions=None, horizon=2, decode_pixels=True):
        device = initial_obs.device
        B, T_ctx = initial_obs.shape[:2]

        # 1) Contexts for observed frames (no future AR here)
        ctx_obs, context_info = self.compute_global_context(
            initial_obs,
            num_context_frames=T_ctx,
            generate_future=False,
        )  # [B, T_ctx, C_ctx]

        # 2) Scan the observed segment (teacher-forced) to get h_{T_ctx-1}, z_{T_ctx-1}, a_{T_ctx-1}
        outputs = {"z": [], "h": [], "a": []}
        h = self.h0.expand(self.number_lstm_layer, B, -1).contiguous()
        c_rnn = self.c0.expand(self.number_lstm_layer, B, -1).contiguous()

        top_block = self.vdvae.decoder.dec_blocks[0]
        C = top_block.zdim
        res = top_block.base

        def sample_top_z_map_and_pooled(h_context: torch.Tensor):
            B_, Hc = h_context.shape
            h_map = h_context.view(B_, Hc, 1, 1).expand(B_, Hc, res, res)
            h_map = self.vdvae.add_coord_no_proj(h_map, scale=0.05)  # matches training :contentReference[oaicite:3]{index=3}
            h_tokens = h_map.permute(0, 2, 3, 1).reshape(B_ * res * res, Hc)
            pz_dist, _ = self.prior(h_tokens)
            z_tokens = pz_dist.sample()
            z_map = z_tokens.view(B_, res, res, C).permute(0, 3, 1, 2).contiguous()
            return z_map, z_map.view(B_, -1)  # z_flat is [B, top_zdim] (NOT [B, C])

        # c_prev is the "previous context" for the prior conditioning when t=0 (c_{-1}); zeros is fine.
        c_prev = torch.zeros(B, self.context_dim, device=device)

        for t in range(T_ctx):
            x_t = initial_obs[:, t]                           # [B,C,H,W]
            c_t = ctx_obs[:, t]                               # [B,C_ctx]
            a_t = actions[:, t] if actions is not None else torch.zeros(B, self.action_dim, device=device)

            x_t_nhwc = x_t.permute(0, 2, 3, 1).contiguous()
            h_context = torch.cat([h[-1], c_prev], dim=-1)

            vdvae_out = self.vdvae(x_t_nhwc, x_t_nhwc, h_context)
            eps = torch.randn_like(vdvae_out["top_q_mean_map"])
            z_map = vdvae_out["top_q_mean_map"] + eps * torch.exp(0.5 * vdvae_out["top_q_logvar_map"])
            z_t = z_map.contiguous().view(B, -1)             # [B, top_zdim]

            rnn_in = torch.cat([z_t, c_t, a_t], dim=-1)
            _, (h, c_rnn) = self._rnn(rnn_in, h, c_rnn, torch.ones(B, device=device))

            c_prev = c_t
            outputs["z"].append(z_t)
            outputs["h"].append(h[-1])
            outputs["a"].append(a_t)

        # Seed "previous-step" variables at t = T_ctx-1
        z_prev = outputs["z"][-1]
        a_prev = outputs["a"][-1]

        # 3) Future imagination (consistent timestamps)
        z_futures, c_futures, a_futures, h_futures = [], [], [], []
        z_top_maps = []

        for k in range(horizon):
            # action at the NEW timestep (t = T_ctx + k)
            a_cur = (
                actions[:, T_ctx + k]
                if actions is not None and actions.shape[1] > (T_ctx + k)
                else torch.zeros(B, self.action_dim, device=device)
            )

            h_top = h[-1]  # this is h_t (top layer) for the *previous* timestep

            # predict c_cur = c_{t+1} from (c_t, a_t, z_t, h_t)  (matches training) :contentReference[oaicite:4]{index=4}
            c_cur = self.context_predictor(
                context_seq=c_prev,
                action_seq=a_prev,
                latent_seq=z_prev,
                hidden_seq=h_top,
            )

            # sample z_cur = z_{t+1} from prior p(z_{t+1} | h_t, c_t)
            h_context = torch.cat([h_top, c_prev], dim=-1)
            z_map, z_cur = sample_top_z_map_and_pooled(h_context)

            # step RNN to get h_{t+1} using CURRENT (z_{t+1}, c_{t+1}, a_{t+1})
            rnn_in = torch.cat([z_cur, c_cur, a_cur], dim=-1)
            _, (h, c_rnn) = self._rnn(rnn_in, h, c_rnn, torch.ones(B, device=device))

            # collect
            z_futures.append(z_cur)
            c_futures.append(c_cur)
            a_futures.append(a_cur)
            h_futures.append(h[-1])
            z_top_maps.append(z_map)

            # advance "previous" variables for next loop
            c_prev = c_cur
            z_prev = z_cur
            a_prev = a_cur

        # 4) Pack outputs (same as your current code)
        z_futures = torch.stack(z_futures, dim=1) if z_futures else torch.empty(B, 0, self.top_zdim, device=device)
        c_futures = torch.stack(c_futures, dim=1) if c_futures else torch.empty(B, 0, self.context_dim, device=device)
        a_futures = torch.stack(a_futures, dim=1) if a_futures else torch.empty(B, 0, self.action_dim, device=device)
        h_futures = torch.stack(h_futures, dim=1) if h_futures else torch.empty(B, 0, h[-1].shape[-1], device=device)

        result = {
            "z_obs": torch.stack(outputs["z"], dim=1),
            "h_obs": torch.stack(outputs["h"], dim=1),
            "a_obs": torch.stack(outputs["a"], dim=1),
            "c_obs": ctx_obs,
            "z_future": z_futures,
            "c_future": c_futures,
            "a_future": a_futures,
            "h_future": h_futures,
        }

        # 5) Pixel rollouts: Perceiver AR (unchanged) + VDVAE decode from sampled top-z maps (unchanged)
        if decode_pixels:
            result["pixels_future"] = self.perceiver_model.generate_autoregressive(
                context_videos=initial_obs,
                num_frames_to_generate=horizon,
                temperature=1.0, top_k=None, top_p=None,
            )

        if len(z_top_maps) > 0:
            z_top = torch.stack(z_top_maps, dim=1)  # [B,Tf,C,res,res]
            B_, Tf = z_top.shape[:2]
            z_top_flat = z_top.view(B_ * Tf, C, res, res)
            latents = [z_top_flat] + [None] * (len(self.vdvae.decoder.dec_blocks) - 1)
            px_z = self.vdvae.decoder.forward_manual_latents(B_ * Tf, latents, t=None)
            x_np = self.vdvae.decoder.out_net.sample(px_z)  # uint8 NHWC
            x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32) / 127.5 - 1.0
            x = x.permute(0, 3, 1, 2).contiguous()
            result["vae_future"] = x.view(B_, Tf, *x.shape[1:])
        else:
            result["vae_future"] = torch.empty(B, 0, self.input_channels, self.image_size, self.image_size, device=device)

        return result

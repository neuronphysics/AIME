from logging import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Gamma, Categorical, Independent, MixtureSameFamily
from typing import Dict, Tuple, Union, Optional, Any
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
from vis_networks import (
    EMA, TemporalDiscriminator,
    AddEpsilon, check_tensor
)
from nvae_architecture import VAEEncoder, VAEDecoder, GramLoss
from VRNN.perceiver.position import RoPE
from VRNN.RGB import RGB
from VRNN.lstm import LSTMLayer

from VRNN.perceiver.video_prediction_perceiverIO import CausalPerceiverIO
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
    Backward-compatible torch.load:
    - On PyTorch >= 2.6, sets weights_only=False explicitly.
    - On older versions, just calls torch.load(path, map_location=...).
    Use ONLY with checkpoints you trust.
    """
    import torch
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

        # Register parameter names
        self.register_parameter('gamma_prior_a', self.gamma_a)
        self.register_parameter('gamma_prior_b', self.gamma_b)
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
    def forward(self, h: torch.Tensor, n_samples: int = 10, use_rand_perm: bool = True, truncation_threshold: float = 0.995) -> Tuple[torch.Tensor, Dict]:
       
        
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
        self.component_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(hidden_dim, 2 * latent_dim * max_components),
            nn.LayerNorm(2 * latent_dim * max_components),
        )
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

        prior_weights = prior_params['pi']
        prior_means = prior_params['means']
        prior_logvars = prior_params['log_vars']
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
            
            alpha_k = alpha[:, k:k+1] if k < alpha.shape[1] else alpha[:, -1:] # Get alpha for current stick

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
        pi, kumar_params = self.stick_breaking(h, n_samples, use_rand_perm=True)
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
def _temporarily_disable_ckpt(*mods):
    flags = []
    for m in mods:
        flags.append(getattr(m, "use_checkpoint", None))
        if hasattr(m, "use_checkpoint"):
            m.use_checkpoint = False
    try:
        yield
    finally:
        for m, f in zip(mods, flags):
            if f is not None:
                m.use_checkpoint = f

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
        img_perceiver_channels:int,
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
        num_encoder_perceiver_layers: int = 4,
        num_perceiver_heads: int = 8,
        num_latent_perceiver: int = 128,
        num_latent_channels_perceiver: int = 128,
        num_codebook_perceiver: int = 1024,
        perceiver_code_dim: int = 256,
        downsample_perceiver: int = 4,
        perceiver_lr_multiplier: float = 0.7,
        use_ctx_checkpoint: bool = True,
        contrastive_weight: float = 0.01,
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
        self.contrastive_weight = contrastive_weight 
        #initialization different parts of the model
        
        self._init_perceiver_context()
        self._init_vrnn_dynamics(use_orthogonal=use_orthogonal,number_lstm_layer=number_lstm_layer)
        self._init_encoder_decoder()
        self._init_discriminators( img_disc_layers, patch_size, num_heads= disc_num_heads)

        self.use_ctx_checkpoint = use_ctx_checkpoint
        # DP-GMM prior
        self.prior = DPGMMPrior(max_components, 
                                self.latent_dim, 
                                hidden_dim + self.context_dim, #This is because that the input for the prior is the hidden state of recurrent model plus context which is coming from perceiver
                                device, prior_alpha=prior_alpha,
                                prior_beta=prior_beta)

        # Initialize weights
        
        self.to(device)
        
        # Setup optimizers
        self.has_optimizers = True
        self._setup_optimizers(learning_rate, weight_decay)

    def _warm_build_unet_adapters(self, batch_size: int = 1, dtype: torch.dtype | None = None):
        if dtype is None:
            dtype = torch.float32

        x_dummy = torch.zeros(
            batch_size,
            self.input_channels,
            self.image_size,
            self.image_size,
            device=self.device,
            dtype=dtype,
        )

        enc_was_training = self.encoder.training
        dec_was_training = self.decoder.training
        self.encoder.eval()
        self.decoder.eval()

        try:
            with torch.no_grad(), _temporarily_disable_ckpt(self.encoder, self.decoder):
                # 1) Run encoder once to get z and the *actual* UNet skips
                z, _, _, _ = self.encoder(x_dummy)
                skips = self.encoder.get_unet_skips(
                    x_dummy, levels=("C1", "C2", "C3", "C4", "C5", "C6"), detach=True
                )

                # 2) Use these real skips to prebuild all 1x1 adapters
                if hasattr(self.decoder, "build_unet_adapters_from_skips"):
                    self.decoder.build_unet_adapters_from_skips(skips)

                # 3) Optional sanity check: dry forward to ensure no missing adapters
                _ = self.decoder(z, skips=skips)

                # We don't want decoder to rely on any stale cached skips
                if hasattr(self.decoder, "_pending_skips"):
                    self.decoder._pending_skips = None
        finally:
            if enc_was_training:
                self.encoder.train()
            if dec_was_training:
                self.decoder.train()
                
    def _init_encoder_decoder(self):
        # Encoder network
        encoder_channels = [64, 64, 96, 128, 128, 192]
        self.encoder = VAEEncoder(
                                  channel_size_per_layer=encoder_channels,
                                  layers_per_block_per_layer=[1,1,2,1,1,1],
                                  latent_size=self.latent_dim,
                                  width=self.image_size,
                                  height=self.image_size,
                                  num_layers_per_resolution=[1,2,2,1],
                                  mlp_hidden_size=self.hidden_dim,
                                  channel_size=64,
                                  input_channels=self.input_channels,
                                  downsample=4,
                                  use_se=False,
                                  dropout= self.dropout,
                                  ).to(self.device)
        # Decoder network 
        self.decoder = VAEDecoder(
                                  latent_size=self.latent_dim,
                                  width=self.image_size,
                                  height= self.image_size,
                                  channel_size_per_layer=[192,128,128,96,64,64],
                                  layers_per_block_per_layer=[1,1,1,2,1,1],
                                  num_layers_per_resolution=[1,2,2,1],
                                  reconstruction_channels=self.input_channels,
                                  downsample=4,
                                  mlp_hidden_size= self.hidden_dim,
                                  dropout= self.dropout,
                                  use_se=False,
                                  encoder_channel_size_per_layer=encoder_channels,
                                  skip_levels_last_n=3
                                  ).to(self.device)
        
        self.gram_loss = GramLoss(
            apply_norm=True,
            img_level=True,             
            remove_neg=True,           # set True to match DINOv3’s “positives-only” option
            remove_only_teacher_neg=False,
        )
        # Warm-build UNet adapters before creating optimizer
        self._warm_build_unet_adapters(batch_size=1)

        self.encoder.gradient_checkpointing_enable()
        self.decoder.gradient_checkpointing_enable()
       
        self.ema_encoder = EMA(self.encoder, decay=0.995, use_num_updates=True)
        self.ema_decoder = EMA(self.decoder, decay=0.995, use_num_updates=True)

        self.apply(self.init_weights)

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
                z_dim=self.latent_dim,         
                device=self.device,
            )

    def _init_vrnn_dynamics(self,use_orthogonal: bool = True, number_lstm_layer: int = 1):
        """Initialize VRNN components with context conditioning"""
        # Feature extractors        
        
        # VRNN recurrence: h_t = f(h_{t-1}, z_t, c_t, a_t)
        self._rnn = LSTMLayer(
            input_size=self.latent_dim + self.action_dim + self.context_dim ,  # z_t + c_t +  a_t
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

    def contrastive_loss(
        self,
        real_features: torch.Tensor,  # [B, T, D]
        fake_features: torch.Tensor,   # [B, T, D]
        temperature: float = 0.1,
        alpha: float = 0.7,
        temporal_margin: int = 5
    ) -> torch.Tensor:
        """
        Temporal coherence 
        """
        B, T, D = real_features.shape
        if T < 2 or B < 2:
            return torch.tensor(0.0, device=real_features.device)
        
        # 1. Normalize features
        real_norm = F.normalize(real_features, p=2, dim=-1)
        fake_norm = F.normalize(fake_features, p=2, dim=-1)
        
        # 2. Intra-video alignment (real_t to fake_t)
        intra_sim = torch.einsum('btd,btd->bt', real_norm, fake_norm) / temperature
        intra_loss = -intra_sim.mean()
        
        # 3. Inter-video separation - FIXED
        # Average pool over time dimension
        real_pooled = real_norm.mean(dim=1)  # [B, D]
        fake_pooled = fake_norm.mean(dim=1)  # [B, D]
        
        # Compute similarity matrix between sequences
        inter_sim = torch.matmul(real_pooled, fake_pooled.T) / temperature  # [B, B]
        
        # Diagonal should be high (same video), off-diagonal low (different videos)
        inter_labels = torch.arange(B, device=real_features.device)
        inter_loss = F.cross_entropy(inter_sim, inter_labels)
        
        # 4. Temporal coherence with margin-based negatives (THIS PART IS GOOD)
        anchors = fake_norm[:, :-1]  # [B, T-1, D]
        positives = fake_norm[:, 1:]  # [B, T-1, D]
        
        # Compute similarities
        pos_sim = torch.sum(anchors * positives, dim=-1) / temperature  # [B, T-1]
        neg_sim = torch.einsum('btd,bkd->btk', anchors, fake_norm) / temperature  # [B, T-1, T]
        
        # Mask negatives within temporal margin
        time_indices = torch.arange(T, device=anchors.device)
        time_diff = torch.abs(time_indices[:-1, None] - time_indices)  # [T-1, T]
        mask = time_diff >= temporal_margin
        mask = mask.unsqueeze(0).expand(B, -1, -1)  # [B, T-1, T]
        
        # Apply mask
        neg_sim = neg_sim.masked_fill(~mask, -1e4)
        
        # Combine positive and negative logits
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # [B, T-1, 1+T]
        labels = torch.zeros(B, T-1, dtype=torch.long, device=logits.device)
        
        temporal_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            labels.view(-1)
        )
        
        # Weighted combination
        return alpha * (intra_loss + inter_loss) + (1 - alpha) * temporal_loss    

    def _setup_optimizers(self, learning_rate: float, weight_decay: float) -> None:
        
        def get_params(*modules, exclude_params=None):
            params = []
            exclude_ids = {id(p) for p in (exclude_params or [])}
            for m in modules:
                if m is None:
                    continue
                if isinstance(m, nn.Module):
                    params.extend([p for p in m.parameters()
                                if p.requires_grad and id(p) not in exclude_ids])
                elif isinstance(m, nn.Parameter):
                    if m.requires_grad and id(m) not in exclude_ids:
                        params.append(m)
            return params

        def split_by_weight_decay(params, wd):
            decay, no_decay = [], []
            for p in params:
                (decay if p.ndim >= 2 else no_decay).append(p)
            groups = []
            if decay:    groups.append({"params": decay,    "weight_decay": wd})
            if no_decay: groups.append({"params": no_decay, "weight_decay": 0.0})
            return groups
        
        perceiver_params = get_params(self.perceiver_model, self.context_predictor, self.frame_slot_pool)
        param_groups = []
        if perceiver_params:  # guard empty
            perceiver_groups = split_by_weight_decay(perceiver_params, weight_decay)
            for g in perceiver_groups:
                g["lr"] = learning_rate * self.perceiver_lr_multiplier
                # Optional: slightly less WD on perceiver
                if g.get("weight_decay", 0.0) > 0: g["weight_decay"] *= 0.5
            param_groups.extend(perceiver_groups)

        trunk_modules = [self.encoder, self.decoder, self.prior, self._rnn, self.rnn_layer_norm, self.gram_loss]

        
        scalar_params = [self.h0, self.c0]

        param_groups.extend(split_by_weight_decay(get_params(*trunk_modules), weight_decay))



        scalars = [p for p in scalar_params if isinstance(p, nn.Parameter) and p.requires_grad]
        if scalars:
            param_groups.append({"params": scalars, "lr": learning_rate, "weight_decay": 0.0001})


        self.gen_optimizer = torch.optim.AdamW(
            param_groups, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8
        )
    
        self.grad_balancer = RGB()

        # tasks in the same order losses were passed in training_step_sequence
        self.grad_balancer.task_name = ["elbo", "perceiver", "adversarial"]
        self.grad_balancer.task_num  = len(self.grad_balancer.task_name)
        self.grad_balancer.device    = self.device
        self.grad_balancer.rep_grad  = False        # operate on shared params (θ), not reps
        self.grad_balancer.alpha_steps = 1  # Reduce from 3 to 1
        self.grad_balancer.update_interval = 2  # Update every 2 steps instead of every step
        self.grad_balancer.lr_inner = 0.05  # Smaller learning rate for stability

        # give RGB access to *generator* params only
        self.grad_balancer.get_share_params = lambda: (
            p for g in self.gen_optimizer.param_groups for p in g["params"]
        )


        # Discriminator optimizer (separate)
        if hasattr(self, 'image_discriminator'):
            disc_params = [p for p in self.image_discriminator.parameters() if p.requires_grad]
            self.img_disc_optimizer = torch.optim.Adam(
                disc_params,
                lr=learning_rate * 0.1,   
                betas=(0.0, 0.9),          # canonical for WGAN-GP
                weight_decay=0.00005
            )
        
        # Setup schedulers
        self._setup_schedulers()

    def _setup_schedulers(self):
        """Setup learning-rate schedulers for all optimizers (Option B)."""

        # Store initial lrs for trunk optimizer
        for group in self.gen_optimizer.param_groups:
            group["initial_lr"] = group["lr"]

        # 1) Trunk scheduler (MGDA step uses this optimizer)
        self.gen_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.gen_optimizer, mode='min', factor=0.2, patience=10, threshold=0.0001, 
        )

        # 3) Discriminator scheduler (unchanged)
        self.img_disc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.img_disc_optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-7
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
            'observations': observations,
            'reconstructions': [],
            'mdl_params': [], 
            'latents': [],
            'z_means': [],
            'z_logvars': [],
            'hidden_states': [],
            'context_states': [],
            'prior_params': [],
            'posterior_params': [],
            'kl_losses': [],
            'kumaraswamy_kl_losses': [],
            'kl_latents': [],
            'reconstruction_losses': [],
            'gram_enc': [],
            'K_eff': [],
        }
        outputs['perceiver_reconstructed_img'] = context_info['reconstruct']
        # Process sequence step by step
        for t in range(seq_len):
            # Get current inputs
            o_t = observations[:, t]
            a_t = actions[:, t]
            # Get context (global or local)
            c_t = context_sequence[:, t]
            
            outputs['context_states'].append(c_t)
            
            #Inference Network q(z_t|o_≤t, z_<t) 
            with self.encoder.enable_caching():
                # Inference Network q(z_t | o_<=t, z_<t)
                z_t, z_mean_t, z_logvar_t, _ = self.encoder(o_t)

                # Use cached skips (no recompute). 
                skips = self.encoder.get_unet_skips(
                    x=None, levels=("C1", "C2", "C3", "C4", "C5", "C6"), detach=False
                )
                # Decoder forward
                logit_probs, means, log_scales, coeffs = self.decoder(z_t, skips=skips)

                q_params = {'mean': z_mean_t, 'logvar': z_logvar_t}
                outputs['posterior_params'].append(q_params)
                # store gram matrix for encoder decoder features
                outputs['gram_enc'].append( self.encoder_gram_loss_student_teacher(o_t, levels=("C1", "C2", "C3", "C4", "C5", "C6")) )

            # Store latent statistics
            outputs['latents'].append(z_t)
            outputs['z_means'].append(z_mean_t)
            outputs['z_logvars'].append(z_logvar_t)
            
            # === Prior Network p(z_t|o_<t, z_<t) ===
            h_context = torch.cat([h[-1], c_prev], dim=-1)
            
            # Get DP-GMM prior distribution
            prior_dist, prior_params = self.prior(h_context)
            outputs['prior_params'].append(prior_params)
            
            outputs['mdl_params'].append({
                    'logit_probs': logit_probs,
                    'means': means,
                    'log_scales': log_scales,
                    'coeffs': coeffs
                })
            reconstruction_t = self.decoder.mdl_head.sample(logit_probs, means, log_scales, coeffs, scale_temp=0.3)
            outputs['reconstructions'].append(reconstruction_t)
                
            # Compute NLL loss instead of MSE
            outputs['reconstruction_losses'].append(self.decoder.mdl_head.nll(o_t, logit_probs, means, log_scales, coeffs).mean())

            # === KL Divergence Computation ===
            # 1. KL between posterior and DP-GMM prior

            kl_z = self.prior.compute_kl_divergence_mc(z_mean_t, z_logvar_t, prior_params)
            outputs['kl_latents'].append(kl_z)
            
            # 2. KL for stick-breaking (Kumaraswamy vs Beta and Gamma prior vs Gamma posterior)
            kumar_beta_kl = self.prior.compute_kl_loss(
                prior_params, 
                prior_params['alpha'], 
                h_context,
            )
            # Stick-breaking KL
            outputs['kumaraswamy_kl_losses'].append(kumar_beta_kl)
            
            # 4. Compute effective number of components and penalty for unused components
            K_eff = self.prior.get_effective_components(prior_params['pi'])


            outputs['K_eff'].append(K_eff.float().mean())
           # Negative for ELBO

            
            # Total KL for this timestep
            total_kl = kl_z + kumar_beta_kl
            outputs['kl_losses'].append(total_kl)
            # === VRNN Recurrence ===
            #build mask based on done signals
            if dones is None:
                mask_t = torch.ones(batch_size, device=self.device)
            else:
                if t == 0:
                    first_t = torch.ones(batch_size, device=self.device)
                else:
                    first_t = dones[:, t-1].to(torch.float32).to(self.device)
                mask_t = 1.0 - first_t

            rnn_input = torch.cat([z_t, c_prev, a_t], dim=-1)
            rnn_output, (h, c) = self._rnn(
                rnn_input, h, c, 
                mask_t
            )
            c_prev = c_t
            rnn_output = self.rnn_layer_norm(rnn_output)
            outputs['hidden_states'].append(h[-1])

        outputs['K_eff'] = torch.stack(outputs['K_eff']).mean() 
        # === Compute aggregated losses ===
        # Stack temporal sequences
        for key in ['reconstructions', 'latents', 'hidden_states', 'context_states']:
            if outputs[key]:
                outputs[key] = torch.stack(outputs[key], dim=1)
        outputs['perceiver_total_loss'] = context_info["total_loss"]
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

            # Hyperparameter for this auxiliary loss
            lambda_ctx = getattr(self, "lambda_ctx_pred", 0.1)

            # Add to perceiver task loss (so it goes into MGDA "perceiver" bucket)
            outputs['perceiver_total_loss'] = outputs['perceiver_total_loss'] + lambda_ctx * ctx_pred_loss

            outputs['context_pred_loss'] = ctx_pred_loss
        # Add auxiliary outputs
        outputs['perceiver_reconstructed_img'] = context_info['reconstruct']
        outputs['perceiver_vq_loss']          = context_info['vq_loss']
        outputs['perceiver_ce_loss']          = context_info['ce_loss']
        outputs['perceiver_lpips_loss']       = context_info['lpips_loss']

        return outputs

    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Encode
        z, z_mean, z_logvar, h = self.encoder(x)

        # Get prior distribution and parameters
        prior_dist, prior_params = self.prior(h)
        skips = self.encoder.get_unet_skips(x, levels=("C1", "C2", "C3", "C4", "C5", "C6"))        # or encoder.get_unet_skips(x, levels=("C3","C4","C5"))
        #self.decoder.set_unet_skips(skips, mode="concat")
        # Decode
        logit_probs, means, log_scales, coeffs = self.decoder(z, skips=skips)

        # Sample reconstruction for visualization
        reconstruction = self.decoder.mdl_head.sample(logit_probs, means, log_scales, coeffs, scale_temp=0.3)


        return reconstruction, {
            'z': z,
            'z_mean': z_mean,
            'z_logvar': z_logvar,
            'prior_dist': prior_dist,
            **prior_params
        }
    
    def compute_total_loss(self, 
                           observations, 
                           actions=None, 
                           beta=1.0,  
                           lambda_recon=1.0, 
                           lambda_gram=0.05,
                           ):
        """
        Corrected loss computation recognizing that kumar_beta_kl already includes alpha prior KL
        """
        outputs = self.forward_sequence(observations, actions)
        losses = {}
        
        # === Reconstruction Term (Positive in Loss) ===
        # This is already negative value -E_q[log p(x|z)] under Gaussian assumption
        losses['recon_loss'] = torch.stack(outputs['reconstruction_losses']).mean()  if outputs['reconstruction_losses'] else torch.tensor(0.0).to(self.device)
        losses['gram_enc_loss'] = torch.stack(outputs['gram_enc']).mean() if outputs['gram_enc'] else torch.tensor(0.0).to(self.device)
        # === KL Divergence Terms ===
        
        # 1. Latent KL: KL[q(z|x) || p(z|h,c)]
        losses['kl_z'] = torch.stack(outputs['kl_latents']).mean() if outputs['kl_latents'] else torch.tensor(0.0).to(self.device)
        
        # 2. Hierarchical KL (includes BOTH stick-breaking AND alpha prior)
        # This is what compute_kl_loss returns
        losses['hierarchical_kl'] = torch.stack(outputs['kumaraswamy_kl_losses']).mean() if outputs['kumaraswamy_kl_losses'] else torch.tensor(0.0).to(self.device)
                
        # Auxiliary losses
        losses['perceiver_loss'] = outputs['perceiver_total_loss']


        
        # === Total Loss (Minimizing Negative ELBO) ===
        losses['total_vae_loss'] = (
            # Reconstruction (positive - we want to minimize error)
            lambda_recon * losses['recon_loss'] +
            # KL terms 
            beta * losses['kl_z'] +
            beta * losses['hierarchical_kl']+
            lambda_gram * losses['gram_enc_loss'] 
        )
        
        return losses, outputs

    
        
    def compute_conditional_entropy(self, logits, params):
        """
        Compute conditional entropy of cluster assignments with temperature scaling.
        H(C|Z) = -E[log P(C|Z)]

        """
        # Ensure temperature stays above minimum value
        eps = torch.finfo(params['pi'].dtype).eps

        # Computer cluster probabilities with temperature scaling
        probs = F.softmax(logits , dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Compute entropy for each sample
        entropy_per_sample = -torch.sum(probs * log_probs, dim=-1)

        # Compute mean entropy
        mean_entropy = entropy_per_sample.mean()
        mixing_proportions = params['pi']
        mixing_proportions_safe = mixing_proportions.clamp(min=eps)
        #The entropy H(π) tells how "spread out" or "diverse" the cluster usage is
        mixing_entropy = -torch.sum(
            mixing_proportions_safe * torch.log(mixing_proportions_safe), 
            dim=-1
        ).mean() 
        # Compute additional clustering statistics
        cluster_stats = {
            'mean_prob_per_cluster': probs.mean(0),
            'mixing_entropy': mixing_entropy.item(),
            'max_prob_per_sample': probs.max(1)[0].mean(),
            'entropy_std': entropy_per_sample.std() if entropy_per_sample.numel() > 1 else torch.tensor(0.0, device=entropy_per_sample.device),
            'active_clusters': (probs.mean(0) > eps).sum().item()
        }

        return mean_entropy + mixing_entropy, cluster_stats


    def encoder_gram_loss_student_teacher(self, x, levels=('C3','C4'), pool=2):
        # Student encoder features
        enc_s = self.encoder.get_unet_skips(x=None, levels=levels, detach=False)  # dict { 'HxW': Tensor[B,C,H,W] }
        if not enc_s:  # cache may be None or empty outside caching context
            enc_s = self.encoder.get_unet_skips(x, levels=levels, detach=False)

        # Teacher (EMA) encoder features
        self.ema_encoder.apply_shadow()
        with torch.inference_mode():
            enc_t = self.encoder.get_unet_skips(x, levels=levels, detach=True)
        self.ema_encoder.restore()

        loss = 0.0
        common = set(enc_s.keys()) & set(enc_t.keys())
        for k in common:
            Fs, Ft = enc_s[k], enc_t[k]              # [B,C,H,W]
            if pool and (Fs.shape[-1] % pool == 0):  # optional downsample 
                Fs = torch.nn.functional.avg_pool2d(Fs, pool)
                Ft = torch.nn.functional.avg_pool2d(Ft, pool)
            # reshape to (B, N, D) = (B, HW, C)
            s = Fs.flatten(2).transpose(1, 2).contiguous()
            t = Ft.flatten(2).transpose(1, 2).contiguous()
            # call the DINOv3 loss (it normalizes internally when apply_norm=True)
            loss = loss + self.gram_loss(s, t, img_level=True)
        return loss

    def compute_gradient_penalty(self, discriminator, real_x, fake_x, z, device):
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

        d_hat = discriminator(x_hat, z=z_hat)["final_score"]  # [B, 1]

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

    def discriminator_step(
        self,
        real_images: torch.Tensor, #[B, T, C, H, W]
        fake_images: torch.Tensor, #[B, T, C, H, W]
        latents: torch.Tensor,  #[B, T, Z]
        WGAN_GP_Coeff: float = 5.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Training step for both discriminators
        """
        seq_len = real_images.shape[1]  # Get sequence length from real images
        # Temporal Image Discriminator
        real_img_outputs = self.image_discriminator(real_images, z=latents.detach())
        fake_img_outputs = self.image_discriminator(fake_images.detach(), z=latents.detach())

        # Extract final scores
        real_img_score = real_img_outputs['final_score']
        fake_img_score = fake_img_outputs['final_score']
        # Gradient penalty requires computing second-order gradients
        # Gradient penalty for images (sample random interpolation points in sequence)
        img_gp = self.compute_gradient_penalty(
            self.image_discriminator,
            real_images,
            fake_images.detach(),
            latents,
            self.device
        )


        img_disc_loss = (
            torch.mean(fake_img_score) - torch.mean(real_img_score) +
            WGAN_GP_Coeff * img_gp/ seq_len
        )

        # Temporal consistency losses
        img_consistency_loss = torch.mean(
            fake_img_outputs['per_frame_scores'].std(dim=1)
        )
    
        if self.has_optimizers and hasattr(self, 'img_disc_optimizer'):
            # Update image discriminator
            self.img_disc_optimizer.zero_grad()
            img_disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.image_discriminator.parameters(), self._grad_clip)
            self.img_disc_optimizer.step()  
    
        return {
            'img_disc_loss': img_disc_loss,
            'img_gp': img_gp,
            'real_img_score': real_img_score.mean(),
            'fake_img_score': fake_img_score.mean(),
            'img_temporal_score': real_img_outputs['temporal_score'].mean(),
            'img_consistency_loss': img_consistency_loss,
        }
    
    def compute_feature_matching_loss(self, real_features, fake_features):
        
        real_mean = real_features.mean(dim=2)  # [B, T, D]
        fake_mean = fake_features.mean(dim=2)

        return F.l1_loss(fake_mean, real_mean.detach())

    def compute_adversarial_losses(
        self,
        x: torch.Tensor, #[B, T, C, H, W  ]
        reconstruction: torch.Tensor, #[B, T, C, H, W]
        z_seq: torch.Tensor, #[B, T, Z]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adversarial losses for both image and latent space
        """
        D = self.image_discriminator
        flags = [p.requires_grad for p in D.parameters()] #freeze Discriminator parameters
        for p in D.parameters():
            p.requires_grad_(False)
        
        fake_img_outputs = self.image_discriminator(reconstruction, z=z_seq, return_features=True)
        img_adv_loss = -torch.mean(fake_img_outputs['final_score'])
        #reward for generating temporally consistent images
    
        real_frame_features = x.reshape(x.shape[0], x.shape[1], -1)  # Flatten spatial dimensions
        #I want it to be fake image generated with size [B, T, C, H, W]
        fake_frame_features = reconstruction.reshape(x.shape[0], x.shape[1], -1)

        temporal_loss_frames =self.contrastive_loss(
        real_features=real_frame_features,
        fake_features=fake_frame_features
        )
            
        # === Feature Matching Loss ===
        # This helps stabilize training
        real_features = self.image_discriminator(x, z=z_seq, return_features=True)['hidden_3d']

        # L1 loss between feature statistics
        feature_match_loss = self.compute_feature_matching_loss(real_features, fake_img_outputs['hidden_3d'])
        # Restore Discriminator parameter gradients
        for p, f in zip(D.parameters(), flags):
            p.requires_grad_(f)

        return img_adv_loss , temporal_loss_frames, feature_match_loss

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
        device = videos.device
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
                                n_critic: int = 3,
                                lambda_img: float = 0.2,
                                lambda_recon: float = 1.0,
                                lambda_gram: float = 0.25,
                                ) -> Dict[str, torch.Tensor]:
        self.train()
        if getattr(self, "image_discriminator", None) is not None:
            self.image_discriminator.train()

        observations = self.prepare_images_for_training(observations)
        warmup_factor = self.get_warmup_factor()
        
        vae_losses, outputs = self.compute_total_loss(
            observations, actions, beta,
            lambda_recon, lambda_gram
        )
        lambda_img_eff = (lambda_img * warmup_factor) if warmup_factor > 0.0 else 0.0

        # ---- 4) Compose task losses for MGDA (5 tasks) ----
        elbo_loss = (
            lambda_recon * vae_losses['recon_loss']
            + beta * vae_losses['kl_z']
            + beta * vae_losses['hierarchical_kl']
            + lambda_gram * vae_losses['gram_enc_loss']
        )
        
        perceiver_loss = vae_losses['perceiver_loss']
        disc_losses_list = []
        for _ in range(n_critic):
            disc_loss = self.discriminator_step(
                real_images=observations,
                fake_images=outputs['reconstructions'],  # discriminator_step detaches internally
                latents=outputs['latents'],
            )
            disc_losses_list.append(disc_loss)

        avg_disc_losses: Dict[str, torch.Tensor] = {}
        if disc_losses_list:
            avg_disc_losses = {
                k: sum(d[k] for d in disc_losses_list) / len(disc_losses_list)
                for k in disc_losses_list[0].keys()
            }
        img_adv_loss, temporal_adv_loss, feat_match_loss = self.compute_adversarial_losses(
            x=observations,
            reconstruction=outputs['reconstructions'],
            z_seq=outputs['latents'],
        )
        adv_base = img_adv_loss + temporal_adv_loss + feat_match_loss
        adv_loss = lambda_img_eff * adv_base  # for logging and total_gen_loss
        # Total generator loss (tensor), used both for logging and LR scheduling
        total_gen_loss = (
            elbo_loss
            + perceiver_loss
            + (lambda_img_eff * (img_adv_loss + temporal_adv_loss + feat_match_loss))
        )
        loss_dict = {
            "elbo":        elbo_loss.reshape([]),
            "perceiver":   perceiver_loss.reshape([]),
            "adversarial": adv_base.reshape([]),  # <-- store unscaled base
        }

        L_elbo      = loss_dict["elbo"]
        L_perceiver = loss_dict["perceiver"]
        L_adv       = lambda_img_eff * loss_dict["adversarial"]  # <-- single scaling

        task_losses = [L_elbo, L_perceiver, L_adv]

        self.gen_optimizer.zero_grad(set_to_none=True)
        # RGB computes per-task grads and writes back the rotated aggregate grad
        self.grad_balancer.backward(task_losses, mode="backward")

        torch.nn.utils.clip_grad_norm_(
            [p for g in self.gen_optimizer.param_groups for p in g["params"]],
            self._grad_clip
        )
        self.gen_optimizer.step()
        
        grad_norm_sq = 0.0
        for group in self.gen_optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    g = p.grad.data
                    grad_norm_sq = grad_norm_sq + float((g.norm(2)).item() ** 2)
        grad_norm = (grad_norm_sq ** 0.5) if grad_norm_sq > 0 else 0.0

        with torch.no_grad():
            if hasattr(self, "ema_encoder"): self.ema_encoder.update()
            if hasattr(self, "ema_decoder"): self.ema_decoder.update()


        avg_max_prob = torch.stack(
            [p["logit_probs"].softmax(1).max(1)[0].mean() for p in outputs["mdl_params"]]
        ).mean()


        eff_comp = outputs['prior_params'][0]['pi'].max(1)[0].mean().item()
        top6_cov = outputs['prior_params'][0]['pi'].topk(6, dim=-1)[0].sum(dim=-1).mean().item()

        return {
            # VAE & auxiliaries
            **vae_losses,
            # Discriminators
            **avg_disc_losses,
            # Adversarial gen
            'img_adv_loss': float(img_adv_loss.item()),
            'temporal_adv_loss': float(temporal_adv_loss.item()),
            'feat_match_loss': float(feat_match_loss.item()),
            # Training stats
            'total_gen_loss': float(total_gen_loss.item()),
            'grad_norm': float(grad_norm),
            'effective_components': eff_comp,
            'Top 6 coverage': top6_cov,
            'mdl_avg_max_mixture_prob': avg_max_prob.item(),
            'mdl_effective_mixtures': (1.0 / avg_max_prob).item(),
        }

    
    def set_epoch(self, epoch: int):
        """Update current epoch for scheduling purposes"""
        self.current_epoch = epoch
    
    def get_warmup_factor(self) -> float:
        """Calculate warmup factor for adversarial losses"""
        if self.current_epoch < self.warmup_epochs:
            return self.current_epoch / self.warmup_epochs
        return 1.0

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Generate samples from the model

        """
        self.ema_encoder.apply_shadow()
        self.ema_decoder.apply_shadow()
        # Sample hidden representation
        h = torch.randn(num_samples, self.hidden_dim + self.context_dim, device=self.device)
        # Get prior distribution
        prior_dist, _ = self.prior(h)

        # Sample latents
        z = prior_dist.sample()

        # Decode
        samples = self.decoder.decode(z, deterministic=False)
        self.ema_encoder.restore()
        self.ema_decoder.restore()
        return samples


    def sample_next_latent(self, h_t, c_t, temperature=1.0):
        """
        Sample z_{t+1} from the DPGMM prior given current hidden state and context
        
        """
        batch_size = h_t.shape[0]
        
        # Combine hidden state and context for prior conditioning
        h_context = torch.cat([h_t, c_t], dim=-1)
        
        # Get DPGMM prior distribution
        prior_dist, prior_params = self.prior(h_context)
        
        # Sample from the mixture with temperature scaling
        if temperature > 0:
            # Sample component assignments from categorical distribution
            pi = prior_params['pi']  # [batch_size, K]
            
            # Apply temperature to mixing proportions
            if temperature != 1.0:
                logits = torch.log(pi + torch.finfo(torch.float32).eps) / temperature
                pi = F.softmax(logits, dim=-1)
            
            # Sample component indices
            component_dist = Categorical(probs=pi)
            selected_components = component_dist.sample()  # [batch_size]
            
            # Get parameters for selected components
            means = prior_params['means']  # [batch_size, K, latent_dim]
            log_vars = prior_params['log_vars']  # [batch_size, K, latent_dim]
            
            # Extract parameters for selected components
            batch_indices = torch.arange(batch_size).to(selected_components.device)
            selected_means = means[batch_indices, selected_components]  # [batch_size, latent_dim]
            selected_log_vars = log_vars[batch_indices, selected_components]  # [batch_size, latent_dim]
            
            # Sample from selected Gaussian components
            std = torch.exp(0.5 * selected_log_vars) * temperature
            z_next = selected_means + std * torch.randn_like(std)
            
        else:
            # Deterministic: use mode of dominant component
            dominant_components = prior_params['pi'].argmax(dim=-1)  # [batch_size]
            batch_indices = torch.arange(batch_size).to(dominant_components.device)
            z_next = prior_params['means'][batch_indices, dominant_components]
            selected_components = dominant_components
        
        prior_info = {
            'pi': prior_params['pi'],
            'selected_components': selected_components,
            'means': prior_params['means'],
            'log_vars': prior_params['log_vars'],
            'prior_dist': prior_dist
        }
        
        return z_next, prior_info


    @torch.no_grad()
    def generate_future_sequence(
        self,
        initial_obs: torch.Tensor,   # [B,T_ctx,C,H,W]
        actions: Optional[torch.Tensor],  # [B, T_total, A] or None
        horizon: int,
        decode_pixels: bool = True    # True: also produce frames for viz
    ) -> Dict[str, torch.Tensor]:
        """
        imagination:
        - consume T_ctx observed frames,
        - roll future for `horizon` steps using PRIOR and ContextPredictor,
        - use offline actions if provided, else zeros (later: policy(h,z,c))
        - optionally decode pixels for visualization
        Returns dict with z, c, a, h (and images if decode_pixels).
        """
        B, T_ctx = initial_obs.shape[:2]
        device = self.device

        # 1) Contexts for observed frames (no future AR here)
        ctx_obs, ctx_info = self.compute_global_context(
            initial_obs, 
            num_context_frames=T_ctx, 
            generate_future=False
        )  # ctx_obs: [B,T_ctx,C_ctx]

        # 2) Build initial RNN state by scanning the observed segment (teacher-forced)
        #    (matches forward_sequence but only up to T_ctx)
        outputs = { 'z': [], 'h': [], 'c': [ctx_obs[:, 0]], 'a': [] }
        h = self.h0.expand(self.number_lstm_layer, B, -1).contiguous()
        c_rnn = self.c0.expand(self.number_lstm_layer, B, -1).contiguous()
        #at t=0, c_prev = ctx_obs[:,t-1] is just ctx_obs[:,0] (initialized above)
        c_prev = torch.zeros(B, self.context_dim, device=device)
        for t in range(T_ctx):
            x_t = initial_obs[:, t]         # [B,C,H,W]
            c_t = ctx_obs[:, t]             # [B,C_ctx]
            a_t = actions[:, t] if actions is not None else torch.zeros(B, self.action_dim, device=device)

            z_t, *_ = self.encoder(x_t)  # z from encoder 

            # step RNN
            rnn_in = torch.cat([z_t, c_t, a_t], dim=-1)
            _, (h, c_rnn) = self._rnn(rnn_in, h, c_rnn, torch.ones(B, device=device))
            outputs['z'].append(z_t)
            outputs['h'].append(h[-1])
            outputs['a'].append(a_t)

        # 3) Future imagination with PRIOR + ContextPredictor
        c_prev = ctx_obs[:, -1]     # last observed context

        z_futures, c_futures, a_futures, h_futures = [], [], [], []

        for t in range(horizon):
            # (a) pick action
            a_t = actions[:, T_ctx + t] if actions is not None and actions.shape[1] > (T_ctx + t) \
                else torch.zeros(B, self.action_dim, device=device)

            # (b) prior over attention/z
            
            pz_dist, pz_params = self.prior(torch.cat([h[-1], c_prev], dim=-1))
            z_t = pz_dist.sample()     # stochastic roll

            # (c) step RNN with imagined inputs
            rnn_in = torch.cat([z_t, c_prev, a_t], dim=-1)
            #
            _, (h, c_rnn) = self._rnn(rnn_in, h, c_rnn, torch.ones(B, device=device))

            # (d) predict next context from (c_prev, a_t)
            c_t = self.context_predictor(context_seq=c_prev, action_seq=a_t, latent_seq=z_t, hidden_seq=h[-1])
            # collect + advance
            z_futures.append(z_t); c_futures.append(c_t); a_futures.append(a_t); h_futures.append(h[-1])
            c_prev = c_t


        # 4) Pack latent imagination outputs
        z_futures = torch.stack(z_futures, dim=1) if len(z_futures) else torch.empty(B,0, self.latent_dim, device=device)
        c_futures = torch.stack(c_futures, dim=1) if len(c_futures) else torch.empty(B,0, self.context_dim, device=device)
        a_futures = torch.stack(a_futures, dim=1) if len(a_futures) else torch.empty(B,0, self.action_dim, device=device)
        h_futures = torch.stack(h_futures, dim=1) if len(h_futures) else torch.empty(B,0, h[-1].shape[-1], device=device)

        result = {
            "z_obs": torch.stack(outputs['z'], dim=1),     # [B,T_ctx, D_z]
            "h_obs": torch.stack(outputs['h'], dim=1),     # [B,T_ctx, D_h]
            "a_obs": torch.stack(outputs['a'], dim=1),     # [B,T_ctx, A]
            "c_obs":  ctx_obs,                             # [B,T_ctx, D_c]
            "z_future": z_futures,                         # [B,H,D_z]
            "c_future": c_futures,                         # [B,H,D_c]
            "a_future": a_futures,                         # [B,H,A]
            "h_future": h_futures,                         # [B,H,D_h]
        }

        # 5) pixel rollout via the Perceiver AR head (for visualization)
        if decode_pixels:
            # Use the Perceiver to generate frames from the observed context only.
            # This remains independent of the *latent* imagination above (standard in Dreamer).
            result["pixels_future"] = self.perceiver_model.generate_autoregressive(
                context_videos=initial_obs, num_frames_to_generate=horizon,
                temperature=1.0, top_k=None, top_p=None
            )  # [B,horizon,C,H,W]

        B, T_future, D = z_futures.shape
        z_flat = z_futures.reshape(B * T_future, D)

        # Use EMA decoder weights for nicer samples (optional but consistent with .sample())
        if hasattr(self, "ema_decoder"):
            self.ema_decoder.apply_shadow()

        x_flat = self.decoder.decode(z_flat, deterministic=False)  # [B*T, C, H, W]

        if hasattr(self, "ema_decoder"):
            self.ema_decoder.restore()

        result["vae_future"] = x_flat.view(B, T_future, *x_flat.shape[1:])  # [B, T_future, C, H, W]

        return result
    

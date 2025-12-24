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
from vis_networks import EMA, TemporalDiscriminator, AddEpsilon, check_tensor
from VRNN.RGB import DynamicWeightAverage
from VRNN.lstm import LSTMLayer
from vdvae.vae import VDVAE
from vdvae.hps import Hyperparams
from vdvae.vae_helpers import mean_from_discretized_mix_logistic

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
        ):
        super().__init__()
        #core dimensions
        self.input_channels = input_channels
        self.image_size = input_dim
        self.max_K = max_components
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
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
        self.use_dwa = use_dwa
        self.use_ctx_checkpoint = use_ctx_checkpoint

        #initialization different parts of the model
        self._init_encoder_decoder(max_components, prior_alpha, prior_beta)
        
        self._init_vrnn_dynamics(use_orthogonal=use_orthogonal,number_lstm_layer=number_lstm_layer)
        self._init_discriminators( img_disc_layers, patch_size, num_heads= disc_num_heads)
        if use_dwa:
            self._init_DynamicWeightAverage(dwa_temperature)

        # Initialize weights
        #self.apply(self.init_weights)
        self.to(device)
        # Setup optimizers
        self.has_optimizers = True
        self._setup_optimizers(learning_rate, weight_decay)


    def _init_DynamicWeightAverage(self, temperature: float = 2.0):

        self.total_weighter = DynamicWeightAverage(
            loss_keys_to_consider=["recon_loss",
                "kl_z",
                "hierarchical_kl",
                "img_adv_loss",
                "temporal_adv_loss",
                "feat_match_loss",
            ],
            temperature=temperature,
        )


    def _init_encoder_decoder(self, max_components: int, prior_alpha: float, prior_beta: float, prior_mc_samples: int = 100):
        """
        Initialize VDVAE + DPGMM prior.
        
        """
        # ---- 1) Build VDVAE hyperparams ----
        H = Hyperparams()
        H.use_checkpoint = self.use_ctx_checkpoint
        H.image_channels = self.input_channels   # usually 3
        H.zdim = self.latent_dim                 # or set explicitly (e.g., 16)
        H.bottleneck_multiple = 0.25 
        H.width = 384
        H.image_size = self.image_size          # e.g. 64
        H.dataset = 'imagenet64'
        H.num_mixtures = 10
        H.skip_threshold = 300.0
        #H.dec_blocks = "1x1,4m1,4x2,8m4,8x4,16m8,16x6,32m16,32x8,64m32,64x3"
        #H.enc_blocks = "64x3,64d2,32x8,32d2,16x6,16d2,8x4,8d2,4x2,4d4,1x3"
        H.dec_blocks = "8x4,16m8,16x4,32m16,32x4,64m32,64x2"
        H.enc_blocks = "64x2,64d2,32x4,32d2,16x4,16d2,8x4"  
        H.custom_width_str = "64:256,32:256,16:320,8:384"
         
        H.no_bias_above = 64
        H.custom_width_str = ""

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
        #
        self.gen_optimizer = torch.optim.Adamax(
            gen_param_groups,
            learning_rate*1.2,
            betas=(0.9, 0.999),
            eps=1e-4,
        )

        #  4) Discriminator optimizer (unchanged)
        if hasattr(self, "image_discriminator"):
            disc_params = [
                p for p in self.image_discriminator.parameters()
                if p.requires_grad
            ]
            if disc_params:
                self.img_disc_optimizer = torch.optim.Adamax(
                    disc_params,
                    lr=learning_rate * 0.05,
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


    def forward_sequence(self, observations, actions=None, dones=None):

        batch_size, seq_len = observations.shape[:2]
        
        # Default actions if not provided
        if actions is None:
            actions = torch.zeros(batch_size, seq_len, self.action_dim).to(self.device)
        
        # Initialize LSTM hidden states
        h = self.h0.expand(self.number_lstm_layer, batch_size, -1).contiguous()
        c = self.c0.expand(self.number_lstm_layer, batch_size, -1).contiguous()
        # Storage for outputs
        outputs = {
            "reconstructions": [],         # [B, C, H, W] per t -> stack to [B, T, C, H, W]
            "reconstruction_samples": [],  # 
            "latents": [],                 # [B, zdim]
            "hidden_states": [],           # [B, hidden_dim]
            "prior_params": [],
            "kumaraswamy_kl_losses": [],
            "kl_latents": [],              # per-step total KL (gauss + DP), scalar
            "reconstruction_losses": [],   # per-step distortion, scalar
            "vae_elbos": [],
            "gauss_rates": [],
            "gm_rates": [],
            "K_eff": [],
            "component_margin": [],
        }

        # Process sequence step by step
        for t in range(seq_len):
            # Get current inputs
            o_t = observations[:, t]
            x_t = o_t.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
            a_t = actions[:, t]
            
            # === Prior Network p(z_t|o_<t, z_<t) ===
            h_context = h[-1]

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
                h_context,
            )
            pi_tok = prior_params["pi"]  # expected [B, K] (or [K]; handle both)
            eps_ent = torch.finfo(torch.float32).eps
            B, C, Ht, Wt = vdvae_out["top_q_mean_map"].shape
            K = pi_tok.shape[1]
            tokens_per_img = Ht * Wt

            # build z_tokens from posterior q(z|x) at top level (reparameterized sample)
            top_q_mean_tokens = vdvae_out["top_q_mean_map"].permute(0, 2, 3, 1).reshape(B * tokens_per_img, C)
            top_q_logvar_tokens = vdvae_out["top_q_logvar_map"].permute(0, 2, 3, 1).reshape(B * tokens_per_img, C)

            noise = torch.randn_like(top_q_mean_tokens)
            z_tokens = top_q_mean_tokens + noise * torch.exp(0.5 * top_q_logvar_tokens)  # [N,C]
            eps_ = torch.finfo(z_tokens.dtype).eps
            # responsibilities under the "prior mixture" for these z_tokens
            resp = self.prior.compute_responsibilities(
                z_tokens=z_tokens,
                prior_params=prior_params,
                eps=eps_,
                temperature=1.0,
            ) # [N,K]

            # Marginal entropy: entropy of average usage across batch
            # token-level MI
            u = resp.mean(dim=0).clamp_min(eps_)             # [K]
            H_marg = -(u * u.log()).sum()
            r = resp.clamp_min(eps_)
            H_cond = - (r * r.log()).sum(dim=-1).mean()
            I = H_marg - H_cond

            outputs["component_margin"].append(I)
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

            rnn_input = torch.cat([z_t, a_t], dim=-1)
            rnn_output, (h, c) = self._rnn(
                rnn_input, h, c, 
                mask_t
            )
            rnn_output = self.rnn_layer_norm(rnn_output)
            outputs['hidden_states'].append(h[-1])
            #print(f"latent z_t range: min {z_t.min().item()}, max {z_t.max().item()}, px_z range: min {vdvae_out['px_z'].min().item()}, max {vdvae_out['px_z'].max().item()}")
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

        # ---- 6) Aggregate across time ----
        # Stack things that are truly sequences
        for key in ["reconstructions", "latents", "hidden_states", "reconstruction_samples"]:
            if len(outputs[key]) > 0 and isinstance(outputs[key][0], torch.Tensor):
                outputs[key] = torch.stack(outputs[key], dim=1)  # [B, T, ...]

        # Scalars that are accumulated per timestep 
        if outputs["K_eff"]:
            outputs["K_eff"] = torch.stack(outputs["K_eff"]).mean()
        return outputs

    def compute_total_loss(
        self,
        observations,
        actions=None,
        beta: float = 1.0,
        lambda_recon: float = 1.0,
    ):
        outputs = self.forward_sequence(observations, actions)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute adversarial losses for both image and latent space
        """
        D = self.image_discriminator
        flags = [p.requires_grad for p in D.parameters()] #freeze Discriminator parameters
        for p in D.parameters():
            p.requires_grad_(False)

        fake_img_outputs = D(reconstruction, z=z_seq.detach(), return_features=True)
        real_img_outputs = D(x,             z=z_seq.detach(), return_features=True)
        img_adv_loss = -torch.mean(fake_img_outputs['final_score'])
        #reward for generating temporally consistent images
    
        real_frame_features = real_img_outputs['hidden_3d'].mean(dim=2)  # Flatten spatial dimensions
        
        fake_frame_features = fake_img_outputs['hidden_3d'].mean(dim=2)

        temporal_loss_frames =self.contrastive_loss(
        real_features=real_frame_features,
        fake_features=fake_frame_features
        )
            
        # === Feature Matching Loss ===
        # This helps stabilize training
        # L1 loss between feature statistics
        feature_match_loss = self.compute_feature_matching_loss(real_img_outputs['hidden_3d'], fake_img_outputs['hidden_3d'])
        # Restore Discriminator parameter gradients
        for p, f in zip(D.parameters(), flags):
            p.requires_grad_(f)

        return img_adv_loss , temporal_loss_frames, feature_match_loss


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
                return images.clamp(-1.0, 1.0)
            
            # If uint8, convert to float [0, 1] first
            if images.dtype == torch.uint8:
                images = images.float() / 255.0
            # If in [0, 255] range
            elif img_max > 1.5:
                images = images / 255.0
            
            # Now images should be in [0, 1], convert to [-1, 1]
            # Check again to be sure
            images = images * 2.0 - 1.0

            return images.clamp(min=-1.0, max=1.0)

    def denormalize_generated_images(self, images):
        """
        Convert generated images from [-1, 1] back to [0, 1] for visualization
        """
        return (images + 1) / 2


    def training_step_sequence(self, 
                            observations: torch.Tensor,
                            actions: torch.Tensor = None,
                            beta: float = 1.0,
                            n_critic: int = 3,
                            lambda_img: float = 0.2,
                            lambda_recon: float = 1.0,
                            batch_idx: Optional[int] = None,
                            ) -> Dict[str, torch.Tensor]:
        """
        Supports:
        - Dynamic Weight Averaging (DWA) over either:
            * ELBO / adv  streams (when grad_balance_method != "none"), or
            * all individual loss terms together (when grad_balance_method == "none").
        - RGB / GradNorm / PCGrad /MGDAmulti-task gradient balancing when requested.
        """
        self.train()
        if getattr(self, "image_discriminator", None) is not None:
            self.image_discriminator.train()

        # 1) Prepare data and compute VAE loss
        warmup_factor = self.get_warmup_factor()

        vae_losses, outputs = self.compute_total_loss(
            observations, actions, beta,
            lambda_recon
        )
        lambda_img_eff = (lambda_img * warmup_factor) if warmup_factor > 0.0 else 0.0

        # 2) Discriminator updates (n_critic)
        disc_losses_list: List[Dict[str, torch.Tensor]] = []
        for _ in range(n_critic):
            disc_loss = self.discriminator_step(
                real_images=observations,
                fake_images=outputs['reconstructions'],  # discriminator_step detaches internally
                latents=torch.cat([outputs['latents'], outputs['hidden_states']], dim=-1)
            )
            disc_losses_list.append(disc_loss)

        avg_disc_losses: Dict[str, torch.Tensor] = {}
        if disc_losses_list:
            avg_disc_losses = {
                k: sum(d[k] for d in disc_losses_list) / len(disc_losses_list)
                for k in disc_losses_list[0].keys()
            }

        # 3) Adversarial generator losses
        img_adv_loss, temporal_adv_loss, feat_match_loss = self.compute_adversarial_losses(
            x=observations,
            reconstruction=outputs['reconstructions'],
            z_seq=torch.cat([outputs['latents'], outputs['hidden_states']], dim=-1),
        )

        # 4) Combine losses with DWA
        if self.use_dwa :
            total_components = {
                "recon_loss":      vae_losses["recon_loss"].reshape([]),
                "kl_z":            vae_losses["kl_z"].reshape([]),
                "hierarchical_kl": vae_losses["hierarchical_kl"].reshape([]),
                "img_adv_loss":      (warmup_factor * img_adv_loss).reshape([]),
                "temporal_adv_loss": (warmup_factor * temporal_adv_loss).reshape([]),
                "feat_match_loss":   feat_match_loss.reshape([]),
                "component_margin": vae_losses["component_margin"].reshape([]),
            }
            total_gen_loss = self.total_weighter.reduce_losses(total_components, batch_idx)

        else:
            # No DWA: fixed scalar weights
            adv_base = (
                lambda_img_eff * img_adv_loss
                + warmup_factor * temporal_adv_loss
                + lambda_img * feat_match_loss
            )
            elbo_loss = (
                lambda_recon * vae_losses['recon_loss']
                + beta * vae_losses['kl_z']
                + beta * vae_losses['hierarchical_kl']
                - vae_losses['component_margin']  # encourage diverse component usage
            )

            total_gen_loss = elbo_loss + adv_base 

        # No multi-task balancer: just backprop total_gen_loss
        total_gen_loss.backward()

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
            self.ema_vdvae.update()

        eff_comp = outputs['prior_params'][0]['pi'].max(1)[0].mean().item()
        top6_cov = outputs['prior_params'][0]['pi'].topk(6, dim=-1)[0].sum(dim=-1).mean().item()

        return {
            # VAE & auxiliaries
            **vae_losses,
            # Discriminators
            **avg_disc_losses,
            # Adversarial gen (cast to float for logging)
            'img_adv_loss': float(img_adv_loss.item()),
            'temporal_adv_loss': float(temporal_adv_loss.item()),
            'feat_match_loss': float(feat_match_loss.item()),
            # Training stats
            'total_gen_loss': float(total_gen_loss.item()),
            'grad_norm': float(grad_norm),
            'effective_components': eff_comp,
            'Top 6 coverage': top6_cov,
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


    @torch.no_grad()
    def generate_future_sequence(
        self,
        initial_obs: torch.Tensor,          # [B, T_ctx, C, H, W] in [-1, 1]
        actions: torch.Tensor | None,       # [B, T_total, action_dim] (needs >= T_ctx + horizon)
        horizon: int,
        dones: torch.Tensor | None = None,  # [B, T_ctx] or [B, T_total] bool/int (optional)
        decoder_temperature: float = 0.85,           # <1 reduces noise
        decode_mode: str = "mean",        # "sample" | "mean"
    ):
        """
        Returns:
        pred_imgs: [B, horizon, C, H, W] in [-1, 1]
        debug: dict with sampled z, h, a
        """
        B, T_ctx = initial_obs.shape[:2]
        device = self.device
        # --- init RNN state ---
        h = self.h0.expand(self.number_lstm_layer, B, -1).contiguous()
        c_rnn = self.c0.expand(self.number_lstm_layer, B, -1).contiguous()

        # --- VDVAE top block info ---
        top_block = self.vdvae.decoder.dec_blocks[0]
        C_top = top_block.zdim
        res = top_block.base

        def _mask_at(t: int):
            # matches training logic: mask_t = 1 - first_t, first_t=1 at t=0 :contentReference[oaicite:5]{index=5}
            if dones is None:
                return torch.ones(B, device=device)
            if t == 0:
                first_t = torch.ones(B, device=device)
            else:
                first_t = dones[:, t - 1].to(torch.float32).to(device)
            return 1.0 - first_t

        def sample_top_z_map_and_flat(h_context: torch.Tensor):
            B_, Hc = h_context.shape
            N = B_ * res * res

            h_map = h_context.view(B_, Hc, 1, 1).expand(B_, Hc, res, res).contiguous()
            h_map = self.vdvae.add_coord_no_proj(h_map, scale=0.05)
            h_tokens = h_map.permute(0, 2, 3, 1).reshape(N, Hc)  # [N, Hc]

            _, prior_params = self.prior(h_tokens)

            pi = prior_params["pi"]                      # [N, K]  (already normalized)
            means = prior_params["means"]                # [N, K, C_top]
            log_vars = prior_params["log_vars"].clamp(-10, 10)  # [N, K, C_top]
            K = pi.shape[1]

            pi_map = pi.view(B_, res, res, K).permute(0, 3, 1, 2).contiguous()  # [B, K, res, res]

            k_tokens = Categorical(probs=pi).sample()  # [N]
            idx = torch.arange(N, device=device)
            mu = means[idx, k_tokens]                                 # [N, C_top]
            std = torch.exp(0.5 * log_vars[idx, k_tokens])            # [N, C_top]
            eps = torch.randn_like(mu)
            z_tokens = mu + (decoder_temperature * std) * eps

            k_map = k_tokens.view(B_, res, res).contiguous()  

            z_map = z_tokens.view(B_, res, res, C_top).permute(0, 3, 1, 2).contiguous()
            z_flat = z_map.view(B_, -1)
            return z_map, z_flat, pi_map, k_map

        # -------------------------
        # 1) Teacher-forced context scan
        # -------------------------
        debug = {"z_obs": [], "h_obs": [], "a_obs": []}

        for t in range(T_ctx):
            x_t = initial_obs[:, t]  # [B,C,H,W]
            a_t = actions[:, t] if actions is not None else torch.zeros(B, self.action_dim, device=device)

            x_t_nhwc = x_t.permute(0, 2, 3, 1).contiguous()
            h_context = h[-1]

            # VDVAE forward: top_q_mean_map/logvar_map are [B,C_top,res,res] 
            vdvae_out = self.vdvae(x_t_nhwc, x_t_nhwc, h_context)

            # sample z_t from posterior q(z|x,h)
            z_t = vdvae_out["top_q_mean_map"].contiguous().view(B, -1)  # [B, top_zdim]

            # step RNN with SAME input structure as training: [z_t, a_t] 
            rnn_in = torch.cat([z_t, a_t], dim=-1)
            mask_t = _mask_at(t)
            _, (h, c_rnn) = self._rnn(rnn_in, h, c_rnn, mask_t)

            debug["z_obs"].append(z_t)
            debug["h_obs"].append(h[-1])
            debug["a_obs"].append(a_t)

        # -------------------------
        # 2) Imagination rollout (prior sampling)
        # -------------------------
        z_top_maps = []
        z_fut, h_fut, a_fut, pi_fut, k_fut = [], [], [], [], []

        for k in range(horizon):
            t_abs = T_ctx + k
            a_k = (
                actions[:, t_abs]
                if actions is not None and actions.shape[1] > t_abs
                else torch.zeros(B, self.action_dim, device=device)
            )

            h_context = h[-1]
            z_map_k, z_k, pi_map_k, k_map_k = sample_top_z_map_and_flat(h_context)  
            z_top_maps.append(z_map_k)
            z_fut.append(z_k)
            a_fut.append(a_k)
            pi_fut.append(pi_map_k)
            k_fut.append(k_map_k)
            
            rnn_in = torch.cat([z_k, a_k], dim=-1)
            if dones is None or dones.shape[1] <= (t_abs - 1):
                mask_k = torch.ones(B, device=device)
            else:
                mask_k = 1.0 - dones[:, t_abs - 1].float().to(device)

            _, (h, c_rnn) = self._rnn(rnn_in, h, c_rnn, mask_k)
            h_fut.append(h[-1])
        debug["h_obs"] = torch.stack(debug["h_obs"], dim=1)
        debug["z_obs"] = torch.stack(debug["z_obs"], dim=1)
        debug["a_obs"] = torch.stack(debug["a_obs"], dim=1)
        debug["pi_future"] = torch.stack(pi_fut, dim=1) # [B, horizon, K, res, res]
        debug["k_future"] = torch.stack(k_fut, dim=1) # [B, horizon, res, res]
        # -------------------------
        # 3) Decode predicted frames from sampled top-level latents
        # -------------------------
        z_top = torch.stack(z_top_maps, dim=1).contiguous()              # [B, horizon, C_top, res, res]

        z_top_flat = z_top.view(B * horizon, C_top, res, res)

        n_blocks = len(self.vdvae.decoder.dec_blocks)
        latents = [z_top_flat] + [None] * (n_blocks - 1)    # <-- NOTHING frozen

        px_z = self.vdvae.decoder.forward_manual_latents(B * horizon, latents, t=decoder_temperature)  

        if decode_mode == "mean":
            # deterministic, less speckle than sampling
            dmol = self.vdvae.decoder.out_net.forward(px_z)
            recon = mean_from_discretized_mix_logistic(dmol, self.vdvae.H.num_mixtures)  # [B*horizon,H,W,C]
            pred = recon.permute(0, 3, 1, 2).contiguous()
        elif decode_mode == "sample":
            # uint8 NHWC -> float NCHW in [-1,1] 
            out_uint8 = self.vdvae.decoder.out_net.sample(px_z)  # uint8 NHWC
            pred = torch.from_numpy(out_uint8).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).contiguous() / 127.5 - 1.0
        else:
            raise ValueError(f"Unknown decode_mode {decode_mode}")

        pred_imgs = pred.view(B, horizon, *pred.shape[1:])       # [B, horizon, C, H, W]

        debug["z_future"] = torch.stack(z_fut, dim=1)
        debug["h_future"] = torch.stack(h_fut, dim=1)
        debug["a_future"] = torch.stack(a_fut, dim=1)
        debug["vae_future"] = pred_imgs
        debug["z_top_future_maps"] = z_top
        return debug

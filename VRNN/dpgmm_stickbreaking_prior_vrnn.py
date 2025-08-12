from logging import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Gamma, Categorical, Independent, MixtureSameFamily
from typing import Dict, Tuple, Union, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import (
    EMA, VAEEncoder, VAEDecoder, 
    TemporalDiscriminator,
    LinearResidual, AttentionPosterior, AttentionPrior,
    AddEpsilon, check_tensor, SelfModelBlock
)
import math
from collections import OrderedDict
from VRNN.lstm import LSTMLayer
import numpy as np
from VRNN.perceiver.Utils import generate_model
from VRNN.perceiver import perceiver_helpers
import VRNN.perceiver.perceiver as perceiver
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
            ('gamma_fc', nn.Linear(hidden_dim, hidden_dim)),
            ('gamma_ln', nn.LayerNorm(hidden_dim)),
            ('gamma_relu', nn.GELU()),
            ('gamma_out', nn.Linear(hidden_dim, 2)),
            ('gamma_ln_out', nn.LayerNorm(2)),
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
        term3 = (a - alpha) * torch.digamma(a)
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

        # Networks for Kumaraswamy parameters a and b
        self.net_a = nn.Sequential(OrderedDict([
            ('kumar_a_fc', nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim))),
            ('kumar_a_ln',nn.LayerNorm(hidden_dim)),
            ('kumar_a_relu', nn.SiLU()),
            ('kumar_a_out', nn.utils.spectral_norm(nn.Linear(hidden_dim , self.K - 1))),
            ('kumar_a_ln_out', nn.LayerNorm(self.K - 1)),
        ]))

        self.net_b = nn.Sequential(OrderedDict([
            ('kumar_b_fc', nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim))),
            ('kumar_b_ln',nn.LayerNorm(hidden_dim)),
            ('kumar_b_relu', nn.SiLU()),
            ('kumar_b_out', nn.utils.spectral_norm(nn.Linear(hidden_dim , self.K - 1))),
            ('kumar_b_ln_out', nn.LayerNorm(self.K - 1)),

        ]))
        self.to(device)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate Kumaraswamy parameters from hidden representation

        Args:
            h: Hidden representation [batch_size, hidden_dim]

        Returns:
            a, b: Kumaraswamy parameters [batch_size, K-1]
        """
        # Add small epsilon to ensure strictly positive parameters
        min_val = -5.0
        max_val = 5.0
        log_a = torch.clamp(self.net_a(h) , min=min_val, max=max_val)
        log_b = torch.clamp(self.net_b(h) , min=min_val, max=max_val)
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
    def compute_stick_breaking_proportions( v: torch.Tensor, max_k:int, perm: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert samples to stick-breaking proportions with optional permutation inversion
        # Compute stick-breaking proportions using vectorized operations
        cumprod_one_minus_v = torch.cumprod(1 - v, dim=-1)  # [batch_size, K-1]
        one_v = F.pad(v, (0, 1), value=1.0)  # [batch_size, K]
        c_one = F.pad(cumprod_one_minus_v, (1, 0), value=1.0)  # [batch_size, K]
        pi = one_v * c_one  # [batch_size, K]
        """
        
        batch_size= v.shape[0]
        check_tensor(v, "stick_breaking_v")
        # Compute cumprod in log space for numerical stability
        log_1minus_v = torch.log1p(-v)  # More stable than log(1-v)
        cumsum_log_1minus = torch.cumsum(log_1minus_v, dim=-1)
            
        # Compute pi components in a numerically stable way
        pi_components = []
        # First component
        pi_components.append(v[:, 0])
    
        # Middle components
        for k in range(1, v.shape[1]):
            pi_k = v[:, k] * torch.exp(cumsum_log_1minus[:, k-1])
            pi_components.append(pi_k)
    
        # Last component
        pi_components.append(torch.exp(cumsum_log_1minus[:, -1]))
    
        # Stack components
        pi = torch.stack(pi_components, dim=1)
        

        # Apply inverse permutation if needed
        if perm is not None:
            # Create full permutation including last component
            full_perm = torch.cat([perm, torch.full((batch_size, 1), max_k-1, device=perm.device)], dim=1)
            inv_perm = torch.argsort(full_perm, dim=1)  # [batch_size, K]
            pi = torch.gather(pi, 1, inv_perm)

        pi = pi / (pi.sum(dim=-1, keepdim=True) + torch.finfo(torch.float32).eps)  # Normalize to sum to 1
        
        return pi

    def forward(self, h: torch.Tensor, n_samples: int = 10, use_rand_perm: bool = True, truncation_threshold: float = 0.995) -> Tuple[torch.Tensor, Dict]:
        """
        Generate mixing proportions using stick-breaking process

        Args:
            h: Hidden representation from encoder [batch_size, hidden_dim]

        Returns:
            pi: Mixing proportions [batch_size, max_components]
            alpha: Concentration parameter
        """
        
        
        # Generate stick-breaking proportions
        # Get Kumaraswamy parameters
        log_kumar_a, log_kumar_b = self.kumar_net(h)
        
        # Sample v from Kumaraswamy for each alpha sample
        v, perm = self.sample_kumaraswamy(log_kumar_a, log_kumar_b, self.max_K, use_rand_perm)  # [n_samples, batch, K-1]


        # Initialize mixing proportions
        pi = self.compute_stick_breaking_proportions(v, self.max_K, perm)
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
        KL(K(a,b) || B(α,β)) = 
            (a-α)/a * (-γ - ψ(b) - 1/b) +     # First part
            ln(a*b) + ln(B(α,β)) -      # Log terms
            (b-1)/b +                         # Additional term
            (β-1)b * E[ln(1-v)]               # Taylor expansion term
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
        term2 += torch.div(-(b - 1), b + eps)
        kl += term1 + term2
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
            nn.LeakyReLU(),
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
        Computes KL[ q(z|x) || p(z) ] where:
            p(z) = GMM(prior_weights, prior_means, prior_log_vars)
            q(z|x) = Gaussian(posterior_means, posterior_log_vars)
    
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
        log_prior_components = log_component_densities + torch.log(prior_weights.unsqueeze(0).clamp(min=eps))
        log_p = torch.logsumexp(log_prior_components, dim=2)  # (n_samples, B)

        # KL divergence: average over samples and batch
        kl_samples = log_q - log_p  # (n_samples, B)
        return kl_samples.mean()  # Scalar
    
    
    def compute_kl_loss(self, params: Dict, alpha: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between Kumaraswamy and Beta distributions and prior gamma distributions
        """
        total_kl = 0.0
        
        # For each stick-breaking weight
        for k in range(self.max_K - 1):

            kumar_a_k = params['kumar_a'][:, k:k+1]  # [batch_size, 1]
            kumar_b_k = params['kumar_b'][:, k:k+1]  # [batch_size, 1]
            
            alpha_k = alpha[:, k:k+1] if k < alpha.shape[1] else alpha[:, -1:] # Get alpha for current stick

            beta_alpha = torch.ones_like(alpha_k).squeeze(-1)  # Beta(1,α)
            beta_beta = alpha_k.squeeze(-1)  # Current stick's α
                        
                        
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
            total_kl += kl

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


class AttentionSchema(nn.Module):
    """
    Complete attention schema implementation with proper spatial modeling
    """
    
    def __init__(
        self,
        image_size: int = 84,
        attention_resolution: int = 21,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        context_dim: int = 128,
        attention_dim: int = 64,
        input_channels: int = 3,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super().__init__()
        
        # Posterior (bottom-up, stimulus-driven attention)
        self.posterior_net = AttentionPosterior(
            image_size, 
            attention_resolution, 
            hidden_dim, 
            context_dim,
            input_channels,
            device=device
        )
        
        # Prior (top-down, predictive attention schema)
        self.prior_net = AttentionPrior(
            attention_resolution, hidden_dim, 
            latent_dim, motion_kernels=8
        )
        
        # Attention state encoder (for RNN integration)
        self.attention_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # Downsample
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, attention_dim)
        )
        self.to(device)
    
    def compute_attention_dynamics_loss(
        self,
        attention_sequence: list,
        predicted_movements: list
    ) -> torch.Tensor:
        """
        Additional loss for smooth attention dynamics
        """
        if len(attention_sequence) < 2:
            return 0.0
        
        # Stack all attention maps into a tensor [T, B, H, W]
        attention_tensor = torch.stack(attention_sequence, dim=0)
        T, B, H, W = attention_tensor.shape
        
        # Compute center of mass for all timesteps at once
        y_coords = torch.arange(H, device=attention_tensor.device).float()
        x_coords = torch.arange(W, device=attention_tensor.device).float()
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Expand grids to match tensor dimensions [1, 1, H, W]
        y_grid = y_grid.unsqueeze(0).unsqueeze(0)
        x_grid = x_grid.unsqueeze(0).unsqueeze(0)
        
        # Compute weighted centers [T, B]
        total_mass = attention_tensor.sum(dim=[2, 3], keepdim=True) + 1e-8
        y_com = (attention_tensor * y_grid).sum(dim=[2, 3]) / total_mass.squeeze(-1).squeeze(-1)
        x_com = (attention_tensor * x_grid).sum(dim=[2, 3]) / total_mass.squeeze(-1).squeeze(-1)
        
        # Stack coordinates [T, B, 2]
        centers = torch.stack([x_com, y_com], dim=-1)
        
        # Compute actual movements between consecutive timesteps [T-1, B, 2]
        actual_movements = centers[1:] - centers[:-1]
        actual_dx = actual_movements[..., 0]  # [T-1, B]
        actual_dy = actual_movements[..., 1]  # [T-1, B]
        
        # Stack predicted movements [T-1, ...]
        pred_movements = torch.stack(predicted_movements, dim=0)  # [T-1, B, H, W] or similar
        
        # Compute predicted movement averages
        pred_dx_mean = pred_movements[..., 0].mean(dim=[2, 3]) if pred_movements.dim() > 3 else pred_movements[..., 0]
        pred_dy_mean = pred_movements[..., 1].mean(dim=[2, 3]) if pred_movements.dim() > 3 else pred_movements[..., 1]
        pred_stack = torch.cat([pred_dx_mean, pred_dy_mean], dim=0)
        actual_stack = torch.cat([actual_dx, actual_dy], dim=0)
        # Compute squared error [T-1, B]
        movement_error = F.huber_loss(
            pred_stack, actual_stack, delta=1.0
            )
        
        # Average over time and batch
        total_loss = movement_error.mean()

        return total_loss

    def _center_of_mass(self, attention_map):
        """Compute attention center of mass for movement tracking"""
        batch_size, H, W = attention_map.shape
        
        # Create coordinate grids
        y_coords = torch.arange(H, device=attention_map.device).float()
        x_coords = torch.arange(W, device=attention_map.device).float()
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Compute weighted average
        total_mass = attention_map.sum(dim=[1, 2], keepdim=True) + torch.finfo(torch.float32).eps  # Avoid division by zero
        y_com = (attention_map * y_grid).sum(dim=[1, 2]) / total_mass.squeeze()
        x_com = (attention_map * x_grid).sum(dim=[1, 2]) / total_mass.squeeze()
        
        return torch.stack([x_com, y_com], dim=-1)
    
class PerceiverContextEmbedding(nn.Module):
    """
    Maintains a learnable context memory that gets updated by attending to Perceiver outputs.
    
    Think of this as having a persistent "understanding" of context that gets
    refined based on global information from the Perceiver, rather than 
    regenerating context from scratch each time.
    """
    def __init__(self, perceiver_latent_dim, context_dim, num_heads=4, dropout=0.1, device='cuda'):
        super().__init__()
        self.perceiver_latent_dim = perceiver_latent_dim
        self.context_dim = context_dim
        self.device = device
        
        # This is our "context memory" - what we're trying to understand/extract
        # Initialize with small random values to break symmetry
        self.context_memory = nn.Parameter(
            torch.randn(context_dim, perceiver_latent_dim) * 0.02
        )
        
        # Layer norm for the context memory to keep it well-behaved
        self.context_norm = nn.LayerNorm(perceiver_latent_dim)
        
        # Cross-attention where context memory queries the Perceiver output
        # This is where the "updating" happens
        self.update_attention = nn.MultiheadAttention(
            embed_dim=perceiver_latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # A gating mechanism to control how much the context changes
        # This helps stability - we don't want context to change too drastically
        self.update_gate = nn.Sequential(
            nn.Linear(perceiver_latent_dim * 2, perceiver_latent_dim),
            nn.Sigmoid()
        )
        
        # Final projection to ensure we output exactly context_dim
        # This also allows the model to learn a different "output representation"
        # of the context if needed
        self.output_projection = nn.Linear(perceiver_latent_dim, 1)
        
        self.to(device)
    
    def forward(self, perceiver_output):
        """
        Updates context memory based on Perceiver output.
        
        Args:
            perceiver_output: [batch*seq_len, num_latents, perceiver_latent_dim]
            
        Returns:
            context: [batch*seq_len, context_dim]
        """
        batch_size = perceiver_output.shape[0]
        
        # Normalize and expand context memory for each batch element
        # Think of this as "preparing our questions"
        context_queries = self.context_norm(self.context_memory)
        context_queries = context_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Query the Perceiver output with our context memory
        # This asks: "Given what the Perceiver sees globally, how should we update our context understanding?"
        updated_context, attention_weights = self.update_attention(
            query=context_queries,     # What we want to know: [batch, context_dim, latent_dim]
            key=perceiver_output,      # What's available: [batch, num_latents, latent_dim]
            value=perceiver_output     # The information to extract
        )
        
        # Optional: Apply gating to blend old and new context
        # This prevents the context from changing too dramatically
        gate_input = torch.cat([context_queries, updated_context], dim=-1)
        gate = self.update_gate(gate_input)
        gated_context = gate * updated_context + (1 - gate) * context_queries
        
        # Project to get final context values
        # [batch, context_dim, latent_dim] -> [batch, context_dim, 1] -> [batch, context_dim]
        context = self.output_projection(gated_context).squeeze(-1)
        
        return context, attention_weights 
    

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
        context_dim: int, # Context dimension for perceiving temporal dynamics
        attention_dim: int,
        action_dim: int,
        sequence_length: int,
        img_disc_channels:int,
        img_disc_layers:int,
        device: torch.device= torch.device('cuda'),
        patch_size: int = 16,
        input_channels: int = 3,  # Number of input channels (e.g., RGB images)
        learning_rate: float = 1e-5,
        grad_clip:float =10.0,
        prior_alpha: float = 1.0,  # Add these parameters
        prior_beta: float = 10.0,
        weight_decay: float = 0.00001,
        use_orthogonal: bool = True,  # Use orthogonal initialization for LSTM,
        number_lstm_layer: int = 2,  # Number of LSTM layers
        HiP_type: str = 'Mini',  # Type of perceiver model
        self_model_weight: float = 0.5,
        use_global_context: bool = True,
        context_window_size: int = 10,
        attention_resolution: int = 21,  # Resolution for attention maps
    ):
        super().__init__()
        #core dimensions
        self.input_channels = input_channels
        self.image_size = input_dim
        self.max_K = max_components
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim ##
        self.attention_dim = attention_dim ##
        self.action_dim = action_dim ##
        self.sequence_length = sequence_length ##
        self.device = device
        # Hyperparameters
        self.HiP_type = HiP_type
        self.min_temperature = 0.1
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.self_model_weight = self_model_weight
        self.use_global_context = use_global_context
        self.context_window_size = context_window_size
        self._lr = learning_rate
        self._grad_clip = grad_clip
        self.attention_resolution = attention_resolution
        self.number_lstm_layer = number_lstm_layer
        self.temporal_attention_temp = nn.Parameter(torch.tensor(1.0))
        self.current_epoch = 0
        #initialization different parts of the model
        self._init_encoder_decoder()
        self._init_perceiver_context()
        self._init_vrnn_dynamics(use_orthogonal=use_orthogonal,number_lstm_layer=number_lstm_layer)
        self._init_attention_schema(attention_resolution)
        self._init_self_model()
        self._init_discriminators( img_disc_layers, patch_size)
        # DP-GMM prior
        self.prior = DPGMMPrior(max_components, 
                                self.latent_dim, 
                                hidden_dim + context_dim, #This is because that the input for the prior is the hidden state of recurrent model plus context which is coming from perceiver
                                device, prior_alpha=prior_alpha,
                                prior_beta=prior_beta)

        # Initialize weights
        
        self.to(device)
        
        # Setup optimizers
        self.has_optimizers = True
        self._setup_optimizers(learning_rate, weight_decay)
         
    def _init_encoder_decoder(self):
        # Encoder network (can reuse your existing encoder architecture)
        self.encoder = VAEEncoder(
                                  channel_size_per_layer=[64,64,128,128,128,128,256,256],
                                  layers_per_block_per_layer=[2,2,2,2,2,2,2,2],
                                  latent_size=self.latent_dim,
                                  width=self.image_size,
                                  height=self.image_size,
                                  num_layers_per_resolution=[2,2,2,2],
                                  mlp_hidden_size=self.hidden_dim,
                                  channel_size=64,
                                  input_channels=self.input_channels,
                                  downsample=4
                                  ).to(self.device)
        # Decoder network (can reuse your existing decoder architecture)
        self.decoder = VAEDecoder(
                                  latent_size=self.latent_dim,
                                  width=self.image_size,
                                  height= self.image_size,
                                  channel_size_per_layer=[256,256,128,128,128,128,64,64],
                                  layers_per_block_per_layer=[2,2,2,2,2,2,2,2],
                                  num_layers_per_resolution=[2,2,2,2],
                                  input_channels=self.input_channels,
                                  downsample=4,
                                  mlp_hidden_size= self.hidden_dim
                                  ).to(self.device)
        self.encoder.gradient_checkpointing_enable()
        self.decoder.gradient_checkpointing_enable()
        

        #self.ema_encoder = EMA(self.encoder, decay=0.995, use_num_updates=True)
        #self.ema_decoder = EMA(self.decoder, decay=0.995, use_num_updates=True)

        self.apply(self.init_weights)

    def _init_perceiver_context(self ):
        """Initialize Hierarchical Perceiver with architecture-aware dimensions"""
    
        # Initialize perceiver
        mock_input = self.generate_mock_input()

        self.perceiver_model = generate_model('HiPClassBottleneck', self.HiP_type, mock_input)
        self.perceiver_model.to(self.device)
        self.out_keys = perceiver_helpers.ModelOutputKeys
        # Extract key dimensions from the processor block (middle block)
        with torch.no_grad():
            # TODO: Can you check why we cut this from computing gradients?
            # Run a forward pass to get actual dimensions
            test_output = self.perceiver_model(mock_input, is_training=False)
            test_latents = test_output[self.out_keys.LATENTS]['image']
            
            # Get actual channel dimension
            perceiver_latent_dim = test_latents.shape[-1]  # [batch, index, channels]
            perceiver_index_dim = test_latents.shape[-2]   # Number of latent vectors
        print(f"Detected Perceiver context output dimension: {perceiver_latent_dim}, {perceiver_index_dim} latent vectors")
   

    
        # Architecture-aware projection
        self.perceiver_projection = PerceiverContextEmbedding(perceiver_latent_dim, 
                                                              self.context_dim, 
                                                              device= self.device)
                

    def _init_discriminators(self, img_disc_layers:int, patch_size: int):
        # Initialize discriminators
        self.image_discriminator = TemporalDiscriminator(
            input_channels=self.input_channels,
            image_size= self.image_size,
            hidden_dim=self.hidden_dim,
            n_layers= img_disc_layers,
            n_heads= 4,
            max_sequence_length=self.sequence_length,
            patch_size= patch_size,
            device= self.device,
            )
        

        

    def _init_vrnn_dynamics(self,use_orthogonal: bool = True, number_lstm_layer: int = 2):
        """Initialize VRNN components with context conditioning"""
        # Feature extractors        
        
        # VRNN recurrence: h_t = f(h_{t-1}, z_t, c_t, A_t, a_t)
        self._rnn = LSTMLayer(
            input_size=self.latent_dim + self.action_dim + self.context_dim + self.hidden_dim,  # z_t + c_t + A_t + a_t
            hidden_size=self.hidden_dim,
            n_lstm_layers= number_lstm_layer,
            use_orthogonal=use_orthogonal
        )
        self.rnn_layer_norm = nn.LayerNorm(self.hidden_dim)
        # Initialize hidden states
        self.h0 = nn.Parameter(torch.zeros(2, 1, self.hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(2, 1, self.hidden_dim))
    
    def _init_attention_schema(self, attention_resolution: int = 21):
        """Initialize attention schema components"""
        # Compute attention state from h, c, z
        self.attention_prior_posterior = AttentionSchema(
                                                        image_size=self.image_size,
                                                        attention_resolution=attention_resolution,  # 64/4
                                                        hidden_dim=self.hidden_dim,
                                                        latent_dim=self.latent_dim,
                                                        context_dim=self.context_dim,
                                                        attention_dim=self.attention_dim,
                                                        input_channels=self.input_channels,
                                                        )
        # Feature extractor for attention
        self.phi_attention = LinearResidual(self.attention_dim + self.hidden_dim//2 + 2, 2*self.hidden_dim)
    
    def _init_self_model(self):
        """Initialize comprehensive self-modeling components"""
        # Hidden state predictor: p(h_{t+1}|h_t, z_t, a_t)
        self.self_model = SelfModelBlock(
            z_dim =self.latent_dim,
            A_dim=self.hidden_dim,
            a_dim=self.action_dim,
            c_dim=self.context_dim,
            h_dim=self.hidden_dim,
            d=48,
            nhead=8,
        )


    def init_weights(self, module: nn.Module):
        """Initialize the weights using the typical initialization schemes """
        if isinstance(module, nn.Linear):
           nn.init.trunc_normal_(module.weight, std=0.02)
           if module.bias is not None:
              nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
           nn.init.trunc_normal_(module.weight, std=0.02)
           if module.padding_idx is not None:
              module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Conv2d):
           nn.init.kaiming_normal_(module.weight,
                                   mode='fan_out',
                                   nonlinearity='relu')
           if module.bias is not None:
              nn.init.zeros_(module.bias)
        elif isinstance(module, nn.ConvTranspose2d):
           nn.init.kaiming_normal_(module.weight,
                                   mode='fan_out',
                                   nonlinearity='relu')
           if module.bias is not None:
              nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
           # Initialize q, k, v projections
           if module.in_proj_weight is not None:
              nn.init.xavier_uniform_(module.in_proj_weight)
           if module.out_proj.weight is not None:
              nn.init.xavier_uniform_(module.out_proj.weight)
           if module.in_proj_bias is not None:
              nn.init.zeros_(module.in_proj_bias)
           if module.out_proj.bias is not None:
              nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
           nn.init.ones_(module.weight)
           nn.init.zeros_(module.bias)

    def contrastive_loss(
        self,
        real_features: torch.Tensor,  # [B, T, D]
        fake_features: torch.Tensor,   # [B, T, D]
        temperature: float = 0.1,
        alpha: float = 0.7,
        temporal_margin: int = 5
    ) -> torch.Tensor:
        """
        Combines:
        1. Intra-video alignment: real_t with fake_t
        2. Inter-video separation: keep videos from different batches apart
        3. Temporal coherence with margin-based negatives
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


    def _setup_optimizers(self, learning_rate=1e-4, weight_decay=0.0001):
        """
        Fixed optimizer setup ensuring all components are properly included
        """
        # Collect all model parameters by component
        param_groups = []
    
        # 1. Perceiver parameters (slow learning)
        perceiver_params = {
            'params': list(self.perceiver_model.parameters()) + 
                      list(self.perceiver_projection.parameters()),
            'lr': learning_rate * 0.5,
            'weight_decay': weight_decay,
            'name': 'perceiver'
        }
        param_groups.append(perceiver_params)
    
        # 2. Encoder/Decoder parameters (normal learning)
        encoder_decoder_params = {
            'params': list(self.encoder.parameters()) + 
                      list(self.decoder.parameters()),
            'lr': learning_rate,
            'weight_decay': weight_decay,
            'name': 'encoder_decoder'
        }
        param_groups.append(encoder_decoder_params)
    
        # 3. Prior network parameters (fast learning)
        prior_params = {
            'params': list(self.prior.parameters()),
            'lr': learning_rate * 0.5,
            'weight_decay': weight_decay * 0.1,
            'name': 'prior'
        }
        param_groups.append(prior_params)
    
        # 4. VRNN dynamics parameters
        vrnn_params = {
            'params': list(self._rnn.parameters()) + 
                      list(self.rnn_layer_norm.parameters()) +
                      list(self.phi_attention.parameters()) +
                      [self.h0, self.c0],  # Don't forget initial states!
            'lr': learning_rate,
            'weight_decay': weight_decay,
            'name': 'vrnn_dynamics'
        }
        param_groups.append(vrnn_params)
    
        # 5. Attention schema parameters (FIXED - was missing attention_computer)
        attention_params = {
            'params': list(self.attention_prior_posterior.parameters()),
            'lr': learning_rate,
            'weight_decay': weight_decay,
            'name': 'attention_schema'
        }
        param_groups.append(attention_params)
    
        # 6. Self-model parameters
        self_model_params = {
            'params': list(self.self_model.parameters()),
            'lr': learning_rate * 1.5,
            'weight_decay': weight_decay * 0.5,
            'name': 'self_model'
        }
        param_groups.append(self_model_params)
    
        # 7. Temperature parameter (special handling)
        temp_params = {
            'params': [self.temperature],
            'lr': learning_rate * 0.1,  # Very slow learning for temperature
            'weight_decay': 0.0,  # No weight decay for temperature
            'name': 'temperature'
        }
        param_groups.append(temp_params)
    
        # Create main optimizer
        self.gen_optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.99),
            eps=1e-7,
            amsgrad=True,  # Use AMSGrad for stability
        )
    
        # Discriminator optimizers (separate for stability)
        self.img_disc_optimizer = torch.optim.AdamW(
            self.image_discriminator.parameters(),
            lr=learning_rate * 0.5,  # Higher LR for discriminators
            betas=(0.9, 0.999),  # Lower beta1 for GANs
            weight_decay=weight_decay,
            eps=1e-8
        )
    
    
        # Setup schedulers
        self._setup_schedulers()
    
        print("Optimizer Setup Complete. Parameter groups:")
        for group in param_groups:
            param_count = sum(p.numel() for p in group['params'] if p.requires_grad)
            print(f"  {group['name']}: {param_count:,} parameters, lr={group['lr']}")

    def _setup_schedulers(self):
        """Setup learning rate schedulers for all optimizers"""
        
        # Store initial learning rates BEFORE creating schedulers
        for group in self.gen_optimizer.param_groups:
            group['initial_lr'] = group['lr']
        
        # 1. Generator scheduler with warm restart
        self.gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.gen_optimizer,
            T_0=1000,  # Initial restart period
            T_mult=2,  # Period multiplier after restart
            eta_min=1e-7  # Minimum learning rate
        )
        
        # 2. Discriminator schedulers with plateau detection
        self.img_disc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.img_disc_optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=True
        )
        
        
        # 3. Initialize warmup tracking
        self.warmup_steps = 1000
        self.current_step = 0

    def optimizer_step(self, loss_dict, is_generator=True):
        """Unified optimizer step with gradient clipping and mixed precision"""
    
        if is_generator:
            optimizer = self.gen_optimizer
            scaler = self.gen_scaler if hasattr(self, 'gen_scaler') else None
            loss = loss_dict['total_loss']
            params = [p for group in optimizer.param_groups for p in group['params']]
        else:
            # Handle discriminator updates
            return self._discriminator_optimizer_step(loss_dict)
    
        # Mixed precision training
        if scaler is not None and torch.cuda.is_available():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=self._grad_clip, norm_type=2.0)
        
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=self._grad_clip, norm_type=2.0)
            optimizer.step()
    
        # Learning rate scheduling
        if self.current_step < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.current_step / self.warmup_steps
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * warmup_factor
        else:
            self.gen_scheduler.step()
    
        self.current_step += 1
    
        return grad_norm
    
    def forward_sequence(self, observations, actions=None):
        """
        Process entire sequence through VRNN with DP-GMM prior
        
        Args:
            observations: [batch_size, seq_len, channels, height, width]
            actions: [batch_size, seq_len, action_dim] (optional)
        
        Returns:
            Dictionary containing all outputs, latents, and losses
        """
        batch_size, seq_len = observations.shape[:2]
        
        # Default actions if not provided
        if actions is None:
            actions = torch.zeros(batch_size, seq_len, self.action_dim).to(self.device)
        
        # Extract global context using Perceiver
        if self.use_global_context:
            context_sequence, cross_attention, perceiver_recon = self.compute_global_context(observations)
        else:
            context_sequence = []
            cross_attention = None
            perceiver_recon = None
        
        # Initialize LSTM hidden states
        h = self.h0.expand(self.number_lstm_layer, batch_size, -1).contiguous()
        c = self.c0.expand(self.number_lstm_layer, batch_size, -1).contiguous()
        
        # Storage for outputs
        outputs = {
            'observations': observations,
            'reconstructions': [],
            'latents': [],
            'z_means': [],
            'z_logvars': [],
            'attention_features': [],
            'attention_states': [],
            'attention_maps': [],
            'hidden_states': [],
            'context_states': [],
            'prior_params': [],
            'posterior_params': [],
            'self_predictions': [], # Predictions made at t for t+1
            'kl_losses': [],
            'attention_losses': [],
            'kumaraswamy_kl_losses': [],
            'kl_latents': [],
            'reconstruction_losses': [],
            'cluster_entropies': [],
            'self_h_prediction_loss': [],
            'self_att_prediction_loss': [],
            'unused_penalties': [],  
            'predicted_movements': [],
        }
        
        # Process sequence step by step
        for t in range(seq_len):
            # Get current inputs
            o_t = observations[:, t]
            a_t = actions[:, t]
            
            # Get context (global or local)
            if self.use_global_context:
                c_t = context_sequence[:, t]
            else:
                c_t = self.compute_local_context(observations, t)
            
            outputs['context_states'].append(c_t)
            
            # === Inference Network q(z_t|o_≤t, z_<t) ===
            z_t, z_mean_t, z_logvar_t = self.encoder(o_t)
            q_params = {'mean': z_mean_t, 'logvar': z_logvar_t}
            outputs['posterior_params'].append(q_params)
            
            # Store latent statistics
            outputs['latents'].append(z_t)
            outputs['z_means'].append(z_mean_t)
            outputs['z_logvars'].append(z_logvar_t)
            
            # === Attention Schema ===
            # Compute attention state A_t
            if t > 0:
                prev_attention = outputs['attention_maps'][-1]
            else:
                # Initialize first attention map
                prev_attention = torch.ones(
                    batch_size, 
                    self.attention_resolution, 
                    self.attention_resolution
                ).to(self.device) / (self.attention_resolution ** 2)
            
            # Posterior attention computation
            attention_map, attention_coords = self.attention_prior_posterior.posterior_net(
                o_t, h[-1], c_t
            )
            outputs['attention_maps'].append(attention_map)
            saliency_features = self.attention_prior_posterior.posterior_net.saliency_net(o_t)
            weighted_visual_features = self.attention_prior_posterior.posterior_net.attention_weighted_features(
            saliency_features, attention_map
           )            
            # Extract attention features
            attention_feat = self.attention_prior_posterior.attention_encoder(
                attention_map.unsqueeze(1)  # Add channel dimension
            )
            #print(f"Attention features shape: {attention_feat.shape}")  # Debugging shape [batch_size, attention_dim]
            attention_features = torch.cat([
             attention_feat,      # Spatial statistics: [B, attention_dim//2]
             weighted_visual_features,         # Content features: [B, hidden_dim//2]
             attention_coords      # Spatial coordinates: [B, 2]
            ], dim=-1)
     
            outputs['attention_features'].append(attention_features)
            attention_state = self.phi_attention(attention_features)
            attention_state_mean, attention_state_logvar = torch.chunk(attention_state, 2, dim=-1)
            attention_state_logvar = torch.clamp(attention_state_logvar, min=-10.0, max=2.0)  # Stability
            attention_state= attention_state_mean + torch.exp(0.5 * attention_state_logvar) * torch.randn_like(attention_state_mean)
            outputs['attention_states'].append(attention_state)
            
            # === Prior Network p(z_t|o_<t, z_<t) ===
            h_context = torch.cat([h[-1], c_t], dim=-1)
            
            # Get DP-GMM prior distribution
            prior_dist, prior_params = self.prior(h_context)
            outputs['prior_params'].append(prior_params)
            # 1. Reconstruction loss for this timestep
            reconstruction_t = self.decoder(z_t)
            outputs['reconstructions'].append(reconstruction_t)

            outputs['reconstruction_losses'].append(F.mse_loss(reconstruction_t, o_t, reduction='mean'))

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

            # Convert K_eff to integer for each batch element
            unused_penalties = []
            for i in range(prior_params['pi'].shape[0]):  # Loop over batch
                k = int(K_eff[i].item())  # Convert to integer
                # Calculate penalty for this batch element
                if k < prior_params['pi'].shape[1]:  # Only if there are unused components
                    unused_penalties.append(torch.sum(prior_params['pi'][i, k:]))
                else:
                    unused_penalties.append(torch.tensor(0.0, device=prior_params['pi'].device))

            unused_penalty = torch.stack(unused_penalties).mean()  # Average over batch


            outputs['unused_penalties'].append(unused_penalty)
            outputs['K_eff'] = K_eff.float().mean()
           # Negative for ELBO

            # 5. Cluster entropy
            entropy_t, cluster_stats = self.compute_conditional_entropy(
                logits=prior_dist.mixture_distribution.logits,
                params=prior_params
            )
            mixing_proportions=prior_params['pi'] 
            mixing_entropy = -torch.sum(mixing_proportions * torch.log(mixing_proportions + torch.finfo(mixing_proportions.dtype).eps), dim=-1).mean()
            entropy_t += mixing_entropy  # Add mixing entropy to cluster entropy
            # Total entropy encourages both:
            # 1. Uncertainty in cluster assignments (exploration)
            # 2. Diversity in cluster usage (avoid mode collapse)
            outputs['cluster_entropies'].append(entropy_t)
            # 3. Attention dynamics KL
            if t > 0  :
                # Prior attention prediction
                z_prev = outputs['latents'][-2] if len(outputs['latents']) > 1 else z_t
                prior_attention, prior_information = self.attention_prior_posterior.prior_net(
                    prev_attention, h[-1], z_prev
                )

                if 'predicted_movement' in prior_information:
                    dx, dy = prior_information['predicted_movement']
                    # Stack dx and dy into a single tensor [B, 2, H, W]
                    movement_tensor = torch.stack([dx, dy], dim=1)
                    outputs['predicted_movements'].append(movement_tensor)

                
                q_dist = Categorical(probs=attention_map.view(batch_size, -1))
                p_dist = Categorical(probs=prior_attention.view(batch_size, -1))
                attention_kl = torch.distributions.kl_divergence(q_dist, p_dist).mean()
                outputs['attention_losses'].append(attention_kl)
            
            # Total KL for this timestep
            total_kl = kl_z + kumar_beta_kl
            outputs['kl_losses'].append(total_kl)

            # === Self-Model Predictions ===
            # TODO:6. Future prediction bonuses (need t and t+tau)? Does this predict next hidden state? 
            if t < seq_len- 1:   # Time delta for future prediction
                # Make self-model predictions
                self_pred = self.self_model_step(
                    h[-1], 
                    outputs['latents'][-1],
                    attention_state, 
                    actions[:, t],
                    current_time=t,
                    context_t=c_t
                )
                outputs['self_predictions'].append(self_pred)
                if t > 0 and len(outputs['self_predictions']) > 0:
                    #the prediction made at t-1 for t
                    prev_pred = outputs['self_predictions'][t-1]
                    h_pred_dist = prev_pred['distributions']['h_dist']
                    h_loss = -h_pred_dist.log_prob(h[-1]).mean()
                    outputs['self_h_prediction_loss'].append(h_loss)
                    
                    att_pred_dist = self_pred['distributions']['attention_dist']
                    att_loss = -att_pred_dist.log_prob(attention_state).mean()
                    outputs['self_att_prediction_loss'].append(att_loss)
                # Compute self-model KL losses
                                
            
            
            # === Update recurrent state ===
            
            rnn_input = torch.cat([z_t, c_t, attention_state, a_t], dim=-1)
            rnn_output, (h, c) = self._rnn(
                rnn_input, h, c, 
                torch.ones(batch_size).to(self.device)
            )
            rnn_output = self.rnn_layer_norm(rnn_output)
            outputs['hidden_states'].append(h[-1])

        if len(outputs['attention_maps']) >= 2 and len(outputs['predicted_movements']) > 0:
            outputs['attention_dynamics_loss'] = self.attention_prior_posterior.compute_attention_dynamics_loss(
                outputs['attention_maps'],  # Still a list here
                outputs['predicted_movements']  # Still a list of tuples here
            )
        else:
            outputs['attention_dynamics_loss'] = torch.tensor(0.0).to(self.device)

        # === Compute aggregated losses ===
        # Stack temporal sequences
        for key in ['reconstructions', 'latents', 'attention_states', 
                    'hidden_states', 'attention_maps']:
            if outputs[key]:
                outputs[key] = torch.stack(outputs[key], dim=1)

        
        # One-step prediction bonuses (negative of losses for ELBO)
        if outputs['self_h_prediction_loss']:
            outputs['future_h_bonus'] = -torch.stack(outputs['self_h_prediction_loss']).mean()
        else:
            outputs['future_h_bonus'] = torch.tensor(0.0).to(self.device)


        if outputs['self_att_prediction_loss']:
            outputs['future_att_bonus'] = -torch.stack(outputs['self_att_prediction_loss']).mean()
        else:
            outputs['future_att_bonus'] = torch.tensor(0.0).to(self.device)
        
        
        
        # Perceiver loss
        obs_flat = observations.reshape(-1, self.image_size * self.image_size, self.input_channels)
        outputs['perceiver_loss'] = F.mse_loss(perceiver_recon, obs_flat, reduction='mean')
        
        # Add auxiliary outputs
        outputs['context_sequence'] = context_sequence
        outputs['cross_attention'] = cross_attention
        outputs['perceiver_recon'] = perceiver_recon
        
        return outputs 

    def _discriminator_optimizer_step(self, loss_dict):
        """Separate handling for discriminator updates"""
    
        # Image discriminator
        self.img_disc_optimizer.zero_grad()
        img_loss = loss_dict['img_disc_loss']
    
        if hasattr(self, 'disc_scaler') and torch.cuda.is_available():
            self.disc_scaler.scale(img_loss).backward()
            self.disc_scaler.unscale_(self.img_disc_optimizer)
            img_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.image_discriminator.parameters(), max_norm=self._grad_clip, norm_type=2.0
            )
            self.disc_scaler.step(self.img_disc_optimizer)
        else:
            img_loss.backward()
            img_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.image_discriminator.parameters(), max_norm=self._grad_clip, norm_type=2.0
            )
            self.img_disc_optimizer.step()
    
            
        return img_grad_norm
    
    def generate_mock_input(self):
        return {
            'image':
                torch.from_numpy(
                    np.random.random((1, self.image_size*self.image_size, self.input_channels)).astype(np.float32)).to(
                    self.device),
        }
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through VAE with DP-GMM prior

        Args:
            x: Input data [batch_size, channels, height, width]

        Returns:
            reconstruction: Reconstructed input
            params: Dictionary containing model parameters and latents
        """
        # Encode
        z, z_mean, z_logvar = self.encoder(x)
        h = self.encoder.hlayer  # Get hidden representation

        # Get prior distribution and parameters
        prior_dist, prior_params = self.prior(h)

        # Decode
        reconstruction = self.decoder(z)

        return reconstruction, {
            'z': z,
            'z_mean': z_mean,
            'z_logvar': z_logvar,
            'prior_dist': prior_dist,
            **prior_params
        }
    
    def compute_total_loss(self, observations, actions=None, beta=1.0, 
                        entropy_weight=0.1, lambda_img=1.0, 
                        lambda_pred=0.1, lambda_att=0.1):
        """
        Corrected loss computation recognizing that kumar_beta_kl already includes alpha prior KL
        """
        outputs = self.forward_sequence(observations, actions)
        losses = {}
        
        # === Reconstruction Term (Positive in Loss) ===
        # This is -E_q[log p(x|z)] under Gaussian assumption
        losses['recon_loss'] = torch.stack(outputs['reconstruction_losses']).mean()  if outputs['reconstruction_losses'] else torch.tensor(0.0).to(self.device)
        
        # === KL Divergence Terms ===
        
        # 1. Latent KL: KL[q(z|x) || p(z|h,c)]
        losses['kl_z'] = torch.stack(outputs['kl_latents']).mean() if outputs['kl_latents'] else torch.tensor(0.0).to(self.device)
        
        # 2. Hierarchical KL (includes BOTH stick-breaking AND alpha prior)
        # This is what your compute_kl_loss returns
        losses['hierarchical_kl'] = torch.stack(outputs['kumaraswamy_kl_losses']).mean() if outputs['kumaraswamy_kl_losses'] else torch.tensor(0.0).to(self.device)
        
        # Note: We do NOT add alpha_prior_kl separately because it's already in hierarchical_kl!
        
        # 3. Attention KL
        losses['attention_kl'] = torch.stack(outputs['attention_losses']).mean() if outputs['attention_losses'] else torch.tensor(0.0).to(self.device)
        # === Other Terms ===
        losses['cluster_entropy'] = torch.stack(outputs['cluster_entropies']).mean() if outputs['cluster_entropies'] else torch.tensor(0.0).to(self.device)
        losses['unused_penalty'] = torch.stack(outputs['unused_penalties']).mean() if outputs['unused_penalties'] else torch.tensor(0.0).to(self.device)
        
        

        # Auxiliary losses
        losses['perceiver_loss'] = outputs['perceiver_loss'] 

        # === Total Loss (Minimizing Negative ELBO) ===
        losses['total_vae_loss'] = (
            # Reconstruction (positive - we want to minimize error)
            lambda_img * losses['recon_loss'] +
            
            # KL terms (positive - we want distributions to be close)
            beta * losses['kl_z'] +
            beta * losses['hierarchical_kl'] +  # This includes both Kumar-Beta AND Alpha-Gamma KL!
            beta * losses['attention_kl'] +
            beta * outputs['attention_dynamics_loss'] +  # Attention dynamics loss
            # Penalties (positive)
            losses['unused_penalty'] +
            
            losses['perceiver_loss'] +
            
            # Entropy (negative - we want to maximize diversity)
            - entropy_weight * losses['cluster_entropy'] +
            
            # Prediction bonuses (negative - rewards for good predictions)
            lambda_pred * outputs['future_h_bonus'] +
            lambda_att * outputs['future_att_bonus']
        )
        #Maximizing ELBO: -recon - KL + entropy 
        # For monitoring, let's also track ELBO
        losses['elbo'] = -losses['total_vae_loss']
        
        return losses, outputs

    def update_temperature(self, epoch: int, anneal_rate: float = 0.003):
        """
         Update temperature parameter with cosine annealing schedule
    
         Args:
            epoch: Current epoch number
            min_temp: Minimum temperature value
            anneal_rate: Rate of temperature annealing
        """
        self.temperature.data = torch.tensor(
            max(
                self.min_temperature,
                1.0 * math.exp(-anneal_rate * epoch)
                ),device=self.device) 
        
    def compute_conditional_entropy(self, logits, params):
        """
        Compute conditional entropy of cluster assignments with temperature scaling.
        H(C|Z) = -E[log P(C|Z)]

        Args:
            logits: Raw logits for cluster assignments [batch_size, n_clusters]
            params: Dictionary containing model parameters and variables

        Returns:
            entropy: Mean conditional entropy
            cluster_stats: Dictionary with clustering statistics
        """
        # Ensure temperature stays above minimum value
        temperature = torch.clamp(self.temperature, min=self.min_temperature)

        # Computer cluster probabilities with temperature scaling
        probs = F.softmax(logits / temperature, dim=-1)
        log_probs = F.log_softmax(logits / temperature, dim=-1)

        # Compute entropy for each sample
        eps = torch.finfo(torch.float32).eps
        entropy_per_sample = -torch.sum(probs * log_probs, dim=-1)

        # Compute mean entropy
        mean_entropy = entropy_per_sample.mean()

        # Compute additional clustering statistics
        cluster_stats = {
            'temperature': temperature.item(),
            'mean_prob_per_cluster': probs.mean(0),
            'max_prob_per_sample': probs.max(1)[0].mean(),
            'entropy_std': entropy_per_sample.std(),
            'active_clusters': (probs.mean(0) > eps).sum().item()
        }

        return mean_entropy, cluster_stats

    def compute_gradient_penalty(self, discriminator, real_samples, fake_samples, device):
        """
        Compute gradient penalty for WGAN-GP
        """
        batch_size = real_samples.size(0)
        if len(real_samples.shape) == 5:  # [B,T,C,H,W]
            alpha = torch.rand(batch_size, 1, 1, 1, 1, device=device)
        elif len(real_samples.shape) == 3:  # [B,T,Z]
            alpha = torch.rand(batch_size, 1, 1, device=device)

        # Get interpolated samples
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)

        # Get discriminator output for interpolated samples
        disc_interpolates = discriminator(interpolates)['final_score']

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates, device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty


    def discriminator_step(
        self,
        real_images: torch.Tensor, #[B, T, C, H, W]
        fake_images: torch.Tensor, #[B, T, C, H, W]
    ) -> Dict[str, torch.Tensor]:
        """
        Training step for both discriminators
        """
        seq_len = real_images.shape[1]  # Get sequence length from real images
        # Temporal Image Discriminator
        real_img_outputs = self.image_discriminator(real_images)
        fake_img_outputs = self.image_discriminator(fake_images.detach())
    
        # Extract final scores
        real_img_score = real_img_outputs['final_score']
        fake_img_score = fake_img_outputs['final_score']
        # Gradient penalty requires computing second-order gradients
        # Gradient penalty for images (sample random interpolation points in sequence)
        img_gp = self.compute_gradient_penalty(
            self.image_discriminator,
            real_images,
            fake_images.detach(),
            self.device
        )


        img_disc_loss = (
            torch.mean(fake_img_score) - torch.mean(real_img_score) +
            10.0 * img_gp
        )

        
        # Temporal consistency losses
        img_consistency_loss = torch.mean(
            fake_img_outputs['per_frame_scores'].std(dim=1)
        )
    
        
        result = {
            'img_disc_loss': img_disc_loss,
            'img_gp': img_gp,
            'real_img_score': real_img_score.mean(),
            'fake_img_score': fake_img_score.mean(),
            'img_temporal_score': real_img_outputs['temporal_score'].mean(),
            'img_consistency_loss': img_consistency_loss,
        }
    
        # Only perform optimization if optimizers exist and we're configured to use them
        if self.has_optimizers and hasattr(self, 'img_disc_optimizer'):
            # Update image discriminator
            self.img_disc_optimizer.zero_grad()
            img_disc_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.image_discriminator.parameters(), self._grad_clip)
            self.img_disc_optimizer.step()

                        
    
        return result
    
    def compute_feature_matching_loss(self, real_features, fake_features):
        """
        Compute feature matching loss between real and fake features
        
        Uses both mean and standard deviation matching for better stability
        """
        feat_match_loss = torch.tensor(0.0, device=self.device)
        num_features = len(real_features)
        
        for i in range(num_features):
            real_feat = real_features[i]
            fake_feat = fake_features[i]
            
            # For CNN features: [B, T, C, H, W]
            if real_feat.dim() == 5:
                # Match mean across spatial dimensions
                real_mean = real_feat.mean(dim=[3, 4])  # [B, T, C]
                fake_mean = fake_feat.mean(dim=[3, 4])
                
                feat_match_loss += F.l1_loss(fake_mean, real_mean.detach())
                
            # For feature vectors: [B, T, D]
            elif real_feat.dim() == 3:
                # Match mean and std across feature dimension
                real_mean = real_feat.mean(dim=2)  # [B, T]
                fake_mean = fake_feat.mean(dim=2)
                
                feat_match_loss += F.l1_loss(fake_mean, real_mean.detach())

        return  feat_match_loss /  num_features  # Normalize by number of features


    def compute_adversarial_losses(
        self,
        x: torch.Tensor, #[B, T, C, H, W  ]
        reconstruction: torch.Tensor, #[B, T, C, H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adversarial losses for both image and latent space
        """
        # Image sequence adversarial loss
        fake_img_outputs = self.image_discriminator(reconstruction, return_features=True)
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
        real_features = self.image_discriminator(x, return_features=True)['spatial_features']
        fake_features = fake_img_outputs['spatial_features']

        # L1 loss between feature statistics
        feature_match_loss = self.compute_feature_matching_loss(real_features, fake_features) 

        return img_adv_loss + temporal_loss_frames, feature_match_loss
      
    def compute_global_context(self, observations):
        """
        Process entire sequence through Perceiver for global context
        
        Args:
            observations: [batch_size, seq_len, channels, height, width]
        
        Returns:
            context: [batch_size, seq_len, context_dim]
            cross_attention: [batch_size, seq_len, num_heads, spatial_dim]
        """
        batch_size, seq_len, channels, H, W = observations.shape
        
        # Flatten observations for perceiver
        obs_flat = observations.reshape(batch_size * seq_len, H*W, self.input_channels)
        
        # Process through perceiver
        perceiver_output = self.perceiver_model({'image': obs_flat}, is_training=True)
        latents = perceiver_output[self.out_keys.LATENTS]['image']
        #Perceiver latents shape: torch.Size([50, 2048, 128]) [batch_size* seq_len, ?, ?]
        # Extract reconstruction for perceiver loss
        perceiver_recon = perceiver_output[self.out_keys.INPUT_RECONSTRUCTION]['image']
        
        # Project to context dimension
        
        context, _ = self.perceiver_projection(latents)
        context = context.reshape(batch_size, seq_len, self.context_dim)
        
        # Extract cross-attention patterns for attention schema
        # Note: This would require modifying perceiver to expose attention weights
        cross_attention = self._extract_perceiver_attention(perceiver_output)
        
        return context, cross_attention, perceiver_recon
    
    def compute_local_context(self, observations, t):
        """
        Compute context using local window around time t
        
        Args:
            observations: [batch_size, seq_len, channels, height, width]
            t: current timestep
        
        Returns:
            context: [batch_size, context_dim]
        """
        batch_size, seq_len = observations.shape[:2]
        
        # Define window boundaries
        window_start = max(0, t - self.context_window_size // 2)
        window_end = min(seq_len, t + self.context_window_size // 2 + 1)
        
        # Extract local observations
        local_obs_flat = observations[:, window_start:window_end].reshape(batch_size * (window_end - window_start), -1, 1)

        # Process through perceiver
        perceiver_output = self.perceiver_model({'image': local_obs_flat}, is_training=True)
        latents = perceiver_output[self.out_keys.LATENTS]['image']
        
        # Average pool over the window and project
        latents = latents.reshape(batch_size, window_end - window_start, -1)
        #compute temporal distances from current time t
        distances = torch.abs(torch.arange(window_start, window_end, dtype=torch.float32, device=observations.device) - t)
        attention_weights= F.softmax(-distances/self.temporal_attention_temp, dim=-1).unsqueeze(0).unsqueeze(-1) # [1, window_size, 1]
        latents = latents* attention_weights.expand(batch_size, -1, latents.shape[-1])  # Apply attention weights
        
        # Average over the window
        latents = latents.sum(dim=1, keepdim=True).transpose(1, 2)   # [batch_size, latent_dim. 1]
        context, _ = self.perceiver_projection(latents).squeeze()
        
        return context
    
 
    
    def self_model_step(self, h_t, z_t, attention_t, a_t, current_time, context_t=None):
        """
        Predict next internal states using self-model
        
        This implements the predictive components:
        - p(h_{t+1}|h_t, z_t, A_t, a_t, c_t)
        - p(A_{t+1}|h_t, z_t, A_t, a_t, c_t)
        
        Args:
            h_t: Current hidden state [batch_size, hidden_dim]
            z_t: Current latent state [batch_size, latent_dim] - NOW PROPERLY USED
            attention_t: Current attention state [batch_size, attention_dim]
            a_t: Current action [batch_size, action_dim]
            c_t: current context (for z prediction) [batch_size, context_dim]
        
        Returns:
            Dictionary with predicted distributions for next timestep
        """
        # 1. Predict h_{t+1} ~ p(h_{t+1}|h_t, z_t, a_t)
        # The hidden state evolution depends on current hidden state, latent, and action
        if context_t is None:
            c_t = torch.zeros(h_t.shape[0], self.context_dim).to(h_t.device)
        else:
            c_t = context_t
        
        t_normalized = torch.tensor(
            [[current_time / self.sequence_length]],
            dtype=torch.float32
        ).to(self.device).expand(h_t.shape[0], -1)


        h_mean, h_logvar , attention_mean, attention_logvar, _ = self.self_model(z_t, attention_t, a_t, c_t, h_t, t_normalized)
        
        return {
            'h_mean': h_mean, 
            'h_logvar': h_logvar, 
            'attention_mean': attention_mean, 
            'attention_logvar': attention_logvar,
            'distributions': {
                'h_dist': Normal(h_mean,torch.exp(0.5 * h_logvar).clamp(min=-10, max=10)),
                'attention_dist': Normal(attention_mean, torch.exp(0.5 * attention_logvar).clamp(min=-10, max=10))
            }
        }
        

    def _extract_perceiver_attention(self, perceiver_output):
        """
        Extract cross-attention weights from perceiver blocks
        
        The Perceiver uses cross-attention between latent arrays and inputs.
        We need to access the attention weights from the HiPCrossAttention modules.
        """
        attention_maps = []
        
        # The perceiver model structure has blocks with cross-attention
        for i, block in enumerate(self.perceiver_model.blocks):
            # Each PerceiverBlock has a projector (HiPCrossAttention)
            if hasattr(block, 'projector') and hasattr(block.projector, '_attention_weights'):
                attention_weights = block.projector._attention_weights
                
                if attention_weights is not None:
                    # attention_weights shape: [batch*groups, num_heads, queries, keys]
                    # We need to reshape and process these weights
                    
                    batch_size, num_groups, num_heads, Q, K = attention_weights.shape
                    
                    # Reshape to separate batch and groups
                    #print(f"Extracting attention weights from block {i} with shape {attention_weights.shape}, batch_size {batch_size}, num_groups {num_groups}")
                    attention_weights = attention_weights.view(
                        batch_size, num_groups, 
                        attention_weights.shape[2],  # num_heads
                        attention_weights.shape[3],  # queries
                        attention_weights.shape[4]   # keys
                    )
                    
                    # Average over heads and groups for visualization
                    attention_avg = attention_weights.mean(dim=[1, 2])  # [batch, queries, keys]
                    
                    attention_maps.append(attention_avg)
        
        # Use the processor block (middle block) attention as primary
        processor_idx = len(self.perceiver_model.blocks) // 2
        
        if processor_idx < len(attention_maps) and len(attention_maps) > 0:
            primary_attention = attention_maps[processor_idx]
            
            # Reshape to spatial format if needed
            # Assuming the keys correspond to flattened spatial positions
            seq_len = primary_attention.shape[-1]
            spatial_size = int(np.sqrt(seq_len))
            
            if spatial_size * spatial_size == seq_len:
                # Reshape to spatial attention map
                primary_attention = primary_attention.view(
                    primary_attention.shape[0], 
                    primary_attention.shape[1], 
                    spatial_size, 
                    spatial_size
                )
            
            return primary_attention
        
        # If no attention found, return None
        return None
    
    def prepare_images_for_training(self, images):
        """
        Normalize images to [-1, 1] range for tanh decoder output
        
        Args:
            images: Input images in [0, 255] or [0, 1] range
        
        Returns:
            Normalized images in [-1, 1] range
        """
        # If images are in [0, 255] range (uint8)
        if images.dtype == torch.uint8 or images.max() > 1.0:
            images = images.float() / 255.0
        
        # Convert from [0, 1] to [-1, 1]
        images = 2 * images - 1
        
        return images

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
                            lambda_pred: float = 0.1,
                            lambda_att: float = 0.1,
                            entropy_weight: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Complete training step for sequence data
        
        Args:
            observations: [batch_size, seq_len, channels, height, width]
            actions: [batch_size, seq_len, action_dim] (optional)
            Other hyperparameters...
        
        Returns:
            Dictionary with all loss values
        """        
        # Normalize observations if needed
        observations = self.prepare_images_for_training(observations)
        self.gen_optimizer.zero_grad()
        # 1. Forward pass and compute all VAE losses
        vae_losses, outputs = self.compute_total_loss(
            observations, actions, beta, entropy_weight, 
            lambda_img, lambda_pred, lambda_att)
        
        # 2. Discriminator training
        
        disc_losses_list = []
        for _ in range(n_critic):
            
            
            disc_loss = self.discriminator_step(
                observations, outputs['reconstructions']
            )
            disc_losses_list.append(disc_loss)
        
        # Average discriminator losses
        avg_disc_losses = {
            k: sum(d[k] for d in disc_losses_list) / len(disc_losses_list)
            for k in disc_losses_list[0].keys()
        }
        
        # 4. Generator training with adversarial losses
        # Get fresh discriminator outputs (no detach)
        img_adv_loss, feat_match_loss = self.compute_adversarial_losses(
            observations,
            outputs['reconstructions'],
        )
        
        # 5. Total generator loss
        total_gen_loss = (
            vae_losses['total_vae_loss'] +
            lambda_img * img_adv_loss +
            feat_match_loss        # Feature matching
        )
        
        total_gen_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self._grad_clip, norm_type=2)
        self.gen_optimizer.step()
        #Update EMA
        #self.ema_encoder.update()
        #self.ema_decoder.update()
        # 6. Optimize generator
        self.update_temperature(self.current_epoch)
            
        # Step scheduler
        if self.current_step >= self.warmup_steps:
            self.gen_scheduler.step()
        self.current_step += 1
        
        # 7. Prepare output dictionary
        return {
            **vae_losses,
            **avg_disc_losses,
            'img_adv_loss': img_adv_loss.item(),
            'feat_match_loss': feat_match_loss.item(),
            'total_gen_loss': total_gen_loss.item(),
            'grad_norm': grad_norm.item() if self.has_optimizers else 0.0,
            'temperature': self.temperature.item(),
            'effective_components': outputs['prior_params'][0]['pi'].max(1)[0].mean().item()  # Avg dominant component prob
        }
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Generate samples from the model

        Args:
            num_samples: Number of samples to generate

        Returns:
            Generated samples
        """
        #self.ema_encoder.apply_shadow()
        #self.ema_decoder.apply_shadow()
        # Sample hidden representation
        h = torch.randn(num_samples, self.hidden_dim + self.context_dim, device=self.device)
        # Get prior distribution
        prior_dist, _ = self.prior(h)

        # Sample latents
        z = prior_dist.sample()

        # Decode
        samples = self.decoder(z)
        #self.ema_encoder.restore()
        #self.ema_decoder.restore()
        return samples

    def get_cluster_statistics(self, x: torch.Tensor) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Get detailed clustering statistics for a batch of data.

        Args:
           x (torch.Tensor): Input batch with shape [batch_size, channels, height, width]

        Returns:
           Dict[str, Union[float, torch.Tensor]]: Dictionary containing clustering statistics:
               - temperature (float): Current temperature parameter
               - mean_prob_per_cluster (torch.Tensor): Average probability for each cluster
               - max_prob_per_sample (float): Average maximum probability across samples
               - entropy_std (float): Standard deviation of entropy values
               - active_clusters (int): Number of active clusters
               - reconstruction_error (float): MSE between input and reconstruction
               - cluster_assignments (torch.Tensor): Cluster assignments for each sample
               - cluster_proportions (torch.Tensor): Average mixing proportions
        """
        with torch.no_grad():
            # Forward pass
            reconstruction, params = self.forward(x)

            # Get entropy and clustering statistics
            _, cluster_stats = self.compute_conditional_entropy(
                params['prior_dist'].mixture_distribution.logits,
                params
            )

            # Compute additional statistics
            cluster_stats.update({
               'reconstruction_error': nn.MSELoss()(reconstruction, x).item(),
               'cluster_assignments': params['prior_dist'].mixture_distribution.probs.argmax(1),
               'cluster_proportions': params['pi'].mean(0)
            })

            # Add dimensionality information
            cluster_stats.update({
              'n_samples': x.shape[0],
              'n_clusters': self.max_K,
              'latent_dim': self.latent_dim
            })

            # Add entropy-based clustering quality metrics
            probs = params['prior_dist'].mixture_distribution.probs
            eps = torch.finfo(torch.float32).eps
            cluster_entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
            cluster_stats.update({
              'min_entropy': cluster_entropy.min().item(),
              'max_entropy': cluster_entropy.max().item(),
              'mean_entropy': cluster_entropy.mean().item()
            })

        return cluster_stats
    
    def sample_next_latent(self, h_t, c_t, temperature=1.0):
        """
        Sample z_{t+1} from the DPGMM prior given current hidden state and context
        
        This follows the generative process:
        p(z_{t+1}|h_t, c_t, π_t) = Σ_k π_{t,k} N(z; μ_k(h_t, c_t), Σ_k(h_t, c_t))
        
        Args:
            h_t: Current hidden state [batch_size, hidden_dim]
            c_t: Current context [batch_size, context_dim]
            temperature: Sampling temperature for stochasticity control
        
        Returns:
            z_next: Sampled next latent state [batch_size, latent_dim]
            prior_info: Dictionary with prior parameters and selected components
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


    def generate_future_sequence(self, initial_obs, initial_action=None, horizon=10, 
                            context_window=None, temperature=1.0):
        """
        Generate a complete future sequence of observations
        
        This method generates future observations by:
        1. Encoding initial observation to get starting state
        2. Using VRNN dynamics to evolve hidden states
        3. Sampling from DPGMM prior for latent states
        4. Decoding latent states to generate observations
        
        Args:
            initial_obs: Starting observation [batch_size, C, H, W]
            initial_action: Initial action [batch_size, action_dim]
            horizon: Number of future steps to generate
            context_window: Previous observations for context [batch_size, window_size, C, H, W]
            temperature: Sampling temperature
        
        Returns:
            Dictionary with generated sequence and intermediate states
        """
        batch_size = initial_obs.shape[0]
        device = initial_obs.device
        
        # Normalize input if needed
        initial_obs = self.prepare_images_for_training(initial_obs)
        
        # Initialize action if not provided
        if initial_action is None:
            initial_action = torch.zeros(batch_size, self.action_dim).to(device)
        
        # Storage for generated sequence
        generated_observations = [initial_obs]
        generated_latents = []
        generated_attentions = []
        generated_attention_coords = []
        attention_uncertainties = []
        hidden_states = []
        
        with torch.no_grad():
            # 1. Encode initial observation
            z_0, _, _ = self.encoder(initial_obs)
            generated_latents.append(z_0)
            
            # 2. Get initial context
            if self.use_global_context and context_window is not None:
                # Use perceiver for global context
                context_sequence, _, _ = self.compute_global_context(context_window)
                c_0 = context_sequence[:, -1]  # Use last context
            else:
                # Use current observation for context
                c_0 = self.compute_local_context(initial_obs.unsqueeze(1), 0)
            
            # 3. Initialize LSTM states
            h = self.h0.expand(self.number_lstm_layer, batch_size, -1).contiguous()
            c = self.c0.expand(self.number_lstm_layer, batch_size, -1).contiguous()
            
            # 4. Initialize attention
            attention_map = torch.ones(
                batch_size, 
                self.attention_resolution, 
                self.attention_resolution
            ).to(device) / (self.attention_resolution ** 2)
            attention_map, attention_coords = self.attention_prior_posterior.posterior_net(
                initial_obs, h[-1], c_0
            )            
            # Process initial step through VRNN
            attention_feat = self.attention_prior_posterior.attention_encoder(
                attention_map.unsqueeze(1)
            )
            saliency_features = self.attention_prior_posterior.posterior_net.saliency_net(
                initial_obs
            )
            weighted_visual_features = self.attention_prior_posterior.posterior_net.attention_weighted_features(
            saliency_features, attention_map
           )
            #Get coordinates directly from posterior attention map
            generated_attention_coords.append(attention_coords)
            attention_features = torch.cat([attention_feat, weighted_visual_features, attention_coords], dim=-1)
            # Feature extraction

            phi_attention = self.phi_attention(attention_features)
            phi_attention_mean, phi_attention_logvar = torch.chunk(phi_attention, 2, dim=-1)
            phi_attention_std = torch.exp(0.5 * phi_attention_logvar.clamp(min=-10, max=5))
            phi_attention = phi_attention_mean + phi_attention_std * torch.randn_like(phi_attention_std).to(device)
            attention_uncertainties.append(phi_attention_std.mean(dim=-1))

            
            # Update LSTM
            rnn_input = torch.cat([z_0, c_0, phi_attention, initial_action], dim=-1)
            _, (h, c) = self._rnn(rnn_input, h, c, torch.ones(batch_size).to(device))
            
            current_h = h[-1]
            current_c_t = c_0
            current_action = initial_action
            
            # 5. Generate future sequence
            for t in range(horizon):
                # Sample next latent from DPGMM prior
                z_next, prior_info = self.sample_next_latent(current_h, current_c_t, temperature)
                generated_latents.append(z_next)
                
                # Decode to observation
                o_next = self.decoder(z_next)
                generated_observations.append(o_next)
                
                # Update attention using prior network
                attention_map, attention_info = self.attention_prior_posterior.prior_net(
                    attention_map, current_h, z_next
                )
                generated_attentions.append(attention_map)
                
                # Extract attention features
                attention_feat = self.attention_prior_posterior.attention_encoder(
                    attention_map.unsqueeze(1)
                )
                #TODO: is this correct? Should we use attention_features or attention_map?
                if 'predicted_movement' in attention_info:
                    dx, dy = attention_info['predicted_movement']
                    # Update attention coordinates based on movement
                    new_coords = attention_coords + torch.stack([dx.mean(dim=[1,2]), 
                                                            dy.mean(dim=[1,2])], dim=-1)
                    new_coords = self.attention_prior_posterior.posterior_net.spatial_regularizer(new_coords)
                    generated_attention_coords.append(new_coords)
                    attention_coords = new_coords
   
                approx_saliency = self.attention_prior_posterior.posterior_net.saliency_net(o_next)
                approx_weighted_features = self.attention_prior_posterior.posterior_net.attention_weighted_features(
                approx_saliency, attention_map
                )
                attention_features = torch.cat([
                    attention_feat,
                    approx_weighted_features,
                    attention_coords
                    ], dim=-1)
                # Update VRNN state
                phi_attention = self.phi_attention(attention_features)
                phi_attention_mean, phi_attention_logvar = torch.chunk(phi_attention, 2, dim=-1)
                phi_attention_std = torch.exp(0.5 * phi_attention_logvar.clamp(min=-10, max=5))
                attention_uncertainties.append(phi_attention_std.mean(dim=-1))
                phi_attention = phi_attention_mean + phi_attention_std * torch.randn_like(phi_attention_std).to(device)

                
                rnn_input = torch.cat([z_next, current_c_t, phi_attention, current_action], dim=-1)
                _, (h, c) = self._rnn(rnn_input, h, c, torch.ones(batch_size).to(device))
                
                current_h = h[-1]
                hidden_states.append(current_h)
                
                # For simplicity, keep context and action constant
                # In practice, you might update these based on the generated sequence
        
        # Stack results
        generated_observations = torch.stack(generated_observations, dim=1)  # [B, T+1, C, H, W]
        generated_latents = torch.stack(generated_latents, dim=1)  # [B, T+1, latent_dim]
        
        if generated_attentions:
            generated_attentions = torch.stack(generated_attentions, dim=1)  # [B, T, H_att, W_att]
        
        # Denormalize images for visualization
        generated_observations = self.denormalize_generated_images(generated_observations)
        
        return {
            'observations': generated_observations,
            'latents': generated_latents,
            'attention_maps': generated_attentions,
            'attention_uncertainties': torch.stack(attention_uncertainties, dim=1),
            'attention_coords': torch.stack(generated_attention_coords, dim=1),
            'hidden_states': torch.stack(hidden_states, dim=1) if hidden_states else None,
            'initial_context': c_0
        }
    
    def visualize_attention_comparison(self, observations, predicted_attention, 
                                    true_attention=None, alpha=0.5):
        """
        Enhanced attention visualization with comparison capability
        
        Args:
            observations: Original images [B, C, H, W]
            predicted_attention: Model predictions [B, H_att, W_att]
            true_attention: Ground truth if available [B, H_att, W_att]
            alpha: Blending factor for overlay
        
        Returns:
            Dictionary with visualization tensors
        """
        batch_size = observations.shape[0]
        
        # Upsample attention to image resolution
        pred_upsampled = F.interpolate(
            predicted_attention.unsqueeze(1),
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        )
        
        # Create heatmap overlays
        pred_heatmap = self._create_heatmap(pred_upsampled, colormap='hot')
        blended_pred = alpha * pred_heatmap + (1 - alpha) * observations
        
        visualizations = {'predicted': blended_pred}
        
        if true_attention is not None:
            true_upsampled = F.interpolate(
                true_attention.unsqueeze(1),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
            true_heatmap = self._create_heatmap(true_upsampled, colormap='cool')
            blended_true = alpha * true_heatmap + (1 - alpha) * observations
            
            # Difference map
            attention_diff = torch.abs(pred_upsampled - true_upsampled)
            diff_heatmap = self._create_heatmap(attention_diff, colormap='seismic')
            
            visualizations.update({
                'ground_truth': blended_true,
                'difference': diff_heatmap,
                'comparison': torch.cat([blended_pred, blended_true], dim=3)
            })
        
        return visualizations

    def _create_heatmap(self, attention_map, colormap='hot'):
        """Convert attention values to RGB heatmap"""
        # Normalize to [0, 1]
        att_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # Apply colormap (simplified - in practice use matplotlib colormaps)
        if colormap == 'hot':
            heatmap = torch.zeros((*attention_map.shape[:-2], 3, *attention_map.shape[-2:]))
            heatmap[:, 0] = att_norm.squeeze(1)  # Red channel
            heatmap[:, 1] = (att_norm.squeeze(1) > 0.5).float()  # Green for high attention
        elif colormap == 'cool':
            heatmap = torch.zeros((*attention_map.shape[:-2], 3, *attention_map.shape[-2:]))
            heatmap[:, 2] = att_norm.squeeze(1)  # Blue channel
            heatmap[:, 1] = (att_norm.squeeze(1) > 0.5).float()  # Green for high attention
        
        return heatmap.to(attention_map.device)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Gamma, Categorical, Independent, MixtureSameFamily
from typing import Dict, Tuple, Union, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import (
    VAEEncoder, VAEDecoder, 
    TemporalDiscriminator, TemporalLatentDiscriminator,
    LinearResidual, AttentionPosterior, AttentionPrior,
    compute_feature_matching_loss, AddEpsilon, check_tensor
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

        a = torch.clamp(a, min=eps, max=50)  # Ensure a is positive
        b = torch.clamp(b, min=eps, max=50)  # Ensure b is positive
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
        
        # Compute squared error [T-1, B]
        movement_error = F.huber_loss(
            torch.stack([pred_dx_mean, pred_dy_mean]), 
            torch.stack([actual_dx, actual_dy]),
            delta=0.1
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
    

class DPGMMVariationalRecurrentEncoder(nn.Module):
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
        latent_disc_layers:int,
        device: torch.device= torch.device('cuda'),
        use_spectral_norm: bool= True,
        input_channels: int = 3,  # Number of input channels (e.g., RGB images)
        learning_rate: float = 1e-5,
        grad_clip:float =10.0,
        prior_alpha: float = 5.0,  # Add these parameters
        prior_beta: float = 1.0,
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
        #initialization different parts of the model
        self._init_encoder_decoder()
        self._init_perceiver_context()
        self._init_vrnn_dynamics(use_orthogonal=use_orthogonal,number_lstm_layer=number_lstm_layer)
        self._init_attention_schema(attention_resolution)
        self._init_self_model()
        self._init_discriminators( img_disc_layers, latent_disc_layers, use_spectral_norm)

        

        # DP-GMM prior
        self.prior = DPGMMPrior(max_components, 
                                latent_dim, 
                                hidden_dim + context_dim, #This is because that the input for the prior is the hidden state of recurrent model plus context which is coming from perceiver
                                device, prior_alpha=prior_alpha,
                                prior_beta=prior_beta)

        # Initialize weights
        self.apply(self.init_weights)
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
                
        
        
        # Attention extractor remains the same
        self.attention_extractor = nn.Sequential(
                                                 nn.Linear(self.context_dim, self.attention_dim),
                                                 nn.LayerNorm(self.attention_dim),
                                                 nn.SiLU(),
                                                 nn.Linear(self.attention_dim, self.attention_dim+self.hidden_dim//2 + 2),
                                                 nn.LayerNorm(self.attention_dim+self.hidden_dim//2 + 2)
                                                 ).to(self.device)

    def _init_discriminators(self, img_disc_layers, latent_disc_layers, use_spectral_norm: bool):
        # Initialize discriminators
        self.image_discriminator = TemporalDiscriminator(
            input_channels=self.input_channels,
            image_size= self.image_size,
            hidden_dim=self.hidden_dim,
            n_layers= img_disc_layers,
            n_heads= 4,
            sequence_length=self.sequence_length,
            use_spectral_norm= use_spectral_norm,
            device= self.device,
            )

        # Latent discriminator operates on flattened latent vectors
        self.latent_discriminator = TemporalLatentDiscriminator(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            n_layers= latent_disc_layers,
            n_heads= 4,
            sequence_length=self.sequence_length,
            device=self.device,
            )

    def _init_vrnn_dynamics(self,use_orthogonal: bool = True, number_lstm_layer: int = 2):
        """Initialize VRNN components with context conditioning"""
        # Feature extractors
        self.phi_z = LinearResidual(self.latent_dim, self.hidden_dim)
        self.phi_a = LinearResidual(self.action_dim, self.hidden_dim)
        self.phi_c = LinearResidual(self.context_dim, self.hidden_dim)
        
        # Context-conditioned prior network
        self.prior_net = nn.Sequential(
            nn.Linear(self.hidden_dim + self.context_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.SiLU(),
            LinearResidual(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim + self.context_dim),
            nn.LayerNorm(self.hidden_dim + self.context_dim)
        )
        
        # VRNN recurrence: h_t = f(h_{t-1}, z_t, c_t, A_t, a_t)
        self._rnn = LSTMLayer(
            input_size=self.hidden_dim * 4,  # phi_z + phi_c + phi_A + phi_a
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
        self.self_model_h = nn.Sequential(
            nn.Linear(self.hidden_dim + self.action_dim + self.latent_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.SiLU(),  
            LinearResidual(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),  # mean and logvar
            nn.LayerNorm(self.hidden_dim * 2)
        )
        
        # Latent predictor: p(z_{t+1}|h_{t+1}, c_{t+1})
        self.self_model_z = nn.Sequential(
            nn.Linear(self.hidden_dim + self.context_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.latent_dim * 2),  # mean and logvar
            nn.LayerNorm(self.latent_dim * 2)  # mean and logvar
        )
        
        # Attention predictor: p(A_{t+1}|h_{t+1}, z_{t+1}, A_t, a_t)
        attention_feature_dim = self.attention_dim + self.hidden_dim//2 + 2
        self.self_model_attention = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim + attention_feature_dim + self.action_dim,
                     self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),  # mean and logvar
            nn.LayerNorm(self.hidden_dim * 2)  # mean and logvar
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

    def _setup_optimizers(self, learning_rate=1e-4, weight_decay=0.0001):
        """
        Fixed optimizer setup ensuring all components are properly included
        """
        # Collect all model parameters by component
        param_groups = []
    
        # 1. Perceiver parameters (slow learning)
        perceiver_params = {
            'params': list(self.perceiver_model.parameters()) + 
                      list(self.perceiver_projection.parameters()) + 
                      list(self.attention_extractor.parameters()),
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
            'params': list(self.prior.parameters()) + 
                      list(self.prior_net.parameters()),
            'lr': learning_rate * 2.0,
            'weight_decay': weight_decay * 0.1,
            'name': 'prior'
        }
        param_groups.append(prior_params)
    
        # 4. VRNN dynamics parameters
        vrnn_params = {
            'params': list(self._rnn.parameters()) + 
                      list(self.rnn_layer_norm.parameters()) +
                      list(self.phi_z.parameters()) + 
                      list(self.phi_a.parameters()) + 
                      list(self.phi_c.parameters()) +
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
            'params': list(self.self_model_h.parameters()) + 
                      list(self.self_model_z.parameters()) + 
                      list(self.self_model_attention.parameters()),
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
            lr=learning_rate * 2.0,  # Higher LR for discriminators
            betas=(0.5, 0.999),  # Lower beta1 for GANs
            weight_decay=weight_decay,
            eps=1e-8
        )
    
        self.latent_disc_optimizer = torch.optim.AdamW(
            self.latent_discriminator.parameters(),
            lr=learning_rate * 2.0,
            betas=(0.5, 0.999),
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
        
        self.latent_disc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.latent_disc_optimizer,
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
            'self_predictions': [],
            'kl_losses': [],
            'self_model_losses': [],
            'attention_losses': [],
            'recognition_losses': [],
            'kumaraswamy_kl_losses': [],
            'kl_latents': [],
            'reconstruction_losses': [],
            'cluster_entropies': [],
            'one_step_h_prediction_loss': [],
            'one_step_z_prediction_loss': [],
            'one_step_att_prediction_loss': [],
            'unused_penalties': [],  
            'self_model_kl_losses': [],
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
            attention_features = self.attention_prior_posterior.attention_encoder(
                attention_map.unsqueeze(1)  # Add channel dimension
            )
            
            attention_features = torch.cat([
             attention_features,      # Spatial statistics: [B, attention_dim//2]
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
            h_prior = self.prior_net(h_context)
            
            # Get DP-GMM prior distribution
            prior_dist, prior_params = self.prior(h_prior)
            outputs['prior_params'].append(prior_params)
            # 1. Reconstruction loss for this timestep
            reconstruction_t = self.decoder(z_t)
            outputs['reconstructions'].append(reconstruction_t)
        
            outputs['reconstruction_losses'].append(F.mse_loss(reconstruction_t, o_t, reduction='none').mean(dim=[1,2,3]))
        
            # === KL Divergence Computation ===
            # 1. KL between posterior and DP-GMM prior

            kl_z = self.prior.compute_kl_divergence_mc(z_mean_t, z_logvar_t, prior_params)
            outputs['kl_latents'].append(kl_z)
            
            # 2. KL for stick-breaking (Kumaraswamy vs Beta and Gamma prior vs Gamma posterior)
            kumar_beta_kl = self.prior.compute_kl_loss(
                prior_params, 
                prior_params['alpha'], 
                h_prior
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

                attention_kl = F.kl_div(
                    F.log_softmax(attention_map.view(batch_size, -1), dim=-1),
                    F.softmax(prior_attention.view(batch_size, -1), dim=-1),
                    reduction='batchmean'
                )
                outputs['attention_losses'].append(attention_kl)
            
            # Total KL for this timestep
            total_kl = kl_z + kumar_beta_kl
            outputs['kl_losses'].append(total_kl)

            if t > 0 and t < seq_len:
                # We made predictions at t-1, now evaluate them against actual values at t
                
                # Get predictions made at previous timestep
                prev_pred = outputs['self_predictions'][t-1]
                
                # Compare with actual values
                # Hidden state prediction loss
                h_pred_dist = Normal(prev_pred['h_mean'], torch.exp(0.5 * prev_pred['h_logvar']))
                h_actual = h[-1]  # Current hidden state
                h_pred_loss = -h_pred_dist.log_prob(h_actual).mean()  # Negative log likelihood
                outputs['one_step_h_prediction_loss'].append(h_pred_loss)
                
                # Latent prediction loss
                z_pred_dist = Normal(prev_pred['z_mean'], torch.exp(0.5 * prev_pred['z_logvar']))
                z_pred_loss = -z_pred_dist.log_prob(z_t).mean()
                outputs['one_step_z_prediction_loss'].append(z_pred_loss)
                
                # Attention prediction loss
                att_pred_dist = Normal(prev_pred['attention_mean'], 
                                    torch.exp(0.5 * prev_pred['attention_logvar']))
                att_pred_loss = -att_pred_dist.log_prob(attention_state).mean()
                outputs['one_step_att_prediction_loss'].append(att_pred_loss)

            # === Self-Model Predictions ===
            # TODO:6. Future prediction bonuses (need t and t+tau)? Does this predict next hidden state? 
            if t < seq_len- 1:
                # Get next context if available
                c_next = context_sequence[:, t+1] if t+1 < seq_len and self.use_global_context else c_t
                
                # Make self-model predictions
                self_pred = self.self_model_step(
                    h[-1], 
                    outputs['latents'][-1],
                    outputs['attention_features'][-1], 
                    actions[:, t],
                    c_next
                )
                outputs['self_predictions'].append(self_pred)
                if t > 0:
                
                    # Compute self-model losses
                    self_loss = self._compute_self_model_loss(
                        outputs['self_predictions'][t-1], # Prediction made at t-1 for time t
                        h[-1],                            # Actual h at time t 
                        z_t,                              # Actual z at time t
                        attention_state                   # Actual attention state at timestep t
                        )
                    outputs['self_model_losses'].append(self_loss)
                # Compute self-model KL losses
                # These compare current states with what was predicted
                h_pred_dist = Normal(self_pred['h_mean'], torch.exp(0.5 * self_pred['h_logvar']))
                
                
                z_pred_dist = Normal(self_pred['z_mean'], torch.exp(0.5 * self_pred['z_logvar']))
                z_actual_dist = Normal(z_t, torch.ones_like(z_t) * 0.1)
                z_kl = torch.distributions.kl_divergence(z_actual_dist, z_pred_dist).mean()
                
                attention_pred_dist = Normal(self_pred['attention_mean'], 
                                        torch.exp(0.5 * self_pred['attention_logvar']))
                attention_actual_dist = Normal(attention_state_mean, 
                                        torch.exp(0.5 * attention_state_logvar))
                attention_kl = torch.distributions.kl_divergence(attention_actual_dist, attention_pred_dist).mean()
                
                outputs['self_model_kl_losses'].append({
                    'h_{t+1}': h_pred_dist,
                    'z_kl': z_kl,
                    'attention_kl': attention_kl,
                    'total': z_kl + attention_kl
                })
    
            # === VRNN State Update ===
            # Feature extraction
            
            # === Update recurrent state ===
            phi_z_t = self.phi_z(z_t)
            phi_c_t = self.phi_c(c_t)
            phi_attention_t = attention_state
            phi_a_t = self.phi_a(a_t)
            
            rnn_input = torch.cat([phi_z_t, phi_c_t, phi_attention_t, phi_a_t], dim=-1)
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
        if outputs['one_step_h_prediction_loss']:
            outputs['future_h_bonus'] = -torch.stack(outputs['one_step_h_prediction_loss']).mean()
        else:
            outputs['future_h_bonus'] = torch.tensor(0.0).to(self.device)
        
        if outputs['one_step_z_prediction_loss']:
            outputs['future_z_bonus'] = -torch.stack(outputs['one_step_z_prediction_loss']).mean()
        else:
            outputs['future_z_bonus'] = torch.tensor(0.0).to(self.device)
        
        if outputs['one_step_att_prediction_loss']:
            outputs['future_att_bonus'] = -torch.stack(outputs['one_step_att_prediction_loss']).mean()
        else:
            outputs['future_att_bonus'] = torch.tensor(0.0).to(self.device)
        
        
        # Schema consistency (if using global context)
        if self.use_global_context and context_sequence is not None:
            perceiver_attention = self.attention_extractor(context_sequence)
            attn_features = torch.stack(outputs['attention_features'], dim=1)
            
            outputs['schema_consistency_loss'] = F.mse_loss(
                attn_features, 
                perceiver_attention,
                reduction='mean'
            )
            
        else:
            outputs['schema_consistency_loss'] = torch.tensor(0.0).to(self.device)
        
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
    
        # Latent discriminator
        self.latent_disc_optimizer.zero_grad()
        latent_loss = loss_dict['latent_disc_loss']
    
        if hasattr(self, 'disc_scaler') and torch.cuda.is_available():
            self.disc_scaler.scale(latent_loss).backward()
            self.disc_scaler.unscale_(self.latent_disc_optimizer)
            latent_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.latent_discriminator.parameters(), max_norm=self._grad_clip, norm_type=2.0
            )
            self.disc_scaler.step(self.latent_disc_optimizer)
            self.disc_scaler.update()
        else:
            latent_loss.backward()
            latent_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.latent_discriminator.parameters(), max_norm=self._grad_clip, norm_type=2.0
            )
            self.latent_disc_optimizer.step()
    
        return img_grad_norm, latent_grad_norm
    
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
                        lambda_pred=0.1, lambda_att=0.1, lambda_schema=0.1):
        """
        Corrected loss computation recognizing that kumar_beta_kl already includes alpha prior KL
        """
        outputs = self.forward_sequence(observations, actions)
        losses = {}
        
        # === Reconstruction Term (Positive in Loss) ===
        # This is -E_q[log p(x|z)] under Gaussian assumption
        losses['recon_loss'] = torch.stack(outputs['reconstruction_losses']).mean() if outputs['reconstruction_losses'] else torch.tensor(0.0).to(self.device)
        
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
        
        # Prediction bonuses (already negative)
        losses['future_pred_bonus'] = outputs['future_h_bonus'] + outputs['future_z_bonus']
        losses['attention_pred_bonus'] = outputs['future_att_bonus']
        
        # Auxiliary losses
        losses['schema_consistency'] = outputs['schema_consistency_loss']
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
            lambda_schema * losses['schema_consistency'] +
            1.0 * losses['perceiver_loss'] +
            
            # Entropy (negative - we want to maximize diversity)
            - entropy_weight * losses['cluster_entropy'] +
            
            # Prediction bonuses (negative - rewards for good predictions)
            lambda_pred * losses['future_pred_bonus'] +
            lambda_att * losses['attention_pred_bonus']
        )
        
        # For monitoring, let's also track ELBO
        losses['elbo'] = -losses['total_vae_loss']
        
        return losses, outputs
    
    def update_temperature(self, epoch: int, min_temp: float = 0.1, anneal_rate: float = 0.003):
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

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty


    def compute_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        params: Dict,
        beta: float = 1.0,
        entropy_weight: float =0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ELBO loss with DP-GMM prior

        Args:
            x: Input data
            reconstruction: Reconstructed input
            params: Dictionary of model parameters
            beta: Weight of KL divergence term
            entropy_weight: Weight for conditional entropy term
        Returns:
            Dictionary containing loss components
        """
        losses={}
        # 1. Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, x, reduction='sum')
        losses['recon_loss'] = recon_loss

        # 2. KL divergence for latent space using Expected KL calculation
        # Similar to ExpectedKLDivergence from InfGaussMMVAE
        
        q_mean = params['z_mean']
        q_logvar = params['z_logvar']
        # Get encoder's hidden state
        h = self.encoder.hlayer

        # Compute KL using Monte Carlo estimation
        kl_z = self.prior.compute_kl_divergence_mc(q_mean, q_logvar, params)
        losses['kl_z'] = kl_z


        # 3. KL divergence between Kumaraswamy and Beta distributions for stick-breaking
        # This is crucial for the DP construction
        kumar_beta_kl = self.prior.compute_kl_loss(params, params['alpha'], h)
        losses['kumar_beta_kl'] = kumar_beta_kl

        # 4. Compute effective number of components and penalty for unused components
        K_eff = self.prior.get_effective_components(params['pi'])

        # Convert K_eff to integer for each batch element
        unused_penalties = []
        for i in range(params['pi'].shape[0]):  # Loop over batch
            k = int(K_eff[i].item())  # Convert to integer
            # Calculate penalty for this batch element
            if k < params['pi'].shape[1]:  # Only if there are unused components
               unused_penalties.append(torch.sum(params['pi'][i, k:]))
            else:
               unused_penalties.append(torch.tensor(0.0, device=params['pi'].device))

        unused_penalty = torch.stack(unused_penalties).mean()  # Average over batch


        losses['unused_penalty'] = unused_penalty
        losses['K_eff'] = K_eff.float().mean()


        # 6. Prior over mixture weights (stick-breaking construction)
        # This ensures the weights sum to 1 and follow the DP construction
        eps = torch.tensor(torch.finfo(torch.float32).eps, device=self.device)
        pi_prior = -torch.sum(params['pi'] * torch.log(params['pi'] + eps), dim=1).mean()
        losses['pi_prior'] = pi_prior

        #compute entropy
        entropy, cluster_stats = self.compute_conditional_entropy(
            params['prior_dist'].mixture_distribution.logits,
            params
        )

        # Add entropy loss and stats to output dictionary
        losses['conditional_entropy'] = entropy
        losses['temperature'] = cluster_stats['temperature']
        losses['active_clusters'] = cluster_stats['active_clusters']

        # Total ELBO loss
        # Note: The negative signs before kumar_beta_kl and alpha_prior are because
        # we want to maximize the ELBO (minimize negative ELBO)
        total_loss = (
            recon_loss +
            beta * kl_z +
            beta * kumar_beta_kl +
            unused_penalty +
            beta * pi_prior+
            entropy_weight * entropy # Add weighted entropy to total loss
        )
        # Verify all loss components are scalars
        for k, v in losses.items():
            if torch.is_tensor(v):
               assert v.dim() == 0, f"Loss component {k} must be a scalar"

        losses['loss'] = total_loss

        return losses


    def discriminator_step(
        self,
        real_images: torch.Tensor, #[B, T, C, H, W]
        fake_images: torch.Tensor, #[B, T, C, H, W]
        real_latents: torch.Tensor, #[B, T, Z]
        fake_latents: torch.Tensor #[B, T, Z]
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

        # Temporal Latent Discriminator
        real_latent_outputs = self.latent_discriminator(real_latents)
        fake_latent_outputs = self.latent_discriminator(fake_latents.detach())
    
        real_latent_score = real_latent_outputs['final_score']
        fake_latent_score = fake_latent_outputs['final_score']

        latent_gp = self.compute_gradient_penalty(
            self.latent_discriminator,
            real_latents,
            fake_latents.detach(),
            self.device
        )

        latent_disc_loss = (
            torch.mean(fake_latent_score) - torch.mean(real_latent_score) +
            10.0 * latent_gp
        )
        # Temporal consistency losses
        img_consistency_loss = torch.mean(
            fake_img_outputs['per_frame_scores'].std(dim=1)
        )
    
        latent_consistency_loss = torch.mean(
            torch.abs(fake_latent_outputs['consistency_score'])
        )
        result = {
            'img_disc_loss': img_disc_loss,
            'latent_disc_loss': latent_disc_loss,
            'img_gp': img_gp,
            'latent_gp': latent_gp,
            'real_img_score': real_img_score.mean(),
            'fake_img_score': fake_img_score.mean(),
            'real_latent_score': real_latent_score.mean(),
            'fake_latent_score': fake_latent_score.mean(),
            'img_temporal_score': real_img_outputs['temporal_score'].mean(),
            'latent_temporal_score': real_latent_outputs['temporal_score'].mean(),
            'img_consistency_loss': img_consistency_loss,
            'latent_consistency_loss': latent_consistency_loss
        }
    
        # Only perform optimization if optimizers exist and we're configured to use them
        if self.has_optimizers and hasattr(self, 'img_disc_optimizer') and hasattr(self, 'latent_disc_optimizer'):
            # Update image discriminator
            self.img_disc_optimizer.zero_grad()
            img_disc_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.image_discriminator.parameters(), self._grad_clip)
            self.img_disc_optimizer.step()

            # Update latent discriminator
            self.latent_disc_optimizer.zero_grad()
            latent_disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.latent_discriminator.parameters(), self._grad_clip)
            self.latent_disc_optimizer.step()
        
            # Step the discriminator schedulers if available
            if hasattr(self, 'img_disc_scheduler') and hasattr(self, 'latent_disc_scheduler'):
                self.img_disc_scheduler.step(img_disc_loss.item())
                self.latent_disc_scheduler.step(latent_disc_loss.item())
                # Add learning rates to result if available
                result['img_disc_lr'] = self.img_disc_scheduler.get_last_lr()[0]
                result['latent_disc_lr'] = self.latent_disc_scheduler.get_last_lr()[0]
        
    
        return result
    
    def compute_feature_matching_loss(self, real_features, fake_features):
        """
        Compute feature matching loss between real and fake features
        
        Uses both mean and standard deviation matching for better stability
        """
        feat_match_loss = 0.0
        num_features = len(real_features)
        
        for i in range(num_features):
            real_feat = real_features[i]
            fake_feat = fake_features[i]
            
            # For CNN features: [B, T, C, H, W]
            if real_feat.dim() == 5:
                # Match mean and std across spatial dimensions
                real_mean = real_feat.mean(dim=[3, 4])  # [B, T, C]
                fake_mean = fake_feat.mean(dim=[3, 4])
                real_std = real_feat.std(dim=[3, 4])
                fake_std = fake_feat.std(dim=[3, 4])
                
                feat_match_loss += F.l1_loss(fake_mean, real_mean.detach())
                feat_match_loss += F.l1_loss(fake_std, real_std.detach())
                
            # For feature vectors: [B, T, D]
            elif real_feat.dim() == 3:
                # Match mean and std across feature dimension
                real_mean = real_feat.mean(dim=2)  # [B, T]
                fake_mean = fake_feat.mean(dim=2)
                real_std = real_feat.std(dim=2)
                fake_std = fake_feat.std(dim=2)
                
                feat_match_loss += F.l1_loss(fake_mean, real_mean.detach())
                feat_match_loss += F.l1_loss(fake_std, real_std.detach())

        return  feat_match_loss /  (2*num_features)  # Normalize by number of features


    def compute_adversarial_losses(
        self,
        x: torch.Tensor, #[B, T, C, H, W  ]
        reconstruction: torch.Tensor, #[B, T, C, H, W]
        z: torch.Tensor, #[B, T, Z]
        prior_z: torch.Tensor #[B, T, Z]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adversarial losses for both image and latent space
        """
        # Image sequence adversarial loss
        fake_img_outputs = self.image_discriminator(reconstruction, return_features=True)
        img_adv_loss = -torch.mean(fake_img_outputs['final_score'])
        #reward for generating temporally consistent images
        temporal_bonus = -fake_img_outputs['temporal_score'].mean()
        
        # Per-frame quality bonus
        # Encourage consistent quality across all frames
        per_frame_scores = fake_img_outputs['per_frame_scores'].squeeze(-1)  # [B, T]
        frame_quality_bonus = -per_frame_scores.mean()
        
        # Variance penalty to encourage temporal smoothness
        frame_variance_penalty = per_frame_scores.var(dim=1).mean()
        
        # Combined image adversarial loss
        total_img_adv = (
            img_adv_loss + 
            0.2 * temporal_bonus + 
            0.1 * frame_quality_bonus + 
            0.1 * frame_variance_penalty
        )
        
        # === Latent Adversarial Loss ===
        fake_latent_outputs = self.latent_discriminator(z)
        
        # Main adversarial loss
        latent_adv_loss = -fake_latent_outputs['final_score'].mean()
        
        # Temporal consistency bonus
        consistency_bonus = -fake_latent_outputs['consistency_score'].mean()
        
        # Sequence quality bonus
        sequence_bonus = -fake_latent_outputs['sequence_score'].mean()
        
        # Combined latent adversarial loss
        total_latent_adv = (
            latent_adv_loss + 
            0.2 * consistency_bonus + 
            0.1 * sequence_bonus
        )
        
        # === Feature Matching Loss ===
        # This helps stabilize training
        real_features = self.image_discriminator(x, return_features=True)['frame_features']
        fake_features = fake_img_outputs['frame_features']
        
        # L1 loss between feature statistics
        feature_match_loss = self.compute_feature_matching_loss(real_features, fake_features) 
        
        return total_img_adv, total_latent_adv, feature_match_loss    

    def generator_step(
        self,
        x: torch.Tensor,
        beta: float = 1.0,
        lambda_img: float = 0.1,
        lambda_latent: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Training step for generator (VAE + Prior)
        """
        # Forward pass
        reconstruction, params = self.forward(x)

        # Sample from prior for latent discriminator
        prior_z = torch.randn_like(params['z'])

        # Compute standard DPGMM-VAE losses
        vae_losses = self.compute_loss(x, reconstruction, params, beta)

        # Compute adversarial losses
        img_adv_loss, latent_adv_loss, feature_match_loss = self.compute_adversarial_losses(
            x, reconstruction, params['z'], prior_z
        )
        # Ensure all loss components are scalars
        assert vae_losses['loss'].dim() == 0, "VAE loss must be a scalar"
        assert img_adv_loss.dim() == 0, "Image adversarial loss must be a scalar"
        assert latent_adv_loss.dim() == 0, "Latent adversarial loss must be a scalar"

        # Combined loss
        total_loss = (
            vae_losses['loss'] +  # Original ELBO and DPGMM terms
            lambda_img * img_adv_loss +  # Image adversarial loss
            lambda_latent * latent_adv_loss + # Latent adversarial loss
            feature_match_loss  # Feature matching loss
        )
        result = {
                **vae_losses,
                'img_adv_loss': img_adv_loss,
                'latent_adv_loss': latent_adv_loss,
                'total_loss': total_loss,
                'feature_match_loss': feature_match_loss
            }
        if self.has_optimizers and hasattr(self, 'gen_optimizer'):
            # Optimize generator
            torch.cuda.empty_cache()

            self.gen_optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), self._grad_clip)
            self.gen_optimizer.step()
        
            if hasattr(self, 'gen_scheduler'):
                self.gen_scheduler.step()
                result['gen_lr'] = self.gen_scheduler.get_last_lr()[0]
        
            # Convert tensor values to items for return value when optimization is performed
            #for key in list(result.keys()):
            #    if torch.is_tensor(result[key]):
            #        result[key] = result[key].item()
    
        return result       
    def compute_global_context(self, observations):
        """
        Process entire sequence through Perceiver for global context
        
        Args:
            observations: [batch_size, seq_len, channels, height, width]
        
        Returns:
            context: [batch_size, seq_len, context_dim]
            cross_attention: [batch_size, seq_len, num_heads, spatial_dim]
        """
        batch_size, seq_len = observations.shape[:2]
        
        # Flatten observations for perceiver
        obs_flat = observations.reshape(batch_size * seq_len, -1, self.input_channels)
        
        # Process through perceiver
        perceiver_output = self.perceiver_model({'image': obs_flat}, is_training=True)
        latents = perceiver_output[self.out_keys.LATENTS]['image']
        #Perceiver latents shape: torch.Size([50, 2048, 128]) [batch_size* seq_len, ?, ?]
        # Extract reconstruction for perceiver loss
        perceiver_recon = perceiver_output[self.out_keys.INPUT_RECONSTRUCTION]['image']
        
        # Project to context dimension
        
        context, weights = self.perceiver_projection(latents)
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
        local_obs = observations[:, window_start:window_end]
        local_obs_flat = local_obs.reshape(batch_size * (window_end - window_start), -1, 1)
        
        # Process through perceiver
        perceiver_output = self.perceiver_model({'image': local_obs_flat}, is_training=True)
        latents = perceiver_output[self.out_keys.LATENTS]['image']
        
        # Average pool over the window and project
        latents = latents.reshape(batch_size, window_end - window_start, -1)
        latents = latents.mean(dim=1, keepdim=True).transpose(1, 2)
        context = self.perceiver_projection(latents).squeeze()
        
        return context
    
 
    
    def self_model_step(self, h_t, z_t, attention_t, a_t, c_next=None):
        """
        Predict next internal states using self-model
        
        This implements the predictive components:
        - p(h_{t+1}|h_t, z_t, a_t) 
        - p(z_{t+1}|h_{t+1}, c_{t+1})
        - p(A_{t+1}|h_{t+1}, z_{t+1}, A_t)
        
        Args:
            h_t: Current hidden state [batch_size, hidden_dim]
            z_t: Current latent state [batch_size, latent_dim] - NOW PROPERLY USED
            attention_t: Current attention state [batch_size, attention_dim]
            a_t: Current action [batch_size, action_dim]
            c_next: Next context (for z prediction) [batch_size, context_dim]
        
        Returns:
            Dictionary with predicted distributions for next timestep
        """
        # 1. Predict h_{t+1} ~ p(h_{t+1}|h_t, z_t, a_t)
        # The hidden state evolution depends on current hidden state, latent, and action
        h_input = torch.cat([h_t, z_t, a_t], dim=-1)
        h_params = self.self_model_h(h_input)
        h_mean, h_logvar = torch.chunk(h_params, 2, dim=-1)
        
        # Ensure logvar is in reasonable range
        h_logvar = torch.clamp(h_logvar, min=-10, max=2)
        h_std = torch.exp(0.5 * h_logvar)
        
        # Sample predicted h_{t+1}
        h_pred = h_mean + h_std * torch.randn_like(h_std)
        
        # 2. Predict z_{t+1} ~ p(z_{t+1}|h_{t+1}, c_{t+1})
        # The latent prediction depends on predicted hidden state and context
        if c_next is not None:
            z_input = torch.cat([h_pred, c_next], dim=-1)
        else:
            # If no future context, use zeros or current context
            z_input = torch.cat([h_pred, torch.zeros(h_pred.shape[0], self.context_dim).to(h_pred.device)], dim=-1)
        
        z_params = self.self_model_z(z_input)
        z_mean, z_logvar = torch.chunk(z_params, 2, dim=-1)
        z_logvar = torch.clamp(z_logvar, min=-10, max=2)
        z_std = torch.exp(0.5 * z_logvar)
        
        # Sample predicted z_{t+1}
        z_pred = z_mean + z_std * torch.randn_like(z_std)
        
        # 3. Predict A_{t+1} ~ p(A_{t+1}|h_{t+1}, z_{t+1}, A_t, a_t)
        # Attention evolution depends on predicted states and current attention
        attention_input = torch.cat([h_pred, z_pred, attention_t, a_t], dim=-1)
        attention_params = self.self_model_attention(attention_input)
        attention_mean, attention_logvar = torch.chunk(attention_params, 2, dim=-1)
        attention_logvar = torch.clamp(attention_logvar, min=-10, max=2)
        
        return {
            'h_mean': h_mean, 
            'h_logvar': h_logvar, 
            'h_pred': h_pred,
            'h_std': h_std,
            'z_mean': z_mean, 
            'z_logvar': z_logvar, 
            'z_pred': z_pred,
            'z_std': z_std,
            'attention_mean': attention_mean, 
            'attention_logvar': attention_logvar,
            'distributions': {
                'h_dist': Normal(h_mean, h_std),
                'z_dist': Normal(z_mean, z_std),
                'attention_dist': Normal(attention_mean, torch.exp(0.5 * attention_logvar))
            }
        }

    def _compute_self_model_loss(self, predictions, h_actual, z_actual, attention_actual):
        """
        Compute self-model losses by comparing predictions with actual next states
        
        This computes the negative log-likelihood of actual states under predicted distributions
        """
        losses = {}
        
        # Hidden state prediction loss
        h_dist = predictions['distributions']['h_dist']
        h_nll = -h_dist.log_prob(h_actual).mean()
        losses['h_prediction_loss'] = h_nll
        
        # Latent state prediction loss
        z_dist = predictions['distributions']['z_dist']
        z_nll = -z_dist.log_prob(z_actual).mean()
        losses['z_prediction_loss'] = z_nll
        
        # Attention state prediction loss
        attention_dist = predictions['distributions']['attention_dist']
        attention_nll = -attention_dist.log_prob(attention_actual).mean()
        losses['attention_prediction_loss'] = attention_nll
        
        # Total self-model loss
        losses['total'] = h_nll + z_nll + attention_nll
        
        return losses
        

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
                            lambda_schema: float = 0.1,
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
        
        # 1. Forward pass and compute all VAE losses
        vae_losses, outputs = self.compute_total_loss(
            observations, actions, beta, entropy_weight, 
            lambda_img, lambda_pred, lambda_att, lambda_schema
        )
        
        # 2. Discriminator training
        
        disc_losses_list = []
        for _ in range(n_critic):
            
            real_latents = torch.randn_like(outputs['latents'])
            
            disc_loss = self.discriminator_step(
                observations, outputs['reconstructions'], real_latents, outputs['latents']
            )
            disc_losses_list.append(disc_loss)
        
        # Average discriminator losses
        avg_disc_losses = {
            k: sum(d[k] for d in disc_losses_list) / len(disc_losses_list)
            for k in disc_losses_list[0].keys()
        }
        
        # 4. Generator training with adversarial losses
        # Get fresh discriminator outputs (no detach)
        img_adv_loss, latent_adv_loss, feat_match_loss = self.compute_adversarial_losses(
            observations,
            outputs['reconstructions'],
            outputs['latents'],
            real_latents
        )
        
        # 5. Total generator loss
        total_gen_loss = (
            vae_losses['total_vae_loss'] +
            lambda_img * img_adv_loss +
            latent_adv_loss +      # Reduced weight for latent adversarial
            feat_match_loss        # Feature matching
        )
        
        
        
        # 6. Optimize generator
        if self.has_optimizers:
            self.gen_optimizer.zero_grad()
            total_gen_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self._grad_clip)
            self.gen_optimizer.step()
            
            # Update temperature
            if hasattr(self, 'current_epoch'):
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
            'latent_adv_loss': latent_adv_loss.item(),
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
        # Sample hidden representation
        h = torch.randn(num_samples, self.encoder.hlayer.shape[1], device=self.device)

        # Get prior distribution
        prior_dist, _ = self.prior(h)

        # Sample latents
        z = prior_dist.sample()

        # Decode
        return self.decoder(z)

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
               'reconstruction_error': F.mse_loss(reconstruction, x, reduction='mean').item(),
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
        
        # Transform through prior network
        h_prior = self.prior_net(h_context)
        
        # Get DPGMM prior distribution
        prior_dist, prior_params = self.prior(h_prior)
        
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
            z_0, z_mean_0, z_logvar_0 = self.encoder(initial_obs)
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
            attention_features = self.attention_prior_posterior.attention_encoder(
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
            attention_features = torch.cat([attention_features, weighted_visual_features, attention_coords], dim=-1)
            # Feature extraction
            phi_z = self.phi_z(z_0)
            phi_c = self.phi_c(c_0)
            phi_attention = self.phi_attention(attention_features)
            phi_attention_mean, phi_attention_logvar = torch.chunk(phi_attention, 2, dim=-1)
            phi_attention_std = torch.exp(0.5 * phi_attention_logvar.clamp(min=-10, max=2))
            phi_attention = phi_attention_mean + phi_attention_std * torch.randn_like(phi_attention_std).to(device)
            attention_uncertainties.append(phi_attention_std.mean(dim=-1))
            phi_a = self.phi_a(initial_action)
            
            # Update LSTM
            rnn_input = torch.cat([phi_z, phi_c, phi_attention, phi_a], dim=-1)
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
                attention_features = self.attention_prior_posterior.attention_encoder(
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
                    attention_features,
                    approx_weighted_features,
                    attention_coords
                    ], dim=-1)
                # Update VRNN state
                phi_z = self.phi_z(z_next)
                phi_c = self.phi_c(current_c_t)
                phi_attention = self.phi_attention(attention_features)
                phi_attention_mean, phi_attention_logvar = torch.chunk(phi_attention, 2, dim=-1)
                phi_attention_std = torch.exp(0.5 * phi_attention_logvar.clamp(min=-10, max=2))
                attention_uncertainties.append(phi_attention_std.mean(dim=-1))
                phi_attention = phi_attention_mean + phi_attention_std * torch.randn_like(phi_attention_std).to(device)
                phi_a = self.phi_a(current_action)
                
                rnn_input = torch.cat([phi_z, phi_c, phi_attention, phi_a], dim=-1)
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
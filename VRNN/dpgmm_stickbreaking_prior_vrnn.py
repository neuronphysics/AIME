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
    ImageDiscriminator, LatentDiscriminator,
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
    Amortized variational posterior for Gamma distribution
    Maps encoder hidden states to Gamma parameters
    """
    def __init__(self, hidden_dim: int, device: torch.device):
        super().__init__()
        # Network for generating Gamma parameters from hidden states
        # Give unique names to each layer
        self.param_net = nn.Sequential(OrderedDict([
            ('gamma_fc', nn.Linear(hidden_dim, hidden_dim)),
            ('gamma_ln', nn.LayerNorm(hidden_dim)),
            ('gamma_relu', nn.LeakyReLU()),
            ('gamma_out', nn.Linear(hidden_dim, 2)),
            ('gamma_softplus', nn.Softplus())
        ]))
        self.device = device
        self.to(device)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate Gamma parameters from hidden representation
        """
        eps = torch.finfo(torch.float32).eps
        params = self.param_net(h) + eps
        concentration, rate = params.split(1, dim=-1)
        return concentration.squeeze(-1), rate.squeeze(-1)

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
            ('kumar_a_fc', nn.Linear(hidden_dim, hidden_dim)),
            ('kumar_a_ln',nn.LayerNorm(hidden_dim)),
            ('kumar_a_relu', nn.LeakyReLU()),
            ('kumar_a_out', nn.Linear(hidden_dim , self.K - 1)),
            ('kumar_a_softplus', nn.Softplus()), # Ensure positive parameters
            ('kumar_a_eps', AddEpsilon(self.eps))
        ]))

        self.net_b = nn.Sequential(OrderedDict([
            ('kumar_b_fc', nn.Linear(hidden_dim, hidden_dim)),
            ('kumar_b_ln',nn.LayerNorm(hidden_dim)),
            ('kumar_b_relu', nn.LeakyReLU()),
            ('kumar_b_out', nn.Linear(hidden_dim , self.K - 1)),
            ('kumar_b_softplus', nn.Softplus()), # Ensure positive parameters
            ('kumar_b_eps', AddEpsilon(self.eps))
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
        min_val =self.eps*10
        max_val = 1e3
        a = torch.clamp(self.net_a(h) , min=min_val, max=max_val)
        b = torch.clamp(self.net_b(h) , min=min_val, max=max_val)
        if torch.isnan(a).any() or torch.isnan(b).any():
            print(f"Warning: NaN in Kumaraswamy parameters: h_range= ({h.min()}, {h.max()})a_range=({a.min()}, {a.max()}), b_range=({b.min()}, {b.max()})")
            mean_a = torch.nanmean(a)  # Compute mean with ignoring NaNs
            a = torch.where(torch.isnan(a), mean_a, a)
            mean_b = torch.nanmean(b)  
            b = torch.where(torch.isnan(b), mean_b, b)

        return a, b

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
                           a: torch.Tensor, 
                           b: torch.Tensor, 
                           max_k: int, 
                           use_rand_perm: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample from Kumaraswamy distribution
        U~Uniform(0,1)
        X=(1-(1-U)**(1/b))**(1/a)
        https://arxiv.org/pdf/2410.00660
        """
        a = a.to(a.device)
        b = b.to(b.device)
        eps = torch.finfo(torch.float32).eps
        tiny = torch.finfo(torch.float16).tiny
        # Clamp parameters for stability
        a = torch.clamp(a, min=eps)
        b = torch.clamp(b, min=eps)
    
        check_tensor(a, "kumar_a")
        check_tensor(b, "kumar_b")
        
        # Optional random permutation
        if use_rand_perm:
            # Generate random permutation indices
            perm = torch.argsort(torch.rand_like(a, device= a.device), dim=-1)
            # Apply permutation
            perm = perm.view(-1, max_k-1)
            
            a = a.gather( dim=1, index=perm) # [batch_size, K-1]
            b = b.gather( dim=1, index=perm) # [batch_size, K-1]
            # Generate full permutation for output
            
        else:
            perm = None
        
        # Sample uniformly with numerical stability
        u = torch.rand_like(a).clamp(tiny, 1.0 - eps).to(a.device)

        # Compute in log space for stability
        # More stable than log(1-u)
        log_1minus_v = torch.clamp(torch.log1p(-u) / b, max=1.0-eps)
    
        # Use log1p for improved numerical stability
        log_v = torch.log1p(-torch.exp(log_1minus_v)) / a
    
        # Convert from log space with clamping
        v = torch.exp(log_v).clamp( min=tiny, max=1.0-eps)
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
        pi =prune_small_components(pi, threshold=1e-5)

        # Apply inverse permutation if needed
        if perm is not None:
            # Create full permutation including last component
            full_perm = torch.cat([perm, torch.full((batch_size, 1), max_k-1, device=perm.device)], dim=1)
            inv_perm = torch.argsort(full_perm, dim=1)  # [batch_size, K]
            pi = torch.gather(pi, 1, inv_perm)

        # Prune small components for numerical stability????
        
        return pi
    
    def forward(self, h: torch.Tensor, n_samples: int = 10, use_rand_perm: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
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
        kumar_a, kumar_b = self.kumar_net(h)
        
        # Sample v from Kumaraswamy for each alpha sample
        v, perm = self.sample_kumaraswamy(kumar_a, kumar_b, self.max_K, use_rand_perm)  # [n_samples, batch, K-1]


        # Initialize mixing proportions
        pi = self.compute_stick_breaking_proportions(v, self.max_K, perm)
        assert pi.shape[-1] == self.max_K  # Add this check
        
        
        return pi, {
            'kumar_a': kumar_a,
            'kumar_b': kumar_b,
            'v': v.squeeze(0) if n_samples == 1 else v,
            'perm':perm
        }


    def concentration_prior(self, h: torch.Tensor, n_samples: int = 10) -> torch.Tensor:
        """
        Compute the log prior probability of the concentration parameter alpha.
        This is part of the hierarchical model where alpha ~ Gamma(gamma_a, gamma_b).

        Args:
            n_samples: Number of samples to use for Monte Carlo estimation

        Returns:
            Log probability of the concentration parameter under the prior
        """
        # Get multiple samples from the variational posterior
        alpha_samples = self.alpha_posterior.sample(h, n_samples)  # [n_samples]

        # Compute log probability under Gamma prior for each sample
        prior_dist = Gamma(self.gamma_a, self.gamma_b)
        log_probs = prior_dist.log_prob(alpha_samples)

        # Average over samples for more stable estimation
        return log_probs.mean()  # Average over all samples


    @staticmethod
    def compute_kumar2beta_kl( a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, n_approx: int, eps: float=torch.finfo(torch.float32).eps) -> torch.Tensor:
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

        a = torch.clamp(a, min=eps)
        b = torch.clamp(b, min=eps)
        alpha = torch.clamp(alpha, min=eps)
        beta = torch.clamp(beta, min=eps)
 
        ab = torch.mul(a, b)
        a_inv = torch.reciprocal(a + eps)
        b_inv = torch.reciprocal(b + eps)

        # Taylor expansion for E[log(1-v)]
                
        log_taylor = torch.logsumexp(torch.stack([beta_fn(m *a_inv, b) - torch.log(m + a * b) for m in range(1, n_approx + 1)], dim=-1), dim=-1)
        kl = torch.mul(torch.mul(beta - 1, b), torch.exp(log_taylor))
        # Add remaining terms
        psi_b = torch.digamma(b + eps)
        term1 = torch.mul(torch.div(a - alpha, a + eps), -EULER_GAMMA - psi_b - b_inv)
        term2 = torch.log(ab + eps) + beta_fn(alpha, beta)
        term2 += torch.div(-(b - 1), b + eps)
        kl += term1 + term2
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
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim * max_components)
        )
        self.to(device)

    def compute_kl_divergence_mc(
                                 self,
                                 q_dist: torch.distributions.Distribution,
                                 prior_params: Dict[str, torch.Tensor],
                                 n_samples: int = 100
                                ) -> torch.Tensor:
        """
        Compute KL(q||p) between neural network posterior q and DPGMM prior p
        using Monte Carlo estimation.

        Args:
            q_dist: Posterior distribution (from encoder)
            prior_params: Dictionary containing DPGMM parameters
            n_samples: Number of MC samples

        Returns:
            Estimated KL divergence
        """
        # Sample from posterior q(z|x)
        batch_size = prior_params['pi'].shape[0]
        z_samples = q_dist.rsample((n_samples,))  # [n_samples, batch_size, latent_dim]

        # Compute log q(z|x)
        log_q = q_dist.log_prob(z_samples)  # [n_samples, batch_size]
        if log_q.dim() > 2:
           log_q = log_q.sum(-1)  # Sum over latent dimensions if needed

        # Prepare samples and parameters for log_p computation
        z_flat = z_samples.view(-1, self.latent_dim)  # [n_samples * batch_size, latent_dim]
        # Properly repeat the parameters for each MC sample
        
        pi_rep = prior_params['pi'].unsqueeze(0).repeat(n_samples, 1, 1)  # [n_samples, batch_size, K]
        pi_rep = pi_rep.reshape(-1, self.max_K)
        
        means_exp = prior_params['means'].unsqueeze(0).expand(n_samples, -1, -1, -1)  # [n_samples, batch_size, K, latent_dim]
        means_rep = means_exp.reshape( -1, self.max_K, self.latent_dim)  # [n_samples * batch_size, K, latent_dim]
        
        vars_rep = torch.exp(prior_params['log_vars'])
        covs_exp = torch.diag_embed(vars_rep).unsqueeze(0).expand(n_samples, -1, -1, -1, -1)  # [n_samples, batch_size, K, latent_dim, latent_dim]
        covs_rep = covs_exp.reshape( -1, self.max_K, self.latent_dim, self.latent_dim)  # [n_samples * batch_size, K, latent_dim, latent_dim]
        
        # Now all parameters have shape [n_samples * batch_size, ...]
        log_p = dp_gmm_log_prob(z_flat, pi_rep, means_rep, covs_rep)
    
        # Reshape back to [n_samples, batch_size]
        log_p = log_p.view(n_samples, batch_size)

        # KL = E_q[log q(z) - log p(z)]
        kl = (log_q - log_p).mean(0)  # Average over MC samples
        return kl.mean()  # Average over batch

    def compute_kl_loss(self, params: Dict, alpha: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between Kumaraswamy and Beta distributions
        """
        total_kl = 0.0
        batch_size = params['kumar_a'].shape[0]
    
        # Reshape alpha to [batch_size, n_samples]

        
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
        alpha_kl = self.stick_breaking.compute_gamma2gamma_kl(h)#TODO
        total_kl = total_kl.mean()+ alpha_kl
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


def log_gaussian(x: torch.Tensor, mu: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
    """
    Optimized computation of log N(x | mu, Sigma) for batched data.
    Args:
        x: [B, K, D] tensor of points
        mu: [B, K, D] tensor of means
        Sigma: [B, K, D, D] tensor of covariance matrices
    Returns:
        [B, K] tensor of log probabilities
    """
    B, K, D = x.shape
    
    # Make tensors contiguous and reshape
    x_flat = x.reshape(-1, D)  # [B*K, D]
    mu_flat = mu.reshape(-1, D)  # [B*K, D]
    Sigma_flat = Sigma.reshape(-1, D, D) # [B*K, D, D]
    
    # Compute difference vectors
    diff = x_flat - mu_flat # [B*K, D]
    
    # First attempt with small epsilon
    eps = torch.finfo(x.dtype).eps
    try:
        # Cholesky decomposition: Sigma = LL^T
        L = torch.linalg.cholesky(
            Sigma_flat + eps * torch.eye(D, device=Sigma_flat.device)[None, :, :]
        )
    except RuntimeError:
        # Fallback with larger epsilon if first attempt fails
        L = torch.linalg.cholesky(
            Sigma_flat + 1e-3 * torch.eye(D, device=Sigma_flat.device)[None, :, :]
        )
    
    # Solve system and compute quadratic term
    y = torch.cholesky_solve(diff.unsqueeze(-1), L)
    quad_term = torch.sum(diff * y.squeeze(-1), dim=-1)
    
    # Compute log determinant from Cholesky factor
    logdet = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)
    
    # Combine terms for final log probability
    log_prob = -0.5 * (quad_term + D * math.log(2 * math.pi) + logdet)
    
    return log_prob.view(B, K)

def dp_gmm_log_prob(x: torch.Tensor, pis: torch.Tensor, mus: torch.Tensor,
                    Sigmas: torch.Tensor) -> torch.Tensor:
    """
    Compute log probability under a Gaussian mixture model.

    Args:
        x: [batch_size, D] tensor of points
        pis: [batch_size, K] tensor of mixture weights
        mus: [batch_size, K, D] tensor of means
        Sigmas: [batch_size, K, D, D] tensor of covariance matrices

    Returns:
        [batch_size] tensor of log probabilities
    """
    batch_size, K, D = mus.shape

    # Expand x for broadcasting with components
    x_expanded = x.unsqueeze(1).expand(-1, K, -1)  # [batch_size, K, D]
    # Ensure inputs have correct shape
    

    # Compute log probabilities for all components
    log_gaussians = log_gaussian(x_expanded, mus, Sigmas)  # [batch_size, K]
    log_pis = torch.log(pis + torch.finfo(torch.float32).eps)

    # Use log-sum-exp trick for numerical stability
    log_components = log_pis + log_gaussians  # [batch_size, K]
    
    max_log_comp = torch.max(log_components, dim=1, keepdim=True)[0]
    return max_log_comp.squeeze(1) + torch.log(
        torch.sum(torch.exp(log_components - max_log_comp), dim=1)
    )

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
        context_dim: int = 128
    ):
        super().__init__()
        
        # Posterior (bottom-up, stimulus-driven attention)
        self.posterior_net = AttentionPosterior(
            image_size, attention_resolution, 
            hidden_dim, context_dim
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
            nn.Linear(32, hidden_dim // 2)
        )
    
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
        
        total_loss = 0.0
        for t in range(1, len(attention_sequence)):
            curr_att = attention_sequence[t]
            prev_att = attention_sequence[t-1]
            
            # Compute actual movement
            prev_com = self._center_of_mass(prev_att)
            curr_com = self._center_of_mass(curr_att)
            actual_dx = curr_com[0] - prev_com[0]
            actual_dy = curr_com[1] - prev_com[1]
            
            # Compare with predicted movement
            pred_dx, pred_dy = predicted_movements[t-1]
            movement_error = (
                (pred_dx.mean(dim=[1,2]) - actual_dx)**2 + 
                (pred_dy.mean(dim=[1,2]) - actual_dy)**2
            ).mean()
            
            total_loss += movement_error
        
        return total_loss / (len(attention_sequence) - 1)
    
    def _center_of_mass(self, attention_map):
        """Compute attention center of mass for movement tracking"""
        batch_size, H, W = attention_map.shape
        
        # Create coordinate grids
        y_coords = torch.arange(H, device=attention_map.device).float()
        x_coords = torch.arange(W, device=attention_map.device).float()
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Compute weighted average
        total_mass = attention_map.sum(dim=[1, 2], keepdim=True) + 1e-8
        y_com = (attention_map * y_grid).sum(dim=[1, 2]) / total_mass.squeeze()
        x_com = (attention_map * x_grid).sum(dim=[1, 2]) / total_mass.squeeze()
        
        return torch.stack([x_com, y_com], dim=-1)


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
        use_actnorm: bool= False,
        input_channels: int = 3,  # Number of input channels (e.g., RGB images)
        learning_rate: float = 4e-5,
        grad_clip:float =1.0,
        prior_alpha: float = 1.0,  # Add these parameters
        prior_beta: float = 1.0,
        weight_decay: float = 0.00001,
        use_orthogonal: bool = True,  # Use orthogonal initialization for LSTM,
        number_lstm_layer: int = 2,  # Number of LSTM layers
        HiP_type: str = 'Mini',  # Type of perceiver model
        attention_temperature: float = 1.0,
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
        self.attention_temperature = attention_temperature
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
        self._init_discriminators(img_disc_channels, img_disc_layers, 
                                 latent_disc_layers, use_actnorm)

        

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
    
        # Get variant configuration
        variant_config = perceiver.VARIANTS[self.HiP_type]
    
        # Extract key dimensions from the processor block (middle block)
        processor_idx = len(variant_config['num_groups']) // 2
        perceiver_latent_dim = variant_config['num_z_channels'][processor_idx]
    
        # Initialize perceiver
        mock_input = self.generate_mock_input()
        self.perceiver_model = generate_model('HiPClassBottleneck', self.HiP_type, mock_input)
        self.perceiver_model.to(self.device)
        self.out_keys = perceiver_helpers.ModelOutputKeys
    
        # Architecture-aware projection
        self.perceiver_projection = nn.Sequential(
                                                  nn.Conv1d(perceiver_latent_dim, perceiver_latent_dim // 2, kernel_size=1),
                                                  nn.ReLU(),
                                                  nn.Conv1d(perceiver_latent_dim // 2, 1, kernel_size=1),
                                                  nn.AdaptiveAvgPool1d(self.context_dim)
                                                  ).to(self.device)
    
        # Attention extractor remains the same
        self.attention_extractor = nn.Sequential(
                                                 nn.Linear(self.context_dim, self.attention_dim * 2),
                                                 nn.LayerNorm(self.attention_dim * 2),
                                                 nn.SiLU(),
                                                 nn.Linear(self.attention_dim * 2, self.attention_dim)
                                                 ).to(self.device)
    
    def _init_discriminators(self, img_disc_channels, img_disc_layers, latent_disc_layers, use_actnorm: bool):        
        # Initialize discriminators
        self.image_discriminator = ImageDiscriminator(
            input_nc=self.input_channels,  # For RGB images
            ndf=img_disc_channels,
            n_layers=img_disc_layers,
            use_actnorm=use_actnorm
        ).to(self.device)

        # Latent discriminator operates on flattened latent vectors
        self.latent_discriminator = LatentDiscriminator(
            input_dims=self.latent_dim,
            num_layers=latent_disc_layers,
            norm_type='layer'
        ).to(self.device)

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
            nn.Linear(self.hidden_dim * 2, self.hidden_dim + self.context_dim)
        )
        
        # VRNN recurrence: h_t = f(h_{t-1}, z_t, c_t, A_t, a_t)
        self._rnn = LSTMLayer(
            input_size=self.hidden_dim * 4,  # phi_z + phi_c + phi_A + phi_a
            hidden_size=self.hidden_dim,
            n_lstm_layers= number_lstm_layer,
            use_orthogonal=use_orthogonal
        )
        
        # Initialize hidden states
        self.h0 = nn.Parameter(torch.zeros(2, 1, self.hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(2, 1, self.hidden_dim))
    
    def _init_attention_schema(self, attention_resolution: int = 21):
        """Initialize attention schema components"""
        # Compute attention state from h, c, z
        self.attention_prior_posterior = AttentionSchema(
                                                        image_size=self.image_size,
                                                        attention_resolution=attention_resolution,  # 84/4
                                                        hidden_dim=self.hidden_dim,
                                                        latent_dim=self.latent_dim,
                                                        context_dim=self.context_dim
                                                        )
        # Feature extractor for attention
        self.phi_attention = LinearResidual(self.attention_dim, self.hidden_dim)
    
    def _init_self_model(self):
        """Initialize comprehensive self-modeling components"""
        # Hidden state predictor: p(h_{t+1}|h_t, z_t, a_t)
        self.self_model_h = nn.Sequential(
            nn.Linear(self.hidden_dim + self.action_dim+ self.latent_dim,self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.SiLU(),  
            LinearResidual(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)  # mean and logvar
        )
        
        # Latent predictor: p(z_{t+1}|h_{t+1}, c_{t+1})
        self.self_model_z = nn.Sequential(
            nn.Linear(self.hidden_dim + self.context_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.latent_dim * 2)  # mean and logvar
        )
        
        # Attention predictor: p(A_{t+1}|h_{t+1}, z_{t+1}, A_t, a_t)
        self.self_model_attention = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim + self.attention_dim + self.action_dim,
                     self.attention_dim * 2),
            nn.LayerNorm(self.attention_dim * 2),
            nn.SiLU(),
            nn.Linear(self.attention_dim * 2, self.attention_dim * 2)  # mean and logvar
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
            betas=(0.9, 0.999),
            eps=1e-8
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
            grad_norm = torch.nn.utils.clip_grad_norm_(params, self._grad_clip)
        
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, self._grad_clip)
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
            'alpha_prior_losses': [],
            'cluster_entropies': [],
            'one_step_h_prediction_loss': [],
            'one_step_z_prediction_loss': [],
            'one_step_att_prediction_loss': [],
            'attention_kl_losses': [],
            'self_model_kl_losses': [],
            'unused_penalties': [],  
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
            attention_map = self.attention_prior_posterior.posterior_net(
                o_t, h[-1], c_t
            )
            outputs['attention_maps'].append(attention_map)
            
            # Extract attention features
            attention_features = self.attention_prior_posterior.attention_encoder(
                attention_map.unsqueeze(1)  # Add channel dimension
            )
            attention_state = self.phi_attention(attention_features)
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
            q_dist = Normal(z_mean_t, torch.exp(0.5 * z_logvar_t))
            kl_z = self.prior.compute_kl_divergence_mc(q_dist, prior_params, n_samples=10)
            outputs['kl_latents'].append(kl_z)
            
            # 2. KL for stick-breaking (Kumaraswamy vs Beta)
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
            # TODO:Alpha prior? What is this?
            alpha_prior = self.prior.stick_breaking.concentration_prior(h_prior, n_samples=10)
            outputs['alpha_prior_losses'].append(-alpha_prior)  # Negative for ELBO

            # 5. Cluster entropy
            pi_t = prior_params['pi']
            entropy_t = -torch.sum(pi_t * torch.log(pi_t + torch.finfo(torch.float32).eps), dim=-1)
            outputs['cluster_entropies'].append(entropy_t)

            # 3. Attention dynamics KL
            if t > 0  :
                # Prior attention prediction
                prior_attention, _ = self.attention_prior_posterior.prior_net(
                    prev_attention, h[-1], z_t
                )
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
                prev_pred = outputs['self_predictions'][-1]
                
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
                att_pred_loss = -att_pred_dist.log_prob(attention_features).mean()
                outputs['one_step_att_prediction_loss'].append(att_pred_loss)

            # === Self-Model Predictions ===
            # TODO:6. Future prediction bonuses (need t and t+tau)? Does this predict next hidden state? 
            if t > 0 and t < seq_len-1:
                # Get next context if available
                c_next = context_sequence[:, t+1] if t+1 < seq_len and self.use_global_context else c_t
                
                # Make self-model predictions
                self_pred = self.self_model_step(
                    h[-1], 
                    outputs['latents'][-1],
                    outputs['attention_states'][-1], 
                    actions[:, t],
                    c_next
                )
                outputs['self_predictions'].append(self_pred)
                
                # Compute self-model losses
                self_loss = self._compute_self_model_loss(
                    self_pred, h[-1], z_t, attention_state
                )
                outputs['self_model_losses'].append(self_loss)
                # Compute self-model KL losses
                # These compare current states with what was predicted
                h_pred_dist = Normal(self_pred['h_mean'], torch.exp(0.5 * self_pred['h_logvar']))
                h_actual_dist = Normal(h[-1], torch.ones_like(h[-1]) * 0.1)
                h_kl = torch.distributions.kl_divergence(h_actual_dist, h_pred_dist).mean()
                
                z_pred_dist = Normal(self_pred['z_mean'], torch.exp(0.5 * self_pred['z_logvar']))
                z_actual_dist = Normal(z_t, torch.ones_like(z_t) * 0.1)
                z_kl = torch.distributions.kl_divergence(z_actual_dist, z_pred_dist).mean()
                
                attention_pred_dist = Normal(self_pred['attention_mean'], 
                                        torch.exp(0.5 * self_pred['attention_logvar']))
                attention_actual_dist = Normal(attention_state, torch.ones_like(attention_state) * 0.1)
                attention_kl = torch.distributions.kl_divergence(attention_actual_dist, attention_pred_dist).mean()
                
                outputs['self_model_kl_losses'].append({
                    'h_kl': h_kl,
                    'z_kl': z_kl,
                    'attention_kl': attention_kl,
                    'total': h_kl + z_kl + attention_kl
                })
    
            # === VRNN State Update ===
            # Feature extraction
            
            # === Update recurrent state ===
            phi_z_t = self.phi_z(z_t)
            phi_c_t = self.phi_c(c_t)
            phi_attention_t = self.phi_attention(attention_features)
            phi_a_t = self.phi_a(a_t)
            
            rnn_input = torch.cat([phi_z_t, phi_c_t, phi_attention_t, phi_a_t], dim=-1)
            rnn_output, (h, c) = self._rnn(
                rnn_input, h, c, 
                torch.ones(batch_size).to(self.device)
            )
            
            outputs['hidden_states'].append(h[-1])
        
        # === Compute aggregated losses ===
        # Stack temporal sequences
        for key in ['reconstructions', 'latents', 'attention_states', 
                    'hidden_states', 'attention_maps']:
            if outputs[key]:
                outputs[key] = torch.stack(outputs[key], dim=1)
        
        # Average losses over time
        for key in ['reconstruction_losses', 'kl_latents', 'kumaraswamy_kl_losses', 'attention_losses',
                    'alpha_prior_losses', 'cluster_entropies', 'unused_penalties']:
            if outputs[key]:
                outputs[key] = torch.stack(outputs[key]).mean()
        
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
        
        # Attention KL average
        if outputs['attention_kl_losses']:
            outputs['attention_kl'] = torch.stack(outputs['attention_kl_losses']).mean()
        else:
            outputs['attention_kl'] = torch.tensor(0.0).to(self.device)
        
        # Schema consistency (if using global context)
        if self.use_global_context and context_sequence is not None:
            perceiver_attention = self.attention_extractor(context_sequence)
            schema_consistency = F.mse_loss(
                outputs['attention_states'][:, :perceiver_attention.shape[1]], 
                perceiver_attention,
                reduction='mean'
            )
            outputs['schema_consistency_loss'] = schema_consistency
        else:
            outputs['schema_consistency_loss'] = torch.tensor(0.0).to(self.device)
        
        # Perceiver loss
        if perceiver_recon is not None:
            obs_flat = observations.reshape(-1, self.image_size * self.image_size * self.input_channels, 1)
            outputs['perceiver_loss'] = F.mse_loss(perceiver_recon, obs_flat, reduction='mean')
        else:
            outputs['perceiver_loss'] = torch.tensor(0.0).to(self.device)
        
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
                self.image_discriminator.parameters(), self._grad_clip
            )
            self.disc_scaler.step(self.img_disc_optimizer)
        else:
            img_loss.backward()
            img_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.image_discriminator.parameters(), self._grad_clip
            )
            self.img_disc_optimizer.step()
    
        # Latent discriminator
        self.latent_disc_optimizer.zero_grad()
        latent_loss = loss_dict['latent_disc_loss']
    
        if hasattr(self, 'disc_scaler') and torch.cuda.is_available():
            self.disc_scaler.scale(latent_loss).backward()
            self.disc_scaler.unscale_(self.latent_disc_optimizer)
            latent_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.latent_discriminator.parameters(), self._grad_clip
            )
            self.disc_scaler.step(self.latent_disc_optimizer)
            self.disc_scaler.update()
        else:
            latent_loss.backward()
            latent_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.latent_discriminator.parameters(), self._grad_clip
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
                        entropy_weight=0.1, lambda_img=0.1, lambda_latent=0.1,
                        lambda_pred=0.1, lambda_att=0.1, lambda_schema=0.1):
        """
        Theoretically-aligned loss computation leveraging pre-computed components
        """
        # Forward pass computes all component losses
        outputs = self.forward_sequence(observations, actions)
        batch_size, seq_len = observations.shape[:2]
        
        losses = {}
        
        # === ELBO Components (already computed in forward_sequence) ===
        
        # 1. Reconstruction fidelity (E_q[log p(o|z)])
        losses['recon_loss'] = outputs['reconstruction_losses'].mean()
        
        # 2. Latent dynamics regularization (KL[q(z|o)||p(z|h,c,π)])
        losses['kl_z'] = torch.stack(outputs['kl_latents']).mean() if outputs['kl_latents'] else 0.0
        
        # 3. Stick-breaking process regularization
        losses['kumar_beta_kl'] = torch.stack(outputs['kumaraswamy_kl_losses']).mean() if outputs['kumaraswamy_kl_losses'] else 0.0
        
        # 4. Concentration hyperprior (sign correction: should be positive KL)
        # Note: alpha_prior_losses stored as negative log prob, so we negate
        losses['alpha_prior'] = -torch.stack(outputs['alpha_prior_losses']).mean() if outputs['alpha_prior_losses'] else 0.0
        
        # 5. Self-model alignment
        if outputs['self_model_kl_losses']:
            self_model_kls = [loss['total'] for loss in outputs['self_model_kl_losses']]
            losses['self_model_kl'] = torch.stack(self_model_kls).mean()
        else:
            losses['self_model_kl'] = torch.tensor(0.0).to(self.device)
        
        # 6. Cluster entropy (negative for maximization)
        losses['cluster_entropy'] = torch.stack(outputs['cluster_entropies']).mean() if outputs['cluster_entropies'] else 0.0
        
        # 7. Future prediction bonuses (already computed with correct sign)
        losses['future_pred_bonus'] = outputs['future_h_bonus'] + outputs['future_z_bonus']
        
        # 8. Attention components
        losses['attention_kl'] = torch.stack(outputs['attention_kl_losses']).mean() if outputs['attention_kl_losses'] else 0.0
        losses['attention_pred_bonus'] = outputs['future_att_bonus']
        
        # 9. Schema consistency
        losses['schema_consistency'] = outputs['schema_consistency_loss']
        
        # 10. Perceiver auxiliary loss
        losses['perceiver_loss'] = outputs['perceiver_loss']
        
        # === Compute ELBO with proper mathematical formulation ===
        # ELBO = E_q[log p(o|z)] - KL_terms + entropy + prediction_bonuses
        losses['elbo'] = (
            -losses['recon_loss']                              # Negative for minimization
            - beta * losses['kl_z']                           # KL penalties
            - beta * losses['kumar_beta_kl']                  
            - beta * losses['alpha_prior']                    
            - beta * losses['self_model_kl']                  
            + entropy_weight * losses['cluster_entropy']      # Entropy bonus
            + lambda_pred * losses['future_pred_bonus']       # Prediction rewards
            - beta * losses['attention_kl']                   # Attention regularization
            + lambda_att * losses['attention_pred_bonus']     # Attention prediction
            - lambda_schema * losses['schema_consistency']    # Schema alignment
            - 0.1 * losses['perceiver_loss']                 # Auxiliary perceiver loss
        )
        
        # Total VAE loss (negative ELBO for minimization)
        losses['total_vae_loss'] = -losses['elbo']
        
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
        alpha = torch.rand(batch_size, 1, device=device)

        # Expand alpha for proper broadcasting
        if len(real_samples.shape) > 2:  # For image discriminator
            alpha = alpha.view(-1, 1, 1, 1)

        # Get interpolated samples
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)

        # Get discriminator output for interpolated samples
        disc_interpolates = discriminator(interpolates)

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
        q_dist = torch.distributions.Normal(
            params['z_mean'],
            torch.exp(0.5 * params['z_logvar'])
        )
        # Get encoder's hidden state
        h = self.encoder.hlayer

        # Compute KL using Monte Carlo estimation
        kl_z = self.prior.compute_kl_divergence_mc(q_dist, params)
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

        # 5. Prior over concentration parameter (alpha)
        alpha_prior = self.prior.stick_breaking.concentration_prior(h, self.max_K-1)
        losses['alpha_prior'] = alpha_prior

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
            unused_penalty -
            alpha_prior + # Negative because we're maximizing ELBO
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
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        real_latents: torch.Tensor,
        fake_latents: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Training step for both discriminators
        """

        # Image discriminator
        real_img_score = self.image_discriminator(real_images)
        fake_img_score = self.image_discriminator(fake_images.detach())

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

        # Latent discriminator
        real_latent_score = self.latent_discriminator(real_latents)
        fake_latent_score = self.latent_discriminator(fake_latents.detach())

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
        result = {
            'img_disc_loss': img_disc_loss,
            'latent_disc_loss': latent_disc_loss,
            'img_gp': img_gp,
            'latent_gp': latent_gp,
            'real_img_score': real_img_score.mean(),
            'fake_img_score': fake_img_score.mean(),
            'real_latent_score': real_latent_score.mean(),
            'fake_latent_score': fake_latent_score.mean()
            }
    
        # Only perform optimization if optimizers exist and we're configured to use them
        if self.has_optimizers and hasattr(self, 'img_disc_optimizer') and hasattr(self, 'latent_disc_optimizer'):
            # Update image discriminator
            self.img_disc_optimizer.zero_grad()
            img_disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.image_discriminator.parameters(), self._grad_clip)
            self.img_disc_optimizer.step()

            # Update latent discriminator
            self.latent_disc_optimizer.zero_grad()
            latent_disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.latent_discriminator.parameters(), self._grad_clip)
            self.latent_disc_optimizer.step()
        
            # Step the discriminator schedulers if available
            if hasattr(self, 'img_disc_scheduler') and hasattr(self, 'latent_disc_scheduler'):
                self.img_disc_scheduler.step()
                self.latent_disc_scheduler.step()
                # Add learning rates to result if available
                result['img_disc_lr'] = self.img_disc_scheduler.get_last_lr()[0]
                result['latent_disc_lr'] = self.latent_disc_scheduler.get_last_lr()[0]
        
            # Convert tensor values to items for return value when optimization is performed
            #for key in list(result.keys()):
            #    if torch.is_tensor(result[key]):
            #        result[key] = result[key].item()
    
        return result

    def compute_adversarial_losses(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        z: torch.Tensor,
        prior_z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adversarial losses for both image and latent space
        """
        # Get features and scores for real/fake images
        real_features = self.image_discriminator(x, get_features=True)
        fake_features = self.image_discriminator(reconstruction, get_features=True) 
    
        # Regular adversarial loss
        fake_img_score = fake_features[-1] # Last feature is discriminator output
        img_adv_loss = -torch.mean(fake_img_score)
    
        # Feature matching loss
        feat_match_loss = compute_feature_matching_loss(
        real_features,
        fake_features
        )
    
        # Latent space adversarial loss stays the same
        fake_latent_score = self.latent_discriminator(z)
        latent_adv_loss = -torch.mean(fake_latent_score)

        return img_adv_loss, latent_adv_loss, feat_match_loss

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
        
        # Extract reconstruction for perceiver loss
        perceiver_recon = perceiver_output[self.out_keys.INPUT_RECONSTRUCTION]['image']
        
        # Project to context dimension
        latents = latents.transpose(1, 2)
        context = self.perceiver_projection(latents)
        context = context.reshape(batch_size, seq_len, -1)
        
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
                    
                    batch_size = attention_weights.shape[0] // block.num_output_groups
                    num_groups = block.num_output_groups
                    
                    # Reshape to separate batch and groups
                    attention_weights = attention_weights.view(
                        batch_size, num_groups, 
                        attention_weights.shape[1],  # num_heads
                        attention_weights.shape[2],  # queries
                        attention_weights.shape[3]   # keys
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
                            lambda_latent: float = 0.1,
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
        batch_size, seq_len = observations.shape[:2]
        
        # Normalize observations if needed
        observations = self.prepare_images_for_training(observations)
        
        # 1. Forward pass and compute all VAE losses
        vae_losses, outputs = self.compute_total_loss(
            observations, actions, beta, entropy_weight, 
            lambda_img, lambda_latent, lambda_pred, lambda_att, lambda_schema
        )
        
        # 2. Discriminator training
        # Sample random timesteps for adversarial training
        t_samples = torch.randint(0, seq_len, (n_critic,))
        
        disc_losses_list = []
        for t in t_samples:
            real_images = observations[:, t]
            fake_images = outputs['reconstructions'][:, t].detach()
            
            # Sample from standard normal as "real" latents
            real_latents = torch.randn_like(outputs['latents'][:, t])
            fake_latents = outputs['latents'][:, t].detach()
            
            disc_loss = self.discriminator_step(
                real_images, fake_images, real_latents, fake_latents
            )
            disc_losses_list.append(disc_loss)
        
        # Average discriminator losses
        avg_disc_losses = {
            k: sum(d[k] for d in disc_losses_list) / len(disc_losses_list)
            for k in disc_losses_list[0].keys()
        }
        
        # 3. Generator training with adversarial losses
        # Sample a timestep for generator adversarial loss
        t_gen = torch.randint(0, seq_len, (1,)).item()
        
        fake_img_score = self.image_discriminator(outputs['reconstructions'][:, t_gen])
        fake_latent_score = self.latent_discriminator(outputs['latents'][:, t_gen])
        
        # Get feature matching loss
        real_features = self.image_discriminator(observations[:, t_gen], get_features=True)
        fake_features = self.image_discriminator(outputs['reconstructions'][:, t_gen], get_features=True)
        feat_match_loss = compute_feature_matching_loss(real_features, fake_features)
        
        # Adversarial losses
        img_adv_loss = -torch.mean(fake_img_score)
        latent_adv_loss = -torch.mean(fake_latent_score)
        
        # 4. Total generator loss
        total_gen_loss = (
            vae_losses['elbo'] +
            lambda_img * img_adv_loss +
            lambda_latent * latent_adv_loss +
            feat_match_loss
        )
        
        # 5. Optimize generator
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
        
        # 6. Prepare output dictionary
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
            
            # Process initial step through VRNN
            attention_features = self.attention_prior_posterior.attention_encoder(
                attention_map.unsqueeze(1)
            )
            
            # Feature extraction
            phi_z = self.phi_z(z_0)
            phi_c = self.phi_c(c_0)
            phi_attention = self.phi_attention(attention_features)
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
                
                # Update VRNN state
                phi_z = self.phi_z(z_next)
                phi_c = self.phi_c(current_c_t)
                phi_attention = self.phi_attention(attention_features)
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
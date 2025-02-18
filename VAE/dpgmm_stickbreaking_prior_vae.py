import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Gamma, Categorical, Independent, MixtureSameFamily
from typing import Dict, Tuple, Union
from models import *
import math
from collections import OrderedDict

# @torch.jit.script
def check_tensor(tensor: torch.Tensor, name: str) -> None:
    """Validate tensor values for debugging"""
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")
    
    
    

# @torch.jit.script
def beta_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    """Compute beta function in log space for numerical stability"""
    eps= 1.1920929e-06
    return torch.lgamma(a + eps) + torch.lgamma(b + eps) - torch.lgamma(a + b + eps)

# @torch.jit.script
def prune_small_components(pi: torch.Tensor, threshold: float = 1e-5) -> torch.Tensor:
    """Prune components with very small mixing proportions"""
    # Find components above threshold
    eps =1.1920929e-07
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

class AddEpsilon(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        return x + self.eps
    
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
    def __init__(self, max_components: int, hidden_dim: int, device: torch.device, dkl_taylor_order:int=10):
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
        self.gamma_a = nn.Parameter(torch.tensor(1.0, device=self.device), requires_grad=True)
        self.gamma_b = nn.Parameter(torch.tensor(1.0, device=self.device), requires_grad=True)

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
    # @torch.jit.script
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
        eps = 1.1920929e-07
        tiny = 1e-5
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
    # @torch.jit.script
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
    # @torch.jit.script
    def compute_kumar2beta_kl( a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, n_approx: int) -> torch.Tensor:
        """
        Compute KL divergence between Kumaraswamy(a,b) and Beta(alpha,beta)
        KL(K(a,b) || B(α,β)) = 
            (a-α)/a * (-γ - ψ(b) - 1/b) +     # First part
            ln(a*b) + ln(B(α,β)) -      # Log terms
            (b-1)/b +                         # Additional term
            (β-1)b * E[ln(1-v)]               # Taylor expansion term
        https://arxiv.org/pdf/1905.12052
        """
        eps = 1.1920929e-06
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
        device: torch.device
    ):
        super().__init__()
        self.max_K = max_components
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Stick-breaking process
        self.stick_breaking = AdaptiveStickBreaking(max_components, hidden_dim, device)

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

class DPGMMVariationalAutoencoder(nn.Module):
    """
    VAE with Dirichlet Process GMM Prior
    """
    def __init__(
        self,
        max_components: int,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        img_disc_channels:int,
        img_disc_layers:int,
        latent_disc_layers:int,
        device: torch.device,
        use_actnorm: bool= False,
        learning_rate: float = 1e-5,
        grad_clip:float =1.0
    ):
        super().__init__()
        self.max_K = max_components
        self.latent_dim = latent_dim
        self.device = device
        self.min_temperature = 0.1
        # self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.temperature = torch.ones(1) * 1.0
        self._lr = learning_rate
        self._grad_clip = grad_clip
        # Encoder network (can reuse your existing encoder architecture)
        self.encoder = VAEEncoder(
            channel_size_per_layer=[64,64,128,128,128,128,256,256],
            layers_per_block_per_layer=[2,2,2,2,2,2,2,2],
            latent_size=latent_dim,
            width=input_dim,
            height=input_dim,
            num_layers_per_resolution=[2,2,2,2],
            mlp_hidden_size=hidden_dim,
            channel_size=64,
            input_channels=3,
            downsample=4
        ).to(device)

        # DP-GMM prior
        self.prior = DPGMMPrior(max_components, latent_dim, hidden_dim, device)

        # Decoder network (can reuse your existing decoder architecture)
        self.decoder = VAEDecoder(
            latent_size=latent_dim,
            width=input_dim,
            height=input_dim,
            channel_size_per_layer=[256,256,128,128,128,128,64,64],
            layers_per_block_per_layer=[2,2,2,2,2,2,2,2],
            num_layers_per_resolution=[2,2,2,2],
            input_channels=3,
            downsample=4,
            mlp_hidden_size=hidden_dim
        ).to(device)

        self.encoder.gradient_checkpointing_enable()
        self.decoder.gradient_checkpointing_enable()
        # Initialize discriminators
        self.image_discriminator = ImageDiscriminator(
            input_nc=3,  # For RGB images
            ndf=img_disc_channels,
            n_layers=img_disc_layers,
            use_actnorm=use_actnorm
        ).to(device)

        # Latent discriminator operates on flattened latent vectors
        self.latent_discriminator = LatentDiscriminator(
            input_dims=latent_dim,
            num_layers=latent_disc_layers,
            norm_type='layer'
        ).to(device)

        # Separate optimizers
        self.gen_optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.prior.parameters()),
            lr= self._lr, 
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        self.img_disc_optimizer = torch.optim.AdamW(
            self.image_discriminator.parameters(),
            lr= self._lr, 
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        self.latent_disc_optimizer = torch.optim.AdamW(
            self.latent_discriminator.parameters(),
            lr= self._lr, 
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        self.gen_scheduler = get_improved_scheduler(self.gen_optimizer)
        self.img_disc_scheduler = get_improved_scheduler(self.img_disc_optimizer)
        self.latent_disc_scheduler = get_improved_scheduler(self.latent_disc_optimizer)

        # Add step counter for schedulers
        self.current_steps = 0
        self.apply(self.init_weights)
        self.to(device)

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
    def update_temperature(self, epoch: int, min_temp: float = 0.1, anneal_rate: float = 0.003):
        """
         Update temperature parameter with cosine annealing schedule
    
         Args:
            epoch: Current epoch number
            min_temp: Minimum temperature value
            anneal_rate: Rate of temperature annealing
        """
        self.temperature = torch.tensor(
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
            beta * pi_prior +
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

        # Update discriminators
        self.img_disc_optimizer.zero_grad()
        img_disc_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.image_discriminator.parameters(), self._grad_clip)
        self.img_disc_optimizer.step()

        self.latent_disc_optimizer.zero_grad()
        latent_disc_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.latent_discriminator.parameters(), self._grad_clip)
        self.latent_disc_optimizer.step()
        # Step the discriminator schedulers
        self.img_disc_scheduler.step()
        self.latent_disc_scheduler.step()
        return {
            'img_disc_loss': img_disc_loss.item(),
            'latent_disc_loss': latent_disc_loss.item(),
            'img_disc_lr': self.img_disc_scheduler.get_last_lr()[0],
            'latent_disc_lr': self.latent_disc_scheduler.get_last_lr()[0],
            'img_gp': img_gp.item(),
            'latent_gp': latent_gp.item(),
            'real_img_score': real_img_score.mean().item(),
            'fake_img_score': fake_img_score.mean().item(),
            'real_latent_score': real_latent_score.mean().item(),
            'fake_latent_score': fake_latent_score.mean().item()
        }

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
        # Image space adversarial loss
        fake_img_score = self.image_discriminator(reconstruction)
        img_adv_loss = -torch.mean(fake_img_score)

        # Latent space adversarial loss
        fake_latent_score = self.latent_discriminator(z)
        latent_adv_loss = -torch.mean(fake_latent_score)

        return img_adv_loss, latent_adv_loss

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
        img_adv_loss, latent_adv_loss = self.compute_adversarial_losses(
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
            lambda_latent * latent_adv_loss  # Latent adversarial loss
        )

        # Optimize generator
        torch.cuda.empty_cache()

        self.gen_optimizer.zero_grad()
        total_loss.backward()

        # mem_alloc = torch.cuda.memory_allocated() / 1024**2  # MB
        # mem_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        # max_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
        # # TODO: return this info somewhere rather than printing it directly
        # print(f"Allocated: {mem_alloc:.2f} MB, Reserved: {mem_reserved:.2f} MB, Max: {max_mem:.2f} MB")

        torch.nn.utils.clip_grad_norm_(self.parameters(), self._grad_clip)
        self.gen_optimizer.step()
        self.gen_scheduler.step()
        return {
            **vae_losses,
            'gen_lr': self.gen_scheduler.get_last_lr()[0],
            'img_adv_loss': img_adv_loss.item(),
            'latent_adv_loss': latent_adv_loss.item(),
            'total_loss': total_loss.item()
        }

    def training_step(self,
                      x: torch.Tensor,
                      beta:float=1.0,
                      n_critic:int=3,
                      lambda_img: float=0.2,
                      lambda_latent: float=0.1,
                      current_step: int = 0) -> Dict[str, torch.Tensor]:
        """
        Perform one training step

        Args:
           x: Input batch
           beta: Weight of KL divergence term
           n_critic: Number of critic iterations per generator step
           lambda_img: Weight of image adversarial loss
           lambda_latent: Weight of latent adversarial loss
           current_step: Current training step for scheduler

        Returns:
           Dictionary with loss values
        """
        # Forward pass
        reconstruction, params = self.forward(x)
        prior_z = torch.randn_like(params['z'])
        latent_vectors= params['z'].detach()
        # Compute losses
        losses = self.compute_loss(x, reconstruction, params)

        # Train discriminators
        disc_losses = []
        for _ in range(n_critic):
            disc_loss = self.discriminator_step(
                x, reconstruction, prior_z, params['z']
            )
            disc_losses.append(disc_loss)

        # Train generator
        gen_losses = self.generator_step(x, beta, lambda_img, lambda_latent)
        gen_lr =  gen_losses['gen_lr']
        img_disc_lr = disc_loss['img_disc_lr']
        latent_disc_lr = disc_loss['latent_disc_lr']
        # Average discriminator losses
        avg_disc_losses = {
            k: sum(d[k] for d in disc_losses) / len(disc_losses)
            for k in disc_losses[0].keys()
        }

        lr_stats = {
        'gen_lr': gen_lr,
        'img_disc_lr': img_disc_lr,
        'latent_disc_lr': latent_disc_lr
        }
        # You might want to log these losses or use them for monitoring
        vae_losses = {
                    'elbo': losses['loss'].item(),
                    'reconstruction': losses['recon_loss'].item(),
                    'kl_z': losses['kl_z'].item(),
                    'kumar_beta_kl': losses['kumar_beta_kl'].item(),
                    'unused_components': losses['unused_penalty'].item(),
                    'effective_components': losses['K_eff'].item()
                  }

        return losses, {
            **gen_losses,
            **avg_disc_losses,
            **vae_losses,
            **lr_stats
        }, latent_vectors

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

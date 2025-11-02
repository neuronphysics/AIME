"""
Dirichlet Process Gaussian Mixture Model (DPGMM) Prior

Implements an adaptive infinite mixture prior for VAE latent space using:
- Dirichlet Process: Non-parametric prior allowing infinite mixtures
- Stick-breaking construction: Generate finite truncation of mixture weights
- Context-dependent components: Gaussian means/variances generated from hidden states
- Monte Carlo KL estimation: Compute KL divergence via sampling

The DPGMM prior p(z|h) is a mixture of Gaussians:
    p(z|h) = Σ_k π_k(h) N(z | μ_k(h), σ²_k(h))

where:
- π_k(h): Mixing weights from stick-breaking
- μ_k(h), σ²_k(h): Component parameters from neural networks
- h: Hidden state providing context

This replaces standard Gaussian prior N(0,I) with adaptive, context-dependent mixtures.

Reference:
- Dirichlet Process: Ferguson (1973)
- Stick-breaking: Sethuraman (1994)
- DPGMM-VAE: Nalisnick & Smyth (2017)
"""

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical, Independent, MixtureSameFamily
from typing import Dict, Tuple
import math

from .stick_breaking import AdaptiveStickBreaking


class DPGMMPrior(nn.Module):
    """
    Dirichlet Process Gaussian Mixture Model prior for VAE.

    Generates a context-dependent mixture of Gaussians prior:
    - Mixing weights π from adaptive stick-breaking
    - Component means and log-variances from neural network
    - Supports Monte Carlo KL divergence estimation

    Args:
        max_components: Maximum number of mixture components (K)
        latent_dim: Dimension of latent variable z
        hidden_dim: Dimension of hidden state h
        device: Device to place parameters on
        prior_alpha: Prior concentration for Gamma hyperprior
        prior_beta: Prior rate for Gamma hyperprior
    """

    def __init__(
        self,
        max_components: int,
        latent_dim: int,
        hidden_dim: int,
        device: torch.device,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0
    ):
        super().__init__()
        self.max_K = max_components
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Stick-breaking process for mixture weights
        self.stick_breaking = AdaptiveStickBreaking(
            max_components, hidden_dim, device,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta
        )

        # Neural network to generate component parameters (means and log-variances)
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
        n_samples: int = 10
    ) -> torch.Tensor:
        """
        Compute KL divergence between Gaussian posterior and DPGMM prior using Monte Carlo.

        KL(q(z|x) || p(z|h)) = E_q[log q(z|x) - log p(z|h)]

        where:
        - q(z|x) = N(z | μ_q, σ²_q) is the VAE posterior
        - p(z|h) = Σ_k π_k N(z | μ_k, σ²_k) is the DPGMM prior

        Uses reparameterization trick for MC estimation.

        Args:
            posterior_mean: Posterior mean μ_q [batch_size, latent_dim]
            posterior_logvar: Posterior log-variance log(σ²_q) [batch_size, latent_dim]
            prior_params: Dictionary with:
                - pi: Mixture weights [batch_size, K]
                - means: Component means [batch_size, K, latent_dim]
                - log_vars: Component log-variances [batch_size, K, latent_dim]
            n_samples: Number of MC samples

        Returns:
            kl: Mean KL divergence [scalar]
        """
        prior_weights = prior_params['pi']
        prior_means = prior_params['means']
        prior_logvars = prior_params['log_vars']

        B, D = posterior_mean.shape
        K = prior_weights.shape[1]
        eps = torch.finfo(torch.float32).eps

        # Sample from posterior using reparameterization: z = μ + ε·σ
        noise = torch.randn(n_samples, B, D, device=posterior_mean.device)
        z = posterior_mean.unsqueeze(0) + noise * torch.exp(0.5 * posterior_logvar).unsqueeze(0)
        # z: [n_samples, batch_size, latent_dim]

        # Compute log q(z|x) for each sample
        log_q = -0.5 * (
            D * math.log(2 * math.pi) +
            posterior_logvar.sum(dim=-1) +
            ((z - posterior_mean.unsqueeze(0)) ** 2 *
             torch.exp(-posterior_logvar.unsqueeze(0))).sum(dim=-1)
        )
        # log_q: [n_samples, batch_size]

        # Compute log p(z|h) = log Σ_k π_k N(z|μ_k,σ²_k)
        z_expanded = z.unsqueeze(2).expand(-1, -1, K, -1)  # [n_samples, B, K, D]

        # Log density for each component
        log_component_densities = -0.5 * (
            D * math.log(2 * math.pi) +
            prior_logvars.sum(dim=-1).unsqueeze(0) +  # [1, B, K]
            ((z_expanded - prior_means.unsqueeze(0)) ** 2 *
             torch.exp(-prior_logvars.unsqueeze(0))).sum(dim=-1)
        )
        # log_component_densities: [n_samples, B, K]

        # Add log mixture weights and marginalize over components
        log_prior_components = log_component_densities + \
            torch.log(prior_weights.unsqueeze(0).clamp(min=eps) + eps)
        log_p = torch.logsumexp(log_prior_components, dim=2)  # [n_samples, B]

        # KL divergence: E_q[log q - log p]
        kl_samples = log_q - log_p  # [n_samples, B]
        return kl_samples.mean()  # Average over samples and batch

    def compute_kl_loss(
        self,
        params: Dict,
        alpha: torch.Tensor,
        h: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between variational and prior distributions.

        This includes:
        1. KL between Kumaraswamy(a,b) and Beta(1,α) for each stick
        2. KL between Gamma posterior and prior for concentration parameter α

        Args:
            params: Dictionary with:
                - kumar_a: Kumaraswamy parameter a [batch_size, K-1]
                - kumar_b: Kumaraswamy parameter b [batch_size, K-1]
            alpha: Concentration parameter samples [batch_size, n_samples]
            h: Hidden states [batch_size, hidden_dim]

        Returns:
            total_kl: Total KL divergence [scalar]
        """
        total_kl = 0.0

        # Average alpha over samples if needed
        alpha_scalar = alpha.mean(dim=1, keepdim=True) if alpha.dim() > 1 else alpha

        # KL divergence for each stick-breaking variable
        for k in range(self.max_K - 1):
            kumar_a_k = params['kumar_a'][:, k:k+1]  # [batch_size, 1]
            kumar_b_k = params['kumar_b'][:, k:k+1]  # [batch_size, 1]

            # Get alpha for current stick (with fallback)
            alpha_k = alpha[:, k:k+1] if k < alpha.shape[1] else alpha[:, -1:]

            # Beta(1, α) prior for stick-breaking
            beta_alpha = torch.ones_like(alpha_scalar).squeeze(-1)
            beta_beta = alpha_scalar.squeeze(-1)

            # Shape assertion for debugging
            assert kumar_a_k.shape == kumar_b_k.shape == beta_alpha.shape == beta_beta.shape, \
                f"Shape mismatch: kumar_a={kumar_a_k.shape}, kumar_b={kumar_b_k.shape}, " \
                f"beta_alpha={beta_alpha.shape}, beta_beta={beta_beta.shape} at component {k} of {self.max_K}"

            # Compute KL(Kumaraswamy || Beta)
            kl = self.stick_breaking.compute_kumar2beta_kl(
                kumar_a_k,
                kumar_b_k,
                beta_alpha,
                beta_beta,
                self.stick_breaking.M
            )
            total_kl = total_kl + kl

        # Add KL for hierarchical prior over concentration parameter α
        alpha_kl = self.stick_breaking.compute_gamma2gamma_kl(h)
        total_kl = total_kl.mean() + alpha_kl

        return total_kl

    def forward(
        self,
        h: torch.Tensor,
        n_samples: int = 10
    ) -> Tuple[MixtureSameFamily, Dict]:
        """
        Generate DPGMM prior distribution from hidden state.

        Args:
            h: Hidden states [batch_size, hidden_dim]
            n_samples: Number of MC samples for concentration parameter

        Returns:
            mixture: PyTorch MixtureSameFamily distribution object
            params: Dictionary with:
                - pi: Mixture weights [batch_size, K]
                - alpha: Concentration parameter [batch_size, n_samples]
                - means: Component means [batch_size, K, latent_dim]
                - log_vars: Component log-variances [batch_size, K, latent_dim]
                - kumar_a, kumar_b: Kumaraswamy parameters
                - v: Stick-breaking variables
                - perm: Permutation indices (if used)
                - active_components: Number of active components
        """
        batch_size = h.shape[0]

        # Get mixing proportions via stick-breaking
        pi, kumar_params = self.stick_breaking(h, n_samples, use_rand_perm=True)

        # Sample concentration parameter from posterior
        alpha = self.stick_breaking.alpha_posterior.sample(h, n_samples)

        # Generate component parameters (means and log-variances)
        params = self.component_nn(h)
        means, log_vars = torch.split(params, self.latent_dim * self.max_K, dim=1)

        # Clamp log-variances for numerical stability
        log_vars = torch.clamp(log_vars, min=-10.0, max=2.0)

        # Reshape to [batch_size, K, latent_dim]
        means = means.view(batch_size, self.max_K, self.latent_dim)
        log_vars = log_vars.view(batch_size, self.max_K, self.latent_dim)

        # Create PyTorch mixture distribution
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

    def get_effective_components(
        self,
        pi: torch.Tensor,
        threshold: float = 1e-3
    ) -> torch.Tensor:
        """
        Determine effective number of components based on mixing proportions.

        Components are considered "effective" if their cumulative weight
        accounts for (1 - threshold) of the total mass.

        Args:
            pi: Mixture weights [batch_size, K]
            threshold: Tail probability threshold (default: 0.001)

        Returns:
            n_effective: Number of effective components [batch_size]
        """
        sorted_pi, _ = torch.sort(pi, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_pi, dim=-1)
        return torch.sum(cumsum < (1.0 - threshold), dim=-1) + 1

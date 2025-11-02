"""
Gamma Posterior Distribution

Amortized variational posterior for Gamma distribution.
Maps encoder hidden states to Gamma concentration and rate parameters.

Reference: https://github.com/threewisemonkeys-as/PyTorch-VAE
"""

import torch
import torch.nn as nn
from torch.distributions import Gamma
from typing import Tuple
from collections import OrderedDict


class AddEpsilon(nn.Module):
    """Add small epsilon for numerical stability"""
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.eps


class GammaPosterior(nn.Module):
    """
    Amortized variational posterior for Gamma distribution.
    Maps encoder hidden states to Gamma parameters.

    The Gamma distribution is parameterized by concentration (α) and rate (β):
    - Concentration: Shape parameter (α > 0)
    - Rate: Inverse scale parameter (β > 0)

    Args:
        hidden_dim: Dimension of input hidden states
        device: Device to place parameters on
        eps: Small epsilon for numerical stability
    """

    def __init__(
        self,
        hidden_dim: int,
        device: torch.device,
        eps: float = torch.finfo(torch.float32).eps
    ):
        super().__init__()

        # Network for generating Gamma parameters from hidden states
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
        Generate Gamma parameters from hidden representation.

        Args:
            h: Hidden states [batch_size, hidden_dim]

        Returns:
            concentration: Shape parameter α [batch_size]
            rate: Rate parameter β [batch_size]
        """
        # Handle NaN/Inf inputs
        if torch.isnan(h).any() or torch.isinf(h).any():
            print(f"Warning: NaN/Inf in GammaPosterior input, replacing with safe values")
            mean_h = torch.nanmean(h)
            h = torch.where(torch.isnan(h) | torch.isinf(h), mean_h, h)

        # Generate parameters
        params = self.param_net(h)
        concentration, rate = params.split(1, dim=-1)

        # Ensure positivity and numerical stability
        concentration = torch.clamp(concentration.squeeze(-1), min=self.eps)
        rate = torch.clamp(rate.squeeze(-1), min=self.eps)

        return concentration, rate

    def sample(self, h: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Sample from the Gamma posterior.

        Args:
            h: Hidden states [batch_size, hidden_dim]
            n_samples: Number of samples to draw

        Returns:
            samples: [batch_size, n_samples]
        """
        concentration, rate = self.forward(h)
        gamma_dist = Gamma(concentration.unsqueeze(-1), rate.unsqueeze(-1))
        return gamma_dist.rsample((n_samples,)).transpose(0, 1)

    def kl_divergence(
        self,
        h: torch.Tensor,
        prior_concentration: torch.Tensor,
        prior_rate: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between Gamma posterior and prior.

        KL(Gamma(α₁,β₁) || Gamma(α₂,β₂)) =
            α₂log(β₁/β₂) - logΓ(α₁) + logΓ(α₂) + (α₁-α₂)ψ(α₁) - (β₁-β₂)(α₁/β₁)

        where ψ is the digamma function.

        Args:
            h: Hidden states [batch_size, hidden_dim]
            prior_concentration: Prior shape parameter α₂ (scalar or tensor)
            prior_rate: Prior rate parameter β₂ (scalar or tensor)

        Returns:
            kl: Mean KL divergence [scalar]
        """
        alpha, beta = self.forward(h)  # Posterior parameters
        a, b = prior_concentration, prior_rate  # Prior parameters

        term1 = a * torch.log(beta / b)
        term2 = -torch.lgamma(alpha) + torch.lgamma(a)
        term3 = (alpha - a) * torch.digamma(alpha)
        term4 = -(beta - b) * (alpha / beta)

        kl = term1 + term2 + term3 + term4
        return kl.mean()

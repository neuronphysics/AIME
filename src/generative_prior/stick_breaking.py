"""
Adaptive Stick-Breaking for Dirichlet Process

Implements adaptive stick-breaking construction for generating mixture weights
in the Dirichlet Process Gaussian Mixture Model (DPGMM).

The stick-breaking process generates mixing proportions π via:
1. Sample v_k ~ Kumaraswamy(a_k, b_k) for k = 1,...,K-1
2. Compute π_k = v_k ∏_{j<k} (1 - v_j)

where the Kumaraswamy parameters (a, b) are generated from hidden states
using neural networks.

Reference:
- Stick-breaking: Sethuraman (1994)
- Kumaraswamy-Beta KL: https://arxiv.org/pdf/1905.12052
- Adaptive truncation: https://arxiv.org/pdf/2410.00660
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from collections import OrderedDict

from .distributions import GammaPosterior, KumaraswamyStable
# Import check_tensor from src.models if needed, otherwise define locally
try:
    from src.models import check_tensor
except ImportError:
    def check_tensor(x: torch.Tensor, name: str = "tensor"):
        """Simple tensor validity check"""
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: {name} contains NaN/Inf values")


def beta_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute beta function in log space for numerical stability.

    Beta function: B(a,b) = Γ(a)Γ(b) / Γ(a+b)
    Log-space: log B(a,b) = lgamma(a) + lgamma(b) - lgamma(a+b)

    Args:
        a: First parameter [any shape]
        b: Second parameter [same shape as a]

    Returns:
        log_beta: Log of beta function [same shape]
    """
    eps = torch.finfo(torch.float32).eps
    return torch.lgamma(a + eps) + torch.lgamma(b + eps) - torch.lgamma(a + b + eps)


class KumaraswamyNetwork(nn.Module):
    """
    Neural network to generate Kumaraswamy distribution parameters.

    Transforms hidden states h_t into log-space Kumaraswamy parameters (log a, log b)
    for K-1 components (the K-th component weight is determined by normalization).

    Architecture:
    - Two parallel networks for log(a) and log(b)
    - Spectral normalization for stability
    - Layer normalization and GELU activations
    - Output clamping to prevent extreme values

    Args:
        hidden_dim: Dimension of input hidden states
        num_components: Maximum number of mixture components (K)
        device: Device to place parameters on
    """

    def __init__(self, hidden_dim: int, num_components: int, device: torch.device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.K = num_components
        self.device = device
        self.eps = torch.tensor(torch.finfo(torch.float32).eps, device=self.device)

        # Initialize network for parameter a
        kumar_a_fc = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(kumar_a_fc.weight, gain=0.5)
        nn.init.constant_(kumar_a_fc.bias, 0.0)

        kumar_a_out = nn.Linear(hidden_dim, self.K - 1)
        nn.init.normal_(kumar_a_out.weight, 0, 0.01)
        nn.init.constant_(kumar_a_out.bias, 0.0)

        # Initialize network for parameter b
        kumar_b_fc = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(kumar_b_fc.weight, gain=0.5)
        nn.init.constant_(kumar_b_fc.bias, 0.0)

        kumar_b_out = nn.Linear(hidden_dim, self.K - 1)
        nn.init.normal_(kumar_b_out.weight, 0, 0.01)
        nn.init.constant_(kumar_b_out.bias, 0.0)

        # Build networks with spectral norm and layer norm
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
        Generate Kumaraswamy parameters from hidden representation.

        Args:
            h: Hidden states [batch_size, hidden_dim]

        Returns:
            log_a: Log of shape parameter [batch_size, K-1]
            log_b: Log of shape parameter [batch_size, K-1]
        """
        # Handle NaN/Inf inputs
        if torch.isnan(h).any() or torch.isinf(h).any():
            print(f"Warning: NaN/Inf in KumaraswamyNetwork input, replacing with safe values")
            mean_h = torch.nanmean(h)
            h = torch.where(torch.isnan(h) | torch.isinf(h), mean_h, h)

        # Generate parameters with clamping
        min_val = -5.0
        max_val = 5.0
        log_a = torch.clamp(self.net_a(h), min=min_val, max=max_val)
        log_b = torch.clamp(self.net_b(h), min=min_val, max=max_val)

        # Safety: replace any remaining NaN/Inf
        log_a = torch.nan_to_num(log_a, nan=0.0, posinf=5.0, neginf=-5.0)
        log_b = torch.nan_to_num(log_b, nan=0.0, posinf=5.0, neginf=-5.0)

        if torch.isnan(log_a).any() or torch.isnan(log_b).any():
            print(f"Warning: NaN in Kumaraswamy parameters: "
                  f"h_range=({h.min()}, {h.max()}), "
                  f"a_range=({log_a.min()}, {log_a.max()}), "
                  f"b_range=({log_b.min()}, {log_b.max()})")
            mean_a = torch.nanmean(log_a)
            log_a = torch.where(torch.isnan(log_a) | torch.isinf(log_a), mean_a, log_a)
            mean_b = torch.nanmean(log_b)
            log_b = torch.where(torch.isnan(log_b) | torch.isinf(log_b), mean_b, log_b)

        return log_a, log_b


class AdaptiveStickBreaking(nn.Module):
    """
    Adaptive stick-breaking construction for Dirichlet Process.

    Generates mixture weights π = (π₁, ..., π_K) using stick-breaking:
    - π_k = v_k ∏_{j<k} (1 - v_j)
    - v_k ~ Kumaraswamy(a_k, b_k)
    - (a_k, b_k) = f(h_t) learned from hidden states

    Features:
    - Adaptive truncation: Automatically prunes small components
    - Random permutation: Optional permutation invariance
    - Variational posterior over DP concentration parameter
    - Numerically stable implementation

    Args:
        max_components: Maximum number of mixture components (K)
        hidden_dim: Dimension of hidden states
        device: Device to place parameters on
        prior_alpha: Prior concentration for Gamma hyperprior
        prior_beta: Prior rate for Gamma hyperprior
        dkl_taylor_order: Taylor series order for Kumar-Beta KL computation
    """

    def __init__(
        self,
        max_components: int,
        hidden_dim: int,
        device: torch.device,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        dkl_taylor_order: int = 10
    ):
        super().__init__()
        self._max_K = max_components
        self.hidden_dim = hidden_dim
        self.device = device
        self.M = dkl_taylor_order  # Taylor series order

        # Variational posterior over concentration parameter α
        self.alpha_posterior = GammaPosterior(hidden_dim, device=device)

        # Gamma hyperprior parameters
        self.gamma_a = nn.Parameter(
            torch.tensor(prior_alpha, device=self.device),
            requires_grad=True
        )
        self.gamma_b = nn.Parameter(
            torch.tensor(prior_beta, device=self.device),
            requires_grad=True
        )

        # Register parameter names for clarity
        self.register_parameter('gamma_prior_a', self.gamma_a)
        self.register_parameter('gamma_prior_b', self.gamma_b)

        # Neural network for generating stick-breaking proportions
        self.kumar_net = KumaraswamyNetwork(hidden_dim, max_components, device)

        self.to(self.device)

    @property
    def max_K(self) -> int:
        """Ensure max_K is always returned as an integer."""
        return int(self._max_K)

    @staticmethod
    def sample_kumaraswamy(
        log_a: torch.Tensor,
        log_b: torch.Tensor,
        max_k: int,
        use_rand_perm: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample from Kumaraswamy distribution using stable reparameterization.

        Kumaraswamy(a, b) reparameterization:
        - U ~ Uniform(0, 1)
        - X = (1 - (1 - U)^(1/b))^(1/a)

        Optional random permutation provides permutation invariance.

        Args:
            log_a: Log of shape parameter [batch_size, K-1]
            log_b: Log of shape parameter [batch_size, K-1]
            max_k: Maximum number of components (K)
            use_rand_perm: Whether to apply random permutation

        Returns:
            v: Sampled stick-breaking variables [batch_size, K-1]
            perm: Permutation indices if used, None otherwise
        """
        log_a = log_a.to(log_a.device)
        log_b = log_b.to(log_b.device)

        check_tensor(log_a, "kumar_a")
        check_tensor(log_b, "kumar_b")

        # Optional random permutation for permutation invariance
        if use_rand_perm:
            # Generate random permutation indices
            perm = torch.argsort(torch.rand_like(log_a, device=log_a.device), dim=-1)
            perm = perm.view(-1, max_k - 1)

            # Apply permutation
            log_a = torch.gather(log_a, dim=1, index=perm)  # [batch_size, K-1]
            log_b = torch.gather(log_b, dim=1, index=perm)  # [batch_size, K-1]
        else:
            perm = None

        # Sample using stable Kumaraswamy implementation
        kumar_dist = KumaraswamyStable(log_a, log_b)
        v = kumar_dist.rsample()

        return v, perm

    @staticmethod
    def compute_stick_breaking_proportions(
        v: torch.Tensor,
        perm: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute mixture weights π from stick-breaking variables v.

        Stick-breaking formula:
        - π_k = v_k ∏_{j<k} (1 - v_j)
        - In log-space: log π_k = log v_k + Σ_{j<k} log(1 - v_j)

        If permutation was applied during sampling, this method inverts it
        to restore the original component ordering.

        Args:
            v: Stick-breaking variables [batch_size, K-1]
            perm: Optional permutation indices [batch_size, K-1]

        Returns:
            pi: Mixture weights [batch_size, K]
        """
        B = v.size(0)
        K = v.size(-1) + 1

        eps = torch.finfo(v.dtype).eps
        v = v.clamp(min=eps, max=1 - eps)

        # Compute cumulative sum of log(1 - v) for log-space stability
        log_prefix = torch.cumsum(torch.log1p(-v), dim=-1)
        log_prefix = F.pad(log_prefix, (1, 0), value=0.)

        # Compute log(π) = log(v) + log_prefix
        log_v_padded = torch.log(F.pad(v, (0, 1), value=1.0))
        pi = torch.exp(log_v_padded + log_prefix)

        # Invert permutation if it was applied
        if perm is not None:
            if perm.size(0) != B or perm.size(-1) != K - 1:
                raise ValueError("perm must be [B, K-1] and match v's batch/K.")
            # Extend permutation to include K-th component
            last_idx = torch.full((B, 1), K - 1, device=perm.device, dtype=perm.dtype)
            full_perm = torch.cat([perm, last_idx], dim=1)
            # Compute inverse permutation
            inv_perm = torch.argsort(full_perm, dim=1)
            # Apply inverse permutation to restore original order
            pi = torch.gather(pi, 1, inv_perm)

        # Renormalize to ensure sum to 1
        pi = pi / (pi.sum(dim=-1, keepdim=True) + eps)
        return pi

    def forward(
        self,
        h: torch.Tensor,
        n_samples: int = 10,
        use_rand_perm: bool = True,
        truncation_threshold: float = 0.995
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Generate mixture weights via adaptive stick-breaking.

        Args:
            h: Hidden states [batch_size, hidden_dim]
            n_samples: Number of samples (currently unused, for future MC)
            use_rand_perm: Whether to apply random permutation
            truncation_threshold: Cumulative weight threshold for truncation

        Returns:
            pi: Mixture weights [batch_size, K]
            info: Dictionary with auxiliary information:
                - kumar_a: Kumaraswamy parameter a
                - kumar_b: Kumaraswamy parameter b
                - v: Stick-breaking variables
                - perm: Permutation indices (if used)
                - active_components: Average number of active components
        """
        # Generate Kumaraswamy parameters from hidden state
        log_kumar_a, log_kumar_b = self.kumar_net(h)

        # Sample stick-breaking variables
        v, perm = self.sample_kumaraswamy(
            log_kumar_a, log_kumar_b, self.max_K, use_rand_perm
        )

        # Compute mixture proportions
        pi = self.compute_stick_breaking_proportions(v, perm)

        # Adaptive truncation: zero out components with small cumulative weight
        pi_sorted, sort_idx = torch.sort(pi, dim=-1, descending=True)
        pi_cumsum = torch.cumsum(pi_sorted, dim=-1)

        # Find truncation point
        truncation_mask = pi_cumsum < truncation_threshold
        # Ensure at least one component beyond threshold is kept
        truncation_mask[:, 1:] = truncation_mask[:, 1:] | truncation_mask[:, :-1]

        # Zero out unused components
        pi_truncated = pi_sorted * truncation_mask.float()

        # Restore original order
        _, unsort_idx = torch.sort(sort_idx, dim=-1)
        pi_final = torch.gather(pi_truncated, 1, unsort_idx)

        # Renormalize
        pi_final = pi_final / (
            pi_final.sum(dim=-1, keepdim=True) + torch.finfo(torch.float32).eps
        )

        return pi, {
            'kumar_a': torch.exp(log_kumar_a),
            'kumar_b': torch.exp(log_kumar_b),
            'v': v,
            'perm': perm,
            'active_components': truncation_mask.sum(dim=-1).float().mean(),
        }

    @staticmethod
    def compute_kumar2beta_kl(
        a: torch.Tensor,
        b: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        n_approx: int,
        eps: float = 100 * torch.finfo(torch.float32).eps
    ) -> torch.Tensor:
        """
        Compute KL divergence between Kumaraswamy(a,b) and Beta(α,β).

        Uses Taylor series approximation for E[log(1-v)] term.

        Reference: https://arxiv.org/pdf/1905.12052

        Args:
            a: Kumaraswamy parameter a
            b: Kumaraswamy parameter b
            alpha: Beta parameter α
            beta: Beta parameter β
            n_approx: Number of terms in Taylor series
            eps: Small epsilon for numerical stability

        Returns:
            kl: KL divergence [same shape as inputs]
        """
        EULER_GAMMA = 0.5772156649
        assert a.shape == b.shape == alpha.shape == beta.shape

        # Clamp parameters for stability
        a = torch.clamp(a, min=eps, max=20)
        b = torch.clamp(b, min=eps, max=20)
        alpha = torch.clamp(alpha, min=eps)
        beta = torch.clamp(beta, min=eps)

        ab = torch.mul(a, b)
        a_inv = torch.reciprocal(a + eps)
        b_inv = torch.reciprocal(b + eps)

        # Taylor expansion for E[log(1-v)]
        log_taylor = torch.logsumexp(
            torch.stack([
                beta_fn(m * a_inv, b) - torch.log(m + ab)
                for m in range(1, n_approx + 1)
            ], dim=-1),
            dim=-1
        )
        kl = torch.mul(torch.mul(beta - 1, b), torch.exp(log_taylor))

        # Add remaining terms
        psi_b = torch.digamma(b + eps)
        term1 = torch.mul(torch.div(a - alpha, a + eps), -EULER_GAMMA - psi_b - b_inv)
        term2 = torch.log(ab + eps) + beta_fn(alpha, beta)
        term2 = term2 + torch.div(-(b - 1), b + eps)
        kl = kl + term1 + term2

        # Debug large KL values
        if torch.any(kl > 1000):
            print(f"WARNING: Large Kumar-Beta KL detected: "
                  f"max={kl.max().item()}, mean={kl.mean().item()}")
            print(f"  a range: ({a.min().item()}, {a.max().item()})")
            print(f"  b range: ({b.min().item()}, {b.max().item()})")
            print(f"  alpha range: ({alpha.min().item()}, {alpha.max().item()})")
            print(f"  beta range: ({beta.min().item()}, {beta.max().item()})")

        return torch.clamp(kl, min=0.0)

    def compute_gamma2gamma_kl(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior over concentration α.

        KL(Gamma(α_q, β_q) || Gamma(α_p, β_p))

        Args:
            h: Hidden states [batch_size, hidden_dim]

        Returns:
            kl: Mean KL divergence [scalar]
        """
        prior_concentration = self.gamma_a
        prior_rate = self.gamma_b
        return self.alpha_posterior.kl_divergence(h, prior_concentration, prior_rate)

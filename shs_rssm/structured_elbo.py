"""Structured variational objective helpers for SHS-RSSM.

The fully Bayesian VB objective uses the same local evidence in both the
forward-backward E-step and the dynamics loss:

    ell_t(k) = E_{q(z_t) q(theta_k)}[ log p(z_t | g_t, theta_k) ].

For fixed q(z), the optimal discrete posterior q*(s) is computed by HMM
forward-backward.  The negative dynamics ELBO is

    - ELBO_dyn = - E_q[log p(z|s,theta)] - H[q(z)]
                 + KL(q(s) || p(s))

and, at the optimum q*(s), its sequence sum is exactly

    -logZ - sum_t H[q(z_t)],

where logZ is the forward log-partition with local evidence ell_t(k).
"""
from __future__ import annotations

import math
import torch

LOG2PIE = math.log(2.0 * math.pi * math.e)


def diag_gauss_entropy(var: torch.Tensor) -> torch.Tensor:
    """Entropy of a diagonal Gaussian q(z)=N(mu, diag(var)).

    Args:
        var: (..., L) diagonal variance.
    Returns:
        (...,) entropy in nats.  Continuous entropies may be negative when the
        variance is very small; this is expected.
    """
    var = var.clamp_min(1e-12)
    return 0.5 * (torch.log(var).sum(-1) + var.shape[-1] * LOG2PIE)


def discrete_path_kl_from_marginals(
    gamma: torch.Tensor,
    xi: torch.Tensor,
    log_init: torch.Tensor,
    log_trans: torch.Tensor,
    is_first: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-step KL(q(s_{1:T}) || p(s_{1:T})) from HMM marginals.

    This mirrors RegimeHead._discrete_path_kl but is functional and useful for
    tests. log_trans may be stationary (K,K) or time-varying (B,T-1,K,K).
    """
    B, T, K = gamma.shape
    dtype, device = gamma.dtype, gamma.device
    log_init = log_init.to(dtype=dtype, device=device)
    if is_first is None:
        isf = torch.zeros(B, T, dtype=dtype, device=device)
    else:
        isf = is_first.reshape(B, T).to(dtype=dtype, device=device).clone()
    isf[:, 0] = 1.0

    out = gamma.new_zeros(B, T)
    init_kl = (gamma * (gamma.clamp_min(1e-30).log() - log_init.view(1, 1, K))).sum(-1)
    out = out + isf * init_kl
    if T <= 1:
        return out

    if log_trans.dim() == 2:
        lt = log_trans.to(dtype=dtype, device=device).view(1, 1, K, K).expand(B, T - 1, K, K)
    elif log_trans.dim() == 3:
        lt = log_trans.to(dtype=dtype, device=device).view(1, T - 1, K, K).expand(B, T - 1, K, K)
    else:
        lt = log_trans.to(dtype=dtype, device=device)

    x = xi.clamp_min(1e-30)
    gp = gamma[:, :-1].clamp_min(1e-30)
    log_q_cond = x.log() - gp[:, :, :, None].log()
    pair_kl = (xi * (log_q_cond - lt)).sum(dim=(-2, -1))
    out[:, 1:] = out[:, 1:] + (1.0 - isf[:, 1:]) * pair_kl
    return out


def vb_kl_per_step(
    evidence: torch.Tensor,
    q_var: torch.Tensor,
    gamma: torch.Tensor,
    discrete_kl: torch.Tensor,
) -> torch.Tensor:
    """Per-step fully Bayesian VB dynamics KL.

    evidence is the exact same ell_t(k) used by the E-step.  The returned value
    is
        -sum_k gamma_tk ell_tk - H[q(z_t)] + discrete_kl_t.
    """
    hq = diag_gauss_entropy(q_var)
    cont = -(gamma.detach() * evidence).sum(-1) - hq
    return cont + discrete_kl


def sequence_kl_from_logZ(logZ: torch.Tensor, q_var: torch.Tensor) -> torch.Tensor:
    """Coordinate-optimal sequence KL: -logZ - sum_t H[q(z_t)]."""
    hq_sum = diag_gauss_entropy(q_var).sum(-1)
    return -logZ - hq_sum

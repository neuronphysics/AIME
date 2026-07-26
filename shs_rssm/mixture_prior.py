"""Mixture prior and the Gaussian-to-mixture KL for the DreamerV3 (Mode B) loss.

Given the previous-step regime responsibilities r_{t-1}(k) and the sticky-HDP mean
transitions E[pi_kl], the prior over z_t collapses to a K-component Gaussian mixture
(NOT K^2 components):

    w_{tl} = sum_k r_{t-1}(k) E[pi_kl]
    p_mix(z_t | z_{t-1}, h_t) = sum_l w_{tl} N(z_t; M_l g_t, Qbar_l(g_t)),

with diagonal component covariances Qbar_l(g_t) = (1 + g_t^T V_l g_t) E[q_l] from the
regime model. The encoder posterior q_phi(z_t|h_t,x_t) is diagonal Gaussian, so the
dynamics KL is the Gaussian-to-mixture upper bound

    KL(q || sum_l w_l p_l) <= sum_l lambda_l [ log(lambda_l / w_l) + KL(q || p_l) ],

tight at the LOCAL auxiliary weights lambda_l ∝ w_l exp(-KL(q||p_l)). These lambda
are computed here in closed form alongside the bound and are distinct from the
smoothed forward-backward marginals.

Everything is diagonal, so each KL(q||p_l) is a sum of 1-D Gaussian KLs.
"""
from __future__ import annotations

import math

import torch
import torch.distributions as D

def mixture_weights(resp_prev: torch.Tensor, Epi: torch.Tensor) -> torch.Tensor:
    """w_{tl} = sum_k r_{t-1}(k) E[pi_kl].

    resp_prev : (..., K)   q(s_{t-1}=k)
    Epi       : (K, K)     E[pi_kl] = transTheta_kl / rowsum  (active block)
    returns   : (..., K)   mixture weights over the current regime l (sum to 1 if
                           resp_prev sums to 1 and Epi rows sum to 1).
    """
    return torch.einsum("...k,kl->...l", resp_prev, Epi)


def _diag_gauss_kl(mu_q, var_q, mu_p, var_p):
    """KL( N(mu_q, diag var_q) || N(mu_p, diag var_p) ), summed over the last dim.

    Shapes broadcast; the last dimension is the latent dimension L.
    returns the KL with the last (L) dim reduced.
    """
    # 0.5 * sum_i [ log(var_p/var_q) + (var_q + (mu_q-mu_p)^2)/var_p - 1 ]
    term = (torch.log(var_p) - torch.log(var_q)
            + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
    return 0.5 * term.sum(-1)


def gaussian_to_mixture_kl(q_mean, q_var, comp_mean, comp_var, w,
                           free_bits: float = 0.0, eps: float = 1e-8,
                           return_aux: bool = False):
    """Variational upper bound on KL(q || mixture), with local auxiliary weights.

    q_mean, q_var : (..., L)        diagonal encoder posterior over z_t
    comp_mean     : (..., K, L)     component means M_l g_t
    comp_var      : (..., K, L)     component diagonal covariances Qbar_l(g_t)
    w             : (..., K)        mixture weights w_{tl}
    returns the bound (...,) ; if return_aux, also (lambda (...,K), per_comp_kl (...,K)).

    free_bits clips each per-component KL below at `free_bits` (per the DreamerV3
    free-bits trick applied inside the bound).
    """
    # per-component KL(q || p_l): broadcast q over the K axis
    qm = q_mean.unsqueeze(-2)                                   # (...,1,L)
    qv = q_var.unsqueeze(-2)                                    # (...,1,L)
    per_comp_kl = _diag_gauss_kl(qm, qv, comp_mean, comp_var)  # (...,K)
    if free_bits > 0.0:
        per_comp_kl = torch.clamp(per_comp_kl, min=free_bits)

    logw = torch.log(w.clamp_min(eps))                         # (...,K)
    # Hershey & Olsen (2007) variational bound, single-Gaussian-q special case. Because
    # the q side has ONE component, the optimal auxiliary weights are closed form,
    #     lambda_l = softmax(log w_l - KL(q||p_l)),
    # so no Blahut-Arimoto iteration is needed, and substituting them collapses their
    # §8 upper bound to
    #     KL(q || sum_l w_l p_l)  <=  -logsumexp_l( log w_l - KL(q||p_l) ).
    # This logsumexp expression is their §7 variational APPROXIMATION; for single-Gaussian
    # q it coincides with the §8 upper bound. It is provably >= 0 (sum_l w_l e^{-KL_l} <=
    # sum_l w_l = 1, since each KL_l >= 0), so the previous clamp_min(0) is unnecessary,
    # and it avoids forming lambda explicitly, which is more numerically stable.
    bound = -torch.logsumexp(logw - per_comp_kl, dim=-1)       # (...,)
    if return_aux:
        log_lambda = torch.log_softmax(logw - per_comp_kl, dim=-1)
        return bound, log_lambda.exp(), per_comp_kl
    return bound


def gaussian_to_lowrank_mixture_kl(q_mean, q_var, comp_mean, comp_d, comp_U, w,
                                   free_bits: float = 0.0, eps: float = 1e-8):
    """KL( N(mu_q, diag var_q) || sum_l w_l N(mu_l, diag(d_l) + U_l U_l^T) ).

    Same Hershey & Olsen variational upper bound as gaussian_to_mixture_kl (single-Gaussian-q
    special case, so the optimal mixing weights are closed form and the bound collapses to
    -logsumexp_l(log w_l - KL_l)), but each per-component KL is the FULL low-rank-plus-diagonal
    KL computed with the Woodbury identity and the matrix-determinant lemma, so the off-diagonal
    correlations of the regime process noise are kept. Cost is O(K L r^2 + K r^3), no dense
    L x L inverse. With comp_U of zero columns this reduces exactly to gaussian_to_mixture_kl.

    Shapes: q_* (...,L); comp_mean, comp_d (...,K,L); comp_U (...,K,L,r); w (...,K).
    """
    from .lowrank import lowrank_kl_diag_q
    qm = q_mean.unsqueeze(-2)                                  # (...,1,L) broadcasts over K
    qv = q_var.unsqueeze(-2)
    per_comp_kl = lowrank_kl_diag_q(qm, qv, comp_mean, comp_d, comp_U)  # (...,K)
    if free_bits > 0.0:
        per_comp_kl = torch.clamp(per_comp_kl, min=free_bits)
    logw = torch.log(w.clamp_min(eps))
    return -torch.logsumexp(logw - per_comp_kl, dim=-1)       # (...,)


def mixture_logprob(z, comp_mean, comp_var, w, eps: float = 1e-8):
    """log sum_l w_l N(z; comp_mean_l, diag comp_var_l)  (for diagnostics/imagination).

    z         : (..., L)
    comp_mean : (..., K, L)
    comp_var  : (..., K, L)
    w         : (..., K)
    returns   : (...,)
    """
    import math
    zt = z.unsqueeze(-2)                                        # (...,1,L)
    log_comp = -0.5 * (
        math.log(2 * math.pi)
        + torch.log(comp_var)
        + (zt - comp_mean) ** 2 / comp_var
    ).sum(-1)                                                   # (...,K)
    return torch.logsumexp(torch.log(w.clamp_min(eps)) + log_comp, dim=-1)


def _diag_gauss_logprob(z, mean, var):
    """log N(z; mean, diag var), summed over the last (L) dim. Shapes broadcast."""
    return -0.5 * (math.log(2 * math.pi) + torch.log(var)
                   + (z - mean) ** 2 / var).sum(-1)


@torch.no_grad()
def mixture_kl_monte_carlo(q_mean, q_var, comp_mean, comp_var, w,
                           n_samples: int = 256, eps: float = 1e-8, generator=None):
    """Unbiased Monte Carlo estimate of the TRUE KL(q || sum_l w_l p_l).

    This is the gold-standard reference of Hershey & Olsen (their §2 / DMC): draw
    z ~ q and average log q(z) - log p_mix(z), which converges to the exact divergence as
    n_samples -> infinity. It is for DIAGNOSTICS and TESTS only -- it is noisy and is not
    used in the training gradient (the variational bound in gaussian_to_mixture_kl is). Its
    purpose is to quantify how loose the variational UPPER bound is on real data: the bound
    should sit at or above this estimate, and the gap is small when the mixture components
    are well separated (the optimal lambda concentrates on one component) and grows when
    they overlap.

    Shapes match gaussian_to_mixture_kl: q_* are (...,L), comp_* are (...,K,L), w is
    (...,K). Returns (...,), the estimated KL per element.
    """
    qm = q_mean.unsqueeze(0)                                    # (1,...,L)
    qsd = q_var.clamp_min(eps).sqrt().unsqueeze(0)
    shape = (n_samples,) + tuple(q_mean.shape)
    noise = torch.randn(shape, dtype=q_mean.dtype, device=q_mean.device, generator=generator)
    z = qm + qsd * noise                                       # (n,...,L)
    log_q = _diag_gauss_logprob(z, qm, q_var.unsqueeze(0))     # (n,...)
    log_p = mixture_logprob(z, comp_mean.unsqueeze(0), comp_var.unsqueeze(0),
                            w.unsqueeze(0), eps=eps)            # (n,...)
    return (log_q - log_p).mean(0)                             # (...,)


@torch.no_grad()
def mixture_entropy_bounds(comp_mean, comp_var, w, eps: float = 1e-8):
    """DETERMINISTIC lower and upper bounds on the entropy of a diagonal-Gaussian
    mixture p(z) = sum_k w_k N(mu_k, diag(v_k)); shapes (...,K,L) and (...,K).

    Upper:  H(p) <= H(w) + sum_k w_k H(N_k)          (joint-entropy bound)
    Lower:  H(p) >= -sum_i w_i log sum_j w_j c_ij,   c_ij = N(mu_i; mu_j, v_i+v_j)
            (Jensen inside -E_{p_i} log p; the Hershey-Olsen pairwise bound)

    Both are closed form -- no sampling. Use these for logged diagnostics instead
    of the Monte Carlo estimator; the true entropy lies in [lower, upper]. The
    lower bound carries an irreducible constant slack of (L/2)(1 - ln 2) nats per
    effective component (it is exactly that loose even for a single Gaussian);
    it is nonetheless a valid deterministic lower bound for any diagonal mixture.
    """
    w = w.clamp_min(eps)
    w = w / w.sum(-1, keepdim=True)
    L = comp_mean.shape[-1]
    # upper: H(w) + sum w_k * 0.5 * sum_l log(2 pi e v_kl)
    Hw = -(w * w.log()).sum(-1)
    Hk = 0.5 * (math.log(2.0 * math.pi * math.e) * L
                + torch.log(comp_var.clamp_min(eps)).sum(-1))          # (...,K)
    upper = Hw + (w * Hk).sum(-1)
    # lower: pairwise convolutions c_ij (diagonal)
    mu_i = comp_mean.unsqueeze(-2)                                      # (...,K,1,L)
    mu_j = comp_mean.unsqueeze(-3)                                      # (...,1,K,L)
    v_ij = comp_var.unsqueeze(-2) + comp_var.unsqueeze(-3)              # (...,K,K,L)
    log_c = -0.5 * (math.log(2.0 * math.pi) * L
                    + torch.log(v_ij.clamp_min(eps)).sum(-1)
                    + ((mu_i - mu_j) ** 2 / v_ij.clamp_min(eps)).sum(-1))   # (...,K,K)
    inner = torch.logsumexp(w.unsqueeze(-2).clamp_min(eps).log() + log_c, dim=-1)  # (...,K)
    lower = -(w * inner).sum(-1)
    return lower, upper


def mixture_kl_variational_bound(q_mean, q_var, comp_mean, comp_var, w, eps: float = 1e-8):
    """Deterministic variational UPPER bound on KL(q || sum_k w_k N_k), no sampling.
    This is the diagonal special case of `gaussian_to_mixture_kl` (single source of
    truth for the -log sum_k w_k exp(-KL(q||p_k)) Hershey-Olsen bound); kept as a named
    entry point. Reduces to KL(q||N_j) when w is one-hot; MC-free replacement for
    mixture_kl_monte_carlo wherever an ELBO needs a mixture KL (round-8 review, issue 4)."""
    return gaussian_to_mixture_kl(q_mean, q_var, comp_mean, comp_var, w, eps=eps)


def mixture_entropy_monte_carlo(comp_mean, comp_var, w, n_samples: int = 64,
                                eps: float = 1e-8, generator=None):
    """Monte Carlo estimate of the TRUE entropy H(sum_l w_l N_l) of the mixture prior.

    There is no closed form for a Gaussian-mixture entropy (this is the regime the
    Hershey & Olsen paper addresses). The moment-matched single Gaussian that
    `get_dist(prior).entropy()` reports is an UPPER bound on this (a Gaussian maximises
    entropy for a given mean and covariance), so the gap H_gauss - H_mix is a regime
    multi-modality signal: ~0 when the prior is effectively unimodal, large when the
    regimes are well separated. For diagnostics/logging only.

    comp_mean, comp_var : (...,K,L) ;  w : (...,K).  Returns (...,) entropy per element.
    """
    
    mix = D.Categorical(probs=w.clamp_min(eps))
    comp = D.Independent(D.Normal(comp_mean, comp_var.clamp_min(eps).sqrt()), 1)
    gmm = D.MixtureSameFamily(mix, comp)
    z = gmm.sample((n_samples,))                              # (n,...,L)
    return -gmm.log_prob(z).mean(0)                           # (...,)


@torch.no_grad()
def mixture_entropy_monte_carlo_lowrank(comp_mean, comp_d, comp_U, w,
                                        n_samples: int = 64, eps: float = 1e-8):
    """Monte Carlo entropy of a mixture with low-rank-plus-diagonal components,
    sum_l w_l N(mu_l, diag(d_l) + U_l U_l^T). Samples a component then draws from its full
    covariance, and scores log p_mix with the Woodbury log-density. Reduces to the diagonal
    estimate when U has zero columns.

    comp_mean, comp_d : (...,K,L) ; comp_U : (...,K,L,r) ; w : (...,K). Returns (...,).
    """
    from .lowrank import lowrank_logpdf
    L = comp_mean.shape[-1]
    r = comp_U.shape[-1]
    idx = torch.distributions.Categorical(probs=w.clamp_min(eps)).sample((n_samples,))  # (n,...)
    sel = idx.unsqueeze(-1).unsqueeze(-1).expand(*idx.shape, 1, L)
    cm = comp_mean.unsqueeze(0).expand(n_samples, *comp_mean.shape)
    cd = comp_d.unsqueeze(0).expand(n_samples, *comp_d.shape)
    smean = torch.gather(cm, -2, sel).squeeze(-2)             # (n,...,L)
    sd = torch.gather(cd, -2, sel).squeeze(-2)
    z = smean + sd.clamp_min(eps).sqrt() * torch.randn_like(smean)
    if r > 0:
        cU = comp_U.unsqueeze(0).expand(n_samples, *comp_U.shape)
        selU = idx.reshape(*idx.shape, 1, 1, 1).expand(*idx.shape, 1, L, r)
        sU = torch.gather(cU, -3, selU).squeeze(-3)           # (n,...,L,r)
        epsr = torch.randn(*smean.shape[:-1], r, device=smean.device, dtype=smean.dtype)
        z = z + torch.einsum("...lr,...r->...l", sU, epsr)
    log_comp = lowrank_logpdf(z.unsqueeze(-2), comp_mean.unsqueeze(0),
                              comp_d.unsqueeze(0), comp_U.unsqueeze(0))   # (n,...,K)
    log_p = torch.logsumexp(torch.log(w.clamp_min(eps)).unsqueeze(0) + log_comp, dim=-1)
    return -log_p.mean(0)                                     # (...,)

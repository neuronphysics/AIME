from __future__ import annotations

import math

import torch
import torch.distributions as D

def mixture_weights(resp_prev: torch.Tensor, Epi: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...k,kl->...l", resp_prev, Epi)


def _diag_gauss_kl(mu_q, var_q, mu_p, var_p):
    term = (torch.log(var_p) - torch.log(var_q)
            + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
    return 0.5 * term.sum(-1)


def gaussian_to_mixture_kl(q_mean, q_var, comp_mean, comp_var, w,
                           free_bits: float = 0.0, eps: float = 1e-8,
                           return_aux: bool = False):
    qm = q_mean.unsqueeze(-2)
    qv = q_var.unsqueeze(-2)
    per_comp_kl = _diag_gauss_kl(qm, qv, comp_mean, comp_var)
    if free_bits > 0.0:
        per_comp_kl = torch.clamp(per_comp_kl, min=free_bits)

    logw = torch.log(w.clamp_min(eps))
    bound = -torch.logsumexp(logw - per_comp_kl, dim=-1)
    if return_aux:
        log_lambda = torch.log_softmax(logw - per_comp_kl, dim=-1)
        return bound, log_lambda.exp(), per_comp_kl
    return bound


def gaussian_to_lowrank_mixture_kl(q_mean, q_var, comp_mean, comp_d, comp_U, w,
                                   free_bits: float = 0.0, eps: float = 1e-8):
    from .lowrank import lowrank_kl_diag_q
    qm = q_mean.unsqueeze(-2)
    qv = q_var.unsqueeze(-2)
    per_comp_kl = lowrank_kl_diag_q(qm, qv, comp_mean, comp_d, comp_U)
    if free_bits > 0.0:
        per_comp_kl = torch.clamp(per_comp_kl, min=free_bits)
    logw = torch.log(w.clamp_min(eps))
    return -torch.logsumexp(logw - per_comp_kl, dim=-1)


def mixture_logprob(z, comp_mean, comp_var, w, eps: float = 1e-8):
    import math
    zt = z.unsqueeze(-2)
    log_comp = -0.5 * (
        math.log(2 * math.pi)
        + torch.log(comp_var)
        + (zt - comp_mean) ** 2 / comp_var
    ).sum(-1)
    return torch.logsumexp(torch.log(w.clamp_min(eps)) + log_comp, dim=-1)


def _diag_gauss_logprob(z, mean, var):
    return -0.5 * (math.log(2 * math.pi) + torch.log(var)
                   + (z - mean) ** 2 / var).sum(-1)


@torch.no_grad()
def mixture_kl_monte_carlo(q_mean, q_var, comp_mean, comp_var, w,
                           n_samples: int = 256, eps: float = 1e-8, generator=None):
    qm = q_mean.unsqueeze(0)
    qsd = q_var.clamp_min(eps).sqrt().unsqueeze(0)
    shape = (n_samples,) + tuple(q_mean.shape)
    noise = torch.randn(shape, dtype=q_mean.dtype, device=q_mean.device, generator=generator)
    z = qm + qsd * noise
    log_q = _diag_gauss_logprob(z, qm, q_var.unsqueeze(0))
    log_p = mixture_logprob(z, comp_mean.unsqueeze(0), comp_var.unsqueeze(0),
                            w.unsqueeze(0), eps=eps)
    return (log_q - log_p).mean(0)


@torch.no_grad()
def mixture_entropy_bounds(comp_mean, comp_var, w, eps: float = 1e-8):
    w = w.clamp_min(eps)
    w = w / w.sum(-1, keepdim=True)
    L = comp_mean.shape[-1]
    Hw = -(w * w.log()).sum(-1)
    Hk = 0.5 * (math.log(2.0 * math.pi * math.e) * L
                + torch.log(comp_var.clamp_min(eps)).sum(-1))
    upper = Hw + (w * Hk).sum(-1)
    mu_i = comp_mean.unsqueeze(-2)
    mu_j = comp_mean.unsqueeze(-3)
    v_ij = comp_var.unsqueeze(-2) + comp_var.unsqueeze(-3)
    log_c = -0.5 * (math.log(2.0 * math.pi) * L
                    + torch.log(v_ij.clamp_min(eps)).sum(-1)
                    + ((mu_i - mu_j) ** 2 / v_ij.clamp_min(eps)).sum(-1))
    inner = torch.logsumexp(w.unsqueeze(-2).clamp_min(eps).log() + log_c, dim=-1)
    lower = -(w * inner).sum(-1)
    return lower, upper


def mixture_kl_variational_bound(q_mean, q_var, comp_mean, comp_var, w, eps: float = 1e-8):
    return gaussian_to_mixture_kl(q_mean, q_var, comp_mean, comp_var, w, eps=eps)


def mixture_entropy_monte_carlo(comp_mean, comp_var, w, n_samples: int = 64,
                                eps: float = 1e-8, generator=None):
    
    mix = D.Categorical(probs=w.clamp_min(eps))
    comp = D.Independent(D.Normal(comp_mean, comp_var.clamp_min(eps).sqrt()), 1)
    gmm = D.MixtureSameFamily(mix, comp)
    z = gmm.sample((n_samples,))
    return -gmm.log_prob(z).mean(0)


@torch.no_grad()
def mixture_entropy_monte_carlo_lowrank(comp_mean, comp_d, comp_U, w,
                                        n_samples: int = 64, eps: float = 1e-8):
    from .lowrank import lowrank_logpdf
    L = comp_mean.shape[-1]
    r = comp_U.shape[-1]
    idx = torch.distributions.Categorical(probs=w.clamp_min(eps)).sample((n_samples,))
    sel = idx.unsqueeze(-1).unsqueeze(-1).expand(*idx.shape, 1, L)
    cm = comp_mean.unsqueeze(0).expand(n_samples, *comp_mean.shape)
    cd = comp_d.unsqueeze(0).expand(n_samples, *comp_d.shape)
    smean = torch.gather(cm, -2, sel).squeeze(-2)
    sd = torch.gather(cd, -2, sel).squeeze(-2)
    z = smean + sd.clamp_min(eps).sqrt() * torch.randn_like(smean)
    if r > 0:
        cU = comp_U.unsqueeze(0).expand(n_samples, *comp_U.shape)
        selU = idx.reshape(*idx.shape, 1, 1, 1).expand(*idx.shape, 1, L, r)
        sU = torch.gather(cU, -3, selU).squeeze(-3)
        epsr = torch.randn(*smean.shape[:-1], r, device=smean.device, dtype=smean.dtype)
        z = z + torch.einsum("...lr,...r->...l", sU, epsr)
    log_comp = lowrank_logpdf(z.unsqueeze(-2), comp_mean.unsqueeze(0),
                              comp_d.unsqueeze(0), comp_U.unsqueeze(0))
    log_p = torch.logsumexp(torch.log(w.clamp_min(eps)).unsqueeze(0) + log_comp, dim=-1)
    return -log_p.mean(0)

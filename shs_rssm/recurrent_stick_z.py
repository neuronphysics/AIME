"""Latent-conditioned stickiness gate:  phi_t = [z_{t-1}, a_{t-1}, 1].

The shipped gate reads phi_t = [tanh(P h_t), a_{t-1}, 1], which is a CONSTANT
with respect to the continuous latent.  That keeps inference cheap but severs
the z -> s edge whenever the deterministic carry is uninformative.

This module replaces h_t with z_{t-1} and drops the tanh.  Dropping the
nonlinearity is what keeps everything closed form: with phi linear in z and
q(z_{t-1}) = N(mu, diag(s2)), the two moments E[phi] and E[phi phi^T] are
available exactly, so

  * the Polya-Gamma sufficient statistics stay conjugate (no Monte Carlo), and
  * the transition potential contributes an exactly Gaussian term to the
    continuous-state Hessian.

With the tanh retained neither expectation has a closed form.

Notation (matching RecurrentStickiness):
    psi_{t,k} = beta_k^T phi_t,      beta_k ~ N(m_k, Sigma_k)      [q(beta)]
    rho_{k,t} = sigma(psi_{t,k})                                   [persistence]
    W_t[k,:]  = rho_{k,t} delta_k + (1 - rho_{k,t}) pi_k

Both beta and phi are random, so
    E[psi]   = m_k^T phibar
    Var[psi] = tr( E[beta beta^T] E[phi phi^T] ) - (m_k^T phibar)^2

E[phi phi^T] = phibar phibar^T + S, where S is diag(s2) padded into the
leading L x L block (the action and bias entries are deterministic), so the
trace never needs the (D_phi x D_phi) outer product materialised.
"""
from __future__ import annotations

import torch


def phi_moments(z_mean, z_var, action=None):
    """First two moments of phi_t = [z_{t-1}, a_{t-1}, 1] under q(z_{t-1}).

    z_mean, z_var: (..., L) posterior mean / diagonal variance of z_{t-1}
    action:        (..., A) or None

    Returns
        phibar : (..., D_phi)            E[phi]
        s2pad  : (..., D_phi)            diagonal of Cov[phi] (zero outside z)
    """
    L = z_mean.shape[-1]
    parts = [z_mean]
    vparts = [z_var if z_var is not None else torch.zeros_like(z_mean)]
    if action is not None:
        parts.append(action.to(z_mean.dtype))
        vparts.append(torch.zeros_like(action, dtype=z_mean.dtype))
    ones = z_mean[..., :1] * 0.0 + 1.0
    parts.append(ones)
    vparts.append(torch.zeros_like(ones))
    return torch.cat(parts, -1), torch.cat(vparts, -1)


def psi_moments(phibar, s2pad, m_beta, Sigma_beta):
    """E[psi] and Var[psi] with BOTH beta and phi random.

    m_beta:     (K, D)          Sigma_beta: (K, D, D)
    Returns m, v each (..., K).
    """
    EbbT = Sigma_beta + torch.einsum("kd,ke->kde", m_beta, m_beta)   # (K,D,D)
    m = torch.einsum("...d,kd->...k", phibar, m_beta)
    # tr(EbbT (phibar phibar^T + diag(s2))) = phibar^T EbbT phibar + sum_d EbbT_dd s2_d
    quad = torch.einsum("...d,kde,...e->...k", phibar, EbbT, phibar)
    diag = torch.einsum("kdd,...d->...k", EbbT, s2pad)
    v = (quad + diag - m * m).clamp_min(0.0)
    return m, v


def jj_potentials(m, v):
    """Jaakkola-Jordan branch potentials, identical in form to the shipped gate."""
    c = torch.sqrt((m * m + v).clamp_min(1e-12))
    const = torch.nn.functional.logsigmoid(c) - 0.5 * c
    return const + 0.5 * m, const - 0.5 * m, m, c


def expected_omega(c, row_weight):
    """E[omega] for omega ~ PG(row_weight, c) = w * tanh(c/2) / (2c).

    Continuous at c -> 0, where the limit is w/4."""
    c = c.abs()
    small = c < 1e-6
    cs = torch.where(small, torch.ones_like(c), c)
    val = (0.5 / cs) * torch.tanh(0.5 * cs)
    val = torch.where(small, torch.full_like(val, 0.25), val)
    return val * row_weight


def pg_stats(phibar, s2pad, r_mass, row_weight, m_beta, Sigma_beta):
    """Conjugate PG natural-parameter statistics, in closed form.

        A_k = sum_{n,t} E[omega_{ntk}] E[phi phi^T]
        h_k = sum_{n,t} (r_{ntk} - w_{ntk}/2) E[phi]

    Shapes: phibar,s2pad (N,D); r_mass,row_weight (N,K).  Returns A (K,D,D), h (K,D).
    """
    m, v = psi_moments(phibar, s2pad, m_beta, Sigma_beta)
    _, _, _, c = jj_potentials(m, v)
    Eom = expected_omega(c, row_weight)                      # (N,K)
    # E[phi phi^T] = phibar phibar^T + diag(s2pad); both terms accumulate cheaply
    A = torch.einsum("nk,nd,ne->kde", Eom, phibar, phibar)
    A = A + torch.diag_embed(torch.einsum("nk,nd->kd", Eom, s2pad))
    h = torch.einsum("nk,nd->kd", r_mass - 0.5 * row_weight, phibar)
    return dict(A=A, h=h, Eom=Eom, m=m, v=v, c=c)


def hessian_blocks(gamma, Eom, m_beta, Sigma_beta, L):
    """Contribution of the transition potential to the continuous-state precision.

    Under PG augmentation the potential is  kappa*psi - omega*psi^2/2  with
    psi = beta^T phi LINEAR in z_{t-1}, so

        -d2/dz2  =  sum_k q(z_t = k) E[omega_{tk}] E[b_k b_k^T],
        b_k = beta_k restricted to its z-block.

    Because psi depends on a SINGLE time index, this lands only on the primary
    block diagonal -- the block-tridiagonal structure of the smoother survives.

    gamma: (N,K) responsibilities;  Eom: (N,K).  Returns (N,L,L), PSD.
    """
    Ebb = (Sigma_beta[:, :L, :L]
           + torch.einsum("kd,ke->kde", m_beta[:, :L], m_beta[:, :L]))   # (K,L,L)
    return torch.einsum("nk,nk,kde->nde", gamma, Eom, Ebb)


# --------------------------------------------------------------------------
# reference implementations used only for verification
# --------------------------------------------------------------------------
@torch.no_grad()
def psi_moments_mc(z_mean, z_var, m_beta, Sigma_beta, action=None, S=200_000,
                   seed=0):
    """Monte-Carlo E[psi], Var[psi] over BOTH q(z) and q(beta)."""
    g = torch.Generator().manual_seed(seed)
    L, K = z_mean.shape[-1], m_beta.shape[0]
    zs = z_mean + z_var.sqrt() * torch.randn(S, *z_mean.shape, generator=g)
    Lb = torch.linalg.cholesky(Sigma_beta + 1e-9 * torch.eye(m_beta.shape[-1]))
    eps = torch.randn(S, K, m_beta.shape[-1], generator=g)
    bs = m_beta.unsqueeze(0) + torch.einsum("skd,ked->ske", eps, Lb)
    phis, _ = phi_moments(zs, torch.zeros_like(zs),
                          None if action is None else action.expand(S, *action.shape))
    psi = torch.einsum("s...d,skd->s...k", phis, bs)
    return psi.mean(0), psi.var(0, unbiased=False)


@torch.no_grad()
def hessian_mc(gamma, z_mean, z_var, m_beta, Sigma_beta, row_weight,
               action=None, S=20_000, eps=1e-3, seed=0):
    """Finite-difference check of the Hessian block on the first sample."""
    L = z_mean.shape[-1]

    pb0, s20 = phi_moments(z_mean, z_var, action)
    m0, v0 = psi_moments(pb0, s20, m_beta, Sigma_beta)
    _, _, _, c0 = jj_potentials(m0, v0)
    Eom = expected_omega(c0, row_weight)        # auxiliary variable: held FIXED

    def pot(zm):
        pb, s2 = phi_moments(zm, z_var, action)
        m, v = psi_moments(pb, s2, m_beta, Sigma_beta)
        return -(0.5 * Eom * (m * m + v) * gamma).sum()

    H = torch.zeros(L, L)
    for i in range(L):
        for j in range(L):
            zpp = z_mean.clone(); zpp[..., i] += eps; zpp[..., j] += eps
            zpm = z_mean.clone(); zpm[..., i] += eps; zpm[..., j] -= eps
            zmp = z_mean.clone(); zmp[..., i] -= eps; zmp[..., j] += eps
            zmm = z_mean.clone(); zmm[..., i] -= eps; zmm[..., j] -= eps
            H[i, j] = (pot(zpp) - pot(zpm) - pot(zmp) + pot(zmm)) / (4 * eps * eps)
    return -H

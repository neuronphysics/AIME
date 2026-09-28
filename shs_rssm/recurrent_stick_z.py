from __future__ import annotations

import torch


def phi_moments(z_mean, z_var, action=None):
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
    EbbT = Sigma_beta + torch.einsum("kd,ke->kde", m_beta, m_beta)
    m = torch.einsum("...d,kd->...k", phibar, m_beta)
    quad = torch.einsum("...d,kde,...e->...k", phibar, EbbT, phibar)
    diag = torch.einsum("kdd,...d->...k", EbbT, s2pad)
    v = (quad + diag - m * m).clamp_min(0.0)
    return m, v


def jj_potentials(m, v):
    """Jaakkola-Jordan branch potentials for the latent-input gate."""
    c = torch.sqrt((m * m + v).clamp_min(1e-12))
    const = torch.nn.functional.logsigmoid(c) - 0.5 * c
    return const + 0.5 * m, const - 0.5 * m, m, c


def expected_omega(c, row_weight):
    """E[omega] for omega ~ PG(row_weight, c)."""
    return (0.5 / c.clamp_min(1e-12)) * torch.tanh(0.5 * c) * row_weight


def pg_stats(phibar, s2pad, r_mass, row_weight, m_beta, Sigma_beta):
    m, v = psi_moments(phibar, s2pad, m_beta, Sigma_beta)
    _, _, _, c = jj_potentials(m, v)
    Eom = expected_omega(c, row_weight)
    A = torch.einsum("nk,nd,ne->kde", Eom, phibar, phibar)
    A = A + torch.diag_embed(torch.einsum("nk,nd->kd", Eom, s2pad))
    h = torch.einsum("nk,nd->kd", r_mass - 0.5 * row_weight, phibar)
    return dict(A=A, h=h, Eom=Eom, m=m, v=v, c=c)


def hessian_blocks(gamma, Eom, m_beta, Sigma_beta, L):
    Ebb = (Sigma_beta[:, :L, :L]
           + torch.einsum("kd,ke->kde", m_beta[:, :L], m_beta[:, :L]))
    return torch.einsum("nk,nk,kde->nde", gamma, Eom, Ebb)


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
    Eom = expected_omega(c0, row_weight)

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

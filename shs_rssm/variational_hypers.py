from __future__ import annotations

import contextlib
import torch


@torch.no_grad()
def calibrate_prior(reg, scale, sF=0.01, lam0_scale=1.0, ard_scale=None):
    scale = torch.as_tensor(scale, dtype=reg.b0.dtype, device=reg.b0.device).reshape(-1)
    if scale.numel() != int(reg.L):
        raise ValueError(f"scale must have L={int(reg.L)} entries, got {scale.numel()}")
    a0 = float(reg.a0)
    if a0 <= 1.0:
        raise ValueError(f"a0={a0} <= 1: E[Q] is undefined, raise a0 (bnpy uses nu0>=D+2)")
    b0_new = sF * (a0 - 1.0) * scale.clamp_min(1e-12)
    if reg.b0.shape != b0_new.shape:
        raise RuntimeError(
            f"regimes.b0 has shape {tuple(reg.b0.shape)} but calibration produces "
            f"{tuple(b0_new.shape)}. Apply the two-line edit in README_PATCH.md.")
    reg.b0.copy_(b0_new)
    if ard_scale is not None:
        reg.lam0_diag.copy_(torch.as_tensor(ard_scale).to(reg.lam0_diag).reshape(-1))
    else:
        reg.lam0_diag.fill_(float(lam0_scale))
    return dict(b0=reg.b0.clone(), lam0_diag=reg.lam0_diag.clone(), sF=float(sF))


@torch.no_grad()
def latent_scale(buffer, mode="var", eps=1e-8):
    acc, n = None, 0
    for b in getattr(buffer, "batches", buffer):
        z = b.stoch if hasattr(b, "stoch") else b[1]
        z = z.reshape(-1, z.shape[-1]).double()
        if mode == "diff":
            z = z[1:] - z[:-1]
        s = z.var(0, unbiased=False) * z.shape[0]
        acc = s if acc is None else acc + s
        n += z.shape[0]
    return (acc / max(n, 1)).clamp_min(eps)


@torch.no_grad()
def hyper_stats(reg):
    a, b = reg.a, reg.b
    E_tau = a / b.clamp_min(1e-30)
    E_logtau = torch.digamma(a) - b.clamp_min(1e-30).log()
    dM = reg.M - reg.M0.unsqueeze(0)
    trV = reg.V.diagonal(dim1=-2, dim2=-1)
    S_w = float((E_tau * dM.pow(2).sum(-1)).sum() + reg.L * trV.sum())
    S_w_g = (E_tau.unsqueeze(-1) * dM.pow(2)).sum(dim=(0, 1)) + reg.L * trV.sum(0)
    return dict(S_tau=E_tau.sum(0), S_logtau=E_logtau.sum(0), S_w=S_w, S_w_g=S_w_g,
                K=int(reg.K), L=int(reg.L), G=int(reg.G))


@torch.no_grad()
def prior_mismatch(reg):
    S = hyper_stats(reg)
    want = S["K"] * float(reg.a0) / S["S_tau"].clamp_min(1e-30)
    return (reg.b0 / want.clamp_min(1e-30))


@torch.no_grad()
def eb_update(reg, per_dim_b0=True, ard=False, update_a0=False, damp=1.0,
              b0_floor=1e-8, lam0_floor=1e-8, _i_know_this_is_ml=False):
    if not _i_know_this_is_ml:
        raise RuntimeError(
            "eb_update is type-II maximum likelihood, not variational inference. "
            "Use calibrate_prior() instead. If you want EB as a comparison arm, "
            "pass _i_know_this_is_ml=True and wrap every sweep in frozen_hypers().")
    if getattr(reg, "_hypers_frozen", False):
        return None
    S = hyper_stats(reg)
    K, L, G = S["K"], S["L"], S["G"]
    a0 = float(reg.a0)
    b0_new = (K * a0 / S["S_tau"].clamp_min(1e-30))
    if not per_dim_b0:
        b0_new = b0_new.new_full((L,), K * L * a0 / float(S["S_tau"].sum().clamp_min(1e-30)))
    b0_new = b0_new.clamp_min(b0_floor)
    lam0_new = ((K * L / S["S_w_g"].clamp_min(1e-30)).clamp_min(lam0_floor) if ard
                else torch.full_like(reg.lam0_diag,
                                     max(K * L * G / max(S["S_w"], 1e-30), lam0_floor)))
    if update_a0:
        target = float((b0_new.log() + S["S_logtau"] / K).mean())
        a = a0
        for _ in range(50):
            f = torch.digamma(torch.tensor(a)).item() - target
            if abs(f) < 1e-10:
                break
            a = max(a - f / torch.polygamma(1, torch.tensor(a)).item(), 1e-3)
        reg.a0.fill_(damp * a + (1 - damp) * a0)
    if reg.b0.shape != b0_new.shape:
        raise RuntimeError(f"regimes.b0 shape {tuple(reg.b0.shape)} != {tuple(b0_new.shape)}")
    reg.b0.copy_(damp * b0_new + (1 - damp) * reg.b0)
    reg.lam0_diag.copy_(damp * lam0_new + (1 - damp) * reg.lam0_diag)
    return dict(a0=float(reg.a0), b0=reg.b0.clone(), lam0_diag=reg.lam0_diag.clone())


@torch.no_grad()
def hierarchical_update(reg, c0=1e-3, d0=1e-3, e0=1e-3, f0=1e-3, per_dim_b0=True):
    S = hyper_stats(reg)
    K, L, G = S["K"], S["L"], S["G"]
    a0 = float(reg.a0)
    c = torch.full_like(reg.b0, c0 + K * a0)
    d = d0 + S["S_tau"]
    if not per_dim_b0:
        c = torch.full_like(reg.b0, c0 + K * L * a0)
        d = torch.full_like(reg.b0, d0 + float(S["S_tau"].sum()))
    e, f = e0 + 0.5 * K * L * G, f0 + 0.5 * S["S_w"]
    reg._hyp_b0 = (c, d, torch.tensor(float(c0)), torch.tensor(float(d0)))
    reg._hyp_v0 = (torch.tensor(float(e)), torch.tensor(float(f)),
                   torch.tensor(float(e0)), torch.tensor(float(f0)))
    reg.b0.copy_(c / d)
    reg.lam0_diag.fill_(float(e / f))
    return dict(b0=reg.b0.clone(), v0=float(e / f))


def _kl_gamma(c, d, c0, d0):
    return ((c - c0) * torch.digamma(c) - torch.lgamma(c) + torch.lgamma(c0)
            + c0 * (d.log() - d0.log()) + c * (d0 - d) / d)


@torch.no_grad()
def hyperprior_kl(reg):
    tot = torch.zeros((), dtype=reg.b0.dtype, device=reg.b0.device)
    if hasattr(reg, "_hyp_b0"):
        c, d, c0, d0 = reg._hyp_b0
        tot = tot + _kl_gamma(c, d, c0.to(c), d0.to(d)).sum()
    if hasattr(reg, "_hyp_v0"):
        e, f, e0, f0 = reg._hyp_v0
        tot = tot + _kl_gamma(e.to(tot), f.to(tot), e0.to(tot), f0.to(tot))
    return tot


@contextlib.contextmanager
def frozen_hypers(*regs):
    prev = [getattr(r, "_hypers_frozen", False) for r in regs]
    for r in regs:
        r._hypers_frozen = True
    try:
        yield
    finally:
        for r, p in zip(regs, prev):
            r._hypers_frozen = p

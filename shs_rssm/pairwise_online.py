"""Online pairwise accumulation for recurrent SHS training.

Recomputes each step's pairwise responsibility q(s_{t-1}, s_t) ONE STEP AT A TIME from the
forward/backward messages and reduces it immediately to the quantities training needs --
the Polya-Gamma persistence stats (r_mass, row_weight per step; Cbase over all steps) and
the per-step discrete-path KL -- so the full O(B T K^2) xi tensor is never allocated
(transient O(B K^2) per step). Produces results identical to reducing a materialised xi.
"""
from __future__ import annotations

import torch


def _resolve_trans(log_trans, t, B, K):
    """Review P2 #11: accept EITHER a full (...,K,K) tensor (indexed) OR a callable
    trans_fn(t)->(B,K,K) that BUILDS the slice on demand, so the full O(BTK^2) tensor
    need never be materialised in the online PG/KL scans."""
    if callable(log_trans):
        return log_trans(t)
    return _trans_at(log_trans, t, B, K)


def _trans_at(log_trans, t, B, K):
    if log_trans.dim() == 2:
        return log_trans.unsqueeze(0).expand(B, K, K)
    if log_trans.dim() == 3:
        return log_trans[t - 1].unsqueeze(0).expand(B, K, K)
    return log_trans[:, t - 1]


@torch.no_grad()
def accumulate_pg_online(log_alpha, log_beta, logZ, log_trans_det, ev, is_first, valid, aux):
    """PG attribution accumulated online (all DETACHED EM quantities). Returns
    r_mass (B,S,K), row_weight (B,S,K), Cbase (K,K) -- identical to attribute_bound on a
    materialised xi, but without storing it."""
    B, T, K = ev.shape
    S = T - 1
    A = aux["A"]                                   # (B,S,K)
    Bd = aux["switch_diag"]                        # (B,S,K)
    w1_frac = torch.exp(A - torch.logaddexp(A, Bd))
    isf = (torch.zeros(B, T, dtype=ev.dtype, device=ev.device) if is_first is None
           else is_first.reshape(B, T).to(ev.dtype).clone())
    isf[:, 0] = 1.0
    val = torch.ones(B, T, dtype=ev.dtype, device=ev.device) if valid is None \
        else valid.reshape(B, T).to(ev.dtype)
    r_mass = ev.new_zeros(B, S, K)
    row_weight = ev.new_zeros(B, S, K)
    Cbase = ev.new_zeros(K, K)
    for t in range(1, T):
        Tt = _resolve_trans(log_trans_det, t, B, K)
        lx = (log_alpha[:, t - 1].unsqueeze(-1) + Tt
              + (ev[:, t] + log_beta[:, t]).unsqueeze(1) - logZ.view(B, 1, 1))
        x = lx.exp() * ((1.0 - isf[:, t]) * val[:, t]).view(B, 1, 1)   # (B,K,K)
        diag_x = torch.diagonal(x, dim1=-2, dim2=-1)                   # (B,K)
        w1 = w1_frac[:, t - 1]
        r_mass[:, t - 1] = diag_x * w1
        row_weight[:, t - 1] = x.sum(-1)
        newdiag = diag_x * (1.0 - w1)
        Cbase = Cbase + (x - torch.diag_embed(diag_x) + torch.diag_embed(newdiag)).sum(0)
    return r_mass, row_weight, Cbase


def pair_kl_online(log_alpha, log_beta, logZ, log_trans_det, ev, is_first, valid,
                   gamma, log_trans_diff):
    """Per-step discrete-path pairwise KL accumulated online. The pairwise q (from the
    DETACHED messages) is detached; `log_trans_diff` is the DIFFERENTIABLE transition, so
    the stickiness projection still receives its transition gradient. Returns (B,S)."""
    B, T, K = ev.shape
    isf = (torch.zeros(B, T, dtype=ev.dtype, device=ev.device) if is_first is None
           else is_first.reshape(B, T).to(ev.dtype).clone())
    isf[:, 0] = 1.0
    val = torch.ones(B, T, dtype=ev.dtype, device=ev.device) if valid is None \
        else valid.reshape(B, T).to(ev.dtype)
    out = gamma.new_zeros(B, T - 1)
    for t in range(1, T):
        Tt = _resolve_trans(log_trans_det, t, B, K)
        with torch.no_grad():
            lx = (log_alpha[:, t - 1].unsqueeze(-1) + Tt
                  + (ev[:, t] + log_beta[:, t]).unsqueeze(1) - logZ.view(B, 1, 1))
            x = (lx.exp() * ((1.0 - isf[:, t]) * val[:, t]).view(B, 1, 1))  # detached q
        gp = gamma[:, t - 1].clamp_min(1e-30)
        ltd = (log_trans_diff(t) if callable(log_trans_diff)
               else log_trans_diff[:, t - 1])                                # differentiable
        log_q_cond = x.clamp_min(1e-30).log() - gp[:, :, None].log()
        out[:, t - 1] = (x * (log_q_cond - ltd)).sum(dim=(-2, -1))
    return out

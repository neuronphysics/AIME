from __future__ import annotations

import torch


def _resolve_trans(log_trans, t, B, K):
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
    B, T, K = ev.shape
    S = T - 1
    A = aux["A"]
    Bd = aux["switch_diag"]
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
        x = lx.exp() * ((1.0 - isf[:, t]) * val[:, t]).view(B, 1, 1)
        diag_x = torch.diagonal(x, dim1=-2, dim2=-1)
        w1 = w1_frac[:, t - 1]
        r_mass[:, t - 1] = diag_x * w1
        row_weight[:, t - 1] = x.sum(-1)
        newdiag = diag_x * (1.0 - w1)
        Cbase = Cbase + (x - torch.diag_embed(diag_x) + torch.diag_embed(newdiag)).sum(0)
    return r_mass, row_weight, Cbase


def pair_kl_online(log_alpha, log_beta, logZ, log_trans_det, ev, is_first, valid,
                   gamma, log_trans_diff):
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
            x = (lx.exp() * ((1.0 - isf[:, t]) * val[:, t]).view(B, 1, 1))
        gp = gamma[:, t - 1].clamp_min(1e-30)
        ltd = (log_trans_diff(t) if callable(log_trans_diff)
               else log_trans_diff[:, t - 1])
        log_q_cond = x.clamp_min(1e-30).log() - gp[:, :, None].log()
        out[:, t - 1] = (x * (log_q_cond - ltd)).sum(dim=(-2, -1))
    return out

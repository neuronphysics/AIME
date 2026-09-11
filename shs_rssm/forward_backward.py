from __future__ import annotations

import math
import torch

NEG_INF = -1e30


def _trans_at(log_trans, t, B, K):
    if log_trans.dim() == 2:
        return log_trans.unsqueeze(0).expand(B, K, K)
    if log_trans.dim() == 3:
        return log_trans[t - 1].unsqueeze(0).expand(B, K, K)
    return log_trans[:, t - 1]


def _mk_trans_resolver(trans_fn, log_trans, B, K):
    if trans_fn is not None:
        return lambda t: trans_fn(t)
    return lambda t: _trans_at(log_trans, t, B, K)


def forward_backward(log_init, log_trans, log_ev, is_first=None, valid=None,
                     assume_start_at_t0=True,
                     prev_msg=None, prev_trans=None, return_pairwise=False,
                     return_final=False, return_boundary=False, return_messages=False,
                     trans_fn=None, ev_mask=None):
    """Masked HMM messages.

    Two independent masks with distinct meanings:

    valid[b, t] = 0    the timestep does not exist (padding, or a row excluded
                       by a caller).  No state, no evidence, no incident edge;
                       the chain restarts at the next existing timestep so that
                       logZ uses exactly the edges the pairwise counts report.
    ev_mask[b, t] = 0  the state exists and its outgoing edge is legal, but its
                       unary evidence is unavailable (a replay-crop boundary:
                       p(z_t | z_{t-1}) is fabricated because z_{t-1} was cropped
                       away).  Evidence is dropped and gamma is zeroed so the
                       timestep contributes no sufficient statistics, but the
                       edge out of it is used by the messages AND counted.

    With both masks the identity  d logZ / d log_trans[i, j] = E[N_ij]  holds.
    """
    B, T, K = log_ev.shape
    _Tof = _mk_trans_resolver(trans_fn, log_trans, B, K)
    if log_init.dim() == 1:
        log_init = log_init.unsqueeze(0).expand(B, K)

    if is_first is not None:
        isf = is_first.reshape(B, T).to(log_ev.dtype).clone()
    else:
        isf = log_ev.new_zeros(B, T)
    isf_raw = isf.clone()          # before the t=0 override, for per-fragment priors
    cont0 = None
    if prev_msg is None:
        if assume_start_at_t0:
            isf[:, 0] = 1.0
        else:
            cont0 = (1.0 - isf[:, 0]).clamp(0.0, 1.0)
            isf[:, 0] = 1.0

    if valid is not None:
        val = valid.reshape(B, T).to(log_ev.dtype)
    else:
        val = log_ev.new_ones(B, T)
    if ev_mask is not None:
        evm = ev_mask.reshape(B, T).to(log_ev.dtype) * val
    else:
        evm = val
    ev = torch.where(evm.unsqueeze(-1) > 0.5, log_ev, torch.zeros_like(log_ev))
    # A timestep that exists but follows a non-existent one starts a fresh chain
    # (no edge is used across the gap, and none is counted).
    resume = log_ev.new_zeros(B, T)
    if valid is not None:
        resume[:, 1:] = (val[:, 1:] * (1.0 - val[:, :-1])).clamp(0.0, 1.0)

    _alpha = [None] * T
    # Boundary tensors are referenced by BOTH the forward loop (per-fragment
    # resume prior) and the backward loop, so they must exist for every entry
    # branch, including the streaming continuation branch.
    unif = log_ev.new_full((B, K), -float(math.log(K)))
    init0 = log_init
    if prev_msg is None:
        if cont0 is not None:
            init0 = torch.where(cont0.unsqueeze(-1) > 0.5, unif, log_init)
        # A non-existent first step must carry unit mass, not unit density, so
        # that a later resume adds no spurious log K to logZ.
        _alpha[0] = torch.where(val[:, 0].unsqueeze(-1) > 0.5, init0 + ev[:, 0],
                                torch.full_like(log_init, -float(math.log(K))))
    else:
        if prev_trans is not None:
            Tb = prev_trans
        elif log_trans.dim() == 2:
            Tb = log_trans.unsqueeze(0).expand(B, K, K)
        else:
            raise ValueError("prev_msg with time-varying log_trans requires prev_trans")
        cont = torch.logsumexp(prev_msg.unsqueeze(-1) + Tb, dim=1) + ev[:, 0]
        fresh = log_init + ev[:, 0]
        f0 = isf[:, 0].unsqueeze(-1)
        _alpha[0] = f0 * fresh + (1.0 - f0) * cont
    for t in range(1, T):
        Tt = _Tof(t)
        m_trans = torch.logsumexp(_alpha[t - 1].unsqueeze(-1) + Tt, dim=1)
        m_reset = log_init + torch.logsumexp(_alpha[t - 1], dim=-1, keepdim=True)
        f = isf[:, t].unsqueeze(-1)
        m = f * m_reset + (1.0 - f) * m_trans
        # Restarting after a gap: the segment is independent, so its boundary
        # potential is init0, but the accumulated mass of the previous segment
        # must still multiply through -- otherwise logZ silently drops it.
        # Each fragment takes the boundary prior of ITS OWN first step: the
        # learned pi0 at a genuine episode start, uninformative otherwise.
        r = resume[:, t].unsqueeze(-1)
        init_t = torch.where(isf_raw[:, t].unsqueeze(-1) > 0.5, log_init, unif)
        m_resume = init_t + torch.logsumexp(_alpha[t - 1], dim=-1, keepdim=True)
        m = r * m_resume + (1.0 - r) * m
        step = m + ev[:, t]
        _alpha[t] = torch.where(val[:, t].unsqueeze(-1) > 0.5, step, _alpha[t - 1])

    log_alpha = torch.stack(_alpha, dim=1)
    logZ = torch.logsumexp(log_alpha[:, T - 1], dim=-1)
    if valid is not None:
        # a row with no existing timestep contributes nothing
        logZ = torch.where(val.sum(1) > 0.5, logZ, torch.zeros_like(logZ))
    _Tb = None
    if prev_msg is not None:
        _Tb = prev_trans if prev_trans is not None else (
            log_trans.unsqueeze(0).expand(B, K, K) if log_trans.dim() == 2 else None)

    log_beta = log_ev.new_zeros(B, T, K)
    for t in range(T - 1, 0, -1):
        Tt = _Tof(t)
        ev_beta = (ev[:, t] + log_beta[:, t])
        msg_trans = torch.logsumexp(Tt + ev_beta.unsqueeze(1), dim=2)
        msg_reset = torch.logsumexp(log_init + ev_beta, dim=-1, keepdim=True).expand(B, K)
        f = isf[:, t].unsqueeze(-1)
        msg = f * msg_reset + (1.0 - f) * msg_trans
        # Across a gap the backward message is state-independent (the future
        # segment's total mass), which is msg_reset built from init0.
        r = resume[:, t].unsqueeze(-1)
        init_t = torch.where(isf_raw[:, t].unsqueeze(-1) > 0.5, log_init, unif)
        msg_resume = torch.logsumexp(init_t + ev_beta, dim=-1, keepdim=True).expand(B, K)
        msg = r * msg_resume + (1.0 - r) * msg
        log_beta[:, t - 1] = torch.where(val[:, t].unsqueeze(-1) > 0.5, msg,
                                         log_beta[:, t])

    log_gamma = log_alpha + log_beta - logZ.view(B, 1, 1)
    # gamma is zeroed only where the state does not exist.  An evidence-masked
    # timestep keeps its (prior-driven) belief -- callers that form sufficient
    # statistics multiply by the evidence mask themselves, so a fabricated
    # timestep contributes none, while imagination starts keep a usable belief.
    gamma = torch.where(val.unsqueeze(-1) > 0.5, log_gamma.exp(),
                        torch.zeros_like(log_gamma))
    _gs = gamma.sum(-1, keepdim=True)
    gamma = torch.where(_gs > 1e-30, gamma / _gs.clamp_min(1e-30), gamma)

    if T >= 2:
        if return_pairwise:
            xi = log_ev.new_zeros(B, T - 1, K, K)
            for t in range(1, T):
                Tt = _Tof(t)
                lx = (log_alpha[:, t - 1].unsqueeze(-1) + Tt
                      + (ev[:, t] + log_beta[:, t]).unsqueeze(1)
                      - logZ.view(B, 1, 1))
                lZxi = torch.logsumexp(lx.reshape(B, -1), dim=-1).view(B, 1, 1)
                lZxi = torch.where(torch.isfinite(lZxi), lZxi, torch.zeros_like(lZxi))
                x = (lx - lZxi).exp()
                x = x * ((1.0 - isf[:, t]) * (1.0 - resume[:, t])
                         * val[:, t] * val[:, t - 1]).view(B, 1, 1)
                xi[:, t - 1] = x
            xicount = xi.sum(dim=(0, 1))
        else:
            xi = None
            xicount = log_ev.new_zeros(K, K)
            for t in range(1, T):
                Tt = _Tof(t)
                lx = (log_alpha[:, t - 1].unsqueeze(-1) + Tt
                      + (ev[:, t] + log_beta[:, t]).unsqueeze(1)
                      - logZ.view(B, 1, 1))
                lZxi = torch.logsumexp(lx.reshape(B, -1), dim=-1).view(B, 1, 1)
                lZxi = torch.where(torch.isfinite(lZxi), lZxi, torch.zeros_like(lZxi))
                x = (lx - lZxi).exp()
                x = x * ((1.0 - isf[:, t]) * (1.0 - resume[:, t])
                         * val[:, t] * val[:, t - 1]).view(B, 1, 1)
                xicount = xicount + x.sum(0)
    else:
        xi = log_ev.new_zeros(B, 0, K, K)
        xicount = log_ev.new_zeros(K, K)

    boundary_xi = None
    if prev_msg is not None and _Tb is not None:
        lxb = (prev_msg.unsqueeze(-1) + _Tb
               + (ev[:, 0] + log_beta[:, 0]).unsqueeze(1)
               - logZ.view(B, 1, 1))
        lZb = torch.logsumexp(lxb.reshape(B, -1), dim=-1).view(B, 1, 1)
        lZb = torch.where(torch.isfinite(lZb), lZb, torch.zeros_like(lZb))
        xb = (lxb - lZb).exp() * ((1.0 - isf[:, 0]) * val[:, 0]).view(B, 1, 1)
        xicount = xicount + xb.sum(0)
        boundary_xi = xb
    if return_final:
        _ar = torch.arange(1, T + 1, device=log_ev.device, dtype=val.dtype)
        last = (val * _ar.view(1, T)).argmax(dim=1).long()
        a_last = log_alpha[torch.arange(B, device=log_ev.device), last]
        final_msg = a_last - torch.logsumexp(a_last, dim=-1, keepdim=True)

    if return_messages:
        return gamma, xicount, logZ, log_alpha, log_beta
    if return_boundary:
        extra = [xi] if return_pairwise else []
        if return_final:
            extra.append(final_msg)
        return (gamma, xicount, logZ, *extra, boundary_xi)
    if return_pairwise and return_final:
        return gamma, xicount, logZ, xi, final_msg
    if return_pairwise:
        return gamma, xicount, logZ, xi
    if return_final:
        return gamma, xicount, logZ, final_msg
    return gamma, xicount, logZ


def start_counts_from(gamma, is_first=None, valid=None, assume_start_at_t0=True):
    B, T, K = gamma.shape
    if is_first is None:
        if not assume_start_at_t0:
            return gamma.new_zeros(K)
        w0 = gamma[:, 0]
        if valid is not None:
            w0 = w0 * valid.reshape(B, T)[:, 0].unsqueeze(-1)
        return w0.sum(0)
    isf = is_first.reshape(B, T).to(gamma.dtype).clone()
    if assume_start_at_t0:
        isf[:, 0] = 1.0
    if valid is not None:
        isf = isf * valid.reshape(B, T).to(gamma.dtype)
    return (gamma * isf.unsqueeze(-1)).sum(dim=(0, 1))


def brute_force(log_init, log_trans, log_ev, is_first=None):
    B, T, K = log_ev.shape
    import itertools
    if log_init.dim() == 1:
        log_init = log_init.unsqueeze(0).expand(B, K)
    if is_first is not None:
        isf = is_first.reshape(B, T).to(log_ev.dtype).clone()
    else:
        isf = log_ev.new_zeros(B, T)
    isf[:, 0] = 1.0
    gamma = log_ev.new_zeros(B, T, K)
    xicount = log_ev.new_zeros(K, K)
    logZ = log_ev.new_zeros(B)
    for b in range(B):
        paths = list(itertools.product(range(K), repeat=T))
        logp = log_ev.new_empty(len(paths))
        for pi, path in enumerate(paths):
            lp = log_init[b, path[0]] + log_ev[b, 0, path[0]]
            for t in range(1, T):
                if isf[b, t] > 0.5:
                    lp = lp + log_init[b, path[t]] + log_ev[b, t, path[t]]
                else:
                    lp = lp + log_trans[path[t - 1], path[t]] + log_ev[b, t, path[t]]
            logp[pi] = lp
        Z = torch.logsumexp(logp, 0)
        logZ[b] = Z
        w = (logp - Z).exp()
        for pi, path in enumerate(paths):
            for t in range(T):
                gamma[b, t, path[t]] += w[pi]
            for t in range(1, T):
                if isf[b, t] < 0.5:
                    xicount[path[t - 1], path[t]] += w[pi]
    return gamma, xicount, logZ

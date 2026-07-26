"""HMM forward-backward in log space, with episode resets and time-varying transitions.

Given, for a batch of B sequences of length T over K regimes,
    log_init  : (K,)  or (B, K)        log initial-state potential log p(s_1)
    log_trans : (K, K)                 stationary log transition, OR
                (T-1, K, K) / (B, T-1, K, K)  time-varying (the transition INTO step t
                                       uses slice t-1); needed for recurrent stickiness
    log_ev    : (B, T, K)              log local evidence log p(x_t | s_t = k)
    is_first  : (B, T) optional        episode-start mask; where is_first[b,t]=1 the
                                       state s_t is independent of s_{t-1} (the chain is
                                       re-seeded from log_init and no transition links
                                       t-1 -> t). t=0 is a start UNLESS a carried chunk
                                       message (prev_msg) continues an episode.
    valid     : (B, T) optional        valid-timestep mask. Where
                                       valid[b,t]=0 the step is padding: it contributes
                                       NO evidence, NO marginal (gamma=0), and NO
                                       transition count, and the forward/backward messages
                                       pass THROUGH it unchanged. A right-padded batch is
                                       therefore bit-identical to the truncated sequence.
    prev_msg  : (B, K) optional        carried FILTERING message q(s_{last of prev chunk})
                                       in log space. For sequences whose
                                       first step continues an episode (is_first[:,0]=0),
                                       step 0 is linked to the previous chunk by a REAL
                                       transition instead of re-seeding from log_init.
    prev_trans: (B, K, K) optional     transition INTO the first step, needed with prev_msg
                                       when log_trans is time-varying (for stationary
                                       log_trans it defaults to log_trans).

Returns gamma (B,T,K), xicount (K,K) summed expected transitions (episode-start and
padding steps contribute nothing), and logZ (B,). Options:
    return_pairwise=True  also returns xi (B, T-1, K, K), the per-step pairwise
                          responsibilities recurrent stickiness needs. When False, xicount
                          is accumulated online WITHOUT ever materialising the O(BTK^2) xi
                          tensor.
    return_final=True     also returns final_msg (B, K), the NORMALISED filtered log-alpha
                          at each sequence's last valid step, to seed the next chunk. 
                          Assumes prefix-validity (complete episode or right-padded chunk).

The recursion is sequential in T (a Python loop, T is small) and fully vectorised over
the batch B and the K x K transitions. Everything is in log space; only the final
marginals/pairwise are exponentiated.
"""
from __future__ import annotations

import torch

NEG_INF = -1e30


def _trans_at(log_trans, t, B, K):
    """Return the (B,K,K) transition INTO step t (uses slice t-1 for time-varying)."""
    if log_trans.dim() == 2:                      # (K,K) stationary
        return log_trans.unsqueeze(0).expand(B, K, K)
    if log_trans.dim() == 3:                      # (T-1,K,K) shared across batch
        return log_trans[t - 1].unsqueeze(0).expand(B, K, K)
    return log_trans[:, t - 1]                    # (B,T-1,K,K) -> (B,K,K)


def _mk_trans_resolver(trans_fn, log_trans, B, K):
    """Review P2 #11: return a per-step transition getter. With trans_fn given, each
    (B,K,K) slice is BUILT ON DEMAND (peak O(BK^2)); otherwise index the full tensor."""
    if trans_fn is not None:
        return lambda t: trans_fn(t)
    return lambda t: _trans_at(log_trans, t, B, K)


def forward_backward(log_init, log_trans, log_ev, is_first=None, valid=None,
                     prev_msg=None, prev_trans=None, return_pairwise=False,
                     return_final=False, return_boundary=False, return_messages=False,
                     trans_fn=None):
    B, T, K = log_ev.shape
    _Tof = _mk_trans_resolver(trans_fn, log_trans, B, K)   # review P2 #11: on-demand slices
    if log_init.dim() == 1:
        log_init = log_init.unsqueeze(0).expand(B, K)          # (B,K)

    if is_first is not None:
        isf = is_first.reshape(B, T).to(log_ev.dtype).clone()
    else:
        isf = log_ev.new_zeros(B, T)
    # t=0 is a fresh start ONLY when there is no carried chunk message; a continued chunk
    # links t=0 to the previous chunk with a transition (item 8), so keep the caller's
    # is_first[:,0] (0 = continue, 1 = new episode at the boundary).
    if prev_msg is None:
        isf[:, 0] = 1.0

    if valid is not None:
        val = valid.reshape(B, T).to(log_ev.dtype)
    else:
        val = log_ev.new_ones(B, T)
    # NaN-safe : -inf log-evidence * 0 would be NaN, so SELECT
    ev = torch.where(val.unsqueeze(-1) > 0.5, log_ev, torch.zeros_like(log_ev))

    # ---------------- forward ----------------
    log_alpha = log_ev.new_empty(B, T, K)
    if prev_msg is None:
        log_alpha[:, 0] = log_init + ev[:, 0]
    else:
        if prev_trans is not None:
            Tb = prev_trans
        elif log_trans.dim() == 2:
            Tb = log_trans.unsqueeze(0).expand(B, K, K)
        else:
            raise ValueError("prev_msg with time-varying log_trans requires prev_trans")
        cont = torch.logsumexp(prev_msg.unsqueeze(-1) + Tb, dim=1) + ev[:, 0]   # (B,K)
        fresh = log_init + ev[:, 0]
        f0 = isf[:, 0].unsqueeze(-1)
        log_alpha[:, 0] = f0 * fresh + (1.0 - f0) * cont
    for t in range(1, T):
        Tt = _Tof(t)                                          # (B,K,K)
        m_trans = torch.logsumexp(log_alpha[:, t - 1].unsqueeze(-1) + Tt, dim=1)  # (B,K)
        # at a reset the transition is log_init[k] (constant in j), which under the
        # standard recursion carries the just-closed episode's partition forward
        m_reset = log_init + torch.logsumexp(log_alpha[:, t - 1], dim=-1, keepdim=True)
        f = isf[:, t].unsqueeze(-1)                           # (B,1)
        m = f * m_reset + (1.0 - f) * m_trans
        step = m + ev[:, t]
        # padding step: carry the previous alpha unchanged (message passes through)
        log_alpha[:, t] = torch.where(val[:, t].unsqueeze(-1) > 0.5, step,
                                      log_alpha[:, t - 1])

    logZ = torch.logsumexp(log_alpha[:, T - 1], dim=-1)       # (B,) (carried past padding)
    _Tb = None
    if prev_msg is not None:
        _Tb = prev_trans if prev_trans is not None else (
            log_trans.unsqueeze(0).expand(B, K, K) if log_trans.dim() == 2 else None)

    # ---------------- backward ----------------
    log_beta = log_ev.new_zeros(B, T, K)
    for t in range(T - 1, 0, -1):
        Tt = _Tof(t)                                          # (B,K,K)
        ev_beta = (ev[:, t] + log_beta[:, t])                 # (B,K)
        msg_trans = torch.logsumexp(Tt + ev_beta.unsqueeze(1), dim=2)   # (B,K) over k
        msg_reset = torch.logsumexp(log_init + ev_beta, dim=-1, keepdim=True).expand(B, K)
        f = isf[:, t].unsqueeze(-1)                           # (B,1)
        msg = f * msg_reset + (1.0 - f) * msg_trans
        # padding step: carry the following beta unchanged
        log_beta[:, t - 1] = torch.where(val[:, t].unsqueeze(-1) > 0.5, msg,
                                         log_beta[:, t])

    # ---------------- marginals ----------------
    log_gamma = log_alpha + log_beta - logZ.view(B, 1, 1)
    gamma = torch.where(val.unsqueeze(-1) > 0.5, log_gamma.exp(),
                        torch.zeros_like(log_gamma))          # padding marginals -> 0

    # ---------------- pairwise / transition counts ----------------
    if T >= 2:
        if return_pairwise:
            # materialise xi and reduce ONCE at the end -- bit-identical to the original
            # (preserves exact float summation order for every existing call site)
            xi = log_ev.new_zeros(B, T - 1, K, K)
            for t in range(1, T):
                Tt = _Tof(t)                                  # (B,K,K)
                lx = (log_alpha[:, t - 1].unsqueeze(-1) + Tt
                      + (ev[:, t] + log_beta[:, t]).unsqueeze(1)
                      - logZ.view(B, 1, 1))                   # (B,K,K)
                x = lx.exp()
                x = x * ((1.0 - isf[:, t]) * val[:, t]).view(B, 1, 1)
                xi[:, t - 1] = x
            xicount = xi.sum(dim=(0, 1))                       # (K,K)
        else:
            # Accumulate xicount ONLINE, never allocating xi (B,T-1,K,K)
            xi = None
            xicount = log_ev.new_zeros(K, K)
            for t in range(1, T):
                Tt = _Tof(t)                                  # (B,K,K)
                lx = (log_alpha[:, t - 1].unsqueeze(-1) + Tt
                      + (ev[:, t] + log_beta[:, t]).unsqueeze(1)
                      - logZ.view(B, 1, 1))                   # (B,K,K)
                x = lx.exp()
                x = x * ((1.0 - isf[:, t]) * val[:, t]).view(B, 1, 1)
                xicount = xicount + x.sum(0)
    else:
        xi = log_ev.new_zeros(B, 0, K, K)
        xicount = log_ev.new_zeros(K, K)

    # BOUNDARY transition: the last state of the previous
    # chunk -> the first state of this chunk is a REAL transition that both the internal
    # xi (t=1..T-1) and start_counts omit; add its pairwise marginal to xicount so
    # full-sequence counts == sum of continued-chunk counts INCLUDING the boundary.
    boundary_xi = None
    if prev_msg is not None and _Tb is not None:
        lxb = (prev_msg.unsqueeze(-1) + _Tb
               + (ev[:, 0] + log_beta[:, 0]).unsqueeze(1)
               - logZ.view(B, 1, 1))                          # (B,K,K)
        xb = lxb.exp() * ((1.0 - isf[:, 0]) * val[:, 0]).view(B, 1, 1)
        xicount = xicount + xb.sum(0)
        boundary_xi = xb
    if return_final:
        # last valid index per sequence (prefix-validity assumption)
        last = (val.sum(1) - 1.0).clamp(min=0).long()         # (B,)
        a_last = log_alpha[torch.arange(B, device=log_ev.device), last]   # (B,K)
        final_msg = a_last - torch.logsumexp(a_last, dim=-1, keepdim=True)

    # expose the forward/backward messages so a caller can
    # recompute each step's pairwise TRANSIENTLY (O(BK^2) memory) and accumulate PG/KL
    # online, instead of materialising the full O(BTK^2) xi tensor.
    if return_messages:
        return gamma, xicount, logZ, log_alpha, log_beta
    # expose the boundary pairwise so recurrent PG / HDP counts
    # at the chunk seam can be attributed by the caller (None when no carried message)
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


def start_counts_from(gamma, is_first=None, valid=None):
    """Expected initial-state counts: sum of gamma over all episode-start steps. gamma is
    already padding-masked; valid is accepted for symmetry and to zero any start that
    lands on a padding step."""
    B, T, K = gamma.shape
    if is_first is None:
        w0 = gamma[:, 0]
        if valid is not None:
            w0 = w0 * valid.reshape(B, T)[:, 0].unsqueeze(-1)
        return w0.sum(0)
    isf = is_first.reshape(B, T).to(gamma.dtype).clone()
    isf[:, 0] = 1.0
    if valid is not None:
        isf = isf * valid.reshape(B, T).to(gamma.dtype)
    return (gamma * isf.unsqueeze(-1)).sum(dim=(0, 1))


def brute_force(log_init, log_trans, log_ev, is_first=None):
    """Reference: exact marginals and counts by enumerating all K^T paths.

    Supports stationary transition and episode resets (is_first). Only for tiny (T,K).
    """
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

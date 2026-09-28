from __future__ import annotations

import torch


def _lead(M, t):
    return torch.einsum("nk,k...->n...", M.to(dtype=t.dtype, device=t.device), t)


def newborn_rows(row_map):
    return (row_map.abs().sum(1) <= 0).nonzero().reshape(-1).tolist()


def remap_stats(live, cand, row_map, scale=None):
    """EMA-scale statistics for the new structure: mapped live rows, seeded newborn rows."""
    names = live.raw_stat_names()
    if getattr(live, "_residual_only", False) or any(not hasattr(cand, n) for n in names):
        return None
    out = {}
    fresh = newborn_rows(row_map)
    for n in names:
        t = _lead(row_map, getattr(live, n))
        if fresh:
            seed = getattr(cand, n)[fresh]
            if scale is not None:
                seed = seed * float(scale)
            t[fresh] = seed.to(t.dtype)
        out[n] = t
    return out


def remap_counts(C_live, s_live, C_cand, s_cand, row_map, scale=None):
    M = row_map.to(dtype=C_live.dtype, device=C_live.device)
    C = M @ C_live @ M.t()
    s = M @ s_live
    fresh = newborn_rows(row_map)
    if fresh:
        f = float(1.0 if scale is None else scale)
        Cc = C_cand.to(C.dtype).to(C.device) * f
        C[fresh, :] = Cc[fresh, :]
        C[:, fresh] = Cc[:, fresh]
        s[fresh] = s_cand.to(s.dtype).to(s.device)[fresh] * f
    return C, s


def remap_rstick(live_rs, cand_rs, row_map, scale=None):
    if live_rs is None:
        return cand_rs
    K_new = int(row_map.shape[0])
    new = live_rs.resized_like(K_new)
    A = _lead(row_map, live_rs.pg_A)
    h = _lead(row_map, live_rs.pg_h)
    fresh = newborn_rows(row_map)
    if fresh and cand_rs is not None:
        f = float(1.0 if scale is None else scale)
        A[fresh] = cand_rs.pg_A[fresh].to(A.dtype) * f
        h[fresh] = cand_rs.pg_h[fresh].to(h.dtype) * f
    new._pg_init = bool(getattr(live_rs, "_pg_init", False))
    if new._pg_init:
        new.pg_set_totals(A, h)
    else:
        new.pg_A.copy_(A)
        new.pg_h.copy_(h)
    return new


def advance_belief_generation(head, row_map, keep=16):
    """Allocate a new structure generation; maps[g] sends a belief of generation g to the new one."""
    old = int(getattr(head, "_belief_gen", 0))
    new = max(int(getattr(head, "_belief_next", old + 1)), old + 1)
    maps = {}
    if row_map is not None:
        M = row_map.detach().to(torch.float32).cpu()
        for g, prev in (getattr(head, "_belief_maps", None) or {}).items():
            if int(prev.shape[0]) == int(M.shape[1]):
                maps[int(g)] = M @ prev
        maps[old] = M
    for g in sorted(maps)[:-keep]:
        maps.pop(g)
    head._belief_maps = maps
    head._belief_gen = new
    head._belief_next = new + 1

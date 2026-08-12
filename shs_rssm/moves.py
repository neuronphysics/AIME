from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

from .regimes import DiagARRegimes
from .sticky_hdp import StickyHDP, merge_rho_omega, drop_rho_omega
from .forward_backward import forward_backward, start_counts_from

import os as _os
BIRTH_DEBUG = _os.environ.get("BIRTH_DEBUG", "0") not in ("0", "", "false", "False")
BIRTH_SEED_HARD = _os.environ.get("BIRTH_SEED_HARD", "1") not in ("0", "", "false", "False")
# Refit recurrent merge candidates before scoring (see merge_move).  Set to 0
# to restore the cheaper pre-fix behaviour when move cost dominates.
_MERGE_REFIT_RECURRENT = _os.environ.get(
    "SHS_MERGE_REFIT_RECURRENT", "1") not in ("0", "", "false", "False")
# Birth split-point criterion: "marginal" scores candidate cuts by the conjugate
# marginal evidence (requires regimes.log_marginal_from_stats; both call sites
# guard on hasattr), anything else falls back to the least-squares AR surrogate
# (_best_cut_ar).  README: "splitting at the cut that maximises the conjugate
# marginal evidence".
_BIRTH_CUT = _os.environ.get("BIRTH_CUT", "marginal")

_MERGE_REFIT = _os.environ.get("MERGE_REFIT", "0") not in ("0", "", "false", "False")
_MERGE_BASE_REFIT = _os.environ.get("MERGE_BASE_REFIT", "0") not in ("0", "", "false", "False")
_DELETE_PLANNER = _os.environ.get("DELETE_PLANNER", "1") not in ("0", "", "false", "False")
_DELETE_MIN_COUNT = float(_os.environ.get("DELETE_MIN_COUNT", "0.01"))

def _delete_records(head):
    rec = getattr(head, "_delete_records", None)
    if rec is None:
        from .delete_planner import FailureRecords
        rec = FailureRecords()
        head._delete_records = rec
    return rec


@torch.no_grad()
def _per_seq_occupancy_from_buffer(head, buf):
    import numpy as _np
    rows = []
    for b in buf.batches:
        gamma, _, _, _ = head.regime_inference(
            b.stoch, b.deter, b.is_first, valid=getattr(b, "valid", None),
            z_var=b.z_var, action=getattr(b, "action", None), cache_estep=False)
        rows.append(gamma.detach().sum(dim=1).cpu().numpy())
    return _np.concatenate(rows, axis=0) if rows else _np.zeros((0, head.K))


def _seq_index_map(buf):
    out = []
    for bi, b in enumerate(buf.batches):
        for r in range(b.stoch.shape[0]):
            out.append((bi, r))
    return out


def _row_mask_from_seqs(buf, seqs):
    idx = _seq_index_map(buf)
    per = {}
    for n in seqs:
        if 0 <= n < len(idx):
            bi, r = idx[n]
            per.setdefault(bi, []).append(r)
    return {bi: torch.zeros(buf.batches[bi].stoch.shape[0], dtype=torch.bool,
                            device=buf.batches[bi].stoch.device).index_fill_(
        0, torch.tensor(rs, device=buf.batches[bi].stoch.device), True)
        for bi, rs in per.items()}


@dataclass
class _Batch:
    stoch: torch.Tensor
    deter: torch.Tensor
    is_first: torch.Tensor | None
    z_var: torch.Tensor | None = None
    step: int = 0
    batch_id: object = None
    repr_version: int | None = None
    valid: torch.Tensor | None = None


def _smoothed_moments(head, b):
    zc = getattr(b, "z_cov", None)
    xc = getattr(b, "zg_xcov", None)
    if zc is None and xc is None:
        return None, None, None
    T = b.stoch.shape[1]
    return (zc,
            None if zc is None else head._shift_cov(zc, b.is_first),
            None if xc is None else head._align_xcov(xc, b.is_first, T=T))


class MoveBuffer:
    def __init__(self, max_batches: int = 8, max_age: int = 0,
                 complete: bool = False, expected_batches: int | None = None,
                 expected_ids=None):
        self.max_batches = max_batches
        self.max_age = max_age
        self.complete = bool(complete)
        self.expected_batches = expected_batches
        self.expected_ids = set(expected_ids) if expected_ids is not None else None
        self.batches: list[_Batch] = []

    def add(self, stoch, deter, is_first=None, z_var=None, step=0,
            batch_id=None, repr_version=None, action=None, valid=None,
            z_cov=None, zg_xcov=None):
        rv = None if repr_version is None else int(repr_version)
        entry = _Batch(
            stoch.detach(), deter.detach(),
            None if is_first is None else is_first.detach(),
            None if z_var is None else z_var.detach(), int(step), batch_id, rv,
            None if valid is None else valid.detach())
        entry.action = None if action is None else action.detach()
        entry.z_cov = None if z_cov is None else z_cov.detach()
        entry.zg_xcov = None if zg_xcov is None else zg_xcov.detach()
        if batch_id is not None:
            for i, b in enumerate(self.batches):
                if b.batch_id == batch_id:
                    self.batches[i] = entry
                    return
        self.batches.append(entry)
        if not self.complete:
            if self.max_age > 0:
                self.batches = [b for b in self.batches if step - b.step <= self.max_age]
            while len(self.batches) > self.max_batches:
                self.batches.pop(0)

    def purge_stale(self, current_version) -> int:
        before = len(self.batches)
        self.batches = [b for b in self.batches
                        if b.repr_version is None
                        or int(b.repr_version) == int(current_version)]
        return before - len(self.batches)

    def is_complete(self) -> bool:
        if not self.complete:
            return False
        ids = [b.batch_id for b in self.batches]
        if any(i is None for i in ids) or len(set(ids)) != len(ids):
            return False
        if self.expected_ids is not None:
            return set(ids) == self.expected_ids
        if self.expected_batches is None:
            return False
        return len(set(ids)) == self.expected_batches

    def __len__(self):
        return len(self.batches)


def _as_buffer(buffer, stoch, deter, is_first, head=None, action=None):
    if buffer is not None and len(buffer) > 0:
        _validate_buffer(head, buffer)
        return buffer
    b = MoveBuffer(max_batches=1)
    b.add(stoch, deter, is_first, action=action)
    return b


def _validate_buffer(head, buf):
    if buf is None:
        return
    if getattr(buf, "complete", False):
        ids = [b.batch_id for b in buf.batches]
        problems = []
        if any(i is None for i in ids):
            problems.append("some batches carry batch_id=None (anonymous data cannot "
                            "certify a corpus)")
        if len(set(ids)) != len(ids):
            problems.append("duplicated batch ids present")
        if buf.expected_ids is not None:
            extras = sorted(set(i for i in ids if i is not None) - set(buf.expected_ids))
            if extras:
                problems.append(f"foreign batch ids present beyond the declared "
                                f"corpus: {extras[:4]} (exact-set contract)")
        if any(b.repr_version is None for b in buf.batches):
            problems.append("some batches lack a repr_version stamp")
        if buf.expected_ids is None and buf.expected_batches is None:
            problems.append("no completeness certificate declared: set expected_ids "
                            "(preferred, membership check) or expected_batches")
        if not problems and not buf.is_complete():
            want = (sorted(buf.expected_ids) if buf.expected_ids is not None
                    else buf.expected_batches)
            problems.append(f"{len(set(ids))} unique ids present, certificate = {want}")
        if problems:
            raise RuntimeError(
                "MoveBuffer(complete=True) contract violated -- refusing to score "
                "moves (a 'gain' would silently mean gain on a partial or "
                "inconsistent corpus):\n  - " + "\n  - ".join(problems))
    if getattr(buf, "complete", False) and head is not None and hasattr(head, "repr_version"):
        
        cur = int(head.repr_version)
        stale = [b.batch_id for b in buf.batches
                 if b.repr_version is not None and b.repr_version != cur]
        if stale:
            raise RuntimeError(
                f"complete MoveBuffer mixes representation version(s) other than the "
                f"head's current {cur} (ids {stale[:4]}...): a whole corpus must be "
                "encoded once under a frozen representation (see fit_offline_corpus).")


def _head_rstick(head):
    if getattr(head, "recurrent", False) and getattr(head, "rstick", None) is not None:
        return head.rstick
    return None


def _clone_rstick(rs, new_K: int):
    return None if rs is None else rs.resized_like(int(new_K))


def _rstick_keep(rs, keep_idx):
    return None if rs is None else rs.select_rows(keep_idx)


def _rstick_merge(rs, i, j):
    """Gate rows for a merge of j into i.

    Selection alone discards row j's persistence evidence. The PG natural-parameter
    statistics are additive over (n, t), so the merged row is their sum and the
    Gaussian posterior is refit from it; see RecurrentStickiness.merge_rows.
    """
    if rs is None:
        return None
    if hasattr(rs, "merge_rows"):
        return rs.merge_rows(i, j)
    return rs.select_rows([k for k in range(rs.K) if k != j])


@torch.no_grad()
def aggregate_bound(head, buffer, regimes=None, hdp=None, rstick=None) -> float:
    local = 0.0
    for b in buffer.batches:
        lz, _, _ = head.bound_local(b.stoch, b.deter, b.is_first, regimes=regimes,
                                    hdp=hdp, rstick=rstick, z_var=b.z_var,
                                    action=getattr(b, "action", None),
                                    valid=getattr(b, "valid", None))
        local += lz
    return float(local + head.bound_global(regimes=regimes, hdp=hdp, rstick=rstick))


@torch.no_grad()
def _accumulate(head, buffer, regimes, hdp, rstick=None, row_mask=None):
    K = regimes.K
    agg = None
    Cc = torch.zeros(K, K, dtype=torch.float64, device=head.z0.device)
    sc = torch.zeros(K, dtype=torch.float64, device=head.z0.device)
    pg_phi, pg_r, pg_w = [], [], []
    for _bi, b in enumerate(buffer.batches):
        if row_mask is not None and _bi not in row_mask:
            continue
        prev = head._prev_stoch(b.stoch, b.is_first)
        g = head.build_g(prev, b.deter,
                         head._shift_action(getattr(b, "action", None), b.is_first))
        g_var = head._g_var_from_z_var(b.z_var, g, is_first=b.is_first)
        ev = regimes.expected_loglik(b.stoch, g, z_var=b.z_var, g_var=g_var).double()
        log_init, log_trans, aux = head._score_potentials(b.deter, hdp, rstick)
        
        _sa0 = not bool(getattr(head, "chunk_boundary_mask", True))
        vm = head._chunk_valid(getattr(b, "valid", None), b.is_first,
                               b.stoch.shape[0], b.stoch.shape[1],
                               b.stoch.dtype, b.stoch.device)
        if row_mask is not None:
            _m = row_mask[_bi].to(ev.dtype).reshape(-1, 1)
            vm = _m.expand(-1, ev.shape[1]) if vm is None else \
                vm.reshape(ev.shape[0], ev.shape[1]).to(ev.dtype) * _m
        gamma, xicount, _, xi = forward_backward(
            log_init, log_trans, ev, is_first=b.is_first, valid=vm,
            assume_start_at_t0=_sa0, return_pairwise=True)
        if aux is not None:
            r_mass, row_weight, counts = rstick.attribute_bound(xi, aux)
            Dp = aux["phi_steps"].shape[-1]
            pg_phi.append(aux["phi_steps"].reshape(-1, Dp))
            pg_r.append(r_mass.reshape(-1, K))
            pg_w.append(row_weight.reshape(-1, K))
        else:
            counts = xicount
        gamma = gamma.to(b.stoch.dtype)
        _zc, _gzc, _xc = _smoothed_moments(head, b)
        st = regimes.stats_from_batch(gamma, b.stoch, g, z_var=b.z_var,
                                      g_z_var=head._shift_var(b.z_var, b.is_first),
                                      z_cov=_zc, g_zcov=_gzc, zg_xcov=_xc)
        agg = st if agg is None else {k: agg[k] + st[k] for k in agg}
        Cc += counts.double()
        sc += start_counts_from(gamma, b.is_first, valid=vm,
                                assume_start_at_t0=_sa0).double()
    pg = None
    if pg_phi:
        pg = dict(phi=torch.cat(pg_phi, 0), r_mass=torch.cat(pg_r, 0),
                  row_weight=torch.cat(pg_w, 0))
    return agg, Cc, sc, pg


@torch.no_grad()
def _refine(head, regimes, hdp, buffer, iters: int = 2, hdp_iters: int = 1, rstick=None):
    K = regimes.K
    _kb = getattr(_refine, "_K_base", None)
    for _it in range(iters):
        agg, Cc, sc, pg = _accumulate(head, buffer, regimes, hdp, rstick=rstick)
        if BIRTH_DEBUG and _kb is not None and K > _kb:
            nb = float(Cc[:, _kb:].sum())
            inc = float(Cc[:, :_kb].sum())
            print(f"[birth] refine {_it}: newborn transition mass = {nb:.4f} "
                  f"({100.0 * nb / max(nb + inc, 1e-12):.2f}% of total)")
        regimes = regimes.clone_with_K(K, agg)
        hdp = hdp.resized_like(K)
        hdp.update(Cc, sc, n_global_iters=hdp_iters)
        if rstick is not None and pg is not None:
            rstick.pg_update_statewise(pg["phi"], pg["r_mass"], pg["row_weight"], lr=None)
    _, Cc, sc, _ = _accumulate(head, buffer, regimes, hdp, rstick=rstick)
    if BIRTH_DEBUG and _kb is not None and K > _kb:
        print(f"[birth] refine final: newborn transition mass = "
              f"{float(Cc[:, _kb:].sum()):.4f}")
    return regimes, hdp, Cc, sc


@torch.no_grad()
def _fit_score(head, regimes, hdp, buffer, iters, rstick=None):
    regimes, hdp, C, s = _refine(head, regimes, hdp, buffer, iters=iters, rstick=rstick)
    bound = aggregate_bound(head, buffer, regimes=regimes, hdp=hdp, rstick=rstick)
    return bound, regimes, hdp, C, s


def _select(stats, idx):
    return {k: v[idx].clone() for k, v in stats.items()}


def _current_stats(regimes: DiagARRegimes):
    Szz = getattr(regimes, "Szz_resid", regimes.Szz)
    s = dict(N=regimes.N.clone(), Sgg=regimes.Sgg.clone(),
             Szg=regimes.Szg.clone(), Szz=Szz.clone())
    if regimes.q_rank > 0:
        s["Szz_full"] = getattr(regimes, "Szz_full_resid", regimes.Szz_full).clone()
    if getattr(regimes, "q_rank", 0) > 0 and hasattr(regimes, "Szf"):
        for nm in ("Szf", "Sfr", "Sfh", "Sff"):
            s[nm] = getattr(regimes, nm).clone()
    return s


def _merge_stats(stats, i, j):
    K = stats["N"].shape[0]
    s = {k: v.clone() for k, v in stats.items()}
    _factor = ("Szf", "Sfr", "Sfh", "Sff")
    dom = i if float(stats["N"][i]) >= float(stats["N"][j]) else j
    for k in s:
        if k in _factor:
            s[k][i] = s[k][dom].clone()
        else:
            s[k][i] = s[k][i] + s[k][j]
    keep = torch.tensor([k for k in range(K) if k != j], device=stats["N"].device)
    return _select(s, keep), keep


def _merge_counts(C, start, i, j):
    K = C.shape[0]
    C = C.clone()
    C[i, :] = C[i, :] + C[j, :]
    C[:, i] = C[:, i] + C[:, j]
    keep = torch.tensor([k for k in range(K) if k != j], device=C.device)
    C = C[keep][:, keep]
    start = start.clone(); start[i] = start[i] + start[j]
    start = start[keep]
    return C, start


def _drop(C, start, k):
    K = C.shape[0]
    keep = torch.tensor([m for m in range(K) if m != k], device=C.device)
    return C[keep][:, keep], start[keep], keep


def _resid_logdet_from(N, Sgg, Szg, Szz, Szz_full, L, ridge=1e-4):
    N = N.clamp_min(1e-6)
    G = Sgg.shape[-1]
    A = Sgg + ridge * torch.eye(G, dtype=Sgg.dtype, device=Sgg.device)
    sol = torch.linalg.solve(A, Szg.transpose(-1, -2))
    explained = Szg @ sol
    if Szz_full is not None:
        Sres = (Szz_full - explained) / N
        Sres = Sres + 1e-6 * torch.eye(L, dtype=Sres.dtype, device=Sres.device)
        return torch.linalg.slogdet(Sres)[1]
    resid = (Szz - torch.diagonal(explained)) / N
    return torch.log(resid.clamp_min(1e-8)).sum()


def _merge_gain_cached(stats, i, j, L):
    has_full = "Szz_full" in stats
    def ld(N, Sgg, Szg, Szz, Szzf):
        return _resid_logdet_from(N, Sgg, Szg, Szz, Szzf, L)
    Ni, Nj = stats["N"][i].clamp_min(1e-6), stats["N"][j].clamp_min(1e-6)
    ld_i = ld(Ni, stats["Sgg"][i], stats["Szg"][i], stats["Szz"][i],
              stats["Szz_full"][i] if has_full else None)
    ld_j = ld(Nj, stats["Sgg"][j], stats["Szg"][j], stats["Szz"][j],
              stats["Szz_full"][j] if has_full else None)
    Nij = Ni + Nj
    ld_ij = ld(Nij, stats["Sgg"][i] + stats["Sgg"][j], stats["Szg"][i] + stats["Szg"][j],
               stats["Szz"][i] + stats["Szz"][j],
               (stats["Szz_full"][i] + stats["Szz_full"][j]) if has_full else None)
    return float(-0.5 * (Nij * ld_ij - Ni * ld_i - Nj * ld_j))


@torch.no_grad()
def _entropy_free_score(head, regimes, hdp, rstick=None):
    fn = getattr(regimes, "data_elbo_from_stats", None)
    if fn is None:
        return None
    ldata = float(fn())
    param_kl = float(regimes.param_kl().sum())
    if rstick is not None:
        param_kl += float(rstick.beta_kl())
    return ldata - param_kl + float(hdp.alloc_elbo())


@torch.no_grad()
def _hughes_merge_shortlist(head, base_reg, base_hdp, baseC, bases, base_rstick=None,
                            hdp_iters: int = 1, pairs_cap: int = 600):
    K = base_reg.K
    if K * (K - 1) // 2 > pairs_cap:
        return None
    base_score = _entropy_free_score(head, base_reg, base_hdp, rstick=base_rstick)
    if base_score is None:
        return None
    stats = _current_stats(base_reg)
    out = []
    for i in range(K):
        for j in range(i + 1, K):
            cand_stats, keep = _merge_stats(stats, i, j)
            C, start = _merge_counts(baseC, bases, i, j)
            cand_reg = base_reg.clone_with_K(K - 1, cand_stats)
            cand_hdp = base_hdp.resized_like(K - 1)
            _r, _o = merge_rho_omega(base_hdp.rho, base_hdp.omega, i, j)
            cand_hdp.seed_rho_omega(_r, _o)
            cand_hdp.update(C, start, n_global_iters=0)
            cand_rstick = _rstick_merge(base_rstick, i, j)
            score = _entropy_free_score(head, cand_reg, cand_hdp, rstick=cand_rstick)
            gain = score - base_score
            if gain > 0.0:
                out.append((i, j, gain))
    out.sort(key=lambda t: -t[2])
    return out


def _candidate(head, regime_stats, C, start, new_K, hdp_iters=1, seed_rho=None):
    regimes = head.regimes.clone_with_K(new_K, regime_stats)
    hdp = head.hdp.resized_like(new_K)
    if seed_rho is not None:
        hdp.seed_rho_omega(seed_rho[0], seed_rho[1])
    hdp.update(C.double(), start.double(), n_global_iters=int(hdp_iters))
    return regimes, hdp


def _rowmap_keep(K_old, keep, sum_into=None, device=None):
    M = torch.zeros(len(keep), K_old, dtype=torch.float64, device=device)
    M[torch.arange(len(keep)), keep.to(M.device)] = 1.0
    if sum_into is not None:
        i, j = sum_into
        pos = int((keep == i).nonzero()[0].item())
        M[pos, j] = 1.0
    return M


def _rowmap_append(K_old, m, device=None):
    return torch.cat([torch.eye(K_old, dtype=torch.float64, device=device),
                      torch.zeros(m, K_old, dtype=torch.float64, device=device)], 0)


def _apply(head, regimes, hdp, C, start, rstick=None, row_map=None):
    K_old = int(head.K)
    head.regimes = regimes
    if hasattr(regimes, "_freeze_C"):
        regimes._freeze_C = False
    head.hdp = hdp
    head.K = regimes.K
    if hasattr(head, "bump_struct_gen"):
        head.bump_struct_gen()
    if row_map is not None:
        head._belief_map = row_map.detach().to(torch.float32).cpu()
    elif regimes.K >= K_old:
        _M = torch.zeros(regimes.K, K_old, dtype=torch.float32)
        _M[:K_old, :K_old] = torch.eye(K_old)
        head._belief_map = _M
    else:
        head._belief_map = None
    if getattr(head, "rstick", None) is not None:
        if rstick is not None:
            head.rstick = rstick
        else:
            head.rstick = head.rstick.resized_like(regimes.K)
    head.register_buffer("ema_trans_counts", C.double().clone())
    head.register_buffer("ema_start_counts", start.double().clone())
    head._counts_initialised = True
    store = getattr(head, "stat_store", None)
    if store is not None and getattr(store, "mode", "legacy_ema") != "legacy_ema":
        if row_map is not None:
            store.remap(row_map)
        elif int(regimes.K) != K_old:
            store.reset()


def _active_size_log_prior(K: int, log_odds: float) -> float:
    return float(log_odds) * float(K)


def _accept(cand_bound: float, base_bound: float, K_cand: int, K_base: int,
            threshold: float, size_log_prior_odds: float,
            shrink_tol_abs: float = 1e-6) -> bool:
    j_cand = cand_bound + _active_size_log_prior(K_cand, size_log_prior_odds)
    j_base = base_bound + _active_size_log_prior(K_base, size_log_prior_odds)

    if K_cand < K_base:
        # bnpy uses a FIXED absolute tolerance of 1e-6 (MemoVBMovesAlg.py:31), and a
        # scale-dependent one over-prunes: at |J| = 1e8 a relative 1e-6 permits ~47
        # nats of deterioration. Reproduce bnpy exactly, raising the floor only if
        # float64 resolution at |j_base| makes 1e-6 unrepresentable, in which case no
        # tie test is meaningful below it.
        eps = float(np.finfo(np.float64).eps) * abs(j_base)
        tol = max(float(shrink_tol_abs), eps)
        return j_cand > j_base + threshold - tol
    return j_cand > j_base + threshold


@torch.no_grad()
def delete_move(head, stoch=None, deter=None, is_first=None, *, buffer=None,
                min_mass=None, threshold=0.0, size_log_prior_odds: float = 0.0,
                refine_iters: int = 2, mode: str = "hughes", delete_topk: int = 3,
                delete_max_seqs: int | None = 10,
                action=None):
    buf = _as_buffer(buffer, stoch, deter, is_first, head=head, action=action)
    if head.K <= 1:
        return False, 0.0
    live_rs = _head_rstick(head)
    soft = torch.zeros(head.K, dtype=torch.float64, device=head.z0.device)
    used = torch.zeros(head.K, dtype=torch.float64, device=head.z0.device)
    for b in buf.batches:
        gamma, _, _, _ = head.regime_inference(b.stoch, b.deter, b.is_first, valid=getattr(b,'valid',None),
                                              z_cov=getattr(b, 'z_cov', None),
                                              zg_xcov=getattr(b, 'zg_xcov', None),
                                               cache_estep=False, z_var=b.z_var,
                                               action=getattr(b, "action", None))
        soft += gamma.sum(dim=(0, 1)).double()
        used += (gamma.sum(dim=1) > 1.0).double().sum(0)
    order = torch.argsort(soft).tolist()
    _plan = None; _rec = None; _tgt_mask = None
    _lapv = float(getattr(head, "_lap", 0.0))
    if _DELETE_PLANNER:
        from .delete_planner import per_sequence_occupancy, plan_delete
        _u = _per_seq_occupancy_from_buffer(head, buf)
        _plan = plan_delete(_u, uids=list(range(head.K)),
                            lap=float(getattr(head, "_lap", 0.0)),
                            records=_delete_records(head), era="2015",
                            min_count=float(_DELETE_MIN_COUNT),
                            max_target_seqs=(10 if delete_max_seqs is None
                                             else int(delete_max_seqs)))
        if _plan is None:
            return False, 0.0
        _rec = _delete_records(head)
        head._lap = _lapv + 1.0
        _tot = _u.sum(0)
        for _uid in _plan.target_uids:
            _rec.record_attempt(int(_uid), float(_tot[int(_uid)]), head._lap)
        _tgt_mask = _row_mask_from_seqs(buf, _plan.target_seqs)
        cands = list(_plan.target_uids)[:max(1, int(delete_topk))]
    elif delete_max_seqs is not None:
        _gated = [int(j) for j in order if float(used[j]) <= float(delete_max_seqs)]
        if _gated:
            order = _gated
        cands = ([int(j) for j in order] if min_mass is None else
                 [int(j) for j in order if float(soft[j]) < float(min_mass)])
        cands = cands[:max(1, int(delete_topk))]
    else:
        cands = ([int(j) for j in order] if min_mass is None else
                 [int(j) for j in order if float(soft[j]) < float(min_mass)])
        cands = cands[:max(1, int(delete_topk))]
    if not cands or head.K <= 1:
        return False, 0.0
    last_gain = 0.0
    for k in cands:
        if head.K <= 1:
            break
        keep = torch.tensor([m for m in range(head.K) if m != k], device=soft.device)
        if mode == "frozen":
            base = aggregate_bound(head, buf, rstick=_clone_rstick(live_rs, head.K))
            stats = _current_stats(head.regimes)
            C, start, _ = _drop(head.ema_trans_counts, head.ema_start_counts, k)
            regimes = head.regimes.clone_with_K(head.K - 1, _select(stats, keep))
            hdp = head.hdp.resized_like(head.K - 1)
            hdp.seed_rho_omega(*drop_rho_omega(head.hdp.rho, head.hdp.omega, k))
            hdp.update(C, start, n_global_iters=3)
            cand_rs = _rstick_keep(_clone_rstick(live_rs, head.K), keep)
            cand = aggregate_bound(head, buf, regimes=regimes, hdp=hdp, rstick=cand_rs)
            if _accept(cand, base, head.K - 1, head.K, threshold, size_log_prior_odds):
                if _rec is not None:
                    _rec.record_success(int(k))
                _apply(head, regimes, hdp, C, start, rstick=cand_rs,
                       row_map=_rowmap_keep(head.K, keep, device=soft.device))
                return True, cand - base
            if _rec is not None:
                _rec.record_fail(int(k), float(soft[k]), float(getattr(head, "_lap", _lapv)))
            last_gain = cand - base
            continue
        base_rs = _clone_rstick(live_rs, head.K)
        base, base_reg, base_hdp, baseC, bases = _fit_score(
            head, head.regimes, head.hdp, buf, iters=refine_iters, rstick=base_rs)
        stats = _current_stats(base_reg)
        C, start, _ = _drop(baseC, bases, k)
        cand_rs = _rstick_keep(base_rs, keep)
        regimes, hdp = _candidate(head, _select(stats, keep), C, start, head.K - 1,
                                  seed_rho=drop_rho_omega(base_hdp.rho,
                                                          base_hdp.omega, k))
        # Restricted refinement (bnpy delete): local step ONLY on the sequences
        # that used state k. Frozen complement statistics anchor everything else,
        # and the two halves add exactly (verified to 7e-15).
        if _plan is not None and _tgt_mask:
            comp = {i: torch.ones(bb.stoch.shape[0], dtype=torch.bool,
                                  device=bb.stoch.device)
                    for i, bb in enumerate(buf.batches)}
            for bi_, mk in _tgt_mask.items():
                comp[bi_] = comp[bi_] & (~mk)
            fr_agg, fr_C, fr_s, _ = _accumulate(head, buf, regimes, hdp,
                                                rstick=cand_rs, row_mask=comp)
            if int(refine_iters) <= 0:
                # With zero refinement the loop below never runs, leaving
                # t_C/t_s unbound at the `C, start = fr_C + t_C` line below
                # (UnboundLocalError).  Accumulate the target half once so the
                # candidate is still scored on complement + target.
                _, t_C, t_s, _ = _accumulate(head, buf, regimes, hdp,
                                             rstick=cand_rs, row_mask=_tgt_mask)
            for _ in range(int(refine_iters)):
                t_agg, t_C, t_s, _ = _accumulate(head, buf, regimes, hdp,
                                                 rstick=cand_rs, row_mask=_tgt_mask)
                agg2 = {kk: fr_agg[kk] + t_agg[kk] for kk in fr_agg}
                regimes = regimes.clone_with_K(head.K - 1, agg2)
                hdp = hdp.resized_like(head.K - 1)
                hdp.update(fr_C + t_C, fr_s + t_s, n_global_iters=1)
            C, start = fr_C + t_C, fr_s + t_s
        else:
            regimes, hdp, C, start = _refine(head, regimes, hdp, buf,
                                             iters=refine_iters, rstick=cand_rs)
        cand = aggregate_bound(head, buf, regimes=regimes, hdp=hdp, rstick=cand_rs)
        if _accept(cand, base, head.K - 1, head.K, threshold, size_log_prior_odds):
            if _rec is not None:
                _rec.record_success(int(k))
            _apply(head, regimes, hdp, C, start, rstick=cand_rs,
                   row_map=_rowmap_keep(head.K, keep, device=soft.device))
            return True, cand - base
        if _rec is not None:
            _rec.record_fail(int(k), float(soft[k]), float(getattr(head, "_lap", _lapv)))
        last_gain = cand - base
    return False, last_gain


@torch.no_grad()
def merge_move(head, stoch=None, deter=None, is_first=None, *, buffer=None,
               threshold=0.0, confirm_top=None, refine_iters=1,
               size_log_prior_odds: float = 0.0, merge_select: str = "hughes",
               max_passes: int = 12, merge_topm: int | None = None,
               action=None):
    buf = _as_buffer(buffer, stoch, deter, is_first, head=head, action=action)
    any_accept = False
    total_gain = 0.0
    for _ in range(max(1, int(max_passes))):
        K = head.K
        if K <= 1:
            break
        L = head.regimes.L
        live_rs = _head_rstick(head)
        base_rs = _clone_rstick(live_rs, K)
        _base_iters = int(refine_iters) if bool(_MERGE_BASE_REFIT) else 0
        base, base_reg, base_hdp, baseC, bases = _fit_score(
            head, head.regimes, head.hdp, buf, iters=_base_iters, rstick=base_rs)
        stats = _current_stats(base_reg)

        _exact_all = bool(head.recurrent) or int(getattr(head.regimes, "q_rank", 0)) > 0
        pairs = None
        if merge_select == "hughes" and not _exact_all:
            pairs = _hughes_merge_shortlist(head, base_reg, base_hdp, baseC, bases,
                                            base_rstick=base_rs)
        if pairs is None:
            ranked = None
            if merge_select == "hughes" and _exact_all:
                ranked = _hughes_merge_shortlist(head, base_reg, base_hdp, baseC, bases,
                                                 base_rstick=base_rs)
            resid = [(i, j, _merge_gain_cached(stats, i, j, L))
                     for i in range(K) for j in range(i + 1, K)]
            resid.sort(key=lambda t: -t[2])
            if ranked:
                seen = {(i, j) for (i, j, _) in ranked}
                pairs = ranked + [p for p in resid if (p[0], p[1]) not in seen]
            else:
                pairs = resid
            if _exact_all and merge_topm is None:
                merge_topm = 40
            if _exact_all and merge_topm is not None:
                pairs = pairs[:int(merge_topm)]
            if confirm_top is not None and not _exact_all:
                pairs = pairs[:confirm_top]
        elif confirm_top is not None:
            pairs = pairs[:confirm_top]

        accepted_this_pass = False
        for (i, j, _) in pairs:
            cand_stats, keep = _merge_stats(stats, i, j)
            C, start = _merge_counts(baseC, bases, i, j)
            # The gate rows must MERGE, not merely be selected: the PG natural
            # parameters are additive over (n, t), so discarding row j drops its
            # persistence evidence.  _rstick_merge sums (pg_A, pg_h) and refits
            # the Gaussian posterior, matching the merge of the regime statistics
            # done by _merge_stats above -- and matching the construction the
            # shortlist screen already scores (_hughes_merge_shortlist).
            cand_rs = _rstick_merge(base_rs, i, j)
            regimes, hdp = _candidate(head, cand_stats, C, start, K - 1,
                                      seed_rho=merge_rho_omega(base_hdp.rho,
                                                               base_hdp.omega, i, j))
            # The PG gate merge (pg_A/pg_h addition) is an approximate
            # INITIALISER for the merged row, not an exact sufficient-statistic
            # merge like the transition counts, so a recurrent merge is refit
            # before scoring even when the MERGE_REFIT speed switch is off.
            # This costs one restricted refinement per candidate pair (up to
            # merge_topm of them per sweep), which is a real slowdown in an RL
            # loop: set SHS_MERGE_REFIT_RECURRENT=0 to restore the cheap,
            # unrefined scoring used before this behaviour was added.
            if not bool(_MERGE_REFIT) and (cand_rs is None
                                           or not _MERGE_REFIT_RECURRENT):
                pass
            else:
                # >= 1 iteration whenever a gate is present: the PG merge is an
                # initialiser, so scoring it unrefined is not a like-for-like
                # bound comparison even if the caller asked for 0 iterations.
                _it = max(1, int(refine_iters)) if cand_rs is not None else int(refine_iters)
                regimes, hdp, C, start = _refine(head, regimes, hdp, buf,
                                                 iters=_it, rstick=cand_rs)
            cand = aggregate_bound(head, buf, regimes=regimes, hdp=hdp, rstick=cand_rs)
            if _accept(cand, base, K - 1, K, threshold, size_log_prior_odds):
                _apply(head, regimes, hdp, C, start, rstick=cand_rs,
                       row_map=_rowmap_keep(K, keep, sum_into=(i, j),
                                            device=keep.device))
                any_accept = True
                accepted_this_pass = True
                total_gain += cand - base
                break
        if not accepted_this_pass:
            break
    return any_accept, total_gain


@torch.no_grad()
def _best_cut(z, w, floor=1e-6):
    W = z.shape[0]
    if W < 4:
        return max(1, W // 2)
    S0 = torch.cumsum(w, 0)
    S1 = torch.cumsum(w[:, None] * z, 0)
    S2 = torch.cumsum(w[:, None] * z * z, 0)
    n1 = S0[:-1].clamp_min(floor)
    n2 = (S0[-1] - S0[:-1]).clamp_min(floor)
    m1 = S1[:-1] / n1[:, None]
    m2 = (S1[-1] - S1[:-1]) / n2[:, None]
    v1 = (S2[:-1] / n1[:, None] - m1 ** 2).clamp_min(floor)
    v2 = ((S2[-1] - S2[:-1]) / n2[:, None] - m2 ** 2).clamp_min(floor)
    score = (-0.5 * (n1[:, None] * v1.log()).sum(-1)
             - 0.5 * (n2[:, None] * v2.log()).sum(-1))
    ok = (n1 > 10 * floor) & (n2 > 10 * floor)
    score = torch.where(ok, score, torch.full_like(score, -1e30))
    return int(score.argmax().item()) + 1


@torch.no_grad()
def _best_cut_marginal(z, g, w, regimes, min_block=3):
    W = z.shape[0]
    if W < 2 * min_block:
        return max(1, W // 2)
    wz = (w[:, None] * z)
    N_c = torch.cumsum(w, 0)
    Sgg_c = torch.cumsum(w[:, None, None] * (g[:, :, None] * g[:, None, :]), 0)
    Szg_c = torch.cumsum(w[:, None, None] * (z[:, :, None] * g[:, None, :]), 0)
    Szz_c = torch.cumsum(wz * z, 0)
    lo, hi = min_block, W - min_block
    idx = torch.arange(lo, hi, device=z.device)
    if idx.numel() == 0:
        return max(1, W // 2)
    Np, Sggp, Szgp, Szzp = N_c[idx - 1], Sgg_c[idx - 1], Szg_c[idx - 1], Szz_c[idx - 1]
    Ns = N_c[-1] - Np
    Sggs, Szgs, Szzs = Sgg_c[-1] - Sggp, Szg_c[-1] - Szgp, Szz_c[-1] - Szzp
    score = (regimes.log_marginal_from_stats(Np, Sggp, Szgp, Szzp)
             + regimes.log_marginal_from_stats(Ns, Sggs, Szgs, Szzs))
    return int(idx[int(torch.argmax(score))])


def _best_cut_ar(z, g, w, L, floor=1e-6):
    W = z.shape[0]
    if W < 6:
        return max(1, W // 2)
    wz = w[:, None] * z
    P_N = torch.cumsum(w, 0)
    P_Sgg = torch.cumsum(w[:, None, None] * (g[:, :, None] * g[:, None, :]), 0)
    P_Szg = torch.cumsum(wz[:, :, None] * g[:, None, :], 0)
    P_Szz = torch.cumsum(wz * z, 0)
    tot = (P_N[-1], P_Sgg[-1], P_Szg[-1], P_Szz[-1])
    best_c, best_s = max(1, W // 2), -float('inf')
    lo, hi = max(2, W // 10), min(W - 2, W - max(2, W // 10))
    for c in range(lo, hi + 1):
        N1 = P_N[c - 1].clamp_min(floor)
        N2 = (tot[0] - P_N[c - 1]).clamp_min(floor)
        ld1 = _resid_logdet_from(N1, P_Sgg[c - 1], P_Szg[c - 1], P_Szz[c - 1], None, L)
        ld2 = _resid_logdet_from(N2, tot[1] - P_Sgg[c - 1], tot[2] - P_Szg[c - 1],
                                 tot[3] - P_Szz[c - 1], None, L)
        sc = float(-0.5 * (N1 * ld1 + N2 * ld2))
        if sc > best_s:
            best_s, best_c = sc, c
    return int(best_c)


def _pad_stats(stats, k_extra):
    out = {}
    for k, v in stats.items():
        z = torch.zeros((k_extra,) + tuple(v.shape[1:]), dtype=v.dtype, device=v.device)
        out[k] = torch.cat([v, z], dim=0)
    return out


def _map_blocks(path, min_len=1):
    p = [int(x) for x in path.reshape(-1).tolist()]
    out, st = [], 0
    for t in range(1, len(p) + 1):
        if t == len(p) or p[t] != p[st]:
            if t - st >= min_len:
                out.append((st, t, p[st]))
            st = t
    return out


@torch.no_grad()
def seqcreate_birth_move(head, stoch=None, deter=None, is_first=None, *, buffer=None,
                         threshold=0.0, K_max=64, n_proposals=5, n_refine=3,
                         min_block=20, max_block=500, seed=None,
                         size_log_prior_odds: float = 0.0, action=None):
    buf = _as_buffer(buffer, stoch, deter, is_first, head=head, action=action)
    rng = np.random.default_rng(seed)
    idx_map = _seq_index_map(buf)
    if not idx_map:
        return False, 0.0
    any_accept, total_gain = False, 0.0
    live_rs = _head_rstick(head)

    for n in rng.permutation(len(idx_map))[:max(1, int(n_proposals))]:
        K = head.K
        if K + 2 > K_max:
            break
        base_rs = _clone_rstick(live_rs, K)
        base = aggregate_bound(head, buf, rstick=base_rs)
        bi, row = idx_map[int(n)]
        b = buf.batches[bi]
        gamma, _, _, _ = head.regime_inference(
            b.stoch, b.deter, b.is_first, valid=getattr(b, "valid", None),
            z_var=b.z_var, action=getattr(b, "action", None), cache_estep=False)
        gamma = gamma.detach()
        blocks = _map_blocks(gamma[row].argmax(-1), min_len=min_block + 1)
        if not blocks:
            continue
        rng.shuffle(blocks)
        s0, s1, _ = blocks[0]
        blen = s1 - s0
        wsize = int(rng.integers(min_block, max(min_block + 1, min(max_block, blen))))
        wstart = s0 + int(rng.integers(0, max(1, blen - wsize)))
        wstop = min(wstart + wsize, s1)
        W = wstop - wstart
        if W < min_block:
            continue

        prev = head._prev_stoch(b.stoch, b.is_first)
        gw = head.build_g(prev, b.deter,
                          head._shift_action(getattr(b, "action", None), b.is_first))
        ones = torch.ones(W, dtype=b.stoch.dtype, device=b.stoch.device)
        _use_marg = (_BIRTH_CUT == "marginal"
                     and hasattr(head.regimes, "log_marginal_from_stats"))
        cut = (_best_cut_marginal(b.stoch[row, wstart:wstop], gw[row, wstart:wstop],
                                  ones, head.regimes)
               if _use_marg else
               _best_cut_ar(b.stoch[row, wstart:wstop].double(),
                            gw[row, wstart:wstop].double(), ones.double(),
                            head.regimes.L))

        new_resp = torch.cat([gamma, gamma.new_zeros(gamma.shape[:2] + (2,))], -1)
        new_resp[row, wstart:wstop, :K] = 0.0
        new_resp[row, wstart:wstart + cut, K] = 1.0
        new_resp[row, wstart + cut:wstop, K + 1] = 1.0

        comp = {i: torch.ones(bb.stoch.shape[0], dtype=torch.bool,
                              device=bb.stoch.device)
                for i, bb in enumerate(buf.batches)}
        comp[bi][row] = False
        fr_agg, fr_C, fr_s, _ = _accumulate(head, buf, head.regimes, head.hdp,
                                            rstick=base_rs, row_mask=comp)
        fr_agg = _pad_stats(fr_agg, 2)
        fr_C = torch.nn.functional.pad(fr_C, (0, 2, 0, 2))
        fr_s = torch.nn.functional.pad(fr_s, (0, 2))

        only = {bi: torch.zeros(b.stoch.shape[0], dtype=torch.bool,
                                device=b.stoch.device)}
        only[bi][row] = True
        _zc, _gzc, _xc = _smoothed_moments(head, b)
        seedr = new_resp * only[bi].to(new_resp.dtype).reshape(-1, 1, 1)
        t_agg = head.regimes.stats_from_batch(
            seedr, b.stoch, gw, z_var=b.z_var,
            g_z_var=head._shift_var(b.z_var, b.is_first),
            z_cov=_zc, g_zcov=_gzc, zg_xcov=_xc)
        rp = head._shift_resp(seedr, b.is_first)
        t_C = torch.einsum("btj,btk->jk", rp[:, 1:], seedr[:, 1:]).double()
        t_s = seedr[:, 0].sum(0).double()

        cand_rs = _clone_rstick(live_rs, K + 2)
        agg = {k: fr_agg[k] + t_agg[k] for k in fr_agg}
        regimes, hdp = _candidate(head, agg, fr_C + t_C, fr_s + t_s, K + 2)

        for _ in range(int(n_refine)):
            t_agg, t_C, t_s, _ = _accumulate(head, buf, regimes, hdp,
                                             rstick=cand_rs, row_mask=only)
            agg = {k: fr_agg[k] + t_agg[k] for k in fr_agg}
            regimes = regimes.clone_with_K(K + 2, agg)
            hdp = hdp.resized_like(K + 2)
            hdp.update(fr_C + t_C, fr_s + t_s, n_global_iters=1)

        cand = aggregate_bound(head, buf, regimes=regimes, hdp=hdp, rstick=cand_rs)
        if _accept(cand, base, K + 2, K, threshold, size_log_prior_odds):
            _apply(head, regimes, hdp, fr_C + t_C, fr_s + t_s, rstick=cand_rs,
                   row_map=_rowmap_append(K, 2, device=head.z0.device))
            any_accept = True
            total_gain += cand - base
    return any_accept, total_gain


def interval_birth_move(head, stoch=None, deter=None, is_first=None, *, buffer=None,
                        threshold=0.0, K_max=64, refine_iters=3, min_window=8,
                        size_log_prior_odds: float = 0.0, action=None):
    buf = _as_buffer(buffer, stoch, deter, is_first, head=head, action=action)
    K = head.K
    if K + 2 > K_max:
        return False, 0.0
    live_rs = _head_rstick(head)
    base_rs = _clone_rstick(live_rs, K)
    base, _, _, _, _ = _fit_score(head, head.regimes, head.hdp, buf,
                                  iters=refine_iters, rstick=base_rs)

    best = None
    gates = []
    for bi, b in enumerate(buf.batches):
        prev = head._prev_stoch(b.stoch, b.is_first)
        g = head.build_g(prev, b.deter,
                         head._shift_action(getattr(b, "action", None), b.is_first))
        g_var = head._g_var_from_z_var(b.z_var, g, is_first=b.is_first)
        ev = head.regimes.expected_loglik(b.stoch, g, z_var=b.z_var, g_var=g_var)
        bestfit = ev.max(dim=-1).values
        scale = bestfit.std() + 1e-6
        gate = torch.sigmoid((bestfit.median() - bestfit) / scale)
        vm = getattr(b, "valid", None)
        if vm is not None:
            gate = gate * vm.reshape(gate.shape).to(gate.dtype)
        gates.append(gate)
        T = gate.shape[1]
        W = max(int(min_window), T // 8)
        if T < W:
            continue
        cs = torch.cumsum(torch.cat([gate.new_zeros(gate.shape[0], 1), gate], 1), 1)
        wsum = cs[:, W:] - cs[:, :-W]
        if b.is_first is not None:
            isf = b.is_first.to(gate.dtype)
            csf = torch.cumsum(torch.cat([isf.new_zeros(isf.shape[0], 1), isf], 1), 1)
            interior = csf[:, W:] - csf[:, 1:T - W + 2]
            wsum = torch.where(interior > 0, torch.full_like(wsum, -1e30), wsum)
        if vm is not None:
            inv = (1.0 - vm.reshape(gate.shape).to(gate.dtype))
            cinv = torch.cumsum(torch.cat([inv.new_zeros(inv.shape[0], 1), inv], 1), 1)
            winv = cinv[:, W:] - cinv[:, :-W]
            wsum = torch.where(winv > 0, torch.full_like(wsum, -1e30), wsum)
        val, flat = wsum.reshape(-1).max(0)
        row, t0 = divmod(int(flat), wsum.shape[1])
        if best is None or float(val) > best[0]:
            best = (float(val), bi, row, int(t0), W)
    if best is None or best[0] < 1.0:
        return False, 0.0
    _, bi_star, row_star, t0, W = best

    agg = None
    Cc = torch.zeros(K + 2, K + 2, dtype=torch.float64, device=head.z0.device)
    sc = torch.zeros(K + 2, dtype=torch.float64, device=head.z0.device)
    for bi, b in enumerate(buf.batches):
        gamma, _, _, _ = head.regime_inference(b.stoch, b.deter, b.is_first, valid=getattr(b,'valid',None),
                                              z_cov=getattr(b, 'z_cov', None),
                                              zg_xcov=getattr(b, 'zg_xcov', None),
                                               cache_estep=False, z_var=b.z_var,
                                               action=getattr(b, "action", None))
        newcols = gamma.new_zeros(gamma.shape[:2] + (2,))
        if bi == bi_star:
            gate = gates[bi]
            _gw = head.build_g(head._prev_stoch(b.stoch, b.is_first), b.deter,
                               head._shift_action(getattr(b, "action", None), b.is_first))
            if _BIRTH_CUT == "marginal" and hasattr(head.regimes, "log_marginal_from_stats"):
                cut = _best_cut_marginal(b.stoch[row_star, t0:t0 + W],
                                         _gw[row_star, t0:t0 + W],
                                         gate[row_star, t0:t0 + W],
                                         head.regimes)
            else:
                cut = _best_cut_ar(b.stoch[row_star, t0:t0 + W].double(),
                                   _gw[row_star, t0:t0 + W].double(),
                                   gate[row_star, t0:t0 + W].double(),
                                   head.regimes.L)
            seed_l = torch.ones_like(gate[row_star, t0:t0 + cut]) if BIRTH_SEED_HARD \
                else gate[row_star, t0:t0 + cut]
            seed_r = torch.ones_like(gate[row_star, t0 + cut:t0 + W]) if BIRTH_SEED_HARD \
                else gate[row_star, t0 + cut:t0 + W]
            newcols[row_star, t0:t0 + cut, 0] = seed_l
            newcols[row_star, t0 + cut:t0 + W, 1] = seed_r
            if BIRTH_DEBUG:
                print(f"[birth] window row={row_star} t0={t0} W={W} cut={cut} "
                      f"gate_mean={gate[row_star, t0:t0 + W].mean().item():.3f} "
                      f"seed_mass={float(newcols.sum()):.3f} hard={BIRTH_SEED_HARD}")
        scale_old = (1.0 - newcols.sum(-1, keepdim=True)).clamp_min(1e-6)
        new_resp = torch.cat([gamma * scale_old, newcols], dim=-1)
        new_resp = new_resp / new_resp.sum(-1, keepdim=True).clamp_min(1e-12)
        g = head.build_g(head._prev_stoch(b.stoch, b.is_first), b.deter,
                         head._shift_action(getattr(b, "action", None), b.is_first))
        _zc, _gzc, _xc = _smoothed_moments(head, b)
        st = head.regimes.stats_from_batch(new_resp, b.stoch, g, z_var=b.z_var,
                                           g_z_var=head._shift_var(b.z_var, b.is_first),
                                           z_cov=_zc, g_zcov=_gzc, zg_xcov=_xc)
        agg = st if agg is None else {k: agg[k] + st[k] for k in agg}
        r_prev = head._shift_resp(new_resp, b.is_first)
        if b.is_first is None:
            Cc += torch.einsum("btj,btk->jk", r_prev[:, 1:], new_resp[:, 1:]).double()
            sc += new_resp[:, 0].sum(0).double()
        else:
            mask = (1.0 - b.is_first[:, 1:].to(new_resp.dtype))
            Cc += torch.einsum("btj,btk,bt->jk", r_prev[:, 1:], new_resp[:, 1:],
                               mask).double()
            sc += (new_resp * b.is_first.reshape(*b.is_first.shape[:2], 1)
                   .to(new_resp.dtype)).sum(dim=(0, 1)).double()

    cand_rs = _clone_rstick(live_rs, K + 2)
    regimes, hdp = _candidate(head, agg, Cc, sc, K + 2)
    _refine._K_base = K
    try:
        regimes, hdp, Cc, sc = _refine(head, regimes, hdp, buf, iters=refine_iters,
                                       rstick=cand_rs)
    finally:
        _refine._K_base = None
    cand = aggregate_bound(head, buf, regimes=regimes, hdp=hdp, rstick=cand_rs)
    if BIRTH_DEBUG:
        print(f"[birth] base={base:.4f} cand={cand:.4f} delta={cand - base:+.4f} "
              f"K {K} -> {K + 2}")
    if _accept(cand, base, K + 2, K, threshold, size_log_prior_odds):
        _apply(head, regimes, hdp, Cc, sc, rstick=cand_rs,
               row_map=_rowmap_append(K, 2, device=head.z0.device))
        return True, cand - base
    return False, cand - base


@torch.no_grad()
def birth_move(head, stoch=None, deter=None, is_first=None, *, buffer=None,
               threshold=0.0, K_max=64, refine_iters=3, min_residual_mass=0.5,
               size_log_prior_odds: float = 0.0,
               action=None):
    buf = _as_buffer(buffer, stoch, deter, is_first, head=head, action=action)
    K = head.K
    if K >= K_max:
        return False, 0.0
    live_rs = _head_rstick(head)
    base_rs = _clone_rstick(live_rs, K)
    base, _, _, _, _ = _fit_score(head, head.regimes, head.hdp, buf,
                                  iters=refine_iters, rstick=base_rs)

    agg = None
    Cc = torch.zeros(K + 1, K + 1, dtype=torch.float64, device=head.z0.device)
    sc = torch.zeros(K + 1, dtype=torch.float64, device=head.z0.device)
    total_new_mass = 0.0
    for b in buf.batches:
        prev = head._prev_stoch(b.stoch, b.is_first)
        g = head.build_g(prev, b.deter,
                         head._shift_action(getattr(b, "action", None), b.is_first))
        g_var = head._g_var_from_z_var(b.z_var, g, is_first=b.is_first)
        ev = head.regimes.expected_loglik(b.stoch, g, z_var=b.z_var, g_var=g_var)
        best = ev.max(dim=-1).values
        ev_s = ev - ev.max(dim=-1, keepdim=True).values
        log_init = head.hdp.expected_log_init().to(ev_s.dtype)
        log_trans = head.hdp.expected_log_trans().to(ev_s.dtype)
        vm = getattr(b, "valid", None)
        gamma, _, _ = forward_backward(log_init, log_trans, ev_s, is_first=b.is_first,
                                       valid=vm)
        scale = best.std() + 1e-6
        gate = torch.sigmoid((best.median() - best) / scale).unsqueeze(-1)
        if vm is not None:
            gate = gate * vm.reshape(*gate.shape[:2], 1).to(gate.dtype)
        new_resp = torch.cat([gamma * (1.0 - gate), gate], dim=-1)
        new_resp = new_resp / new_resp.sum(-1, keepdim=True).clamp_min(1e-12)
        total_new_mass += float(new_resp[..., -1].sum())
        _zc, _gzc, _xc = _smoothed_moments(head, b)
        st = head.regimes.stats_from_batch(new_resp, b.stoch, g, z_var=b.z_var,
                                           g_z_var=head._shift_var(b.z_var, b.is_first),
                                           z_cov=_zc, g_zcov=_gzc, zg_xcov=_xc)
        agg = st if agg is None else {k: agg[k] + st[k] for k in agg}
        r_prev = head._shift_resp(new_resp, b.is_first)
        if b.is_first is None:
            Cc += torch.einsum("btj,btk->jk", r_prev[:, 1:], new_resp[:, 1:]).double()
        else:
            mask = (1.0 - b.is_first[:, 1:].to(new_resp.dtype))
            Cc += torch.einsum("btj,btk,bt->jk",
                               r_prev[:, 1:], new_resp[:, 1:], mask).double()
        sc += new_resp[:, 0].sum(0).double() if b.is_first is None else \
            (new_resp * b.is_first.reshape(*b.is_first.shape[:2], 1).to(new_resp.dtype)
             ).sum(dim=(0, 1)).double()

    if total_new_mass < min_residual_mass:
        return False, 0.0

    cand_rs = _clone_rstick(live_rs, K + 1)
    regimes, hdp = _candidate(head, agg, Cc, sc, K + 1)
    regimes, hdp, Cc, sc = _refine(head, regimes, hdp, buf, iters=refine_iters,
                                   rstick=cand_rs)
    cand = aggregate_bound(head, buf, regimes=regimes, hdp=hdp, rstick=cand_rs)
    if _accept(cand, base, K + 1, K, threshold, size_log_prior_odds):
        _apply(head, regimes, hdp, Cc, sc, rstick=cand_rs,
               row_map=_rowmap_append(K, 1, device=head.z0.device))
        return True, cand - base
    return False, cand - base


@torch.no_grad()
def split_move(head, stoch=None, deter=None, is_first=None, *, buffer=None,
               threshold=0.0, K_max=64, refine_iters=3, min_child_mass=0.5,
               confirm_top=2, size_log_prior_odds: float = 0.0, action=None):
    buf = _as_buffer(buffer, stoch, deter, is_first, head=head, action=action)
    K = head.K
    if K >= K_max:
        return False, 0.0
    live_rs = _head_rstick(head)
    base_rs = _clone_rstick(live_rs, K)
    base, _, _, _, _ = _fit_score(head, head.regimes, head.hdp, buf,
                                  iters=refine_iters, rstick=base_rs)
    L = head.regimes.L

    soft = torch.zeros(K, dtype=torch.float64, device=head.z0.device)
    for b in buf.batches:
        gamma, _, _, _ = head.regime_inference(b.stoch, b.deter, b.is_first, valid=getattr(b,'valid',None),
                                              z_cov=getattr(b, 'z_cov', None),
                                              zg_xcov=getattr(b, 'zg_xcov', None),
                                               cache_estep=False, z_var=b.z_var,
                                               action=getattr(b, "action", None))
        soft += gamma.sum(dim=(0, 1)).double()
    for kpar in torch.argsort(soft, descending=True).tolist()[:confirm_top]:
        if float(soft[kpar]) < 2 * min_child_mass:
            continue
        n = torch.zeros((), dtype=torch.float64, device=head.z0.device)
        msum = torch.zeros(L, dtype=torch.float64, device=head.z0.device)
        ssum = torch.zeros(L, L, dtype=torch.float64, device=head.z0.device)
        for b in buf.batches:
            gamma, _, _, _ = head.regime_inference(b.stoch, b.deter, b.is_first, valid=getattr(b,'valid',None),
                                              z_cov=getattr(b, 'z_cov', None),
                                              zg_xcov=getattr(b, 'zg_xcov', None),
                                                   cache_estep=False, z_var=b.z_var,
                                               action=getattr(b, "action", None))
            g = head.build_g(head._prev_stoch(b.stoch, b.is_first), b.deter,
                         head._shift_action(getattr(b, "action", None), b.is_first))
            comp_mean, _ = head.regimes.predictive(g)
            resid = (b.stoch - comp_mean[..., kpar, :]).double().reshape(-1, L)
            w = gamma[..., kpar].double().reshape(-1)
            n += w.sum(); msum += (w[:, None] * resid).sum(0)
            ssum += torch.einsum("n,ni,nj->ij", w, resid, resid)
        if float(n) < 2 * min_child_mass:
            continue
        mean = msum / n.clamp_min(1e-6)
        cov = ssum / n.clamp_min(1e-6) - mean[:, None] * mean[None, :]
        evecs = torch.linalg.eigh(
            cov + 1e-6 * torch.eye(L, dtype=cov.dtype, device=cov.device))[1]
        v = evecs[:, -1]

        agg = None
        Cc = torch.zeros(K + 1, K + 1, dtype=torch.float64, device=head.z0.device)
        sc = torch.zeros(K + 1, dtype=torch.float64, device=head.z0.device)
        child_mass = 0.0
        for b in buf.batches:
            gamma, _, _, _ = head.regime_inference(b.stoch, b.deter, b.is_first, valid=getattr(b,'valid',None),
                                              z_cov=getattr(b, 'z_cov', None),
                                              zg_xcov=getattr(b, 'zg_xcov', None),
                                                   cache_estep=False, z_var=b.z_var,
                                               action=getattr(b, "action", None))
            g = head.build_g(head._prev_stoch(b.stoch, b.is_first), b.deter,
                         head._shift_action(getattr(b, "action", None), b.is_first))
            comp_mean, _ = head.regimes.predictive(g)
            resid = b.stoch - comp_mean[..., kpar, :]
            proj = torch.einsum("btl,l->bt", resid, v.to(resid.dtype))
            sA = torch.sigmoid(proj / (proj.std() + 1e-6))
            wpar = gamma[..., kpar]
            new = gamma.clone(); new[..., kpar] = wpar * sA
            new = torch.cat([new, (wpar * (1.0 - sA)).unsqueeze(-1)], dim=-1)
            new = new / new.sum(-1, keepdim=True).clamp_min(1e-12)
            child_mass += float(new[..., -1].sum())
            _zc, _gzc, _xc = _smoothed_moments(head, b)
            st = head.regimes.stats_from_batch(new, b.stoch, g, z_var=b.z_var,
                                               g_z_var=head._shift_var(b.z_var, b.is_first),
                                               z_cov=_zc, g_zcov=_gzc, zg_xcov=_xc)
            agg = st if agg is None else {k: agg[k] + st[k] for k in agg}
            r_prev = head._shift_resp(new, b.is_first)
            if b.is_first is None:
                Cc += torch.einsum("btj,btk->jk", r_prev[:, 1:], new[:, 1:]).double()
            else:
                mask = (1.0 - b.is_first[:, 1:].to(new.dtype))
                Cc += torch.einsum("btj,btk,bt->jk", r_prev[:, 1:], new[:, 1:], mask).double()
            sc += new[:, 0].sum(0).double() if b.is_first is None else \
                (new * b.is_first.reshape(*b.is_first.shape[:2], 1).to(new.dtype)
                 ).sum(dim=(0, 1)).double()
        if child_mass < min_child_mass:
            continue
        cand_rs = _clone_rstick(live_rs, K + 1)
        regimes, hdp = _candidate(head, agg, Cc, sc, K + 1)
        regimes, hdp, Cc, sc = _refine(head, regimes, hdp, buf,
                                       iters=max(3, refine_iters), rstick=cand_rs)
        cand = aggregate_bound(head, buf, regimes=regimes, hdp=hdp, rstick=cand_rs)
        if _accept(cand, base, K + 1, K, threshold, size_log_prior_odds):
            _apply(head, regimes, hdp, Cc, sc, rstick=cand_rs,
                   row_map=_rowmap_append(K, 1, device=head.z0.device))
            return True, float(cand - base)
    return False, 0.0


@torch.no_grad()
def sweep_moves(head, stoch=None, deter=None, is_first=None, *, buffer=None,
                do_birth=True, do_split=True, threshold=0.0, create_bonus=0.0,
                size_log_prior_odds=None, confirm_top=None, refine_iters=3,
                delete_mode: str = "hughes", merge_select: str = "hughes",
                merge_passes: int = 12, birth_style: str = "interval",
                delete_topk: int = 3, merge_topm: int | None = None, action=None,
                seqcreate_proposals: int = 5, birth_proposals: int | None = None,
                lap: float | None = None):
    if lap is not None:
        head._lap = float(lap)
    else:
        head._lap = float(getattr(head, "_lap", 0.0)) + 1.0
    _st = getattr(head, "stat_store", None)
    if (do_birth or do_split) and _st is not None and getattr(_st, "mode", None) == "streaming" \
            and not getattr(head, "_warned_stream_birth", False):
        import warnings
        warnings.warn("birth/split on a streaming aggregate are APPROXIMATE (historical "
                      "mass outside the move buffer is not re-split); use offline_memoized "
                      "consolidation for exact model selection.")
        head._warned_stream_birth = True
    buf = _as_buffer(buffer, stoch, deter, is_first, head=head, action=action)
    if birth_proposals is not None:
        seqcreate_proposals = int(birth_proposals)
    import os as _os2
    _do_checks = _os2.environ.get("SHS_CHECKS", "off") != "off"

    if lap is not None:
        head._lap = float(lap)

    s = create_bonus if size_log_prior_odds is None else size_log_prior_odds
    cheap = max(1, refine_iters // 2)
    log = {}
    if lap is not None:
        head._lap = float(lap)
    import os as _o
    _chk = _o.environ.get("SHS_CHECKS", "off") != "off"
    def _verify(tag, res):
        if _chk and isinstance(res, tuple) and res[0]:
            from .checks import verify_bound_tracking
            base_now = aggregate_bound(head, buf, rstick=_clone_rstick(_head_rstick(head), head.K))
            verify_bound_tracking(head, buf, base_now, tag=tag,
                                  aggregate_fn=lambda h, b: aggregate_bound(
                                      h, b, rstick=_clone_rstick(_head_rstick(h), h.K)))
        return res
    # Prune-first ordering (Hughes 2015 Sec. 4; settings-bnpyHDPHMMdelmerge):
    # shrink moves run before growth so a birth is proposed against an already
    # consolidated model, and a state born this lap cannot be merged away in the
    # same lap before it has been fit.
    log["merge"] = _verify("merge", merge_move(head, buffer=buf, threshold=threshold, refine_iters=cheap,
                              confirm_top=confirm_top, size_log_prior_odds=s,
                              merge_select=merge_select, max_passes=merge_passes, merge_topm=merge_topm))
    log["delete"] = _verify("delete", delete_move(head, buffer=buf, threshold=threshold,
                                size_log_prior_odds=s, refine_iters=cheap,
                                mode=delete_mode, delete_topk=delete_topk))
    if do_birth:
        if birth_style == "seqcreate":
            log["birth"] = _verify("birth", seqcreate_birth_move(
                head, buffer=buf, threshold=threshold, size_log_prior_odds=s,
                n_proposals=int(seqcreate_proposals)))
        elif birth_style == "interval":
            log["birth"] = _verify("birth", interval_birth_move(
                head, buffer=buf, threshold=threshold,
                refine_iters=max(3, refine_iters), size_log_prior_odds=s))
        else:
            log["birth"] = _verify("birth", birth_move(
                head, buffer=buf, threshold=threshold,
                refine_iters=max(3, refine_iters), size_log_prior_odds=s))
    if do_split:
        log["split"] = _verify("split", split_move(
            head, buffer=buf, threshold=threshold, refine_iters=max(3, refine_iters),
            confirm_top=confirm_top, size_log_prior_odds=s))
    accepted_any = any(isinstance(v, tuple) and bool(v[0]) for v in log.values())
    if accepted_any and buf is not None and getattr(buf, "complete", False):
        head.resync_store_from_buffer(buf)
    return log
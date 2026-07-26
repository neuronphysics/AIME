"""Adaptive-K structure moves for the regime mixture: birth, split, merge, and delete.

Each move proposes a change to the number of active regimes K, refines the candidate with a
short local variational E/M loop on a held-out buffer of recent latents, and accepts it only
if it improves the structured variational bound aggregated over that buffer. The acceptance
follows Hughes et al. (NIPS 2015): data-driven proposals of any form, verified by the EXACT
whole-buffer surrogate ELBO. Seeds are data-driven heuristics (as Hughes' births are); the
accept/reject decision never is.

Fidelity notes relative to the previous revision:
  * Global complexity terms (regime param KL, stickiness beta KL, HDP allocation) are counted
    ONCE per scored set, not once per batch, matching Hughes' L = Ldata+Lentropy+Llocal+Lglobal
    on the aggregated corpus. The allocation term uses the exact linear-slack form, so frozen
    (not-refit) globals are scored by their true ELBO.
  * When recurrent stickiness is active, base and candidates are scored under the SAME
    fully variational recurrent model (PG/JJ-bounded augmented potentials): candidates carry a
    properly row-mapped, buffer-refit stickiness posterior and pay its KL symmetrically.
  * Merge candidate selection implements Hughes' entropy-free test (Sec. 4 of the NIPS 2015
    paper): a pair is shortlisted iff L'_data + L'_alloc - param_kl' exceeds the base value,
    a criterion that is guaranteed to contain every ELBO-improving merge because the path
    entropy can only decrease under a merge. Acceptance still re-verifies with the exact
    bound. The older residual-log-det ranking remains available as merge_select="residual".
  * Delete follows Hughes: drop the state's row/column from the counts and its sufficient
    statistics, REFIT the candidate on the buffer (data-driven reassignment of the deleted
    state's mass), and accept iff the exact whole-buffer bound improves. The previous
    frozen-globals variant is available as delete_mode="frozen" (now scored exactly, thanks
    to the slack term).
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from .regimes import DiagARRegimes
from .sticky_hdp import StickyHDP
from .forward_backward import forward_backward, start_counts_from


# ----------------------------------------------------------------- scoring set
@dataclass
class _Batch:
    stoch: torch.Tensor
    deter: torch.Tensor
    is_first: torch.Tensor | None
    z_var: torch.Tensor | None = None
    step: int = 0
    batch_id: object = None
    repr_version: int | None = None


class MoveBuffer:
    """Fixed held-out batches the move ELBO is aggregated over.

    This is the 'memoized' scoring set: the variational bound a move must improve is
    aggregated over THESE batches, not a single training minibatch. Keep it stable
    across a move sweep (Dreamer's replay is non-stationary, so do NOT feed fresh
    training minibatches as the score).

    Because the encoder/RSSM keep changing, latents added long ago no longer reflect the
    current representation. `max_age` (in world-model updates) evicts entries older than that
    so the score stays representation-consistent; 0 disables age-based eviction and only the
    `max_batches` ring cap applies.
    """
    def __init__(self, max_batches: int = 8, max_age: int = 0,
                 complete: bool = False, expected_batches: int | None = None,
                 expected_ids=None):
        """complete=True turns the buffer into a WHOLE-CORPUS contract: batches carry
        stable ids (same id replaces, never duplicates), ring eviction is disabled,
        and moves refuse to score until every one of `expected_batches` ids is
        present. Under that contract an accepted move's positive gain IS a
        whole-corpus ELBO improvement (the Hughes guarantee). The default ring
        buffer remains a recent-window surrogate, as documented."""
        self.max_batches = max_batches
        self.max_age = max_age
        self.complete = bool(complete)
        self.expected_batches = expected_batches
        self.expected_ids = set(expected_ids) if expected_ids is not None else None
        self.batches: list[_Batch] = []

    def add(self, stoch, deter, is_first=None, z_var=None, step=0,
            batch_id=None, repr_version=None, action=None):
        rv = None if repr_version is None else int(repr_version)
        entry = _Batch(
            stoch.detach(), deter.detach(),
            None if is_first is None else is_first.detach(),
            None if z_var is None else z_var.detach(), int(step), batch_id, rv)
        entry.action = None if action is None else action.detach()
        if batch_id is not None:
            for i, b in enumerate(self.batches):
                if b.batch_id == batch_id:
                    self.batches[i] = entry          # stable-id replace, never duplicate
                    return
        self.batches.append(entry)
        if not self.complete:
            if self.max_age > 0:
                self.batches = [b for b in self.batches if step - b.step <= self.max_age]
            while len(self.batches) > self.max_batches:
                self.batches.pop(0)

    def purge_stale(self, current_version) -> int:
        """Drop every buffered batch whose repr_version differs from the current
        one. A live ring can never legitimately hold pre-bump latents (the raw
        observations are gone), so staleness is PURGED, never silently mixed. 
        Returns the number of entries dropped."""
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
            return False                       # anonymous or duplicated partitions
        if self.expected_ids is not None:
            return set(ids) == self.expected_ids     # EXACT set: no foreign/missing ids
        if self.expected_batches is None:
            return False                       # no certificate declared -> never complete
        return len(set(ids)) == self.expected_batches

    def __len__(self):
        return len(self.batches)


def _as_buffer(buffer, stoch, deter, is_first, head=None, action=None):
    if buffer is not None and len(buffer) > 0:
        _validate_buffer(head, buffer)
        return buffer
    b = MoveBuffer(max_batches=1)
    b.add(stoch, deter, is_first, action=action)   # review Important #3: fallback API carries the action
    return b


def _validate_buffer(head, buf):
    """Enforce the completeness contract and representation consistency.

    * complete=True buffers must actually be complete before any move is scored;
      otherwise 'positive gain' would silently mean gain on a partial corpus.
    * If batches are stamped with a representation version, every stamped batch
      must match the head's current `repr_version`: latents produced by different
      encoder states are not one corpus, and scoring across them voids the
      whole-corpus interpretation of the bound.
    """
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
        # The single-version requirement applies ONLY to a COMPLETE (whole-corpus)
        # buffer, whose "gain on the whole corpus" reading needs one encoder version.
        # A live (non-complete) ring is a recent-data proposal buffer scored by exact
        # acceptance; mixing recent versions in it is sound and must NOT raise.
        cur = int(head.repr_version)
        stale = [b.batch_id for b in buf.batches
                 if b.repr_version is not None and b.repr_version != cur]
        if stale:
            raise RuntimeError(
                f"complete MoveBuffer mixes representation version(s) other than the "
                f"head's current {cur} (ids {stale[:4]}...): a whole corpus must be "
                "encoded once under a frozen representation (see fit_offline_corpus).")


# ----------------------------------------------------------- recurrent-stickiness helpers
def _head_rstick(head):
    """The live stickiness module if recurrence is on, else None."""
    if getattr(head, "recurrent", False) and getattr(head, "rstick", None) is not None:
        return head.rstick
    return None


def _clone_rstick(rs, new_K: int):
    """A detached working copy at truncation new_K (rows 0..min-1 preserved)."""
    return None if rs is None else rs.resized_like(int(new_K))


def _rstick_keep(rs, keep_idx):
    """A detached working copy keeping exactly the rows `keep_idx` (delete/merge)."""
    return None if rs is None else rs.select_rows(keep_idx)


# ----------------------------------------------------------------- aggregated bound
@torch.no_grad()
def aggregate_bound(head, buffer, regimes=None, hdp=None, rstick=None) -> float:
    """Exact frozen variational ELBO of the WHOLE buffer under the given globals.

    Local log-partitions are summed per batch; the global complexity terms (regime
    param KL, stickiness beta KL when recurrence is active, and the exact-slack HDP
    allocation term) are added ONCE, so the score is the surrogate ELBO of the
    aggregated corpus, the quantity Hughes' moves must improve. Low-rank Q uses the
    exact Woodbury evidence, not a diagonal scoring shortcut.
    """
    local = 0.0
    for b in buffer.batches:
        lz, _, _ = head.bound_local(b.stoch, b.deter, b.is_first, regimes=regimes,
                                    hdp=hdp, rstick=rstick, z_var=b.z_var,
                                    action=getattr(b, "action", None))
        local += lz
    return float(local + head.bound_global(regimes=regimes, hdp=hdp, rstick=rstick))


# ----------------------------------------------------------------- local accumulation
@torch.no_grad()
def _accumulate(head, buffer, regimes, hdp, rstick=None):
    """One E-step over the buffer with the given candidate globals.

    Runs forward-backward under the candidate's potentials -- the PG/JJ-bounded
    time-varying augmented potentials when a stickiness module is supplied, else the
    stationary base E[log pi]. Returns summed regime sufficient statistics, summed
    BASE-branch transition counts, summed start counts, and (when recurrent) the
    concatenated Polya-Gamma statistics (phi, r_mass, row_weight) for the stickiness
    M-step, so candidate refinement is coordinate ascent on the full recurrent ELBO.
    """
    K = regimes.K
    agg = None
    Cc = torch.zeros(K, K, dtype=torch.float64, device=head.z0.device)
    sc = torch.zeros(K, dtype=torch.float64, device=head.z0.device)
    pg_phi, pg_r, pg_w = [], [], []
    for b in buffer.batches:
        prev = head._prev_stoch(b.stoch, b.is_first)
        g = head.build_g(prev, b.deter,
                         head._shift_action(getattr(b, "action", None), b.is_first))
        g_var = head._g_var_from_z_var(b.z_var, g, is_first=b.is_first)
        ev = regimes.expected_loglik(b.stoch, g, z_var=b.z_var, g_var=g_var).double()
        log_init, log_trans, aux = head._score_potentials(b.deter, hdp, rstick)
        gamma, xicount, _, xi = forward_backward(
            log_init, log_trans, ev, is_first=b.is_first, return_pairwise=True)
        if aux is not None:
            r_mass, row_weight, counts = rstick.attribute_bound(xi, aux)
            Dp = aux["phi_steps"].shape[-1]
            pg_phi.append(aux["phi_steps"].reshape(-1, Dp))
            pg_r.append(r_mass.reshape(-1, K))
            pg_w.append(row_weight.reshape(-1, K))
        else:
            counts = xicount
        gamma = gamma.to(b.stoch.dtype)
        st = regimes.stats_from_batch(gamma, b.stoch, g, z_var=b.z_var,
                                      g_z_var=head._shift_var(b.z_var, b.is_first))
        agg = st if agg is None else {k: agg[k] + st[k] for k in agg}
        Cc += counts.double()
        sc += start_counts_from(gamma, b.is_first).double()
    pg = None
    if pg_phi:
        pg = dict(phi=torch.cat(pg_phi, 0), r_mass=torch.cat(pg_r, 0),
                  row_weight=torch.cat(pg_w, 0))
    return agg, Cc, sc, pg


@torch.no_grad()
def _refine(head, regimes, hdp, buffer, iters: int = 2, hdp_iters: int = 3, rstick=None):
    """Restricted local VB at the candidate K: alternate the E-step (forward-backward
    under the candidate potentials) with closed-form regime, HDP, and (when recurrent)
    Polya-Gamma stickiness M-steps on the buffer. Turns a one-shot birth/merge/delete
    SEED into a local optimum of the full variational bound BEFORE it is scored.
    `rstick` is refit IN PLACE (callers pass a working clone, never the live module).
    """
    K = regimes.K
    for _ in range(iters):
        agg, Cc, sc, pg = _accumulate(head, buffer, regimes, hdp, rstick=rstick)
        regimes = regimes.clone_with_K(K, agg)          # installs stats + closed-form M-step
        hdp = hdp.resized_like(K)
        hdp.update(Cc, sc, n_global_iters=hdp_iters)
        if rstick is not None and pg is not None:
            rstick.pg_update_statewise(pg["phi"], pg["r_mass"], pg["row_weight"], lr=None)
    # one final E-step so the returned counts are consistent with the returned globals
    _, Cc, sc, _ = _accumulate(head, buffer, regimes, hdp, rstick=rstick)
    return regimes, hdp, Cc, sc


@torch.no_grad()
def _fit_score(head, regimes, hdp, buffer, iters, rstick=None):
    """Fit the given globals to the buffer at their K, then return (bound, globals).

    Used for BOTH the current structure and a candidate so the two ELBOs are
    comparable: each is the exact variational bound of the SAME data at its own K,
    including symmetric parameter-complexity terms. When recurrence is active the
    stickiness posterior is refit on the buffer too (callers pass a working clone).
    """
    regimes, hdp, C, s = _refine(head, regimes, hdp, buffer, iters=iters, rstick=rstick)
    bound = aggregate_bound(head, buffer, regimes=regimes, hdp=hdp, rstick=rstick)
    return bound, regimes, hdp, C, s


# ----------------------------------------------------------------- stat surgery
def _select(stats, idx):
    return {k: v[idx].clone() for k, v in stats.items()}


def _current_stats(regimes: DiagARRegimes):
    # Under the shared-carry dynamics the regime regression is on the carry residual
    # z~ = z - C h~ (C tied across regimes, so z~ is regime-independent); the residual-logdet
    # merge/delete scoring must therefore use the C-residualised diagonal Szz_resid, not the raw
    # second moment. DiagARRegimes has no Szz_resid, so it falls back to its own Szz.
    Szz = getattr(regimes, "Szz_resid", regimes.Szz)
    s = dict(N=regimes.N.clone(), Sgg=regimes.Sgg.clone(),
             Szg=regimes.Szg.clone(), Szz=Szz.clone())
    if regimes.q_rank > 0:
        # shared carry exposes the C-residualised full moment; DiagARRegimes has only Szz_full
        s["Szz_full"] = getattr(regimes, "Szz_full_resid", regimes.Szz_full).clone()
    if getattr(regimes, "q_rank", 0) > 0 and hasattr(regimes, "Szf"):
        # Bayes-U factor statistics ride the candidate stats, so merge/delete
        # candidates refit q(U) from REAL evidence via _update_qU instead of
        # collapsing every regime to the zero-loading saddle 
        for nm in ("Szf", "Sfr", "Sfh", "Sff"):
            s[nm] = getattr(regimes, nm).clone()
    return s


def _merge_stats(stats, i, j):
    K = stats["N"].shape[0]
    s = {k: v.clone() for k, v in stats.items()}
    # The factor statistics Szf/Sfr/Sfh/Sff live in each regime's ARBITRARY factor
    # basis (the low-rank model is invariant under U_k -> U_k R, f_tk -> R^T f_tk for
    # orthogonal R), so summing them across two regimes is basis-inconsistent. 
    #Correctness is already guaranteed downstream --
    # _refine recomputes q(f) and every factor moment from the buffer under the merged
    # U, so the SCORED bound does not depend on the seed basis -- but we also make the
    # SEED basis-consistent: the merged row inherits the dominant parent's factor
    # statistics (its basis becomes the seed basis) instead of an incoherent sum.
    _factor = ("Szf", "Sfr", "Sfh", "Sff")
    dom = i if float(stats["N"][i]) >= float(stats["N"][j]) else j
    for k in s:
        if k in _factor:
            s[k][i] = s[k][dom].clone()           # basis-consistent seed; refine recomputes
        else:
            s[k][i] = s[k][i] + s[k][j]           # regression stats are basis-free: sum
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
    """log|Sigma_res| for a component with the given (cached) sufficient statistics, where
    Sigma_res = (E[zz^T] - Szg Sgg^{-1} Szg^T) / N is the residual covariance after the
    optimal linear fit. This is the data-fit part of the component's variational evidence and
    is a pure function of the memoized statistics: no E-step over data. Diagonal Q uses only
    the diagonal of the residual; low-rank uses the full L x L residual log-det.
    """
    N = N.clamp_min(1e-6)
    G = Sgg.shape[-1]
    A = Sgg + ridge * torch.eye(G, dtype=Sgg.dtype, device=Sgg.device)
    sol = torch.linalg.solve(A, Szg.transpose(-1, -2))        # (G,L) = Sgg^{-1} Szg^T
    explained = Szg @ sol                                     # (L,L)
    if Szz_full is not None:
        Sres = (Szz_full - explained) / N
        Sres = Sres + 1e-6 * torch.eye(L, dtype=Sres.dtype, device=Sres.device)
        return torch.linalg.slogdet(Sres)[1]
    resid = (Szz - torch.diagonal(explained)) / N
    return torch.log(resid.clamp_min(1e-8)).sum()


def _merge_gain_cached(stats, i, j, L):
    """Change in completed-data evidence from merging regimes i and j, from cached stats only
    (no E-step). Returns -1/2 [N_ij log|Sres_ij| - N_i log|Sres_i| - N_j log|Sres_j|], which is
    <= 0 (merging never improves the data fit); the LEAST-negative pair is the most mergeable.
    Kept as the merge_select="residual" fast ranking; the default Hughes test below is the
    guaranteed-complete selection criterion.
    """
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


# --------------------------------------------- Hughes entropy-free merge selection
@torch.no_grad()
def _entropy_free_score(head, regimes, hdp, rstick=None):
    """Ldata + Lalloc - param_kl from CACHED statistics (no E-step, no entropy term).

    This is the left/right side of Hughes' merge-selection inequality (NIPS 2015,
    Sec. 4), and it reproduces bnpy's `calcHardMergeGap` exactly. Key accounting
    point, verified numerically against bnpy in tests_bnpy_parity: because the
    candidate HDP is refit to its own counts (theta = M + alpha*E[beta] + kappa*I),
    bnpy's slack term  <M + prior - theta, E[log pi]>  is IDENTICALLY ZERO, so the
    linear allocation term collapses to  L_top - c_Dir(transTheta) - c_Dir(startTheta),
    i.e. exactly `StickyHDP.alloc_elbo()` (the afterGlobalStep form). The transition
    count-mass <M, E[log pi]> lives in bnpy's (cancelled) slack and in the FB
    log-partition, NOT in this data-free merge score. Using `exact_alloc_elbo` here
    would wrongly subtract <M,P> + <s,P0>, whose value depends on how counts split
    across the merge candidates, and could flip the shortlist ranking. `alloc_elbo`
    is the bnpy-faithful term.

    Requires the regime model to expose `data_elbo_from_stats`; returns None
    otherwise (caller falls back to the residual ranking).
    """
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
    """All pairs passing Hughes' entropy-free test, sorted by descending gain.

    For each pair (i, j) the candidate globals come from ONE global step on the merged
    summaries (pooled regime statistics, closed-form regime M-step inside clone_with_K;
    merged counts, one HDP global update) -- exactly Hughes' candidate construction --
    and the pair is kept iff the entropy-free score improves. The exact confirmation
    with the full bound (entropy included, via forward-backward on the buffer) happens
    in merge_move.
    """
    K = base_reg.K
    if K * (K - 1) // 2 > pairs_cap:
        # each pair costs one HDP global step (L-BFGS on rho/omega); above the cap the
        # caller falls back to the fast residual ranking, whose top candidates are still
        # verified with the exact bound (sound, not guaranteed complete at this scale)
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
            cand_hdp.update(C, start, n_global_iters=hdp_iters)
            cand_rstick = _rstick_keep(base_rstick, keep)
            score = _entropy_free_score(head, cand_reg, cand_hdp, rstick=cand_rstick)
            gain = score - base_score
            if gain > 0.0:
                out.append((i, j, gain))
    out.sort(key=lambda t: -t[2])
    return out


def _candidate(head, regime_stats, C, start, new_K, hdp_iters=3):
    regimes = head.regimes.clone_with_K(new_K, regime_stats)
    hdp = head.hdp.resized_like(new_K)
    hdp.update(C.double(), start.double(), n_global_iters=hdp_iters)
    return regimes, hdp


def _rowmap_keep(K_old, keep, sum_into=None, device=None):
    """(K_new, K_old) map for delete/merge: gather kept rows; for a merge, the row that
    keeps index i also receives old row j (statistic caches SUM under a merge, unlike
    the rstick posterior which selects row i as-is)."""
    M = torch.zeros(len(keep), K_old, dtype=torch.float64, device=device)
    M[torch.arange(len(keep)), keep.to(M.device)] = 1.0
    if sum_into is not None:
        i, j = sum_into
        pos = int((keep == i).nonzero()[0].item())
        M[pos, j] = 1.0
    return M


def _rowmap_append(K_old, m, device=None):
    """(K_old+m, K_old) map for birth/split: identity plus m all-zero fresh rows."""
    return torch.cat([torch.eye(K_old, dtype=torch.float64, device=device),
                      torch.zeros(m, K_old, dtype=torch.float64, device=device)], 0)


def _apply(head, regimes, hdp, C, start, rstick=None, row_map=None):
    K_old = int(head.K)
    head.regimes = regimes
    if hasattr(regimes, "_freeze_C"):
        # Scoring clones freeze the tied carry C; the
        # ACCEPTED model must resume learning C or every later episode trains
        # against a frozen drift.
        regimes._freeze_C = False
    head.hdp = hdp
    head.K = regimes.K
    if hasattr(head, "bump_struct_gen"):
        head.bump_struct_gen()   # re-identify states for async deltas
    if getattr(head, "rstick", None) is not None:
        if rstick is not None:
            # install the candidate's buffer-refit, row-mapped stickiness posterior
            head.rstick = rstick
        else:
            # legacy path (e.g. demo): resize, preserving common rows
            head.rstick = head.rstick.resized_like(regimes.K)
    head.register_buffer("ema_trans_counts", C.double().clone())
    head.register_buffer("ema_start_counts", start.double().clone())
    head._counts_initialised = True
    # keep the online sufficient-statistic ledger consistent with the new row layout
    store = getattr(head, "stat_store", None)
    if store is not None and getattr(store, "mode", "legacy_ema") != "legacy_ema":
        if row_map is not None:
            store.remap(row_map)
        elif int(regimes.K) != K_old:
            store.reset()      # unknown mapping: rebuild the ledger over the next pass


# --------------------------------------------------- model-size prior on active regimes
def _active_size_log_prior(K: int, log_odds: float) -> float:
    """log p(K) up to an additive constant, for an exponential/geometric prior on the number
    of active regimes with per-regime log-odds `log_odds` (proper up to the truncation K_max).

    log_odds == 0 is the neutral prior, so the penalized objective below reduces to the plain
    structured ELBO. log_odds > 0 favours MORE regimes (used early to resist premature
    collapse), log_odds < 0 favours fewer (an Occam/MDL bias).
    """
    return float(log_odds) * float(K)


def _accept(cand_bound: float, base_bound: float, K_cand: int, K_base: int,
            threshold: float, size_log_prior_odds: float) -> bool:
    """Accept a structural move iff it increases the SIZE-PENALIZED ELBO

        J(model) = L(model) + log p(K_active),     log p(K) = size_log_prior_odds * K,

    by more than `threshold`. For fixed size_log_prior_odds this is exact monotone ascent on
    a single well-defined objective (the ELBO under an explicit prior on the number of active
    regimes); the acceptance is a MAP structure decision, not a heuristic threshold offset.
    Annealing size_log_prior_odds -> 0 over training is a continuation / graduated-optimization
    schedule: monotone within each phase, converging to the plain-ELBO objective. With
    size_log_prior_odds == 0 it is strictly monotone on L itself.
    """
    j_cand = cand_bound + _active_size_log_prior(K_cand, size_log_prior_odds)
    j_base = base_bound + _active_size_log_prior(K_base, size_log_prior_odds)
    return j_cand > j_base + threshold


# ----------------------------------------------------------------- moves
@torch.no_grad()
def delete_move(head, stoch=None, deter=None, is_first=None, *, buffer=None,
                min_mass=None, threshold=0.0, size_log_prior_odds: float = 0.0,
                refine_iters: int = 2, mode: str = "hughes", delete_topk: int = 3,
                action=None):
    """Remove a rarely-used regime, Hughes-style.

    An occupancy gate proposes the least-used regime (Hughes gates delete candidates by
    usage too, Sec. 4). mode="hughes" (default): both sides are refit on the buffer --
    the base at K, the candidate at K-1 after dropping the state's sufficient
    statistics and its count row/column -- so the deleted state's mass is reassigned in
    a data-driven way by the candidate's restricted E/M loop, and the move is accepted
    iff the exact whole-buffer bound improves. mode="frozen" keeps the previous
    behaviour (both sides scored on the installed globals, no refit, isolating the
    question 'is this state load-bearing right now?'); with the exact-slack allocation
    term this score is now the true ELBO of the frozen model rather than an
    afterGlobalStep approximation.
    """
    buf = _as_buffer(buffer, stoch, deter, is_first, head=head, action=action)
    if head.K <= 1:
        return False, 0.0
    live_rs = _head_rstick(head)
    # soft occupancy across the whole buffer under the installed globals
    soft = torch.zeros(head.K, dtype=torch.float64, device=head.z0.device)
    for b in buf.batches:
        gamma, _, _, _ = head.regime_inference(b.stoch, b.deter, b.is_first,
                                               cache_estep=False, z_var=b.z_var,
                                               action=getattr(b, "action", None))
        soft += gamma.sum(dim=(0, 1)).double()
    # Hughes gates delete candidates by usage ("states used in 10 or fewer
    # sequences" / target mass 0.01); the soft-mass analogue: any state below 1% of
    # total corpus mass is a candidate (min_mass=None), and up to `delete_topk` of the
    # least-occupied candidates are each independently verified with the exact
    # whole-buffer bound -- not only the single least-used state.
    if min_mass is None:
        min_mass = 0.01 * float(soft.sum())
    order = torch.argsort(soft).tolist()
    cands = [int(j) for j in order if float(soft[j]) < float(min_mass)]
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
            hdp.update(C, start, n_global_iters=3)
            cand_rs = _rstick_keep(live_rs, keep)
            cand = aggregate_bound(head, buf, regimes=regimes, hdp=hdp, rstick=cand_rs)
            if _accept(cand, base, head.K - 1, head.K, threshold, size_log_prior_odds):
                _apply(head, regimes, hdp, C, start, rstick=cand_rs,
                       row_map=_rowmap_keep(head.K, keep, device=soft.device))
                return True, cand - base
            last_gain = cand - base
            continue
        # ---- Hughes mode: symmetric buffer refit on both sides ----
        base_rs = _clone_rstick(live_rs, head.K)
        base, base_reg, base_hdp, baseC, bases = _fit_score(
            head, head.regimes, head.hdp, buf, iters=refine_iters, rstick=base_rs)
        stats = _current_stats(base_reg)
        C, start, _ = _drop(baseC, bases, k)
        cand_rs = _rstick_keep(base_rs, keep)
        regimes, hdp = _candidate(head, _select(stats, keep), C, start, head.K - 1)
        regimes, hdp, C, start = _refine(head, regimes, hdp, buf, iters=refine_iters,
                                         rstick=cand_rs)
        cand = aggregate_bound(head, buf, regimes=regimes, hdp=hdp, rstick=cand_rs)
        if _accept(cand, base, head.K - 1, head.K, threshold, size_log_prior_odds):
            _apply(head, regimes, hdp, C, start, rstick=cand_rs,
                   row_map=_rowmap_keep(head.K, keep, device=soft.device))
            return True, cand - base
        last_gain = cand - base
    return False, last_gain


@torch.no_grad()
def merge_move(head, stoch=None, deter=None, is_first=None, *, buffer=None,
               threshold=0.0, confirm_top=None, refine_iters=1,
               size_log_prior_odds: float = 0.0, merge_select: str = "hughes",
               max_passes: int = 3, merge_topm: int | None = None,
               action=None):
    """Merge redundant regimes.

    Selection (merge_select="hughes", default): every pair is tested with Hughes'
    entropy-free criterion from cached statistics -- complete, because a merge can only
    lower the path entropy, so any ELBO-improving merge must pass it. Confirmation:
    each shortlisted pair (best gain first, up to `confirm_top`; None = all) is refit
    on the buffer and accepted iff the EXACT whole-buffer bound improves. After an
    accept the pass restarts on the new base (up to `max_passes`), so multiple
    disjoint merges can land in one sweep, as in Hughes' laps. merge_select="residual"
    restores the previous fast residual-log-det ranking (sound shortlist, not
    guaranteed complete).
    """
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
        # Fit the CURRENT structure to this buffer first, and rank/seed merges from those
        # buffer-consistent statistics, not the installed online EMA.
        base, base_reg, base_hdp, baseC, bases = _fit_score(
            head, head.regimes, head.hdp, buf, iters=refine_iters, rstick=base_rs)
        stats = _current_stats(base_reg)

        # The entropy-free shortlist is a valid necessary condition ONLY for the
        # diagonal stationary model. For the recurrent model it omits the PG/JJ expected
        # transition-likelihood term, and for q_rank>0 the factor evidence is a diagonal
        # surrogate -- either can reject a pair that improves the full bound. So for those
        # models we evaluate EVERY pair with the exact aggregated bound: 
        # correct, at O(K^2) exact confirmations.
        _exact_all = bool(head.recurrent) or int(getattr(head.regimes, "q_rank", 0)) > 0
        pairs = None
        if merge_select == "hughes" and not _exact_all:
            pairs = _hughes_merge_shortlist(head, base_reg, base_hdp, baseC, bases,
                                            base_rstick=base_rs)
        if pairs is None:                                   # fallback: residual ranking
            pairs = [(i, j, _merge_gain_cached(stats, i, j, L))
                     for i in range(K) for j in range(i + 1, K)]
            pairs.sort(key=lambda t: -t[2])
            if _exact_all and merge_topm is not None:
                # Hughes top-M SCREENING for the recurrent /
                # low-rank models. The residual ranking is a PROPOSAL heuristic only --
                # acceptance below is still the exact refit + whole-buffer bound, so a
                # harmful merge can never be accepted; a beneficial one can be missed.
                # Cost drops from O(R N K^4) (all pairs refit) to O(N K^2 + M R N K^2).
                pairs = pairs[:int(merge_topm)]
            if confirm_top is not None and not _exact_all:
                pairs = pairs[:confirm_top]
        elif confirm_top is not None:
            pairs = pairs[:confirm_top]

        accepted_this_pass = False
        for (i, j, _) in pairs:
            cand_stats, keep = _merge_stats(stats, i, j)
            C, start = _merge_counts(baseC, bases, i, j)   # buffer-refit transition counts
            cand_rs = _rstick_keep(base_rs, keep)
            regimes, hdp = _candidate(head, cand_stats, C, start, K - 1)
            regimes, hdp, C, start = _refine(head, regimes, hdp, buf, iters=refine_iters,
                                             rstick=cand_rs)
            cand = aggregate_bound(head, buf, regimes=regimes, hdp=hdp, rstick=cand_rs)
            if _accept(cand, base, K - 1, K, threshold, size_log_prior_odds):
                _apply(head, regimes, hdp, C, start, rstick=cand_rs,
                       row_map=_rowmap_keep(K, keep, sum_into=(i, j),
                                            device=keep.device))
                any_accept = True
                accepted_this_pass = True
                total_gain += cand - base
                break                                       # restart pass on the new base
        if not accepted_this_pass:
            break
    return any_accept, total_gain


@torch.no_grad()
def _best_cut(z, w, floor=1e-6):
    """Linear-time optimal two-block cut (the Hughes NIPS'15 birth construction ships a
    linear-time search maximizing the data term): choose the cut c maximizing the
    weighted diagonal-Gaussian log-likelihood of the window split into two contiguous
    blocks. Prefix sums of (w, w z, w z^2) give every cut's block moments, so the whole
    search is O(W L) -- dynamic programming, not the O(W^2 L) naive rescan."""
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


def interval_birth_move(head, stoch=None, deter=None, is_first=None, *, buffer=None,
                        threshold=0.0, K_max=64, refine_iters=3, min_window=8,
                        size_log_prior_odds: float = 0.0, action=None):
    """Hughes/x-hdphmm-style CONTIGUOUS-INTERVAL birth: locate the single
    worst-explained contiguous window in the buffer, seed TWO fresh regimes on its
    first and second halves (a K+2 candidate), refine with restricted VB, and accept
    iff the exact aggregated bound improves. The two-substate window seed is the
    NIPS-2015 HMM birth construction; like every proposal here it is data-driven and
    the acceptance is always the exact bound."""
    buf = _as_buffer(buffer, stoch, deter, is_first, head=head, action=action)
    K = head.K
    if K + 2 > K_max:
        return False, 0.0
    live_rs = _head_rstick(head)
    base_rs = _clone_rstick(live_rs, K)
    base, _, _, _, _ = _fit_score(head, head.regimes, head.hdp, buf,
                                  iters=refine_iters, rstick=base_rs)

    # locate the worst-explained window across all sequences in the buffer
    best = None                              # (score, batch_idx, row, t0, W)
    gates = []
    for bi, b in enumerate(buf.batches):
        prev = head._prev_stoch(b.stoch, b.is_first)
        g = head.build_g(prev, b.deter,
                         head._shift_action(getattr(b, "action", None), b.is_first))
        g_var = head._g_var_from_z_var(b.z_var, g, is_first=b.is_first)
        ev = head.regimes.expected_loglik(b.stoch, g, z_var=b.z_var, g_var=g_var)
        bestfit = ev.max(dim=-1).values                       # (B,T)
        scale = bestfit.std() + 1e-6
        gate = torch.sigmoid((bestfit.median() - bestfit) / scale)   # (B,T) in (0,1)
        gates.append(gate)
        T = gate.shape[1]
        W = max(int(min_window), T // 8)
        if T < W:
            continue
        cs = torch.cumsum(torch.cat([gate.new_zeros(gate.shape[0], 1), gate], 1), 1)
        wsum = cs[:, W:] - cs[:, :-W]                         # (B,T-W+1)
        if b.is_first is not None:
            # a birth window must lie WITHIN one episode: mask any window containing an
            # internal episode start (is_first strictly inside (t0, t0+W))
            isf = b.is_first.to(gate.dtype)
            csf = torch.cumsum(torch.cat([isf.new_zeros(isf.shape[0], 1), isf], 1), 1)
            interior = csf[:, W:] - csf[:, 1:T - W + 2]
            wsum = torch.where(interior > 0, torch.full_like(wsum, -1e30), wsum)
        val, flat = wsum.reshape(-1).max(0)
        row, t0 = divmod(int(flat), wsum.shape[1])
        if best is None or float(val) > best[0]:
            best = (float(val), bi, row, int(t0), W)
    if best is None or best[0] < 1.0:                         # no real window evidence
        return False, 0.0
    _, bi_star, row_star, t0, W = best

    # accumulate K+2 seed statistics: outside the window everything keeps gamma;
    # inside it, the gated mass moves to the two new states (halves of the window)
    agg = None
    Cc = torch.zeros(K + 2, K + 2, dtype=torch.float64, device=head.z0.device)
    sc = torch.zeros(K + 2, dtype=torch.float64, device=head.z0.device)
    for bi, b in enumerate(buf.batches):
        gamma, _, _, _ = head.regime_inference(b.stoch, b.deter, b.is_first,
                                               cache_estep=False, z_var=b.z_var,
                                               action=getattr(b, "action", None))
        newcols = gamma.new_zeros(gamma.shape[:2] + (2,))
        if bi == bi_star:
            gate = gates[bi]
            cut = _best_cut(b.stoch[row_star, t0:t0 + W].double(),
                            gate[row_star, t0:t0 + W].double())
            newcols[row_star, t0:t0 + cut, 0] = gate[row_star, t0:t0 + cut]
            newcols[row_star, t0 + cut:t0 + W, 1] = gate[row_star, t0 + cut:t0 + W]
        scale_old = (1.0 - newcols.sum(-1, keepdim=True)).clamp_min(1e-6)
        new_resp = torch.cat([gamma * scale_old, newcols], dim=-1)
        new_resp = new_resp / new_resp.sum(-1, keepdim=True).clamp_min(1e-12)
        g = head.build_g(head._prev_stoch(b.stoch, b.is_first), b.deter,
                         head._shift_action(getattr(b, "action", None), b.is_first))
        st = head.regimes.stats_from_batch(new_resp, b.stoch, g, z_var=b.z_var,
                                           g_z_var=head._shift_var(b.z_var, b.is_first))
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
    regimes, hdp, Cc, sc = _refine(head, regimes, hdp, buf, iters=refine_iters,
                                   rstick=cand_rs)
    cand = aggregate_bound(head, buf, regimes=regimes, hdp=hdp, rstick=cand_rs)
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
    """Seed a fresh regime from the WORST-EXPLAINED timesteps (largest predictive
    residual under the current regimes), refine it with restricted local VB at K+1,
    accept iff the exact aggregated bound improves. The residual-gate SEED is a
    data-driven proposal in Hughes' sense ('proposals can flexibly take any form');
    acceptance is always the exact bound, under the recurrent model when active.
    """
    buf = _as_buffer(buffer, stoch, deter, is_first, head=head, action=action)
    K = head.K
    if K >= K_max:
        return False, 0.0
    live_rs = _head_rstick(head)
    base_rs = _clone_rstick(live_rs, K)
    base, _, _, _, _ = _fit_score(head, head.regimes, head.hdp, buf,
                                  iters=refine_iters, rstick=base_rs)

    # seed K+1 responsibilities from the residual gate, accumulate emission/transition stats
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
        best = ev.max(dim=-1).values                             # (B,T) best-regime fit
        ev_s = ev - ev.max(dim=-1, keepdim=True).values
        log_init = head.hdp.expected_log_init().to(ev_s.dtype)
        log_trans = head.hdp.expected_log_trans().to(ev_s.dtype)
        gamma, _, _ = forward_backward(log_init, log_trans, ev_s, is_first=b.is_first)
        # residual gate: ~1 where even the best regime fits poorly (below typical)
        scale = best.std() + 1e-6
        gate = torch.sigmoid((best.median() - best) / scale).unsqueeze(-1)   # (B,T,1)
        new_resp = torch.cat([gamma * (1.0 - gate), gate], dim=-1)
        new_resp = new_resp / new_resp.sum(-1, keepdim=True).clamp_min(1e-12)
        total_new_mass += float(new_resp[..., -1].sum())
        st = head.regimes.stats_from_batch(new_resp, b.stoch, g, z_var=b.z_var,
                                           g_z_var=head._shift_var(b.z_var, b.is_first))
        agg = st if agg is None else {k: agg[k] + st[k] for k in agg}
        # Seed-only pair counts (outer products of shifted responsibilities). With
        # recurrence these are only an initialisation; _refine re-derives the proper
        # base-branch counts from the bounded-potential E-step before scoring.
        r_prev = head._shift_resp(new_resp, b.is_first)
        if b.is_first is None:
            Cc += torch.einsum("btj,btk->jk", r_prev[:, 1:], new_resp[:, 1:]).double()
        else:
            # mask episode resets: no transition is created across an is_first boundary
            mask = (1.0 - b.is_first[:, 1:].to(new_resp.dtype))
            Cc += torch.einsum("btj,btk,bt->jk",
                               r_prev[:, 1:], new_resp[:, 1:], mask).double()
        sc += new_resp[:, 0].sum(0).double() if b.is_first is None else \
            (new_resp * b.is_first.reshape(*b.is_first.shape[:2], 1).to(new_resp.dtype)
             ).sum(dim=(0, 1)).double()

    if total_new_mass < min_residual_mass:        # no real residual evidence -> skip
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
    """Sub-cluster split: re-differentiate a BROAD, high-occupancy regime by partitioning its
    responsibility-weighted one-step residual along the residual's top principal component into
    two children, accepting the K+1 model iff the exact bound improves. Birth seeds from
    globally worst-explained timesteps and cannot fire when one broad regime explains its data
    'well enough' (no residual outliers) -- the post-collapse / low-rank-Q case. Split recovers
    structure from INSIDE such a regime (Chang-Fisher / Dinari-Freifeld two-way seed).
    """
    buf = _as_buffer(buffer, stoch, deter, is_first, head=head, action=action)
    K = head.K
    if K >= K_max:
        return False, 0.0
    live_rs = _head_rstick(head)
    base_rs = _clone_rstick(live_rs, K)
    base, _, _, _, _ = _fit_score(head, head.regimes, head.hdp, buf,
                                  iters=refine_iters, rstick=base_rs)
    L = head.regimes.L

    # occupancy under the installed globals; attempt to split the broadest regimes first
    soft = torch.zeros(K, dtype=torch.float64, device=head.z0.device)
    for b in buf.batches:
        gamma, _, _, _ = head.regime_inference(b.stoch, b.deter, b.is_first,
                                               cache_estep=False, z_var=b.z_var,
                                               action=getattr(b, "action", None))
        soft += gamma.sum(dim=(0, 1)).double()
    for kpar in torch.argsort(soft, descending=True).tolist()[:confirm_top]:
        if float(soft[kpar]) < 2 * min_child_mass:
            continue
        # ---- pass 1: parent-weighted residual covariance -> top principal component ----
        n = torch.zeros((), dtype=torch.float64, device=head.z0.device)
        msum = torch.zeros(L, dtype=torch.float64, device=head.z0.device)
        ssum = torch.zeros(L, L, dtype=torch.float64, device=head.z0.device)
        for b in buf.batches:
            gamma, _, _, _ = head.regime_inference(b.stoch, b.deter, b.is_first,
                                                   cache_estep=False, z_var=b.z_var,
                                               action=getattr(b, "action", None))
            g = head.build_g(head._prev_stoch(b.stoch, b.is_first), b.deter,
                         head._shift_action(getattr(b, "action", None), b.is_first))
            comp_mean, _ = head.regimes.predictive(g)               # (B,T,K,L)
            resid = (b.stoch - comp_mean[..., kpar, :]).double().reshape(-1, L)
            w = gamma[..., kpar].double().reshape(-1)               # parent responsibility
            n += w.sum(); msum += (w[:, None] * resid).sum(0)
            ssum += torch.einsum("n,ni,nj->ij", w, resid, resid)
        if float(n) < 2 * min_child_mass:
            continue
        mean = msum / n.clamp_min(1e-6)
        cov = ssum / n.clamp_min(1e-6) - mean[:, None] * mean[None, :]
        evecs = torch.linalg.eigh(
            cov + 1e-6 * torch.eye(L, dtype=cov.dtype, device=cov.device))[1]
        v = evecs[:, -1]                                            # top PC of the residual

        # ---- pass 2: soft 2-way split of the parent column, accumulate K+1 stats ----
        agg = None
        Cc = torch.zeros(K + 1, K + 1, dtype=torch.float64, device=head.z0.device)
        sc = torch.zeros(K + 1, dtype=torch.float64, device=head.z0.device)
        child_mass = 0.0
        for b in buf.batches:
            gamma, _, _, _ = head.regime_inference(b.stoch, b.deter, b.is_first,
                                                   cache_estep=False, z_var=b.z_var,
                                               action=getattr(b, "action", None))
            g = head.build_g(head._prev_stoch(b.stoch, b.is_first), b.deter,
                         head._shift_action(getattr(b, "action", None), b.is_first))
            comp_mean, _ = head.regimes.predictive(g)
            resid = b.stoch - comp_mean[..., kpar, :]              # (B,T,L)
            proj = torch.einsum("btl,l->bt", resid, v.to(resid.dtype))
            sA = torch.sigmoid(proj / (proj.std() + 1e-6))         # parent keeps the A-share
            wpar = gamma[..., kpar]
            new = gamma.clone(); new[..., kpar] = wpar * sA
            new = torch.cat([new, (wpar * (1.0 - sA)).unsqueeze(-1)], dim=-1)  # (B,T,K+1)
            new = new / new.sum(-1, keepdim=True).clamp_min(1e-12)
            child_mass += float(new[..., -1].sum())
            st = head.regimes.stats_from_batch(new, b.stoch, g, z_var=b.z_var,
                                               g_z_var=head._shift_var(b.z_var, b.is_first))
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
                merge_passes: int = 3, birth_style: str = "interval",
                delete_topk: int = 3, merge_topm: int | None = None, action=None):
    """One birth / split / merge / delete pass (Hughes lap structure: grow during the visit, clean up after). Returns a log {move: (accepted, elbo_gain)},
    where elbo_gain is the change in the PLAIN structured ELBO (so the log is transparent even
    when a move is accepted under a non-neutral size prior).

    `create_bonus` (== `size_log_prior_odds`) is the per-regime log-odds of an explicit
    exponential/geometric prior on the number of active regimes. Every move is accepted iff it
    increases the SIZE-PENALIZED ELBO  J = L + create_bonus * K_active  (see `_accept`). This is
    NOT a heuristic threshold offset: for a fixed create_bonus it is exact monotone ascent on
    a single well-defined objective, the ELBO under a prior on model size (a MAP structure
    decision). Annealing create_bonus -> 0 over training (via the curriculum) is a
    continuation / graduated-optimization schedule: monotone within each phase and converging
    to the plain-ELBO objective. With create_bonus == 0 every accepted move strictly
    increases L -- now including the recurrent-stickiness factors when recurrence is active,
    since base and candidates are scored under the same PG/JJ-bounded recurrent ELBO.
    """
    # (honest labeling): birth/split redistribute the data
    # currently in the move BUFFER exactly, but when the persistent globals are a
    # constant-memory STREAMING aggregate, historical mass OUTSIDE the buffer cannot
    # be re-split -- so on a streaming store birth/split are APPROXIMATE model
    # selection. Exact whole-history selection requires the offline_memoized frozen
    # corpus. (A fixed-Kmax + active-mask rewrite that makes moves shape-stable and
    # labels the historical transport is the fuller fix and is NOT implemented.)
    _st = getattr(head, "stat_store", None)
    if (do_birth or do_split) and _st is not None and getattr(_st, "mode", None) == "streaming" \
            and not getattr(head, "_warned_stream_birth", False):
        import warnings
        warnings.warn("birth/split on a streaming aggregate are APPROXIMATE (historical "
                      "mass outside the move buffer is not re-split); use offline_memoized "
                      "consolidation for exact model selection.")
        head._warned_stream_birth = True
    buf = _as_buffer(buffer, stoch, deter, is_first, head=head, action=action)
    s = create_bonus if size_log_prior_odds is None else size_log_prior_odds
    cheap = max(1, refine_iters // 2)
    log = {}
    if do_birth:
        if birth_style == "interval":
            log["birth"] = interval_birth_move(head, buffer=buf, threshold=threshold,
                                               refine_iters=max(3, refine_iters),
                                               size_log_prior_odds=s)
        else:
            log["birth"] = birth_move(head, buffer=buf, threshold=threshold,
                                      refine_iters=max(3, refine_iters),
                                      size_log_prior_odds=s)
    if do_split:
        log["split"] = split_move(head, buffer=buf, threshold=threshold,
                                  refine_iters=max(3, refine_iters), confirm_top=confirm_top,
                                  size_log_prior_odds=s)
    # cleanup phase (Hughes lap structure: merges and deletes after the growth moves)
    log["merge"] = merge_move(head, buffer=buf, threshold=threshold, refine_iters=cheap,
                              confirm_top=confirm_top, size_log_prior_odds=s,
                              merge_select=merge_select, max_passes=merge_passes, merge_topm=merge_topm)
    log["delete"] = delete_move(head, buffer=buf, threshold=threshold,
                                size_log_prior_odds=s, refine_iters=cheap,
                                mode=delete_mode, delete_topk=delete_topk)
    accepted_any = any(isinstance(v, tuple) and bool(v[0]) for v in log.values())
    if accepted_any and buf is not None and getattr(buf, "complete", False):
        # Hughes post-move consistency: rebuild the memoized ledger under the INSTALLED
        # candidate over the certified complete buffer, so subsequent global updates
        # start from the accepted model's own whole-corpus statistics instead of the
        # row-remapped pre-move summaries. The row remap in
        # _apply remains the mid-sweep safety net between multiple accepts. The rebuild
        # finishes with one global step from the COMPLETE totals.
        head.resync_store_from_buffer(buf)
    return log

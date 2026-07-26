"""State-dependent recurrent stickiness for the SHS-RSSM.

This implements the recurrent sticky HDP-HMM transition in a disentangled form. At
an interior transition into time t, the previous regime i has its own persistence
probability

    rho_{t,i} = sigmoid(w_i^T phi_t + b_i),

and the effective transition is

    M_t[i,j] = rho_{t,i} 1[j=i] + (1-rho_{t,i}) Pi[i,j].

The base is the NON-sticky HDP-HMM transition posterior. The Bernoulli persistence
weights {w_i,b_i}_{i=1}^K are not optimized by Adam; they have Gaussian variational
posteriors updated by K independent Polya-Gamma logistic-regression M-steps from
forward-backward pairwise marginals.

Fully variational E-step (no heuristics): the forward-backward potentials come from
`bound_log_trans`, the Polya-Gamma / Jaakkola-Jordan lower bound on the expected log
augmented transition, evaluated at the coordinate-ascent-optimal PG variational
parameter c_{t,i}^2 = E_q[psi_{t,i}^2]. The base branch uses the raw sub-stochastic
E[log pibar] potential (never softmax-renormalised), so the forward log-partition is
a genuine lower bound on the marginal evidence and the whole objective remains a
single ELBO shared by the E-step, the losses, and structure-move acceptance. The
probit `sigma()` / `effective_log_trans()` path is retained strictly for generative
(imagination) rollouts, where a posterior-predictive moment approximation is
appropriate and no bound is claimed.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class RecurrentStickiness(nn.Module):
    def __init__(self, K: int, feat_dim: int, prior_persist: float = 0.9,
                 weight_prior_var: float = 1.0, bias_prior_var: float = 4.0,
                 pg_iters: int = 4, uncertainty_correction: bool = True,
                 device=None, dtype=torch.float64):
        """
        K: number of active regimes. Each previous-state row has its own logistic
           persistence model.
        feat_dim: D, feature dimension before appending the bias; beta has D+1 dims.
        prior_persist: prior mean self-persistence; sets the prior bias logit.
        """
        super().__init__()
        self.K = int(K)
        self.D = int(feat_dim)
        self.pg_iters = int(pg_iters)
        self.uncorr = bool(uncertainty_correction)
        self.prior_persist = float(prior_persist)
        self.weight_prior_var = float(weight_prior_var)
        self.bias_prior_var = float(bias_prior_var)
        self._dtype = dtype

        Dp = self.D + 1
        bias0 = math.log(prior_persist / (1.0 - prior_persist))
        m0 = torch.zeros(Dp, dtype=dtype, device=device)
        m0[-1] = bias0
        s0 = torch.full((Dp,), float(weight_prior_var), dtype=dtype, device=device)
        s0[-1] = float(bias_prior_var)
        self.register_buffer("m0", m0)
        self.register_buffer("sigma0_diag", s0)

        # q(beta_i) = N(m_beta[i], Sigma_beta[i]) for each origin state i.
        self.register_buffer("m_beta", m0.view(1, Dp).repeat(self.K, 1).clone())
        self.register_buffer("Sigma_beta", torch.diag(s0).view(1, Dp, Dp).repeat(self.K, 1, 1).clone())
        # EMA of data-side PG natural statistics, one set per origin state.
        self.register_buffer("pg_A", torch.zeros(self.K, Dp, Dp, dtype=dtype, device=device))
        self.register_buffer("pg_h", torch.zeros(self.K, Dp, dtype=dtype, device=device))
        self._pg_init = False

    # These buffers are deliberately high precision for the logistic/Cholesky math and
    # must NOT be downcast if a caller does model.float()/.half()/.to(dtype) (e.g. for eval
    # or export). Device moves still propagate; only the numerical dtype is pinned so the
    # buffers and self._dtype cannot desync (which otherwise crashes pg_update_statewise).
    def _apply(self, fn, *args, **kwargs):
        module = super()._apply(fn, *args, **kwargs)
        for name in ("m0", "sigma0_diag", "m_beta", "Sigma_beta", "pg_A", "pg_h"):
            buf = getattr(module, name, None)
            if buf is not None and buf.dtype != module._dtype:
                setattr(module, name, buf.to(module._dtype))
        return module

    # ----------------------------------------------------------------- persistence
    def _psi_moments(self, phi):
        """E[psi_{t,i}] and Var[psi_{t,i}] for phi (...,D+1).

        Returns mu,v with shape (...,K).
        """
        mb = self.m_beta.to(phi.dtype)          # (K,D+1)
        Sb = self.Sigma_beta.to(phi.dtype)      # (K,D+1,D+1)
        mu = torch.einsum("...d,kd->...k", phi, mb)
        v = torch.einsum("...d,kde,...e->...k", phi, Sb, phi)
        return mu, v.clamp_min(0.0)

    def sigma(self, phi):
        """PREDICTIVE-ONLY state-specific persistence E_q[sigmoid(psi_{t,i})]; returns (...,K).

        This is the MacKay probit moment approximation of the posterior-predictive
        persistence probability. It is used only for generative rollouts
        (imagination). It is NOT part of any variational bound: the E-step and the
        ELBO use `bound_log_trans`, which is a proper Polya-Gamma / Jaakkola-Jordan
        lower bound rather than a moment approximation.
        """
        mu, v = self._psi_moments(phi)
        if self.uncorr:
            z = mu / torch.sqrt(1.0 + (math.pi / 8.0) * v)
        else:
            z = mu
        return torch.sigmoid(z)

    # -------------------------------------------- variational (PG/JJ) E-step potentials
    def jj_branch_potentials(self, phi):
        """Per-(link, origin-state) branch log-potentials of the augmented Bernoulli.

        For every origin state i at a link with feature phi, the augmented model draws
        w ~ Bern(sigmoid(psi_i)), psi_i = beta_i^T phi. Under q(beta_i) = N(m_i, S_i)
        and the Polya-Gamma augmentation with q(omega_{t,i}) = PG(1, c_{t,i}) at the
        coordinate-ascent optimum c^2 = E_q[psi^2] = m^2 + v (equivalently the
        Jaakkola-Jordan bound at its optimal tilt), the expected log Bernoulli factors
        are bounded in closed form:

            E_q[log p(w=1 | psi)] >= +m/2 + log sigmoid(c) - c/2  =: A     (persist)
            E_q[log p(w=0 | psi)] >= -m/2 + log sigmoid(c) - c/2  =: B0    (switch)

        (the quadratic term -lambda(c)(m^2+v-c^2) vanishes at the optimal c). These are
        true lower bounds, so a forward-backward log-partition built from them is a
        valid ELBO term. Returns (A, B0, m, c), each (..., K).
        """
        m, v = self._psi_moments(phi)
        c = torch.sqrt((m * m + v).clamp_min(1e-12))
        const = torch.nn.functional.logsigmoid(c) - 0.5 * c
        A = const + 0.5 * m
        B0 = const - 0.5 * m
        return A, B0, m, c

    def bound_log_trans(self, base_elogpi, phi_steps):
        """Time-varying transition LOG-POTENTIALS that lower-bound the recurrent model.

        base_elogpi : (K,K) E_q[log pibar_{ij}] of the NON-sticky base HDP rows
                      (raw Dirichlet-posterior expectations; deliberately NOT
                      renormalised -- the sub-stochastic potential is what keeps the
                      forward-backward log-partition a lower bound on the evidence and
                      preserves the HDP's K-penalising normalisation mass).
        phi_steps   : (..., D+1) features for the links (typically (B, T-1, D+1)).

        The augmented transition marginalises the Bernoulli branch exactly inside the
        potential:

            pot[i, j] = logaddexp(A_i + log 1[j==i],  B0_i + E[log pibar_{ij}])

        i.e. off-diagonal entries carry only the switch branch, the diagonal carries
        both. Forward-backward over these potentials is exact structured VB over the
        joint chain (s_t, w_t); pairwise marginals xi decompose into the two branches
        via `attribute_bound`. Everything is differentiable in phi (for the stickiness
        projection gradient). Returns (log_trans (...,K,K), aux dict).
        """
        K = base_elogpi.shape[0]
        elog = base_elogpi.to(phi_steps.dtype)
        A, B0, m, c = self.jj_branch_potentials(phi_steps)     # (...,K)
        switch = B0[..., :, None] + elog                       # (...,K,K)
        diag_persist = A                                        # (...,K)
        eye = torch.eye(K, dtype=switch.dtype, device=switch.device)
        big_neg = torch.finfo(switch.dtype).min / 4.0
        persist_full = diag_persist[..., :, None] + (1.0 - eye) * big_neg
        log_trans = torch.logaddexp(persist_full, switch)
        aux = dict(A=A, B0=B0, m=m, c=c,
                   switch_diag=torch.diagonal(switch, dim1=-2, dim2=-1),
                   base_elogpi=elog)
        return log_trans, aux

    def bound_aux_only(self, base_elogpi, phi_steps):
        """ROUND-24 review P2 #11: the transition AUX (A, B0, m, c, switch_diag,
        base_elogpi) WITHOUT materialising the O(BTK^2) log_trans. Every returned tensor is
        O(BTK) or (K,K); each (B,K,K) transition slice is rebuilt ON DEMAND from this aux via
        trans_slice_from_aux, so recurrent forward-backward never holds the full tensor."""
        elog = base_elogpi.to(phi_steps.dtype)
        A, B0, m, c = self.jj_branch_potentials(phi_steps)              # each (...,K), O(BTK)
        switch_diag = B0 + torch.diagonal(elog)                        # (...,K)
        return dict(A=A, B0=B0, m=m, c=c, switch_diag=switch_diag, base_elogpi=elog)

    @staticmethod
    def trans_slice_from_aux(aux, t):
        """Rebuild the single (B,K,K) transition log-potential for link t from the aux
        (differentiable in A/B0, hence in phi). Identical to bound_log_trans(...)[:, t-1]."""
        A = aux["A"][:, t - 1]                                         # (B,K)
        B0 = aux["B0"][:, t - 1]                                       # (B,K)
        elog = aux["base_elogpi"]                                      # (K,K)
        K = A.shape[-1]
        switch = B0[..., :, None] + elog                              # (B,K,K)
        eye = torch.eye(K, dtype=A.dtype, device=A.device)
        big_neg = torch.finfo(A.dtype).min / 4.0
        persist = A[..., :, None] + (1.0 - eye) * big_neg             # (B,K,K)
        return torch.logaddexp(persist, switch)

    @staticmethod
    def attribute_bound(xi, aux):
        """Exact branch decomposition of the pairwise marginals under the PG/JJ potentials.

        xi  : (B,S,K,K) pairwise q(s_{t-1}=i, s_t=j) from forward-backward run on
              `bound_log_trans` potentials.
        aux : dict from `bound_log_trans`.

        Returns:
          r_mass     : (B,S,K) expected persistence (w=1) mass per origin state,
                       q(s_{t-1}=i, s_t=i, w=1).
          row_weight : (B,S,K) q(s_{t-1}=i): the Bernoulli trial weight (every interior
                       link draws a w for its origin state).
          Cbase      : (K,K) expected switch-branch (w=0) transition counts, the
                       sufficient statistic of the base non-sticky HDP rows.
        """
        A = aux["A"]                                            # (B,S,K)
        Bd = aux["switch_diag"]                                 # (B,S,K)
        w1_frac = torch.exp(A - torch.logaddexp(A, Bd))         # q(w=1 | i -> i)
        diag_xi = torch.diagonal(xi, dim1=-2, dim2=-1)          # (B,S,K)
        r_mass = diag_xi * w1_frac
        row_weight = xi.sum(dim=-1)                             # (B,S,K)
        Cbase = xi.clone()
        newdiag = diag_xi * (1.0 - w1_frac)
        Cbase = Cbase - torch.diag_embed(diag_xi) + torch.diag_embed(newdiag)
        return r_mass, row_weight, Cbase.sum(dim=(0, 1))

    def effective_log_trans(self, base_elogpi, phi_steps):
        """PREDICTIVE-ONLY effective transition (probit-approximate posterior mean).

        Kept for generative rollouts and diagnostics. Training, the ELBO, and move
        scoring use `bound_log_trans`. Returns log M (B,S,K,K), rho (B,S,K), Pi (K,K).
        """
        K = base_elogpi.shape[0]
        Pi = torch.softmax(base_elogpi, dim=-1)                 # (K,K)
        sig = self.sigma(phi_steps)                            # (B,S,K)
        eye = torch.eye(K, dtype=Pi.dtype, device=Pi.device)
        s = sig[..., :, None]                                  # (B,S,K,1), row-specific
        M = s * eye + (1.0 - s) * Pi                           # (B,S,K,K)
        return M.clamp_min(1e-30).log(), sig, Pi

    # ----------------------------------------- responsibilities and disentangled counts
    @staticmethod
    def attribute(xi, sig, Pi):
        """Split pairwise responsibilities into persistence and base-transition mass.

        xi  : (B,S,K,K), q(s_{t-1}=i, s_t=j) from forward-backward.
        sig : (B,S,K), rho_{t,i}.
        Pi  : (K,K), base transition.

        Returns:
          r_mass     : (B,S,K), expected persistence mass attributed to row i.
          row_weight : (B,S,K), q(s_{t-1}=i), the Bernoulli trial weight for row i.
          Cbase      : (K,K), transition counts attributable to the base HDP transition.
        """
        K = Pi.shape[0]
        eye = torch.eye(K, dtype=Pi.dtype, device=Pi.device)
        s = sig[..., :, None]
        M = (s * eye + (1.0 - s) * Pi).clamp_min(1e-30)        # (B,S,K,K)

        diag_xi = torch.diagonal(xi, dim1=-2, dim2=-1)         # (B,S,K)
        diag_M = torch.diagonal(M, dim1=-2, dim2=-1)           # (B,S,K)
        r_mass = diag_xi * (sig / diag_M)                     # (B,S,K)
        row_weight = xi.sum(dim=-1)                            # (B,S,K)

        base_frac = ((1.0 - s) * Pi) / M                       # (B,S,K,K)
        Cbase = (xi * base_frac).sum(dim=(0, 1))               # (K,K)
        return r_mass, row_weight, Cbase

    # ------------------------------------------------------------ Polya-Gamma update
    @torch.no_grad()
    def pg_update_statewise(self, phi, r_mass, row_weight, lr=None):
        """K independent variational PG logistic-regression updates.

        phi        : (N,D+1) features.
        r_mass     : (N,K), expected positive/persistence mass for each origin state.
        row_weight : (N,K), q(s_{t-1}=i), Bernoulli trial weight for each origin state.
        lr         : None for full-batch update; float for online EMA of natural stats.

        For each regime i, this is the fractional-binomial logistic update with
        y_i mass r_mass[:,i] out of row_weight[:,i] trials:

            h_i = sum_n (r_{n,i} - 0.5 a_{n,i}) phi_n
            A_i = sum_n a_{n,i} E[omega_{n,i}] phi_n phi_n^T.
        """
        # Match the LIVE buffer dtype (m_beta et al. are pinned high-precision by _apply);
        # keying off the buffer rather than self._dtype means the einsums below cannot
        # dtype-desync even if a caller ever bypasses the _apply pin.
        wd = self.m_beta.dtype
        phi = phi.to(wd)
        r_mass = r_mass.to(wd)
        row_weight = row_weight.to(wd).clamp_min(0.0)

        K, Dp = self.K, phi.shape[-1]
        Sig0_inv = torch.diag(1.0 / self.sigma0_diag)
        rhs_prior = Sig0_inv @ self.m0

        def solve_spd(prec, rhs):
            # Cholesky with escalating-jitter retry: the PG precision is SPD in exact
            # arithmetic but can lose definiteness numerically under extreme E[omega].
            base = prec.diagonal().abs().mean().clamp_min(1e-12)
            Lc = None
            for j in range(4):
                try:
                    pm = prec if j == 0 else prec + (10.0 ** (j - 8)) * base * torch.eye(
                        prec.shape[-1], dtype=prec.dtype, device=prec.device)
                    Lc = torch.linalg.cholesky(pm)
                    break
                except Exception:
                    if j == 3:
                        raise
            Sigma = torch.cholesky_inverse(Lc)
            return Sigma, Sigma @ rhs

        new_m = self.m_beta.clone()
        new_S = self.Sigma_beta.clone()
        new_A = self.pg_A.clone()
        new_h = self.pg_h.clone()

        for k in range(K):
            w = row_weight[:, k]
            r = r_mass[:, k]
            m_cur = self.m_beta[k].clone()
            S_cur = self.Sigma_beta[k].clone()

            # If a row has essentially no responsibility in this batch, keep its online stats.
            if float(w.sum()) <= 1e-12 and lr is None:
                continue

            try:
                for _ in range(self.pg_iters):
                    EbbT = S_cur + torch.outer(m_cur, m_cur)
                    c = torch.einsum("nd,de,ne->n", phi, EbbT, phi).clamp_min(1e-12).sqrt()
                    Eom = (0.5 / c) * torch.tanh(0.5 * c) * w
                    A = torch.einsum("n,nd,ne->de", Eom, phi, phi)
                    h = torch.einsum("n,nd->d", r - 0.5 * w, phi)

                    if lr is None:
                        A_eff, h_eff = A, h
                    else:
                        if not self._pg_init:
                            A_eff, h_eff = A, h
                        else:
                            A_eff = self.pg_A[k] * (1.0 - lr) + lr * A
                            h_eff = self.pg_h[k] * (1.0 - lr) + lr * h

                    S_cur, m_cur = solve_spd(Sig0_inv + A_eff, rhs_prior + h_eff)
                cand = (S_cur, m_cur,
                        A_eff if lr is not None else A,
                        h_eff if lr is not None else h)
                if not all(bool(torch.isfinite(t).all()) for t in cand):
                    raise FloatingPointError("nonfinite PG candidate")
            except Exception:
                # TRANSACTIONAL install: this row keeps its previous valid posterior
                # and natural statistics; the failure is counted, not propagated.
                self.n_pg_guard_rejects = getattr(self, "n_pg_guard_rejects", 0) + 1
                continue
            new_m[k] = m_cur
            new_S[k] = S_cur
            if lr is None:
                new_A[k] = A
                new_h[k] = h
            else:
                new_A[k] = A_eff
                new_h[k] = h_eff

        self.m_beta.copy_(new_m)
        self.Sigma_beta.copy_(new_S)
        self.pg_A.copy_(new_A)
        self.pg_h.copy_(new_h)
        self._pg_init = True

    @torch.no_grad()
    def pg_update(self, phi, r, weight=None, lr=None):
        """Compatibility wrapper for older scalar-persistence callers.

        It broadcasts a scalar soft label to all rows. New code should call
        pg_update_statewise() with r_mass and row_weight from attribute().
        """
        if weight is None:
            weight = torch.ones_like(r)
        r_mass = r[:, None].expand(-1, self.K) / max(self.K, 1)
        row_weight = weight[:, None].expand(-1, self.K) / max(self.K, 1)
        return self.pg_update_statewise(phi, r_mass, row_weight, lr=lr)

    # ---------------------------------------------- memoized / streaming primitives
    @torch.no_grad()
    def pg_stats_from_batch(self, phi, r_mass, row_weight):
        """This batch's Polya-Gamma NATURAL statistics (A_k, h_k), evaluated at the
        CURRENT posterior's E[omega] (Hughes-style batch summary: stale until the
        batch is revisited). Totals over batches are installed with pg_set_totals."""
        wd = self.m_beta.dtype
        phi = phi.to(wd)
        r = r_mass.to(wd)
        w = row_weight.to(wd).clamp_min(0.0)
        EbbT = self.Sigma_beta + torch.einsum("kd,ke->kde", self.m_beta, self.m_beta)
        c = torch.einsum("nd,kde,ne->nk", phi, EbbT, phi).clamp_min(1e-12).sqrt()
        Eom = (0.5 / c) * torch.tanh(0.5 * c) * w                       # (N,K)
        A = torch.einsum("nk,nd,ne->kde", Eom, phi, phi)                # (K,D+1,D+1)
        h = torch.einsum("nk,nd->kd", r - 0.5 * w, phi)                 # (K,D+1)
        return dict(A=A, h=h)

    @torch.no_grad()
    def pg_set_totals(self, A, h):
        """Install summed PG natural statistics and solve every row posterior exactly:
        Sigma_k = (Sigma_0^{-1} + A_k)^{-1},  m_k = Sigma_k (Sigma_0^{-1} m_0 + h_k).
        Used by the streaming/memoized store; the c-parameter refresh happens the
        next time each batch's summary is recomputed (memoized semantics)."""
        if not (bool(torch.isfinite(A).all()) and bool(torch.isfinite(h).all())):
            # TRANSACTIONAL: refuse nonfinite summed naturals outright -- keep the
            # previous valid row posteriors and count (round-4 review, rec. 8)
            self.n_pg_guard_rejects = getattr(self, 'n_pg_guard_rejects', 0) + 1
            return
        wd = self.m_beta.dtype
        A = A.to(wd)
        h = h.to(wd)
        Sig0_inv = torch.diag(1.0 / self.sigma0_diag)
        rhs_prior = Sig0_inv @ self.m0
        # ATOMIC (round-5 review, issue 7): solve every row into TEMP tensors with
        # escalating Cholesky jitter; install nothing unless all K rows succeed and are
        # finite, so a failure on a late row cannot leave earlier rows updated.
        Dp = self.m_beta.shape[-1]
        eye = torch.eye(Dp, dtype=wd, device=A.device)
        scale = float(torch.diagonal(Sig0_inv).abs().mean()) + 1.0
        newSigma = torch.empty_like(self.Sigma_beta)
        newm = torch.empty_like(self.m_beta)
        for k in range(self.K):
            prec = Sig0_inv + A[k]
            Sigma = None
            for tries in range(5):
                jit = 0.0 if tries == 0 else (10.0 ** (tries - 6)) * scale
                try:
                    Lc = torch.linalg.cholesky(prec + jit * eye)
                    cand = torch.cholesky_inverse(Lc)
                    if bool(torch.isfinite(cand).all()):
                        Sigma = cand
                        break
                except Exception:
                    continue
            if Sigma is None:
                self.n_pg_guard_rejects = getattr(self, "n_pg_guard_rejects", 0) + 1
                return                                    # keep previous posteriors entirely
            newSigma[k] = Sigma
            newm[k] = Sigma @ (rhs_prior + h[k])
        if not (bool(torch.isfinite(newSigma).all()) and bool(torch.isfinite(newm).all())):
            self.n_pg_guard_rejects = getattr(self, "n_pg_guard_rejects", 0) + 1
            return
        self.Sigma_beta.copy_(newSigma)
        self.m_beta.copy_(newm)
        self.pg_A.copy_(A)
        self.pg_h.copy_(h)
        self._pg_init = True

    # ------------------------------------------------------------------- ELBO piece
    @torch.no_grad()
    def beta_kl(self):
        """Sum_i KL(q(beta_i) || p(beta_i)) for state-specific persistence weights."""
        Dp = self.m_beta.shape[-1]
        inv0 = 1.0 / self.sigma0_diag
        dm = self.m_beta - self.m0.view(1, Dp)
        tr = (torch.diagonal(self.Sigma_beta, dim1=-2, dim2=-1) * inv0.view(1, Dp)).sum(-1)
        maha = (dm.pow(2) * inv0.view(1, Dp)).sum(-1)
        logdet0 = torch.log(self.sigma0_diag).sum()
        logdetq = torch.linalg.slogdet(self.Sigma_beta).logabsdet
        return 0.5 * (tr + maha - Dp + logdet0 - logdetq).sum()

    @torch.no_grad()
    def select_rows(self, keep_idx):
        """A new module whose origin-state rows are this module's rows `keep_idx`.

        Used by delete/merge candidates: dropping regime j (or merging j into i) must
        drop ROW j of the per-origin persistence posteriors, not truncate from the
        end. Rows are copied with their PG natural statistics so a candidate refit
        starts from the matching posterior.
        """
        keep = torch.as_tensor(keep_idx, device=self.m0.device, dtype=torch.long)
        new = RecurrentStickiness(
            K=int(keep.numel()), feat_dim=self.D, prior_persist=self.prior_persist,
            weight_prior_var=self.weight_prior_var, bias_prior_var=self.bias_prior_var,
            pg_iters=self.pg_iters, uncertainty_correction=self.uncorr,
            device=self.m0.device, dtype=self._dtype,
        )
        new.m_beta.copy_(self.m_beta[keep])
        new.Sigma_beta.copy_(self.Sigma_beta[keep])
        new.pg_A.copy_(self.pg_A[keep])
        new.pg_h.copy_(self.pg_h[keep])
        new._pg_init = self._pg_init
        return new

    @torch.no_grad()
    def resized_like(self, new_K: int):
        """Resize to a new K, preserving rows 0..min(K,new_K)-1 and prior-initialising new rows."""
        new = RecurrentStickiness(
            K=int(new_K), feat_dim=self.D, prior_persist=self.prior_persist,
            weight_prior_var=self.weight_prior_var, bias_prior_var=self.bias_prior_var,
            pg_iters=self.pg_iters, uncertainty_correction=self.uncorr,
            device=self.m0.device, dtype=self._dtype,
        )
        n = min(self.K, int(new_K))
        new.m_beta[:n].copy_(self.m_beta[:n])
        new.Sigma_beta[:n].copy_(self.Sigma_beta[:n])
        new.pg_A[:n].copy_(self.pg_A[:n])
        new.pg_h[:n].copy_(self.pg_h[:n])
        new._pg_init = self._pg_init
        return new

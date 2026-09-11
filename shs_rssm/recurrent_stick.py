from __future__ import annotations

import math
import torch
import torch.nn as nn
from scipy.special import digamma, polygamma

class RecurrentStickiness(nn.Module):
    def __init__(self, K: int, feat_dim: int, prior_persist: float = 0.9,
                 moment_match: bool = False, prior_conc: float = 10.0,
                 weight_prior_var: float = 1.0, bias_prior_var: float = 4.0,
                 pg_iters: int = 4, uncertainty_correction: bool = True,
                 hier_bias: bool = False, hier_a0: float = 2.0,
                 hier_b0: float | None = None, hier_used_only: bool = False,
                 hier_min_evidence: float = 1.0,
                 device=None, dtype=torch.float64):
        """Persistence gate  rho_{k,t} = sigmoid(beta_k^T phi_t),  q(beta_k) = N(m_k, Sigma_k).

        Prior on beta_k = [weights ; bias]:
            weights ~ N(0, weight_prior_var I)                        (always)
            bias    ~ N(bias0, bias_prior_var)                       (hier_bias=False, fixed)
            bias    ~ N(mu, 1/lam),  mu ~ N(bias0, bias_prior_var),
                      lam ~ Gamma(hier_a0, hier_b0)                   (hier_bias=True)

        The hierarchical form is the disentangled-sticky construction carried
        over to the recurrent gate: E[mu] is the population-level persistence
        (feature 2 of Zhou et al.), 1/E[lam] is how much persistence varies
        across regimes (feature 3).  Both are learned by conjugate mean-field
        updates from the K bias marginals of q(beta); see hier_update().  The
        fixed prior is recovered as the special case in which q(mu), q(lam) are
        frozen at their initial values.  `hier_b0=None` sets b0 = a0 *
        bias_prior_var so that the EFFECTIVE bias precision at initialisation,
        E[lam] = a0/b0, equals 1/bias_prior_var: the first coefficient update is
        identical to the fixed gate's.  The MARGINAL prior on a bias is wider
        than the fixed gate's (a0=2: E[1/lam] = 2*bias_prior_var, plus Var[mu]),
        so the two models are not the same prior, only the same first step.
        """
        super().__init__()
        self.K = int(K)
        self.D = int(feat_dim)
        self.pg_iters = int(pg_iters)
        self.uncorr = bool(uncertainty_correction)
        self.prior_persist = float(prior_persist)
        self.weight_prior_var = float(weight_prior_var)
        self.bias_prior_var = float(bias_prior_var)
        self.moment_match = bool(moment_match)
        self.prior_conc = float(prior_conc)
        self.hier_bias = bool(hier_bias)
        self.hier_a0 = float(hier_a0)
        self._hier_b0_arg = (None if hier_b0 is None else float(hier_b0))
        self.hier_used_only = bool(hier_used_only)
        self.hier_min_evidence = float(hier_min_evidence)
        self._dtype = dtype

        Dp = self.D + 1
        # Two ways to turn a persistence prior into a Gaussian prior on the
        # logit bias b_j.  Default (moment_match=False) is the plain
        # logit-of-the-mean heuristic, logit(E[kappa]); it is NOT the logit
        # moment of a Beta and is kept as the default only because changing it
        # changes every recurrent result.  With moment_match=True and a
        # concentration, the manuscript's exact conversion is used:
        #     kappa_j ~ Beta(rho1, rho2),  b_j = logit(kappa_j)
        #     E[b_j]   = psi(rho1) - psi(rho2)
        #     Var(b_j) = psi'(rho1) + psi'(rho2)
        # Either way this is logistic-normal, i.e. an approximation to the
        # logit-Beta by moment matching, not an exact re-parameterisation.
        if moment_match:
            
            c = float(prior_conc)
            r1, r2 = self.prior_persist * c, (1.0 - self.prior_persist) * c
            bias0 = float(digamma(r1) - digamma(r2))
            bias_prior_var = float(polygamma(1, r1) + polygamma(1, r2))
        else:
            bias0 = math.log(prior_persist / (1.0 - prior_persist))
        m0 = torch.zeros(Dp, dtype=dtype, device=device)
        m0[-1] = bias0
        s0 = torch.full((Dp,), float(weight_prior_var), dtype=dtype, device=device)
        s0[-1] = float(bias_prior_var)
        self.register_buffer("m0", m0)
        self.register_buffer("sigma0_diag", s0)

        self.register_buffer("m_beta", m0.view(1, Dp).repeat(self.K, 1).clone())
        self.register_buffer("Sigma_beta", torch.diag(s0).view(1, Dp, Dp).repeat(self.K, 1, 1).clone())
        self.register_buffer("pg_A", torch.zeros(self.K, Dp, Dp, dtype=dtype, device=device))
        self.register_buffer("pg_h", torch.zeros(self.K, Dp, dtype=dtype, device=device))
        self._pg_init = False

        # Hierarchical bias prior: hyper-prior (m00, v00, a0, b0) and the
        # variational hyper-posterior q(mu) = N(mu_m, mu_v), q(lam) = Gamma(a, b).
        # They exist (and are checkpointed) even when hier_bias=False so that a
        # run can be resumed with the flag flipped either way.
        # b0 defaults off the bias prior variance ACTUALLY in force (which may be
        # the moment-matched one), so E[lam] = a0/b0 = 1/bias_prior_var at init.
        self.hier_b0 = (self._hier_b0_arg if self._hier_b0_arg is not None
                        else self.hier_a0 * float(bias_prior_var))
        sc = lambda x: torch.tensor(float(x), dtype=dtype, device=device)
        self.register_buffer("hyp_m00", sc(bias0))
        self.register_buffer("hyp_v00", sc(bias_prior_var))
        self.register_buffer("hyp_a0", sc(self.hier_a0))
        self.register_buffer("hyp_b0", sc(self.hier_b0))
        self.register_buffer("hyp_mu_m", sc(bias0))
        self.register_buffer("hyp_mu_v", sc(bias_prior_var))
        self.register_buffer("hyp_lam_a", sc(self.hier_a0))
        self.register_buffer("hyp_lam_b", sc(self.hier_b0))
        self._register_load_state_dict_pre_hook(self._hier_pre_load_hook)
        if self.hier_bias:
            self._hier_sync()

    _HYPER_BUFFERS = ("hyp_m00", "hyp_v00", "hyp_a0", "hyp_b0",
                      "hyp_mu_m", "hyp_mu_v", "hyp_lam_a", "hyp_lam_b")

    def _apply(self, fn, *args, **kwargs):
        module = super()._apply(fn, *args, **kwargs)
        for name in ("m0", "sigma0_diag", "m_beta", "Sigma_beta", "pg_A", "pg_h") + self._HYPER_BUFFERS:
            buf = getattr(module, name, None)
            if buf is not None and buf.dtype != module._dtype:
                setattr(module, name, buf.to(module._dtype))
        return module

    # ------------------------------------------------------------------ hier
    def _ctor_kwargs(self):
        return dict(prior_persist=self.prior_persist, moment_match=self.moment_match,
                    prior_conc=self.prior_conc,
                    weight_prior_var=self.weight_prior_var,
                    bias_prior_var=self.bias_prior_var, pg_iters=self.pg_iters,
                    uncertainty_correction=self.uncorr, hier_bias=self.hier_bias,
                    hier_a0=self.hier_a0, hier_b0=self._hier_b0_arg,
                    hier_used_only=self.hier_used_only,
                    hier_min_evidence=self.hier_min_evidence)

    @torch.no_grad()
    def _copy_hyper_from(self, other):
        for name in self._HYPER_BUFFERS:
            getattr(self, name).copy_(getattr(other, name))
        if self.hier_bias:
            self._hier_sync()

    def _hier_pre_load_hook(self, state_dict, prefix, *args):
        # Checkpoints written before the hierarchical prior existed lack these
        # scalars; seed them from the live module so a strict load succeeds.
        for name in self._HYPER_BUFFERS:
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = getattr(self, name).detach().clone()

    @torch.no_grad()
    def _hier_sync(self):
        """Write E[mu] and 1/E[lam] into the prior slots every other method reads.

        Everything downstream (PG refits, birth seeding, merge refits, KL) uses
        m0 / sigma0_diag as the bias prior, so keeping those two slots equal to
        the hyper-posterior means is what makes the hierarchy take effect
        without touching any other code path.
        """
        self.m0[-1] = self.hyp_mu_m
        self.sigma0_diag[-1] = self.hyp_lam_b / self.hyp_lam_a

    def hier_rows(self):
        """Rows that enter the hyper-posterior update."""
        if not self.hier_used_only or not self._pg_init:
            return torch.ones(self.K, dtype=torch.bool, device=self.m0.device)
        # pg_A[k, -1, -1] = sum_t E[omega_tk] (phi's bias slot is 1): the gate
        # evidence accumulated by row k.  Rows below the threshold sit at the
        # prior and carry no information about the population.
        return self.pg_A[:, -1, -1] >= self.hier_min_evidence

    @torch.no_grad()
    def hier_update(self, n_iter: int = 1):
        """Mean-field coordinate ascent for q(mu), q(lam) given q(beta).

            q(mu)  = N( (m00/v00 + E[lam] sum_k E[r_k]) / (1/v00 + K E[lam]),
                        1 / (1/v00 + K E[lam]) )
            q(lam) = Gamma( a0 + K/2,
                            b0 + 1/2 sum_k ( Var[r_k] + (E[r_k] - E[mu])^2 + Var[mu] ) )

        with r_k the bias coordinate of beta_k.  Cheap (O(K)), run after every
        gate refit.  Empty rows sit at the current prior (E[r_k] = E[mu]), so
        they cancel exactly from the q(mu) fixed point and enter the q(lam)
        fixed point only through K_empty * Var[mu] / 2 (small, but not zero:
        under the factorised approximation the precision estimate does depend
        weakly on how many unobserved rows are instantiated).  hier_used_only=True
        removes them from BOTH hier_update() and beta_kl(), which defines a
        hierarchy over evidence-bearing rows only; it is a different (consistent)
        objective, not a shortcut to the same one.
        """
        if not self.hier_bias:
            return
        sel = self.hier_rows()
        Kc = int(sel.sum())
        if Kc == 0:
            return
        r_m = self.m_beta[sel, -1]
        r_v = self.Sigma_beta[sel, -1, -1].clamp_min(0.0)
        m00, v00, a0, b0 = self.hyp_m00, self.hyp_v00, self.hyp_a0, self.hyp_b0
        a, b = self.hyp_lam_a.clone(), self.hyp_lam_b.clone()
        for _ in range(max(1, int(n_iter))):
            Elam = a / b
            prec = 1.0 / v00 + Kc * Elam
            mu_m = (m00 / v00 + Elam * r_m.sum()) / prec
            mu_v = 1.0 / prec
            S = (r_v + (r_m - mu_m) ** 2).sum() + Kc * mu_v
            a = a0 + 0.5 * Kc
            b = b0 + 0.5 * S
        if not all(bool(torch.isfinite(t)) for t in (mu_m, mu_v, a, b)):
            self.n_pg_guard_rejects = getattr(self, "n_pg_guard_rejects", 0) + 1
            return
        self.hyp_mu_m.copy_(mu_m); self.hyp_mu_v.copy_(mu_v)
        self.hyp_lam_a.copy_(a); self.hyp_lam_b.copy_(b)
        self._hier_sync()

    @torch.no_grad()
    def hier_summary(self):
        """(E[mu], 1/E[lam], sigmoid(E[mu])).

        1/E[lam] = b/a is the EFFECTIVE prior variance used in the coefficient
        update (the Gaussian update needs E[lam]); it is not the expected
        conditional variance E[1/lam] = b/(a-1), and a NEW row's predictive
        variance is E[1/lam] + Var[mu].  sigmoid(E[mu]) is a plug-in "typical"
        persistence, not E[sigmoid(r)].  See hier_report() for all of them."""
        Elam = self.hyp_lam_a / self.hyp_lam_b
        return (float(self.hyp_mu_m), float(1.0 / Elam), float(torch.sigmoid(self.hyp_mu_m)))

    @torch.no_grad()
    def hier_report(self):
        a, b = self.hyp_lam_a, self.hyp_lam_b
        E_inv_lam = b / (a - 1.0) if float(a) > 1.0 else b / a
        pred_var = E_inv_lam + self.hyp_mu_v
        # E[sigmoid(r)] for a new row under N(E[mu], pred_var), probit approximation
        pers = torch.sigmoid(self.hyp_mu_m / torch.sqrt(1.0 + math.pi * pred_var / 8.0))
        return dict(logit_mean=float(self.hyp_mu_m), logit_mean_var=float(self.hyp_mu_v),
                    eff_var=float(b / a), cond_var=float(E_inv_lam), pred_var=float(pred_var),
                    persist_plugin=float(torch.sigmoid(self.hyp_mu_m)), persist_expected=float(pers))

    def _psi_moments(self, phi):
        mb = self.m_beta.to(phi.dtype)
        Sb = self.Sigma_beta.to(phi.dtype)
        mu = torch.einsum("...d,kd->...k", phi, mb)
        v = torch.einsum("...d,kde,...e->...k", phi, Sb, phi)
        return mu, v.clamp_min(0.0)

    def sigma(self, phi):
        mu, v = self._psi_moments(phi)
        if self.uncorr:
            z = mu / torch.sqrt(1.0 + (math.pi / 8.0) * v)
        else:
            z = mu
        return torch.sigmoid(z)

    def jj_branch_potentials(self, phi):
        m, v = self._psi_moments(phi)
        c = torch.sqrt((m * m + v).clamp_min(1e-12))
        const = torch.nn.functional.logsigmoid(c) - 0.5 * c
        A = const + 0.5 * m
        B0 = const - 0.5 * m
        return A, B0, m, c

    def bound_log_trans(self, base_elogpi, phi_steps):
        K = base_elogpi.shape[0]
        elog = base_elogpi.to(phi_steps.dtype)
        A, B0, m, c = self.jj_branch_potentials(phi_steps)
        switch = B0[..., :, None] + elog
        diag_persist = A
        eye = torch.eye(K, dtype=switch.dtype, device=switch.device)
        big_neg = torch.finfo(switch.dtype).min / 4.0
        persist_full = diag_persist[..., :, None] + (1.0 - eye) * big_neg
        log_trans = torch.logaddexp(persist_full, switch)
        aux = dict(A=A, B0=B0, m=m, c=c,
                   switch_diag=torch.diagonal(switch, dim1=-2, dim2=-1),
                   base_elogpi=elog)
        return log_trans, aux

    def bound_aux_only(self, base_elogpi, phi_steps):
        elog = base_elogpi.to(phi_steps.dtype)
        A, B0, m, c = self.jj_branch_potentials(phi_steps)
        switch_diag = B0 + torch.diagonal(elog)
        return dict(A=A, B0=B0, m=m, c=c, switch_diag=switch_diag, base_elogpi=elog)

    @staticmethod
    def trans_slice_from_aux(aux, t):
        A = aux["A"][:, t - 1]
        B0 = aux["B0"][:, t - 1]
        elog = aux["base_elogpi"]
        K = A.shape[-1]
        switch = B0[..., :, None] + elog
        eye = torch.eye(K, dtype=A.dtype, device=A.device)
        big_neg = torch.finfo(A.dtype).min / 4.0
        persist = A[..., :, None] + (1.0 - eye) * big_neg
        return torch.logaddexp(persist, switch)

    @staticmethod
    def attribute_bound(xi, aux):
        A = aux["A"]
        Bd = aux["switch_diag"]
        w1_frac = torch.exp(A - torch.logaddexp(A, Bd))
        diag_xi = torch.diagonal(xi, dim1=-2, dim2=-1)
        r_mass = diag_xi * w1_frac
        row_weight = xi.sum(dim=-1)
        Cbase = xi.clone()
        newdiag = diag_xi * (1.0 - w1_frac)
        Cbase = Cbase - torch.diag_embed(diag_xi) + torch.diag_embed(newdiag)
        return r_mass, row_weight, Cbase.sum(dim=(0, 1))

    def effective_log_trans(self, base_elogpi, phi_steps):
        K = base_elogpi.shape[0]
        Pi = torch.softmax(base_elogpi, dim=-1)
        sig = self.sigma(phi_steps)
        eye = torch.eye(K, dtype=Pi.dtype, device=Pi.device)
        s = sig[..., :, None]
        M = s * eye + (1.0 - s) * Pi
        return M.clamp_min(1e-30).log(), sig, Pi

    @staticmethod
    def attribute(xi, sig, Pi):
        K = Pi.shape[0]
        eye = torch.eye(K, dtype=Pi.dtype, device=Pi.device)
        s = sig[..., :, None]
        M = (s * eye + (1.0 - s) * Pi).clamp_min(1e-30)

        diag_xi = torch.diagonal(xi, dim1=-2, dim2=-1)
        diag_M = torch.diagonal(M, dim1=-2, dim2=-1)
        r_mass = diag_xi * (sig / diag_M)
        row_weight = xi.sum(dim=-1)

        base_frac = ((1.0 - s) * Pi) / M
        Cbase = (xi * base_frac).sum(dim=(0, 1))
        return r_mass, row_weight, Cbase

    @torch.no_grad()
    def pg_update_statewise(self, phi, r_mass, row_weight, lr=None):
        wd = self.m_beta.dtype
        phi = phi.to(wd)
        r_mass = r_mass.to(wd)
        row_weight = row_weight.to(wd).clamp_min(0.0)

        K, Dp = self.K, phi.shape[-1]
        Sig0_inv = torch.diag(1.0 / self.sigma0_diag)
        rhs_prior = Sig0_inv @ self.m0

        def solve_spd(prec, rhs):
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
        self.hier_update()

    @torch.no_grad()
    def pg_update(self, phi, r, weight=None, lr=None):
        if weight is None:
            weight = torch.ones_like(r)
        r_mass = r[:, None].expand(-1, self.K) / max(self.K, 1)
        row_weight = weight[:, None].expand(-1, self.K) / max(self.K, 1)
        return self.pg_update_statewise(phi, r_mass, row_weight, lr=lr)

    @torch.no_grad()
    def pg_stats_from_batch(self, phi, r_mass, row_weight):
        wd = self.m_beta.dtype
        phi = phi.to(wd)
        r = r_mass.to(wd)
        w = row_weight.to(wd).clamp_min(0.0)
        EbbT = self.Sigma_beta + torch.einsum("kd,ke->kde", self.m_beta, self.m_beta)
        c = torch.einsum("nd,kde,ne->nk", phi, EbbT, phi).clamp_min(1e-12).sqrt()
        Eom = (0.5 / c) * torch.tanh(0.5 * c) * w
        A = torch.einsum("nk,nd,ne->kde", Eom, phi, phi)
        h = torch.einsum("nk,nd->kd", r - 0.5 * w, phi)
        return dict(A=A, h=h)

    @torch.no_grad()
    def pg_set_totals(self, A, h):
        if not (bool(torch.isfinite(A).all()) and bool(torch.isfinite(h).all())):
            self.n_pg_guard_rejects = getattr(self, 'n_pg_guard_rejects', 0) + 1
            return
        wd = self.m_beta.dtype
        A = A.to(wd)
        h = h.to(wd)
        Sig0_inv = torch.diag(1.0 / self.sigma0_diag)
        rhs_prior = Sig0_inv @ self.m0
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
                return
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
        self.hier_update()

    @torch.no_grad()
    def beta_kl(self):
        Dp = self.m_beta.shape[-1]
        inv0 = 1.0 / self.sigma0_diag
        dm = self.m_beta - self.m0.view(1, Dp)
        tr = (torch.diagonal(self.Sigma_beta, dim1=-2, dim2=-1) * inv0.view(1, Dp)).sum(-1)
        maha = (dm.pow(2) * inv0.view(1, Dp)).sum(-1)
        logdet0 = torch.log(self.sigma0_diag).sum()
        logdetq = torch.linalg.slogdet(self.Sigma_beta).logabsdet
        kl = 0.5 * (tr + maha - Dp + logdet0 - logdetq).sum()
        if not self.hier_bias:
            return kl
        # Hierarchical correction.  The plug-in term above scored the bias slot
        # against N(E[mu], 1/E[lam]); the exact mean-field expectation of
        # log p(r_k | mu, lam) differs by the E[log lam] vs -log E[1/lam]... terms
        # and by Var[mu].  Replace the bias slot's contribution and add the
        # hyper-KLs KL(q(mu)||p(mu)) + KL(q(lam)||p(lam)).
        # Rows outside hier_rows() (only when hier_used_only=True) are NOT part
        # of the hierarchy: they keep the plug-in prior N(E[mu], 1/E[lam]) already
        # scored above, so the bound and hier_update() optimise the same
        # objective.  With the default hier_used_only=False every row is in.
        sel = self.hier_rows()
        r_m = self.m_beta[sel, -1]
        r_v = torch.diagonal(self.Sigma_beta, dim1=-2, dim2=-1)[sel, -1]
        Elam = self.hyp_lam_a / self.hyp_lam_b
        Eloglam = torch.special.digamma(self.hyp_lam_a) - torch.log(self.hyp_lam_b)
        mu_m, mu_v = self.hyp_mu_m, self.hyp_mu_v
        # remove plug-in bias-slot cross-entropy: -1/2 log(2 pi s0) - 1/2 (v + (m - m0)^2)/s0
        s0 = self.sigma0_diag[-1]
        plug = (-0.5 * torch.log(2 * math.pi * s0)
                - 0.5 * (r_v + (r_m - self.m0[-1]) ** 2) / s0).sum()
        exact = (0.5 * Eloglam - 0.5 * math.log(2 * math.pi)
                 - 0.5 * Elam * (r_v + (r_m - mu_m) ** 2 + mu_v)).sum()
        kl = kl + (plug - exact)                      # KL = E[log q] - E[log p]
        kl_mu = 0.5 * ((mu_v + (mu_m - self.hyp_m00) ** 2) / self.hyp_v00 - 1.0
                       + torch.log(self.hyp_v00 / mu_v))
        a, b, a0, b0 = self.hyp_lam_a, self.hyp_lam_b, self.hyp_a0, self.hyp_b0
        kl_lam = ((a - a0) * torch.special.digamma(a) - torch.lgamma(a) + torch.lgamma(a0)
                  + a0 * (torch.log(b) - torch.log(b0)) + a * (b0 - b) / b)
        return kl + kl_mu + kl_lam

    @torch.no_grad()
    def select_rows(self, keep_idx):
        keep = torch.as_tensor(keep_idx, device=self.m0.device, dtype=torch.long)
        new = RecurrentStickiness(
            K=int(keep.numel()), feat_dim=self.D,
            device=self.m0.device, dtype=self._dtype, **self._ctor_kwargs(),
        )
        new.m0.copy_(self.m0)
        new.sigma0_diag.copy_(self.sigma0_diag)
        new.m_beta.copy_(self.m_beta[keep])
        new.Sigma_beta.copy_(self.Sigma_beta[keep])
        new.pg_A.copy_(self.pg_A[keep])
        new.pg_h.copy_(self.pg_h[keep])
        new._pg_init = self._pg_init
        new._copy_hyper_from(self)
        return new

    @torch.no_grad()
    def merge_rows(self, i, j):
        """Merge gate row j into row i, then drop j.

        `select_rows` alone is wrong for a merge: it retains row i's gate and
        discards row j's, so the merged state inherits only one parent's
        persistence evidence and the other parent's PG statistics are lost. The
        natural-parameter statistics (pg_A, pg_h) are additive over (n, t) --- they
        are the G^(2) and G^(1) accumulators of the JJ bound --- so the merged row
        is their sum, and the Gaussian posterior is refit from it:

            A' = A_i + A_j,   h' = h_i + h_j
            Sigma' = (Sigma_theta^-1 + A')^-1,   m' = Sigma' (Sigma_theta^-1 mu_theta + h')

        This is the gate analogue of M'_ii = M_ii + M_jj + M_ij + M_ji for the
        transition counts: both are sums of per-step evidence, so both merge by
        addition rather than by selection.
        """
        i, j = int(i), int(j)
        if i == j:
            raise ValueError("merge_rows requires i != j")
        # NON-MUTATING.  Merge candidates are scored in a loop over many (i, j)
        # pairs against one baseline object; mutating self here would leak row j's
        # evidence into the baseline, so pair (0,2) would be scored on top of the
        # already-merged row 0 from pair (0,1) -- contaminating every later
        # shortlist score and candidate.  Work on a full clone instead.
        out = self.select_rows(list(range(self.K)))
        out.pg_A[i] = out.pg_A[i] + out.pg_A[j]
        out.pg_h[i] = out.pg_h[i] + out.pg_h[j]
        Sig0_inv = torch.diag(1.0 / out.sigma0_diag)
        Sig = torch.linalg.inv(Sig0_inv + out.pg_A[i])
        out.Sigma_beta[i] = Sig
        out.m_beta[i] = Sig @ (Sig0_inv @ out.m0 + out.pg_h[i])
        keep = [k for k in range(out.K) if k != j]
        return out.select_rows(keep)

    @torch.no_grad()
    def resized_like(self, new_K: int):
        new = RecurrentStickiness(
            K=int(new_K), feat_dim=self.D,
            device=self.m0.device, dtype=self._dtype, **self._ctor_kwargs(),
        )
        new.m0.copy_(self.m0)
        new.sigma0_diag.copy_(self.sigma0_diag)
        new._copy_hyper_from(self)
        n = min(self.K, int(new_K))
        new.m_beta[:n].copy_(self.m_beta[:n])
        new.Sigma_beta[:n].copy_(self.Sigma_beta[:n])
        new.pg_A[:n].copy_(self.pg_A[:n])
        new.pg_h[:n].copy_(self.pg_h[:n])
        # Rows beyond n are newborns: the constructor seeded them from the
        # DEFAULT prior, so re-seed them from the prior actually in force
        # (which may be moment-matched).  Without this a birth silently
        # reintroduces the logit-of-the-mean heuristic one row at a time.
        if int(new_K) > n:
            new.m_beta[n:].copy_(new.m0.unsqueeze(0).expand(int(new_K) - n, -1))
            new.Sigma_beta[n:].copy_(
                torch.diag(new.sigma0_diag).unsqueeze(0).expand(
                    int(new_K) - n, -1, -1))
        new._pg_init = self._pg_init
        return new

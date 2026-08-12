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
                 device=None, dtype=torch.float64):
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

    def _apply(self, fn, *args, **kwargs):
        module = super()._apply(fn, *args, **kwargs)
        for name in ("m0", "sigma0_diag", "m_beta", "Sigma_beta", "pg_A", "pg_h"):
            buf = getattr(module, name, None)
            if buf is not None and buf.dtype != module._dtype:
                setattr(module, name, buf.to(module._dtype))
        return module

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

    @torch.no_grad()
    def beta_kl(self):
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
        keep = torch.as_tensor(keep_idx, device=self.m0.device, dtype=torch.long)
        new = RecurrentStickiness(
            K=int(keep.numel()), feat_dim=self.D, prior_persist=self.prior_persist,
            weight_prior_var=self.weight_prior_var, bias_prior_var=self.bias_prior_var,
            pg_iters=self.pg_iters, uncertainty_correction=self.uncorr,
            device=self.m0.device, dtype=self._dtype,
        )
        new.m0.copy_(self.m0)
        new.sigma0_diag.copy_(self.sigma0_diag)
        new.m_beta.copy_(self.m_beta[keep])
        new.Sigma_beta.copy_(self.Sigma_beta[keep])
        new.pg_A.copy_(self.pg_A[keep])
        new.pg_h.copy_(self.pg_h[keep])
        new._pg_init = self._pg_init
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
            K=int(new_K), feat_dim=self.D, prior_persist=self.prior_persist,
            weight_prior_var=self.weight_prior_var, bias_prior_var=self.bias_prior_var,
            pg_iters=self.pg_iters, uncertainty_correction=self.uncorr,
            device=self.m0.device, dtype=self._dtype,
        )
        new.m0.copy_(self.m0)
        new.sigma0_diag.copy_(self.sigma0_diag)
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

"""Sticky HDP-HMM global update: the transition posterior q(pi_k) and the
stick-breaking root q(u).

Two coupled updates, alternated (exactly as bnpy's HDPHMM.update_global_params_VB):

(a) Conjugate sticky Dirichlet update of the transition rows, given the expected
    transition counts M_kl from forward-backward and the current sticks:
        transTheta_kl = M_kl + alpha * E[beta_l] + kappa * 1[k==l],
    with an extra remainder column l=K+1 carrying alpha * E[beta_{K+1}].

(b) Non-conjugate update of the stick-breaking root q(u_k)=Beta(rho_k omega_k,
    (1-rho_k) omega_k), by maximising the ELBO over (rho, omega). The objective is
    the sticky (kappa>0) branch of bnpy's OptimizerRhoOmega.objFunc_constrained;
    we optimise it with L-BFGS through an unconstrained reparameterisation and rely
    on autograd (the objective uses only digamma/lgamma, which torch differentiates).

E[log pi] from the Dirichlet posterior feeds both the forward-backward transition
potential and step (b)'s sumLogPi statistic.

Convention note: the prior on u_k is Beta(1, gamma), and beta = rho2beta(rho) is the
stick-breaking map; E[u_k]=rho_k, "concentration" omega_k. All buffers, no optimizer
exposure (updated by the closed-form/numerical M-step, not SGD).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

EPS = 1e-8


def rho2beta(rho: torch.Tensor) -> torch.Tensor:
    """Stick-breaking map rho (K,) -> beta (K+1,), including the remainder mass.

    beta_k = rho_k * prod_{m<k}(1-rho_m) for k=0..K-1,  beta_K = prod_{m<K}(1-rho_m).
    Equivalent to bnpy's StickBreakUtil.rho2beta (K -> K+1).
    """
    one_minus = torch.cumprod(1.0 - rho, dim=-1)              # prod_{m<=k}(1-rho_m)
    shifted = torch.cat([rho.new_ones(1), one_minus[:-1]], dim=-1)  # prod_{m<k}(1-rho_m)
    beta_active = rho * shifted                               # (K,)
    beta_rem = one_minus[-1:]                                 # (1,)
    return torch.cat([beta_active, beta_rem], dim=-1)        # (K+1,)


_rho2beta_safe = rho2beta  # alias used below


def beta2rho(beta: torch.Tensor, K: int) -> torch.Tensor:
    """Inverse stick-breaking: beta (K+1,) -> rho (K,).

    rho_k = beta_k / (1 - sum_{m<k} beta_m).  Exact inverse of `rho2beta`, and the
    closed form used by Hughes et al. (NIPS 2015) supp. F.1 Proposal Step 4/4 to build
    a merge candidate's top-level sticks without re-running the numerical optimiser.
    """
    beta = beta[:K + 1]
    cum = torch.cumsum(beta, dim=-1)
    prev = torch.cat([beta.new_zeros(1), cum[:-1]], dim=-1)   # sum_{m<k} beta_m
    denom = (1.0 - prev[:K]).clamp_min(EPS)
    return (beta[:K] / denom).clamp(EPS, 1.0 - EPS)


def merge_rho_omega(rho: torch.Tensor, omega: torch.Tensor, i: int, j: int):
    """Closed-form (rho', omega') for merging state j into state i (i < j).

    beta'_i = beta_i + beta_j, other actives unchanged, remainder unchanged;
    omega'_i = omega_i + omega_j (Hughes supp. F.1 heuristic for the concentration).
    """
    K = rho.shape[0]
    if i > j:
        i, j = j, i
    beta = rho2beta(rho)                       # (K+1,)
    keep = [k for k in range(K) if k != j]
    b_act = beta[keep].clone()
    pos_i = keep.index(i)
    b_act[pos_i] = b_act[pos_i] + beta[j]
    b_new = torch.cat([b_act, beta[K:K + 1]], dim=-1)
    b_new = b_new / b_new.sum().clamp_min(EPS)
    om = omega[keep].clone()
    om[pos_i] = om[pos_i] + omega[j]
    return beta2rho(b_new, K - 1), om


def drop_rho_omega(rho: torch.Tensor, omega: torch.Tensor, k: int):
    """Closed-form (rho', omega') for deleting state k: its beta mass goes to the
    remainder stick, which is what a delete means under the HDP prior."""
    K = rho.shape[0]
    beta = rho2beta(rho)
    keep = [m for m in range(K) if m != k]
    b_new = torch.cat([beta[keep], (beta[K] + beta[k]).reshape(1)], dim=-1)
    b_new = b_new / b_new.sum().clamp_min(EPS)
    return beta2rho(b_new, K - 1), omega[keep].clone()



def kvec(K: int, device=None, dtype=None) -> torch.Tensor:
    """Descending [K, K-1, ..., 1]."""
    return torch.arange(K, 0, -1, device=device, dtype=dtype)


def c_Beta(g1: torch.Tensor, g0: torch.Tensor) -> torch.Tensor:
    """Summed Beta log-cumulant: sum_k lgamma(g1+g0) - lgamma(g1) - lgamma(g0)."""
    return (torch.lgamma(g1 + g0) - torch.lgamma(g1) - torch.lgamma(g0)).sum()


def c_Dir(theta: torch.Tensor) -> torch.Tensor:
    """Summed Dirichlet log-cumulant over rows.

    For theta (K, K+1): sum_k [ lgamma(sum_l theta_kl) - sum_l lgamma(theta_kl) ].
    For theta (K+1,):   lgamma(sum theta) - sum lgamma(theta).  (bnpy c_Dir)
    """
    if theta.dim() == 1:
        return torch.lgamma(theta.sum()) - torch.lgamma(theta).sum()
    return (torch.lgamma(theta.sum(-1)) - torch.lgamma(theta).sum(-1)).sum()


def L_top(rho, omega, alpha, gamma, kappa, start_alpha):
    """Top-level term of the HDP-HMM surrogate ELBO (bnpy HDPHMMUtil.L_top, kappa>0).

    This carries the K-dependent allocation constants (tAlpha, tKappa, tBeta) and the
    stick-breaking Beta cumulants that make the move-acceptance bound penalise K.
    """
    K = rho.shape[0]
    eta1 = rho * omega
    eta0 = (1.0 - rho) * omega
    dig_om = torch.digamma(omega)
    ElogU = torch.digamma(eta1) - dig_om
    Elog1mU = torch.digamma(eta0) - dig_om

    cB_prior = (torch.lgamma(torch.as_tensor(1.0 + gamma, dtype=rho.dtype, device=rho.device))
                - torch.lgamma(torch.as_tensor(1.0, dtype=rho.dtype, device=rho.device))
                - torch.lgamma(torch.as_tensor(gamma, dtype=rho.dtype, device=rho.device)))
    diff_cBeta = K * cB_prior - c_Beta(eta1, eta0)

    la = torch.log(torch.as_tensor(alpha, dtype=rho.dtype, device=rho.device))
    lsa = torch.log(torch.as_tensor(start_alpha, dtype=rho.dtype, device=rho.device))
    tAlpha = K * K * la + K * lsa

    coefU = (K + 1.0) - eta1
    coef1mU = K * kvec(K, rho.device, rho.dtype) + 1.0 + gamma - eta0
    if kappa > 0:
        beta_active = rho2beta(rho)[:K]
        sumEBeta = beta_active.sum()
        lak = torch.log(torch.as_tensor(alpha + kappa, dtype=rho.dtype, device=rho.device))
        lk = torch.log(torch.as_tensor(kappa, dtype=rho.dtype, device=rho.device))
        tBeta = sumEBeta * (lak - lk)
        tKappa = K * (lk - lak)
    else:
        # non-sticky HDP-HMM (bnpy HDPHMMUtil.L_top, kappa==0 branch): the init
        # distribution folds in as an extra row, so the U-coefficients gain (K+1)
        coefU = (K + 1.0) + 1.0 - eta1
        coef1mU = (K + 1.0) * kvec(K, rho.device, rho.dtype) + gamma - eta0
        tBeta = torch.zeros((), dtype=rho.dtype, device=rho.device)
        tKappa = torch.zeros((), dtype=rho.dtype, device=rho.device)

    diff_logU = torch.inner(coefU, ElogU) + torch.inner(coef1mU, Elog1mU)
    return tAlpha + tKappa + tBeta + diff_cBeta + diff_logU


class StickyHDP(nn.Module):
    def __init__(
        self,
        K: int,
        gamma: float = 5.0,
        alpha: float = 1.0,
        kappa: float = 50.0,
        start_alpha: float = 1.0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float64,
    ):
        super().__init__()
        self.K = K
        self.gamma = float(gamma)        # top-level DP concentration (prior u_k ~ Beta(1,gamma))
        self.alpha = float(alpha)        # second-level DP concentration (transAlpha)
        self.kappa = float(kappa)        # stickiness
        self.start_alpha = float(start_alpha)

        # initial rho implies near-uniform beta with small remainder mass
        remMass = min(0.1, 1.0 / (K * K))
        delta = (-1.0 + remMass) * torch.arange(0, K, dtype=dtype, device=device) 
        rho0 = (1.0 - remMass) / (K + delta)
        self.register_buffer("rho", rho0.clamp(EPS, 1 - EPS))
        self.register_buffer("omega", (1.0 + self.gamma) * torch.ones(K, dtype=dtype, device=device))
        self.lbfgs_max_iter = 60   # warm-started root update needs far fewer than 200
        self.register_buffer("trans_theta", torch.zeros(K, K + 1, dtype=dtype, device=device))
        self.register_buffer("start_theta", torch.zeros(K + 1, dtype=dtype, device=device))
        self._dtype = dtype
        # initialise theta from the prior sticks with zero counts
        with torch.no_grad():
            tt, st = self._calc_theta(torch.zeros(K, K, dtype=dtype, device=self.rho.device),
                                      torch.zeros(K, dtype=dtype, device=self.rho.device))
            self.trans_theta.copy_(tt)
            self.start_theta.copy_(st)

    # The HDP posterior buffers run in float64 for the digamma/lgamma optimisation; pin their
    # dtype so a module-wide model.float()/.half()/.to(dtype) cannot silently downcast them
    # (device moves still propagate). Keeps self._dtype authoritative and consistent.
    def _apply(self, fn, *args, **kwargs):
        module = super()._apply(fn, *args, **kwargs)
        for name in ("rho", "omega", "trans_theta", "start_theta"):
            buf = getattr(module, name, None)
            if buf is not None and buf.dtype != module._dtype:
                setattr(module, name, buf.to(module._dtype))
        return module

    #  stick-breaking
    def Ebeta(self) -> torch.Tensor:
        return _rho2beta_safe(self.rho)

    #  conjugate theta update
    def _calc_theta(self, trans_counts: torch.Tensor, start_counts: torch.Tensor):
        """transTheta (K,K+1) and startTheta (K+1,) from counts and current rho."""
        K = self.K
        Ebeta = _rho2beta_safe(self.rho)                       # (K+1,)
        alphaEBeta = self.alpha * Ebeta                        # (K+1,)
        trans_theta = alphaEBeta.unsqueeze(0).repeat(K, 1)     # (K,K+1)
        eye = torch.eye(K, dtype=trans_theta.dtype, device=trans_theta.device)
        trans_theta = trans_theta.clone()
        trans_theta[:, :K] = trans_theta[:, :K] + trans_counts + self.kappa * eye
        start_theta = self.start_alpha * Ebeta                 # (K+1,)
        start_theta = start_theta.clone()
        start_theta[:K] = start_theta[:K] + start_counts
        return trans_theta, start_theta

    @staticmethod
    def expected_log_pi(theta: torch.Tensor) -> torch.Tensor:
        """E[log pi] = digamma(theta) - digamma(rowsum)."""
        return torch.digamma(theta) - torch.digamma(theta.sum(-1, keepdim=True))

    def expected_log_trans(self, include_remainder: bool = False) -> torch.Tensor:
        """Expected log transition matrix for forward-backward.

        Returns the active K x K block of E[log pi] by default (the potential among
        instantiated regimes); set include_remainder to also return the K+1 column.
        """
        elp = self.expected_log_pi(self.trans_theta)           # (K, K+1)
        return elp if include_remainder else elp[:, :self.K]

    def expected_log_init(self) -> torch.Tensor:
        """E[log pi_0] over the active K (for the forward-backward initial potential)."""
        elp = torch.digamma(self.start_theta) - torch.digamma(self.start_theta.sum())
        return elp[:self.K]

    # ----------------------------------------------- rho/omega ELBO objective
    def _neg_elbo(self, rho: torch.Tensor, omega: torch.Tensor,
                  sumLogPi: torch.Tensor, startAlphaLogPi: torch.Tensor) -> torch.Tensor:
        """Negative ELBO in (rho, omega): the sticky (kappa>0) branch of bnpy's
        objFunc_constrained. sumLogPi and startAlphaLogPi are length K+1."""
        K = self.K
        g1 = rho * omega
        g0 = (1.0 - rho) * omega
        dig_omega = torch.digamma(omega)
        Elogu = torch.digamma(g1) - dig_omega
        Elog1mu = torch.digamma(g0) - dig_omega

        kv = kvec(K, device=rho.device, dtype=rho.dtype)
        import math
        if self.kappa > 0:
            ONcoef = (K + 1.0) - g1                                   # (K,)
            OFFcoef = K * kv + 1.0 + self.gamma - g0                  # (K,)
            Tvec = self.alpha * sumLogPi + startAlphaLogPi            # (K+1,)
            Tvec = Tvec.clone()
            Tvec[:-1] = Tvec[:-1] + (math.log(self.alpha + self.kappa) - math.log(self.kappa))
        else:
            # non-sticky HDP-HMM. This is bnpy's kappa==0 objFunc multiplied through by
            # nDoc=K+1 (same optimum), which matches the kappa==0 L_top term by term.
            ONcoef = (K + 1.0) + 1.0 - g1                             # (K,)
            OFFcoef = (K + 1.0) * kv + self.gamma - g0                # (K,)
            Tvec = self.alpha * sumLogPi + startAlphaLogPi            # (K+1,) no sticky shift
        Ebeta = _rho2beta_safe(rho)                                  # (K+1,)
        elbo_local = (Ebeta * Tvec).sum()

        elbo = (-c_Beta(g1, g0)
                + (ONcoef * Elogu).sum()
                + (OFFcoef * Elog1mu).sum()
                + elbo_local)
        return -elbo

    #  numerical optimisation
    def optimize_rho_omega(self, sumLogPi, startAlphaLogPi, n_iter: int = 200,
                           lr: float = 0.3):
        n_iter = int(min(n_iter, getattr(self, 'lbfgs_max_iter', n_iter)))
        """Maximise the ELBO over (rho, omega) via L-BFGS on an unconstrained
        reparameterisation rho=sigmoid(a), omega=softplus(b)."""
        dtype, device = self.rho.dtype, self.rho.device
        # init from current values
        a = torch.logit(self.rho.clamp(1e-4, 1 - 1e-4)).clone().detach().requires_grad_(True)
        b = torch.log(torch.expm1(self.omega.clamp(min=1e-3))).clone().detach().requires_grad_(True)

        opt = torch.optim.LBFGS([a, b], lr=lr, max_iter=n_iter,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                                tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            rho = torch.sigmoid(a).clamp(EPS, 1 - EPS)
            omega = torch.nn.functional.softplus(b) + 1e-6
            loss = self._neg_elbo(rho, omega, sumLogPi, startAlphaLogPi)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            rho = torch.sigmoid(a).clamp(EPS, 1 - EPS)
            omega = torch.nn.functional.softplus(b) + 1e-6
        return rho.detach(), omega.detach()

    # full M-step
    @torch.no_grad()
    def update(self, trans_counts: torch.Tensor, start_counts: torch.Tensor,
               n_global_iters: int = 3):
        """Alternate the conjugate theta update and the rho/omega optimisation."""
        trans_counts = trans_counts.to(dtype=self._dtype, device=self.rho.device)
        start_counts = start_counts.to(dtype=self._dtype, device=self.rho.device)
        tt, st = self._calc_theta(trans_counts, start_counts)
        if not (bool(torch.isfinite(tt).all()) and bool(torch.isfinite(st).all())):
            # TRANSACTIONAL guard: refuse to install a nonfinite Dirichlet state and
            # keep the previous valid (theta, rho, omega) entirely.
            self.n_guard_rejects = getattr(self, "n_guard_rejects", 0) + 1
            return
        self.trans_theta.copy_(tt)
        self.start_theta.copy_(st)
        for _ in range(n_global_iters):
            elp = self.expected_log_pi(self.trans_theta)               # (K,K+1)
            sumLogPi = elp.sum(0)                                      # (K+1,)
            startELogPi = (torch.digamma(self.start_theta)
                           - torch.digamma(self.start_theta.sum()))    # (K+1,)
            startAlphaLogPi = self.start_alpha * startELogPi
            rho0 = self.rho.clone()
            omega0 = self.omega.clone()
            obj0 = float(self.alloc_elbo())
            try:
                with torch.enable_grad():
                    rho, omega = self.optimize_rho_omega(sumLogPi, startAlphaLogPi)
            except Exception:
                # L-BFGS itself raised: restore the snapshot
                # (rho0, omega0) and their conjugate theta, count, and stop iterating.
                self.rho.copy_(rho0)
                self.omega.copy_(omega0)
                tt, st = self._calc_theta(trans_counts, start_counts)
                self.trans_theta.copy_(tt)
                self.start_theta.copy_(st)
                self.n_guard_rejects = getattr(self, "n_guard_rejects", 0) + 1
                break
            if not (bool(torch.isfinite(rho).all()) and bool(torch.isfinite(omega).all())
                    and bool((omega > 0).all()) and bool((rho > 0).all())
                    and bool((rho < 1).all())):
                # keep the previous (rho, omega); the theta installed above is a valid
                # conjugate half-step under them, so the retained state is consistent.
                self.n_guard_rejects = getattr(self, "n_guard_rejects", 0) + 1
                break
            self.rho.copy_(rho)
            self.omega.copy_(omega)
            tt, st = self._calc_theta(trans_counts, start_counts)
            if not (bool(torch.isfinite(tt).all()) and bool(torch.isfinite(st).all())):
                # nonfinite Dirichlet params under the NEW (rho, omega): restore the
                # snapshot too, not just break.
                self.rho.copy_(rho0)
                self.omega.copy_(omega0)
                tt0, st0 = self._calc_theta(trans_counts, start_counts)
                self.trans_theta.copy_(tt0)
                self.start_theta.copy_(st0)
                self.n_guard_rejects = getattr(self, "n_guard_rejects", 0) + 1
                break
            self.trans_theta.copy_(tt)
            self.start_theta.copy_(st)
            obj1 = float(self.alloc_elbo())
            if (not math.isfinite(obj1)) or obj1 < obj0 - 1e-6 * (abs(obj0) + 1.0):
                # OBJECTIVE ROLLBACK: an L-BFGS overshoot
                # is never installed -- restore the previous (rho, omega) and their
                # conjugate theta, count the rejection, stop iterating.
                self.rho.copy_(rho0)
                self.omega.copy_(omega0)
                tt, st = self._calc_theta(trans_counts, start_counts)
                self.trans_theta.copy_(tt)
                self.start_theta.copy_(st)
                self.n_guard_rejects = getattr(self, 'n_guard_rejects', 0) + 1
                break

    @torch.no_grad()
    def root_kl(self) -> torch.Tensor:
        """sum_k KL( q(u_k)=Beta(rho_k omega_k,(1-rho_k)omega_k) || Beta(1, gamma) ).

        The stick-breaking complexity: instantiating more regimes costs more here, so
        this is the term that penalises K in birth/merge/delete acceptance.
        """
        aq = self.rho * self.omega
        bq = (1.0 - self.rho) * self.omega
        ap = torch.ones_like(aq)
        bp = self.gamma * torch.ones_like(bq)
        logB_q = torch.lgamma(aq) + torch.lgamma(bq) - torch.lgamma(aq + bq)
        logB_p = torch.lgamma(ap) + torch.lgamma(bp) - torch.lgamma(ap + bp)
        kl = (logB_p - logB_q
              + (aq - ap) * torch.digamma(aq)
              + (bq - bp) * torch.digamma(bq)
              + (ap - aq + bp - bq) * torch.digamma(aq + bq))
        return kl.sum()

    @torch.no_grad()
    def alloc_elbo(self) -> torch.Tensor:
        """Allocation ELBO term L_top - c_Dir(transTheta) - c_Dir(startTheta).

        This is the part of the HDP-HMM ELBO that is constant/linear in the suff
        stats at the optimum (bnpy calcELBO_LinearTerms with afterGlobalStep=True,
        whose slack term vanishes). It is what a correct birth/merge/delete bound must
        include so that instantiating a regime is properly penalised; it replaces the
        ad-hoc root_kl in the acceptance score.
        """
        Ltop = L_top(self.rho, self.omega, self.alpha, self.gamma,
                     self.kappa, self.start_alpha)
        return Ltop - c_Dir(self.trans_theta) - c_Dir(self.start_theta)

    @torch.no_grad()
    def exact_alloc_elbo(self) -> torch.Tensor:
        """EXACT allocation ELBO for ARBITRARY theta: L_top - c_Dir + linear slack.

        The full HDP-HMM surrogate ELBO decomposes, for q(s) computed by
        forward-backward under the potentials P = E_q[log pi], as

            L = sum_b logZ_b - param_kl
                + L_top - c_Dir(transTheta) - c_Dir(startTheta)
                + sum_{k,l} (alpha E[beta_l] + kappa 1[k==l] - theta_kl) P_kl
                + sum_l    (start_alpha E[beta_l] - theta0_l) P0_l.

        All count dependence enters through logZ (the FB identity
        logZ = Ldata + <M,P> + <s,P0> + H[q(s)] cancels the <M,P> term of
        E_q[log p(s|pi)]), so this term is exact for ANY Dirichlet posterior theta,
        not only one refit on the scored counts. `alloc_elbo` plus the historical
        "logZ - <M,P> - <s,P0>" pattern is the afterGlobalStep=True specialisation
        (theta = counts + prior mean), where the slack collapses to -<M,P> - <s,P0>;
        the two agree in that case. Frozen-globals scoring (delete) and EMA-fitted
        base models need this exact form.
        """
        K = self.K
        Ebeta = _rho2beta_safe(self.rho)                       # (K+1,)
        prior_trans = self.alpha * Ebeta.unsqueeze(0).repeat(K, 1)
        prior_trans[:, :K] = prior_trans[:, :K] + self.kappa * torch.eye(
            K, dtype=self._dtype, device=self.rho.device)
        P = self.expected_log_pi(self.trans_theta)             # (K,K+1)
        slack_trans = ((prior_trans - self.trans_theta) * P).sum()

        prior_start = self.start_alpha * Ebeta
        P0 = torch.digamma(self.start_theta) - torch.digamma(self.start_theta.sum())
        slack_start = ((prior_start - self.start_theta) * P0).sum()

        Ltop = L_top(self.rho, self.omega, self.alpha, self.gamma,
                     self.kappa, self.start_alpha)
        return (Ltop - c_Dir(self.trans_theta) - c_Dir(self.start_theta)
                + slack_trans + slack_start)

    @torch.no_grad()
    @torch.no_grad()
    def seed_rho_omega(self, rho_new: torch.Tensor, omega_new: torch.Tensor):
        """Install closed-form (rho, omega) without any numerical optimisation.

        Used for merge/delete CANDIDATE scoring so the shortlist ranking is a
        deterministic function of the merged statistics rather than of how far a
        cold-started L-BFGS happened to converge.
        """
        self.rho.copy_(rho_new.to(self.rho.dtype).clamp(EPS, 1 - EPS))
        self.omega.copy_(omega_new.to(self.omega.dtype).clamp_min(1e-3))
        return self

    def resized_like(self, new_K: int):
        """A fresh StickyHDP at a different truncation with the same hyperparameters."""
        return StickyHDP(K=new_K, gamma=self.gamma, alpha=self.alpha, kappa=self.kappa,
                         start_alpha=self.start_alpha, dtype=self._dtype,
                         device=self.rho.device)

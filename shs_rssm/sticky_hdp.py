from __future__ import annotations

import math
import torch
import torch.nn as nn

EPS = 1e-8


def rho2beta(rho: torch.Tensor) -> torch.Tensor:
    one_minus = torch.cumprod(1.0 - rho, dim=-1)
    shifted = torch.cat([rho.new_ones(1), one_minus[:-1]], dim=-1)
    beta_active = rho * shifted
    beta_rem = one_minus[-1:]
    return torch.cat([beta_active, beta_rem], dim=-1)


_rho2beta_safe = rho2beta


def beta2rho(beta: torch.Tensor, K: int) -> torch.Tensor:
    beta = beta[:K + 1]
    cum = torch.cumsum(beta, dim=-1)
    prev = torch.cat([beta.new_zeros(1), cum[:-1]], dim=-1)
    denom = (1.0 - prev[:K]).clamp_min(EPS)
    return (beta[:K] / denom).clamp(EPS, 1.0 - EPS)


def merge_rho_omega(rho: torch.Tensor, omega: torch.Tensor, i: int, j: int):
    K = rho.shape[0]
    if i > j:
        i, j = j, i
    beta = rho2beta(rho)
    keep = [k for k in range(K) if k != j]
    b_act = beta[keep].clone()
    pos_i = keep.index(i)
    b_act[pos_i] = b_act[pos_i] + beta[j]
    b_new = torch.cat([b_act, beta[K:K + 1]], dim=-1)
    b_new = b_new / b_new.sum().clamp_min(EPS)
    om = omega[keep].clone()
    return beta2rho(b_new, K - 1), om


def drop_rho_omega(rho: torch.Tensor, omega: torch.Tensor, k: int):
    K = rho.shape[0]
    beta = rho2beta(rho)
    keep = [m for m in range(K) if m != k]
    b_new = torch.cat([beta[keep], (beta[K] + beta[k]).reshape(1)], dim=-1)
    b_new = b_new / b_new.sum().clamp_min(EPS)
    return beta2rho(b_new, K - 1), omega[keep].clone()


def kvec(K: int, device=None, dtype=None) -> torch.Tensor:
    return torch.arange(K, 0, -1, device=device, dtype=dtype)


def c_Beta(g1: torch.Tensor, g0: torch.Tensor) -> torch.Tensor:
    return (torch.lgamma(g1 + g0) - torch.lgamma(g1) - torch.lgamma(g0)).sum()


def c_Dir(theta: torch.Tensor) -> torch.Tensor:
    if theta.dim() == 1:
        return torch.lgamma(theta.sum()) - torch.lgamma(theta).sum()
    return (torch.lgamma(theta.sum(-1)) - torch.lgamma(theta).sum(-1)).sum()


def L_top(rho, omega, alpha, gamma, kappa, start_alpha):
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
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.kappa = float(kappa)
        self.start_alpha = float(start_alpha)

        remMass = min(0.1, 1.0 / (K * K))
        delta = (-1.0 + remMass) * torch.arange(0, K, dtype=dtype, device=device) 
        rho0 = (1.0 - remMass) / (K + delta)
        self.register_buffer("rho", rho0.clamp(EPS, 1 - EPS))
        self.register_buffer("omega", (1.0 + self.gamma) * torch.ones(K, dtype=dtype, device=device))
        self.lbfgs_max_iter = 60
        self.register_buffer("trans_theta", torch.zeros(K, K + 1, dtype=dtype, device=device))
        self.register_buffer("start_theta", torch.zeros(K + 1, dtype=dtype, device=device))
        self._dtype = dtype
        with torch.no_grad():
            tt, st = self._calc_theta(torch.zeros(K, K, dtype=dtype, device=self.rho.device),
                                      torch.zeros(K, dtype=dtype, device=self.rho.device))
            self.trans_theta.copy_(tt)
            self.start_theta.copy_(st)

    def _apply(self, fn, *args, **kwargs):
        module = super()._apply(fn, *args, **kwargs)
        for name in ("rho", "omega", "trans_theta", "start_theta"):
            buf = getattr(module, name, None)
            if buf is not None and buf.dtype != module._dtype:
                setattr(module, name, buf.to(module._dtype))
        return module

    def Ebeta(self) -> torch.Tensor:
        return _rho2beta_safe(self.rho)

    def _calc_theta(self, trans_counts: torch.Tensor, start_counts: torch.Tensor):
        K = self.K
        Ebeta = _rho2beta_safe(self.rho)
        alphaEBeta = self.alpha * Ebeta
        trans_theta = alphaEBeta.unsqueeze(0).repeat(K, 1)
        eye = torch.eye(K, dtype=trans_theta.dtype, device=trans_theta.device)
        trans_theta = trans_theta.clone()
        trans_theta[:, :K] = trans_theta[:, :K] + trans_counts + self.kappa * eye
        start_theta = self.start_alpha * Ebeta
        start_theta = start_theta.clone()
        start_theta[:K] = start_theta[:K] + start_counts
        return trans_theta, start_theta

    @staticmethod
    def expected_log_pi(theta: torch.Tensor) -> torch.Tensor:
        return torch.digamma(theta) - torch.digamma(theta.sum(-1, keepdim=True))

    def expected_log_trans(self, include_remainder: bool = False) -> torch.Tensor:
        elp = self.expected_log_pi(self.trans_theta)
        return elp if include_remainder else elp[:, :self.K]

    def expected_log_init(self) -> torch.Tensor:
        elp = torch.digamma(self.start_theta) - torch.digamma(self.start_theta.sum())
        return elp[:self.K]

    def _neg_elbo(self, rho: torch.Tensor, omega: torch.Tensor,
                  sumLogPi: torch.Tensor, startAlphaLogPi: torch.Tensor) -> torch.Tensor:
        K = self.K
        g1 = rho * omega
        g0 = (1.0 - rho) * omega
        dig_omega = torch.digamma(omega)
        Elogu = torch.digamma(g1) - dig_omega
        Elog1mu = torch.digamma(g0) - dig_omega

        kv = kvec(K, device=rho.device, dtype=rho.dtype)
        import math
        if self.kappa > 0:
            ONcoef = (K + 1.0) - g1
            OFFcoef = K * kv + 1.0 + self.gamma - g0
            Tvec = self.alpha * sumLogPi + startAlphaLogPi
            Tvec = Tvec.clone()
            Tvec[:-1] = Tvec[:-1] + (math.log(self.alpha + self.kappa) - math.log(self.kappa))
        else:
            ONcoef = (K + 1.0) + 1.0 - g1
            OFFcoef = (K + 1.0) * kv + self.gamma - g0
            Tvec = self.alpha * sumLogPi + startAlphaLogPi
        Ebeta = _rho2beta_safe(rho)
        elbo_local = (Ebeta * Tvec).sum()

        elbo = (-c_Beta(g1, g0)
                + (ONcoef * Elogu).sum()
                + (OFFcoef * Elog1mu).sum()
                + elbo_local)
        return -elbo

    def optimize_rho_omega(self, sumLogPi, startAlphaLogPi, n_iter: int = 200,
                           lr: float = 0.3):
        n_iter = int(min(n_iter, getattr(self, 'lbfgs_max_iter', n_iter)))
        dtype, device = self.rho.dtype, self.rho.device
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

    @torch.no_grad()
    def update(self, trans_counts: torch.Tensor, start_counts: torch.Tensor,
               n_global_iters: int = 3):
        trans_counts = trans_counts.to(dtype=self._dtype, device=self.rho.device)
        start_counts = start_counts.to(dtype=self._dtype, device=self.rho.device)
        tt, st = self._calc_theta(trans_counts, start_counts)
        if not (bool(torch.isfinite(tt).all()) and bool(torch.isfinite(st).all())):
            self.n_guard_rejects = getattr(self, "n_guard_rejects", 0) + 1
            return
        self.trans_theta.copy_(tt)
        self.start_theta.copy_(st)
        for _ in range(n_global_iters):
            elp = self.expected_log_pi(self.trans_theta)
            sumLogPi = elp.sum(0)
            startELogPi = (torch.digamma(self.start_theta)
                           - torch.digamma(self.start_theta.sum()))
            startAlphaLogPi = self.start_alpha * startELogPi
            rho0 = self.rho.clone()
            omega0 = self.omega.clone()
            obj0 = float(self.alloc_elbo())
            try:
                with torch.enable_grad():
                    rho, omega = self.optimize_rho_omega(sumLogPi, startAlphaLogPi)
            except Exception:
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
                self.n_guard_rejects = getattr(self, "n_guard_rejects", 0) + 1
                break
            self.rho.copy_(rho)
            self.omega.copy_(omega)
            tt, st = self._calc_theta(trans_counts, start_counts)
            if not (bool(torch.isfinite(tt).all()) and bool(torch.isfinite(st).all())):
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
                self.rho.copy_(rho0)
                self.omega.copy_(omega0)
                tt, st = self._calc_theta(trans_counts, start_counts)
                self.trans_theta.copy_(tt)
                self.start_theta.copy_(st)
                self.n_guard_rejects = getattr(self, 'n_guard_rejects', 0) + 1
                break

    @torch.no_grad()
    def root_kl(self) -> torch.Tensor:
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
        Ltop = L_top(self.rho, self.omega, self.alpha, self.gamma,
                     self.kappa, self.start_alpha)
        return Ltop - c_Dir(self.trans_theta) - c_Dir(self.start_theta)

    @torch.no_grad()
    def exact_alloc_elbo(self) -> torch.Tensor:
        K = self.K
        Ebeta = _rho2beta_safe(self.rho)
        prior_trans = self.alpha * Ebeta.unsqueeze(0).repeat(K, 1)
        prior_trans[:, :K] = prior_trans[:, :K] + self.kappa * torch.eye(
            K, dtype=self._dtype, device=self.rho.device)
        P = self.expected_log_pi(self.trans_theta)
        slack_trans = ((prior_trans - self.trans_theta) * P).sum()

        prior_start = self.start_alpha * Ebeta
        P0 = torch.digamma(self.start_theta) - torch.digamma(self.start_theta.sum())
        slack_start = ((prior_start - self.start_theta) * P0).sum()

        Ltop = L_top(self.rho, self.omega, self.alpha, self.gamma,
                     self.kappa, self.start_alpha)
        return (Ltop - c_Dir(self.trans_theta) - c_Dir(self.start_theta)
                + slack_trans + slack_start)

    @torch.no_grad()
    def seed_rho_omega(self, rho_new: torch.Tensor, omega_new: torch.Tensor):
        self.rho.copy_(rho_new.to(self.rho.dtype).clamp(EPS, 1 - EPS))
        self.omega.copy_(omega_new.to(self.omega.dtype).clamp_min(1e-3))
        return self

    def resized_like(self, new_K: int):
        return StickyHDP(K=new_K, gamma=self.gamma, alpha=self.alpha, kappa=self.kappa,
                         start_alpha=self.start_alpha, dtype=self._dtype,
                         device=self.rho.device)

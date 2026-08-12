from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn

LOG2PI = math.log(2.0 * math.pi)


def _chol_jitter(mat, tries=4):
    base = mat.diagonal(dim1=-2, dim2=-1).abs().mean().clamp_min(1e-12)
    for j in range(tries):
        try:
            m = mat if j == 0 else mat + (10.0 ** (j - 8)) * base * torch.eye(
                mat.shape[-1], dtype=mat.dtype, device=mat.device)
            return torch.linalg.cholesky(m)
        except Exception:
            if j == tries - 1:
                raise


class DiagARRegimes(nn.Module):
    def __init__(
        self,
        K: int,
        L: int,
        G: int,
        action_dim: int = 0,
        a0: float = 3.0,
        b0: float = 2.0,
        learn_b0: bool = False,
        b0_c0: float = 2.0,
        v0_scale: float = 1.0,
        ard: bool = True,
        identity_init: bool = True,
        jitter: float = 1e-6,
        q_rank: int = 0,
        infl_max: float = 10.0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float32,
    ):
        self.action_dim = int(action_dim)
        super().__init__()
        self.K, self.L, self.G = K, L, G
        self.ard = ard
        self.noise_var_floor = 1e-6
        import os as _os
        self.precision_floor_frac = float(_os.environ.get("REG_PREC_FLOOR", "0.0"))
        self.warmup_laps = int(_os.environ.get("REG_WARMUP_LAPS", "0"))
        self.reseed_empty = _os.environ.get("REG_RESEED_EMPTY", "0") == "1"
        self.empty_thresh = float(_os.environ.get("REG_EMPTY_THRESH", "1e-3"))
        self._mstep_count = 0
        self.jitter = jitter
        self.infl_max = float(infl_max)

        self.register_buffer("a0", torch.tensor(float(a0), dtype=dtype, device=device))
        self.register_buffer("b0", torch.full((L,), float(b0), dtype=dtype, device=device))
        self.learn_b0 = bool(learn_b0)
        self.register_buffer("b0_c0", torch.tensor(float(b0_c0), dtype=dtype, device=device))
        self.register_buffer("b0_d0", torch.tensor(float(b0_c0) / float(b0), dtype=dtype, device=device))
        self.register_buffer("b0_chat", torch.full((L,), float(b0_c0), dtype=dtype, device=device))
        self.register_buffer("b0_dhat", torch.full((L,), float(b0_c0) / float(b0), dtype=dtype, device=device))
        self.register_buffer("lam0_diag", torch.full((G,), float(v0_scale), dtype=dtype, device=device))

        M0 = torch.zeros(L, G, dtype=dtype, device=device)
        if identity_init and G >= L:
            M0[:, :L] = torch.eye(L, dtype=dtype, device=device)
        self.register_buffer("M0", M0)

        self.register_buffer("M", M0.clone().unsqueeze(0).repeat(K, 1, 1))
        self.register_buffer("lam", torch.diag_embed(self.lam0_diag).unsqueeze(0).repeat(K, 1, 1))
        self.register_buffer("a", torch.full((K, L), float(a0), dtype=dtype, device=device))
        self.register_buffer("b", torch.full((K, L), float(b0), dtype=dtype, device=device))
        self.register_buffer("V", torch.diag_embed(1.0 / self.lam0_diag).unsqueeze(0).repeat(K, 1, 1))

        self.register_buffer("N", torch.zeros(K, dtype=dtype, device=device))
        self.register_buffer("Sgg", torch.zeros(K, G, G, dtype=dtype, device=device))
        self.register_buffer("Szg", torch.zeros(K, L, G, dtype=dtype, device=device))
        self.register_buffer("Szz", torch.zeros(K, L, dtype=dtype, device=device))
        self.q_rank = int(q_rank)
        if self.q_rank > 0:
            raise ValueError(
                "q_rank>0 on the non-shared path (DiagARRegimes) evaluated an inflated "
                "PREDICTIVE density instead of the ELBO likelihood and carried a "
                "point-estimated U with no posterior or KL. "
                "Use shared_carry=True: SharedCarryRegimes implements the fully "
                "variational factor-augmented low-rank noise (local f_t ~ N(0,I), "
                "Gaussian q(U) with prior and KL, exact conditional q(f)). "
                "Config knobs: shs_q_rank / shs_shared_carry.")
        if self.q_rank > 0:
            self.register_buffer("Szz_full", torch.zeros(K, L, L, dtype=dtype, device=device))
            self.register_buffer("Ufac", torch.zeros(K, L, self.q_rank, dtype=dtype, device=device))
            self.register_buffer("q_Ddiag", torch.full((K, L), float(b0), dtype=dtype, device=device))
        self._stats_initialised = False
        self._refresh_cache()

    def _refresh_cache(self):
        eye = torch.eye(self.G, device=self.lam.device, dtype=self.lam.dtype)
        lam = self.lam + self.jitter * eye
        chol = torch.linalg.cholesky(lam)
        self.V = torch.cholesky_inverse(chol)
        self._lam_chol = chol

    @property
    def Omega(self) -> torch.Tensor:
        return self.a / self.b

    def E_logdet_prec(self) -> torch.Tensor:
        return (torch.digamma(self.a) - torch.log(self.b)).sum(-1)

    def expected_loglik(self, z: torch.Tensor, g: torch.Tensor,
                        z_var: torch.Tensor = None, g_var: torch.Tensor = None,
                        diag_score: bool = False,
                        zg_xcov: torch.Tensor = None,
                        g_zcov: torch.Tensor = None,
                        z_cov: torch.Tensor = None) -> torch.Tensor:
        if zg_xcov is not None:
            zg_xcov = zg_xcov.to(self.M.dtype)
        if g_zcov is not None:
            g_zcov = g_zcov.to(self.M.dtype)
        if z_cov is not None:
            z_cov = z_cov.to(self.M.dtype)
        if z_var is None and z_cov is not None:
            z_var = torch.diagonal(z_cov, dim1=-2, dim2=-1).clamp_min(0)
        if g_zcov is not None and g_var is not None:
            g_var = torch.cat([torch.zeros_like(g_var[..., :self.L]),
                               g_var[..., self.L:]], dim=-1)
        mu = torch.einsum("klg,...g->...kl", self.M, g)
        Vg = torch.einsum("kgh,...h->...kg", self.V, g)
        gVg = torch.einsum("...g,...kg->...k", g, Vg)
        if g_zcov is not None:
            gVg = gVg + torch.einsum("kjl,...lj->...k", self.V[:, :self.L, :self.L], g_zcov)
        if g_var is not None:
            Vdiag = torch.diagonal(self.V, dim1=-2, dim2=-1)
            gVg = gVg + torch.einsum("kg,...g->...k", Vdiag, g_var)
        if self.q_rank > 0:
            infl = (1.0 + gVg).clamp(max=self.infl_max)
            if diag_score:
                var = infl.unsqueeze(-1) * (self.q_Ddiag + (self.Ufac ** 2).sum(-1))
                mean_resid2 = (z.unsqueeze(-2) - mu) ** 2
                if g_var is not None:
                    M2gvar = torch.einsum("klg,...g->...kl", self.M ** 2, g_var)
                    mean_resid2 = mean_resid2 + M2gvar
                if g_zcov is not None:
                    A_ = self.M[..., :self.L]
                    mean_resid2 = mean_resid2 + torch.einsum(
                        "kij,...jl,kil->...ki", A_, g_zcov, A_)
                if zg_xcov is not None:
                    A_ = self.M[..., :self.L]
                    mean_resid2 = (mean_resid2 - 2.0 * torch.einsum(
                        "kij,...ij->...ki", A_, zg_xcov)).clamp_min(0.0)
                quad = (mean_resid2 / var).sum(-1)
                if z_var is not None:
                    quad = quad + (z_var.unsqueeze(-2) / var).sum(-1)
                return -0.5 * (self.L * LOG2PI + torch.log(var).sum(-1) + quad)
            from .lowrank import lowrank_logpdf, lowrank_inv_diag, lowrank_quadform_cols
            d = infl.unsqueeze(-1) * self.q_Ddiag
            U = infl.clamp(min=1e-6).sqrt().unsqueeze(-1).unsqueeze(-1) * self.Ufac
            zc = z.unsqueeze(-2).expand_as(mu)
            ll = lowrank_logpdf(zc, mu, d, U)
            need_Q0inv = (z_cov is not None or g_zcov is not None
                          or zg_xcov is not None)
            if need_Q0inv:
                eyeL = torch.eye(self.L, dtype=self.M.dtype, device=self.M.device)
                Q0 = (torch.diag_embed(self.q_Ddiag)
                      + torch.einsum("kif,kjf->kij", self.Ufac, self.Ufac)
                      + 1e-8 * eyeL)
                Q0inv = torch.cholesky_inverse(torch.linalg.cholesky(Q0))
                A_ = self.M[..., :self.L]
            inv_diag = None
            if z_var is not None or g_var is not None:
                inv_diag = lowrank_inv_diag(d, U)
            if z_cov is not None:
                tr = torch.einsum("kij,...ji->...k", Q0inv, z_cov) / infl
                ll = ll - 0.5 * tr
            elif z_var is not None:
                tr = (inv_diag * z_var.unsqueeze(-2)).sum(-1)
                ll = ll - 0.5 * tr
            if g_var is not None:
                colQ = lowrank_quadform_cols(d, U, self.M)
                tr_g = torch.einsum("...g,...kg->...k", g_var, colQ)
                ll = ll - 0.5 * tr_g
            if g_zcov is not None:
                G_ = torch.einsum("kji,kjl,klm->kim", A_, Q0inv, A_)
                ll = ll - 0.5 * torch.einsum("kim,...mi->...k", G_, g_zcov) / infl
            if zg_xcov is not None:
                H_ = torch.einsum("kij,kjl->kil", Q0inv, A_)
                ll = ll + torch.einsum("kij,...ij->...k", H_, zg_xcov) / infl
            return ll
        resid2 = (z.unsqueeze(-2) - mu) ** 2
        if g_var is not None:
            resid2 = resid2 + torch.einsum("klg,...g->...kl", self.M ** 2, g_var)
        if g_zcov is not None:
            A_ = self.M[..., :self.L]
            resid2 = resid2 + torch.einsum("kij,...jl,kil->...ki", A_, g_zcov, A_)
        if zg_xcov is not None:
            A_ = self.M[..., :self.L]
            resid2 = resid2 - 2.0 * torch.einsum("kij,...ij->...ki", A_, zg_xcov)
            resid2 = resid2.clamp_min(0.0)
        prec = self.Omega
        quad = (prec * resid2).sum(-1)
        if z_var is not None:
            quad = quad + (prec * z_var.unsqueeze(-2)).sum(-1)
        elogdet = self.E_logdet_prec()
        out = (
            -0.5 * self.L * LOG2PI
            + 0.5 * elogdet
            - 0.5 * quad
            - 0.5 * self.L * gVg
        )
        return out

    def predictive(self, g: torch.Tensor):
        mean = torch.einsum("klg,...g->...kl", self.M, g)
        Vg = torch.einsum("kgh,...h->...kg", self.V, g)
        gVg = torch.einsum("...g,...kg->...k", g, Vg)
        if self.q_rank > 0:
            marg = self.q_Ddiag + (self.Ufac ** 2).sum(-1)
            var = (1.0 + gVg).clamp(max=self.infl_max).unsqueeze(-1) * marg
            return mean, var
        Eq = self.b / torch.clamp(self.a - 1.0, min=1e-4)
        var = (1.0 + gVg).clamp(max=self.infl_max).unsqueeze(-1) * Eq
        return mean, var

    def predictive_moments(self, g: torch.Tensor, g_var: torch.Tensor = None):
        mean, var = self.predictive(g)
        if g_var is None:
            return mean, var
        Vdiag = torch.diagonal(self.V, dim1=-2, dim2=-1)
        extra_gVg = torch.einsum("kg,...g->...k", Vdiag, g_var)
        if self.q_rank > 0:
            marg = self.q_Ddiag + (self.Ufac ** 2).sum(-1)
            var = var + extra_gVg.unsqueeze(-1) * marg
        else:
            Eq = self.b / torch.clamp(self.a - 1.0, min=1e-4)
            var = var + extra_gVg.unsqueeze(-1) * Eq
        var = var + torch.einsum("klg,...g->...kl", self.M ** 2, g_var)
        return mean, var.clamp_min(1e-8)

    def predictive_cov_moments(self, g: torch.Tensor, g_var: torch.Tensor = None):
        mean, d, U = self.predictive_cov(g)
        if g_var is None:
            return mean, d, U
        Vdiag = torch.diagonal(self.V, dim1=-2, dim2=-1)
        extra_gVg = torch.einsum("kg,...g->...k", Vdiag, g_var)
        if self.q_rank > 0:
            d = d + extra_gVg.unsqueeze(-1) * self.q_Ddiag
            U_extra = torch.einsum("klg,...g->...klg", self.M, g_var.clamp_min(0.0).sqrt())
            U = torch.cat([U, U_extra], dim=-1)
        else:
            Eq = self.b / torch.clamp(self.a - 1.0, min=1e-4)
            d = d + extra_gVg.unsqueeze(-1) * Eq
            d = d + torch.einsum("klg,...g->...kl", self.M ** 2, g_var)
        return mean, d.clamp_min(1e-8), U

    def predictive_cov(self, g: torch.Tensor):
        mean = torch.einsum("klg,...g->...kl", self.M, g)
        Vg = torch.einsum("kgh,...h->...kg", self.V, g)
        gVg = torch.einsum("...g,...kg->...k", g, Vg)
        infl = (1.0 + gVg).clamp(min=1e-6, max=self.infl_max)
        if self.q_rank > 0:
            d = infl.unsqueeze(-1) * self.q_Ddiag
            U = infl.sqrt().unsqueeze(-1).unsqueeze(-1) * self.Ufac
        else:
            Eq = self.b / torch.clamp(self.a - 1.0, min=1e-4)
            d = infl.unsqueeze(-1) * Eq
            U = d.new_zeros(d.shape + (0,))
        return mean, d, U

    @torch.no_grad()
    def log_marginal_from_stats(self, N, Sgg, Szg, Szz):
        batched = (Szz.dim() == 2)
        if not batched:
            N = N.reshape(1) if torch.is_tensor(N) else torch.as_tensor([float(N)])
            Sgg, Szg, Szz = Sgg.unsqueeze(0), Szg.unsqueeze(0), Szz.unsqueeze(0)
        N = N.to(self.M0.dtype).reshape(-1)
        Sgg, Szg, Szz = (t.to(self.M0.dtype) for t in (Sgg, Szg, Szz))

        G, L = self.G, self.L
        lam0 = torch.diag(self.lam0_diag)
        eyeG = torch.eye(G, dtype=lam0.dtype, device=lam0.device)
        lam = lam0.unsqueeze(0) + Sgg + self.jitter * eyeG

        rhs = (self.M0 @ lam0).unsqueeze(0) + Szg
        chol = torch.linalg.cholesky(lam)
        V = torch.cholesky_inverse(chol)
        MN = torch.einsum("blg,bgh->blh", rhs, V)

        logdet_lam = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        logdet_lam0 = torch.log(self.lam0_diag).sum()

        m0lam0m0 = torch.einsum("ig,gh,ih->i", self.M0, lam0, self.M0)
        lamM = torch.einsum("bgh,blh->blg", lam, MN)
        MlamM = torch.einsum("blg,blg->bl", MN, lamM)

        b0 = self._b0_rate().reshape(1, L)
        a0 = self.a0
        aN = a0 + 0.5 * N.reshape(-1, 1)
        bN = b0 + 0.5 * (Szz + m0lam0m0.reshape(1, L) - MlamM)
        bN = torch.clamp(bN, min=1e-12)

        out = (-0.5 * N.reshape(-1, 1) * float(np.log(2.0 * np.pi))
               + 0.5 * (logdet_lam0 - logdet_lam).reshape(-1, 1)
               + a0 * torch.log(b0) - aN * torch.log(bN)
               + torch.lgamma(aN) - torch.lgamma(torch.as_tensor(
                   a0, dtype=bN.dtype, device=bN.device)))
        out = out.sum(-1)
        return out if batched else out[0]

    @torch.no_grad()
    def calibrate_b0_from_data(self, z, sF: float = 0.1, valid=None):
        """Scale the Gamma rate prior to the data, as bnpy does (b0 = sF * Cov(x)).

        b0 is a fixed rate added to the residual sum of squares:
            E[sigma^2_k] = (b0 + SSR_k/2) / (a0 + N_k/2)
        so its effect is relative to SSR. On ToyARK13 (near-noiseless AR,
        SSR ~ 0.5-10) the default b0=2.0 dominates, inflating a small state's
        variance ~7x versus ~2.8x for a large one, which flattens small states
        out of the E-step. Measured on ToyARK13 N=24, K=25, flat prior:
            b0=2.0   -> occupied 8,  hamming 0.309
            b0=1e-3  -> occupied 9,  hamming 0.251
            plain-EM AR-HMM ref.      10,          0.297
        Call once after construction, before fitting.
        """
        x = z.reshape(-1, z.shape[-1]).to(self.b0.dtype)
        if valid is not None:
            m = valid.reshape(-1).to(torch.bool)
            if m.numel() == x.shape[0]:
                x = x[m]
        var = x.var(dim=0, unbiased=False).clamp_min(1e-8)
        self.b0.copy_(float(sF) * var)
        self.b.copy_(self.b0.unsqueeze(0).expand_as(self.b).clone())
        return self.b0

    def stats_from_batch(self, resp: torch.Tensor, z: torch.Tensor, g: torch.Tensor,
                         z_var: torch.Tensor = None, g_z_var: torch.Tensor = None,
                         zg_xcov: torch.Tensor = None, g_zcov: torch.Tensor = None,
                         z_cov: torch.Tensor = None):
        if zg_xcov is not None:
            zg_xcov = zg_xcov.to(self.M.dtype)
        if g_zcov is not None:
            g_zcov = g_zcov.to(self.M.dtype)
        Kr = resp.shape[-1]
        r = resp.reshape(-1, Kr)
        zf = z.reshape(-1, self.L)
        gf = g.reshape(-1, self.G)
        N = r.sum(0)
        Sgg = torch.einsum("mk,mg,mh->kgh", r, gf, gf)
        Szg = torch.einsum("mk,mi,mg->kig", r, zf, gf)
        Szz = torch.einsum("mk,mi->ki", r, zf * zf)
        out = dict(N=N, Sgg=Sgg, Szg=Szg, Szz=Szz)
        if self.q_rank > 0:
            out["Szz_full"] = torch.einsum("mk,mi,mj->kij", r, zf, zf)
        if z_var is not None:
            vf = z_var.reshape(-1, self.L)
            rv = torch.einsum("mk,mi->ki", r, vf)
            out["Szz"] = out["Szz"] + rv
            if self.q_rank > 0:
                out["Szz_full"] = out["Szz_full"] + torch.diag_embed(rv)
        if g_z_var is not None and g_zcov is None:
            gv = g_z_var.reshape(-1, self.L)
            rgv = torch.einsum("mk,mi->ki", r, gv)
            out["Sgg"][:, :self.L, :self.L] = (
                out["Sgg"][:, :self.L, :self.L] + torch.diag_embed(rgv))
        if g_zcov is not None:
            gc = g_zcov.reshape(-1, self.L, self.L)
            out["Sgg"][:, :self.L, :self.L] = (
                out["Sgg"][:, :self.L, :self.L]
                + torch.einsum("mk,mij->kij", r, gc))
        if zg_xcov is not None:
            xc = zg_xcov.reshape(-1, self.L, self.L)
            out["Szg"][:, :, :self.L] = (
                out["Szg"][:, :, :self.L] + torch.einsum("mk,mij->kij", r, xc))
        if z_cov is not None and self.q_rank > 0:
            zc = z_cov.reshape(-1, self.L, self.L)
            off = zc - torch.diag_embed(torch.diagonal(zc, dim1=-2, dim2=-1))
            out["Szz_full"] = out["Szz_full"] + torch.einsum("mk,mij->kij", r, off)
        return out

    def set_stats(self, stats):
        self.N.copy_(stats["N"])
        self.Sgg.copy_(stats["Sgg"])
        self.Szg.copy_(stats["Szg"])
        self.Szz.copy_(stats["Szz"])
        if self.q_rank > 0:
            self.Szz_full.copy_(stats["Szz_full"])
        self._stats_initialised = True

    def ema_update_stats(self, stats, tau: float):
        if not self._stats_initialised:
            self.set_stats(stats)
            return
        pairs = [("N", self.N), ("Sgg", self.Sgg), ("Szg", self.Szg), ("Szz", self.Szz)]
        if self.q_rank > 0:
            pairs.append(("Szz_full", self.Szz_full))
        for name, S in pairs:
            S.mul_(1.0 - tau).add_(tau * stats[name])

    @torch.no_grad()
    def calibrate_b0_from_data(self, z, g=None, sF: float = 0.1, valid=None):
        """Scale the Gamma rate prior to the data, as bnpy does (b0 = sF * Cov).

        A fixed b0 is not scale free. On near-noiseless data (ToyARK13: per-dim
        SSR ~ 0.5-10) the default b0=2 dominates the residual and inflates the
        posterior variance of SMALL states far more than large ones
        (N=200 -> ~7x, N=3000 -> ~2.8x), flattening them out of the E-step.
        Measured effect on ToyARK13, K=25, N=24:

            b0=2.0   -> occupied 8,  hamming 0.309
            b0=0.001 -> occupied 9,  hamming 0.251     (reference: 10 / 0.297)

        b0 = sF * Var(z) per output dim reproduces the ML-limit behaviour while
        staying a proper prior. bnpy's convention is `sF` with `ECovMat eye`;
        its ToyARK13 settings use sF 0.1.
        """
        zz = z.reshape(-1, self.L) if z.dim() > 2 else z
        if valid is not None:
            m = valid.reshape(-1).to(torch.bool)
            zz = zz[m]
        var = zz.to(self.b0_chat.dtype).var(dim=0, unbiased=False).clamp_min(1e-8)
        b0_new = (float(sF) * var).clamp_min(1e-8)
        self.b0_dhat.copy_(self.b0_chat / b0_new)
        self.b0_d0.copy_((self.b0_chat / b0_new).mean())
        # `_b0_rate` reads self.b0 unless learn_b0 is set, so write there too.
        # self.b0 may be scalar (shared) or per-dim; handle both.
        if self.b0.dim() == 0:
            self.b0 = b0_new.mean().clone()
        else:
            self.b0 = b0_new.clone()
        return b0_new

    @torch.no_grad()
    def m_step(self):
        eyeG = torch.eye(self.G, device=self.M.device, dtype=self.M.dtype)
        lam0 = torch.diag(self.lam0_diag)

        lam = lam0.unsqueeze(0) + self.Sgg
        lam = lam + self.jitter * eyeG
        chol = _chol_jitter(lam)
        V = torch.cholesky_inverse(chol)

        rhs = (self.M0 @ lam0).unsqueeze(0) + self.Szg
        M = torch.einsum("klg,kgh->klh", rhs, V)

        a = self.a0 + 0.5 * self.N.unsqueeze(-1)
        a = a.expand(self.K, self.L).clone()

        m0lam0m0 = torch.einsum("ig,gh,ih->i", self.M0, lam0, self.M0)
        lamM = torch.einsum("kgh,klh->klg", lam, M)
        MlamM = torch.einsum("klg,klg->kl", M, lamM)
        b = self._b0_rate().unsqueeze(0) + 0.5 * (self.Szz + m0lam0m0.unsqueeze(0) - MlamM)
        b = torch.clamp(b, min=1e-6)

        self.M.copy_(M)
        self.lam.copy_(lam)
        b = torch.maximum(b, a * self.noise_var_floor)

        frac = float(getattr(self, "precision_floor_frac", 0.0))
        if frac > 0.0:
            w = self.N.clamp_min(0.0).unsqueeze(-1)
            var_k = (b / a.clamp_min(1e-12))
            pooled = (w * var_k).sum(0) / w.sum().clamp_min(1e-12)
            b = torch.maximum(b, a * (frac * pooled).unsqueeze(0))

        wl = int(getattr(self, "warmup_laps", 0))
        if wl > 0 and int(getattr(self, "_mstep_count", 0)) < wl:
            t = (int(getattr(self, "_mstep_count", 0)) + 1) / float(wl)
            a = self.a0 + t * (a - self.a0)
            b = self._b0_rate().unsqueeze(0) + t * (b - self._b0_rate().unsqueeze(0))
            b = torch.clamp(b, min=1e-6)
        self._mstep_count = int(getattr(self, "_mstep_count", 0)) + 1

        if bool(getattr(self, "reseed_empty", False)):
            Nk = self.N.clamp_min(0.0)
            empty = Nk <= float(getattr(self, "empty_thresh", 1e-3))
            if bool(empty.any()) and bool((~empty).any()):
                w = Nk[~empty].unsqueeze(-1)
                a_live = (w * a[~empty]).sum(0) / w.sum().clamp_min(1e-12)
                b_live = (w * b[~empty]).sum(0) / w.sum().clamp_min(1e-12)
                a = a.clone(); b = b.clone()
                a[empty] = a_live
                b[empty] = b_live

        self.a.copy_(a)
        self.b.copy_(b)
        self.V.copy_(V)
        self._lam_chol = chol

        if self.ard:
            self._ard_step()
        if self.q_rank > 0:
            self._fit_lowrank_Q(M)
        if getattr(self, "learn_b0", False):
            self.b0_chat.copy_(self.b0_c0 + float(self.K) * self.a0)
            self.b0_dhat.copy_(self.b0_d0 + (self.a / self.b).sum(0))

    @torch.no_grad()
    def _fit_lowrank_Q(self, M):
        N = self.N.clamp(min=1.0)
        SzgMt = torch.einsum("klg,kmg->klm", self.Szg, M)
        MSgg = torch.einsum("klg,kgh->klh", M, self.Sgg)
        MSggMt = torch.einsum("klh,kmh->klm", MSgg, M)
        S = (self.Szz_full - SzgMt - SzgMt.transpose(-1, -2) + MSggMt) / N.view(-1, 1, 1)
        eyeL = torch.eye(self.L, device=S.device, dtype=S.dtype)
        S = 0.5 * (S + S.transpose(-1, -2)) + self.jitter * eyeL
        evals, evecs = torch.linalg.eigh(S)
        top_val = evals[..., -self.q_rank:].clamp(min=0.0)
        top_vec = evecs[..., -self.q_rank:]
        U = top_vec * top_val.sqrt().unsqueeze(-2)
        d = torch.diagonal(S, dim1=-2, dim2=-1) - (U ** 2).sum(-1)
        d = d.clamp(min=1e-4)
        empty = (self.N <= 1e-6)
        if empty.any():
            U[empty] = 0.0
            d[empty] = float(self._b0_rate().mean())
        self.Ufac.copy_(U)
        self.q_Ddiag.copy_(d)

    @torch.no_grad()
    def _ard_step(self):
        Vdiag = torch.diagonal(self.V, dim1=-2, dim2=-1)
        Om = self.Omega
        Mj2 = self.M ** 2
        weighted = torch.einsum("ki,kig->kg", Om, Mj2)
        occ = self.N > 1.0
        n_occ = int(occ.sum())
        if n_occ == 0:
            return
        denom = (self.L * Vdiag + weighted)[occ].sum(0)
        alpha = (n_occ * self.L) / torch.clamp(denom, min=1e-8)
        alpha = torch.clamp(alpha, min=1e-4, max=1e3)
        self.lam0_diag.copy_(alpha)

    @torch.no_grad()
    def param_kl(self) -> torch.Tensor:
        K, L, G = self.K, self.L, self.G
        lam0 = self.lam0_diag
        chol = self._lam_chol
        logdet_lam = 2.0 * torch.log(
            torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        logdet_lam0 = torch.log(lam0).sum()
        Vdiag = torch.diagonal(self.V, dim1=-2, dim2=-1)
        tr_term = (lam0.unsqueeze(0) * Vdiag).sum(-1)

        diff = self.M - self.M0.unsqueeze(0)
        maha = (lam0.view(1, 1, G) * diff ** 2).sum(-1)
        Etau = self.a / self.b
        gauss_kl = (0.5 * (logdet_lam - logdet_lam0 - G + tr_term).unsqueeze(-1)
                    + 0.5 * Etau * maha)
        a_q, b_q = self.a, self.b
        a_p = self.a0
        if getattr(self, "learn_b0", False):
            Elogb0 = (torch.digamma(self.b0_chat)
                      - torch.log(self.b0_dhat)).unsqueeze(0)
            Eb0 = (self.b0_chat / self.b0_dhat).unsqueeze(0)
        else:
            Elogb0 = torch.log(self.b0).unsqueeze(0)
            Eb0 = self.b0.unsqueeze(0)
        gamma_kl = ((a_q - a_p) * torch.digamma(a_q)
                    - torch.lgamma(a_q) + torch.lgamma(a_p)
                    + a_p * (torch.log(b_q) - Elogb0)
                    + a_q * (Eb0 - b_q) / b_q)
        return (gauss_kl + gamma_kl).sum(-1)

    def _b0_rate(self):
        if getattr(self, "learn_b0", False):
            return self.b0_chat / self.b0_dhat.clamp_min(1e-12)
        return self.b0

    def hyper_kl(self) -> torch.Tensor:
        if not getattr(self, "learn_b0", False):
            return torch.zeros((), dtype=self.b.dtype, device=self.b.device)
        aq, bq = self.b0_chat, self.b0_dhat
        ap, bp = self.b0_c0, self.b0_d0
        return ((aq - ap) * torch.digamma(aq) - torch.lgamma(aq)
                + torch.lgamma(ap) + ap * (torch.log(bq) - torch.log(bp))
                + aq * (bp - bq) / bq).sum()

    @torch.no_grad()
    def data_elbo_from_stats(self) -> torch.Tensor:
        N = self.N.clamp_min(0.0)
        elog = torch.digamma(self.a) - torch.log(self.b)
        MSzg = (self.M * self.Szg).sum(-1)
        MSggM = torch.einsum("klg,kgh,klh->kl", self.M, self.Sgg, self.M)
        if self.q_rank > 0:
            var = self.q_Ddiag + (self.Ufac ** 2).sum(-1)
            quad = (self.Szz - 2.0 * MSzg + MSggM) / var
            per = (-0.5 * N.unsqueeze(-1) * LOG2PI
                   - 0.5 * N.unsqueeze(-1) * torch.log(var)
                   - 0.5 * quad)
            return per.sum()
        Etau = self.a / self.b
        per = (-0.5 * N.unsqueeze(-1) * LOG2PI
               + 0.5 * N.unsqueeze(-1) * elog
               - 0.5 * Etau * (self.Szz - 2.0 * MSzg + MSggM))
        Vdiag_tr = torch.einsum("kgh,khg->k", self.V, self.Sgg)
        return per.sum() - 0.5 * self.L * Vdiag_tr.sum()

    @torch.no_grad()
    def clone_with_K(self, new_K: int, stats=None):
        new = DiagARRegimes(
            K=new_K, L=self.L, G=self.G,
            a0=float(self.a0), b0=float(self.b0.mean()), v0_scale=1.0,
            ard=False, identity_init=False, jitter=self.jitter,
            q_rank=self.q_rank,
            learn_b0=getattr(self, "learn_b0", False),
            b0_c0=float(getattr(self, "b0_c0", torch.tensor(2.0))),
            device=self.M.device, dtype=self.M.dtype,
        )
        new.b0.copy_(self.b0)
        if getattr(self, "learn_b0", False):
            new.b0_chat.copy_(self.b0_chat)
            new.b0_dhat.copy_(self.b0_dhat)
        new.lam0_diag.copy_(self.lam0_diag)
        new.M0.copy_(self.M0)
        if stats is not None:
            new.set_stats(stats)
            new.m_step()
        return new

    @torch.no_grad()
    def fit_full_batch(self, resp, z, g, n_iter: int = 1):
        stats = self.stats_from_batch(resp, z, g)
        self.set_stats(stats)
        for _ in range(max(1, n_iter)):
            self.m_step()


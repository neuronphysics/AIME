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


class SharedCarryRegimes(nn.Module):
    def __init__(
        self,
        K: int,
        L: int,
        G: int,
        a0: float = 3.0,
        b0: float = 2.0,
        learn_b0: bool = False,
        b0_c0: float = 2.0,
        v0_scale: float = 1.0,
        vC0_scale: float = 1.0,
        ard: bool = True,
        identity_init: bool = True,
        jitter: float = 1e-6,
        q_rank: int = 0,
        action_dim: int = 0,
        infl_max: float = 10.0,
        n_block_iters: int = 3,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float32,
    ):
        super().__init__()
        self.K, self.L, self.G = K, L, G
        self.action_dim = int(action_dim)
        self.Hp = G - L - self.action_dim - 1
        assert self.Hp >= 0, "G must be >= L+1 (regressor [z; h~; 1])"
        self.Lr = L + self.action_dim + 1
        self.ard = ard
        self.noise_var_floor = 1e-6
        self.jitter = jitter
        self.q_rank = int(q_rank)
        self.infl_max = float(infl_max)
        self.n_block_iters = int(n_block_iters)

        self.register_buffer("a0", torch.tensor(float(a0), dtype=dtype, device=device))
        self.register_buffer("b0", torch.tensor(float(b0), dtype=dtype, device=device))
        self.learn_b0 = bool(learn_b0)
        self.register_buffer("b0_c0", torch.tensor(float(b0_c0), dtype=dtype, device=device))
        self.register_buffer("b0_d0", torch.tensor(float(b0_c0) / float(b0), dtype=dtype, device=device))
        self.register_buffer("b0_chat", torch.tensor(float(b0_c0), dtype=dtype, device=device))
        self.register_buffer("b0_dhat", torch.tensor(float(b0_c0) / float(b0), dtype=dtype, device=device))
        self.register_buffer("lam0_diag", torch.full((self.Lr,), float(v0_scale), dtype=dtype, device=device))
        M0 = torch.zeros(L, self.Lr, dtype=dtype, device=device)
        if identity_init:
            M0[:, :L] = torch.eye(L, dtype=dtype, device=device)
        self.register_buffer("M0", M0)

        self.register_buffer("M", M0.clone().unsqueeze(0).repeat(K, 1, 1))
        self.register_buffer("lam", torch.diag_embed(self.lam0_diag).unsqueeze(0).repeat(K, 1, 1))
        self.register_buffer("V", torch.diag_embed(1.0 / self.lam0_diag).unsqueeze(0).repeat(K, 1, 1))
        self.register_buffer("a", torch.full((K, L), float(a0), dtype=dtype, device=device))
        self.register_buffer("b", torch.full((K, L), float(b0), dtype=dtype, device=device))

        self.register_buffer("vC0", torch.tensor(float(vC0_scale), dtype=dtype, device=device))
        self.register_buffer("Cmean", torch.zeros(L, self.Hp, dtype=dtype, device=device))
        self.register_buffer("Ccov", (1.0 / float(vC0_scale)) *
                             torch.eye(self.Hp, dtype=dtype, device=device).unsqueeze(0).repeat(L, 1, 1))
        self.register_buffer("C0mean", torch.zeros(L, self.Hp, dtype=dtype, device=device))

        self.register_buffer("N", torch.zeros(K, dtype=dtype, device=device))
        self.register_buffer("Srr", torch.zeros(K, self.Lr, self.Lr, dtype=dtype, device=device))
        self.register_buffer("Szr", torch.zeros(K, L, self.Lr, dtype=dtype, device=device))
        self.register_buffer("Szz", torch.zeros(K, L, dtype=dtype, device=device))
        self.register_buffer("Shh", torch.zeros(K, self.Hp, self.Hp, dtype=dtype, device=device))
        self.register_buffer("Szh", torch.zeros(K, L, self.Hp, dtype=dtype, device=device))
        self.register_buffer("Srh", torch.zeros(K, self.Lr, self.Hp, dtype=dtype, device=device))

        self.register_buffer("Sgg", self.Srr.clone())
        self.register_buffer("Szg", self.Szr.clone())
        self.register_buffer("Szz_resid", self.Szz.clone())

        if self.q_rank > 0:
            self.register_buffer("Szz_full", torch.zeros(K, L, L, dtype=dtype, device=device))
            self.register_buffer("Szz_full_resid", torch.zeros(K, L, L, dtype=dtype, device=device))
            self.register_buffer("Ufac", torch.zeros(K, L, self.q_rank, dtype=dtype, device=device))
            self.register_buffer("q_Ddiag", torch.full((K, L), float(b0), dtype=dtype, device=device))
            self.register_buffer("_q_taudiag", torch.full((K, L), float(b0), dtype=dtype, device=device))
            self.register_buffer("_q_Udiag", torch.zeros((K, L), dtype=dtype, device=device))
            self.u_prior_scale = 0.5
            F = self.q_rank
            self.register_buffer("Umean", 0.01 * torch.randn(K, L, F, dtype=dtype, device=device))
            self.register_buffer("Ucov", (self.u_prior_scale
                                          * torch.eye(F, dtype=dtype, device=device)
                                          ).expand(K, L, F, F).contiguous().clone())
            self.register_buffer("Szf", torch.zeros(K, L, F, dtype=dtype, device=device))
            self.register_buffer("Sfr", torch.zeros(K, F, self.Lr, dtype=dtype, device=device))
            self.register_buffer("Sfh", torch.zeros(K, F, self.Hp, dtype=dtype, device=device))
            self.register_buffer("Sff", torch.zeros(K, F, F, dtype=dtype, device=device))

        self._freeze_C = False
        self._stats_initialised = False
        self._refresh_cache()

    @torch.no_grad()
    def log_marginal_from_stats(self, N, Sgg, Szg, Szz):
        L, Lr = self.L, self.Lr
        Sgg_r = Sgg[..., :Lr, :Lr] if Sgg.shape[-1] != Lr else Sgg
        Szg_r = Szg[..., :, :Lr] if Szg.shape[-1] != Lr else Szg
        batched = (Szz.dim() == 2)
        if not batched:
            N = N.reshape(1) if torch.is_tensor(N) else torch.as_tensor([float(N)])
            Sgg_r, Szg_r, Szz = Sgg_r.unsqueeze(0), Szg_r.unsqueeze(0), Szz.unsqueeze(0)
        N = N.to(self.lam0_diag.dtype).reshape(-1)
        Sgg_r, Szg_r, Szz = (t.to(self.lam0_diag.dtype) for t in (Sgg_r, Szg_r, Szz))
        lam0 = torch.diag(self.lam0_diag)
        eye = torch.eye(Lr, dtype=lam0.dtype, device=lam0.device)
        lam = lam0.unsqueeze(0) + Sgg_r + self.jitter * eye
        chol = torch.linalg.cholesky(lam)
        V = torch.cholesky_inverse(chol)
        MN = torch.einsum("blg,bgh->blh", Szg_r, V)
        logdet_lam = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        logdet_lam0 = torch.log(self.lam0_diag).sum()
        lamM = torch.einsum("bgh,blh->blg", lam, MN)
        MlamM = torch.einsum("blg,blg->bl", MN, lamM)
        _b0 = self._b0_rate() if hasattr(self, "_b0_rate") else self.b0
        b0 = _b0.expand(L).reshape(1, L) if _b0.dim() == 0 else _b0.reshape(1, L)
        a0 = float(self.a0)
        aN = a0 + 0.5 * N.reshape(-1, 1)
        bN = torch.clamp(b0 + 0.5 * (Szz - MlamM), min=1e-12)
        out = (-0.5 * N.reshape(-1, 1) * float(np.log(2.0 * np.pi))
               + 0.5 * (logdet_lam0 - logdet_lam).reshape(-1, 1)
               + a0 * torch.log(b0) - aN * torch.log(bN)
               + torch.lgamma(aN) - torch.lgamma(torch.as_tensor(
                   a0, dtype=bN.dtype, device=bN.device)))
        out = out.sum(-1)
        return out if batched else out[0]

    def _split_g(self, g):
        z = g[..., : self.L]
        htil = g[..., self.L : self.L + self.Hp]
        act = g[..., self.L + self.Hp : -1]
        one = g[..., -1:]
        r = torch.cat([z, act, one], dim=-1)
        return r, htil

    def _refresh_cache(self):
        eye = torch.eye(self.Lr, device=self.lam.device, dtype=self.lam.dtype)
        lam = self.lam + self.jitter * eye
        chol = torch.linalg.cholesky(lam)
        self.V = torch.cholesky_inverse(chol)
        self._lam_chol = chol

    @property
    def Omega(self) -> torch.Tensor:
        return self.a / self.b

    def E_logdet_prec(self) -> torch.Tensor:
        return (torch.digamma(self.a) - torch.log(self.b)).sum(-1)

    def _Ch(self, htil):
        return torch.einsum("ih,...h->...i", self.Cmean, htil)

    def _hVCh(self, htil):
        Vh = torch.einsum("ihj,...j->...ih", self.Ccov, htil)
        return torch.einsum("...h,...ih->...i", htil, Vh)

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
        r, htil = self._split_g(g)
        mu = torch.einsum("klr,...r->...kl", self.M, r) + self._Ch(htil).unsqueeze(-2)
        Vr = torch.einsum("krs,...s->...kr", self.V, r)
        rVr = torch.einsum("...r,...kr->...k", r, Vr)
        r_var = None
        if g_var is not None:
            r_var = torch.cat([g_var[..., : self.L],
                           torch.zeros_like(g[..., self.L + self.Hp :])], dim=-1)
            Vdiag = torch.diagonal(self.V, dim1=-2, dim2=-1)
            if g_zcov is not None:
                r_var = torch.cat([torch.zeros_like(r_var[..., : self.L]),
                                   r_var[..., self.L :]], dim=-1)
            rVr = rVr + torch.einsum("kr,...r->...k", Vdiag, r_var)
        if g_zcov is not None:
            rVr = rVr + torch.einsum("kjl,...lj->...k",
                                     self.V[:, : self.L, : self.L], g_zcov)
        hVCh = self._hVCh(htil)
        if self.q_rank > 0 and diag_score:
            var = self.q_Ddiag + (self.Ufac ** 2).sum(-1)
            mean_resid2 = (z.unsqueeze(-2) - mu) ** 2
            if r_var is not None:
                mean_resid2 = mean_resid2 + torch.einsum("klr,...r->...kl", self.M ** 2, r_var)
            quad = (mean_resid2 / var).sum(-1)
            if z_var is not None:
                quad = quad + (z_var.unsqueeze(-2) / var).sum(-1)
            quad = quad + (hVCh.unsqueeze(-2) / var).sum(-1)
            quad = quad + rVr * (self.q_Ddiag / var).sum(-1)
            return -0.5 * (self.L * LOG2PI + torch.log(var).sum(-1) + quad)
        prec = self.Omega
        resid2 = (z.unsqueeze(-2) - mu) ** 2
        if r_var is not None:
            resid2 = resid2 + torch.einsum("klr,...r->...kl", self.M ** 2, r_var)
        if g_zcov is not None:
            A_ = self.M[..., : self.L]
            resid2 = resid2 + torch.einsum("kij,...jl,kil->...ki", A_, g_zcov, A_)
        if zg_xcov is not None:
            A_ = self.M[..., : self.L]
            resid2 = (resid2 - 2.0 * torch.einsum("kij,...ij->...ki", A_, zg_xcov)
                      ).clamp_min(0.0)
        quad = (prec * resid2).sum(-1)
        if z_var is not None:
            quad = quad + (prec * z_var.unsqueeze(-2)).sum(-1)
        carry_fluc = torch.einsum("kl,...l->...k", prec, hVCh)
        elogdet = self.E_logdet_prec()
        out = (
            -0.5 * self.L * LOG2PI
            + 0.5 * elogdet
            - 0.5 * quad
            - 0.5 * self.L * rVr
            - 0.5 * carry_fluc
        )
        if self.q_rank > 0:
            F = self.q_rank
            EuuT = self.Ucov + torch.einsum("kif,kig->kifg", self.Umean, self.Umean)
            P = (torch.eye(F, dtype=prec.dtype, device=prec.device)
                 + torch.einsum("ki,kifg->kfg", prec, EuuT))
            cholP = _chol_jitter(P)
            Pinv = torch.cholesky_inverse(cholP)
            logdetP = 2.0 * torch.log(torch.diagonal(cholP, dim1=-2, dim2=-1)).sum(-1)
            Rm = z.unsqueeze(-2) - mu
            bf = torch.einsum("ki,kif,...ki->...kf", prec, self.Umean, Rm)
            quadf = torch.einsum("...kf,kfg,...kg->...k", bf, Pinv, bf)
            if (z_cov is not None or z_var is not None
                    or g_zcov is not None or zg_xcov is not None):
                A_ = self.M[..., : self.L]
                if z_cov is not None:
                    SR = z_cov.unsqueeze(-3).expand(*z_cov.shape[:-2], self.K,
                                                    self.L, self.L).clone()
                elif z_var is not None:
                    SR = torch.diag_embed(z_var).unsqueeze(-3).expand(
                        *z_var.shape[:-1], self.K, self.L, self.L).clone()
                else:
                    SR = torch.zeros(*Rm.shape[:-1], self.L, self.L,
                                     dtype=Rm.dtype, device=Rm.device)
                SR = SR.to(self.M.dtype)
                if g_zcov is not None:
                    SR = SR + torch.einsum("kij,...jl,kml->...kim",
                                           A_, g_zcov, A_)
                if zg_xcov is not None:
                    AX = torch.einsum("kij,...lj->...kli", A_, zg_xcov)
                    SR = SR - AX.transpose(-1, -2) - AX
                W = prec.unsqueeze(-1) * self.Umean
                WSW = torch.einsum("kif,...kij,kjg->...kfg", W, SR, W)
                quadf = quadf + torch.einsum("kfg,...kgf->...k", Pinv, WSW)
            out = out + 0.5 * (quadf - logdetP)
        return out

    def predictive(self, g: torch.Tensor):
        r, htil = self._split_g(g)
        mean = torch.einsum("klr,...r->...kl", self.M, r) + self._Ch(htil).unsqueeze(-2)
        Vr = torch.einsum("krs,...s->...kr", self.V, r)
        rVr = torch.einsum("...r,...kr->...k", r, Vr)
        infl = (1.0 + rVr).clamp(max=self.infl_max).unsqueeze(-1)
        carry_var = self._hVCh(htil).unsqueeze(-2)
        if self.q_rank > 0:
            factor = (self.Ufac ** 2).sum(-1)
            var = infl * self._q_taudiag + self._q_Udiag + factor + carry_var
        else:
            Eq = self.b / torch.clamp(self.a - 1.0, min=1e-4)
            var = infl * Eq + carry_var
        return mean, var.clamp_min(1e-8)

    def predictive_moments(self, g: torch.Tensor, g_var: torch.Tensor = None):
        mean, var = self.predictive(g)
        if g_var is None:
            return mean, var
        r_var = torch.cat([g_var[..., : self.L],
                           torch.zeros_like(g[..., self.L + self.Hp :])], dim=-1)
        Vdiag = torch.diagonal(self.V, dim1=-2, dim2=-1)
        extra_rVr = torch.einsum("kr,...r->...k", Vdiag, r_var)
        if self.q_rank > 0:
            var = var + extra_rVr.unsqueeze(-1) * self._q_taudiag
        else:
            Eq = self.b / torch.clamp(self.a - 1.0, min=1e-4)
            var = var + extra_rVr.unsqueeze(-1) * Eq
        var = var + torch.einsum("klr,...r->...kl", self.M ** 2, r_var)
        return mean, var.clamp_min(1e-8)

    def predictive_cov_moments(self, g: torch.Tensor, g_var: torch.Tensor = None):
        mean, d, U = self.predictive_cov(g)
        if g_var is None:
            return mean, d, U
        r_var = torch.cat([g_var[..., : self.L],
                           torch.zeros_like(g[..., self.L + self.Hp :])], dim=-1)
        Vdiag = torch.diagonal(self.V, dim1=-2, dim2=-1)
        extra_rVr = torch.einsum("kr,...r->...k", Vdiag, r_var)
        if self.q_rank > 0:
            d = d + extra_rVr.unsqueeze(-1) * self._q_taudiag
            U_extra = torch.einsum("klr,...r->...klr", self.M, r_var.clamp_min(0.0).sqrt())
            U = torch.cat([U, U_extra], dim=-1)
        else:
            Eq = self.b / torch.clamp(self.a - 1.0, min=1e-4)
            d = d + extra_rVr.unsqueeze(-1) * Eq
            d = d + torch.einsum("klr,...r->...kl", self.M ** 2, r_var)
        return mean, d.clamp_min(1e-8), U

    def predictive_cov(self, g: torch.Tensor):
        r, htil = self._split_g(g)
        mean = torch.einsum("klr,...r->...kl", self.M, r) + self._Ch(htil).unsqueeze(-2)
        Vr = torch.einsum("krs,...s->...kr", self.V, r)
        rVr = torch.einsum("...r,...kr->...k", r, Vr)
        infl = (1.0 + rVr).clamp(min=1e-6, max=self.infl_max)
        carry_var = self._hVCh(htil).unsqueeze(-2)
        if self.q_rank > 0:
            d = infl.unsqueeze(-1) * self._q_taudiag + self._q_Udiag + carry_var
            U = torch.ones_like(infl).unsqueeze(-1).unsqueeze(-1) * self.Ufac
        else:
            Eq = self.b / torch.clamp(self.a - 1.0, min=1e-4)
            d = infl.unsqueeze(-1) * Eq + carry_var
            U = d.new_zeros(d.shape + (0,))
        return mean, d.clamp_min(1e-8), U

    def stats_from_batch(self, resp: torch.Tensor, z: torch.Tensor, g: torch.Tensor,
                         z_var: torch.Tensor = None, g_z_var: torch.Tensor = None,
                         zg_xcov: torch.Tensor = None, g_zcov: torch.Tensor = None,
                         z_cov: torch.Tensor = None):
        if zg_xcov is not None:
            zg_xcov = zg_xcov.to(self.M.dtype)
        if g_zcov is not None:
            g_zcov = g_zcov.to(self.M.dtype)
        Kr = resp.shape[-1]
        rsp = resp.reshape(-1, Kr)
        zf = z.reshape(-1, self.L)
        gf = g.reshape(-1, self.G)
        rf = torch.cat([gf[:, : self.L],
                        gf[:, self.L + self.Hp :]], dim=-1)
        hf = gf[:, self.L : self.L + self.Hp]
        N = rsp.sum(0)
        Srr = torch.einsum("mk,mr,ms->krs", rsp, rf, rf)
        Szr = torch.einsum("mk,mi,mr->kir", rsp, zf, rf)
        Szz = torch.einsum("mk,mi->ki", rsp, zf * zf)
        Shh = torch.einsum("mk,mh,mg->khg", rsp, hf, hf)
        Szh = torch.einsum("mk,mi,mh->kih", rsp, zf, hf)
        Srh = torch.einsum("mk,mr,mh->krh", rsp, rf, hf)
        out = dict(N=N, Srr=Srr, Szr=Szr, Szz=Szz, Shh=Shh, Szh=Szh, Srh=Srh)
        if self.q_rank > 0:
            out["Szz_full"] = torch.einsum("mk,mi,mj->kij", rsp, zf, zf)
        if z_var is not None:
            vf = z_var.reshape(-1, self.L)
            rv = torch.einsum("mk,mi->ki", rsp, vf)
            out["Szz"] = out["Szz"] + rv
            if self.q_rank > 0:
                out["Szz_full"] = out["Szz_full"] + torch.diag_embed(rv)
        if self.q_rank > 0:
            Kf, F = self.K, self.q_rank
            Om = self.Omega
            EuuT = self.Ucov + torch.einsum("kif,kig->kifg", self.Umean, self.Umean)
            P = (torch.eye(F, dtype=zf.dtype, device=zf.device)
                 + torch.einsum("ki,kifg->kfg", Om, EuuT))
            Pinv = torch.cholesky_inverse(_chol_jitter(P))
            Ch = hf @ self.Cmean.transpose(0, 1)
            Mr = torch.einsum("klr,mr->mkl", self.M, rf)
            Rm = zf.unsqueeze(1) - Mr - Ch.unsqueeze(1)
            bf = torch.einsum("ki,kif,mki->mkf", Om, self.Umean, Rm)
            m_f = torch.einsum("kfg,mkg->mkf", Pinv, bf)
            rspK = rsp[:, :Kf]
            Szf = torch.einsum("mk,mi,mkf->kif", rspK, zf, m_f)
            Sfr = torch.einsum("mk,mkf,mr->kfr", rspK, m_f, rf)
            Sfh = torch.einsum("mk,mkf,mh->kfh", rspK, m_f, hf)
            Sff = (torch.einsum("mk,mkf,mkg->kfg", rspK, m_f, m_f)
                   + rspK.sum(0).view(-1, 1, 1) * Pinv)
            _any_mom = (z_var is not None or z_cov is not None or zg_xcov is not None
                        or g_zcov is not None or g_z_var is not None)
            if _any_mom:
                Mtot = zf.shape[0]
                A_ = self.M[..., : self.L]
                if z_cov is not None:
                    Czz = z_cov.reshape(-1, self.L, self.L)
                elif z_var is not None:
                    Czz = torch.diag_embed(z_var.reshape(-1, self.L))
                else:
                    Czz = zf.new_zeros(Mtot, self.L, self.L)
                if g_zcov is not None:
                    Cpp = g_zcov.reshape(-1, self.L, self.L)
                elif g_z_var is not None:
                    Cpp = torch.diag_embed(g_z_var.reshape(-1, self.L))
                else:
                    Cpp = zf.new_zeros(Mtot, self.L, self.L)
                Cxp = (zg_xcov.reshape(-1, self.L, self.L) if zg_xcov is not None
                       else zf.new_zeros(Mtot, self.L, self.L))
                W = Om.unsqueeze(-1) * self.Umean
                WP = torch.einsum("klf,kfg->klg", W, Pinv)
                CxpAT = torch.einsum("mij,klj->mkil", Cxp, A_)
                ACpp = torch.einsum("kij,mjl->mkil", A_, Cpp)
                ACppAT = torch.einsum("mkil,kjl->mkij", ACpp, A_)
                S_R = (Czz.unsqueeze(1) + ACppAT - CxpAT
                       - CxpAT.transpose(-1, -2))
                CzR = Czz.unsqueeze(1) - CxpAT
                CRzp = Cxp.unsqueeze(1) - ACpp
                Szf = Szf + torch.einsum("mk,mkij,kjf->kif", rspK, CzR, WP)
                tmp = torch.einsum("mk,kjf,mkjl->kfl", rspK, W, CRzp)
                Sfr = Sfr.clone()
                Sfr[:, :, : self.L] = Sfr[:, :, : self.L] + torch.einsum(
                    "kfg,kgl->kfl", Pinv, tmp)
                inner = torch.einsum("mk,kif,mkij,kjg->kfg", rspK, W, S_R, W)
                Sff = Sff + torch.einsum("kfa,kab,kbg->kfg", Pinv, inner, Pinv)
            if Kr > Kf:
                def _pad(t):
                    return torch.cat([t, torch.zeros((Kr - Kf,) + t.shape[1:],
                                                     dtype=t.dtype, device=t.device)], 0)
                Szf, Sfr, Sfh, Sff = _pad(Szf), _pad(Sfr), _pad(Sfh), _pad(Sff)
            out.update(Szf=Szf, Sfr=Sfr, Sfh=Sfh, Sff=Sff)
        if g_z_var is not None and g_zcov is None:
            gv = g_z_var.reshape(-1, self.L)
            rgv = torch.einsum("mk,mi->ki", rsp, gv)
            out["Srr"][:, : self.L, : self.L] = (
                out["Srr"][:, : self.L, : self.L] + torch.diag_embed(rgv))
        if g_zcov is not None:
            gc = g_zcov.reshape(-1, self.L, self.L)
            out["Srr"][:, : self.L, : self.L] = (
                out["Srr"][:, : self.L, : self.L]
                + torch.einsum("mk,mij->kij", rsp, gc))
        if zg_xcov is not None:
            xc = zg_xcov.reshape(-1, self.L, self.L)
            out["Szr"][:, :, : self.L] = (
                out["Szr"][:, :, : self.L] + torch.einsum("mk,mij->kij", rsp, xc))
        if z_cov is not None and self.q_rank > 0:
            zc = z_cov.reshape(-1, self.L, self.L)
            off = zc - torch.diag_embed(torch.diagonal(zc, dim1=-2, dim2=-1))
            out["Szz_full"] = out["Szz_full"] + torch.einsum("mk,mij->kij", rsp, off)
        return out

    def set_stats(self, stats):
        self.N.copy_(stats["N"])
        if "Srr" in stats:
            self.Srr.copy_(stats["Srr"]); self.Szr.copy_(stats["Szr"]); self.Szz.copy_(stats["Szz"])
            self.Shh.copy_(stats["Shh"]); self.Szh.copy_(stats["Szh"]); self.Srh.copy_(stats["Srh"])
            if self.q_rank > 0:
                self.Szz_full.copy_(stats["Szz_full"])
                for _nm in ("Szf", "Sfr", "Sfh", "Sff"):
                    if _nm in stats:
                        getattr(self, _nm).copy_(stats[_nm])
            self._residual_only = False
        else:
            self.Sgg.copy_(stats["Sgg"]); self.Szg.copy_(stats["Szg"]); self.Szz_resid.copy_(stats["Szz"])
            if self.q_rank > 0:
                self.Szz_full_resid.copy_(stats["Szz_full"])
                for _nm in ("Szf", "Sfr", "Sfh", "Sff"):
                    if _nm in stats:
                        getattr(self, _nm).copy_(stats[_nm])
            self._residual_only = True
        self._stats_initialised = True

    def ema_update_stats(self, stats, tau: float):
        if not self._stats_initialised:
            self.set_stats(stats); return
        names = ["N", "Srr", "Szr", "Szz", "Shh", "Szh", "Srh"]
        if self.q_rank > 0:
            names += ["Szz_full", "Szf", "Sfr", "Sfh", "Sff"]
        for name in names:
            getattr(self, name).mul_(1.0 - tau).add_(tau * stats[name])

    @torch.no_grad()
    def _regime_mstep_from_resid(self, Szg, Szz_resid):
        eye = torch.eye(self.Lr, device=self.M.device, dtype=self.M.dtype)
        lam0 = torch.diag(self.lam0_diag)
        lam = lam0.unsqueeze(0) + self.Sgg + self.jitter * eye
        chol = _chol_jitter(lam)  
        V = torch.cholesky_inverse(chol)
        rhs = (self.M0 @ lam0).unsqueeze(0) + Szg
        M = torch.einsum("klr,krs->kls", rhs, V)
        a = (self.a0 + 0.5 * self.N.unsqueeze(-1)).expand(self.K, self.L).clone()
        m0lam0m0 = torch.einsum("ir,rs,is->i", self.M0, lam0, self.M0)
        lamM = torch.einsum("krs,kls->klr", lam, M)
        MlamM = torch.einsum("klr,klr->kl", M, lamM)
        b = self._b0_rate() + 0.5 * (Szz_resid + m0lam0m0.unsqueeze(0) - MlamM)
        b = torch.clamp(b, min=1e-6)
        b = torch.maximum(b, a * self.noise_var_floor)
        self.M.copy_(M); self.lam.copy_(lam); self.V.copy_(V)
        self.a.copy_(a); self.b.copy_(b); self._lam_chol = chol

    @torch.no_grad()
    def _C_mstep(self):
        Om = self.Omega
        eyeH = torch.eye(self.Hp, device=self.M.device, dtype=self.M.dtype)
        Lam = self.vC0 * eyeH.unsqueeze(0) + torch.einsum("ki,khg->ihg", Om, self.Shh)
        Lam = Lam + self.jitter * eyeH
        GSrh = torch.einsum("klr,krh->klh", self.M, self.Srh)
        resid_cross = self.Szh - GSrh
        if self.q_rank > 0:
            resid_cross = resid_cross - torch.einsum("kif,kfh->kih", self.Umean, self.Sfh)
        eta = self.vC0 * self.C0mean + torch.einsum("ki,kih->ih", Om, resid_cross)
        cholC = torch.linalg.cholesky(Lam)
        Vc = torch.cholesky_inverse(cholC)
        m = torch.einsum("ihg,ig->ih", Vc, eta)
        self.Ccov.copy_(Vc); self.Cmean.copy_(m)

    @torch.no_grad()
    def _update_qU(self):
        Om = self.Omega
        F = self.q_rank
        Syf = self.Szf - torch.einsum("ih,kfh->kif", self.Cmean, self.Sfh)
        lin = Om.unsqueeze(-1) * (Syf - torch.einsum("klr,kfr->klf", self.M, self.Sfr))
        eyeF = torch.eye(F, dtype=self.M.dtype, device=self.M.device)
        Prec = (eyeF / self.u_prior_scale
                + Om.unsqueeze(-1).unsqueeze(-1) * self.Sff.unsqueeze(1))
        cov = torch.cholesky_inverse(_chol_jitter(Prec))
        mean = torch.einsum("kifg,kig->kif", cov, lin)
        evid = (self.Sff.abs().sum(dim=(-2, -1)) > 0)
        self.Ucov.copy_(torch.where(evid.view(-1, 1, 1, 1), cov, self.Ucov))
        self.Umean.copy_(torch.where(evid.view(-1, 1, 1), mean, self.Umean))

    @torch.no_grad()
    def _f_adjusted_resid_stats(self, Szg, Szz_resid):
        Syf = self.Szf - torch.einsum("ih,kfh->kif", self.Cmean, self.Sfh)
        SzgF = Szg - torch.einsum("kif,kfr->kir", self.Umean, self.Sfr)
        EuuT = self.Ucov + torch.einsum("kif,kig->kifg", self.Umean, self.Umean)
        SzzF = (Szz_resid - 2.0 * (self.Umean * Syf).sum(-1)
                + torch.einsum("kifg,kfg->ki", EuuT, self.Sff))
        return SzgF, SzzF.clamp_min(0.0)

    @torch.no_grad()
    def _residualise_stats(self):
        Szg = self.Szr - torch.einsum("ih,krh->kir", self.Cmean, self.Srh)
        CSzh = torch.einsum("ih,kih->ki", self.Cmean, self.Szh)
        ShhC = torch.einsum("khg,ig->kih", self.Shh, self.Cmean)
        CShhC = torch.einsum("ih,kih->ki", self.Cmean, ShhC)
        trVCShh = torch.einsum("ihg,khg->ki", self.Ccov, self.Shh)
        Szz_resid = self.Szz - 2.0 * CSzh + CShhC + trVCShh
        self.Sgg.copy_(self.Srr)
        self.Szg.copy_(Szg)
        self.Szz_resid.copy_(Szz_resid.clamp_min(0.0))
        if self.q_rank > 0:
            SzhC = torch.einsum("kih,jh->kij", self.Szh, self.Cmean)
            ShhC2 = torch.einsum("khg,jg->khj", self.Shh, self.Cmean)
            CShhC2 = torch.einsum("ih,khj->kij", self.Cmean, ShhC2)
            full = self.Szz_full - SzhC - SzhC.transpose(-1, -2) + CShhC2
            full = full + torch.diag_embed(trVCShh)
            self.Szz_full_resid.copy_(full)
        return Szg, self.Szz_resid

    @torch.no_grad()
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
                + aq * (bp - bq) / bq)

    def m_step(self):
        if getattr(self, "_residual_only", False):
            if self.q_rank > 0:
                self._update_qU()
                SzgF, SzzF = self._f_adjusted_resid_stats(self.Szg, self.Szz_resid)
                self._regime_mstep_from_resid(SzgF, SzzF)
            else:
                self._regime_mstep_from_resid(self.Szg, self.Szz_resid)
            if self.ard:
                self._ard_step()
            if self.q_rank > 0:
                self._refresh_lowrank_predictive_cache()
            return
        for _ in range(max(1, self.n_block_iters)):
            Szg, Szz_resid = self._residualise_stats()
            if self.q_rank > 0:
                self._update_qU()
                SzgF, SzzF = self._f_adjusted_resid_stats(Szg, Szz_resid)
                self._regime_mstep_from_resid(SzgF, SzzF)
            else:
                self._regime_mstep_from_resid(Szg, Szz_resid)
            if not self._freeze_C:
                self._C_mstep()
        self._residualise_stats()
        if self.ard:
            self._ard_step()
        if self.q_rank > 0:
            self._refresh_lowrank_predictive_cache()

        if getattr(self, "learn_b0", False):
            self.b0_chat.copy_(self.b0_c0 + float(self.K * self.L) * self.a0)
            self.b0_dhat.copy_(self.b0_d0 + (self.a / self.b).sum())

    @torch.no_grad()
    def _refresh_lowrank_predictive_cache(self):
        Einv_tau = self.b / (self.a - 1.0).clamp_min(1e-3)
        trU = torch.einsum("kiff->ki", self.Ucov)
        self.q_Ddiag.copy_((Einv_tau + trU).clamp_min(1e-8))
        self._q_taudiag.copy_(Einv_tau.clamp_min(1e-8))
        self._q_Udiag.copy_(trU.clamp_min(0.0))
        self.Ufac.copy_(self.Umean)

    @torch.no_grad()
    def reset_slot(self, k):
        k = int(k)
        self.M[k] = 0.0
        self.lam[k] = torch.diag(self.lam0_diag)
        self.V[k] = torch.diag(1.0 / self.lam0_diag)
        self.a[k] = float(self.a0)
        self.b[k] = float(self._b0_rate())
        if hasattr(self, "N"):
            self.N[k] = 0.0
        for name, buf in self.named_buffers():
            if torch.is_tensor(buf) and buf.dim() >= 1 and buf.shape[0] == self.K \
                    and name.startswith("S"):
                buf[k] = 0.0
        if self.q_rank > 0:
            self.Umean[k] = 0.01 * torch.randn_like(self.Umean[k])
            self.Ucov[k] = (self.u_prior_scale
                            * torch.eye(self.Umean.shape[-1], dtype=self.Ucov.dtype,
                                        device=self.Ucov.device))
        return self

    @torch.no_grad()
    def _ard_step(self):
        Vdiag = torch.diagonal(self.V, dim1=-2, dim2=-1)
        Om = self.Omega
        Mj2 = self.M ** 2
        weighted = torch.einsum("ki,kir->kr", Om, Mj2)
        occ = self.N > 1.0
        n_occ = int(occ.sum())
        if n_occ == 0:
            return
        denom = (self.L * Vdiag + weighted)[occ].sum(0)
        alpha = (n_occ * self.L) / torch.clamp(denom, min=1e-8)
        self.lam0_diag.copy_(torch.clamp(alpha, min=1e-4, max=1e3))

    @torch.no_grad()
    def param_kl(self) -> torch.Tensor:
        K, L, Lr, Hp = self.K, self.L, self.Lr, self.Hp
        lam0 = self.lam0_diag
        chol = self._lam_chol
        logdet_lam = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        logdet_lam0 = torch.log(lam0).sum()
        Vdiag = torch.diagonal(self.V, dim1=-2, dim2=-1)
        tr_term = (lam0.unsqueeze(0) * Vdiag).sum(-1)
        diff = self.M - self.M0.unsqueeze(0)
        maha = (lam0.view(1, 1, Lr) * diff ** 2).sum(-1)
        Etau = self.a / self.b
        gauss_kl = (0.5 * (logdet_lam - logdet_lam0 - Lr + tr_term).unsqueeze(-1)
                    + 0.5 * Etau * maha)
        a_q, b_q = self.a, self.b
        a_p = self.a0
        if getattr(self, "learn_b0", False):
            Elogb0 = torch.digamma(self.b0_chat) - torch.log(self.b0_dhat)
            Eb0 = self.b0_chat / self.b0_dhat
        else:
            Elogb0 = torch.log(self.b0)
            Eb0 = self.b0
        gamma_kl = ((a_q - a_p) * torch.digamma(a_q)
                    - torch.lgamma(a_q) + torch.lgamma(a_p)
                    + a_p * (torch.log(b_q) - Elogb0)
                    + a_q * (Eb0 - b_q) / b_q)
        regime_kl = (gauss_kl + gamma_kl).sum(-1)

        cholC = torch.linalg.cholesky(self.Ccov + self.jitter *
                                      torch.eye(Hp, device=self.M.device, dtype=self.M.dtype))
        logdetVc = 2.0 * torch.log(torch.diagonal(cholC, dim1=-2, dim2=-1)).sum(-1)
        trVc = torch.diagonal(self.Ccov, dim1=-2, dim2=-1).sum(-1)
        dC = self.Cmean - self.C0mean
        maha_C = (dC ** 2).sum(-1)
        C_kl = 0.5 * (-logdetVc - Hp * torch.log(self.vC0) - Hp
                      + self.vC0 * trVc + self.vC0 * maha_C).sum()
        total = regime_kl + C_kl / K
        if self.q_rank > 0:
            cu = float(self.u_prior_scale)
            F = self.q_rank
            cholU = _chol_jitter(self.Ucov)
            logdetU = 2.0 * torch.log(torch.diagonal(cholU, dim1=-2, dim2=-1)).sum(-1)
            u_kl = 0.5 * ((torch.einsum("kiff->ki", self.Ucov)
                           + (self.Umean ** 2).sum(-1)) / cu
                          - F + F * math.log(cu) - logdetU)
            total = total + u_kl.sum(-1)
        return total

    @torch.no_grad()
    @torch.no_grad()
    def data_elbo_from_stats(self) -> torch.Tensor:
        if getattr(self, "_residual_only", False):
            Szg, Szz_res = self.Szg, self.Szz_resid
        else:
            Szg, Szz_res = self._residualise_stats()
        N = self.N.clamp_min(0.0)
        MSzg = (self.M * Szg).sum(-1)
        MSggM = torch.einsum("klr,krs,kls->kl", self.M, self.Sgg, self.M)
        if self.q_rank > 0:
            var = self.q_Ddiag + (self.Ufac ** 2).sum(-1)
            quad = (Szz_res - 2.0 * MSzg + MSggM) / var
            per = (-0.5 * N.unsqueeze(-1) * LOG2PI
                   - 0.5 * N.unsqueeze(-1) * torch.log(var)
                   - 0.5 * quad)
            return per.sum()
        elog = torch.digamma(self.a) - torch.log(self.b)
        Etau = self.a / self.b
        per = (-0.5 * N.unsqueeze(-1) * LOG2PI
               + 0.5 * N.unsqueeze(-1) * elog
               - 0.5 * Etau * (Szz_res - 2.0 * MSzg + MSggM))
        Vtr = torch.einsum("krs,ksr->k", self.V, self.Sgg)
        return per.sum() - 0.5 * self.L * Vtr.sum()

    @torch.no_grad()
    def clone_with_K(self, new_K: int, stats=None):
        new = SharedCarryRegimes(
            K=new_K, L=self.L, G=self.G, a0=float(self.a0), b0=float(self.b0),
            v0_scale=1.0, vC0_scale=float(self.vC0), ard=False, identity_init=False,
            jitter=self.jitter, q_rank=self.q_rank, action_dim=self.action_dim,
            learn_b0=getattr(self, "learn_b0", False),
            b0_c0=float(getattr(self, "b0_c0", torch.tensor(2.0))),
            device=self.M.device, dtype=self.M.dtype,
        )
        if getattr(self, "learn_b0", False):
            new.b0_chat.copy_(self.b0_chat)
            new.b0_dhat.copy_(self.b0_dhat)
        new.lam0_diag.copy_(self.lam0_diag)
        new.M0.copy_(self.M0)
        new.Cmean.copy_(self.Cmean); new.Ccov.copy_(self.Ccov); new.C0mean.copy_(self.C0mean)
        new._freeze_C = True
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

from __future__ import annotations

import math
import torch

LOG2PI = math.log(2.0 * math.pi)


def expected_potentials(reg, gamma, c, g_mean=None):
    shared = hasattr(reg, "Lr")
    L = int(reg.L)
    K = int(reg.K)
    dtp, dev = reg.M.dtype, reg.M.device
    gamma = gamma.to(dtp)
    c = c.to(dtp)
    B, T, Dc = c.shape
    q_rank = int(getattr(reg, "q_rank", 0))

    A = reg.M[..., :L]
    Vzz = reg.V[:, :L, :L]

    if shared:
        Hp = int(reg.Hp)
        htil = c[..., :Hp]
        tail = c[..., Hp:]
        m = (torch.einsum("kld,btd->btkl", reg.M[..., L:], tail)
             + torch.einsum("lh,bth->btl", reg.Cmean, htil).unsqueeze(2))
        Vztail = torch.einsum("kld,btd->btkl", reg.V[:, :L, L:], tail)
    else:
        tail = c
        m = torch.einsum("kld,btd->btkl", reg.M[..., L:], c)
        Vztail = torch.einsum("kld,btd->btkl", reg.V[:, :L, L:], c)

    Et = (reg.a / reg.b.clamp_min(1e-30))

    if q_rank == 0:
        Prec_diag = Et
        EtA = Et.unsqueeze(-1) * A
        PhiA_k = torch.einsum("kil,kim->klm", EtA, A) + float(L) * Vzz
        OmM = torch.diag_embed(torch.einsum("btk,ki->bti", gamma, Prec_diag))
        PsiA = torch.einsum("btk,kil->btil", gamma, EtA)
        PhiA = torch.einsum("btk,klm->btlm", gamma, PhiA_k)
        bz = torch.einsum("btk,btkl->btl", gamma, Et.view(1, 1, K, L) * m)
        bp = torch.einsum("btk,btkl->btl", gamma,
                          torch.einsum("kil,btki->btkl", EtA, m)
                          + float(L) * Vztail)
        return dict(OmM=OmM, PsiA=PsiA, PhiA=PhiA, bz=bz, bp=bp)

    if shared:
        Om = reg.Omega
        Um = reg.Umean
        EuuT = reg.Ucov + torch.einsum("kif,kig->kifg", Um, Um)
        F = Um.shape[-1]
        P = (torch.eye(F, dtype=dtp, device=dev)
             + torch.einsum("ki,kifg->kfg", Om, EuuT))
        W = Om.unsqueeze(-1) * Um
        sol = torch.cholesky_solve(W.transpose(-1, -2),
                                   torch.linalg.cholesky(P))
        Prec_k = torch.diag_embed(Om) - torch.einsum("klf,kfm->klm", W, sol)
        Prec = Prec_k.view(1, 1, K, L, L).expand(B, T, K, L, L)
        fluct_PhiA = float(L) * Vzz
        fluct_bp = float(L) * Vztail
    else:
        if g_mean is None:
            raise ValueError(
                "expected_potentials: q_rank > 0 on DiagARRegimes needs `g_mean` "
                "(the current posterior-mean regressor) to evaluate the "
                "parameter-fluctuation inflation (1 + g^T V g).")
        g_mean = g_mean.to(dtp)
        Q_k = torch.diag_embed(reg.q_Ddiag) + torch.einsum(
            "klr,kmr->klm", reg.Ufac, reg.Ufac)
        eyeL = torch.eye(L, dtype=dtp, device=dev)
        Qinv_k = torch.cholesky_inverse(torch.linalg.cholesky(
            Q_k + 1e-10 * eyeL))
        Vg = torch.einsum("kgh,bth->btkg", reg.V, g_mean)
        gVg = torch.einsum("btg,btkg->btk", g_mean, Vg)
        infl = (1.0 + gVg).clamp(min=1e-6, max=float(reg.infl_max))
        Prec = Qinv_k.view(1, 1, K, L, L) / infl[..., None, None]
        fluct_PhiA = None
        fluct_bp = None

    PrecA = torch.einsum("btklm,kmj->btklj", Prec, A)
    OmM = torch.einsum("btk,btklm->btlm", gamma, Prec)
    PsiA = torch.einsum("btk,btklj->btlj", gamma, PrecA)
    PhiA_k = torch.einsum("kil,btkij->btklj", A, PrecA)
    if fluct_PhiA is not None:
        PhiA_k = PhiA_k + fluct_PhiA.view(1, 1, K, L, L)
    PhiA = torch.einsum("btk,btklj->btlj", gamma, PhiA_k)
    Pm = torch.einsum("btklm,btkm->btkl", Prec, m)
    bz = torch.einsum("btk,btkl->btl", gamma, Pm)
    bp_k = torch.einsum("kil,btki->btkl", A, Pm)
    if fluct_bp is not None:
        bp_k = bp_k + fluct_bp
    bp = torch.einsum("btk,btkl->btl", gamma, bp_k)
    return dict(OmM=OmM, PsiA=PsiA, PhiA=PhiA, bz=bz, bp=bp)


def chain_potentials(reg, gamma, g_full):
    L = int(reg.L)
    c = g_full[..., L:]
    needs_g = (int(getattr(reg, "q_rank", 0)) > 0) and not hasattr(reg, "Lr")
    return expected_potentials(reg, gamma, c, g_mean=g_full if needs_g else None)


def build_blocks(pot, enc_mean, enc_prec, prior_mean=None, prior_prec=None,
                 is_first=None, valid=None):
    OmM, PsiA, PhiA = pot['OmM'], pot['PsiA'], pot['PhiA']
    bz, bp = pot['bz'], pot['bp']
    B, T, L, _ = OmM.shape
    dev, dt = OmM.device, OmM.dtype
    eye = torch.eye(L, device=dev, dtype=dt)

    D = torch.zeros(B, T, L, L, dtype=dt, device=dev)
    U = torch.zeros(B, max(T - 1, 1), L, L, dtype=dt, device=dev)
    h = torch.zeros(B, T, L, dtype=dt, device=dev)

    D += torch.diag_embed(enc_prec.to(dt))
    h += enc_prec.to(dt) * enc_mean.to(dt)

    if prior_prec is not None:
        prior_prec = prior_prec.to(dt)
        pm = (torch.zeros_like(enc_mean[:, 0], dtype=dt) if prior_mean is None
              else prior_mean.to(dt))
        if prior_prec.dim() == 2:
            prior_prec = prior_prec.unsqueeze(1).expand(B, T, L)
        if pm.dim() == 2:
            pm = pm.unsqueeze(1).expand(B, T, L)
        anchor = torch.zeros(B, T, 1, dtype=dt, device=dev)
        anchor[:, 0] = 1.0
        if is_first is not None:
            anchor = torch.maximum(anchor, is_first.reshape(B, T, 1).to(dt))
        D = D + torch.diag_embed(anchor * prior_prec)
        h = h + anchor * prior_prec * pm

    if T > 1:
        keep = torch.ones(B, T - 1, dtype=dt, device=dev)
        if is_first is not None:
            keep = keep * (1.0 - is_first[:, 1:].to(dt))
        if valid is not None:
            v = valid.to(dt)
            keep = keep * v[:, :-1] * v[:, 1:]
        kw = keep[..., None, None]

        D[:, 1:] = D[:, 1:] + kw * OmM[:, 1:]
        D[:, :-1] = D[:, :-1] + kw * PhiA[:, 1:]
        U[:] = -kw * PsiA[:, 1:].transpose(-1, -2)
        h[:, 1:] = h[:, 1:] + keep[..., None] * bz[:, 1:]
        h[:, :-1] = h[:, :-1] - keep[..., None] * bp[:, 1:]

    D = D + 1e-8 * eye
    return D, U, h


def smooth(D, U, h, return_logdet=False):
    B, T, L, _ = D.shape
    dev, dt = D.device, D.dtype

    Lc = [torch.linalg.cholesky(D[:, 0])]
    for t in range(1, T):
        Wt = torch.cholesky_solve(U[:, t - 1], Lc[t - 1])
        Dp_t = D[:, t] - U[:, t - 1].transpose(-1, -2) @ Wt
        Lc.append(torch.linalg.cholesky(Dp_t))

    y = [h[:, 0]]
    for t in range(1, T):
        Wt = torch.cholesky_solve(U[:, t - 1], Lc[t - 1])
        y.append(h[:, t] - torch.einsum('blm,bl->bm', Wt, y[t - 1]))

    mean_l = [None] * T
    mean_l[T - 1] = torch.cholesky_solve(y[T - 1].unsqueeze(-1), Lc[T - 1]).squeeze(-1)
    for t in range(T - 2, -1, -1):
        rhs = y[t] - torch.einsum('blm,bm->bl', U[:, t], mean_l[t + 1])
        mean_l[t] = torch.cholesky_solve(rhs.unsqueeze(-1), Lc[t]).squeeze(-1)
    mean = torch.stack(mean_l, dim=1)

    eye = torch.eye(L, device=dev, dtype=dt).expand(B, L, L)
    cov_l = [None] * T
    xcov_l = [None] * max(T - 1, 1)
    cov_l[T - 1] = torch.cholesky_solve(eye, Lc[T - 1])
    for t in range(T - 2, -1, -1):
        Dinv = torch.cholesky_solve(eye, Lc[t])
        xc = -Dinv @ U[:, t] @ cov_l[t + 1]
        xcov_l[t] = xc
        cov_l[t] = Dinv - xc @ U[:, t].transpose(-1, -2) @ Dinv
    cov = torch.stack(cov_l, dim=1)
    xcov = (torch.stack(xcov_l, dim=1) if T > 1
            else torch.zeros(B, 1, L, L, dtype=dt, device=dev))

    logdet = 2.0 * torch.log(torch.diagonal(torch.stack(Lc, dim=1),
                                            dim1=-2, dim2=-1)).sum((-1, -2))
    quad = (h * mean).sum((-1, -2))
    logZ = 0.5 * quad - 0.5 * logdet + 0.5 * T * L * LOG2PI
    if return_logdet:
        return mean, cov, xcov, logZ, logdet
    return mean, cov, xcov, logZ


@torch.no_grad()
def smoothed_stats(reg, gamma, mean, cov, xcov, c, valid=None,
                   is_first=None, z0_mean=None, z0_var=None):
    L = int(reg.L)
    B, T, _ = mean.shape
    Dc = c.shape[-1]
    G = L + Dc
    w = gamma / gamma.sum(-1, keepdim=True).clamp_min(1e-30)
    if valid is not None:
        v = valid.to(w.dtype)
        pair_v = (v[:, 1:] * v[:, :-1]).unsqueeze(-1)
    else:
        pair_v = None
    w = w[:, 1:]
    if pair_v is not None:
        w = w * pair_v
    m, mp = mean[:, 1:], mean[:, :-1]
    cov_t, cov_p = cov[:, 1:], cov[:, :-1]
    xc = xcov.transpose(-1, -2)

    if is_first is not None:
        isf = is_first[:, 1:].reshape(B, T - 1, 1).to(mean.dtype)
        z0m = (mean.new_zeros(L) if z0_mean is None else z0_mean.to(mean)).view(1, 1, L)
        z0c = torch.diag_embed(
            (mean.new_zeros(L) if z0_var is None else z0_var.to(mean))).view(1, 1, L, L)
        mp = torch.where(isf > 0.5, z0m.expand_as(mp), mp)
        isf4 = isf.unsqueeze(-1)
        cov_p = torch.where(isf4 > 0.5, z0c.expand_as(cov_p), cov_p)
        xc = torch.where(isf4 > 0.5, torch.zeros_like(xc), xc)

    if hasattr(reg, "Lr"):
        g_pair = torch.cat([mp, c[:, 1:].to(mean.dtype)], dim=-1)
        return reg.stats_from_batch(
            w, mean[:, 1:], g_pair,
            z_var=torch.diagonal(cov_t, dim1=-2, dim2=-1).clamp_min(0),
            zg_xcov=xc, g_zcov=cov_p, z_cov=cov_t)

    Czz = cov_t + torch.einsum('btl,btm->btlm', m, m)
    Cpp = cov_p + torch.einsum('btl,btm->btlm', mp, mp)
    Cxp = xc + torch.einsum('btl,btm->btlm', m, mp)
    cc = c[:, 1:]

    Egg = torch.zeros(B, T - 1, G, G, dtype=mean.dtype, device=mean.device)
    Egg[..., :L, :L] = Cpp
    Egg[..., :L, L:] = torch.einsum('btl,btd->btld', mp, cc)
    Egg[..., L:, :L] = Egg[..., :L, L:].transpose(-1, -2)
    Egg[..., L:, L:] = torch.einsum('btd,bte->btde', cc, cc)
    Ezg = torch.cat([Cxp, torch.einsum('btl,btd->btld', m, cc)], dim=-1)

    Sgg = torch.einsum('btk,btlm->klm', w, Egg)
    Szg = torch.einsum('btk,btlg->klg', w, Ezg)
    Szz = torch.einsum('btk,btl->kl', w, torch.diagonal(Czz, dim1=-2, dim2=-1))
    N = w.sum((0, 1))
    out = dict(N=N, Sgg=Sgg, Szg=Szg, Szz=Szz)
    if int(getattr(reg, "q_rank", 0)) > 0:
        out["Szz_full"] = torch.einsum('btk,btlm->klm', w, Czz)
    return out


@torch.no_grad()
def verify_against_dense(B=2, T=7, L=3, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(B, T, L, L, generator=g, dtype=dtype)
    D = D @ D.transpose(-1, -2) + 3.0 * L * torch.eye(L, dtype=dtype)
    U = 0.15 * torch.randn(B, T - 1, L, L, generator=g, dtype=dtype)
    h = torch.randn(B, T, L, generator=g, dtype=dtype)
    mean, cov, xcov, logZ = smooth(D, U, h)

    errs = {}
    for bi in range(B):
        J = torch.zeros(T * L, T * L, dtype=dtype)
        for t in range(T):
            J[t*L:(t+1)*L, t*L:(t+1)*L] = D[bi, t]
        for t in range(T - 1):
            J[t*L:(t+1)*L, (t+1)*L:(t+2)*L] = U[bi, t]
            J[(t+1)*L:(t+2)*L, t*L:(t+1)*L] = U[bi, t].T
        hv = h[bi].reshape(-1)
        Sig = torch.linalg.inv(J)
        mu = Sig @ hv
        lz = 0.5 * hv @ mu - 0.5 * torch.logdet(J) + 0.5 * T * L * LOG2PI
        e_mu = (mu.reshape(T, L) - mean[bi]).abs().max().item()
        e_cv = max((Sig[t*L:(t+1)*L, t*L:(t+1)*L] - cov[bi, t]).abs().max().item()
                   for t in range(T))
        e_xc = max((Sig[t*L:(t+1)*L, (t+1)*L:(t+2)*L] - xcov[bi, t]).abs().max().item()
                   for t in range(T - 1))
        e_lz = abs(lz.item() - logZ[bi].item())
        errs[bi] = dict(mean=e_mu, cov=e_cv, xcov=e_xc, logZ=e_lz)
    return errs


@torch.no_grad()
def gamma_norm_error(gamma, valid=None):
    s = gamma.sum(-1)
    e = (s - 1.0).abs()
    if valid is not None:
        e = e * valid.to(e.dtype)
    return float(e.max())

def sample_chain(D, U, h, n_samples=1, generator=None):
    B, T, L, _ = D.shape
    dev, dt = D.device, D.dtype
    Lc = [torch.linalg.cholesky(D[:, 0])]
    for t in range(1, T):
        Wt = torch.cholesky_solve(U[:, t - 1], Lc[t - 1])
        Lc.append(torch.linalg.cholesky(D[:, t] - U[:, t - 1].transpose(-1, -2) @ Wt))
    y = [h[:, 0]]
    for t in range(1, T):
        Wt = torch.cholesky_solve(U[:, t - 1], Lc[t - 1])
        y.append(h[:, t] - torch.einsum('blm,bl->bm', Wt, y[t - 1]))

    S = int(n_samples)
    eps = torch.randn(S, B, T, L, dtype=dt, device=dev, generator=generator)
    def _white(t, e):
        return torch.linalg.solve_triangular(
            Lc[t].transpose(-1, -2).unsqueeze(0).expand(S, B, L, L),
            e.unsqueeze(-1), upper=True).squeeze(-1)

    z = [None] * T
    mT = torch.cholesky_solve(y[T - 1].unsqueeze(-1), Lc[T - 1]).squeeze(-1)
    z[T - 1] = mT.unsqueeze(0) + _white(T - 1, eps[:, :, T - 1])
    for t in range(T - 2, -1, -1):
        rhs = y[t].unsqueeze(0) - torch.einsum('blm,sbm->sbl', U[:, t], z[t + 1])
        mt = torch.cholesky_solve(
            rhs.reshape(S * B, L, 1),
            Lc[t].unsqueeze(0).expand(S, B, L, L).reshape(S * B, L, L)
        ).reshape(S, B, L)
        z[t] = mt + _white(t, eps[:, :, t])
    return torch.stack(z, dim=2)

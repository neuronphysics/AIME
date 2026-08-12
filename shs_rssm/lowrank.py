import torch


def _woodbury_solve(d, U, x):
    r = U.shape[-1]
    Dinv = 1.0 / d
    if r == 0:
        return Dinv.unsqueeze(-1) * x if x.dim() == d.dim() + 1 else Dinv * x, None
    DinvU = Dinv.unsqueeze(-1) * U
    cap = torch.einsum("...lr,...ls->...rs", U, DinvU)
    eye_r = torch.eye(r, device=d.device, dtype=d.dtype)
    cap = cap + eye_r
    Lcap = torch.linalg.cholesky(cap)
    vec = x.dim() == d.dim()
    xm = x.unsqueeze(-1) if vec else x
    Dinv_x = Dinv.unsqueeze(-1) * xm
    Ut_Dinv_x = torch.einsum("...lr,...lm->...rm", U, Dinv_x)
    sol = torch.cholesky_solve(Ut_Dinv_x, Lcap)
    corr = torch.einsum("...lr,...rm->...lm", DinvU, sol)
    out = Dinv_x - corr
    return (out.squeeze(-1) if vec else out), Lcap


def lowrank_logdet(d, U):
    r = U.shape[-1]
    logdet = torch.log(d).sum(-1)
    if r == 0:
        return logdet
    Dinv = 1.0 / d
    cap = torch.einsum("...lr,...ls->...rs", U, Dinv.unsqueeze(-1) * U)
    cap = cap + torch.eye(r, device=d.device, dtype=d.dtype)
    Lcap = torch.linalg.cholesky(cap)
    logdet = logdet + 2.0 * torch.log(torch.diagonal(Lcap, dim1=-2, dim2=-1)).sum(-1)
    return logdet


def lowrank_diag(d, U):
    if U.shape[-1] == 0:
        return d
    return d + (U ** 2).sum(-1)


def lowrank_inv_diag(d, U):
    Dinv = 1.0 / d
    if U.shape[-1] == 0:
        return Dinv
    DinvU = Dinv.unsqueeze(-1) * U
    cap = torch.einsum("...lr,...ls->...rs", U, DinvU)
    cap = cap + torch.eye(U.shape[-1], device=d.device, dtype=d.dtype)
    Lcap = torch.linalg.cholesky(cap)
    sol = torch.cholesky_solve(DinvU.transpose(-1, -2), Lcap)
    diag_corr = torch.einsum("...lr,...rl->...l", DinvU, sol)
    return Dinv - diag_corr


def lowrank_logpdf(z, mean, d, U):
    L = z.shape[-1]
    e = (z - mean)
    Qinv_e, _ = _woodbury_solve(d, U, e)
    quad = (e * Qinv_e).sum(-1)
    logdet = lowrank_logdet(d, U)
    return -0.5 * (L * torch.log(torch.tensor(2 * torch.pi, dtype=d.dtype, device=d.device))
                   + logdet + quad)


def lowrank_kl_diag_q(mu_q, var_q, mu_p, d_p, U_p):
    L = mu_q.shape[-1]
    diff = mu_q - mu_p
    Qinv_diff, _ = _woodbury_solve(d_p, U_p, diff)
    quad = (diff * Qinv_diff).sum(-1)
    tr = (var_q * lowrank_inv_diag(d_p, U_p)).sum(-1)
    logdet_p = lowrank_logdet(d_p, U_p)
    logdet_q = torch.log(var_q).sum(-1)
    return 0.5 * (logdet_p - logdet_q - L + tr + quad)


def lowrank_quadform_cols(d, U, M):
    Dinv = 1.0 / d
    quadD = torch.einsum("...kl,klg->...kg", Dinv, M ** 2)
    r = U.shape[-1]
    if r == 0:
        return quadD
    t = torch.einsum("...klr,...kl,klg->...krg", U, Dinv, M)
    W = torch.einsum("...klr,...kl,...kls->...krs", U, Dinv, U)
    eye = torch.eye(r, device=d.device, dtype=d.dtype)
    W = W + eye
    x = torch.linalg.solve(W, t)
    corr = torch.einsum("...krg,...krg->...kg", t, x)
    return quadD - corr


def lowrank_trace_prod(d, U, S):
    Dinv = 1.0 / d
    quadD = (torch.diagonal(S, dim1=-2, dim2=-1) * Dinv).sum(-1)
    r = U.shape[-1]
    if r == 0:
        return quadD
    DinvU = Dinv.unsqueeze(-1) * U
    W = torch.einsum("...lr,...ls->...rs", U, DinvU)
    W = W + torch.eye(r, device=d.device, dtype=d.dtype)
    Lcap = torch.linalg.cholesky(W)
    T1 = torch.einsum("...lr,...lm->...rm", DinvU, S)
    T2 = torch.einsum("...rm,...ms->...rs", T1, DinvU)
    sol = torch.cholesky_solve(T2, Lcap)
    corr = torch.diagonal(sol, dim1=-2, dim2=-1).sum(-1)
    return quadD - corr

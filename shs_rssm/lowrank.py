"""Low-rank-plus-diagonal Gaussian operations for the regime process noise.

Process noise covariance per regime is  Q = diag(d) + U U^T  with U of shape (L, r),
r << L. All operations use the Woodbury identity and the matrix-determinant lemma so the
cost is O(L r^2 + r^3) for the shared factors plus O(L r) per evaluated point, instead of
the O(L^3) a dense covariance would need. With r = 0 (no factor) these reduce to the
plain diagonal case.

Shapes: leading dims (...) broadcast; d is (..., L); U is (..., L, r); vectors are (..., L).
"""
import torch


def _woodbury_solve(d, U, x):
    """Return Q^{-1} x and the r x r capacitance Cholesky, for Q = diag(d) + U U^T.

    Q^{-1} = D^{-1} - D^{-1} U (I_r + U^T D^{-1} U)^{-1} U^T D^{-1}.
    x : (..., L) or (..., L, m). Returns (Qinv_x, L_cap) where L_cap is chol of the cap.
    """
    r = U.shape[-1]
    Dinv = 1.0 / d                                           # (...,L)
    if r == 0:
        return Dinv.unsqueeze(-1) * x if x.dim() == d.dim() + 1 else Dinv * x, None
    DinvU = Dinv.unsqueeze(-1) * U                           # (...,L,r)
    cap = torch.einsum("...lr,...ls->...rs", U, DinvU)       # U^T D^{-1} U  (...,r,r)
    eye_r = torch.eye(r, device=d.device, dtype=d.dtype)
    cap = cap + eye_r
    Lcap = torch.linalg.cholesky(cap)                        # (...,r,r)
    vec = x.dim() == d.dim()
    xm = x.unsqueeze(-1) if vec else x                       # (...,L,m)
    Dinv_x = Dinv.unsqueeze(-1) * xm                         # (...,L,m)
    Ut_Dinv_x = torch.einsum("...lr,...lm->...rm", U, Dinv_x)        # (...,r,m)
    sol = torch.cholesky_solve(Ut_Dinv_x, Lcap)                     # (...,r,m)
    corr = torch.einsum("...lr,...rm->...lm", DinvU, sol)           # (...,L,m)
    out = Dinv_x - corr
    return (out.squeeze(-1) if vec else out), Lcap


def lowrank_logdet(d, U):
    """log det(diag(d) + U U^T) = sum log d + log det(I_r + U^T D^{-1} U)."""
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
    """Diagonal of diag(d) + U U^T  (for moment-matching the mixture marginal)."""
    if U.shape[-1] == 0:
        return d
    return d + (U ** 2).sum(-1)


def lowrank_inv_diag(d, U):
    """Diagonal of (diag(d) + U U^T)^{-1} via Woodbury, i.e. (Q^{-1})_{ii}.

    Used for the trace term tr(Q^{-1} diag(v)) = sum_i v_i (Q^{-1})_{ii} that appears in
    both the KL and the expected-log-likelihood (E-step) corrections.
    """
    Dinv = 1.0 / d
    if U.shape[-1] == 0:
        return Dinv
    DinvU = Dinv.unsqueeze(-1) * U                                    # (...,L,r)
    cap = torch.einsum("...lr,...ls->...rs", U, DinvU)
    cap = cap + torch.eye(U.shape[-1], device=d.device, dtype=d.dtype)
    Lcap = torch.linalg.cholesky(cap)
    sol = torch.cholesky_solve(DinvU.transpose(-1, -2), Lcap)        # (...,r,L)
    diag_corr = torch.einsum("...lr,...rl->...l", DinvU, sol)
    return Dinv - diag_corr


def lowrank_logpdf(z, mean, d, U):
    """log N(z; mean, diag(d) + U U^T), summed over the L dim. Leading dims broadcast."""
    L = z.shape[-1]
    e = (z - mean)
    Qinv_e, _ = _woodbury_solve(d, U, e)
    quad = (e * Qinv_e).sum(-1)
    logdet = lowrank_logdet(d, U)
    return -0.5 * (L * torch.log(torch.tensor(2 * torch.pi, dtype=d.dtype, device=d.device))
                   + logdet + quad)


def lowrank_kl_diag_q(mu_q, var_q, mu_p, d_p, U_p):
    """KL( N(mu_q, diag var_q) || N(mu_p, diag(d_p) + U_p U_p^T) ), summed over L.

    = 0.5 [ logdet(Q_p) - sum log var_q - L + tr(Q_p^{-1} diag var_q)
            + (mu_q-mu_p)^T Q_p^{-1} (mu_q-mu_p) ].
    """
    L = mu_q.shape[-1]
    diff = mu_q - mu_p
    Qinv_diff, _ = _woodbury_solve(d_p, U_p, diff)
    quad = (diff * Qinv_diff).sum(-1)
    # tr(Q_p^{-1} diag var_q) = sum_i var_q_i * (Q_p^{-1})_{ii}
    tr = (var_q * lowrank_inv_diag(d_p, U_p)).sum(-1)
    logdet_p = lowrank_logdet(d_p, U_p)
    logdet_q = torch.log(var_q).sum(-1)
    return 0.5 * (logdet_p - logdet_q - L + tr + quad)


def lowrank_quadform_cols(d, U, M):
    """Column quadratic forms  colQ[...,k,g] = M[k,:,g]^T Q_k^{-1} M[k,:,g]  for
    Q_k = diag(d_k) + U_k U_k^T, evaluated for every regressor column g of M.

    d : (...,K,L)       diagonal factor (may carry batch leading dims via inflation)
    U : (...,K,L,r)     low-rank factor
    M : (K,L,G)         per-regime linear maps (no batch leading dims; broadcast)

    Returns (...,K,G).  This is the exact contraction used for the regressor-uncertainty
    trace  tr(Q^{-1} M diag(g_var) M^T) = sum_g g_var_g * colQ[...,k,g], because
    M diag(g_var) M^T = sum_g g_var_g m_g m_g^T.  Woodbury keeps every intermediate at
    most (...,K,r,G), never the dense (...,K,L,G).  For r == 0 it is the plain
    diagonal contraction sum_i M_{k,i,g}^2 / d_{k,i}.
    """
    Dinv = 1.0 / d                                                  # (...,K,L)
    quadD = torch.einsum("...kl,klg->...kg", Dinv, M ** 2)          # (...,K,G)
    r = U.shape[-1]
    if r == 0:
        return quadD
    # t_{...,k,r,g} = sum_i U_{k,i,r} Dinv_{k,i} M_{k,i,g}
    t = torch.einsum("...klr,...kl,klg->...krg", U, Dinv, M)        # (...,K,r,G)
    # W_k = I_r + U_k^T Dinv_k U_k
    W = torch.einsum("...klr,...kl,...kls->...krs", U, Dinv, U)     # (...,K,r,r)
    eye = torch.eye(r, device=d.device, dtype=d.dtype)
    W = W + eye
    x = torch.linalg.solve(W, t)                                   # (...,K,r,G) = W^{-1} t
    corr = torch.einsum("...krg,...krg->...kg", t, x)              # (...,K,G)
    return quadD - corr

"""Per-regime Bayesian linear-Gaussian dynamics with DIAGONAL process noise.

Each regime l models the one-step latent transition

    z_t = Gamma_l g_t + eps,   eps ~ N(0, Q_l),   Q_l = diag(q_{l,1..L}),

with the stacked regressor g_t = [z_{t-1}; (P h_t); 1] in R^G and Gamma_l in R^{L x G}.

Because Q_l is diagonal the matrix-normal-Wishart factorises across output
dimensions into L independent Normal-Gamma regressions that SHARE a single
G x G design precision Lambda_l per regime:

    tau_{l,i}            ~ Gamma(a0, b0)                 (precision 1/q_{l,i})
    w_{l,i} | tau_{l,i}  ~ N(m0_i, (tau_{l,i} Lambda0)^{-1})

Posterior, from responsibility-weighted sufficient statistics
    N_l    = sum_t r_t(l)
    Sgg_l  = sum_t r_t(l) g_t g_t^T            (G x G, shared across i)
    Szg_l  = sum_t r_t(l) z_t g_t^T            (L x G)
    Szz_l  = sum_t r_t(l) z_t.^2               (L,, diagonal only)
is
    Lambda_l = Lambda0 + Sgg_l                                   (G x G)
    M_l      = (M0 Lambda0 + Szg_l) Lambda_l^{-1}                (L x G)
    a_{l,i}  = a0 + N_l/2
    b_{l,i}  = b0 + 0.5 ( Szz_{l,i} + m0_i^T Lambda0 m0_i - M_{l,i}^T Lambda_l M_{l,i} )

This is the diagonal specialisation of the manuscript's MNIW update; Lambda_l is
the column PRECISION (bnpy's `Post.V` convention), V_l = Lambda_l^{-1} the column
covariance used in the fluctuation term.

All tensors are torch; the posterior is held in buffers so the module moves with
`.to(device)` and is excluded from the optimizer (it is updated by closed form,
not by gradient descent).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

LOG2PI = math.log(2.0 * math.pi)



def _chol_jitter(mat, tries=4):
    """SPD Cholesky with escalating diagonal jitter (transactional-guard support):
    retries at 1e-7/1e-6/1e-5 of the mean |diagonal| before letting the error rise."""
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
        v0_scale: float = 1.0,
        ard: bool = True,
        identity_init: bool = True,
        jitter: float = 1e-6,
        q_rank: int = 0,
        infl_max: float = 10.0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float32,
    ):
        self.action_dim = int(action_dim)   # informational; G already includes it
        """
        K : max number of regimes (truncation)
        L : latent dimension (dim of z)
        G : regressor dimension (= L + H' + 1)
        a0, b0 : Gamma(shape, rate) prior on each output-dim precision
        v0_scale : scales the prior weight precision Lambda0 = v0_scale * I (or ARD diag)
        ard : if True, Lambda0 diagonal is learned per input dim by a MacKay step (m_step)
        identity_init : seed M0 so that the z_{t-1} block of Gamma is identity
                        (a near-stationary "z_t ~ z_{t-1}" prior, as DreamerV3 favours)
        q_rank : if > 0, the process-noise covariance is low-rank-plus-diagonal
                 Q_l = diag(d_l) + U_l U_l^T with U_l of shape (L, q_rank), fit by ML
                 factor analysis on the responsibility-weighted residuals (q_rank=0 keeps
                 the diagonal Bayesian path unchanged). Only the E-step emission uses the
                 full Q; the predictive marginal stays diagonal so imagination / the KL
                 gradient are interface-identical.
        """
        super().__init__()
        self.K, self.L, self.G = K, L, G
        self.ard = ard
        self.noise_var_floor = 1e-6  # E[tau] cap = 1/floor; Q-collapse safeguard (see m_step)
        self.jitter = jitter
        # cap on the predictive-variance inflation (1 + g^T V g): the M-coefficient-uncertainty
        # term grows quadratically in g, so an unbounded value lets the actor-imagination
        # rollout self-amplify (large z -> large g -> larger variance -> larger z) and diverge.
        # Capping it bounds the predictive uncertainty without changing the fitted mean.
        self.infl_max = float(infl_max)

        # ---- prior hyperparameters (buffers; ard diag may be updated) ----
        self.register_buffer("a0", torch.tensor(float(a0), dtype=dtype, device=device))
        self.register_buffer("b0", torch.tensor(float(b0), dtype=dtype, device=device))
        # Lambda0 stored as its diagonal (G,), kept diagonal so ARD is a per-dim scalar.
        self.register_buffer("lam0_diag", torch.full((G,), float(v0_scale), dtype=dtype, device=device))

        # M0 : (L, G). Optionally identity on the z_{t-1} block (first L cols).
        M0 = torch.zeros(L, G, dtype=dtype, device=device)
        if identity_init and G >= L:
            M0[:, :L] = torch.eye(L, dtype=dtype, device=device)
        self.register_buffer("M0", M0)

        # ---- posterior parameters (buffers) ----
        self.register_buffer("M", M0.clone().unsqueeze(0).repeat(K, 1, 1))      # (K,L,G)
        self.register_buffer("lam", torch.diag_embed(self.lam0_diag).unsqueeze(0).repeat(K, 1, 1))  # (K,G,G)
        self.register_buffer("a", torch.full((K, L), float(a0), dtype=dtype, device=device))   # (K,L)
        self.register_buffer("b", torch.full((K, L), float(b0), dtype=dtype, device=device))   # (K,L)
        # cached column covariance V_l = lam^{-1} and its Cholesky for solves
        self.register_buffer("V", torch.diag_embed(1.0 / self.lam0_diag).unsqueeze(0).repeat(K, 1, 1))  # (K,G,G)

        # ---- EMA sufficient statistics (buffers) ----
        self.register_buffer("N", torch.zeros(K, dtype=dtype, device=device))                  # (K,)
        self.register_buffer("Sgg", torch.zeros(K, G, G, dtype=dtype, device=device))          # (K,G,G)
        self.register_buffer("Szg", torch.zeros(K, L, G, dtype=dtype, device=device))          # (K,L,G)
        self.register_buffer("Szz", torch.zeros(K, L, dtype=dtype, device=device))             # (K,L)
        # ---- optional low-rank-plus-diagonal process noise Q = diag(d) + U U^T ----
        self.q_rank = int(q_rank)
        if self.q_rank > 0:
            raise ValueError(
                "q_rank>0 on the non-shared path (DiagARRegimes) evaluated an inflated "
                "PREDICTIVE density instead of the ELBO likelihood and carried a "
                "point-estimated U with no posterior or KL (external review, round 3). "
                "Use shared_carry=True: SharedCarryRegimes implements the fully "
                "variational factor-augmented low-rank noise (local f_t ~ N(0,I), "
                "Gaussian q(U) with prior and KL, exact conditional q(f)). "
                "Config knobs: shs_q_rank / shs_shared_carry.")
        if self.q_rank > 0:
            # full second moment of z (needed for the residual covariance), the fitted
            # factor U, and the fitted diagonal d
            self.register_buffer("Szz_full", torch.zeros(K, L, L, dtype=dtype, device=device))   # (K,L,L)
            self.register_buffer("Ufac", torch.zeros(K, L, self.q_rank, dtype=dtype, device=device))  # (K,L,r)
            self.register_buffer("q_Ddiag", torch.full((K, L), float(b0), dtype=dtype, device=device))  # (K,L)
        self._stats_initialised = False
        self._refresh_cache()

    # ------------------------------------------------------------------ utils
    def _refresh_cache(self):
        """Recompute V = lam^{-1} from lam, with a Cholesky for numerical solves."""
        eye = torch.eye(self.G, device=self.lam.device, dtype=self.lam.dtype)
        lam = self.lam + self.jitter * eye
        chol = torch.linalg.cholesky(lam)                       # (K,G,G)
        self.V = torch.cholesky_inverse(chol)                   # (K,G,G)
        self._lam_chol = chol

    @property
    def Omega(self) -> torch.Tensor:
        """E[Q_l^{-1}] diagonal = a/b  ->  (K, L)."""
        return self.a / self.b

    def E_logdet_prec(self) -> torch.Tensor:
        """sum_i E[log tau_{l,i}] = sum_i (digamma(a) - log b)  ->  (K,)."""
        return (torch.digamma(self.a) - torch.log(self.b)).sum(-1)

    # -------------------------------------------------------- local evidence
    def expected_loglik(self, z: torch.Tensor, g: torch.Tensor,
                        z_var: torch.Tensor = None, g_var: torch.Tensor = None,
                        diag_score: bool = False) -> torch.Tensor:
        """E_q[ log N(z_t; Gamma_l g_t, Q_l) ] for every regime l.

        z : (..., L)   one-step targets z_t
        g : (..., G)   regressors g_t = [z_{t-1}; P h_t; 1]
        z_var: (..., L) optional diagonal posterior variance Var_q(z_t). When given, `z` is
               the posterior MEAN of the target.
        g_var: (..., G) optional diagonal posterior variance Var_q(g_t). Only the first
               L entries are normally non-zero, corresponding to uncertainty in z_{t-1};
               the projected carry and bias are deterministic. When supplied, the evidence
               is the exact factorised-Gaussian VB expectation over both TARGET and
               REGRESSOR:
                   E_{q(z_t)q(g_t)q(theta_l)}[log p(z_t | g_t, theta_l)].
               This removes the remaining deterministic-regressor approximation.
        returns (..., K) the local evidence psi_t(l) used by forward-backward.

        E_q[log N] = -L/2 log(2pi) + 1/2 sum_i (psi(a)-log b)
                     - 1/2 sum_i (a/b) E[(z_i - M_{l,i}.g)^2]
                     - L/2 E[g^T V_l g].
        """
        # predicted means per regime: mu[...,l,i] = M_l[i,:] . g
        # M: (K,L,G); g: (...,G) -> (...,K,L)
        mu = torch.einsum("klg,...g->...kl", self.M, g)
        # fluctuation g^T V_l g  -> (...,K)
        Vg = torch.einsum("kgh,...h->...kg", self.V, g)        # (...,K,G)
        gVg = torch.einsum("...g,...kg->...k", g, Vg)          # (...,K)
        # Full factorised-Gaussian expectation over the regressor.  With
        # g_var=None this reduces exactly to the older deterministic-regressor path.
        if g_var is not None:
            # E[g^T V_l g] = mu_g^T V_l mu_g + tr(V_l diag Var[g]).
            Vdiag = torch.diagonal(self.V, dim1=-2, dim2=-1)    # (K,G)
            gVg = gVg + torch.einsum("kg,...g->...k", Vdiag, g_var)
        if self.q_rank > 0:
            infl = (1.0 + gVg).clamp(max=self.infl_max)         # (...,K), capped
            if diag_score:
                # fast diagonal-marginal evidence for the structure search (birth/merge/
                # delete): scoring K is insensitive to the off-diagonal noise, so use
                # var = diag(Q_l) = (1+gVg)(D_l + sum_r U_lr^2) and a diagonal log-density,
                # which is O(K L) instead of the O(K L r^2) Woodbury path. The full
                # low-rank Q is still fit by the online M-step.
                var = infl.unsqueeze(-1) * (self.q_Ddiag + (self.Ufac ** 2).sum(-1))  # (...,K,L)
                mean_resid2 = (z.unsqueeze(-2) - mu) ** 2
                if g_var is not None:
                    M2gvar = torch.einsum("klg,...g->...kl", self.M ** 2, g_var)
                    mean_resid2 = mean_resid2 + M2gvar
                quad = (mean_resid2 / var).sum(-1)
                if z_var is not None:
                    quad = quad + (z_var.unsqueeze(-2) / var).sum(-1)
                return -0.5 * (self.L * LOG2PI + torch.log(var).sum(-1) + quad)
            # full low-rank-plus-diagonal emission: log N(z; mu_l, (1+gVg)(D_l + U_l U_l^T))
            from .lowrank import lowrank_logpdf, lowrank_inv_diag, lowrank_quadform_cols
            d = infl.unsqueeze(-1) * self.q_Ddiag              # (...,K,L)
            U = infl.clamp(min=1e-6).sqrt().unsqueeze(-1).unsqueeze(-1) * self.Ufac  # (...,K,L,r)
            zc = z.unsqueeze(-2).expand_as(mu)
            ll = lowrank_logpdf(zc, mu, d, U)                  # (...,K)
            inv_diag = None
            if z_var is not None or g_var is not None:
                inv_diag = lowrank_inv_diag(d, U)              # (...,K,L)
            if z_var is not None:
                # -1/2 tr(Q_l^{-1} diag(Var_q(z_t)))
                tr = (inv_diag * z_var.unsqueeze(-2)).sum(-1)
                ll = ll - 0.5 * tr
            if g_var is not None:
                # Exact regressor-uncertainty trace for the low-rank residual model:
                #   tr(Q_l^{-1} M_l diag(g_var) M_l^T) = sum_g g_var_g m_{l,g}^T Q_l^{-1} m_{l,g},
                # since M_l diag(g_var) M_l^T = sum_g g_var_g m_{l,g} m_{l,g}^T. Because that
                # outer-product matrix is dense, the OFF-diagonal of Q_l^{-1} contributes; the
                # Woodbury column quadratic captures it in full. Reduces to the diagonal
                # q_rank=0 term (inv_diag . M^2 g_var) when U has no columns.
                colQ = lowrank_quadform_cols(d, U, self.M)        # (...,K,G)
                tr_g = torch.einsum("...g,...kg->...k", g_var, colQ)
                ll = ll - 0.5 * tr_g
            return ll
        resid2 = (z.unsqueeze(-2) - mu) ** 2                    # (...,K,L)
        if g_var is not None:
            # E[(z_i - M_i^T g)^2] adds M_i^T Var[g] M_i.  The encoder posterior is
            # factorised across time, so Cov(z_t, z_{t-1})=0 in this amortised VB family.
            resid2 = resid2 + torch.einsum("klg,...g->...kl", self.M ** 2, g_var)
        prec = self.Omega                                      # (K,L)
        quad = (prec * resid2).sum(-1)                         # (...,K)
        if z_var is not None:
            # E_q[(z-mu)^2] adds Var_q to the residual: + sum_i Omega_{l,i} Var_q_i
            quad = quad + (prec * z_var.unsqueeze(-2)).sum(-1)
        elogdet = self.E_logdet_prec()                         # (K,)
        out = (
            -0.5 * self.L * LOG2PI
            + 0.5 * elogdet
            - 0.5 * quad
            - 0.5 * self.L * gVg
        )
        return out

    # ----------------------------------------------- predictive (mixture prior)
    def predictive(self, g: torch.Tensor):
        """Posterior-predictive mean and DIAGONAL covariance per regime.

        g : (..., G)
        returns mean (..., K, L) and var (..., K, L) where
            mean_{l} = M_l g
            var_{l,i}(g) = (1 + g^T V_l g) * E[q_{l,i}],   E[q] = b/(a-1).
        """
        mean = torch.einsum("klg,...g->...kl", self.M, g)      # (...,K,L)
        Vg = torch.einsum("kgh,...h->...kg", self.V, g)
        gVg = torch.einsum("...g,...kg->...k", g, Vg)          # (...,K)
        if self.q_rank > 0:
            # marginal diag of (1+gVg)(D_l + U_l U_l^T) = (1+gVg)(d_l + sum_r U_{l,:,r}^2)
            marg = self.q_Ddiag + (self.Ufac ** 2).sum(-1)     # (K,L)
            var = (1.0 + gVg).clamp(max=self.infl_max).unsqueeze(-1) * marg   # (...,K,L)
            return mean, var
        Eq = self.b / torch.clamp(self.a - 1.0, min=1e-4)      # (K,L) = E[q]
        var = (1.0 + gVg).clamp(max=self.infl_max).unsqueeze(-1) * Eq         # (...,K,L)
        return mean, var

    def predictive_moments(self, g: torch.Tensor, g_var: torch.Tensor = None):
        """Posterior-predictive Gaussian moments allowing uncertain regressors.

        If g_var is supplied, the returned variance integrates q(g) as well as q(theta):
            Var[z_i] = E_g Var[z_i|g,Data] + Var_g E[z_i|g,Data].
        With g_var=None it is exactly `predictive(g)`.  This is used by imagination when
        the previous latent state carries a variance, keeping rollout and training on the
        same Bayesian moment model rather than treating z_{t-1} as deterministic.
        """
        mean, var = self.predictive(g)
        if g_var is None:
            return mean, var
        # E_g[g^T Vg] replaces g^T Vg inside the posterior-predictive covariance.
        Vdiag = torch.diagonal(self.V, dim1=-2, dim2=-1)        # (K,G)
        extra_gVg = torch.einsum("kg,...g->...k", Vdiag, g_var) # (...,K)
        if self.q_rank > 0:
            marg = self.q_Ddiag + (self.Ufac ** 2).sum(-1)
            var = var + extra_gVg.unsqueeze(-1) * marg
        else:
            Eq = self.b / torch.clamp(self.a - 1.0, min=1e-4)
            var = var + extra_gVg.unsqueeze(-1) * Eq
        # Var_g[M g]
        var = var + torch.einsum("klg,...g->...kl", self.M ** 2, g_var)
        return mean, var.clamp_min(1e-8)

    def predictive_cov_moments(self, g: torch.Tensor, g_var: torch.Tensor = None):
        """Full-covariance-factor counterpart of predictive_moments, EXACT for q_rank>0.

        The regressor-integrated predictive covariance is
            E_g[Q(g)] + M diag(g_var) M^T,
        where the MEAN-regressor Student-t factor (1+E[gVg])(diag(D)+U_fac U_fac^T) comes
        from predictive_cov, and the regressor-UNCERTAINTY term adds only:
          - the diagonal inflation extra_gVg * D of the diagonal noise;
          - the cross-output columns M diag(sqrt(g_var)) (= M diag(g_var) M^T).
        It does NOT scale the factor U_fac U_fac^T (round-12 review item 11: f indep. tau).
        The Dreamer low-rank sampler reads the factor width dynamically, so the wider U is
        drawn correctly. For q_rank==0 the covariance is genuinely diagonal and the
        off-diagonal cannot be represented in a zero-column factor, so that path keeps the
        diagonal moment (matching predictive_moments).
        """
        mean, d, U = self.predictive_cov(g)
        if g_var is None:
            return mean, d, U
        Vdiag = torch.diagonal(self.V, dim1=-2, dim2=-1)
        extra_gVg = torch.einsum("kg,...g->...k", Vdiag, g_var)
        if self.q_rank > 0:
            # ROUND-12 review, item 11: regressor UNCERTAINTY (g_var) inflates the diagonal
            # noise (extra_gVg * q_Ddiag) and adds the mean-map cross-output covariance
            # M diag(g_var) M^T (U_extra), but does NOT re-scale the factor covariance
            # U0 U0^T -- the factor f is independent of the noise precision tau. The old
            # U_infl append (sqrt(extra_gVg) * Ufac) wrongly scaled E[U U^T] and is removed.
            d = d + extra_gVg.unsqueeze(-1) * self.q_Ddiag                     # inflate diag noise
            U_extra = torch.einsum("klg,...g->...klg", self.M, g_var.clamp_min(0.0).sqrt())    # (...,K,L,G)
            U = torch.cat([U, U_extra], dim=-1)
        else:
            Eq = self.b / torch.clamp(self.a - 1.0, min=1e-4)
            d = d + extra_gVg.unsqueeze(-1) * Eq
            d = d + torch.einsum("klg,...g->...kl", self.M ** 2, g_var)         # diagonal-only fallback
        return mean, d.clamp_min(1e-8), U

    def predictive_cov(self, g: torch.Tensor):
        """Posterior-predictive mean and FULL covariance FACTORS per regime.

        Returns (mean, d, U) with the per-regime predictive covariance
            Cov_l(g) = diag(d_l) + U_l U_l^T = (1 + g^T V_l g)(diag(D_l) + U_l U_l^T),
        i.e. the same (1+gVg)-inflated low-rank-plus-diagonal process noise the emission
        likelihood uses, not just its diagonal marginal. For q_rank == 0, U has zero
        columns and d is the diagonal predictive variance (so the low-rank KL path reduces
        to the diagonal one). This is what feeds the Woodbury-based mixture KL.
        """
        mean = torch.einsum("klg,...g->...kl", self.M, g)         # (...,K,L)
        Vg = torch.einsum("kgh,...h->...kg", self.V, g)
        gVg = torch.einsum("...g,...kg->...k", g, Vg)             # (...,K)
        infl = (1.0 + gVg).clamp(min=1e-6, max=self.infl_max)     # (...,K), capped
        if self.q_rank > 0:
            d = infl.unsqueeze(-1) * self.q_Ddiag                 # (...,K,L)
            U = infl.sqrt().unsqueeze(-1).unsqueeze(-1) * self.Ufac  # (...,K,L,r)
        else:
            Eq = self.b / torch.clamp(self.a - 1.0, min=1e-4)     # (K,L)
            d = infl.unsqueeze(-1) * Eq                           # (...,K,L)
            U = d.new_zeros(d.shape + (0,))                       # (...,K,L,0)
        return mean, d, U

    # ------------------------------------------------------ sufficient stats
    def stats_from_batch(self, resp: torch.Tensor, z: torch.Tensor, g: torch.Tensor,
                         z_var: torch.Tensor = None, g_z_var: torch.Tensor = None):
        """Responsibility-weighted sufficient statistics for one batch.

        resp : (..., Kr) q(s_t = l), nonnegative, sums to 1 over last dim. Kr need not
               equal self.K: birth_move passes K+1 columns to seed a candidate.
        z    : (..., L)  targets.
        g    : (..., G)  regressors.
        z_var: (..., L)  optional diagonal posterior variance of the TARGET z_t. When given,
               `z` is treated as the posterior MEAN and the second-moment statistics use the
               exact Gaussian moment E[z z^T] = z z^T + diag(z_var) instead of the outer
               product of a single reparameterised sample (Rao-Blackwellised / analytic-VB
               M-step). Without the diag(z_var) term the regime noise is underestimated.
        g_z_var: (..., L) optional diagonal posterior variance of the REGRESSOR z-block
               z_{t-1} (the first L entries of g; the projected-deter and bias entries are
               deterministic). When given, the Gram matrix uses E[g g^T] with diag(Var z_{t-1})
               added to its z-block, i.e. it accounts for the uncertainty of the regressor, not
               only the target. This is the fixable half of the input-uncertainty correction:
               it shrinks the fitted M toward smaller magnitudes when z_{t-1} is uncertain. The
               cross moment E[z_t z_{t-1}^T] in Szg stays mu_t mu_{t-1}^T, because under the
               factorised encoder posterior Cov(z_t, z_{t-1}) = 0 and that term is not
               recoverable without a smoothed joint posterior.
        returns dict of (N, Sgg, Szg, Szz [, Szz_full]) summed over all leading dims.
        """
        Kr = resp.shape[-1]
        r = resp.reshape(-1, Kr)                               # (M,Kr)
        zf = z.reshape(-1, self.L)                              # (M,L)
        gf = g.reshape(-1, self.G)                              # (M,G)
        N = r.sum(0)                                            # (Kr,)
        # Sgg_l = sum_m r_{m,l} g_m g_m^T
        Sgg = torch.einsum("mk,mg,mh->kgh", r, gf, gf)         # (Kr,G,G)
        Szg = torch.einsum("mk,mi,mg->kig", r, zf, gf)         # (Kr,L,G)
        Szz = torch.einsum("mk,mi->ki", r, zf * zf)            # (Kr,L)
        out = dict(N=N, Sgg=Sgg, Szg=Szg, Szz=Szz)
        if self.q_rank > 0:
            out["Szz_full"] = torch.einsum("mk,mi,mj->kij", r, zf, zf)  # (Kr,L,L)
        if z_var is not None:
            vf = z_var.reshape(-1, self.L)
            rv = torch.einsum("mk,mi->ki", r, vf)              # sum_m r_{m,l} var_{m,i}
            out["Szz"] = out["Szz"] + rv                       # diag E[z^2] = mu^2 + var
            if self.q_rank > 0:
                out["Szz_full"] = out["Szz_full"] + torch.diag_embed(rv)
        if g_z_var is not None:
            gv = g_z_var.reshape(-1, self.L)
            rgv = torch.einsum("mk,mi->ki", r, gv)             # (Kr,L) regressor z-block var
            out["Sgg"][:, :self.L, :self.L] = (
                out["Sgg"][:, :self.L, :self.L] + torch.diag_embed(rgv))
        return out

    def set_stats(self, stats):
        """Overwrite EMA statistics (used for a full-batch / analytic fit)."""
        self.N.copy_(stats["N"])
        self.Sgg.copy_(stats["Sgg"])
        self.Szg.copy_(stats["Szg"])
        self.Szz.copy_(stats["Szz"])
        if self.q_rank > 0:
            self.Szz_full.copy_(stats["Szz_full"])
        self._stats_initialised = True

    def ema_update_stats(self, stats, tau: float):
        """Exponential moving average of the sufficient statistics (SVI).

        On the first call (or if not yet initialised) the batch statistics are
        adopted directly so the running average is not biased toward the prior-empty
        zero state.
        """
        if not self._stats_initialised:
            self.set_stats(stats)
            return
        pairs = [("N", self.N), ("Sgg", self.Sgg), ("Szg", self.Szg), ("Szz", self.Szz)]
        if self.q_rank > 0:
            pairs.append(("Szz_full", self.Szz_full))
        for name, S in pairs:
            S.mul_(1.0 - tau).add_(tau * stats[name])

    # --------------------------------------------------------- closed-form M-step
    @torch.no_grad()
    def m_step(self):
        """Closed-form MNIW (diagonal) update of the posterior from current stats."""
        eyeG = torch.eye(self.G, device=self.M.device, dtype=self.M.dtype)
        lam0 = torch.diag(self.lam0_diag)                       # (G,G)

        # Lambda_l = Lambda0 + Sgg_l
        lam = lam0.unsqueeze(0) + self.Sgg                      # (K,G,G)
        lam = lam + self.jitter * eyeG
        chol = _chol_jitter(lam)                        # (K,G,G)
        V = torch.cholesky_inverse(chol)                       # (K,G,G)

        # M_l = (M0 Lambda0 + Szg_l) Lambda_l^{-1}
        rhs = (self.M0 @ lam0).unsqueeze(0) + self.Szg         # (K,L,G)
        M = torch.einsum("klg,kgh->klh", rhs, V)               # (K,L,G)

        # a_{l,i} = a0 + N_l/2  (broadcast over i)
        a = self.a0 + 0.5 * self.N.unsqueeze(-1)               # (K,L)
        a = a.expand(self.K, self.L).clone()

        # b_{l,i} = b0 + 0.5 ( Szz_{l,i} + m0_i^T Lambda0 m0_i - M_{l,i}^T Lambda_l M_{l,i} )
        m0lam0m0 = torch.einsum("ig,gh,ih->i", self.M0, lam0, self.M0)        # (L,)
        # M_{l,i}^T Lambda_l M_{l,i}
        lamM = torch.einsum("kgh,klh->klg", lam, M)                            # (K,L,G)
        MlamM = torch.einsum("klg,klg->kl", M, lamM)                           # (K,L)
        b = self.b0 + 0.5 * (self.Szz + m0lam0m0.unsqueeze(0) - MlamM)
        b = torch.clamp(b, min=1e-6)

        self.M.copy_(M)
        self.lam.copy_(lam)
        # degenerate-collapse safeguard: cap E[tau_{l,i}] = a/b at 1/noise_var_floor.
        # A junk state that transiently captures near-duplicate points under structure
        # churn drives its residual to ~0 and KL(q(tau)||p(tau)) to ~1e9 for one lap
        # (measured: the ENTIRE ARD-on ELBO spike in the Lorenz component probe was
        # this single term; logZ/alloc/beta_kl moved by nats). An absolute floor on b
        # cannot help: with a = a0 + N/2 ~ 1e3, E[tau] = a/b still reaches 1e9 at
        # b = 1e-6. The floor must be RELATIVE to a. Inactive on healthy fits
        # (floor std 1e-3 in standardized units, ~3 orders below fitted Q here).
        b = torch.maximum(b, a * self.noise_var_floor)
        self.a.copy_(a)
        self.b.copy_(b)
        self.V.copy_(V)
        self._lam_chol = chol

        if self.ard:
            self._ard_step()
        if self.q_rank > 0:
            self._fit_lowrank_Q(M)

    @torch.no_grad()
    def _fit_lowrank_Q(self, M):
        """ML factor-analysis fit of Q_l = diag(d_l) + U_l U_l^T from the weighted
        residual covariance S_l = E_l[(z - M_l g)(z - M_l g)^T].

        S_l = (Szz_full - Szg M^T - M Szg^T + M Sgg M^T) / N_l. We take the top-q_rank
        eigenvectors for U_l and set d_l = diag(S_l) - diag(U_l U_l^T) (>= floor), so the
        marginal diag(Q_l) reproduces diag(S_l) exactly while q_rank directions capture the
        dominant residual correlations.
        """
        N = self.N.clamp(min=1.0)
        SzgMt = torch.einsum("klg,kmg->klm", self.Szg, M)                 # Szg M^T (K,L,L)
        MSgg = torch.einsum("klg,kgh->klh", M, self.Sgg)
        MSggMt = torch.einsum("klh,kmh->klm", MSgg, M)                    # M Sgg M^T
        S = (self.Szz_full - SzgMt - SzgMt.transpose(-1, -2) + MSggMt) / N.view(-1, 1, 1)
        eyeL = torch.eye(self.L, device=S.device, dtype=S.dtype)
        S = 0.5 * (S + S.transpose(-1, -2)) + self.jitter * eyeL          # symmetrise + jitter
        evals, evecs = torch.linalg.eigh(S)                              # ascending
        top_val = evals[..., -self.q_rank:].clamp(min=0.0)               # (K,r)
        top_vec = evecs[..., -self.q_rank:]                              # (K,L,r)
        U = top_vec * top_val.sqrt().unsqueeze(-2)                       # (K,L,r)
        d = torch.diagonal(S, dim1=-2, dim2=-1) - (U ** 2).sum(-1)       # (K,L)
        d = d.clamp(min=1e-4)
        # regimes with no mass keep an isotropic prior-scale diagonal and zero factor
        empty = (self.N <= 1e-6)
        if empty.any():
            U[empty] = 0.0
            d[empty] = float(self.b0)
        self.Ufac.copy_(U)
        self.q_Ddiag.copy_(d)

    @torch.no_grad()
    def _ard_step(self):
        """Shared-ARD MacKay fixed point for the diagonal of Lambda0.

        alpha_j <- (K L) / sum_k ( L [V_k]_jj + [M_k]_{:,j}^T Omega_k [M_k]_{:,j} ).
        Uses the same L*[V]_jj fluctuation term as the manuscript (consistency
        with Xi_k = M^T Omega M + L V).
        """
        Vdiag = torch.diagonal(self.V, dim1=-2, dim2=-1)        # (K,G)
        # [M_k]_{:,j}^T Omega_k [M_k]_{:,j} = sum_i Omega_{k,i} M_{k,i,j}^2
        Om = self.Omega                                        # (K,L)
        Mj2 = self.M ** 2                                      # (K,L,G)
        weighted = torch.einsum("ki,kig->kg", Om, Mj2)
        # MacKay evidence counting over OCCUPIED states only: empty states have
        # posterior == prior (V ~ Lambda0^{-1}, M ~ M0), so including them creates a
        # ratchet alpha -> K L / (E L / alpha + S_occ) that diverges to the clamp as
        # structural moves leave transiently empty rows. EB on zero-count states is
        # undefined; they keep the shared prior and contribute no evidence.
        occ = self.N > 1.0
        n_occ = int(occ.sum())
        if n_occ == 0:
            return         # (K,G)
        denom = (self.L * Vdiag + weighted)[occ].sum(0)             # (G,)
        alpha = (n_occ * self.L) / torch.clamp(denom, min=1e-8)
        # keep ARD in a sane range
        alpha = torch.clamp(alpha, min=1e-4, max=1e3)
        self.lam0_diag.copy_(alpha)

    @torch.no_grad()
    def param_kl(self) -> torch.Tensor:
        """Per-regime KL( q(theta_l) || p(theta_l) ) for the diagonal Normal-Gamma model.

        Factorises per output dim i into a Gamma KL on tau_{l,i} plus a Gaussian KL on
        w_{l,i} | tau (which shares the design precision Lambda_l across i). Returns (K,).
        Used by birth/merge/delete as the parameter-complexity side of the bound.
        """
        K, L, G = self.K, self.L, self.G
        lam0 = self.lam0_diag                                   # (G,)
        chol = self._lam_chol                                   # (K,G,G) = chol(Lambda_l)
        logdet_lam = 2.0 * torch.log(
            torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)     # (K,)
        logdet_lam0 = torch.log(lam0).sum()                    # scalar
        Vdiag = torch.diagonal(self.V, dim1=-2, dim2=-1)       # (K,G)
        tr_term = (lam0.unsqueeze(0) * Vdiag).sum(-1)          # (K,) = tr(Lam0 Lam_l^{-1})

        diff = self.M - self.M0.unsqueeze(0)                   # (K,L,G)
        maha = (lam0.view(1, 1, G) * diff ** 2).sum(-1)        # (K,L) = (M-m0)^T Lam0 (M-m0)
        Etau = self.a / self.b                                 # (K,L)
        # Gaussian KL per (l,i): logdet/tr shared across i, mean term per i
        gauss_kl = (0.5 * (logdet_lam - logdet_lam0 - G + tr_term).unsqueeze(-1)
                    + 0.5 * Etau * maha)                       # (K,L)
        # Gamma KL per (l,i): KL( Gamma(a_q,b_q) || Gamma(a_p,b_p) ), rate parameterisation
        a_q, b_q = self.a, self.b
        a_p, b_p = self.a0, self.b0
        gamma_kl = ((a_q - a_p) * torch.digamma(a_q)
                    - torch.lgamma(a_q) + torch.lgamma(a_p)
                    + a_p * (torch.log(b_q) - torch.log(b_p))
                    + a_q * (b_p - b_q) / b_q)                 # (K,L)
        return (gauss_kl + gamma_kl).sum(-1)                   # (K,)

    @torch.no_grad()
    def data_elbo_from_stats(self) -> torch.Tensor:
        """Summed expected data log-likelihood as a PURE FUNCTION of the cached statistics.

        For the diagonal model (q_rank == 0) this is exactly
            sum_t sum_k r_tk E_q[log N(z_t; M_k g_t, Q_k)]
        evaluated from the installed sufficient statistics and posterior:

            sum_{k,i} [ -N_k/2 log 2pi + N_k/2 (psi(a_ki) - log b_ki)
                        - (a/b)_ki/2 (Szz_ki - 2 <M_ki, Szg_ki> + M_ki Sgg_k M_ki^T) ]
            - 1/2 sum_k L * tr(V_k Sgg_k),

        identical (to numerical precision) to summing `expected_loglik` weighted by
        the responsibilities that produced the stats, because z_var / g_var
        corrections are already folded into Szz / Sgg by `stats_from_batch`. This is
        the Ldata term of Hughes et al. (NIPS 2015)'s entropy-free merge-selection
        test, computable without touching the data. For q_rank > 0 the Woodbury
        evidence is not linear in these statistics; the diagonal-marginal surrogate
        (var = D + diag(U U^T), no (1+gVg) inflation) is returned instead, which is
        used only for the merge SHORTLIST; acceptance always re-verifies with the
        exact bound on the buffer.
        """
        N = self.N.clamp_min(0.0)                              # (K,)
        elog = torch.digamma(self.a) - torch.log(self.b)       # (K,L)
        MSzg = (self.M * self.Szg).sum(-1)                     # (K,L) <M_ki, Szg_ki>
        MSggM = torch.einsum("klg,kgh,klh->kl", self.M, self.Sgg, self.M)  # (K,L)
        if self.q_rank > 0:
            var = self.q_Ddiag + (self.Ufac ** 2).sum(-1)      # (K,L) diagonal surrogate
            quad = (self.Szz - 2.0 * MSzg + MSggM) / var
            per = (-0.5 * N.unsqueeze(-1) * LOG2PI
                   - 0.5 * N.unsqueeze(-1) * torch.log(var)
                   - 0.5 * quad)
            return per.sum()
        Etau = self.a / self.b                                 # (K,L)
        per = (-0.5 * N.unsqueeze(-1) * LOG2PI
               + 0.5 * N.unsqueeze(-1) * elog
               - 0.5 * Etau * (self.Szz - 2.0 * MSzg + MSggM))
        Vdiag_tr = torch.einsum("kgh,khg->k", self.V, self.Sgg)  # tr(V_k Sgg_k)
        return per.sum() - 0.5 * self.L * Vdiag_tr.sum()

    @torch.no_grad()
    def clone_with_K(self, new_K: int, stats=None):
        """A fresh DiagARRegimes at a different K with the SAME prior (a0,b0,Lambda0,M0).

        If `stats` (dict of N,Sgg,Szg,Szz at the new K) is given, they are installed and
        a closed-form M-step is run so the clone is ready to score. Used by the moves.
        """
        new = DiagARRegimes(
            K=new_K, L=self.L, G=self.G,
            a0=float(self.a0), b0=float(self.b0), v0_scale=1.0,
            ard=False, identity_init=False, jitter=self.jitter,
            q_rank=self.q_rank,
            device=self.M.device, dtype=self.M.dtype,
        )
        # copy the (possibly ARD-updated) prior exactly; candidate keeps it frozen
        # so the bound comparison is against a fixed prior
        new.lam0_diag.copy_(self.lam0_diag)
        new.M0.copy_(self.M0)
        if stats is not None:
            new.set_stats(stats)
            new.m_step()
        return new

    # ------------------------------------------------------------------ convenience
    @torch.no_grad()
    def fit_full_batch(self, resp, z, g, n_iter: int = 1):
        """Convenience: set stats from a batch and run the closed-form M-step.

        For ARD, a few iterations let Lambda0 and the posterior co-adapt.
        """
        stats = self.stats_from_batch(resp, z, g)
        self.set_stats(stats)
        for _ in range(max(1, n_iter)):
            self.m_step()

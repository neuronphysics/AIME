"""Per-regime Bayesian linear-Gaussian dynamics with a SHARED carry drift (tied C).

This is the collapse-resistant dynamics of the manuscript (Sec. 3, Prop. 1/2, Sec. 5.5).
The one-step latent transition is

    z_t = Gamma_l r_t + C h~_t + eps,   eps ~ N(0, Q_l),  Q_l = diag(q_{l,1..L}),

with the REGIME regressor r_t = [z_{t-1}; 1] in R^{Lr}, Lr = L+1, the regime map
Gamma_l = [A_l, b_l] in R^{L x Lr}, and a SINGLE carry-drift map C in R^{L x H'} that is
TIED across regimes (outside the regime plate). The carry h~_t = P h_t enters every
component identically, so it cannot launder a regime-specific linear readout of the GRU
state (Prop. 1); the regimes carry only the residual affine dynamics that differ across
behaviours, which is what the sticky prior persists (Remark 2).

Drop-in for `DiagARRegimes`: it accepts the SAME stacked regressor g_t = [z_{t-1}; h~_t; 1]
that `RegimeHead.build_g` already produces, and splits it internally into r_t and h~_t. The
public surface (`expected_loglik`, `predictive`, `predictive_cov`, `stats_from_batch`,
`set_stats`, `ema_update_stats`, `m_step`, `param_kl`, `clone_with_K`, `Omega`,
`E_logdet_prec`, and the N/Sgg/Szg/Szz buffers the moves read) matches `DiagARRegimes`,
so `RegimeHead`, `mixture_prior`, and `moves` need no change beyond selecting this class.

Globals and their updates (all closed-form, no grad, EMA sufficient statistics):

* Regime blocks Gamma_l, Q_l : diagonal MNIW (Normal-Gamma) regression of the CARRY
  RESIDUAL z~_t = z_t - C h~_t on r_t, with responsibility weights (Eqs 24-26).
* Shared drift C : a row-factorised variational GAUSSIAN posterior q(C_i)=N(m^C_i, V^C_i),
  the exact CAVI coordinate (Eqs 29-31): a precision-weighted Bayesian ridge of the regime
  residual d_{t,l}=z_t-Gamma_l r_t on h~_t, summed over regimes with weights r_t(l) ω_{l,i}.
  It is a distribution, not a MAP point: V^C_i is kept and feeds the predictive variance
  (Eq 34) and the local evidence (Eq 20).

The two blocks couple through sum_t r_t(l) h~_t r_t^T, so m_step alternates them a few times
(block coordinate ascent on the same concave quadratic, Prop. 2). Diagonal Q only.
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


class SharedCarryRegimes(nn.Module):
    def __init__(
        self,
        K: int,
        L: int,
        G: int,                       # full stacked-regressor dim = L + Hp + 1
        a0: float = 3.0,
        b0: float = 2.0,
        v0_scale: float = 1.0,        # prior precision of the regime maps (Lambda0 diag)
        vC0_scale: float = 1.0,       # prior precision of the shared-C rows
        ard: bool = True,
        identity_init: bool = True,
        jitter: float = 1e-6,
        q_rank: int = 0,              # only diagonal supported on the shared-carry path
        action_dim: int = 0,          # action joins the regime regressor r
        infl_max: float = 10.0,
        n_block_iters: int = 3,       # block-coordinate (regime <-> C) sweeps per m_step
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float32,
    ):
        super().__init__()
        self.K, self.L, self.G = K, L, G
        self.action_dim = int(action_dim)
        self.Hp = G - L - self.action_dim - 1    # carry-projection dim H'
        assert self.Hp >= 0, "G must be >= L+1 (regressor [z; h~; 1])"
        self.Lr = L + self.action_dim + 1        # regime regressor [z_{t-1}; a_{t-1}; 1]
        self.ard = ard
        self.noise_var_floor = 1e-6  # E[tau] cap = 1/floor; Q-collapse safeguard (see m_step)
        self.jitter = jitter
        self.q_rank = int(q_rank)                # 0 = diagonal Q; >0 = diag(d)+U U^T on the C-residual
        self.infl_max = float(infl_max)
        self.n_block_iters = int(n_block_iters)

        # ---- regime prior hyperparameters ----
        self.register_buffer("a0", torch.tensor(float(a0), dtype=dtype, device=device))
        self.register_buffer("b0", torch.tensor(float(b0), dtype=dtype, device=device))
        self.register_buffer("lam0_diag", torch.full((self.Lr,), float(v0_scale), dtype=dtype, device=device))
        M0 = torch.zeros(L, self.Lr, dtype=dtype, device=device)
        if identity_init:
            M0[:, :L] = torch.eye(L, dtype=dtype, device=device)   # z_t ~ z_{t-1} prior
        self.register_buffer("M0", M0)

        # ---- regime posterior ----
        self.register_buffer("M", M0.clone().unsqueeze(0).repeat(K, 1, 1))                 # (K,L,Lr)
        self.register_buffer("lam", torch.diag_embed(self.lam0_diag).unsqueeze(0).repeat(K, 1, 1))  # (K,Lr,Lr)
        self.register_buffer("V", torch.diag_embed(1.0 / self.lam0_diag).unsqueeze(0).repeat(K, 1, 1))
        self.register_buffer("a", torch.full((K, L), float(a0), dtype=dtype, device=device))
        self.register_buffer("b", torch.full((K, L), float(b0), dtype=dtype, device=device))

        # ---- shared carry drift C : row-factorised Gaussian q(C_i)=N(m^C_i, V^C_i) ----
        self.register_buffer("vC0", torch.tensor(float(vC0_scale), dtype=dtype, device=device))
        self.register_buffer("Cmean", torch.zeros(L, self.Hp, dtype=dtype, device=device))    # (L,Hp)
        self.register_buffer("Ccov", (1.0 / float(vC0_scale)) *
                             torch.eye(self.Hp, dtype=dtype, device=device).unsqueeze(0).repeat(L, 1, 1))  # (L,Hp,Hp)
        self.register_buffer("C0mean", torch.zeros(L, self.Hp, dtype=dtype, device=device))

        # ---- raw EMA sufficient statistics (responsibility-weighted, per regime) ----
        self.register_buffer("N", torch.zeros(K, dtype=dtype, device=device))                 # (K,)
        self.register_buffer("Srr", torch.zeros(K, self.Lr, self.Lr, dtype=dtype, device=device))   # Σ r̂ r r^T
        self.register_buffer("Szr", torch.zeros(K, L, self.Lr, dtype=dtype, device=device))    # Σ r̂ z r^T
        self.register_buffer("Szz", torch.zeros(K, L, dtype=dtype, device=device))             # Σ r̂ z⊙z (diag)
        self.register_buffer("Shh", torch.zeros(K, self.Hp, self.Hp, dtype=dtype, device=device))   # Σ r̂ h~ h~^T
        self.register_buffer("Szh", torch.zeros(K, L, self.Hp, dtype=dtype, device=device))    # Σ r̂ z_i h~
        self.register_buffer("Srh", torch.zeros(K, self.Lr, self.Hp, dtype=dtype, device=device))   # Σ r̂ r h~^T

        # ---- residualised regime stats EXPOSED to moves (Sgg/Szg/Szz at G=Lr) ----
        # refreshed at every m_step from the raw stats + current C; the moves treat these as
        # an ordinary regime regression (residual after the shared drift).
        self.register_buffer("Sgg", self.Srr.clone())            # alias: the r-Gram is C-free
        self.register_buffer("Szg", self.Szr.clone())            # = Σ r̂ z~ r^T  (C-residualised)
        self.register_buffer("Szz_resid", self.Szz.clone())      # = Σ r̂ z~⊙z~

        # ---- optional low-rank-plus-diagonal process noise Q = diag(d) + U U^T ----
        # Fit by ML factor analysis on the CARRY-RESIDUAL covariance (z~ = z - C h~), so the
        # tied drift is removed first and the regimes carry only the residual correlated noise.
        if self.q_rank > 0:
            # raw full second moment Σ r̂ z z^T, its C-residualised form, the fitted factor U
            # and diagonal d. Szz_full_resid is the (L,L) analogue of Szz_resid and is what the
            # moves read for the residual log-det when q_rank>0.
            self.register_buffer("Szz_full", torch.zeros(K, L, L, dtype=dtype, device=device))
            self.register_buffer("Szz_full_resid", torch.zeros(K, L, L, dtype=dtype, device=device))
            self.register_buffer("Ufac", torch.zeros(K, L, self.q_rank, dtype=dtype, device=device))
            self.register_buffer("q_Ddiag", torch.full((K, L), float(b0), dtype=dtype, device=device))
            # review Important #1: keep the tau-noise E[tau^-1] and the INDEPENDENT U
            # uncertainty tr Cov(U) SEPARATE, so only the noise is inflated by (1+rVr).
            self.register_buffer("_q_taudiag", torch.full((K, L), float(b0), dtype=dtype, device=device))
            self.register_buffer("_q_Udiag", torch.zeros((K, L), dtype=dtype, device=device))
            # fully variational factor augmentation : z = A r + C h~ + U f + eps,
            # f_t ~ N(0, I_F) local latents, rows u_{k,i} ~ N(0, u_prior_scale I) with Gaussian
            # posterior q(u_{k,i}) = N(Umean[k,i], Ucov[k,i]). (Ufac, q_Ddiag) above become
            # DERIVED moment-matched predictive caches; the fitting path never reads them.
            self.u_prior_scale = 0.5
            F = self.q_rank
            # small random init: Umean = 0 is a SADDLE of the block ascent (zero loadings
            # give zero factor means give zero cross statistics), exactly as in factor
            # analysis; symmetry must be broken at init, never by the updates.
            self.register_buffer("Umean", 0.01 * torch.randn(K, L, F, dtype=dtype, device=device))
            self.register_buffer("Ucov", (self.u_prior_scale
                                          * torch.eye(F, dtype=dtype, device=device)
                                          ).expand(K, L, F, F).contiguous().clone())
            # gamma-weighted local-factor sufficient statistics (all K-leading: ledger/remap safe)
            self.register_buffer("Szf", torch.zeros(K, L, F, dtype=dtype, device=device))
            # review P0 #2: the local-factor regressor stat has width Lr = L+action_dim+1
            # (the regressor r_t = [z_{t-1}; a_{t-1}; 1]); L+1 crashed when action_dim>0.
            self.register_buffer("Sfr", torch.zeros(K, F, self.Lr, dtype=dtype, device=device))
            self.register_buffer("Sfh", torch.zeros(K, F, self.Hp, dtype=dtype, device=device))
            self.register_buffer("Sff", torch.zeros(K, F, F, dtype=dtype, device=device))

        # a clone made for move scoring freezes C (moves change K; C is K-independent)
        self._freeze_C = False
        self._stats_initialised = False
        self._refresh_cache()

    # ------------------------------------------------------------------ helpers
    def _split_g(self, g):
        """g=[z; h~; 1] -> regime regressor r=[z;1] (...,Lr) and carry h~ (...,Hp)."""
        z = g[..., : self.L]
        htil = g[..., self.L : self.L + self.Hp]
        act = g[..., self.L + self.Hp : -1]      # action block (empty if action_dim=0)
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
        """E[Q_l^{-1}] diagonal = a/b -> (K, L)."""
        return self.a / self.b

    def E_logdet_prec(self) -> torch.Tensor:
        return (torch.digamma(self.a) - torch.log(self.b)).sum(-1)

    def _Ch(self, htil):
        """Mean carry drift  C̄ h~_t  -> (...,L)."""
        return torch.einsum("ih,...h->...i", self.Cmean, htil)

    def _hVCh(self, htil):
        """Per output dim i: h~^T V^C_i h~  -> (...,L). The carry-drift uncertainty term."""
        Vh = torch.einsum("ihj,...j->...ih", self.Ccov, htil)     # (...,L,Hp)
        return torch.einsum("...h,...ih->...i", htil, Vh)         # (...,L)

    # -------------------------------------------------------- local evidence
    def expected_loglik(self, z: torch.Tensor, g: torch.Tensor,
                        z_var: torch.Tensor = None, g_var: torch.Tensor = None,
                        diag_score: bool = False) -> torch.Tensor:
        """E_q[ log N(z_t; Gamma_l r_t + C h~_t, Q_l) ] for every regime l -> (...,K).

        Matches DiagARRegimes' fully-Bayesian evidence, with the regime mean Gamma_l r_t
        shifted by the shared drift C̄ h~_t, the regime-map fluctuation taken over r (not g),
        and the extra carry-drift fluctuation -1/2 Σ_i ω_{l,i} (h~^T V^C_i h~) from Eq (20).
        If g_var is supplied, the first L entries encode Var_q(z_{t-1}) and the evidence is
        the exact factorised-Gaussian expectation over both target z_t and regressor r_t.
        """
        r, htil = self._split_g(g)
        mu = torch.einsum("klr,...r->...kl", self.M, r) + self._Ch(htil).unsqueeze(-2)  # (...,K,L)
        Vr = torch.einsum("krs,...s->...kr", self.V, r)
        rVr = torch.einsum("...r,...kr->...k", r, Vr)            # (...,K) regime-map fluctuation
        r_var = None
        if g_var is not None:
            # r=[z_{t-1}; 1]; only the z block is uncertain, the bias is deterministic.
            r_var = torch.cat([g_var[..., : self.L],
                           torch.zeros_like(g[..., self.L + self.Hp :])], dim=-1)
            Vdiag = torch.diagonal(self.V, dim1=-2, dim2=-1)     # (K,Lr)
            rVr = rVr + torch.einsum("kr,...r->...k", Vdiag, r_var)
        hVCh = self._hVCh(htil)                                   # (...,L) carry-drift fluctuation
        if self.q_rank > 0 and diag_score:
            # O(KL) diagonal-marginal PROPOSAL score for the structure search only
            # (birth/merge/delete shortlists), from the moment-matched predictive cache;
            # acceptance always uses the exact bound.
            var = self.q_Ddiag + (self.Ufac ** 2).sum(-1)        # marginal diag, (K,L)
            mean_resid2 = (z.unsqueeze(-2) - mu) ** 2
            if r_var is not None:
                mean_resid2 = mean_resid2 + torch.einsum("klr,...r->...kl", self.M ** 2, r_var)
            quad = (mean_resid2 / var).sum(-1)
            if z_var is not None:
                quad = quad + (z_var.unsqueeze(-2) / var).sum(-1)
            quad = quad + (hVCh.unsqueeze(-2) / var).sum(-1)
            quad = quad + rVr * (self.q_Ddiag / var).sum(-1)
            return -0.5 * (self.L * LOG2PI + torch.log(var).sum(-1) + quad)
        prec = self.Omega                                        # (K,L)
        resid2 = (z.unsqueeze(-2) - mu) ** 2                     # (...,K,L)
        if r_var is not None:
            resid2 = resid2 + torch.einsum("klr,...r->...kl", self.M ** 2, r_var)
        quad = (prec * resid2).sum(-1)                          # (...,K)
        if z_var is not None:
            quad = quad + (prec * z_var.unsqueeze(-2)).sum(-1)
        carry_fluc = torch.einsum("kl,...l->...k", prec, hVCh)  # Σ_i ω_{l,i} h~^T V^C_i h~
        elogdet = self.E_logdet_prec()                          # (K,)
        out = (
            -0.5 * self.L * LOG2PI
            + 0.5 * elogdet
            - 0.5 * quad
            - 0.5 * self.L * rVr
            - 0.5 * carry_fluc
        )
        if self.q_rank > 0:
            # Fully variational low-rank noise by FACTOR AUGMENTATION:
            # z = A r + C h~ + U f + eps with f_t ~ N(0, I_F) local and Gaussian q(U).
            # The per-(t,k) evidence is the f-PROFILED bound
            #   ell = E_{q(f*)}[E_q log N(z; Ar+Ch+Uf, diag(1/tau))] - KL(q(f*)||N(0,I))
            # whose closed form at the optimal q(f*) = N(P^-1 b, P^-1) is the diagonal
            # evidence above PLUS  0.5 (b^T P^-1 b - log|P|):
            #   P_k = I + sum_i E[tau_ki] E[u_ki u_ki^T]   (constant over t: one KxFxF
            #         Cholesky per call -- O(K F^3), F <= 4),
            #   b_tk = sum_i E[tau_ki] E[u_ki] * mean-resid_tki   (only means couple: the
            #         q blocks are independent, so every variance correction of the
            #         diagonal path carries over unchanged).
            # At q(U) = delta(0) this reduces EXACTLY to the diagonal ELBO (tested).
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
            out = out + 0.5 * (quadf - logdetP)
        return out

    # predictive (mixture prior)
    def predictive(self, g: torch.Tensor):
        """Posterior-predictive mean and DIAGONAL covariance per regime (Eq 33/34).

        mean_l = Gamma_l r_t + C̄ h~_t
        var_{l,i} = (1 + r^T V_l r) * marg_{l,i} + (h~^T V^C_i h~),  with marg = E[q] (diagonal Q)
        or marg = d_l + sum_r U_{l,:,r}^2 (low-rank Q).
        """
        r, htil = self._split_g(g)
        mean = torch.einsum("klr,...r->...kl", self.M, r) + self._Ch(htil).unsqueeze(-2)
        Vr = torch.einsum("krs,...s->...kr", self.V, r)
        rVr = torch.einsum("...r,...kr->...k", r, Vr)            # (...,K)
        infl = (1.0 + rVr).clamp(max=self.infl_max).unsqueeze(-1)  # (...,K,1)
        carry_var = self._hVCh(htil).unsqueeze(-2)               # (...,1,L) shared across regimes
        if self.q_rank > 0:
            # q(U) is INDEPENDENT of (A,tau), so the factor
            # variance sum(Ufac^2) = E[U]E[U]^T diagonal is NOT inflated by the
            # coefficient-uncertainty (1+rVr); only the tau-scaled diagonal noise is.
            # This makes predictive() consistent with the (already-fixed) predictive_cov.
            factor = (self.Ufac ** 2).sum(-1)                    # (K,L) E[U]E[U]^T diag, unscaled
            # review Important #1: only E[tau^-1] is inflated by (1+rVr); tr Cov(U) and
            # the E[U] factor are independent of (A,tau) and stay un-inflated.
            var = infl * self._q_taudiag + self._q_Udiag + factor + carry_var
        else:
            Eq = self.b / torch.clamp(self.a - 1.0, min=1e-4)    # (K,L)
            var = infl * Eq + carry_var
        return mean, var.clamp_min(1e-8)

    def predictive_moments(self, g: torch.Tensor, g_var: torch.Tensor = None):
        """Posterior-predictive moments allowing uncertain z_{t-1} in the regressor.

        With g_var=None this is exactly predictive(g).  With g_var, the first L entries are
        Var_q(z_{t-1}); the shared carry and bias remain deterministic.  We add both
        E[regime-map parameter uncertainty] and Var_r[Gamma_l r].
        """
        mean, var = self.predictive(g)
        if g_var is None:
            return mean, var
        r_var = torch.cat([g_var[..., : self.L],
                           torch.zeros_like(g[..., self.L + self.Hp :])], dim=-1)
        Vdiag = torch.diagonal(self.V, dim1=-2, dim2=-1)        # (K,Lr)
        extra_rVr = torch.einsum("kr,...r->...k", Vdiag, r_var)
        if self.q_rank > 0:
            # review blocker 6 + Important #1: regressor uncertainty inflates the
            # tau-NOISE only, not tr Cov(U) or the E[U] factor (matches predictive_cov_moments).
            var = var + extra_rVr.unsqueeze(-1) * self._q_taudiag
        else:
            Eq = self.b / torch.clamp(self.a - 1.0, min=1e-4)
            var = var + extra_rVr.unsqueeze(-1) * Eq
        var = var + torch.einsum("klr,...r->...kl", self.M ** 2, r_var)
        return mean, var.clamp_min(1e-8)

    def predictive_cov_moments(self, g: torch.Tensor, g_var: torch.Tensor = None):
        """EXACT regressor-integrated covariance factors for q_rank>0 (see DiagARRegimes).

        E_r[Q(r)] + M diag(r_var) M^T.  The regressor-uncertainty term inflates the
        DIAGONAL noise (extra_rVr * q_Ddiag) and adds the cross-output columns
        M diag(r_var) M^T (kept in full as extra U columns); it does NOT scale the factor
        covariance U0 U0^T (f is independent of tau).
        """
        mean, d, U = self.predictive_cov(g)
        if g_var is None:
            return mean, d, U
        r_var = torch.cat([g_var[..., : self.L],
                           torch.zeros_like(g[..., self.L + self.Hp :])], dim=-1)
        Vdiag = torch.diagonal(self.V, dim1=-2, dim2=-1)
        extra_rVr = torch.einsum("kr,...r->...k", Vdiag, r_var)
        if self.q_rank > 0:
            # regressor UNCERTAINTY (r_var) propagates through the
            # mean map -- giving the cross-output term M diag(r_var) M^T (U_extra) and the
            # diagonal inflation extra_rVr * q_Ddiag -- but it must NOT re-scale the
            # independent factor covariance U0 U0^T (the factor f is independent of the
            # noise precision tau, so the E[1/tau] Student-t inflation from the MEAN
            # regressor in predictive_cov is a separate effect). The previous U_infl append
            # (sqrt(extra_rVr) * Ufac) wrongly multiplied E[U U^T] by the regressor
            # uncertainty and is removed.
            d = d + extra_rVr.unsqueeze(-1) * self._q_taudiag                  # review Important #1: tau-noise only
            U_extra = torch.einsum("klr,...r->...klr", self.M, r_var.clamp_min(0.0).sqrt())    # (...,K,L,Lr)
            U = torch.cat([U, U_extra], dim=-1)
        else:
            Eq = self.b / torch.clamp(self.a - 1.0, min=1e-4)
            d = d + extra_rVr.unsqueeze(-1) * Eq
            d = d + torch.einsum("klr,...r->...kl", self.M ** 2, r_var)         # diagonal-only fallback
        return mean, d.clamp_min(1e-8), U

    def predictive_cov(self, g: torch.Tensor):
        """Mean and FULL covariance factors: Cov_l = diag(d_l) + U_l U_l^T.

        Diagonal Q -> U has zero columns and d is the diagonal predictive variance (so the
        low-rank mixture KL reduces to the diagonal one). Low-rank Q -> the (1+rVr)-inflated
        diag(d_l)+U_l U_l^T plus the diagonal carry-drift uncertainty, the same noise the
        evidence uses, fed to the Woodbury mixture KL.
        """
        r, htil = self._split_g(g)
        mean = torch.einsum("klr,...r->...kl", self.M, r) + self._Ch(htil).unsqueeze(-2)
        Vr = torch.einsum("krs,...s->...kr", self.V, r)
        rVr = torch.einsum("...r,...kr->...k", r, Vr)
        infl = (1.0 + rVr).clamp(min=1e-6, max=self.infl_max)
        carry_var = self._hVCh(htil).unsqueeze(-2)               # (...,1,L)
        if self.q_rank > 0:
            # review Important #1: inflate tau-noise only; add tr Cov(U) un-inflated.
            d = infl.unsqueeze(-1) * self._q_taudiag + self._q_Udiag + carry_var    # (...,K,L)
            # q(U) is INDEPENDENT of the (A,tau) Normal-Gamma block, so the
            # factor loading U0 U0^T is NOT scaled by the coefficient-uncertainty inflation
            # (1+rVr) -- only the tau-scaled diagonal noise is inflated (f is indep. of tau).
            U = torch.ones_like(infl).unsqueeze(-1).unsqueeze(-1) * self.Ufac  # (...,K,L,r) UNSCALED
        else:
            Eq = self.b / torch.clamp(self.a - 1.0, min=1e-4)
            d = infl.unsqueeze(-1) * Eq + carry_var
            U = d.new_zeros(d.shape + (0,))
        return mean, d.clamp_min(1e-8), U

    # sufficient statistics for one batch
    def stats_from_batch(self, resp: torch.Tensor, z: torch.Tensor, g: torch.Tensor,
                         z_var: torch.Tensor = None, g_z_var: torch.Tensor = None):
        """Responsibility-weighted RAW sufficient statistics for one batch.

        Returns the raw (C-free) regime + carry cross statistics; residualisation against the
        current C happens inside m_step. resp may carry K+1 columns (birth seeds a candidate).
        z_var adds diag(Var z_t) to the second moments (analytic/Rao-Blackwellised); g_z_var
        adds diag(Var z_{t-1}) to the r-Gram z-block.
        """
        Kr = resp.shape[-1]
        rsp = resp.reshape(-1, Kr)                               # (M,Kr)
        zf = z.reshape(-1, self.L)
        gf = g.reshape(-1, self.G)
        rf = torch.cat([gf[:, : self.L],
                        gf[:, self.L + self.Hp :]], dim=-1)      # (M,Lr)=[z,a,1]
        hf = gf[:, self.L : self.L + self.Hp]                    # (M,Hp)
        N = rsp.sum(0)                                           # (Kr,)
        Srr = torch.einsum("mk,mr,ms->krs", rsp, rf, rf)
        Szr = torch.einsum("mk,mi,mr->kir", rsp, zf, rf)
        Szz = torch.einsum("mk,mi->ki", rsp, zf * zf)
        Shh = torch.einsum("mk,mh,mg->khg", rsp, hf, hf)
        Szh = torch.einsum("mk,mi,mh->kih", rsp, zf, hf)
        Srh = torch.einsum("mk,mr,mh->krh", rsp, rf, hf)
        out = dict(N=N, Srr=Srr, Szr=Szr, Szz=Szz, Shh=Shh, Szh=Szh, Srh=Srh)
        if self.q_rank > 0:
            out["Szz_full"] = torch.einsum("mk,mi,mj->kij", rsp, zf, zf)   # (Kr,L,L) raw
        if z_var is not None:
            vf = z_var.reshape(-1, self.L)
            rv = torch.einsum("mk,mi->ki", rsp, vf)
            out["Szz"] = out["Szz"] + rv                        # diag E[z z^T]
            if self.q_rank > 0:
                out["Szz_full"] = out["Szz_full"] + torch.diag_embed(rv)
        if self.q_rank > 0:
            # ---- local factor E-step (exact conditional): q(f_tk) = N(m_ftk, P_k^{-1}) with
            # P_k = I + sum_i E[tau_ki] E[u_ki u_ki^T]  (data-independent -> shared over t) and
            # m_ftk = P_k^{-1} sum_i E[tau_ki] E[u_ki] resid_mean_tki. Only MEANS couple to f
            # (independent q blocks), so variance corrections stay in the base statistics.
            Kf, F = self.K, self.q_rank
            Om = self.Omega                                                    # (K,L) E[tau]
            EuuT = self.Ucov + torch.einsum("kif,kig->kifg", self.Umean, self.Umean)
            P = (torch.eye(F, dtype=zf.dtype, device=zf.device)
                 + torch.einsum("ki,kifg->kfg", Om, EuuT))
            Pinv = torch.cholesky_inverse(_chol_jitter(P))                     # (K,F,F)
            Ch = hf @ self.Cmean.transpose(0, 1)                               # (M,L)
            Mr = torch.einsum("klr,mr->mkl", self.M, rf)
            Rm = zf.unsqueeze(1) - Mr - Ch.unsqueeze(1)                        # (M,K,L)
            bf = torch.einsum("ki,kif,mki->mkf", Om, self.Umean, Rm)
            m_f = torch.einsum("kfg,mkg->mkf", Pinv, bf)                       # (M,K,F)
            rspK = rsp[:, :Kf]
            Szf = torch.einsum("mk,mi,mkf->kif", rspK, zf, m_f)
            Sfr = torch.einsum("mk,mkf,mr->kfr", rspK, m_f, rf)
            Sfh = torch.einsum("mk,mkf,mh->kfh", rspK, m_f, hf)
            Sff = (torch.einsum("mk,mkf,mkg->kfg", rspK, m_f, m_f)
                   + rspK.sum(0).view(-1, 1, 1) * Pinv)
            if Kr > Kf:
                # candidate columns (birth seeding) start at q(u)=prior, E[f]=0: zero stats
                def _pad(t):
                    return torch.cat([t, torch.zeros((Kr - Kf,) + t.shape[1:],
                                                     dtype=t.dtype, device=t.device)], 0)
                Szf, Sfr, Sfh, Sff = _pad(Szf), _pad(Sfr), _pad(Sfh), _pad(Sff)
            out.update(Szf=Szf, Sfr=Sfr, Sfh=Sfh, Sff=Sff)
        if g_z_var is not None:
            gv = g_z_var.reshape(-1, self.L)
            rgv = torch.einsum("mk,mi->ki", rsp, gv)            # Var(z_{t-1}) on the r z-block
            out["Srr"][:, : self.L, : self.L] = (
                out["Srr"][:, : self.L, : self.L] + torch.diag_embed(rgv))
        return out

    def set_stats(self, stats):
        """Install statistics. Accepts either RAW stats (full shared-carry dict) or the
        residualised regime-only dict {N,Sgg,Szg,Szz} that the moves pass to a frozen-C clone.
        """
        self.N.copy_(stats["N"])
        if "Srr" in stats:                                       # raw, full shared-carry stats
            self.Srr.copy_(stats["Srr"]); self.Szr.copy_(stats["Szr"]); self.Szz.copy_(stats["Szz"])
            self.Shh.copy_(stats["Shh"]); self.Szh.copy_(stats["Szh"]); self.Srh.copy_(stats["Srh"])
            if self.q_rank > 0:
                self.Szz_full.copy_(stats["Szz_full"])           # raw Σ r̂ z z^T
                for _nm in ("Szf", "Sfr", "Sfh", "Sff"):
                    if _nm in stats:
                        getattr(self, _nm).copy_(stats[_nm])
            self._residual_only = False
        else:                                                    # residualised regime-only (moves)
            self.Sgg.copy_(stats["Sgg"]); self.Szg.copy_(stats["Szg"]); self.Szz_resid.copy_(stats["Szz"])
            if self.q_rank > 0:
                # the moves hand the ALREADY-residualised full second moment here
                self.Szz_full_resid.copy_(stats["Szz_full"])
                # the factor statistics are RAW (C-free) accumulations in BOTH
                # conventions -- _update_qU forms the residual combination itself
                # (Syf = Szf - C Sfh) -- so a candidate clone installs them as-is.
                # Dropping them here was the collapse: the clone
                # refit q(U) from zero factor evidence and every surviving regime
                # lost its learned loadings on installation.
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

    # --------------------------------------------------------- closed-form M-step
    @torch.no_grad()
    def _regime_mstep_from_resid(self, Szg, Szz_resid):
        """Diagonal Normal-Gamma update of the regime blocks from the C-RESIDUAL stats
        (N, Srr, Szg=Σ r̂ z~ r^T, Szz_resid=Σ r̂ z~⊙z~). Updates M, lam, V, a, b in place."""
        eye = torch.eye(self.Lr, device=self.M.device, dtype=self.M.dtype)
        lam0 = torch.diag(self.lam0_diag)
        # Sgg == Srr on the raw path (kept in sync by _residualise_stats); on the frozen-C
        # move path only Sgg is populated, so read Sgg to cover both.
        lam = lam0.unsqueeze(0) + self.Sgg + self.jitter * eye           # (K,Lr,Lr)
        chol = _chol_jitter(lam)  
        V = torch.cholesky_inverse(chol)
        rhs = (self.M0 @ lam0).unsqueeze(0) + Szg                        # (K,L,Lr)
        M = torch.einsum("klr,krs->kls", rhs, V)
        a = (self.a0 + 0.5 * self.N.unsqueeze(-1)).expand(self.K, self.L).clone()
        m0lam0m0 = torch.einsum("ir,rs,is->i", self.M0, lam0, self.M0)   # (L,)
        lamM = torch.einsum("krs,kls->klr", lam, M)
        MlamM = torch.einsum("klr,klr->kl", M, lamM)
        b = self.b0 + 0.5 * (Szz_resid + m0lam0m0.unsqueeze(0) - MlamM)
        b = torch.clamp(b, min=1e-6)
        # degenerate-collapse safeguard: the ABSOLUTE clamp above is insufficient
        # (E[tau] = a/b ~ 1e9 at a ~ 1e3, b = 1e-6); cap E[tau] RELATIVE to a.
        # See regimes.py m_step for the measured failure mode.
        b = torch.maximum(b, a * self.noise_var_floor)
        self.M.copy_(M); self.lam.copy_(lam); self.V.copy_(V)
        self.a.copy_(a); self.b.copy_(b); self._lam_chol = chol

    @torch.no_grad()
    def _C_mstep(self):
        """Row-factorised variational Gaussian update of the shared drift C (Eqs 29-31).

        For each output dim i, with omega_{l,i}=a_{l,i}/b_{l,i}:
            Lambda^C_i = vC0 I + Σ_l ω_{l,i} Shh_l
            eta^C_i    = vC0 m^C_0i + Σ_l ω_{l,i} ( Szh_{l,i} - Gamma_l[i,:] Srh_l )
            V^C_i = (Lambda^C_i)^-1,  m^C_i = V^C_i eta^C_i.
        """
        Om = self.Omega                                                 # (K,L)
        eyeH = torch.eye(self.Hp, device=self.M.device, dtype=self.M.dtype)
        # precision: vC0 I + Σ_l ω_{l,i} Shh_l  -> (L,Hp,Hp)
        Lam = self.vC0 * eyeH.unsqueeze(0) + torch.einsum("ki,khg->ihg", Om, self.Shh)
        Lam = Lam + self.jitter * eyeH
        # Gamma_l[i,:] @ Srh_l : regime-explained carry cross term -> (K,L,Hp)
        GSrh = torch.einsum("klr,krh->klh", self.M, self.Srh)
        # rhs per (l,i): Szh_{l,i} - (Gamma_l Srh_l)_i ; weight by ω_{l,i} and sum over l
        resid_cross = self.Szh - GSrh                                   # (K,L,Hp)
        if self.q_rank > 0:
            # subtract the factor-explained carry cross moment E[(U f) h~^T]
            resid_cross = resid_cross - torch.einsum("kif,kfh->kih", self.Umean, self.Sfh)
        eta = self.vC0 * self.C0mean + torch.einsum("ki,kih->ih", Om, resid_cross)  # (L,Hp)
        cholC = torch.linalg.cholesky(Lam)
        Vc = torch.cholesky_inverse(cholC)                             # (L,Hp,Hp)
        m = torch.einsum("ihg,ig->ih", Vc, eta)                        # (L,Hp)
        self.Ccov.copy_(Vc); self.Cmean.copy_(m)

    @torch.no_grad()
    def _update_qU(self):
        """Exact Gaussian coordinate update of q(U) given q(f) (batch f-stats), q(A,tau)
        (Omega, M) and q(C): row u_{k,i} has precision I/c_u + E[tau_ki] Sff_k and linear
        term E[tau_ki] (Syf_ki - Gamma_ki Sfr_k^T), Syf = Szf - Cbar Sfh^T (C-residualised
        on the fly). Conjugate: exact block maximisation of the augmented ELBO."""
        Om = self.Omega
        F = self.q_rank
        Syf = self.Szf - torch.einsum("ih,kfh->kif", self.Cmean, self.Sfh)
        lin = Om.unsqueeze(-1) * (Syf - torch.einsum("klr,kfr->klf", self.M, self.Sfr))
        eyeF = torch.eye(F, dtype=self.M.dtype, device=self.M.device)
        Prec = (eyeF / self.u_prior_scale
                + Om.unsqueeze(-1).unsqueeze(-1) * self.Sff.unsqueeze(1))      # (K,L,F,F)
        cov = torch.cholesky_inverse(_chol_jitter(Prec))
        mean = torch.einsum("kifg,kig->kif", cov, lin)
        # Rows with NO factor evidence (all-zero Sff: freshly born states, or a
        # clone handed statistics from before a state existed) KEEP their current
        # q(U): overwriting them with the exact zero of empty stats would pin them
        # to the loading saddle forever. Constructor
        # initialisation is saddle-broken, so kept rows acquire evidence at their
        # next E-step.
        evid = (self.Sff.abs().sum(dim=(-2, -1)) > 0)                  # (K,)
        self.Ucov.copy_(torch.where(evid.view(-1, 1, 1, 1), cov, self.Ucov))
        self.Umean.copy_(torch.where(evid.view(-1, 1, 1), mean, self.Umean))

    @torch.no_grad()
    def _f_adjusted_resid_stats(self, Szg, Szz_resid):
        """Moments of the f-adjusted target z - C h~ - U f, so the UNMODIFIED conjugate
        Normal-Gamma block fits (A_k, tau_k) on them:
          Szg~   = Szg - E[U] Sfr
          Szz~_i = Szz_i - 2 E[u_i].Syf_i + <E[u_i u_i^T], Sff>."""
        Syf = self.Szf - torch.einsum("ih,kfh->kif", self.Cmean, self.Sfh)
        SzgF = Szg - torch.einsum("kif,kfr->kir", self.Umean, self.Sfr)
        EuuT = self.Ucov + torch.einsum("kif,kig->kifg", self.Umean, self.Umean)
        SzzF = (Szz_resid - 2.0 * (self.Umean * Syf).sum(-1)
                + torch.einsum("kifg,kfg->ki", EuuT, self.Sff))
        return SzgF, SzzF.clamp_min(0.0)

    @torch.no_grad()
    def _residualise_stats(self):
        """Form the C-residualised regime stats Szg=Σ r̂ z~ r^T and the EXPECTED residual
        second moment E_q(C)[Σ r̂ (z - C h~)_i^2] from raw stats and the current q(C), and
        refresh the Sgg/Szg/Szz buffers the M-step and the moves read.

        Exactness note: because the rows c_i of C carry a Gaussian posterior
        q(c_i)=N(C̄_i, V^C_i), the expected squared residual is

            E_q(C)[(z - C h~)_i^2] = z_i^2 - 2 C̄_i·(z_i h~) + h~^T (C̄_i C̄_i^T + V^C_i) h~,

        so the residual statistic must include the carry-COVARIANCE trace
        tr(V^C_i S_hh,k) in addition to the mean quadratic C̄_i S_hh C̄_i. This is the
        same fluctuation `expected_loglik` charges per sample (its hVCh term, Eq. 20);
        omitting it here would make the Q/regime M-step and the evidence optimize
        inconsistent objectives. Rows of q(C) are independent, so the cross-dimension
        residual covariance gets no such term (the trace enters the DIAGONAL only)."""
        # Szg = Szr - C̄ @ Srh^T
        Szg = self.Szr - torch.einsum("ih,krh->kir", self.Cmean, self.Srh)
        # Szz_resid_i = Szz_i - 2 C̄_i·Szh_{l,i} + C̄_i Shh_l C̄_i + tr(V^C_i Shh_l)
        CSzh = torch.einsum("ih,kih->ki", self.Cmean, self.Szh)        # C̄_i·Σ r̂ z_i h~
        ShhC = torch.einsum("khg,ig->kih", self.Shh, self.Cmean)       # (K,L,Hp)
        CShhC = torch.einsum("ih,kih->ki", self.Cmean, ShhC)           # C̄_i Shh_l C̄_i
        trVCShh = torch.einsum("ihg,khg->ki", self.Ccov, self.Shh)     # tr(V^C_i Shh_l)
        Szz_resid = self.Szz - 2.0 * CSzh + CShhC + trVCShh
        self.Sgg.copy_(self.Srr)
        self.Szg.copy_(Szg)
        self.Szz_resid.copy_(Szz_resid.clamp_min(0.0))
        if self.q_rank > 0:
            # full residual second moment E_q(C)[Σ r̂ (z - Ch~)(z - Ch~)^T] for the
            # factor-analysis Q fit; row-independent q(C) puts the covariance trace on
            # the diagonal only
            SzhC = torch.einsum("kih,jh->kij", self.Szh, self.Cmean)    # Σ r̂ z_i (C̄h~)_j
            ShhC2 = torch.einsum("khg,jg->khj", self.Shh, self.Cmean)   # (K,Hp,L)
            CShhC2 = torch.einsum("ih,khj->kij", self.Cmean, ShhC2)     # Σ r̂ (C̄h~)_i (C̄h~)_j
            full = self.Szz_full - SzhC - SzhC.transpose(-1, -2) + CShhC2
            full = full + torch.diag_embed(trVCShh)
            self.Szz_full_resid.copy_(full)
        return Szg, self.Szz_resid

    @torch.no_grad()
    def m_step(self):
        """Block-coordinate VB M-step: alternate the regime blocks (on the C-residual) and the
        shared drift C (on the regime residual). A few sweeps suffice (Prop. 2)."""
        if getattr(self, "_residual_only", False):
            # frozen-C clone made by the moves: only the regime blocks update, from the
            # residualised stats it was handed.
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
                # block order: q(U) | old (A,tau)  ->  (A,tau) | new U  ->  C | both.
                # Each is an exact conditional maximiser: monotone coordinate ascent.
                self._update_qU()
                SzgF, SzzF = self._f_adjusted_resid_stats(Szg, Szz_resid)
                self._regime_mstep_from_resid(SzgF, SzzF)
            else:
                self._regime_mstep_from_resid(Szg, Szz_resid)
            if not self._freeze_C:
                self._C_mstep()
        # final residualised snapshot for the moves + ARD
        self._residualise_stats()
        if self.ard:
            self._ard_step()
        if self.q_rank > 0:
            self._refresh_lowrank_predictive_cache()

    @torch.no_grad()
    def _refresh_lowrank_predictive_cache(self):
        """Moment-matched PREDICTIVE noise cache from the variational posterior:
        E[Q_k] = E[diag(1/tau_k)] + E[U_k U_k^T]
               = diag(b/(a-1) + tr Ucov_{k,i}) + Umean_k Umean_k^T,
        stored as (q_Ddiag, Ufac) so predictive()/predictive_moments()/diag_score keep
        their existing low-rank-plus-diagonal reading. GENERATION-ONLY: the fitting path
        (evidence, M-steps, KLs, move scores' exact part) never touches this cache."""
        Einv_tau = self.b / (self.a - 1.0).clamp_min(1e-3)
        trU = torch.einsum("kiff->ki", self.Ucov)
        self.q_Ddiag.copy_((Einv_tau + trU).clamp_min(1e-8))    # full marginal (back-compat)
        self._q_taudiag.copy_(Einv_tau.clamp_min(1e-8))         # tau-noise (inflatable)
        self._q_Udiag.copy_(trU.clamp_min(0.0))                 # U uncertainty (NOT inflatable)
        self.Ufac.copy_(self.Umean)

    @torch.no_grad()
    def reset_slot(self, k):
        """reset regime slot k to its PRIOR (a fresh regime) and clear
        its sufficient statistics -- the parameter side of a fixed-Kmax birth (activate a
        spare + reset it) or delete (deactivate + clear), so structural moves never resize."""
        k = int(k)
        self.M[k] = 0.0
        self.lam[k] = torch.diag(self.lam0_diag)
        self.V[k] = torch.diag(1.0 / self.lam0_diag)
        self.a[k] = float(self.a0)
        self.b[k] = float(self.b0)
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
        """Shared-ARD MacKay fixed point for the diagonal of the regime prior precision."""
        Vdiag = torch.diagonal(self.V, dim1=-2, dim2=-1)        # (K,Lr)
        Om = self.Omega                                         # (K,L)
        Mj2 = self.M ** 2                                       # (K,L,Lr)
        weighted = torch.einsum("ki,kir->kr", Om, Mj2)
        # MacKay evidence counting over OCCUPIED states only: empty states have
        # posterior == prior (V ~ Lambda0^{-1}, M ~ M0), so including them creates a
        # ratchet alpha -> K L / (E L / alpha + S_occ) that diverges to the clamp as
        # structural moves leave transiently empty rows. EB on zero-count states is
        # undefined; they keep the shared prior and contribute no evidence.
        occ = self.N > 1.0
        n_occ = int(occ.sum())
        if n_occ == 0:
            return          # (K,Lr)
        denom = (self.L * Vdiag + weighted)[occ].sum(0)              # (Lr,)
        alpha = (n_occ * self.L) / torch.clamp(denom, min=1e-8)
        self.lam0_diag.copy_(torch.clamp(alpha, min=1e-4, max=1e3))  # ceiling bounds param_kl's lam0*(M-M0)^2 term

    @torch.no_grad()
    def param_kl(self) -> torch.Tensor:
        """Per-regime KL(q(theta_l)||p(theta_l)) PLUS the shared-C KL spread over regimes.

        The regime Normal-Gamma KL is the diagonal MNIW term over r; the shared drift is a
        single global object, so its Gaussian KL is added once (split evenly across the K
        regimes) so the per-regime move bound still sees a clean K-dependent allocation cost.
        Returns (K,).
        """
        K, L, Lr, Hp = self.K, self.L, self.Lr, self.Hp
        lam0 = self.lam0_diag
        chol = self._lam_chol
        logdet_lam = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)   # (K,)
        logdet_lam0 = torch.log(lam0).sum()
        Vdiag = torch.diagonal(self.V, dim1=-2, dim2=-1)
        tr_term = (lam0.unsqueeze(0) * Vdiag).sum(-1)
        diff = self.M - self.M0.unsqueeze(0)
        maha = (lam0.view(1, 1, Lr) * diff ** 2).sum(-1)        # (K,L)
        Etau = self.a / self.b
        gauss_kl = (0.5 * (logdet_lam - logdet_lam0 - Lr + tr_term).unsqueeze(-1)
                    + 0.5 * Etau * maha)                        # (K,L)
        a_q, b_q = self.a, self.b
        a_p, b_p = self.a0, self.b0
        gamma_kl = ((a_q - a_p) * torch.digamma(a_q)
                    - torch.lgamma(a_q) + torch.lgamma(a_p)
                    + a_p * (torch.log(b_q) - torch.log(b_p))
                    + a_q * (b_p - b_q) / b_q)                  # (K,L)
        regime_kl = (gauss_kl + gamma_kl).sum(-1)               # (K,)

        # shared-C Gaussian KL: Σ_i KL(N(m^C_i,V^C_i)||N(m^C_0i, vC0^-1 I))
        cholC = torch.linalg.cholesky(self.Ccov + self.jitter *
                                      torch.eye(Hp, device=self.M.device, dtype=self.M.dtype))
        logdetVc = 2.0 * torch.log(torch.diagonal(cholC, dim1=-2, dim2=-1)).sum(-1)     # (L,)
        trVc = torch.diagonal(self.Ccov, dim1=-2, dim2=-1).sum(-1)                      # (L,)
        dC = self.Cmean - self.C0mean
        maha_C = (dC ** 2).sum(-1)                              # (L,) with prior prec vC0 I
        C_kl = 0.5 * (-logdetVc - Hp * torch.log(self.vC0) - Hp
                      + self.vC0 * trVc + self.vC0 * maha_C).sum()   # scalar
        total = regime_kl + C_kl / K
        if self.q_rank > 0:
            # factor-loading KL: sum_ki KL( N(Umean_ki, Ucov_ki) || N(0, c_u I_F) ).
            # Fully counted in the bound (this is what makes q_rank>0 a true ELBO path).
            cu = float(self.u_prior_scale)
            F = self.q_rank
            cholU = _chol_jitter(self.Ucov)
            logdetU = 2.0 * torch.log(torch.diagonal(cholU, dim1=-2, dim2=-1)).sum(-1)
            u_kl = 0.5 * ((torch.einsum("kiff->ki", self.Ucov)
                           + (self.Umean ** 2).sum(-1)) / cu
                          - F + F * math.log(cu) - logdetU)      # (K,L)
            total = total + u_kl.sum(-1)
        return total

    @torch.no_grad()
    @torch.no_grad()
    def data_elbo_from_stats(self) -> torch.Tensor:
        """C-residualised expected data log-likelihood from cached statistics.

        Same closed form as DiagARRegimes.data_elbo_from_stats, on the residual
        regression stats (Sgg = r-Gram, Szg / Szz residualised against the frozen C).
        Since `_residualise_stats` folds the carry-covariance trace tr(V^C_i S_hh,k)
        into Szz_resid, the responsibility-summed carry-drift fluctuation
        -1/2 Σ_t r̂_tk ω_{k,i} h~^T V^C_i h~ IS captured here through E[τ]·Szz_resid
        (exactly, because ω enters the evidence linearly against that statistic). What
        cannot be a function of cached statistics is only the per-sample coupling of
        the (1+rVr) inflation with the residual under q_rank > 0; there the diagonal
        surrogate is used, and — as with the DiagAR case — this remains a Hughes-style
        SHORTLIST criterion whose selections are always re-verified by the exact
        acceptance bound.
        """
        if getattr(self, "_residual_only", False):
            Szg, Szz_res = self.Szg, self.Szz_resid
        else:
            Szg, Szz_res = self._residualise_stats()
        N = self.N.clamp_min(0.0)
        MSzg = (self.M * Szg).sum(-1)                          # (K,L)
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
        """A fresh SharedCarryRegimes at a different K, sharing this object's C (frozen).

        Used by the moves: K changes, the shared drift does not, so the candidate keeps C
        fixed and only its regime blocks are fit (from the residualised stats handed in).
        """
        new = SharedCarryRegimes(
            K=new_K, L=self.L, G=self.G, a0=float(self.a0), b0=float(self.b0),
            v0_scale=1.0, vC0_scale=float(self.vC0), ard=False, identity_init=False,
            jitter=self.jitter, q_rank=self.q_rank, action_dim=self.action_dim,
            device=self.M.device, dtype=self.M.dtype,   # keep action_dim
        )
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

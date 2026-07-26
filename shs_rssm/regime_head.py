"""RegimeHead: the SHS-RSSM machinery that plugs into a DreamerV3 RSSM.

It owns the three trained-by-closed-form globals (the diagonal regime model and the
sticky-HDP transition) plus the ONE neural parameter it adds, the carry projection P.
It exposes everything the world model needs:

  build_g(prev_stoch, deter)      -> g_t = [z_{t-1}, P h_t, 1]
  regime_inference(stoch, deter,  -> responsibilities gamma, transition counts,
                   is_first)          start counts, and the cached regressors g
  dynamics_kl(q_mean, q_std,      -> the structured variational KL (dyn/rep balanced),
              stoch, deter, ...)      replacing DreamerV3's Gaussian-Gaussian dyn KL
  update_globals(...)             -> EMA sufficient statistics + closed-form M-step
                                     on the regime and sticky-HDP globals (no grad)

Contract: gradients touch only P (and, through the loss, the encoder/decoder/GRU
that produced q and the deter). The regime and sticky-HDP globals are buffers,
updated by closed form on EMA statistics, never by SGD. The two sides communicate
only through the dynamics KL and the responsibilities.
"""
from __future__ import annotations

import copy
import threading
import torch
import torch.nn as nn

from .regimes import DiagARRegimes
from .regimes_shared import SharedCarryRegimes
from .sticky_hdp import StickyHDP
from .forward_backward import forward_backward, start_counts_from
from .mixture_prior import mixture_weights, _diag_gauss_kl
from .recurrent_stick import RecurrentStickiness
from .structured_elbo import diag_gauss_entropy


class RegimeHead(nn.Module):
    def __init__(
        self,
        stoch: int,          # L, latent dim
        deter: int,          # H, GRU carry dim
        K: int = 16,         # truncation (max regimes)
        proj_dim: int | None = 64,   # H'; set to `deter` to disable projection
        action_dim: int = 0,         # A; >0 adds a per-regime action term B_k a_{t-1} 
        # regime (Normal-Gamma) hyperparameters
        a0: float = 3.0, b0: float = 2.0, v0_scale: float = 1.0, ard: bool = True,
        identity_init: bool = True,
        q_rank: int = 0,
        shared_carry: bool = False,   # tie the carry drift C across regimes (Prop. 1 fix)
        # sticky-HDP hyperparameters
        gamma: float = 5.0, alpha: float = 1.0, kappa: float = 50.0, start_alpha: float = 1.0,
        # recurrent stickiness (h-dependent persistence with Polya-Gamma)
        recurrent: bool = False, prior_persist: float = 0.9, pg_iters: int = 4,
        rstick_dim: int | None = 8, rstick_stopgrad: bool = True,
        # online schedule
        ema_tau: float = 0.02, hdp_iters: int = 2,
        online_mode: str = "ema",          # "ema" | "full_batch" | "streaming" | "memoized"
        expected_batches: int | None = None,   # memoized completeness certificate (count)
        expected_ids: set | None = None,       # memoized completeness certificate (id set)
        strict_stream: bool = False,           # streaming: no id fallback, contiguous offsets
        hdp_every: int = 1,                    # item 18: run HDP root-stick opt every Nth global step
        pg_every: int = 1,                     # item 18: refresh the PG block every Nth global step
        dtype=torch.float32,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        # replica construction: every ctor arg, so make_worker can build a
        # DETACHED worker head structurally identical to the master
        self._ctor_kwargs = dict(
            stoch=stoch, deter=deter, K=K, proj_dim=proj_dim,
            action_dim=action_dim, a0=a0, b0=b0,
            v0_scale=v0_scale, ard=ard, identity_init=identity_init, q_rank=q_rank,
            shared_carry=shared_carry, gamma=gamma, alpha=alpha, kappa=kappa,
            start_alpha=start_alpha, recurrent=recurrent, prior_persist=prior_persist,
            pg_iters=pg_iters, rstick_dim=rstick_dim, rstick_stopgrad=rstick_stopgrad,
            ema_tau=ema_tau, hdp_iters=hdp_iters, online_mode=online_mode,
            expected_batches=expected_batches,
            expected_ids=(set(expected_ids) if expected_ids is not None else None),
            strict_stream=strict_stream, hdp_every=hdp_every, pg_every=pg_every,
            dtype=dtype, device=device)
        self.L = stoch
        self.H = deter
        self.K = K
        self.Hp = deter if (proj_dim is None or proj_dim >= deter) else proj_dim
        self.use_proj = self.Hp < deter
        self.action_dim = int(action_dim)
        self.G = self.L + self.Hp + self.action_dim + 1
        self.ema_tau = ema_tau
        self.online_mode = online_mode
        if online_mode == "ema":
            self.stat_store = None
        else:
            from .online_vb import SuffStatStore
            self.stat_store = SuffStatStore(
                mode=("legacy_ema" if online_mode == "ema" else online_mode),
                ema_tau=ema_tau, expected_batches=expected_batches,
                expected_ids=expected_ids, strict_stream=strict_stream)
        # representation version: bump whenever the encoder/GRU that produced the
        # latents changes (e.g. every world-model gradient step, or per consolidation
        # freeze). Move buffers stamp it so structure moves can refuse to score a
        # corpus mixing incompatible representations.
        self.register_buffer("repr_version", torch.zeros((), dtype=torch.long))
        self._register_load_state_dict_pre_hook(self._structural_pre_load_hook)
        self.register_load_state_dict_post_hook(self._post_load_refresh)
        self.hdp_iters = hdp_iters
        self.recurrent = recurrent
        self.rstick_stopgrad = bool(rstick_stopgrad)
        self.rstick_dim = int(min(deter, rstick_dim)) if rstick_dim is not None else self.Hp

        # the only neural parameter added: linear carry projection (no bias)
        if self.use_proj:
            self.P = nn.Linear(deter, self.Hp, bias=False)
        else:
            self.P = None

        # Separate low-dimensional feature map for recurrent stickiness. The feature is
        # deliberately smaller than the dynamics regressor and can stop the gradient into the
        # Dreamer carry while still learning this projection from the transition ELBO.
        if self.rstick_dim < deter:
            self.P_stick = nn.Linear(deter, self.rstick_dim, bias=False)
            nn.init.orthogonal_(self.P_stick.weight, gain=0.5)
        else:
            self.P_stick = None

        self.shared_carry = bool(shared_carry)
        if self.shared_carry:
            # tied-C dynamics: the carry enters every regime through one shared drift, so the
            # regressor still arrives as g=[z; P h; 1] but the regime maps act on r=[z; 1] only.
            # Low-rank process noise is fit on the carry residual when q_rank > 0.
            self.regimes = SharedCarryRegimes(
                K=K, L=self.L, G=self.G, action_dim=self.action_dim,
                a0=a0, b0=b0, v0_scale=v0_scale, ard=ard,
                identity_init=identity_init, q_rank=q_rank, dtype=dtype, device=device,
            )
        else:
            self.regimes = DiagARRegimes(
                K=K, L=self.L, G=self.G, action_dim=self.action_dim,
                a0=a0, b0=b0, v0_scale=v0_scale, ard=ard,
                identity_init=identity_init, q_rank=q_rank, dtype=dtype, device=device,
            )
        # sticky-HDP runs in float64 internally for the digamma/lgamma optimisation.
        # When recurrent stickiness is on, the base transition must be NON-sticky
        # (kappa=0): the Bernoulli persistence carries all the stickiness, so a kappa>0
        # base would double-count it. The kappa=0 branch is the faithful HDP-HMM.
        base_kappa = 0.0 if recurrent else kappa
        self.hdp = StickyHDP(K=K, gamma=gamma, alpha=alpha, kappa=base_kappa,
                             start_alpha=start_alpha, dtype=torch.float64, device=device)

        # recurrent stickiness head (input-dependent dwell, Polya-Gamma logistic). ALWAYS
        # constructed -- even when the run starts with recurrent=False -- so a curriculum can
        # switch it ON after the regimes have stabilised. It simply sits at its prior and is
        # not consulted while self.recurrent is False. Feature is the carry projection
        # [P h_t; 1] (dim Hp); when recurrent is on, the base kappa is forced to 0 (below /
        # in the curriculum) so the Bernoulli persistence is not double-counted.
        self.rstick = RecurrentStickiness(
            K=K, feat_dim=self.rstick_dim, prior_persist=prior_persist, pg_iters=pg_iters,
            dtype=torch.float64, device=device,
        )
        self._estep = None  # cache of recurrent-E-step quantities for the global update
        self._struct_cache = None  # cached xi/gamma for the structured sequence KL
        self.hdp_every = int(hdp_every)   # item 18: staggered global-block schedule
        self.pg_every = int(pg_every)
        self._gstep_count = 0             # global-step counter for the schedule
        self._episode_cursor = 0          # monotonic completed-episode stream offset (item 1)
        self._repr_frozen = False         # blocker 3: representation-epoch version pin
        # fixed-Kmax active-state mask. None => all K active
        # (legacy shape-changing behaviour). A bool (K,) tensor => inactive slots receive
        # NO responsibility, so K can be held at a fixed capacity while regimes
        # activate/deactivate WITHOUT changing tensor shapes.
        self.active_mask = None
        self._shs_online_pairwise = False # blocker 8: accumulate PG/KL without full xi
        self._async_version = 0    # SDA-Bayes master posterior version
        self._struct_gen = 0       # structural generation (bumped per accepted move)
        self._commit_lock = threading.RLock()  # serializes master commits 

        # learned initial latent z_0 (eq:init, mu0); covariance handled by the regime prior.
        # nn.Parameter (not a buffer) so Adam actually trains it.
        self.z0 = nn.Parameter(torch.zeros(self.L, dtype=dtype, device=device))

        # EMA transition / start counts, kept so birth/merge/delete can rebuild the HDP
        self.register_buffer("ema_trans_counts", torch.zeros(K, K, dtype=torch.float64, device=device))
        self.register_buffer("ema_start_counts", torch.zeros(K, dtype=torch.float64, device=device))
        self._counts_initialised = False

    # ----------------------------------------------------------------- regressor
    def _proj(self, deter):
        return self.P(deter) if self.use_proj else deter

    def build_g(self, prev_stoch, deter, action=None):
        """g_t = [z_{t-1}, P h_t, a_{t-1}, 1] (action block only when action_dim>0).

        a per-regime action term enters the CONJUGATE dynamics
        regressor, so  z_t = A_k z_{t-1} + B_k a_{t-1} + b_k + C P h_t (+ U_k f_t) + eps
        keeps exact Normal-Gamma updates -- B_k is simply extra columns of the regime
        regressor block (shared-carry keeps the tied C on P h_t; the action joins r).
        `action` must already be aligned as the action that PRODUCED step t (Dreamer's
        prev-action convention) and is zeroed at episode starts via _shift_action."""
        htil = self._proj(deter)
        ones = prev_stoch[..., :1] * 0.0 + 1.0
        parts = [prev_stoch, htil]
        if self.action_dim > 0:
            if action is None:
                action = prev_stoch.new_zeros(*prev_stoch.shape[:-1], self.action_dim)
            parts.append(action.to(prev_stoch.dtype))
        parts.append(ones)
        return torch.cat(parts, dim=-1)

    def _shift_action(self, action, is_first):
        """Zero the action regressor at episode starts (no a_{t-1} exists there)."""
        if action is None or self.action_dim == 0:
            return None
        if is_first is None:
            return action
        isf = is_first.reshape(*action.shape[:-1], 1).to(action.dtype)
        return action * (1.0 - isf)

    def build_stick_phi(self, deter):
        """Low-dimensional recurrent-stickiness feature phi_t=[S h_t; 1].

        This is intentionally separate from the dynamics regressor g_t. With
        rstick_stopgrad=True, gradients from the transition term train S but do not push the
        Dreamer GRU carry to encode switching shortcuts.
        """
        h = deter.detach() if self.rstick_stopgrad else deter
        htil = self.P_stick(h) if self.P_stick is not None else h
        htil = torch.tanh(htil[..., :self.rstick_dim])
        ones = htil[..., :1] * 0.0 + 1.0
        return torch.cat([htil, ones], dim=-1)

    def _prev_stoch(self, stoch, is_first):
        """z_{t-1} from stoch (shift by one), reset to z0 at episode starts.

        Built with cat/where (no in-place writes) so it is safe in the autograd graph.
        stoch (B,T,L), is_first (B,T) or (B,T,1).
        """
        B, T, L = stoch.shape
        z0 = self.z0.view(1, 1, L).expand(B, 1, L)
        prev = torch.cat([z0, stoch[:, :-1]], dim=1) if T > 1 else z0
        if is_first is not None:
            isf = is_first.reshape(B, T, 1).to(stoch.dtype)
            prev = torch.where(isf > 0.5, self.z0.view(1, 1, L), prev)
        return prev

    def _g_var_from_z_var(self, z_var, g, is_first=None):
        """Diagonal Var[g_t] for g_t=[z_{t-1}; P h_t; 1].

        Only z_{t-1} is random in the amortised continuous posterior.  The projected carry
        P h_t and the bias are deterministic.  At episode starts z_{t-1}=z0 is the learned
        deterministic initial latent, so the regressor variance is zero on those links.
        """
        if z_var is None:
            return None
        prev_var = self._shift_var(z_var, is_first)
        zeros_tail = g[..., self.L:] * 0.0
        return torch.cat([prev_var, zeros_tail], dim=-1)

    # ------------------------------------------------------------ regime E-step
    def _transition_logpotentials(self, deter, dtype=None, device=None):
        """Return log p(s_1) and log p(s_t|s_{t-1}) for the current transition model.

        If recurrent stickiness is active, the transition tensor is time varying with
        shape (B,T-1,K,K). Otherwise it is stationary with shape (K,K).
        """
        dtype = dtype if dtype is not None else deter.dtype
        device = device if device is not None else deter.device
        log_init = self.hdp.expected_log_init().to(dtype=dtype, device=device)
        base_elogpi = self.hdp.expected_log_trans().to(dtype=dtype, device=device)
        if self.recurrent:
            # Fully variational PG/JJ-bounded augmented potentials (no softmax, no
            # probit moment approximation): forward-backward on these is exact
            # structured VB over the joint (regime, persistence-indicator) chain and
            # its log-partition is a valid ELBO term.
            phi = self.build_stick_phi(deter.to(dtype))
            log_trans, aux = self.rstick.bound_log_trans(base_elogpi, phi[:, 1:])
            aux = dict(aux, phi_steps=phi[:, 1:])
            log_init, log_trans = self._mask_logpotentials(log_init, log_trans)  # P1 #9
            return log_init, log_trans, aux
        log_init, base_elogpi = self._mask_logpotentials(log_init, base_elogpi)  # P1 #9
        return log_init, base_elogpi, None

    def _transition_potentials_ondemand(self, deter, dtype=None, device=None):
        """Recurrent transition potentials WITHOUT materialising the
        O(BTK^2) log_trans. Returns (log_init, aux, trans_fn) where aux is O(BTK) and
        trans_fn(t) builds a single (B,K,K) slice on demand (active-mask aware). Recurrent
        only (uses the stickiness features)."""
        dtype = dtype if dtype is not None else deter.dtype
        device = device if device is not None else deter.device
        log_init = self.hdp.expected_log_init().to(dtype=dtype, device=device)
        base = self.hdp.expected_log_trans().to(dtype=dtype, device=device)
        phi = self.build_stick_phi(deter.to(dtype))
        aux = dict(self.rstick.bound_aux_only(base, phi[:, 1:]), phi_steps=phi[:, 1:])
        if self.active_mask is not None:
            m = self.active_mask.to(log_init.device)
            log_init = torch.where(m, log_init, log_init.new_full((), -1e30))
        am = self.active_mask
        rstick = self.rstick

        def trans_fn(t):
            Tt = rstick.trans_slice_from_aux(aux, t)                  # (B,K,K), O(BK^2)
            if am is not None:
                mm = am.to(Tt.device); K = int(mm.shape[0])
                Tt = torch.where((mm.view(1, K, 1) & mm.view(1, 1, K)), Tt,
                                 Tt.new_full((), -1e30))
            return Tt
        return log_init, aux, trans_fn

    def _vb_evidence(self, z_mean, deter, is_first=None, z_var=None, action=None):
        """Fully Bayesian VB local evidence ell_t(k).

        This is the ONLY evidence used by both the E-step and the dynamics loss:

            ell_t(k) = E_{q(z_t) q(z_{t-1}) q(theta_k)}
                       [ log p(z_t | z_{t-1}, h_t, theta_k) ].

        Thus BOTH the target z_t and the stochastic-regressor block z_{t-1} are integrated
        analytically under the factorised amortised Gaussian posterior.  The projected carry
        h_t and the bias remain deterministic model context; there is no longer a point
        approximation for the previous stochastic latent.
        """
        prev = self._prev_stoch(z_mean, is_first)
        act = self._shift_action(action, is_first)
        g = self.build_g(prev, deter, act)
        zvar = None if z_var is None else z_var
        gvar = self._g_var_from_z_var(zvar, g, is_first=is_first)
        ev = self.regimes.expected_loglik(z_mean, g, z_var=zvar, g_var=gvar)
        return ev, g, gvar

    def regime_inference(self, stoch, deter, is_first=None, cache_estep: bool = False,
                         z_var=None,
                         action=None, valid=None):
        """Structured VB E-step for the discrete regime path.

        The local potentials are the fully Bayesian expected log likelihoods
        under q(z_t) and q(theta_k).  The exact same function `_vb_evidence` is
        used later by `dynamics_kl`, so the E-step and loss cannot drift apart as
        regime noise parameters specialise.
        """
        with torch.amp.autocast("cuda", enabled=False):
            z = stoch.float()
            d = deter.float()
            zvar = None if z_var is None else z_var.float()
            ev_raw, g, g_var = self._vb_evidence(z, d, is_first=is_first, z_var=zvar, action=action)  # (B,T,K)
            # Row-wise shifts improve numerical stability and do not alter gamma/xi
            # (but DO shift logZ by the shift sum -- added back below for the ELBO).
            ev_raw = self._apply_active_mask(ev_raw)   # Blocker 4: inactive slots -> -inf
            _ev_shift = ev_raw.max(dim=-1, keepdim=True).values     # (B,T,1)
            ev = ev_raw - _ev_shift
            _online_pw = bool(getattr(self, "_shs_online_pairwise", False)) and self.recurrent
            _trans_fn = None
            if _online_pw:
                # blocker 8 + review P2 #11: keep the MESSAGES and build each transition
                # slice ON DEMAND -- neither the O(BTK^2) xi NOR the O(BTK^2) transition
                # tensor is ever materialised (peak drops to O(BTK)).
                log_init, trans_aux, _trans_fn = self._transition_potentials_ondemand(
                    d, dtype=torch.float32, device=d.device)
                gamma, counts_base, _logZ_re, _la, _lb = forward_backward(
                    log_init, None, ev, is_first=is_first, valid=valid,
                    return_messages=True, trans_fn=_trans_fn)
                xi = None
            else:
                log_init, log_trans, trans_aux = self._transition_logpotentials(
                    d, dtype=torch.float32, device=d.device)
                gamma, counts_base, _logZ_re, xi = forward_backward(
                    log_init, log_trans, ev, is_first=is_first, valid=valid,
                    return_pairwise=True)
            # blocker 8 + review Important #5: true logZ = shifted logZ + sum_t(shift),
            # but the shift must be masked by VALID (padding steps used zero evidence
            # in forward-backward, so their shift must NOT be added back), and kept as
            # a TENSOR -- converting to float here would force a GPU sync on every
            # replay inference that does not need the value.
            if valid is not None:
                _vmask = valid.reshape(_ev_shift.shape[0], _ev_shift.shape[1]).to(_ev_shift.dtype)
            else:
                _vmask = _ev_shift.new_ones(_ev_shift.shape[0], _ev_shift.shape[1])
            self._last_logZ = (_logZ_re
                               + (_ev_shift.squeeze(-1) * _vmask).sum(dim=1)).detach()  # (B,)

            if self.recurrent:
                # Exact conditional split of the pairwise marginals into persistence
                # (w=1) vs base-switch (w=0) branches under the bounded potentials.
                if _online_pw:
                    from shs_rssm.pairwise_online import accumulate_pg_online
                    r_mass, row_weight, counts = accumulate_pg_online(
                        _la, _lb, _logZ_re, _trans_fn, ev, is_first, valid, trans_aux)
                else:
                    r_mass, row_weight, counts = self.rstick.attribute_bound(xi, trans_aux)
                if cache_estep:
                    self._estep = dict(
                        phi_steps=trans_aux["phi_steps"].detach(),
                        r_mass=r_mass.detach(),
                        row_weight=row_weight.detach(),
                        sig=(r_mass / row_weight.clamp_min(1e-12)).detach(),
                        is_first=is_first, valid=valid,
                    )
            else:
                counts = counts_base
                if cache_estep:
                    self._estep = None

            if cache_estep:
                if _online_pw:
                    self._struct_cache = dict(
                        gamma=gamma.detach(), online=True,
                        messages=(_la.detach(), _lb.detach(), _logZ_re.detach()),
                        trans_aux={k: (v.detach() if torch.is_tensor(v) else v)
                                   for k, v in trans_aux.items()},   # O(BTK) aux, NOT O(BTK^2)
                        is_first=is_first, valid=valid,
                        evidence=ev_raw.detach(), evidence_shifted=ev.detach(),
                    )
                else:
                    self._struct_cache = dict(
                        gamma=gamma.detach(), xi=xi.detach(), is_first=is_first,
                        evidence=ev_raw.detach(), evidence_shifted=ev.detach(),
                    )

        start_counts = start_counts_from(gamma, is_first, valid=valid)
        return gamma, counts, start_counts, g

    # ------------------------------------------------------------ structured KL
    def _discrete_path_kl(self, gamma, xi, deter, is_first=None):
        """Per-step KL(q(s_{1:T}) || p(s_{1:T}|phi)) under detached q(s).

        This is the discrete part of the structured variational objective.  Its
        entropy terms are detached EM quantities; the log-transition term remains
        differentiable when recurrent stickiness is active, so the low-dimensional
        stickiness projection receives the correct transition-gradient.
        """
        B, T, K = gamma.shape
        dtype, device = gamma.dtype, gamma.device
        log_init = self.hdp.expected_log_init().to(dtype=dtype, device=device)
        if is_first is None:
            isf = torch.zeros(B, T, dtype=dtype, device=device)
        else:
            isf = is_first.reshape(B, T).to(dtype=dtype, device=device).clone()
        isf[:, 0] = 1.0

        gclamp = gamma.clamp_min(1e-30)
        out = gamma.new_zeros(B, T)

        # Initial-state KL at true starts / resets.
        init_kl = (gamma * (gclamp.log() - log_init.view(1, 1, K))).sum(-1)
        out = out + isf * init_kl

        if T <= 1:
            return out

        base_elogpi = self.hdp.expected_log_trans().to(dtype=dtype, device=device)
        if self.recurrent:
            phi = self.build_stick_phi(deter)
            log_trans, _ = self.rstick.bound_log_trans(base_elogpi, phi[:, 1:])
        else:
            log_trans = base_elogpi.view(1, 1, K, K).expand(B, T - 1, K, K)

        x = xi.clamp_min(1e-30)
        gp = gamma[:, :-1].clamp_min(1e-30)
        log_q_cond = x.log() - gp[:, :, :, None].log()
        pair_kl = (xi * (log_q_cond - log_trans)).sum(dim=(-2, -1))
        out[:, 1:] = out[:, 1:] + (1.0 - isf[:, 1:]) * pair_kl
        return out

    def _discrete_path_kl_online(self, gamma, deter, is_first, cache):
        """The discrete-path KL WITHOUT a materialised xi. The
        initial-state term needs only gamma; the pairwise term is accumulated per step by
        pair_kl_online using the cached messages and the DIFFERENTIABLE transition."""
        from shs_rssm.pairwise_online import pair_kl_online
        B, T, K = gamma.shape
        dtype, device = gamma.dtype, gamma.device
        log_init = self.hdp.expected_log_init().to(dtype=dtype, device=device)
        isf = (torch.zeros(B, T, dtype=dtype, device=device) if is_first is None
               else is_first.reshape(B, T).to(dtype=dtype, device=device).clone())
        isf[:, 0] = 1.0
        gclamp = gamma.clamp_min(1e-30)
        out = gamma.new_zeros(B, T)
        init_kl = (gamma * (gclamp.log() - log_init.view(1, 1, K))).sum(-1)
        out = out + isf * init_kl
        if T <= 1:
            return out
        la, lb, lZ = cache["messages"]
        val = cache.get("valid", None)
        ev = cache["evidence_shifted"]
        base_elogpi = self.hdp.expected_log_trans().to(dtype=dtype, device=device)
        if self.recurrent:
            # review P2 #11: build each transition slice ON DEMAND (no O(BTK^2) tensor).
            # The DIFFERENTIABLE slices come from freshly-recomputed aux (for the stickiness
            # gradient); the DETACHED slices (for the xi recomputation) come from the cached
            # detached aux -- both are O(BTK) not O(BTK^2).
            phi = self.build_stick_phi(deter)
            aux_diff = self.rstick.bound_aux_only(base_elogpi, phi[:, 1:])
            trans_diff_fn = lambda t: self.rstick.trans_slice_from_aux(aux_diff, t)
            _caux = cache.get("trans_aux", None)
            if _caux is not None:
                trans_det_fn = lambda t: self.rstick.trans_slice_from_aux(_caux, t)
            else:
                trans_det_fn = lambda t: trans_diff_fn(t).detach()
        else:
            _lt = base_elogpi.view(1, 1, K, K).expand(B, T - 1, K, K)
            trans_diff_fn = _lt
            trans_det_fn = _lt.detach()
        pair_kl = pair_kl_online(la, lb, lZ, trans_det_fn, ev, is_first, val,
                                 gamma, trans_diff_fn)
        out[:, 1:] = out[:, 1:] + (1.0 - isf[:, 1:]) * pair_kl
        return out

    # ------------------------------------------------------------ dynamics KL
    def dynamics_kl(self, q_mean, q_std, stoch, deter, gamma, is_first=None,
                    free: float = 1.0, dyn_scale: float = 0.5, rep_scale: float = 0.1,
                    strict_elbo: bool = False, global_scale: float = 1.0, action=None):
        """Fully Bayesian structured VB dynamics objective.

        This uses the exact same local evidence as the E-step:

            ell_t(k) = E_{q(z_t) q(theta_k)}[log p(z_t | g_t, theta_k)].

        Given the forward-backward posterior q*(s) from `regime_inference`, the
        per-step negative dynamics ELBO is

            -sum_k gamma_tk ell_t(k) - H[q(z_t)]
            + KL_t(q(s) || p(s)).

        This removes the previous E-step/loss gap: regime responsibilities and
        gradients are now based on one identical Bayesian likelihood, including
        regime-specific noise and parameter uncertainty.  The returned tensors keep
        Dreamer's (B,T) KL-loss interface.  `dyn` and `rep` intentionally report the
        same VB continuous term, because there is no separate predictive-Gaussian
        dyn/rep KL in the fully Bayesian objective.  To run the mathematically exact
        objective without free-bits, set kl_free=0.
        """
        q_var = (q_std ** 2).clamp_min(1e-8)
        gam = gamma.detach()

        # Same target and same evidence convention as the analytic E-step: posterior
        # mean plus the trace correction from q_var.  `stoch` is kept in the signature
        # for API compatibility with Dreamer/RSSM callers.
        # pass actions so the OPTIMISED loss evidence matches the
        # action-conditioned E-step evidence (otherwise they diverge when action_dim>0)
        evidence, _, _ = self._vb_evidence(q_mean, deter, is_first=is_first, z_var=q_var,
                                           action=action)
        hq = diag_gauss_entropy(q_var)
        cont_vb = -(gam * evidence).sum(-1) - hq

        # Discrete-path KL from the cached E-step.  When the cache is absent or stale
        # (read-only diagnostics), recompute xi with the SAME evidence definition so the
        # loss remains internally consistent rather than silently mixing criteria.
        _c = getattr(self, "_struct_cache", None)
        if _c is not None and _c.get("online", False) \
                and _c.get("gamma", None) is not None and _c["gamma"].shape == gamma.shape:
            # blocker 8: online discrete-path KL, no materialised xi
            disc = self._discrete_path_kl_online(gam, deter, is_first, _c)
        else:
            xi = None
            if _c is not None and _c.get("gamma", None) is not None \
                    and _c["gamma"].shape == gamma.shape:
                xi = _c.get("xi", None)
            if xi is None:
                with torch.no_grad():
                    ev = evidence.detach()
                    ev = ev - ev.max(dim=-1, keepdim=True).values
                    log_init, log_trans, _ = self._transition_logpotentials(
                        deter.detach(), dtype=ev.dtype, device=ev.device)
                    _, _, _, xi = forward_backward(
                        log_init, log_trans, ev, is_first=is_first, return_pairwise=True)
            disc = self._discrete_path_kl(gam, xi.to(gam.device, gam.dtype), deter, is_first)

        # Global KL for the state-specific recurrent-stickiness variational posterior.
        # This is a true ELBO complexity term.  It is a scalar buffer-side penalty (the PG
        # posterior is updated in closed form), so it mainly makes the reported objective and
        # move scores honest; gradients to P_stick still come from the transition term above.
        rstick_kl = q_mean.new_tensor(0.0)
        if self.recurrent and getattr(self, "rstick", None) is not None:
            rstick_kl = self.rstick.beta_kl().to(dtype=q_mean.dtype, device=q_mean.device)
            rstick_kl = rstick_kl / max(1, q_mean.shape[0] * q_mean.shape[1])

        # Free bits are a Dreamer stabilizer, not part of the pure ELBO.  They are
        # applied only to the continuous VB term and can be disabled via kl_free=0.
        if strict_elbo:
            free = 0.0                       # no free bits inside a strict ELBO
        cont_for_loss = torch.clamp(cont_vb, min=free) if free is not None and free > 0 else cont_vb
        vb = cont_vb + disc + rstick_kl
        dyn = cont_for_loss + disc + rstick_kl
        rep = cont_for_loss
        value = vb
        # rstick_kl is already an ELBO KL term; include it unweighted in the optimization.
        loss = dyn_scale * (cont_for_loss + disc) + rep_scale * rep + rstick_kl
        if strict_elbo:
            # One analytic objective: the per-step negative LOCAL switching-head ELBO
            # (unit weight, no free bits, no representation-KL reweighting) plus the
            # GLOBAL complexity terms distributed uniformly over the scored steps so
            # that loss.sum() == -[ local ELBO + global_scale * global ELBO ] exactly.
            # global_scale should be (steps in this batch)/(steps in the corpus) for
            # minibatch SGD, and 1.0 for a full-batch fit.
            # NOTE: rstick beta_kl is part of bound_global, so it must NOT be added
            # separately here (that would double-count the stickiness complexity).
            loss = cont_vb + disc
            with torch.no_grad():
                gneg = -self.bound_global()
            loss = loss + (global_scale * gneg / max(loss.numel(), 1)) * torch.ones_like(loss)
        return loss, value, dyn, rep

    def _Epi(self):
        """E[pi_kl] restricted and renormalised over the active K regimes.

        transTheta has a K+1-th remainder column (mass to uninstantiated regimes);
        for the fixed-K active mixture we condition on the active set and renormalise
        each row to sum to 1, so the mixture weights form a proper distribution.
        """
        active = self.hdp.trans_theta[:, :self.K]
        return active / active.sum(-1, keepdim=True)

    @staticmethod
    def _shift_resp(gamma, is_first=None):
        """gamma_{t-1}; uniform at t=0 and at episode starts."""
        B, T, K = gamma.shape
        u = gamma.new_full((B, 1, K), 1.0 / K)
        prev = torch.cat([u, gamma[:, :-1]], dim=1) if T > 1 else u
        if is_first is not None:
            isf = is_first.reshape(B, T, 1).to(gamma.dtype)
            prev = torch.where(isf > 0.5, gamma.new_full((B, T, K), 1.0 / K), prev)
        return prev

    # ------------------------------------------------------------ global M-step
    @staticmethod
    def _shift_var(v, is_first=None):
        """Var(z_{t-1}): one-step shift of a per-step diagonal variance, with ZERO at episode
        resets (where z_{t-1} is the deterministic learned z0). Mirrors _prev_stoch but for a
        variance, so the regressor-uncertainty (Gram) correction is consistent across the
        online M-step, move refinement, and birth seeding."""
        if v is None:
            return None
        zeros = torch.zeros_like(v[:, :1])
        sv = torch.cat([zeros, v[:, :-1]], dim=1) if v.shape[1] > 1 else zeros
        if is_first is not None:
            isf = is_first.reshape(*is_first.shape[:2], 1).to(v.dtype)
            sv = torch.where(isf > 0.5, torch.zeros_like(sv), sv)
        return sv

    @torch.no_grad()
    @torch.no_grad()
    def get_extra_state(self):
        """Serialize the online sufficient-statistic ledger with the module state_dict,
        so memoized totals AND per-batch summaries survive checkpoint round trips
        (previously the globals survived but the ledger silently reset)."""
        return dict(stat_store=(self.stat_store.state_dict()
                                if self.stat_store is not None else None),
                    async_version=int(getattr(self, '_async_version', 0)),
                    struct_gen=int(getattr(self, '_struct_gen', 0)),
                    gstep_count=int(getattr(self, '_gstep_count', 0)),
                    episode_cursor=int(getattr(self, '_episode_cursor', 0)),  
                    # the EMA/blend-vs-overwrite flags MUST persist, else the
                    # first update after resume OVERWRITES the loaded stats instead of blending.
                    counts_initialised=bool(getattr(self, '_counts_initialised', False)),
                    stats_initialised=bool(getattr(self.regimes, '_stats_initialised', False)),
                    pg_init=bool(getattr(getattr(self, 'rstick', None), '_pg_init', False))
                    if getattr(self, 'rstick', None) is not None else False,
                    active_mask=(self.active_mask.tolist()
                                 if getattr(self, 'active_mask', None) is not None else None))

    def set_extra_state(self, state):
        self._async_version = int((state or {}).get("async_version", 0))
        self._struct_gen = int((state or {}).get("struct_gen", 0))
        self._gstep_count = int((state or {}).get("gstep_count", 0))
        self._episode_cursor = int((state or {}).get("episode_cursor", 0))
        # review P1 #8: restore the blend flags so accumulated stats are not overwritten
        st = state or {}
        self._counts_initialised = bool(st.get("counts_initialised", False))
        if getattr(self, 'regimes', None) is not None:
            self.regimes._stats_initialised = bool(st.get("stats_initialised", False))
        if getattr(self, 'rstick', None) is not None:
            self.rstick._pg_init = bool(st.get("pg_init", False))
        # review P1 #9: restore the fixed-Kmax active mask (registered as a buffer below,
        # but the bool list here covers older checkpoints / cross-device loads)
        _am = st.get("active_mask", None)
        if _am is not None:
            self.active_mask = torch.tensor(_am, dtype=torch.bool, device=self.M_device())
        payload = (state or {}).get("stat_store")
        if payload is not None and self.stat_store is not None:
            self.stat_store.load_state_dict(payload, device=self.ema_trans_counts.device)

    @torch.no_grad()
    def _structural_pre_load_hook(self, state_dict, prefix, local_metadata, strict,
                                  missing_keys, unexpected_keys, error_msgs):
        """Adaptive-K checkpoints: an accepted move changes every K-shaped tensor, but a
        restarted run reconstructs the CONFIGURED K, so vanilla load_state_dict would
        die on shape mismatch. Before tensor copy, reshape every registered
        parameter/buffer whose incoming shape differs (shapes read from the incoming
        state_dict itself -- no semantic dimension mapping, so K==L coincidences are
        safe) and update the structural K attributes."""
        resized = 0
        for name, buf in list(self.named_buffers(recurse=True)):
            key = prefix + name
            if key in state_dict and tuple(state_dict[key].shape) != tuple(buf.shape):
                owner = self
                parts = name.split(".")
                for p in parts[:-1]:
                    owner = getattr(owner, p)
                owner._buffers[parts[-1]] = torch.empty_like(state_dict[key],
                                                            device=buf.device)
                resized += 1
        for name, par in list(self.named_parameters(recurse=True)):
            key = prefix + name
            if key in state_dict and tuple(state_dict[key].shape) != tuple(par.shape):
                owner = self
                parts = name.split(".")
                for p in parts[:-1]:
                    owner = getattr(owner, p)
                setattr(owner, parts[-1],
                        torch.nn.Parameter(torch.empty_like(state_dict[key],
                                                            device=par.device),
                                           requires_grad=par.requires_grad))
                resized += 1
        key = prefix + "ema_trans_counts"
        if key in state_dict:
            k_new = int(state_dict[key].shape[0])
            if k_new != int(self.K):
                for obj in (self, self.regimes, self.hdp,
                            getattr(self, "rstick", None)):
                    if obj is not None and hasattr(obj, "K"):
                        obj.K = k_new

    def _post_load_refresh(self, module, incompatible_keys):
        """Recompute derived caches (Cholesky of the design precision) after load."""
        try:
            lam = self.regimes.lam
            self.regimes._lam_chol = torch.linalg.cholesky(
                lam + 1e-8 * torch.eye(lam.shape[-1], device=lam.device,
                                       dtype=lam.dtype))
        except Exception:
            self.regimes._lam_chol = None

    def begin_full_batch_pass(self):
        if self.stat_store is not None:
            self.stat_store.begin_full_batch_pass()

    @torch.no_grad()
    def finalize_full_batch_pass(self):
        """Force the single deferred global step of an OPEN full-batch pass from the
        accumulated totals (corpora without a declared expected_batches, or a ragged
        final pass). Complement of begin_full_batch_pass()."""
        st = self.stat_store
        if st is None or st.mode != "full_batch":
            raise ValueError(
                "finalize_full_batch_pass() requires online_mode='full_batch' "
                f"(store mode is {None if st is None else st.mode!r}) ")
        if getattr(st, "_fb_finalized", False):
            raise ValueError(
                "full_batch pass already finalized; call begin_full_batch_pass() "
                "before accumulating and finalizing a new pass")
        if st._tot_stats is None:
            raise ValueError("cannot finalize an EMPTY full_batch pass -- no batches "
                             "have been accumulated")
        ok = self.global_step_from_totals()
        if ok:
            st._fb_finalized = True   # FINALIZED only after a SUCCESSFUL global update
        return ok

    @torch.no_grad()
    def stream_episode(self, stoch, deter, is_first=None, z_var=None, action=None,
                       valid=None):
        """Completed-episode streaming DRIVER. Absorbs one COMPLETE
        episode exactly once under a MONOTONIC INTEGER offset -- constant memory (integer
        offsets keep no per-id set) -- advancing an internal cursor that is checkpointed
        via extra_state. This is the ingestion primitive a Dreamer collection loop calls
        once per finished episode; random replay minibatches must NOT be routed here (they
        would double-count). Returns the absorb_episode diagnostics plus the offset used.

        NOTE (honest scope): this driver and its absorb-once/constant-memory/checkpoint
        contract are unit-tested, but wiring the CALL-SITE into dreamer.py's environment
        rollout loop (detect episode completion -> encode under a stable representation ->
        stream_episode) is Dreamer-loop integration that is not exercised here."""
        off = int(getattr(self, "_episode_cursor", 0))
        diag = self.absorb_episode(off, stoch, deter, is_first=is_first, z_var=z_var,
                                   action=action, valid=valid)
        self._episode_cursor = off + 1
        diag = dict(diag); diag["stream_offset"] = off
        return diag

    @torch.no_grad()
    def absorb_episode(self, episode_id, stoch, deter, is_first=None, z_var=None,
                       action=None, valid=None):
        """The explicit unique-new-episode streaming interface.

        Contract: `stoch`/`deter` (and optional `z_var`, `action`) are ONE complete
        episode already encoded under the CURRENT representation; `episode_id` is its
        permanent id. The episode is absorbed exactly once -- the streaming store's
        absorb-once cursor/ledger rejects a duplicate id BEFORE any totals change -- its
        sufficient statistics are added atomically to the persistent totals, one global
        posterior step runs, and the switching-head ELBO contribution (scored under the
        pre-update globals) plus diagnostics are returned. The live Dreamer replay loop
        must NOT call this on randomly sampled replay minibatches (that path is
        live_ema); checkpointing the head checkpoints the cursor and ledger."""
        st = self.stat_store
        if st is None or st.mode != "streaming":
            raise ValueError(
                "absorb_episode requires online_mode='episode_stream' (streaming); got "
                f"{None if st is None else st.mode!r}")
        # blocker 8: ONE E-step. regime_inference runs forward-backward once and stashes
        # logZ; the pre-update ELBO is that logZ (same globals as a separate bound_local
        # call would use), so the previous double forward-backward is gone.
        gamma, tc, sc, _ = self.regime_inference(stoch, deter, is_first=is_first,
                                                 cache_estep=True, z_var=z_var,
                                                 action=action, valid=valid)
        _lz = getattr(self, "_last_logZ", None)
        elbo = float(_lz.sum()) if torch.is_tensor(_lz) else float(_lz or 0.0)
        self.update_globals(stoch, deter, gamma, tc, sc, is_first=is_first,
                            z_var=z_var, batch_id=episode_id, action=action)
        occ = gamma.detach().sum(dim=tuple(range(gamma.dim() - 1)))
        return dict(episode_id=episode_id, elbo=elbo, K=int(self.K),
                    occupancy=occ.cpu(), n_steps=int(gamma.shape[-2] * gamma.shape[0]
                                                     if gamma.dim() == 3 else gamma.shape[0]),
                    stream_count=int(getattr(st, "_stream_count", 0)))

    #  SDA-Bayes asynchronous master/worker (replica-based)
    def _lock(self):
        lk = getattr(self, "_commit_lock", None)
        if lk is None:
            lk = threading.RLock()
            self._commit_lock = lk
        return lk

    def bump_struct_gen(self):
        """Advance the structural GENERATION. Called after ANY accepted structural move
        (birth/split/merge/delete). A K equality check alone cannot detect a birth
        followed by a merge that returns to the same K with different state identities;
        the generation counter can."""
        self._struct_gen = int(getattr(self, "_struct_gen", 0)) + 1

    def master_snapshot(self):
        """SDA-Bayes: a COMPLETE copy of everything a worker needs to reproduce master
        inference: the FULL module state dict (emission Normal-Gamma, sticky-HDP and
        recurrent-PG posteriors, the learned carry projection P, recurrent projection
        P_stick, learned initial state z0, EMA count buffers) PLUS the mutable RUNTIME
        MODE (`recurrent`, which curricula flip on the live head) and structural metadata.
        Taken under the master lock so a concurrent commit cannot interleave a half-updated
        posterior into the copy."""
        with self._lock():
            return dict(
                version=int(getattr(self, "_async_version", 0)),
                struct_gen=int(getattr(self, "_struct_gen", 0)),
                repr_version=int(self.repr_version), K=int(self.K),
                recurrent=bool(self.recurrent),
                state={k: (v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v))
                       for k, v in self.state_dict().items()})

    def _snap_meta_of(self, snap):
        return dict(version=int(snap["version"]), struct_gen=int(snap["struct_gen"]),
                    repr_version=int(snap["repr_version"]), K=int(snap["K"]),
                    recurrent=bool(snap.get("recurrent", self.recurrent)))

    def load_snapshot(self, snap):
        """Install a master snapshot into THIS head (worker-replica use): tensors,
        runtime mode, counters, and the immutable snapshot identity `_snap_meta` that
        async_worker_delta tags deltas from. Reloading a reusable worker goes through
        here, so its posterior and its identity can never disagree."""
        self.load_state_dict(snap["state"])
        self.K = int(snap["K"])
        self.recurrent = bool(snap.get("recurrent", self.recurrent))
        self._async_version = int(snap["version"])
        self._struct_gen = int(snap["struct_gen"])
        self._snap_meta = self._snap_meta_of(snap)
        return self

    def make_worker(self, snapshot=None):
        """Build a DETACHED worker replica: a separate RegimeHead constructed with the
        master's constructor arguments and loaded from the snapshot (the structural
        pre-load hook resizes K-shaped tensors, so a snapshot taken after moves loads
        cleanly). Local inference on the replica cannot touch the master's parameters,
        `_struct_cache`, `_estep`, or any other live state. Build ONE
        replica per worker thread and refresh it by passing a new snapshot to
        async_worker_delta; commits serialize at the master's lock."""
        snap = self.master_snapshot() if snapshot is None else snapshot
        w = RegimeHead(**self._ctor_kwargs)
        return w.load_snapshot(snap)

    @torch.no_grad()
    def async_worker_delta(self, stoch, deter, is_first=None, z_var=None,
                           snapshot=None, worker=None, data_repr_version=None,
                           local_iters: int = 1, action=None, valid=None):
        """SDA-Bayes WORKER: run LOCAL forward-backward on a DETACHED replica and return
        the sufficient-statistic delta d_xi, tagged from the replica's OWN immutable
        snapshot identity `_snap_meta` -- never from a passed snapshot the replica might
        not hold. Reuse contract: pass `worker=` to reuse a replica; if
        `snapshot=` is also given, the snapshot is RELOADED into the replica first; with
        `worker=` alone the replica's existing (possibly deliberately stale) posterior is
        used and truthfully tagged.

        `data_repr_version` declares which representation encoded `stoch`/`deter`; it
        must match the replica's snapshot. It is REQUIRED under strict streaming; 
        otherwise optional, because raw-tensor provenance is
        not verifiable -- the declared contract is what can be checked."""
        if worker is None:
            snap = self.master_snapshot() if snapshot is None else snapshot
            w = self.make_worker(snap)
        else:
            w = worker
            if not hasattr(w, "_snap_meta"):
                raise ValueError(
                    "reusable worker was not built by make_worker()/load_snapshot(); it "
                    "has no snapshot identity to tag deltas from")
            if snapshot is not None:
                w.load_snapshot(snapshot)
        meta = w._snap_meta
        if data_repr_version is None:
            if self.stat_store is not None and getattr(self.stat_store, "strict_stream", False):
                raise ValueError(
                    "strict streaming REQUIRES data_repr_version: declare which "
                    "representation encoded these tensors")
        elif int(data_repr_version) != int(meta["repr_version"]):
            raise ValueError(
                f"worker data declares repr_version {int(data_repr_version)} but the "
                f"replica's snapshot is repr_version {int(meta['repr_version'])}; "
                "re-encode the data or load the matching snapshot")
        gamma, tc, sc, _ = w.regime_inference(stoch, deter, is_first=is_first,
                                              cache_estep=True, z_var=z_var,
                                              action=action, valid=valid)
        prev = w._prev_stoch(stoch, is_first)
        g = w.build_g(prev, deter, w._shift_action(action, is_first))
        g_z_var = w._shift_var(None if z_var is None else z_var, is_first)
        stats = w.regimes.stats_from_batch(
            gamma, stoch, g, z_var=z_var, g_z_var=g_z_var)
        pg = None
        if w.recurrent and w._estep is not None:
            es = w._estep
            Dp = es["phi_steps"].shape[-1]
            pg = w.rstick.pg_stats_from_batch(
                es["phi_steps"].reshape(-1, Dp), es["r_mass"].reshape(-1, w.K),
                es["row_weight"].reshape(-1, w.K))
        em_backup = None
        if int(local_iters) > 1:
            # Local coordinate iterations mutate the REPLICA's
            # emission posterior; restore it after the delta so a reused worker's next
            # task starts from its snapshot, not from this episode's local optimum.
            em_backup = ({k: v.detach().clone()
                          for k, v in w.regimes.state_dict().items()},
                         bool(getattr(w.regimes, "_stats_initialised", False)))
        for _it in range(max(0, int(local_iters) - 1)):
            # ITERATED LOCAL BatchVB ( cost/quality knob): coordinate-ascent on
            # the EMISSION block of the LOCAL posterior -- base prior + THIS batch's
            # statistics (the SDA-Bayes framework-1 primitive A(C, xi_0), warm-started
            # at the snapshot posterior's responsibilities). The transition and PG
            # blocks stay at the snapshot: a PARTIAL local CAVI, stated as such. With
            # local_iters=1 this reduces exactly to the previous single-E-step SSU.
            w.regimes.set_stats(stats)
            w.regimes.m_step()
            gamma, tc, sc, _ = w.regime_inference(stoch, deter, is_first=is_first,
                                                  cache_estep=True, z_var=z_var,
                                                  action=action, valid=valid)
            stats = w.regimes.stats_from_batch(
                gamma, stoch, g, z_var=z_var, g_z_var=g_z_var)
            if w.recurrent and w._estep is not None:
                es = w._estep
                Dp = es["phi_steps"].shape[-1]
                pg = w.rstick.pg_stats_from_batch(
                    es["phi_steps"].reshape(-1, Dp), es["r_mass"].reshape(-1, w.K),
                    es["row_weight"].reshape(-1, w.K))
        if em_backup is not None:
            w.regimes.load_state_dict(em_backup[0])
            w.regimes._stats_initialised = em_backup[1]
            if hasattr(w.regimes, "_lam_chol"):
                w.regimes._lam_chol = None
        w._estep = None
        return dict(s=stats, C=tc.detach().double(), v=sc.detach().double(), p=pg,
                    base_version=int(meta["version"]),
                    struct_gen=int(meta["struct_gen"]),
                    repr_version=int(meta["repr_version"]), K=int(meta["K"]),
                    recurrent=bool(meta["recurrent"]))

    @torch.no_grad()
    def async_commit(self, offset, delta, refresh: bool = True, tolerate_stale: bool = True,
                     max_stale: int | None = None):
        """SDA-Bayes MASTER: validate REQUIRED delta metadata (posterior version,
        structural generation, representation version, K, runtime mode -- a delta missing
        any is rejected, no defaults), commit to the totals, refresh the global
        posterior, bump the version. The WHOLE commit runs under the master lock, so
        concurrent commits cannot interleave backups and installs. 
        If the refresh fails, the store cursor, totals, EVERY tensor of the
        head (including the EMA count buffers) and the mutated plain attributes
        (`_counts_initialised`, the emission stats/cholesky caches, the PG init flag and
        guard counter) are ROLLED BACK. Posterior staleness
        is tolerated Hogwild-style unless tolerate_stale=False; structural,
        representation or runtime-mode mismatch is never tolerated."""
        with self._lock():
            for key in ("base_version", "struct_gen", "repr_version", "K", "recurrent"):
                if key not in delta:
                    raise ValueError(
                        f"async delta is missing required metadata {key!r}; deltas must "
                        "come from async_worker_delta")
            if int(delta["K"]) != int(self.K):
                raise ValueError(
                    f"async delta was computed at K={int(delta['K'])} but the master is "
                    f"now K={int(self.K)}; a structural change invalidates it")
            if int(delta["struct_gen"]) != int(getattr(self, "_struct_gen", 0)):
                raise ValueError(
                    f"async delta was computed at structural generation "
                    f"{int(delta['struct_gen'])} but the master is at generation "
                    f"{int(getattr(self, '_struct_gen', 0))}; an accepted "
                    "birth/merge/delete re-identifies the states even at equal K")
            if int(delta["repr_version"]) != int(self.repr_version):
                raise ValueError(
                    f"async delta was computed at representation version "
                    f"{int(delta['repr_version'])} but the master is now "
                    f"{int(self.repr_version)}; recompute against the current "
                    "representation")
            if bool(delta["recurrent"]) != bool(self.recurrent):
                raise ValueError(
                    f"async delta was computed with recurrent={bool(delta['recurrent'])} "
                    f"but the master now runs recurrent={bool(self.recurrent)}; the "
                    "runtime mode changed (curriculum switch) so the delta's statistics "
                    "are structurally wrong")
            cur = int(getattr(self, "_async_version", 0))
            if max_stale is not None and cur - int(delta["base_version"]) > int(max_stale):
                raise ValueError(
                    f"async delta is {cur - int(delta['base_version'])} versions stale at "
                    f"COMMIT time (> max_stale={int(max_stale)}); recompute against a "
                    "fresher snapshot")
            if not tolerate_stale and int(delta["base_version"]) != cur:
                raise ValueError(
                    f"async delta is stale (base_version {int(delta['base_version'])} != "
                    f"master {cur}) and tolerate_stale=False")
            st = self.stat_store

            def _cl(x):
                if x is None:
                    return None
                if torch.is_tensor(x):
                    return x.clone()
                if isinstance(x, dict):
                    return {k: _cl(v) for k, v in x.items()}
                if isinstance(x, (tuple, list)):
                    return type(x)(_cl(v) for v in x)
                return x

            store_backup = dict(
                tot=(_cl(st._tot_stats), _cl(st._tot_C), _cl(st._tot_s), _cl(st._tot_pg)),
                lo=st._async_lo, ahead=set(st._async_ahead), cnt=st._stream_count,
                nup=st.n_updates, api=st._stream_api)
            _MISSING = object()
            head_backup = None
            attr_backup = None
            if refresh:
                head_backup = {k: (v.detach().clone() if torch.is_tensor(v)
                                   else copy.deepcopy(v))
                               for k, v in self.state_dict().items()}
                attr_sites = [(self, "_counts_initialised"),
                              (self.regimes, "_stats_initialised"),
                              (self.regimes, "_lam_chol")]
                if self.recurrent:
                    attr_sites += [(self.rstick, "_pg_init"),
                                   (self.rstick, "n_pg_guard_rejects")]
                attr_backup = [(obj, name, _cl(getattr(obj, name, _MISSING)))
                               for obj, name in attr_sites]
            st.async_commit(offset, dict(s=delta["s"], C=delta["C"], v=delta["v"],
                                         p=delta["p"]))
            if refresh:
                try:
                    self.global_step_from_totals()
                except Exception as e:
                    st._tot_stats, st._tot_C, st._tot_s, st._tot_pg = store_backup["tot"]
                    st._async_lo = store_backup["lo"]
                    st._async_ahead = store_backup["ahead"]
                    st._stream_count = store_backup["cnt"]
                    st.n_updates = store_backup["nup"]
                    st._stream_api = store_backup["api"]
                    self.load_state_dict(head_backup)
                    for obj, name, val in attr_backup:
                        if val is _MISSING:
                            if hasattr(obj, name):
                                delattr(obj, name)
                        else:
                            setattr(obj, name, val)
                    raise ValueError(
                        f"async commit ROLLED BACK: the global refresh failed ({e}); "
                        "the offset was not consumed and the master -- tensors, EMA "
                        "counts, caches and flags -- is unchanged")
            self._async_version = cur + 1
            return int(self._async_version)

    def _hdp_update_scheduled(self, C, s):
        """run the (iterative, expensive) HDP root-stick optimisation
        only every `hdp_every` global steps -- the transition COUNTS still accumulate
        every step, so this trades a slightly staler transition posterior for much less
        compute per episode. hdp_every=1 (default) runs every step (unchanged)."""
        self._gstep_count = int(getattr(self, "_gstep_count", 0)) + 1
        every = int(getattr(self, "hdp_every", 1))
        if every <= 1 or ((self._gstep_count - 1) % every) == 0:
            self.hdp.update(C, s, n_global_iters=self.hdp_iters)
            return True
        return False

    def _pg_set_scheduled(self, A, h):
        """refresh the recurrent PG block every `pg_every` global steps
        (the totals A/h are always current; this only gates the posterior refresh)."""
        every = int(getattr(self, "pg_every", 1))
        if every <= 1 or ((int(getattr(self, "_gstep_count", 1)) - 1) % every) == 0:
            self.rstick.pg_set_totals(A, h)
            return True
        return False

    def global_step_from_totals(self):
        """One coordinate-ascent global update from the store's CURRENT totals without
        touching the per-batch ledger (the final global step of a post-move rebuild:
        fresh complete summaries first, then exactly one global update from them)."""
        st = self.stat_store
        if st is None or st._tot_stats is None:
            return False
        self.regimes.set_stats(st._tot_stats)
        self.regimes.m_step()
        self.ema_trans_counts.copy_(st._tot_C)
        self.ema_start_counts.copy_(st._tot_s)
        self._counts_initialised = True
        self._hdp_update_scheduled(st._tot_C, st._tot_s)
        if self.recurrent and st._tot_pg is not None:
            self._pg_set_scheduled(st._tot_pg["A"], st._tot_pg["h"])
        return True

    def resync_store_from_buffer(self, buffer):
        """After an ACCEPTED structural move: rebuild the memoized ledger from the
        complete move buffer under the INSTALLED (candidate) model, so the ledger
        equals the accepted candidate's whole-corpus statistics rather than
        row-remapped pre-move summaries. The final update_globals call re-runs the
        global M-step from the fresh totals, making parameters and ledger a
        consistent coordinate-ascent state (Hughes' post-move fresh summaries)."""
        store = getattr(self, "stat_store", None)
        if store is None or store.mode != "memoized" or buffer is None:
            return False
        batches = list(getattr(buffer, "batches", []))
        if not batches or any(b.batch_id is None for b in batches):
            return False
        store.reset()
        ids = []
        for b in batches:
            _bact = getattr(b, "action", None)   # blocker 5: post-move rebuild uses actions
            gamma, counts, sc, _ = self.regime_inference(
                b.stoch, b.deter, b.is_first, cache_estep=True, z_var=b.z_var,
                action=_bact)
            self.update_globals(b.stoch, b.deter, gamma, counts, sc,
                                is_first=b.is_first, z_var=b.z_var, batch_id=b.batch_id,
                                stats_only=True, action=_bact)
            ids.append(b.batch_id)
        store.expected_ids = set(ids)
        store.expected_batches = len(ids)
        # final global step from the COMPLETE rebuilt totals (never from a partial
        # ledger): a pure coordinate-ascent update, so the accepted candidate's exact
        # gain is preserved and then improved upon.
        self.global_step_from_totals()
        return True

    def begin_repr_epoch(self):
        """freeze the representation VERSION for a streaming/
        consolidation epoch so every episode absorbed during the epoch shares one latent
        coordinate system and the store accepts them (it otherwise rejects mixed
        versions after the first). The CALLER must ALSO hold the encoder fixed for the
        epoch (a frozen target copy -- see WorldModel.begin_shs_repr_epoch); this method
        only pins the version the SHS sufficient statistics are stamped with."""
        self._repr_frozen = True
        return int(self.repr_version)

    def end_repr_epoch(self):
        """Close the epoch: the next representation is a NEW version (so a later epoch's
        statistics are never mixed with this one's)."""
        self._repr_frozen = False
        self.repr_version += 1
        return int(self.repr_version)

    def enable_fixed_kmax(self, n_active=None):
        """Blocker 4: switch to fixed-Kmax active masking at the current capacity K. The
        first `n_active` slots (default: all) are active; the rest are inactive spares that
        birth can later activate. Tensor shapes never change again."""
        n = int(self.K if n_active is None else n_active)
        m = torch.zeros(self.K, dtype=torch.bool, device=self.M_device())
        m[:max(0, min(n, self.K))] = True
        self.active_mask = m
        return self

    def M_device(self):
        try:
            return self.regimes.M.device
        except Exception:
            return torch.device("cpu")

    def n_active(self):
        """Effective number of live regimes (K_eff)."""
        return int(self.K if self.active_mask is None else int(self.active_mask.sum()))

    def activate_slot(self):
        """Blocker 4 birth: activate the lowest inactive slot; returns its index or None if
        the fixed capacity is full."""
        if self.active_mask is None:
            return None
        idx = (~self.active_mask).nonzero(as_tuple=False)
        if idx.numel() == 0:
            return None
        k = int(idx[0])
        self.active_mask[k] = True
        return k

    def deactivate_slot(self, k):
        """Blocker 4 delete: mark slot k inactive (shape unchanged)."""
        if self.active_mask is not None and 0 <= int(k) < self.K:
            self.active_mask[int(k)] = False
        return self

    @torch.no_grad()
    def birth_into_slot(self):
        """fixed-Kmax BIRTH -- activate the lowest spare slot AND
        reset its dynamics parameters to the prior (a fresh regime), so K is unchanged and
        tensor shapes are stable. Returns the slot index, or None if capacity is full."""
        k = self.activate_slot()
        if k is not None and hasattr(self.regimes, "reset_slot"):
            self.regimes.reset_slot(k)
        return k

    @torch.no_grad()
    def delete_slot(self, k):
        """fixed-Kmax DELETE -- deactivate slot k AND clear its
        parameters/stats, shape-stable."""
        self.deactivate_slot(k)
        if hasattr(self.regimes, "reset_slot"):
            self.regimes.reset_slot(int(k))
        return self

    def _mask_logpotentials(self, log_init, log_trans):
        """extend the fixed-Kmax active mask to the TRANSITION
        potentials (not just the E-step evidence), so INACTIVE regimes get zero mass in
        the initial distribution AND as both transition origin and target. Applied on the
        E-step path and (via masked weights) in imagination."""
        if self.active_mask is None:
            return log_init, log_trans
        m = self.active_mask.to(log_init.device)
        neg = log_init.new_full((), -1e30)
        log_init = torch.where(m, log_init, neg)
        K = int(m.shape[0])
        row = m.view(*([1] * (log_trans.dim() - 2)), K, 1)
        col = m.view(*([1] * (log_trans.dim() - 2)), 1, K)
        log_trans = torch.where(row & col, log_trans, log_trans.new_full((), -1e30))
        return log_init, log_trans

    def _mask_weights(self, w):
        """Zero INACTIVE regimes in a mixture-weight vector and renormalise (P1 #9)."""
        if self.active_mask is None:
            return w
        m = self.active_mask.to(w.device, w.dtype)
        w = w * m
        return w / w.sum(-1, keepdim=True).clamp_min(1e-12)

    def _apply_active_mask(self, ev):
        """Set inactive regimes' evidence to -inf so forward-backward gives them zero
        responsibility (Blocker 4). No-op when active_mask is None."""
        if self.active_mask is None:
            return ev
        m = self.active_mask.to(ev.device)
        neg = torch.full_like(ev, -1e30)
        return torch.where(m.view(*([1] * (ev.dim() - 1)), -1), ev, neg)

    def bump_repr_version(self):
        """Signal that the upstream representation changed (invalidates stored latents)."""
        if getattr(self, "_repr_frozen", False):
            return   # blocker 3: version pinned during a representation epoch
        self.repr_version += 1

    def update_globals(self, stoch, deter, gamma, trans_counts, start_counts,
                       is_first=None, z_var=None, batch_id=None, stats_only=False,
                       repr_version=None,
                       action=None, valid=None):
        """EMA sufficient statistics + closed-form update of the regime and HDP globals.

        When `z_var` (the diagonal posterior variance) is supplied, `stoch` is taken to be
        the posterior MEAN and the regime M-step uses the exact responsibility-weighted
        second moments E[z z^T] = z z^T + diag(z_var) rather than the outer product of a
        single encoder sample (analytic-VB / Rao-Blackwellised update; same fixed point,
        lower variance).
        """
        if valid is not None:
            # padding marginals must be zero (regime_inference already masks them; this
            # is a defensive re-mask so a hand-built gamma is also safe)
            gamma = gamma * valid.reshape(*gamma.shape[:-1], 1).to(gamma.dtype)
        prev = self._prev_stoch(stoch, is_first)
        act = self._shift_action(action, is_first)
        g = self.build_g(prev, deter, act)
        # regressor z-block variance Var(z_{t-1}) for the Gram correction (zero at resets)
        g_z_var = self._shift_var(None if z_var is None else z_var.detach(), is_first)
        # regime model: EMA the responsibility-weighted statistics, then closed-form
        stats = self.regimes.stats_from_batch(
            gamma.detach(), stoch.detach(), g.detach(),
            z_var=None if z_var is None else z_var.detach(), g_z_var=g_z_var)
        tc = trans_counts.detach().double()
        sc = start_counts.detach().double()
        if self.stat_store is not None:
            # explicit online-VB semantics (full_batch / streaming / memoized) 
            pg = None
            if self.recurrent and self._estep is not None:
                es = self._estep
                Dp = es["phi_steps"].shape[-1]
                pg = self.rstick.pg_stats_from_batch(
                    es["phi_steps"].reshape(-1, Dp),
                    es["r_mass"].reshape(-1, self.K),
                    es["row_weight"].reshape(-1, self.K))
            if batch_id is None and self.stat_store.mode == "memoized":
                raise ValueError(
                    "online_mode='memoized' requires a STABLE batch_id per corpus "
                    "partition (Hughes replace semantics). Pass update_globals(..., "
                    "batch_id=...) or SHSRSSM.set_batch_id(...); an auto-counter would "
                    "append every replay visit as a new batch and silently break "
                    "memoized/streaming guarantees.")
            # streaming/full_batch: an auto id encodes absorb-once / per-pass semantics
            if batch_id is not None:
                bid = batch_id
            elif self.stat_store.mode == "streaming":
                bid = int(self.stat_store.n_updates)   # integer offset => O(1) cursor
            else:
                bid = f"auto{self.stat_store.n_updates}"
            st_tot, C_tot, s_tot, pg_tot = self.stat_store.add_batch(
                bid, stats, tc, sc, pg,
                repr_version=int(self.repr_version if repr_version is None
                                 else repr_version))
            if (self.stat_store.mode == "full_batch" and not stats_only
                    and (self.stat_store.expected_batches is None
                         or self.stat_store.n_batches < self.stat_store.expected_batches)):
                # full-batch CAVI: globals stay FROZEN throughout the pass. The single
                # global step fires ONLY when a declared count is reached (below) or via
                # finalize_full_batch_pass(). An UNDECLARED (expected_batches=None) or
                # ragged pass therefore never does corpus-prefix updates -- it stays
                # frozen until finalize is called.
                self._estep = None
                return
            if stats_only:
                # Ledger-only write (post-move resync): record this batch's summaries
                # under the CURRENT parameters and change nothing else, so the installed
                # accepted candidate is preserved exactly. Running the M-step / HDP / ARD
                # here would fire on a partially rebuilt ledger (the store was reset),
                # i.e. a fraction of the corpus against full-corpus priors.
                self._estep = None
                return
            self.regimes.set_stats(st_tot)
            self.regimes.m_step()
            self.ema_trans_counts.copy_(C_tot)
            self.ema_start_counts.copy_(s_tot)
            self._counts_initialised = True
            self._hdp_update_scheduled(C_tot, s_tot)
            if self.recurrent and pg_tot is not None:
                self._pg_set_scheduled(pg_tot["A"], pg_tot["h"])   # pg_every on main path
            if self.stat_store.mode == "full_batch":
                # declared boundary FINALIZED only AFTER the M-step/HDP/PG all
                # succeeded: a failure above leaves the
                # pass NOT finalized, so it can be retried cleanly
                self.stat_store._fb_finalized = True
            self._estep = None
            return
        # ---- legacy EMA path (Dreamer online default; forgetting-factor estimator) ----
        self.regimes.ema_update_stats(stats, self.ema_tau)
        self.regimes.m_step()
        # EMA the transition / start counts, then update the sticky-HDP from the smoothed counts
        if not self._counts_initialised:
            self.ema_trans_counts.copy_(tc)
            self.ema_start_counts.copy_(sc)
            self._counts_initialised = True
        else:
            self.ema_trans_counts.mul_(1 - self.ema_tau).add_(self.ema_tau * tc)
            self.ema_start_counts.mul_(1 - self.ema_tau).add_(self.ema_tau * sc)
        self._hdp_update_scheduled(self.ema_trans_counts, self.ema_start_counts)

        # recurrent stickiness: one online Polya-Gamma M-step from state-specific
        # soft persistence labels. row_weight is already zero on episode-reset links
        # because xi carries no transition mass there.
        if self.recurrent and self._estep is not None:
            es = self._estep
            phi = es["phi_steps"]              # (B,S,D+1)
            r_mass = es["r_mass"]              # (B,S,K)
            row_weight = es["row_weight"]      # (B,S,K)
            B, S, K = r_mass.shape
            self.rstick.pg_update_statewise(
                phi.reshape(-1, phi.shape[-1]),
                r_mass.reshape(-1, K),
                row_weight.reshape(-1, K),
                lr=self.ema_tau,
            )
            self._estep = None

    # acceptance bound
    def _score_potentials(self, deter, hdp, rstick, dtype=torch.float64):
        """Log potentials (init, trans, aux) for scoring under GIVEN globals.

        With `rstick` (a RecurrentStickiness at hdp.K) the transition potentials are
        the PG/JJ-bounded time-varying augmented potentials; otherwise the stationary
        base E[log pi]. Used by bound_local and by the moves' candidate E-steps so
        that base and candidate are always scored under the SAME model class.
        """
        device = deter.device
        log_init = hdp.expected_log_init().to(dtype=dtype, device=device)
        base_elogpi = hdp.expected_log_trans().to(dtype=dtype, device=device)
        if rstick is not None:
            phi = self.build_stick_phi(deter).to(dtype)
            log_trans, aux = rstick.bound_log_trans(base_elogpi, phi[:, 1:])
            return log_init, log_trans, dict(aux, phi_steps=phi[:, 1:])
        return log_init, base_elogpi, None

    def _score_rstick(self, regimes=None, rstick=None):
        """The stickiness module a score should use: explicit candidate, or the live
        module when recurrence is on and the live globals are being scored."""
        if rstick is not None:
            return rstick
        if self.recurrent and (regimes is None or regimes is self.regimes):
            return getattr(self, "rstick", None)
        return None

    @torch.no_grad()
    def bound_local(self, stoch, deter, is_first=None, regimes=None, hdp=None,
                    rstick=None, z_var=None, action=None, valid=None):
        """Local part of the frozen variational ELBO on one batch: sum_b logZ_b.

        Also returns the expected BASE-branch transition counts and start counts under
        the same posterior (needed by callers that refit candidates). With the exact
        allocation slack (`StickyHDP.exact_alloc_elbo`) the whole count dependence of
        the ELBO enters through logZ, so the local score is just the log-partition of
        forward-backward run under E[log pi] (and, when recurrent, the PG/JJ-bounded
        augmented potentials). Computed in float64, no max-subtraction, so scores are
        absolute and comparable across candidates.
        """
        regimes = regimes if regimes is not None else self.regimes
        hdp = hdp if hdp is not None else self.hdp
        rstick = self._score_rstick(regimes=regimes, rstick=rstick)
        prev = self._prev_stoch(stoch, is_first)
        act = self._shift_action(action, is_first)
        g = self.build_g(prev, deter, act)
        g_var = self._g_var_from_z_var(z_var, g, is_first=is_first)
        ev = regimes.expected_loglik(stoch, g, z_var=z_var, g_var=g_var).double()  # (B,T,K)
        log_init, log_trans, aux = self._score_potentials(deter, hdp, rstick)
        gamma, xicount, logZ, xi = forward_backward(
            log_init, log_trans, ev, is_first=is_first, valid=valid,
            return_pairwise=True)
        if aux is not None:
            _, _, counts = rstick.attribute_bound(xi, aux)
        else:
            counts = xicount
        start_counts = start_counts_from(gamma, is_first, valid=valid)
        return float(logZ.sum()), counts.double(), start_counts.double()

    @torch.no_grad()
    def bound_global(self, regimes=None, hdp=None, rstick=None):
        """Global part of the frozen variational ELBO (counted ONCE per scored set):

            - sum_l KL(q(theta_l)||p(theta_l))  - sum_i KL(q(beta_i)||p(beta_i))
            + [L_top - c_Dir(theta) - c_Dir(theta0) + linear slack]  (exact_alloc_elbo)

        The slack form makes the score the true surrogate ELBO for arbitrary theta,
        including frozen-globals scoring where theta was NOT refit on the scored
        counts. When a candidate is scored with recurrence active, the candidate's
        (resized) stickiness module contributes its own KL, so base and candidate pay
        symmetric complexity.
        """
        regimes = regimes if regimes is not None else self.regimes
        hdp = hdp if hdp is not None else self.hdp
        rstick = self._score_rstick(regimes=regimes, rstick=rstick)
        param_kl = regimes.param_kl().sum().double()
        if rstick is not None:
            param_kl = param_kl + rstick.beta_kl().double()
        return float(hdp.exact_alloc_elbo() - param_kl)

    @torch.no_grad()
    def bound(self, stoch, deter, is_first=None, regimes=None, hdp=None, z_var=None,
              rstick=None):
        """Frozen variational ELBO on a SINGLE batch (local + global).

        L = logZ - param_kl - rstick_kl + exact allocation term. Exactly the
        structured surrogate ELBO of Hughes et al. (NIPS 2015) extended with the
        PG/JJ-bounded recurrent-stickiness factors when recurrence is active; the
        recurrent terms are inside logZ, so the same acceptance rule "accept iff L
        improves" verifies moves under the true (recurrent) model. When aggregating
        several batches, use bound_local per batch and bound_global once (see
        moves.aggregate_bound) so global complexity is not multiple-counted.
        """
        local, _, _ = self.bound_local(stoch, deter, is_first=is_first, regimes=regimes,
                                       hdp=hdp, rstick=rstick, z_var=z_var)
        return local + self.bound_global(regimes=regimes, hdp=hdp, rstick=rstick)

    # imagination prior
    def imagine_prior(self, prev_stoch, deter, resp_prev, sample=True, prev_var=None,
                      action=None, mode=None):
        """One-step mixture prior for actor-critic rollouts.

        Returns next-z (B,L), the new regime responsibilities (B,K), and the
        moment-matched mixture mean and std (B,L) so the state's mean/std diagnostics
        stay consistent with the actual mixture (not the unused base Gaussian head).
        Uses the effective (h-dependent) transition when recurrent stickiness is on.

        NOT wrapped in no_grad: during actor learning the value gradient must flow
        through the imagined dynamics to the policy. The regime globals are buffers and
        the world-model parameters are frozen by the actor loop, but the graph through
        the actions/states (and the reparameterised z = mean + std * eps) must remain.
        """
        g = self.build_g(prev_stoch, deter, action)
        if prev_var is not None:
            zeros_tail = g[..., self.L:] * 0.0
            g_var = torch.cat([prev_var.to(g.dtype), zeros_tail], dim=-1)
        else:
            g_var = None
        base_elogpi = self.hdp.expected_log_trans().to(prev_stoch.dtype)
        if self.recurrent:
            phi = self.build_stick_phi(deter)                       # (B,D+1)
            Pi = torch.softmax(base_elogpi, dim=-1)
            sig = self.rstick.sigma(phi)                            # (B,K), row-specific
            eye = torch.eye(self.K, dtype=prev_stoch.dtype, device=prev_stoch.device)
            M = sig[..., :, None] * eye + (1.0 - sig[..., :, None]) * Pi  # (B,K,K)
            w = torch.einsum("bk,bkl->bl", resp_prev, M).clamp_min(1e-8)
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-8)
            w = self._mask_weights(w)   # P1 #9: no mass on inactive regimes in imagination
        else:
            Epi = self._Epi().to(prev_stoch.dtype)
            w = mixture_weights(resp_prev, Epi).clamp_min(1e-8)        # (B,K)
            w = w / w.sum(-1, keepdim=True)  # renormalise
            w = self._mask_weights(w)   # P1 #9
        comp_mean, comp_var = self.regimes.predictive_moments(g, g_var=g_var)  # (B,K,L)
        # marginal (moment-matched) mean and variance over the mixture
        mean = (w.unsqueeze(-1) * comp_mean).sum(-2)              # (B,L)
        dev = comp_mean - mean.unsqueeze(-2)                      # (B,K,L)
        var = (w.unsqueeze(-1) * (comp_var + dev.pow(2))).sum(-2)
        var = var.clamp_min(1e-8)
        std = var.sqrt()

        # EXPLICIT imagination modes.
        #   actor_moment    -> differentiable moment-matched single Gaussian; the value
        #                      gradient flows through mean/std (correct for imag_gradient=
        #                      'dynamics'). Belief propagates as the full mixture vector w.
        #   eval_sample     -> draw a regime ~ Cat(w) and a reparameterised Gaussian from
        #                      it; the belief becomes the SAMPLED one-hot, so the next
        #                      step conditions on the drawn regime (a temporally coherent
        #                      sticky path). Nondifferentiable discrete pick (stop-grad).
        #   reinforce_sample-> eval_sample PLUS the categorical log-prob is stashed
        #                      (self._last_imag_logprob) for a score-function estimator.
        #                      HONEST (review Important #7): the actor loss does NOT yet
        #                      consume this log-prob, so reinforce_sample currently behaves
        #                      like eval_sample plus a stashed term a REINFORCE actor would
        #                      read; it is a hook, not a wired score-function estimator.
        _mode = mode
        if _mode is None:
            _mode = "eval_sample" if getattr(self, "imag_sample_mixture", False) else "actor_moment"
        if _mode not in ("actor_moment", "eval_sample", "reinforce_sample"):
            raise ValueError(f"unknown imagination mode {_mode!r}")
        resp_out = w
        if (not sample) or _mode == "actor_moment":
            z = mean if not sample else (mean + std * torch.randn_like(mean))
        else:
            # Sample the TRUE mixture: draw a component ~ Categorical(w), then a
            # reparameterised Gaussian from that component. The continuous draw keeps the
            # value gradient (it flows through the chosen component's mean/std); the
            # discrete component pick is stop-grad. This is faithful when the prior is
            # genuinely multimodal, where the moment-matched single Gaussian would place
            # mass in the low-density region between the regimes. (Default off; the
            # moment-matched sampler below keeps a single reparameterised Gaussian, which
            # is lower-variance for the actor.)
            idx = torch.distributions.Categorical(probs=w).sample()        # (...)
            L = comp_mean.shape[-1]
            sel = idx.unsqueeze(-1).unsqueeze(-1).expand(*idx.shape, 1, L)  # (...,1,L)
            sel_mean = torch.gather(comp_mean, -2, sel).squeeze(-2)         # (...,L)
            if self.regimes.q_rank > 0:
                # draw from the FULL low-rank-plus-diagonal component covariance
                #   z = mu + sqrt(d) eps_L + U eps_r,  Cov = diag(d) + U U^T
                _, comp_d, comp_U = self.regimes.predictive_cov_moments(g, g_var=g_var)
                r = comp_U.shape[-1]
                sel_d = torch.gather(comp_d, -2, sel).squeeze(-2)           # (...,L)
                selU = idx.reshape(*idx.shape, 1, 1, 1).expand(*idx.shape, 1, L, r)
                sel_U = torch.gather(comp_U, -3, selU).squeeze(-3)          # (...,L,r)
                eps_r = torch.randn(*sel_mean.shape[:-1], r,
                                    device=sel_mean.device, dtype=sel_mean.dtype)
                z = (sel_mean + sel_d.sqrt() * torch.randn_like(sel_mean)
                     + torch.einsum("...lr,...r->...l", sel_U, eps_r))
            else:
                sel_var = torch.gather(comp_var, -2, sel).squeeze(-2)
                z = sel_mean + sel_var.sqrt() * torch.randn_like(sel_mean)
            # coherent STICKY path: propagate the SAMPLED regime as a one-hot belief so
            # the next transition conditions on the draw, not the smeared mixture.
            resp_out = torch.zeros_like(w).scatter_(-1, idx.unsqueeze(-1), 1.0)
            if _mode == "reinforce_sample":
                self._last_imag_logprob = torch.log(
                    w.gather(-1, idx.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12))
        # Safety net for actor-imagination rollouts: the AR map is (near-)marginally stable,
        # so latent plus noise can slowly random-walk toward overflow over a long horizon
        # (especially under float16). Clamp to a range far outside the encoder's latent scale
        # so this never affects normal dynamics or gradients but cannot diverge.
        zmax = 20.0
        std = std.clamp(max=5.0)
        mean = mean.clamp(-zmax, zmax)
        z = z.clamp(-zmax, zmax)
        # propagate the regime belief one step (mixture vector for actor_moment, sampled
        # one-hot for eval_sample/reinforce_sample)
        return z, resp_out, mean, std

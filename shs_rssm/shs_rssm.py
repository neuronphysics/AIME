"""Drop-in RSSM subclass for dreamerv3-torch that swaps the single-Gaussian dynamics
prior for the sticky-HDP switching mixture.

Usage (in dreamerv3-torch `models.py`, WorldModel.__init__):

    from shs_rssm.shs_rssm import SHSRSSM
    self.dynamics = SHSRSSM(
        # ... all the usual RSSM kwargs from config ...
        shs_K=16, shs_proj_dim=64, shs_kappa=50.0, shs_gamma=5.0,
    )

Nothing else in the training loop changes: WorldModel._train still calls
`observe(...)` and then `kl_loss(post, prior, free, dyn_scale, rep_scale)`. This
subclass:
  * keeps `obs_step` (the amortised encoder q_phi(z_t|h_t,x_t)) and the GRU verbatim;
  * stashes `is_first` during `observe` (kl_loss does not otherwise receive it);
  * overrides `kl_loss` to (a) run the regime forward-backward, (b) return the
    structured variational switching KL in place of the Gaussian-Gaussian dynamics KL, and
    (c) update the regime and sticky-HDP globals by EMA + closed form (no grad);
  * overrides `img_step` so imagination samples the mixture prior.

Only continuous latents are supported (discrete=False), which is the SHS-RSSM case.
"""
from __future__ import annotations

import torch

import networks  # dreamerv3-torch
from .regime_head import RegimeHead
from .moves import sweep_moves, MoveBuffer


class SHSRSSM(networks.RSSM):
    def __init__(self, *args,
                 shs_K: int = 16,
                 shs_proj_dim: int | None = 64,
                 shs_gamma: float = 5.0,
                 shs_alpha: float = 1.0,
                 shs_kappa: float = 50.0,
                 shs_start_alpha: float = 1.0,
                 shs_a0: float = 3.0, shs_b0: float = 2.0, shs_v0_scale: float = 1.0,
                 shs_ard: bool = False, shs_ema_tau: float = 0.02,
                 shs_hdp_iters: int = 2, shs_global_update_every: int = 1,
                 shs_free: float = 1.0,
                 shs_recurrent: bool = False, shs_prior_persist: float = 0.9,
                 shs_pg_iters: int = 4, shs_rstick_dim: int | None = 8,
                 shs_rstick_stopgrad: bool = True,
                 shs_move_every: int = 0, shs_move_warmup: int = 2000,
                 shs_move_birth: bool = True, shs_move_buffer: int = 8,
                 shs_move_max_age: int = 0, shs_move_split: bool = True,
                 shs_move_confirm_top: int | None = None,
                 shs_delete_mode: str = "hughes", shs_merge_select: str = "hughes",
                 shs_merge_passes: int = 3,
                 shs_q_rank: int = 0, shs_imag_sample_mixture: bool = False,
                 shs_imag_mode: str = "actor_moment",
                 shs_online_pairwise: bool = False,
                 shs_analytic_estep: bool = True,
                 shs_shared_carry: bool = True,
                 shs_curriculum: list | None = None,
                 shs_online_mode: str = "ema",
                 shs_expected_batches: int | None = None,
                 shs_strict_stream: bool = False,
                 shs_hdp_every: int = 1,
                 shs_pg_every: int = 1,
                 shs_action_dim: int = 0,
                 shs_merge_topm: int | None = None,
                 shs_global_source: str = "replay_ema",
                 shs_strict_elbo: bool = False,
                 shs_birth_style: str = "interval",
                 shs_corpus_batches: int | None = None,
                 shs_entropy_mode: str = "bounds",
                 shs_moves_complete: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert not self._discrete, "SHSRSSM supports continuous latents only"
        self.regime = RegimeHead(
            stoch=self._stoch, deter=self._deter, K=shs_K, proj_dim=shs_proj_dim,
            a0=shs_a0, b0=shs_b0, v0_scale=shs_v0_scale, ard=shs_ard,
            gamma=shs_gamma, alpha=shs_alpha, kappa=shs_kappa,
            start_alpha=shs_start_alpha, ema_tau=shs_ema_tau, hdp_iters=shs_hdp_iters,
            recurrent=shs_recurrent, prior_persist=shs_prior_persist, pg_iters=shs_pg_iters,
            rstick_dim=shs_rstick_dim, rstick_stopgrad=shs_rstick_stopgrad,
            q_rank=shs_q_rank, shared_carry=shs_shared_carry,
            action_dim=shs_action_dim,
            online_mode=shs_online_mode, expected_batches=shs_expected_batches,
            strict_stream=shs_strict_stream,
            hdp_every=shs_hdp_every, pg_every=shs_pg_every,
            device=self._device,
        )
        # strict-ELBO profile: the returned training loss is one complete analytic
        # switching-head ELBO (no free bits, unit dyn scale, no representation-KL
        # reweighting, globals included). Reject silently-incompatible settings
        # rather than overriding them behind the caller's back.
        self._shs_strict_elbo = bool(shs_strict_elbo)
        if self._shs_strict_elbo:
            problems = []
            if shs_free not in (0, 0.0, None):
                problems.append(f"shs_free={shs_free!r} -> set `shs_free: 0` "
                                "(free bits change the objective)")
            if shs_ard:
                problems.append("shs_ard=True -> set `shs_ard: False` (MacKay "
                                "empirical-Bayes hyperprior updates are not ELBO ascent)")
            if shs_q_rank and int(shs_q_rank) > 0 and not shs_shared_carry:
                problems.append("shs_q_rank>0 without shs_shared_carry -> either set "
                                "`shs_q_rank: 0` or enable `shs_shared_carry: true`; only "
                                "the shared-carry path carries the fully variational "
                                "factor-augmented low-rank noise (Gaussian q(U) + KL, "
                                "local f). The non-shared low-rank path has no q(U).")
            if shs_curriculum:
                problems.append("shs_curriculum non-empty -> set `shs_curriculum: []` "
                                "(staged objective changes are not one ELBO)")
            if shs_online_mode not in ("memoized", "full_batch"):
                problems.append(f"shs_online_mode={shs_online_mode!r} -> use 'memoized' "
                                "(stable ids) or 'full_batch' (explicit pass boundaries); "
                                "'ema' is a forgetting-factor estimator, not coordinate "
                                "ascent on one fixed-corpus ELBO, and 'streaming' is "
                                "incompatible with replay revisits")
            if shs_move_every and int(shs_move_every) > 0:
                problems.append("shs_move_every>0 -> set `shs_move_every: 0` (live "
                                "ring-buffer moves carry only a buffer-local guarantee; "
                                "strict permits complete-corpus consolidation moves only)")
            if shs_rstick_stopgrad:
                problems.append("shs_rstick_stopgrad=True -> set `shs_rstick_stopgrad: "
                                "False` (stopped gradients make the training gradient "
                                "differ from the gradient of the declared ELBO)")
            if not shs_corpus_batches or int(shs_corpus_batches) < 1:
                problems.append("shs_corpus_batches unset -> declare how many replay "
                                "minibatches constitute ONE corpus so the global KL is "
                                "charged once per corpus, not once per minibatch "
                                "(`shs_corpus_batches: 1` for full-batch training)")
            if problems:
                raise ValueError(
                    "shs_strict_elbo=True requires a strict-compatible configuration:\n  - "
                    + "\n  - ".join(problems))
        self._shs_corpus_batches = shs_corpus_batches
        self._global_scale = (1.0 / float(shs_corpus_batches)
                              if (self._shs_strict_elbo and shs_corpus_batches) else 1.0)
        self._shs_entropy_mode = str(shs_entropy_mode)
        self._shs_moves_complete = bool(shs_moves_complete)
        self._shs_birth_style = shs_birth_style
        # opt-in: sample the true mixture (not the moment-matched Gaussian) in imagination
        self.regime.imag_sample_mixture = bool(shs_imag_sample_mixture)
        # blocker 7: default the imagination mode to the DIFFERENTIABLE moment path
        # (correct for imag_gradient='dynamics'); 'eval_sample' for true categorical
        # rollouts at eval; None falls back to the legacy imag_sample_mixture flag.
        self._shs_imag_mode = shs_imag_mode
        # blocker 8: accumulate PG/KL online (no O(BTK^2) xi) in recurrent training
        self.regime._shs_online_pairwise = bool(shs_online_pairwise)
        self._shs_merge_topm = shs_merge_topm
        # WHERE the persistent SHS globals come from.
        #   replay_ema              -> replay minibatch stats update the globals (default)
        #   completed_episode_stream-> replay computes the LOCAL loss only; the globals
        #                              are fed ONLY by stream_completed_episodes (absorb-
        #                              once), so replay revisits never enter the store
        #   offline_memoized        -> fixed-corpus memoized laps (set_batch_id)
        self._shs_global_source = str(shs_global_source)
        self._shs_action_dim = int(shs_action_dim)
        self._last_action = None   # (B,T,A) stash from observe
        # use the expected log-likelihood under q(z) for the E-step (Rao-Blackwellised)
        self._shs_analytic_estep = bool(shs_analytic_estep)
        self._shs_free = shs_free
        self._shs_update_every = shs_global_update_every
        self._shs_move_every = shs_move_every
        self._shs_move_warmup = shs_move_warmup
        self._shs_move_birth = shs_move_birth
        self._shs_move_split = bool(shs_move_split)
        self._shs_move_confirm_top = shs_move_confirm_top
        self._shs_delete_mode = shs_delete_mode
        self._shs_merge_select = shs_merge_select
        self._shs_merge_passes = shs_merge_passes
        self._last_move_log = {}               # outcome of the most recent sweep (logged)
        # ---- optional in-run curriculum: stage the SHS knobs in ONE training run ----
        # `shs_curriculum` is a list of phase dicts, each {"until": <step or -1>, plus any
        # of "move_every", "kappa", "move_threshold", "create_bonus"} that OVERRIDE the base
        # value for that phase. `until` is in WORLD-MODEL GRADIENT STEPS -- the count
        # self._shs_step increments once per kl_loss/training update, the SAME unit
        # shs_move_warmup uses, NOT env steps. Phases are matched in order: the first whose
        # `until` exceeds the current step (or is -1) is active. Empty/None -> as before.
        # This replaces running the model several times with different configs.
        self._curriculum = list(shs_curriculum) if shs_curriculum else []
        self._curr_phase = -1                 # last applied phase index (logged)
        self._move_threshold = 0.0            # live move-acceptance margin (annealable)
        self._move_create_bonus = 0.0         # live over-create bias for birth/split (annealable)
        _curric_moves = any(float(p.get("move_every", shs_move_every)) > 0
                            for p in self._curriculum)
        # Persistent held-out scoring set for the moves. Moves score the complete-data
        # ELBO over the LAST `shs_move_buffer` minibatches rather than the single current
        # one (Dreamer's replay is non-stationary), which is the "memoized" set the move
        # acceptance is summed over. Allocated when adaptive K is on at construction OR is
        # turned on by any curriculum phase (so the later phase starts with a full buffer).
        self._move_buffer = MoveBuffer(max_batches=shs_move_buffer,
                                       max_age=shs_move_max_age) \
            if (shs_move_every > 0 or _curric_moves) else None
        self._shs_step = 0
        self._last_is_first = None
        self._pending = None  # deferred global update (applied after the next backward)
        self._n_live_sweeps_skipped = 0  # scheduled live sweeps declined (under-filled/unsound ring)
        self._shs_min_live_batches = 4   # min recent minibatches before a live sweep may run
        self._next_batch_id = None  # stable identity of the NEXT minibatch (set_batch_id)

    # keep is_first for kl_loss
    def _aligned_action(self, ref):
        """Slice the stashed observe() action to ref's (B,T); None when unused."""
        if self._shs_action_dim == 0 or self._last_action is None:
            return None
        act = self._last_action
        if act.shape[1] < ref.shape[1]:
            # A too-short action is an ERROR when action_dim>0, not a
            # silent None that would make the E-step and loss use different likelihoods
            raise ValueError(
                f"stashed action has T={act.shape[1]} < required T={ref.shape[1]} "
                f"(shs_action_dim={self._shs_action_dim}); cannot align")
        return act[:, : ref.shape[1]]

    def observe(self, embed, action, is_first, state=None, sample=True):
        self._last_is_first = is_first  # (B, T)
        if self._shs_action_dim > 0:
            self._last_action = action.detach()
        # forward sample= (base RSSM.observe accepts it); the frozen-target
        # SHS encode calls observe(sample=False) -- the override previously dropped it,
        # raising TypeError on the streaming/consolidation paths.
        return super().observe(embed, action, is_first, state, sample=sample)

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        """carry a FILTERED regime belief across online steps.

        The base obs_step returns a posterior with NO `regime_resp`, so the next img_step
        resets to a uniform regime prior every step. Here we PREDICT (the prior's propagated
        belief from img_step) then UPDATE it with this step's dynamics evidence (a one-step
        analytic regime filter), and attach it to the posterior so beliefs persist online.
        Guarded: on any issue we fall back to the predicted belief (still not uniform) and
        never break inference."""
        post, prior = super().obs_step(prev_state, prev_action, embed, is_first, sample=sample)
        resp_pred = prior.get("regime_resp", None) if isinstance(prior, dict) else None
        if resp_pred is not None:
            try:
                R = self.regime
                z = (post["mean"] if "mean" in post else post["stoch"]).float()   # (B,L)
                deter = post["deter"].float()                                     # (B,H)
                zv = (post["std"].float() ** 2) if "std" in post else None
                prev_z = prev_state.get("stoch") if isinstance(prev_state, dict) else None
                prev_z = torch.zeros_like(z) if prev_z is None else prev_z.float()
                act = None
                _a = getattr(self, "_last_action", None)
                if self._shs_action_dim > 0 and _a is not None:
                    act = _a[:, -1].float() if _a.dim() == 3 else _a.float()
                g = R.build_g(prev_z, deter, act)                                 # (B,G)
                ev = R.regimes.expected_loglik(z, g, z_var=zv)                    # (B,K)
                resp_post = torch.softmax(resp_pred.float().clamp_min(1e-30).log() + ev, -1)
                if is_first is not None:
                    m = is_first.reshape(-1, 1).to(resp_post.dtype)              # reset -> predicted
                    resp_post = m * resp_pred.float() + (1.0 - m) * resp_post
                post["regime_resp"] = resp_post.to(post["deter"].dtype)
            except Exception:
                post["regime_resp"] = resp_pred
        return post, prior

    def _apply_curriculum(self):
        """Stage the SHS knobs in a SINGLE run, from self._shs_step (WM gradient steps).
        Sets the live move cadence, base stickiness kappa, and move-acceptance margin from
        the active phase. No-op when no curriculum is configured. Only mutates values that
        are SAFE to change mid-run -- move_every (read each step), hdp.kappa (read live in
        the sticky-Dirichlet update), and the move threshold. Structural choices such as
        `recurrent` are set once at construction and are deliberately NOT scheduled, since
        toggling them mid-run would rebuild the stickiness path."""
        if not self._curriculum:
            return
        step = self._shs_step
        idx = next((i for i, p in enumerate(self._curriculum)
                    if int(p.get("until", -1)) < 0 or step < int(p["until"])),
                   len(self._curriculum) - 1)
        phase = self._curriculum[idx]
        if "move_every" in phase:
            self._shs_move_every = int(phase["move_every"])
        if "move_threshold" in phase:
            self._move_threshold = float(phase["move_threshold"])
        if "create_bonus" in phase:
            self._move_create_bonus = float(phase["create_bonus"])
        # Switch recurrent stickiness ON/OFF by phase. rstick is always constructed, so this
        # is safe mid-run; it is simply not consulted while recurrent is False. This is what
        # lets the run STABILISE first (kappa-sticky, no recurrence) and only then turn on
        # input-dependent dwell once the regimes are differentiated.
        if "recurrent" in phase and getattr(self.regime, "rstick", None) is not None:
            self.regime.recurrent = bool(phase["recurrent"])
        # Base stickiness kappa governs persistence while recurrent is OFF. When recurrent is
        # ON the Bernoulli persistence carries the stickiness, so the base is forced
        # non-sticky (kappa=0) to avoid double-counting (mirrors base_kappa at construction).
        if getattr(self.regime, "hdp", None) is not None:
            if self.regime.recurrent:
                self.regime.hdp.kappa = 0.0
            elif "kappa" in phase:
                self.regime.hdp.kappa = float(phase["kappa"])
        if idx != self._curr_phase:
            self._curr_phase = idx
            print(f"[SHS] curriculum -> phase {idx} @ grad-step {step}: "
                  f"move_every={self._shs_move_every}, recurrent={self.regime.recurrent}, "
                  f"kappa={getattr(self.regime.hdp, 'kappa', None)}, "
                  f"move_threshold={self._move_threshold}, "
                  f"create_bonus={self._move_create_bonus}")

    def curriculum_state(self) -> dict:
        """Live curriculum knobs + last move-sweep outcome, for logging so the staging and
        the adaptive-K moves are both visible in metrics."""
        out = {
            "shs_curriculum_phase": float(self._curr_phase),
            "shs_move_every_live": float(self._shs_move_every),
            "shs_kappa_live": float(getattr(self.regime.hdp, "kappa", 0.0)),
            "shs_move_threshold_live": float(self._move_threshold),
            "shs_create_bonus_live": float(self._move_create_bonus),
            "shs_recurrent_live": float(bool(self.regime.recurrent)),
        }
        # transactional-guard + ledger-hygiene counters: nonzero values mean a
        # numerically invalid candidate update was REJECTED (previous state kept)
        # or stale-representation batches were auto-invalidated -- visible, not silent.
        out["shs_guard_rejects_hdp"] = float(getattr(self.regime.hdp, "n_guard_rejects", 0))
        out["shs_guard_rejects_pg"] = float(getattr(self.regime.rstick, "n_pg_guard_rejects", 0))
        _st = getattr(self.regime, "stat_store", None)
        out["shs_stale_invalidated"] = float(getattr(_st, "n_stale_invalidated", 0) if _st is not None else 0)
        out["shs_live_sweeps_skipped_stale"] = float(getattr(self, "_n_live_sweeps_skipped", 0))
        # last sweep: per-move accepted flag (1/0) and ELBO gain, so birth/split that get
        # immediately merged away are visible rather than silently discarded
        for m, (ok, gain) in (self._last_move_log or {}).items():
            out[f"shs_move_{m}_accepted"] = float(bool(ok))
            out[f"shs_move_{m}_gain"] = float(gain)
        return out

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        if self._shs_strict_elbo and (float(free) != 0.0 or float(dyn_scale) != 1.0
                                      or float(rep_scale) != 0.0):
            raise ValueError(
                f"shs_strict_elbo=True but the RUNTIME KL profile is (kl_free={free}, "
                f"dyn_scale={dyn_scale}, rep_scale={rep_scale}); strict requires "
                "(0, 1, 0). Set kl_free: 0.0, dyn_scale: 1.0, rep_scale: 0.0 (the "
                "`shs_strict` preset does this) -- the constructor cannot see these "
                "training-loop values, so they are validated here on every call.")
        # Apply any deferred global update from the previous step. Its autograd graph
        # has been freed by the optimizer step, so modifying the buffers in place now
        # is safe; doing it inside the same forward/backward would corrupt the graph.
        if self._pending is not None:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                _rv = self._pending[8] if len(self._pending) > 8 else None
                self.regime.update_globals(*self._pending[:7],
                                           batch_id=self._pending[7],
                                           action=(self._pending[9] if len(self._pending) > 9 else None),
                                           repr_version=_rv)
            self._pending = None

        # encoder posterior and carry for the whole segment
        q_mean = post["mean"].float()
        q_std = post["std"].float()
        stoch = post["stoch"].float()
        deter = post["deter"].float()
        is_first = self._last_is_first
        self._shs_step += 1

        # in-run curriculum: stage the move cadence / base stickiness / acceptance margin
        # from the active phase before anything reads them this step.
        self._apply_curriculum()

        # Adaptive-K move sweep on a slow schedule. The persistent buffer is fed EVERY step
        # whenever it exists (decoupled from the live move_every) so that a curriculum phase
        # that turns moves ON starts with a populated, RECENT buffer; staleness eviction
        # (max_age) drops the pre-convergence latents on its own. The sweep itself fires
        # only when moves are active in the current phase (move_every>0), past the warmup.
        if self._move_buffer is not None:
            with torch.no_grad():
                if self._shs_analytic_estep:
                    # store the posterior mean + variance so move scoring uses the same
                    # expected-log-likelihood E-step as training, not a single sample
                    self._move_buffer.add(
                        q_mean.detach(), deter.detach(),
                        None if is_first is None else is_first.detach(),
                        (q_std ** 2).detach(), step=self._shs_step,
                        batch_id=f"live{self._shs_step}",
                        repr_version=int(self.regime.repr_version),
                        action=(self._aligned_action(q_mean).detach()
                                if self._shs_action_dim > 0 else None))   # current-step action, not cleared _pending
                else:
                    self._move_buffer.add(
                        stoch.detach(), deter.detach(),
                        None if is_first is None else is_first.detach(),
                        step=self._shs_step,
                        batch_id=f"live{self._shs_step}",
                        repr_version=int(self.regime.repr_version),
                        action=(self._aligned_action(q_mean).detach()
                                if self._shs_action_dim > 0 else None))   # non-analytic branch
        _scheduled = (self._shs_move_every > 0
                      and self._shs_step >= self._shs_move_warmup
                      and self._shs_step % self._shs_move_every == 0)
        if _scheduled and self._live_sweep_allowed():
            with torch.no_grad():
                self._last_move_log = sweep_moves(
                    self.regime, buffer=self._move_buffer,
                    do_birth=self._shs_move_birth, do_split=self._shs_move_split,
                    threshold=self._move_threshold, create_bonus=self._move_create_bonus,
                    confirm_top=self._shs_move_confirm_top,
                    delete_mode=self._shs_delete_mode,
                    merge_select=self._shs_merge_select,
                    merge_passes=self._shs_merge_passes,
                    birth_style=self._shs_birth_style,
                    delete_topk=getattr(self, "_shs_delete_topk", 3),
                merge_topm=self._shs_merge_topm)
            accepted = {m: round(float(g), 3) for m, (ok, g) in self._last_move_log.items() if ok}
            if accepted:
                print(f"[SHS] move sweep @ grad-step {self._shs_step}: K={self.regime.K} "
                      f"accepted={accepted}")
        elif _scheduled:
            # A scheduled sweep we DECLINE: too few consistent minibatches to be a real
            # sample, or a memoized/streaming ledger is active (its arithmetic needs one 
            # encoder version, so moves there run on complete consolidation buffers, not 
            # the live ring). Skip with a visible counter; fall through to the loss -- 
            # never early-return (that dropped the gradient) and never purge the bounded ring.
            self._n_live_sweeps_skipped = getattr(self, "_n_live_sweeps_skipped", 0) + 1
            self._last_move_log = {}

        # regime E-step (no grad): responsibilities + counts. cache_estep=True caches the
        # per-step persistence quantities for the deferred recurrent-stickiness update (all
        # read-only callers leave cache_estep False so they never clobber it). With the
        # analytic E-step on, the evidence is the expected log-likelihood under q(z_t) -- the
        # posterior mean plus the -1/2 tr(Q^-1 diag(var)) correction -- instead of one encoder
        # sample, matching the analytic M-step (Rao-Blackwellised responsibilities).
        # Pin the regime E-step to float32: it caches the sufficient statistics that feed
        # the closed-form PG / HDP / regime updates, and the AMP grad-scaler does not police
        # forward activations. The .float() casts on q_mean/deter are not enough on their own
        # because autocast re-downcasts the internal matmuls; disabling autocast does pin it.
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            # Fully Bayesian VB E-step: always use q(z_t)=N(mean,std^2) and the
            # Rao-Blackwellised expected log likelihood.  This keeps the E-step
            # evidence exactly identical to the dynamics loss evidence.  The old
            # sample-based E-step is intentionally bypassed here because it would
            # optimize a different criterion from the fully Bayesian loss.
            gamma, counts, start_counts, _ = self.regime.regime_inference(
                q_mean.float(), deter.float(), is_first, cache_estep=True,
                z_var=(q_std ** 2).float(),
                action=self._aligned_action(q_mean))
        # expose the inferred regime belief so actor imagination starts from it
        # (img_step reads state["regime_resp"]) instead of a uniform prior
        post["regime_resp"] = gamma.detach()

        # Structured variational KL (carries grad to encoder via q and to P/GRU via g)
        if self._shs_strict_elbo and not (float(dyn_scale) == 1.0 and float(rep_scale) == 0.0):
            raise ValueError(
                "shs_strict_elbo=True requires dyn_scale=1.0 and rep_scale=0.0 "
                f"(got {dyn_scale}, {rep_scale}); scaled KL terms are not one ELBO.")
        loss, value, dyn, rep = self.regime.dynamics_kl(
            q_mean, q_std, stoch, deter, gamma, is_first,
            free=free if free is not None else self._shs_free,
            dyn_scale=dyn_scale, rep_scale=rep_scale,
            strict_elbo=self._shs_strict_elbo,
            global_scale=self._global_scale,
            action=self._aligned_action(q_mean),   # loss uses SAME action as E-step
        )

        # stash this batch for a deferred closed-form M-step (applied next call). Pass the
        # posterior MEAN and VARIANCE (not the encoder sample) so the regime M-step uses the
        # exact analytic moments E[z]=mu, E[z z^T]=mu mu^T + diag(var) -- a Rao-Blackwellised
        # update with the same fixed point but lower variance than one-sample Monte Carlo EM.
        # replay stash happens for replay_ema (EMA update) AND offline_memoized (so the
        # memoized store's stable-id guard still fires loudly if replay is misused);
        # ONLY completed_episode_stream skips it (episodes are fed via stream_episode,
        # so replay revisits must not write to the persistent store).
        if (self._shs_step % self._shs_update_every == 0
                and self._shs_global_source != "completed_episode_stream"):
            # blocker 1: in completed_episode_stream mode this stash is SKIPPED, so
            # replay minibatches never write to the persistent store (no double-count
            # against absorb-once episodes); the local loss above is still returned.
            self._pending = (q_mean.detach(), deter.detach(), gamma.detach(),
                             counts.detach(), start_counts.detach(),
                             None if is_first is None else is_first.detach(),
                             (q_std ** 2).detach(),
                             self._next_batch_id,
                             int(self.regime.repr_version),
                             None if self._last_action is None else self._aligned_action(q_mean).detach())
            self._next_batch_id = None            # consume-once

        return loss, value.detach(), dyn.detach(), rep.detach()

    @torch.no_grad()
    def annotate_regime_resp(self, post, is_first=None, action=None):
        """Attach the posterior regime belief q(s_t) to an OBSERVED trajectory.

        `observe` produces the continuous posterior but, unlike `kl_loss`, never runs the
        regime forward-backward, so a trajectory returned by `observe` alone carries no
        `regime_resp`. Open-loop tools (`video_pred`, the diagnostics hook) seed
        imagination from the LAST observed state; without this annotation that seed falls
        back to the uniform regime prior in `img_step`, so the rollout starts from the
        wrong regime. This runs the SAME no-grad E-step `kl_loss` uses -- a forward-backward
        over the encoder samples with episode resets at `is_first` -- and stores the
        responsibilities. It does NOT update the regime/HDP globals, does NOT run a move
        sweep, and does NOT touch the dynamics KL: it is pure annotation and is safe to
        call in eval / under no_grad.

        Args:
            post: trajectory dict from `observe`, holding `stoch` (B,T,L) and `deter`
                (B,T,H) (plus the usual continuous-posterior entries).
            is_first: (B,T) episode-start mask so the chain resets at episode boundaries
                exactly as in training; pass `data["is_first"]` for the same window.
        Returns:
            The same `post`, with `post["regime_resp"]` set to gamma, shape (B,T,K),
            each timestep summing to 1 over the K regimes.
        """
        deter = post["deter"].float()
        # match the training E-step: fully Bayesian expected log-likelihood under
        # q(z) whenever posterior mean/std are available.
        if "mean" in post and "std" in post:
            z = post["mean"].float()
            z_var = post["std"].float() ** 2
        else:
            z = post["stoch"].float()
            z_var = None
        gamma, _, _, _ = self.regime.regime_inference(
            z, deter, is_first, cache_estep=False, z_var=z_var,
            action=(action if self._shs_action_dim > 0 else None))   # blocker 5
        post["regime_resp"] = gamma
        return post

    @torch.no_grad()
    def prior_entropy_bounds(self, state):
        """Deterministic (lower, upper) mixture-entropy bounds; no Monte Carlo."""
        lo = self.prior_entropy(state, mode="bounds")
        return lo, self._last_entropy_bounds[1]

    def set_batch_id(self, batch_id):
        """Declare the stable identity of the NEXT training minibatch.

        Consumed once by the deferred global M-step. Required by
        `shs_online_mode: memoized` (Hughes replace semantics need a fixed corpus
        partition -- in Dreamer that is the consolidation path or offline fitting;
        the live replay loop has no stable identities, so memoized mode without ids
        fails loudly instead of silently appending every visit as a new batch).
        `streaming`/`full_batch` auto-id each call (absorb-once / per-pass semantics)."""
        self._next_batch_id = batch_id

    def get_extra_state(self):
        """Serialize the deferred global update so a checkpoint taken between a
        backward and the next kl_loss does not silently drop the newest statistics."""
        def cpu(o):
            if o is None:
                return None
            if torch.is_tensor(o):
                return o.detach().cpu()
            if isinstance(o, (tuple, list)):
                return tuple(cpu(v) for v in o)
            return o
        return dict(pending=cpu(self._pending),
                    next_batch_id=self._next_batch_id,
                    shs_step=int(self._shs_step))

    def set_extra_state(self, state):
        dev = self.regime.ema_trans_counts.device
        def mv(o):
            if o is None:
                return None
            if torch.is_tensor(o):
                return o.to(dev)
            if isinstance(o, (tuple, list)):
                return tuple(mv(v) for v in o)
            return o
        p = state.get("pending")
        self._pending = None if p is None else tuple(mv(v) for v in p)
        self._next_batch_id = state.get("next_batch_id")
        self._shs_step = int(state.get("shs_step", 0))

    def begin_full_batch_pass(self):
        """Explicit epoch boundary for shs_online_mode='full_batch' (offline use)."""
        self.regime.begin_full_batch_pass()

    def finalize_full_batch_pass(self):
        """Proxy: force the single deferred global step of an open full-batch
        pass from the accumulated totals (ragged/undeclared corpora)."""
        return self.regime.finalize_full_batch_pass()

    def _live_sweep_allowed(self) -> bool:
        """Whether a scheduled LIVE move sweep may run on the current ring.

        The live (non-complete) ring is a bounded window of recent regime-head inputs
        used as a PROPOSAL buffer scored by EXACT acceptance; mixing recent encoder
        versions in it is sound (acceptance is exact on its contents, and it never
        claims to be a whole corpus). Gates: (a) it must hold enough minibatches to be a
        real sample rather than one step's data ; (b) the store must NOT be memoized/streaming, 
        whose replace / absorb-once arithmetic requires a single encoder version (those modes run moves
        on complete consolidation buffers). A complete buffer bypasses both via its own
        whole-corpus certificate.
        """
        buf = self._move_buffer
        if buf is None:
            return False
        if getattr(buf, "complete", False):
            return buf.is_complete()
        if self._shs_moves_complete:
            return False
        enough = len(buf.batches) >= int(getattr(self, "_shs_min_live_batches", 4))
        store = getattr(self.regime, "stat_store", None)
        ema_mode = (store is None) or (store.mode == "legacy_ema")
        return enough and ema_mode

    def bump_repr_version(self):
        """Advance the representation version after an encoder/GRU update.

        The live ring is deliberately NOT purged here: it is a bounded recent-data
        proposal buffer scored by exact acceptance, and dropping it every optimizer step
        left a single minibatch to score.
        The strict single-version requirement applies only to COMPLETE consolidation
        buffers and is enforced in _validate_buffer; a complete buffer that spans a bump
        raises there, directing the caller to encode the corpus once under a frozen
        representation (fit_offline_corpus does exactly this).
        """
        self.regime.bump_repr_version()

    def prior_entropy(self, state, n_samples: int = 64, mode: str | None = None):
        """True (Monte Carlo) entropy of the SHS MIXTURE prior over z_t, per (B,T) step.

        DreamerV3's logged `prior_ent = get_dist(prior).entropy()` reports the entropy of
        the moment-matched single Gaussian, which is an UPPER bound on the mixture entropy,
        not the mixture entropy itself (this only affects the logged scalar -- the training
        KL now uses the structured regime objective). This recomputes the mixture from the
        state and estimates its entropy directly; logging both gives the regime
        multi-modality gap H_gauss - H_mix for free. Uses the exact recurrent sticky mixture
        weights when recurrent stickiness is on (matching the training prior), the closed-form
        HDP weights otherwise. Diagnostics only.
        """
        from .mixture_prior import (mixture_weights, mixture_entropy_monte_carlo,
                                    mixture_entropy_monte_carlo_lowrank)
        R = self.regime
        stoch = state["stoch"].float()
        deter = state["deter"].float()
        isf = state.get("is_first", None)
        prev = R._prev_stoch(stoch, isf)
        # use the SAME action-conditioned regressor as training when
        # action_dim>0 (guarded -- diagnostics must never crash on a stale/short action).
        _act = None
        if self._shs_action_dim > 0:
            try:
                _act = R._shift_action(self._aligned_action(stoch), isf)
            except Exception:
                _act = None
        g = R.build_g(prev, deter, _act)
        if "std" in state:
            g_var = R._g_var_from_z_var(state["std"].float().pow(2), g, is_first=isf)
        else:
            g_var = None
        comp_mean, comp_var = R.regimes.predictive_moments(g, g_var=g_var)  # (B,T,K,L)
        rr = state.get("regime_resp", None)
        # Adaptive-K guard: a move may have changed K after this state's regime_resp was
        # produced, so its last dim can disagree with the current K. Fall back to a uniform
        # prior at the current K rather than mismatch the transition matrix.
        if rr is None or rr.shape[-1] != R.K:
            rr = stoch.new_full(stoch.shape[:2] + (R.K,), 1.0 / R.K)
        rr_prev = R._shift_resp(rr.float(), isf)
        if R.recurrent:
            # exact state-specific sticky weights w = gamma_prev @
            # (rho_i I + (1-rho_i) Pi), matching the structured training prior.
            phi = R.build_stick_phi(deter)
            Pi = torch.softmax(R.hdp.expected_log_trans().to(g.dtype), dim=-1)
            sig = R.rstick.sigma(phi)
            eye = torch.eye(R.K, dtype=g.dtype, device=g.device)
            M = sig[..., :, None] * eye + (1.0 - sig[..., :, None]) * Pi
            w = torch.einsum("...k,...kl->...l", rr_prev, M).clamp_min(1e-8)
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-8)
        else:
            w = mixture_weights(rr_prev, R._Epi().to(g.dtype))
        mode = self._shs_entropy_mode if mode is None else mode
        if getattr(self, "_shs_strict_elbo", False) and mode != "bounds":
            # strict ELBO is provably Monte-Carlo-free: force the deterministic
            # mixture-entropy bounds, never a sampling estimator 
            mode = "bounds"
        if mode == "bounds":
            # deterministic mixture-entropy bounds (Hershey-Olsen style): no sampling.
            # Components use the diagonal predictive marginals, so with q_rank>0 these
            # bound the diagonalised mixture's entropy (labelled diagnostic).
            from .mixture_prior import mixture_entropy_bounds
            lo, hi = mixture_entropy_bounds(comp_mean, comp_var, w)
            self._last_entropy_bounds = (lo.detach(), hi.detach())
            return lo.clamp_min(1e-8)
        if R.regimes.q_rank > 0:
            _, comp_d, comp_U = R.regimes.predictive_cov_moments(g, g_var=g_var)
            return mixture_entropy_monte_carlo_lowrank(comp_mean, comp_d, comp_U, w,
                                                       n_samples=n_samples)
        return mixture_entropy_monte_carlo(comp_mean, comp_var, w, n_samples=n_samples)

    def _deter_step(self, prev_state, prev_action):
        """Just the deterministic GRU update from networks.RSSM.img_step (no unused
        Gaussian prior head / sampling)."""
        x = torch.cat([prev_state["stoch"], prev_action], -1)
        x = self._img_in_layers(x)
        deter = prev_state["deter"]
        for _ in range(self._rec_depth):
            x, deter = self._cell(x, [deter]); deter = deter[0]
        return deter

    def img_step(self, prev_state, prev_action, sample=True):
        # only the GRU carry is needed from the base RSSM; the SHS mixture replaces the
        # Gaussian prior entirely, so we skip _img_out_layers / suff_stats / sampling
        deter = self._deter_step(prev_state, prev_action)
        B = deter.shape[0]
        resp_prev = prev_state.get("regime_resp", None)
        if resp_prev is None or resp_prev.shape[-1] != self.regime.K:
            # a belief carried across a structural move has the
            # OLD K; reset it to the current-K uniform prior so actor imagination cannot
            # crash on a dimension mismatch (the weaker fix the reviewer suggested; a
            # fixed Kmax + active mask would preserve history but is a larger change)
            resp_prev = deter.new_full((B, self.regime.K), 1.0 / self.regime.K)
        prev_var = None
        if "std" in prev_state:
            prev_var = prev_state["std"].float().pow(2)
        z, resp, mean, std = self.regime.imagine_prior(
            prev_state["stoch"].float(), deter.float(), resp_prev.float(), sample=sample,
            prev_var=prev_var,
            action=(prev_action.float() if self._shs_action_dim > 0 else None),
            mode=getattr(self, "_shs_imag_mode", None),
        )
        dt = deter.dtype
        return {"stoch": z.to(dt), "deter": deter, "mean": mean.to(dt),
                "std": std.to(dt), "regime_resp": resp.to(dt)}

    def initial(self, batch_size):
        state = super().initial(batch_size)
        # the SHS initial latent is the learned z0 (the base Gaussian head is unused);
        # set stoch/mean/std consistently and seed a uniform regime belief
        z0 = self.regime.z0.to(state["stoch"].dtype)
        state["stoch"] = z0.view(1, -1).expand(batch_size, -1).contiguous()
        if "mean" in state:
            state["mean"] = state["stoch"].clone()
            # z0 is a learned deterministic initial latent in the SHS prior.  Keep its
            # uncertainty zero so the first imagined regressor is not spuriously noisy.
            state["std"] = torch.zeros_like(state["stoch"])
        state["regime_resp"] = torch.full(
            (batch_size, self.regime.K), 1.0 / self.regime.K, device=self._device
        )
        return state


# config wiring
# shs_* keys legitimately consumed OUTSIDE the SHSRSSM constructor (WorldModel loop,
# diagnostics, consolidation cadence). Everything else must be a constructor kwarg;
# unknown shs_* keys raise, so "option exists in YAML but is silently dead at the
# construction boundary" is now a hard error.
SHS_NON_CTOR_KEYS = {
    "shs_diag_log", "shs_diag_figures",
    "shs_stream_episodes",   # absorb-once completed-episode streaming (loop-level)
    "shs_repr_epoch_steps",  # frozen-target representation-epoch refresh cadence
    "shs_consolidate_every_episodes", "shs_consolidate_warmup",
    "shs_consolidate_batches", "shs_consolidate_sweeps", "shs_consolidate_ep_len",
}


def shs_kwargs_from_config(config):
    """Map EVERY shs_* attribute of `config` to an SHSRSSM constructor kwarg.

    Consume-all guard: any shs_* attribute that is neither an SHSRSSM kwarg nor in
    SHS_NON_CTOR_KEYS raises ValueError naming the dead keys, instead of being
    silently dropped at the models.py boundary."""
    import inspect
    accepted = {n for n in inspect.signature(SHSRSSM.__init__).parameters
                if n.startswith("shs_")}
    try:
        items = {k: v for k, v in vars(config).items() if k.startswith("shs_")}
    except TypeError:
        items = {k: getattr(config, k) for k in dir(config) if k.startswith("shs_")}
    unknown = sorted(k for k in items if k not in accepted and k not in SHS_NON_CTOR_KEYS)
    if unknown:
        raise ValueError(
            "dead config wiring: these shs_* keys are not SHSRSSM constructor kwargs "
            f"and not registered loop-level keys: {unknown}. Either add them to "
            "SHSRSSM.__init__ or to SHS_NON_CTOR_KEYS in shs_rssm/shs_rssm.py.")
    kw = {k: v for k, v in items.items() if k in accepted}
    # shs_action_dim = -1 means AUTO -> config.num_actions, so a
    # preset can enable the per-regime action term without hard-coding the action count.
    # Default 0 (off) is unchanged; -1 is an explicit opt-in.
    if int(kw.get("shs_action_dim", 0)) == -1:
        kw["shs_action_dim"] = int(getattr(config, "num_actions", 0))
    return kw

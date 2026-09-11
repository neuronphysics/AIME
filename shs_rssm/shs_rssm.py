from __future__ import annotations

import numpy as np
import torch
import warnings
import networks
from .regime_head import RegimeHead
from .moves import sweep_moves, MoveBuffer
from .checks import check_occupancy_vs_beta
from .mixture_prior import mixture_weights, mixture_entropy_monte_carlo, mixture_entropy_monte_carlo_lowrank

class SHSRSSM(networks.RSSM):
    def __init__(self, *args,
                 shs_K: int = 16,
                 shs_proj_dim: int | None = 64,
                 shs_gamma: float = 5.0,
                 shs_alpha: float = 1.0,
                 shs_kappa: float = 50.0,
                 shs_start_alpha: float = 1.0,
                 shs_a0: float = 3.0, shs_b0: float = 2.0, shs_v0_scale: float = 1.0,
                 shs_calibrate_b0: bool = True, shs_calibrate_sF: float = 1.0,
                 shs_learn_b0: bool = False, shs_b0_strength: float = 2.0,
                 shs_ard: bool = False, shs_ema_tau: float = 0.02,
                 shs_hdp_iters: int = 2, shs_global_update_every: int = 1,
                 shs_free: float = 1.0,
                 shs_recurrent: bool = False, shs_prior_persist: float = 0.9,
                 shs_pg_iters: int = 4, shs_rstick_dim: int | None = 8,
                 shs_rstick_stopgrad: bool = True,
                 shs_rstick_weight_var: float = 1.0,
                 shs_rstick_bias_var: float = 4.0,
                 shs_rstick_use_action: bool = False,
                 shs_rstick_hier: bool = True,
                 shs_rstick_hier_a0: float = 2.0,
                 shs_rstick_hier_b0: float | None = None,
                 shs_rstick_hier_used_only: bool = False,
                 shs_rstick_hier_min_evidence: float = 1.0,
                 shs_live_moves: bool = True,
                 shs_ckpt_buffer: int = 8,
                 shs_posterior_samples: int = 4,
                 shs_move_every: int = 0, shs_move_warmup: int = 2000,
                 shs_teacher: bool = True, shs_teacher_tau: float = 0.005,
                 shs_teacher_drift_tol: float = 0.05,
                 shs_grad_per_env: float | None = None,
                 shs_move_warmup_env: float | None = None,
                 shs_ema_memory_env: float | None = None,
                 shs_move_birth: bool = True, shs_move_buffer: int = 8,
                 shs_move_max_age: int = 0, shs_move_split: bool = True,
                 shs_move_confirm_top: int | None = None,
                 shs_delete_mode: str = "hughes", shs_merge_select: str = "hughes",
                 shs_merge_passes: int = 3,
                 shs_q_rank: int = 0, shs_imag_sample_mixture: bool = False,
                 shs_imag_mode: str = "actor_moment",
                 shs_online_pairwise: bool = False,
                 shs_analytic_estep: bool = True,
                 shs_moment_consistent: bool = True,
                 shs_gate_input: str = "carry",
                 shs_gate_inner_iters: int = 1,
                 shs_chunk_boundary_mask: bool = True,
                 shs_shared_carry: bool = True,
                 shs_curriculum: list | None = None,
                 shs_online_mode: str = "ema",
                 shs_expected_batches: int | None = None,
                 shs_strict_stream: bool = False,
                 shs_stream_local_iters: int = 1,
                 shs_stream_discount: float = 1.0,
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
        # ---- batch-invariant schedule units --------------------------------
        # Keys ending in *_env are ENVIRONMENT (policy) steps, converted once
        # to grad-steps via grad-per-env = train_ratio/(batch_size*batch_length)
        # (auto-injected by shs_kwargs_from_config).  Plain grad-step keys are
        # unchanged for backward compatibility.
        self._grad_per_env = (float(shs_grad_per_env)
                              if shs_grad_per_env else None)

        def _env2gs(v):
            if self._grad_per_env is None:
                raise ValueError(
                    "*_env schedule keys require shs_grad_per_env (auto-set "
                    "by shs_kwargs_from_config)")
            return max(1, int(round(float(v) * self._grad_per_env)))
        self._env2gs = _env2gs

        if shs_ema_memory_env is not None:
            if self._grad_per_env is None:
                raise ValueError("shs_ema_memory_env requires shs_grad_per_env")
            _tau = 1.0 / max(1.0, float(shs_ema_memory_env) * self._grad_per_env)
            shs_ema_tau = float(min(0.5, max(1e-5, _tau)))
            print(f"[SHS] ema tau derived from memory_env={shs_ema_memory_env} "
                  f"env-steps: tau={shs_ema_tau:.5f}")
        if shs_move_warmup_env is not None:
            shs_move_warmup = _env2gs(shs_move_warmup_env)
        super().__init__(*args, **kwargs)
        assert not self._discrete, "SHSRSSM supports continuous latents only"
        self.regime = RegimeHead(
            stoch=self._stoch, deter=self._deter, K=shs_K, proj_dim=shs_proj_dim,
            a0=shs_a0, b0=shs_b0, learn_b0=shs_learn_b0, b0_strength=shs_b0_strength, v0_scale=shs_v0_scale, ard=shs_ard,
            gamma=shs_gamma, alpha=shs_alpha, kappa=shs_kappa,
            start_alpha=shs_start_alpha, ema_tau=shs_ema_tau, hdp_iters=shs_hdp_iters,
            recurrent=shs_recurrent, prior_persist=shs_prior_persist, pg_iters=shs_pg_iters,
            rstick_dim=shs_rstick_dim, rstick_stopgrad=shs_rstick_stopgrad,
            rstick_weight_var=shs_rstick_weight_var,
            rstick_bias_var=shs_rstick_bias_var,
            rstick_use_action=shs_rstick_use_action,
            rstick_hier=shs_rstick_hier, rstick_hier_a0=shs_rstick_hier_a0,
            rstick_hier_b0=shs_rstick_hier_b0,
            rstick_hier_used_only=shs_rstick_hier_used_only,
            rstick_hier_min_evidence=shs_rstick_hier_min_evidence,
            q_rank=shs_q_rank, shared_carry=shs_shared_carry,
            gate_input=shs_gate_input, gate_inner_iters=shs_gate_inner_iters,
            action_dim=shs_action_dim,
            online_mode=shs_online_mode, expected_batches=shs_expected_batches,
            strict_stream=shs_strict_stream,
            stream_local_iters=shs_stream_local_iters,
            stream_discount=shs_stream_discount,
            hdp_every=shs_hdp_every, pg_every=shs_pg_every,
            device=self._device,
        )
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
        self.regime.imag_sample_mixture = bool(shs_imag_sample_mixture)
        self._shs_imag_mode = shs_imag_mode
        self.regime._shs_online_pairwise = bool(shs_online_pairwise)
        self._shs_merge_topm = shs_merge_topm
        self._shs_global_source = str(shs_global_source)
        self._shs_action_dim = int(shs_action_dim)
        self._last_action = None
        self._shs_analytic_estep = bool(shs_analytic_estep)
        self._shs_moment_consistent = bool(shs_moment_consistent)
        self._shs_calibrate_b0 = bool(shs_calibrate_b0)
        self._shs_calibrate_sF = float(shs_calibrate_sF)
        self._shs_b0_calibrated = False
        self._shs_kappa_cfg = float(shs_kappa)
        self.regime.chunk_boundary_mask = bool(shs_chunk_boundary_mask)
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
        self._last_move_log = {}
        self._curriculum = list(shs_curriculum) if shs_curriculum else []
        self._curr_phase = -1
        self._shs_teacher_cfg = dict(enabled=bool(shs_teacher),
                                     tau=float(shs_teacher_tau),
                                     drift_tol=float(shs_teacher_drift_tol))
        self._teacher = None
        for _p in self._curriculum:
            if "until_env" in _p:
                _p["until"] = (-1 if float(_p["until_env"]) < 0
                               else self._env2gs(_p["until_env"]))
            if "move_every_env" in _p:
                _p["move_every"] = (0 if float(_p["move_every_env"]) <= 0
                                    else self._env2gs(_p["move_every_env"]))
        if self._grad_per_env is not None and self._curriculum:
            print("[SHS] schedule units: grad/env=%.5f  curriculum(grad-steps)=%s"
                  % (self._grad_per_env,
                     [int(q.get("until", -1)) for q in self._curriculum]))
        self._move_threshold = 0.0
        self._move_create_bonus = 0.0
        _curric_moves = any(float(p.get("move_every", shs_move_every)) > 0
                            for p in self._curriculum)
        self._move_buffer = MoveBuffer(max_batches=shs_move_buffer,
                                       max_age=shs_move_max_age) \
            if (shs_move_every > 0 or _curric_moves) else None
        self._shs_step = 0
        self._last_is_first = None
        self._pending = None
        self._n_live_sweeps_skipped = 0
        self._n_live_sweeps_run = 0
        self._shs_min_live_batches = 4
        # False disables scheduled live sweeps entirely, so K stays at shs_K.
        self._shs_live_moves = bool(shs_live_moves)
        # How many of the most recent move-buffer batches travel in the
        # checkpoint, so a resumed run can sweep without waiting for a refill;
        # 0 disables.
        self._shs_ckpt_buffer = int(shs_ckpt_buffer)
        # Structural bookkeeping for the single acting stream.
        self._shs_struct_gen = 0
        self._shs_acting_gen = 0
        self._shs_belief_maps = {}
        self._shs_buffer_ckpt_version = None
        self.register_load_state_dict_post_hook(
            lambda mod, keys: mod._finish_buffer_restore())
        # Extra sequential draws of the recognition posterior, used only for the
        # global-statistics update (regime M-step, transition counts, gate PG
        # statistics).  Each draw re-runs the GRU and posterior head on the
        # cached embeddings and the S draws are averaged: the same estimator with
        # 1/S of the path-sample variance.  The loss, the actor features and the
        # move buffer use the single draw the world model trains on.
        self._shs_posterior_samples = max(1, int(shs_posterior_samples))
        self._last_embed = None
        self._next_batch_id = None

    def _aligned_action(self, ref):
        if self._shs_action_dim == 0 or self._last_action is None:
            return None
        act = self._last_action
        if act.shape[1] < ref.shape[1]:
            raise ValueError(
                f"stashed action has T={act.shape[1]} < required T={ref.shape[1]} "
                f"(shs_action_dim={self._shs_action_dim}); cannot align")
        return act[:, : ref.shape[1]]

    def observe(self, embed, action, is_first, state=None, sample=True,
                episode_first=None):
        self._last_is_first = is_first
        # "no predecessor in this crop" (crop boundary) vs "a genuine episode
        # started here".  Falls back to is_first so old call sites still work.
        self._last_episode_first = (episode_first if episode_first is not None
                                    else is_first)
        if self._shs_action_dim > 0:
            self._last_action = action.detach()
        self._last_embed = embed.detach()
        return super().observe(embed, action, is_first, state, sample=sample)

    def _zero_action_like(self, ref, prev_action):
        """A correctly shaped zero action for the first step of a stream, so
        the dynamics and the gate are conditioned on the same action."""
        if prev_action is not None or self._shs_action_dim <= 0:
            return prev_action
        return torch.zeros(ref.shape[0], int(self._shs_action_dim),
                           device=ref.device, dtype=ref.dtype)

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        prev_action = self._zero_action_like(embed, prev_action)
        # A structural move may have happened since this state was produced;
        # reconcile BEFORE super().obs_step, whose reset loop blends rows of
        # prev_state with rows of initial() and needs matching widths.
        prev_state = self._sync_state_gen(prev_state)
        post, prior = super().obs_step(prev_state, prev_action, embed, is_first, sample=sample)
        resp_pred = prior.get("regime_resp", None) if isinstance(prior, dict) else None
        if resp_pred is not None:
            try:
                R = self.regime
                z = (post["mean"] if "mean" in post else post["stoch"]).float()
                deter = post["deter"].float()
                zv = (post["std"].float() ** 2) if "std" in post else None
                prev_z = prev_state.get("stoch") if isinstance(prev_state, dict) else None
                prev_z = torch.zeros_like(z) if prev_z is None else prev_z.float()
                act = None
                if self._shs_action_dim > 0:
                    act = (prev_action.reshape(z.shape[0], -1).float()
                           if prev_action is not None
                           else torch.zeros(z.shape[0], int(self._shs_action_dim),
                                            device=z.device, dtype=z.dtype))
                if is_first is not None:
                    isf_col = is_first.reshape(-1, 1)
                    if bool((isf_col > 0.5).any()):
                        z0 = R.z0.to(prev_z.dtype).view(1, -1).expand_as(prev_z)
                        prev_z = torch.where(isf_col > 0.5, z0, prev_z)
                        if act is not None:
                            act = act * (1.0 - isf_col.to(act.dtype))
                        li = R.hdp.expected_log_init().float()[: R.K]
                        am = getattr(R, "active_mask", None)
                        if am is not None:
                            am = am.to(device=li.device, dtype=li.dtype)[: R.K]
                            li = torch.where(am > 0.5, li,
                                             torch.full_like(li, -1e30))
                        start = torch.softmax(li, -1).view(1, -1).to(resp_pred.dtype)
                        resp_pred = torch.where(isf_col > 0.5,
                                                start.expand_as(resp_pred), resp_pred)
                        # The returned PRIOR at an episode start must come from
                        # the learned initial regime law, not from uniform
                        # weights advanced through the transition matrix.
                        if isinstance(prior, dict) and "mean" in prior:
                            _z0, _r0, _m0, _s0 = R.imagine_prior(
                                prev_z, deter, resp_pred, sample=sample, action=act,
                                weights=start.expand_as(resp_pred))
                            _c = isf_col.to(prior["mean"].dtype)
                            _r0 = start.expand_as(resp_pred)     # predictive law, not a draw
                            for _k, _v in (("mean", _m0), ("std", _s0), ("stoch", _z0),
                                           ("regime_resp", _r0)):
                                if _k in prior:
                                    prior[_k] = torch.where(
                                        _c > 0.5, _v.to(prior[_k].dtype), prior[_k])
                g = R.build_g(prev_z, deter, act)
                ev = R.regimes.expected_loglik(z, g, z_var=zv)
                resp_post = torch.softmax(resp_pred.float().clamp_min(1e-30).log() + ev, -1)
                post["regime_resp"] = resp_post.to(post["deter"].dtype)
            except Exception:
                import traceback
                import warnings
                self._filter_guard_hits = int(getattr(self, "_filter_guard_hits", 0)) + 1
                if not getattr(self, "_filter_guard_warned", False):
                    self._filter_guard_warned = True
                    warnings.warn(
                        "SHS online regime filter failed; falling back to the predicted "
                        "belief (counted in _filter_guard_hits). First traceback:\n"
                        + traceback.format_exc())
                post["regime_resp"] = resp_pred
        return post, prior

    def _reconcile_resp(self, rr, gen):
        """Bring a belief formed at structural generation `gen` up to date.

        Composes every map recorded since `gen`.  If the chain is unusable (a
        map was dropped, or the width no longer matches) the belief is rebuilt
        from the learned initial regime law rather than silently reinterpreted
        under new identities."""
        if rr is None:
            return None
        cur = int(getattr(self, "_shs_struct_gen", 0))
        K = int(self.regime.K)
        if int(gen) >= cur:
            if int(rr.shape[-1]) == K:
                return rr
            gen = -1                       # unknown provenance: rebuild
        hist = getattr(self, "_shs_belief_maps", None) or {}
        M = None
        g = int(gen)
        while g < cur:
            m = hist.get(g)
            if m is None:
                M = None
                break
            m = m.to(device=rr.device, dtype=rr.dtype)
            M = m if M is None else m @ M
            g += 1
        if M is not None and int(M.shape[1]) == int(rr.shape[-1]) and int(M.shape[0]) == K:
            out = (rr.reshape(-1, rr.shape[-1]) @ M.t()).reshape(*rr.shape[:-1], K)
            out = out.clamp_min(1e-8)
            return out / out.sum(-1, keepdim=True)
        law = self._initial_regime_law(rr.dtype, rr.device)
        return law.view(*([1] * (rr.dim() - 1)), K).expand(*rr.shape[:-1], K).contiguous()

    def _initial_regime_law(self, dtype, device):
        li = self.regime.hdp.expected_log_init().to(dtype=torch.float32, device=device)
        return torch.softmax(li[: self.regime.K], -1).to(dtype)

    def sync_acting_state(self, state):
        """Reconcile the ACTING state that the agent carries across training
        updates.  There is exactly one acting stream, so its structural
        generation is tracked here rather than inside the state dict (adding a
        key would change the state contract shared with imagination, replay and
        checkpoints).  Safe to call every step: a no-op when no move has
        happened, and it also catches a move that preserved K."""
        if not isinstance(state, dict) or "regime_resp" not in state:
            return state
        cur = int(getattr(self, "_shs_struct_gen", 0))
        gen = int(self._shs_acting_gen)
        if gen == cur and int(state["regime_resp"].shape[-1]) == int(self.regime.K):
            self._shs_acting_gen = cur          # register: this stream is current
            return state
        state = dict(state)
        state["regime_resp"] = self._reconcile_resp(state["regime_resp"], gen)
        self._shs_acting_gen = cur
        return state

    def _sync_state_gen(self, state, batch_size=None):
        """Width-level safety net for any state entering obs_step / img_step:
        a belief whose width no longer matches K is mapped forward, or rebuilt
        from the learned initial law if the map chain is unusable.  This is what
        keeps a partial environment reset from blending mismatched widths."""
        if not isinstance(state, dict) or "regime_resp" not in state:
            return state
        rr = state["regime_resp"]
        if int(rr.shape[-1]) == int(self.regime.K):
            return state
        state = dict(state)
        state["regime_resp"] = self._reconcile_resp(rr, int(getattr(self, "_shs_acting_gen", 0)))
        return state

    def _map_regime_resp(self, rr):
        R = self.regime
        bm = getattr(R, "_belief_map", None)
        if rr is None or bm is None:
            return None
        if int(bm.shape[0]) != int(R.K) or int(bm.shape[1]) != int(rr.shape[-1]):
            return None
        out = rr.float() @ bm.t().to(device=rr.device, dtype=torch.float32)
        out = out.clamp_min(1e-8)
        return out / out.sum(-1, keepdim=True)

    def _apply_curriculum(self):
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
        # Per-phase birth/split flags gate which proposal types run in this
        # phase; unspecified phases keep the constructor defaults.
        if "birth" in phase:
            self._shs_move_birth = bool(phase["birth"])
        if "split" in phase:
            self._shs_move_split = bool(phase["split"])
        if "recurrent" in phase and getattr(self.regime, "rstick", None) is not None:
            self.regime.recurrent = bool(phase["recurrent"])
        if getattr(self.regime, "hdp", None) is not None:
            if self.regime.recurrent:
                # the gate supplies the self-transition bias, so the Dirichlet
                # sticky mass must be off
                self.regime.hdp.kappa = 0.0
            else:
                # a phase with the gate off uses the configured sticky mass
                self.regime.hdp.kappa = float(phase.get("kappa", self._shs_kappa_cfg))
        if idx != self._curr_phase:
            self._curr_phase = idx
            try:
                _tc = getattr(self.regime, "ema_trans_counts", None)
                _sc = getattr(self.regime, "ema_start_counts", None)
                if _tc is not None and _sc is not None and float(_tc.sum()) > 0:
                    self.regime.hdp.update(_tc.double(), _sc.double(),
                                           n_global_iters=1)
            except Exception as _e:
                print(f"[SHS] curriculum theta refresh skipped: {_e}")
            print(f"[SHS] curriculum -> phase {idx} @ grad-step {step}: "
                  f"move_every={self._shs_move_every}, recurrent={self.regime.recurrent}, "
                  f"kappa={getattr(self.regime.hdp, 'kappa', None)}, "
                  f"move_threshold={self._move_threshold}, "
                  f"create_bonus={self._move_create_bonus}, "
                  f"birth={self._shs_move_birth}, split={self._shs_move_split}")

    def curriculum_state(self) -> dict:
        out = {
            "shs_curriculum_phase": float(self._curr_phase),
            "shs_move_every_live": float(self._shs_move_every),
            "shs_kappa_live": float(getattr(self.regime.hdp, "kappa", 0.0)),
            "shs_move_threshold_live": float(self._move_threshold),
            "shs_create_bonus_live": float(self._move_create_bonus),
            "shs_move_birth_live": float(bool(self._shs_move_birth)),
            "shs_move_split_live": float(bool(self._shs_move_split)),
            "shs_recurrent_live": float(bool(self.regime.recurrent)),
        }
        out["shs_guard_rejects_hdp"] = float(getattr(self.regime.hdp, "n_guard_rejects", 0))
        out["shs_guard_rejects_pg"] = float(getattr(self.regime.rstick, "n_pg_guard_rejects", 0))
        _st = getattr(self.regime, "stat_store", None)
        out["shs_stale_invalidated"] = float(getattr(_st, "n_stale_invalidated", 0) if _st is not None else 0)
        out["shs_live_sweeps_skipped_stale"] = float(getattr(self, "_n_live_sweeps_skipped", 0))
        out["shs_live_sweeps_run"] = float(getattr(self, "_n_live_sweeps_run", 0))
        out["shs_repr_version"] = float(int(self.regime.repr_version))
        out["shs_move_buffer_len"] = float(len(self._move_buffer.batches) if self._move_buffer is not None else 0)
        for m, (ok, gain) in (self._last_move_log or {}).items():
            out[f"shs_move_{m}_accepted"] = float(bool(ok))
            out[f"shs_move_{m}_gain"] = float(gain)
        return out

    @torch.no_grad()
    def _calibrate_noise_prior(self, q_mean, is_first=None):
        reg = self.regime.regimes
        z = q_mean.detach()
        if z.shape[1] < 2:
            return
        dz = z[:, 1:] - z[:, :-1]
        # Calibration needs real innovations: a batch whose steps are all
        # episode starts contains no legal adjacent pair.
        _MIN_PAIRS = 32
        if is_first is not None:
            keep = (1.0 - is_first.reshape(z.shape[0], z.shape[1])[:, 1:]).to(dz.dtype)
            w = keep.reshape(-1)
            n_valid = float(w.sum())
            if n_valid < max(2.0, float(_MIN_PAIRS)):
                return                       # keep the prior and the unset latch
            d = dz.reshape(-1, dz.shape[-1])
            n = w.sum().clamp_min(1.0)
            mu = (w[:, None] * d).sum(0) / n
            var = (w[:, None] * (d - mu) ** 2).sum(0) / n
        else:
            if dz.shape[0] * dz.shape[1] < max(2, _MIN_PAIRS):
                return
            var = dz.reshape(-1, dz.shape[-1]).var(0)
        var = var.clamp_min(1e-8).to(reg.b0.dtype if torch.is_tensor(reg.b0)
                                     else torch.float32)
        a0 = float(reg.a0) if not torch.is_tensor(reg.a0) else float(reg.a0.reshape(-1)[0])
        b_new = self._shs_calibrate_sF * (a0 - 1.0) * var
        old = float(reg.b0.reshape(-1)[0]) if torch.is_tensor(reg.b0) else float(reg.b0)
        if torch.is_tensor(reg.b0):
            # b0 is per-output-dimension for the diagonal head and a single shared
            # scalar for the tied-carry head; reduce when the target is scalar rather
            # than reshaping, which is not defined for a length-L vector.
            tgt = b_new.to(reg.b0.dtype)
            reg.b0.copy_(tgt.mean() if reg.b0.dim() == 0
                         else tgt.reshape(reg.b0.shape))
        else:
            reg.b0 = float(b_new.mean())
        self._shs_b0_calibrated = True
        eq = (b_new / (a0 - 1.0))
        print("[SHS] noise prior calibrated from data: E[Q_ii] %s "
              "(was %.4g); ratio %.3g" % (
                  np.array2string(eq.detach().cpu().numpy(), precision=5),
                  old / (a0 - 1.0), float(eq.mean()) / max(old / (a0 - 1.0), 1e-12)))

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        if self._shs_strict_elbo and (float(free) != 0.0 or float(dyn_scale) != 1.0
                                      or float(rep_scale) != 0.0):
            raise ValueError(
                f"shs_strict_elbo=True but the RUNTIME KL profile is (kl_free={free}, "
                f"dyn_scale={dyn_scale}, rep_scale={rep_scale}); strict requires "
                "(0, 1, 0). Set kl_free: 0.0, dyn_scale: 1.0, rep_scale: 0.0 (the "
                "`shs_strict` preset does this) -- the constructor cannot see these "
                "training-loop values, so they are validated here on every call.")
        if self._pending is not None:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                _rv = self._pending[8] if len(self._pending) > 8 else None
                self.regime.update_globals(*self._pending[:7],
                                           batch_id=self._pending[7],
                                           action=(self._pending[9] if len(self._pending) > 9 else None),
                                           prev_stoch=(self._pending[10] if len(self._pending) > 10 else None),
                                           episode_first=getattr(self, "_pending_episode_first", None),
                                           repr_version=_rv)
            self._pending = None

        q_mean = post["mean"].float()
        q_std = post["std"].float()
        is_first = self._last_is_first
        episode_first = getattr(self, "_last_episode_first", None)
        if episode_first is not None and is_first is not None \
                and torch.equal(episode_first, is_first):
            episode_first = None
        stoch = post["stoch"].float()
        deter = post["deter"].float()
        self._shs_step += 1

        # b0 is the Gamma RATE prior on the regime noise precision, so its effect is
        # relative to the residual sum of squares: E[sigma^2_k] = (b0 + SSR_k/2)/(a0 + N_k/2).
        # Calibrating it on the first gradient step sets it from a RANDOMLY INITIALISED
        # encoder, i.e. from the variance of noise rather than of dynamics, and
        # _shs_b0_calibrated then latches so it is never revisited. Defer to the end of
        # the encoder-only curriculum phase, which is the first point at which the latent
        # scale has stopped moving and the last point before any structural move depends
        # on it.
        _b0_at = int(self._curriculum[0].get("until", 0)) if self._curriculum else 0
        if (self._shs_calibrate_b0 and not self._shs_b0_calibrated
                and self._shs_step >= max(1, _b0_at)):
            self._calibrate_noise_prior(q_mean, is_first)

        _pz = stoch if getattr(self, "_shs_moment_consistent", True) else None

        self._apply_curriculum()

        if self._move_buffer is not None:
            with torch.no_grad():
                if self._shs_analytic_estep:
                    self._move_buffer.add(
                        q_mean.detach(), deter.detach(),
                        None if is_first is None else is_first.detach(),
                        (q_std ** 2).detach(), step=self._shs_step,
                        batch_id=f"live{self._shs_step}",
                        repr_version=int(self.regime.repr_version),
                        action=(self._aligned_action(q_mean).detach()
                                if self._shs_action_dim > 0 else None),
                        episode_first=(None if episode_first is None
                                       else episode_first.detach()),
                        prev_stoch=(None if _pz is None else _pz.detach()))
                else:
                    self._move_buffer.add(
                        stoch.detach(), deter.detach(),
                        None if is_first is None else is_first.detach(),
                        step=self._shs_step,
                        batch_id=f"live{self._shs_step}",
                        repr_version=int(self.regime.repr_version),
                        action=(self._aligned_action(q_mean).detach()
                                if self._shs_action_dim > 0 else None),
                        episode_first=(None if episode_first is None
                                       else episode_first.detach()))
        _scheduled = (self._shs_move_every > 0
                      and self._shs_step >= self._shs_move_warmup
                      and self._shs_step % self._shs_move_every == 0)
        if _scheduled and self._live_sweep_allowed():
            self._n_live_sweeps_run += 1
            self.regime._belief_map = None      # maps compose within one sweep only
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
                    lap=float(self._shs_step) / max(1, self._shs_move_every),
                    delete_topk=getattr(self, "_shs_delete_topk", 3),
                merge_topm=self._shs_merge_topm)
            accepted = {m: round(float(g), 3) for m, (ok, g) in self._last_move_log.items() if ok}
            if getattr(self, "_teacher", None) is not None:
                self._teacher.mark_move()
            if accepted:
                print(f"[SHS] move sweep @ grad-step {self._shs_step}: K={self.regime.K} "
                      f"accepted={accepted}")
            # Beliefs produced this step were formed under the pre-sweep regime
            # identities, so remap them once from this sweep's map.
            # Structural handoff.  Beliefs held by ANY consumer -- the replay
            # posterior here, and the acting state that Dreamer keeps across
            # training updates -- were formed under the pre-sweep regime
            # identities.  Bump a structural generation, keep the composed map
            # keyed by the generation it maps FROM, and reconcile lazily: any
            # belief carrying an older generation is mapped (or rebuilt) the
            # next time it is used, including when K is unchanged.
            _bm = getattr(self.regime, "_belief_map", None)
            if _bm is not None:
                _g0 = int(getattr(self, "_shs_struct_gen", 0))
                self._shs_struct_gen = _g0 + 1
                # One-step maps: entry g maps generation g -> g+1.  The chain
                # is composed at lookup by _reconcile_resp.
                _hist = getattr(self, "_shs_belief_maps", None) or {}
                _hist[_g0] = _bm.detach().clone()
                _oldest = min(int(getattr(self, "_shs_acting_gen", _g0)), _g0)
                for _g in [k for k in _hist if k < _oldest - 1]:
                    _hist.pop(_g)                       # nothing can still need it
                self._shs_belief_maps = _hist
                if isinstance(post, dict) and "regime_resp" in post:
                    post["regime_resp"] = self._reconcile_resp(
                        post["regime_resp"], self._shs_struct_gen - 1)
            self.regime._belief_map = None
        elif _scheduled:
            self._n_live_sweeps_skipped = getattr(self, "_n_live_sweeps_skipped", 0) + 1
            self._last_move_log = {}

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            gamma, counts, start_counts, _ = self.regime.regime_inference(
                q_mean.float(), deter.float(), is_first, cache_estep=True,
                z_var=(q_std ** 2).float(),
                prev_stoch=None if _pz is None else _pz.float(),
                action=self._aligned_action(q_mean),
                episode_first=episode_first)
        post["regime_resp"] = gamma.detach()

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
            action=self._aligned_action(q_mean),
            prev_stoch=_pz,
            episode_first=episode_first,
        )

        if (self._shs_step % self._shs_update_every == 0
                and self._shs_global_source != "completed_episode_stream"):
            self._pending_episode_first = getattr(
                self, "_last_episode_first", None)
            _act = None if self._last_action is None else self._aligned_action(q_mean).detach()
            pend = [q_mean.detach(), deter.detach(), gamma.detach(),
                    counts.detach(), start_counts.detach(),
                    None if is_first is None else is_first.detach(),
                    (q_std ** 2).detach(), _act, None if _pz is None else _pz.detach()]
            S = int(self._shs_posterior_samples)
            if (S > 1 and _pz is not None and self._last_embed is not None
                    and self._last_action is not None and is_first is not None
                    and getattr(self.regime, "_estep", None) is not None):
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                    es_main = {k: (v.clone() if torch.is_tensor(v) else v)
                               for k, v in self.regime._estep.items()}
                    # The S-1 extra draws are independent, so they run as one
                    # replicated batch: a single length-T recurrence and a single
                    # forward-backward instead of S-1 of each.
                    R_ = S - 1
                    _emb_r = self._last_embed.float().repeat(R_, 1, 1)
                    _act_r = self._last_action.float().repeat(R_, 1, 1)
                    _isf_r = is_first.float().repeat(R_, 1)
                    _epf_r = (None if episode_first is None
                              else episode_first.repeat(R_, 1))
                    m_s, s_s, z_s, d_s = self._posterior_redraw(_emb_r, _act_r, _isf_r)
                    _act_al = None if _act is None else _act.repeat(R_, 1, 1)
                    g_s, c_s, sc_s, _ = self.regime.regime_inference(
                        m_s, d_s, _isf_r, cache_estep=True, z_var=s_s ** 2,
                        prev_stoch=z_s, action=_act_al, episode_first=_epf_r)
                    es_extra = {k: (v.clone() if torch.is_tensor(v) else v)
                                for k, v in self.regime._estep.items()}
                    # Batch-concatenate the draws; every global statistic is a
                    # gamma-weighted sum over batch x time, so scaling the
                    # responsibilities / counts / gate masses by 1/S makes the
                    # concatenation the S-average of the per-draw statistics.
                    _cat = lambda a, b: (None if a is None or b is None
                                         else torch.cat([a, b], 0))
                    pend = [_cat(pend[0], m_s), _cat(pend[1], d_s),
                            _cat(pend[2], g_s) / S,
                            (pend[3] + c_s) / S, (pend[4] + sc_s) / S,
                            _cat(pend[5], _isf_r.detach()), _cat(pend[6], s_s ** 2),
                            (None if _act is None else _cat(_act, _act_al.detach())),
                            _cat(pend[8], z_s)]
                    _B0 = int(q_mean.shape[0])
                    es = {}
                    for k, v in es_main.items():
                        w = es_extra.get(k)
                        if (torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == _B0
                                and torch.is_tensor(w)):
                            es[k] = torch.cat([v, w], 0)
                        else:
                            es[k] = v
                    es["r_mass"] = es["r_mass"] / S
                    es["row_weight"] = es["row_weight"] / S
                    es["sig"] = es["r_mass"] / es["row_weight"].clamp_min(1e-12)
                    self.regime._estep = es
                    if self._pending_episode_first is not None:
                        self._pending_episode_first = torch.cat(
                            [self._pending_episode_first] * S, 0)
            self._pending = (pend[0], pend[1], pend[2], pend[3], pend[4], pend[5], pend[6],
                             self._next_batch_id, int(self.regime.repr_version),
                             pend[7], pend[8])
            self._next_batch_id = None
            self._last_embed = None      # nothing else needs it; do not pin the batch

        return loss, value.detach(), dyn.detach(), rep.detach()

    @torch.no_grad()
    def annotate_regime_resp(self, post, is_first=None, action=None):
        deter = post["deter"].float()
        if "mean" in post and "std" in post:
            z = post["mean"].float()
            z_var = post["std"].float() ** 2
        else:
            z = post["stoch"].float()
            z_var = None
        gamma, _, _, _ = self.regime.regime_inference(
            z, deter, is_first, cache_estep=False, z_var=z_var,
            action=(action if self._shs_action_dim > 0 else None))   
        post["regime_resp"] = gamma
        return post

    @torch.no_grad()
    def prior_entropy_bounds(self, state):
        lo = self.prior_entropy(state, mode="bounds")
        return lo, self._last_entropy_bounds[1]

    def set_batch_id(self, batch_id):
        self._next_batch_id = batch_id

    # Scalar bookkeeping that lives outside state_dict but affects behaviour:
    # first-batch latches decide whether a global update initialises or blends,
    # and the step counters drive the pg_every / hdp_every schedules.
    _SCALAR_STATE = (
        ("regime", ("_counts_initialised", "_gstep_count", "_struct_gen", "_lap",
                    "_curr_phase",
                    "_episode_cursor", "_async_version", "_repr_frozen",
                    "_async_repr_rejects")),
        ("regime.regimes", ("_stats_initialised", "_mstep_count")),
        ("regime.rstick", ("_pg_init", "n_pg_guard_rejects")),
        ("regime.stat_store", ("n_updates", "_stream_count", "pass_id",
                               "n_stale_invalidated")),
        ("", ("_lap", "_shs_step", "_curr_phase", "_n_live_sweeps_run",
              "_n_live_sweeps_skipped", "_shs_last_teacher_version",
              "_next_move_step", "_shs_b0_calibrated", "_shs_struct_gen",
              "_shs_acting_gen")),
    )

    def _scalar_owner(self, path):
        obj = self
        for part in [p for p in path.split(".") if p]:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj

    def _collect_scalars(self):
        # The effective curriculum phase settings travel with the checkpoint so
        # a pending transaction is validated against the phase that produced it.
        out = {"regime|repr_version": int(self.regime.repr_version),
               "phase|recurrent": bool(self.regime.recurrent),
               "phase|kappa": float(self.regime.hdp.kappa),
               "phase|move_every": int(getattr(self, "_move_every", 0) or 0),
               "phase|move_threshold": float(getattr(self, "_move_threshold", 0.0)),
               "phase|birth": bool(getattr(self, "_shs_move_birth", True)),
               "phase|split": bool(getattr(self, "_shs_move_split", True)),
               "phase|create_bonus": float(getattr(self, "_shs_create_bonus", 0.0))}
        for path, names in self._SCALAR_STATE:
            obj = self._scalar_owner(path)
            if obj is None:
                continue
            for n in names:
                if hasattr(obj, n):
                    v = getattr(obj, n)
                    if isinstance(v, (bool, int, float)):
                        out[f"{path}|{n}"] = v
        return out

    def _apply_scalars(self, d):
        if not d:
            return
        rv = d.get("regime|repr_version")
        if rv is not None:
            self.regime.repr_version.fill_(int(rv))
        if "phase|recurrent" in d:
            self.regime.recurrent = bool(d["phase|recurrent"])
            self.regime.hdp.kappa = float(d["phase|kappa"])
            self._move_every = int(d.get("phase|move_every", 0))
            self._move_threshold = float(d.get("phase|move_threshold", 0.0))
            self._shs_move_birth = bool(d.get("phase|birth", True))
            self._shs_move_split = bool(d.get("phase|split", True))
            self._shs_create_bonus = float(d.get("phase|create_bonus", 0.0))
        for key, v in d.items():
            if key == "regime|repr_version" or key.startswith("phase|"):
                continue
            path, _, n = key.partition("|")
            obj = self._scalar_owner(path)
            if obj is not None and hasattr(obj, n):
                try:
                    setattr(obj, n, type(getattr(obj, n))(v))
                except Exception:
                    setattr(obj, n, v)

    def get_extra_state(self):
        def cpu(o):
            if o is None:
                return None
            if torch.is_tensor(o):
                return o.detach().cpu()
            if isinstance(o, (tuple, list)):
                return tuple(cpu(v) for v in o)
            return o
        _es = getattr(self.regime, "_estep", None)
        _rs = getattr(self.regime, "rstick", None)
        _pend_rec = bool(self.regime.recurrent) if self._pending is not None else None
        # Persist the most recent move-buffer batches in half precision; these
        # are scoring inputs, not parameters.
        _buf = []
        if (self._move_buffer is not None and self._move_buffer.batches
                and int(getattr(self, "_shs_ckpt_buffer", 0)) > 0):
            _cfg_keep = int(getattr(self, "_shs_ckpt_buffer", 0))
            _keep = 0 if _cfg_keep <= 0 else max(
                int(getattr(self, "_shs_min_live_batches", 4)), _cfg_keep)
            for _b in self._move_buffer.batches[-_keep:]:
                _e = {}
                for _f in ("stoch", "deter", "is_first", "z_var", "valid", "action",
                           "episode_first", "prev_stoch", "z_cov", "zg_xcov"):
                    _v = getattr(_b, _f, None)
                    if torch.is_tensor(_v):
                        _e[_f] = _v.detach().to(torch.float16).cpu()
                _e["step"] = int(_b.step)
                _e["batch_id"] = _b.batch_id
                _e["repr_version"] = _b.repr_version
                _buf.append(_e)
        _ss = getattr(self.regime, "stat_store", None)
        return dict(move_buffer=_buf,
                    buffer_repr_version=int(self.regime.repr_version),
                    scalars=self._collect_scalars(),
                    init_flags=dict(
                        pg_init=bool(getattr(_rs, "_pg_init", False)) if _rs is not None else False,
                        counts_initialised=bool(getattr(self.regime, "_counts_initialised", False)),
                        store_n_updates=int(getattr(_ss, "n_updates", 0)) if _ss is not None else 0),
                    pending=cpu(self._pending),
                    pending_estep=(None if _es is None
                                   else {k: cpu(v) for k, v in _es.items()}),
                    pending_episode_first=cpu(getattr(self, "_pending_episode_first", None)),
                    pending_recurrent=_pend_rec,
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
        _es = state.get("pending_estep")
        # The gate E-step cache belongs to the same pending transaction as the
        # continuous statistics; if either half is missing the transaction is
        # dropped rather than half-applied.
        self.regime._estep = None if _es is None else {k: mv(v) for k, v in _es.items()}
        self._pending_episode_first = mv(state.get("pending_episode_first"))
        # Only a transaction produced by a recurrent phase carries a gate cache.
        _pr = state.get("pending_recurrent")
        if _pr is None:
            _pr = bool(getattr(self.regime, "recurrent", False))
        if self._pending is not None and bool(_pr) and self.regime._estep is None:
            self._pending = None
            self._pending_episode_first = None
        # The first global update of a run initialises statistics from the full
        # batch instead of blending, so these latches travel with the state.
        # Restore the effective phase before validating the pending transaction.
        self._apply_scalars(state.get("scalars"))
        _fl = state.get("init_flags") or {}
        _rs = getattr(self.regime, "rstick", None)
        if _rs is not None and _fl.get("pg_init"):
            _rs._pg_init = True
        if _fl.get("counts_initialised"):
            self.regime._counts_initialised = True
        _ss = getattr(self.regime, "stat_store", None)
        if _ss is not None and _fl.get("store_n_updates"):
            try:
                _ss.n_updates = max(int(_ss.n_updates), int(_fl["store_n_updates"]))
            except Exception:
                pass
        _buf = state.get("move_buffer") or []
        _bv = state.get("buffer_repr_version")
        if _bv is not None:
            # entries that were already stale when the checkpoint was written stay stale
            _buf = [_e for _e in _buf if _e.get("repr_version") == _bv]
        if _buf and self._move_buffer is not None and not self._move_buffer.batches:
            _dev = self.regime.z0.device
            _dt = self.regime.z0.dtype
            for _e in _buf:
                _t = lambda k: (_e[k].to(device=_dev, dtype=_dt) if k in _e else None)
                try:
                    self._move_buffer.add(
                        _t("stoch"), _t("deter"), _t("is_first"), _t("z_var"),
                        step=int(_e.get("step", 0)), batch_id=_e.get("batch_id"),
                        repr_version=_e.get("repr_version"), action=_t("action"),
                        valid=_t("valid"), z_cov=_t("z_cov"), zg_xcov=_t("zg_xcov"),
                        episode_first=_t("episode_first"), prev_stoch=_t("prev_stoch"))
                except Exception:
                    break
            # The restored batches belong to the representation being restored
            # with them.  Their stamp is translated to the version in force once
            # the whole load has settled (_finish_buffer_restore); any change
            # after that invalidates them in the ordinary way.
            self._shs_buffer_ckpt_version = _bv
        self._next_batch_id = state.get("next_batch_id")
        self._shs_step = int(state.get("shs_step", 0))

    def _finish_buffer_restore(self):
        """Translate restored move-buffer stamps to the representation version
        in force at the end of loading, once and only for the batches that were
        valid in the checkpoint."""
        _bv = getattr(self, "_shs_buffer_ckpt_version", None)
        if _bv is None or self._move_buffer is None:
            return
        _rv = int(self.regime.repr_version)
        for _b in self._move_buffer.batches:
            if _b.repr_version == _bv:
                _b.repr_version = _rv
        self._shs_buffer_ckpt_version = None

    def begin_full_batch_pass(self):
        self.regime.begin_full_batch_pass()

    def finalize_full_batch_pass(self):
        return self.regime.finalize_full_batch_pass()

    def _live_sweep_allowed(self) -> bool:
        buf = self._move_buffer
        if buf is None or not getattr(self, "_shs_live_moves", True):
            return False
        if getattr(buf, "complete", False):
            return buf.is_complete()
        if self._shs_moves_complete:
            return False
        # Drop batches encoded under a superseded representation, then require
        # (i) enough same-version batches and (ii) a teacher drift small enough
        # that the window is a single coordinate system.  A skipped sweep costs
        # nothing; a move scored across representations costs the run.
        try:
            dropped = buf.purge_stale(int(self.regime.repr_version))
            if dropped:
                self._shs_stale_dropped = getattr(self, "_shs_stale_dropped", 0) + dropped
        except Exception:
            pass
        enough = len(buf.batches) >= int(getattr(self, "_shs_min_live_batches", 4))
        store = getattr(self.regime, "stat_store", None)
        ema_mode = (store is None) or (store.mode == "legacy_ema")
        t = getattr(self, "_teacher", None)
        drift_ok = True if t is None else bool(t.safe_for_moves())
        if not drift_ok:
            self._shs_drift_skips = getattr(self, "_shs_drift_skips", 0) + 1
        return enough and ema_mode and drift_ok

    def bump_repr_version(self):
        self.regime.bump_repr_version()

    def prior_entropy(self, state, n_samples: int = 64, mode: str | None = None):
        
        R = self.regime
        stoch = state["stoch"].float()
        deter = state["deter"].float()
        isf = state.get("is_first", None)
        prev = R._prev_stoch(stoch, isf)
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
        comp_mean, comp_var = R.regimes.predictive_moments(g, g_var=g_var)
        rr = state.get("regime_resp", None)
        if rr is not None and rr.shape[-1] != R.K:
            rr = self._map_regime_resp(rr)
        if rr is None or rr.shape[-1] != R.K:
            rr = stoch.new_full(stoch.shape[:2] + (R.K,), 1.0 / R.K)
        rr_prev = R._shift_resp(rr.float(), isf)
        if R.recurrent:
            phi = (R.rstick.pack(*R.rstick.build_phi_moments(
                       prev, torch.zeros_like(prev),
                       _act if R.rstick_action_dim > 0 else None))
                   if getattr(R, 'gate_input', 'carry') == 'latent'
                   else R.build_stick_phi(deter, _act))
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
            mode = "bounds"
        if mode == "bounds":
            from .mixture_prior import mixture_entropy_bounds
            lo, hi = mixture_entropy_bounds(comp_mean, comp_var, w)
            self._last_entropy_bounds = (lo.detach(), hi.detach())
            return lo.clamp_min(1e-8)
        if R.regimes.q_rank > 0:
            _, comp_d, comp_U = R.regimes.predictive_cov_moments(g, g_var=g_var)
            return mixture_entropy_monte_carlo_lowrank(comp_mean, comp_d, comp_U, w,
                                                       n_samples=n_samples)
        return mixture_entropy_monte_carlo(comp_mean, comp_var, w, n_samples=n_samples)

    @torch.no_grad()
    def _posterior_redraw(self, embed, action, is_first):
        """One fresh sequential draw z_{1:T} ~ q(z_{1:T}) = prod_t q(z_t | h_t),
        h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1}), from cached embeddings.  Returns
        (mean, std, stoch, deter), each (B, T, .)."""
        B, T = embed.shape[:2]
        state = None
        means, stds, stochs, deters = [], [], [], []
        for t in range(T):
            e, a, f = embed[:, t], action[:, t], is_first[:, t]
            if state is None or float(f.sum()) == len(f):
                state = self.initial(B)
                a = torch.zeros_like(a)
            elif float(f.sum()) > 0:
                fr = f[:, None]
                a = a * (1.0 - fr)
                init = self.initial(B)
                state = {k: v * (1.0 - fr.reshape(fr.shape + (1,) * (v.dim() - 2)))
                         + init[k] * fr.reshape(fr.shape + (1,) * (v.dim() - 2))
                         for k, v in state.items()}
            deter = self._deter_step(state, a)
            x = self._obs_out_layers(torch.cat([deter, e], -1))
            stats = self._suff_stats_layer("obs", x)
            stoch = self.get_dist(stats).sample()
            state = dict(state); state.update(stats); state["stoch"] = stoch; state["deter"] = deter
            means.append(stats["mean"]); stds.append(stats["std"]); stochs.append(stoch); deters.append(deter)
        return (torch.stack(means, 1), torch.stack(stds, 1), torch.stack(stochs, 1), torch.stack(deters, 1))

    def _deter_step(self, prev_state, prev_action):
        x = torch.cat([prev_state["stoch"], prev_action], -1)
        x = self._img_in_layers(x)
        deter = prev_state["deter"]
        for _ in range(self._rec_depth):
            x, deter = self._cell(x, [deter]); deter = deter[0]
        return deter

    def img_step(self, prev_state, prev_action, sample=True):
        prev_action = self._zero_action_like(prev_state["deter"], prev_action)
        # Same reconciliation as obs_step: a rollout state may predate a move,
        # including one that preserved K.
        prev_state = self._sync_state_gen(prev_state)
        deter = self._deter_step(prev_state, prev_action)
        B = deter.shape[0]
        resp_prev = prev_state.get("regime_resp", None)
        if resp_prev is not None and resp_prev.shape[-1] != self.regime.K:
            resp_prev = self._map_regime_resp(resp_prev)
        if resp_prev is None or resp_prev.shape[-1] != self.regime.K:
            resp_prev = self._initial_regime_law(deter.dtype, deter.device).view(1, -1).expand(B, -1)
        # In a sampled rollout the predecessor sample already realises its own
        # uncertainty, so the predecessor variance must not be propagated on top
        # of it.  Only the mean-propagated (sample=False) rollout carries a
        # distribution forward analytically, so only then is prev_var used.
        prev_var = None
        if (not sample) and "std" in prev_state:
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
        z0 = self.regime.z0.to(state["stoch"].dtype)
        state["stoch"] = z0.view(1, -1).expand(batch_size, -1).contiguous()
        if "mean" in state:
            state["mean"] = state["stoch"].clone()
            state["std"] = torch.zeros_like(state["stoch"])
        state["regime_resp"] = torch.full(
            (batch_size, self.regime.K), 1.0 / self.regime.K, device=self._device
        )
        return state


SHS_NON_CTOR_KEYS = {
    "shs_calibrate_b0",
    "shs_calibrate_sF",
    "shs_diag_log", "shs_diag_figures",
    "shs_stream_episodes",
    "shs_repr_epoch_steps",
    "shs_consolidate_every_episodes", "shs_consolidate_warmup",
    "shs_consolidate_batches", "shs_consolidate_sweeps", "shs_consolidate_ep_len",
    "shs_stream_chunked",
    "shs_inference_mode",
    "shs_ckpt_reservoir",
}

# Mirror of SHS_NON_CTOR_KEYS for the other direction.
SHS_PINNED_KEYS = set()


def shs_kwargs_from_config(config):
    import inspect
    accepted = {n for n in inspect.signature(SHSRSSM.__init__).parameters
                if n.startswith("shs_")}
    try:
        items = {k: v for k, v in vars(config).items() if k.startswith("shs_")}
    except TypeError:
        items = {k: getattr(config, k) for k in dir(config) if k.startswith("shs_")}
    try:
        # train_ratio is gradient steps per agent step, while the *_env schedule
        # keys are written in raw environment frames (the same units as `steps:`
        # in configs.yaml, which dreamer.py divides by action_repeat).  Convert
        # per frame so curriculum boundaries land where the config says.
        items.setdefault(
            "shs_grad_per_env",
            float(config.train_ratio)
            / (float(config.batch_size) * float(config.batch_length))
            / max(1.0, float(getattr(config, "action_repeat", 1) or 1)))
    except Exception:
        pass
    unknown = sorted(k for k in items if k not in accepted and k not in SHS_NON_CTOR_KEYS)
    if unknown:
        raise ValueError(
            "dead config wiring: these shs_* keys are not SHSRSSM constructor kwargs "
            f"and not registered loop-level keys: {unknown}. Either add them to "
            "SHSRSSM.__init__ or to SHS_NON_CTOR_KEYS in shs_rssm/shs_rssm.py.")
    unreachable = sorted(accepted - set(items) - SHS_PINNED_KEYS)
    if unreachable:
        raise ValueError(
            "unreachable knobs: these SHSRSSM constructor kwargs have no "
            f"configs.yaml entry: {unreachable}. They are pinned at their "
            "Python defaults and cannot be set from the command line, because "
            "dreamer.py builds its --flags from the merged config dict. Add "
            "them to the `defaults:` block of configs.yaml, or to "
            "SHS_PINNED_KEYS in shs_rssm/shs_rssm.py if the Python default is "
            "intentionally not user-facing.")
    kw = {k: v for k, v in items.items() if k in accepted}
    if int(kw.get("shs_action_dim", 0)) == -1:
        n_act = int(getattr(config, "num_actions", 0))
        if n_act <= 0:
            raise ValueError(
                "shs_action_dim: -1 means 'inherit config.num_actions', but "
                f"num_actions resolved to {n_act}. dreamer.py sets it from the "
                "env action space before the world model is built, so this "
                "means SHSRSSM is being constructed off that path. Set "
                "shs_action_dim to the action width explicitly.")
        kw["shs_action_dim"] = n_act
    return kw
from __future__ import annotations

import math
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
                 shs_moment_consistent: bool = False,
                 shs_gate_input: str = "carry",
                 shs_gate_inner_iters: int = 2,
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
                 shs_move_version: str = "repr",
                 shs_move_first: int | None = None,
                 shs_move_first_env: float | None = None,
                 shs_imag_propagate_var: bool = False,
                 shs_move_K_headroom: int | None = None,
                 shs_delete_empty: bool = False,
                 shs_move_holdout: int = 0,
                 shs_move_holdout_z: float = 2.0,
                 shs_move_holdout_horizon: int = 15,
                 shs_move_holdout_tol: float = 1.0,
                 shs_move_commit: str = "remap",
                 shs_structure_moves: bool = False,
                 shs_ema_gate: bool = True,
                 shs_ema_gate_ref: float = 0.0,
                 shs_ema_gate_floor: float = 0.0,
                 shs_forget_mode: str = "fixed",
                 shs_schedule_unit: str = "agent",
                 **kwargs):
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
        if shs_move_first_env is not None:
            if shs_move_first is not None:
                raise ValueError("set shs_move_first or shs_move_first_env, not both")
            shs_move_first = _env2gs(shs_move_first_env)
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
        if (shs_move_every and int(shs_move_every) > 0
                and bool(shs_moment_consistent)):
            warnings.warn(
                "shs_move_every>0 with shs_moment_consistent=True: live moves are "
                "scored on the analytic bound while training optimises the "
                "sample-conditional one. Set `shs_moment_consistent: False` to put "
                "both on one objective, or run moves through consolidation.",
                RuntimeWarning, stacklevel=2)       
        self._shs_corpus_batches = shs_corpus_batches
        self._global_scale = (1.0 / float(shs_corpus_batches)
                              if (self._shs_strict_elbo and shs_corpus_batches) else 1.0)
        self._shs_entropy_mode = str(shs_entropy_mode)
        self._shs_moves_complete = bool(shs_moves_complete)
        self._shs_move_version = str(shs_move_version)
        if self._shs_move_version not in ("repr", "teacher"):
            raise ValueError(
                f"shs_move_version must be 'repr' or 'teacher', got {shs_move_version!r}")
        if self._shs_move_version == "teacher" and not shs_teacher:
            raise ValueError("shs_move_version='teacher' requires shs_teacher=True")
        self._shs_structure_moves = bool(shs_structure_moves)
        if not self._shs_structure_moves:
            shs_move_every = 0
            shs_move_first = None
        self._move_first = None if shs_move_first is None else int(shs_move_first)
        if self._move_first is not None and self._move_first < 1:
            raise ValueError(f"shs_move_first must be >= 1 gradient step, got {shs_move_first!r}")
        self._move_first_pending = self._move_first is not None
        self._shs_birth_style = shs_birth_style
        self.regime.imag_sample_mixture = bool(shs_imag_sample_mixture)
        self._shs_imag_mode = (None if shs_imag_mode in (None, "", "none", "null") else str(shs_imag_mode))
        self._shs_imag_propagate_var = bool(shs_imag_propagate_var)
        self._move_K_headroom = None if shs_move_K_headroom is None else int(shs_move_K_headroom)
        self._move_K_cap = (None if self._move_K_headroom is None
                            else int(shs_K) + self._move_K_headroom)
        self.regime._shs_online_pairwise = bool(shs_online_pairwise)
        self._shs_merge_topm = shs_merge_topm
        self._shs_global_source = str(shs_global_source)
        self._shs_action_dim = int(shs_action_dim)
        self._last_action = None
        self._last_acq_time = None
        self._last_episode_uid = None
        self._shs_analytic_estep = bool(shs_analytic_estep)
        self._shs_moment_consistent = bool(shs_moment_consistent)
        self._shs_calibrate_b0 = bool(shs_calibrate_b0)
        self._shs_calibrate_sF = float(shs_calibrate_sF)
        self._shs_b0_calibrated = False
        self.regime.chunk_boundary_mask = bool(shs_chunk_boundary_mask)
        self.regime.ema_gate = bool(shs_ema_gate)
        self.regime.ema_gate_ref = float(shs_ema_gate_ref)
        self.regime.ema_gate_floor = float(min(1.0, max(0.0, shs_ema_gate_floor)))
        if str(shs_forget_mode) not in ("fixed", "evidence"):
            raise ValueError(f"shs_forget_mode must be 'fixed' or 'evidence', got {shs_forget_mode!r}")
        self.regime.forget_mode = str(shs_forget_mode)
        if str(shs_move_commit) not in ("remap", "window"):
            raise ValueError(f"shs_move_commit must be 'remap' or 'window', got {shs_move_commit!r}")
        self._shs_move_commit = str(shs_move_commit)
        self._shs_move_holdout_tol = float(shs_move_holdout_tol)
        self._shs_schedule_unit = str(shs_schedule_unit)
        self._shs_free = shs_free
        self._shs_update_every = shs_global_update_every
        self._shs_move_every = shs_move_every
        self._shs_move_warmup = shs_move_warmup
        self._shs_move_birth = shs_move_birth
        self._shs_move_split = bool(shs_move_split)
        self._shs_move_confirm_top = shs_move_confirm_top
        self._shs_delete_mode = shs_delete_mode
        self._shs_delete_empty = bool(shs_delete_empty)
        self._shs_move_holdout = int(shs_move_holdout)
        self._shs_move_holdout_z = float(shs_move_holdout_z)
        self._shs_move_holdout_horizon = int(shs_move_holdout_horizon)
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
            print("[SHS] schedule units: %s steps, grad/step=%.5f  curriculum(grad-steps)=%s"
                  % (self._shs_schedule_unit, self._grad_per_env,
                     [int(q.get("until", -1)) for q in self._curriculum]))
        self._move_threshold = 0.0
        self._move_shrink_threshold = None
        self._move_create_bonus = 0.0
        _curric_moves = self._shs_structure_moves and any(
            float(p.get("move_every", shs_move_every)) > 0 for p in self._curriculum)
        self._move_buffer = MoveBuffer(max_batches=shs_move_buffer,
                                       max_age=shs_move_max_age) \
            if (shs_move_every > 0 or _curric_moves) else None
        self._shs_step = 0
        self._last_is_first = None
        self._pending = None
        self._n_live_sweeps_skipped = 0
        self._shs_min_live_batches = 4
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
                episode_first=None, acq_time=None, episode_uid=None):
        self._last_is_first = is_first
        self._last_episode_first = (episode_first if episode_first is not None
                                    else is_first)
        self._last_action = action.detach()
        self._last_acq_time = None if acq_time is None else acq_time.detach()
        self._last_episode_uid = None if episode_uid is None else episode_uid.detach()
        return super().observe(embed, action, is_first, state, sample=sample)

    def _gen_like(self, ref):
        return torch.full(tuple(ref.shape[:-1]) + (1,), float(getattr(self.regime, "_belief_gen", 0)),
                          dtype=torch.float32, device=ref.device)

    def aligned_resp(self, state):
        """Regime belief of `state` in the current structure, or None when it cannot be mapped."""
        R = self.regime
        rr = state.get("regime_resp") if isinstance(state, dict) else None
        if rr is None:
            return None
        K, cur = int(R.K), int(getattr(R, "_belief_gen", 0))
        gen = state.get("regime_gen")
        if gen is None:
            return rr if int(rr.shape[-1]) == K else None
        g = gen[..., 0].round().long()
        if bool((g == cur).all()):
            return rr if int(rr.shape[-1]) == K else None
        out = torch.full(tuple(rr.shape[:-1]) + (K,), 1.0 / K, dtype=torch.float32, device=rr.device)
        maps = getattr(R, "_belief_maps", None) or {}
        for v in torch.unique(g).tolist():
            sel = g == v
            if v == cur and int(rr.shape[-1]) == K:
                out[sel] = rr[sel].float()
                continue
            M = maps.get(int(v))
            if M is None or tuple(M.shape) != (K, int(rr.shape[-1])):
                continue
            m = (rr[sel].float() @ M.t().to(device=rr.device)).clamp_min(1e-8)
            out[sel] = m / m.sum(-1, keepdim=True)
        return out

    def _align_belief(self, state):
        rr = None if not isinstance(state, dict) else state.get("regime_resp")
        if rr is None:
            return state
        al = self.aligned_resp(state)
        if al is None:
            al = torch.full(tuple(rr.shape[:-1]) + (int(self.regime.K),),
                            1.0 / float(self.regime.K), dtype=torch.float32, device=rr.device)
        state["regime_resp"] = al.to(rr.dtype)
        state["regime_gen"] = self._gen_like(al)
        return state

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        prev_state = self._align_belief(prev_state)
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
                if self._shs_action_dim > 0 and prev_action is not None:
                    act = prev_action.reshape(z.shape[0], -1).float()
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
            post["regime_gen"] = self._gen_like(post["regime_resp"])
        return post, prior

    def _apply_curriculum(self):
        if not self._curriculum:
            return
        step = self._shs_step
        idx = next((i for i, p in enumerate(self._curriculum)
                    if int(p.get("until", -1)) < 0 or step < int(p["until"])),
                   len(self._curriculum) - 1)
        phase = self._curriculum[idx]
        if "move_every" in phase:
            self._shs_move_every = (int(phase["move_every"])
                                    if getattr(self, "_shs_structure_moves", True) else 0)
        if "move_threshold" in phase:
            self._move_threshold = float(phase["move_threshold"])
        if "shrink_threshold" in phase:
            _dt = phase["shrink_threshold"]
            self._move_shrink_threshold = None if _dt is None else float(_dt)
        if "create_bonus" in phase:
            self._move_create_bonus = float(phase["create_bonus"])
        if "birth" in phase:
            self._shs_move_birth = bool(phase["birth"])
        if "split" in phase:
            self._shs_move_split = bool(phase["split"])
        if "recurrent" in phase and getattr(self.regime, "rstick", None) is not None:
            self.regime.recurrent = bool(phase["recurrent"])
        if getattr(self.regime, "hdp", None) is not None:
            if self.regime.recurrent:
                self.regime.hdp.kappa = 0.0
            elif "kappa" in phase:
                self.regime.hdp.kappa = float(phase["kappa"])
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
                  f"shrink_threshold={self._move_shrink_threshold}, "
                  f"create_bonus={self._move_create_bonus}, "
                  f"birth={self._shs_move_birth}, split={self._shs_move_split}")

    def curriculum_state(self) -> dict:
        out = {
            "shs_curriculum_phase": float(self._curr_phase),
            "shs_move_every_live": float(self._shs_move_every),
            "shs_structure_moves": float(bool(getattr(self, "_shs_structure_moves", True))),
            "shs_kappa_live": float(getattr(self.regime.hdp, "kappa", 0.0)),
            "shs_move_threshold_live": float(self._move_threshold),
            "shs_shrink_threshold_live": float(self._move_threshold
                                               if self._move_shrink_threshold is None
                                               else self._move_shrink_threshold),
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
        out["shs_empty_regimes_removed"] = float(getattr(self.regime, "_n_empty_removed", 0))
        out["shs_move_refit_gain"] = float(getattr(self.regime, "_refit_gain", 0.0))
        out["shs_move_holdout_vetoes"] = float(getattr(self.regime, "_holdout_vetoes", 0))
        out["shs_move_fit_vetoes"] = float(getattr(self.regime, "_fit_vetoes", 0))
        out["shs_move_installed_gain"] = float(getattr(self.regime, "_installed_gain", 0.0))
        out["shs_move_holdout_base"] = float(getattr(self.regime, "_holdout_base_score", float("nan")))
        out["shs_move_holdout_episodes"] = float(getattr(self.regime, "_holdout_episodes", 0))
        out["shs_move_sweeps_deferred"] = float(getattr(self.regime, "_sweeps_deferred", 0))
        out["shs_active_regimes_window"] = float(getattr(self.regime, "_active_window", -1))
        _g = getattr(self.regime, "_ema_gate_last", None)
        if _g is not None:
            out["shs_ema_gate_mean"] = float(_g.mean())
            out["shs_ema_gate_at_floor"] = float((_g <= float(self.regime.ema_gate_floor) + 1e-12).sum())
        _errs = []
        for _f, _info in (getattr(self.regime, "_retention_last", None) or {}).items():
            out[f"shs_retention_{_f}_mean"] = float(_info["r"].mean())
            out[f"shs_retention_{_f}_min"] = float(_info["r"].min())
            out[f"shs_drift_prob_{_f}_max"] = float(_info["P"].max())
            out[f"shs_drift_rate_{_f}"] = float(_info["rate"])
            out[f"shs_drift_exposure_{_f}_mean"] = float(_info["usage"].mean())
            _errs.append(float(_info["err"]))
        for _f, _post in (getattr(self.regime, "_drift", None) or {}).items():
            out[f"shs_drift_events_{_f}"] = float(_post.events)
        if _errs:
            out["shs_retention_quad_err"] = max(_errs)
        out["shs_retention_guard_hits"] = float(getattr(self.regime, "_forget_guard_hits", 0))
        for _nm, _mod in (("hdp", self.regime.hdp), ("pg", self.regime.rstick)):
            _att = float(getattr(_mod, "n_guard_attempts", 0))
            _rej = float(getattr(_mod, f"n_{'pg_' if _nm == 'pg' else ''}guard_rejects", 0))
            out[f"shs_guard_reject_rate_{_nm}"] = _rej / max(_att, 1.0)
        _cap = getattr(self, "_move_K_cap", None)
        out["shs_move_K_cap"] = float(-1 if _cap is None else _cap)
        out["shs_move_K_capped"] = float(_cap is not None and int(self.regime.K) >= _cap)
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
        if is_first is not None:
            keep = (1.0 - is_first.reshape(z.shape[0], z.shape[1])[:, 1:]).to(dz.dtype)
            w = keep.reshape(-1)
            d = dz.reshape(-1, dz.shape[-1])
            n = w.sum().clamp_min(1.0)
            mu = (w[:, None] * d).sum(0) / n
            var = (w[:, None] * (d - mu) ** 2).sum(0) / n
        else:
            var = dz.reshape(-1, dz.shape[-1]).var(0)
        var = var.clamp_min(1e-8).to(reg.b0.dtype if torch.is_tensor(reg.b0)
                                     else torch.float32)
        a0 = float(reg.a0) if not torch.is_tensor(reg.a0) else float(reg.a0.reshape(-1)[0])
        b_new = self._shs_calibrate_sF * (a0 - 1.0) * var
        old = float(reg.b0.reshape(-1)[0]) if torch.is_tensor(reg.b0) else float(reg.b0)
        if torch.is_tensor(reg.b0):
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
            self._flush_pending()

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
                        repr_version=self._move_stamp(),
                        action=(self._aligned_action(q_mean).detach()
                                if self._shs_action_dim > 0 else None),
                        rollout_action=self._rollout_action(q_mean),
                        episode_first=(None if episode_first is None
                                       else episode_first.detach()),
                        acq_time=self._last_acq_time, episode_uid=self._last_episode_uid)
                else:
                    self._move_buffer.add(
                        stoch.detach(), deter.detach(),
                        None if is_first is None else is_first.detach(),
                        step=self._shs_step,
                        batch_id=f"live{self._shs_step}",
                        repr_version=self._move_stamp(),
                        action=(self._aligned_action(q_mean).detach()
                                if self._shs_action_dim > 0 else None),
                        rollout_action=self._rollout_action(q_mean),
                        episode_first=(None if episode_first is None
                                       else episode_first.detach()),
                        acq_time=self._last_acq_time, episode_uid=self._last_episode_uid)
        _periodic = (self._shs_move_every > 0
                     and self._shs_step >= self._shs_move_warmup
                     and self._shs_step % self._shs_move_every == 0)
        _first_due = (getattr(self, "_move_first_pending", False)
                      and self._shs_move_every > 0
                      and self._shs_step >= self._move_first)
        _scheduled = _periodic or _first_due
        if _scheduled and self._live_sweep_allowed():
            if _first_due:
                print(f"[SHS] first structure sweep (shs_move_first={self._move_first}) "
                      f"@ grad-step {self._shs_step}")
            self._move_first_pending = False
            with torch.no_grad():
                _grow = self._move_K_cap is None or int(self.regime.K) < self._move_K_cap
                self._last_move_log = sweep_moves(
                    self.regime, buffer=self._move_buffer,
                    do_birth=self._shs_move_birth and _grow, do_split=self._shs_move_split and _grow,
                    threshold=self._move_threshold, create_bonus=self._move_create_bonus,
                    shrink_threshold=self._move_shrink_threshold,
                    confirm_top=self._shs_move_confirm_top,
                    delete_mode=self._shs_delete_mode,
                    delete_empty=self._shs_delete_empty,
                    holdout=self._shs_move_holdout,
                    holdout_score=self.holdout_predictive_score,
                    holdout_z=self._shs_move_holdout_z,
                    holdout_tol=self._shs_move_holdout_tol,
                    K_max=self._move_K_cap, commit=self._shs_move_commit,
                    merge_select=self._shs_merge_select,
                    merge_passes=self._shs_merge_passes,
                    birth_style=self._shs_birth_style,
                    lap=float(self._shs_step) / max(1, self._shs_move_every),
                    delete_topk=getattr(self, "_shs_delete_topk", 3),
                merge_topm=self._shs_merge_topm)
            if not self._last_move_log and getattr(self.regime, "_last_defer_reason", None):
                print(f"[SHS] sweep @ grad-step {self._shs_step} deferred: "
                      f"{self.regime._last_defer_reason}")
                self.regime._last_defer_reason = None
            accepted = {m: round(float(g), 3) for m, (ok, g) in self._last_move_log.items() if ok}
            if getattr(self, "_teacher", None) is not None:
                self._teacher.mark_move()
            if accepted:
                print(f"[SHS] move sweep @ grad-step {self._shs_step}: K={self.regime.K} "
                      f"accepted={accepted} installed_gain="
                      f"{float(getattr(self.regime, '_installed_gain', 0.0)):.3f}")
        elif _scheduled:
            if _periodic:
                self._n_live_sweeps_skipped = getattr(self, "_n_live_sweeps_skipped", 0) + 1
            elif not getattr(self, "_move_first_waiting", False):
                self._move_first_waiting = True
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
        post["regime_gen"] = self._gen_like(gamma)

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
            self._pending = (q_mean.detach(), deter.detach(), gamma.detach(),
                             counts.detach(), start_counts.detach(),
                             None if is_first is None else is_first.detach(),
                             (q_std ** 2).detach(),
                             self._next_batch_id,
                             int(self.regime.repr_version),
                             None if self._last_action is None else self._aligned_action(q_mean).detach(),
                             None if _pz is None else _pz.detach())
            self._next_batch_id = None

        return loss, value.detach(), dyn.detach(), rep.detach()

    @torch.no_grad()
    def _flush_pending(self):
        """Apply the deferred global update; re-run its E-step if the structure changed meanwhile."""
        p = self._pending
        self._pending = None
        if p is None:
            return
        with torch.amp.autocast("cuda", enabled=False):
            q_mean, deter, gamma, counts, start, is_first, z_var = p[:7]
            act = p[9] if len(p) > 9 else None
            pz = p[10] if len(p) > 10 else None
            epf = getattr(self, "_pending_episode_first", None)
            if int(gamma.shape[-1]) != int(self.regime.K):
                gamma, counts, start, _ = self.regime.regime_inference(
                    q_mean.float(), deter.float(), is_first, cache_estep=True,
                    z_var=None if z_var is None else z_var.float(),
                    prev_stoch=None if pz is None else pz.float(), action=act,
                    episode_first=epf)
                self._pending_recomputed = int(getattr(self, "_pending_recomputed", 0)) + 1
            self.regime.update_globals(q_mean, deter, gamma, counts, start, is_first, z_var,
                                       batch_id=p[7], action=act, prev_stoch=pz,
                                       episode_first=epf,
                                       repr_version=(p[8] if len(p) > 8 else None))

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
        post["regime_gen"] = self._gen_like(gamma)
        return post

    @torch.no_grad()
    def prior_entropy_bounds(self, state):
        lo = self.prior_entropy(state, mode="bounds")
        return lo, self._last_entropy_bounds[1]

    def set_batch_id(self, batch_id):
        self._next_batch_id = batch_id

    def get_extra_state(self):
        def cpu(o):
            if o is None:
                return None
            if torch.is_tensor(o):
                return o.detach().cpu()
            if isinstance(o, (tuple, list)):
                return tuple(cpu(v) for v in o)
            return o
        return dict(pending=cpu(self._pending),
                    pending_episode_first=cpu(getattr(self, "_pending_episode_first", None)),
                    next_batch_id=self._next_batch_id,
                    shs_step=int(self._shs_step),
                    move_first_pending=bool(getattr(self, "_move_first_pending", False)),
                    move_K_cap=getattr(self, "_move_K_cap", None),
                    b0_calibrated=bool(self._shs_b0_calibrated),
                    empty_removed=int(getattr(self.regime, "_n_empty_removed", 0)),
                    holdout_vetoes=int(getattr(self.regime, "_holdout_vetoes", 0)),
                    fit_vetoes=int(getattr(self.regime, "_fit_vetoes", 0)),
                    sweeps_deferred=int(getattr(self.regime, "_sweeps_deferred", 0)))

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
        self._pending_episode_first = mv(state.get("pending_episode_first"))
        self._next_batch_id = state.get("next_batch_id")
        self.regime._sweeps_deferred = int(state.get("sweeps_deferred", 0))
        self._shs_step = int(state.get("shs_step", 0))
        _first = getattr(self, "_move_first", None)
        self._move_first_pending = bool(state.get(
            "move_first_pending", _first is not None and self._shs_step < _first))
        cap = state.get("move_K_cap")
        if cap is not None and getattr(self, "_move_K_cap", None) is not None:
            self._move_K_cap = int(cap)
        if "b0_calibrated" in state:
            self._shs_b0_calibrated = bool(state["b0_calibrated"])
        self.regime._n_empty_removed = int(state.get("empty_removed", 0))
        self.regime._holdout_vetoes = int(state.get("holdout_vetoes", 0))
        self.regime._fit_vetoes = int(state.get("fit_vetoes", 0))

    def begin_full_batch_pass(self):
        self.regime.begin_full_batch_pass()

    def finalize_full_batch_pass(self):
        return self.regime.finalize_full_batch_pass()

    def _live_sweep_allowed(self) -> bool:
        buf = self._move_buffer
        if buf is None:
            return False
        if getattr(buf, "complete", False):
            return buf.is_complete()
        if self._shs_moves_complete:
            return False
        current = self._move_stamp()
        if current is None:
            return False
        try:
            dropped = buf.purge_stale(int(current))
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

    def live_state(self) -> dict:
        """Runtime SHS state that is not a tensor (saved beside the weights)."""
        return dict(recurrent=bool(self.regime.recurrent),
                    kappa=float(getattr(self.regime.hdp, "kappa", 0.0)),
                    b0_calibrated=bool(self._shs_b0_calibrated),
                    shs_step=int(self._shs_step), phase=int(self._curr_phase),
                    K=int(self.regime.K),
                    move_first_pending=bool(getattr(self, "_move_first_pending", False)),
                    move_K_cap=getattr(self, "_move_K_cap", None))

    def load_live_state(self, live, runtime=None, encoder=None):
        """Restore the non-tensor runtime saved by live_state() and the teacher, if any."""
        live = dict(live or {})
        runtime = dict(runtime or {})
        regime = self.regime
        if "shs_step" in live:
            self._shs_step = int(live["shs_step"])
        self._shs_b0_calibrated = bool(live.get("b0_calibrated",
                                                runtime.get("b0_calibrated", self._shs_b0_calibrated)))
        if "recurrent" in live and getattr(regime, "rstick", None) is not None:
            regime.recurrent = bool(live["recurrent"])
        if "kappa" in live:
            regime.hdp.kappa = 0.0 if regime.recurrent else float(live["kappa"])
        if "move_first_pending" in live:
            self._move_first_pending = bool(live["move_first_pending"])
        if live.get("move_K_cap") is not None and getattr(self, "_move_K_cap", None) is not None:
            self._move_K_cap = int(live["move_K_cap"])
        if self._curriculum:
            self._curr_phase = next((i for i, p in enumerate(self._curriculum)
                                     if int(p.get("until", -1)) < 0
                                     or self._shs_step < int(p["until"])),
                                    len(self._curriculum) - 1)
            self._apply_curriculum()
        teacher, probe = runtime.get("teacher"), runtime.get("teacher_probe")
        cfg = self._shs_teacher_cfg
        if encoder is not None and teacher is not None and probe is not None and cfg.get("enabled"):
            from .teacher import EMATeacher
            device = next(encoder.parameters()).device
            self._teacher = EMATeacher(encoder, tau=cfg["tau"], drift_tol=cfg["drift_tol"],
                                       probe={k: v.to(device) for k, v in probe.items()})
            self._teacher.load_state_dict(teacher)
        return self.live_state()

    def _move_stamp(self):
        """Representation stamp for live move-buffer batches (see shs_move_version)."""
        if getattr(self, "_shs_move_version", "repr") == "teacher":
            t = getattr(self, "_teacher", None)
            return None if t is None else int(t.version)
        return int(self.regime.repr_version)

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
        rr = self.aligned_resp(state)
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

    def _deter_step(self, prev_state, prev_action):
        x = torch.cat([prev_state["stoch"], prev_action], -1)
        x = self._img_in_layers(x)
        deter = prev_state["deter"]
        for _ in range(self._rec_depth):
            x, deter = self._cell(x, [deter]); deter = deter[0]
        return deter

    def _rollout_action(self, ref):
        act = self._last_action
        if act is None or act.shape[1] < ref.shape[1]:
            return None
        return act[:, : ref.shape[1]].detach()

    @torch.no_grad()
    def holdout_predictive_score(self, batches):
        H = max(1, int(self._shs_move_holdout_horizon))
        out = []
        with torch.amp.autocast("cuda", enabled=False):
            for b in batches:
                act = getattr(b, "rollout_action", None)
                stoch, deter = b.stoch.float(), b.deter.float()
                B, T, L = stoch.shape
                if act is None or T < H + 2:
                    continue
                act = act.float()
                zv = None if b.z_var is None else b.z_var.float()
                ef = b.episode_first if b.episode_first is not None else b.is_first
                ef = None if ef is None else ef.reshape(B, T).float()
                bact = getattr(b, "action", None)
                bval = getattr(b, "valid", None)
                cut = lambda x, t: None if x is None else x[:, :t + 1]
                n_starts = max(1, min(3, T - H - 1))
                starts = torch.linspace(1, T - H - 1, n_starts).round().long().unique().tolist()
                score = stoch.new_zeros(B)
                count = stoch.new_zeros(B)
                for t0 in starts:
                    belief = self.regime.regime_inference(
                        stoch[:, :t0 + 1], deter[:, :t0 + 1], cut(b.is_first, t0),
                        cache_estep=False, z_var=cut(zv, t0), action=cut(bact, t0),
                        valid=cut(bval, t0), episode_first=cut(b.episode_first, t0))[0]
                    belief = belief[:, -1].float()
                    ok = (stoch.new_ones(B) if ef is None
                          else (ef[:, t0 + 1:t0 + H + 1].sum(1) == 0).float())
                    std0 = (zv[:, t0].clamp_min(1e-8).sqrt() if zv is not None
                            else torch.zeros_like(stoch[:, t0]))
                    state = {"stoch": stoch[:, t0], "deter": deter[:, t0], "mean": stoch[:, t0],
                             "std": std0, "regime_resp": belief, "regime_gen": self._gen_like(belief)}
                    lp = stoch.new_zeros(B)
                    for h in range(1, H + 1):
                        state = self.img_step(state, act[:, t0 + h], sample=False)
                        mean = state["mean"].float()
                        std = state["std"].float().clamp_min(1e-4)
                        z = stoch[:, t0 + h]
                        lp = lp + (-0.5 * ((z - mean) / std).pow(2) - std.log()
                                   - 0.5 * math.log(2 * math.pi)).sum(-1)
                    score = score + ok * lp / H
                    count = count + ok
                keep = count > 0
                out.append(score[keep] / count[keep])
        if not out:
            return torch.zeros(0)
        return torch.cat(out)

    def img_step(self, prev_state, prev_action, sample=True):
        deter = self._deter_step(prev_state, prev_action)
        B = deter.shape[0]
        resp_prev = self.aligned_resp(prev_state)
        if resp_prev is None or resp_prev.shape[-1] != self.regime.K:
            resp_prev = deter.new_full((B, self.regime.K), 1.0 / self.regime.K)
        prev_var = None
        if "std" in prev_state and (not sample or self._shs_imag_propagate_var):
            prev_var = prev_state["std"].float().pow(2)
        z, resp, mean, std = self.regime.imagine_prior(
            prev_state["stoch"].float(), deter.float(), resp_prev.float(), sample=sample,
            prev_var=prev_var,
            action=(prev_action.float() if self._shs_action_dim > 0 else None),
            mode=getattr(self, "_shs_imag_mode", None),
        )
        dt = deter.dtype
        return {"stoch": z.to(dt), "deter": deter, "mean": mean.to(dt),
                "std": std.to(dt), "regime_resp": resp.to(dt), "regime_gen": self._gen_like(resp)}

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
        state["regime_gen"] = self._gen_like(state["regime_resp"])
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
        per_agent_step = (float(config.train_ratio)
                          / (float(config.batch_size) * float(config.batch_length)))
        unit = str(items.get("shs_schedule_unit", "agent"))
        if unit not in ("agent", "frame"):
            raise ValueError(f"shs_schedule_unit must be 'agent' or 'frame', got {unit!r}")
        if unit == "frame":
            per_agent_step /= float(getattr(config, "action_repeat", 1) or 1)
        items.setdefault("shs_grad_per_env", per_agent_step)
    except AttributeError:
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
    first = kw.get("shs_move_first")
    if isinstance(first, str):
        text = first.strip().lower()
        if text in ("", "none", "null"):
            first = None
        elif text == "pretrain":
            first = int(getattr(config, "pretrain", 0))
        else:
            first = int(float(text))
    kw["shs_move_first"] = first
    first_env = kw.get("shs_move_first_env")
    if isinstance(first_env, str):
        text = first_env.strip().lower()
        first_env = None if text in ("", "none", "null") else float(text)
    kw["shs_move_first_env"] = first_env
    headroom = kw.get("shs_move_K_headroom")
    if isinstance(headroom, str):
        text = headroom.strip().lower()
        headroom = None if text in ("", "none", "null") else int(float(text))
    kw["shs_move_K_headroom"] = headroom
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
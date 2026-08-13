from __future__ import annotations

import numpy as np
import torch

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
                 shs_moment_consistent: bool = True,
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
        super().__init__(*args, **kwargs)
        assert not self._discrete, "SHSRSSM supports continuous latents only"
        self.regime = RegimeHead(
            stoch=self._stoch, deter=self._deter, K=shs_K, proj_dim=shs_proj_dim,
            a0=shs_a0, b0=shs_b0, learn_b0=shs_learn_b0, b0_strength=shs_b0_strength, v0_scale=shs_v0_scale, ard=shs_ard,
            gamma=shs_gamma, alpha=shs_alpha, kappa=shs_kappa,
            start_alpha=shs_start_alpha, ema_tau=shs_ema_tau, hdp_iters=shs_hdp_iters,
            recurrent=shs_recurrent, prior_persist=shs_prior_persist, pg_iters=shs_pg_iters,
            rstick_dim=shs_rstick_dim, rstick_stopgrad=shs_rstick_stopgrad,
            rstick_weight_var=shs_rstick_weight_var, rstick_use_action=config.shs_rstick_use_action,
            q_rank=shs_q_rank, shared_carry=shs_shared_carry,
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

    def observe(self, embed, action, is_first, state=None, sample=True):
        self._last_is_first = is_first
        if self._shs_action_dim > 0:
            self._last_action = action.detach()
        return super().observe(embed, action, is_first, state, sample=sample)

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
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
        return post, prior

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
                  f"create_bonus={self._move_create_bonus}")

    def curriculum_state(self) -> dict:
        out = {
            "shs_curriculum_phase": float(self._curr_phase),
            "shs_move_every_live": float(self._shs_move_every),
            "shs_kappa_live": float(getattr(self.regime.hdp, "kappa", 0.0)),
            "shs_move_threshold_live": float(self._move_threshold),
            "shs_create_bonus_live": float(self._move_create_bonus),
            "shs_recurrent_live": float(bool(self.regime.recurrent)),
        }
        out["shs_guard_rejects_hdp"] = float(getattr(self.regime.hdp, "n_guard_rejects", 0))
        out["shs_guard_rejects_pg"] = float(getattr(self.regime.rstick, "n_pg_guard_rejects", 0))
        _st = getattr(self.regime, "stat_store", None)
        out["shs_stale_invalidated"] = float(getattr(_st, "n_stale_invalidated", 0) if _st is not None else 0)
        out["shs_live_sweeps_skipped_stale"] = float(getattr(self, "_n_live_sweeps_skipped", 0))
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
                                           repr_version=_rv)
            self._pending = None

        q_mean = post["mean"].float()
        q_std = post["std"].float()
        is_first = self._last_is_first
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
                                if self._shs_action_dim > 0 else None))
                else:
                    self._move_buffer.add(
                        stoch.detach(), deter.detach(),
                        None if is_first is None else is_first.detach(),
                        step=self._shs_step,
                        batch_id=f"live{self._shs_step}",
                        repr_version=int(self.regime.repr_version),
                        action=(self._aligned_action(q_mean).detach()
                                if self._shs_action_dim > 0 else None))
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
                    lap=float(self._shs_step) / max(1, self._shs_move_every),
                    delete_topk=getattr(self, "_shs_delete_topk", 3),
                merge_topm=self._shs_merge_topm)
            accepted = {m: round(float(g), 3) for m, (ok, g) in self._last_move_log.items() if ok}
            if accepted:
                print(f"[SHS] move sweep @ grad-step {self._shs_step}: K={self.regime.K} "
                      f"accepted={accepted}")
        elif _scheduled:
            self._n_live_sweeps_skipped = getattr(self, "_n_live_sweeps_skipped", 0) + 1
            self._last_move_log = {}

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            gamma, counts, start_counts, _ = self.regime.regime_inference(
                q_mean.float(), deter.float(), is_first, cache_estep=True,
                z_var=(q_std ** 2).float(),
                prev_stoch=None if _pz is None else _pz.float(),
                action=self._aligned_action(q_mean))
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
        )

        if (self._shs_step % self._shs_update_every == 0
                and self._shs_global_source != "completed_episode_stream"):
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
        enough = len(buf.batches) >= int(getattr(self, "_shs_min_live_batches", 4))
        store = getattr(self.regime, "stat_store", None)
        ema_mode = (store is None) or (store.mode == "legacy_ema")
        return enough and ema_mode

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
            phi = R.build_stick_phi(deter, _act)
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

    def img_step(self, prev_state, prev_action, sample=True):
        deter = self._deter_step(prev_state, prev_action)
        B = deter.shape[0]
        resp_prev = prev_state.get("regime_resp", None)
        if resp_prev is not None and resp_prev.shape[-1] != self.regime.K:
            resp_prev = self._map_regime_resp(resp_prev)
        if resp_prev is None or resp_prev.shape[-1] != self.regime.K:
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


def shs_kwargs_from_config(config):
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
    if int(kw.get("shs_action_dim", 0)) == -1:
        kw["shs_action_dim"] = int(getattr(config, "num_actions", 0))
    return kw

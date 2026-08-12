import copy
import numpy as np
import torch
from torch import nn

import networks
import tools
from shs_rssm import shs_diagnostics as D
import pathlib
from shs_rssm.shs_filmstrip import plot_regime_filmstrip
to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        _imode = str(getattr(config, "shs_inference_mode", "filtered"))
        if _imode != "filtered":
            raise NotImplementedError(
                f"shs_inference_mode='{_imode}' is NOT wired into the training loop. "
                "The structured-SVAE components exist and are unit-tested "
                "(RegimeHead.svae_estep / svae_local_bound, continuous_smoother."
                "sample_chain), but nothing calls them: there is no recognition-potential "
                "head, the world-model heads still consume the filtered posterior, and "
                "dynamics_kl has not been replaced by the structured objective. Running "
                "with this flag would silently produce FILTERED results under an SVAE "
                "label. Use shs_inference_mode='filtered' until those are implemented.")
        if getattr(config, "use_shs", False):
            assert config.dyn_discrete == 0, (
                "use_shs requires dyn_discrete: 0 (SHS-RSSM is continuous-latent only)"
            )
            from shs_rssm.shs_rssm import SHSRSSM, shs_kwargs_from_config
            if config.shs_move_every > 0:
                print(f"[SHS] adaptive-K moves ON: shs_move_every={config.shs_move_every}, "
                      f"warmup={config.shs_move_warmup}, birth={config.shs_move_birth}.")
            WorldModel._normalize_and_validate_shs(config)
            self.dynamics = SHSRSSM(
                config.dyn_stoch,
                config.dyn_deter,
                config.dyn_hidden,
                config.dyn_rec_depth,
                config.dyn_discrete,
                config.act,
                config.norm,
                config.dyn_mean_act,
                config.dyn_std_act,
                config.dyn_min_std,
                config.unimix_ratio,
                config.initial,
                config.num_actions,
                self.embed_size,
                config.device,
                **shs_kwargs_from_config(config),
            )
        else:
            self.dynamics = networks.RSSM(
                config.dyn_stoch,
                config.dyn_deter,
                config.dyn_hidden,
                config.dyn_rec_depth,
                config.dyn_discrete,
                config.act,
                config.norm,
                config.dyn_mean_act,
                config.dyn_std_act,
                config.dyn_min_std,
                config.unimix_ratio,
                config.initial,
                config.num_actions,
                self.embed_size,
                config.device,
            )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    def _train(self, data):
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())
        if hasattr(self.dynamics, "bump_repr_version"):
            self.dynamics.bump_repr_version()

        metrics.update(
            {f"{name}_loss": to_np(torch.mean(loss)) for name, loss in losses.items()}
        )
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(torch.mean(dyn_loss))
        metrics["rep_loss"] = to_np(torch.mean(rep_loss))
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    def preprocess(self, obs):
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        obs["image"] = obs["image"] / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            obs["discount"] = obs["discount"].unsqueeze(-1)
        assert "is_first" in obs
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        if hasattr(self.dynamics, "annotate_regime_resp"):
            self.dynamics.annotate_regime_resp(states, data["is_first"][:6, :5],
                                              action=(data["action"][:6, :5] if getattr(self.dynamics, "_shs_action_dim", 0) > 0 else None))
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)

    @torch.no_grad()
    def shs_diagnostics(self, data, step, logdir, context=5, make_figures=True):
        if not hasattr(self.dynamics, "annotate_regime_resp"):
            return {}
        import pathlib
        data = self.preprocess(data)
        embed = self.encoder(data)
        B = min(6, embed.shape[0])
        T = embed.shape[1]
        ctx = min(context, T)

        states, prior = self.dynamics.observe(
            embed[:B, :ctx], data["action"][:B, :ctx], data["is_first"][:B, :ctx]
        )
        self.dynamics.annotate_regime_resp(states, data["is_first"][:B, :ctx],
                                          action=(data["action"][:B, :ctx] if getattr(self.dynamics, "_shs_action_dim", 0) > 0 else None))
        gamma = states["regime_resp"].float()
        stoch = states["stoch"].float()

        occ = gamma.reshape(-1, gamma.shape[-1]).sum(0)
        occ = occ / occ.sum().clamp_min(1e-12)
        entropy = float(-(occ.clamp_min(1e-12) * occ.clamp_min(1e-12).log()).sum())
        active = int((occ > 0.01).sum().item())
        Epi = self.dynamics.regime._Epi()
        self_trans = float(torch.diagonal(Epi).mean())
        metrics = {
            "shs_active_regimes": float(active),
            "shs_current_K": float(self.dynamics.regime.K),
            "shs_regime_entropy": entropy,
            "shs_expected_self_transition": self_trans,
        }
        prior["is_first"] = data["is_first"][:B, :ctx]
        if hasattr(self.dynamics, "prior_entropy_bounds"):
            _ent_lo, _ent_hi = self.dynamics.prior_entropy_bounds(prior)
            metrics["shs_prior_mixture_ent"] = float(_ent_lo.mean())
            metrics["shs_prior_mixture_ent_ub"] = float(_ent_hi.mean())
        else:
            metrics["shs_prior_mixture_ent"] = float(self.dynamics.prior_entropy(prior).mean())
        if hasattr(self.dynamics, "curriculum_state"):
            metrics.update(self.dynamics.curriculum_state())

        if not make_figures:
            return metrics

        try:
            from shs_rssm import shs_diagnostics as D
        except Exception as e:
            print(f"[SHS] diagnostics figures skipped ({type(e).__name__}: {e})")
            return metrics
        out = pathlib.Path(logdir) / "shs_diagnostics" / f"step_{int(step)}"
        out.mkdir(parents=True, exist_ok=True)

        D.plot_latent_clustering(
            stoch, gamma, str(out / "latent_clustering.png"),
            title=f"SHS-RSSM regimes @ step {int(step)}")

        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:B, ctx:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        model0 = torch.cat([recon[0], openl[0]], 0)
        true0 = data["image"][0, :model0.shape[0]]
        D.plot_reconstructions(
            true0, model0, str(out / "reconstruction.png"),
            context=ctx, title=f"SHS-RSSM reconstruction @ step {int(step)}")
        return metrics

    @torch.no_grad()
    def shs_regime_filmstrip(self, data, step, logdir, episode=0, window=None):
        if not hasattr(self.dynamics, "annotate_regime_resp"):
            return {}


        data = self.preprocess(data)
        b = int(episode)
        T = data["image"].shape[1] if window is None else min(window, data["image"].shape[1])
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[b:b + 1, :T], data["action"][b:b + 1, :T], data["is_first"][b:b + 1, :T])
        self.dynamics.annotate_regime_resp(states, data["is_first"][b:b + 1, :T],
                                          action=(data["action"][b:b + 1, :T] if getattr(self.dynamics, "_shs_action_dim", 0) > 0 else None))

        gamma = states["regime_resp"][0].float()
        z = states["mean"][0].float() if "mean" in states else states["stoch"][0].float()
        frames = data["image"][b, :T]

        out = pathlib.Path(logdir) / "shs_diagnostics" / f"step_{int(step)}"
        out.mkdir(parents=True, exist_ok=True)
        try:
            stats = plot_regime_filmstrip(
                frames, gamma, str(out / "regime_filmstrip.png"), z=z,
                max_frames=12, min_dwell=2,
                title=f"SHS regime filmstrip @ step {int(step)}")
        except Exception as e:
            print(f"[SHS] regime filmstrip skipped (non-fatal): {type(e).__name__}: {e}")
            return {}
        return {f"shs_filmstrip_{k}": float(v) for k, v in stats.items()}

    @staticmethod
    def _extract_shape_factors(data):
        if "factor_n_present" not in data:
            return None

        def to(v):
            arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            return arr.reshape(arr.shape[0], arr.shape[1])
        n_present = to(data["factor_n_present"])
        per = {"alpha": [], "present": [], "x": [], "y": []}
        i = 0
        while f"factor_shape{i}_alpha" in data:
            for key in per:
                per[key].append(to(data[f"factor_shape{i}_{key}"]))
            i += 1
        factors = {"n_present": n_present}
        for key, cols in per.items():
            if cols:
                factors[key] = np.stack(cols, axis=-1)
        return factors

    @torch.no_grad()
    def shs_disentangle_diagnostics(self, data, step, logdir, make_figures=True):
        if not hasattr(self.dynamics, "annotate_regime_resp"):
            return {}
        factors = self._extract_shape_factors(data)
        if factors is None:
            return {}
        import pathlib
        data = self.preprocess(data)
        embed = self.encoder(data)
        states, _ = self.dynamics.observe(embed, data["action"], data["is_first"])
        self.dynamics.annotate_regime_resp(states, data["is_first"],
                                          action=(data["action"] if getattr(self.dynamics, "_shs_action_dim", 0) > 0 else None))
        stoch = states["stoch"].float()
        gamma = states["regime_resp"].float()
        f = {k: torch.as_tensor(v) for k, v in factors.items()}

        from shs_rssm import shs_disentangle as DZ
        align = DZ.regime_factor_alignment(gamma, f)
        decode = DZ.factor_decodability(stoch, f, key="n_present")
        _, _, mig = DZ.latent_factor_mi_matrix(stoch, f, keys=["n_present", "present"])
        metrics = {
            "shs_composition_decodability": decode["accuracy"],
            "shs_composition_decode_chance": decode["baseline"],
            "shs_latent_factor_mig": mig,
            "shs_regime_composition_nmi": align["regime_composition_nmi"],
            "shs_boundary_f1": align["boundary_f1"],
        }
        if make_figures:
            out = pathlib.Path(logdir) / "shs_diagnostics" / f"step_{int(step)}"
            out.mkdir(parents=True, exist_ok=True)
            DZ.plot_tsne_disentangle(
                stoch, gamma, f, str(out / "tsne_disentangle.png"),
                factor_key="n_present", title=f"SHS-RSSM latents @ step {int(step)}")
            DZ.plot_mi_matrix(stoch, f, str(out / "latent_factor_mi.png"),
                              keys=["n_present", "present", "x", "y"])
        return metrics

    @torch.no_grad()
    @staticmethod
    def _normalize_and_validate_shs(config):
        alias = {"episode_stream": "completed_episode_stream",
                 "stream": "completed_episode_stream", "completed": "completed_episode_stream",
                 "replay": "replay_ema", "ema": "replay_ema",
                 "memoized": "offline_memoized", "offline": "offline_memoized"}
        gs = alias.get(getattr(config, "shs_global_source", "replay_ema"),
                       getattr(config, "shs_global_source", "replay_ema"))
        valid = {"replay_ema", "completed_episode_stream", "offline_memoized"}
        if gs not in valid:
            raise ValueError(
                f"unknown shs_global_source; expected one of {sorted(valid)} "
                "(an unknown source would silently disable global updates).")
        config.shs_global_source = gs
        need = {"replay_ema": "ema", "completed_episode_stream": "streaming",
                "offline_memoized": "memoized"}[gs]
        om = getattr(config, "shs_online_mode", "ema")
        if om != need:
            raise ValueError(
                f"shs_global_source={gs!r} requires shs_online_mode={need!r} but got {om!r} "
                "( e.g. completed_episode_stream+ema silently produces NO global "
                "updates). Pair them: replay_ema<->ema, completed_episode_stream<->streaming, "
                "offline_memoized<->memoized.")
        stream_flag = bool(getattr(config, "shs_stream_episodes", False))
        if gs == "completed_episode_stream" and not stream_flag:
            raise ValueError(
                "shs_global_source='completed_episode_stream' requires "
                "shs_stream_episodes=True; otherwise no episode is ever ingested and "
                "the switching globals silently freeze.")
        if stream_flag and gs != "completed_episode_stream":
            raise ValueError(
                "shs_stream_episodes=True requires "
                "shs_global_source='completed_episode_stream' (mode 'streaming'); "
                f"with {gs!r} the queue would drain and commit episodes that "
                "stream_completed_episodes() ignores.")
        return gs

    @staticmethod
    def _tensor_snapshot(module):
        snap = {n: p.detach().clone() for n, p in module.named_parameters()}
        snap.update({n: b.detach().clone() for n, b in module.named_buffers()})
        return snap

    @staticmethod
    def _tensor_snapshot_params(module):
        return {n: p.detach().clone() for n, p in module.named_parameters()}

    def _shs_swap_target_in(self):
        tgt = getattr(self, "_shs_target_state", None)
        if tgt is None:
            return None
        return {"encoder": self._tensor_swap_in(self.encoder, tgt["encoder"]),
                "dynamics": self._tensor_swap_in(self.dynamics, tgt["dynamics"])}

    def _shs_swap_target_out(self, saved):
        if saved is None:
            return
        self._tensor_restore(self.encoder, saved["encoder"])
        self._tensor_restore(self.dynamics, saved["dynamics"])

    @staticmethod
    def _tensor_swap_in(module, snap):
        live = dict(module.named_parameters()); live.update(dict(module.named_buffers()))
        saved = {}
        for n, t in snap.items():
            if n in live:
                saved[n] = live[n].detach().clone()
                live[n].data.copy_(t)
        return saved

    @staticmethod
    def _tensor_restore(module, saved):
        live = dict(module.named_parameters()); live.update(dict(module.named_buffers()))
        for n, t in saved.items():
            if n in live:
                live[n].data.copy_(t)

    def begin_shs_repr_epoch(self):
        if not hasattr(self.dynamics, "regime") or getattr(self, "_shs_target_state", None) is not None:
            return
        self._shs_target_state = {
            "encoder": self._tensor_snapshot(self.encoder),
            "dynamics": self._tensor_snapshot_params(self.dynamics),
        }
        self.dynamics.regime.begin_repr_epoch()

    def end_shs_repr_epoch(self):
        if not hasattr(self.dynamics, "regime") or getattr(self, "_shs_target_state", None) is None:
            return
        self._shs_target_state = None
        self.dynamics.regime.end_repr_epoch()

    def get_extra_state(self):
        res = getattr(self, "_shs_reservoir", None) or []
        keep = bool(getattr(getattr(self, "_config", None), "shs_ckpt_reservoir", False))
        return {"shs_reservoir": (res if keep else None),
                "shs_reservoir_len": int(len(res)),
                "shs_reservoir_persisted": keep}

    def set_extra_state(self, state):
        st = state or {}
        res = st.get("shs_reservoir", None)
        if res:
            self._shs_reservoir = list(res)
        else:
            self._shs_reservoir = []
            n = int(st.get("shs_reservoir_len", 0))
            if n:
                print(f"[SHS] checkpoint carried {n} reservoir episodes but "
                      "shs_ckpt_reservoir=False, so they were NOT saved. The next "
                      "representation refresh will discard accumulated statistics; "
                      "history restarts from newly observed episodes only.")

    def _shs_reservoir_add(self, data_batches, max_keep=64):
        if not hasattr(self, "_shs_reservoir"):
            self._shs_reservoir = []
        self._shs_reservoir.extend(data_batches)
        if len(self._shs_reservoir) > max_keep:
            self._shs_reservoir = self._shs_reservoir[-max_keep:]

    def rebuild_stats_from_reservoir(self):
        res = getattr(self, "_shs_reservoir", None)
        head = self.dynamics.regime
        store = getattr(head, "stat_store", None)
        if store is None:
            return 0
        if not res:
            store.reset()
            head._episode_cursor = 0
            print("[SHS] representation refresh with an EMPTY reservoir: accumulated "
                  "sufficient statistics have been DISCARDED (they cannot be re-expressed "
                  "in the new latent coordinates). Streaming restarts from zero history. "
                  "This is expected immediately after a checkpoint resume.")
            return 0
        store.reset()
        head._episode_cursor = 0
        return int(self.stream_completed_episodes(list(res)).get("shs_stream_episodes", 0))

    def refresh_shs_repr_epoch(self):
        self.end_shs_repr_epoch()
        self.begin_shs_repr_epoch()
        self.rebuild_stats_from_reservoir()

    def _shs_target_encode(self, data, state=None):
        with torch.no_grad():
            embed = self.encoder(data)
            post, _ = self.dynamics.observe(embed, data["action"], data["is_first"],
                                            state=state, sample=False)
        nxt = {k: v[:, -1].detach() for k, v in post.items()
               if torch.is_tensor(v) and v.dim() >= 2}
        return post, nxt

    def stream_completed_episodes(self, data_batches, episode_ids=None):
        if not hasattr(self.dynamics, "regime"):
            return {}
        head = self.dynamics.regime
        if getattr(head, "stat_store", None) is None or head.stat_store.mode != "streaming":
            return {}
        dyn = self.dynamics
        analytic = bool(getattr(dyn, "_shs_analytic_estep", False))
        n, elbo_sum = 0, 0.0
        _own_epoch = not getattr(head, "_repr_frozen", False)
        if _own_epoch:
            head.begin_repr_epoch()
        _snap = head.master_snapshot()
        _swapped = self._shs_swap_target_in()
        _use_ids = (episode_ids is not None
                    and hasattr(data_batches, "__len__")
                    and len(episode_ids) == len(data_batches)
                    and not bool(getattr(head.stat_store, "strict_stream", False)))
        try:
          for _bi, data in enumerate(data_batches):
            _eid = episode_ids[_bi] if _use_ids else None
            if isinstance(data, (list, tuple)):
                _cst, _cdicts = None, []
                for _ck in data:
                    _ck = self.preprocess(_ck)
                    _cp, _cst = self._shs_target_encode(_ck, state=_cst)
                    _cdicts.append(dict(
                        stoch=(_cp["mean"].float() if (analytic and "mean" in _cp)
                               else _cp["stoch"].float()),
                        deter=_cp["deter"].float(),
                        is_first=_ck["is_first"].float(),
                        z_var=((_cp["std"].float() ** 2)
                               if (analytic and "std" in _cp) else None),
                        action=(_ck["action"].float()
                                if int(getattr(dyn, "_shs_action_dim", 0)) > 0
                                else None)))
                diag = head.stream_episode_chunks(_cdicts, episode_id=_eid)
            else:
                data = self.preprocess(data)
                post, _ = self._shs_target_encode(data)
                isf = data["is_first"].float()
                act = data["action"].float() if int(getattr(dyn, "_shs_action_dim", 0)) > 0 else None
                zvar = (post["std"].float() ** 2) if (analytic and "std" in post) else None
                stoch = post["mean"].float() if (analytic and "mean" in post) else post["stoch"].float()
                _aid = (episode_ids[_bi] if (episode_ids is not None
                        and hasattr(data_batches, "__len__")
                        and len(episode_ids) == len(data_batches)) else None)
                diag = head.stream_episode(stoch, post["deter"].float(), is_first=isf,
                                           z_var=zvar, action=act, episode_id=_eid,
                                           audit_id=_aid)
            _e = float(diag.get("elbo", 0.0))
            import math as _math
            if not _math.isfinite(_e):
                raise FloatingPointError("non-finite ELBO during streaming ingestion")
            elbo_sum += _e; n += 1
        except Exception:
            head.load_snapshot(_snap)
            raise
        finally:
            self._shs_swap_target_out(_swapped)
            if _own_epoch:
                head.end_repr_epoch()
        return {"shs_stream_episodes": float(n),
                "shs_stream_cursor": float(getattr(head, "_episode_cursor", 0)),
                "shs_stream_elbo_mean": float(elbo_sum / max(1, n))}

    def consolidate_regimes(self, data_batches, n_sweeps: int = 3):
        if not hasattr(self.dynamics, "regime"):
            return {}
        from shs_rssm.moves import MoveBuffer, sweep_moves
        head = self.dynamics.regime
        dyn = self.dynamics
        analytic = bool(getattr(dyn, "_shs_analytic_estep", False))
        if hasattr(dyn, "bump_repr_version"):
            dyn.bump_repr_version()
        rv = int(head.repr_version)
        buf = MoveBuffer(max_batches=len(data_batches) + 1,
                         complete=True, expected_batches=len(data_batches))
        for ci, data in enumerate(data_batches):
            data = self.preprocess(data)
            embed = self.encoder(data)
            post, _ = self.dynamics.observe(embed, data["action"], data["is_first"], sample=False)
            isf = data["is_first"].float()
            _act = (data["action"].float()
                    if int(getattr(self.dynamics, "_shs_action_dim", 0)) > 0 else None)
            if analytic and "mean" in post:
                buf.add(post["mean"].float(), post["deter"].float(), isf,
                        post["std"].float() ** 2, step=0,
                        batch_id=f"consol{ci}", repr_version=rv, action=_act)
            else:
                buf.add(post["stoch"].float(), post["deter"].float(), isf, step=0,
                        batch_id=f"consol{ci}", repr_version=rv, action=_act)
        K_before = head.K
        for sweep_i in range(max(1, n_sweeps)):
            sweep_moves(head, buffer=buf, lap=float(sweep_i),
                        do_birth=getattr(dyn, "_shs_move_birth", True),
                        do_split=getattr(dyn, "_shs_move_split", True),
                        confirm_top=getattr(dyn, "_shs_move_confirm_top", 8),
                        delete_mode=getattr(dyn, "_shs_delete_mode", "hughes"),
                        merge_select=getattr(dyn, "_shs_merge_select", "hughes"),
                        merge_passes=getattr(dyn, "_shs_merge_passes", 12),
                        birth_style=getattr(dyn, "_shs_birth_style", "seqcreate"),
                        birth_proposals=getattr(dyn, "_shs_birth_proposals", 5))
        return {
            "shs_consolidate_K_before": float(K_before),
            "shs_consolidate_K_after": float(head.K),
            "shs_consolidate_active_regimes": float(int((head.regimes.N > 1.0).sum())),
        }

    def save_latent_snapshot(self, step, stoch, gamma, factors, max_keep=5):
        if not hasattr(self, "_shs_snapshots"):
            self._shs_snapshots = []
        b = min(4, stoch.shape[0])
        snap = {
            "step": int(step),
            "label": f"step {int(step)}",
            "stoch": stoch[:b].detach().cpu(),
            "gamma": gamma[:b].detach().cpu(),
            "factors": {k: (v[:b] if hasattr(v, "__getitem__") else v)
                        for k, v in factors.items()},
        }
        self._shs_snapshots.append(snap)
        while len(self._shs_snapshots) > max_keep:
            self._shs_snapshots.pop(0)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.device)
            )
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective,
    ):
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics
        _R = getattr(dynamics, "regime", None)
        if (_R is not None
                and getattr(dynamics, "_shs_imag_mode", "") == "reinforce_sample"):
            _R._imag_logprob_trace = []
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])
        else:
            adv = target - base

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        _R = getattr(self._world_model.dynamics, "regime", None)
        _tr = getattr(_R, "_imag_logprob_trace", None) if _R is not None else None
        if _tr:
            _lp = torch.stack(_tr, 0)
            _n = policy.log_prob(imag_action).shape[0]
            if _lp.shape[0] > _n:
                _lp = _lp[:_n]
            actor_target = actor_target + (
                _lp[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach())
            metrics["regime_reinforce_lp"] = to_np(torch.mean(_lp))
            _R._imag_logprob_trace = None
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1

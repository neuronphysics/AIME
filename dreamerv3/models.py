import copy
import torch
from torch import nn
from torch.distributions import kl_divergence, Normal, Independent, Categorical
import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
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
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.amp.autocast("cuda", enabled=self._use_amp):
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

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.amp.autocast("cuda", enabled=self._use_amp):
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

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        obs["image"] = obs["image"] / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


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
            # register ema_vals to nn.Module for enabling torch.save and torch.load
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
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled by ema or sym_log.
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
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
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
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

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
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1

class LatentAGACBehavior(ImagBehavior):
    """
    Dreamer-style behavior class with AGAC-like exploration in latent space.

    Modes:
      agac.kl_mode = "none"
        -> identical to ImagBehavior (no AGAC terms)

      agac.kl_mode = "recursive"
        -> Design B:
           - critic sees KL-shaped reward r_t + beta_kl * K_t
           - actor gets standard Dreamer loss + extra REINFORCE term
             with advantage bonus beta_delta * delta_t

      agac.kl_mode = "actor_horizon"
        -> Design A':
           - critic stays purely extrinsic
           - actor gets standard Dreamer loss + extra REINFORCE term
             with horizon KL bonus beta_kl * G_t^KL

    Old policy options:
      agac.old_policy_mode = "ema"
        -> EMA lagged actor/adversary 

      agac.old_policy_mode = "snapshot"
        -> snapshot actor/adversary once per _train() call
           (closer to strict PPO-style use of π_old)
    """

    def __init__(self, config, world_model):
        # Initialize ImagBehavior (actor, value, slow_value, ema, etc.)
        super().__init__(config, world_model)
        self._agac_cfg = getattr(config, "agac", None)
        
        def _cfg_get(obj, key, default):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        if self._agac_cfg is None:
            # No AGAC section: fall back to defaults
            self._agac_cfg = {}


        # KL framework hyperparameters
        self._kl_mode = _cfg_get(self._agac_cfg, "kl_mode", "none")
        self._beta_kl = _cfg_get(self._agac_cfg, "beta_kl", 0.0)
        self._beta_delta = _cfg_get(self._agac_cfg, "beta_delta", 0.0)
        self._gamma_kl = _cfg_get(self._agac_cfg, "gamma_kl", config.discount)

        # Old-policy handling (for KL and adversary)
        self._old_policy_mode = _cfg_get(self._agac_cfg, "old_policy_mode", "ema")
        assert self._old_policy_mode in ["ema", "snapshot"]
        self._slow_actor_fraction = _cfg_get(self._agac_cfg, "slow_actor_fraction", 0.01)
        self._slow_actor_update = _cfg_get(self._agac_cfg, "slow_actor_update", 1)
        print("[AGAC] kl_mode:", self._kl_mode, "beta_kl:", self._beta_kl, "beta_delta:", self._beta_delta, "old_policy_mode:", self._old_policy_mode)

        # Latent feature size (same as ImagBehavior)
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter

        # Adversary policy in latent space (same action dist as actor)
        self.adversary = networks.MLP(
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
            name="Adversary",
        )

        # Optimizer for adversary (reuse actor optimizer settings)
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._adv_opt = tools.Optimizer(
            "adversary",
            self.adversary.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer adversary_opt has "
            f"{sum(p.numel() for p in self.adversary.parameters())} variables."
        )

        # Slow (old) copies of actor and adversary
        self._slow_actor = copy.deepcopy(self.actor)
        self._slow_adversary = copy.deepcopy(self.adversary)
        self._agac_updates = 0  # counter for EMA updates

    # ------------------------------------------------------------------
    # Slow targets: critic (from ImagBehavior) + old policies (AGAC)
    # ------------------------------------------------------------------
    def _update_slow_target(self):
        """
        Called once per behavior update.

        1) Keeps ImagBehavior behavior for slow critic.
        2) Additionally maintains slow actor/adversary if in EMA mode.
        """
        # 1) slow critic (ImagBehavior logic)
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1

        # 2) EMA old policies
        if self._old_policy_mode == "ema":
            self._update_slow_policies_ema()

    def _update_slow_policies_ema(self):
        """EMA update for _slow_actor and _slow_adversary."""
        mix = self._slow_actor_fraction
        if self._agac_updates % self._slow_actor_update == 0:
            for s, d in zip(self.actor.parameters(), self._slow_actor.parameters()):
                d.data.mul_(1.0 - mix).add_(mix * s.data)
            for s, d in zip(self.adversary.parameters(), self._slow_adversary.parameters()):
                d.data.mul_(1.0 - mix).add_(mix * s.data)
        self._agac_updates += 1

    def _snapshot_policies(self):
        """
        Snapshot current actor/adversary into _slow_*.

        This is closer to strict PPO/AGAC semantics:
        freeze π_old once per batch, and use it for KL and adversary loss.
        """
        self._slow_actor.load_state_dict(self.actor.state_dict())
        self._slow_adversary.load_state_dict(self.adversary.state_dict())

    # AGAC-specific helpers
    def _compute_kl_terms(self, imag_feat, imag_action):
        """
        Compute per-step KL and log-ratio δ_t between *slow* actor and adversary:

          K_t    = D_KL(pi_slow(·|s_t) || adv_slow(·|s_t))
          δ_t    = log pi_slow(a_t|s_t) − log adv_slow(a_t|s_t)

        Uses no gradients (π_old, adv_old are treated as fixed).
        """
        with torch.no_grad():
            pi = self._slow_actor(imag_feat)
            adv = self._slow_adversary(imag_feat)

            a_sample = imag_action.detach()
            logp_pi = pi.log_prob(a_sample)
            logp_adv = adv.log_prob(a_sample)
            # shape (time, batch, 1)
            delta_t = (logp_pi - logp_adv)[..., None]
            inner_pi  = pi._dist           # ContDist._dist
            inner_adv = adv._dist

            # state-wise KL
            if hasattr(pi, "logits") and hasattr(adv, "logits"):
                old_pi_distribution = Categorical(logits=pi.logits)
                adv_old_pi_distribution = Categorical(logits=adv.logits)
                K_raw = kl_divergence(old_pi_distribution, adv_old_pi_distribution)  # (T,B)
                K_t = K_raw[..., None]
                
            elif isinstance(inner_pi, Independent) and isinstance(inner_pi.base_dist, Normal):
                base_pi  = inner_pi.base_dist
                base_adv = inner_adv.base_dist
                # KL over action dimensions, keep (T,B,1)
                K_t_raw = kl_divergence(base_pi, base_adv)       # (T,B,action_dim)
                K_t = K_t_raw.sum(-1, keepdim=True)              # (T,B,1)

        return K_t, delta_t

    def _horizon_kl_return(self, e_t, discount):
        """
        Discount-aware horizon KL:

          G_t^KL = Σ_{n>=0} (γ_kl * discount_{t+n})^n * e_{t+n}

        Args:
          e_t:       (T, B, 1) per-step KL signal K_t
          discount:  (T, B, 1) continuation discount from world model
        """
        T = e_t.shape[0]
        G = torch.zeros_like(e_t)
        running = torch.zeros_like(e_t[-1])
        for t in reversed(range(T)):
            running = e_t[t] + self._gamma_kl * discount[t] * running
            G[t] = running
        return G

    # Main training: extends ImagBehavior._train with AGAC bonuses
    def _train(
        self,
        start,
        objective,
    ):
        """
        start: posterior states from WorldModel._train.
        objective: function(imag_feat, imag_state, imag_action) -> reward_t.

        Behavior:
          - Always calls ImagBehavior._compute_target and _compute_actor_loss
            as the base Dreamer actor-critic.
          - Adds AGAC-style extra policy gradients on top:
              * recursive: log π * beta_delta * δ_t
              * actor_horizon: log π * beta_kl * G_t^KL
          - Uses slow actor/adversary for KL and δ_t.
        """
        # 1) slow critic + (if EMA) slow policies
        self._update_slow_target()
        # 2) snapshot old policies if requested
        if self._old_policy_mode == "snapshot":
            self._snapshot_policies()

        metrics = {}

        # Actor (Dreamer + AGAC bonus) 
        with tools.RequiresGrad(self.actor), tools.RequiresGrad(self.adversary):
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                # Imagine rollout in latent space
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()

                # For actor_horizon we need discount explicitly
                if self._kl_mode == "actor_horizon":
                    if "cont" in self._world_model.heads:
                        inp = self._world_model.dynamics.get_feat(imag_state)
                        discount = (
                            self._config.discount
                            * self._world_model.heads["cont"](inp).mean
                        )
                    else:
                        discount = self._config.discount * torch.ones_like(reward)
                else:
                    discount = None

                # KL terms (only if any KL mode is active)
                if self._kl_mode != "none":
                    K_t, delta_t = self._compute_kl_terms(imag_feat, imag_action)
                else:
                    K_t, delta_t = None, None

                # Critic reward:
                #  - recursive: r + beta_kl * K_t (KL-shaped MDP)
                #  - others:    pure extrinsic reward
                if self._kl_mode == "recursive" and K_t is not None:
                    reward_for_critic = reward + self._beta_kl * K_t
                else:
                    reward_for_critic = reward

                # Base Dreamer target, weights, baseline (uses reward_for_critic)
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward_for_critic
                )

                # Base Dreamer actor loss (dynamics / reinforce / both)
                actor_loss_main, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )

                # Extra AGAC policy gradient
                # We add an extra REINFORCE-style term:
                #   L_bonus ~ - E[ weights * log π(a|s) * bonus_adv_t ]
                # where bonus_adv_t is:
                #   - beta_kl * G_t^KL   for actor_horizon
                #   - beta_delta * δ_t   for recursive
                actor_loss_bonus = 0.0
                if self._kl_mode in ["actor_horizon", "recursive"]:
                    policy = self.actor(imag_feat.detach())
                    logp = policy.log_prob(imag_action)  # (T,B) or (T,B,1)
                    # Align shapes to (T-1, B, 1) as in ImagBehavior
                    logp = logp[:-1]
                    if logp.ndim == 2:
                        logp = logp[:, :, None]

                    if self._kl_mode == "actor_horizon":
                        # Discount-aware horizon KL bonus, critic stays extrinsic
                        assert K_t is not None and discount is not None
                        G_kl = self._horizon_kl_return(K_t, discount)  # (T,B,1)
                        bonus_adv = self._beta_kl * G_kl[1:].detach()  # (T-1,B,1)
                    elif self._kl_mode == "recursive":
                        # Log-ratio δ_t bonus (AGAC-style)
                        assert delta_t is not None
                        bonus_adv = self._beta_delta * delta_t[1:].detach()  # (T-1,B,1)
                    else:
                        bonus_adv = None

                    if bonus_adv is not None:
                        actor_target_bonus = logp * bonus_adv
                        actor_loss_bonus = -weights[:-1] * actor_target_bonus

                # Ensure same shape as main actor_loss tensor
                if not torch.is_tensor(actor_loss_bonus):
                    # scalar 0.0 -> broadcastable tensor of zeros
                    actor_loss_bonus = torch.zeros_like(actor_loss_main)

                # Combine main + bonus, then add entropy and reduce
                actor_loss = actor_loss_main + actor_loss_bonus
                actor_loss -= (
                    self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                )
                actor_loss = torch.mean(actor_loss)

                metrics.update(mets)
                metrics["actor_loss_main"] = to_np(actor_loss_main.mean())
                metrics["actor_loss_bonus"] = to_np(actor_loss_bonus.mean())
                metrics["actor_entropy"] = to_np(actor_ent.mean())
                metrics["imag_state_entropy"] = to_np(state_ent.mean())
                value_input = imag_feat

        # Critic (Dreamer, possibly shaped) 
        with tools.RequiresGrad(self.value):
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                value = self.value(value_input[:-1].detach())
                target_tensor = torch.stack(target, dim=1)
                value_loss = -value.log_prob(target_tensor.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        # ----------------- Adversary (track old actor) -----------------
        with tools.RequiresGrad(self.adversary):
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                # 1) Get old actor distribution (no grads)
                with torch.no_grad():
                    pi_old = self._slow_actor(imag_feat.detach())

                # 2) Current adversary distribution (we DO want grads here)
                adv_dist = self.adversary(imag_feat.detach())

                # ---- Discrete case: exact Categorical KL ----
                if hasattr(pi_old, "logits") and hasattr(adv_dist, "logits"):
                    p = Categorical(logits=pi_old.logits)   # π_old
                    q = Categorical(logits=adv_dist.logits) # π_adv
                    kl = kl_divergence(p, q)                # (T,B)
                    adv_loss = torch.mean(kl)               # scalar

                else:
                    # ---- Continuous Gaussian case: exact Normal KL if possible ----
                    inner_pi  = getattr(pi_old, "_dist", None)
                    inner_adv = getattr(adv_dist, "_dist", None)
                    if (
                        isinstance(inner_pi, Independent)
                        and isinstance(inner_pi.base_dist, Normal)
                        and isinstance(inner_adv, Independent)
                        and isinstance(inner_adv.base_dist, Normal)
                    ):
                        base_pi  = inner_pi.base_dist      # Normal(loc_old, scale_old)
                        base_adv = inner_adv.base_dist     # Normal(loc_adv, scale_adv)
                        # KL over action dims: (T,B,action_dim) -> (T,B)
                        kl_raw = kl_divergence(base_pi, base_adv)   # (T,B,action_dim)
                        kl = kl_raw.sum(-1)                         # (T,B)
                        adv_loss = torch.mean(kl)                   # scalar
                    else:
                        # Fallback: MC estimate as before
                        a_sample = imag_action.detach()
                        logp_old = pi_old.log_prob(a_sample)
                        logp_adv = adv_dist.log_prob(a_sample)
                        adv_loss = torch.mean(logp_old - logp_adv)

        # ----------------- Optimizer steps & metrics -----------------
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
            metrics.update(self._adv_opt(adv_loss, self.adversary.parameters()))

        metrics["actor_loss"] = to_np(actor_loss)
        metrics["value_loss"] = to_np(value_loss)
        metrics["adv_loss"] = to_np(adv_loss)
        metrics["kl_mode"] = {
            "none": 0.0,
            "recursive": 1.0,
            "actor_horizon": 2.0,
        }[self._kl_mode]
        metrics["old_policy_mode"] = {
            "ema": 0.0,
            "snapshot": 1.0,
        }[self._old_policy_mode]

        # Same return signature as ImagBehavior._train
        return imag_feat, imag_state, imag_action, weights, metrics

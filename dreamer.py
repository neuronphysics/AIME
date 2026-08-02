import argparse
import functools
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
# newer torch removed the top-level alias dreamerv3-torch expects
if not hasattr(torch, "softplus"):
    torch.softplus = torch.nn.functional.softplus
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset, train_eps=None):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._train_eps = train_eps        # live replay cache, for whole-episode consolidation
        # frozen-representation regime consolidation (episode-aligned, off by default)
        self._episodes_seen = 0
        self._last_consolidate_ep = 0
        self._shs_consolidate_every = int(getattr(config, "shs_consolidate_every_episodes", 0))
        self._shs_consolidate_warmup = int(getattr(config, "shs_consolidate_warmup", 0))
        self._shs_consolidate_batches = int(getattr(config, "shs_consolidate_batches", 8))
        self._shs_consolidate_sweeps = int(getattr(config, "shs_consolidate_sweeps", 3))
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        from shs_rssm.episode_stream import CompletedEpisodeQueue
        self._shs_episode_queue = CompletedEpisodeQueue()   # review P0 #2/#3: exactly-once completed-episode queue
        self._shs_epoch_step = 0   # review P0 #4: step of the last representation-epoch refresh
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if getattr(config, "use_shs", False) and config.compile:
            print("[SHS] disabling torch.compile: SHS-RSSM does Python-side HDP/L-BFGS "
                  "updates, buffer mutation, and shape-changing moves that compile can't trace.")
        if (
            config.compile and os.name != "nt"
            and not getattr(config, "use_shs", False)
        ):  # compilation is not supported on windows, nor with SHS-RSSM
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def _shs_stream_enqueue(self, env_id, key, payload):
        """Exactly-once enqueue at the REAL episode-completion event
        (called from tools.simulate). Pushes the DURABLE replay key AND the exact payload so
        the transactional drain ingests precisely the finished episode. Active only when SHS
        completed-episode streaming is configured."""
        if (getattr(self._config, "use_shs", False)
                and getattr(self._config, "shs_stream_episodes", False)):
            self._shs_episode_queue.push(payload, replay_key=key)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                # SHS-RSSM regime diagnostics (scalars always; figures to logdir).
                # Enable with `shs_diag_log: True` in the config; no-op otherwise and
                # for the plain Gaussian RSSM. Read-only w.r.t. the regime globals.
                if getattr(self._config, "use_shs", False) and \
                        getattr(self._config, "shs_diag_log", False):
                    shs_metrics = self._wm.shs_diagnostics(
                        next(self._dataset), self._step, self._config.logdir,
                        make_figures=getattr(self._config, "shs_diag_figures", True))
                    for _k, _v in shs_metrics.items():
                        self._logger.scalar(_k, _v)
                    # latent-disentanglement diagnostics vs the shapes ground truth
                    # (no-op for envs without the `log_*` factor channels)
                    dz_metrics = self._wm.shs_disentangle_diagnostics(
                        next(self._dataset), self._step, self._config.logdir,
                        make_figures=getattr(self._config, "shs_diag_figures", True))
                    for _k, _v in dz_metrics.items():
                        self._logger.scalar(_k, _v)
                    # long-horizon regime filmstrip: real frames + z_t under the ribbon
                    fs_metrics = self._wm.shs_regime_filmstrip(
                        next(self._dataset), self._step, self._config.logdir)
                    for _k, _v in fs_metrics.items():
                        self._logger.scalar(_k, _v)
                self._logger.write(fps=True)

            # Episode-aligned frozen-representation regime consolidation. Whole episodes are
            # the right unit because regimes are within-episode segments and the HMM
            # transition counts need complete trajectories; we therefore trigger on completed
            # episodes (summed across envs), not on a step count. Heavy, so it runs rarely;
            # the cheap streaming sweep still handles fast adaptation. Off unless
            # `shs_consolidate_every_episodes > 0`.
            self._episodes_seen += int(np.sum(reset))
            if (getattr(self._config, "use_shs", False)
                    and self._shs_consolidate_every > 0
                    and self._train_eps is not None
                    and self._step >= self._shs_consolidate_warmup
                    and self._episodes_seen - self._last_consolidate_ep
                        >= self._shs_consolidate_every):
                self._last_consolidate_ep = self._episodes_seen
                ep_len = int(getattr(self._config, "shs_consolidate_ep_len", 500))
                batches = tools.sample_whole_episodes(
                    self._train_eps, self._shs_consolidate_batches,
                    max_len=ep_len, seed=self._episodes_seen)
                if batches:
                    cons = self._wm.consolidate_regimes(
                        batches, n_sweeps=self._shs_consolidate_sweeps)
                    for _k, _v in cons.items():
                        self._logger.scalar(_k, _v)

            # absorb-ONCE persistent streaming of completed
            # episodes into the SHS sufficient-statistic archive (separate from replay
            # training and from consolidation moves). Off unless `shs_stream_episodes` and a
            # streaming online_mode.
            #
            # (push side + long-horizon epoch): each COMPLETED episode
            # (reset>0) is PUSHED to the exactly-once queue; a long-horizon representation
            # epoch is opened around the drain so all episodes ingested in one burst share a
            # FROZEN target encoder; the queue drains exactly once (monotonic ids, watermark
            # checkpointed) and routes to stream_completed_episodes.
            #
            # episodes are ENQUEUED at the real completion event in
            # tools.simulate (on_episode_complete -> _shs_stream_enqueue) with their DURABLE
            # replay keys, so here we drain TRANSACTIONALLY and fetch the EXACT episodes by
            # key -- never a random replay sample. The count, watermark and payloads are all
            # exact and checkpointed. (Running this still requires the DMC rollout loop.)
            if (getattr(self._config, "use_shs", False)
                    and getattr(self._config, "shs_stream_episodes", False)
                    and self._train_eps is not None
                    and self._step >= self._shs_consolidate_warmup
                    and len(self._shs_episode_queue) > 0):
                # review P0 #5: TRANSACTIONAL drain -- reserve, stream, then commit ONLY on
                # success; on any failure abort so the watermark/totals/cursor do not move.
                _res = self._shs_episode_queue.reserve()
                if _res:
                    try:
                        _ep_len = int(getattr(self._config, "shs_consolidate_ep_len", 500))
                        # review P0 #3: fetch each reserved episode by its DURABLE key
                        _sbatches = []
                        for (_id, _payload, _key) in _res:
                            _ep = _payload if isinstance(_payload, dict) else (
                                tools.load_episode_by_key(self._config.traindir, _key)
                                if _key is not None else None)
                            _b = tools.episode_to_batch(_ep, max_len=_ep_len)
                            if _b is not None:
                                _sbatches.append(_b)
                        if _sbatches:
                            self._wm._shs_reservoir_add(_sbatches)   # review P1 #7: retain for rebuild
                            # ONE persistent frozen-target epoch (idempotent begin,
                            # NOT closed per drain) so accumulated stats never mix two versions;
                            # refresh only on a long cadence (rebuild-from-reservoir belongs there).
                            self._wm.begin_shs_repr_epoch()
                            _refresh = int(getattr(self._config, "shs_repr_epoch_steps", 0))
                            if _refresh > 0 and (self._step - getattr(self, "_shs_epoch_step", 0)) >= _refresh:
                                self._wm.refresh_shs_repr_epoch()
                                self._shs_epoch_step = int(self._step)
                            _sm = self._wm.stream_completed_episodes(_sbatches)
                            if not all(np.isfinite(v) for v in _sm.values()):
                                raise FloatingPointError("non-finite SHS stream update")
                            for _k, _v in _sm.items():
                                self._logger.scalar(_k, _v)
                        self._shs_episode_queue.commit()      # atomic: advance watermark
                    except Exception as _e:
                        self._shs_episode_queue.abort()       # roll back: nothing moves
                        self._logger.scalar("shs_stream_aborted", 1.0)
                    self._logger.scalar("shs_stream_watermark",
                                        float(self._shs_episode_queue.consumed_watermark))

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "shapes":
        from envs.shapes import ShapesEnv

        env = ShapesEnv(task, size=config.size, seed=config.seed + id,
                        time_limit=config.time_limit)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    elif suite == "metaworld":
        from envs.metaworld import MetaWorld

        env = MetaWorld(
            task,
            action_repeat=config.action_repeat,
            size=config.size,
            seed=config.seed + id,
            camera=config.mw_camera,
            render=config.mw_render,
            randomize_goal=config.mw_randomize_goal,
            terminate_on_success=config.mw_terminate_on_success,
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "procgen":
        from envs.procgen import ProcGen

        env = ProcGen(
            task,
            action_repeat=config.action_repeat,
            size=config.size,
            seed=config.seed + id,
            distribution_mode=config.procgen_distribution_mode,
            num_levels=config.procgen_num_levels,
            start_level=config.procgen_start_level,
        )
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode, id: make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    # A discrete (one-hot-wrapped) action space needs a one-hot actor; otherwise the learned
    # policy emits continuous values the OneHotAction wrapper rejects. Only override a
    # continuous dist so an explicit onehot/onehot_gumble choice is preserved.
    if hasattr(acts, "discrete") and config.actor["dist"] not in ("onehot", "onehot_gumble"):
        print(f"[env] discrete action space ({config.num_actions} actions): "
              f"setting actor dist '{config.actor['dist']}' -> 'onehot'")
        config.actor["dist"] = "onehot"

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
        train_eps,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        if checkpoint.get("shs_episode_queue") is not None \
                and hasattr(agent, "_shs_episode_queue"):
            agent._shs_episode_queue.load_state_dict(checkpoint["shs_episode_queue"])
        agent._shs_epoch_step = int(checkpoint.get("shs_epoch_step", 0))
        if checkpoint.get("shs_target_state") is not None:
            agent._wm._shs_target_state = checkpoint["shs_target_state"]
        agent._should_pretrain._once = False

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
            on_episode_complete=agent._shs_stream_enqueue,   # review P0 #3
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            # review P0 #3: persist the completed-episode queue (next id, watermark,
            # pending ids) so exactly-once ingestion survives checkpoint/resume.
            "shs_episode_queue": agent._shs_episode_queue.state_dict()
            if hasattr(agent, "_shs_episode_queue") else None,
            "shs_epoch_step": int(getattr(agent, "_shs_epoch_step", 0)),
            # review P1 #7: persist the frozen representation target so a mid-epoch resume
            # continues with the SAME coordinate system (not a fresh live snapshot).
            "shs_target_state": getattr(agent._wm, "_shs_target_state", None),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    yaml_loader = yaml.YAML(typ="safe", pure=True)
    root = pathlib.Path(sys.argv[0]).parent
    configs = yaml_loader.load((root / "configs.yaml").read_text())
    # Per-benchmark configs live in benchmarks/<name>/configs.yaml and are merged
    # into the same flat namespace, so `--configs metaworld_shs` works exactly
    # like the configs defined in the root file. Root definitions win on a name
    # clash so nothing that already works can be silently overridden.
    for extra in sorted((root / "benchmarks").glob("*/configs.yaml")):
        for name, cfg in (yaml_loader.load(extra.read_text()) or {}).items():
            if name in configs:
                raise ValueError(
                    f"config '{name}' in {extra} collides with configs.yaml"
                )
            configs[name] = cfg

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))

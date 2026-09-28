import argparse
import functools
import json
import os
import pathlib
import random
import signal
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
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._train_eps = train_eps
        self._episodes_seen = 0
        self._last_consolidate_ep = 0
        self._shs_consolidate_every = int(getattr(config, "shs_consolidate_every_episodes", 0))
        self._shs_consolidate_warmup = int(getattr(config, "shs_consolidate_warmup", 0))
        self._shs_consolidate_batches = int(getattr(config, "shs_consolidate_batches", 8))
        self._shs_consolidate_sweeps = int(getattr(config, "shs_consolidate_sweeps", 3))
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        from shs_rssm.episode_stream import CompletedEpisodeQueue
        self._shs_episode_queue = CompletedEpisodeQueue()
        self._shs_epoch_step = 0
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if getattr(config, "use_shs", False) and config.compile:
            print("[SHS] disabling torch.compile: SHS-RSSM does Python-side HDP/L-BFGS "
                  "updates, buffer mutation, and shape-changing moves that compile can't trace.")
        if (
            config.compile and os.name != "nt"
            and not getattr(config, "use_shs", False)
        ):
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def _shs_stream_enqueue(self, env_id, key, payload):
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
                if getattr(self._config, "use_shs", False) and \
                        getattr(self._config, "shs_diag_log", False):
                    shs_metrics = self._wm.shs_diagnostics(
                        next(self._dataset), self._step, self._config.logdir,
                        make_figures=getattr(self._config, "shs_diag_figures", True))
                    for _k, _v in shs_metrics.items():
                        self._logger.scalar(_k, _v)
                    dz_metrics = self._wm.shs_disentangle_diagnostics(
                        next(self._dataset), self._step, self._config.logdir,
                        make_figures=getattr(self._config, "shs_diag_figures", True))
                    for _k, _v in dz_metrics.items():
                        self._logger.scalar(_k, _v)
                    fs_metrics = self._wm.shs_regime_filmstrip(
                        next(self._dataset), self._step, self._config.logdir)
                    for _k, _v in fs_metrics.items():
                        self._logger.scalar(_k, _v)
                self._logger.write(fps=True)

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

            if (getattr(self._config, "use_shs", False)
                    and getattr(self._config, "shs_stream_episodes", False)
                    and self._train_eps is not None
                    and self._step >= self._shs_consolidate_warmup
                    and len(self._shs_episode_queue) > 0):
                _res = self._shs_episode_queue.reserve()
                if _res:
                    try:
                        _ep_len = int(getattr(self._config, "shs_consolidate_ep_len", 500))
                        _sbatches = []
                        _sids = []
                        _skipped = 0
                        for (_id, _payload, _key) in _res:
                            _ep = _payload if isinstance(_payload, dict) else (
                                tools.load_episode_by_key(self._config.traindir, _key)
                                if _key is not None else None)
                            if bool(getattr(self._config, "shs_stream_chunked", True)):
                                _b = tools.episode_to_chunks(_ep, chunk_len=_ep_len)
                            else:
                                _b = tools.episode_to_batch(_ep, max_len=_ep_len)
                            if _b is not None:
                                _sbatches.append(_b)
                                _sids.append(_id)
                            else:
                                _skipped += 1
                        if _sbatches:
                            self._wm.begin_shs_repr_epoch()
                            _refresh = int(getattr(self._config, "shs_repr_epoch_steps", 0))
                            if _refresh > 0 and (self._step - getattr(self, "_shs_epoch_step", 0)) >= _refresh:
                                self._wm.refresh_shs_repr_epoch()
                                self._shs_epoch_step = int(self._step)
                            self._logger.scalar(
                                "shs_repr_lag",
                                float(self._step - getattr(self, "_shs_epoch_step", 0)))
                            _sm = self._wm.stream_completed_episodes(_sbatches,
                                                                     episode_ids=_sids)
                            if not all(np.isfinite(v) for v in _sm.values()):
                                raise FloatingPointError("non-finite SHS stream update")
                            self._wm._shs_reservoir_add(_sbatches)
                            for _k, _v in _sm.items():
                                self._logger.scalar(_k, _v)
                        self._shs_episode_queue.commit()
                        if _skipped:
                            self._logger.scalar("shs_stream_skipped", float(_skipped))
                    except Exception as _e:
                        self._shs_episode_queue.abort()
                        self._logger.scalar("shs_stream_aborted", 1.0)
                        import traceback as _tb
                        print("[SHS] streaming ingestion ABORTED -- batch rolled back, "
                              f"queue watermark unchanged: {type(_e).__name__}: {_e}")
                        _tb.print_exc()
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
            action = torch.nn.functional.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            ).float()
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


def make_dataset(episodes, config, seed=0):
    generator = tools.sample_episodes(episodes, config.batch_length, seed)
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
    elif suite == "procgen":
        import envs.procgen as procgen
        env = procgen.ProcGen(task, config.size, seed=config.seed + id)
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
    elif suite == "carl":
        import envs.carl as carl

        cfg = config.carl
        is_train = "train" in mode
        env = carl.CARL(
            task,
            config.action_repeat,
            config.size,
            seed=config.seed + id,
            contexts=cfg["train_contexts"] if is_train else cfg["eval_contexts"],
            selector=cfg["train_selector"] if is_train else cfg["eval_selector"],
            start=id,
        )
        env = wrappers.NormalizeActions(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


OPTIONAL_STATE_SUFFIXES = ("z0_logvar", "b0_c0", "b0_d0", "b0_chat", "b0_dhat")


def _shs_adapt_state_shapes(agent, sd):
    resized = []
    for mname, mod in agent.named_modules():
        for bname, buf in list(mod.named_buffers(recurse=False)):
            full = f"{mname}.{bname}" if mname else bname
            t = sd.get(full)
            if torch.is_tensor(t) and torch.is_tensor(buf) and tuple(t.shape) != tuple(buf.shape):
                mod.register_buffer(bname, torch.zeros(tuple(t.shape), dtype=buf.dtype, device=buf.device))
                resized.append(f"{full}: {tuple(buf.shape)} -> {tuple(t.shape)}")
    if resized:
        print("[SHS] resume: adapted buffer shapes to checkpoint:")
        for r in resized:
            print("   ", r)
    return resized


RESUME_EXIT = 75
CONFIG_EXIT = 78
STOP_POLL = 100
SCHEDULES = ("_should_log", "_should_train", "_should_reset")
RUN_CONTROL_KEYS = frozenset((
    "logdir", "traindir", "evaldir", "offline_traindir", "offline_evaldir", "steps", "eval_every",
    "eval_episode_num", "log_every", "video_pred_log", "shs_diag_log", "shs_diag_figures",
    "device", "compile", "parallel", "debug"))


class StopRequest:
    def __init__(self):
        self.signum = None
        for name in ("SIGUSR1", "SIGTERM"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), self._request)

    def _request(self, signum, frame):
        self.signum = signum

    def __bool__(self):
        return self.signum is not None


def rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda = state.get("cuda")
    if cuda is not None and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(cuda)


def derived_seed(*entropy):
    return int(np.random.SeedSequence([int(e) for e in entropy]).generate_state(1)[0] % (2**31 - 1))


def schedule_state(agent):
    state = {name: getattr(agent, name)._last for name in SCHEDULES}
    state["pretrain_pending"] = bool(agent._should_pretrain._once)
    if hasattr(agent._task_behavior, "_updates"):
        state["behavior_updates"] = int(agent._task_behavior._updates)
    return state


def load_schedule_state(agent, state):
    for name in SCHEDULES:
        if name in state:
            getattr(agent, name)._last = state[name]
    agent._should_pretrain._once = bool(state.get("pretrain_pending", False))
    if "behavior_updates" in state and hasattr(agent._task_behavior, "_updates"):
        agent._task_behavior._updates = int(state["behavior_updates"])


def atomic_save(items, path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(items, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def reconcile_replay(traindir, kept, logdir):
    kept = set(kept)
    present = {p.name: p for p in traindir.glob("*.npz")}
    missing = sorted(kept - set(present))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} replay episodes listed in the checkpoint are missing from {traindir}, "
            f"first {missing[0]}")
    extra = sorted(name for name in present if name not in kept)
    if extra:
        aside = logdir / "train_eps_after_checkpoint"
        aside.mkdir(exist_ok=True)
        for name in extra:
            os.replace(present[name], aside / name)
        print(f"[resume] moved {len(extra)} episodes recorded after the checkpoint to {aside}")


def check_resume_config(stored, current):
    if not isinstance(stored, dict):
        return
    shared = (set(stored) & set(current)) - RUN_CONTROL_KEYS
    changed = sorted(k for k in shared if stored[k] != current[k])
    if changed:
        print("latest.pt was trained with other settings for: " + ", ".join(changed)
              + "; resume with the same settings or use another logdir", file=sys.stderr)
        raise SystemExit(CONFIG_EXIT)
    single = sorted((set(stored) ^ set(current)) - RUN_CONTROL_KEYS)
    if single:
        print(f"[resume] keys present in only one of the two configurations: {', '.join(single)}")


def truncate_metrics(logdir, size):
    path = logdir / "metrics.jsonl"
    if size is None or not path.exists() or path.stat().st_size <= size:
        return
    with path.open("rb") as f:
        f.seek(size)
        tail = f.read()
    with (logdir / "metrics_after_checkpoint.jsonl").open("ab") as f:
        f.write(tail)
    with path.open("rb+") as f:
        f.truncate(size)
    records = tail.count(b"\n")
    print(f"[resume] moved {records} metric records written after the checkpoint to metrics_after_checkpoint.jsonl")


def record_config(logdir, snapshot):
    text = json.dumps(snapshot, indent=2, sort_keys=True, default=str) + "\n"
    (logdir / "run_config.json").write_text(text)
    first = logdir / "launch_config.json"
    if not first.exists():
        first.write_text(text)
        return
    launched = json.loads(first.read_text())
    changed = sorted(k for k in set(launched) | set(snapshot) if launched.get(k) != snapshot.get(k))
    if changed:
        print(f"[config] this launch differs from launch_config.json in: {', '.join(changed)}")


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    stop = StopRequest()
    config_snapshot = json.loads(json.dumps(vars(config), default=str))
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    if hasattr(config, "shs_consolidate_warmup"):
        config.shs_consolidate_warmup //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    record_config(logdir, config_snapshot)
    checkpoint = None
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt", weights_only=False)
        check_resume_config(checkpoint.get("config"), config_snapshot)
    progress = checkpoint is not None and "step" in checkpoint
    resumes = int(checkpoint.get("resumes", 0)) + 1 if progress else 0
    if progress:
        if not config.offline_traindir:
            reconcile_replay(config.traindir, checkpoint["episode_files"], logdir)
        truncate_metrics(logdir, checkpoint.get("metrics_bytes"))
        step = int(checkpoint["step"])
        logger = tools.Logger(logdir, int(checkpoint["logger_step"]),
                              purge_step=int(checkpoint["logger_step"]) + 1)
        print(f"[resume] agent step {step} ({logger.step} frames), start number {resumes + 1}")
    else:
        step = count_steps(config.traindir)
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
    env_config = config
    if resumes:
        env_config = argparse.Namespace(**vars(config))
        env_config.seed = derived_seed(config.seed, resumes, 1)
    make = lambda mode, id: make_env(env_config, mode, id)
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
    if hasattr(acts, "discrete") and config.actor["dist"] not in ("onehot", "onehot_gumble"):
        print(f"[env] discrete action space ({config.num_actions} actions): "
              f"setting actor dist '{config.actor['dist']}' -> 'onehot'")
        config.actor["dist"] = "onehot"

    state = None
    if not config.offline_traindir and not progress:
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
    train_dataset = make_dataset(train_eps, config, derived_seed(config.seed, resumes, 2) if resumes else 0)
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
    remaining = 0
    if checkpoint is not None:
        _sd = checkpoint["agent_state_dict"]
        _shs_adapt_state_shapes(agent, _sd)
        _NEW_KEY_SUFFIXES = OPTIONAL_STATE_SUFFIXES
        _missing = [k for k, v in agent.state_dict().items()
                    if k.endswith(_NEW_KEY_SUFFIXES) and k not in _sd
                    and torch.is_tensor(v)]
        if _missing:
            import warnings
            for _k in _missing:
                _sd[_k] = agent.state_dict()[_k].clone()
            warnings.warn(
                f"checkpoint has no {sorted(_missing)}; using their initial values")
        agent.load_state_dict(_sd)
        _ints = checkpoint.get("shs_int_attrs") or {}
        for _n, _m in agent.named_modules():
            _k = _ints.get(f"{_n}.K" if _n else "K")
            if _k is not None and isinstance(getattr(_m, "K", None), int):
                _m.K = int(_k)
        agent._update_count = int(checkpoint.get("update_count", agent._update_count))
        agent._episodes_seen = int(checkpoint.get("episodes_seen", agent._episodes_seen))
        agent._last_consolidate_ep = int(
            checkpoint.get("last_consolidate_ep", agent._last_consolidate_ep))
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        if checkpoint.get("shs_episode_queue") is not None \
                and hasattr(agent, "_shs_episode_queue"):
            agent._shs_episode_queue.load_state_dict(checkpoint["shs_episode_queue"])
        agent._shs_epoch_step = int(checkpoint.get("shs_epoch_step", 0))
        if checkpoint.get("shs_target_state") is not None:
            agent._wm._shs_target_state = checkpoint["shs_target_state"]
        if hasattr(agent._wm.dynamics, "load_live_state") and checkpoint.get("shs_live"):
            _live = agent._wm.dynamics.load_live_state(
                checkpoint["shs_live"], checkpoint.get("transfer_runtime"), agent._wm.encoder)
            print(f"[SHS] runtime restored: {_live}")
        agent._should_pretrain._once = False
        if progress:
            load_schedule_state(agent, checkpoint.get("schedules") or {})
            remaining = int(checkpoint.get("train_remaining", 0))
            if checkpoint.get("rng") is not None:
                set_rng_state(checkpoint["rng"])
        del _sd
    checkpoint = None

    def save_checkpoint(train_remaining):
        metrics = logdir / "metrics.jsonl"
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            "shs_episode_queue": agent._shs_episode_queue.state_dict()
            if hasattr(agent, "_shs_episode_queue") else None,
            "shs_epoch_step": int(getattr(agent, "_shs_epoch_step", 0)),
            "shs_target_state": getattr(agent._wm, "_shs_target_state", None),
            "shs_int_attrs": {(f"{_n}.K" if _n else "K"): int(_m.K)
                              for _n, _m in agent.named_modules()
                              if isinstance(getattr(_m, "K", None), int)},
            "update_count": int(getattr(agent, "_update_count", 0)),
            "episodes_seen": int(getattr(agent, "_episodes_seen", 0)),
            "last_consolidate_ep": int(getattr(agent, "_last_consolidate_ep", 0)),
            "task": config.task,
            "config": config_snapshot,
            "step": int(agent._step),
            "logger_step": int(logger.step),
            "train_remaining": int(train_remaining),
            "resumes": resumes,
            "episode_files": sorted(p.name for p in config.traindir.glob("*.npz")),
            "metrics_bytes": metrics.stat().st_size if metrics.exists() else 0,
            "rng": rng_state(),
            "schedules": schedule_state(agent),
        }
        _dyn = agent._wm.dynamics
        if hasattr(_dyn, "live_state"):
            _teacher = getattr(_dyn, "_teacher", None)
            _probe = getattr(_teacher, "probe", None) if _teacher is not None else None
            items_to_save["shs_live"] = _dyn.live_state()
            items_to_save["transfer_runtime"] = {
                "b0_calibrated": bool(getattr(_dyn, "_shs_b0_calibrated", False)),
                "teacher": _teacher.state_dict() if _teacher is not None else None,
                "teacher_probe": ({k: v.detach().cpu() for k, v in _probe.items()}
                                  if isinstance(_probe, dict) else None),
            }
        atomic_save(items_to_save, logdir / "latest.pt")

    try:
        while remaining > 0 or agent._step < config.steps + config.eval_every:
            if remaining == 0:
                logger.write()
                if stop:
                    save_checkpoint(0)
                    break
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
                remaining = config.eval_every
                print("Start training.")
            else:
                print(f"Resume training ({remaining} agent steps left before the next evaluation).")
            while remaining > 0 and not stop:
                chunk = min(STOP_POLL, remaining)
                state = tools.simulate(
                    agent,
                    train_envs,
                    train_eps,
                    config.traindir,
                    logger,
                    limit=config.dataset_size,
                    steps=chunk,
                    state=state,
                    on_episode_complete=agent._shs_stream_enqueue,
                )
                remaining -= chunk
            save_checkpoint(remaining)
            if stop:
                break
    finally:
        for env in train_envs + eval_envs:
            try:
                env.close()
            except Exception:
                pass
    if stop:
        print(f"[checkpoint] {signal.Signals(stop.signum).name}: saved at agent step {agent._step} "
              f"({logger.step} frames); exit {RESUME_EXIT} to resume in the next job")
        return RESUME_EXIT
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    yaml_loader = yaml.YAML(typ="safe", pure=True)
    configs = yaml_loader.load(
       (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

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
    sys.exit(main(parser.parse_args(remaining)))
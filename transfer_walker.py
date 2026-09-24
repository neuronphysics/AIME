"""Fresh-replay Walker transfer for the two supplied Dreamer/AIME projects.

Copy beside dreamer.py. Reuses that project's agent, environment and training
methods. See WALKER_TRANSFER.md for exact configuration and experiment controls.
Checkpoints are trusted local training artifacts, not untrusted downloads.
"""
from __future__ import annotations

import argparse
import collections
import copy
import functools
import hashlib
import inspect
import json
import os
import pathlib
import sys
import warnings

import numpy as np
import torch
from ruamel.yaml import YAML

_requested_gl_backend = os.environ.get("MUJOCO_GL")
import dreamer
if _requested_gl_backend is not None:
    # The legacy baseline unconditionally sets OSMesa at import time.
    os.environ["MUJOCO_GL"] = _requested_gl_backend
import tools
from parallel import Damy, Parallel


def sha256_of(path, chunk=1 << 20):
    """Digest of a file; hashlib.file_digest needs Python 3.11, the Mila env has 3.10."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def canonical_state(state):
    """Normalize torch.compile wrappers without dropping model tensors."""
    result = {}
    for key, value in state.items():
        name = ".".join(part for part in key.split(".") if part != "_orig_mod")
        if name in result:
            raise ValueError(f"Ambiguous compiled checkpoint key: {name}")
        result[name] = value
    return result


def component(state, prefix):
    result = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not result:
        raise ValueError(f"Checkpoint has no component {prefix!r}")
    return result


def make_agent(obs_space, act_space, config, logger, dataset, train_eps=None):
    # Legacy MultiEncoder/Decoder omit the device argument to MLP, whose
    # constructor defaults to CUDA. Supply the selected device during creation
    # without modifying either repository's network implementation.
    import networks
    original = networks.MLP.__init__
    signature = inspect.signature(original)

    def init(self, *args, **kwargs):
        if "device" not in signature.bind_partial(self, *args, **kwargs).arguments:
            kwargs["device"] = config.device
        original(self, *args, **kwargs)

    networks.MLP.__init__ = init
    try:
        kwargs = {"train_eps": train_eps} if "train_eps" in inspect.signature(dreamer.Dreamer).parameters else {}
        return dreamer.Dreamer(obs_space, act_space, config, logger, dataset, **kwargs).to(config.device)
    finally:
        networks.MLP.__init__ = original


def strict_load(module, state, label):
    # AIME's structural pre-hook can resize parameters. Check first to prevent
    # a wrong architecture from silently replacing optimizer-owned parameters.
    target = module.state_dict()
    missing = sorted(set(target) - set(state))
    extra = sorted(set(state) - set(target))
    shapes = [f"{k}: saved {tuple(v.shape)}, constructed {tuple(target[k].shape)}"
              for k, v in state.items() if k in target and torch.is_tensor(v)
              and torch.is_tensor(target[k]) and v.shape != target[k].shape]
    if missing or extra or shapes:
        raise ValueError(
            f"Incompatible {label}; reuse the source architecture/configuration.\n"
            f"Missing: {missing[:10]}\nUnexpected: {extra[:10]}\n"
            + "\n".join(shapes[:10]))
    module.load_state_dict(state, strict=True)


def checkpoint_kind(state):
    return "aime" if "_wm.dynamics.regime.ema_trans_counts" in state else "dreamer"


def prepare_architecture(config, checkpoint):
    state = canonical_state(checkpoint["agent_state_dict"])
    is_aime = checkpoint_kind(state) == "aime"
    if is_aime != bool(getattr(config, "use_shs", False)):
        raise ValueError("Load an AIME checkpoint into AIME, and a Dreamer checkpoint into Dreamer.")
    if is_aime:
        counts = state["_wm.dynamics.regime.ema_trans_counts"]
        if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
            raise ValueError("Invalid saved regime count matrix")
        # Instantiate every K-dependent object correctly before optimizers exist.
        config.shs_K = int(counts.shape[0])
        print(f"[transfer] instantiate saved regime count K={config.shs_K}")
    return state


def configure_loaded_shs(agent, checkpoint, *, resume=False, fixed_regimes=False):
    wm, dynamics = agent._wm, agent._wm.dynamics
    if not hasattr(dynamics, "regime"):
        return
    runtime = checkpoint.get("transfer_runtime", {})
    # b0 is already stored as a buffer. Legacy checkpoints omit the latch;
    # preserve the saved prior instead of calibrating it again on the new task.
    dynamics._shs_b0_calibrated = bool(runtime.get("b0_calibrated", True))
    # The curriculum phase is chosen from dynamics._shs_step, a plain Python
    # counter that AIME does not serialise (its own resume path has the same
    # gap). Without restoring it the loaded model silently drops back to phase
    # 0: gate off, kappa re-added, moves off. _shs_step advances once per
    # world-model update, as does update_count, so restore it from there.
    saved_updates = int(checkpoint.get("update_count", 0))
    if saved_updates > 0:
        dynamics._shs_step = saved_updates
    else:
        warnings.warn("Checkpoint has no update_count; forcing the final curriculum phase")
        dynamics._shs_step = max([int(p.get("until", -1)) for p in dynamics._curriculum] + [0]) + 1
    phase = next((i for i, p in enumerate(dynamics._curriculum)
                  if int(p.get("until", -1)) < 0
                  or dynamics._shs_step < int(p["until"])),
                 len(dynamics._curriculum) - 1)
    dynamics._curr_phase = phase
    # Apply runtime gate/move settings without the phase-change HDP refit.
    dynamics._apply_curriculum()
    if fixed_regimes:
        dynamics._shs_move_every = 0
        for p in dynamics._curriculum:
            p["move_every"] = 0
    store = dynamics.regime.stat_store
    if store is not None:
        if store.mode != "legacy_ema":
            raise ValueError("This transfer runner supports AIME replay EMA, not fixed-corpus/strict-stream stores.")
        # The serialized store otherwise overwrites an explicitly swept rate.
        store.ema_tau = float(dynamics.regime.ema_tau)
    if not resume:
        dynamics._pending = None
        dynamics._next_batch_id = None
        wm._shs_reservoir = []
        wm._shs_target_state = None
    elif checkpoint.get("shs_target_state") is not None:
        wm._shs_target_state = checkpoint["shs_target_state"]
    teacher = runtime.get("teacher")
    if teacher is not None:
        from shs_rssm.teacher import EMATeacher
        cfg = dynamics._shs_teacher_cfg
        dynamics._teacher = EMATeacher(wm.encoder, tau=cfg["tau"],
                                       drift_tol=cfg["drift_tol"])
        dynamics._teacher.load_state_dict(teacher)
    elif dynamics._shs_teacher_cfg.get("enabled"):
        warnings.warn("Source checkpoint has no EMA-teacher snapshot. The supplied legacy saver omits it; "
                      "the target teacher will initialize from the loaded encoder.")
    print(f"[transfer] SHS curriculum phase={dynamics._curr_phase}/{len(dynamics._curriculum)-1}, "
          f"recurrent gate={getattr(dynamics.regime, 'recurrent', None)}, "
          f"kappa={getattr(getattr(dynamics.regime, 'hdp', None), 'kappa', None)}")
    print(f"[transfer] SHS step={dynamics._shs_step}, "
          f"statistic tau={dynamics.regime.ema_tau}, "
          f"statistics initialized={dynamics.regime.regimes._stats_initialised}")


def initialize_agent(agent, checkpoint, mode, *, resume=False, fixed_regimes=False):
    """Load each shared module once, so aliases cannot undo partial transfer."""
    state = canonical_state(checkpoint["agent_state_dict"])
    wm_state = copy.deepcopy(component(state, "_wm."))
    if mode == "world_model":
        # A new reward definition requires a new reward model in this ablation.
        reward_state = agent._wm.heads["reward"].state_dict()
        for key in list(wm_state):
            if key.startswith("heads.reward."):
                wm_state[key] = reward_state[key[len("heads.reward."):]]
    strict_load(agent._wm, wm_state, "world model")
    if mode == "full":
        behavior = agent._task_behavior
        for name in ("actor", "value", "_slow_value"):
            if hasattr(behavior, name):
                strict_load(getattr(behavior, name),
                            component(state, f"_task_behavior.{name}."), name)
        for name, buffer in behavior.named_buffers(recurse=False):
            key = f"_task_behavior.{name}"
            if key not in state or state[key].shape != buffer.shape:
                raise ValueError(f"Missing/incompatible behavior buffer {key}")
            buffer.copy_(state[key].to(buffer))
    configure_loaded_shs(agent, checkpoint, resume=resume, fixed_regimes=fixed_regimes)
    if resume:
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._update_count = int(checkpoint.get("update_count", 0))
        for attr, key in [("_episodes_seen", "episodes_seen"),
                          ("_last_consolidate_ep", "last_consolidate_ep"),
                          ("_shs_epoch_step", "shs_epoch_step")]:
            if hasattr(agent, attr):
                setattr(agent, attr, int(checkpoint.get(key, 0)))
        agent._task_behavior._updates = int(checkpoint.get("behavior_updates", 0))
        agent._should_pretrain._once = False
    # Fresh transfer keeps freshly constructed optimizers and gives every arm
    # the same configured target pretraining updates after random prefill.


def save_checkpoint(agent, path, config, metadata):
    dynamics = agent._wm.dynamics
    teacher = getattr(dynamics, "_teacher", None)
    payload = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        "shs_int_attrs": {f"{n}.K" if n else "K": int(m.K)
                          for n, m in agent.named_modules()
                          if isinstance(getattr(m, "K", None), int)},
        "update_count": int(agent._update_count),
        "episodes_seen": int(getattr(agent, "_episodes_seen", 0)),
        "last_consolidate_ep": int(getattr(agent, "_last_consolidate_ep", 0)),
        "shs_epoch_step": int(getattr(agent, "_shs_epoch_step", 0)),
        "shs_target_state": getattr(agent._wm, "_shs_target_state", None),
        "behavior_updates": int(getattr(agent._task_behavior, "_updates", 0)),
        "target_env_steps": int(agent._logger.step),
        "task": config.task,
        "transfer_metadata": metadata,
        "transfer_runtime": {
            "b0_calibrated": bool(getattr(dynamics, "_shs_b0_calibrated", False)),
            "teacher": teacher.state_dict() if teacher is not None else None,
        },
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=[])
    parser.add_argument("--config-json", help="Resolved source configuration; remaining flags override it")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--transfer-mode", choices=["full", "world_model", "scratch"], default="full")
    parser.add_argument("--source-task", default="dmc_walker_walk")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--fixed-regimes", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume this runner's existing target run")
    args, remaining = parser.parse_known_args(argv)
    catalog = YAML(typ="safe").load((pathlib.Path(__file__).parent / "configs.yaml").read_text())
    defaults = {}

    def merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = copy.deepcopy(v)

    for name in ["defaults", *args.configs]:
        merge(defaults, catalog[name])
    if args.config_json:
        merge(defaults, json.loads(pathlib.Path(args.config_json).read_text()))
    cfgparser = argparse.ArgumentParser()
    for name, value in defaults.items():
        parse = tools.args_type(value)
        cfgparser.add_argument(f"--{name}", type=parse, default=parse(value))
    config = cfgparser.parse_args(remaining)
    config.compile = False  # checkpoint prefixes are normalized on input
    if not config.logdir:
        raise ValueError("Supply a separate --logdir for this target experiment")
    if config.task not in ("dmc_walker_walk", "dmc_walker_run", "dmc_walker_stand"):
        raise ValueError("Use standard DMC Walker tasks here. carl_dmc_walker_run is not implemented by the supplied CARL wrapper.")
    if config.expl_behavior != "greedy":
        raise ValueError("This runner supports the supplied Walker greedy actor-critic configuration")
    if any(getattr(config, k, None) for k in ("traindir", "evaldir", "offline_traindir", "offline_evaldir")):
        raise ValueError("Use target-local replay: omit traindir/evaldir/offline directory overrides")
    if getattr(config, "use_shs", False):
        if getattr(config, "shs_global_source", "replay_ema") != "replay_ema":
            raise ValueError("This runner's AIME protocol requires shs_global_source=replay_ema")
        if getattr(config, "shs_online_mode", "ema") not in ("ema", "live_ema", "legacy_ema"):
            raise ValueError("Choose an AIME EMA configuration, not a fixed-corpus/strict-stream mode")
        if config.shs_stream_episodes:
            raise ValueError("Disable completed-episode streaming for this replay-EMA protocol")
    if args.resume and args.checkpoint:
        raise ValueError("Use --resume or --checkpoint, not both")
    if not args.resume and ((args.transfer_mode == "scratch") == bool(args.checkpoint)):
        raise ValueError("Supply --checkpoint for full/world_model; omit it for scratch")
    return args, config


def run(args, config):
    logdir = pathlib.Path(config.logdir).expanduser().resolve()
    if not args.resume and logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Fresh transfer requires an empty target directory: {logdir}")
    source = logdir / "latest.pt" if args.resume else pathlib.Path(args.checkpoint).expanduser()
    checkpoint = None
    if args.resume or args.checkpoint:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        if args.resume and "target_env_steps" not in checkpoint:
            raise ValueError("--resume requires a checkpoint written by this transfer runner")
        if args.resume and checkpoint["task"] != config.task:
            raise ValueError("Resume must keep the target task unchanged")
        if not args.resume and checkpoint.get("task", args.source_task) != args.source_task:
            raise ValueError("--source-task disagrees with checkpoint metadata")
        prepare_architecture(config, checkpoint)
    if args.fixed_regimes and getattr(config, "use_shs", False):
        config.shs_move_every = 0
        config.shs_consolidate_every_episodes = 0
        for p in config.shs_curriculum:
            p["move_every"] = 0
            p["move_every_env"] = 0
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    raw_config = copy.deepcopy(vars(config))
    budget = int(config.steps)
    repeat, nenv = int(config.action_repeat), int(config.envs)
    if repeat < 1 or nenv < 1 or budget < 0 or budget % (repeat * nenv):
        raise ValueError("steps must be a nonnegative multiple of action_repeat * envs")
    if config.eval_every < repeat * nenv:
        raise ValueError("eval_every must cover at least one vector environment step")
    config.eval_every = max(nenv, int(config.eval_every) // (repeat * nenv) * nenv)
    config.log_every = max(1, int(config.log_every) // repeat)
    config.time_limit = int(config.time_limit) // repeat
    config.steps = budget // repeat
    if hasattr(config, "shs_consolidate_warmup"):
        config.shs_consolidate_warmup //= repeat
    config.traindir, config.evaldir = logdir / "train_eps", logdir / "eval_eps"
    for directory in (logdir, config.traindir, config.evaldir):
        directory.mkdir(parents=True, exist_ok=True)
    metadata = dict(mode=args.transfer_mode, source_task=args.source_task,
                    target_task=config.task, source_checkpoint=str(source.resolve()) if checkpoint else None,
                    source_sha256=None, optimizer="reset for fresh transfer",
                    replay="fresh target replay", fixed_regimes=args.fixed_regimes,
                    source_cost_included_in_target_steps=False)
    if checkpoint and not args.resume:
        metadata["source_sha256"] = sha256_of(source)
    if args.resume:
        metadata = checkpoint["transfer_metadata"]
    (logdir / "transfer_config.json").write_text(json.dumps(raw_config, indent=2, default=str) + "\n")
    (logdir / "transfer_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    start_step = int(checkpoint["target_env_steps"]) if args.resume else 0
    if start_step > budget and not args.evaluate_only:
        raise ValueError("The requested total target budget is below the resumed checkpoint step")
    logger = tools.Logger(logdir, start_step)
    train_eps = tools.load_episodes(config.traindir, limit=config.dataset_size) if args.resume else collections.OrderedDict()
    eval_eps = collections.OrderedDict()
    train_envs, eval_envs = [], []
    try:
        wrap = (lambda env: Parallel(env, "process")) if config.parallel else Damy
        train_envs = [wrap(dreamer.make_env(config, "train", i)) for i in range(nenv)]
        eval_envs = [wrap(dreamer.make_env(config, "eval", 10000 + i)) for i in range(nenv)]
        config.num_actions = int(train_envs[0].action_space.shape[0])
        agent = make_agent(train_envs[0].observation_space, train_envs[0].action_space,
                           config, logger, dreamer.make_dataset(train_eps, config), train_eps)
        if checkpoint:
            initialize_agent(agent, checkpoint, "full" if args.resume else args.transfer_mode,
                             resume=args.resume, fixed_regimes=args.fixed_regimes)
        agent.requires_grad_(False)

        def evaluate():
            if config.eval_episode_num <= 0:
                return
            # Match the original agent's evaluation path (training=False).
            tools.simulate(functools.partial(agent, training=False), eval_envs, eval_eps,
                           config.evaldir, logger, is_eval=True, episodes=config.eval_episode_num)

        evaluate()  # Before any target interaction/update for a fresh transfer.
        if args.evaluate_only:
            return
        state = None
        if not train_eps:
            prefill = min(config.steps - agent._step,
                          int(np.ceil(int(config.prefill) / nenv)) * nenv)
            if prefill < nenv * int(config.batch_length):
                raise ValueError("Target budget/prefill must supply at least batch_length transitions per environment")
            rng = np.random.default_rng(config.seed)
            acts = train_envs[0].action_space

            def random_agent(obs, done, recurrent):
                action = rng.uniform(acts.low, acts.high, size=(nenv, config.num_actions)).astype(np.float32)
                return {"action": torch.from_numpy(action),
                        "logprob": torch.full((nenv,), float(-np.log(acts.high - acts.low).sum()))}, None

            state = tools.simulate(random_agent, train_envs, train_eps, config.traindir,
                                   logger, limit=config.dataset_size, steps=prefill)
            agent._step += prefill
            logger.step = agent._step * repeat
        while agent._step < config.steps:
            chunk = min(config.eval_every, config.steps - agent._step)
            state = tools.simulate(agent, train_envs, train_eps, config.traindir,
                                   logger, limit=config.dataset_size, steps=chunk, state=state)
            save_checkpoint(agent, logdir / "latest.pt", config, metadata)
            evaluate()
        logger.write()
    finally:
        for env in train_envs + eval_envs:
            try:
                env.close()
            except Exception:
                pass
        logger._writer.close()


if __name__ == "__main__":
    run(*parse_config())

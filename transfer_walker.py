"""Walker walk -> run transfer with fresh replay. The source setup is adopted from the .pt;
run with --help for the arms, the curriculum clock and the resume rules."""
from __future__ import annotations

import argparse
import collections
import copy
import functools
import hashlib
import inspect
import json
import math
import os
import pathlib
import re
import signal
import sys
import warnings

import numpy as np
import torch
from ruamel.yaml import YAML

_requested_gl_backend = os.environ.get("MUJOCO_GL")
import dreamer
if _requested_gl_backend is not None:
    os.environ["MUJOCO_GL"] = _requested_gl_backend
import tools
from parallel import Damy, Parallel


EXIT_RESUME = 75

DEFAULT_SOURCE_RECIPES = (
    "dmc_vision,dmc_walker_shs,shs_walker_recipe",
    "dmc_vision,dmc_walker_shs_moves",
    "dmc_vision,dmc_walker_shs_moves,shs_small",
    "dmc_vision,dmc_walker_shs,walker_walk_launch",
    "dmc_vision,dmc_walker_shs,walker_walk_launch_minimum",
)
RUN_CONTROL_KEYS = {
    "steps", "eval_every", "eval_episode_num", "log_every", "device", "parallel",
    "compile", "video_pred_log", "debug", "logdir", "traindir", "evaldir",
}
RUN_CONTROL_PREFIXES = ("shs_diag",)

TARGET_KEYS = {
    "logdir", "traindir", "evaldir", "offline_traindir", "offline_evaldir", "seed",
    "deterministic_run", "steps", "parallel", "eval_every", "eval_episode_num",
    "log_every", "reset_every", "device", "compile", "precision", "debug",
    "video_pred_log", "task", "envs", "action_repeat", "time_limit", "prefill",
    "carl", "batch_size", "batch_length", "train_ratio", "pretrain", "dataset_size",
    "num_actions", "expl_behavior", "expl_until", "expl_extr_scale",
    "expl_intr_scale", "transfer_curriculum_clock",
}
TARGET_PREFIXES = (
    "disag_", "shs_curriculum", "shs_move", "shs_merge", "shs_delete", "shs_birth",
    "shs_consolidate", "shs_diag", "shs_stream", "shs_repr_epoch", "shs_ckpt_",
    "shs_expected_", "shs_corpus_", "shs_strict", "shs_ema_gate", "shs_structure_moves",
    "shs_forget_", "shs_schedule_unit",
)


def is_target_key(key):
    return key in TARGET_KEYS or key.startswith(TARGET_PREFIXES)


def sha256_of(path, chunk=1 << 20):
    """SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(obj):
    """Plain JSON types only, so checkpoints stay loadable with weights_only=True."""
    return json.loads(json.dumps(obj, default=str))


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


def backfill_optional_state(module, state, label):
    """Fill state keys the checkpoint lacks from the constructed module."""
    suffixes = getattr(dreamer, "OPTIONAL_STATE_SUFFIXES", ())
    target = module.state_dict()
    filled = sorted(k for k, v in target.items()
                    if k not in state and k.endswith(suffixes) and torch.is_tensor(v))
    for key in filled:
        state[key] = target[key].detach().clone()
    if filled:
        warnings.warn(f"{label}: checkpoint has no {filled}; using their initial values")
    return filled


def strict_load(module, state, label):
    target = module.state_dict()
    missing = sorted(set(target) - set(state))
    extra = sorted(set(state) - set(target))
    shapes = [f"{k}: saved {tuple(v.shape)}, constructed {tuple(target[k].shape)}"
              for k, v in state.items() if k in target and torch.is_tensor(v)
              and torch.is_tensor(target[k]) and v.shape != target[k].shape]
    if missing or extra or shapes:
        raise ValueError(
            f"Incompatible {label}: the checkpoint differs from the constructed model in a "
            "way the setup adoption does not cover (see the adopted-setup table above).\n"
            f"Missing: {missing[:10]}\nUnexpected: {extra[:10]}\n"
            + "\n".join(shapes[:10]))
    module.load_state_dict(state, strict=True)


def checkpoint_kind(state):
    return "aime" if "_wm.dynamics.regime.ema_trans_counts" in state else "dreamer"



def _merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


def resolve_presets(names, config_json=None):
    catalog = YAML(typ="safe").load((pathlib.Path(__file__).parent / "configs.yaml").read_text())
    merged = {}
    for name in ["defaults", *names]:
        if name not in catalog:
            raise ValueError(f"Unknown config preset {name!r} (not in configs.yaml)")
        _merge(merged, catalog[name])
    if config_json:
        _merge(merged, json.loads(pathlib.Path(config_json).read_text()))
    return merged


def namespace_from(merged, argv):
    cfgparser = argparse.ArgumentParser()
    for name, value in merged.items():
        parse = tools.args_type(value)
        cfgparser.add_argument(f"--{name}", type=parse, default=parse(value))
    return cfgparser.parse_args(argv)


def explicit_keys(argv):
    """Config keys the user typed on the command line (they beat adopted values)."""
    return {tok[2:].split("=", 1)[0] for tok in argv if tok.startswith("--")}


def _get(config, key):
    top, _, sub = key.partition(".")
    value = getattr(config, top, None)
    if sub:
        return value.get(sub) if isinstance(value, dict) else None
    return value


def _assign(config, key, value):
    top, _, sub = key.partition(".")
    if sub:
        getattr(config, top)[sub] = value
        return
    current = getattr(config, top, None)
    if isinstance(current, tuple) and isinstance(value, list):
        value = tuple(value)
    setattr(config, top, value)


def _same(a, b):
    a, b = jsonable(a), jsonable(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b



def infer_architecture(state, config):
    """{config key: value} fixed by the checkpoint's tensor shapes."""
    def shape(key):
        value = state.get(key)
        return tuple(value.shape) if torch.is_tensor(value) else None

    def mlp(prefix, name):
        pat = re.compile(re.escape(prefix) + rf"layers\.{name}_linear(\d+)\.weight")
        idx = [int(m.group(1)) for m in map(pat.fullmatch, state) if m]
        if not idx:
            return None
        return len(idx), shape(f"{prefix}layers.{name}_linear0.weight")[0]

    out = {}
    aime = checkpoint_kind(state) == "aime"
    dyn = "_wm.dynamics."
    W, gru = shape(dyn + "W"), shape(dyn + "_cell.layers.GRU_linear.weight")
    if W is not None:
        out["dyn_deter"], out["initial"] = W[-1], "learned"
    elif gru is not None:
        out["dyn_deter"] = gru[0] // 3
        if getattr(config, "initial", None) == "learned":
            out["initial"] = "zeros"
    hidden = shape(dyn + "_img_in_layers.0.weight")
    if hidden is not None:
        out["dyn_hidden"] = hidden[0]
    stat = shape(dyn + "_imgs_stat_layer.weight")
    if aime:
        out["dyn_discrete"] = 0
        out["dyn_stoch"] = shape(dyn + "regime.z0")[0]
    elif stat is not None:
        disc = int(getattr(config, "dyn_discrete", 0) or 0)
        if disc > 0 and stat[0] % disc == 0:
            out["dyn_stoch"] = stat[0] // disc
        elif disc == 0:
            out["dyn_stoch"] = stat[0] // 2
    for key, (prefix, name) in {"reward_head": ("_wm.heads.reward.", "Reward"),
                                "cont_head": ("_wm.heads.cont.", "Cont"),
                                "actor": ("_task_behavior.actor.", "Actor"),
                                "critic": ("_task_behavior.value.", "Value")}.items():
        found = mlp(prefix, name)
        if found is not None:
            out[f"{key}.layers"] = found[0]
            if key == "reward_head":
                out["units"] = found[1]
    enc = shape("_wm.encoder._cnn.layers.0.weight")
    if enc is not None:
        out["encoder.cnn_depth"], out["encoder.kernel_size"] = enc[0], enc[-1]
    pat = re.compile(r"_wm\.heads\.decoder\._cnn\.layers\.(\d+)\.weight")
    convs = [(int(m.group(1)), m.group(0)) for m in map(pat.fullmatch, state)
             if m and state[m.group(0)].dim() == 4]
    if convs:
        last = shape(max(convs)[1])
        out["decoder.cnn_depth"], out["decoder.kernel_size"] = last[0], last[-1]
    if not aime:
        return out

    reg = dyn + "regime."
    L, deter = out["dyn_stoch"], out["dyn_deter"]
    out["shs_K"] = shape(reg + "ema_trans_counts")[0]
    proj = shape(reg + "P.weight")
    Hp = proj[0] if proj is not None else deter
    out["shs_proj_dim"] = Hp
    shared = (reg + "regimes.Cmean") in state
    out["shs_shared_carry"] = shared
    ufac = shape(reg + "regimes.Ufac")
    out["shs_q_rank"] = ufac[-1] if ufac is not None else 0
    if shared:
        A = shape(reg + "regimes.lam0_diag")[0] - L - 1
    else:
        A = shape(reg + "regimes.M")[-1] - L - Hp - 1
    out["shs_action_dim"] = A
    stick = shape(reg + "P_stick.weight")
    rdim = stick[0] if stick is not None else deter
    out["shs_rstick_dim"] = rdim
    feat = shape(reg + "rstick.m0")[0] - 1
    preferred = str(getattr(config, "shs_gate_input", "latent"))
    options = []
    for gate in dict.fromkeys([preferred, "latent", "carry"]):
        r_act = feat - (L if gate == "latent" else rdim)
        if r_act == 0 or (A > 0 and r_act == A):
            options.append((gate, r_act > 0))
    if not options:
        raise ValueError(f"Cannot interpret the checkpoint's gate features (width {feat}, "
                         f"stoch {L}, rstick_dim {rdim}, action_dim {A})")
    out["shs_gate_input"], out["shs_rstick_use_action"] = options[0]
    return out


def _effective(key, value, config, arch):
    """Value as the constructor would use it (None/oversized widths collapse)."""
    deter = int(arch.get("dyn_deter", getattr(config, "dyn_deter", 0)))
    if key == "shs_proj_dim":
        return deter if value is None or int(value) >= deter else int(value)
    if key == "shs_rstick_dim":
        if value is None:
            return _effective("shs_proj_dim", getattr(config, "shs_proj_dim", None), config, arch)
        return min(deter, int(value))
    if key == "shs_action_dim" and int(value) == -1:
        return int(getattr(config, "num_actions", -1) or -1)
    return value


def _recipe_mismatches(state, cfg):
    """Architecture keys on which a candidate recipe disagrees with the checkpoint."""
    if (checkpoint_kind(state) == "aime") != bool(getattr(cfg, "use_shs", False)):
        return ["use_shs: recipe and checkpoint are different model kinds"]
    arch = infer_architecture(state, cfg)
    return [f"{k}: recipe {_get(cfg, k)!r}, checkpoint {v!r}" for k, v in arch.items()
            if k != "shs_K"
            and not _same(_effective(k, _get(cfg, k), cfg, arch), v)]


def select_recipe(checkpoint, specs):
    """Pick the source launch recipe consistent with a checkpoint that stores no config."""
    state = canonical_state(checkpoint["agent_state_dict"])
    tried = []
    for spec in specs:
        names = [n for n in spec.split(",") if n]
        merged = resolve_presets(names)
        cfg = namespace_from(merged, [])
        bad = _recipe_mismatches(state, cfg)
        if not bad:
            print(f"[transfer] source recipe: {spec} (architecture matches the checkpoint)")
            return dict(spec=spec, names=names, merged=merged, cfg=cfg, consistent=True)
        tried.append((spec, bad))
    detail = "; ".join(f"{spec}: {', '.join(bad[:4])}" for spec, bad in tried)
    warnings.warn(
        "No source recipe matches the checkpoint's architecture (" + detail + "). The "
        "architecture is still adopted from tensor shapes, but hyperparameters the checkpoint "
        "does not record (shs_ema_tau, shs_b0, ...) fall back to the target presets. Add the "
        "launch overrides the source was trained with as a preset and pass it in --source-recipes.")
    names = [n for n in specs[0].split(",") if n]
    return dict(spec=specs[0], names=names, merged=None,
                cfg=namespace_from(resolve_presets(names), []), consistent=False,
                mismatches={spec: bad for spec, bad in tried})


def adopt_setup(config, checkpoint, explicit, *, label, k_from="counts", recipe=None,
                resume=False):
    """Rewrite `config` so the constructed model matches `checkpoint`; returns what was adopted."""
    state = canonical_state(checkpoint["agent_state_dict"])
    is_aime = checkpoint_kind(state) == "aime"
    if is_aime != bool(getattr(config, "use_shs", False)):
        raise ValueError("Load an AIME checkpoint into AIME (use_shs: True), and a Dreamer "
                         "checkpoint into Dreamer.")
    rows, conflicts = [], []
    saved, origin = checkpoint.get("config"), "stored config"
    if not isinstance(saved, dict) and recipe is not None and recipe.get("merged") is not None:
        saved, origin = recipe["merged"], f"source recipe {recipe['names'][-1]}"
    if resume and not isinstance(saved, dict):
        warnings.warn("The target checkpoint does not record its configuration (written by an "
                      "older runner); resuming with the current presets and flags unchecked.")
    if isinstance(saved, dict):
        for key, value in saved.items():
            run_control = key in RUN_CONTROL_KEYS or key.startswith(RUN_CONTROL_PREFIXES)
            if (not hasattr(config, key) or _same(getattr(config, key), value)
                    or (is_target_key(key) and not resume) or (resume and run_control)):
                continue
            if key in explicit:
                if resume:
                    conflicts.append(f"--{key} {getattr(config, key)!r} (run was started with {value!r})")
                else:
                    rows.append(dict(key=key, value=getattr(config, key), was=value,
                                     source="explicit CLI value kept", was_label="source had"))
                continue
            current = getattr(config, key)
            if isinstance(value, dict) and isinstance(current, dict):
                for sub, sub_value in value.items():
                    if not _same(current.get(sub), sub_value):
                        rows.append(dict(key=f"{key}.{sub}", value=sub_value,
                                         was=current.get(sub), source=origin))
                        current[sub] = copy.deepcopy(sub_value)
                continue
            rows.append(dict(key=key, value=value, was=current, source=origin))
            _assign(config, key, copy.deepcopy(value))
    if conflicts:
        raise ValueError("Resume would change the experiment this target run was started with:\n  "
                         + "\n  ".join(conflicts) + "\nDrop those flags, or start a fresh run in a "
                         "new --logdir. (steps, eval/log cadence and device may change on resume.)")
    if isinstance(saved, dict) and "action_repeat" in saved \
            and not _same(saved["action_repeat"], config.action_repeat):
        warnings.warn(f"Source used action_repeat={saved['action_repeat']}, target uses "
                      f"{config.action_repeat}: the transferred dynamics run at a different timescale.")
    arch = infer_architecture(state, config)
    if k_from == "config" and isinstance(saved, dict) and "shs_K" in saved:
        arch["shs_K"] = int(saved["shs_K"])
    for key, value in arch.items():
        current = _get(config, key)
        if _same(_effective(key, current, config, arch), value):
            continue
        if key in explicit:
            warnings.warn(f"Ignoring --{key} {current}: {label} was built with {key}={value}.")
        rows.append(dict(key=key, value=value, was=current, source="tensor shapes"))
        _assign(config, key, value)
    width = max([len(r["key"]) for r in rows] + [10])
    print(f"[transfer] setup adopted from {label}:")
    if not rows:
        print("[transfer]   (presets/CLI already match the checkpoint)")
    for r in rows:
        print(f"[transfer]   {r['key']:<{width}} = {r['value']!r:<8} "
              f"({r.pop('was_label', 'config had')} {r['was']!r}; {r['source']})")
    return jsonable(rows)


def _phase_schedule(curriculum, grad_per_env):
    """Mirror SHSRSSM's conversion of *_env schedule keys to gradient steps."""
    phases = []
    for p in curriculum or ():
        q = dict(p)
        if "until_env" in q:
            q["until"] = (-1 if float(q["until_env"]) < 0
                          else max(1, int(round(float(q["until_env"]) * grad_per_env))))
        phases.append(q)
    return phases


def _logged_live_state(ckpt_path):
    """Last live gate state logged beside the checkpoint (needs shs_diag_log)."""
    path = pathlib.Path(ckpt_path).parent / "metrics.jsonl"
    if not path.exists():
        return None
    last = None
    with path.open() as f:
        for line in f:
            if '"shs_recurrent_live"' in line:
                last = line
    if last is None:
        return None
    rec = json.loads(last)
    return dict(env_step=rec.get("step"), recurrent=bool(rec["shs_recurrent_live"]),
                kappa=float(rec.get("shs_kappa_live", float("nan"))),
                phase=rec.get("shs_curriculum_phase"))


def source_live_state(checkpoint, source_cfg, source_label, ckpt_path):
    """Gate/kappa/b0-latch/clock of the source model at the moment it was saved."""
    live = checkpoint.get("shs_live")
    if isinstance(live, dict) and "recurrent" in live:
        out = dict(live)
        out["provenance"] = "recorded in checkpoint"
        return out
    state = canonical_state(checkpoint["agent_state_dict"])
    dyn_extra = state.get("_wm.dynamics._extra_state") or {}
    step = int(dyn_extra.get("shs_step") or checkpoint.get("update_count", 0) or 0)
    gpe = float(source_cfg.train_ratio) / (float(source_cfg.batch_size) * float(source_cfg.batch_length))
    if str(getattr(source_cfg, "shs_schedule_unit", "agent")) == "frame":
        gpe /= float(getattr(source_cfg, "action_repeat", 1) or 1)
    phases = _phase_schedule(source_cfg.shs_curriculum, gpe)
    if step <= 0:
        warnings.warn("Checkpoint records no SHS step; assuming the final source curriculum phase.")
        step = max([int(p.get("until", -1)) for p in phases] + [0]) + 1
    recurrent = bool(source_cfg.shs_recurrent)
    kappa = 0.0 if recurrent else float(source_cfg.shs_kappa)
    idx = -1
    if phases:
        idx = next((i for i, p in enumerate(phases)
                    if int(p.get("until", -1)) < 0 or step < int(p["until"])), len(phases) - 1)
        for p in phases[:idx + 1]:
            if "recurrent" in p:
                recurrent = bool(p["recurrent"])
            if recurrent:
                kappa = 0.0
            elif "kappa" in p:
                kappa = float(p["kappa"])
    b0_at = max(1, int(phases[0].get("until", 0))) if phases else 1
    calibrated = (not bool(source_cfg.shs_calibrate_b0)) or step >= b0_at
    out = dict(recurrent=recurrent, kappa=kappa, b0_calibrated=calibrated, shs_step=step,
               phase=idx, K=int(state["_wm.dynamics.regime.ema_trans_counts"].shape[0]),
               provenance=f"reconstructed from {source_label} curriculum at grad step {step}")
    pg_init = (state.get("_wm.dynamics.regime._extra_state") or {}).get("pg_init")
    if pg_init and not recurrent:
        warnings.warn("The checkpoint's persistence gate has been trained (pg_init) but the "
                      f"reconstructed source phase {idx} has the gate off; check --source-recipes.")
    logged = _logged_live_state(ckpt_path)
    if logged is not None:
        out["logged"] = logged
        kappa_ok = recurrent or math.isnan(logged["kappa"]) or math.isclose(logged["kappa"], kappa)
        if logged["recurrent"] != recurrent or not kappa_ok:
            warnings.warn(f"Source metrics.jsonl last logged recurrent={logged['recurrent']} "
                          f"kappa={logged['kappa']} (env step {logged['env_step']}), but the "
                          f"reconstruction gives recurrent={recurrent} kappa={kappa}. The log can "
                          "lag the checkpoint by up to log_every; otherwise check --source-recipes.")
    return out


def check_action_dim(checkpoint, config):
    state = canonical_state(checkpoint["agent_state_dict"])
    w = state.get("_wm.dynamics._img_in_layers.0.weight")
    if not torch.is_tensor(w):
        return
    stoch = int(config.dyn_stoch) * (int(config.dyn_discrete) or 1)
    saved = int(w.shape[1]) - stoch
    if saved != int(config.num_actions):
        raise ValueError(f"The checkpoint was trained with {saved} actions but the target "
                         f"environment {config.task} has {config.num_actions}.")



def configure_loaded_shs(agent, checkpoint, live, *, resume=False, clock="source"):
    wm, dynamics = agent._wm, agent._wm.dynamics
    if not hasattr(dynamics, "regime"):
        return None
    regime = dynamics.regime
    runtime = checkpoint.get("transfer_runtime") or {}
    if resume:
        state = {k: v for k, v in (live or {}).items() if k in ("recurrent", "kappa")}
        state.update(runtime.get("target_live") or {})
        state.setdefault("shs_step", dynamics._shs_step)
        state.setdefault("b0_calibrated", runtime.get("b0_calibrated", True))
        state.setdefault("recurrent", regime.recurrent)
        state.setdefault("kappa", regime.hdp.kappa)
    else:
        state = dict(live)
        if clock == "target":
            state["shs_step"] = 0
    dynamics._shs_step = int(state["shs_step"])
    dynamics._shs_b0_calibrated = bool(state["b0_calibrated"])
    first = getattr(dynamics, "_move_first", None)
    if resume:
        if "move_first_pending" in state:
            dynamics._move_first_pending = bool(state["move_first_pending"])
    else:
        dynamics._move_first_pending = first is not None
        headroom = getattr(dynamics, "_move_K_headroom", None)
        dynamics._move_K_cap = None if headroom is None else int(regime.K) + int(headroom)
    regime.recurrent = bool(state["recurrent"])
    regime.hdp.kappa = 0.0 if regime.recurrent else float(state["kappa"])
    if dynamics._curriculum:
        dynamics._curr_phase = next((i for i, p in enumerate(dynamics._curriculum)
                                     if int(p.get("until", -1)) < 0
                                     or dynamics._shs_step < int(p["until"])),
                                    len(dynamics._curriculum) - 1)
    dynamics._apply_curriculum()
    store = regime.stat_store
    if store is not None:
        if store.mode != "legacy_ema":
            raise ValueError("This transfer runner supports AIME replay EMA, not fixed-corpus/strict-stream stores.")
        store.ema_tau = float(regime.ema_tau)
    if not resume:
        dynamics._pending = None
        dynamics._next_batch_id = None
        regime._drift, regime._retention_last = {}, {}
        wm._shs_reservoir = []
        wm._shs_target_state = None
    elif checkpoint.get("shs_target_state") is not None:
        wm._shs_target_state = checkpoint["shs_target_state"]
    teacher, probe = runtime.get("teacher"), runtime.get("teacher_probe")
    cfg = dynamics._shs_teacher_cfg
    if resume and teacher is not None and probe is not None and cfg.get("enabled"):
        from shs_rssm.teacher import EMATeacher
        device = next(wm.encoder.parameters()).device
        dynamics._teacher = EMATeacher(wm.encoder, tau=cfg["tau"], drift_tol=cfg["drift_tol"],
                                       probe={k: v.to(device) for k, v in probe.items()})
        dynamics._teacher.load_state_dict(teacher)
        print(f"[transfer] EMA teacher restored (version {dynamics._teacher.version})")
    elif cfg.get("enabled"):
        dynamics._teacher = None
        print("[transfer] EMA teacher: rebuilt from the loaded encoder on the first target update")
    print(f"[transfer] SHS clock={clock} step={dynamics._shs_step} "
          f"phase={dynamics._curr_phase}/{len(dynamics._curriculum) - 1} "
          f"recurrent gate={regime.recurrent} kappa={regime.hdp.kappa} K={regime.K}")
    first_note = (f"first sweep at grad step {first}, then "
                  if first is not None and getattr(dynamics, "_move_first_pending", False) else "")
    print(f"[transfer] SHS moves: {first_note}every {dynamics._shs_move_every} grad steps, "
          f"birth={dynamics._shs_move_birth} split={dynamics._shs_move_split} merge/delete="
          f"{dynamics._shs_move_every > 0} window={getattr(dynamics, '_shs_move_version', 'repr')}; "
          f"b0 calibrated={dynamics._shs_b0_calibrated} (b0={regime.regimes.b0.flatten()[:1].tolist()}), "
          f"statistic tau={regime.ema_tau}")
    if dynamics._shs_move_every > 0 and getattr(dynamics, "_shs_move_version", "repr") == "repr":
        warnings.warn("Structure moves are scheduled with shs_move_version=repr; in replay-EMA "
                      "training every live sweep is skipped. Use shs_move_version: teacher.")
    return dynamics.live_state() if hasattr(dynamics, "live_state") else None


def initialize_agent(agent, checkpoint, mode, live, *, resume=False,
                     clock="source"):
    """Load each shared module once, so aliases cannot undo partial transfer."""
    state = canonical_state(checkpoint["agent_state_dict"])
    wm_state = copy.deepcopy(component(state, "_wm."))
    if mode == "world_model":
        reward_state = agent._wm.heads["reward"].state_dict()
        for key in list(wm_state):
            if key.startswith("heads.reward."):
                wm_state[key] = reward_state[key[len("heads.reward."):]]
    backfill_optional_state(agent._wm, wm_state, "world model")
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
    start = configure_loaded_shs(agent, checkpoint, live, resume=resume, clock=clock)
    if resume:
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._update_count = int(checkpoint.get("update_count", 0))
        for attr, key in [("_episodes_seen", "episodes_seen"),
                          ("_last_consolidate_ep", "last_consolidate_ep"),
                          ("_shs_epoch_step", "shs_epoch_step")]:
            if hasattr(agent, attr):
                setattr(agent, attr, int(checkpoint.get(key, 0)))
        agent._task_behavior._updates = int(checkpoint.get("behavior_updates", 0))
        agent._should_pretrain._once = not bool(
            (checkpoint.get("progress") or {}).get("pretrained", True))
    return start


def keep_metrics_through(logdir, last_step):
    path = logdir / "metrics.jsonl"
    if not path.exists():
        return
    kept, moved = [], []
    for line in path.read_text().splitlines(keepends=True):
        try:
            step = json.loads(line).get("step", 0)
        except ValueError:
            step = 0
        (kept if step <= last_step else moved).append(line)
    if moved:
        with (logdir / "metrics_after_checkpoint.jsonl").open("a") as f:
            f.writelines(moved)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text("".join(kept))
        os.replace(temporary, path)
        print(f"[transfer] moved {len(moved)} metric records written after the checkpoint to metrics_after_checkpoint.jsonl")


def save_checkpoint(agent, path, config, config_snapshot, metadata, progress=None, resumes=0):
    dynamics = agent._wm.dynamics
    teacher = getattr(dynamics, "_teacher", None)
    probe = getattr(teacher, "probe", None) if teacher is not None else None
    live = dynamics.live_state() if hasattr(dynamics, "live_state") else None
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
        "config": config_snapshot,
        "shs_live": live,
        "transfer_metadata": metadata,
        "transfer_runtime": {
            "b0_calibrated": bool(getattr(dynamics, "_shs_b0_calibrated", False)),
            "teacher": teacher.state_dict() if teacher is not None else None,
            "teacher_probe": ({k: v.detach().cpu() for k, v in probe.items()}
                              if isinstance(probe, dict) else None),
            "target_live": live,
        },
        "progress": dict(progress or {}),
        "episode_files": sorted(p.name for p in pathlib.Path(config.traindir).glob("*.npz")),
        "metrics_bytes": (path.parent / "metrics.jsonl").stat().st_size
        if (path.parent / "metrics.jsonl").exists() else 0,
        "rng": dreamer.rng_state(),
        "resumes": int(resumes),
    }
    dreamer.atomic_save(payload, path)



def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=[])
    parser.add_argument("--config-json", help="Resolved source configuration; remaining flags override it")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--transfer-mode", choices=["full", "world_model", "scratch"], default="full")
    parser.add_argument("--source-task", default="dmc_walker_walk")
    parser.add_argument("--source-recipes", nargs="+", default=list(DEFAULT_SOURCE_RECIPES),
                        help="Candidate launch recipes (comma-separated preset lists) for "
                             "checkpoints that do not store their config; the one whose "
                             "architecture matches the checkpoint supplies the hyperparameters "
                             "the weights do not record and the source curriculum")
    parser.add_argument("--setup-from", default="",
                        help="scratch only: adopt this checkpoint's architecture, not its weights")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume this runner's existing target run")
    parser.add_argument("--prefill-policy", choices=["random", "transferred"], default="random",
                        help="policy that collects the initial replay: uniform random actions, or "
                             "the loaded actor sampling its own actions")
    parser.add_argument("--save-every", type=int, default=2000,
                        help="also checkpoint every this many env frames (0: only at evaluations); "
                             "bounds the work lost to a hard kill such as preemption")
    parser.add_argument("--stop-check-every", type=int, default=100,
                        help="agent steps between checks for SIGUSR1/SIGTERM; on either, the runner "
                             f"checkpoints and exits with status {EXIT_RESUME} (resume in a new slice)")
    args, remaining = parser.parse_known_args(argv)
    config = namespace_from(resolve_presets(args.configs, args.config_json), remaining)
    config.compile = False
    args.explicit = explicit_keys(remaining)
    if not config.logdir:
        raise ValueError("Supply a separate --logdir for this target experiment")
    if config.task not in ("dmc_walker_walk", "dmc_walker_run", "dmc_walker_stand"):
        raise ValueError("Supported tasks: dmc_walker_walk, dmc_walker_run, dmc_walker_stand")
    if config.expl_behavior != "greedy":
        raise ValueError("This runner requires expl_behavior: greedy")
    if any(getattr(config, k, None) for k in ("traindir", "evaldir", "offline_traindir", "offline_evaldir")):
        raise ValueError("Use target-local replay: omit traindir/evaldir/offline directory overrides")
    if getattr(config, "transfer_curriculum_clock", "source") not in ("source", "target"):
        raise ValueError("transfer_curriculum_clock must be 'source' or 'target'")
    if getattr(config, "use_shs", False):
        if getattr(config, "shs_global_source", "replay_ema") != "replay_ema":
            raise ValueError("This runner's AIME protocol requires shs_global_source=replay_ema")
        if getattr(config, "shs_online_mode", "ema") not in ("ema", "live_ema", "legacy_ema"):
            raise ValueError("Choose an AIME EMA configuration, not a fixed-corpus/strict-stream mode")
        if config.shs_stream_episodes:
            raise ValueError("Disable completed-episode streaming for this replay-EMA protocol")
    if args.resume and (args.checkpoint or args.setup_from):
        raise ValueError("Use --resume alone; the target run's own checkpoint defines its setup")
    if not args.resume and ((args.transfer_mode == "scratch") == bool(args.checkpoint)):
        raise ValueError("Supply --checkpoint for full/world_model; omit it for scratch")
    if args.setup_from and args.transfer_mode != "scratch":
        raise ValueError("--setup-from is for the scratch arm; transfer arms adopt --checkpoint")
    return args, config


def run(args, config):
    logdir = pathlib.Path(config.logdir).expanduser().resolve()
    if not args.resume and logdir.exists() and any(logdir.iterdir()):
        raise FileExistsError(f"Fresh transfer requires an empty target directory: {logdir}")
    clock = getattr(config, "transfer_curriculum_clock", "source")
    source = logdir / "latest.pt" if args.resume else pathlib.Path(args.checkpoint).expanduser()
    checkpoint, live, adopted, setup_checkpoint, recipe = None, None, [], None, None

    def recipe_for(ck):
        if isinstance(ck.get("config"), dict):
            return None
        return select_recipe(ck, args.source_recipes)

    if args.resume or args.checkpoint:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        if args.resume and "target_env_steps" not in checkpoint:
            raise ValueError("--resume requires a checkpoint written by this transfer runner")
        if args.resume and checkpoint["task"] != config.task:
            raise ValueError("Resume must keep the target task unchanged")
        if not args.resume and checkpoint.get("task", args.source_task) != args.source_task:
            raise ValueError("--source-task disagrees with checkpoint metadata")
        recipe = None if args.resume else recipe_for(checkpoint)
        adopted = adopt_setup(config, checkpoint, args.explicit, recipe=recipe, resume=args.resume,
                              label="target checkpoint" if args.resume else f"source {source.name}")
        clock = getattr(config, "transfer_curriculum_clock", "source")
        if getattr(config, "use_shs", False):
            if args.resume:
                live = (checkpoint.get("transfer_metadata") or {}).get("source_live")
            else:
                saved = checkpoint.get("config")
                if isinstance(saved, dict):
                    source_cfg, source_label = namespace_from(saved, []), "stored config"
                else:
                    source_cfg, source_label = recipe["cfg"], recipe["spec"]
                live = source_live_state(checkpoint, source_cfg, source_label, source)
                print(f"[transfer] source live state ({live['provenance']}): "
                      f"recurrent={live['recurrent']} kappa={live['kappa']} "
                      f"b0_calibrated={live['b0_calibrated']} phase={live['phase']} K={live['K']}")
    elif args.setup_from:
        setup_checkpoint = pathlib.Path(args.setup_from).expanduser()
        ck = torch.load(setup_checkpoint, map_location="cpu", weights_only=False)
        recipe = recipe_for(ck)
        adopted = adopt_setup(config, ck, args.explicit, label=f"setup {setup_checkpoint.name}",
                              k_from="config", recipe=recipe)
        del ck
    if args.resume and getattr(config, "use_shs", False):
        _stored = checkpoint.get("config") if isinstance(checkpoint.get("config"), dict) else {}
        _meta = checkpoint.get("transfer_metadata") or {}
        if "shs_structure_moves" not in _stored and _meta.get("fixed_regimes") is False:
            config.shs_structure_moves = True
    if getattr(config, "use_shs", False) and not getattr(config, "shs_structure_moves", False):
        config.shs_consolidate_every_episodes = 0
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    raw_config = jsonable(vars(config))
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
                    replay="fresh target replay", prefill_policy=args.prefill_policy,
                    structure_moves=bool(getattr(config, "shs_structure_moves", False)),
                    forget_mode=str(getattr(config, "shs_forget_mode", "fixed")),
                    drift_rate_posterior=("resumed" if args.resume else "fresh target posterior"),
                    move_commit=str(getattr(config, "shs_move_commit", "remap")),
                    imag_mode=str(getattr(config, "shs_imag_mode", "actor_moment")),
                    curriculum_clock=clock, adopted_setup=adopted,
                    source_live=jsonable(live) if live else None,
                    setup_from=str(setup_checkpoint.resolve()) if setup_checkpoint else None,
                    source_recipe=(None if recipe is None else jsonable(
                        {k: recipe.get(k) for k in ("spec", "consistent", "mismatches")})),
                    source_cost_included_in_target_steps=False)
    if checkpoint and not args.resume:
        metadata["source_sha256"] = sha256_of(source)
    if args.resume:
        metadata = checkpoint["transfer_metadata"]
    (logdir / "transfer_config.json").write_text(json.dumps(raw_config, indent=2, default=str) + "\n")
    if not (logdir / "launch_config.json").exists():
        (logdir / "launch_config.json").write_text(json.dumps(raw_config, indent=2, default=str) + "\n")
    (logdir / "transfer_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    start_step = int(checkpoint["target_env_steps"]) if args.resume else 0
    if start_step > budget and not args.evaluate_only:
        raise ValueError("The requested total target budget is below the resumed checkpoint step")
    resumes = int(checkpoint.get("resumes", 0)) + 1 if args.resume else 0
    if args.resume and "episode_files" in checkpoint:
        dreamer.reconcile_replay(config.traindir, checkpoint["episode_files"], logdir)
        if "metrics_bytes" in checkpoint:
            dreamer.truncate_metrics(logdir, checkpoint["metrics_bytes"])
        else:
            keep_metrics_through(logdir, start_step)
    logger = tools.Logger(logdir, start_step, purge_step=start_step + 1 if args.resume else None)
    train_eps = tools.load_episodes(config.traindir, limit=config.dataset_size) if args.resume else collections.OrderedDict()
    env_config = config
    if resumes:
        env_config = argparse.Namespace(**vars(config))
        env_config.seed = dreamer.derived_seed(config.seed, resumes, 1)
    eval_eps = collections.OrderedDict()
    train_envs, eval_envs = [], []
    try:
        wrap = (lambda env: Parallel(env, "process")) if config.parallel else Damy
        train_envs = [wrap(dreamer.make_env(env_config, "train", i)) for i in range(nenv)]
        eval_envs = [wrap(dreamer.make_env(env_config, "eval", 10000 + i)) for i in range(nenv)]
        config.num_actions = int(train_envs[0].action_space.shape[0])
        if checkpoint:
            check_action_dim(checkpoint, config)
        agent = make_agent(train_envs[0].observation_space, train_envs[0].action_space,
                           config, logger,
                           dreamer.make_dataset(train_eps, config, dreamer.derived_seed(config.seed, resumes, 2) if resumes else 0),
                           train_eps)
        if checkpoint:
            start = initialize_agent(agent, checkpoint, "full" if args.resume else args.transfer_mode,
                                     live, resume=args.resume,
                                     clock=clock)
            if start is not None and not args.resume:
                metadata["target_start_live"] = jsonable(start)
                (logdir / "transfer_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            if args.resume and checkpoint.get("rng") is not None:
                dreamer.set_rng_state(checkpoint["rng"])
        agent.requires_grad_(False)

        def evaluate():
            if config.eval_episode_num <= 0:
                return
            tools.simulate(functools.partial(agent, training=False), eval_envs, eval_eps,
                           config.evaldir, logger, is_eval=True, episodes=config.eval_episode_num)

        E = int(config.eval_every)
        S = max(0, int(args.save_every) // (repeat * nenv) * nenv)
        poll = max(nenv, int(args.stop_check_every) // nenv * nenv)
        default_anchor = min(config.steps, int(np.ceil(int(config.prefill) / nenv)) * nenv)
        if args.resume:
            saved = checkpoint.get("progress") or {}
            anchor = int(saved.get("eval_anchor", default_anchor))
            last_eval = int(saved.get("last_eval_step", agent._step))
        else:
            anchor, last_eval = default_anchor, -1

        def due_eval(step):
            """Latest evaluation point <= step on the grid anchor + k*E (k >= 1), plus the end."""
            if step >= config.steps:
                return config.steps
            return anchor + (step - anchor) // E * E if step >= anchor + E else 0

        def next_eval(step):
            k = 1 if step < anchor + E else (step - anchor) // E + 1
            return min(anchor + k * E, config.steps)

        def evaluate_at_step():
            nonlocal last_eval
            evaluate()
            last_eval = int(agent._step)

        def checkpoint_now():
            save_checkpoint(agent, logdir / "latest.pt", config, raw_config, metadata, progress=dict(
                last_eval_step=last_eval, eval_anchor=anchor,
                pretrained=not agent._should_pretrain._once), resumes=resumes)

        if not args.resume or last_eval < due_eval(agent._step):
            evaluate_at_step()
        if args.evaluate_only:
            return 0
        stop = {"signal": None}

        def request_stop(signum, _frame):
            stop["signal"] = signum

        for sig in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(sig, request_stop)
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

            def transferred_agent(obs, done, recurrent):
                with torch.no_grad():
                    return agent._policy(obs, recurrent, training=True)

            prefill_agent = transferred_agent if args.prefill_policy == "transferred" else random_agent
            state = tools.simulate(prefill_agent, train_envs, train_eps, config.traindir,
                                   logger, limit=config.dataset_size, steps=prefill)
            agent._step += prefill
            logger.step = agent._step * repeat
        upcoming_eval = next_eval(agent._step)
        upcoming_save = (agent._step // S + 1) * S if S else None
        while agent._step < config.steps:
            if stop["signal"] is not None:
                checkpoint_now()
                logger.write()
                print(f"[transfer] {signal.Signals(stop['signal']).name}: checkpointed at env step "
                      f"{logger.step}; exit {EXIT_RESUME} to resume in a new slice", flush=True)
                return EXIT_RESUME
            limit = min(upcoming_eval, upcoming_save or config.steps)
            state = tools.simulate(agent, train_envs, train_eps, config.traindir, logger,
                                   limit=config.dataset_size,
                                   steps=min(poll, limit - agent._step), state=state)
            saved_here = False
            if upcoming_save and agent._step >= upcoming_save:
                checkpoint_now()
                saved_here = True
                upcoming_save = (agent._step // S + 1) * S
            if agent._step >= upcoming_eval:
                if not saved_here:
                    checkpoint_now()
                if stop["signal"] is None:
                    evaluate_at_step()
                upcoming_eval = next_eval(agent._step)
        logger.write()
        return 0
    finally:
        for env in train_envs + eval_envs:
            try:
                env.close()
            except Exception:
                pass
        logger._writer.close()


if __name__ == "__main__":
    sys.exit(run(*parse_config()) or 0)

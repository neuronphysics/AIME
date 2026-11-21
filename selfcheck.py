#!/usr/bin/env python3
"""Runs an arm through every phase of its schedule on the training device, in a few minutes.

    python selfcheck.py --configs dmc_vision shs_rebuttal rb_full --task dmc_walker_walk

It takes the arguments of dreamer.py (the launchers pass the training command's own). It
builds the world model and the actor-critic with the configured sizes on the configured
device and trains them on random images with the schedule compressed, so that calibration,
re-seed, structure sweeps with births and splits, and the freeze all happen within about 70
updates. An arm with births or splits then gets a few sweeps in which growth candidates are
accepted whatever their score, so that the code behind an accepted birth or split runs as well
(random data rarely leads to one); the drift test that empties the sweep buffer is off, since
an untrained encoder on random images would trip it all the time. Then it runs the logging
diagnostics and the open-loop error. Last, the state of the world model and the runtime state
of the regime head are written to a file and restored into a fresh model, which is compared
with what was saved (tensors and K) and updated once more.

What it does not cover: learning (the data are random); the environments and the replay; the
actor's and the optimizers' state at a resume (dreamer.py's own checkpoint does that). No
environment is created; with --selfcheck_report only that file is written.

Verdicts, printed and written to --selfcheck_report FILE:
  [selfcheck] OK          every part this arm has was reached                      exit 0
  [selfcheck] INCOMPLETE  no error, but a part was not reached (see the notes)     exit 3
  [selfcheck] FAILED      an exception; the stage and the traceback follow         exit 1
  [selfcheck] NO VERDICT  cut off by --selfcheck_limit seconds (2700) or a signal  exit 1
The launchers call it once before a fresh rebuttal run. With SELFCHECK=1 (their default) they
start training whatever it reports; with SELFCHECK=strict a FAILED check stops the run before
training; SELFCHECK=0 skips the check.
"""
import argparse
import copy
import faulthandler
import gc
import pathlib
import sys
import tempfile
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ACTIONS = {"walker": 6, "cheetah": 6, "hopper": 4, "pendulum": 1, "acrobot": 1, "finger": 2,
           "reacher": 2, "quadruped": 12, "humanoid": 21, "fish": 5, "cup": 2, "cartpole": 1}


def parse_config(argv):
    """The command line of dreamer.py, resolved the way dreamer.py resolves it."""
    import ruamel.yaml as yaml
    import tools
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args(argv)
    configs = yaml.YAML(typ="safe", pure=True).load((ROOT / "configs.yaml").read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value
    defaults = {}
    for name in ["defaults", *(args.configs or [])]:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    return parser.parse_args(remaining)


class Space:
    def __init__(self, shape, low=None, high=None, dtype=None):
        self.shape, self.low, self.high, self.dtype = tuple(shape), low, high, dtype


class DictSpace:
    def __init__(self, spaces):
        self.spaces = spaces


def run(argv):
    import numpy as np
    import torch
    import models
    from shs_rssm.shs_rssm import shs_kwargs_from_config

    opts = argparse.ArgumentParser(add_help=False)
    opts.add_argument("--selfcheck_small", action="store_true", help="tiny networks (to try the script on a CPU)")
    opts.add_argument("--selfcheck_updates", type=int, default=0, help="number of updates (default: enough for every phase)")
    opts.add_argument("--selfcheck_report", default="", help="file that receives the verdict")
    opts.add_argument("--selfcheck_limit", type=int, default=2700, help="seconds after which the check gives up")
    own, argv = opts.parse_known_args(argv)
    REPORT[0] = own.selfcheck_report
    report("[selfcheck] NO VERDICT: started " + time.strftime("%Y-%m-%d %H:%M:%S")
           + ", then stopped by its time limit or by a signal before it finished")
    if own.selfcheck_limit > 0:                                      # also ends a check that hangs inside a library call
        faulthandler.dump_traceback_later(own.selfcheck_limit, exit=True)
    config = parse_config(argv)
    config.compile = False
    use_shs = bool(getattr(config, "use_shs", False))
    task = str(config.task)
    domain = task.split("_")[2] if task.startswith("carl_dmc_") else (task.split("_")[1] if "_" in task else task)
    A = ACTIONS.get(domain, 6)
    config.num_actions = A
    if own.selfcheck_small:
        config.dyn_deter = config.dyn_hidden = 64
        config.dyn_stoch, config.units = 8, 32
        config.encoder = dict(config.encoder, cnn_depth=8, mlp_layers=1, mlp_units=16)
        config.decoder = dict(config.decoder, cnn_depth=8, mlp_layers=1, mlp_units=16)
        if use_shs:
            config.shs_proj_dim, config.shs_rstick_dim, config.shs_move_holdout_horizon = 16, 8, 8
    B, T = int(config.batch_size), int(config.batch_length)
    device = str(config.device)

    # Compress the schedule: the first boundary (calibration and re-seed) at update 20.
    first, n_updates = None, 12
    if use_shs:
        kw = shs_kwargs_from_config(copy.deepcopy(config))
        per_unit = float(kw["shs_grad_per_env"])
        bounds = [float(p["until_env"]) for p in config.shs_curriculum if float(p.get("until_env", -1)) > 0]
        base_scale = float(getattr(config, "shs_schedule_scale", 1.0))
        if bounds:
            first = min(bounds) * per_unit / base_scale          # updates until the first boundary, as configured
            config.shs_schedule_scale = base_scale * max(1.0, first / 20.0)
            last = max(bounds + [float(getattr(config, "shs_freeze_globals_env", None) or 0)])
            n_updates = int(round(last * per_unit / config.shs_schedule_scale)) + 12
            # a structure sweep every 10 of these updates: each move type gets a few sweeps, and
            # the check stays short (a sweep costs far more than an update)
            every = 10.0 * config.shs_schedule_scale / per_unit
            config.shs_curriculum = tuple(
                dict(p, move_every_env=every) if float(p.get("move_every_env", 0) or 0) > 0 else dict(p)
                for p in config.shs_curriculum)
    if own.selfcheck_updates:
        n_updates = int(own.selfcheck_updates)

    spaces = {"image": Space(tuple(config.size) + (3,), dtype=np.uint8)}
    carl = task.startswith("carl_")
    with_context = "context" in str(config.encoder.get("mlp_keys", ""))
    if with_context:
        spaces["context"] = Space((14,))
    obs_space = DictSpace(spaces)
    act_space = Space((A,), low=-np.ones(A, np.float32), high=np.ones(A, np.float32))
    torch.manual_seed(0)
    np.random.seed(0)

    def make_model():
        wm = models.WorldModel(obs_space, act_space, 0, config).to(device)
        return wm, models.ImagBehavior(config, wm).to(device)

    def batch(step):
        first_ = torch.zeros(B, T)
        first_[:, 0] = 1.0
        if T > 12:
            first_[0, T // 2] = 1.0                               # a sequence that runs into another episode
        uid = (1000.0 * step + torch.arange(B, dtype=torch.float64))[:, None].expand(B, T).clone()
        data = {
            "image": torch.randint(0, 256, (B, T) + tuple(config.size) + (3,), dtype=torch.uint8),
            "action": torch.rand(B, T, A) * 2 - 1, "is_first": first_, "is_terminal": torch.zeros(B, T),
            "reward": torch.randn(B, T), "cont": torch.ones(B, T), "discount": torch.ones(B, T),
            "episode_first": first_.clone(), "episode_uid": uid,
            "acq_time": torch.full((B, T), float(step), dtype=torch.float64),
        }
        if carl:
            data["context_id"] = (torch.arange(B) % 3).float()[:, None].expand(B, T).clone()
            data["context"] = torch.randn(B, 1, 14).expand(B, T, 14).clone()
        return {k: v.numpy() for k, v in data.items()}               # numpy, as the replay delivers it

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    wm, behavior = make_model()
    dyn = wm.dynamics
    print(f"[selfcheck] {task} on {device}, batch {B} x {T}, {n_updates} updates"
          + (f"; schedule divided by {config.shs_schedule_scale:g} (first boundary at update "
             f"{int(round(first / max(1.0, first / 20.0)))} instead of {int(round(first))})" if first else ""), flush=True)
    reward_fn = lambda f, s, a: wm.heads["reward"](wm.dynamics.get_feat(s)).mode()
    kl = float("nan")
    STAGE[0] = "training through the schedule"
    for step in range(1, n_updates + 1):
        post, _, mets = wm._train(batch(step))
        kl = float(np.mean(mets.get("kl", float("nan"))))
        if not np.isfinite(kl):
            raise FloatingPointError(f"non-finite KL at update {step}")
        behavior._train(post, reward_fn)
        teacher = getattr(dyn, "_teacher", None)
        if teacher is not None:
            # An untrained encoder on random images drifts much faster than one in training, and
            # every drift step would empty the buffer that structure sweeps need.  The staleness
            # test is therefore off in this check, so that the sweeps are reached.
            teacher.drift_tol = float("inf")
    # Births and splits are rarely accepted on random data.  So that the code behind an accepted
    # one also runs here (install, acceptance checks, resized buffers, updates and a checkpoint
    # with the new K), up to two sweeps per move type accept growth candidates whatever their
    # score, with room for two new regimes each; then the settings of the arm are restored.
    forced = None
    if (use_shs and bool(getattr(config, "shs_structure_moves", False)) and getattr(dyn, "_curriculum", None)
            and not dyn._globals_frozen()):
        last_phase = dyn._curriculum[-1]
        if last_phase.get("birth") or last_phase.get("split"):
            STAGE[0] = "forced growth (birth and split candidates accepted whatever their score)"
            original = dict(last_phase)
            cap0, z0, K0 = dyn._move_K_cap, dyn._shs_move_holdout_z, int(dyn.regime.K)

            def accepted(move):
                return int(dyn.curriculum_state().get(f"shs_moves_{move}_accepted", 0))

            def one_update(step):
                post, _, mets = wm._train(batch(step))
                value = float(np.mean(mets.get("kl", float("nan"))))
                if not np.isfinite(value):
                    raise FloatingPointError(f"non-finite KL at update {step}")
                behavior._train(post, reward_fn)
                return value

            base = {m: accepted(m) for m in ("birth", "split")}
            step = n_updates
            try:
                dyn._shs_move_holdout_z = -1e15
                for want in ("birth", "split"):
                    if not original.get(want):
                        continue
                    last_phase.update(move_threshold=-1e15, shrink_threshold=1e15, move_every=2,
                                      birth=(want == "birth"), split=(want == "split"))
                    dyn._move_K_cap = int(dyn.regime.K) + 2
                    for _ in range(4):
                        step += 1
                        kl = one_update(step)
                        if accepted(want) > base[want]:
                            break
            finally:
                last_phase.clear()
                last_phase.update(original)
                dyn._move_K_cap, dyn._shs_move_holdout_z = cap0, z0
            for _ in range(2):                                           # the arm's own settings, the new K
                step += 1
                kl = one_update(step)
            n_updates = step
            forced = {m: accepted(m) - base[m] for m in ("birth", "split")}
            forced["K"] = f"{K0} -> {int(dyn.regime.K)}"
    STAGE[0] = "logging, open-loop error, checkpoint"
    train_s = time.time() - t0

    info = {"updates": n_updates, "seconds": round(train_s, 1), "kl": round(kl, 3)}
    with tempfile.TemporaryDirectory() as tmp:
        if use_shs:
            for name in ("shs_diagnostics", "shs_disentangle_diagnostics"):
                if hasattr(wm, name):
                    getattr(wm, name)(batch(n_updates + 1), n_updates, tmp, make_figures=False)
            if hasattr(wm, "shs_regime_filmstrip"):
                wm.shs_regime_filmstrip(batch(n_updates + 2), n_updates, tmp)
            cs = dyn.curriculum_state()
            info.update(phase=int(dyn._curr_phase), phases=len(dyn._curriculum), K=int(dyn.regime.K),
                        reseeds=int(cs.get("shs_reseeds", 0)), sweeps=int(cs.get("shs_sweeps_run", 0)),
                        frozen=bool(cs.get("shs_globals_frozen", 0)), calibrated=bool(dyn._shs_b0_calibrated))
            moves = {m: "/".join(str(int(cs.get(f"shs_moves_{m}_{k}", 0))) for k in ("attempts", "tried", "accepted"))
                     for m in ("merge", "delete", "birth", "split")}
            if info["sweeps"]:
                info["moves_searched_installed_kept"] = moves
            if forced is not None:
                info["forced_growth"] = forced
        wm.video_pred(batch(n_updates + 3))
        om = models.openl_summary([wm.openl_error(batch(n_updates + 4))], "eval_")
        info["openl_steps"] = int(om.get("eval_openl_steps", 0))
        if device.startswith("cuda"):
            info["peak_gpu_mib"] = int(torch.cuda.max_memory_allocated(device) / 2 ** 20)

        # The state of the world model and the runtime state of the regime head go through a
        # file into a fresh model, as at a resume.  Before anything else runs on the restored
        # model, its tensors and its K are compared with what was saved; then it gets one more
        # update.  The first model is released before the second is built, so this stage needs
        # no more memory than training.  The actor's and the optimizers' state, the replay and
        # the environments are not part of this stage.
        STAGE[0] = "saving the world model to a file and restoring it into a fresh model"
        saved_K = info.get("K")
        path = pathlib.Path(tmp) / "selfcheck_state.pt"
        torch.save({"state": wm.state_dict(), "live": dyn.live_state() if hasattr(dyn, "live_state") else None}, path)
        wm = behavior = dyn = post = mets = teacher = reward_fn = None
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        saved = torch.load(path, map_location=device, weights_only=False)
        sd, live = saved["state"], saved["live"]
        wm2, behavior2 = make_model()
        for mname, mod in wm2.named_modules():                       # buffers whose shape follows K
            for bname, buf in list(mod.named_buffers(recurse=False)):
                full = f"{mname}.{bname}" if mname else bname
                t = sd.get(full)
                if torch.is_tensor(t) and tuple(t.shape) != tuple(buf.shape):
                    mod.register_buffer(bname, torch.zeros(tuple(t.shape), dtype=buf.dtype, device=buf.device))
        wm2.load_state_dict(sd)
        restored = wm2.state_dict()
        differing = [k for k, v in sd.items()                        # repr_version counts loads, by design
                     if torch.is_tensor(v) and not k.endswith("repr_version") and (
                         k not in restored or tuple(restored[k].shape) != tuple(v.shape)
                         or not torch.equal(restored[k], v))]
        if differing:
            raise RuntimeError(f"{len(differing)} tensors differ after restoring, first: {differing[:3]}")
        if live is not None:
            wm2.dynamics.load_live_state(live)
        if use_shs and int(wm2.dynamics.regime.K) != saved_K:
            raise RuntimeError(f"K is {int(wm2.dynamics.regime.K)} after restoring, {saved_K} was saved")
        STAGE[0] = "an update of the restored model"
        post2, _, mets2 = wm2._train(batch(n_updates + 5))
        if not np.isfinite(float(np.mean(mets2.get("kl", float("nan"))))):
            raise FloatingPointError("non-finite KL in the update of the restored model")
        behavior2._train(post2, lambda f, s, a: wm2.heads["reward"](wm2.dynamics.get_feat(s)).mode())
        if device.startswith("cuda"):
            info["peak_gpu_mib_restored"] = int(torch.cuda.max_memory_allocated(device) / 2 ** 20)
    info["total_seconds"] = round(time.time() - t0, 1)
    notes = []
    if use_shs:
        if not info["calibrated"]:
            notes.append("the noise prior was not calibrated (first boundary not reached)")
        if info["phase"] != info["phases"] - 1:
            notes.append(f"stopped in phase {info['phase']} of {info['phases'] - 1}")
        if bool(getattr(config, "shs_reseed_at_calibration", False)) and info["reseeds"] != 1:
            notes.append(f"{info['reseeds']} re-seeds instead of 1")
        if bool(getattr(config, "shs_structure_moves", False)) and not info["frozen"]:
            if info["sweeps"] == 0:
                notes.append("no structure sweep ran")
            else:
                idle = [m for m in ("merge", "birth", "split")
                        if int(cs.get(f"shs_moves_{m}_attempts", 0)) == 0]
                if idle:
                    notes.append("never searched: " + ", ".join(idle))
            if forced is not None and forced["birth"] + forced["split"] == 0:
                notes.append("no birth or split was accepted even when forced, so the code after an "
                             "accepted growth move did not run")
        if getattr(config, "shs_freeze_globals_env", None) not in (None, "None") and not info["frozen"]:
            notes.append("the regime factors were not frozen")
    # OK: every part that this arm has was reached.  INCOMPLETE: no error, but a part was not
    # reached (the notes say which), so the check says nothing about it.
    word = "INCOMPLETE" if notes else "OK"
    verdict = f"[selfcheck] {word} " + str(info) + ((" | note: " + "; ".join(notes)) if notes else "")
    print(verdict, flush=True)
    report(verdict)
    return 3 if notes else 0


REPORT = [""]
STAGE = ["setup"]


def report(text):
    if not REPORT[0]:
        return
    try:
        path = pathlib.Path(REPORT[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.rstrip() + "\n")
    except OSError as e:
        print(f"[selfcheck] could not write {REPORT[0]}: {e}", flush=True)


def main():
    try:
        return run(sys.argv[1:])
    except BaseException:                                            # also SystemExit of argparse, KeyboardInterrupt
        tb = traceback.format_exc()
        print(tb, flush=True)
        print(f"[selfcheck] FAILED during: {STAGE[0]}. The traceback above is from the check, on this "
              "device, before any training. Send it along.", flush=True)
        report(f"[selfcheck] FAILED during: {STAGE[0]}\n" + tb)
        return 1
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    sys.exit(main())

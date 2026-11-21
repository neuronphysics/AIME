#!/usr/bin/env python3
"""Open-loop prediction error of every arm on one and the same set of held-out trajectories.

    python score_prediction.py logdir/rebuttal --task dmc_walker_walk --md
    python score_prediction.py logdir/rebuttal --task carl_walker --arms rb_gate,rb_ema,rb_power,rb_frozen
    python score_prediction.py logdir/rebuttal --task dmc_walker_walk --episodes-from dv3_gauss
    python score_prediction.py logdir/rebuttal --task dmc_walker_walk --episodes /path/to/episodes
    python score_prediction.py logdir/rebuttal --task dmc_walker_walk --checkpoint checkpoint_500000.pt

The errors that dreamer.py logs during training (openl_mse, eval_openl_mse) are measured on
the trajectories of the agent that is being trained, so two arms are scored on different data.
This script scores the saved world model of every run (latest.pt) on one common set:

  * the set is built from evaluation episodes, which no run trains on.  By default every run
    of the task contributes its newest evaluation episodes in equal number (per context on
    CARL); --episodes-from ARM takes the episodes of one arm only, so that no other arm is
    scored on data of its own policy; --episodes DIR takes the .npz episodes of a directory;
  * the episodes are cut into windows of --length steps at fixed positions; every model sees
    the same windows, filters the first --context frames and predicts the rest open loop with
    the recorded actions (WorldModel.openl_error, the function behind the logged numbers);
  * on CARL the error is the mean of the per-context means.

Per arm it prints the mean and the standard deviation over seeds of: the error on the whole
set, on the first 15 predicted steps, and, when the set is pooled from the runs, the error on
the windows that the run produced itself and on those of all other runs.  --csv writes one row
per run and source arm.  A run is expected as <root>/<task>/<arm>/seed_<n>/ with latest.pt and
eval_eps/.  Only checkpoints are read: it can be run at any time, on a GPU or (slowly) a CPU.

latest.pt moves on with a run.  To compare arms at a fixed budget while runs continue, the
launchers make dreamer.py keep checkpoint_<frames>.pt when a run passes that many frames
(--checkpoint_at); --checkpoint checkpoint_500000.pt scores those files, with the evaluation
episodes written up to the moment each of them was saved.
"""
import argparse
import contextlib
import csv
import io
import math
import os
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

NOT_OBSERVED = ("action", "reward", "discount", "is_first", "is_last", "is_terminal", "logprob")


class Space:
    def __init__(self, shape):
        self.shape = tuple(shape)


class DictSpace:
    def __init__(self, spaces):
        self.spaces = spaces


def find_runs(roots, task, arms, checkpoint="latest.pt"):
    """<root>/<task>/<arm>/seed_<n> directories that hold the checkpoint file."""
    runs, seen = [], set()
    for root in roots:
        root = pathlib.Path(root)
        for d, _, files in os.walk(root, followlinks=True):
            if checkpoint not in files:
                continue
            d = pathlib.Path(d)
            s = re.fullmatch(r"seed_?(\d+)", d.name)
            if not s or len(d.relative_to(root).parts) < 3:
                continue
            real = d.resolve()
            run = dict(path=d, task=d.parent.parent.name, arm=d.parent.name, seed=int(s.group(1)))
            if real in seen or (task and run["task"] != task) or (arms and run["arm"] not in arms):
                continue
            seen.add(real)
            runs.append(run)
    return sorted(runs, key=lambda r: (r["task"], r["arm"], r["seed"]))


def newest_episodes(directory, per_context, scan=400, not_after=None):
    """The newest episodes of a directory, per_context of them for every context id found among
    the newest `scan` files (one group when the episodes carry no context id).  With not_after
    (a file time), only episodes written up to then count: the ones a kept checkpoint goes with."""
    files = sorted(pathlib.Path(directory).glob("*.npz"), reverse=True)
    if not_after is not None:
        files = [f for f in files if f.stat().st_mtime <= not_after]
    files = files[:scan]
    groups = {}
    for f in files:
        try:
            with np.load(f) as ep:
                if "image" not in ep or "action" not in ep:
                    continue
                ctx = int(np.asarray(ep["context_id"]).reshape(-1)[0]) if "context_id" in ep else -1
        except Exception as e:                                      # a file that is being written
            print(f"  skipped {f.name}: {e}")
            continue
        g = groups.setdefault(ctx, [])
        if len(g) < per_context:
            g.append(f)
    return {c: sorted(v) for c, v in sorted(groups.items())}


def cut_windows(path, length, stride):
    """Windows of `length` steps at positions 0, stride, 2 stride, ... of one episode, as
    dreamer.py's replay returns them (the first step of a window is marked is_first)."""
    with np.load(path) as ep:
        episode = {k: ep[k] for k in ep.keys() if "log_" not in k}
    total = len(episode["action"])
    out = []
    for start in range(0, total - length + 1, stride):
        w = {k: v[start:start + length].copy() for k, v in episode.items()}
        w["is_first"] = w["is_first"].copy()
        w["is_first"][0] = True
        out.append(w)
    return out


def build_set(runs, args):
    """[(source label, source run path or None, [windows])] for one task."""
    if args.episodes:
        groups = newest_episodes(args.episodes, 10 ** 9, scan=10 ** 9)
        files = [f for v in groups.values() for f in v]
        wins = [w for f in files for w in cut_windows(f, args.length, args.stride)]
        return [("given", None, wins)], len(files)
    sources = [r for r in runs if not args.episodes_from or r["arm"] in args.episodes_from]
    if not sources:
        sys.exit(f"no run of the arms {sorted(args.episodes_from)} to take the episodes from")
    out, n_files, short = [], 0, []
    for r in sources:
        # latest.pt goes with the newest episodes; a kept checkpoint with those written up to
        # the moment it was saved (a minute of slack for the clocks of file servers)
        limit = None if args.checkpoint == "latest.pt" else (r["path"] / args.checkpoint).stat().st_mtime + 60.0
        groups = newest_episodes(r["path"] / "eval_eps", args.episodes_per_run, not_after=limit)
        files = [f for v in groups.values() for f in v]
        if not files or any(len(v) < args.episodes_per_run for v in groups.values()):
            short.append(f"{r['arm']}/seed_{r['seed']}")
        wins = [w for f in files for w in cut_windows(f, args.length, args.stride)]
        if wins:
            out.append((r["arm"], r["path"], wins))
            n_files += len(files)
    if short:
        print(f"  fewer than {args.episodes_per_run} evaluation episodes (per context) in: {', '.join(short)}")
    return out, n_files


def load_world_model(run, device, checkpoint="latest.pt"):
    """The world model of a run, rebuilt from the configuration stored in its checkpoint."""
    import torch
    import models
    import tools
    ck = torch.load(run["path"] / checkpoint, map_location=device, weights_only=False)
    cfg = dict(ck["config"])
    ns = argparse.Namespace(**{k: tools.args_type(v)(v) for k, v in cfg.items()})
    ns.device, ns.compile = device, False
    for key in ("steps", "eval_every", "log_every", "time_limit", "shs_consolidate_warmup"):
        if hasattr(ns, key) and isinstance(getattr(ns, key), (int, float)):
            setattr(ns, key, getattr(ns, key) // ns.action_repeat)      # as dreamer.py does before it builds the agent
    state = {k[len("_wm."):]: v for k, v in ck["agent_state_dict"].items() if k.startswith("_wm.")}
    return ck, ns, state


def instantiate(ck, ns, state, example, device):
    import torch
    import models
    ns.num_actions = int(example["action"].shape[-1])
    obs_space = DictSpace({k: Space(v.shape[1:]) for k, v in example.items() if k not in NOT_OBSERVED})
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            wm = models.WorldModel(obs_space, Space((ns.num_actions,)), int(ck.get("step", 0)), ns).to(device)
            for mname, mod in wm.named_modules():                       # buffers whose shape follows K
                for bname, buf in list(mod.named_buffers(recurse=False)):
                    t = state.get(f"{mname}.{bname}" if mname else bname)
                    if torch.is_tensor(t) and tuple(t.shape) != tuple(buf.shape):
                        mod.register_buffer(bname, torch.zeros(tuple(t.shape), dtype=buf.dtype, device=buf.device))
            own = wm.state_dict()
            for k, v in own.items():                                    # tensors that older checkpoints lack
                if k not in state and torch.is_tensor(v) and k.endswith(
                        ("z0_logvar", "b0_c0", "b0_d0", "b0_chat", "b0_dhat")):
                    state[k] = v.clone()
            wm.load_state_dict(state)
            ints = ck.get("shs_int_attrs") or {}
            for n, m in wm.named_modules():
                k = ints.get(f"_wm.{n}.K" if n else "_wm.K")
                if k is not None and isinstance(getattr(m, "K", None), int):
                    m.K = int(k)
            if hasattr(wm.dynamics, "load_live_state") and ck.get("shs_live"):
                wm.dynamics.load_live_state(ck["shs_live"], ck.get("transfer_runtime"), wm.encoder)
    except Exception:
        sys.stdout.write(log.getvalue()[-4000:])
        raise
    wm.requires_grad_(False)
    return wm


def score(wm, groups, args):
    """Sums of openl_error per source group: {label: [parts]}."""
    import torch
    out = {}
    for gi, (label, path, wins) in enumerate(groups):
        parts = []
        for rep in range(args.samples):
            torch.manual_seed(1000 * rep + gi)                          # the same noise for every model
            for i in range(0, len(wins), args.batch):
                chunk = wins[i:i + args.batch]
                batch = {k: np.stack([w[k] for w in chunk]) for k in chunk[0]}
                with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
                    parts.append(wm.openl_error(batch, context=args.context))
        out.setdefault((label, path), []).extend(parts)
    return out


def mean_std(values):
    v = [x for x in values if x is not None and math.isfinite(x)]
    if not v:
        return math.nan, math.nan, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else math.nan), len(v)


def cell(values, digits=5):
    m, s, n = mean_std(values)
    if n == 0:
        return "-"
    return f"{m:.{digits}f}" + (f" ± {s:.{digits}f}" if n > 1 else "")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("roots", nargs="+", help="directories with <task>/<arm>/seed_<n>/latest.pt")
    p.add_argument("--task", default="", help="task directory name (needed when the roots hold several tasks)")
    p.add_argument("--arms", default="", help="comma-separated arms to score (default: all); the set of "
                   "episodes does not depend on it")
    p.add_argument("--episodes-from", default="", help="comma-separated arms whose evaluation episodes form the set")
    p.add_argument("--episodes", default="", help="directory of .npz episodes that form the set")
    p.add_argument("--checkpoint", default="latest.pt", help="checkpoint file to score in every run, e.g. "
                   "checkpoint_500000.pt (kept by dreamer.py --checkpoint_at); its episodes are the newest "
                   "ones written before it")
    p.add_argument("--episodes-per-run", type=int, default=2, help="newest evaluation episodes per run (and context)")
    p.add_argument("--length", type=int, default=64, help="steps per window")
    p.add_argument("--stride", type=int, default=0, help="steps between window starts (default: --length)")
    p.add_argument("--context", type=int, default=5, help="filtered frames before the open-loop forecast")
    p.add_argument("--samples", type=int, default=1, help="forecasts per window (latent samples)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="")
    p.add_argument("--md", action="store_true", help="Markdown table")
    p.add_argument("--csv", default="", help="write one row per run and source arm")
    args = p.parse_args()
    args.stride = args.stride or args.length
    args.arms = {a for a in args.arms.split(",") if a}
    args.episodes_from = {a for a in args.episodes_from.split(",") if a}

    import torch
    import models
    from rebuttal_report import ARM_NOTE, arm_key
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    if os.sep in args.checkpoint or not args.checkpoint.endswith(".pt"):
        sys.exit("--checkpoint is a file name inside each run directory, e.g. checkpoint_500000.pt")
    runs = find_runs(args.roots, args.task, args.arms, args.checkpoint)
    tasks = sorted({r["task"] for r in runs})
    if not runs:
        sys.exit(f"no run with a {args.checkpoint} found (expected <root>/<task>/<arm>/seed_<n>/{args.checkpoint})")
    if len(tasks) > 1:
        sys.exit(f"several tasks found, choose one with --task: {', '.join(tasks)}")
    source_runs = runs if not args.arms else find_runs(args.roots, tasks[0], set(), args.checkpoint)
    without = [f"{r['arm']}/seed_{r['seed']}" for r in find_runs(args.roots, tasks[0], args.arms)
               if not (r["path"] / args.checkpoint).exists()] if args.checkpoint != "latest.pt" else []
    print(f"task {tasks[0]}: {len(runs)} runs to score ({args.checkpoint}) on {device}")
    if without:
        print(f"  no {args.checkpoint} yet in: {', '.join(without)}")
    groups, n_files = build_set(source_runs, args)
    n_win = sum(len(w) for _, _, w in groups)
    if not n_win:
        sys.exit("no window could be cut: no evaluation episode of at least --length steps")
    contexts = sorted({int(np.asarray(w["context_id"]).reshape(-1)[0]) for _, _, ws in groups for w in ws
                       if "context_id" in w})
    where = ("the directory given" if args.episodes else
             f"the runs of {', '.join(sorted(args.episodes_from))}" if args.episodes_from else "every run of the task")
    print(f"common set: {n_win} windows of {args.length} steps from {n_files} evaluation episodes of {where}"
          + (f", contexts {contexts}" if contexts else "")
          + f"; {args.context} context frames, {args.length - args.context} predicted")
    example = groups[0][2][0]

    rows, frames = [], {}
    for r in runs:
        try:
            ck, ns, state = load_world_model(r, device, args.checkpoint)
            wm = instantiate(ck, ns, state, example, device)
        except Exception as e:
            print(f"  {r['arm']}/seed_{r['seed']}: could not be loaded ({type(e).__name__}: {str(e)[:300]})")
            continue
        frames[(r["arm"], r["seed"])] = int(ck.get("logger_step", 0))
        parts = score(wm, groups, args)
        own = [x for (label, path), ps in parts.items() if path is not None and path.resolve() == r["path"].resolve() for x in ps]
        other = [x for (label, path), ps in parts.items() if path is None or path.resolve() != r["path"].resolve() for x in ps]
        everything = [x for ps in parts.values() for x in ps]

        def summary(ps):
            s = models.openl_summary(ps) if ps else {}
            return s.get("openl_mse"), s.get("openl_mse_15"), s.get("openl_steps", 0.0), s

        allm, all15, steps, full = summary(everything)
        row = dict(task=r["task"], arm=r["arm"], seed=r["seed"], frames=frames[(r["arm"], r["seed"])],
                   source="all", mse=allm, mse_15=all15, steps=steps,
                   contexts=full.get("openl_contexts", ""), own=summary(own)[0], other=summary(other)[0],
                   K=int(getattr(getattr(wm.dynamics, "regime", None), "K", 0)) or "")
        for c in contexts:
            row[f"mse_ctx{c}"] = full.get(f"openl_mse_ctx{c}")
        rows.append(row)
        by_arm = {}
        for (label, path), ps in parts.items():
            by_arm.setdefault(label, []).extend(ps)
        for label, ps in sorted(by_arm.items()):
            m, m15, st, _ = summary(ps)
            rows.append(dict(task=r["task"], arm=r["arm"], seed=r["seed"], frames=row["frames"],
                             source=label, mse=m, mse_15=m15, steps=st))
        print(f"  {r['arm']}/seed_{r['seed']}: {row['frames']} frames, mse {allm:.5f}, first 15 steps {all15:.5f}")
        del wm, state, ck
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    main_rows = [r for r in rows if r["source"] == "all"]
    if not main_rows:
        sys.exit("no model could be scored")
    top = max(frames.values())
    behind = sorted(f"{a}/seed_{s} ({f})" for (a, s), f in frames.items() if f < 0.95 * top)
    if behind:
        print(f"\nNOTE: these checkpoints are more than 5% behind the furthest one ({top} frames), so the "
              f"comparison mixes training budgets: {', '.join(behind)}")
    pooled = not args.episodes and not args.episodes_from
    arms = sorted({r["arm"] for r in main_rows}, key=arm_key)
    head = ["arm", "", "seeds", "frames", "MSE", "MSE, first 15 steps"]
    if not args.episodes:
        head += ["on its own episodes", "on the others' episodes"]
    if contexts:
        head += ["contexts scored"]
    table = []
    for a in arms:
        rs = [r for r in main_rows if r["arm"] == a]
        line = [a, ARM_NOTE.get(a, ""), str(len(rs)), f"{int(np.mean([r['frames'] for r in rs]))}",
                cell([r["mse"] for r in rs]), cell([r["mse_15"] for r in rs])]
        if not args.episodes:
            line += [cell([r["own"] for r in rs]), cell([r["other"] for r in rs])]
        if contexts:
            line += [f"{int(min(r['contexts'] or 0 for r in rs))} of {len(contexts)}"]
        table.append(line)
    print(f"\nOpen-loop prediction error on the common set ({tasks[0]}; pixel MSE, images in [0, 1]; "
          f"mean ± std over seeds)")
    if args.md:
        print("| " + " | ".join(head) + " |")
        print("|" + "|".join("---" for _ in head) + "|")
        for line in table:
            print("| " + " | ".join(line) + " |")
    else:
        widths = [max(len(str(x[i])) for x in [head] + table) for i in range(len(head))]
        for line in [head] + table:
            print("  ".join(str(c).ljust(w) for c, w in zip(line, widths)).rstrip())
    if pooled:
        print("\nEvery model is scored on the same windows.  The set holds episodes of every run in equal "
              "number, so each\nmodel has produced a small share of it itself ('on its own episodes'); "
              "'on the others' episodes' leaves that share out.")
    elif args.episodes_from:
        print("\nEvery model is scored on the same windows.  'on its own episodes' is filled only for the "
              "runs whose episodes form the set.")
    if args.csv:
        keys = ["task", "arm", "seed", "frames", "K", "source", "mse", "mse_15", "steps", "contexts", "own", "other"] \
            + [f"mse_ctx{c}" for c in contexts]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()

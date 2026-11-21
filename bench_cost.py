#!/usr/bin/env python3
"""Time and memory of a training update against the number of regimes.

    python bench_cost.py                    # on the GPU of the current job, a few minutes
    python bench_cost.py --K 1 8 16 32 --updates 200 --md

For DreamerV3 (categorical and Gaussian latents) and for SHS-RSSM with K regimes (preset
shs_rebuttal with the state-dependent gate, no structure moves) it prints the time of one
full update (world model, then actor-critic on imagined rollouts; batch 16 x 64 of 64 x 64
images, as in training) and the peak GPU memory.  Then it runs the full model (rb_full) on a
shortened schedule and times the structure sweeps of each phase.  The data are random: the
cost of an update does not depend on what the regimes have learned.

Run it from the repository root, on the same GPU type as the training runs.
"""
import argparse
import pathlib
import statistics
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import ruamel.yaml as yaml  # noqa: E402

import models  # noqa: E402
import tools  # noqa: E402


class Space:
    def __init__(self, shape, low=None, high=None, dtype=np.float32):
        self.shape, self.low, self.high, self.dtype = tuple(shape), low, high, dtype


class DictSpace:
    def __init__(self, spaces):
        self.spaces = spaces


def build_config(presets, overrides):
    """defaults + presets + overrides, resolved as dreamer.py resolves its command line."""
    configs = yaml.YAML(typ="safe", pure=True).load((ROOT / "configs.yaml").read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value
    defaults = {}
    for name in ["defaults", *presets]:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    argv = []
    for k, v in overrides.items():
        argv += [f"--{k}", str(v)]
    return parser.parse_args(argv)


def make_batch(step, B, T, A, device):
    img = torch.randint(0, 256, (B, T, 64, 64, 3), dtype=torch.uint8)
    first = torch.zeros(B, T)
    first[:, 0] = 1.0
    uid = (1000.0 * step + torch.arange(B, dtype=torch.float64))[:, None].expand(B, T)
    return {
        "image": img, "action": torch.rand(B, T, A) * 2 - 1, "is_first": first,
        "is_terminal": torch.zeros(B, T), "reward": torch.randn(B, T), "cont": torch.ones(B, T),
        "discount": torch.ones(B, T), "episode_first": first.clone(),
        "episode_uid": uid.clone(), "acq_time": torch.full((B, T), float(step), dtype=torch.float64),
    }


class Model:
    def __init__(self, presets, overrides, args):
        over = {"device": args.device, "compile": "False", "logdir": "/tmp/bench_cost",
                "batch_size": args.batch_size, "batch_length": args.batch_length, **overrides}
        if args.small:                      # only to try the script on a CPU
            over.update({"dyn_deter": 64, "dyn_hidden": 64, "dyn_stoch": 8, "units": 32,
                         "shs_rstick_dim": 8, "shs_proj_dim": 16, "precision": 32})
        self.config = cfg = build_config(presets, over)
        if args.small:
            cfg.encoder = dict(cfg.encoder, cnn_depth=8, mlp_layers=1, mlp_units=16)
            cfg.decoder = dict(cfg.decoder, cnn_depth=8, mlp_layers=1, mlp_units=16)
        if args.frames_per_update:          # shortened schedule for the sweep timing
            cfg.train_ratio = float(cfg.batch_size * cfg.batch_length * cfg.action_repeat) / float(args.frames_per_update)
        cfg.num_actions = args.actions
        obs = DictSpace({"image": Space((64, 64, 3), dtype=np.uint8)})
        act = Space((args.actions,), low=-np.ones(args.actions), high=np.ones(args.actions))
        torch.manual_seed(0)
        self.wm = models.WorldModel(obs, act, 0, cfg).to(args.device)
        self.behavior = models.ImagBehavior(cfg, self.wm).to(args.device)
        self.args, self.step = args, 0

    def update(self):
        a = self.args
        self.step += 1
        data = make_batch(self.step, a.batch_size, a.batch_length, a.actions, a.device)
        if a.device.startswith("cuda"):
            torch.cuda.synchronize(a.device)
        t0 = time.perf_counter()
        post, _, _ = self.wm._train(data)
        reward = lambda f, s, act: self.wm.heads["reward"](self.wm.dynamics.get_feat(s)).mode()
        self.behavior._train(post, reward)
        if a.device.startswith("cuda"):
            torch.cuda.synchronize(a.device)
        return time.perf_counter() - t0

    def regime_state_mib(self):
        """Size of the conjugate regime factors and their statistics (buffers of the regime
        head, not parameters): the part of the model state that grows with K."""
        regime = getattr(self.wm.dynamics, "regime", None)
        if regime is None:
            return float("nan")
        return sum(b.numel() * b.element_size() for b in regime.buffers()) / 2 ** 20


def peak_mib(device):
    if not device.startswith("cuda"):
        return float("nan")
    return torch.cuda.max_memory_allocated(device) / 2 ** 20


def show(title, head, rows, md):
    print()
    if md:
        print(f"**{title}**\n")
        print("| " + " | ".join(head) + " |")
        print("|" + "|".join("---" for _ in head) + "|")
        for r in rows:
            print("| " + " | ".join(r) + " |")
        return
    print(title)
    w = [max(len(r[i]) for r in rows + [head]) for i in range(len(head))]
    line = "  ".join(c.ljust(w[i]) for i, c in enumerate(head))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(r)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    ap.add_argument("--updates", type=int, default=100, help="timed updates per model (after --warmup)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--batch_length", type=int, default=64)
    ap.add_argument("--actions", type=int, default=6, help="action dimension (walker: 6, quadruped: 12)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-sweeps", action="store_true", help="skip the structure-sweep timing")
    ap.add_argument("--sweep-updates", type=int, default=330, help="updates of the shortened rb_full run")
    ap.add_argument("--sweep-frames-per-update", type=float, default=1250.0,
                    help="frames one update stands for in the shortened run (100k frames = 80 updates)")
    ap.add_argument("--small", action="store_true", help="tiny networks, to try the script without a GPU")
    ap.add_argument("--md", action="store_true", help="Markdown tables")
    args = ap.parse_args()
    args.frames_per_update = None
    dev = args.device
    print(f"device {dev}" + (f" ({torch.cuda.get_device_name(dev)})" if dev.startswith("cuda") else "")
          + f", batch {args.batch_size} x {args.batch_length}, {args.actions} action dimensions, "
          f"{args.updates} timed updates per model")

    variants = [("DreamerV3, categorical", ["dmc_vision", "dmc_vanilla_compare"], {"dyn_discrete": 32}),
                ("DreamerV3, Gaussian", ["dmc_vision", "dmc_vanilla_compare"], {"dyn_discrete": 0})]
    variants += [(f"SHS-RSSM, K = {k}", ["dmc_vision", "shs_rebuttal", "rb_k1" if k == 1 else "rb_gate"], {"shs_K": k})
                 for k in args.K]
    rows, base = [], None
    for name, presets, over in variants:
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
        m = Model(presets, {"task": "dmc_walker_walk", **over}, args)
        for _ in range(args.warmup):
            m.update()
        ts = [m.update() for _ in range(args.updates)]
        ms = 1000 * statistics.median(ts)
        base = ms if base is None else base
        # the behaviour module holds the world model, so count every parameter once
        n_par = sum(p.numel() for p in {id(p): p for p in [*m.wm.parameters(), *m.behavior.parameters()]}.values())
        mem = peak_mib(dev)
        reg = m.regime_state_mib()
        rows.append([name, f"{ms:.1f}", f"{1000 * statistics.mean(ts):.1f}", f"{ms / base:.2f}x",
                     "-" if mem != mem else f"{mem:.0f}", f"{n_par / 1e6:.1f}M", "-" if reg != reg else f"{reg:.2f}"])
        print(f"  {name}: {ms:.1f} ms per update", flush=True)
        del m
    show("One full update (world model + actor-critic).  Peak GPU memory is what the measured process allocated on "
         "the device; 'regime state' is the size of the regime factors and statistics, which are buffers and not "
         "counted among the parameters",
         ["model", "ms per update (median)", "mean", "relative", "peak GPU memory (MiB)", "parameters", "regime state (MiB)"],
         rows, args.md)

    if args.no_sweeps:
        return
    # ---- structure sweeps: rb_full on a shortened schedule --------------------------------------
    args.frames_per_update = args.sweep_frames_per_update
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    m = Model(["dmc_vision", "shs_rebuttal", "rb_full"], {"task": "dmc_walker_walk"}, args)
    dyn = m.wm.dynamics
    plain, sweeps = [], {}
    for _ in range(args.sweep_updates):
        before = int(dyn._n_sweeps_run)
        t = m.update()
        if int(dyn._n_sweeps_run) > before:
            sweeps.setdefault(int(dyn._curr_phase), []).append((t, int(dyn.regime.K)))
        else:
            plain.append(t)
    med = statistics.median(plain)
    label = {1: "merge, delete", 2: "merge, delete, birth", 3: "merge, delete, birth, split"}
    # How many updates lie between two sweeps in training, from the presets themselves:
    # updates per frame = train_ratio / (batch_size * batch_length * action_repeat).
    def between_sweeps(presets):
        c = build_config(presets, {})
        frames = max(float(p.get("move_every_env", 0) or 0) for p in c.shs_curriculum)
        return frames, frames * float(c.train_ratio) / (c.batch_size * c.batch_length * c.action_repeat)
    per_frames, dmc_updates = between_sweeps(["dmc_vision", "shs_rebuttal", "rb_full"])
    _, carl_updates = between_sweeps(["carl_walker_shs", "shs_rebuttal", "rb_full"])
    rows = []
    for ph in sorted(sweeps):
        extra = [max(0.0, t - med) for t, _ in sweeps[ph]]
        ks = [k for _, k in sweeps[ph]]
        s = statistics.mean(extra)
        rows.append([str(ph), label.get(ph, "-"), str(len(extra)), f"{s:.2f}", f"{max(extra):.2f}",
                     f"{min(ks)}-{max(ks)}", f"{100 * s / (dmc_updates * med):.2f}%",
                     f"{100 * s / (carl_updates * med):.2f}%"])
    mem = peak_mib(dev)
    if rows:
        show(f"Structure sweeps of the full model (one every {per_frames:.0f} frames: every {dmc_updates:.0f} updates on DMC, "
             f"every {carl_updates:.0f} on CARL; an update without a sweep takes {1000 * med:.1f} ms"
             + ("" if mem != mem else f"; peak GPU memory with sweeps {mem:.0f} MiB") + ")",
             ["phase", "moves searched", "sweeps timed", "seconds per sweep (mean)", "max", "K during the sweeps",
              "added training time, DMC", "CARL"], rows, args.md)
    else:
        print("\nno structure sweep ran: raise --sweep-updates")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import warnings
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

KNOWN_ENVS = [
    "acrobot", "cartpole", "cheetah", "cup", "finger", "hopper", "humanoid",
    "pendulum", "quadruped", "reacher", "walker", "dog", "fish", "manipulator",
    "swimmer", "point_mass", "ball_in_cup",
]
SEED_RE = re.compile(r"(?:seed|s)[_-]?(\d+)", re.I)

OKABE_ITO = [
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#009E73",  # Bluish Green
    "#CC79A7",  # Purple
    "#E69F00",  # Yellow-Orange
    "#56B4E9",  # Sky Blue
    "#B22222",  # Firebrick
    "#000000",  # Black
]

METRIC_LABELS = {
    "eval_return": "Evaluation return",
    "train_return": "Training return",
}

# Fixed colors per method/configuration; all use solid lines ("-")
CANONICAL_COLORS = {
    "SHS-RSSM": "#0072B2",          # Blue
    "SHS-RSSM (v2)": "#009E73",     # Green
    "SHS-RSSM (gate)": "#CC79A7",   # Purple
    "DreamerV3": "#D55E00",         # Vermillion / Orange
    "DreamerV3 (v2)": "#E69F00",    # Yellow-Orange
    "Vanilla Dreamer": "#D55E00",   # Vermillion / Orange
}


def format_method(raw: str) -> str:
    if not raw:
        return ""
    r = raw.lower().replace("-", "_")

    if r.startswith("shs"):
        rest = r[3:].strip("_")
        if not rest or rest == "rssm":
            return "SHS-RSSM"
        if rest.startswith("rssm_"):
            rest = rest[5:]
        return f"SHS-RSSM ({rest})"

    if any(r.startswith(k) for k in ("dreamerv3", "dreamer_v3", "vanilla_dreamer")):
        rest = re.sub(r"^(?:dreamerv3|dreamer_v3|vanilla_dreamer)_?", "", r).strip("_")
        return f"DreamerV3 ({rest})" if rest else "DreamerV3"

    if r.startswith("dreamer"):
        rest = r[7:].strip("_")
        if not rest or rest in ("v3", "3"):
            return "DreamerV3"
        if rest in ("v2", "2"):
            return "DreamerV2"
        return f"DreamerV3 ({rest})"

    if r.startswith("vanilla"):
        rest = r[7:].strip("_")
        return f"DreamerV3 ({rest})" if rest else "DreamerV3"

    if r.startswith("baseline"):
        rest = r[8:].strip("_")
        return f"Baseline ({rest})" if rest else "Baseline"

    if r.startswith("rssm"):
        rest = r[4:].strip("_")
        return f"RSSM ({rest})" if rest else "RSSM"

    return raw.replace("_", " ").title()


def infer_run_details(path: str, logdir: str) -> tuple[str, str, str | None]:
    """Extracts (env, method, seed) from a file path."""
    rel = os.path.relpath(path, logdir if os.path.isdir(logdir) else os.path.dirname(logdir))
    parts = [p for p in rel.replace("\\", "/").split("/") if p]
    if parts and (parts[-1].endswith(".jsonl") or parts[-1].endswith(".json")):
        parts = parts[:-1]
    if not parts:
        parts = [os.path.splitext(os.path.basename(path))[0]]

    seed = None
    if parts and (SEED_RE.fullmatch(parts[-1]) or re.fullmatch(r"\d+", parts[-1])):
        m = SEED_RE.search(parts[-1])
        seed = m.group(1) if m else parts[-1]
        parts = parts[:-1]

    if seed is None:
        for i in reversed(range(len(parts))):
            m = re.search(r"[_-](?:seed|s)[_-]?(\d+)$", parts[i], re.I)
            if m:
                seed = m.group(1)
                parts[i] = parts[i][:m.start()]
                break

    if seed is None:
        m = SEED_RE.search(path)
        if m:
            seed = m.group(1)

    comp = next((p for p in parts if any(e in p.lower() for e in KNOWN_ENVS)), None)
    if comp is None:
        comp = parts[-1] if parts else "run"

    name = comp.lower()

    visible = False
    if "_visible" in name or "-visible" in name:
        visible = True
        name = re.sub(r"[_-]visible\b", "", name)

    name = re.sub(r"^(?:carl_|dmc_)", "", name)
    name = re.sub(r"[_-](?:seed|s)[_-]?\d+$", "", name, flags=re.I)

    method_pattern = re.compile(
        r"^(.*?)[_-]("
        r"shs(?:[_-].*)?|"
        r"dreamerv3(?:[_-].*)?|"
        r"dreamer[_-]v3(?:[_-].*)?|"
        r"dreamer(?:[_-].*)?|"
        r"vanilla(?:[_-].*)?|"
        r"baseline(?:[_-].*)?|"
        r"rssm(?:[_-].*)?"
        r")$",
        re.I,
    )

    m = method_pattern.match(name)
    if m:
        env_raw = m.group(1)
        method_raw = m.group(2)
    else:
        env_raw = name
        method_raw = ""

    method = format_method(method_raw)
    if visible:
        method = f"{method} (ctx visible)" if method else "ctx visible"

    env = env_raw.strip("_-")
    return env, method, seed


def method_sort_key(m: str) -> tuple[int, str]:
    m_low = m.lower()
    if m == "SHS-RSSM":
        return (0, m)
    if "shs" in m_low:
        return (1, m)
    if any(k in m_low for k in ("dreamer", "vanilla", "baseline")):
        return (3, m)
    return (2, m)


def method_style(method: str, i: int) -> tuple[str, str]:
    # Always return a solid line ("-") and assign a distinct color
    if method in CANONICAL_COLORS:
        return CANONICAL_COLORS[method], "-"
    base = method.replace(" (ctx visible)", "").strip()
    if base in CANONICAL_COLORS:
        return CANONICAL_COLORS[base], "-"
    color = OKABE_ITO[i % len(OKABE_ITO)]
    return color, "-"


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 11,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "0.15",
        "axes.axisbelow": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "grid.color": "0.87",
        "grid.linewidth": 0.6,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.8",
        "legend.fancybox": False,
    })


def style_axis(ax) -> None:
    for s in ax.spines.values():
        s.set_visible(True)
    ax.grid(True, which="major")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", top=False, right=False)


def pretty_env(env: str) -> str:
    env_clean = env.replace("-", "_")
    for ke in KNOWN_ENVS:
        if env_clean.lower().startswith(ke):
            task = env_clean[len(ke):].strip("_")
            domain = ke.title()
            if task:
                return f"{domain} ({task.replace('_', ' ')})"
            return domain
    return env.replace("_", " ").title()


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").capitalize())


def find_metric_files(logdir: str, filename: str = "metrics.jsonl") -> list[str]:
    hits = []
    for root, _dirs, files in os.walk(logdir):
        for f in files:
            if f == filename or f.endswith("_" + filename):
                hits.append(os.path.join(root, f))
    if os.path.isfile(logdir):
        hits.append(logdir)
    return sorted(set(hits))


def load_series(path: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    steps, vals, bad = [], [], 0
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if metric in d and d[metric] is not None and "step" in d:
                try:
                    steps.append(float(d["step"]))
                    vals.append(float(d[metric]))
                except (TypeError, ValueError):
                    bad += 1
    if bad:
        print(f"    ! {bad} unparseable/!numeric lines skipped in {os.path.basename(path)}",
              file=sys.stderr)
    o = np.argsort(steps, kind="stable")
    s, v = np.asarray(steps)[o], np.asarray(vals)[o]
    if len(s):
        keep = np.concatenate([s[1:] != s[:-1], [True]])
        s, v = s[keep], v[keep]
    return s, v


def smooth(y: np.ndarray, w: int) -> np.ndarray:
    if w < 2 or len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def slope_per_1m(steps: np.ndarray, vals: np.ndarray, frac: float = 0.25) -> float:
    n = len(vals)
    if n < 4:
        return float("nan")
    k = max(2, int(n * frac))
    return float(np.polyfit(steps[-k:], vals[-k:], 1)[0] * 1e6)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", required=True, help="root to search (or a single file)")
    ap.add_argument("--filename", default="metrics.jsonl")
    ap.add_argument("--metric", default="eval_return",
                    help="eval_return (default) | train_return | any logged key")
    ap.add_argument("--envs", nargs="*", default=None,
                    help="substring filter, e.g. --envs cheetah hopper")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="keep only these methods, e.g. --methods SHS-RSSM DreamerV3")
    ap.add_argument("--smooth", type=int, default=0, help="moving-average window")
    ap.add_argument("--ref", default=None,
                    help='JSON of reference scores, e.g. \'{"cheetah": 880}\'')
    ap.add_argument("--separate", action="store_true", help="one panel per environment")
    ap.add_argument("--no-aggregate", action="store_true", help="draw every seed")
    ap.add_argument("--max-steps", type=float, default=None)
    ap.add_argument("--out", default="returns.png")
    ap.add_argument("--csv", default=None, help="also write a summary CSV here")
    args = ap.parse_args()

    files = find_metric_files(args.logdir, args.filename)
    if not files:
        print(f"No '{args.filename}' found under {args.logdir}", file=sys.stderr)
        return 1

    runs: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for f in files:
        env, method, seed = infer_run_details(f, args.logdir)
        if args.envs and not any(e.lower() in env.lower() or e.lower() in f.lower()
                                 for e in args.envs):
            continue
        if args.methods and not any(m.lower() in method.lower() for m in args.methods):
            continue
        s, v = load_series(f, args.metric)
        if len(s) == 0:
            print(f"    ! no '{args.metric}' in {f}", file=sys.stderr)
            continue
        if args.max_steps:
            m = s <= args.max_steps
            s, v = s[m], v[m]
        runs[env][method].append((seed, s, v, f))
        print(f"  {env:<22} {method or '-':<18} seed={seed or '-':<4} n={len(s):<6} "
              f"steps<= {s[-1]:>12,.0f}  {os.path.relpath(f, args.logdir)}")

    if not runs:
        print("Nothing matched the filters.", file=sys.stderr)
        return 1

    set_style()
    ref = json.loads(args.ref) if args.ref else {}
    envs = sorted(runs)
    colors = {e: OKABE_ITO[i % len(OKABE_ITO)] for i, e in enumerate(envs)}
    ylab = metric_label(args.metric)

    if args.separate:
        n = len(envs)
        ncol = min(3, n)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 2.8 * nrow),
                                 squeeze=False, constrained_layout=True)
        axes = axes.ravel()
    else:
        fig, ax0 = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
        axes = [ax0] * len(envs)

    summary = []
    for i, env in enumerate(envs):
        ax = axes[i]
        sorted_methods = sorted(runs[env].items(), key=lambda kv: method_sort_key(kv[0]))
        for mi, (method, series) in enumerate(sorted_methods):
            mc, ls = method_style(method, mi)
            c = mc if (args.separate or len(envs) == 1) else colors[env]
            if not method:
                c, ls = colors[env], "-"
            label = pretty_env(env) if not method else (method if args.separate else f"{pretty_env(env)} — {method}")

            if len(series) > 1 and not args.no_aggregate:
                grid = np.unique(np.concatenate([s for _, s, _, _ in series]))
                lo = min(s[0] for _, s, _, _ in series)
                hi = max(s[-1] for _, s, _, _ in series)
                grid = grid[(grid >= lo) & (grid <= hi)]
                if len(grid) == 0:
                    grid = np.unique(np.concatenate([s for _, s, _, _ in series]))

                M = np.vstack([np.interp(grid, s, v, left=np.nan, right=np.nan)
                               for _, s, v, _ in series])

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    med = np.nanmedian(M, axis=0)
                    q1 = np.nanpercentile(M, 25, axis=0)
                    q3 = np.nanpercentile(M, 75, axis=0)

                valid = ~np.isnan(med)
                if np.any(valid):
                    gp = grid[valid]
                    med_v, q1_v, q3_v = med[valid], q1[valid], q3[valid]
                    mp, q1p, q3p = (smooth(a, args.smooth) for a in (med_v, q1_v, q3_v))
                    gp = gp[len(gp) - len(mp):]

                    ax.fill_between(gp / 1e6, q1p, q3p, color=c, alpha=.16, lw=0, zorder=2)
                    ax.plot(gp / 1e6, mp, color=c, ls="-", lw=2.0, zorder=3,
                            label=f"{label}, $n={len(series)}$")
                    s_ref, v_ref = gp, mp
                else:
                    s_ref, v_ref = series[0][1], series[0][2]
            else:
                for sd, s, v, _ in series:
                    y = smooth(v, args.smooth)
                    x = s[len(s) - len(y):]
                    lab = label if sd is None else f"{label} s{sd}"
                    ax.plot(x / 1e6, y, color=c, ls="-", lw=1.5, alpha=.85, zorder=3, label=lab)
                s_ref, v_ref = series[0][1], series[0][2]

            if env in ref or any(k in env for k in ref):
                key = env if env in ref else next(k for k in ref if k in env)
                ax.axhline(ref[key], color="0.25", ls=(0, (5, 3)), lw=1.1, zorder=1)
                ax.text(.995, ref[key], f"ref {ref[key]:g} ",
                        transform=ax.get_yaxis_transform(),
                        va="top", ha="right", fontsize=7.5, color="0.25")

            sl = slope_per_1m(s_ref, v_ref)
            k = max(2, int(len(v_ref) * .25))
            summary.append(dict(env=env, method=method or "-", n_runs=len(series), steps=float(s_ref[-1]),
                                final=float(np.nanmean(v_ref[-k:])),
                                std=float(np.nanstd(v_ref[-k:])), best=float(np.nanmax(v_ref)),
                                slope_per_1M=sl,
                                status="converged" if abs(sl) < 100 else "still changing"))

        if args.separate:
            style_axis(ax)
            ax.set_title(pretty_env(env), loc="left", fontweight="bold")
            finals = [r for r in summary if r["env"] == env]
            ax.text(0.98, 0.04, "\n".join(f"{r['method']}: {r['final']:.0f}$\\pm${r['std']:.0f}" for r in finals),
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="0.35")
            ax.set_xlabel(r"Environment steps ($\times 10^6$)")
            ax.set_ylabel(ylab)
            if len(runs[env]) > 1 or any(runs[env]):
                ax.legend(borderaxespad=0.6)

    if not args.separate:
        style_axis(ax0)
        ax0.set_xlabel(r"Environment steps ($\times 10^6$)")
        ax0.set_ylabel(ylab)
        ax0.set_title(ylab + ("  (median $\\pm$ IQR over seeds)"
                              if not args.no_aggregate else ""),
                      loc="left", fontweight="bold")
        ax0.legend(ncols=1 if len(envs) <= 4 else 2, borderaxespad=0.6)
        ax0.margins(x=0.01)
    else:
        for j in range(len(envs), len(axes)):
            axes[j].axis("off")

    fig.savefig(args.out, bbox_inches="tight")
    print(f"\nwrote {args.out}")

    print(f"\n{'env':<22}{'method':<18}{'runs':>5}{'steps':>13}{'final':>10}{'best':>9}"
          f"{'slope/1M':>11}  status")
    for r in summary:
        print(f"{r['env']:<22}{r['method']:<18}{r['n_runs']:>5}{r['steps']:>13,.0f}{r['final']:>10.1f}"
              f"{r['best']:>9.1f}{r['slope_per_1M']:>11.0f}  {r['status']}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Strip + ribbon figure for the FitzHugh-Nagumo multi-step comparison.

Produces four panels:

  A  phase portrait: truth vs each model's 30-step open-loop rollout
  B  ribbon of the per-trajectory error distribution against horizon
     (median line, 25-75 and 10-90 bands) -- the spread matters as much as
     the mean, because a model can look good on average while failing badly
     on a subset of initial phases
  C  strip: the inferred regime sequence during the rollout, one row per
     model, which is where the gate variants differ visibly even though
     their error curves do not
  D  gate diagnostic: rho_{k,t} against the latent, showing whether the
     persistence gate is state-dependent at all

Run after fhn_benchmark.py has populated fhn_bench/, or standalone (it will
refit the SHS variants itself, which takes a couple of minutes).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

import fhn_benchmark as F                                   # noqa: E402
from shs_rssm.offline_trainer import fit_offline_corpus     # noqa: E402
from shs_rssm.regime_head import RegimeHead                 # noqa: E402

OUT = _HERE / "fhn_bench"
OUT.mkdir(exist_ok=True)

VARIANTS = [
    ("no gate",     dict(recurrent=False),                              "tab:blue"),
    ("gate=carry",  dict(recurrent=True, gate_input="carry"),           "tab:purple"),
    ("gate=latent", dict(recurrent=True, gate_input="latent"),          "tab:red"),
]


def fit_variant(Xs, kw, K0=12, laps=18, seed=0, sF=0.1):
    """Fit and roll out, returning predictions, regime path and the head."""
    torch.manual_seed(seed)
    n, T, D = Xs.shape
    TT = F.T_TRAIN
    z = torch.tensor(Xs[:, :TT], dtype=torch.float32)
    d = torch.zeros(n, TT, 2)
    isf = torch.zeros(n, TT); isf[:, 0] = 1.0
    zv = torch.full((n, TT, D), 1e-4)
    head = RegimeHead(stoch=D, deter=2, K=K0, proj_dim=None, action_dim=0,
                      a0=3.0, b0=sF * float(z.var()),
                      online_mode="memoized", expected_batches=1,
                      device=torch.device("cpu"), **kw)
    fit_offline_corpus(head, encode_fn=lambda: [("c", z, d, isf, zv, None)],
                       laps=laps, sweep_every=3,
                       sweep_kwargs=dict(threshold=0.0, refine_iters=3))
    with torch.no_grad():
        gam, _, _, _ = head.regime_inference(z, d, is_first=isf, z_var=zv)
        resp, x = gam[:, -1], z[:, -1]
        d1 = torch.zeros(n, 2)
        preds, path = [], []
        for _ in range(F.T_HOLDOUT):
            x, resp, mean, _ = head.imagine_prior(x, d1, resp, sample=False)
            preds.append(mean.clone()); path.append(resp.argmax(-1).clone())
            x = mean
        P = torch.stack(preds, 1).numpy()
        Z = torch.stack(path, 1).numpy()
    return P, Z, head


def gate_rho(head, Xs):
    """rho_{k,t} over the training corpus, or None for the non-recurrent path."""
    if not head.recurrent:
        return None
    n, T, D = Xs.shape
    z = torch.tensor(Xs[:, :F.T_TRAIN], dtype=torch.float32)
    zv = torch.full_like(z, 1e-4)
    isf = torch.zeros(n, F.T_TRAIN); isf[:, 0] = 1.0
    with torch.no_grad():
        if getattr(head, "gate_input", "carry") == "latent":
            phi = head.build_stick_phi_z(z, zv, None, is_first=isf)
        else:
            phi = head.build_stick_phi(torch.zeros(n, F.T_TRAIN, 2), None,
                                       is_first=isf)
        m, v = head.rstick._psi_moments(phi)
        return torch.sigmoid(m / torch.sqrt(1 + (np.pi / 8) * v)).numpy()


def main(n_traj=100, seed=0, laps=18):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    X = F.simulate_fhn(n_traj=n_traj, seed=seed)
    Xs, _, _ = F.standardise(X)
    true = Xs[:, F.T_TRAIN:]

    cache = OUT / f"ribbon_cache_n{n_traj}_s{seed}_l{laps}.npz"
    fits = {}
    if cache.exists():
        blob = np.load(cache, allow_pickle=True)
        fits = blob["fits"].item()
        print(f"  loaded cached fits from {cache.name}")
    else:
        for name, kw, col in VARIANTS:
            print(f"  fitting {name} ...", flush=True)
            P, Z, head = fit_variant(Xs, kw, laps=laps, seed=seed)
            fits[name] = dict(P=P, Z=Z, col=col, rho=gate_rho(head, Xs),
                              K=int(head.K))
        np.savez(cache, fits=fits)
        print(f"  cached to {cache.name}")

    # external baselines, if fhn_benchmark has produced them
    ext = {}
    for k, lab, col in [("trslds", "TrSLDS", "tab:green"),
                        ("linear", "single LDS", "0.35")]:
        f = OUT / f"{k}_seed{seed}.npz"
        if f.exists():
            ext[lab] = dict(curve=np.load(f, allow_pickle=True)["curve"], col=col)

    fig = plt.figure(figsize=(13.5, 9.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.15, 0.9], hspace=0.42,
                          wspace=0.24)

    # ---- A: phase portrait ------------------------------------------------
    axA = fig.add_subplot(gs[0, 0])
    # show the trajectory that actually MOVES over the holdout; trajectory 0 can
    # sit on the slow manifold, where every model looks identical
    i = int(np.argmax(true.std(1).sum(-1)))
    hist = Xs[i, F.T_TRAIN - 60:F.T_TRAIN]
    axA.plot(hist[:, 0], hist[:, 1], color="0.7", lw=1.2, label="history")
    axA.plot(true[i, :, 0], true[i, :, 1], "k-", lw=2.2, label="truth")
    for name in fits:
        P = fits[name]["P"]
        axA.plot(P[i, :, 0], P[i, :, 1], "-", color=fits[name]["col"], lw=1.6,
                 alpha=0.9, label=f"SHS {name}")
    axA.plot(*hist[-1], "ko", ms=6)
    axA.set_xlabel("$v$ (standardised)"); axA.set_ylabel("$w$")
    axA.set_title("A  30-step open-loop rollout, phase plane", fontsize=10,
                  loc="left")
    axA.legend(fontsize=7, frameon=False, loc="best"); axA.grid(alpha=0.25)
    axA.set_aspect("equal", adjustable="datalim")

    # ---- B: error ribbon --------------------------------------------------
    axB = fig.add_subplot(gs[0, 1])
    h = np.arange(1, F.T_HOLDOUT + 1)
    denom = (true ** 2).sum(-1).mean()
    for name in fits:
        e = np.sqrt(((fits[name]["P"] - true) ** 2).sum(-1) / max(denom, 1e-12))
        med = np.median(e, 0)
        c = fits[name]["col"]
        axB.fill_between(h, np.percentile(e, 10, 0), np.percentile(e, 90, 0),
                         color=c, alpha=0.10, lw=0)
        axB.fill_between(h, np.percentile(e, 25, 0), np.percentile(e, 75, 0),
                         color=c, alpha=0.22, lw=0)
        axB.plot(h, med, "-", color=c, lw=2, label=f"SHS {name}")
    for lab, d in ext.items():
        axB.plot(h, d["curve"], "--", color=d["col"], lw=1.6, label=lab)
    axB.axhline(1.0, color="k", lw=0.6, alpha=0.4)
    axB.text(F.T_HOLDOUT * 0.62, 1.02, "predicting the mean", fontsize=7,
             alpha=0.6)
    axB.set_xlabel("steps predicted into the future ($h$)")
    axB.set_ylabel("normalised error")
    axB.set_title("B  per-trajectory error: median, 25-75, 10-90", fontsize=10,
                  loc="left")
    axB.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    axB.grid(alpha=0.25); axB.set_xlim(1, F.T_HOLDOUT)

    # ---- C: regime strips -------------------------------------------------
    axC = fig.add_subplot(gs[1, :])
    Kmax = max(f["K"] for f in fits.values())
    cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, max(Kmax, 2))))
    rows, labels = [], []
    for name in fits:
        rows.append(fits[name]["Z"][:24])
        labels.append(f"SHS {name}\n(K={fits[name]['K']})")
    strip = np.concatenate([np.vstack([r, np.full((2, r.shape[1]), np.nan)])
                            for r in rows], 0)
    im = axC.imshow(strip, aspect="auto", interpolation="nearest", cmap=cmap,
                    vmin=-0.5, vmax=max(Kmax, 2) - 0.5)
    for j, lab in enumerate(labels):
        axC.text(-0.7, j * 26 + 12, lab, ha="right", va="center", fontsize=8)
        if j:
            axC.axhline(j * 26 - 1.5, color="k", lw=1.0)
    cb = fig.colorbar(im, ax=axC, pad=0.01, fraction=0.022,
                      ticks=range(max(Kmax, 2)))
    cb.set_label("regime", fontsize=8); cb.ax.tick_params(labelsize=7)
    axC.set_yticks([]); axC.set_xlabel("steps into the rollout")
    axC.set_title("C  regime path during the rollout (24 trajectories per block)",
                  fontsize=10, loc="left")
    axC.annotate("every variant collapses to ONE regime by ~step 16:\n"
                 "the gate sets dwell time, not destination, so the\n"
                 "chain cannot keep cycling",
                 xy=(0.62, 0.5), xycoords="axes fraction", fontsize=8,
                 ha="left", va="center",
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.88,
                           ec="0.5"))

    # ---- D: gate diagnostic ----------------------------------------------
    axD = fig.add_subplot(gs[2, 0])
    any_rho = False
    for name in fits:
        rho = fits[name]["rho"]
        if rho is None:
            continue
        any_rho = True
        sd = rho.std((0, 1))
        axD.bar(np.arange(len(sd)) + (0.35 if "latent" in name else 0.0),
                sd, width=0.33, color=fits[name]["col"], alpha=0.85,
                label=f"{name}  (rho sd across t)")
    if any_rho:
        axD.set_xlabel("regime $k$"); axD.set_ylabel("sd of $\\rho_{k,t}$")
        axD.set_title("D  is the persistence gate state-dependent?", fontsize=10,
                      loc="left")
        axD.legend(fontsize=7, frameon=False); axD.grid(alpha=0.25, axis="y")
    else:
        axD.axis("off")

    # ---- table ------------------------------------------------------------
    axT = fig.add_subplot(gs[2, 1]); axT.axis("off")
    hs = [1, 5, 10, 15, 30]
    tbl = [["model"] + [f"h={x}" for x in hs]]
    for name in fits:
        c = F.nrmse_by_horizon(fits[name]["P"], true)
        tbl.append([f"SHS {name}"] + [f"{c[x-1]:.3f}" for x in hs])
    for lab, d in ext.items():
        tbl.append([lab] + [f"{d['curve'][x-1]:.3f}" for x in hs])
    t = axT.table(cellText=tbl[1:], colLabels=tbl[0], loc="center",
                  cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1.06, 1.5)
    for (r, c), cell in t.get_celld().items():
        cell.set_width(0.30 if c == 0 else 0.135)
    axT.set_title("NRMSE by horizon", fontsize=10, loc="left")

    fig.suptitle("FitzHugh-Nagumo, 30-step open-loop prediction "
                 f"({n_traj} trajectories, Becker-Ehmck et al. 2019 protocol)",
                 fontsize=11)
    fig.savefig(OUT / "fhn_ribbon.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote {OUT / 'fhn_ribbon.png'}")

    print("\ngate diagnostic:")
    for name in fits:
        rho = fits[name]["rho"]
        if rho is None:
            print(f"  {name:12s} non-recurrent (no gate)")
        else:
            print(f"  {name:12s} rho mean {np.round(rho.mean((0,1)), 3)}  "
                  f"sd {np.round(rho.std((0,1)), 4)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traj", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--laps", type=int, default=18)
    a = ap.parse_args()
    main(n_traj=a.n_traj, seed=a.seed, laps=a.laps)

from __future__ import annotations

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
            "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]


@torch.no_grad()
def regime_responsibilities(rssm, post, is_first=None):
    stoch = post["stoch"].float()
    deter = post["deter"].float()
    gamma, _, _, _ = rssm.regime.regime_inference(stoch, deter, is_first,
                                                  cache_estep=False)
    return gamma


def plot_latent_clustering(stoch, gamma, path, title="Latent regime clustering",
                           true_labels=None):
    B, T, L = stoch.shape
    K = gamma.shape[-1]
    Z = stoch.reshape(-1, L).detach().cpu().numpy()
    R = gamma.reshape(-1, K).detach().cpu().numpy()
    lab = R.argmax(-1)
    occ = R.sum(0); occ = occ / occ.sum()
    used = np.where(occ > 0.01)[0]

    Zc = Z - Z.mean(0)
    try:
        U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        emb = Zc @ Vt[:2].T
    except np.linalg.LinAlgError:
        emb = Zc[:, :2]

    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1, 1.5])

    ax = fig.add_subplot(gs[0, 0])
    for k in used:
        m = lab == k
        ax.scatter(emb[m, 0], emb[m, 1], s=5, c=_PALETTE[k % len(_PALETTE)],
                   alpha=0.4, label=f"regime {k}")
    ax.set_title(f"latents (PCA-2) by inferred regime\n{len(used)} active of K={K}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(range(K), occ, color=[_PALETTE[k % len(_PALETTE)] for k in range(K)])
    ax.set_title("regime occupancy"); ax.set_xlabel("regime"); ax.set_ylabel("fraction")
    ax.axhline(0.01, ls="--", color="0.6", lw=1)

    ax = fig.add_subplot(gs[0, 2])
    seq = gamma[0].argmax(-1).detach().cpu().numpy()
    for t in range(T):
        ax.axvspan(t - 0.5, t + 0.5, ymin=(0.0 if true_labels is None else 0.0),
                   ymax=(1.0 if true_labels is None else 0.45),
                   color=_PALETTE[int(seq[t]) % len(_PALETTE)], lw=0)
    if true_labels is not None:
        tl = true_labels[0]
        for t in range(T):
            ax.axvspan(t - 0.5, t + 0.5, ymin=0.55, ymax=1.0,
                       color=_PALETTE[int(tl[t]) % len(_PALETTE)], lw=0)
        ax.set_yticks([0.22, 0.78]); ax.set_yticklabels(["inferred", "true"])
    else:
        ax.set_yticks([0.5]); ax.set_yticklabels(["inferred"])
    ax.set_xlim(-0.5, T - 0.5); ax.set_title("regime timeline (one episode)")
    ax.set_xlabel("time step")
    fig.suptitle(title, y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return dict(active=len(used), occupancy=occ)


def _to_img(x):
    x = x.detach().cpu().float().numpy() if hasattr(x, "detach") else np.asarray(x)
    if x.ndim == 3:
        if x.shape[-1] in (1, 3, 4):
            return np.clip(x[..., 0] if x.shape[-1] == 1 else x, 0, 1)
        if x.shape[0] in (1, 3, 4):
            y = np.transpose(x, (1, 2, 0))
            return np.clip(y[..., 0] if y.shape[-1] == 1 else y, 0, 1)
    return np.clip(x, 0, 1)


def plot_reconstructions(true_frames, recon_frames, path, imagined_frames=None,
                         n=8, context=None, title="Frame reconstruction"):
    T = true_frames.shape[0]
    idx = np.linspace(0, T - 1, min(n, T)).round().astype(int)
    rows = 2 if imagined_frames is None else 3
    fig, axes = plt.subplots(rows, len(idx), figsize=(1.5 * len(idx), 1.7 * rows))
    if len(idx) == 1:
        axes = axes.reshape(rows, 1)
    row_names = ["true", "reconstructed"] + (["imagined"] if imagined_frames is not None else [])
    series = [true_frames, recon_frames] + ([imagined_frames] if imagined_frames is not None else [])
    for r, (name, frames) in enumerate(zip(row_names, series)):
        for c, t in enumerate(idx):
            ax = axes[r, c]
            ti = min(int(t), frames.shape[0] - 1)
            ax.imshow(_to_img(frames[ti]), cmap="magma", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={t}", fontsize=9)
            if c == 0:
                ax.set_ylabel(name, fontsize=11)
    if context is not None:
        for c, t in enumerate(idx):
            if t >= context:
                axes[-1, c].add_patch(plt.Rectangle((0, 0), 1, 1, transform=axes[-1, c].transAxes,
                                                     fill=False, edgecolor="cyan", lw=2))
    fig.suptitle(title + ("" if context is None else
                 f"   (imagination open-loop after t={context-1}, cyan)"), y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)

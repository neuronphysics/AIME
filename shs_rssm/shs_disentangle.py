from __future__ import annotations

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score

_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
            "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94"]


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _subsample(n, max_points, seed=0):
    if n <= max_points:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, max_points, replace=False)


def tsne_embedding(Z, max_points=2500, perplexity=30.0, seed=0):
    Z = _np(Z).astype(np.float64)
    idx = _subsample(Z.shape[0], max_points, seed)
    Zs = Z[idx]
    perp = float(min(perplexity, max(5, (len(idx) - 1) // 3)))
    emb = TSNE(n_components=2, perplexity=perp, init="pca",
               learning_rate="auto", random_state=seed).fit_transform(Zs)
    return emb, idx


def _discretize(x, bins=8):
    x = _np(x).reshape(-1)
    if np.allclose(x, x[0]):
        return np.zeros_like(x, dtype=int)
    qs = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
    return np.digitize(x, qs)


def _entropy(labels):
    _, c = np.unique(labels, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def regime_factor_alignment(gamma, factors, boundary_tol=2):
    gamma = _np(gamma)
    B, T, K = gamma.shape
    regime = gamma.argmax(-1)
    npres = _np(factors["n_present"]).astype(int)

    nmi = normalized_mutual_info_score(regime.reshape(-1), npres.reshape(-1))

    flat_r, flat_n = regime.reshape(-1), npres.reshape(-1)
    purity = 0.0
    for k in np.unique(flat_r):
        m = flat_r == k
        if m.any():
            vals, cnts = np.unique(flat_n[m], return_counts=True)
            purity += cnts.max()
    purity /= flat_r.size

    tp = fp = fn = 0
    for b in range(B):
        r_sw = np.where(np.diff(regime[b]) != 0)[0] + 1
        n_ch = np.where(np.diff(npres[b]) != 0)[0] + 1
        matched_events = set()
        for s in r_sw:
            near = np.where(np.abs(n_ch - s) <= boundary_tol)[0]
            if len(near):
                tp += 1; matched_events.add(near[0])
            else:
                fp += 1
        fn += len(set(range(len(n_ch))) - matched_events)
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)

    return dict(regime_composition_nmi=float(nmi), regime_purity=float(purity),
                boundary_precision=float(prec), boundary_recall=float(rec),
                boundary_f1=float(f1), n_regimes_used=int(len(np.unique(regime))))


def factor_decodability(stoch, factors, key="n_present", k=15, n_splits=3, seed=0):
    from sklearn.neighbors import KNeighborsClassifier
    Z = _np(stoch).reshape(-1, _np(stoch).shape[-1])
    v = _np(factors[key])
    y = (v if v.ndim == 2 else v.sum(-1)).reshape(-1)
    y = np.rint(y).astype(int) if key == "n_present" else _discretize(y, 5)
    rng = np.random.default_rng(seed)
    N = len(y); perm = rng.permutation(N)
    folds = np.array_split(perm, n_splits)
    accs = []
    for i in range(n_splits):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        knn = KNeighborsClassifier(n_neighbors=k).fit(Z[tr], y[tr])
        pred = knn.predict(Z[te])
        cls = np.unique(y[te]); acc = np.mean([(pred[y[te] == c] == c).mean() for c in cls])
        accs.append(acc)
    vals, cnts = np.unique(y, return_counts=True)
    baseline = 1.0 / len(vals)
    return dict(accuracy=float(np.mean(accs)), baseline=float(baseline), key=key)


def latent_factor_mi_matrix(stoch, factors, keys=None, bins=8):
    Z = _np(stoch).reshape(-1, _np(stoch).shape[-1])
    L = Z.shape[1]
    cols, names = [], []
    keys = keys or ["n_present"]
    for key in keys:
        v = _np(factors[key])
        if v.ndim == 2:
            cols.append(v.reshape(-1)); names.append(key)
        else:
            for i in range(v.shape[-1]):
                cols.append(v[..., i].reshape(-1)); names.append(f"{key}{i}")
    F = len(cols)
    Zb = np.stack([_discretize(Z[:, d], bins) for d in range(L)], 1)
    M = np.zeros((L, F))
    Hf = np.zeros(F)
    for j, col in enumerate(cols):
        cb = _discretize(col, bins)
        Hf[j] = _entropy(cb) + 1e-12
        for d in range(L):
            M[d, j] = mutual_info_score(Zb[:, d], cb) / Hf[j]
    migs = []
    for j in range(F):
        s = np.sort(M[:, j])[::-1]
        if len(s) >= 2:
            migs.append(s[0] - s[1])
    mig = float(np.mean(migs)) if migs else 0.0
    return M, names, mig


def plot_tsne_disentangle(stoch, gamma, factors, path, factor_key="n_present",
                          trajectories=True, max_points=2500, seed=0,
                          title="SHS-RSSM latent space"):
    Z = _np(stoch); B, T, L = Z.shape
    K = _np(gamma).shape[-1]
    Zf = Z.reshape(-1, L)
    regime = _np(gamma).argmax(-1).reshape(-1)
    fac = _np(factors[factor_key])
    fac = fac if fac.ndim == 2 else fac.sum(-1)
    facf = fac.reshape(-1)

    emb, idx = tsne_embedding(Zf, max_points=max_points, seed=seed)
    seq_of = np.repeat(np.arange(B), T)
    t_of = np.tile(np.arange(T), B)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    used = np.unique(regime[idx])

    ax = axes[0]
    if trajectories:
        pos = {r: (emb[i, 0], emb[i, 1]) for i, r in enumerate(idx)}
        for b in range(B):
            rows = [i for i, r in enumerate(idx) if seq_of[r] == b]
            rows.sort(key=lambda i: t_of[idx[i]])
            for a, c in zip(rows[:-1], rows[1:]):
                if t_of[idx[c]] - t_of[idx[a]] == 1:
                    ax.plot([emb[a, 0], emb[c, 0]], [emb[a, 1], emb[c, 1]],
                            color="0.75", lw=0.4, alpha=0.5, zorder=1)
    for k in used:
        m = regime[idx] == k
        ax.scatter(emb[m, 0], emb[m, 1], s=9, color=_PALETTE[k % len(_PALETTE)],
                   alpha=0.75, zorder=2, label=f"regime {k}")
    ax.set_title(f"latents by inferred regime ({len(used)} active of K={K})")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=8, framealpha=0.9, ncol=2)

    ax = axes[1]
    sc = ax.scatter(emb[:, 0], emb[:, 1], s=9, c=facf[idx], cmap="viridis", alpha=0.8)
    ax.set_title(f"same latents by ground truth: {factor_key}")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=factor_key)

    metrics = regime_factor_alignment(gamma, factors)
    fig.suptitle(f"{title}    "
                 f"regime↔composition NMI={metrics['regime_composition_nmi']:.2f}, "
                 f"boundary F1={metrics['boundary_f1']:.2f}", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return metrics


def plot_mi_matrix(stoch, factors, path, keys=None, bins=8,
                   title="Latent ↔ factor mutual information"):
    M, names, mig = latent_factor_mi_matrix(stoch, factors, keys=keys, bins=bins)
    L, F = M.shape
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * F + 3), max(4, 0.28 * L + 1.5)))
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0)
    ax.set_xticks(range(F)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(L)); ax.set_yticklabels([f"z{d}" for d in range(L)], fontsize=7)
    ax.set_xlabel("ground-truth factor"); ax.set_ylabel("latent dimension")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="normalized MI")
    ax.set_title(f"{title}   (MIG={mig:.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return M, names, mig


def plot_tsne_evolution(snapshots, path, color_by="regime", factor_key="n_present",
                        max_points=1500, seed=0,
                        title="Latent t-SNE evolution over training"):
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4))
    if n == 1:
        axes = [axes]
    sc = None
    for ax, snap in zip(axes, snapshots):
        Z = _np(snap["stoch"]).reshape(-1, _np(snap["stoch"]).shape[-1])
        emb, idx = tsne_embedding(Z, max_points=max_points, seed=seed)
        if color_by == "regime":
            regime = _np(snap["gamma"]).argmax(-1).reshape(-1)
            for k in np.unique(regime[idx]):
                m = regime[idx] == k
                ax.scatter(emb[m, 0], emb[m, 1], s=7,
                           color=_PALETTE[k % len(_PALETTE)], alpha=0.7)
            sub = f"{len(np.unique(regime[idx]))} regimes"
        else:
            fac = _np(snap["factors"][factor_key])
            fac = fac if fac.ndim == 2 else fac.sum(-1)
            sc = ax.scatter(emb[:, 0], emb[:, 1], s=7, c=fac.reshape(-1)[idx],
                            cmap="viridis", alpha=0.8)
            sub = factor_key
        lab = snap.get("label", f"step {int(snap.get('step', 0))}")
        ax.set_title(f"{lab}  ({sub})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    if sc is not None:
        fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02, label=factor_key)
    fig.suptitle(title, y=1.02, fontsize=13)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)

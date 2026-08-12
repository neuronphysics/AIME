"""Publication figures for the comparison suite.

Three figure types:

1. ``regime_ribbons``   -- the regime-change figure: for each shown sequence a
   ground-truth colour ribbon, one ribbon per model (labels Hungarian-matched
   to truth so colours agree), and one observation channel underneath so the
   reader can see the change-points against the raw signal.
2. ``latent_partition`` -- NASCAR only: inferred continuous latents coloured
   by inferred regime, one panel per model, next to the true latents.  This is
   the classic "oval track" panel from Linderman et al. / Nassar et al.
3. ``objective_traces`` -- each model's own training objective on its own axis
   (bounds and log-likelihoods are NOT cross-comparable; the panel shows
   convergence, nothing more, and says so in the axis title).
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CMAP, VMAX = "tab20", 19


def _remap(z, mapping, K_pad=None):
    z = np.asarray(z).astype(int)
    hi = max(int(z.max()) + 1, (K_pad or 0))
    lut = np.array([mapping.get(k, k) for k in range(hi)])
    return lut[z]


def _ribbon(ax, row, label):
    ax.imshow(np.asarray(row)[None, :], aspect="auto", cmap=CMAP, vmin=0,
              vmax=VMAX, interpolation="nearest")
    ax.set_yticks([])
    ax.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center")


def state_usage(bundle, model_rows, out_png, title=None):
    """Per-state occupancy, truth vs each model, on the matched label space.

    The ribbon figure understates how many states a fit actually uses: a state
    that appears as many short fragments is nearly invisible there, even though
    it carries real mass.  This panel plots that mass directly -- one bar per
    state -- so "K states are in use" is visible rather than only tabulated.
    Bars use the ribbon colour map, so a bar and its ribbon colour are the same
    state.  Counts run over the WHOLE corpus, not just the sequences drawn in
    the ribbon figure.
    """
    rows = [("truth", np.concatenate(bundle["z_true"]), None)]
    rows += [(label, np.concatenate([np.asarray(s) for s in z_seqs]), mapping)
             for label, z_seqs, mapping in model_rows]
    def _mapped(z, m):
        return z if m is None else _remap(z, m)

    K = 1 + max(int(_mapped(z, m).max()) for _, z, m in rows)

    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 1.5 * len(rows)),
                             squeeze=False, sharex=True)
    axes = axes.ravel()
    for ax, (label, z, mapping) in zip(axes, rows):
        z = _mapped(z, mapping)
        frac = np.bincount(z, minlength=K) / len(z)
        used = int((frac > 0.005).sum())
        ax.bar(np.arange(K), frac,
               color=[plt.get_cmap(CMAP)(k % 20 / VMAX) for k in range(K)],
               edgecolor="k", linewidth=0.3)
        ax.axhline(0.005, color="0.6", lw=0.6, ls=":")
        ax.set_ylabel(f"{label}\n{used} states", fontsize=7, rotation=0,
                      ha="right", va="center")
        ax.tick_params(labelsize=6)
    axes[-1].set_xlabel("state (Hungarian-matched to truth)", fontsize=8)
    axes[-1].set_xticks(np.arange(K))
    fig.suptitle(title or f"{bundle['name']}: state usage over the whole corpus "
                          "(dotted line = 0.5% occupancy)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"figure -> {out_png}")


def regime_ribbons(bundle, model_rows, out_png, n_show=3, obs_dim=0,
                   title=None):
    """model_rows: list of (label, z_pred_per_seq, mapping)."""
    n_show = min(n_show, len(bundle["seqs"]))
    rows_per_seq = 1 + len(model_rows) + 1
    fig, axes = plt.subplots(n_show * rows_per_seq, 1,
                             figsize=(12, 0.62 * n_show * rows_per_seq),
                             squeeze=False)
    axes = axes.ravel()
    i = 0
    for n in range(n_show):
        T = len(bundle["z_true"][n])
        _ribbon(axes[i], bundle["z_true"][n], f"seq{n}\ntruth"); i += 1
        for label, z_seqs, mapping in model_rows:
            _ribbon(axes[i], _remap(z_seqs[n], mapping), label); i += 1
        ax = axes[i]; i += 1
        ax.plot(bundle["seqs"][n][:, obs_dim], lw=0.6, color="k")
        ax.set_xlim(0, T)
        ax.set_ylabel(f"obs[{obs_dim}]", fontsize=7, rotation=0,
                      ha="right", va="center")
        ax.set_yticks([])
    for ax in axes[:-1]:
        ax.set_xticks([])
    fig.suptitle(title or f"{bundle['name']}: regime changes, truth vs models "
                          "(labels Hungarian-matched)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"figure -> {out_png}")


def latent_partition(bundle, model_pts, out_png, title=None):
    """model_pts: list of (label, x (T,2), z (T,), mapping).  Truth first."""
    xt = np.concatenate(bundle["x_true"], 0)
    zt = np.concatenate(bundle["z_true"], 0)
    panels = [("truth", xt, zt, None)] + list(model_pts)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.2),
                             squeeze=False)
    for ax, (label, x, z, mapping) in zip(axes.ravel(), panels):
        z = z if mapping is None else _remap(z, mapping)
        ax.scatter(x[:, 0], x[:, 1], c=np.asarray(z) % 20, cmap=CMAP, vmin=0,
                   vmax=VMAX, s=1.5, linewidths=0)
        ax.set_title(label, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title or f"{bundle['name']}: latent trajectories coloured by "
                          "regime", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"figure -> {out_png}")


def objective_traces(traces, out_png, dataset):
    """traces: dict label -> 1-D array.  One subplot per model, own scale."""
    traces = {k: v for k, v in traces.items() if v is not None and len(v)}
    if not traces:
        return
    fig, axes = plt.subplots(1, len(traces), figsize=(3.4 * len(traces), 2.6),
                             squeeze=False)
    for ax, (label, tr) in zip(axes.ravel(), traces.items()):
        ax.plot(np.asarray(tr), lw=1.0)
        ax.set_title(f"{label} objective\n(own scale; not cross-comparable)",
                     fontsize=8)
        ax.set_xlabel("iteration / lap", fontsize=7)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{dataset}: per-model training objectives", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"figure -> {out_png}")
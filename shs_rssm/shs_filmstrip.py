"""Regime filmstrip: a long-horizon view of the inferred switching sequence with the
actual RGB frames placed under the regime ribbon, plus the aligned z_t evolution.

The existing `plot_latent_clustering` only renders a 5-step timeline (the eval context
window) and never shows the observation, so it cannot tell you whether a regime boundary
corresponds to anything visible in the rollout. This module annotates a long OBSERVED
window (posterior regimes over real frames, not open-loop imagination) and answers the
question directly: at each switch, what does the walker look like, and does z_t actually
move when the regime changes?

Self-contained: pass tensors + a decode/frames array, no DreamerV3 imports.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch

_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
            "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
            "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#5254a3"]


def _col(k):
    return _PALETTE[int(k) % len(_PALETTE)]


def _to_img(x):
    """(C,H,W)/(H,W,C)/(H,W) tensor-or-array -> HxWx3 (or HxW) float in [0,1]."""
    x = x.detach().cpu().float().numpy() if hasattr(x, "detach") else np.asarray(x, np.float32)
    if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        x = np.transpose(x, (1, 2, 0))           # CHW -> HWC
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    return np.clip(x, 0.0, 1.0)


def _segments(lab):
    """Contiguous runs of equal label -> list of (start, end_exclusive, label)."""
    segs, s = [], 0
    for t in range(1, len(lab)):
        if lab[t] != lab[t - 1]:
            segs.append((s, t, int(lab[s]))); s = t
    segs.append((s, len(lab), int(lab[s])))
    return segs


def _merge_flicker(segs, min_dwell):
    """Absorb runs shorter than `min_dwell` into the longer neighbour, for FRAME SELECTION
    only (the ribbon still shows the raw sequence). With low stickiness the raw sequence is
    full of 1-step flickers; this keeps the filmstrip readable and surfaces the persistent
    segments. Returns merged (start, end, label)."""
    if min_dwell <= 1 or len(segs) <= 1:
        return segs
    segs = [list(s) for s in segs]
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, (s, e, _) in enumerate(segs):
            if e - s < min_dwell:
                left = segs[i - 1] if i > 0 else None
                right = segs[i + 1] if i + 1 < len(segs) else None
                # merge into whichever neighbour is longer (ties -> left)
                ln = (left[1] - left[0]) if left else -1
                rn = (right[1] - right[0]) if right else -1
                if ln >= rn and left is not None:
                    left[1] = e
                    segs.pop(i)
                elif right is not None:
                    right[0] = s
                    segs.pop(i)
                changed = True
                break
    return [tuple(s) for s in segs]


def _pick_indices(lab, max_frames, min_dwell):
    """Frame times to show: t=0, every (merged) boundary, every segment midpoint, last t.
    If that exceeds `max_frames`, keep t=0/last/all boundaries and drop midpoints of the
    shortest segments first (boundaries are the diagnostically important times)."""
    T = len(lab)
    merged = _merge_flicker(_segments(lab), min_dwell)
    boundaries = [s for (s, _, _) in merged]               # includes 0
    mids = [(s + e - 1) // 2 for (s, e, _) in merged]
    kind = {}                                              # t -> 'start'|'switch'|'mid'
    for b in boundaries:
        kind[b] = "start" if b == 0 else "switch"
    for (s, e, _), m in zip(merged, mids):
        kind.setdefault(m, "mid")
    kind.setdefault(T - 1, kind.get(T - 1, "mid"))
    chosen = sorted(kind)
    if len(chosen) > max_frames:
        forced = {0, T - 1} | set(boundaries)
        optional = sorted(set(chosen) - forced,
                          key=lambda t: -next(e - s for (s, e, _) in merged if s <= t < e))
        keep = (forced | set(optional[: max(0, max_frames - len(forced))]))
        chosen = sorted(keep)[:max_frames]
    return chosen, kind, merged


def plot_regime_filmstrip(frames, gamma, path, z=None, true_labels=None,
                          max_frames=12, min_dwell=2, frame_stride=1,
                          title="SHS regime filmstrip"):
    """Regime ribbon over an observed window with frames + z_t evolution underneath.

    frames : (T,C,H,W) or (T,H,W) RGB for ONE observed episode/window.
    gamma  : (T,K) posterior responsibilities, or (T,) hard labels.
    z      : optional (T,L) latent (posterior mean or sample) for the evolution heatmap.
    true_labels : optional (T,) ground-truth segmentation drawn as a second ribbon.
    max_frames  : cap on filmstrip frames (boundaries are always kept).
    min_dwell   : runs shorter than this are merged for frame selection (not in the ribbon).
    frame_stride: subsample frames along time before anything (use >1 for long episodes).

    Returns dict(active, n_segments_raw, n_segments_merged, switch_rate).
    """
    g = gamma.detach().cpu().numpy() if hasattr(gamma, "detach") else np.asarray(gamma)
    frames = frames[::frame_stride]
    if g.ndim == 2:
        lab = g.argmax(-1).astype(int)[::frame_stride]
        conf = np.clip(g.max(-1), 0.0, 1.0)[::frame_stride]  # float32 max can be 1.0000x -> alpha>1
        K = g.shape[-1]
    else:
        lab = g.astype(int)[::frame_stride]
        conf = np.ones_like(lab, float)
        K = int(lab.max()) + 1
    T = len(lab)
    if z is not None:
        z = (z.detach().cpu().numpy() if hasattr(z, "detach") else np.asarray(z))[::frame_stride]
    if true_labels is not None:
        tl = (true_labels.detach().cpu().numpy() if hasattr(true_labels, "detach")
              else np.asarray(true_labels))[::frame_stride].astype(int)
    else:
        tl = None

    idx, kind, merged = _pick_indices(lab, max_frames, min_dwell)
    raw_segs = _segments(lab)

    # ---- layout: ribbon(s) + optional z heatmap on a shared time axis, frames below ----
    n_ribbon_rows = 1 + (tl is not None)
    has_z = z is not None
    fig = plt.figure(figsize=(max(11.0, 1.05 * len(idx)), 6.4 if has_z else 5.0))
    height_ratios = ([0.5 * n_ribbon_rows] + ([1.15] if has_z else []) + [2.7])
    gs = fig.add_gridspec(len(height_ratios), 1, height_ratios=height_ratios, hspace=0.32)

    # ribbon
    ax_r = fig.add_subplot(gs[0, 0])
    if tl is None:
        for t in range(T):
            ax_r.axvspan(t - 0.5, t + 0.5, color=_col(lab[t]), alpha=0.35 + 0.65 * conf[t], lw=0)
        ax_r.set_yticks([0.5]); ax_r.set_yticklabels(["inferred"])
    else:
        for t in range(T):
            ax_r.axvspan(t - 0.5, t + 0.5, ymin=0.0, ymax=0.45,
                         color=_col(lab[t]), alpha=0.35 + 0.65 * conf[t], lw=0)
            ax_r.axvspan(t - 0.5, t + 0.5, ymin=0.55, ymax=1.0, color=_col(tl[t]), lw=0)
        ax_r.set_yticks([0.22, 0.78]); ax_r.set_yticklabels(["inferred", "true"])
    for t in idx:                                          # ticks where frames are sampled
        ax_r.axvline(t, color="k", lw=0.8, ymin=-0.05, ymax=1.05, clip_on=False)
    ax_r.set_xlim(-0.5, T - 0.5); ax_r.set_ylim(0, 1)
    ax_r.set_xticks([]); ax_r.set_title("regime ribbon (alpha = posterior confidence)",
                                        fontsize=10)

    # z_t evolution heatmap (latent dim x time), boundaries overlaid
    if has_z:
        ax_z = fig.add_subplot(gs[1, 0])
        zc = z.T                                           # (L, T)
        vlim = np.percentile(np.abs(zc), 99) + 1e-6
        ax_z.imshow(zc, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                    extent=[-0.5, T - 0.5, zc.shape[0] - 0.5, -0.5], interpolation="nearest")
        for (s, _, _) in merged[1:]:                       # merged switch lines
            ax_z.axvline(s - 0.5, color="k", lw=1.1, alpha=0.9)
        ax_z.set_xlim(-0.5, T - 0.5)
        ax_z.set_ylabel("z dim", fontsize=9)
        ax_z.set_xlabel("time step", fontsize=9)
        ax_z.set_title("z_t evolution (black = merged regime switch)", fontsize=10)
    else:
        ax_r.set_xlabel("time step", fontsize=9)

    # filmstrip
    ax_f = fig.add_subplot(gs[-1, 0]); ax_f.axis("off")
    n = len(idx)
    pad, w = 0.012, None
    w = (1.0 - pad * (n + 1)) / n
    for j, t in enumerate(idx):
        x0 = pad + j * (w + pad)
        sub = ax_f.inset_axes([x0, 0.0, w, 0.9])
        img = _to_img(frames[min(t, len(frames) - 1)])
        sub.imshow(img, cmap=None if img.ndim == 3 else "magma", vmin=0, vmax=1)
        sub.set_xticks([]); sub.set_yticks([])
        for s in sub.spines.values():
            s.set_edgecolor(_col(lab[t])); s.set_linewidth(3.0)
        tag = {"start": "start", "switch": "switch", "mid": "mid"}[kind[t]]
        sub.set_title(f"t={t}  r{lab[t]}\n{tag}", fontsize=8,
                      color="k" if kind[t] == "mid" else _col(lab[t]),
                      fontweight="bold" if kind[t] != "mid" else "normal")
        # connector from the ribbon tick down to the frame
        con = ConnectionPatch(xyA=(t, 0), xyB=(0.5, 1.0), coordsA=ax_r.transData,
                              coordsB=sub.transAxes, color="0.5", lw=0.7, alpha=0.7)
        fig.add_artist(con)

    active = int(len(np.unique(lab)))
    switch_rate = (len(raw_segs) - 1) / max(1, T - 1)
    fig.suptitle(f"{title}    |  {active} active regimes, "
                 f"{len(raw_segs)} raw segments, {len(merged)} after flicker-merge, "
                 f"switch rate {switch_rate:.2f}/step", y=0.99, fontsize=11)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return dict(active=active, n_segments_raw=len(raw_segs),
                n_segments_merged=len(merged), switch_rate=float(switch_rate))
